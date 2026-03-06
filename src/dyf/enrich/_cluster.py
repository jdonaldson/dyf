"""Level 1 → 2: Louvain clustering enrichment."""

import json
import re

import numpy as np

from dyf.colors import spatial_rgb_map
from dyf.lazy_index import LazyIndex, rewrite_lazy_index
from dyf.provenance import create_provenance, provenance_to_dict

from ._labeling import (
    annotate_cluster_names,
    label_clusters,
    transfer_labels_majority_vote,
)


def fit_birch(data, target_k, max_iters=10):
    """Fit BIRCH with enough subclusters, then agglomerative merge."""
    from scipy.cluster.hierarchy import linkage, fcluster
    from sklearn.cluster import Birch

    lo, hi = 1e-4, float(np.linalg.norm(data.max(axis=0) - data.min(axis=0)))
    best_birch, best_n = None, 0

    for _ in range(max_iters):
        mid = (lo + hi) / 2
        birch = Birch(n_clusters=None, threshold=mid, branching_factor=50)
        birch.fit(data)
        n = len(birch.subcluster_centers_)
        if n >= target_k and (best_birch is None or n < best_n):
            best_birch, best_n = birch, n
        if n < target_k:
            hi = mid
        else:
            lo = mid
        if target_k <= n <= target_k * 3:
            break

    if best_birch is None:
        best_birch = Birch(n_clusters=None, threshold=lo, branching_factor=50)
        best_birch.fit(data)

    n_subs = len(best_birch.subcluster_centers_)
    if n_subs <= target_k:
        return best_birch

    birch = Birch(n_clusters=target_k, threshold=best_birch.threshold,
                  branching_factor=50)
    birch.fit(data)
    return birch


def merge_tiny_clusters(labels, coords, min_pct=0.005):
    """Merge clusters smaller than min_pct into nearest large neighbor."""
    from collections import Counter
    min_size = max(10, int(len(labels) * min_pct))
    counts = Counter(labels)
    tiny = {cid for cid, cnt in counts.items() if cnt < min_size}
    if not tiny:
        return labels

    labels = labels.copy()
    big = sorted(set(labels) - tiny)
    centroids = {}
    for cid in big:
        mask = labels == cid
        centroids[cid] = coords[mask].mean(axis=0)
    for tcid in sorted(tiny):
        mask = labels == tcid
        cent = coords[mask].mean(axis=0)
        dists = {c: np.linalg.norm(cent - v) for c, v in centroids.items()}
        nearest = min(dists, key=dists.get)
        labels[mask] = nearest
        print(f"    Merged cluster {tcid} ({counts[tcid]} pts) → {nearest}")

    old_ids = sorted(set(labels.tolist()))
    id_map = {old: new for new, old in enumerate(old_ids)}
    labels = np.array([id_map[l] for l in labels])
    return labels


def enrich_cluster(dyf_path, model="gpt-oss:20b",
                   output_path=None, force=False, domain=None,
                   resolution=1.0):
    """Add Louvain cluster labels + dendrogram to a .dyf file (Level 1 → 2)."""
    print(f"\n=== Level 2: Louvain Clustering (dendrogram) ===")
    print(f"  Input: {dyf_path}")

    with LazyIndex(dyf_path) as idx:
        level = idx.detect_enrichment_level()
        if level < 1:
            print(f"  ERROR: Need level 1 (UMAP coords), got level {level}. "
                  f"Run 'project' first.")
            return
        if level >= 2 and not force:
            print(f"  Already at level {level} (has clusters), skipping. "
                  f"Use --force to re-run.")
            return

    # Extract data
    with LazyIndex(dyf_path) as idx:
        data = idx.extract_all_fields()
    n = len(data['embeddings'])

    umap_x = data['fields']['umap_x']
    umap_y = data['fields']['umap_y']
    umap_z = data['fields']['umap_z']
    coords = np.column_stack([umap_x, umap_y, umap_z])

    titles = data['fields'].get('title')
    if titles is None:
        titles = [f"Item {i}" for i in range(n)]
    embeddings = data['embeddings']

    if domain is None:
        domain = data['metadata'].get('domain')
    if domain:
        print(f"  Domain: {domain}")

    label_cache_data = json.loads(data['metadata'].get('_label_cache', '{}'))
    if label_cache_data:
        print(f"  Loaded {len(label_cache_data)} label cache entries from .dyf")

    split_kw_json = data['metadata'].get('split_keywords')
    split_kw_data = None
    if split_kw_json:
        split_kw_data = json.loads(split_kw_json)
        n_splits = len(split_kw_data.get('splits', {}))
        print(f"  Using split keywords ({n_splits} splits) for label context")

    tree_maps = None
    if split_kw_data:
        try:
            from dyf.splits import build_tree_maps
            with LazyIndex(dyf_path) as idx_tree:
                tree_maps = build_tree_maps(idx_tree)
            print(f"  Loaded tree structure for cluster-tree DAG")
        except Exception as e:
            print(f"  WARNING: Could not load tree structure: {e}")

    # Check text diversity
    from dyf.splits import assess_text_diversity, label_clusters_frequency
    title_list = titles if isinstance(titles, list) else list(titles)
    diversity = assess_text_diversity(title_list)
    use_frequency_labels = not diversity.is_diverse
    if use_frequency_labels:
        print(f"  LOW TEXT DIVERSITY: {diversity.reason}")
        print(f"    ({diversity.unique_token_count} unique tokens, "
              f"token/item={diversity.token_item_ratio:.6f}, "
              f"title ratio={diversity.unique_title_ratio:.4f})")
        print(f"    Using frequency-based labeling (no LLM)")

    new_sf = {}
    new_meta = {}

    # ── Louvain: natural communities with dendrogram ──
    print(f"\n  Computing Louvain communities from tree leaves...")
    from dyf.agglomerate import compute_louvain_hierarchy
    with LazyIndex(dyf_path) as idx_agg:
        hierarchy = compute_louvain_hierarchy(idx_agg, coords, embeddings,
                                                 resolution=resolution)

    if hierarchy is None:
        print("    Skipped: tree has fewer than 2 leaves")
    else:
        point_labels = hierarchy['point_labels']
        natural_k = hierarchy['natural_k']
        print(f"    Natural k = {natural_k}")

        print(f"  Labeling {natural_k} communities...")
        if use_frequency_labels:
            community_names = label_clusters_frequency(
                title_list, point_labels)
        else:
            community_names = label_clusters(
                titles, coords, point_labels, embeddings,
                model=model, cache_data=label_cache_data,
                cache_key="louvain_communities",
                split_keywords=split_kw_data,
                domain=domain)
        label_cache_data["louvain_communities"] = {
            str(k): v for k, v in community_names.items()}

        rgb = spatial_rgb_map(point_labels.tolist(), embeddings)

        community_centroids = {}
        for cid in sorted(set(point_labels.tolist())):
            mask = point_labels == cid
            cent = coords[mask].mean(axis=0)
            community_centroids[str(cid)] = [
                round(float(cent[0]), 4),
                round(float(cent[1]), 4),
                round(float(cent[2]), 4) if coords.shape[1] > 2 else 0.0]

        new_meta['louvain_leaf_communities'] = json.dumps({
            'leaf_to_community': {
                str(k): v for k, v in
                hierarchy['leaf_to_community'].items()},
            'natural_k': natural_k,
            'resolution': hierarchy['resolution'],
        })

        new_meta['louvain_dendrogram'] = json.dumps({
            'Z': hierarchy['Z'].tolist(),
            'community_names': {
                str(k): v for k, v in community_names.items()},
            'community_colors': {
                str(k): v for k, v in rgb.items()},
            'community_centroids': community_centroids,
            'community_sizes': {
                str(k): v for k, v in
                hierarchy['community_sizes'].items()},
        })

        new_meta['leaf_item_map'] = json.dumps({
            str(k): v for k, v in
            hierarchy['leaf_item_map'].items()
        })

        new_sf['community_id'] = hierarchy['point_labels'].astype(np.int32)
        new_sf['centroid_dist'] = hierarchy['centroid_dist']
        new_sf['nearest_other_dist'] = hierarchy['nearest_other_dist']

        dendro_extra = json.loads(new_meta['louvain_dendrogram'])
        dendro_extra['community_cohesion'] = {
            str(k): round(v, 6) for k, v in
            hierarchy['community_cohesion'].items()}
        dendro_extra['community_embedding_centroids'] = {
            str(k): v.tolist() for k, v in zip(
                hierarchy['unique_community_ids'],
                hierarchy['community_embedding_centroids'])}
        new_meta['louvain_dendrogram'] = json.dumps(dendro_extra)

        print(f"    Stored dendrogram metadata "
              f"(Z: {len(hierarchy['Z'])} merges, "
              f"{len(hierarchy['leaf_item_map'])} leaves)")
        print(f"    Stored per-point fields: community_id, "
              f"centroid_dist, nearest_other_dist")

    # ── LSH tree-leaf agglomeration (50-bucket layer) ──
    from dyf.colors import tree_rgb_map

    print("\n  Computing agglomerated DYF tree buckets...")
    from dyf.agglomerate import agglomerate_tree_leaves
    with LazyIndex(dyf_path) as idx_agg:
        lsh_labels, lsh_names, lsh_label_data, item_leaf_map, tree_struct = \
            agglomerate_tree_leaves(idx_agg, coords, embeddings, n_groups=50)

    if lsh_labels is not None:
        n_lsh = len(set(lsh_labels.tolist()))
        print(f"    {n_lsh} agglomerated buckets")

        print("  Labeling agglomerated buckets...")
        if use_frequency_labels:
            bucket_names = label_clusters_frequency(title_list, lsh_labels)
        else:
            bucket_names = label_clusters(
                titles, coords, lsh_labels, embeddings,
                model=model, cache_data=label_cache_data,
                cache_key="lsh_buckets", domain=domain)
        label_cache_data["lsh_buckets"] = {
            str(k): v for k, v in bucket_names.items()}

        for entry in lsh_label_data:
            cid = entry["cid"]
            if cid in bucket_names:
                entry["text"] = bucket_names[cid][:50]

        new_sf['lsh_bucket_ids'] = lsh_labels.astype(np.int32)

        new_meta['lsh_bucket_names'] = json.dumps(
            {str(k): v for k, v in bucket_names.items()})
        centroids_lsh = {}
        for entry in lsh_label_data:
            centroids_lsh[str(entry["cid"])] = [
                round(entry["x"], 4),
                round(entry["y"], 4),
                round(entry["z"], 4)]
        new_meta['lsh_bucket_centroids'] = json.dumps(centroids_lsh)

        lsh_colors = tree_rgb_map(lsh_labels, tree_struct, item_leaf_map)
        new_meta['lsh_bucket_colors'] = json.dumps(
            {str(k): v for k, v in lsh_colors.items()})

        print(f"    Stored LSH bucket IDs, names, centroids, and colors")
    else:
        print("    Skipped: tree has fewer than 2 leaves")

    # Strip stale level 3 metadata when re-clustering
    if force and level >= 3:
        for stale_key in ['edge_pairs', 'edge_paths_2d', 'tour_narration',
                          '_provenance_level_3']:
            new_meta[stale_key] = None
        print("  Stripped stale level 3 metadata (re-run 'viz' to regenerate)")

    new_meta['_label_cache'] = json.dumps(label_cache_data)

    # Stamp provenance for Level 2
    new_meta['_provenance_level_2'] = json.dumps(provenance_to_dict(
        create_provenance(
            artifact_type="dyf",
            n_items=n,
            source_paths=[str(dyf_path)],
            params={"mode": "louvain_dendrogram",
                    "model": model},
        )
    ))

    # Identify stale BIRCH cluster_* fields to drop
    with LazyIndex(dyf_path) as idx_stale:
        stale_fields = {f for f in idx_stale.stored_field_names
                        if f.startswith('cluster_')}
    for key in list(data['metadata'].keys()):
        if (key.startswith('cluster_names_')
                or key.startswith('cluster_centroids_')
                or key.startswith('cluster_colors_')):
            new_meta[key] = None

    if stale_fields:
        print(f"  Dropping stale fields: {sorted(stale_fields)}")

    out = output_path or dyf_path
    print(f"\n  Writing enriched file: {out}")
    rewrite_lazy_index(dyf_path, new_stored_fields=new_sf,
                       new_metadata=new_meta, output_path=out,
                       drop_fields=stale_fields)
    print(f"  Done. Level 1 → 2 (dual 2D/3D)")
