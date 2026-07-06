"""Level 1 → 2: Louvain clustering enrichment."""

import json
import logging

import numpy as np

logger = logging.getLogger(__name__)

from dyf.colors import spatial_rgb_map
from dyf.lazy_index import LazyIndex, rewrite_lazy_index
from dyf.provenance import create_provenance, provenance_to_dict

from ._labeling import (
    label_clusters,
)


def fit_birch(data: np.ndarray, target_k: int, max_iters: int = 10):
    """Fit BIRCH with enough subclusters, then agglomerative merge."""
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

    birch = Birch(n_clusters=target_k, threshold=best_birch.threshold, branching_factor=50)
    birch.fit(data)
    return birch


def merge_tiny_clusters(labels: np.ndarray, coords: np.ndarray, min_pct: float = 0.005) -> np.ndarray:
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
        logger.info(f"    Merged cluster {tcid} ({counts[tcid]} pts) → {nearest}")

    old_ids = sorted(set(labels.tolist()))
    id_map = {old: new for new, old in enumerate(old_ids)}
    labels = np.array([id_map[l] for l in labels])
    return labels


def _extract_cluster_inputs(dyf_path, domain):
    """Extract data, fields, caches, and text diversity info from a .dyf file.

    Args:
        dyf_path: Path to the .dyf file.
        domain: Optional domain string (overrides metadata if provided).

    Returns:
        dict with keys: n, coords, titles, embeddings, domain,
        label_cache_data, split_kw_data, use_frequency_labels, data,
        title_list.
    """
    from dyf.splits import assess_text_diversity

    with LazyIndex(dyf_path) as idx:
        data = idx.extract_all_fields()
    n = len(data["embeddings"])

    umap_x = data["fields"]["umap_x"]
    umap_y = data["fields"]["umap_y"]
    umap_z = data["fields"]["umap_z"]
    coords = np.column_stack([umap_x, umap_y, umap_z])

    titles = data["fields"].get("title")
    if titles is None:
        titles = [f"Item {i}" for i in range(n)]
    embeddings = data["embeddings"]

    if domain is None:
        domain = data["metadata"].get("domain")
    if domain:
        logger.info(f"  Domain: {domain}")

    label_cache_data = json.loads(data["metadata"].get("_label_cache", "{}"))
    if label_cache_data:
        logger.info(f"  Loaded {len(label_cache_data)} label cache entries from .dyf")

    split_kw_json = data["metadata"].get("split_keywords")
    split_kw_data = None
    if split_kw_json:
        split_kw_data = json.loads(split_kw_json)
        n_splits = len(split_kw_data.get("splits", {}))
        logger.info(f"  Using split keywords ({n_splits} splits) for label context")

    tree_maps = None  # noqa: F841
    if split_kw_data:
        try:
            from dyf.splits import build_tree_maps

            with LazyIndex(dyf_path) as idx_tree:
                tree_maps = build_tree_maps(idx_tree)  # noqa: F841
            logger.info("  Loaded tree structure for cluster-tree DAG")
        except Exception as e:
            logger.warning("Could not load tree structure: %s", e)

    # Check text diversity
    title_list = titles if isinstance(titles, list) else list(titles)
    diversity = assess_text_diversity(title_list)
    use_frequency_labels = not diversity.is_diverse
    if use_frequency_labels:
        logger.info(f"  LOW TEXT DIVERSITY: {diversity.reason}")
        logger.info(
            f"    ({diversity.unique_token_count} unique tokens, "
            f"token/item={diversity.token_item_ratio:.6f}, "
            f"title ratio={diversity.unique_title_ratio:.4f})"
        )
        logger.info("    Using frequency-based labeling (no LLM)")

    return {
        "n": n,
        "coords": coords,
        "titles": titles,
        "embeddings": embeddings,
        "domain": domain,
        "label_cache_data": label_cache_data,
        "split_kw_data": split_kw_data,
        "use_frequency_labels": use_frequency_labels,
        "data": data,
        "title_list": title_list,
    }


def _run_louvain_phase(
    dyf_path,
    coords,
    embeddings,
    titles,
    title_list,
    model,
    domain,
    label_cache_data,
    split_kw_data,
    use_frequency_labels,
    resolution,
    new_sf,
    new_meta,
):
    """Run Louvain community detection, label communities, and populate metadata.

    Modifies new_sf, new_meta, and label_cache_data dicts in place.

    Args:
        dyf_path: Path to the .dyf file.
        coords: (N, 3) UMAP coordinates.
        embeddings: (N, D) embedding matrix.
        titles: Titles array/list.
        title_list: Titles as a plain list.
        model: LLM model name for labeling.
        domain: Optional domain string.
        label_cache_data: Mutable label cache dict.
        split_kw_data: Split keywords data or None.
        use_frequency_labels: Whether to use frequency-based labeling.
        resolution: Louvain resolution parameter.
        new_sf: Mutable dict for new stored fields.
        new_meta: Mutable dict for new metadata.
    """
    from dyf.splits import label_clusters_frequency

    logger.info("  Computing Louvain communities from tree leaves...")
    from dyf.agglomerate import compute_louvain_hierarchy

    with LazyIndex(dyf_path) as idx_agg:
        hierarchy = compute_louvain_hierarchy(idx_agg, coords, embeddings, resolution=resolution)

    if hierarchy is None:
        logger.info("    Skipped: tree has fewer than 2 leaves")
        return

    point_labels = hierarchy["point_labels"]
    natural_k = hierarchy["natural_k"]
    logger.info(f"    Natural k = {natural_k}")

    logger.info(f"  Labeling {natural_k} communities...")
    if use_frequency_labels:
        community_names = label_clusters_frequency(title_list, point_labels)
    else:
        community_names = label_clusters(
            titles,
            coords,
            point_labels,
            embeddings,
            model=model,
            cache_data=label_cache_data,
            cache_key="louvain_communities",
            split_keywords=split_kw_data,
            domain=domain,
        )
    label_cache_data["louvain_communities"] = {str(k): v for k, v in community_names.items()}

    rgb = spatial_rgb_map(point_labels.tolist(), embeddings)

    community_centroids = {}
    for cid in sorted(set(point_labels.tolist())):
        mask = point_labels == cid
        cent = coords[mask].mean(axis=0)
        community_centroids[str(cid)] = [
            round(float(cent[0]), 4),
            round(float(cent[1]), 4),
            round(float(cent[2]), 4) if coords.shape[1] > 2 else 0.0,
        ]

    new_meta["louvain_leaf_communities"] = json.dumps(
        {
            "leaf_to_community": {str(k): v for k, v in hierarchy["leaf_to_community"].items()},
            "natural_k": natural_k,
            "resolution": hierarchy["resolution"],
        }
    )

    new_meta["louvain_dendrogram"] = json.dumps(
        {
            "Z": hierarchy["Z"].tolist(),
            "community_names": {str(k): v for k, v in community_names.items()},
            "community_colors": {str(k): v for k, v in rgb.items()},
            "community_centroids": community_centroids,
            "community_sizes": {str(k): v for k, v in hierarchy["community_sizes"].items()},
        }
    )

    new_meta["leaf_item_map"] = json.dumps({str(k): v for k, v in hierarchy["leaf_item_map"].items()})

    new_sf["community_id"] = hierarchy["point_labels"].astype(np.int32)
    new_sf["centroid_dist"] = hierarchy["centroid_dist"]
    new_sf["nearest_other_dist"] = hierarchy["nearest_other_dist"]

    dendro_extra = json.loads(new_meta["louvain_dendrogram"])
    dendro_extra["community_cohesion"] = {str(k): round(v, 6) for k, v in hierarchy["community_cohesion"].items()}
    dendro_extra["community_embedding_centroids"] = {
        str(k): v.tolist()
        for k, v in zip(hierarchy["unique_community_ids"], hierarchy["community_embedding_centroids"])
    }
    new_meta["louvain_dendrogram"] = json.dumps(dendro_extra)

    logger.info(
        f"    Stored dendrogram metadata (Z: {len(hierarchy['Z'])} merges, {len(hierarchy['leaf_item_map'])} leaves)"
    )
    logger.info("    Stored per-point fields: community_id, centroid_dist, nearest_other_dist")


def _run_lsh_agglomeration_phase(
    dyf_path,
    coords,
    embeddings,
    titles,
    title_list,
    model,
    domain,
    label_cache_data,
    use_frequency_labels,
    new_sf,
    new_meta,
):
    """Run LSH tree-leaf agglomeration, label buckets, and populate metadata.

    Modifies new_sf, new_meta, and label_cache_data dicts in place.

    Args:
        dyf_path: Path to the .dyf file.
        coords: (N, 3) UMAP coordinates.
        embeddings: (N, D) embedding matrix.
        titles: Titles array/list.
        title_list: Titles as a plain list.
        model: LLM model name for labeling.
        domain: Optional domain string.
        label_cache_data: Mutable label cache dict.
        use_frequency_labels: Whether to use frequency-based labeling.
        new_sf: Mutable dict for new stored fields.
        new_meta: Mutable dict for new metadata.
    """
    from dyf.colors import tree_rgb_map
    from dyf.splits import label_clusters_frequency

    logger.info("  Computing agglomerated DYF tree buckets...")
    from dyf.agglomerate import agglomerate_tree_leaves

    with LazyIndex(dyf_path) as idx_agg:
        lsh_labels, lsh_names, lsh_label_data, item_leaf_map, tree_struct = agglomerate_tree_leaves(
            idx_agg, coords, embeddings, n_groups=50
        )

    if lsh_labels is not None:
        n_lsh = len(set(lsh_labels.tolist()))
        logger.info(f"    {n_lsh} agglomerated buckets")

        logger.info("  Labeling agglomerated buckets...")
        if use_frequency_labels:
            bucket_names = label_clusters_frequency(title_list, lsh_labels)
        else:
            bucket_names = label_clusters(
                titles,
                coords,
                lsh_labels,
                embeddings,
                model=model,
                cache_data=label_cache_data,
                cache_key="lsh_buckets",
                domain=domain,
            )
        label_cache_data["lsh_buckets"] = {str(k): v for k, v in bucket_names.items()}

        for entry in lsh_label_data:
            cid = entry["cid"]
            if cid in bucket_names:
                entry["text"] = bucket_names[cid][:50]

        new_sf["lsh_bucket_ids"] = lsh_labels.astype(np.int32)

        new_meta["lsh_bucket_names"] = json.dumps({str(k): v for k, v in bucket_names.items()})
        centroids_lsh = {}
        for entry in lsh_label_data:
            centroids_lsh[str(entry["cid"])] = [round(entry["x"], 4), round(entry["y"], 4), round(entry["z"], 4)]
        new_meta["lsh_bucket_centroids"] = json.dumps(centroids_lsh)

        lsh_colors = tree_rgb_map(lsh_labels, tree_struct, item_leaf_map)
        new_meta["lsh_bucket_colors"] = json.dumps({str(k): v for k, v in lsh_colors.items()})

        logger.info("    Stored LSH bucket IDs, names, centroids, and colors")
    else:
        logger.info("    Skipped: tree has fewer than 2 leaves")


def enrich_cluster(dyf_path, model="gpt-oss:20b", output_path=None, force=False, domain=None, resolution=1.0):
    """Add Louvain cluster labels + dendrogram to a .dyf file (Level 1 → 2)."""
    logger.info("=== Level 2: Louvain Clustering (dendrogram) ===")
    logger.info(f"  Input: {dyf_path}")

    with LazyIndex(dyf_path) as idx:
        level = idx.detect_enrichment_level()
        if level < 1:
            logger.warning(f"  Need level 1 (UMAP coords), got level {level}. Run 'project' first.")
            return
        if level >= 2 and not force:
            logger.info(f"  Already at level {level} (has clusters), skipping. Use --force to re-run.")
            return

    inputs = _extract_cluster_inputs(dyf_path, domain)
    n = inputs["n"]
    coords = inputs["coords"]
    embeddings = inputs["embeddings"]
    titles = inputs["titles"]
    title_list = inputs["title_list"]
    domain = inputs["domain"]
    label_cache_data = inputs["label_cache_data"]
    split_kw_data = inputs["split_kw_data"]
    use_frequency_labels = inputs["use_frequency_labels"]
    data = inputs["data"]

    new_sf = {}
    new_meta = {}

    _run_louvain_phase(
        dyf_path,
        coords,
        embeddings,
        titles,
        title_list,
        model,
        domain,
        label_cache_data,
        split_kw_data,
        use_frequency_labels,
        resolution,
        new_sf,
        new_meta,
    )

    _run_lsh_agglomeration_phase(
        dyf_path,
        coords,
        embeddings,
        titles,
        title_list,
        model,
        domain,
        label_cache_data,
        use_frequency_labels,
        new_sf,
        new_meta,
    )

    # Strip stale level 3 metadata when re-clustering
    if force and level >= 3:
        for stale_key in ["edge_pairs", "edge_paths_2d", "tour_narration", "_provenance_level_3"]:
            new_meta[stale_key] = None
        logger.info("  Stripped stale level 3 metadata (re-run 'viz' to regenerate)")

    new_meta["_label_cache"] = json.dumps(label_cache_data)

    # Stamp provenance for Level 2
    new_meta["_provenance_level_2"] = json.dumps(
        provenance_to_dict(
            create_provenance(
                artifact_type="dyf",
                n_items=n,
                source_paths=[str(dyf_path)],
                params={"mode": "louvain_dendrogram", "model": model},
            )
        )
    )

    # Identify stale BIRCH cluster_* fields to drop
    with LazyIndex(dyf_path) as idx_stale:
        stale_fields = {f for f in idx_stale.stored_field_names if f.startswith("cluster_")}
    for key in list(data["metadata"].keys()):
        if (
            key.startswith("cluster_names_")
            or key.startswith("cluster_centroids_")
            or key.startswith("cluster_colors_")
        ):
            new_meta[key] = None

    if stale_fields:
        logger.info(f"  Dropping stale fields: {sorted(stale_fields)}")

    out = output_path or dyf_path
    logger.info(f"  Writing enriched file: {out}")
    rewrite_lazy_index(
        dyf_path, new_stored_fields=new_sf, new_metadata=new_meta, output_path=out, drop_fields=stale_fields
    )
    logger.info("  Done. Level 1 → 2 (dual 2D/3D)")
