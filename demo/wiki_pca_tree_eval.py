"""
Evaluate clustering: BIRCH (2D) vs PCA tree (high-D) vs DYF LSH (high-D)

All methods share the same DYF-parameterized UMAP 2D projection for spatial
metrics.  Clustering comes from four sources:
  1. BIRCH on 2D UMAP coords (baseline)
  2. PCA tree on high-D embeddings (cut_tree_to_labels)
  3. DYF single-level: LSH buckets with num_bits tuned for ~target_k
  4. DYF hierarchical: two-tier LSH (global buckets + local facets)

Metrics:
  - Silhouette (2D): spatial cluster coherence
  - kNN purity (2D): fraction of spatial neighbors sharing cluster label
  - Intra-cluster cosine sim (high-D): do clusters correspond to real topics?
  - Inter-cluster cosine sim (high-D): are clusters semantically distinct?
  - Sim gap: intra - inter (positive = semantically meaningful clusters)
  - Cluster fragmentation: how many spatial blobs per cluster?

Usage:
    python demo/wiki_pca_tree_eval.py demo/wiki_simple_50k.parquet [--sample 8000]
"""

import argparse
import time
from collections import defaultdict

import numpy as np
import polars as pl
import umap
from sklearn.cluster import Birch, AgglomerativeClustering
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors
from scipy.sparse.csgraph import connected_components

from dyf.pca_tree import build_pca_tree, cut_tree_to_labels
from dyf.dyf_tree import build_dyf_tree, cut_dyf_tree_to_labels


def load_and_dedup(parquet_path, sample=None):
    from dyf_rs import DensityClassifier
    from dyf.chunks import deduplicate_chunks

    df = pl.read_parquet(parquet_path)
    if sample and sample < len(df):
        df = df.sample(sample, seed=42)

    titles = df["title"].to_list()
    embeddings = np.array(df["embedding"].to_list(), dtype=np.float32)

    clf = DensityClassifier(embedding_dim=embeddings.shape[1], num_bits=12, seed=42)
    clf.fit(embeddings)
    bucket_ids = np.asarray(clf.get_bucket_ids())
    dedup_mask = deduplicate_chunks(bucket_ids, np.asarray(titles))

    titles = [t for t, keep in zip(titles, dedup_mask) if keep]
    embeddings = embeddings[dedup_mask]
    print(f"Loaded {len(titles)} points after dedup")
    return titles, embeddings


def suggest_n_neighbors(embeddings):
    from dyf_rs import DensityClassifier
    clf = DensityClassifier(embedding_dim=embeddings.shape[1], num_bits=12, seed=42)
    clf.fit(embeddings)
    mean_size = np.array(clf.get_bucket_sizes()).mean()
    return int(np.clip(mean_size, 15, 100))


def run_umap(embeddings, n_neighbors):
    reducer = umap.UMAP(
        n_components=2, n_neighbors=n_neighbors,
        min_dist=0.1, n_jobs=-1, verbose=False,
    )
    coords = np.asarray(reducer.fit_transform(embeddings))
    nan_mask = np.isnan(coords).any(axis=1)
    if nan_mask.any():
        nn = NearestNeighbors(n_neighbors=1, metric='cosine')
        nn.fit(embeddings[~nan_mask])
        _, idx = nn.kneighbors(embeddings[nan_mask])
        coords[nan_mask] = coords[~nan_mask][idx.ravel()]
    median = np.nanmedian(coords, axis=0)
    mad = np.nanmedian(np.abs(coords - median), axis=0)
    scale = float(np.fmax(np.nanmax(mad), 1e-8))
    coords = (coords - median) / scale
    return coords


def fit_birch(data, target_k, max_iters=10):
    lo, hi = 1e-4, float(np.linalg.norm(data.max(axis=0) - data.min(axis=0)))
    best_birch, best_diff = None, float('inf')
    for _ in range(max_iters):
        mid = (lo + hi) / 2
        birch = Birch(n_clusters=None, threshold=mid, branching_factor=50)
        birch.fit(data)
        n = len(birch.subcluster_centers_)
        diff = abs(n - target_k)
        if diff < best_diff:
            best_birch, best_diff = birch, diff
        if n > target_k:
            lo = mid
        else:
            hi = mid
        if target_k // 2 <= n <= target_k * 2:
            break
    return best_birch


def dyf_single_level(embeddings, target_k, seed=42):
    """LSH bucketing with num_bits chosen to approximate target_k clusters."""
    from dyf_rs import DensityClassifier

    dim = embeddings.shape[1]
    best_labels, best_diff = None, float('inf')
    best_bits = None

    # Try bits from 3 to 8, pick closest to target_k populated buckets
    for bits in range(3, 9):
        clf = DensityClassifier(embedding_dim=dim, num_bits=bits, seed=seed)
        clf.fit(embeddings)
        bucket_ids = np.array(clf.get_bucket_ids())
        n_populated = len(set(bucket_ids.tolist()))
        diff = abs(n_populated - target_k)
        if diff < best_diff:
            best_labels = bucket_ids
            best_diff = diff
            best_bits = bits

    # Relabel to contiguous 0..n_clusters-1
    unique = sorted(set(best_labels.tolist()))
    remap = {old: new for new, old in enumerate(unique)}
    labels = np.array([remap[b] for b in best_labels])
    print(f"  Best: {best_bits} bits -> {len(unique)} clusters")
    return labels


def dyf_hierarchical(embeddings, target_k, seed=42):
    """Two-tier DYF: global LSH buckets, then local facets in dense buckets.

    Produces fine-grained facet IDs, then merges them to target_k using
    agglomerative clustering on facet centroids.
    """
    from dyf_rs import DensityClassifier

    n, dim = embeddings.shape

    # Global tier: use enough bits for moderate granularity
    global_bits = 8
    global_clf = DensityClassifier(embedding_dim=dim, num_bits=global_bits, seed=seed)
    global_clf.fit(embeddings)
    global_ids = np.array(global_clf.get_bucket_ids())

    bucket_to_indices = defaultdict(list)
    for idx, bid in enumerate(global_ids):
        bucket_to_indices[bid].append(idx)

    bucket_sizes = {bid: len(idxs) for bid, idxs in bucket_to_indices.items()}
    size_vals = list(bucket_sizes.values())
    dense_threshold = max(np.percentile(size_vals, 75), 10)

    # Assign fine-grained facet IDs
    facet_labels = np.full(n, -1, dtype=np.int32)
    facet_id = 0

    for bid, indices in bucket_to_indices.items():
        indices = np.array(indices)
        bucket_size = len(indices)

        if bucket_size >= dense_threshold and bucket_size >= 10:
            # Facet this bucket
            bucket_emb = embeddings[indices]
            bits = 4 if bucket_size < 50 else (6 if bucket_size < 200 else 8)
            try:
                facet_clf = DensityClassifier(
                    embedding_dim=dim, num_bits=bits, seed=seed)
                facet_clf.fit(bucket_emb)
                local_ids = np.array(facet_clf.get_bucket_ids())
                local_unique = set(local_ids.tolist())
                local_remap = {old: facet_id + i
                               for i, old in enumerate(sorted(local_unique))}
                for li, gi in enumerate(indices):
                    facet_labels[gi] = local_remap[local_ids[li]]
                facet_id += len(local_unique)
            except Exception:
                # Fallback: whole bucket is one facet
                for gi in indices:
                    facet_labels[gi] = facet_id
                facet_id += 1
        else:
            # Small bucket: one facet
            for gi in indices:
                facet_labels[gi] = facet_id
            facet_id += 1

    n_facets = len(set(facet_labels.tolist()))
    print(f"  {n_facets} facets from {len(bucket_to_indices)} global buckets")

    if n_facets <= target_k:
        # Already fewer than target — just relabel contiguously
        unique = sorted(set(facet_labels.tolist()))
        remap = {old: new for new, old in enumerate(unique)}
        return np.array([remap[f] for f in facet_labels])

    # Merge facets to target_k using agglomerative clustering on centroids
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    emb_n = embeddings / np.maximum(norms, 1e-10)

    facet_unique = sorted(set(facet_labels.tolist()))
    centroids = np.zeros((len(facet_unique), dim), dtype=np.float32)
    facet_remap = {old: new for new, old in enumerate(facet_unique)}

    for old_id, new_id in facet_remap.items():
        mask = facet_labels == old_id
        cent = emb_n[mask].mean(axis=0)
        cent /= max(np.linalg.norm(cent), 1e-10)
        centroids[new_id] = cent

    # Cosine distance for agglomerative clustering
    agg = AgglomerativeClustering(
        n_clusters=target_k, metric='cosine', linkage='average')
    facet_to_cluster = agg.fit_predict(centroids)

    # Map points: facet_label -> facet_remap -> facet_to_cluster
    labels = np.array([facet_to_cluster[facet_remap[f]] for f in facet_labels])
    n_final = len(set(labels.tolist()))
    print(f"  Merged to {n_final} clusters")
    return labels


def evaluate(name, coords, labels, embeddings):
    """Compute metrics for a clustering result projected onto 2D coords."""
    n = len(labels)
    unique_labels = sorted(set(labels))
    n_clusters = len(unique_labels)

    # Normalize embeddings
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    emb_n = embeddings / np.maximum(norms, 1e-10)

    # 1. Silhouette (2D)
    silh = silhouette_score(coords, labels, sample_size=min(5000, n))

    # 2. kNN purity (2D, k=15)
    k_purity = 15
    nn = NearestNeighbors(n_neighbors=k_purity + 1)
    nn.fit(coords)
    _, indices = nn.kneighbors(coords)
    purities = []
    for i in range(n):
        neighbors = indices[i, 1:]
        same = np.sum(labels[neighbors] == labels[i])
        purities.append(same / k_purity)
    mean_purity = np.mean(purities)

    # 3. Intra-cluster cosine similarity (high-D)
    rng = np.random.default_rng(42)
    intra_sims = []
    for c in unique_labels:
        mask = labels == c
        cluster_embs = emb_n[mask]
        nc = cluster_embs.shape[0]
        if nc < 2:
            continue
        if nc > 200:
            idx = rng.choice(nc, 200, replace=False)
            cluster_embs = cluster_embs[idx]
            nc = 200
        sim_matrix = cluster_embs @ cluster_embs.T
        triu = np.triu(np.ones((nc, nc), dtype=bool), k=1)
        intra_sims.append(float(sim_matrix[triu].mean()))
    mean_intra = np.mean(intra_sims)

    # 4. Inter-cluster cosine similarity (high-D centroids)
    centroids = []
    for c in unique_labels:
        cent = emb_n[labels == c].mean(axis=0)
        cent /= max(np.linalg.norm(cent), 1e-10)
        centroids.append(cent)
    centroids = np.array(centroids)
    inter_matrix = centroids @ centroids.T
    nc = len(centroids)
    inter_triu = np.triu(np.ones((nc, nc), dtype=bool), k=1)
    mean_inter = float(inter_matrix[inter_triu].mean())

    # 5. Cluster fragmentation (in 2D)
    nn1 = NearestNeighbors(n_neighbors=2)
    nn1.fit(coords)
    dists1, _ = nn1.kneighbors(coords)
    threshold = np.median(dists1[:, 1]) * 3

    frag_counts = []
    for c in unique_labels:
        mask = labels == c
        cluster_coords = coords[mask]
        nc_pts = cluster_coords.shape[0]
        if nc_pts < 2:
            frag_counts.append(1)
            continue
        nn_c = NearestNeighbors(radius=threshold)
        nn_c.fit(cluster_coords)
        adj = nn_c.radius_neighbors_graph(cluster_coords, mode='connectivity')
        n_comp, _ = connected_components(adj, directed=False)
        frag_counts.append(n_comp)

    mean_frag = np.mean(frag_counts)
    pct_single = 100 * sum(1 for f in frag_counts if f == 1) / len(frag_counts)
    total_frag = sum(frag_counts)

    # Print
    print(f"\n  {name} ({n_clusters} clusters):")
    print(f"    Silhouette (2D):       {silh:>7.3f}")
    print(f"    kNN purity (k=15):     {mean_purity:>7.3f}")
    print(f"    Intra-cluster sim:     {mean_intra:>7.3f}  (higher = tighter topics)")
    print(f"    Inter-cluster sim:     {mean_inter:>7.3f}  (lower = more distinct)")
    print(f"    Sim gap:               {mean_intra - mean_inter:>7.3f}  (positive = good)")
    print(f"    Fragmentation:         {mean_frag:>7.1f} avg  ({pct_single:.0f}% single-blob)")
    print(f"    Total fragments:       {total_frag:>5d}  (ideal: {n_clusters})")

    return {
        'silh': silh, 'purity': mean_purity,
        'intra': mean_intra, 'inter': mean_inter,
        'gap': mean_intra - mean_inter,
        'frag': mean_frag, 'total_frag': total_frag,
        'pct_single': pct_single,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("parquet_path")
    parser.add_argument("--sample", type=int, default=8000)
    parser.add_argument("--n-clusters", type=int, default=25)
    parser.add_argument("--max-depth", type=int, default=12,
                        help="PCA tree depth")
    args = parser.parse_args()

    titles, embeddings = load_and_dedup(args.parquet_path, args.sample)
    n = len(titles)

    # DYF-parameterized UMAP (shared 2D projection for all methods)
    dyf_k = suggest_n_neighbors(embeddings)
    print(f"\n=== DYF UMAP (n_neighbors={dyf_k}) ===")
    t0 = time.time()
    coords = run_umap(embeddings, n_neighbors=dyf_k)
    t_umap = time.time() - t0
    print(f"  UMAP done in {t_umap:.1f}s")

    target_k = args.n_clusters
    results = {}

    # Method 1: BIRCH on 2D coords
    print(f"\n=== BIRCH on 2D coords (target_k={target_k}) ===")
    t0 = time.time()
    birch = fit_birch(coords, target_k)
    labels_birch = birch.predict(coords)
    t_birch = time.time() - t0
    print(f"  {len(set(labels_birch))} clusters in {t_birch:.2f}s")

    # Method 2: PCA tree on high-D embeddings
    print(f"\n=== PCA tree on high-D (depth={args.max_depth}, "
          f"target_k={target_k}) ===")
    t0 = time.time()
    tree = build_pca_tree(embeddings, args.max_depth)
    labels_pca = cut_tree_to_labels(tree, args.max_depth, n, target_k)
    t_pca = time.time() - t0
    print(f"  {len(set(labels_pca))} clusters in {t_pca:.2f}s")

    # Method 3: DYF single-level LSH
    print(f"\n=== DYF single-level LSH (target_k~{target_k}) ===")
    t0 = time.time()
    labels_dyf1 = dyf_single_level(embeddings, target_k)
    t_dyf1 = time.time() - t0
    print(f"  {t_dyf1:.2f}s")

    # Method 4: DYF hierarchical (two-tier + merge)
    print(f"\n=== DYF hierarchical (target_k={target_k}) ===")
    t0 = time.time()
    labels_dyf2 = dyf_hierarchical(embeddings, target_k)
    t_dyf2 = time.time() - t0
    print(f"  {t_dyf2:.2f}s")

    # Method 5: DYF tree (recursive k-ary LSH + agglomerative merge)
    dyf_tree_bits = 3
    dyf_tree_depth = 4
    print(f"\n=== DYF tree (depth={dyf_tree_depth}, bits={dyf_tree_bits}, "
          f"target_k={target_k}) ===")
    t0 = time.time()
    dtree = build_dyf_tree(embeddings, max_depth=dyf_tree_depth,
                           num_bits=dyf_tree_bits, min_leaf_size=4)
    labels_dtree = cut_dyf_tree_to_labels(dtree, n, target_k, embeddings)
    t_dtree = time.time() - t0
    n_dtree = len(set(labels_dtree.tolist()))
    print(f"  {n_dtree} clusters in {t_dtree:.2f}s")

    # Evaluate all
    print(f"\n{'=' * 70}")
    print(f"  Clustering Comparison — {n} points, target_k={target_k}")
    print(f"{'=' * 70}")

    results['BIRCH-2D'] = evaluate(
        "BIRCH on 2D UMAP coords", coords, labels_birch, embeddings)
    results['PCA-tree'] = evaluate(
        "PCA tree on high-D embeddings", coords, labels_pca, embeddings)
    results['DYF-1lvl'] = evaluate(
        "DYF single-level LSH", coords, labels_dyf1, embeddings)
    results['DYF-hier'] = evaluate(
        "DYF hierarchical (2-tier + merge)", coords, labels_dyf2, embeddings)
    results['DYF-tree'] = evaluate(
        "DYF tree (recursive LSH + merge)", coords, labels_dtree, embeddings)

    # Summary table
    methods = ['BIRCH-2D', 'PCA-tree', 'DYF-1lvl', 'DYF-hier', 'DYF-tree']
    metrics = [('silh', 'Silhouette'), ('purity', 'kNN purity'),
               ('intra', 'Intra sim'), ('inter', 'Inter sim'),
               ('gap', 'Sim gap'), ('frag', 'Fragmentation'),
               ('pct_single', '% single-blob')]

    print(f"\n  {'Metric':<18s}", end="")
    for m in methods:
        print(f"  {m:>10s}", end="")
    print()
    print(f"  {'-'*18}", end="")
    for _ in methods:
        print(f"  {'-'*10}", end="")
    print()

    for key, label in metrics:
        print(f"  {label:<18s}", end="")
        vals = [results[m][key] for m in methods]
        # Highlight best: for sim gap, higher is better; for inter/frag, lower is better
        for v in vals:
            print(f"  {v:>10.3f}", end="")
        print()

    print()


if __name__ == "__main__":
    main()
