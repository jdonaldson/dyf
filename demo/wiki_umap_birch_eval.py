"""
Evaluate UMAP + BIRCH: standard vs DYF-parameterized

Metrics:
  - Silhouette (2D): spatial cluster coherence
  - kNN purity (2D): fraction of spatial neighbors sharing cluster label
  - Intra-cluster cosine sim (high-D): do clusters correspond to real topics?
  - Inter-cluster cosine sim (high-D): are clusters semantically distinct?
  - Cluster fragmentation: how many spatial blobs per cluster?

Usage:
    python demo/wiki_umap_birch_eval.py demo/wiki_simple_50k.parquet [--sample 8000]
"""

import argparse
import time
from pathlib import Path

import numpy as np
import polars as pl
import umap
from sklearn.cluster import Birch
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors
from scipy.sparse.csgraph import connected_components


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
    bucket_ids = clf.get_bucket_ids()
    dedup_mask = deduplicate_chunks(bucket_ids, np.asarray(titles))

    titles = [t for t, keep in zip(titles, dedup_mask) if keep]
    embeddings = embeddings[dedup_mask]
    print(f"Loaded {len(titles)} points after dedup")
    return titles, embeddings


def suggest_n_neighbors(embeddings):
    from dyf_rs import DensityClassifier
    clf = DensityClassifier(embedding_dim=embeddings.shape[1], num_bits=12, seed=42)
    clf.fit(embeddings)
    mean_size = clf.get_bucket_sizes().mean()
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


def evaluate(name, coords, labels, embeddings):
    """Compute metrics for a UMAP + clustering result."""
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

    # 5. Cluster fragmentation
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
    args = parser.parse_args()

    titles, embeddings = load_and_dedup(args.parquet_path, args.sample)

    # Standard UMAP
    print("\n=== Standard UMAP (k=15) ===")
    t0 = time.time()
    coords_std = run_umap(embeddings, n_neighbors=15)
    t_std = time.time() - t0

    # DYF UMAP
    print("\n=== DYF UMAP ===")
    dyf_k = suggest_n_neighbors(embeddings)
    print(f"  Suggested n_neighbors={dyf_k}")
    t0 = time.time()
    coords_dyf = run_umap(embeddings, n_neighbors=dyf_k)
    t_dyf = time.time() - t0

    # BIRCH on both
    target_k = args.n_clusters

    birch_std = fit_birch(coords_std, target_k)
    labels_std = birch_std.predict(coords_std)

    birch_dyf = fit_birch(coords_dyf, target_k)
    labels_dyf = birch_dyf.predict(coords_dyf)

    print(f"\n{'=' * 60}")
    print(f"  UMAP + BIRCH Comparison ({len(titles)} points)")
    print(f"{'=' * 60}")
    print(f"  Standard: k=15, {t_std:.1f}s")
    print(f"  DYF:      k={dyf_k}, {t_dyf:.1f}s")

    r_std = evaluate("Standard UMAP + BIRCH", coords_std, labels_std, embeddings)
    r_dyf = evaluate(f"DYF UMAP (k={dyf_k}) + BIRCH", coords_dyf, labels_dyf, embeddings)

    # Delta summary
    print(f"\n  {'Metric':<25s}  {'Standard':>8s}  {'DYF':>8s}  {'Delta':>8s}")
    print(f"  {'-'*25}  {'-'*8}  {'-'*8}  {'-'*8}")
    for key, label in [('silh', 'Silhouette'),
                       ('purity', 'kNN purity'),
                       ('intra', 'Intra-cluster sim'),
                       ('inter', 'Inter-cluster sim'),
                       ('gap', 'Sim gap'),
                       ('frag', 'Fragmentation'),
                       ('pct_single', '% single-blob')]:
        s, d = r_std[key], r_dyf[key]
        delta = d - s
        sign = "+" if delta > 0 else ""
        print(f"  {label:<25s}  {s:>8.3f}  {d:>8.3f}  {sign}{delta:>7.3f}")

    print()


if __name__ == "__main__":
    main()
