"""
UMAP + BIRCH comparison: standard vs DYF-parameterized

Side-by-side HTML outputs:
  1. Standard UMAP (n_neighbors=15) + BIRCH
  2. DYF-parameterized UMAP (n_neighbors from LSH density) + BIRCH

Usage:
    python demo/wiki_umap_birch.py demo/wiki_simple_50k.parquet [--sample 8000]
"""

import argparse
import colorsys
import time
from pathlib import Path

import numpy as np
import polars as pl
import plotly.graph_objects as go
import umap
from sklearn.cluster import Birch
from sklearn.neighbors import NearestNeighbors


def load_and_dedup(parquet_path, sample=None):
    """Load parquet, optionally sample, dedup via LSH."""
    print(f"Loading {parquet_path}...")
    df = pl.read_parquet(parquet_path)
    if sample and sample < len(df):
        df = df.sample(sample, seed=42)

    titles = df["title"].to_list()
    embeddings = np.array(df["embedding"].to_list(), dtype=np.float32)

    from dyf_rs import DensityClassifier
    from dyf.chunks import deduplicate_chunks

    clf = DensityClassifier(embedding_dim=embeddings.shape[1], num_bits=12, seed=42)
    clf.fit(embeddings)
    bucket_ids = clf.get_bucket_ids()
    dedup_mask = deduplicate_chunks(bucket_ids, np.asarray(titles))

    n_before = len(titles)
    titles = [t for t, keep in zip(titles, dedup_mask) if keep]
    embeddings = embeddings[dedup_mask]
    print(f"  {n_before} -> {len(titles)} after dedup")
    return titles, embeddings


def suggest_n_neighbors(embeddings, num_bits=12, min_k=15, max_k=100):
    """Use DYF LSH bucket density to suggest UMAP n_neighbors."""
    from dyf_rs import DensityClassifier

    clf = DensityClassifier(embedding_dim=embeddings.shape[1], num_bits=num_bits, seed=42)
    clf.fit(embeddings)
    bucket_sizes = clf.get_bucket_sizes()
    mean_size = bucket_sizes.mean()
    suggested = int(np.clip(mean_size, min_k, max_k))
    n_buckets = len(set(clf.get_bucket_ids()))
    print(f"  DYF: {n_buckets} buckets, mean_size={mean_size:.0f}, "
          f"suggested n_neighbors={suggested}")
    return suggested


def run_umap(embeddings, n_neighbors=15):
    """Run UMAP and return normalized 2D coords."""
    print(f"  Running UMAP (n_neighbors={n_neighbors})...")
    t0 = time.time()
    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=n_neighbors,
        min_dist=0.1,
        n_jobs=-1,
        verbose=False,
    )
    coords = np.asarray(reducer.fit_transform(embeddings))

    # Fix NaN coords
    nan_mask = np.isnan(coords).any(axis=1)
    if nan_mask.any():
        print(f"    Replacing {nan_mask.sum()} NaN coords")
        nn = NearestNeighbors(n_neighbors=1, metric='cosine')
        nn.fit(embeddings[~nan_mask])
        _, idx = nn.kneighbors(embeddings[nan_mask])
        coords[nan_mask] = coords[~nan_mask][idx.ravel()]

    # Normalize: median-center + MAD scaling
    median = np.nanmedian(coords, axis=0)
    mad = np.nanmedian(np.abs(coords - median), axis=0)
    scale = float(np.fmax(np.nanmax(mad), 1e-8))
    coords = (coords - median) / scale
    print(f"    Done in {time.time() - t0:.1f}s")
    return coords


def fit_birch(data, target_k, max_iters=10):
    """Fit BIRCH with binary search on threshold to get ~target_k subclusters."""
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


def golden_ratio_colors(labels):
    """Generate hierarchical colors using golden ratio hue spacing."""
    unique = sorted(set(labels))
    n = len(unique)
    hues = [(i * 0.618033988749895) % 1.0 for i in range(n)]
    label_to_hue = {lbl: hues[i] for i, lbl in enumerate(unique)}

    colors = []
    for lbl in labels:
        hue = label_to_hue[lbl]
        r, g, b = colorsys.hls_to_rgb(hue, 0.45, 0.6)
        colors.append(f"#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}")
    return colors


def build_figure(coords, titles, labels, colors, title_str):
    """Build dark-theme Plotly scatter figure."""
    n_clusters = len(set(labels))

    # Compute cluster centroids and names
    centroids = {}
    names = {}
    for cid in set(labels):
        mask = labels == cid
        pts = np.where(mask)[0]
        centroid = coords[pts].mean(axis=0)
        centroids[cid] = centroid
        dists = np.linalg.norm(coords[pts] - centroid, axis=1)
        names[cid] = titles[pts[np.argmin(dists)]]

    fig = go.Figure()
    fig.add_trace(go.Scattergl(
        x=coords[:, 0], y=coords[:, 1],
        mode="markers",
        marker=dict(size=4, color=colors, opacity=0.7),
        text=[f"{titles[i]}<br>Cluster {labels[i]}" for i in range(len(titles))],
        hoverinfo="text",
        showlegend=False,
    ))

    for cid, name in names.items():
        cx, cy = centroids[cid]
        fig.add_annotation(
            x=cx, y=cy, text=name[:25], showarrow=False,
            font=dict(size=9, color="white", family="Arial Black"),
            bgcolor="rgba(40,40,40,0.85)",
            bordercolor="rgba(100,100,100,0.6)",
            borderwidth=1, borderpad=3,
        )

    fig.update_layout(
        title=title_str,
        plot_bgcolor="#2a2a2a", paper_bgcolor="#2a2a2a",
        font=dict(color="white"),
        xaxis=dict(showgrid=False, showticklabels=False, zeroline=False),
        yaxis=dict(showgrid=False, showticklabels=False, zeroline=False,
                   scaleanchor="x", scaleratio=1),
        dragmode="pan",
        margin=dict(l=20, r=20, t=60, b=20),
        width=1400, height=1000,
    )
    return fig


def main():
    parser = argparse.ArgumentParser(
        description="Compare standard vs DYF-parameterized UMAP + BIRCH")
    parser.add_argument("parquet_path", help="Path to embeddings parquet")
    parser.add_argument("--sample", type=int, default=8000)
    parser.add_argument("--n-clusters", type=int, default=25,
                        help="Target number of BIRCH clusters")
    args = parser.parse_args()

    outdir = Path(args.parquet_path).parent
    titles, embeddings = load_and_dedup(args.parquet_path, args.sample)
    n = len(titles)
    titles_arr = np.array(titles)

    # ── Standard UMAP (n_neighbors=15) ────────────────────────────────────
    print("\n=== Standard UMAP (n_neighbors=15) ===")
    coords_std = run_umap(embeddings, n_neighbors=15)

    # ── DYF-parameterized UMAP ────────────────────────────────────────────
    print("\n=== DYF-parameterized UMAP ===")
    dyf_k = suggest_n_neighbors(embeddings)
    coords_dyf = run_umap(embeddings, n_neighbors=dyf_k)

    # ── BIRCH clustering (on 2D coords, same target_k for both) ──────────
    target_k = args.n_clusters
    print(f"\nFitting BIRCH (target_k={target_k}) on both projections...")

    birch_std = fit_birch(coords_std, target_k)
    labels_std = birch_std.predict(coords_std)
    n_std = len(set(labels_std))
    print(f"  Standard: {n_std} clusters")

    birch_dyf = fit_birch(coords_dyf, target_k)
    labels_dyf = birch_dyf.predict(coords_dyf)
    n_dyf = len(set(labels_dyf))
    print(f"  DYF: {n_dyf} clusters")

    # ── Build and save figures ────────────────────────────────────────────
    colors_std = golden_ratio_colors(labels_std)
    colors_dyf = golden_ratio_colors(labels_dyf)

    fig_std = build_figure(
        coords_std, titles_arr, labels_std, colors_std,
        f"Standard UMAP (k=15) + BIRCH — {n_std} clusters, {n} points",
    )
    fig_dyf = build_figure(
        coords_dyf, titles_arr, labels_dyf, colors_dyf,
        f"DYF UMAP (k={dyf_k}) + BIRCH — {n_dyf} clusters, {n} points",
    )

    path_std = str(outdir / "rog_2d_standard_birch.html")
    path_dyf = str(outdir / "rog_2d_dyf_birch.html")

    fig_std.write_html(path_std, config={"scrollZoom": True})
    fig_dyf.write_html(path_dyf, config={"scrollZoom": True})
    print(f"\nWrote {path_std}")
    print(f"Wrote {path_dyf}")

    import subprocess
    subprocess.run(["open", path_std])
    subprocess.run(["open", path_dyf])


if __name__ == "__main__":
    main()
