"""
BIRCH (2D) vs DYF tree (high-D) clustering comparison.

Both visualizations share the same DYF-parameterized UMAP 2D projection.
BIRCH clusters on the 2D coordinates; DYF tree clusters on the original
high-dimensional embeddings, then colors are mapped onto the shared layout.

Renders standalone HTML files using scatter-gl (Three.js) with
zoom-dependent cluster labels.

Usage:
    python demo/wiki_clustering_viz.py demo/wiki_simple_50k.parquet [--sample 8000]
"""

import argparse
import colorsys
import json
import math
import re
import subprocess
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import polars as pl
import umap
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.ndimage import gaussian_filter
from sklearn.cluster import Birch
from sklearn.neighbors import NearestNeighbors

from dyf.dyf_tree import (
    build_dyf_tree,
    refine_dyf_tree,
    cut_dyf_tree_to_labels,
    refine_clusters,
)


# ── Helpers ──────────────────────────────────────────────────────────────


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
    bucket_ids = np.asarray(clf.get_bucket_ids())
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
    bucket_sizes = np.array(clf.get_bucket_sizes())
    mean_size = bucket_sizes.mean()
    suggested = int(np.clip(mean_size, min_k, max_k))
    n_buckets = len(set(clf.get_bucket_ids()))
    print(f"  DYF: {n_buckets} buckets, mean_size={mean_size:.0f}, "
          f"suggested n_neighbors={suggested}")
    return suggested


def run_umap(embeddings, n_neighbors=15, n_components=2, densmap=False):
    """Run UMAP and return normalized coords."""
    label = "densMAP" if densmap else "UMAP"
    print(f"  Running {label} (n_neighbors={n_neighbors}, {n_components}D)...")
    t0 = time.time()
    reducer = umap.UMAP(
        n_components=n_components,
        n_neighbors=n_neighbors,
        min_dist=0.1,
        densmap=densmap,
        n_jobs=-1,
        verbose=False,
    )
    coords = np.asarray(reducer.fit_transform(embeddings))

    nan_mask = np.isnan(coords).any(axis=1)
    if nan_mask.any():
        print(f"    Replacing {nan_mask.sum()} NaN coords")
        nn = NearestNeighbors(n_neighbors=1, metric='cosine')
        nn.fit(embeddings[~nan_mask])
        _, idx = nn.kneighbors(embeddings[nan_mask])
        coords[nan_mask] = coords[~nan_mask][idx.ravel()]

    median = np.nanmedian(coords, axis=0)
    mad = np.nanmedian(np.abs(coords - median), axis=0)
    scale = float(np.fmax(np.nanmax(mad), 1e-8))
    coords = (coords - median) / scale
    print(f"    Done in {time.time() - t0:.1f}s")
    return coords


def orient_landscape(coords):
    """Rotate XY plane so the widest spread aligns with the X axis."""
    xy = coords[:, :2]
    cov = np.cov(xy, rowvar=False)
    theta = 0.5 * np.arctan2(2 * cov[0, 1], cov[0, 0] - cov[1, 1])
    c, s = np.cos(-theta), np.sin(-theta)
    rot = xy @ np.array([[c, s], [-s, c]])
    # Check if Y is still wider; if so, rotate 90 more
    if np.ptp(rot[:, 1]) > np.ptp(rot[:, 0]):
        c2, s2 = np.cos(np.pi / 2), np.sin(np.pi / 2)
        rot = rot @ np.array([[c2, s2], [-s2, c2]])
    out = coords.copy()
    out[:, :2] = rot
    xr = np.ptp(out[:, 0])
    yr = np.ptp(out[:, 1])
    print(f"    Landscape orient: rotated {np.degrees(theta):.1f}°, "
          f"spread X={xr:.2f} Y={yr:.2f} (ratio {xr/yr:.2f})")
    return out


# ── 3D HAMMER edge bundling ──────────────────────────────────────────


def bundle_edges_3d(node_coords, edges, n_points=20, attraction=0.7):
    """3D edge bundling via midpoint attraction with cubic spline output.

    For each edge, computes a midpoint pulled toward the centroid of all
    edge midpoints that share a node (i.e. edges from the same cluster hub
    get bundled). The pulled midpoint becomes a spline control point,
    producing smooth curves that converge at shared hubs.

    Args:
        node_coords: (N, 3) array of node positions
        edges: list of (source_idx, target_idx) tuples
        n_points: number of output points per edge path
        attraction: how strongly midpoints pull toward shared hub (0-1)

    Returns:
        list of (n_points, 3) arrays — one smooth path per edge
    """
    from scipy.interpolate import CubicSpline

    if not edges:
        return []

    coords = np.asarray(node_coords, dtype=np.float32)

    # Compute the centroid of all edge midpoints sharing each node
    # This creates natural "hubs" where edges from the same cluster converge
    node_mid_sum = defaultdict(lambda: np.zeros(3, dtype=np.float64))
    node_mid_count = defaultdict(int)
    edge_midpoints = []
    for src, dst in edges:
        mid = (coords[src] + coords[dst]) / 2
        edge_midpoints.append(mid)
        node_mid_sum[src] += mid
        node_mid_count[src] += 1
        node_mid_sum[dst] += mid
        node_mid_count[dst] += 1

    # For each node, the hub point is the average midpoint of its edges
    node_hub = {}
    for nid in node_mid_sum:
        node_hub[nid] = (node_mid_sum[nid] / node_mid_count[nid]).astype(np.float32)

    # Build smooth spline paths
    result = []
    t_ctrl = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
    t_out = np.linspace(0, 1, n_points)

    for i, (src, dst) in enumerate(edges):
        p0 = coords[src]
        p4 = coords[dst]
        mid = edge_midpoints[i]

        # Pull the midpoint toward both endpoint hubs
        hub_src = node_hub[src]
        hub_dst = node_hub[dst]
        attracted_mid = mid + attraction * ((hub_src + hub_dst) / 2 - mid)

        # Quarter points interpolate between endpoint and attracted midpoint
        q1 = p0 * 0.5 + attracted_mid * 0.5
        q3 = p4 * 0.5 + attracted_mid * 0.5

        ctrl = np.array([p0, q1, attracted_mid, q3, p4], dtype=np.float32)
        cs = CubicSpline(t_ctrl, ctrl, bc_type='clamped')
        path = cs(t_out).astype(np.float32)
        result.append(path)

    return result


def compute_bridge_edges_3d(coords, embeddings, labels, n_clusters):
    """Compute cross-cluster bridge edges using ROG ontology.

    Returns (bundled_3d, bundled_2d, pair_counts, centroids) where
    bundled_3d and bundled_2d are lists of bundled edge paths for
    3D and 2D rendering respectively.
    """
    from collections import defaultdict
    import dyf

    print("  Building ROG ontology for bridge detection...")
    result = dyf.build_rog_ontology(
        embeddings,
        initial_threshold=0.55,
        min_threshold=0.35,
        target_coverage=0.95,
        verbose=False,
    )

    ont = result.ontology
    pair_counts = defaultdict(int)

    for parent, children_list in ont.children.items():
        for child, sim, div_gap in children_list:
            c1, c2 = int(labels[parent]), int(labels[child])
            if c1 != c2:
                pair = (min(c1, c2), max(c1, c2))
                pair_counts[pair] += 1

    cross = sum(pair_counts.values())
    print(f"  Found {len(pair_counts)} cluster pairs, {cross:,} cross-cluster edges")

    # Compute cluster centroids in 3D
    centroids = np.zeros((n_clusters, coords.shape[1]), dtype=np.float32)
    for c in range(n_clusters):
        mask = labels == c
        if mask.any():
            centroids[c] = coords[mask].mean(axis=0)

    # Build edge list (deduplicated cluster pairs)
    edge_list = sorted(pair_counts.keys(), key=lambda p: -pair_counts[p])

    if not edge_list:
        return [], {}, centroids

    # 2D: datashader hammer_bundle (proven KDEEB on dense 2D grid)
    print(f"  Bundling {len(edge_list)} bridge edges in 2D (datashader)...")
    import pandas as pd
    from datashader.bundling import hammer_bundle as ds_hammer_bundle

    centroids_2d = centroids[:, :2]
    nodes_df = pd.DataFrame({
        "x": centroids_2d[:, 0].astype(float),
        "y": centroids_2d[:, 1].astype(float),
    })
    edges_df = pd.DataFrame({
        "source": [e[0] for e in edge_list],
        "target": [e[1] for e in edge_list],
    })
    bundled_df = ds_hammer_bundle(nodes_df, edges_df)

    # Parse datashader output: NaN-separated edge segments → list of paths
    bundled_2d = []
    current_path = []
    for _, row in bundled_df.iterrows():
        if pd.isna(row["x"]) or pd.isna(row["y"]):
            if current_path:
                bundled_2d.append(np.array(current_path, dtype=np.float32))
                current_path = []
        else:
            current_path.append([row["x"], row["y"], 0.0])
    if current_path:
        bundled_2d.append(np.array(current_path, dtype=np.float32))

    print(f"  Got {len(bundled_2d)} 2D bundled paths")

    return bundled_2d, pair_counts, centroids


def fit_birch(data, target_k, max_iters=10):
    """Fit BIRCH with enough subclusters, then agglomerative merge to target_k."""
    # Binary search for a threshold that produces >= target_k subclusters
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
            hi = mid  # need smaller threshold for more subclusters
        else:
            lo = mid
        if target_k <= n <= target_k * 3:
            break

    if best_birch is None:
        # Couldn't get enough subclusters — use lowest threshold result
        best_birch = Birch(n_clusters=None, threshold=lo, branching_factor=50)
        best_birch.fit(data)

    # Agglomerative merge subclusters down to exactly target_k
    n_subs = len(best_birch.subcluster_centers_)
    if n_subs > target_k:
        birch = Birch(n_clusters=target_k, threshold=best_birch.threshold,
                      branching_factor=50)
        birch.fit(data)
        return birch

    return best_birch


CLUSTER_LEVELS = (5, 12, 25)


def build_label_hierarchy(coords, titles_arr, birch, embeddings=None,
                          model=None):
    """Build multi-level label hierarchy from BIRCH subclusters via Ward linkage.

    If model is provided, generates LLM labels via contrastive TF-IDF.
    Returns dict mapping level -> list of label dicts with centroid, name, size.
    """
    ndim = coords.shape[1]
    sub_centers = birch.subcluster_centers_
    n_subs = len(sub_centers)
    titles = list(titles_arr)

    # Ward linkage on subcluster centers
    Z = linkage(sub_centers, method='ward')

    # Map each point to its subcluster
    sub_labels = birch.predict(coords)

    levels = {}
    for k in CLUSTER_LEVELS:
        if k >= n_subs:
            continue
        # Cut dendrogram at k clusters
        micro_labels = fcluster(Z, k, criterion='maxclust') - 1
        # Map points through subclusters to final labels
        point_labels = np.array([micro_labels[s] for s in sub_labels])

        # Get LLM labels if model provided
        cluster_names = None
        if model and embeddings is not None:
            print(f"  Labeling level {k}...")
            cluster_names = label_clusters(
                titles, coords, point_labels, embeddings, model=model)

        label_data = []
        for cid in sorted(set(point_labels)):
            mask = point_labels == cid
            if not mask.any():
                continue
            pts = np.where(mask)[0]
            centroid = coords[pts].mean(axis=0)
            if cluster_names and cid in cluster_names:
                name = cluster_names[cid]
            else:
                # Fallback: nearest title to centroid
                dists = np.linalg.norm(coords[pts] - centroid, axis=1)
                name = str(titles_arr[pts[np.argmin(dists)]])
            rec = {
                "x": float(centroid[0]),
                "y": float(centroid[1]),
                "z": float(centroid[2]) if ndim >= 3 else 0.0,
                "text": name[:50],
                "size": int(mask.sum()),
            }
            label_data.append(rec)
        levels[k] = label_data
        print(f"  Level {k}: {len(label_data)} labels")

    return levels


def golden_ratio_colors(labels):
    """Generate colors using golden ratio hue spacing. Returns per-label hex."""
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


def golden_ratio_color_map(labels):
    """Return dict mapping label -> hex color."""
    unique = sorted(set(labels))
    hues = [(i * 0.618033988749895) % 1.0 for i in range(len(unique))]
    cmap = {}
    for i, lbl in enumerate(unique):
        r, g, b = colorsys.hls_to_rgb(hues[i], 0.45, 0.6)
        cmap[int(lbl)] = f"#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}"
    return cmap


def golden_ratio_rgb_map(labels):
    """Return dict mapping label -> [r, g, b] (0-255)."""
    unique = sorted(set(labels))
    hues = [(i * 0.618033988749895) % 1.0 for i in range(len(unique))]
    cmap = {}
    for i, lbl in enumerate(unique):
        r, g, b = colorsys.hls_to_rgb(hues[i], 0.45, 0.6)
        cmap[int(lbl)] = [int(r * 255), int(g * 255), int(b * 255)]
    return cmap


# ── Contrastive cluster labeling ──────────────────────────────────────


def _compute_tfidf_keywords(titles, labels, n_clusters, top_k=10, min_df=1):
    """TF-IDF keywords per cluster for contrastive labeling."""
    stop_words = {
        'the', 'a', 'an', 'of', 'in', 'on', 'at', 'to', 'for', 'and', 'or',
        'is', 'was', 'are', 'were', 'be', 'been', 'by', 'with', 'from', 'as',
        'it', 'its', 'this', 'that', 'not', 'but', 'has', 'had', 'have', 'do',
        'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might',
        'list', 'disambiguation', 'episode', 'season',
    }

    def tokenize(text):
        words = re.findall(r'[a-z]+', text.lower())
        return [w for w in words if len(w) > 2 and w not in stop_words]

    cluster_titles = defaultdict(list)
    for i, label in enumerate(labels):
        cluster_titles[int(label)].append(titles[i])

    word_df = defaultdict(int)
    cluster_word_counts = {}
    for cid in range(n_clusters):
        word_counts = defaultdict(int)
        words_in_cluster = set()
        for title in cluster_titles[cid]:
            for word in tokenize(title):
                word_counts[word] += 1
                words_in_cluster.add(word)
        cluster_word_counts[cid] = word_counts
        for word in words_in_cluster:
            word_df[word] += 1

    vocab = {w for w, df in word_df.items() if min_df <= df < n_clusters}
    idf = {w: math.log((n_clusters + 1) / (word_df[w] + 1)) for w in vocab}

    cluster_keywords = {}
    for cid in range(n_clusters):
        word_counts = cluster_word_counts[cid]
        total_words = sum(word_counts.values())
        if total_words == 0:
            cluster_keywords[cid] = []
            continue
        scores = []
        for word in vocab:
            tf = word_counts.get(word, 0) / total_words
            score = tf * idf[word]
            if score > 0:
                scores.append((word, score))
        scores.sort(key=lambda x: -x[1])
        cluster_keywords[cid] = scores[:top_k]

    return cluster_keywords


def _find_nearest_cluster(cluster_id, centroids):
    """Find nearest cluster by centroid L2 distance."""
    target = centroids[cluster_id]
    min_dist = float('inf')
    nearest = 0
    for i, centroid in enumerate(centroids):
        if i != cluster_id:
            dist = np.linalg.norm(target - centroid)
            if dist < min_dist:
                min_dist = dist
                nearest = i
    return nearest


def _sample_spatial(point_indices, coords, k):
    """Farthest-point sampling in projection space for spatial coverage."""
    pts = np.array(point_indices)
    if len(pts) <= k:
        return pts.tolist()
    cluster_coords = coords[pts]
    chosen = [np.random.randint(len(pts))]
    for _ in range(k - 1):
        chosen_coords = cluster_coords[chosen]
        dists = np.min(
            np.linalg.norm(
                cluster_coords[:, None, :] - chosen_coords[None, :, :],
                axis=2,
            ),
            axis=1,
        )
        dists[chosen] = -1
        chosen.append(int(np.argmax(dists)))
    return pts[chosen].tolist()


def label_clusters(titles, coords, labels, embeddings, model="gemma2:9b",
                   n_samples=20):
    """Label clusters via contrastive TF-IDF + local Ollama LLM.

    For each cluster: spatially samples titles, computes contrastive TF-IDF
    keywords against the nearest high-D neighbor, and asks the LLM for a
    short topic label.  Runs Ollama calls in parallel.

    Returns:
        dict mapping cluster_id -> label string
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    unique_labels = sorted(set(int(l) for l in labels))
    n_clusters = len(unique_labels)
    label_arr = np.asarray(labels)

    cluster_points = defaultdict(list)
    for i, cid in enumerate(label_arr):
        cluster_points[int(cid)].append(i)

    # High-D centroids for nearest-neighbor finding
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    emb_n = embeddings / np.maximum(norms, 1e-10)
    hd_centroids = np.zeros((n_clusters, embeddings.shape[1]), dtype=np.float32)
    cid_to_idx = {cid: idx for idx, cid in enumerate(unique_labels)}
    for cid in unique_labels:
        pts = cluster_points[cid]
        cent = emb_n[pts].mean(axis=0)
        norm = np.linalg.norm(cent)
        if norm > 1e-10:
            cent /= norm
        hd_centroids[cid_to_idx[cid]] = cent

    print(f"  Labeling {n_clusters} clusters with contrastive LLM ({model})...")

    tasks = []
    for cid in unique_labels:
        pts = cluster_points[cid]
        if not pts:
            continue

        # Farthest-point spatial sampling in projection space
        sample_indices = _sample_spatial(pts, coords, n_samples * 3)
        seen = set()
        sample_titles = []
        for idx in sample_indices:
            t = titles[idx]
            if t not in seen:
                seen.add(t)
                sample_titles.append(t)
                if len(sample_titles) >= n_samples:
                    break

        # Contrastive TF-IDF against nearest high-D neighbor
        nearest_idx = _find_nearest_cluster(cid_to_idx[cid], hd_centroids)
        nearest_cid = unique_labels[nearest_idx]
        neighbor_pts = cluster_points[nearest_cid]

        kw_str = ""
        if neighbor_pts:
            combined_titles = ([titles[p] for p in pts]
                               + [titles[p] for p in neighbor_pts])
            combined_labels = np.zeros(len(pts) + len(neighbor_pts), dtype=int)
            combined_labels[len(pts):] = 1
            kw = _compute_tfidf_keywords(combined_titles, combined_labels, 2,
                                         top_k=8, min_df=1)
            keywords = [w for w, _ in kw.get(0, [])][:8]
            if keywords:
                kw_str = (f"\nDistinguishing keywords (vs neighbor): "
                          f"{', '.join(keywords)}")

        prompt = (
            f"You are labeling clusters in an embedding space. "
            f"This cluster has {len(pts)} items.\n"
            f"{kw_str}\n"
            f"Sample items from across this cluster:\n"
            + "\n".join(f"- {t}" for t in sample_titles)
            + "\n\n"
            "Give a short (2-5 word) label that DISTINGUISHES this cluster "
            "from similar ones. Use the distinguishing keywords and specific "
            "product/item names to find what makes this group unique.\n\n"
            "BAD labels (too vague): \"Medical Devices\", "
            "\"Surgical Instruments\", \"General Products\"\n"
            "GOOD labels: \"Spinal Fixation Screws\", \"Dental Crowns & Bridges\", "
            "\"Compression Stockings\", \"Hearing Aid Components\"\n\n"
            "Reply with ONLY the label, nothing else."
        )
        tasks.append((cid, prompt))

    # Parallel Ollama calls
    cluster_names = {cid: f"Cluster {cid}" for cid in unique_labels}

    def _call_ollama(task):
        cid, prompt = task
        try:
            result = subprocess.run(
                ["ollama", "run", model, prompt],
                capture_output=True, text=True, timeout=30,
            )
            label = result.stdout.strip().split('\n')[0][:50]
            label = label.strip('"\'').strip()
            return cid, label if label else f"Cluster {cid}"
        except Exception:
            return cid, f"Cluster {cid}"

    n_workers = min(4, len(tasks))
    completed = 0
    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        futures = {executor.submit(_call_ollama, t): t for t in tasks}
        for future in as_completed(futures):
            cid, label = future.result()
            cluster_names[cid] = label
            completed += 1
            if completed % 5 == 0 or completed == len(tasks):
                print(f"    Labeled {completed}/{len(tasks)} clusters...",
                      flush=True)

    for cid in unique_labels:
        n_pts = len(cluster_points[cid])
        print(f"    [{cid:2d}] {cluster_names[cid]:<35s} ({n_pts} pts)")

    return cluster_names


# ── Scatter-GL normalization (mirrors generatePointPositionArray) ────────

SCATTER_PLOT_CUBE_LENGTH = 2


def scatter_gl_normalize(coords):
    """Replicate scatter-gl's point normalization. Returns normalized coords."""
    x_ext = [float(coords[:, 0].min()), float(coords[:, 0].max())]
    y_ext = [float(coords[:, 1].min()), float(coords[:, 1].max())]
    x_range = x_ext[1] - x_ext[0]
    y_range = y_ext[1] - y_ext[0]
    max_range = max(x_range, y_range)
    half = SCATTER_PLOT_CUBE_LENGTH / 2

    def scale_linear(val, extent, scale):
        t = (val - extent[0]) / (extent[1] - extent[0]) if extent[1] != extent[0] else 0.5
        return scale[0] + t * (scale[1] - scale[0])

    x_scale = [-half * x_range / max_range, half * x_range / max_range]
    y_scale = [-half * y_range / max_range, half * y_range / max_range]

    norm = np.zeros_like(coords)
    for i in range(len(coords)):
        norm[i, 0] = scale_linear(coords[i, 0], x_ext, x_scale)
        norm[i, 1] = scale_linear(coords[i, 1], y_ext, y_scale)
    return norm


# ── HTML template ────────────────────────────────────────────────────────

HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>%(title)s</title>
<style>
* { margin: 0; padding: 0; box-sizing: border-box; }
html, body { width: 100%%; height: 100%%; overflow: hidden; background: #1e1e1e; }
#container { width: 100%%; height: 100%%; position: relative; }
#header {
  position: absolute; top: 0; left: 0; right: 0; z-index: 10;
  padding: 12px 20px; background: rgba(30,30,30,0.9);
  border-bottom: 1px solid #444; color: #ddd;
  font: 14px -apple-system, 'Segoe UI', sans-serif;
}
#header h1 { font-size: 16px; font-weight: 600; margin-bottom: 2px; }
#header .sub { font-size: 12px; color: #999; }
.cluster-label {
  position: absolute; pointer-events: none; z-index: 5;
  color: #fff; font: bold 10px -apple-system, 'Segoe UI', sans-serif;
  background: rgba(30,30,30,0.88); border: 1px solid rgba(120,120,120,0.5);
  padding: 2px 5px; border-radius: 3px;
  white-space: nowrap; transform: translate(-50%%, -50%%);
  transition: opacity 0.12s;
  text-shadow: 0 1px 2px rgba(0,0,0,0.6);
}
</style>
</head>
<body>
<div id="header">
  <h1>%(title)s</h1>
  <div class="sub">Scroll to zoom &middot; Drag to pan &middot; Hover for details</div>
</div>
<div id="container"></div>

<script src="https://cdn.jsdelivr.net/npm/three@0.125.0/build/three.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/scatter-gl@0.0.13/lib/scatter-gl.min.js"></script>
<script>
(function() {
  // ── Embedded data ─────────────────────────────────────────────────────
  const POINTS = %(points_json)s;
  const META   = %(meta_json)s;
  const COLORS = %(colors_json)s;
  const CENTROIDS = %(centroids_json)s;

  // ── Build dataset ─────────────────────────────────────────────────────
  const dataPoints = POINTS.map(p => [p[0], p[1]]);
  const metadata = META.map(m => ({ label: m.title + ' [cluster ' + m.cid + ']' }));
  const dataset = new ScatterGL.Dataset(dataPoints, metadata);

  // ── Create scatter-gl ─────────────────────────────────────────────────
  const container = document.getElementById('container');
  const scatterGL = new ScatterGL(container, {
    rotateOnStart: false,
    selectEnabled: false,
    showLabelsOnHover: true,
    renderMode: 'POINT',
    styles: {
      backgroundColor: 0x1e1e1e,
      fog: { enabled: false },
      point: {
        colorNoSelection: 'rgba(150,150,150,0.5)',
        scaleDefault: 1.0,
        scaleHover: 2.0,
      },
      label: {
        fillColorHover: '#ffffff',
        strokeColorHover: '#1e1e1e',
        strokeWidth: 3,
        fillWidth: 6,
        fontSize: 12,
      },
    },
    pointColorer: function(i, selectedIndices, hoverIndex) {
      if (hoverIndex === i) return '#ffffff';
      return COLORS[i];
    },
  });
  scatterGL.render(dataset);
  scatterGL.setDimensions(2);
  scatterGL.setPanMode();

  // Increase zoom speed (default is too sluggish with orthographic camera)
  (function tweakZoom() {
    try {
      var ctrl = scatterGL.scatterPlot.orbitCameraControls;
      if (ctrl) { ctrl.zoomSpeed = 3.0; }
      else { setTimeout(tweakZoom, 100); }
    } catch(e) { setTimeout(tweakZoom, 100); }
  })();

  // ── Zoom-dependent label overlay ──────────────────────────────────────
  // Create label DOM elements
  const labelEls = CENTROIDS.map(function(c) {
    var el = document.createElement('div');
    el.className = 'cluster-label';
    el.textContent = c.label;
    el.style.borderLeftColor = c.color;
    el.style.borderLeftWidth = '3px';
    container.appendChild(el);
    return el;
  });

  // Access scatter-gl internal camera (TS private = JS public at runtime)
  function getCamera() {
    try { return scatterGL.scatterPlot.camera; }
    catch(e) { return null; }
  }

  // Project world point to screen coords using camera matrices
  function projectToScreen(wx, wy, cam, w, h) {
    // matrixWorldInverse: world -> camera space
    var m = cam.matrixWorldInverse.elements;
    var vx = m[0]*wx + m[4]*wy + m[8]*0 + m[12];
    var vy = m[1]*wx + m[5]*wy + m[9]*0 + m[13];
    var vz = m[2]*wx + m[6]*wy + m[10]*0 + m[14];
    // projectionMatrix: camera -> clip space
    var p = cam.projectionMatrix.elements;
    var cx = p[0]*vx + p[4]*vy + p[8]*vz + p[12];
    var cy = p[1]*vx + p[5]*vy + p[9]*vz + p[13];
    var cw = p[3]*vx + p[7]*vy + p[11]*vz + p[15];
    // NDC to screen
    var ndcX = cx / cw;
    var ndcY = cy / cw;
    return {
      x: (ndcX + 1) / 2 * w,
      y: (1 - ndcY) / 2 * h,
      visible: ndcX >= -1.2 && ndcX <= 1.2 && ndcY >= -1.2 && ndcY <= 1.2
    };
  }

  // Compute visible world-space area for label density control
  function getVisibleArea(cam) {
    // For ortho camera: area = (right-left)/zoom * (top-bottom)/zoom
    // Default frustum half-extent is 1.2, so full range = 2.4
    var zoom = cam.zoom || 1;
    var ar = container.clientWidth / container.clientHeight;
    var w, h;
    if (ar > 1) { w = 2.4 * ar / zoom; h = 2.4 / zoom; }
    else { w = 2.4 / zoom; h = 2.4 / ar / zoom; }
    return w * h;
  }

  // How many labels to show at a given area
  // Default view area ~ 2.4*ar * 2.4 ~ 7-10 for a wide screen
  var defaultArea = 0;
  var MAX_LABELS = CENTROIDS.length;
  function maxLabelsForArea(area) {
    if (defaultArea === 0) return MAX_LABELS;
    var ratio = defaultArea / Math.max(area, 0.01);
    // ratio=1 at default zoom, >1 when zoomed in
    if (ratio < 0.8) return 0;              // zoomed out beyond default
    if (ratio < 1.5) return Math.min(8, MAX_LABELS);  // near default
    if (ratio < 3)   return Math.min(15, MAX_LABELS); // moderate zoom
    return MAX_LABELS;                       // zoomed in
  }

  // Label update loop
  var frameId = 0;
  function updateLabels() {
    frameId = requestAnimationFrame(updateLabels);
    var cam = getCamera();
    if (!cam) return;

    if (defaultArea === 0) defaultArea = getVisibleArea(cam);

    var w = container.clientWidth;
    var h = container.clientHeight;
    var area = getVisibleArea(cam);
    var maxShow = maxLabelsForArea(area);

    // CENTROIDS are sorted by size descending in Python
    for (var i = 0; i < CENTROIDS.length; i++) {
      var c = CENTROIDS[i];
      var el = labelEls[i];
      if (i >= maxShow) {
        el.style.opacity = '0';
        continue;
      }
      var p = projectToScreen(c.nx, c.ny, cam, w, h);
      if (!p.visible) {
        el.style.opacity = '0';
        continue;
      }
      el.style.left = p.x + 'px';
      el.style.top = p.y + 'px';
      el.style.opacity = '1';
    }
  }

  // Start label loop after first render settles
  setTimeout(function() { updateLabels(); }, 200);

  // Handle window resize
  window.addEventListener('resize', function() { scatterGL.resize(); });
})();
</script>
</body>
</html>
"""


def build_html(coords, titles_arr, labels, color_map, title_str,
               cluster_names=None):
    """Build scatter-gl HTML string with zoom-dependent labels."""
    labels_list = labels.tolist() if hasattr(labels, 'tolist') else list(labels)

    # Scatter-gl normalized coordinates (mirrors generatePointPositionArray)
    norm_coords = scatter_gl_normalize(coords)

    # Points as [x, y] in original data space (scatter-gl normalizes internally)
    points = [[float(coords[i, 0]), float(coords[i, 1])] for i in range(len(coords))]

    # Per-point metadata and colors
    meta = [{"title": str(titles_arr[i]), "cid": int(labels_list[i])}
            for i in range(len(titles_arr))]
    colors = [color_map[int(labels_list[i])] for i in range(len(labels_list))]

    # Cluster centroids (in scatter-gl normalized space)
    centroids = []
    for cid in sorted(set(labels_list)):
        mask = np.array(labels_list) == cid
        pts = np.where(mask)[0]
        centroid = coords[pts].mean(axis=0)
        norm_centroid = norm_coords[pts].mean(axis=0)
        dists = np.linalg.norm(coords[pts] - centroid, axis=1)
        if cluster_names and cid in cluster_names:
            name = cluster_names[cid]
        else:
            name = str(titles_arr[pts[np.argmin(dists)]])
        centroids.append({
            "nx": float(norm_centroid[0]),
            "ny": float(norm_centroid[1]),
            "label": name[:45],
            "cluster": int(cid),
            "size": int(mask.sum()),
            "color": color_map[int(cid)],
        })

    # Sort by size descending (largest clusters get label priority)
    centroids.sort(key=lambda c: -c["size"])

    return HTML_TEMPLATE % {
        "title": title_str,
        "points_json": json.dumps(points),
        "meta_json": json.dumps(meta),
        "colors_json": json.dumps(colors),
        "centroids_json": json.dumps(centroids),
    }


# ── Pydeck builder ───────────────────────────────────────────────────────


def build_pydeck(coords, titles_arr, labels, rgb_map, title_str, out_path,
                 cluster_names=None, ws_port=8766, label_levels=None,
                 bundled_edges=None):
    """Build a pydeck 3D point cloud with HTML overlay labels."""
    import pydeck as pdk

    labels_list = labels.tolist() if hasattr(labels, 'tolist') else list(labels)
    ndim = coords.shape[1]

    # Point data
    point_data = []
    for i in range(len(titles_arr)):
        cid = int(labels_list[i])
        rgb = rgb_map[cid]
        rec = {
            "x": float(coords[i, 0]),
            "y": float(coords[i, 1]),
            "z": float(coords[i, 2]) if ndim >= 3 else 0.0,
            "r": rgb[0], "g": rgb[1], "b": rgb[2],
            "title": str(titles_arr[i]),
            "cluster": cid,
        }
        point_data.append(rec)

    # Centroid labels — single level fallback
    label_data = []
    for cid in sorted(set(labels_list)):
        mask = np.array(labels_list) == cid
        pts = np.where(mask)[0]
        centroid = coords[pts].mean(axis=0)
        dists = np.linalg.norm(coords[pts] - centroid, axis=1)
        if cluster_names and cid in cluster_names:
            name = cluster_names[cid]
        else:
            name = str(titles_arr[pts[np.argmin(dists)]])
        rgb = rgb_map[int(cid)]
        rec = {
            "x": float(centroid[0]),
            "y": float(centroid[1]),
            "z": float(centroid[2]) if ndim >= 3 else 0.0,
            "text": name[:45],
            "size": int(mask.sum()),
            "r": rgb[0], "g": rgb[1], "b": rgb[2],
        }
        label_data.append(rec)

    # Multi-level label hierarchy (if provided, use it; otherwise single-level)
    if label_levels:
        levels_data = label_levels
    else:
        # Wrap single-level as the only level
        n_clusters = len(label_data)
        levels_data = {n_clusters: label_data}

    # Data is median-centered by run_umap, so origin is the natural center
    target = [0, 0, 0]

    # Add default full-opacity alpha
    for rec in point_data:
        rec["a"] = 255

    point_layer = pdk.Layer(
        "PointCloudLayer",
        data=point_data,
        get_position=["x", "y", "z"],
        get_color=["r", "g", "b", "a"],
        get_normal=[0, 0, 1],
        point_size=2,
        pickable=True,
        auto_highlight=True,
    )

    layers = [point_layer]

    # Add empty edge layer (populated in 2D mode by JS rebuildLayer)
    if bundled_edges:
        edge_layer = pdk.Layer(
            "PathLayer",
            data=[],
            get_path="path",
            get_color="color",
            width_min_pixels=1,
            width_max_pixels=2,
            pickable=False,
        )
        layers.append(edge_layer)

    # TextLayer + OrbitView has a known sizing bug (deck.gl #6808).
    # We inject HTML overlay labels instead — see reset_script below.

    view_state = pdk.ViewState(
        target=target,
        controller=True,
        rotation_x=15,
        rotation_orbit=30,
        zoom=5.5,
    )

    view = pdk.View(type="OrbitView", controller=True)

    deck = pdk.Deck(
        layers=layers,
        initial_view_state=view_state,
        views=[view],
        tooltip={"text": "{title}\nCluster {cluster}"},
    )

    deck.to_html(out_path, css_background_color="#1e1e1e", open_browser=False)

    # Inject viewport meta, title overlay, and label system into pydeck HTML
    html = Path(out_path).read_text()
    html = html.replace(
        '<head>',
        '<head><meta name="viewport" content="width=device-width, initial-scale=1">',
        1,
    )
    label_json = json.dumps(label_data)
    # Multi-level labels: keys are level numbers (as strings in JSON)
    levels_json = json.dumps({str(k): v for k, v in levels_data.items()})
    point_json = json.dumps(point_data)
    # Edge path data for JS (2D bundled edges only, shown in 2D mode)
    edge_paths_json = "[]"
    if bundled_edges:
        edge_paths = [[[float(p[0]), float(p[1]), float(p[2])]
                       for p in path] for path in bundled_edges]
        edge_paths_json = json.dumps(edge_paths)

    # Expose pydeck's local deckInstance as a global
    html = html.replace(
        "const deckInstance = createDeck(",
        "window.deckInstance = createDeck(",
    )

    # Build overlay: header, control panel, labels, and logic
    overlay_html = f"""
<!-- Header -->
<div id="header" style="position:absolute;top:0;left:0;right:260px;z-index:10;
  padding:12px 20px;font:14px -apple-system,'Segoe UI',sans-serif;">
  <div style="font-size:16px;font-weight:600">{title_str}</div>
  <div id="header-sub" class="sub" style="font-size:12px;">
    Scroll to zoom &middot; Drag to orbit &middot; Hover for details
  </div>
</div>

<!-- Control panel -->
<div id="panel" style="position:absolute;top:0;right:0;bottom:0;width:260px;
  z-index:10;font:13px -apple-system,'Segoe UI',sans-serif;
  overflow-y:auto;padding:16px;">

  <div style="font-weight:600;margin-bottom:12px;
    color:var(--fg-section);text-transform:uppercase;letter-spacing:1px;font-size:11px;">
    Controls
  </div>

  <div style="display:flex;gap:8px;margin-bottom:16px;">
    <button id="reset-btn" class="panel-btn">Reset View</button>
    <button id="mode-btn" class="panel-btn">2D Mode</button>
  </div>
  <div style="display:flex;gap:8px;margin-bottom:16px;">
    <button id="theme-btn" class="panel-btn">Light Mode</button>
  </div>

  <div style="margin-bottom:12px;">
    <label style="display:flex;align-items:center;gap:8px;cursor:pointer;">
      <input type="checkbox" id="toggle-labels" checked
        style="accent-color:var(--accent);width:16px;height:16px;">
      <span>Show labels</span>
    </label>
  </div>

  <div style="margin-bottom:12px;">
    <label style="display:flex;align-items:center;gap:8px;cursor:pointer;">
      <input type="checkbox" id="toggle-edges" checked
        style="accent-color:var(--accent);width:16px;height:16px;">
      <span>Show bridge edges</span>
    </label>
  </div>

  <div style="margin-bottom:16px;">
    <div style="margin-bottom:6px;">Point size: <span id="ps-val">2</span></div>
    <input type="range" id="point-size" min="1" max="8" value="2" step="0.5"
      style="width:100%;accent-color:var(--accent);">
  </div>

  <div style="font-weight:600;font-size:11px;margin-bottom:8px;
    color:var(--fg-section);text-transform:uppercase;letter-spacing:1px;">
    Clusters
  </div>
  <div id="cluster-list" style="font-size:12px;line-height:1.8;"></div>
</div>

<style>
:root {{
  --bg: #1e1e1e; --bg-panel: rgba(28,28,28,0.95); --bg-header: rgba(30,30,30,0.92);
  --bg-label: rgba(30,30,30,0.88); --bg-btn: #444; --bg-btn-hover: #555;
  --fg: #ddd; --fg-muted: #999; --fg-section: #aaa;
  --border: #444; --border-label: rgba(120,120,120,0.5);
  --range-bg: #555; --range-thumb: #aaa; --accent: #888;
  --shadow-label: rgba(0,0,0,0.6);
}}
body.light {{
  --bg: #f5f5f5; --bg-panel: rgba(245,245,245,0.97); --bg-header: rgba(250,250,250,0.95);
  --bg-label: rgba(255,255,255,0.92); --bg-btn: #ddd; --bg-btn-hover: #ccc;
  --fg: #222; --fg-muted: #666; --fg-section: #555;
  --border: #ccc; --border-label: rgba(100,100,100,0.3);
  --range-bg: #ccc; --range-thumb: #666; --accent: #666;
  --shadow-label: rgba(0,0,0,0.15);
}}
.cl {{
  position:absolute; pointer-events:none; z-index:5;
  color:var(--fg); font:bold 14px -apple-system,"Segoe UI",sans-serif;
  background:var(--bg-label);
  border:1px solid var(--border-label);
  padding:2px 6px; border-radius:3px; white-space:nowrap;
  transform:translate(-50%,-50%);
  text-shadow:0 1px 2px var(--shadow-label);
  transition:opacity 0.15s;
}}
.cl.level-coarse {{ font-size:15px; font-weight:800; border-width:2px; }}
.cl.level-mid    {{ font-size:13px; font-weight:700; border-width:1px; }}
.cl.level-fine   {{ font-size:11px; font-weight:600; border-width:1px; opacity:0.85; }}
#header {{ background:var(--bg-header); border-bottom:1px solid var(--border); color:var(--fg); }}
#header .sub {{ color:var(--fg-muted); }}
#panel {{ background:var(--bg-panel); border-left:1px solid var(--border); color:var(--fg); }}
#panel input[type="range"] {{
  -webkit-appearance:none; height:4px; background:var(--range-bg); border-radius:2px;
  outline:none;
}}
#panel input[type="range"]::-webkit-slider-thumb {{
  -webkit-appearance:none; width:14px; height:14px;
  background:var(--range-thumb); border-radius:50%; cursor:pointer;
}}
.panel-btn {{
  flex:1; background:var(--bg-btn); color:var(--fg);
  border:1px solid var(--border); padding:8px 0; border-radius:4px;
  cursor:pointer; font-size:13px; font-family:inherit;
}}
.panel-btn:hover {{ background:var(--bg-btn-hover); }}
</style>

<script>
(function() {{
  var labels = {label_json};
  var labelLevels = {levels_json};
  var allPoints = {point_json};
  var edgePaths = {edge_paths_json};
  var labelsVisible = true;
  var edgesVisible = true;

  // Cluster visibility state: null=all visible, Set=only those visible
  var hiddenClusters = new Set();
  var isolatedCluster = null;  // null=normal mode, number=isolated cluster

  // 2D/3D mode state
  var viewMode = "3d";
  var currentTheme = "dark";
  var zBackup = allPoints.map(function(p) {{ return p.z; }});

  // ── Multi-level label system ─────────────────────────────────────────
  // Parse levels: keys are cluster counts (as strings), values are label arrays
  var levelKeys = Object.keys(labelLevels).map(Number).sort(function(a,b) {{ return a - b; }});
  // Store z backups per level for 2D/3D toggle
  var zLevelsBackup = {{}};
  levelKeys.forEach(function(k) {{
    zLevelsBackup[k] = labelLevels[k].map(function(c) {{ return c.z; }});
  }});
  // Level style classes: coarsest=coarse, finest=fine, middle=mid
  function levelClass(k) {{
    var idx = levelKeys.indexOf(k);
    if (idx === 0) return "level-coarse";
    if (idx === levelKeys.length - 1) return "level-fine";
    return "level-mid";
  }}
  // Pre-create a pool of reusable label DOM elements
  var MAX_VISIBLE_LABELS = 40;
  var labelPool = [];
  for (var _lp = 0; _lp < MAX_VISIBLE_LABELS; _lp++) {{
    var e = document.createElement("div");
    e.className = "cl";
    e.style.opacity = "0";
    document.body.appendChild(e);
    labelPool.push(e);
  }}

  function isClusterVisible(cid) {{
    if (isolatedCluster !== null) return cid === isolatedCluster;
    return !hiddenClusters.has(cid);
  }}

  // Build edge path layer data (for toggling)
  function edgeColor() {{
    return (currentTheme === "light") ? [30, 80, 180, 90] : [255, 255, 255, 60];
  }}
  var edgePathData = edgePaths.map(function(path) {{
    return {{ path: path, color: edgeColor() }};
  }});

  function rebuildLayer() {{
    var dk = getDeck();
    if (!dk || !dk.props) return;
    var is2d = (viewMode === "2d");
    var ec = edgeColor();
    edgePathData = edgePaths.map(function(path) {{
      return {{ path: path, color: ec }};
    }});
    var visible = allPoints.filter(function(p) {{ return isClusterVisible(p.cluster); }});
    if (is2d) {{
      visible = visible.map(function(p) {{
        return {{ x: p.x, y: p.y, z: 0, r: p.r, g: p.g, b: p.b,
                  a: 255, title: p.title, cluster: p.cluster }};
      }});
    }}
    var layers = dk.props.layers;
    if (!layers || !layers.length) return;
    var oldLayer = layers[0];
    var newLayer = oldLayer.clone({{ data: visible }});
    var newLayers = [newLayer];
    // Add edge layer in 2D mode only (bundled edges are 2D)
    if (is2d && edgesVisible && edgePaths.length > 0 && layers.length > 1) {{
      var edgeLayer = layers[1];
      newLayers.push(edgeLayer.clone({{ data: edgePathData }}));
    }}
    dk.setProps({{ layers: newLayers }});
    updateRowStyles();
    // Trigger depth alpha recalc (3D only)
    if (!is2d) {{
      var vps = dk.getViewports ? dk.getViewports() : null;
      if (vps && vps.length) {{
        setTimeout(function() {{ updatePointAlpha(dk, vps[0]); }}, 50);
      }}
    }}
  }}

  function updateRowStyles() {{
    rows.forEach(function(row, i) {{
      var cid = rowClusterIds[i];
      if (isolatedCluster !== null) {{
        row.style.opacity = (cid === isolatedCluster) ? "1" : "0.3";
        row.style.textDecoration = (cid === isolatedCluster) ? "none" : "line-through";
      }} else if (hiddenClusters.has(cid)) {{
        row.style.opacity = "0.3";
        row.style.textDecoration = "line-through";
      }} else {{
        row.style.opacity = "1";
        row.style.textDecoration = "none";
      }}
    }});
  }}

  // Populate cluster list in panel
  var listEl = document.getElementById("cluster-list");
  var rows = [];
  var rowClusterIds = [];
  (function() {{
    var uniqueCids = [];
    var cset = {{}};
    allPoints.forEach(function(p) {{
      if (!(p.cluster in cset)) {{ cset[p.cluster] = true; uniqueCids.push(p.cluster); }}
    }});
    uniqueCids.sort(function(a,b) {{ return a - b; }});

    labels.forEach(function(c, i) {{
      var cid = i < uniqueCids.length ? uniqueCids[i] : i;
      var row = document.createElement("div");
      row.style.cursor = "pointer";
      row.style.padding = "2px 0";
      row.style.transition = "opacity 0.15s";
      row.innerHTML =
        '<span style="display:inline-block;width:10px;height:10px;border-radius:2px;' +
        'background:rgb(' + c.r + ',' + c.g + ',' + c.b + ');margin-right:6px;' +
        'vertical-align:middle;"></span>' +
        '<span style="vertical-align:middle;">' + c.text +
        ' <span style="color:var(--fg-muted);">(' + c.size + ')</span></span>';

      // Single click: toggle hide
      row.addEventListener("click", function(e) {{
        e.preventDefault();
        if (isolatedCluster !== null) return;  // in isolation mode, use dblclick
        if (hiddenClusters.has(cid)) {{
          hiddenClusters.delete(cid);
        }} else {{
          hiddenClusters.add(cid);
        }}
        rebuildLayer();
      }});

      // Double click: isolate or reset
      row.addEventListener("dblclick", function(e) {{
        e.preventDefault();
        if (isolatedCluster === cid) {{
          // Reset: show all
          isolatedCluster = null;
          hiddenClusters.clear();
        }} else {{
          // Isolate this cluster
          isolatedCluster = cid;
          hiddenClusters.clear();
        }}
        rebuildLayer();
      }});

      listEl.appendChild(row);
      rows.push(row);
      rowClusterIds.push(cid);
    }});
  }})();

  // Deck access
  function getDeck() {{
    var d = window.deckInstance;
    return d && d.deck ? d.deck : d || null;
  }}

  // Depth-based point alpha (debounced — rebuilds layer when view settles)
  var depthTimer = null;
  var lastViewJson = "";
  function updatePointAlpha(dk, vp) {{
    if (viewMode === "2d") return;
    var visible = allPoints.filter(function(p) {{ return isClusterVisible(p.cluster); }});
    if (!visible.length) return;
    var depths = [];
    for (var i = 0; i < visible.length; i++) {{
      try {{
        var sp = vp.project([visible[i].x, visible[i].y, visible[i].z]);
        depths.push(sp[2] || 0);
      }} catch(e) {{ depths.push(0); }}
    }}
    var minD = depths[0], maxD = depths[0];
    for (var i = 1; i < depths.length; i++) {{
      if (depths[i] < minD) minD = depths[i];
      if (depths[i] > maxD) maxD = depths[i];
    }}
    var rangeD = maxD - minD || 1;
    var updated = visible.map(function(p, i) {{
      var t = (depths[i] - minD) / rangeD;
      var alpha = Math.round(255 - t * 200);
      return {{ x: p.x, y: p.y, z: p.z, r: p.r, g: p.g, b: p.b,
                a: alpha, title: p.title, cluster: p.cluster }};
    }});
    var layers = dk.props.layers;
    if (layers && layers.length) {{
      var newLayers = [layers[0].clone({{ data: updated }})];
      // Preserve edge layer (index 1) if present
      for (var li = 1; li < layers.length; li++) {{
        newLayers.push(layers[li]);
      }}
      dk.setProps({{ layers: newLayers }});
    }}
  }}

  // ── Multi-level zoom-aware label placement ───────────────────────────
  // Zoom thresholds: map zoom level to which cluster levels to show
  // deck.gl OrbitView zoom ~5.5 default; higher = more zoomed in
  // All levels always active; spatial separation + label pool (40 max)
  // naturally prioritize coarse labels and fill gaps with finer ones.
  // Isolated clusters always get labeled regardless of zoom.
  var ZOOM_THRESHOLDS = [
    {{ zoom: 0, levels: levelKeys }}
  ];

  function getActiveLevels(zoom) {{
    var active = ZOOM_THRESHOLDS[0].levels;
    for (var i = 0; i < ZOOM_THRESHOLDS.length; i++) {{
      if (zoom >= ZOOM_THRESHOLDS[i].zoom) active = ZOOM_THRESHOLDS[i].levels;
    }}
    return active;
  }}

  // Label placement: project, cull off-screen, spatial separation
  function updateLabels(vp, zoom) {{
    if (!labelsVisible) {{
      for (var i = 0; i < MAX_VISIBLE_LABELS; i++) labelPool[i].style.opacity = "0";
      return;
    }}
    var activeLevels = getActiveLevels(zoom);
    var is2d = (viewMode === "2d");
    var w = window.innerWidth - 260;  // account for panel
    var h = window.innerHeight;

    // Minimum screen-space separation squared (pixels)
    var minSepSq = Math.pow(Math.min(w, h) * 0.06, 2);

    var placed = [];  // {{sx, sy, text, levelKey, depth}}

    // Process levels coarsest first
    for (var li = 0; li < activeLevels.length; li++) {{
      var lk = activeLevels[li];
      var lvlLabels = labelLevels[lk];
      if (!lvlLabels) continue;
      var cls = levelClass(lk);

      for (var j = 0; j < lvlLabels.length; j++) {{
        var c = lvlLabels[j];
        var lz = is2d ? 0 : c.z;
        var sp;
        try {{ sp = vp.project([c.x, c.y, lz]); }} catch(e) {{ continue; }}
        var sx = sp[0], sy = sp[1];

        // Cull off-screen
        if (sx < -30 || sx > w + 30 || sy < -30 || sy > h + 30) continue;

        // Check separation against already-placed labels
        var tooClose = false;
        for (var p = 0; p < placed.length; p++) {{
          var dx = sx - placed[p].sx, dy = sy - placed[p].sy;
          if (dx * dx + dy * dy < minSepSq) {{ tooClose = true; break; }}
        }}
        if (tooClose) continue;

        if (placed.length >= MAX_VISIBLE_LABELS) break;
        placed.push({{ sx: sx, sy: sy, text: c.text, cls: cls, depth: sp[2] || 0 }});
      }}
      if (placed.length >= MAX_VISIBLE_LABELS) break;
    }}

    // Sort by depth for z-ordering (farther = lower z-index)
    placed.sort(function(a, b) {{ return b.depth - a.depth; }});
    var minD = placed.length ? placed[placed.length - 1].depth : 0;
    var maxD = placed.length ? placed[0].depth : 1;
    var rangeD = maxD - minD || 1;

    // Apply to pool elements
    for (var i = 0; i < MAX_VISIBLE_LABELS; i++) {{
      var el = labelPool[i];
      if (i < placed.length) {{
        var pl = placed[i];
        el.textContent = pl.text;
        el.className = "cl " + pl.cls;
        el.style.left = pl.sx + "px";
        el.style.top = pl.sy + "px";
        el.style.zIndex = 10 + i;
        if (is2d) {{
          el.style.opacity = "1";
        }} else {{
          var t = (pl.depth - minD) / rangeD;
          el.style.opacity = (1.0 - t * 0.7).toFixed(2);
        }}
      }} else {{
        el.style.opacity = "0";
      }}
    }}
  }}

  // Main render loop
  function update() {{
    requestAnimationFrame(update);
    var dk = getDeck();
    if (!dk || !dk.getViewports) return;
    var vps = dk.getViewports();
    if (!vps || !vps.length) return;
    var vp = vps[0];

    // Get current zoom from view state
    var zoom = 5.5;
    if (dk.viewManager) {{
      try {{ zoom = dk.viewManager.getViewState().zoom || 5.5; }} catch(e) {{}}
    }}

    // Debounced point alpha update
    var vs = dk.viewManager ? JSON.stringify(dk.viewManager.getViewState()) : "";
    if (vs !== lastViewJson) {{
      lastViewJson = vs;
      clearTimeout(depthTimer);
      depthTimer = setTimeout(function() {{ updatePointAlpha(dk, vp); }}, 150);
    }}

    updateLabels(vp, zoom);
  }}

  setTimeout(function() {{
    var dk = getDeck();
    if (dk && dk.getViewports) {{
      var vps = dk.getViewports();
      if (vps && vps.length) updatePointAlpha(dk, vps[0]);
    }}
    update();
  }}, 1500);

  // Reset view
  document.getElementById("reset-btn").addEventListener("click", function() {{
    var dk = getDeck();
    if (dk && dk.setProps) {{
      dk.setProps({{ initialViewState: {{
        target: [0,0,0], rotationX: 15, rotationOrbit: 30, zoom: 5.5,
        transitionDuration: 300
      }} }});
    }}
  }});

  // Toggle labels
  document.getElementById("toggle-labels").addEventListener("change", function(e) {{
    labelsVisible = e.target.checked;
  }});

  // Toggle bridge edges
  document.getElementById("toggle-edges").addEventListener("change", function(e) {{
    edgesVisible = e.target.checked;
    rebuildLayer();
  }});

  // Point size slider
  document.getElementById("point-size").addEventListener("input", function(e) {{
    var sz = parseFloat(e.target.value);
    document.getElementById("ps-val").textContent = sz;
    var dk = getDeck();
    if (!dk || !dk.props) return;
    var layers = dk.props.layers;
    if (!layers || !layers.length) return;
    // Clone layer props with new pointSize
    var newLayers = layers.map(function(l) {{
      if (l.constructor && l.constructor.layerName === "PointCloudLayer") {{
        return l.clone({{ pointSize: sz }});
      }}
      return l;
    }});
    dk.setProps({{ layers: newLayers }});
  }});

  // ── Dark/light theme toggle ──────────────────────────────────────────
  function setTheme(theme) {{
    currentTheme = theme;
    var isLight = (theme === "light");
    document.body.classList.toggle("light", isLight);
    var btn = document.getElementById("theme-btn");
    if (btn) btn.textContent = isLight ? "Dark Mode" : "Light Mode";
    // Update deck.gl canvas background
    var bg = isLight ? "#f5f5f5" : "#1e1e1e";
    document.body.style.background = bg;
    var canvases = document.querySelectorAll("canvas");
    canvases.forEach(function(c) {{ c.style.background = bg; }});
    // Update pydeck wrapper div
    var deckDiv = document.getElementById("deck-container");
    if (deckDiv) deckDiv.style.background = bg;
    var deckWrapper = document.querySelector("#deckgl-wrapper");
    if (deckWrapper) deckWrapper.style.background = bg;
    // Rebuild edge layer with theme-appropriate edge color
    rebuildLayer();
  }}

  document.getElementById("theme-btn").addEventListener("click", function() {{
    setTheme(currentTheme === "dark" ? "light" : "dark");
  }});

  // ── 2D/3D mode toggle ────────────────────────────────────────────────
  function setViewMode(mode) {{
    viewMode = mode;
    var btn = document.getElementById("mode-btn");
    if (btn) btn.textContent = (mode === "2d") ? "3D Mode" : "2D Mode";
    var sub = document.getElementById("header-sub");
    if (sub) sub.textContent = (mode === "2d")
      ? "Scroll to zoom \u00b7 Drag to pan \u00b7 Hover for details"
      : "Scroll to zoom \u00b7 Drag to orbit \u00b7 Hover for details";
    var dk = getDeck();
    if (!dk || !dk.setProps) return;
    if (mode === "2d") {{
      // Flatten Z (XY already landscape-oriented from Python)
      for (var i = 0; i < allPoints.length; i++) allPoints[i].z = 0;
      // Flatten Z in all label levels
      levelKeys.forEach(function(k) {{
        labelLevels[k].forEach(function(c) {{ c.z = 0; }});
      }});
      // Top-down view, lock rotation, pan-only controller
      dk.setProps({{
        initialViewState: {{
          target: [0, 0, 0], rotationX: 90, rotationOrbit: 0, zoom: 5.5,
          minRotationX: 90, maxRotationX: 90,
          transitionDuration: 400
        }},
        controller: {{ dragMode: "pan" }}
      }});
    }} else {{
      // Restore Z
      for (var i = 0; i < allPoints.length; i++) allPoints[i].z = zBackup[i];
      // Restore Z in all label levels
      levelKeys.forEach(function(k) {{
        var backup = zLevelsBackup[k];
        labelLevels[k].forEach(function(c, j) {{ c.z = backup[j]; }});
      }});
      // Restore orbit controls
      dk.setProps({{
        initialViewState: {{
          target: [0, 0, 0], rotationX: 15, rotationOrbit: 30, zoom: 5.5,
          minRotationX: -90, maxRotationX: 90,
          transitionDuration: 400
        }},
        controller: {{ dragMode: "rotate" }}
      }});
    }}
    rebuildLayer();
  }}

  document.getElementById("mode-btn").addEventListener("click", function() {{
    setViewMode(viewMode === "3d" ? "2d" : "3d");
  }});

  // ── WebSocket bridge ──────────────────────────────────────────────────
  (function connectWS() {{
    var wsUrl = "ws://" + location.host + "/ws";
    var ws;
    try {{ ws = new WebSocket(wsUrl); }} catch(e) {{ return; }}
    ws.onmessage = function(e) {{
      var msg;
      try {{ msg = JSON.parse(e.data); }} catch(err) {{ return; }}
      switch (msg.cmd) {{
        case "hide":
          hiddenClusters.add(msg.cluster);
          isolatedCluster = null;
          rebuildLayer();
          break;
        case "show":
          hiddenClusters.delete(msg.cluster);
          isolatedCluster = null;
          rebuildLayer();
          break;
        case "isolate":
          isolatedCluster = msg.cluster;
          hiddenClusters.clear();
          rebuildLayer();
          break;
        case "show_all":
          isolatedCluster = null;
          hiddenClusters.clear();
          rebuildLayer();
          break;
        case "reset_view":
          var dk = getDeck();
          if (dk && dk.setProps) {{
            dk.setProps({{ initialViewState: {{
              target: [0,0,0], rotationX: 15, rotationOrbit: 30, zoom: 5.5,
              transitionDuration: 300
            }} }});
          }}
          break;
        case "point_size":
          var dk2 = getDeck();
          if (dk2 && dk2.props && dk2.props.layers && dk2.props.layers.length) {{
            var newLayers = dk2.props.layers.map(function(l) {{
              if (l.constructor && l.constructor.layerName === "PointCloudLayer") {{
                return l.clone({{ pointSize: msg.size || 2 }});
              }}
              return l;
            }});
            dk2.setProps({{ layers: newLayers }});
            var slider = document.getElementById("point-size");
            var valEl = document.getElementById("ps-val");
            if (slider) slider.value = msg.size || 2;
            if (valEl) valEl.textContent = msg.size || 2;
          }}
          break;
        case "labels":
          labelsVisible = !!msg.visible;
          var cb = document.getElementById("toggle-labels");
          if (cb) cb.checked = labelsVisible;
          break;
        case "highlight":
          if (msg.indices && msg.indices.length) {{
            var idxSet = new Set(msg.indices);
            var highlighted = allPoints.map(function(p, i) {{
              var vis = isClusterVisible(p.cluster);
              if (!vis) return null;
              var isHl = idxSet.has(i);
              return {{ x: p.x, y: p.y, z: p.z,
                        r: isHl ? 255 : p.r,
                        g: isHl ? 255 : p.g,
                        b: isHl ? 0   : p.b,
                        a: isHl ? 255 : 80,
                        title: p.title, cluster: p.cluster }};
            }}).filter(function(p) {{ return p !== null; }});
            var dk3 = getDeck();
            if (dk3 && dk3.props && dk3.props.layers && dk3.props.layers.length) {{
              var hlLayers = [dk3.props.layers[0].clone({{ data: highlighted }})];
              for (var hli = 1; hli < dk3.props.layers.length; hli++) hlLayers.push(dk3.props.layers[hli]);
              dk3.setProps({{ layers: hlLayers }});
            }}
            // Restore after 3 seconds
            setTimeout(function() {{ rebuildLayer(); }}, 3000);
          }}
          break;
        case "set_mode":
          if (msg.mode === "2d" || msg.mode === "3d") {{
            setViewMode(msg.mode);
          }}
          break;
        case "set_theme":
          if (msg.theme === "light" || msg.theme === "dark") {{
            setTheme(msg.theme);
          }}
          break;
      }}
    }};
    ws.onclose = function() {{ setTimeout(connectWS, 2000); }};
    ws.onerror = function() {{}};  // suppress console errors when server not running
  }})();
}})();
</script>
"""

    html = html.replace("</html>", overlay_html + "\n</html>", 1)
    Path(out_path).write_text(html)
    print(f"Wrote {out_path}")


# ── Main ─────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Compare BIRCH (2D) vs DYF tree (high-D) clustering")
    parser.add_argument("parquet_path", help="Path to embeddings parquet")
    parser.add_argument("--sample", type=int, default=8000)
    parser.add_argument("--n-clusters", type=int, default=25,
                        help="Target number of clusters")
    parser.add_argument("--dyf-depth", type=int, default=8,
                        help="DYF tree max depth")
    parser.add_argument("--dyf-bits", type=int, default=3,
                        help="DYF tree LSH bits per level")
    parser.add_argument("--renderer", choices=["scattergl", "pydeck"],
                        default="scattergl",
                        help="Rendering backend (default: scattergl)")
    parser.add_argument("--refine", action=argparse.BooleanOptionalAction,
                        default=True,
                        help="Refine incoherent clusters (default: on)")
    parser.add_argument("--no-label", action="store_true",
                        help="Skip LLM cluster labeling")
    parser.add_argument("--model", default="gemma2:9b",
                        help="Ollama model for cluster labeling (default: gemma2:9b)")
    parser.add_argument("--densmap", action="store_true",
                        help="Use densMAP for density-preserving projection")
    parser.add_argument("--no-edges", action="store_true",
                        help="Skip bridge edge bundling")
    parser.add_argument("--port", type=int, default=8766,
                        help="WebSocket port for viz_server bridge (default: 8766)")
    args = parser.parse_args()

    outdir = Path(args.parquet_path).parent
    titles, embeddings = load_and_dedup(args.parquet_path, args.sample)
    n = len(titles)
    titles_arr = np.array(titles)
    target_k = args.n_clusters
    use_3d = args.renderer == "pydeck"
    n_components = 3 if use_3d else 2

    # ── Shared projection (DYF-parameterized UMAP) ───────────────────────
    print(f"\n=== DYF-parameterized UMAP ({n_components}D) ===")
    dyf_k = suggest_n_neighbors(embeddings)
    coords = run_umap(embeddings, n_neighbors=dyf_k, n_components=n_components,
                       densmap=args.densmap)
    coords = orient_landscape(coords)

    # ── BIRCH clustering on projected coords ─────────────────────────────
    print(f"\nFitting BIRCH (target_k={target_k}) on {n_components}D coords...")
    birch = fit_birch(coords, target_k)
    labels_birch = birch.predict(coords)
    n_birch = len(set(labels_birch))
    print(f"  BIRCH: {n_birch} clusters")

    # ── Multi-level label hierarchy from BIRCH ───────────────────────────
    print(f"\nBuilding label hierarchy from BIRCH subclusters...")
    hierarchy_model = args.model if not args.no_label else None
    birch_levels = build_label_hierarchy(
        coords, titles_arr, birch, embeddings=embeddings,
        model=hierarchy_model,
    )

    # ── DYF tree clustering on high-D embeddings ─────────────────────────
    print(f"\nBuilding DYF tree (depth={args.dyf_depth}, bits={args.dyf_bits}) "
          f"on high-D embeddings...")
    t0 = time.time()
    tree = build_dyf_tree(
        embeddings, max_depth=args.dyf_depth,
        num_bits=args.dyf_bits, min_leaf_size=4,
    )

    if args.refine:
        stats = refine_dyf_tree(tree, embeddings)
        print(f"  Refined {stats['n_refined']} leaves "
              f"(coherence {stats['coherence_before']:.3f} -> "
              f"{stats['coherence_after']:.3f}, "
              f"leaves {stats['n_leaves_before']} -> {stats['n_leaves_after']})")

    labels_dyf = cut_dyf_tree_to_labels(tree, n, target_k, embeddings)

    if args.refine:
        labels_dyf = refine_clusters(labels_dyf, embeddings)

    n_dyf = len(set(labels_dyf.tolist()))
    print(f"  DYF tree: {n_dyf} clusters ({time.time() - t0:.1f}s)")

    dim_label = "3D" if use_3d else "2D"

    # ── Contrastive cluster labeling via LLM ─────────────────────────────
    names_birch = names_dyf = None
    if not args.no_label:
        print("\n=== Labeling BIRCH clusters ===")
        names_birch = label_clusters(
            titles, coords, labels_birch, embeddings, model=args.model)
        print("\n=== Labeling DYF tree clusters ===")
        names_dyf = label_clusters(
            titles, coords, labels_dyf, embeddings, model=args.model)

    # Add base BIRCH labels as finest hierarchy level so all panel labels
    # are reachable when zoomed in (Ward linkage cuts differ from BIRCH merge)
    if names_birch:
        ndim = coords.shape[1]
        base_level = []
        for cid in sorted(set(labels_birch.tolist()
                               if hasattr(labels_birch, 'tolist')
                               else list(labels_birch))):
            mask = labels_birch == cid
            pts = np.where(mask)[0]
            centroid = coords[pts].mean(axis=0)
            name = names_birch.get(cid, f"Cluster {cid}")
            base_level.append({
                "x": float(centroid[0]),
                "y": float(centroid[1]),
                "z": float(centroid[2]) if ndim >= 3 else 0.0,
                "text": name[:50],
                "size": int(mask.sum()),
            })
        birch_levels[n_birch] = base_level
        print(f"  Added base BIRCH level ({n_birch}): {len(base_level)} labels")

    # ── Bridge edge bundling (2D only) ──────────────────────────────────
    bundled_birch_2d = None
    if args.renderer == "pydeck" and not args.no_edges:
        print("\n=== Computing bridge edges (BIRCH) ===")
        bundled_birch_2d, pair_info, _ = compute_bridge_edges_3d(
            coords, embeddings, labels_birch, n_birch)
        if bundled_birch_2d:
            print(f"  Bundled {len(bundled_birch_2d)} 2D bridge edges")

    if args.renderer == "pydeck":
        rgb_birch = golden_ratio_rgb_map(labels_birch)
        rgb_dyf = golden_ratio_rgb_map(labels_dyf.tolist())

        path_birch = str(outdir / "rog_3d_birch_clusters.html")
        path_dyf = str(outdir / "rog_3d_dyf_tree_clusters.html")

        build_pydeck(
            coords, titles_arr, labels_birch, rgb_birch,
            f"BIRCH on {dim_label} (k={dyf_k} UMAP) — {n_birch} clusters, {n:,} pts",
            path_birch, cluster_names=names_birch, ws_port=args.port,
            label_levels=birch_levels,
            bundled_edges=bundled_birch_2d,
        )
        build_pydeck(
            coords, titles_arr, labels_dyf, rgb_dyf,
            f"DYF Tree (depth={args.dyf_depth}, bits={args.dyf_bits}) "
            f"— {n_dyf} clusters, {n:,} pts",
            path_dyf, cluster_names=names_dyf, ws_port=args.port,
            label_levels=birch_levels,
            bundled_edges=bundled_birch_2d,
        )
    else:
        cmap_birch = golden_ratio_color_map(labels_birch)
        cmap_dyf = golden_ratio_color_map(labels_dyf.tolist())

        html_birch = build_html(
            coords, titles_arr, labels_birch, cmap_birch,
            f"BIRCH on {dim_label} (k={dyf_k} UMAP) \u2014 {n_birch} clusters, {n:,} points",
            cluster_names=names_birch,
        )
        html_dyf = build_html(
            coords, titles_arr, labels_dyf, cmap_dyf,
            f"DYF Tree on high-D (depth={args.dyf_depth}, bits={args.dyf_bits}) "
            f"\u2014 {n_dyf} clusters, {n:,} points",
            cluster_names=names_dyf,
        )

        path_birch = str(outdir / "rog_2d_birch_clusters.html")
        path_dyf = str(outdir / "rog_2d_dyf_tree_clusters.html")

        Path(path_birch).write_text(html_birch)
        Path(path_dyf).write_text(html_dyf)
        print(f"\nWrote {path_birch}")
        print(f"Wrote {path_dyf}")

    subprocess.run(["open", path_birch])
    subprocess.run(["open", path_dyf])


if __name__ == "__main__":
    main()
