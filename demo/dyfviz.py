"""
BIRCH (2D) vs DYF tree (high-D) clustering comparison.

Both visualizations share the same DYF-parameterized UMAP 3D projection.
BIRCH clusters on the 2D coordinates; DYF tree clusters on the original
high-dimensional embeddings, then colors are mapped onto the shared layout.

Renders standalone pydeck HTML files with WebSocket bridge for live control,
bridge edges, guided tours, and 2D/3D toggle.

Usage:
    python demo/dyfviz.py demo/gudid_50k_titled.dyf [--sample 8000]
"""

import argparse
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

from dyf.colors import spatial_rgb_map, spatial_color_map, tree_rgb_map
from dyf.provenance import provenance_from_dict, check_compatible
from dyf.dyf_tree import (
    build_dyf_tree,
    refine_dyf_tree,
    cut_dyf_tree_to_labels,
    refine_clusters,
)


# ── Helpers ──────────────────────────────────────────────────────────────


def load_and_dedup(parquet_path, sample=None, pre_labels=None, extra_cols=None):
    """Load parquet, optionally sample, dedup via LSH.

    If pre_labels is provided (ndarray same length as parquet), it is
    filtered through the same sample + dedup pipeline and returned as
    the third element of the tuple.

    If extra_cols is provided (list of column names), those columns are
    extracted from the parquet and filtered through the same pipeline,
    returned as a dict[str, list] in the fourth element.
    """
    print(f"Loading {parquet_path}...")
    df = pl.read_parquet(parquet_path)
    if sample and sample < len(df):
        df = df.sample(sample, seed=42)
        if pre_labels is not None:
            # pl.DataFrame.sample with seed=42 uses polars internal RNG;
            # we need to replicate the same row selection.  Polars sample
            # returns rows in their original order, so reconstruct indices.
            idx = df.with_row_index("__idx__")["__idx__"].to_numpy()
            pre_labels = pre_labels[idx]

    titles = df["title"].to_list()
    embeddings = np.array(df["embedding"].to_list(), dtype=np.float32)

    # Extract extra columns before dedup filtering
    extra_data = {}
    if extra_cols:
        for col in extra_cols:
            if col in df.columns:
                extra_data[col] = df[col].to_list()

    from dyf_rs import DensityClassifier
    from dyf.chunks import deduplicate_chunks

    clf = DensityClassifier(embedding_dim=embeddings.shape[1], num_bits=12, seed=42)
    clf.fit(embeddings)
    bucket_ids = np.asarray(clf.get_bucket_ids())
    dedup_mask = deduplicate_chunks(bucket_ids, np.asarray(titles))

    n_before = len(titles)
    titles = [t for t, keep in zip(titles, dedup_mask) if keep]
    embeddings = embeddings[dedup_mask]
    if pre_labels is not None:
        pre_labels = pre_labels[dedup_mask]
    for col in extra_data:
        extra_data[col] = [v for v, keep in zip(extra_data[col], dedup_mask) if keep]
    print(f"  {n_before} -> {len(titles)} after dedup")
    return titles, embeddings, pre_labels, extra_data


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
        random_state=42,
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
        return [], [], {}, centroids

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
    # Each path corresponds to edge_list[i] = (c1, c2)
    bundled_2d = []
    current_path = []
    for _, row in bundled_df.iterrows():
        if pd.isna(row["x"]) or pd.isna(row["y"]):
            if current_path:
                bundled_2d.append(current_path)
                current_path = []
        else:
            current_path.append([row["x"], row["y"]])
    if current_path:
        bundled_2d.append(current_path)

    # Add z=0 to 2D bundled paths for flat rendering
    bundled_2d_flat = []
    for path in bundled_2d:
        bundled_2d_flat.append(np.array([[x, y, 0.0] for x, y in path], dtype=np.float32))

    # Generate catenary curves between centroids for 3D
    # Sag increases with distance, stronger edges arc higher
    n_segments = 20  # points per curve
    max_count = max(pair_counts.values()) if pair_counts else 1

    bundled_3d = []
    for c1, c2 in edge_list:
        p1 = centroids[c1]
        p2 = centroids[c2]
        count = pair_counts.get((c1, c2), 1)

        # Distance between centroids
        dist = np.linalg.norm(p2 - p1)

        # Sag: proportional to distance, stronger edges sag less (arc higher)
        strength = count / max_count  # 0-1
        base_sag = 0.15 * dist  # sag proportional to distance
        sag = base_sag * (1.0 - 0.5 * strength)  # stronger = less sag (higher arc)

        # Generate catenary-like curve (parabolic approximation)
        path = []
        for j in range(n_segments + 1):
            t = j / n_segments
            # Linear interpolation for x, y, z
            pt = p1 + t * (p2 - p1)
            # Parabolic arc in z (maximum at t=0.5)
            sag_amount = 4 * sag * t * (1 - t)  # positive = upward arc
            pt = pt.copy()
            pt[2] += sag_amount
            path.append(pt.tolist())

        bundled_3d.append(np.array(path, dtype=np.float32))

    print(f"  Got {len(bundled_2d_flat)} 2D bundled + {len(bundled_3d)} 3D catenary paths")

    return bundled_2d_flat, bundled_3d, pair_counts, centroids


def fit_birch(data, target_k, max_iters=10, embeddings=None):
    """Fit BIRCH with enough subclusters, then agglomerative merge to target_k.

    If *embeddings* is provided, the agglomerative merge uses Ward linkage
    on L2-normalised embedding-space centroids (cosine-like) instead of the
    projected-coord subcluster centres.  This gives semantically coherent
    clusters while BIRCH handles the spatial partitioning.
    """
    from scipy.cluster.hierarchy import linkage, fcluster

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
    if n_subs <= target_k:
        return best_birch

    if embeddings is not None:
        # Hybrid merge: combine embedding similarity with UMAP spatial
        # proximity so clusters are both semantically coherent AND
        # visually contiguous (no isolated fragments).
        sub_labels = best_birch.predict(data)

        # Embedding centroids (L2-normalised → cosine-like)
        sub_emb_centroids = np.zeros((n_subs, embeddings.shape[1]),
                                     dtype=np.float64)
        # UMAP-space centroids
        sub_spatial_centroids = np.zeros((n_subs, data.shape[1]),
                                         dtype=np.float64)
        for sid in range(n_subs):
            mask = sub_labels == sid
            if mask.any():
                sub_emb_centroids[sid] = embeddings[mask].mean(axis=0)
                sub_spatial_centroids[sid] = data[mask].mean(axis=0)

        # L2-normalise embeddings so euclidean Ward ≈ cosine Ward
        norms = np.linalg.norm(sub_emb_centroids, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        sub_emb_normed = sub_emb_centroids / norms

        # Normalise spatial centroids to unit variance per dimension
        # so they're on comparable scale to the embedding features
        spatial_std = sub_spatial_centroids.std(axis=0, keepdims=True)
        spatial_std[spatial_std == 0] = 1.0
        sub_spatial_normed = sub_spatial_centroids / spatial_std

        # Concatenate: embedding features + spatial features (weighted)
        # spatial_weight controls how much UMAP proximity matters
        # 0.5 = balanced; higher = more visually contiguous clusters
        spatial_weight = 0.5
        hybrid = np.hstack([sub_emb_normed,
                            sub_spatial_normed * spatial_weight])

        Z = linkage(hybrid, method='ward')
        merge_labels = fcluster(Z, t=target_k, criterion='maxclust') - 1

        # Attach the merge mapping so predict() returns final cluster IDs
        # while predict_sub() preserves access to raw subcluster IDs
        # (needed by build_label_hierarchy).
        best_birch._emb_merge_labels = merge_labels
        best_birch._target_k = target_k
        best_birch.predict_sub = best_birch.predict  # raw subcluster predict

        _orig_predict = best_birch.predict

        def _merged_predict(X, _orig=_orig_predict, _map=merge_labels):
            sub_ids = _orig(X)
            return np.array([_map[s] for s in sub_ids])

        best_birch.predict = _merged_predict
        print(f"  BIRCH: {n_subs} subclusters → {target_k} via "
              f"hybrid (embedding + spatial w={spatial_weight}) Ward linkage")
        return best_birch

    # Fallback: agglomerative merge on projected coords (original behaviour)
    birch = Birch(n_clusters=target_k, threshold=best_birch.threshold,
                  branching_factor=50)
    birch.fit(data)
    return birch


def cluster_levels(target_k):
    """Generate hierarchy zoom levels below target_k (base level added separately)."""
    if target_k <= 6:
        return (max(2, target_k // 2),)
    lo = max(2, round(target_k * 0.2))
    mid = max(lo + 1, round(target_k * 0.5))
    return (lo, mid)


def build_label_hierarchy(coords, titles_arr, birch, embeddings=None,
                          model=None, cluster_data=None, base_labels=None,
                          target_k=25):
    """Build multi-level label hierarchy from BIRCH subclusters via Ward linkage.

    If model is provided, generates LLM labels via contrastive TF-IDF.
    cluster_data: the data BIRCH was fit on (for predict); defaults to coords.
    Returns dict mapping level -> list of label dicts with centroid, name, size.
    """
    ndim = coords.shape[1]
    sub_centers = birch.subcluster_centers_
    n_subs = len(sub_centers)
    titles = list(titles_arr)

    # Ward linkage on subcluster centers
    Z = linkage(sub_centers, method='ward')

    # Map each point to its subcluster (predict on same space BIRCH was fit on)
    predict_data = cluster_data if cluster_data is not None else coords
    # Use raw subcluster predict if available (embedding-agglom mode)
    _predict = getattr(birch, 'predict_sub', birch.predict)
    sub_labels = _predict(predict_data)

    levels = {}
    for k in cluster_levels(target_k):
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
                "cid": int(cid),
            }
            if base_labels is not None:
                rec["leaf_cids"] = sorted(set(int(c) for c in base_labels[mask]))
            label_data.append(rec)
        levels[k] = label_data
        print(f"  Level {k}: {len(label_data)} labels")

    return levels


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


def label_clusters(titles, coords, labels, embeddings, model="gpt-oss:20b",
                   n_samples=20, cache_file=None, cache_key=None):
    """Label clusters via contrastive TF-IDF + local Ollama LLM.

    For each cluster: spatially samples titles, computes contrastive TF-IDF
    keywords against the nearest high-D neighbor, and asks the LLM for a
    short topic label.  Runs Ollama calls in parallel.

    If cache_file is provided, loads labels from cache (under cache_key)
    when available, and saves newly generated labels back to the cache.

    Returns:
        dict mapping cluster_id -> label string
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    unique_labels = sorted(set(int(l) for l in labels))

    # Check label cache
    if cache_file:
        cache_path = Path(cache_file)
        if cache_path.exists():
            cache_data = json.loads(cache_path.read_text())
            cached = cache_data.get(cache_key or "default", {})
            if cached and len(cached) == len(unique_labels):
                cluster_names = {int(k): v for k, v in cached.items()}
                print(f"  Loaded {len(cluster_names)} labels from cache ({cache_file})")
                for cid in sorted(cluster_names.keys()):
                    n_pts = int(np.sum(np.asarray(labels) == cid))
                    print(f"    [{cid:2d}] {cluster_names[cid]:<35s} ({n_pts} pts)")
                return cluster_names

    # Diversity gate: skip LLM if text has insufficient variety
    from dyf.splits import assess_text_diversity, label_clusters_frequency
    diversity = assess_text_diversity(titles)
    if not diversity.is_diverse:
        print(f"  LOW TEXT DIVERSITY: {diversity.reason}")
        print(f"    Using frequency-based labeling (no LLM)")
        return label_clusters_frequency(titles, labels)

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
            import urllib.request
            payload = json.dumps({
                "model": model,
                "prompt": prompt,
                "stream": False,
            }).encode()
            req = urllib.request.Request(
                "http://localhost:11434/api/generate",
                data=payload,
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(req, timeout=120) as resp:
                data = json.loads(resp.read().decode())
                label = data.get("response", "").strip()
            label = label.split('\n')[0][:50].strip('"\'').strip()
            return cid, label if label else f"Cluster {cid}"
        except Exception as e:
            print(f"    WARNING: Ollama failed for cluster {cid}: {e}")
            return cid, f"Cluster {cid}"

    n_workers = min(2, len(tasks))
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

    # ── Second pass: re-label duplicates with sibling context ───────────
    # Find clusters that share the same label
    from collections import Counter
    label_counts = Counter(cluster_names.values())
    duplicates = {lbl for lbl, cnt in label_counts.items() if cnt > 1}

    if duplicates:
        dup_cids = [cid for cid in unique_labels
                    if cluster_names[cid] in duplicates]
        taken = sorted(set(cluster_names.values()))
        print(f"    Re-labeling {len(dup_cids)} clusters with duplicate names...")

        dup_tasks = []
        for cid in dup_cids:
            pts = cluster_points[cid]
            if not pts:
                continue
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
                combined_labels_arr = np.zeros(
                    len(pts) + len(neighbor_pts), dtype=int)
                combined_labels_arr[len(pts):] = 1
                kw = _compute_tfidf_keywords(
                    combined_titles, combined_labels_arr, 2, top_k=8, min_df=1)
                keywords = [w for w, _ in kw.get(0, [])][:8]
                if keywords:
                    kw_str = (f"\nDistinguishing keywords (vs neighbor): "
                              f"{', '.join(keywords)}")

            # List sibling labels so the LLM avoids them
            siblings = [l for l in taken if l != cluster_names[cid]]
            sibling_str = ", ".join(f'"{s}"' for s in siblings[:15])

            prompt = (
                f"You are labeling clusters in an embedding space. "
                f"This cluster has {len(pts)} items.\n"
                f"{kw_str}\n"
                f"Sample items from across this cluster:\n"
                + "\n".join(f"- {t}" for t in sample_titles)
                + "\n\n"
                f"These labels are ALREADY TAKEN by other clusters: "
                f"{sibling_str}\n\n"
                "Give a short (2-5 word) label that is DIFFERENT from all "
                "taken labels. Focus on what makes THIS cluster UNIQUE — "
                "specific device types, body parts, or use cases.\n\n"
                "Reply with ONLY the label, nothing else."
            )
            dup_tasks.append((cid, prompt))

        n_workers = min(2, len(dup_tasks))
        with ThreadPoolExecutor(max_workers=n_workers) as executor:
            futures = {executor.submit(_call_ollama, t): t for t in dup_tasks}
            for future in as_completed(futures):
                cid, label = future.result()
                cluster_names[cid] = label

    for cid in unique_labels:
        n_pts = len(cluster_points[cid])
        print(f"    [{cid:2d}] {cluster_names[cid]:<35s} ({n_pts} pts)")

    # Save to cache
    if cache_file:
        cache_path = Path(cache_file)
        cache_data = {}
        if cache_path.exists():
            cache_data = json.loads(cache_path.read_text())
        cache_data[cache_key or "default"] = {
            str(k): v for k, v in cluster_names.items()
        }
        cache_path.write_text(json.dumps(cache_data, indent=2))
        print(f"  Saved {len(cluster_names)} labels to cache ({cache_file})")

    return cluster_names



def compute_cluster_shapes(coords, titles, labels):
    """Compute shape descriptors for each cluster for narration context.

    For each cluster, computes:
    - Aspect ratio via PCA eigenvalues → "compact", "elongated", "very elongated"
    - Tendril detection (points >2σ from centroid forming directional chains)
    - Outlier titles (farthest from centroid) and core titles (nearest to centroid)

    Returns:
        dict mapping cluster_id -> {
            shape: str, aspect_ratio: float, n_tendrils: int,
            outlier_titles: list[str], core_titles: list[str],
            outlier_indices: list[int], core_indices: list[int]
        }
    """
    from sklearn.decomposition import PCA

    label_arr = np.asarray(labels)
    coords_arr = np.asarray(coords)
    unique_cids = sorted(set(int(c) for c in label_arr))

    shapes = {}
    for cid in unique_cids:
        mask = label_arr == cid
        pts_idx = np.where(mask)[0]
        if len(pts_idx) < 3:
            shapes[cid] = {
                "shape": "compact", "aspect_ratio": 1.0, "n_tendrils": 0,
                "outlier_titles": [], "core_titles": [],
                "outlier_indices": [], "core_indices": [],
            }
            continue

        pts = coords_arr[pts_idx]
        centroid = pts.mean(axis=0)
        dists = np.linalg.norm(pts - centroid, axis=1)

        # Aspect ratio via PCA
        n_comp = min(3, pts.shape[1], len(pts))
        pca = PCA(n_components=n_comp)
        pca.fit(pts)
        eigenvalues = pca.explained_variance_
        if eigenvalues[-1] > 0:
            aspect_ratio = float(eigenvalues[0] / eigenvalues[-1])
        else:
            aspect_ratio = float(eigenvalues[0] / (eigenvalues[-1] + 1e-10))

        if aspect_ratio > 8:
            shape = "very elongated"
        elif aspect_ratio > 3:
            shape = "elongated"
        else:
            shape = "compact"

        # Tendril detection: points beyond 2σ from centroid
        mean_dist = dists.mean()
        std_dist = dists.std()
        threshold = mean_dist + 2 * std_dist
        far_mask = dists > threshold
        far_idx = pts_idx[far_mask]

        n_tendrils = 0
        if len(far_idx) >= 2:
            # Cluster far points by direction from centroid using simple angular binning
            far_pts = coords_arr[far_idx]
            directions = far_pts - centroid
            norms = np.linalg.norm(directions, axis=1, keepdims=True)
            norms[norms == 0] = 1
            unit_dirs = directions / norms
            # Use first PCA component to project directions
            if n_comp >= 1:
                proj = unit_dirs @ pca.components_[0]
                # Count sign groups as rough tendril estimate
                n_pos = (proj > 0.3).sum()
                n_neg = (proj < -0.3).sum()
                if n_pos >= 2:
                    n_tendrils += 1
                if n_neg >= 2:
                    n_tendrils += 1

        # Core titles (nearest to centroid)
        n_core = min(5, len(pts_idx))
        core_order = np.argsort(dists)[:n_core]
        core_indices = pts_idx[core_order].tolist()
        core_titles = [str(titles[i]) for i in core_indices]

        # Outlier titles (farthest from centroid)
        n_outlier = min(5, len(pts_idx))
        outlier_order = np.argsort(dists)[-n_outlier:][::-1]
        outlier_indices = pts_idx[outlier_order].tolist()
        outlier_titles = [str(titles[i]) for i in outlier_indices]

        shapes[cid] = {
            "shape": shape,
            "aspect_ratio": round(aspect_ratio, 1),
            "n_tendrils": n_tendrils,
            "outlier_titles": outlier_titles,
            "core_titles": core_titles,
            "outlier_indices": outlier_indices,
            "core_indices": core_indices,
        }

    return shapes


def _number_to_words(n):
    """Convert an integer to English words for TTS-friendly output."""
    if n < 0:
        return "negative " + _number_to_words(-n)
    if n == 0:
        return "zero"

    ones = ["", "one", "two", "three", "four", "five", "six", "seven",
            "eight", "nine", "ten", "eleven", "twelve", "thirteen",
            "fourteen", "fifteen", "sixteen", "seventeen", "eighteen", "nineteen"]
    tens = ["", "", "twenty", "thirty", "forty", "fifty",
            "sixty", "seventy", "eighty", "ninety"]

    if n < 20:
        return ones[n]
    if n < 100:
        return tens[n // 10] + ("" if n % 10 == 0 else " " + ones[n % 10])
    if n < 1000:
        remainder = n % 100
        if remainder == 0:
            return ones[n // 100] + " hundred"
        return ones[n // 100] + " hundred and " + _number_to_words(remainder)
    if n < 1_000_000:
        thousands = _number_to_words(n // 1000)
        rest = n % 1000
        if rest == 0:
            return thousands + " thousand"
        elif rest < 100:
            return thousands + " thousand and " + _number_to_words(rest)
        else:
            return thousands + " thousand, " + _number_to_words(rest)
    # Millions (unlikely but safe)
    millions = _number_to_words(n // 1_000_000)
    rest = n % 1_000_000
    if rest == 0:
        return millions + " million"
    return millions + " million, " + _number_to_words(rest)


def _approx_number_words(n):
    """Convert a number to approximate TTS-friendly English, e.g. 'about four thousand seven hundred'."""
    if n < 20:
        return _number_to_words(n)
    if n < 100:
        # Round to nearest 10
        rounded = round(n / 10) * 10
        if rounded == n:
            return _number_to_words(n)
        return "about " + _number_to_words(rounded)
    if n < 1000:
        # Round to nearest 50
        rounded = round(n / 50) * 50
        return "about " + _number_to_words(rounded)
    if n < 10000:
        # Round to nearest 100
        rounded = round(n / 100) * 100
        return "about " + _number_to_words(rounded)
    # Round to nearest 1000
    rounded = round(n / 1000) * 1000
    return "about " + _number_to_words(rounded)


def generate_tour_narration(cluster_names, titles, labels, edge_pairs=None,
                            model=None, cluster_shapes=None, label_centroids=None,
                            title=None, narration_file=None):
    """Generate tour narration for each cluster.

    If narration_file is provided, loads pre-written narration keyed by cluster
    name. Falls back to a simple default for any cluster name not found in the
    file. The file should be a JSON dict with cluster names as keys and
    narration strings as values, plus optional _intro_template and _outro.
    """
    label_arr = np.asarray(labels)
    cluster_points = defaultdict(list)
    for i, cid in enumerate(label_arr):
        cluster_points[int(cid)].append(i)

    print(f"\n=== Generating tour narration ===")
    print(f"  Generating narration for {len(cluster_names)} clusters...")

    total_pts = sum(len(v) for v in cluster_points.values())
    sorted_cids = sorted(cluster_names.keys(),
                         key=lambda c: len(cluster_points.get(c, [])),
                         reverse=True)

    # Load pre-written narration if available
    prewritten = {}
    if narration_file:
        narration_path = Path(narration_file)
        if narration_path.exists():
            prewritten = json.loads(narration_path.read_text())
            print(f"  Loaded {len(prewritten)} entries from {narration_file}")

    narration = {}
    name_seen = defaultdict(int)  # track duplicates for _2 suffix lookup

    for cid in sorted_cids:
        name = cluster_names[cid]
        n_pts = len(cluster_points.get(cid, []))
        n_approx = _approx_number_words(n_pts)

        # Look up by name, then name_2, name_3 for duplicates
        name_seen[name] += 1
        lookup_key = name if name_seen[name] == 1 else f"{name}_{name_seen[name]}"

        if lookup_key in prewritten:
            narration[cid] = prewritten[lookup_key]
        elif name in prewritten and name_seen[name] == 1:
            narration[cid] = prewritten[name]
        else:
            # Simple fallback
            narration[cid] = (
                f"{name}. With {n_approx} products, Curvo's language model "
                f"grouped these items together based on shared patterns "
                f"in their descriptions."
            )
            print(f"    WARNING: fallback narration for [{cid}] '{lookup_key}'")

    done = len(narration)
    print(f"    Generated {done}/{len(cluster_names)} narrations...")
    for cid in sorted_cids[:3]:
        print(f"    [{cid:2d}] {narration[cid][:70]}...")

    # --- Intro ---
    n_clusters = len(cluster_names)
    n_words = _number_to_words(n_clusters)
    top3 = [cluster_names[c] for c in sorted_cids[:3]]
    total_words = _approx_number_words(total_pts)

    intro_template = prewritten.get("_intro_template", "")
    if intro_template:
        narration["intro"] = intro_template.format(
            title=title or "GUDID Medical Device Landscape",
            total=total_words,
            n_clusters=n_words,
            top1=top3[0] if len(top3) > 0 else "",
            top2=top3[1] if len(top3) > 1 else "",
            top3=top3[2] if len(top3) > 2 else "",
        )
    elif title:
        narration["intro"] = (
            f"{title}. What you see is a landscape built by Curvo — "
            f"{total_words} items, organized into {n_words} clusters. "
            f"Curvo's language model read every product description "
            f"and grouped items by the meaning it found. "
            f"The biggest regions are {top3[0]}"
            + (f", {top3[1]}" if len(top3) > 1 else "")
            + (f", and {top3[2]}" if len(top3) > 2 else "")
            + ". Let's take a closer look."
        )
    else:
        narration["intro"] = (
            f"Welcome. What you see is a landscape built by Curvo — "
            f"{total_words} medical devices, organized into {n_words} clusters. "
            f"Curvo's language model read every product description "
            f"and grouped items by the meaning it found. "
            f"The biggest regions are {top3[0]}"
            + (f", {top3[1]}" if len(top3) > 1 else "")
            + (f", and {top3[2]}" if len(top3) > 2 else "")
            + ". Let's take a closer look."
        )
    print(f"    [intro] {narration['intro'][:70]}...")

    # --- Outro ---
    narration["outro"] = prewritten.get("_outro",
        "And that completes our tour. "
        "What Curvo has built here is a map of relationships — products grouped not by "
        "a human-defined taxonomy, but by the patterns Curvo's language model found "
        "in how they're described. "
        "Clusters that sit close together share deeper similarities, "
        "and the bridges between them reveal how one category shades into the next. "
        "This is what Curvo can do with any product catalogue."
    )
    print(f"    [outro] {narration['outro'][:70]}...")

    return narration


def _generate_narration_ollama(cluster_names, titles, labels, coords,
                                model="gpt-oss:20b", title=None):
    """Generate tour narration inline via Ollama (no pre-written file needed).

    For each cluster, samples ~10 titles using spatial sampling, asks Ollama
    for a 2-3 sentence narration, then generates intro/outro from templates.

    Returns dict mapping cluster_id (and "intro"/"outro") -> narration text.
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    label_arr = np.asarray(labels)
    cluster_points = defaultdict(list)
    for i, cid in enumerate(label_arr):
        cluster_points[int(cid)].append(i)

    total_pts = sum(len(v) for v in cluster_points.values())
    sorted_cids = sorted(cluster_names.keys(),
                         key=lambda c: len(cluster_points.get(c, [])),
                         reverse=True)

    print(f"\n=== Generating tour narration via Ollama ({model}) ===")
    print(f"  Generating narration for {len(cluster_names)} clusters...")

    # Build tasks: (cid, prompt)
    tasks = []
    for cid in sorted_cids:
        name = cluster_names[cid]
        pts = cluster_points.get(cid, [])
        n_pts = len(pts)
        n_approx = _approx_number_words(n_pts)

        # Sample ~10 titles using spatial sampling
        sample_indices = _sample_spatial(pts, coords, 30)
        seen = set()
        sample_titles = []
        for idx in sample_indices:
            t = titles[idx]
            if t not in seen:
                seen.add(t)
                sample_titles.append(t)
                if len(sample_titles) >= 10:
                    break

        items_str = "\n".join(f"- {t}" for t in sample_titles)
        prompt = (
            f'Write a 2-3 sentence tour narration for a cluster of devices '
            f'called "{name}" containing approximately {n_approx} products.\n\n'
            f'Sample items:\n{items_str}\n\n'
            f'Rules:\n'
            f'- Start with the cluster name\n'
            f'- Be specific about what distinguishes this group\n'
            f'- Write for text-to-speech (British male, calm documentary style)\n'
            f'- Keep under 45 words\n'
            f'- No quotation marks or special characters\n'
        )
        tasks.append((cid, prompt))

    # Parallel Ollama calls
    narration = {}

    def _call_ollama(task):
        cid, prompt = task
        try:
            result = subprocess.run(
                ["ollama", "run", model, prompt],
                capture_output=True, text=True, timeout=60,
            )
            text = result.stdout.strip()
            # Clean up: remove quotes, collapse whitespace
            text = text.strip('"\'').strip()
            text = re.sub(r'\s+', ' ', text)
            return cid, text if text else None
        except Exception as e:
            print(f"    WARNING: Ollama failed for cluster {cid}: {e}")
            return cid, None

    n_workers = min(4, len(tasks))
    completed = 0
    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        futures = {executor.submit(_call_ollama, t): t for t in tasks}
        for future in as_completed(futures):
            cid, text = future.result()
            if text:
                narration[cid] = text
            else:
                # Fallback
                name = cluster_names[cid]
                n_approx = _approx_number_words(
                    len(cluster_points.get(cid, [])))
                narration[cid] = (
                    f"{name}. With {n_approx} products, Curvo's language model "
                    f"grouped these items together based on shared patterns "
                    f"in their descriptions."
                )
            completed += 1
            if completed % 5 == 0 or completed == len(tasks):
                print(f"    Narrated {completed}/{len(tasks)} clusters...",
                      flush=True)

    for cid in sorted_cids[:3]:
        print(f"    [{cid:2d}] {narration[cid][:70]}...")

    # --- Intro ---
    n_clusters = len(cluster_names)
    n_words = _number_to_words(n_clusters)
    top3 = [cluster_names[c] for c in sorted_cids[:3]]
    total_words = _approx_number_words(total_pts)

    if title:
        narration["intro"] = (
            f"{title}. What you see is a landscape built by Curvo — "
            f"{total_words} items, organized into {n_words} clusters. "
            f"Curvo's language model read every product description "
            f"and grouped items by the meaning it found. "
            f"The biggest regions are {top3[0]}"
            + (f", {top3[1]}" if len(top3) > 1 else "")
            + (f", and {top3[2]}" if len(top3) > 2 else "")
            + ". Let's take a closer look."
        )
    else:
        narration["intro"] = (
            f"Welcome. What you see is a landscape built by Curvo — "
            f"{total_words} items, organized into {n_words} clusters. "
            f"Curvo's language model read every product description "
            f"and grouped items by the meaning it found. "
            f"The biggest regions are {top3[0]}"
            + (f", {top3[1]}" if len(top3) > 1 else "")
            + (f", and {top3[2]}" if len(top3) > 2 else "")
            + ". Let's take a closer look."
        )
    print(f"    [intro] {narration['intro'][:70]}...")

    # --- Outro ---
    narration["outro"] = (
        "And that completes our tour. "
        "What Curvo has built here is a map of relationships — products grouped not by "
        "a human-defined taxonomy, but by the patterns Curvo's language model found "
        "in how they're described. "
        "Clusters that sit close together share deeper similarities, "
        "and the bridges between them reveal how one category shades into the next. "
        "This is what Curvo can do with any product catalogue."
    )
    print(f"    [outro] {narration['outro'][:70]}...")

    return narration


def generate_tour_audio(narration, voice="bm_george"):
    """Generate audio for tour narration using Kokoro TTS.

    Args:
        narration: dict mapping cluster_id -> narration text
        voice: Kokoro voice ID (default: bm_george for British male)

    Returns:
        dict mapping cluster_id -> {"data": base64 WAV, "duration": ms}
    """
    import base64
    import io
    try:
        import soundfile as sf
        from kokoro import KPipeline
    except (ImportError, Exception) as e:
        print(f"  [TTS] Kokoro not available ({type(e).__name__}: {e}), skipping audio generation")
        return {}

    print(f"\n=== Generating tour audio ===")
    print(f"  Rendering {len(narration)} audio clips with voice '{voice}'...")

    # Initialize Kokoro pipeline (British English)
    lang_code = 'b' if voice.startswith('b') else 'a'
    pipeline = KPipeline(lang_code=lang_code)

    audio_data = {}
    done = 0
    for cid, text in narration.items():
        try:
            # Generate audio
            for _, _, audio in pipeline(text, voice=voice):
                # Calculate duration in ms (24000 Hz sample rate)
                duration_ms = int(len(audio) / 24000 * 1000)
                # Convert to WAV bytes
                buf = io.BytesIO()
                sf.write(buf, audio, 24000, format='WAV')
                buf.seek(0)
                # Store both data and duration
                audio_data[cid] = {
                    "data": base64.b64encode(buf.read()).decode('ascii'),
                    "duration": duration_ms
                }
                break  # Only need first chunk
        except Exception as e:
            print(f"    [TTS] Failed for cluster {cid}: {e}")

        done += 1
        if done % 5 == 0 or done == len(narration):
            print(f"    Rendered {done}/{len(narration)} clips...")

    return audio_data


# ── Pydeck builder ───────────────────────────────────────────────────────


def build_pydeck(coords, titles_arr, labels, rgb_map, title_str, out_path,
                 cluster_names=None, ws_port=8766, label_levels=None,
                 bundled_edges_2d=None, bundled_edges_3d=None, edge_pairs=None,
                 logo_path=None, tour_narration=None, tour_audio=None,
                 tour_callouts=None, tour_title=None, subtitle_str="",
                 multi_level_data=None, embeddings=None):
    """Build a pydeck 3D point cloud with HTML overlay labels."""
    import base64
    import pyarrow as pa
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
            "cid": int(cid),
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

    # Compute initial zoom from 2-sigma extent (matches JS defaultZoom formula)
    import math as _math
    sigma = 1.5
    stds = [float(coords[:, d].std()) for d in range(min(ndim, 3))]
    max_extent = max(2 * sigma * s for s in stds) or 1.0
    # Assume ~800px viewport as reasonable default
    initial_zoom = _math.log2(800 * 1.3 / max_extent)
    initial_zoom = max(4.0, min(12.0, initial_zoom))

    # Add default full-opacity alpha
    for rec in point_data:
        rec["a"] = 255

    point_layer = pdk.Layer(
        "PointCloudLayer",
        data=[],
        get_position=["x", "y", "z"],
        get_color=["r", "g", "b", "a"],
        get_normal=[0, 0, 1],
        point_size=3,
        pickable=True,
        auto_highlight=True,
    )

    layers = [point_layer]

    # Add empty edge layer (populated by JS rebuildLayer)
    if bundled_edges_2d or bundled_edges_3d:
        edge_layer = pdk.Layer(
            "PathLayer",
            data=[],
            get_path="path",
            get_color="color",
            get_width="width",
            width_scale=1,
            width_min_pixels=1,
            width_max_pixels=6,
            pickable=False,
        )
        layers.append(edge_layer)

    # TextLayer + OrbitView has a known sizing bug (deck.gl #6808).
    # We inject HTML overlay labels instead — see reset_script below.

    view_state = pdk.ViewState(
        target=target,
        controller=True,
        rotation_x=90,
        rotation_orbit=0,
        zoom=initial_zoom,
        min_rotation_x=90,
        max_rotation_x=90,
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
    html = html.replace('<html>', '<html lang="en">', 1)
    html = html.replace(
        '<head>',
        '<head><meta name="viewport" content="width=device-width, initial-scale=1">',
        1,
    )
    html = html.replace('<title>pydeck</title>', f'<title>dyfviz — {title_str}</title>', 1)
    label_json = json.dumps(label_data)
    # Multi-level labels: keys are level numbers (as strings in JSON)
    levels_json = json.dumps({str(k): v for k, v in levels_data.items()})
    # Tour narration: cluster_id -> narration text
    narration_json = json.dumps({str(k): v for k, v in (tour_narration or {}).items()})
    # Tour audio: cluster_id -> base64 WAV
    audio_json = json.dumps({str(k): v for k, v in (tour_audio or {}).items()})
    # Tour callouts: cluster_id -> {indices: [...], labels: [...]}
    callouts_json = json.dumps({str(k): v for k, v in (tour_callouts or {}).items()})

    # Multi-level cluster metadata (when ROG cache provided)
    if multi_level_data:
        cluster_meta = {
            "levels": {},
            "default": str(multi_level_data["default"]),
            "multiLevel": True,
        }
        for lvl in sorted(multi_level_data["label_data"].keys()):
            cluster_meta["levels"][str(lvl)] = {
                "names": {str(k): v for k, v in multi_level_data["names"][lvl].items()},
                "label_data": multi_level_data["label_data"][lvl],
            }
        if multi_level_data.get("lsh_labels") is not None:
            cluster_meta["lsh"] = {
                "names": {str(k): v for k, v in multi_level_data["lsh_names"].items()},
                "label_data": multi_level_data["lsh_label_data"],
            }
        # Dual 2D/3D cluster data
        if multi_level_data.get("has_dual_clusters"):
            cluster_meta["hasDualClusters"] = True
            cluster_meta["levels_3d"] = {}
            for lvl in sorted(multi_level_data["label_data_3d"].keys()):
                cluster_meta["levels_3d"][str(lvl)] = {
                    "names": {str(k): v for k, v in multi_level_data["names_3d"][lvl].items()},
                    "label_data": multi_level_data["label_data_3d"][lvl],
                }
        # Spatial color maps per level for JS level switcher.
        # Use pre-stored maps from .dyf metadata when available,
        # fall back to computing from embeddings.
        pre_colors = multi_level_data.get("color_maps", {})
        color_maps = {}
        for lvl in sorted(multi_level_data["labels"].keys()):
            slvl = str(lvl)
            if slvl in pre_colors:
                color_maps[slvl] = pre_colors[slvl]
            elif embeddings is not None:
                lvl_labels = multi_level_data["labels"][lvl]
                smap = spatial_rgb_map(lvl_labels, embeddings)
                color_maps[slvl] = {str(k): v for k, v in smap.items()}
        if multi_level_data.get("lsh_labels") is not None:
            if "lsh" in pre_colors:
                color_maps["lsh"] = pre_colors["lsh"]
            elif (multi_level_data.get("item_leaf_map") is not None
                  and multi_level_data.get("tree_structure") is not None):
                smap = tree_rgb_map(
                    multi_level_data["lsh_labels"],
                    multi_level_data["tree_structure"],
                    multi_level_data["item_leaf_map"])
                color_maps["lsh"] = {str(k): v for k, v in smap.items()}
            elif embeddings is not None:
                smap = spatial_rgb_map(multi_level_data["lsh_labels"], embeddings)
                color_maps["lsh"] = {str(k): v for k, v in smap.items()}
        if color_maps:
            cluster_meta["colorMaps"] = color_maps

        cluster_meta_json = json.dumps(cluster_meta)
    else:
        cluster_meta_json = "null"

    # ── Arrow IPC for compact binary transfer ───────────────────────
    if multi_level_data:
        # Multi-level mode: store per-level cluster IDs instead of RGB
        batch_dict = {
            "x": pa.array([p["x"] for p in point_data], type=pa.float32()),
            "y": pa.array([p["y"] for p in point_data], type=pa.float32()),
            "z": pa.array([p["z"] for p in point_data], type=pa.float32()),
            "a": pa.array([p["a"] for p in point_data], type=pa.uint8()),
        }
        for lvl in sorted(multi_level_data["labels"].keys()):
            col_name = f"cluster_{lvl}"
            batch_dict[col_name] = pa.array(
                multi_level_data["labels"][lvl].tolist(), type=pa.int32())
        # Add 3D cluster columns if dual clusters
        if multi_level_data.get("has_dual_clusters"):
            for lvl in sorted(multi_level_data["labels_3d"].keys()):
                col_name = f"cluster_{lvl}_3d"
                batch_dict[col_name] = pa.array(
                    multi_level_data["labels_3d"][lvl].tolist(),
                    type=pa.int32())
        if multi_level_data.get("lsh_labels") is not None:
            batch_dict["cluster_lsh"] = pa.array(
                multi_level_data["lsh_labels"].tolist(), type=pa.int32())
        batch_dict["title"] = pa.array(
            [p["title"] for p in point_data], type=pa.utf8())
        points_batch = pa.record_batch(batch_dict)
    else:
        points_batch = pa.record_batch({
            "x": pa.array([p["x"] for p in point_data], type=pa.float32()),
            "y": pa.array([p["y"] for p in point_data], type=pa.float32()),
            "z": pa.array([p["z"] for p in point_data], type=pa.float32()),
            "r": pa.array([p["r"] for p in point_data], type=pa.uint8()),
            "g": pa.array([p["g"] for p in point_data], type=pa.uint8()),
            "b": pa.array([p["b"] for p in point_data], type=pa.uint8()),
            "a": pa.array([p["a"] for p in point_data], type=pa.uint8()),
            "cluster": pa.array([p["cluster"] for p in point_data], type=pa.int32()),
            "title": pa.array([p["title"] for p in point_data], type=pa.utf8()),
        })
    sink = pa.BufferOutputStream()
    with pa.ipc.new_stream(sink, points_batch.schema) as writer:
        writer.write_batch(points_batch)
    import gzip as _gzip
    points_ipc_b64 = base64.b64encode(
        _gzip.compress(sink.getvalue().to_pybytes())
    ).decode()

    # Edge paths as Arrow IPC: 2D bundled + 3D catenary + weights (gzip-compressed)
    edges_2d_ipc_b64 = ""
    edges_3d_ipc_b64 = ""
    edge_pairs_json = "[]"
    if edge_pairs and (bundled_edges_2d or bundled_edges_3d):
        sorted_pairs = sorted(edge_pairs.keys(), key=lambda p: -edge_pairs[p])
        max_weight = max(edge_pairs.values()) if edge_pairs else 1
        weights = []
        for i in range(len(sorted_pairs)):
            pair = sorted_pairs[i]
            weights.append(edge_pairs[pair] / max_weight)

        # Serialize 2D bundled paths
        if bundled_edges_2d:
            path_arrays_2d = []
            for path in bundled_edges_2d:
                flat = []
                for pt in path:
                    flat.extend([float(pt[0]), float(pt[1]), float(pt[2])])
                path_arrays_2d.append(flat)
            edges_2d_batch = pa.record_batch({
                "path": pa.array(path_arrays_2d, type=pa.list_(pa.float32())),
                "weight": pa.array(weights[:len(path_arrays_2d)], type=pa.float32()),
            })
            sink_2d = pa.BufferOutputStream()
            with pa.ipc.new_stream(sink_2d, edges_2d_batch.schema) as writer:
                writer.write_batch(edges_2d_batch)
            edges_2d_ipc_b64 = base64.b64encode(
                _gzip.compress(sink_2d.getvalue().to_pybytes())
            ).decode()

        # Serialize 3D catenary paths
        if bundled_edges_3d:
            path_arrays_3d = []
            for path in bundled_edges_3d:
                flat = []
                for pt in path:
                    flat.extend([float(pt[0]), float(pt[1]), float(pt[2])])
                path_arrays_3d.append(flat)
            edges_3d_batch = pa.record_batch({
                "path": pa.array(path_arrays_3d, type=pa.list_(pa.float32())),
                "weight": pa.array(weights[:len(path_arrays_3d)], type=pa.float32()),
            })
            sink_3d = pa.BufferOutputStream()
            with pa.ipc.new_stream(sink_3d, edges_3d_batch.schema) as writer:
                writer.write_batch(edges_3d_batch)
            edges_3d_ipc_b64 = base64.b64encode(
                _gzip.compress(sink_3d.getvalue().to_pybytes())
            ).decode()

        edge_pairs_json = json.dumps([[int(a), int(b)] for a, b in sorted_pairs])

    # Expose pydeck's local deckInstance as a global
    html = html.replace(
        "const deckInstance = createDeck(",
        "window.deckInstance = createDeck(",
    )

    # Optional client logo (e.g. --logo curvo.png)
    client_logo_html = ""
    if logo_path and Path(logo_path).exists():
        logo_data = Path(logo_path).read_bytes()
        logo_b64 = base64.b64encode(logo_data).decode()
        client_logo_html = (
            f'<img src="data:image/png;base64,{logo_b64}" '
            f'class="header-logo" '
            f'style="height:28px;margin-left:12px;vertical-align:middle;">'
        )

    overlay_html = build_pydeck_overlay(
        points_ipc_b64=points_ipc_b64,
        edges_2d_ipc_b64=edges_2d_ipc_b64,
        edges_3d_ipc_b64=edges_3d_ipc_b64,
        label_json=label_json,
        levels_json=levels_json,
        edge_pairs_json=edge_pairs_json,
        narration_json=narration_json,
        callouts_json=callouts_json,
        audio_json=audio_json,
        title_str=title_str,
        subtitle_str=subtitle_str,
        client_logo_html=client_logo_html,
        tour_title=tour_title,
        cluster_meta_json=cluster_meta_json,
    )

    html = html.replace("</html>", overlay_html + "\n</html>", 1)
    Path(out_path).write_text(html)
    print(f"Wrote {out_path}")


def build_pydeck_overlay(*, points_ipc_b64, edges_2d_ipc_b64, edges_3d_ipc_b64,
                         label_json, levels_json, edge_pairs_json,
                         narration_json, callouts_json, audio_json,
                         title_str, subtitle_str, client_logo_html, tour_title,
                         cluster_meta_json="null"):
    """Render the JS overlay template. Can be called standalone for patching."""
    dyf_logo_svg = (
        '<svg class="dyf-logo" viewBox="0 0 340 105" width="85" height="26">'
        '<defs><linearGradient id="dyf-grad" x1="0%" y1="0%" x2="100%" y2="100%">'
        '<stop offset="0%" stop-color="#e94560"/>'
        '<stop offset="100%" stop-color="#f9a826"/>'
        '</linearGradient></defs>'
        '<text x="4" y="82" font-family="Montserrat,sans-serif"'
        ' font-size="88" font-weight="700">'
        '<tspan class="dyf-letter">d</tspan>'
        '<tspan fill="url(#dyf-grad)">\u028e</tspan>'
        '<tspan class="dyf-letter">f</tspan>'
        '<tspan class="dyf-muted">viz</tspan>'
        '</text>'
        '</svg>'
    )
    html_shell = f"""
<!-- DYF_OVERLAY_START -->
<!-- Highlighter canvas overlay -->
<canvas id="hl-canvas" style="position:absolute;top:0;left:0;width:100%;height:100%;
  z-index:6;pointer-events:none;"></canvas>
<!-- Header -->
<div id="header" style="position:absolute;top:0;left:0;right:260px;z-index:20;
  padding:12px 20px;font:14px -apple-system,'Segoe UI',sans-serif;">
  <div style="margin-bottom:6px;">{client_logo_html}</div>
  <div style="font-size:28px;font-weight:700;letter-spacing:0.01em">{title_str}</div>
  <div style="font-size:12px;opacity:0.6;margin-top:2px;">{subtitle_str}</div>
  <div id="header-sub" class="sub" style="font-size:11px;">
    Scroll to zoom &middot; Drag to orbit &middot; Hover for details
    <span id="session-id" style="margin-left:12px;opacity:0.7;font-family:monospace;"></span>
    <span id="build-stamp" style="margin-left:12px;opacity:0.5;font-family:monospace;font-size:9px;">v9-dedup</span>
  </div>
</div>
<!-- Tour cluster label overlay (positioned at centroid) -->
<div id="tour-label" style="display:none;position:absolute;z-index:50;padding:8px 16px;
  background:rgba(0,0,0,0.85);border-radius:6px;
  font:600 18px -apple-system,'Segoe UI',sans-serif;color:#fff;
  text-align:center;pointer-events:none;white-space:nowrap;
  box-shadow:0 2px 12px rgba(0,0,0,0.5);
  transform:translate(-50%,-100%) translateY(-12px);"></div>
<!-- Container for edge centroid labels during tour -->
<div id="tour-edge-labels" style="display:none;"></div>
<!-- Container for point callout labels during tour -->
<div id="tour-callout-labels" style="display:none;"></div>
<!-- Camera state debug overlay -->
<div id="camera-debug" style="display:none;position:absolute;bottom:10px;left:10px;z-index:20;
  padding:8px 12px;background:rgba(0,0,0,0.8);border-radius:4px;
  font:11px monospace;color:#0f0;pointer-events:none;white-space:pre;"></div>
<style>
.tour-edge-label {{
  position:absolute;z-index:14;padding:4px 10px;
  background:rgba(60,80,160,0.9);border-radius:4px;
  font:500 12px -apple-system,'Segoe UI',sans-serif;color:#fff;
  text-align:center;pointer-events:none;white-space:nowrap;
  box-shadow:0 1px 6px rgba(0,0,0,0.4);
  transform:translate(-50%,-50%);
  border:1px solid rgba(100,140,255,0.5);
}}
body.light .tour-edge-label {{
  background:rgba(70,100,180,0.95);
  border-color:rgba(50,80,150,0.6);
}}
.tour-callout-label {{
  position:absolute;z-index:16;padding:3px 8px;
  background:rgba(255,255,255,0.92);border-radius:3px;
  font:500 11px -apple-system,'Segoe UI',sans-serif;color:#1a1a1a;
  text-align:left;pointer-events:none;white-space:nowrap;
  box-shadow:0 1px 8px rgba(0,0,0,0.5);
  transform:translateY(-50%);
  border:1px solid rgba(255,220,80,0.8);
  opacity:0;transition:opacity 0.3s ease-in;
  max-width:200px;overflow:hidden;text-overflow:ellipsis;
}}
.tour-callout-label.visible {{
  opacity:1;
}}
.tour-callout-label.core {{
  border-color:rgba(100,200,255,0.8);
}}
.tour-callout-label.outlier {{
  border-color:rgba(255,180,60,0.8);
}}
body.light .tour-callout-label {{
  background:rgba(255,255,255,0.95);
  color:#111;
}}
#tour-label.hero {{
  font-size:clamp(42px, 7vw, 84px);
  padding:32px 56px;
  border-radius:14px;
  background:rgba(0,0,0,0.8);
  box-shadow:0 6px 60px rgba(0,0,0,0.7);
  transform:translate(-50%,-50%);
  letter-spacing:0.02em;
}}
</style>

<!-- Panel toggle tab -->
<div id="panel-toggle" onclick="togglePanel()" style="position:absolute;top:50%;right:260px;
  transform:translateY(-50%);z-index:11;width:20px;height:60px;
  background:var(--bg-panel);border:1px solid var(--border);border-right:none;
  border-radius:6px 0 0 6px;cursor:pointer;display:flex;align-items:center;
  justify-content:center;font-size:14px;color:var(--fg);transition:right 0.3s ease;">
  <span id="panel-toggle-arrow">▶</span>
</div>

<!-- Palette panel -->
<div id="panel" style="position:absolute;top:0;right:0;bottom:0;width:260px;
  z-index:10;font:13px -apple-system,'Segoe UI',sans-serif;
  overflow-y:auto;transition:right 0.3s ease;">

  <!-- DYF logo -->
  <div style="padding:10px 12px 6px;text-align:center;">
    {dyf_logo_svg}
  </div>

  <!-- View palette -->
  <div class="palette">
    <div class="palette-header" onclick="this.parentElement.classList.toggle('collapsed')">
      <span class="palette-arrow"></span>View
    </div>
    <div class="palette-body">
      <div style="display:flex;gap:6px;margin-bottom:8px;">
        <button id="reset-btn" class="panel-btn">↻ Reset</button>
        <button id="mode-btn" class="panel-btn">□ 2D</button>
        <button id="fullscreen-btn" class="panel-btn">⛶ Fullscreen</button>
      </div>
      <div style="display:flex;gap:6px;">
        <button id="theme-btn" class="panel-btn">☼ Light</button>
      </div>
    </div>
  </div>

  <!-- Display palette -->
  <div class="palette collapsed">
    <div class="palette-header" onclick="this.parentElement.classList.toggle('collapsed')">
      <span class="palette-arrow"></span>Display
    </div>
    <div class="palette-body">
      <label class="palette-check">
        <input type="checkbox" id="toggle-labels" checked>
        <span>Labels</span>
      </label>
      <label class="palette-check">
        <input type="checkbox" id="toggle-edges" checked>
        <span>Bridge edges</span>
      </label>
      <label class="palette-check">
        <input type="checkbox" id="toggle-sheen">
        <span>Specular sweep</span>
      </label>
      <label class="palette-check">
        <input type="checkbox" id="toggle-orbit">
        <span>Auto-orbit</span>
      </label>
      <label class="palette-check">
        <input type="checkbox" id="toggle-outliers" checked>
        <span>Show outliers</span>
      </label>
      <div style="margin-top:8px;">
        <div style="display:flex;justify-content:space-between;margin-bottom:4px;">
          <span>Point size</span><span id="ps-val">2</span>
        </div>
        <input type="range" id="point-size" min="1" max="8" value="2" step="0.5"
          style="width:100%;">
      </div>
    </div>
  </div>

  <!-- Clusters palette -->
  <div class="palette collapsed">
    <div class="palette-header" onclick="this.parentElement.classList.toggle('collapsed')">
      <span class="palette-arrow"></span>Clusters
    </div>
    <div class="palette-body">
      <div id="level-selector" style="display:none;margin-bottom:8px;">
        <div style="font-size:10px;opacity:0.6;margin-bottom:4px;">Cluster level</div>
        <div id="level-buttons" style="display:flex;flex-wrap:wrap;gap:3px;"></div>
      </div>
      <div id="cluster-list" style="font-size:12px;line-height:1.8;max-height:400px;overflow-y:auto;"></div>
    </div>
  </div>

  <!-- Legend palette -->
  <div class="palette collapsed">
    <div class="palette-header" onclick="this.parentElement.classList.toggle('collapsed')">
      <span class="palette-arrow"></span>Legend
    </div>
    <div class="palette-body">
      <div style="font-size:11px;line-height:1.9;">
        <div style="font-size:10px;opacity:0.6;margin-bottom:2px;">Size</div>
        <div>⭑ big hub &nbsp;⭒ small hub</div>
        <div style="font-size:10px;opacity:0.6;margin-top:6px;margin-bottom:2px;">Purity</div>
        <div>≈ mixed (flagged outliers)</div>
      </div>
    </div>
  </div>

  <!-- Tour palette -->
  <div class="palette collapsed">
    <div class="palette-header" onclick="this.parentElement.classList.toggle('collapsed')">
      <span class="palette-arrow"></span>Tour
    </div>
    <div class="palette-body">
      <button id="tour-btn" class="panel-btn" style="width:100%;margin-bottom:8px;">
        ▶ Start Tour
      </button>
      <div id="tour-list" style="font-size:11px;line-height:1.6;max-height:200px;overflow-y:auto;"></div>
    </div>
  </div>

  <!-- MCP Debug palette -->
  <div class="palette collapsed">
    <div class="palette-header" onclick="this.parentElement.classList.toggle('collapsed')">
      <span class="palette-arrow"></span>MCP Debug
    </div>
    <div class="palette-body">
      <div id="mcp-log" style="font-size:10px;font-family:monospace;max-height:200px;overflow-y:auto;overflow-x:hidden;word-break:break-all;"></div>
    </div>
  </div>
</div>

<link href="https://fonts.googleapis.com/css2?family=Montserrat:wght@700&display=swap" rel="stylesheet">
<style>
:root {{
  --bg: #1e1e1e; --bg-panel: rgba(28,28,28,0.95); --bg-header: rgba(30,30,30,0.75);
  --bg-label: rgba(30,30,30,0.88); --bg-btn: #444; --bg-btn-hover: #555;
  --bg-palette-header: rgba(40,40,40,0.95);
  --fg: #ddd; --fg-muted: #999; --fg-section: #aaa;
  --border: #444; --border-label: rgba(120,120,120,0.5);
  --range-bg: #555; --range-thumb: #aaa; --accent: #888;
  --shadow-label: rgba(0,0,0,0.6);
}}
body.light {{
  --bg: #f5f5f5; --bg-panel: rgba(245,245,245,0.97); --bg-header: rgba(250,250,250,0.75);
  --bg-label: rgba(255,255,255,0.92); --bg-btn: #ddd; --bg-btn-hover: #ccc;
  --bg-palette-header: rgba(230,230,230,0.95);
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
#header {{ background:var(--bg-header); border-bottom:1px solid var(--border); color:var(--fg);
  backdrop-filter:blur(8px); -webkit-backdrop-filter:blur(8px); }}
#header .sub {{ color:var(--fg-muted); }}
.dyf-logo .dyf-letter {{ fill: #dddddd; }}
body.light .dyf-logo .dyf-letter {{ fill: #333333; }}
.dyf-logo .dyf-muted {{ fill: #999999; }}
body.light .dyf-logo .dyf-muted {{ fill: #666666; }}
.header-logo {{ filter:grayscale(1) invert(1) brightness(1.8); }}
body.light .header-logo {{ filter:grayscale(1) brightness(0.3); }}
#panel {{ background:var(--bg-panel); border-left:1px solid var(--border); color:var(--fg); }}
#panel input[type="range"] {{
  -webkit-appearance:none; height:4px; background:var(--range-bg); border-radius:2px;
  outline:none; accent-color:var(--accent);
}}
#panel input[type="range"]::-webkit-slider-thumb {{
  -webkit-appearance:none; width:14px; height:14px;
  background:var(--range-thumb); border-radius:50%; cursor:pointer;
}}
.panel-btn {{
  flex:1; background:var(--bg-btn); color:var(--fg);
  border:1px solid var(--border); padding:6px 0; border-radius:3px;
  cursor:pointer; font-size:11px; font-family:inherit;
}}
.panel-btn:hover {{ background:var(--bg-btn-hover); }}
.level-btn {{
  padding:3px 8px; border-radius:3px; border:1px solid var(--border);
  background:var(--bg-btn); color:var(--fg); cursor:pointer;
  font-size:10px; font-family:monospace; transition:all 0.15s;
}}
.level-btn:hover {{ background:var(--bg-btn-hover); }}
.level-btn.active {{
  background:var(--fg-section); color:var(--bg); border-color:var(--fg-section);
  font-weight:700;
}}
.palette {{
  border-bottom:1px solid var(--border);
}}
.palette-header {{
  display:flex; align-items:center; gap:6px;
  padding:8px 12px; cursor:pointer; user-select:none;
  font-size:11px; font-weight:600; text-transform:uppercase;
  letter-spacing:0.5px; color:var(--fg-section);
  background:var(--bg-palette-header);
}}
.palette-header:hover {{ background:var(--bg-btn); }}
.palette-arrow {{
  display:inline-block; width:0; height:0;
  border-left:4px solid transparent; border-right:4px solid transparent;
  border-top:5px solid var(--fg-muted);
  transition:transform 0.15s;
}}
.palette.collapsed .palette-arrow {{
  transform:rotate(-90deg);
}}
.palette-body {{
  padding:10px 12px;
}}
.palette.collapsed .palette-body {{
  display:none;
}}
.palette-check {{
  display:flex; align-items:center; gap:8px; cursor:pointer;
  padding:3px 0;
}}
.palette-check input {{
  accent-color:var(--accent); width:14px; height:14px;
}}
.tour-item {{
  opacity: 0.6;
  transition: opacity 0.2s;
  position: relative;
  overflow: hidden;
}}
.tour-item:hover {{
  opacity: 0.9;
  background: rgba(255,255,255,0.1);
}}
.tour-item.active {{
  opacity: 1;
  color: #fff;
  font-weight: 600;
  background: linear-gradient(to right, var(--accent) var(--progress, 0%), rgba(255,255,255,0.15) var(--progress, 0%));
}}
</style>

<script>
// Panel toggle (exposed on window for module script access)
var panelHidden = false;
function togglePanel() {{
  panelHidden = !panelHidden;
  var panel = document.getElementById("panel");
  var toggle = document.getElementById("panel-toggle");
  var arrow = document.getElementById("panel-toggle-arrow");
  var header = document.getElementById("header");
  if (panelHidden) {{
    panel.style.right = "-260px";
    toggle.style.right = "0";
    arrow.textContent = "◀";
    if (header) header.style.right = "0";
  }} else {{
    panel.style.right = "0";
    toggle.style.right = "260px";
    arrow.textContent = "▶";
    if (header) header.style.right = "260px";
  }}
}}
window.togglePanel = togglePanel;
</script>
"""

    # Inject data via window.__DYF_DATA__ then load overlay module
    _js_path = Path(__file__).parent / "pydeck_overlay.js"
    js_src = _js_path.read_text()
    tour_title_json = json.dumps(tour_title or "GUDID Medical Device Landscape")
    data_script = (
        '<script>\n'
        'window.__DYF_DATA__ = {\n'
        f'  clusterMeta: {cluster_meta_json},\n'
        f'  pointsIpcB64: "{points_ipc_b64}",\n'
        f'  edges2dIpcB64: "{edges_2d_ipc_b64}",\n'
        f'  edges3dIpcB64: "{edges_3d_ipc_b64}",\n'
        f'  labels: {label_json},\n'
        f'  labelLevels: {levels_json},\n'
        f'  edgePairs: {edge_pairs_json},\n'
        f'  tourNarration: {narration_json},\n'
        f'  tourCallouts: {callouts_json},\n'
        f'  tourAudio: {audio_json},\n'
        f'  tourTitle: {tour_title_json},\n'
        '};\n'
        '</script>\n'
    )
    return (html_shell
            + data_script
            + '<script type="module">\n' + js_src + '</script>\n'
            + '<!-- DYF_OVERLAY_END -->\n')


# ── Agglomerated DYF tree buckets ─────────────────────────────────────


def _agglomerate_tree_leaves(idx, coords, embeddings, n_groups=50):
    """Thin wrapper — delegates to ``dyf.agglomerate.agglomerate_tree_leaves``."""
    from dyf.agglomerate import agglomerate_tree_leaves
    return agglomerate_tree_leaves(idx, coords, embeddings, n_groups=n_groups)


# ── Main ─────────────────────────────────────────────────────────────────


def _render_from_dyf(dyf_path, args):
    """Render directly from an enriched .dyf file (Level 1+). No compute."""
    from dyf.lazy_index import LazyIndex

    print(f"\n=== Rendering from .dyf: {dyf_path} ===")
    with LazyIndex(dyf_path) as idx:
        level = idx.detect_enrichment_level()
        print(f"  Enrichment level: {level}")
        if level < 1:
            print(f"  ERROR: .dyf file is level 0 (no UMAP coords). "
                  f"Run 'python demo/dyf_enrich.py project {dyf_path}' first.")
            return
        data = idx.extract_all_fields()

    # Check for stored category graphs
    from dyf.categorical import load_category_graphs
    cat_graphs = load_category_graphs(data.get('metadata', {}))
    if cat_graphs:
        names = ", ".join(cat_graphs.keys())
        print(f"  Category graphs detected: {names}")

    n = len(data['embeddings'])
    umap_x = data['fields']['umap_x']
    umap_y = data['fields']['umap_y']
    umap_z = data['fields']['umap_z']
    coords = np.column_stack([umap_x, umap_y, umap_z])

    titles_list = data['fields'].get('title')
    if titles_list is None:
        titles_list = [f"Item {i}" for i in range(n)]
    titles_arr = np.array(titles_list)

    # Pick best cluster level — handle cluster_{k}, cluster_{k}_2d, cluster_{k}_3d
    _cluster_re = re.compile(r'^cluster_(\d+)(?:_(2d|3d))?$')

    def _cluster_sort_key(f: str) -> tuple[int, str]:
        m = _cluster_re.match(f)
        assert m is not None
        return (int(m.group(1)), m.group(2) or '')

    all_cluster_fields = sorted(
        [f for f in data['fields'] if _cluster_re.match(f)],
        key=_cluster_sort_key)

    # Separate into 2d, 3d, and bare fields
    cluster_fields_2d = [f for f in all_cluster_fields
                         if f.endswith('_2d')]
    cluster_fields_3d = [f for f in all_cluster_fields
                         if f.endswith('_3d')]
    cluster_fields_bare = [f for f in all_cluster_fields
                           if not f.endswith('_2d')
                           and not f.endswith('_3d')]
    has_dual_clusters = len(cluster_fields_2d) > 0

    # For backward compat, cluster_fields is the primary set (2d if dual,
    # bare otherwise)
    cluster_fields = cluster_fields_2d if has_dual_clusters \
        else cluster_fields_bare

    # Check for tree labels in metadata
    tree_labels_meta = None
    for mk in sorted(data['metadata'].keys()):
        if mk.startswith('tree_labels_depth_'):
            tree_labels_meta = json.loads(data['metadata'][mk])
            print(f"  Found tree labels: {mk}")
            break

    if not cluster_fields and tree_labels_meta:
        # Use tree labels to create cluster assignments
        print("  Building clusters from tree labels...")
        from dyf.lazy_index import LazyIndex
        with LazyIndex(dyf_path) as idx2:
            tree_struct = idx2.get_tree_structure()
        # Map each point to its ancestor at the child level
        child_labels_map = tree_labels_meta.get('child_labels', {})
        branch_labels_map = tree_labels_meta.get('branch_labels', {})
        # Build node_id→cluster_id mapping for labeled children
        labeled_nodes = sorted(child_labels_map.keys(), key=int)
        node_to_cluster = {int(nid): i for i, nid in enumerate(labeled_nodes)}
        # Map each point to its tree leaf, then walk up to a labeled node
        parent_of = {n['node_id']: n['parent_id'] for n in tree_struct}
        leaf_nodes = [n for n in tree_struct if n['is_leaf']]
        labels_birch = np.full(n, -1, dtype=np.int32)
        with LazyIndex(dyf_path) as idx2:
            for ln in leaf_nodes:
                if ln['batch_index'] < 0:
                    continue
                batch = idx2.get_leaf(ln['batch_index'])
                item_indices = batch.column('item_index').to_numpy()
                # Walk up to find a labeled ancestor
                nid = ln['node_id']
                while nid is not None:
                    if str(nid) in child_labels_map:
                        labels_birch[item_indices] = node_to_cluster[nid]
                        break
                    nid = parent_of.get(nid)
                else:
                    # Assign to nearest branch
                    nid = ln['node_id']
                    while nid is not None:
                        if str(nid) in branch_labels_map:
                            # Use branch as a fallback cluster
                            if nid not in node_to_cluster:
                                node_to_cluster[nid] = len(node_to_cluster)
                                child_labels_map[str(nid)] = \
                                    branch_labels_map[str(nid)]
                            labels_birch[item_indices] = node_to_cluster[nid]
                            break
                        nid = parent_of.get(nid)
        # Handle any unassigned points
        unassigned = labels_birch == -1
        if unassigned.any():
            fallback_id = max(node_to_cluster.values()) + 1
            labels_birch[unassigned] = fallback_id
            child_labels_map[str(fallback_id)] = "Other"
            node_to_cluster[-1] = fallback_id
        n_birch = len(set(labels_birch.tolist()))
        names_birch = {}
        for str_nid, cid in node_to_cluster.items():
            label = child_labels_map.get(str(str_nid), f"Cluster {cid}")
            names_birch[cid] = label
        print(f"  {n_birch} clusters from tree labels")
    elif not cluster_fields:
        # Level 1 only: do a quick inline BIRCH
        print("  No cluster fields found, running quick BIRCH...")
        target_k = getattr(args, 'n_clusters', 25)
        birch = fit_birch(coords[:, :2], target_k)
        labels_birch = birch.predict(coords[:, :2])
        n_birch = len(set(labels_birch.tolist()))
        names_birch = {i: f"Cluster {i}" for i in range(n_birch)}
    else:
        # Use the cluster level closest to --n-clusters
        target_k = getattr(args, 'n_clusters', 25)
        best_field = min(cluster_fields,
                         key=lambda f: abs(
                             _cluster_sort_key(f)[0] - target_k))
        best_k = _cluster_sort_key(best_field)[0]
        suffix = '_2d' if has_dual_clusters else ''
        print(f"  Using cluster level: {best_field}")

        labels_birch = np.asarray(data['fields'][best_field])
        n_birch = len(set(labels_birch.tolist()))

        # Load cluster names from metadata
        names_key = f'cluster_names_{best_k}{suffix}'
        names_json = data['metadata'].get(names_key, '{}')
        names_birch = {int(k): v for k, v in json.loads(names_json).items()}
        if not names_birch:
            names_birch = {i: f"Cluster {i}"
                           for i in sorted(set(labels_birch.tolist()))}

    # Multi-level data: from tree labels or cluster_* fields
    multi_level_data = None
    if tree_labels_meta and not cluster_fields:
        # Build two-level view from tree: branches (coarse) + children (fine)
        branch_labels_map = tree_labels_meta.get('branch_labels', {})
        # Coarse level: branch labels
        # Assign each point to its branch (grandparent of leaf)
        branch_labels = np.full(n, -1, dtype=np.int32)
        branch_nids = sorted(branch_labels_map.keys(), key=int)
        branch_to_id = {int(nid): i for i, nid in enumerate(branch_nids)}

        with LazyIndex(dyf_path) as idx2:
            tree_struct2 = idx2.get_tree_structure()
        parent_of2 = {n2['node_id']: n2['parent_id'] for n2 in tree_struct2}
        leaf_nodes2 = [n2 for n2 in tree_struct2 if n2['is_leaf']]

        with LazyIndex(dyf_path) as idx2:
            for ln in leaf_nodes2:
                if ln['batch_index'] < 0:
                    continue
                batch = idx2.get_leaf(ln['batch_index'])
                item_indices = batch.column('item_index').to_numpy()
                nid = ln['node_id']
                while nid is not None:
                    if str(nid) in branch_labels_map:
                        branch_labels[item_indices] = branch_to_id[nid]
                        break
                    nid = parent_of2.get(nid)

        unassigned = branch_labels == -1
        if unassigned.any():
            branch_labels[unassigned] = len(branch_to_id)

        branch_names = {i: branch_labels_map[nid]
                        for nid, i in branch_to_id.items()}
        n_branches = len(set(branch_labels.tolist()))

        ndim = coords.shape[1]
        ml_labels = {}
        ml_names = {}
        ml_label_data = {}

        # Coarse level (branches)
        branch_level_labels = []
        for cid in sorted(set(branch_labels.tolist())):
            mask = branch_labels == cid
            pts_idx = np.where(mask)[0]
            centroid = coords[pts_idx].mean(axis=0)
            name = branch_names.get(cid, f"Branch {cid}")
            branch_level_labels.append({
                "x": float(centroid[0]),
                "y": float(centroid[1]),
                "z": float(centroid[2]) if ndim >= 3 else 0.0,
                "text": str(name)[:50],
                "size": int(mask.sum()),
                "cid": int(cid),
                "leaf_cids": [int(cid)],
            })
        ml_labels[n_branches] = branch_labels
        ml_names[n_branches] = branch_names
        ml_label_data[n_branches] = branch_level_labels

        # Fine level (children = labels_birch from above)
        child_level_labels = []
        for cid in sorted(set(labels_birch.tolist())):
            mask = labels_birch == cid
            pts_idx = np.where(mask)[0]
            centroid = coords[pts_idx].mean(axis=0)
            name = names_birch.get(cid, f"Cluster {cid}")
            child_level_labels.append({
                "x": float(centroid[0]),
                "y": float(centroid[1]),
                "z": float(centroid[2]) if ndim >= 3 else 0.0,
                "text": str(name)[:50],
                "size": int(mask.sum()),
                "cid": int(cid),
                "leaf_cids": [int(cid)],
            })
        ml_labels[n_birch] = labels_birch
        ml_names[n_birch] = names_birch
        ml_label_data[n_birch] = child_level_labels

        multi_level_data = {
            "labels": ml_labels,
            "names": ml_names,
            "label_data": ml_label_data,
            "lsh_labels": None,
            "lsh_names": {},
            "lsh_label_data": [],
            "default": n_birch,
        }

        # Load pre-baked LSH buckets, or compute on the fly
        if 'lsh_bucket_ids' in data['fields']:
            lsh_labels = np.asarray(data['fields']['lsh_bucket_ids'],
                                    dtype=np.int32)
            lsh_names_json = data['metadata'].get('lsh_bucket_names', '{}')
            lsh_names = {int(k): v
                         for k, v in json.loads(lsh_names_json).items()}
            ndim_lsh = coords.shape[1]
            lsh_label_data = []
            for cid in sorted(set(lsh_labels.tolist())):
                mask = lsh_labels == cid
                pts = np.where(mask)[0]
                centroid = coords[pts].mean(axis=0)
                name = lsh_names.get(cid, f"Bucket {cid}")
                lsh_label_data.append({
                    "x": float(centroid[0]),
                    "y": float(centroid[1]),
                    "z": float(centroid[2]) if ndim_lsh >= 3 else 0.0,
                    "text": str(name)[:50],
                    "size": int(mask.sum()),
                    "cid": int(cid),
                    "leaf_cids": [int(cid)],
                })
            multi_level_data["lsh_labels"] = lsh_labels
            multi_level_data["lsh_names"] = lsh_names
            multi_level_data["lsh_label_data"] = lsh_label_data
            # Load pre-baked colors
            lsh_colors_json = data['metadata'].get('lsh_bucket_colors')
            if lsh_colors_json:
                multi_level_data.setdefault("color_maps", {})["lsh"] = {
                    str(k): v
                    for k, v in json.loads(lsh_colors_json).items()}
            print(f"  LSH: {len(set(lsh_labels.tolist()))} pre-baked buckets")
        else:
            print("  Computing agglomerated DYF tree buckets...")
            with LazyIndex(dyf_path) as idx_agg:
                lsh_labels, lsh_names, lsh_label_data, item_leaf_map, tree_struct = \
                    _agglomerate_tree_leaves(
                        idx_agg, coords, data['embeddings'], n_groups=50)
            if lsh_labels is not None:
                model = getattr(args, 'model', 'gpt-oss:20b')
                cache_file = getattr(args, 'label_cache', None)
                print("  Labeling agglomerated buckets...")
                bucket_names = label_clusters(
                    titles_arr, coords, lsh_labels, data['embeddings'],
                    model=model, cache_file=cache_file,
                    cache_key="lsh_buckets")
                lsh_names = bucket_names
                for entry in lsh_label_data:
                    cid = entry["cid"]
                    if cid in bucket_names:
                        entry["text"] = bucket_names[cid][:50]
                multi_level_data["lsh_labels"] = lsh_labels
                multi_level_data["lsh_names"] = lsh_names
                multi_level_data["lsh_label_data"] = lsh_label_data
                multi_level_data["item_leaf_map"] = item_leaf_map
                multi_level_data["tree_structure"] = tree_struct
                print(f"    {len(set(lsh_labels.tolist()))} agglomerated buckets")

    elif len(cluster_fields) > 1:
        ndim = coords.shape[1]
        suffix = '_2d' if has_dual_clusters else ''

        def _build_level_data(fields_list, name_suffix):
            """Build ml_labels/names/label_data from a list of cluster fields."""
            _labels = {}
            _names = {}
            _label_data = {}
            for cf in fields_list:
                m = _cluster_re.match(cf)
                assert m is not None
                lvl = int(m.group(1))
                arr = np.asarray(data['fields'][cf])
                _labels[lvl] = arr
                nk = f'cluster_names_{lvl}{name_suffix}'
                nj = data['metadata'].get(nk, '{}')
                _names[lvl] = {int(k): v
                               for k, v in json.loads(nj).items()}
                level_labels = []
                for cid in sorted(set(arr.tolist())):
                    mask = arr == cid
                    pts = np.where(mask)[0]
                    centroid = coords[pts].mean(axis=0)
                    name = _names[lvl].get(cid, f"Cluster {cid}")
                    level_labels.append({
                        "x": float(centroid[0]),
                        "y": float(centroid[1]),
                        "z": float(centroid[2]) if ndim >= 3 else 0.0,
                        "text": str(name)[:50],
                        "size": int(mask.sum()),
                        "cid": int(cid),
                        "leaf_cids": [int(cid)],
                    })
                _label_data[lvl] = level_labels
            return _labels, _names, _label_data

        ml_labels, ml_names, ml_label_data = _build_level_data(
            cluster_fields, suffix)

        multi_level_data = {
            "labels": ml_labels,
            "names": ml_names,
            "label_data": ml_label_data,
            "lsh_labels": None,
            "lsh_names": {},
            "lsh_label_data": [],
            "default": target_k,
        }

        # Load pre-baked LSH buckets, or compute on the fly
        if 'lsh_bucket_ids' in data['fields']:
            lsh_labels = np.asarray(data['fields']['lsh_bucket_ids'],
                                    dtype=np.int32)
            lsh_names_json = data['metadata'].get('lsh_bucket_names', '{}')
            lsh_names_agg = {int(k): v
                             for k, v in json.loads(lsh_names_json).items()}
            ndim_lsh = coords.shape[1]
            lsh_label_data = []
            for cid in sorted(set(lsh_labels.tolist())):
                mask = lsh_labels == cid
                pts = np.where(mask)[0]
                centroid = coords[pts].mean(axis=0)
                name = lsh_names_agg.get(cid, f"Bucket {cid}")
                lsh_label_data.append({
                    "x": float(centroid[0]),
                    "y": float(centroid[1]),
                    "z": float(centroid[2]) if ndim_lsh >= 3 else 0.0,
                    "text": str(name)[:50],
                    "size": int(mask.sum()),
                    "cid": int(cid),
                    "leaf_cids": [int(cid)],
                })
            multi_level_data["lsh_labels"] = lsh_labels
            multi_level_data["lsh_names"] = lsh_names_agg
            multi_level_data["lsh_label_data"] = lsh_label_data
            # Load pre-baked colors
            lsh_colors_json = data['metadata'].get('lsh_bucket_colors')
            if lsh_colors_json:
                multi_level_data.setdefault("color_maps", {})["lsh"] = {
                    str(k): v
                    for k, v in json.loads(lsh_colors_json).items()}
            print(f"  LSH: {len(set(lsh_labels.tolist()))} pre-baked buckets")
        else:
            print("  Computing agglomerated DYF tree buckets...")
            with LazyIndex(dyf_path) as idx_agg:
                lsh_labels, lsh_names_agg, lsh_label_data, item_leaf_map, tree_struct = \
                    _agglomerate_tree_leaves(idx_agg, coords, data['embeddings'],
                                             n_groups=50)
            if lsh_labels is not None:
                model = getattr(args, 'model', 'gpt-oss:20b')
                cache_file = getattr(args, 'label_cache', None)
                print("  Labeling agglomerated buckets...")
                bucket_names = label_clusters(
                    titles_arr, coords, lsh_labels, data['embeddings'],
                    model=model, cache_file=cache_file,
                    cache_key="lsh_buckets")
                lsh_names_agg = bucket_names
                for entry in lsh_label_data:
                    cid = entry["cid"]
                    if cid in bucket_names:
                        entry["text"] = bucket_names[cid][:50]
                multi_level_data["lsh_labels"] = lsh_labels
                multi_level_data["lsh_names"] = lsh_names_agg
                multi_level_data["lsh_label_data"] = lsh_label_data
                multi_level_data["item_leaf_map"] = item_leaf_map
                multi_level_data["tree_structure"] = tree_struct
                print(f"    {len(set(lsh_labels.tolist()))} agglomerated buckets")

        # Build 3D cluster data if dual clusters exist
        if has_dual_clusters and cluster_fields_3d:
            ml_labels_3d, ml_names_3d, ml_label_data_3d = \
                _build_level_data(cluster_fields_3d, '_3d')
            multi_level_data["labels_3d"] = ml_labels_3d
            multi_level_data["names_3d"] = ml_names_3d
            multi_level_data["label_data_3d"] = ml_label_data_3d
            multi_level_data["has_dual_clusters"] = True

        # Load pre-stored spatial color maps from .dyf metadata
        pre_color_maps = {}
        for lvl in sorted(multi_level_data["labels"].keys()):
            ckey = f'cluster_colors_{lvl}{suffix}'
            cjson = data['metadata'].get(ckey)
            if cjson:
                pre_color_maps[str(lvl)] = {
                    k: v for k, v in json.loads(cjson).items()}
        if pre_color_maps:
            existing = multi_level_data.get("color_maps", {})
            existing.update(pre_color_maps)
            multi_level_data["color_maps"] = existing

    # Edge data from metadata
    bundled_birch_2d = None
    bundled_birch_3d = None
    birch_pair_info = None
    edge_pairs_json = data['metadata'].get('edge_pairs')
    if edge_pairs_json:
        raw_pairs = json.loads(edge_pairs_json)
        birch_pair_info = {(p[0], p[1]): p[2] for p in raw_pairs}

        # 2D edge paths from metadata
        paths_2d_json = data['metadata'].get('edge_paths_2d')
        if paths_2d_json:
            paths_2d = json.loads(paths_2d_json)
            bundled_birch_2d = [
                np.array([[x, y, 0.0] for x, y in path], dtype=np.float32)
                for path in paths_2d
            ]

        # Generate 3D catenary paths from centroids
        n_cls = max(labels_birch) + 1
        centroids = np.zeros((n_cls, coords.shape[1]), dtype=np.float32)
        for c in range(n_cls):
            mask = labels_birch == c
            if mask.any():
                centroids[c] = coords[mask].mean(axis=0)
        edge_list = sorted(birch_pair_info.keys(),
                           key=lambda p: -birch_pair_info[p])
        max_count = max(birch_pair_info.values()) if birch_pair_info else 1
        bundled_birch_3d = []
        for c1, c2 in edge_list:
            if c1 >= n_cls or c2 >= n_cls:
                continue
            p1, p2 = centroids[c1], centroids[c2]
            count = birch_pair_info.get((c1, c2), 1)
            dist = np.linalg.norm(p2 - p1)
            strength = count / max_count
            sag = 0.15 * dist * (1.0 - 0.5 * strength)
            path = []
            for j in range(21):
                t = j / 20
                pt = p1 + t * (p2 - p1)
                pt = pt.copy()
                pt[2] += 4 * sag * t * (1 - t)
                path.append(pt.tolist())
            bundled_birch_3d.append(np.array(path, dtype=np.float32))

    # Tour narration from metadata
    narration_birch = {}
    narration_json = data['metadata'].get('tour_narration')
    if narration_json:
        raw = json.loads(narration_json)
        for k, v in raw.items():
            try:
                narration_birch[int(k)] = v
            except ValueError:
                narration_birch[k] = v  # "intro", "outro"

    # Generate audio if narration exists
    audio_birch = {}
    if narration_birch:
        audio_birch = generate_tour_audio(narration_birch)

    outdir = Path(dyf_path).parent
    display_title = getattr(args, 'title', None) or "Embedding Landscape"

    # Try pre-stored color map from .dyf metadata, fall back to computing
    colors_key = f'cluster_colors_{best_k}{suffix}' if cluster_fields else ''
    colors_json = data['metadata'].get(colors_key) if colors_key else None
    if colors_json:
        rgb_birch = {int(k): v for k, v in json.loads(colors_json).items()}
    else:
        rgb_birch = spatial_rgb_map(labels_birch, data['embeddings'])
    path_birch = str(outdir / "rog_3d_birch_clusters.html")
    subtitle = f"BIRCH — {n_birch} clusters, {n:,} pts (from .dyf)"

    build_pydeck(
        coords, titles_arr, labels_birch, rgb_birch,
        display_title, path_birch,
        cluster_names=names_birch,
        ws_port=getattr(args, 'port', 8766),
        bundled_edges_2d=bundled_birch_2d,
        bundled_edges_3d=bundled_birch_3d,
        edge_pairs=birch_pair_info,
        logo_path=getattr(args, 'logo', None),
        tour_narration=narration_birch,
        tour_audio=audio_birch,
        tour_title=getattr(args, 'title', None),
        subtitle_str=subtitle,
        multi_level_data=multi_level_data,
        embeddings=data['embeddings'],
    )
    print(f"\nWrote {path_birch}")
    subprocess.run(["open", path_birch])


def main():
    parser = argparse.ArgumentParser(
        description="Compare BIRCH (2D) vs DYF tree (high-D) clustering")
    parser.add_argument("parquet_path",
                        help="Path to embeddings parquet or enriched .dyf")
    parser.add_argument("--sample", type=int, default=8000)
    parser.add_argument("--n-clusters", type=int, default=25,
                        help="Target number of clusters")
    parser.add_argument("--dyf-depth", type=int, default=8,
                        help="DYF tree max depth")
    parser.add_argument("--dyf-bits", type=int, default=3,
                        help="DYF tree LSH bits per level")
    parser.add_argument("--refine", action=argparse.BooleanOptionalAction,
                        default=True,
                        help="Refine incoherent clusters (default: on)")
    parser.add_argument("--model", default="gpt-oss:20b",
                        help="Ollama model for cluster labeling (default: gpt-oss:20b)")
    parser.add_argument("--densmap", action="store_true",
                        help="Use densMAP for density-preserving projection")
    parser.add_argument("--no-edges", action="store_true",
                        help="Skip bridge edge bundling")
    parser.add_argument("--port", type=int, default=8766,
                        help="WebSocket port for viz_server bridge (default: 8766)")
    parser.add_argument("--cluster-2d", action="store_true",
                        help="Run BIRCH clustering on 2D projection (x,y only)")
    parser.add_argument("--pre-clusters", default=None,
                        help="NPY file with pre-computed cluster labels "
                             "(one int per parquet row, bypasses BIRCH)")
    parser.add_argument("--logo", default=None,
                        help="Path to logo image (PNG) to embed in header")
    parser.add_argument("--title", default=None,
                        help="Title text shown across top and as tour welcome label")
    parser.add_argument("--label-cache", default=None,
                        help="JSON file to cache cluster labels (avoids re-running Ollama)")
    parser.add_argument("--narrate", action="store_true",
                        help="Generate tour narration via Ollama (no narration file needed)")
    parser.add_argument("--rog-cache", default=None,
                        help="Path to ROG preprocessing cache (.pkl) for multi-level cluster toggle")
    parser.add_argument("--fisher-col", default=None,
                        help="Parquet column for Fisher dimension weighting "
                             "(e.g. gmdn_terms). Applied before UMAP and tree building.")
    args = parser.parse_args()

    # .dyf fast path: skip all compute, go straight to render
    if args.parquet_path.endswith('.dyf'):
        _render_from_dyf(args.parquet_path, args)
        return

    outdir = Path(args.parquet_path).parent
    pre_labels = None
    if args.pre_clusters:
        pre_labels = np.load(args.pre_clusters)
        print(f"  Loaded pre-computed clusters: {len(pre_labels)} labels, "
              f"{len(set(pre_labels))} unique")
    extra_cols = [args.fisher_col] if args.fisher_col else None
    titles, embeddings, pre_labels, extra_data = load_and_dedup(
        args.parquet_path, args.sample, pre_labels=pre_labels,
        extra_cols=extra_cols)

    # ── Optional Fisher dimension weighting ──────────────────────────────
    if args.fisher_col and args.fisher_col in extra_data:
        from dyf.fisher import extract_fisher_labels, compute_fisher_weights, apply_fisher_weights
        fisher_labels = extract_fisher_labels(extra_data[args.fisher_col])
        fisher_weights = compute_fisher_weights(embeddings, fisher_labels)
        embeddings = apply_fisher_weights(embeddings, fisher_weights)
        print(f"  Fisher weighting applied ({args.fisher_col}): "
              f"top-5 dims {np.argsort(fisher_weights)[-5:][::-1]}")

    n = len(titles)
    titles_arr = np.array(titles)
    target_k = args.n_clusters
    n_components = 3

    # ── Shared projection (DYF-parameterized UMAP) ───────────────────────
    print(f"\n=== DYF-parameterized UMAP ({n_components}D) ===")
    dyf_k = suggest_n_neighbors(embeddings)
    coords = run_umap(embeddings, n_neighbors=dyf_k, n_components=n_components,
                       densmap=args.densmap)
    coords = orient_landscape(coords)

    # ── Clustering on projected coords ────────────────────────────────────
    cluster_coords = coords[:, :2] if args.cluster_2d else coords
    cluster_dim = 2 if args.cluster_2d else n_components

    if pre_labels is not None:
        # Use pre-computed cluster labels (e.g. from HDBSCAN pipeline)
        labels_birch = pre_labels.astype(int)
        # Relabel to contiguous 0-based IDs
        old_ids = sorted(set(labels_birch))
        id_map = {old: new for new, old in enumerate(old_ids)}
        labels_birch = np.array([id_map[l] for l in labels_birch])
        n_birch = len(set(labels_birch))
        print(f"\nUsing pre-computed clusters: {n_birch} clusters, {n} points")

        # Build a simple single-level hierarchy (no BIRCH subclusters)
        birch_levels = {}
    else:
        print(f"\nFitting BIRCH (target_k={target_k}) on {cluster_dim}D coords...")
        birch = fit_birch(cluster_coords, target_k)
        labels_birch = birch.predict(cluster_coords)
        n_birch = len(set(labels_birch))
        print(f"  BIRCH: {n_birch} clusters")

        # Merge tiny clusters (< 0.5% of total) into nearest large neighbor
        min_cluster_size = max(10, int(len(labels_birch) * 0.005))
        from collections import Counter
        cluster_counts = Counter(labels_birch)
        tiny_cids = {cid for cid, cnt in cluster_counts.items()
                     if cnt < min_cluster_size}
        if tiny_cids:
            big_cids = sorted(set(labels_birch) - tiny_cids)
            big_centroids = {}
            for cid in big_cids:
                mask = labels_birch == cid
                big_centroids[cid] = cluster_coords[mask].mean(axis=0)
            for tcid in sorted(tiny_cids):
                mask = labels_birch == tcid
                centroid = cluster_coords[mask].mean(axis=0)
                dists = {cid: np.linalg.norm(centroid - c)
                         for cid, c in big_centroids.items()}
                nearest = min(dists, key=lambda k: dists[k])
                labels_birch[mask] = nearest
                print(f"  Merged tiny cluster {tcid} "
                      f"({cluster_counts[tcid]} pts) -> {nearest}")
            old_ids = sorted(set(labels_birch))
            id_map = {old: new for new, old in enumerate(old_ids)}
            labels_birch = np.array([id_map[l] for l in labels_birch])
            n_birch = len(set(labels_birch))
            print(f"  After merge: {n_birch} clusters")

        # Multi-level label hierarchy from BIRCH subclusters
        print(f"\nBuilding label hierarchy from BIRCH subclusters...")
        hierarchy_model = args.model
        birch_levels = build_label_hierarchy(
            coords, titles_arr, birch, embeddings=embeddings,
            model=hierarchy_model, base_labels=labels_birch,
            target_k=target_k,
            cluster_data=cluster_coords if args.cluster_2d else None,
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

    # ── Contrastive cluster labeling via LLM ─────────────────────────────
    label_cache = getattr(args, 'label_cache', None)
    print("\n=== Labeling BIRCH clusters ===")
    names_birch = label_clusters(
        titles, coords, labels_birch, embeddings, model=args.model,
        cache_file=label_cache, cache_key="birch")
    print("\n=== Labeling DYF tree clusters ===")
    names_dyf = label_clusters(
        titles, coords, labels_dyf, embeddings, model=args.model,
        cache_file=label_cache, cache_key="dyf")

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
                "cid": int(cid),
                "leaf_cids": [int(cid)],
            })
        birch_levels[n_birch] = base_level
        print(f"  Added base BIRCH level ({n_birch}): {len(base_level)} labels")

    # ── Bridge edge bundling (2D bundled + 3D catenary) ─────────────────
    bundled_birch_2d = None
    bundled_birch_3d = None
    birch_pair_info = None
    if not args.no_edges:
        print("\n=== Computing bridge edges (BIRCH) ===")
        bundled_birch_2d, bundled_birch_3d, birch_pair_info, _ = compute_bridge_edges_3d(
            coords, embeddings, labels_birch, n_birch)
        if bundled_birch_2d:
            print(f"  Bundled {len(bundled_birch_2d)} 2D + {len(bundled_birch_3d)} 3D bridge edges")

    # ── Load ROG cache for multi-level cluster toggle ───────────────────
    multi_level_data = None
    if args.rog_cache:
        import pickle
        cache_path = Path(args.rog_cache)
        if not cache_path.exists():
            print(f"WARNING: --rog-cache {cache_path} not found, skipping multi-level")
        else:
            print(f"\n=== Loading ROG cache: {cache_path} ===")
            with open(cache_path, "rb") as f:
                rog_cache = pickle.load(f)
            cr = rog_cache['cluster_result']
            lsh = rog_cache['lsh_data']
            cache_n = len(cr['labels'][next(iter(cr['labels']))])

            # Provenance check — fail loud on mismatch
            cache_prov_dict = rog_cache.get('_provenance')
            if cache_prov_dict:
                prov = provenance_from_dict(cache_prov_dict)
                ok, warnings = check_compatible(
                    prov,
                    downstream_n_items=n,
                    downstream_sample_n=args.sample,
                )
                if not ok:
                    print(f"\n  ERROR: ROG cache is incompatible with current pipeline:")
                    for w in warnings:
                        print(f"    - {w}")
                    print(f"  Cache: sample={prov.sample_n}, n_items={prov.n_items}")
                    print(f"  Current: sample={args.sample}, n_items={n}")
                    print(f"  Rebuild ROG cache with matching parameters.")
                    import sys
                    sys.exit(1)
            elif cache_n != n:
                # Fallback for caches without provenance
                print(f"  WARNING: ROG cache has {cache_n} points, "
                      f"current pipeline has {n}. Use --sample 0 or "
                      f"matching --sample to align point counts.")

            # Extract multi-level cluster labels
            ml_labels = {}
            ml_names = {}
            for lvl in sorted(cr['labels'].keys()):
                arr = np.asarray(cr['labels'][lvl])
                if len(arr) == n:
                    ml_labels[lvl] = arr
                    # names can be list (indexed by position) or dict (keyed by cid)
                    raw_names = cr['names'].get(lvl, {})
                    if isinstance(raw_names, list):
                        ml_names[lvl] = {i: name for i, name in enumerate(raw_names)}
                    else:
                        ml_names[lvl] = raw_names
                    print(f"  Level {lvl}: {len(set(arr.tolist()))} clusters")
                else:
                    print(f"  Level {lvl}: length mismatch ({len(arr)} vs {n}), skipping")

            # Extract LSH bucket IDs → remap sparse uint64 to contiguous int32
            raw_bids = np.asarray(lsh['bucket_ids'], dtype=np.int64)
            if len(raw_bids) == n:
                unique_bids = sorted(set(raw_bids.tolist()))
                bid_remap = {old: new for new, old in enumerate(unique_bids)}
                lsh_labels = np.array([bid_remap[int(b)] for b in raw_bids], dtype=np.int32)
                lsh_names = {}
                for old_bid, name in lsh.get('bucket_names', {}).items():
                    new_bid = bid_remap.get(int(old_bid))
                    if new_bid is not None:
                        lsh_names[new_bid] = name
                print(f"  LSH: {len(unique_bids)} buckets")
            else:
                lsh_labels = None
                lsh_names = {}
                print(f"  LSH: length mismatch ({len(raw_bids)} vs {n}), skipping")

            # Build label_data dicts for each level (centroid + name + size)
            ndim = coords.shape[1]
            ml_label_data = {}
            for lvl, arr in ml_labels.items():
                names_dict = ml_names[lvl]
                level_labels = []
                for cid in sorted(set(arr.tolist())):
                    mask = arr == cid
                    pts = np.where(mask)[0]
                    centroid = coords[pts].mean(axis=0)
                    name = names_dict.get(cid, f"Cluster {cid}")
                    if isinstance(name, list):
                        name = name[0] if name else f"Cluster {cid}"
                    level_labels.append({
                        "x": float(centroid[0]),
                        "y": float(centroid[1]),
                        "z": float(centroid[2]) if ndim >= 3 else 0.0,
                        "text": str(name)[:50],
                        "size": int(mask.sum()),
                        "cid": int(cid),
                        "leaf_cids": [int(cid)],
                    })
                ml_label_data[lvl] = level_labels

            # LSH label data
            lsh_label_data = []
            if lsh_labels is not None:
                for cid in sorted(set(lsh_labels.tolist())):
                    mask = lsh_labels == cid
                    pts = np.where(mask)[0]
                    centroid = coords[pts].mean(axis=0)
                    name = lsh_names.get(cid, f"Bucket {cid}")
                    lsh_label_data.append({
                        "x": float(centroid[0]),
                        "y": float(centroid[1]),
                        "z": float(centroid[2]) if ndim >= 3 else 0.0,
                        "text": str(name)[:50],
                        "size": int(mask.sum()),
                        "cid": int(cid),
                        "leaf_cids": [int(cid)],
                    })

            multi_level_data = {
                "labels": ml_labels,          # {5: ndarray, 12: ..., 25: ..., 50: ...}
                "names": ml_names,            # {5: {cid: name}, ...}
                "label_data": ml_label_data,  # {5: [label_dicts], ...}
                "lsh_labels": lsh_labels,     # ndarray or None
                "lsh_names": lsh_names,       # {cid: name}
                "lsh_label_data": lsh_label_data,
                "default": target_k,
            }
            print(f"  Multi-level data ready (default level: {target_k})")

    rgb_birch = spatial_rgb_map(labels_birch, embeddings)
    rgb_dyf = spatial_rgb_map(labels_dyf.tolist(), embeddings)

    # Compute cluster shape analysis for narration context
    print("\n=== Computing cluster shapes ===")
    coords_2d = coords[:, :2]
    shapes_birch = compute_cluster_shapes(coords_2d, titles, labels_birch)
    shapes_dyf = compute_cluster_shapes(coords_2d, titles, labels_dyf)
    print(f"  BIRCH: {sum(1 for s in shapes_birch.values() if s['shape'] != 'compact')}/{len(shapes_birch)} non-compact clusters")
    print(f"  DYF: {sum(1 for s in shapes_dyf.values() if s['shape'] != 'compact')}/{len(shapes_dyf)} non-compact clusters")

    # Build callout data for tour visualization
    def _build_callouts(shapes):
        callouts = {}
        for cid, s in shapes.items():
            indices = []
            labels_list = []
            # Core representative points only (nearest to centroid = best examples)
            for idx, title in zip(s['core_indices'][:5], s['core_titles'][:5]):
                indices.append(idx)
                labels_list.append(title[:40])
            if indices:
                callouts[cid] = {"indices": indices, "labels": labels_list}
        return callouts

    callouts_birch = _build_callouts(shapes_birch)
    callouts_dyf = _build_callouts(shapes_dyf)

    # Generate tour narration for TTS
    if args.narrate:
        # Inline narration via Ollama — no pre-written file needed
        narration_birch = _generate_narration_ollama(
            names_birch, titles, labels_birch, coords,
            model=args.model, title=args.title)
        narration_dyf = _generate_narration_ollama(
            names_dyf, titles, labels_dyf, coords,
            model=args.model, title=args.title)
    else:
        # Look for narration file next to the input parquet or in demo/
        narration_file = None
        for candidate in [
            Path(args.parquet_path).parent / "tour_narration.json",
            outdir / "tour_narration.json",
        ]:
            if candidate.exists():
                narration_file = str(candidate)
                break

        narration_birch = generate_tour_narration(
            names_birch, titles, labels_birch,
            title=args.title, narration_file=narration_file)
        narration_dyf = generate_tour_narration(
            names_dyf, titles, labels_dyf,
            title=args.title, narration_file=narration_file)

    # Generate audio for tour narration
    audio_birch = generate_tour_audio(narration_birch)
    audio_dyf = generate_tour_audio(narration_dyf)

    path_birch = str(outdir / "rog_3d_birch_clusters.html")
    path_dyf = str(outdir / "rog_3d_dyf_tree_clusters.html")

    birch_title = args.title or "GUDID Energy Devices"
    birch_subtitle = f"BIRCH on 3D (k={dyf_k} UMAP) — {n_birch} clusters, {n:,} pts"
    dyf_title = args.title or "GUDID Energy Devices"
    dyf_subtitle = (
        f"DYF Tree (depth={args.dyf_depth}, bits={args.dyf_bits}) "
        f"— {n_dyf} clusters, {n:,} pts"
    )

    build_pydeck(
        coords, titles_arr, labels_birch, rgb_birch,
        birch_title,
        path_birch, cluster_names=names_birch, ws_port=args.port,
        label_levels=birch_levels,
        bundled_edges_2d=bundled_birch_2d,
        bundled_edges_3d=bundled_birch_3d,
        edge_pairs=birch_pair_info,
        logo_path=args.logo,
        tour_narration=narration_birch,
        tour_audio=audio_birch,
        tour_callouts=callouts_birch,
        tour_title=args.title,
        subtitle_str=birch_subtitle,
        multi_level_data=multi_level_data,
        embeddings=embeddings,
    )
    build_pydeck(
        coords, titles_arr, labels_dyf, rgb_dyf,
        dyf_title,
        path_dyf, cluster_names=names_dyf, ws_port=args.port,
        label_levels=birch_levels,
        bundled_edges_2d=bundled_birch_2d,
        bundled_edges_3d=bundled_birch_3d,
        edge_pairs=birch_pair_info,
        logo_path=args.logo,
        tour_narration=narration_dyf,
        tour_audio=audio_dyf,
        tour_callouts=callouts_dyf,
        subtitle_str=dyf_subtitle,
        tour_title=args.title,
        multi_level_data=multi_level_data,
        embeddings=embeddings,
    )

    subprocess.run(["open", path_birch])
    subprocess.run(["open", path_dyf])


if __name__ == "__main__":
    main()
