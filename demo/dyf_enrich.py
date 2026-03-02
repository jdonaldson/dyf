"""
Enrich a .dyf file through progressive levels:

    Level 0 → 1 (project): Add UMAP coordinates as stored fields
    Level 1 → 2 (cluster): Add BIRCH cluster labels + LLM names
    Level 2 → 3 (viz):     Add bridge edges + tour narration/audio

Usage:
    python demo/dyf_enrich.py project demo/gudid_50k_titled.dyf
    python demo/dyf_enrich.py cluster demo/gudid_50k_titled.dyf
    python demo/dyf_enrich.py viz demo/gudid_50k_titled.dyf
    python demo/dyf_enrich.py all demo/gudid_50k_titled.dyf
"""

import argparse
import json
import math
import re
import time
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from dyf.colors import spatial_rgb_map
from dyf.lazy_index import LazyIndex, rewrite_lazy_index
from dyf.provenance import create_provenance, provenance_to_dict


# ── UMAP projection (Level 0 → 1) ──────────────────────────────────────


def suggest_n_neighbors(embeddings, num_bits=12, min_k=15, max_k=100):
    """Use DYF LSH bucket density to suggest UMAP n_neighbors."""
    from dyf_rs import DensityClassifier

    clf = DensityClassifier(
        embedding_dim=embeddings.shape[1], num_bits=num_bits, seed=42)
    clf.fit(embeddings)
    bucket_sizes = np.array(clf.get_bucket_sizes())
    mean_size = bucket_sizes.mean()
    suggested = int(np.clip(mean_size, min_k, max_k))
    n_buckets = len(set(clf.get_bucket_ids()))
    print(f"  DYF: {n_buckets} buckets, mean_size={mean_size:.0f}, "
          f"suggested n_neighbors={suggested}")
    return suggested


def run_umap(embeddings, n_neighbors=15, n_components=3, densmap=False):
    """Run UMAP and return median-centered, MAD-scaled coords."""
    import umap
    from sklearn.neighbors import NearestNeighbors

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
    if np.ptp(rot[:, 1]) > np.ptp(rot[:, 0]):
        c2, s2 = np.cos(np.pi / 2), np.sin(np.pi / 2)
        rot = rot @ np.array([[c2, s2], [-s2, c2]])
    out = coords.copy()
    out[:, :2] = rot
    xr = np.ptp(out[:, 0])
    yr = np.ptp(out[:, 1])
    print(f"    Landscape orient: rotated {np.degrees(theta):.1f}°, "
          f"spread X={xr:.2f} Y={yr:.2f} (ratio {xr / yr:.2f})")
    return out


def enrich_project(dyf_path, n_components=3, densmap=False, output_path=None,
                   fisher_col=None, fisher_parquet=None,
                   diagnose_parquet=None):
    """Add UMAP coordinates to a .dyf file (Level 0 → 1).

    If fisher_col is set, loads the column from fisher_parquet (or falls
    back to stored fields in the .dyf), computes sqrt(Fisher ratio) weights,
    and applies them to embeddings before UMAP.
    """
    print(f"\n=== Level 1: UMAP Projection ===")
    print(f"  Input: {dyf_path}")

    with LazyIndex(dyf_path) as idx:
        level = idx.detect_enrichment_level()
        if level >= 1:
            print(f"  Already at level {level} (has UMAP coords), skipping.")
            return
        n = idx.total_items
        print(f"  {n:,} items, dim={idx.embedding_dim}")

    # Extract embeddings
    with LazyIndex(dyf_path) as idx:
        data = idx.extract_all_fields()
    embeddings = data['embeddings']

    # Optional Fisher dimension weighting
    fisher_weights = None
    if fisher_col:
        from dyf.fisher import extract_fisher_labels, compute_fisher_weights, apply_fisher_weights
        import polars as pl

        if fisher_parquet:
            df = pl.read_parquet(fisher_parquet)
            if fisher_col in df.columns:
                raw_vals = df[fisher_col].to_list()
            else:
                print(f"  WARNING: column '{fisher_col}' not in {fisher_parquet}, "
                      f"skipping Fisher weighting")
                raw_vals = None
        elif fisher_col in data.get('stored_fields', {}):
            raw_vals = data['stored_fields'][fisher_col]
        else:
            print(f"  WARNING: fisher_col='{fisher_col}' not found, "
                  f"skipping Fisher weighting")
            raw_vals = None

        if raw_vals is not None:
            fisher_labels = extract_fisher_labels(raw_vals)
            fisher_weights = compute_fisher_weights(embeddings, fisher_labels)
            embeddings = apply_fisher_weights(embeddings, fisher_weights)
            print(f"  Fisher weighting applied ({fisher_col}): "
                  f"top-5 dims {np.argsort(fisher_weights)[-5:][::-1]}")

    # Optional axis diagnostics sanity check
    if diagnose_parquet:
        import polars as pl
        from dyf.categorical import discover_categorical_columns, diagnose_axes

        diag_path = Path(diagnose_parquet)
        if diag_path.exists():
            diag_df = pl.read_parquet(diag_path)
            label_cols = discover_categorical_columns(diag_df, text_col="text")
            if label_cols:
                diags = diagnose_axes(embeddings, label_cols)
                print(f"  Axis diagnostics ({len(diags)} axes):")
                for d in diags:
                    flag = " ⚠ UNDER-SERVED" if d.lift < 3.0 else ""
                    print(f"    {d.name}: lift={d.lift:.1f}x  "
                          f"purity={d.knn_purity:.3f}{flag}")
                under = [d for d in diags if d.lift < 3.0]
                if under:
                    print(f"  WARNING: {len(under)} axis(es) under-served. "
                          f"Consider re-embedding with --diagnose in gudid_embeddings.py")
        else:
            print(f"  WARNING: --diagnose-parquet={diag_path} not found, skipping")

    # Compute UMAP
    dyf_k = suggest_n_neighbors(embeddings)
    coords = run_umap(embeddings, n_neighbors=dyf_k,
                       n_components=n_components, densmap=densmap)
    coords = orient_landscape(coords)

    # Write back
    new_sf = {
        'umap_x': coords[:, 0].astype(np.float32),
        'umap_y': coords[:, 1].astype(np.float32),
        'umap_z': (coords[:, 2].astype(np.float32) if n_components >= 3
                   else np.zeros(len(coords), dtype=np.float32)),
    }
    new_meta = {
        'umap_n_neighbors': str(dyf_k),
        'umap_n_components': str(n_components),
        'umap_densmap': str(densmap).lower(),
    }
    if fisher_weights is not None:
        new_meta['fisher_col'] = fisher_col
        new_meta['fisher_weights'] = json.dumps(fisher_weights.tolist())
        # Store a CategoryGraph for downstream multi-level use
        from dyf.categorical import CategoryGraph, store_category_graph
        graph = CategoryGraph.from_single_level(fisher_labels)
        new_meta.update(store_category_graph(graph, fisher_col))

    # Stamp provenance for Level 1
    new_meta['_provenance_level_1'] = json.dumps(provenance_to_dict(
        create_provenance(
            artifact_type="dyf",
            n_items=len(embeddings),
            source_paths=[str(dyf_path)],
            params={"n_components": n_components, "densmap": densmap,
                    "fisher_col": fisher_col},
        )
    ))

    out = output_path or dyf_path
    print(f"  Writing enriched file: {out}")
    rewrite_lazy_index(dyf_path, new_stored_fields=new_sf,
                       new_metadata=new_meta, output_path=out)
    print(f"  Done. Level 0 → 1")


# ── BIRCH clustering (Level 1 → 2) ─────────────────────────────────────


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

    # Relabel to contiguous 0-based IDs
    old_ids = sorted(set(labels.tolist()))
    id_map = {old: new for new, old in enumerate(old_ids)}
    labels = np.array([id_map[l] for l in labels])
    return labels


def transfer_labels_majority_vote(labels_primary, names_primary,
                                   labels_secondary):
    """Transfer cluster names from primary to secondary via majority vote.

    For each secondary cluster, count which primary cluster owns the most
    members. Assign the winning primary cluster's name. Disambiguate
    collisions by appending a numeric suffix: "Forceps (2)".

    Args:
        labels_primary: int array, per-point cluster IDs (e.g. 2D clusters).
        names_primary: dict {cluster_id: name_str} for primary clusters.
        labels_secondary: int array, per-point cluster IDs (e.g. 3D clusters).

    Returns:
        dict {secondary_cluster_id: name_str}
    """
    labels_p = np.asarray(labels_primary)
    labels_s = np.asarray(labels_secondary)

    secondary_names = {}
    for s_cid in sorted(set(labels_s.tolist())):
        mask = labels_s == s_cid
        primary_ids = labels_p[mask]
        most_common = Counter(primary_ids.tolist()).most_common(1)[0][0]
        secondary_names[s_cid] = names_primary.get(most_common,
                                                    f"Cluster {s_cid}")

    # Disambiguate collisions with numeric suffix
    name_counts = Counter(secondary_names.values())
    duplicates = {name for name, cnt in name_counts.items() if cnt > 1}
    if duplicates:
        seen = {}  # name -> running counter
        for s_cid in sorted(secondary_names.keys()):
            name = secondary_names[s_cid]
            if name in duplicates:
                idx = seen.get(name, 0) + 1
                seen[name] = idx
                if idx > 1:
                    secondary_names[s_cid] = f"{name} ({idx})"

    return secondary_names


# ── Contrastive cluster labeling ────────────────────────────────────────


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
        wc = cluster_word_counts[cid]
        total = sum(wc.values())
        if total == 0:
            cluster_keywords[cid] = []
            continue
        scores = []
        for word in vocab:
            tf = wc.get(word, 0) / total
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
    """Farthest-point sampling in projection space."""
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
                axis=2),
            axis=1)
        dists[chosen] = -1
        chosen.append(int(np.argmax(dists)))
    return pts[chosen].tolist()


def _get_cluster_path_context(point_indices, split_keywords, titles,
                              sample_size=50, top_k=3):
    """Get the majority tree path context for a cluster's points.

    Samples points from the cluster, traces each through the split tree,
    and returns the most common path as a readable string.

    Args:
        point_indices: List of item indices in this cluster.
        split_keywords: Parsed split_keywords metadata dict.
        titles: Full titles list (unused but kept for API consistency).
        sample_size: How many points to sample for majority vote.
        top_k: Keywords per path step.

    Returns:
        String like "screw,plate,fixation → pedicle,cervical,spine" or ""
    """
    splits = split_keywords.get('splits', {})
    if not splits:
        return ""

    # Sample a subset for efficiency
    rng = np.random.default_rng(42)
    pts = np.asarray(point_indices)
    if len(pts) > sample_size:
        pts = rng.choice(pts, size=sample_size, replace=False)

    # For each split node, find which child each point belongs to
    # by checking the 'count' ranges. We trace majority child per split.
    # Build a node → child_id mapping for each point by checking membership.
    split_votes: dict[str, Counter] = defaultdict(Counter)

    for nid_str, split in splits.items():
        children = split.get('children', {})
        if not children:
            continue

        # Build a quick lookup: which points belong to which child
        # We don't have the tree/leaf_batches here, so use a heuristic:
        # count how many sample points have titles matching each child's
        # top keywords.
        for cid_str, cinfo in children.items():
            unigrams = [w for w, _ in cinfo.get('unigrams', [])[:top_k]]
            if not unigrams:
                continue
            # Count how many sample points match this child's keywords
            kw_set = set(unigrams)
            match_count = 0
            for idx in pts:
                if idx < len(titles):
                    words = set(re.findall(r'[a-z]{3,}', titles[idx].lower()))
                    if words & kw_set:
                        match_count += 1
            split_votes[nid_str][cid_str] = match_count

    # Build path from the winning child at each depth
    path_steps = []
    # Sort splits by depth
    sorted_splits = sorted(
        splits.items(),
        key=lambda x: x[1].get('depth', 0)
    )
    for nid_str, split in sorted_splits:
        votes = split_votes.get(nid_str)
        if not votes:
            continue
        # Pick the child with most keyword matches
        winner = votes.most_common(1)[0][0]
        children = split.get('children', {})
        winner_info = children.get(winner, {})
        words = [w for w, _ in winner_info.get('unigrams', [])[:top_k]]
        if words:
            path_steps.append(','.join(words))

    if not path_steps:
        return ""
    return ' → '.join(path_steps)


def label_clusters(titles, coords, labels, embeddings, model="gemma2:9b",
                   n_samples=20, cache_file=None, cache_key=None,
                   cache_data=None, split_keywords=None,
                   path_labels=None, sibling_keywords=None):
    """Label clusters via contrastive TF-IDF + local Ollama LLM.

    Label context priority (highest first):
    1. ``path_labels`` + ``sibling_keywords`` from cluster-tree DAG
    2. ``split_keywords`` tree path heuristic (deprecated)
    3. Contrastive TF-IDF vs nearest neighbor (baseline fallback)
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from collections import Counter

    unique_labels = sorted(set(int(l) for l in labels))

    # Check cache: cache_data (in-memory) → cache_file (on-disk)
    _cache_key = cache_key or "default"
    if cache_data is not None:
        cached = cache_data.get(_cache_key, {})
        if cached and len(cached) == len(unique_labels):
            cluster_names = {int(k): v for k, v in cached.items()}
            print(f"  Loaded {len(cluster_names)} labels from cache")
            return cluster_names
    elif cache_file:
        cache_path = Path(cache_file)
        if cache_path.exists():
            file_cache = json.loads(cache_path.read_text())
            cached = file_cache.get(_cache_key, {})
            if cached and len(cached) == len(unique_labels):
                cluster_names = {int(k): v for k, v in cached.items()}
                print(f"  Loaded {len(cluster_names)} labels from cache")
                return cluster_names

    n_clusters = len(unique_labels)
    label_arr = np.asarray(labels)

    cluster_points = defaultdict(list)
    for i, cid in enumerate(label_arr):
        cluster_points[int(cid)].append(i)

    # High-D centroids
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

        # Build context string — priority:
        # 1. DAG path labels + sibling keywords (best)
        # 2. Split keyword tree path heuristic (deprecated)
        # 3. Contrastive TF-IDF vs nearest neighbor (baseline)
        kw_str = ""
        if path_labels is not None and sibling_keywords is not None:
            from dyf.cluster_tree import format_cluster_context
            pl = path_labels.get(cid, "")
            sk = sibling_keywords.get(cid, [])
            ctx = format_cluster_context(cid, pl, sk)
            if ctx:
                kw_str = f"\n{ctx}"

        if not kw_str and split_keywords:
            # Deprecated: heuristic tree path via keyword matching
            path_context = _get_cluster_path_context(
                pts, split_keywords, titles)
            if path_context:
                kw_str = (f"\nTree path context (root → leaf): "
                          f"{path_context}")

        if not kw_str:
            # Fallback: contrastive TF-IDF vs nearest neighbor
            nearest_idx = _find_nearest_cluster(cid_to_idx[cid], hd_centroids)
            nearest_cid = unique_labels[nearest_idx]
            neighbor_pts = cluster_points[nearest_cid]

            if neighbor_pts:
                combined = ([titles[p] for p in pts]
                            + [titles[p] for p in neighbor_pts])
                combined_labels = np.zeros(
                    len(pts) + len(neighbor_pts), dtype=int)
                combined_labels[len(pts):] = 1
                kw = _compute_tfidf_keywords(combined, combined_labels, 2,
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
            "GOOD labels: \"Spinal Fixation Screws\", "
            "\"Dental Crowns & Bridges\", "
            "\"Compression Stockings\", \"Hearing Aid Components\"\n\n"
            "Reply with ONLY the label, nothing else."
        )
        tasks.append((cid, prompt))

    cluster_names = {cid: f"Cluster {cid}" for cid in unique_labels}

    def _label_one(task):
        cid, prompt = task
        resp = _call_ollama(model, prompt)
        label = resp.split('\n')[0][:50].strip('"\'').strip()
        return cid, label if label else f"Cluster {cid}"

    n_workers = min(4, len(tasks))
    completed = 0
    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        futures = {executor.submit(_label_one, t): t for t in tasks}
        for future in as_completed(futures):
            cid, label = future.result()
            cluster_names[cid] = label
            completed += 1
            if completed % 5 == 0 or completed == len(tasks):
                print(f"    Labeled {completed}/{len(tasks)}...", flush=True)

    # Re-label duplicates with sibling context
    label_counts = Counter(cluster_names.values())
    duplicates = {lbl for lbl, cnt in label_counts.items() if cnt > 1}
    if duplicates:
        dup_cids = [c for c in unique_labels if cluster_names[c] in duplicates]
        taken = sorted(set(cluster_names.values()))
        print(f"    Re-labeling {len(dup_cids)} duplicates...")
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
            siblings = [l for l in taken if l != cluster_names[cid]]
            sibling_str = ", ".join(f'"{s}"' for s in siblings[:15])
            prompt = (
                f"You are labeling clusters in an embedding space. "
                f"This cluster has {len(pts)} items.\n"
                f"Sample items:\n"
                + "\n".join(f"- {t}" for t in sample_titles) + "\n\n"
                f"These labels are ALREADY TAKEN: {sibling_str}\n\n"
                "Give a short (2-5 word) label DIFFERENT from all taken "
                "labels. Reply with ONLY the label, nothing else."
            )
            dup_tasks.append((cid, prompt))
        with ThreadPoolExecutor(max_workers=min(4, len(dup_tasks))) as executor:
            futures = {executor.submit(_label_one, t): t for t in dup_tasks}
            for future in as_completed(futures):
                cid, label = future.result()
                cluster_names[cid] = label

    for cid in unique_labels:
        n_pts = len(cluster_points[cid])
        print(f"    [{cid:2d}] {cluster_names[cid]:<35s} ({n_pts} pts)")

    # Save to cache (only write file if cache_file was used, not cache_data)
    if cache_file and cache_data is None:
        cache_path = Path(cache_file)
        file_cache = {}
        if cache_path.exists():
            file_cache = json.loads(cache_path.read_text())
        file_cache[_cache_key] = {
            str(k): v for k, v in cluster_names.items()
        }
        cache_path.write_text(json.dumps(file_cache, indent=2))
        print(f"  Saved labels to cache ({cache_file})")

    return cluster_names


# ── Bottom-up tree labeling ─────────────────────────────────────────


def _collect_descendant_indices(node_id, children_of, leaf_batches):
    """Recursively collect all item indices under a node."""
    if node_id in leaf_batches:
        return leaf_batches[node_id]
    kids = children_of.get(node_id, [])
    if not kids:
        return np.array([], dtype=int)
    return np.concatenate([
        _collect_descendant_indices(k, children_of, leaf_batches)
        for k in kids
    ])


def _call_ollama(model, prompt, timeout=300):
    """Call Ollama HTTP API and return text response."""
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
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode())
            return data.get("response", "").strip()
    except Exception as e:
        print(f"    Ollama error: {e}")
        return ""


def label_tree_bottomup(idx, titles, model="gemma2:9b", target_depth=3,
                        samples_per_child=8, min_child_size=20,
                        cache_file=None, cache_data=None):
    """Label tree nodes bottom-up using the DYF tree hierarchy.

    For each internal node at target_depth, samples titles from each child,
    presents them to the LLM, and asks for:
      - One specific label per child
      - One summary label for the parent branch

    Args:
        idx: LazyIndex (open).
        titles: List/array of titles indexed by item_index.
        model: Ollama model name.
        target_depth: Depth of nodes to label (3 = first split = 16 nodes).
        samples_per_child: How many titles to sample from each child.
        min_child_size: Skip children smaller than this.
        cache_file: Optional JSON cache path.
        cache_data: Optional in-memory cache dict (takes priority over cache_file).

    Returns:
        dict with:
            branch_labels: {node_id: label} for internal nodes at target_depth
            child_labels: {node_id: label} for their children
            hierarchy: {branch_node_id: [child_node_ids]} mapping
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    tree = idx.get_tree_structure()
    by_id = {n['node_id']: n for n in tree}

    # Build parent→children mapping
    children_of = defaultdict(list)
    for n in tree:
        if n['parent_id'] is not None:
            children_of[n['parent_id']].append(n['node_id'])

    # Build leaf batch index
    leaf_batches = {}
    for n in tree:
        if n['is_leaf'] and n['batch_index'] >= 0:
            batch = idx.get_leaf(n['batch_index'])
            leaf_batches[n['node_id']] = batch.column('item_index').to_numpy()

    # Get nodes at target depth
    target_nodes = [n for n in tree if n['depth'] == target_depth]
    target_nodes.sort(key=lambda n: -n['num_items'])

    print(f"  Tree labeling: {len(target_nodes)} branches at depth {target_depth}")

    # Check cache: cache_data (in-memory) → cache_file (on-disk)
    _cache_key = f"tree_depth_{target_depth}"
    if cache_data is not None:
        cached = cache_data.get(_cache_key, {})
        if cached.get("branch_labels") and cached.get("child_labels"):
            print(f"  Loaded tree labels from cache")
            return cached
    elif cache_file:
        cache_path = Path(cache_file)
        if cache_path.exists():
            file_cache = json.loads(cache_path.read_text())
            cached = file_cache.get(_cache_key, {})
            if cached.get("branch_labels") and cached.get("child_labels"):
                print(f"  Loaded tree labels from cache")
                return cached

    rng = np.random.default_rng(42)
    branch_labels = {}
    child_labels = {}
    hierarchy = {}

    def _label_branch(node):
        nid = node['node_id']
        kids = children_of[nid]
        kids_sorted = sorted(kids, key=lambda k: -by_id[k]['num_items'])

        # Sample titles from each child
        child_samples = {}
        for kid_id in kids_sorted:
            kn = by_id[kid_id]
            if kn['num_items'] < min_child_size:
                continue
            all_idx = _collect_descendant_indices(
                kid_id, children_of, leaf_batches)
            if len(all_idx) < 3:
                continue

            n_sample = min(samples_per_child, len(all_idx))
            sample_idx = rng.choice(all_idx, size=n_sample, replace=False)
            sample_titles = [titles[j][:120] for j in sample_idx]

            # Deduplicate
            seen = set()
            unique = []
            for t in sample_titles:
                if t not in seen:
                    seen.add(t)
                    unique.append(t)
            child_samples[kid_id] = (kn['num_items'], unique)

        if not child_samples:
            return nid, f"Group {nid}", {}

        # Build prompt
        prompt_parts = [
            "You are labeling groups in an embedding-space tree. Below are "
            "sample titles from each child group within a branch.\n"
        ]
        child_ids_ordered = list(child_samples.keys())
        for i, kid_id in enumerate(child_ids_ordered):
            count, samples = child_samples[kid_id]
            prompt_parts.append(f"--- Group {i+1} ({count} items) ---")
            for t in samples:
                prompt_parts.append(f"  - {t}")
            prompt_parts.append("")

        prompt_parts.append(
            "For each group, give a SHORT (2-5 word) descriptive label that "
            "captures what makes it distinctive FROM THE OTHER GROUPS.\n"
            "Then give ONE summary label (2-5 words) for the entire branch.\n\n"
            "BAD labels (too vague): 'General Items', 'Miscellaneous', "
            "'Various Topics'\n"
            "GOOD labels (specific): 'Marine Biology', 'European Monarchs', "
            "'Spinal Fixation Screws', 'Video Game Consoles'\n\n"
            "Reply in this EXACT format (one line per group, then summary):\n"
        )
        for i in range(len(child_ids_ordered)):
            prompt_parts.append(f"Group {i+1}: <label>")
        prompt_parts.append("Branch: <summary label>")

        prompt = "\n".join(prompt_parts)
        response = _call_ollama(model, prompt)

        # Parse response
        kid_labels = {}
        branch_label = f"Branch {nid}"
        for line in response.split('\n'):
            line = line.strip()
            if not line:
                continue
            if line.lower().startswith('branch:'):
                branch_label = line.split(':', 1)[1].strip().strip('"\'')[:50]
            else:
                for i, kid_id in enumerate(child_ids_ordered):
                    prefix = f"Group {i+1}:"
                    if line.startswith(prefix) or line.lower().startswith(
                            prefix.lower()):
                        label = line.split(':', 1)[1].strip().strip('"\'')[:50]
                        kid_labels[kid_id] = label
                        break

        # Fill in any missing child labels
        for kid_id in child_ids_ordered:
            if kid_id not in kid_labels:
                kid_labels[kid_id] = f"Subgroup {kid_id}"

        return nid, branch_label, kid_labels

    # Run labeling (sequential for visibility)
    for i, node in enumerate(target_nodes):
        nid, b_label, k_labels = _label_branch(node)
        branch_labels[nid] = b_label
        child_labels.update(k_labels)
        hierarchy[nid] = sorted(k_labels.keys())

        n_kids = len(k_labels)
        print(f"    [{i+1}/{len(target_nodes)}] {b_label:<35s} "
              f"({node['num_items']:>6,} items, {n_kids} children)")
        for kid_id, kid_label in sorted(k_labels.items(),
                                         key=lambda x: -by_id[x[0]]['num_items']):
            kn = by_id[kid_id]
            print(f"      └─ {kid_label:<30s} ({kn['num_items']:>5,} items)")

    result = {
        'branch_labels': {str(k): v for k, v in branch_labels.items()},
        'child_labels': {str(k): v for k, v in child_labels.items()},
        'hierarchy': {str(k): v for k, v in hierarchy.items()},
    }

    # Save to cache (only write file if cache_file was used, not cache_data)
    if cache_file and cache_data is None:
        cache_path = Path(cache_file)
        file_cache = {}
        if cache_path.exists():
            file_cache = json.loads(cache_path.read_text())
        file_cache[_cache_key] = result
        cache_path.write_text(json.dumps(file_cache, indent=2))
        print(f"  Saved tree labels to cache ({cache_file})")

    return result


def enrich_tree(dyf_path, model="gemma2:9b", target_depth=3,
                samples_per_child=8, output_path=None):
    """Add tree-based hierarchical labels to a .dyf file.

    Label cache is stored in .dyf metadata under '_label_cache'.

    Uses the existing DYF tree structure for bottom-up labeling:
    each branch at target_depth gets a summary label, and each of
    its children gets a specific label. Much cheaper than per-cluster
    contrastive labeling.

    Args:
        dyf_path: Path to .dyf file (Level 0+).
        model: Ollama model name.
        target_depth: Tree depth for branches (3 = 16 branches for 4-bit tree).
        samples_per_child: Titles to sample per child node.
        output_path: Output path (defaults to overwriting input).
    """
    print(f"\n=== Tree Labeling (depth={target_depth}) ===")
    print(f"  Input: {dyf_path}")

    with LazyIndex(dyf_path) as idx:
        # Get titles
        sf_names = idx.stored_field_names
        if 'title' not in sf_names:
            print("  ERROR: No 'title' stored field found.")
            return

        data = idx.extract_all_fields()
        titles = data['fields']['title']

        # Load label cache from .dyf metadata
        label_cache_data = json.loads(data['metadata'].get('_label_cache', '{}'))
        if label_cache_data:
            print(f"  Loaded {len(label_cache_data)} label cache entries from .dyf")

        result = label_tree_bottomup(
            idx, titles, model=model, target_depth=target_depth,
            samples_per_child=samples_per_child,
            cache_data=label_cache_data)
        label_cache_data[f"tree_depth_{target_depth}"] = result

    # Store results in metadata
    new_meta = {
        f'tree_labels_depth_{target_depth}': json.dumps(result),
        '_label_cache': json.dumps(label_cache_data),
    }

    out = output_path or dyf_path
    print(f"\n  Writing labels to: {out}")
    rewrite_lazy_index(dyf_path, new_metadata=new_meta, output_path=out)
    print(f"  Done. Tree labels at depth {target_depth} stored.")


def annotate_cluster_names(names, labels, embeddings,
                           max_sample=200, seed=42):
    """Prepend glyph annotations to cluster names for size and purity.

    Size (point count, z-score based):
        ⭑ big hub   — count > 1 stdev above mean
        ⭒ small hub — count above mean
        (no star)   — below mean

    Purity (intra-cluster cosine similarity, z-score based):
        ≈ appended when purity > 1 stdev below mean (impure flag)
    """
    rng = np.random.RandomState(seed)
    cluster_ids = sorted(set(int(c) for c in labels))

    # ── Per-cluster size ──
    sizes = {}
    for cid in cluster_ids:
        sizes[cid] = int(np.sum(labels == cid))

    # ── Per-cluster purity (sampled cosine similarity) ──
    purities = {}
    for cid in cluster_ids:
        mask = labels == cid
        embs = embeddings[mask]
        if len(embs) < 2:
            purities[cid] = 1.0
            continue
        if len(embs) > max_sample:
            idx = rng.choice(len(embs), max_sample, replace=False)
            embs = embs[idx]
        norms = np.linalg.norm(embs, axis=1, keepdims=True)
        embs_normed = embs / np.maximum(norms, 1e-8)
        gram = embs_normed @ embs_normed.T
        n_pts = len(embs)
        triu_idx = np.triu_indices(n_pts, k=1)
        purities[cid] = float(gram[triu_idx].mean())

    # ── Z-score thresholds ──
    size_vals = np.array([sizes[c] for c in cluster_ids], dtype=float)
    size_mean = float(size_vals.mean())
    size_std = float(size_vals.std())
    size_threshold = size_mean + size_std  # capital if > 1σ above mean

    purity_vals = np.array([purities[c] for c in cluster_ids])
    pur_mean = float(purity_vals.mean())
    pur_std = float(purity_vals.std())
    pur_threshold = pur_mean - pur_std  # impure if > 1σ below mean

    n_capitals = int(np.sum(size_vals > size_threshold))
    n_impure = int(np.sum(purity_vals < pur_threshold))
    print(f"    Size: mean={size_mean:.0f}, σ={size_std:.0f}, "
          f"threshold={size_threshold:.0f} → {n_capitals} capitals")
    print(f"    Purity: mean={pur_mean:.3f}, σ={pur_std:.3f}, "
          f"threshold={pur_threshold:.3f} → {n_impure} impure")

    # ── Annotate ──
    annotated = {}
    for cid in cluster_ids:
        if sizes[cid] > size_threshold:
            star = '⭑'
        elif sizes[cid] > size_mean:
            star = '⭒'
        else:
            star = ''
        impure = '≈' if purities[cid] < pur_threshold else ''
        prefix = f'{star}{impure} ' if (star or impure) else ''

        name = names.get(cid, names.get(str(cid), f"Cluster {cid}"))
        annotated[cid] = f'{prefix}{name}'

    n_annotated = sum(1 for c in cluster_ids
                      if annotated[c] != names.get(c, names.get(str(c), '')))
    print(f"    Annotated {n_annotated}/{len(cluster_ids)} clusters")
    return annotated


def enrich_cluster(dyf_path, n_clusters_list=None, model="gemma2:9b",
                   output_path=None, force=False):
    """Add BIRCH cluster labels to a .dyf file (Level 1 → 2).

    Label cache is stored in .dyf metadata under '_label_cache'.

    Args:
        dyf_path: Path to .dyf file (must be at least Level 1).
        n_clusters_list: List of cluster counts, e.g. [12, 25, 50].
            Defaults to [12, 25, 50].
        model: Ollama model for LLM labeling.
        output_path: Output path (defaults to overwriting input).
        force: Re-run even if already at level 2+.
            Strips stale level 3 metadata (edges, narration).
    """
    if n_clusters_list is None:
        n_clusters_list = [12, 25, 50]

    print(f"\n=== Level 2: BIRCH Clustering ===")
    print(f"  Input: {dyf_path}")
    print(f"  Cluster levels: {n_clusters_list}")

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

    # Get titles if available
    titles = data['fields'].get('title')
    if titles is None:
        titles = [f"Item {i}" for i in range(n)]
    embeddings = data['embeddings']

    # Load label cache from .dyf metadata
    label_cache_data = json.loads(data['metadata'].get('_label_cache', '{}'))
    if label_cache_data:
        print(f"  Loaded {len(label_cache_data)} label cache entries from .dyf")

    # Load split keywords if available (for improved cluster labeling context)
    split_kw_json = data['metadata'].get('split_keywords')
    split_kw_data = None
    if split_kw_json:
        split_kw_data = json.loads(split_kw_json)
        n_splits = len(split_kw_data.get('splits', {}))
        print(f"  Using split keywords ({n_splits} splits) for label context")

    # Load tree structure for cluster-tree DAG (when split keywords available)
    tree_maps = None
    if split_kw_data:
        try:
            from dyf.splits import build_tree_maps
            with LazyIndex(dyf_path) as idx_tree:
                tree_maps = build_tree_maps(idx_tree)
            print(f"  Loaded tree structure for cluster-tree DAG")
        except Exception as e:
            print(f"  WARNING: Could not load tree structure: {e}")

    # Cluster at each level — dual 2D/3D clustering
    new_sf = {}
    new_meta = {}
    cluster_2d = coords[:, :2]
    cluster_3d = coords[:, :3]

    for target_k in n_clusters_list:
        # ── 2D clustering (primary — gets LLM labels) ──
        print(f"\n  Clustering 2D at k={target_k}...")
        birch_2d = fit_birch(cluster_2d, target_k)
        labels_2d = birch_2d.predict(cluster_2d)
        labels_2d = merge_tiny_clusters(labels_2d, cluster_2d)
        n_2d = len(set(labels_2d.tolist()))
        print(f"    2D: {n_2d} clusters after merge")

        new_sf[f'cluster_{target_k}_2d'] = labels_2d.astype(np.int32)

        # 2D centroids (using full 3D coords for centroid positions)
        centroids_2d = {}
        for cid in sorted(set(labels_2d.tolist())):
            mask = labels_2d == cid
            cent = coords[mask].mean(axis=0)
            centroids_2d[cid] = [round(float(cent[0]), 4),
                                 round(float(cent[1]), 4),
                                 round(float(cent[2]), 4)]
        new_meta[f'cluster_centroids_{target_k}_2d'] = json.dumps(
            centroids_2d)

        # Build cluster-tree DAG for this cluster level (when tree available)
        dag_path_labels = None
        dag_sibling_kw = None
        if tree_maps and split_kw_data:
            try:
                from dyf.cluster_tree import (
                    build_cluster_tree_dag,
                    compute_sibling_keywords,
                    derive_path_labels,
                )
                tree_list, cmap_tree, lbatch_tree = tree_maps
                dag = build_cluster_tree_dag(
                    tree_list, cmap_tree, lbatch_tree,
                    labels_2d, target_k)
                dag_path_labels = derive_path_labels(
                    dag, split_kw_data, target_k)
                dag_sibling_kw = compute_sibling_keywords(
                    dag, titles if isinstance(titles, list)
                    else list(titles),
                    labels_2d, target_k)
                print(f"    Built cluster-tree DAG: "
                      f"{len(dag_path_labels)} path labels, "
                      f"{len(dag_sibling_kw)} sibling keyword sets")
                # Store DAG and path labels in metadata
                new_meta[f'cluster_tree_dag_{target_k}_2d'] = json.dumps(
                    dag.to_dict())
                new_meta[f'cluster_path_labels_{target_k}_2d'] = json.dumps(
                    {str(k): v for k, v in dag_path_labels.items()})
            except Exception as e:
                print(f"    WARNING: cluster-tree DAG failed: {e}")

        # LLM-label 2D clusters
        print(f"  Labeling 2D k={target_k} clusters...")
        names_2d_raw = label_clusters(
            titles, coords, labels_2d, embeddings,
            model=model, cache_data=label_cache_data,
            cache_key=f"cluster_{target_k}_2d",
            split_keywords=split_kw_data,
            path_labels=dag_path_labels,
            sibling_keywords=dag_sibling_kw)
        label_cache_data[f"cluster_{target_k}_2d"] = {
            str(k): v for k, v in names_2d_raw.items()}
        names_2d = annotate_cluster_names(
            names_2d_raw, labels_2d, embeddings,
)
        new_meta[f'cluster_names_{target_k}_2d'] = json.dumps(
            {str(k): v for k, v in names_2d.items()})

        # ── 3D clustering (secondary — labels transferred) ──
        print(f"  Clustering 3D at k={target_k}...")
        birch_3d = fit_birch(cluster_3d, target_k)
        labels_3d = birch_3d.predict(cluster_3d)
        labels_3d = merge_tiny_clusters(labels_3d, cluster_3d)
        n_3d = len(set(labels_3d.tolist()))
        print(f"    3D: {n_3d} clusters after merge")

        new_sf[f'cluster_{target_k}_3d'] = labels_3d.astype(np.int32)

        # 3D centroids
        centroids_3d = {}
        for cid in sorted(set(labels_3d.tolist())):
            mask = labels_3d == cid
            cent = coords[mask].mean(axis=0)
            centroids_3d[cid] = [round(float(cent[0]), 4),
                                 round(float(cent[1]), 4),
                                 round(float(cent[2]), 4)]
        new_meta[f'cluster_centroids_{target_k}_3d'] = json.dumps(
            centroids_3d)

        # Transfer 2D labels to 3D via majority vote (use raw names)
        names_3d_raw = transfer_labels_majority_vote(
            labels_2d, names_2d_raw, labels_3d)
        names_3d = annotate_cluster_names(
            names_3d_raw, labels_3d, embeddings,
)
        new_meta[f'cluster_names_{target_k}_3d'] = json.dumps(
            {str(k): v for k, v in names_3d.items()})

        print(f"    Transferred labels to {n_3d} 3D clusters")

        # Spatial color maps (embedding-derived hue ordering)
        rgb_2d = spatial_rgb_map(labels_2d.tolist(), embeddings)
        new_meta[f'cluster_colors_{target_k}_2d'] = json.dumps(
            {str(k): v for k, v in rgb_2d.items()})
        rgb_3d = spatial_rgb_map(labels_3d.tolist(), embeddings)
        new_meta[f'cluster_colors_{target_k}_3d'] = json.dumps(
            {str(k): v for k, v in rgb_3d.items()})
        print(f"    Stored color maps for 2D ({len(rgb_2d)}) "
              f"and 3D ({len(rgb_3d)}) clusters")

    # Strip stale level 3 metadata when re-clustering (None = delete key)
    if force and level >= 3:
        for stale_key in ['edge_pairs', 'edge_paths_2d', 'tour_narration',
                          '_provenance_level_3']:
            new_meta[stale_key] = None
        print("  Stripped stale level 3 metadata (re-run 'viz' to regenerate)")

    # Store label cache in metadata
    new_meta['_label_cache'] = json.dumps(label_cache_data)

    # Stamp provenance for Level 2
    new_meta['_provenance_level_2'] = json.dumps(provenance_to_dict(
        create_provenance(
            artifact_type="dyf",
            n_items=n,
            source_paths=[str(dyf_path)],
            params={"n_clusters_list": n_clusters_list, "model": model},
        )
    ))

    out = output_path or dyf_path
    print(f"\n  Writing enriched file: {out}")
    rewrite_lazy_index(dyf_path, new_stored_fields=new_sf,
                       new_metadata=new_meta, output_path=out)
    print(f"  Done. Level 1 → 2 (dual 2D/3D)")


def reannotate(dyf_path, output_path=None):
    """Re-run glyph annotations on existing cluster names without re-clustering.

    Reads raw names from _label_cache, recomputes purity/connectivity glyphs,
    writes updated cluster_names_* metadata. Does not touch clusters, edges,
    or narration.
    """
    print(f"\n=== Reannotate Cluster Glyphs ===")
    print(f"  Input: {dyf_path}")

    with LazyIndex(dyf_path) as idx:
        level = idx.detect_enrichment_level()
        if level < 2:
            print(f"  ERROR: Need level 2 (clusters), got level {level}.")
            return
        data = idx.extract_all_fields()

    label_cache_data = json.loads(data['metadata'].get('_label_cache', '{}'))
    if not label_cache_data:
        print("  ERROR: No _label_cache in metadata. Run 'cluster' first.")
        return

    embeddings = data['embeddings']

    # Discover cluster levels (e.g. 12, 25, 50) from stored fields
    cluster_ks = sorted({
        parts[1]
        for sf_name in data['fields']
        if sf_name.startswith('cluster_')
        for parts in [sf_name.split('_')]
        if len(parts) == 3 and parts[2] in ('2d', '3d')
    })

    new_meta = {}
    for target_k in cluster_ks:
        # Get 2D raw names from cache
        cache_key_2d = f"cluster_{target_k}_2d"
        raw_2d = label_cache_data.get(cache_key_2d)
        if raw_2d is None:
            raw_2d = label_cache_data.get(f"cluster_{target_k}")
        if raw_2d is None:
            print(f"  Skipping k={target_k}: no raw names in cache")
            continue
        raw_2d = {int(k): v for k, v in raw_2d.items()}

        # Reannotate 2D
        sf_2d = f'cluster_{target_k}_2d'
        if sf_2d in data['fields']:
            labels_2d = data['fields'][sf_2d].astype(np.int32)
            print(f"  Reannotating {sf_2d}...")
            ann_2d = annotate_cluster_names(
                raw_2d, labels_2d, embeddings)
            new_meta[f'cluster_names_{target_k}_2d'] = json.dumps(
                {str(k): v for k, v in ann_2d.items()})

        # Reannotate 3D (transfer from 2D raw names, then annotate)
        sf_3d = f'cluster_{target_k}_3d'
        if sf_3d in data['fields'] and sf_2d in data['fields']:
            labels_3d = data['fields'][sf_3d].astype(np.int32)
            raw_3d = transfer_labels_majority_vote(
                labels_2d, raw_2d, labels_3d)
            print(f"  Reannotating {sf_3d}...")
            ann_3d = annotate_cluster_names(
                raw_3d, labels_3d, embeddings)
            new_meta[f'cluster_names_{target_k}_3d'] = json.dumps(
                {str(k): v for k, v in ann_3d.items()})

    out = output_path or dyf_path
    print(f"\n  Writing: {out}")
    rewrite_lazy_index(dyf_path, new_metadata=new_meta, output_path=out)
    print("  Done.")


# ── Split keyword enrichment ───────────────────────────────────────────


def enrich_splits(dyf_path, max_depth=3, bigram_check=False, output_path=None,
                  domain_threshold=0.10, min_child_items=50):
    """Compute tree split keywords and store in .dyf metadata.

    For each internal node (up to max_depth), computes discriminative TF-IDF
    keywords for each child side of the split. These keywords provide
    deterministic, LLM-free context for cluster labeling.

    Args:
        dyf_path: Path to .dyf file (needs at least titles stored).
        max_depth: Maximum depth from root to compute keywords for.
        bigram_check: Enable PMI-based compound meaning detection.
        output_path: Output path (defaults to overwriting input).
        domain_threshold: Fraction threshold for domain stop words (0.0-1.0).
        min_child_items: Skip children with fewer items than this.
    """
    from dyf.splits import (
        build_tree_maps, compute_domain_stopwords, compute_split_keywords,
    )

    print(f"\n=== Split Keywords ===")
    print(f"  Input: {dyf_path}")

    with LazyIndex(dyf_path) as idx:
        tree, children_map, leaf_batches = build_tree_maps(idx)

    # Extract titles
    with LazyIndex(dyf_path) as idx:
        data = idx.extract_all_fields()
    titles = data['fields'].get('title')
    if titles is None:
        titles = [f"Item {i}" for i in range(len(data['embeddings']))]
    if isinstance(titles, np.ndarray):
        titles = titles.tolist()

    n = len(titles)
    print(f"  {n:,} items, tree has {len(tree)} nodes")

    # Compute domain stop words
    domain_sw = compute_domain_stopwords(titles, threshold=domain_threshold)
    print(f"  {len(domain_sw)} domain stop words "
          f"(e.g. {sorted(domain_sw)[:5]})")

    # Compute split keywords
    result = compute_split_keywords(
        titles, tree, leaf_batches, children_map,
        max_depth_from_root=max_depth,
        min_child_items=min_child_items,
        domain_stopwords=domain_sw,
        bigram_check=bigram_check,
    )

    n_splits = len(result['splits'])
    print(f"  Computed keywords for {n_splits} splits "
          f"(depth 0-{max_depth - 1})")

    if bigram_check:
        needed = sum(1 for s in result['splits'].values()
                     if s.get('bigram_needed'))
        print(f"  Bigram needed: {needed}/{n_splits} splits")

    # Serialize: convert tuple keys/values for JSON
    serializable = {
        'domain_stopwords': result['domain_stopwords'],
        'splits': {},
    }
    for nid, split in result['splits'].items():
        s = {
            'depth': split['depth'],
            'children': {},
        }
        if 'bigram_needed' in split:
            s['bigram_needed'] = split['bigram_needed']
        for cid, cinfo in split['children'].items():
            entry = {
                'count': cinfo['count'],
                'unigrams': cinfo['unigrams'],
            }
            if 'bigrams' in cinfo:
                entry['bigrams'] = cinfo['bigrams']
            s['children'][str(cid)] = entry
        serializable['splits'][str(nid)] = s

    new_meta = {
        'split_keywords': json.dumps(serializable),
    }

    out = output_path or dyf_path
    print(f"  Writing enriched file: {out}")
    rewrite_lazy_index(dyf_path, new_metadata=new_meta, output_path=out)
    print(f"  Done.")


# ── Bridge edges + narration (Level 2 → 3) ─────────────────────────────


def compute_bridge_edges(coords, embeddings, labels, n_clusters):
    """Compute cross-cluster bridge edges using ROG ontology."""
    import dyf

    print("  Building ROG ontology for bridge detection...")
    result = dyf.build_rog_ontology(
        embeddings, initial_threshold=0.55, min_threshold=0.35,
        target_coverage=0.95, verbose=False)

    ont = result.ontology
    pair_counts = defaultdict(int)
    for parent, children_list in ont.children.items():
        for child, sim, div_gap in children_list:
            c1, c2 = int(labels[parent]), int(labels[child])
            if c1 != c2:
                pair = (min(c1, c2), max(c1, c2))
                pair_counts[pair] += 1

    cross = sum(pair_counts.values())
    print(f"  {len(pair_counts)} cluster pairs, {cross:,} cross-cluster edges")

    # Cluster centroids
    centroids = np.zeros((n_clusters, coords.shape[1]), dtype=np.float32)
    for c in range(n_clusters):
        mask = labels == c
        if mask.any():
            centroids[c] = coords[mask].mean(axis=0)

    edge_list = sorted(pair_counts.keys(), key=lambda p: -pair_counts[p])
    if not edge_list:
        return [], {}

    # Serialize edge pairs: [[src, dst, weight], ...]
    edge_pairs = [[int(c1), int(c2), int(pair_counts[(c1, c2)])]
                  for c1, c2 in edge_list]

    # 2D bundled paths via datashader
    import pandas as pd
    from datashader.bundling import hammer_bundle

    centroids_2d = centroids[:, :2]
    nodes_df = pd.DataFrame({
        "x": centroids_2d[:, 0].astype(float),
        "y": centroids_2d[:, 1].astype(float),
    })
    edges_df = pd.DataFrame({
        "source": [e[0] for e in edge_list],
        "target": [e[1] for e in edge_list],
    })
    bundled_df = hammer_bundle(nodes_df, edges_df)

    edge_paths_2d = []
    current_path = []
    for _, row in bundled_df.iterrows():
        if pd.isna(row["x"]) or pd.isna(row["y"]):
            if current_path:
                edge_paths_2d.append(current_path)
                current_path = []
        else:
            current_path.append([round(row["x"], 4), round(row["y"], 4)])
    if current_path:
        edge_paths_2d.append(current_path)

    print(f"  Bundled {len(edge_paths_2d)} 2D edge paths")
    return edge_pairs, edge_paths_2d


def enrich_viz(dyf_path, cluster_level=25, model="gpt-oss:20b",
               title=None, output_path=None, force=False):
    """Add bridge edges and tour narration (Level 2 → 3).

    Args:
        dyf_path: Path to .dyf file (must be at least Level 2).
        cluster_level: Which cluster_N level to use for edges/narration.
        model: Ollama model for narration generation.
        title: Title for intro narration.
        output_path: Output path (defaults to overwriting input).
        force: Re-run even if already at level 3.
    """
    print(f"\n=== Level 3: Viz Enrichment ===")
    print(f"  Input: {dyf_path}")

    with LazyIndex(dyf_path) as idx:
        level = idx.detect_enrichment_level()
        if level < 2:
            print(f"  ERROR: Need level 2 (clusters), got level {level}. "
                  f"Run 'cluster' first.")
            return
        if level >= 3 and not force:
            print(f"  Already at level {level} (viz-ready), skipping. "
                  f"Use --force to re-run.")
            return

    # Extract data
    with LazyIndex(dyf_path) as idx:
        data = idx.extract_all_fields()
    n = len(data['embeddings'])

    coords = np.column_stack([
        data['fields']['umap_x'],
        data['fields']['umap_y'],
        data['fields']['umap_z'],
    ])
    embeddings = data['embeddings']

    # Try dual-cluster naming first (cluster_{k}_2d), fallback to bare
    cluster_field_2d = f'cluster_{cluster_level}_2d'
    cluster_field_bare = f'cluster_{cluster_level}'
    if cluster_field_2d in data['fields']:
        cluster_field = cluster_field_2d
        names_key = f'cluster_names_{cluster_level}_2d'
    elif cluster_field_bare in data['fields']:
        cluster_field = cluster_field_bare
        names_key = f'cluster_names_{cluster_level}'
    else:
        available = [f for f in data['fields'] if f.startswith('cluster_')]
        print(f"  ERROR: cluster_{cluster_level} not found. "
              f"Available: {available}")
        return
    labels = data['fields'][cluster_field]
    n_clusters = len(set(labels.tolist()))

    # Get cluster names
    names_json = data['metadata'].get(names_key, '{}')
    cluster_names = {int(k): v for k, v in json.loads(names_json).items()}

    # Bridge edges
    print(f"\n  Computing bridge edges for {n_clusters} clusters...")
    edge_pairs, edge_paths_2d = compute_bridge_edges(
        coords, embeddings, labels, n_clusters)

    # Generate narration via Ollama (with sample-title fallback)
    titles = data['fields'].get('title')
    if titles is None:
        titles = [f"Item {i}" for i in range(n)]
    narration = _generate_narration(
        cluster_names, titles, labels, coords, model=model, title=title)

    new_meta = {
        'edge_pairs': json.dumps(edge_pairs),
        'edge_paths_2d': json.dumps(edge_paths_2d),
        'tour_narration': json.dumps(
            {str(k): v for k, v in narration.items()}),
    }

    # Stamp provenance for Level 3
    new_meta['_provenance_level_3'] = json.dumps(provenance_to_dict(
        create_provenance(
            artifact_type="dyf",
            n_items=n,
            source_paths=[str(dyf_path)],
            params={"cluster_level": cluster_level, "model": model},
        )
    ))

    out = output_path or dyf_path
    print(f"\n  Writing enriched file: {out}")
    rewrite_lazy_index(dyf_path, new_metadata=new_meta, output_path=out)
    print(f"  Done. Level 2 → 3")


def _number_to_words(n):
    """Convert a small integer to English words."""
    words = {
        1: "one", 2: "two", 3: "three", 4: "four", 5: "five",
        6: "six", 7: "seven", 8: "eight", 9: "nine", 10: "ten",
        11: "eleven", 12: "twelve", 13: "thirteen", 14: "fourteen",
        15: "fifteen", 16: "sixteen", 17: "seventeen", 18: "eighteen",
        19: "nineteen", 20: "twenty", 25: "twenty-five", 30: "thirty",
        40: "forty", 50: "fifty",
    }
    return words.get(n, str(n))


def _approx_number_words(n):
    """Convert a count to approximate spoken form."""
    if n < 100:
        return f"{n}"
    elif n < 1000:
        hundreds = round(n / 100) * 100
        return f"about {hundreds}"
    elif n < 10000:
        thousands = round(n / 100) * 100
        return f"about {thousands:,}"
    else:
        thousands = round(n / 1000) * 1000
        return f"about {thousands:,}"


def _call_ollama_chat(prompt, model="gpt-oss:20b", timeout=30):
    """Call Ollama via HTTP API. Returns response text or None."""
    import json as _json
    payload = _json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
    }).encode()
    req = urllib.request.Request(
        "http://localhost:11434/api/chat",
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = _json.loads(resp.read())
            return body.get("message", {}).get("content", "").strip()
    except Exception:
        return None


def _generate_narration(cluster_names, titles, labels, coords,
                        model="gpt-oss:20b", title=None):
    """Generate tour narration using Ollama, with sample-title fallback."""
    from concurrent.futures import ThreadPoolExecutor, as_completed

    label_arr = np.asarray(labels)
    cluster_points = defaultdict(list)
    for i, cid in enumerate(label_arr):
        cluster_points[int(cid)].append(i)

    total_pts = sum(len(v) for v in cluster_points.values())
    sorted_cids = sorted(cluster_names.keys(),
                         key=lambda c: len(cluster_points.get(c, [])),
                         reverse=True)

    # Check Ollama availability
    ollama_ok = _call_ollama_chat("Say OK.", model=model, timeout=30) is not None
    if ollama_ok:
        print(f"\n  Generating narration via Ollama ({model})...")
    else:
        print(f"\n  Ollama not available — using sample-title narration")

    narration = {}
    tasks = []

    for cid in sorted_cids:
        name = cluster_names[cid]
        pts = cluster_points.get(cid, [])
        n_pts = len(pts)
        n_approx = _approx_number_words(n_pts)

        # Sample diverse titles from this cluster
        sample_idx = _sample_spatial(pts, coords, 20)
        seen = set()
        sample_titles = []
        for idx in sample_idx:
            t = titles[idx] if hasattr(titles, '__getitem__') else str(idx)
            if t not in seen:
                seen.add(t)
                sample_titles.append(t)
                if len(sample_titles) >= 12:
                    break

        if ollama_ok:
            items_str = "\n".join(f"  - {t}" for t in sample_titles)
            prompt = (
                f'You are narrating a guided tour of an FDA medical '
                f'device landscape for a general audience.\n\n'
                f'Cluster name: "{name}"\n'
                f'Size: {n_approx} devices\n\n'
                f'Sample FDA product listings (these are raw registry '
                f'entries — do NOT recite product codes or model '
                f'numbers):\n{items_str}\n\n'
                f'Write 2-3 sentences that:\n'
                f'1. Start with "{name}."\n'
                f'2. Explain in plain language what this category of '
                f'medical device does and why it matters clinically\n'
                f'3. Say roughly how many devices are in this group '
                f'(use "{n_approx}")\n\n'
                f'Style: calm British documentary narrator. '
                f'Written for text-to-speech — spell out all numbers, '
                f'no abbreviations, no special characters, no quotes. '
                f'Do NOT list product names or model numbers.\n'
            )
            tasks.append((cid, prompt, sample_titles))
        else:
            # Fallback without LLM
            narration[cid] = (
                f"{name}. {n_approx} devices in this category.")

    if ollama_ok and tasks:
        completed = 0

        def _do_one(task):
            cid, prompt, samples = task
            text = _call_ollama_chat(prompt, model=model, timeout=120)
            if text:
                # Clean up LLM quirks
                text = re.sub(r'\s+', ' ', text).strip().strip('"\'')
                return cid, text
            # Fallback for this cluster
            name = cluster_names[cid]
            n_approx = _approx_number_words(
                len(cluster_points.get(cid, [])))
            return cid, f"{name}. {n_approx} devices in this category."

        n_workers = min(2, len(tasks))
        with ThreadPoolExecutor(max_workers=n_workers) as executor:
            futures = {executor.submit(_do_one, t): t for t in tasks}
            for future in as_completed(futures):
                cid, text = future.result()
                narration[cid] = text
                completed += 1
                if completed % 5 == 0 or completed == len(tasks):
                    print(f"    Narrated {completed}/{len(tasks)} "
                          f"clusters...", flush=True)

    # Intro
    n_clusters = len(cluster_names)
    n_words = _number_to_words(n_clusters)
    top3 = [cluster_names[c] for c in sorted_cids[:3]]
    total_words = _approx_number_words(total_pts)
    display_title = title or "Embedding Landscape"
    narration["intro"] = (
        f"{display_title}. {total_words} items organized into "
        f"{n_words} clusters. The largest regions are {top3[0]}"
        + (f", {top3[1]}" if len(top3) > 1 else "")
        + (f", and {top3[2]}" if len(top3) > 2 else "")
        + ". Let's take a look."
    )
    narration["outro"] = (
        "That completes our tour. Clusters nearby share deeper "
        "similarities, and the bridges between them trace where one "
        "category shades into the next."
    )

    # Preview
    for cid in sorted_cids[:3]:
        print(f"    [{cid:2d}] {narration[cid][:80]}...")

    return narration


# ── CLI ─────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Enrich a .dyf file through progressive levels")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # project
    p_proj = subparsers.add_parser(
        "project", help="Level 0→1: Add UMAP coordinates")
    p_proj.add_argument("dyf_path", help="Path to .dyf file")
    p_proj.add_argument("--n-components", type=int, default=3,
                        help="UMAP dimensions (default: 3)")
    p_proj.add_argument("--densmap", action="store_true",
                        help="Use densMAP")
    p_proj.add_argument("--fisher-col", default=None,
                        help="Column name for Fisher dimension weighting (e.g. gmdn_terms)")
    p_proj.add_argument("--fisher-parquet", default=None,
                        help="Parquet file containing the Fisher column "
                             "(if not stored in the .dyf)")
    p_proj.add_argument("--diagnose-parquet", default=None,
                        help="Parquet file for axis diagnostics sanity check "
                             "(discover columns, report under-served axes)")
    p_proj.add_argument("-o", "--output", default=None,
                        help="Output path (default: overwrite input)")

    # cluster
    p_clust = subparsers.add_parser(
        "cluster", help="Level 1→2: Add BIRCH clusters + LLM labels")
    p_clust.add_argument("dyf_path", help="Path to .dyf file")
    p_clust.add_argument("--n-clusters", default="12,25,50",
                         help="Comma-separated cluster counts (default: 12,25,50)")
    p_clust.add_argument("--model", default="gemma2:9b",
                         help="Ollama model for labeling")
    p_clust.add_argument("--force", action="store_true",
                         help="Re-run even if already at level 2+")
    p_clust.add_argument("-o", "--output", default=None)

    # viz
    p_viz = subparsers.add_parser(
        "viz", help="Level 2→3: Add bridge edges + narration")
    p_viz.add_argument("dyf_path", help="Path to .dyf file")
    p_viz.add_argument("--cluster-level", type=int, default=25,
                       help="Which cluster level for edges (default: 25)")
    p_viz.add_argument("--model", default="gpt-oss:20b")
    p_viz.add_argument("--title", default=None,
                       help="Title for intro narration")
    p_viz.add_argument("--force", action="store_true",
                       help="Re-run even if already at level 3")
    p_viz.add_argument("-o", "--output", default=None)

    # tree
    p_tree = subparsers.add_parser(
        "tree", help="Add tree-based hierarchical labels")
    p_tree.add_argument("dyf_path", help="Path to .dyf file")
    p_tree.add_argument("--depth", type=int, default=3,
                        help="Tree depth for branches (default: 3)")
    p_tree.add_argument("--samples", type=int, default=8,
                        help="Titles to sample per child (default: 8)")
    p_tree.add_argument("--model", default="gemma2:9b",
                        help="Ollama model for labeling")
    p_tree.add_argument("-o", "--output", default=None)

    # splits
    p_splits = subparsers.add_parser(
        "splits", help="Compute tree split keywords (deterministic, no LLM)")
    p_splits.add_argument("dyf_path", help="Path to .dyf file")
    p_splits.add_argument("--depth", type=int, default=3,
                          help="Max depth from root (default: 3)")
    p_splits.add_argument("--bigram-check", action="store_true",
                          help="Enable PMI-based compound meaning detection")
    p_splits.add_argument("-o", "--output", default=None)

    # reannotate
    p_reann = subparsers.add_parser(
        "reannotate", help="Re-run glyph annotations without re-clustering")
    p_reann.add_argument("dyf_path", help="Path to .dyf file")
    p_reann.add_argument("-o", "--output", default=None)

    # all
    p_all = subparsers.add_parser(
        "all", help="Run all enrichment levels in sequence")
    p_all.add_argument("dyf_path", help="Path to .dyf file")
    p_all.add_argument("--n-clusters", default="12,25,50")
    p_all.add_argument("--model", default="gemma2:9b")
    p_all.add_argument("--title", default=None)
    p_all.add_argument("--fisher-col", default=None,
                       help="Column name for Fisher dimension weighting")
    p_all.add_argument("--fisher-parquet", default=None,
                       help="Parquet file containing the Fisher column")
    p_all.add_argument("--diagnose-parquet", default=None,
                       help="Parquet file for axis diagnostics sanity check")
    p_all.add_argument("-o", "--output", default=None)

    args = parser.parse_args()

    if args.command == "project":
        enrich_project(args.dyf_path, n_components=args.n_components,
                       densmap=args.densmap, output_path=args.output,
                       fisher_col=args.fisher_col,
                       fisher_parquet=args.fisher_parquet,
                       diagnose_parquet=args.diagnose_parquet)

    elif args.command == "cluster":
        levels = [int(x) for x in args.n_clusters.split(",")]
        enrich_cluster(args.dyf_path, n_clusters_list=levels,
                       model=args.model,
                       output_path=args.output, force=args.force)

    elif args.command == "viz":
        enrich_viz(args.dyf_path, cluster_level=args.cluster_level,
                   model=args.model, title=args.title,
                   output_path=args.output, force=args.force)

    elif args.command == "splits":
        enrich_splits(args.dyf_path, max_depth=args.depth,
                      bigram_check=args.bigram_check,
                      output_path=args.output)

    elif args.command == "reannotate":
        reannotate(args.dyf_path, output_path=args.output)

    elif args.command == "tree":
        enrich_tree(args.dyf_path, model=args.model,
                    target_depth=args.depth,
                    samples_per_child=args.samples,
                    output_path=args.output)

    elif args.command == "all":
        levels = [int(x) for x in args.n_clusters.split(",")]
        out = args.output or args.dyf_path
        enrich_project(args.dyf_path, output_path=out,
                       fisher_col=getattr(args, 'fisher_col', None),
                       fisher_parquet=getattr(args, 'fisher_parquet', None),
                       diagnose_parquet=getattr(args, 'diagnose_parquet', None))
        enrich_cluster(out, n_clusters_list=levels,
                       model=args.model,
                       output_path=out)
        enrich_viz(out, cluster_level=levels[1] if len(levels) > 1
                   else levels[0],
                   model=args.model, title=args.title, output_path=out)


if __name__ == "__main__":
    main()
