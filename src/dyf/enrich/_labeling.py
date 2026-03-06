"""Shared cluster labeling helpers for the enrich pipeline."""

import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from ._ollama import _call_ollama, _make_domain_context


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
    """Get the majority tree path context for a cluster's points."""
    splits = split_keywords.get('splits', {})
    if not splits:
        return ""

    rng = np.random.default_rng(42)
    pts = np.asarray(point_indices)
    if len(pts) > sample_size:
        pts = rng.choice(pts, size=sample_size, replace=False)

    split_votes: dict[str, Counter] = defaultdict(Counter)

    for nid_str, split in splits.items():
        children = split.get('children', {})
        if not children:
            continue

        for cid_str, cinfo in children.items():
            unigrams = [w for w, _ in cinfo.get('unigrams', [])[:top_k]]
            if not unigrams:
                continue
            kw_set = set(unigrams)
            match_count = 0
            for idx in pts:
                if idx < len(titles):
                    words = set(re.findall(r'[a-z]{3,}', titles[idx].lower()))
                    if words & kw_set:
                        match_count += 1
            split_votes[nid_str][cid_str] = match_count

    path_steps = []
    sorted_splits = sorted(
        splits.items(),
        key=lambda x: x[1].get('depth', 0)
    )
    for nid_str, split in sorted_splits:
        votes = split_votes.get(nid_str)
        if not votes:
            continue
        winner = votes.most_common(1)[0][0]
        children = split.get('children', {})
        winner_info = children.get(winner, {})
        words = [w for w, _ in winner_info.get('unigrams', [])[:top_k]]
        if words:
            path_steps.append(','.join(words))

    if not path_steps:
        return ""
    return ' → '.join(path_steps)


def label_clusters(titles, coords, labels, embeddings, model="gpt-oss:20b",
                   n_samples=20, cache_file=None, cache_key=None,
                   cache_data=None, split_keywords=None,
                   path_labels=None, sibling_keywords=None,
                   domain=None):
    """Label clusters via contrastive TF-IDF + local Ollama LLM."""
    from concurrent.futures import ThreadPoolExecutor, as_completed

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

        kw_str = ""
        if path_labels is not None and sibling_keywords is not None:
            from dyf.cluster_tree import format_cluster_context
            pl = path_labels.get(cid, "")
            sk = sibling_keywords.get(cid, [])
            ctx = format_cluster_context(cid, pl, sk)
            if ctx:
                kw_str = f"\n{ctx}"

        if not kw_str and split_keywords:
            path_context = _get_cluster_path_context(
                pts, split_keywords, titles)
            if path_context:
                kw_str = (f"\nTree path context (root → leaf): "
                          f"{path_context}")

        if not kw_str:
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

        dc = _make_domain_context(domain)
        prompt = (
            f"You are labeling clusters of {dc['domain']} in an embedding "
            f"space. This cluster has {len(pts)} {dc['items']}.\n"
            f"{kw_str}\n"
            f"Sample {dc['items']} from across this cluster:\n"
            + "\n".join(f"- {t}" for t in sample_titles)
            + "\n\n"
            "Give a short (2-5 word) label that DISTINGUISHES this cluster "
            "from similar ones. Use the distinguishing keywords and specific "
            "names to find what makes this group unique.\n\n"
            f"BAD labels (too vague): \"{dc['domain'].title()}\", "
            "\"General Items\", \"Miscellaneous\"\n"
            "GOOD labels: specific, distinguishing sub-category names\n\n"
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


def annotate_cluster_names(names, labels, embeddings,
                           max_sample=200, seed=42):
    """Compute size/purity glyphs and return clean names + separate glyph dict."""
    rng = np.random.RandomState(seed)
    cluster_ids = sorted(set(int(c) for c in labels))

    sizes = {}
    for cid in cluster_ids:
        sizes[cid] = int(np.sum(labels == cid))

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

    size_vals = np.array([sizes[c] for c in cluster_ids], dtype=float)
    size_mean = float(size_vals.mean())
    size_std = float(size_vals.std())
    size_threshold = size_mean + size_std

    purity_vals = np.array([purities[c] for c in cluster_ids])
    pur_mean = float(purity_vals.mean())
    pur_std = float(purity_vals.std())
    pur_threshold = pur_mean - pur_std

    n_capitals = int(np.sum(size_vals > size_threshold))
    n_impure = int(np.sum(purity_vals < pur_threshold))
    print(f"    Size: mean={size_mean:.0f}, σ={size_std:.0f}, "
          f"threshold={size_threshold:.0f} → {n_capitals} capitals")
    print(f"    Purity: mean={pur_mean:.3f}, σ={pur_std:.3f}, "
          f"threshold={pur_threshold:.3f} → {n_impure} impure")

    clean_names = {}
    glyphs_dict = {}
    for cid in cluster_ids:
        name = names.get(cid, names.get(str(cid), f"Cluster {cid}"))
        clean_names[cid] = name

        if sizes[cid] > size_threshold:
            star = '⭑'
        elif sizes[cid] > size_mean:
            star = '⭒'
        else:
            star = ''
        impure = '≈' if purities[cid] < pur_threshold else ''
        glyphs_dict[cid] = {"size": star, "purity": impure}

    n_flagged = sum(1 for g in glyphs_dict.values()
                    if g['size'] or g['purity'])
    print(f"    Flagged {n_flagged}/{len(cluster_ids)} clusters with glyphs")
    return clean_names, glyphs_dict


def transfer_labels_majority_vote(labels_primary, names_primary,
                                   labels_secondary):
    """Transfer cluster names from primary to secondary via majority vote."""
    labels_p = np.asarray(labels_primary)
    labels_s = np.asarray(labels_secondary)

    secondary_names = {}
    for s_cid in sorted(set(labels_s.tolist())):
        mask = labels_s == s_cid
        primary_ids = labels_p[mask]
        most_common = Counter(primary_ids.tolist()).most_common(1)[0][0]
        secondary_names[s_cid] = names_primary.get(most_common,
                                                    f"Cluster {s_cid}")

    name_counts = Counter(secondary_names.values())
    duplicates = {name for name, cnt in name_counts.items() if cnt > 1}
    if duplicates:
        seen = {}
        for s_cid in sorted(secondary_names.keys()):
            name = secondary_names[s_cid]
            if name in duplicates:
                idx = seen.get(name, 0) + 1
                seen[name] = idx
                if idx > 1:
                    secondary_names[s_cid] = f"{name} ({idx})"

    return secondary_names
