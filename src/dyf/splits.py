"""Split-based TF-IDF keyword extraction from DYF tree structure.

Each PCA-based LSH split in the DYF tree cleanly separates distinct vocabularies.
This module computes discriminative unigram (and optionally bigram) keywords for
each side of each split, enabling deterministic, LLM-free labeling of tree paths.

Two label types:
- **Split specification** (path): TF-IDF keywords per side, deterministic, no LLM
- **Cluster identity** (BIRCH): LLM names dense regions, using path as context
"""

from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass

import numpy as np

from dyf.lazy_index import TreeNode

# English stop words (compact set covering common function words)
_ENGLISH_STOP = frozenset(
    {
        "the",
        "a",
        "an",
        "of",
        "in",
        "on",
        "at",
        "to",
        "for",
        "and",
        "or",
        "is",
        "was",
        "are",
        "were",
        "be",
        "been",
        "by",
        "with",
        "from",
        "as",
        "it",
        "its",
        "this",
        "that",
        "not",
        "but",
        "has",
        "had",
        "have",
        "do",
        "does",
        "did",
        "will",
        "would",
        "could",
        "should",
        "may",
        "might",
        "can",
        "shall",
        "each",
        "which",
        "their",
        "there",
        "than",
        "into",
        "such",
        "other",
        "also",
        "about",
        "more",
        "these",
        "some",
        "them",
        "then",
        "what",
        "when",
        "where",
        "how",
        "who",
        "whom",
        "all",
        "any",
        "both",
        "most",
        "many",
        "very",
        "just",
        "only",
        "own",
        "same",
        "being",
        "because",
        "through",
        "during",
        "before",
        "after",
        "above",
        "below",
        "between",
        "under",
        "over",
        "again",
        "further",
        "once",
        "here",
        "why",
        "out",
        "off",
        "down",
        "up",
        "too",
        "nor",
        "yet",
        "so",
        "if",
        "while",
        "until",
        "list",
        "disambiguation",
        "episode",
        "season",
        "use",
        "used",
    }
)


def tokenize(text: str) -> list[str]:
    """Lowercase, extract words of 3+ chars, filter stop words."""
    words = re.findall(r"[a-z]{3,}", text.lower())
    return [w for w in words if w not in _ENGLISH_STOP]


# ── Tree traversal utilities ──────────────────────────────────────────


def build_tree_maps(idx):
    """Extract tree structure, children_map, and leaf_batches from a LazyIndex.

    Args:
        idx: An open LazyIndex instance.

    Returns:
        (tree, children_map, leaf_batches) where:
        - tree: list of node dicts from idx.get_tree_structure()
        - children_map: {parent_node_id: [child_node_id, ...]}
        - leaf_batches: {node_id: np.ndarray of item indices}
    """
    tree = idx.get_tree_structure()

    children_map: dict[int, list[int]] = defaultdict(list)
    for node in tree:
        if node["parent_id"] is not None:
            children_map[node["parent_id"]].append(node["node_id"])

    leaf_batches: dict[int, np.ndarray] = {}
    for node in tree:
        if node["is_leaf"] and node["batch_index"] >= 0:
            batch = idx.get_leaf(node["batch_index"])
            leaf_batches[node["node_id"]] = batch.column("item_index").to_numpy()

    return tree, dict(children_map), leaf_batches


def collect_descendant_indices(
    node_id: int,
    children_map: dict[int, list[int]],
    leaf_batches: dict[int, np.ndarray],
) -> np.ndarray:
    """Recursively collect all item indices under a node."""
    if node_id in leaf_batches:
        return leaf_batches[node_id]
    kids = children_map.get(node_id, [])
    if not kids:
        return np.array([], dtype=int)
    return np.concatenate([collect_descendant_indices(k, children_map, leaf_batches) for k in kids])


def _compute_depth_from_root(tree, children_map):
    """Compute depth-from-root for every node via BFS."""
    root_id = next(n["node_id"] for n in tree if n["parent_id"] is None)
    depth_from_root = {root_id: 0}
    queue = [root_id]
    while queue:
        nid = queue.pop(0)
        for child_id in children_map.get(nid, []):
            depth_from_root[child_id] = depth_from_root[nid] + 1
            queue.append(child_id)
    return depth_from_root


# ── Domain stopwords ──────────────────────────────────────────────────


def compute_domain_stopwords(
    titles: list[str],
    threshold: float = 0.10,
) -> set[str]:
    """Words appearing in >threshold fraction of titles. Corpus-level stop words.

    Args:
        titles: List of title strings.
        threshold: Fraction threshold (0.0-1.0). Words appearing in more than
            this fraction of titles are considered domain stop words.

    Returns:
        Set of domain-specific stop word strings.
    """
    n = len(titles)
    if n == 0:
        return set()

    # Count document frequency (how many titles contain each word)
    doc_freq: Counter = Counter()
    for title in titles:
        unique_words = set(tokenize(title))
        doc_freq.update(unique_words)

    cutoff = n * threshold
    return {w for w, count in doc_freq.items() if count > cutoff}


# ── Text diversity assessment ─────────────────────────────────────


@dataclass
class TextDiversityReport:
    """Result of text diversity assessment for a collection of titles."""

    unique_token_count: int
    token_item_ratio: float
    unique_title_ratio: float
    n_items: int
    is_diverse: bool
    reason: str


def assess_text_diversity(
    titles: list[str],
    *,
    min_unique_tokens: int = 50,
    min_token_ratio: float = 0.01,
    min_unique_title_ratio: float = 0.05,
) -> TextDiversityReport:
    """Cheaply assess whether a title collection has enough diversity for LLM labeling.

    Single O(n) pass: collects unique tokens (via tokenize()) and unique title strings.
    Returns a report with three signals and a boolean gate.

    Args:
        titles: List of title strings.
        min_unique_tokens: Minimum unique token count to be considered diverse.
        min_token_ratio: Minimum ratio of unique tokens to total items.
        min_unique_title_ratio: Minimum ratio of unique titles to total titles.

    Returns:
        TextDiversityReport with is_diverse=True if all thresholds pass.
    """
    n = len(titles)
    if n == 0:
        return TextDiversityReport(
            unique_token_count=0,
            token_item_ratio=0.0,
            unique_title_ratio=0.0,
            n_items=0,
            is_diverse=False,
            reason="empty title list",
        )

    unique_tokens: set[str] = set()
    unique_titles: set[str] = set()
    for title in titles:
        unique_titles.add(title)
        unique_tokens.update(tokenize(title))

    utc = len(unique_tokens)
    token_ratio = utc / n
    title_ratio = len(unique_titles) / n

    reasons = []
    if utc < min_unique_tokens:
        reasons.append(f"only {utc} unique tokens (need {min_unique_tokens})")
    if token_ratio < min_token_ratio:
        reasons.append(f"token/item ratio {token_ratio:.6f} < {min_token_ratio}")
    if title_ratio < min_unique_title_ratio:
        reasons.append(f"unique title ratio {title_ratio:.4f} < {min_unique_title_ratio}")

    is_diverse = len(reasons) == 0
    reason = "OK" if is_diverse else "; ".join(reasons)

    return TextDiversityReport(
        unique_token_count=utc,
        token_item_ratio=token_ratio,
        unique_title_ratio=title_ratio,
        n_items=n,
        is_diverse=is_diverse,
        reason=reason,
    )


# ── Frequency-based cluster labeling ─────────────────────────────


def label_clusters_frequency(
    titles: list[str],
    labels: np.ndarray,
) -> dict[int, str]:
    """Label clusters using TF-IDF token frequency (no LLM).

    Per-cluster TF-IDF: tokens frequent in one cluster but rare globally win.
    Falls back to raw title frequency if tokenization yields nothing
    (e.g. numeric-only titles like "Digit 7"). Disambiguates identical
    labels across clusters with ``(2)``, ``(3)`` suffixes.

    Args:
        titles: List of title strings, one per item.
        labels: Integer cluster assignments, same length as titles.

    Returns:
        dict mapping cluster_id -> label string.
    """
    label_arr = np.asarray(labels)
    unique_labels = sorted(set(int(l) for l in label_arr))
    n_items = len(titles)

    # Gather per-cluster data
    cluster_titles: dict[int, list[str]] = defaultdict(list)
    cluster_tokens: dict[int, Counter] = defaultdict(Counter)
    for i, cid in enumerate(label_arr):
        cid = int(cid)
        cluster_titles[cid].append(titles[i])
        cluster_tokens[cid].update(tokenize(titles[i]))

    # Global document frequency (fraction of items containing each token)
    global_df: Counter = Counter()
    for i in range(n_items):
        global_df.update(set(tokenize(titles[i])))

    n_clusters = len(unique_labels)
    cluster_names: dict[int, str] = {}

    for cid in unique_labels:
        tokens = cluster_tokens[cid]
        n_cluster = len(cluster_titles[cid])

        if tokens:
            # TF-IDF: tf = count / cluster_size, idf = log(n_clusters / df_across_clusters)
            # Use cluster-level DF: how many clusters contain this token
            token_cluster_df: Counter = Counter()
            for tok in tokens:
                for other_cid in unique_labels:
                    if tok in cluster_tokens[other_cid]:
                        token_cluster_df[tok] += 1

            scores = []
            for tok, count in tokens.items():
                tf = count / n_cluster
                idf = math.log(n_clusters / (1 + token_cluster_df[tok]))
                if idf > 0:  # skip tokens in all clusters
                    scores.append((tok, tf * idf))

            if scores:
                scores.sort(key=lambda x: -x[1])
                # Take top 2-3 tokens as label
                top = [w for w, _ in scores[:3]]
                cluster_names[cid] = " ".join(w.capitalize() for w in top)
                continue

        # Fallback: most common raw title
        title_counts = Counter(cluster_titles[cid])
        most_common_title = title_counts.most_common(1)[0][0]
        # Truncate to 50 chars
        cluster_names[cid] = most_common_title[:50]

    # Disambiguate identical labels
    label_counts = Counter(cluster_names.values())
    dups = {lbl for lbl, cnt in label_counts.items() if cnt > 1}
    if dups:
        seen: dict[str, int] = {}
        for cid in unique_labels:
            name = cluster_names[cid]
            if name in dups:
                seen[name] = seen.get(name, 0) + 1
                if seen[name] > 1:
                    cluster_names[cid] = f"{name} ({seen[name]})"

    return cluster_names


# ── Split keyword computation ─────────────────────────────────────────


def _compute_child_tfidf(child_data, n_children, top_k, bigram_check):
    """Compute TF-IDF unigrams (and optionally bigrams) for each child of a split node.

    For each child, scores its terms against the cross-child IDF and returns
    the top-k unigrams (and bigrams if requested).

    Args:
        child_data: {child_id: {'count': int, 'word_counts': Counter,
            'total_words': int, 'bigram_counts': Counter, 'total_bigrams': int}}.
        n_children: Number of children (for IDF denominator).
        top_k: Number of top keywords to return per child.
        bigram_check: Whether to compute bigram TF-IDF as well.

    Returns:
        dict mapping child_id -> {'count': int, 'unigrams': [...], 'bigrams': [...]}.
    """
    # Compute unigram document frequency across children
    word_df: Counter = Counter()
    for cdata in child_data.values():
        for word in cdata["word_counts"]:
            word_df[word] += 1

    # IDF: log(n_children / (1 + df))
    idf = {
        w: math.log(n_children / (1 + df))
        for w, df in word_df.items()
        if df < n_children  # skip words in ALL children (no discrimination)
    }

    # Compute TF-IDF per child for unigrams
    children_result = {}
    for cid, cdata in child_data.items():
        total = cdata["total_words"]
        if total == 0:
            children_result[cid] = {
                "count": cdata["count"],
                "unigrams": [],
            }
            if bigram_check:
                children_result[cid]["bigrams"] = []
            continue

        scores = []
        for word, count in cdata["word_counts"].items():
            if word in idf:
                tf = count / total
                scores.append((word, tf * idf[word]))

        scores.sort(key=lambda x: -x[1])
        entry: dict = {
            "count": cdata["count"],
            "unigrams": [(w, round(s, 6)) for w, s in scores[:top_k]],
        }

        # Bigram TF-IDF if requested
        if bigram_check:
            bigram_df: Counter = Counter()
            for cd in child_data.values():
                for bg in cd["bigram_counts"]:
                    bigram_df[bg] += 1

            bg_idf = {bg: math.log(n_children / (1 + df)) for bg, df in bigram_df.items() if df < n_children}

            bg_total = cdata["total_bigrams"]
            if bg_total > 0:
                bg_scores = []
                for bg, count in cdata["bigram_counts"].items():
                    if bg in bg_idf:
                        tf = count / bg_total
                        bg_scores.append((bg, tf * bg_idf[bg]))
                bg_scores.sort(key=lambda x: -x[1])
                entry["bigrams"] = [(bg, round(s, 6)) for bg, s in bg_scores[:top_k]]
            else:
                entry["bigrams"] = []

        children_result[cid] = entry

    return children_result


def compute_split_keywords(
    titles: list[str],
    tree: list[TreeNode],
    leaf_batches: dict[int, np.ndarray],
    children_map: dict[int, list[int]],
    *,
    max_depth_from_root: int = 3,
    min_child_items: int = 50,
    top_k: int = 10,
    domain_stopwords: set[str] | None = None,
    bigram_check: bool = False,
) -> dict:
    """Compute discriminative TF-IDF keywords for each side of each tree split.

    For each internal node (up to max_depth_from_root), treats each child as a
    "document class" and computes TF-IDF scores to find discriminative words.

    Args:
        titles: List of title strings, indexed by item index.
        tree: Tree structure from idx.get_tree_structure().
        leaf_batches: {node_id: item_indices} for leaf nodes.
        children_map: {parent_id: [child_ids]} for internal nodes.
        max_depth_from_root: Maximum tree depth to process.
        min_child_items: Skip children with fewer items than this.
        top_k: Number of top keywords to return per child.
        domain_stopwords: Additional stop words to filter.
        bigram_check: Enable PMI-based compound meaning detection.

    Returns:
        dict with keys:
            "domain_stopwords": list of domain stop words used
            "splits": {node_id: {
                "depth": int,
                "children": {child_id: {
                    "count": int,
                    "unigrams": [("word", score), ...],
                    "bigrams": [("w1_w2", score), ...],  # if bigram_check
                }},
                "bigram_needed": bool,  # if bigram_check
            }}
    """
    all_stopwords = _ENGLISH_STOP | (domain_stopwords or set())

    # Compute depth-from-root for each node via BFS.
    depth_from_root = _compute_depth_from_root(tree, children_map)

    # Find internal nodes within depth range from root.
    internal_nodes = [
        n
        for n in tree
        if depth_from_root.get(n["node_id"], 999) < max_depth_from_root
        and children_map.get(n["node_id"])  # has children
    ]

    splits = {}

    for node in internal_nodes:
        nid = node["node_id"]
        child_ids = children_map[nid]

        # Collect titles for each child
        child_data: dict[int, dict] = {}
        for cid in child_ids:
            indices = collect_descendant_indices(cid, children_map, leaf_batches)
            if len(indices) < min_child_items:
                continue

            # Tokenize all titles in this child
            word_counts: Counter = Counter()
            bigram_counts: Counter = Counter()
            total_words = 0
            total_bigrams = 0
            for idx in indices:
                if idx < len(titles):
                    words = [w for w in tokenize(titles[idx]) if w not in all_stopwords]
                    word_counts.update(words)
                    total_words += len(words)
                    if bigram_check:
                        # Re-tokenize filtering stopwords for bigrams
                        bigrams = [f"{words[i]}_{words[i + 1]}" for i in range(len(words) - 1)]
                        bigram_counts.update(bigrams)
                        total_bigrams += len(bigrams)

            child_data[cid] = {
                "count": int(len(indices)),
                "word_counts": word_counts,
                "total_words": total_words,
                "bigram_counts": bigram_counts,
                "total_bigrams": total_bigrams,
            }

        if len(child_data) < 2:
            continue

        n_children = len(child_data)
        children_result = _compute_child_tfidf(child_data, n_children, top_k, bigram_check)

        split_entry: dict = {
            "depth": depth_from_root.get(nid, 0),
            "children": children_result,
        }

        # Bigram necessity check
        if bigram_check:
            split_entry["bigram_needed"] = _check_bigram_needed(children_result, top_k=20)

        splits[nid] = split_entry

    return {
        "domain_stopwords": sorted(domain_stopwords or []),
        "splits": splits,
    }


def _project_terms_onto_hyperplane(term_items, embeddings, hp, all_stopwords, min_term_count):
    """Project vocabulary terms onto a hyperplane and rank them by projection score.

    Computes a "term embedding" for each term (centroid of item embeddings whose
    titles contain that term), then projects onto the hyperplane direction.

    Args:
        term_items: {term: [item_index, ...]} mapping terms to their item indices.
        embeddings: (n, dim) embedding array.
        hp: (dim,) float64 normalized hyperplane direction.
        all_stopwords: Set of stop words (unused here, kept for API clarity).
        min_term_count: Minimum item count for a term to be included.

    Returns:
        dict mapping term -> float projection score, or empty dict if no terms qualify.
    """
    # Filter by min_term_count
    filtered = {w: idxs for w, idxs in term_items.items() if len(idxs) >= min_term_count}

    if not filtered:
        return {}

    # Compute term embeddings (centroid of item embeddings containing each term)
    # and project onto hyperplane
    term_scores: dict[str, float] = {}
    for word, idxs in filtered.items():
        idx_arr = np.array(idxs)
        term_centroid = embeddings[idx_arr].mean(axis=0).astype(np.float64)
        term_scores[word] = float(term_centroid @ hp)

    return term_scores


def compute_embedding_keywords(
    titles: list[str],
    embeddings: np.ndarray,
    tree: list[TreeNode],
    leaf_batches: dict[int, np.ndarray],
    children_map: dict[int, list[int]],
    hyperplanes: dict[int, np.ndarray],
    *,
    max_depth_from_root: int = 3,
    min_child_items: int = 50,
    top_k: int = 10,
    domain_stopwords: set[str] | None = None,
    min_term_count: int = 3,
) -> dict:
    """Compute discriminative keywords by projecting term embeddings onto split hyperplanes.

    For each internal node, builds a "term embedding" (centroid of all item embeddings
    whose titles contain that term), then projects onto the node's PCA hyperplane to
    find terms most aligned with each child's side of the split.

    Drop-in replacement for compute_split_keywords() — same output format.

    Args:
        titles: List of title strings, indexed by item index.
        embeddings: (n, dim) float32 array of full embeddings.
        tree: Tree structure from idx.get_tree_structure().
        leaf_batches: {node_id: item_indices} for leaf nodes.
        children_map: {parent_id: [child_ids]} for internal nodes.
        hyperplanes: {node_id: (num_bits, dim)} from idx.get_split_hyperplanes().
        max_depth_from_root: Maximum tree depth to process.
        min_child_items: Skip children with fewer items than this.
        top_k: Number of top keywords to return per child.
        domain_stopwords: Additional stop words to filter.
        min_term_count: Minimum number of items containing a term to include it.

    Returns:
        dict with keys:
            "domain_stopwords": list of domain stop words used
            "splits": {node_id: {
                "depth": int,
                "children": {child_id: {
                    "count": int,
                    "unigrams": [("word", score), ...],
                }},
            }}
    """
    all_stopwords = _ENGLISH_STOP | (domain_stopwords or set())

    # Compute depth-from-root for each node via BFS.
    depth_from_root = _compute_depth_from_root(tree, children_map)

    # Find internal nodes within depth range that have hyperplanes
    internal_nodes = [
        n
        for n in tree
        if depth_from_root.get(n["node_id"], 999) < max_depth_from_root
        and children_map.get(n["node_id"])
        and n["node_id"] in hyperplanes
    ]

    splits = {}

    for node in internal_nodes:
        nid = node["node_id"]
        child_ids = children_map[nid]

        # Get first PCA direction (highest variance), normalized
        hp = hyperplanes[nid][0].astype(np.float64)
        hp_norm = np.linalg.norm(hp)
        if hp_norm < 1e-12:
            continue
        hp = hp / hp_norm

        # Collect all descendant item indices for this node
        all_indices = collect_descendant_indices(nid, children_map, leaf_batches)
        if len(all_indices) == 0:
            continue

        # Build term → item indices mapping from titles under this node
        term_items: dict[str, list[int]] = defaultdict(list)
        for idx in all_indices:
            if idx < len(titles):
                words = tokenize(titles[idx])
                for w in words:
                    if w not in all_stopwords:
                        term_items[w].append(idx)

        # Project terms onto hyperplane and rank by projection score
        term_scores = _project_terms_onto_hyperplane(term_items, embeddings, hp, all_stopwords, min_term_count)

        if not term_scores:
            continue

        # Compute child centroid projections onto hyperplane
        child_data: dict[int, dict] = {}
        for cid in child_ids:
            child_indices = collect_descendant_indices(cid, children_map, leaf_batches)
            if len(child_indices) < min_child_items:
                continue
            child_centroid = embeddings[child_indices].mean(axis=0).astype(np.float64)
            child_proj = float(child_centroid @ hp)
            child_data[cid] = {
                "count": int(len(child_indices)),
                "proj": child_proj,
                "indices": child_indices,
            }

        if len(child_data) < 2:
            continue

        # Assign terms to the child whose centroid projection is closest
        child_projs = {cid: cd["proj"] for cid, cd in child_data.items()}

        children_result = {}
        for cid, cd in child_data.items():
            children_result[cid] = {
                "count": cd["count"],
                "unigrams": [],
            }

        # For each term, assign to nearest child by projection
        for word, score in term_scores.items():
            best_cid = min(child_projs, key=lambda c: abs(score - child_projs[c]))
            # Score = distance from midpoint of other children's projections
            magnitude = abs(score - child_projs[best_cid])
            children_result[best_cid]["unigrams"].append((word, magnitude))

        # Sort by score descending and keep top_k per child
        for cid in children_result:
            unigrams = children_result[cid]["unigrams"]
            unigrams.sort(key=lambda x: -x[1])
            children_result[cid]["unigrams"] = [(w, round(s, 6)) for w, s in unigrams[:top_k]]

        splits[nid] = {
            "depth": depth_from_root.get(nid, 0),
            "children": children_result,
        }

    return {
        "domain_stopwords": sorted(domain_stopwords or []),
        "splits": splits,
    }


def _check_bigram_needed(
    children_result: dict,
    top_k: int = 20,
) -> bool:
    """Check if bigrams resolve ambiguous unigrams.

    A unigram is "ambiguous" if it appears in the top-k keywords of
    multiple children. Returns True if >2 ambiguous unigrams exist
    and bigrams resolve them (appear in top-20 of only one child).
    """
    # Collect top unigrams per child
    child_top_words: dict[int, set[str]] = {}
    for cid, entry in children_result.items():
        child_top_words[cid] = {w for w, _ in entry["unigrams"][:top_k]}

    # Find words appearing in top-k of multiple children
    word_child_count: Counter = Counter()
    for words in child_top_words.values():
        word_child_count.update(words)
    ambiguous = {w for w, c in word_child_count.items() if c > 1}

    if len(ambiguous) <= 2:
        return False

    # Check if bigrams resolve ambiguity
    child_top_bigrams: dict[int, set[str]] = {}
    for cid, entry in children_result.items():
        bgs = entry.get("bigrams", [])
        child_top_bigrams[cid] = {bg for bg, _ in bgs[:top_k]}

    # For each ambiguous unigram, check if a bigram containing it
    # appears in only one child
    resolved = 0
    for word in ambiguous:
        for cid, bgs in child_top_bigrams.items():
            matching_bgs = {bg for bg in bgs if word in bg.split("_")}
            if matching_bgs:
                # Check if these bigrams are unique to this child
                other_bgs = set()
                for other_cid, other_set in child_top_bigrams.items():
                    if other_cid != cid:
                        other_bgs.update(other_set)
                if matching_bgs - other_bgs:
                    resolved += 1
                    break

    return resolved > 2


# ── Path formatting ───────────────────────────────────────────────────


def format_split_path(
    item_index: int,
    split_keywords: dict,
    tree: list[TreeNode],
    leaf_batches: dict[int, np.ndarray],
    children_map: dict[int, list[int]],
    *,
    top_k: int = 3,
) -> list[str]:
    """Return the keyword path for a single item from root to leaf.

    Traces which child the item belongs to at each split level and
    returns the top-k keywords for that child.

    Args:
        item_index: The item's index in the dataset.
        split_keywords: Output of compute_split_keywords().
        tree: Tree structure list.
        leaf_batches: {node_id: item_indices}.
        children_map: {parent_id: [child_ids]}.
        top_k: Number of keywords per path step.

    Returns:
        List of comma-separated keyword strings, one per split level.
        E.g.: ["screw,plate,fixation", "pedicle,cervical,spine"]
    """
    splits = split_keywords.get("splits", {})
    if not splits:
        return []

    by_id = {n["node_id"]: n for n in tree}

    # Find which leaf contains this item
    item_leaf = None
    for nid, indices in leaf_batches.items():
        if item_index in indices:
            item_leaf = nid
            break
    if item_leaf is None:
        return []

    # Build path from leaf to root
    leaf_to_root = []
    current = item_leaf
    while current is not None:
        leaf_to_root.append(current)
        current = by_id[current].get("parent_id")
    root_to_leaf = list(reversed(leaf_to_root))

    # At each split node, find which child the item belongs to
    path = []
    for nid in root_to_leaf:
        if nid not in splits:
            continue
        split = splits[nid]
        # Find which child the item is under
        for child_id, child_info in split["children"].items():
            child_id_int = int(child_id)
            # Check if item_index is a descendant of this child
            desc = collect_descendant_indices(child_id_int, children_map, leaf_batches)
            if item_index in desc:
                words = [w for w, _ in child_info["unigrams"][:top_k]]
                if words:
                    path.append(",".join(words))
                break

    return path
