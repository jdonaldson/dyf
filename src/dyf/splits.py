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

import numpy as np

# English stop words (compact set covering common function words)
_ENGLISH_STOP = frozenset({
    'the', 'a', 'an', 'of', 'in', 'on', 'at', 'to', 'for', 'and', 'or',
    'is', 'was', 'are', 'were', 'be', 'been', 'by', 'with', 'from', 'as',
    'it', 'its', 'this', 'that', 'not', 'but', 'has', 'had', 'have', 'do',
    'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might',
    'can', 'shall', 'each', 'which', 'their', 'there', 'than', 'been',
    'into', 'such', 'other', 'also', 'about', 'more', 'these', 'some',
    'them', 'then', 'what', 'when', 'where', 'how', 'who', 'whom',
    'all', 'any', 'both', 'most', 'many', 'very', 'just', 'only',
    'own', 'same', 'being', 'because', 'through', 'during', 'before',
    'after', 'above', 'below', 'between', 'under', 'over', 'again',
    'further', 'once', 'here', 'why', 'out', 'off', 'down', 'up',
    'too', 'nor', 'yet', 'so', 'if', 'while', 'until',
    'list', 'disambiguation', 'episode', 'season', 'use', 'used',
})


def _tokenize(text: str) -> list[str]:
    """Lowercase, extract words of 3+ chars, filter stop words."""
    words = re.findall(r'[a-z]{3,}', text.lower())
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
        if node['parent_id'] is not None:
            children_map[node['parent_id']].append(node['node_id'])

    leaf_batches: dict[int, np.ndarray] = {}
    for node in tree:
        if node['is_leaf'] and node['batch_index'] >= 0:
            batch = idx.get_leaf(node['batch_index'])
            leaf_batches[node['node_id']] = batch.column('item_index').to_numpy()

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
    return np.concatenate([
        collect_descendant_indices(k, children_map, leaf_batches)
        for k in kids
    ])


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
        unique_words = set(_tokenize(title))
        doc_freq.update(unique_words)

    cutoff = n * threshold
    return {w for w, count in doc_freq.items() if count > cutoff}


# ── Split keyword computation ─────────────────────────────────────────


def compute_split_keywords(
    titles: list[str],
    tree: list[dict],
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

    # Find the root node (parent_id is None) and compute depth-from-root
    # for each node. DYF trees may use either convention (root at max or
    # min depth value), so we compute depth_from_root via BFS.
    root_id = next(n['node_id'] for n in tree if n['parent_id'] is None)

    depth_from_root: dict[int, int] = {root_id: 0}
    queue = [root_id]
    while queue:
        nid = queue.pop(0)
        for child_id in children_map.get(nid, []):
            depth_from_root[child_id] = depth_from_root[nid] + 1
            queue.append(child_id)

    # Find internal nodes within depth range from root.
    internal_nodes = [
        n for n in tree
        if depth_from_root.get(n['node_id'], 999) < max_depth_from_root
        and children_map.get(n['node_id'])  # has children
    ]

    splits = {}

    for node in internal_nodes:
        nid = node['node_id']
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
                    words = [w for w in _tokenize(titles[idx])
                             if w not in all_stopwords]
                    word_counts.update(words)
                    total_words += len(words)
                    if bigram_check:
                        # Re-tokenize filtering stopwords for bigrams
                        bigrams = [
                            f"{words[i]}_{words[i+1]}"
                            for i in range(len(words) - 1)
                        ]
                        bigram_counts.update(bigrams)
                        total_bigrams += len(bigrams)

            child_data[cid] = {
                'count': int(len(indices)),
                'word_counts': word_counts,
                'total_words': total_words,
                'bigram_counts': bigram_counts,
                'total_bigrams': total_bigrams,
            }

        if len(child_data) < 2:
            continue

        n_children = len(child_data)

        # Compute unigram document frequency across children
        word_df: Counter = Counter()
        for cdata in child_data.values():
            for word in cdata['word_counts']:
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
            total = cdata['total_words']
            if total == 0:
                children_result[cid] = {
                    'count': cdata['count'],
                    'unigrams': [],
                }
                if bigram_check:
                    children_result[cid]['bigrams'] = []
                continue

            scores = []
            for word, count in cdata['word_counts'].items():
                if word in idf:
                    tf = count / total
                    scores.append((word, tf * idf[word]))

            scores.sort(key=lambda x: -x[1])
            entry: dict = {
                'count': cdata['count'],
                'unigrams': [(w, round(s, 6)) for w, s in scores[:top_k]],
            }

            # Bigram TF-IDF if requested
            if bigram_check:
                bigram_df: Counter = Counter()
                for cd in child_data.values():
                    for bg in cd['bigram_counts']:
                        bigram_df[bg] += 1

                bg_idf = {
                    bg: math.log(n_children / (1 + df))
                    for bg, df in bigram_df.items()
                    if df < n_children
                }

                bg_total = cdata['total_bigrams']
                if bg_total > 0:
                    bg_scores = []
                    for bg, count in cdata['bigram_counts'].items():
                        if bg in bg_idf:
                            tf = count / bg_total
                            bg_scores.append((bg, tf * bg_idf[bg]))
                    bg_scores.sort(key=lambda x: -x[1])
                    entry['bigrams'] = [
                        (bg, round(s, 6)) for bg, s in bg_scores[:top_k]
                    ]
                else:
                    entry['bigrams'] = []

            children_result[cid] = entry

        split_entry: dict = {
            'depth': depth_from_root.get(nid, 0),
            'children': children_result,
        }

        # Bigram necessity check
        if bigram_check:
            split_entry['bigram_needed'] = _check_bigram_needed(
                children_result, top_k=20)

        splits[nid] = split_entry

    return {
        'domain_stopwords': sorted(domain_stopwords or []),
        'splits': splits,
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
        child_top_words[cid] = {w for w, _ in entry['unigrams'][:top_k]}

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
        bgs = entry.get('bigrams', [])
        child_top_bigrams[cid] = {bg for bg, _ in bgs[:top_k]}

    # For each ambiguous unigram, check if a bigram containing it
    # appears in only one child
    resolved = 0
    for word in ambiguous:
        for cid, bgs in child_top_bigrams.items():
            matching_bgs = {bg for bg in bgs
                           if word in bg.split('_')}
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
    tree: list[dict],
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
    splits = split_keywords.get('splits', {})
    if not splits:
        return []

    by_id = {n['node_id']: n for n in tree}

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
        current = by_id[current].get('parent_id')
    root_to_leaf = list(reversed(leaf_to_root))

    # At each split node, find which child the item belongs to
    path = []
    for nid in root_to_leaf:
        if nid not in splits:
            continue
        split = splits[nid]
        # Find which child the item is under
        for child_id, child_info in split['children'].items():
            child_id_int = int(child_id)
            # Check if item_index is a descendant of this child
            desc = collect_descendant_indices(
                child_id_int, children_map, leaf_batches)
            if item_index in desc:
                words = [w for w, _ in child_info['unigrams'][:top_k]]
                if words:
                    path.append(','.join(words))
                break

    return path
