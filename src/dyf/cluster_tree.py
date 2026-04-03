"""Cluster-Tree DAG for hierarchical cluster labeling.

Connects BIRCH spatial clusters to DYF tree nodes via item overlap,
producing a DAG that enables:
- **Path labels**: deterministic tree-path context per cluster
- **Sibling keywords**: contrastive TF-IDF within path-sharing groups

Pure functions, no I/O. Depends on ``dyf.categorical.CategoryGraph``
and tokenization utilities from ``dyf.splits``.
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict

import numpy as np

from dyf.categorical import CategoryGraph
from dyf.splits import collect_descendant_indices, tokenize

# ── DAG construction ──────────────────────────────────────────────────


def build_cluster_tree_dag(
    tree: list[dict],
    children_map: dict[int, list[int]],
    leaf_batches: dict[int, np.ndarray],
    cluster_labels: np.ndarray,
    n_clusters: int,
    *,
    straddle_threshold: float = 0.15,
    max_tree_depth: int | None = None,
) -> CategoryGraph:
    """Build a DAG connecting BIRCH clusters to tree nodes they overlap with.

    Parameters
    ----------
    tree : list[dict]
        Tree structure from ``idx.get_tree_structure()``.
    children_map : dict[int, list[int]]
        ``{parent_id: [child_ids]}``.
    leaf_batches : dict[int, np.ndarray]
        ``{node_id: item_indices}`` for leaf nodes.
    cluster_labels : (n,) int array
        Per-item BIRCH cluster ID.
    n_clusters : int
        The cluster resolution (e.g. 25), used for node naming.
    straddle_threshold : float
        Minimum overlap fraction to create a cluster→tree edge.
    max_tree_depth : int or None
        If set, only consider tree nodes up to this depth from root.

    Returns
    -------
    CategoryGraph
        DAG with tree-internal edges and cluster-to-tree edges.
        Node names: ``"tree_0"``, ``"tree_1"``, ... for tree nodes;
        ``"cluster_25_0"``, ``"cluster_25_14"``, ... for clusters.
    """
    cluster_labels = np.asarray(cluster_labels)

    # Compute depth_from_root via BFS
    root_id = next(n['node_id'] for n in tree if n['parent_id'] is None)
    depth_from_root: dict[int, int] = {root_id: 0}
    queue = [root_id]
    while queue:
        nid = queue.pop(0)
        for child_id in children_map.get(nid, []):
            depth_from_root[child_id] = depth_from_root[nid] + 1
            queue.append(child_id)

    # Identify internal nodes (have children) within depth limit
    internal_nodes = []
    for node in tree:
        nid = node['node_id']
        if not children_map.get(nid):
            continue  # leaf
        d = depth_from_root.get(nid, 999)
        if max_tree_depth is not None and d >= max_tree_depth:
            continue
        internal_nodes.append(nid)

    # Pre-compute descendant items for the *deepest* internal nodes.
    # We want the most specific tree nodes that still have children.
    # "Deepest internal" = internal nodes whose children are all leaves
    # OR internal nodes at max_tree_depth - 1.
    # Actually, the plan says "deepest internal level" — let's compute
    # overlap at ALL internal nodes but only keep the deepest match.

    # Pre-compute descendant item sets for each internal node
    descendant_items: dict[int, set[int]] = {}
    for nid in internal_nodes:
        desc = collect_descendant_indices(nid, children_map, leaf_batches)
        descendant_items[nid] = set(desc.tolist())

    # Also compute descendants for children of internal nodes
    # (needed to connect clusters to the right child, not the parent)
    child_descendant_items: dict[int, set[int]] = {}
    for nid in internal_nodes:
        for cid in children_map.get(nid, []):
            if cid not in child_descendant_items:
                desc = collect_descendant_indices(cid, children_map, leaf_batches)
                child_descendant_items[cid] = set(desc.tolist())

    # Pre-compute cluster item sets
    unique_clusters = sorted(set(int(c) for c in cluster_labels))
    cluster_items: dict[int, set[int]] = {}
    for cid in unique_clusters:
        cluster_items[cid] = set(np.where(cluster_labels == cid)[0].tolist())

    # Build edges
    edges: list[tuple[str, str, float]] = []

    # Tree-internal edges: parent → child
    for parent_nid in internal_nodes:
        for child_nid in children_map.get(parent_nid, []):
            edges.append((
                f"tree_{parent_nid}",
                f"tree_{child_nid}",
                1.0,
            ))

    # Cluster → tree edges: connect each cluster to tree nodes at the
    # deepest level where overlap exceeds threshold.
    # Strategy: for each cluster, check overlap with all internal nodes'
    # children. Connect to the deepest tree node with sufficient overlap.

    # Collect all tree node IDs that are children of internal nodes
    # (these are the "attachment points" for clusters)
    attachment_nodes: list[int] = []
    for nid in internal_nodes:
        attachment_nodes.extend(children_map.get(nid, []))
    attachment_nodes = sorted(set(attachment_nodes))

    # Sort by depth descending — prefer deeper matches
    attachment_by_depth = sorted(
        attachment_nodes,
        key=lambda nid: depth_from_root.get(nid, 0),
        reverse=True,
    )

    for cid in unique_clusters:
        c_items = cluster_items[cid]
        c_size = len(c_items)
        if c_size == 0:
            continue

        # Track which items are already "claimed" by a deeper tree node
        claimed: set[int] = set()
        cluster_node_name = f"cluster_{n_clusters}_{cid}"

        for tnid in attachment_by_depth:
            if tnid in child_descendant_items:
                t_items = child_descendant_items[tnid]
            elif tnid in descendant_items:
                t_items = descendant_items[tnid]
            else:
                # Leaf node with no pre-computed descendants
                desc = collect_descendant_indices(tnid, children_map, leaf_batches)
                t_items = set(desc.tolist())

            unclaimed_overlap = c_items & t_items - claimed
            overlap_frac = len(unclaimed_overlap) / c_size

            if overlap_frac >= straddle_threshold:
                edges.append((
                    f"tree_{tnid}",
                    cluster_node_name,
                    round(overlap_frac, 4),
                ))
                claimed |= unclaimed_overlap

        # If no tree node claimed this cluster, attach to root
        if not any(e[1] == cluster_node_name for e in edges):
            edges.append((
                f"tree_{root_id}",
                cluster_node_name,
                1.0,
            ))

    return CategoryGraph.from_edges(edges)


# ── Path labels ───────────────────────────────────────────────────────


def derive_path_labels(
    dag: CategoryGraph,
    split_keywords: dict,
    n_clusters: int,
    *,
    top_k: int = 3,
) -> dict[int, str]:
    """Derive deterministic path labels for each cluster from the DAG.

    Parameters
    ----------
    dag : CategoryGraph
        DAG from ``build_cluster_tree_dag()``.
    split_keywords : dict
        Output of ``compute_split_keywords()``.
    n_clusters : int
        Cluster resolution (e.g. 25).
    top_k : int
        Number of keywords per path step.

    Returns
    -------
    dict[int, str]
        ``{cluster_id: "keyword1, keyword2 / keyword3, keyword4"}``.
        Single path → ``"cardiac / pacemaker"``.
        Straddling → ``"cardiac / {pacemaker, defibrillator}"``.
    """
    splits = split_keywords.get('splits', {})

    # Build a quick lookup: tree node → its parent tree node
    tree_parent: dict[str, str] = {}
    for node_name in dag.all_nodes():
        if node_name.startswith("tree_"):
            parents = dag.get_parents(node_name)
            tree_parents = [p for p in parents if p.startswith("tree_")]
            if tree_parents:
                tree_parent[node_name] = tree_parents[0]

    # For each split node, build a mapping: child_tree_node → keywords
    # splits dict is keyed by node_id (int or str), children by child_id
    node_keywords: dict[str, str] = {}
    for nid_key, split_data in splits.items():
        children = split_data.get('children', {})
        for cid_key, cinfo in children.items():
            child_nid = int(cid_key)
            words = [w for w, _ in cinfo.get('unigrams', [])[:top_k]]
            if words:
                node_keywords[f"tree_{child_nid}"] = ", ".join(words)

    result: dict[int, str] = {}

    for cid in range(max(1, n_clusters * 2)):  # iterate over possible cluster IDs
        cluster_name = f"cluster_{n_clusters}_{cid}"
        if cluster_name not in dag.all_nodes():
            continue

        # Get parent tree nodes
        parents = dag.get_parents(cluster_name)
        tree_parents = [p for p in parents if p.startswith("tree_")]

        if not tree_parents:
            result[cid] = ""
            continue

        # Trace each parent to root, collecting keywords along the way
        paths: list[list[str]] = []
        for tp in tree_parents:
            path: list[str] = []
            current = tp
            while current is not None:
                if current in node_keywords:
                    path.append(node_keywords[current])
                current = tree_parent.get(current)
            path.reverse()
            paths.append(path)

        if len(paths) == 1:
            # Single path
            result[cid] = " / ".join(paths[0]) if paths[0] else ""
        elif len(paths) > 1:
            # Straddling: find common prefix, show divergence
            # Find the longest common prefix
            min_len = min(len(p) for p in paths)
            common_prefix_len = 0
            for i in range(min_len):
                if all(p[i] == paths[0][i] for p in paths):
                    common_prefix_len = i + 1
                else:
                    break

            prefix_parts = paths[0][:common_prefix_len]

            # Collect diverging parts
            divergent = []
            for p in paths:
                remaining = p[common_prefix_len:]
                if remaining:
                    divergent.append(remaining[0])

            if divergent:
                brace = "{" + ", ".join(sorted(set(divergent))) + "}"
                parts = prefix_parts + [brace]
            else:
                parts = prefix_parts

            result[cid] = " / ".join(parts) if parts else ""
        else:
            result[cid] = ""

    return result


# ── Sibling keywords ──────────────────────────────────────────────────


def compute_sibling_keywords(
    dag: CategoryGraph,
    titles: list[str],
    cluster_labels: np.ndarray,
    n_clusters: int,
    *,
    top_k: int = 8,
) -> dict[int, list[tuple[str, float]]]:
    """Compute contrastive TF-IDF keywords for clusters sharing tree parents.

    Parameters
    ----------
    dag : CategoryGraph
    titles : list[str]
        Per-item title strings.
    cluster_labels : (n,) int array
        Per-item BIRCH cluster ID.
    n_clusters : int
        Cluster resolution.
    top_k : int
        Keywords per cluster.

    Returns
    -------
    dict[int, list[tuple[str, float]]]
        ``{cluster_id: [("word", score), ...]}``.
    """
    cluster_labels = np.asarray(cluster_labels)
    unique_clusters = sorted(set(int(c) for c in cluster_labels))

    # Map cluster IDs to their parent tree-node sets
    cluster_parent_set: dict[int, frozenset[str]] = {}
    for cid in unique_clusters:
        cluster_name = f"cluster_{n_clusters}_{cid}"
        parents = dag.get_parents(cluster_name)
        tree_parents = frozenset(p for p in parents if p.startswith("tree_"))
        cluster_parent_set[cid] = tree_parents

    # Group clusters by shared parent set (siblings)
    parent_groups: dict[frozenset[str], list[int]] = defaultdict(list)
    for cid, pset in cluster_parent_set.items():
        parent_groups[pset].append(cid)

    # Pre-compute cluster item indices
    cluster_items: dict[int, np.ndarray] = {}
    for cid in unique_clusters:
        cluster_items[cid] = np.where(cluster_labels == cid)[0]

    result: dict[int, list[tuple[str, float]]] = {}

    for pset, sibling_cids in parent_groups.items():
        if len(sibling_cids) < 2:
            # Lone cluster — fallback to corpus-wide TF-IDF
            for cid in sibling_cids:
                result[cid] = _corpus_tfidf(
                    titles, cluster_items[cid], top_k)
            continue

        # Contrastive TF-IDF within sibling group
        kw = _sibling_group_tfidf(titles, sibling_cids, cluster_items, top_k)
        result.update(kw)

    return result


def _sibling_group_tfidf(
    titles: list[str],
    sibling_cids: list[int],
    cluster_items: dict[int, np.ndarray],
    top_k: int,
) -> dict[int, list[tuple[str, float]]]:
    """Compute contrastive TF-IDF for a group of sibling clusters."""
    n_siblings = len(sibling_cids)

    # Tokenize per cluster
    cluster_word_counts: dict[int, Counter] = {}
    cluster_total_words: dict[int, int] = {}

    for cid in sibling_cids:
        wc: Counter = Counter()
        for idx in cluster_items[cid]:
            if idx < len(titles):
                words = tokenize(titles[idx])
                wc.update(words)
        cluster_word_counts[cid] = wc
        cluster_total_words[cid] = sum(wc.values())

    # Document frequency across siblings
    word_df: Counter = Counter()
    for wc in cluster_word_counts.values():
        for word in wc:
            word_df[word] += 1

    # IDF: only keep words that don't appear in ALL siblings
    idf = {
        w: math.log(n_siblings / (1 + df))
        for w, df in word_df.items()
        if df < n_siblings
    }

    result: dict[int, list[tuple[str, float]]] = {}
    for cid in sibling_cids:
        total = cluster_total_words[cid]
        if total == 0:
            result[cid] = []
            continue

        scores = []
        for word, count in cluster_word_counts[cid].items():
            if word in idf:
                tf = count / total
                scores.append((word, tf * idf[word]))

        scores.sort(key=lambda x: -x[1])
        result[cid] = [(w, round(s, 6)) for w, s in scores[:top_k]]

    return result


def _corpus_tfidf(
    titles: list[str],
    cluster_indices: np.ndarray,
    top_k: int,
) -> list[tuple[str, float]]:
    """Fallback TF-IDF: this cluster vs entire corpus."""
    # Cluster word counts
    cluster_wc: Counter = Counter()
    for idx in cluster_indices:
        if idx < len(titles):
            words = tokenize(titles[idx])
            cluster_wc.update(words)
    cluster_total = sum(cluster_wc.values())
    if cluster_total == 0:
        return []

    # Corpus word counts (all items)
    corpus_wc: Counter = Counter()
    for title in titles:
        corpus_wc.update(tokenize(title))
    corpus_total = sum(corpus_wc.values())
    if corpus_total == 0:
        return []

    # Simple TF-IDF: tf(cluster) * log(corpus_total / (1 + corpus_count))
    scores = []
    for word, count in cluster_wc.items():
        tf = count / cluster_total
        idf = math.log(corpus_total / (1 + corpus_wc.get(word, 0)))
        scores.append((word, tf * idf))

    scores.sort(key=lambda x: -x[1])
    return [(w, round(s, 6)) for w, s in scores[:top_k]]


# ── Formatting ────────────────────────────────────────────────────────


def format_cluster_context(
    cluster_id: int,
    path_label: str,
    sibling_keywords: list[tuple[str, float]],
    sibling_labels: dict[int, str] | None = None,
) -> str:
    """Format cluster context for an LLM labeling prompt.

    Parameters
    ----------
    cluster_id : int
    path_label : str
        From ``derive_path_labels()``.
    sibling_keywords : list[tuple[str, float]]
        From ``compute_sibling_keywords()``.
    sibling_labels : dict[int, str] or None
        Labels of sibling clusters (for disambiguation context).

    Returns
    -------
    str
        Formatted context string.
    """
    parts = []

    if path_label:
        parts.append(f"Tree path: {path_label}")

    if sibling_keywords:
        kw_str = ", ".join(w for w, _ in sibling_keywords)
        parts.append(
            f"Distinguishing keywords (vs path siblings): {kw_str}")

    if sibling_labels:
        others = [f'"{v}"' for k, v in sorted(sibling_labels.items())
                  if k != cluster_id]
        if others:
            parts.append(f"Sibling clusters: {', '.join(others[:10])}")

    return "\n".join(parts)
