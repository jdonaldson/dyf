"""Categorical DAG structures for hierarchical label management.

Provides a reusable CategoryGraph abstraction for domain taxonomies
(GMDN, ICD, NAICS, etc.), general label coarsening, multi-level
Fisher dimension weighting, and two-pass embedding diagnostics.
"""

from __future__ import annotations

import json
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Callable, Optional, Union

import numpy as np


# ── CategoryGraph ────────────────────────────────────────────────────────


@dataclass
class CategoryGraph:
    """DAG of string-labeled category nodes with weighted edges.

    A hierarchy is a DAG where each node has at most one parent (a tree).
    This structure supports general DAGs (multiple parents) but most
    domain taxonomies will be trees.

    Attributes
    ----------
    children : dict mapping parent → list of (child, weight)
    parents  : dict mapping child → list of (parent, weight)
    roots    : nodes with no parents
    leaves   : nodes with no children
    """

    children: dict[str, list[tuple[str, float]]] = field(default_factory=dict)
    parents: dict[str, list[tuple[str, float]]] = field(default_factory=dict)
    roots: list[str] = field(default_factory=list)
    leaves: list[str] = field(default_factory=list)

    # ── Navigation ───────────────────────────────────────────────────

    def all_nodes(self) -> set[str]:
        """Return all nodes in the graph."""
        nodes = set(self.children.keys()) | set(self.parents.keys())
        for lst in self.children.values():
            nodes.update(c for c, _ in lst)
        for lst in self.parents.values():
            nodes.update(p for p, _ in lst)
        nodes.update(self.roots)
        nodes.update(self.leaves)
        return nodes

    def get_children(self, node: str) -> list[str]:
        """Direct children of *node*."""
        return [c for c, _ in self.children.get(node, [])]

    def get_parents(self, node: str) -> list[str]:
        """Direct parents of *node*."""
        return [p for p, _ in self.parents.get(node, [])]

    def get_ancestors(self, node: str, max_depth: int = 100) -> set[str]:
        """All ancestors via BFS upward."""
        ancestors: set[str] = set()
        frontier = {node}
        for _ in range(max_depth):
            next_frontier: set[str] = set()
            for n in frontier:
                for parent, _ in self.parents.get(n, []):
                    if parent not in ancestors:
                        ancestors.add(parent)
                        next_frontier.add(parent)
            if not next_frontier:
                break
            frontier = next_frontier
        return ancestors

    def get_descendants(self, node: str, max_depth: int = 100) -> set[str]:
        """All descendants via BFS downward."""
        descendants: set[str] = set()
        frontier = {node}
        for _ in range(max_depth):
            next_frontier: set[str] = set()
            for n in frontier:
                for child, _ in self.children.get(n, []):
                    if child not in descendants:
                        descendants.add(child)
                        next_frontier.add(child)
            if not next_frontier:
                break
            frontier = next_frontier
        return descendants

    def get_depth(self, node: str) -> int:
        """Shortest path length from any root to *node* (0 for roots)."""
        if node in self.roots:
            return 0
        # BFS from roots
        visited: dict[str, int] = {r: 0 for r in self.roots}
        queue = deque(self.roots)
        while queue:
            current = queue.popleft()
            d = visited[current]
            for child, _ in self.children.get(current, []):
                if child not in visited:
                    visited[child] = d + 1
                    queue.append(child)
        return visited.get(node, -1)

    def lca_depth(self, node_a: str, node_b: str) -> int:
        """Depth of the lowest common ancestor of two nodes.

        Returns the depth of the deepest node that is an ancestor of both
        *node_a* and *node_b* (or one of them, if one is an ancestor of the
        other).  Returns -1 if either node is absent or they share no
        common ancestor.
        """
        all_nodes = self.all_nodes()
        if node_a not in all_nodes or node_b not in all_nodes:
            return -1
        if node_a == node_b:
            return self.get_depth(node_a)
        ancestors_a = self.get_ancestors(node_a) | {node_a}
        ancestors_b = self.get_ancestors(node_b) | {node_b}
        common = ancestors_a & ancestors_b
        if not common:
            return -1
        return max(self.get_depth(c) for c in common)

    def max_depth(self) -> int:
        """Maximum depth across all nodes."""
        if not self.roots:
            return 0
        visited: dict[str, int] = {r: 0 for r in self.roots}
        queue = deque(self.roots)
        while queue:
            current = queue.popleft()
            d = visited[current]
            for child, _ in self.children.get(current, []):
                if child not in visited:
                    visited[child] = d + 1
                    queue.append(child)
        return max(visited.values()) if visited else 0

    def nodes_at_depth(self, depth: int) -> list[str]:
        """All nodes whose shortest root distance equals *depth*."""
        if not self.roots:
            return []
        visited: dict[str, int] = {r: 0 for r in self.roots}
        queue = deque(self.roots)
        while queue:
            current = queue.popleft()
            d = visited[current]
            for child, _ in self.children.get(current, []):
                if child not in visited:
                    visited[child] = d + 1
                    queue.append(child)
        return [n for n, d in visited.items() if d == depth]

    def is_tree(self) -> bool:
        """True if every node has at most one parent."""
        for plist in self.parents.values():
            if len(plist) > 1:
                return False
        return True

    def summary(self) -> str:
        """Human-readable stats."""
        nodes = self.all_nodes()
        n_edges = sum(len(v) for v in self.children.values())
        return (
            f"CategoryGraph: {len(nodes)} nodes, {n_edges} edges, "
            f"{len(self.roots)} roots, {len(self.leaves)} leaves, "
            f"max_depth={self.max_depth()}, is_tree={self.is_tree()}"
        )

    # ── Item resolution ──────────────────────────────────────────────

    def items_at_depth(
        self,
        depth: int,
        item_labels: np.ndarray,
    ) -> np.ndarray:
        """Resolve each item's label to the requested depth.

        For items whose finest label is deeper than *depth*, walk parents
        until a node at the target depth is found.  For items already at
        or above *depth*, use their label as-is.

        Parameters
        ----------
        depth : int
            Target depth (0 = roots).
        item_labels : (n,) str array
            Per-item labels (typically the finest/leaf labels).

        Returns
        -------
        resolved : (n,) str array
        """
        # Pre-compute depth for all nodes
        node_depths: dict[str, int] = {r: 0 for r in self.roots}
        queue = deque(self.roots)
        while queue:
            current = queue.popleft()
            d = node_depths[current]
            for child, _ in self.children.get(current, []):
                if child not in node_depths:
                    node_depths[child] = d + 1
                    queue.append(child)

        # Cache: label → resolved label at depth
        cache: dict[str, str] = {}

        def _resolve(label: str) -> str:
            if label in cache:
                return cache[label]
            d = node_depths.get(label, -1)
            if d == -1:
                # Unknown node — return as-is
                cache[label] = label
                return label
            if d <= depth:
                cache[label] = label
                return label
            # Walk parents until we reach target depth
            frontier = {label}
            for _ in range(d - depth):
                next_frontier: set[str] = set()
                for n in frontier:
                    for parent, _ in self.parents.get(n, []):
                        next_frontier.add(parent)
                if not next_frontier:
                    break
                frontier = next_frontier
            # Pick the first parent at target depth (deterministic sort)
            at_depth = sorted(
                n for n in frontier if node_depths.get(n, -1) == depth
            )
            result = at_depth[0] if at_depth else label
            cache[label] = result
            return result

        item_labels = np.asarray(item_labels, dtype=str)
        return np.array([_resolve(lbl) for lbl in item_labels], dtype=str)

    # ── Constructors ─────────────────────────────────────────────────

    @classmethod
    def from_edges(
        cls,
        edges: list[tuple[str, str, float]],
    ) -> CategoryGraph:
        """Build from explicit edge list [(parent, child, weight)].

        If tuples have 2 elements, weight defaults to 1.0.
        """
        children: dict[str, list[tuple[str, float]]] = {}
        parents: dict[str, list[tuple[str, float]]] = {}
        all_parents: set[str] = set()
        all_children: set[str] = set()

        for edge in edges:
            if len(edge) == 2:
                p, c = edge  # type: ignore[misc]
                w = 1.0
            else:
                p, c, w = edge
            children.setdefault(p, []).append((c, w))
            parents.setdefault(c, []).append((p, w))
            all_parents.add(p)
            all_children.add(c)

        all_nodes = all_parents | all_children
        roots = sorted(all_nodes - all_children)
        leaves = sorted(all_nodes - all_parents)

        return cls(children=children, parents=parents, roots=roots, leaves=leaves)

    @classmethod
    def from_levels(
        cls,
        level_columns: list[np.ndarray | list],
    ) -> CategoryGraph:
        """Build DAG from co-occurring level arrays [L0, L1, L2, ...].

        L0 is the coarsest (root-adjacent) level, L_last is finest.
        Edges are created between adjacent levels based on co-occurrence.
        Edge weights reflect the fraction of items in the parent that
        belong to each child.
        """
        n_levels = len(level_columns)
        if n_levels < 1:
            return cls()

        cols = [np.asarray(c, dtype=str) for c in level_columns]
        n_items = len(cols[0])
        for c in cols:
            if len(c) != n_items:
                raise ValueError("All level columns must have the same length")

        edges: list[tuple[str, str, float]] = []
        # Synthetic root connecting to L0
        root_label = "_root_"
        l0_unique = np.unique(cols[0])
        for lbl in l0_unique:
            count = int(np.sum(cols[0] == lbl))
            edges.append((root_label, lbl, count / n_items))

        # Adjacent-level edges
        for i in range(n_levels - 1):
            parent_col = cols[i]
            child_col = cols[i + 1]
            # Count co-occurrences
            pair_counts: dict[tuple[str, str], int] = {}
            parent_counts: dict[str, int] = {}
            for p_val, c_val in zip(parent_col, child_col):
                pair_counts[(p_val, c_val)] = pair_counts.get((p_val, c_val), 0) + 1
                parent_counts[p_val] = parent_counts.get(p_val, 0) + 1
            for (p_val, c_val), count in pair_counts.items():
                w = count / parent_counts[p_val]
                edges.append((p_val, c_val, w))

        return cls.from_edges(edges)

    @classmethod
    def from_single_level(
        cls,
        labels: np.ndarray | list,
    ) -> CategoryGraph:
        """Depth-1 graph: ``_root_`` → unique labels.

        This is the degenerate case for flat categorical columns.
        """
        labels_arr = np.asarray(labels, dtype=str)
        unique = np.unique(labels_arr)
        n = len(labels_arr)
        edges: list[tuple[str, str, float]] = []
        for lbl in unique:
            count = int(np.sum(labels_arr == lbl))
            edges.append(("_root_", lbl, count / n))
        return cls.from_edges(edges)

    # ── Serialization ────────────────────────────────────────────────

    def to_dict(self) -> dict:
        """JSON-serializable dict (edge list format)."""
        edges = []
        for parent, child_list in self.children.items():
            for child, weight in child_list:
                edges.append({"parent": parent, "child": child, "weight": weight})
        return {"edges": edges}

    @classmethod
    def from_dict(cls, data: dict) -> CategoryGraph:
        """Reconstruct from ``to_dict()`` output."""
        edges = [
            (e["parent"], e["child"], e.get("weight", 1.0))
            for e in data["edges"]
        ]
        return cls.from_edges(edges)

    def to_json(self) -> str:
        """Serialize to JSON string."""
        return json.dumps(self.to_dict())

    @classmethod
    def from_json(cls, s: str) -> CategoryGraph:
        """Reconstruct from JSON string."""
        return cls.from_dict(json.loads(s))


# ── Label coarsening ─────────────────────────────────────────────────────


def coarsen(
    values: list | np.ndarray,
    strategy: Union[str, Callable] = "first_term",
) -> np.ndarray:
    """General label extraction / coarsening.

    Parameters
    ----------
    values : list or array
        Raw column values.  Elements may be strings, lists of strings,
        None, or NaN floats.
    strategy : str or callable
        ``"first_term"`` — split on comma, take first token, lowercase.
        ``"raw"`` — convert to str as-is.
        ``"prefix_N"`` (e.g. ``"prefix_3"``) — first N characters.
        A callable ``f(str) -> str`` — applied after unwrapping.

    Returns
    -------
    labels : (n,) str array
    """
    # Resolve strategy
    if callable(strategy):
        transform = strategy
    elif strategy == "first_term":
        def transform(s: str) -> str:
            term = s.split(",")[0].strip().lower()
            return term if term else "_unknown_"
    elif strategy == "raw":
        def transform(s: str) -> str:
            return s
    elif isinstance(strategy, str) and strategy.startswith("prefix_"):
        try:
            n_chars = int(strategy.split("_", 1)[1])
        except (ValueError, IndexError):
            raise ValueError(f"Invalid prefix strategy: {strategy!r}")

        def transform(s: str) -> str:
            prefix = s[:n_chars].strip().lower()
            return prefix if prefix else "_unknown_"
    else:
        raise ValueError(f"Unknown coarsen strategy: {strategy!r}")

    out: list[str] = []
    for v in values:
        if v is None or (isinstance(v, float) and np.isnan(v)):
            out.append("_unknown_")
            continue

        # Unwrap list/array → first element
        if isinstance(v, (list, np.ndarray)):
            v = v[0] if len(v) > 0 else ""

        s = str(v)
        result = transform(s)
        out.append(result if result else "_unknown_")

    return np.array(out, dtype=str)


# ── Multi-level Fisher weighting ─────────────────────────────────────────


def multi_level_fisher_weights(
    embeddings: np.ndarray,
    graph: CategoryGraph,
    item_labels: np.ndarray,
    min_count: int = 50,
) -> np.ndarray:
    """Compute Fisher weights averaged across multiple DAG depths.

    At each depth of the graph, item labels are resolved via
    ``graph.items_at_depth()``, then ``compute_fisher_weights`` is called.
    The per-depth weight vectors are combined via weighted average
    (weighted by number of surviving classes at that depth) and
    re-normalized.

    Falls back to single-level Fisher if the graph has depth <= 1.

    Parameters
    ----------
    embeddings : (n, d) float32
    graph : CategoryGraph
    item_labels : (n,) str array
        Per-item leaf labels (finest level).
    min_count : int
        Passed through to ``compute_fisher_weights``.

    Returns
    -------
    weights : (d,) float32
        L2-normalized combined Fisher weights.
    """
    from dyf.fisher import compute_fisher_weights

    embeddings = np.asarray(embeddings, dtype=np.float32)
    item_labels = np.asarray(item_labels, dtype=str)
    _, d = embeddings.shape

    md = graph.max_depth()
    if md <= 1:
        # Single level — just use the item labels directly
        return compute_fisher_weights(embeddings, item_labels, min_count=min_count)

    # Collect per-depth weights
    depth_weights: list[tuple[np.ndarray, int]] = []
    for depth in range(1, md + 1):
        resolved = graph.items_at_depth(depth, item_labels)
        _, counts = np.unique(resolved, return_counts=True)
        n_classes = int(np.sum(counts >= min_count))
        if n_classes < 2:
            continue
        w = compute_fisher_weights(embeddings, resolved, min_count=min_count)
        depth_weights.append((w, n_classes))

    if not depth_weights:
        # No depth produced enough classes — uniform
        w = np.ones(d, dtype=np.float32)
        w /= np.linalg.norm(w)
        return w

    # Weighted average by number of surviving classes
    total_classes = sum(nc for _, nc in depth_weights)
    combined = np.zeros(d, dtype=np.float64)
    for w, nc in depth_weights:
        combined += w.astype(np.float64) * (nc / total_classes)

    combined = combined.astype(np.float32)
    norm = np.linalg.norm(combined)
    if norm > 0:
        combined /= norm
    return combined


# ── Axis diagnostics ─────────────────────────────────────────────────────


@dataclass
class AxisDiagnostic:
    """Diagnostic result for a single categorical axis.

    Attributes
    ----------
    name : str
        Column / axis name.
    knn_purity : float
        Observed k-NN purity (fraction of each point's neighbors sharing
        its label, averaged over all points).
    random_baseline : float
        Herfindahl index — expected purity under random label assignment
        (sum of squared class proportions).
    lift : float
        ``knn_purity / random_baseline``.  Low lift means the embedding
        barely improves on chance for this axis.
    n_classes : int
        Number of unique label values.
    """

    name: str
    knn_purity: float
    random_baseline: float
    lift: float
    n_classes: int

    def __repr__(self) -> str:
        return (
            f"AxisDiagnostic({self.name!r}, purity={self.knn_purity:.3f}, "
            f"baseline={self.random_baseline:.3f}, lift={self.lift:.1f}x, "
            f"classes={self.n_classes})"
        )


def diagnose_axes(
    embeddings: np.ndarray,
    label_columns: dict[str, np.ndarray | list],
    k: int = 15,
    sample_n: int = 5000,
    seed: int = 42,
) -> list[AxisDiagnostic]:
    """Detect which categorical axes are under-served by the embedding.

    For each axis, computes k-NN purity (do nearby points share the same
    label?) and compares to the Herfindahl random baseline.  Low lift =
    the embedding doesn't actively separate this axis = candidate for
    promotion to an explicit text field before re-embedding.

    Parameters
    ----------
    embeddings : (n, d) float32
        Embedding matrix.
    label_columns : dict[str, array-like]
        Mapping of axis name → per-item labels (length n).
    k : int
        Number of neighbors for k-NN purity (default 15).
    sample_n : int
        Subsample to this many points for speed (0 = no subsampling).
        Default 5000 — brute-force k-NN on 5k points takes ~1s.
    seed : int
        Random seed for subsampling.

    Returns
    -------
    diagnostics : list[AxisDiagnostic]
        Sorted by lift ascending (worst-performing axis first).

    Examples
    --------
    >>> diags = diagnose_axes(embeddings, {"gmdn": gmdn_labels, "polarity": pol_labels})
    >>> for d in diags:
    ...     print(f"{d.name}: lift={d.lift:.1f}x  purity={d.knn_purity:.3f}")
    polarity: lift=2.2x  purity=0.978
    gmdn: lift=23.0x  purity=0.929
    """
    rng = np.random.default_rng(seed)
    embeddings = np.asarray(embeddings, dtype=np.float32)
    n = embeddings.shape[0]

    # Coerce all label columns to numpy arrays
    label_arrays: dict[str, np.ndarray] = {
        name: np.asarray(labels, dtype=str)
        for name, labels in label_columns.items()
    }

    # Subsample for speed
    if sample_n > 0 and n > sample_n:
        idx = rng.choice(n, sample_n, replace=False)
        embeddings = embeddings[idx]
        label_arrays = {name: labels[idx] for name, labels in label_arrays.items()}
        n = sample_n

    # Brute-force k-NN via squared L2 distances
    # ||a-b||^2 = ||a||^2 + ||b||^2 - 2*a.b
    norms_sq = (embeddings**2).sum(axis=1)
    dists = norms_sq[:, None] + norms_sq[None, :] - 2.0 * (embeddings @ embeddings.T)
    np.fill_diagonal(dists, np.inf)

    # Indices of k nearest neighbors per point
    knn_idx = np.argpartition(dists, k, axis=1)[:, :k]

    results: list[AxisDiagnostic] = []
    for name, labels in label_arrays.items():
        # k-NN purity: for each point, fraction of neighbors sharing its label
        neighbor_labels = labels[knn_idx]  # (n, k)
        same = (neighbor_labels == labels.reshape(-1, 1)).sum(axis=1)  # (n,)
        knn_purity = float(same.mean() / k)

        # Herfindahl index (random baseline)
        _, counts = np.unique(labels, return_counts=True)
        props = counts / counts.sum()
        herfindahl = float((props**2).sum())

        lift = knn_purity / herfindahl if herfindahl > 0 else float("inf")

        results.append(
            AxisDiagnostic(
                name=name,
                knn_purity=knn_purity,
                random_baseline=herfindahl,
                lift=lift,
                n_classes=len(counts),
            )
        )

    results.sort(key=lambda x: x.lift)
    return results


# ── Metadata helpers ─────────────────────────────────────────────────────


def store_category_graph(
    graph: CategoryGraph,
    name: str,
    field_mapping: Optional[dict[str, str]] = None,
    existing_metadata: Optional[dict[str, str]] = None,
) -> dict[str, str]:
    """Serialize a CategoryGraph for .dyf metadata.

    Parameters
    ----------
    graph : CategoryGraph
    name : str
        Identifier for this graph (e.g. ``"gmdn"``).
    field_mapping : dict, optional
        Maps depth → stored field name (e.g. ``{0: "gmdn_family", 1: "gmdn_term"}``).
    existing_metadata : dict, optional
        When provided and contains ``category_graphs``, the new graph is
        merged into the existing dict rather than replacing it.

    Returns
    -------
    metadata : dict[str, str]
        Suitable for passing to ``write_lazy_index`` or ``rewrite_lazy_index``
        as metadata entries.
    """
    # Start from existing graphs if available
    payload: dict = {}
    if existing_metadata:
        raw = existing_metadata.get("category_graphs")
        if raw:
            payload = json.loads(raw)

    payload[name] = {
        "graph": graph.to_dict(),
        "field_mapping": field_mapping or {},
    }
    return {"category_graphs": json.dumps(payload)}


def load_category_graphs(
    metadata: dict[str, str],
) -> dict[str, tuple[CategoryGraph, dict[str, str]]]:
    """Deserialize CategoryGraphs from .dyf metadata.

    Parameters
    ----------
    metadata : dict[str, str]
        Metadata dict from a .dyf file.

    Returns
    -------
    graphs : dict mapping name → (CategoryGraph, field_mapping)
    """
    raw = metadata.get("category_graphs")
    if not raw:
        return {}

    data = json.loads(raw)
    result = {}
    for name, entry in data.items():
        graph = CategoryGraph.from_dict(entry["graph"])
        field_mapping = entry.get("field_mapping", {})
        result[name] = (graph, field_mapping)
    return result


# ── Two-pass embedding pipeline ──────────────────────────────────────────


def discover_categorical_columns(
    df: Any,
    text_col: str = "text",
    max_cardinality: int = 500,
    min_cardinality: int = 2,
) -> dict[str, np.ndarray]:
    """Auto-detect categorical columns from a polars DataFrame.

    String columns with bounded cardinality are treated as categorical
    axes. List[str] columns are coarsened via ``coarsen(strategy='first_term')``.
    High-cardinality string columns (likely free text) and the embedding
    column are skipped.

    Parameters
    ----------
    df : polars.DataFrame
        Input dataframe.
    text_col : str
        Name of the text column used for embedding (excluded from axes).
    max_cardinality : int
        Columns with more unique values than this are skipped.
    min_cardinality : int
        Columns with fewer unique values than this are skipped.

    Returns
    -------
    label_columns : dict[str, np.ndarray]
        Mapping of column name → per-row string labels.
    """
    import polars as pl

    skip_cols = {text_col, "embedding"}
    result: dict[str, np.ndarray] = {}

    for col_name in df.columns:
        if col_name in skip_cols:
            continue

        dtype = df[col_name].dtype

        # List[str] columns → coarsen with first_term
        if dtype == pl.List(pl.Utf8) or dtype == pl.List(pl.String):
            raw_vals = df[col_name].to_list()
            labels = coarsen(raw_vals, strategy="first_term")
            n_unique = len(np.unique(labels))
            if min_cardinality <= n_unique <= max_cardinality:
                result[col_name] = labels
            continue

        # String columns
        if dtype in (pl.Utf8, pl.String):
            n_unique = df[col_name].n_unique()
            if min_cardinality <= n_unique <= max_cardinality:
                # Check if it's likely free text (high cardinality or long strings)
                avg_len = df[col_name].str.len_chars().mean()
                if avg_len is not None and avg_len > 100:
                    continue  # skip free-text columns
                result[col_name] = np.array(
                    df[col_name].fill_null("_unknown_").to_list(), dtype=str
                )
            continue

        # Categorical columns (polars Categorical type)
        if dtype == pl.Categorical:
            vals = df[col_name].cast(pl.String).fill_null("_unknown_").to_list()
            n_unique = len(set(vals))
            if min_cardinality <= n_unique <= max_cardinality:
                result[col_name] = np.array(vals, dtype=str)

    return result


def embed_with_diagnostics(
    embeddings: np.ndarray,
    text_col: list[str],
    label_columns: dict[str, np.ndarray | list],
    embed_fn: Callable[[list[str]], np.ndarray],
    lift_threshold: float = 3.0,
    prefix: str = "",
) -> tuple[np.ndarray, list[AxisDiagnostic], list[AxisDiagnostic], list[str]]:
    """Two-pass embedding with axis diagnostics.

    1. Run ``diagnose_axes`` on the provided embeddings.
    2. If any axis has lift < ``lift_threshold``, rebuild text with
       explicit labeled fields for under-served axes.
    3. Re-embed with the structured text via ``embed_fn``.
    4. Re-diagnose to confirm improvement.

    Parameters
    ----------
    embeddings : (n, d) float32
        Baseline embeddings (already computed by the caller).
    text_col : list[str]
        Original text strings used for embedding (one per row).
    label_columns : dict[str, array-like]
        Mapping of axis name → per-item labels (length n).
    embed_fn : callable
        ``fn(texts: list[str]) -> np.ndarray`` — re-embeds a list of texts.
    lift_threshold : float
        Axes with lift below this are promoted to explicit text fields.
    prefix : str
        Prefix prepended to each structured text (e.g. ``"search_document: "``).

    Returns
    -------
    embeddings : (n, d) float32
        Final embeddings (original if no axes promoted, re-embedded otherwise).
    before_diags : list[AxisDiagnostic]
        Diagnostics from the first pass.
    after_diags : list[AxisDiagnostic]
        Diagnostics from the second pass (same as before if no re-embedding).
    texts : list[str]
        Final text strings (original or structured).
    """
    # Coerce label columns to numpy
    label_arrays: dict[str, np.ndarray | list] = {
        name: np.asarray(labels, dtype=str)
        for name, labels in label_columns.items()
    }

    # First pass: diagnose
    before_diags = diagnose_axes(embeddings, label_arrays)

    # Identify under-served axes
    promote_axes = [d.name for d in before_diags if d.lift < lift_threshold]

    if not promote_axes:
        # Nothing to promote — return originals
        return embeddings, before_diags, before_diags, list(text_col)

    # Build structured text with promoted columns prepended
    n = len(text_col)
    structured_texts: list[str] = []
    skip_values = {"_unknown_", "other", "unspecified", ""}

    for i in range(n):
        parts: list[str] = []
        for axis_name in promote_axes:
            val = str(label_arrays[axis_name][i])
            if val.lower() not in skip_values:
                parts.append(f"{axis_name}: {val}")
        parts.append(text_col[i])
        structured_texts.append(prefix + ". ".join(parts))

    # Re-embed
    new_embeddings = embed_fn(structured_texts)

    # Second pass: diagnose the new embeddings
    after_diags = diagnose_axes(new_embeddings, label_arrays)

    return new_embeddings, before_diags, after_diags, structured_texts


def diagnostics_to_metadata(
    before: list[AxisDiagnostic],
    after: list[AxisDiagnostic],
) -> dict[str, str]:
    """Serialize axis diagnostics for .dyf metadata.

    Parameters
    ----------
    before : list[AxisDiagnostic]
        Diagnostics from the first pass.
    after : list[AxisDiagnostic]
        Diagnostics from the second pass.

    Returns
    -------
    metadata : dict[str, str]
        Keys ``axis_diagnostics_before`` and ``axis_diagnostics_after``,
        each a JSON string.
    """
    def _serialize(diags: list[AxisDiagnostic]) -> str:
        return json.dumps([
            {
                "name": d.name,
                "knn_purity": round(d.knn_purity, 4),
                "random_baseline": round(d.random_baseline, 4),
                "lift": round(d.lift, 2),
                "n_classes": d.n_classes,
            }
            for d in diags
        ])

    return {
        "axis_diagnostics_before": _serialize(before),
        "axis_diagnostics_after": _serialize(after),
    }
