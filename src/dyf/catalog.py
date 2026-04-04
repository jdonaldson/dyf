"""CatalogSpace: Multi-catalog matching with cross-catalog coherence.

Generalizes ANCHOR's single-ontology matching to N catalogs with
joint coherence scoring. Uses CategoryGraph for hierarchy navigation
instead of code-prefix logic.

Key concepts:
- ontological_z: Does the query belong in this catalog at all?
- node_z: How distinctive is each matched node within its catalog?
- path_alignment: Does the matched node's path to root reflect semantics?
- coherence_score: Do matches across catalogs agree via cross-mappings?
"""

from __future__ import annotations

import json
import math
from collections import Counter
from dataclasses import dataclass, field
from itertools import product as itertools_product

import numpy as np

from .categorical import CategoryGraph
from .splits import tokenize

# ── Dataclasses ──────────────────────────────────────────────────────────


@dataclass
class CatalogConfig:
    """Configuration for a single catalog in CatalogSpace.

    Parameters
    ----------
    name : str
        Catalog identifier (e.g., "unspsc", "broadjump", "curvo").
    graph : CategoryGraph
        Hierarchy structure. Use ``CategoryGraph.from_single_level()``
        for flat catalogs.
    embeddings : np.ndarray
        (n_nodes, d) pre-computed embeddings for catalog nodes.
    node_ids : np.ndarray
        (n_nodes,) string identifiers for each node.
    node_names : np.ndarray
        (n_nodes,) human-readable names for each node.
    """

    name: str
    graph: CategoryGraph
    embeddings: np.ndarray
    node_ids: np.ndarray
    node_names: np.ndarray

    def __post_init__(self):
        self.embeddings = np.asarray(self.embeddings, dtype=np.float32)
        self.node_ids = np.asarray(self.node_ids, dtype=str)
        self.node_names = np.asarray(self.node_names, dtype=str)
        n = len(self.node_ids)
        if self.embeddings.shape[0] != n:
            raise ValueError(
                f"embeddings rows ({self.embeddings.shape[0]}) != "
                f"node_ids length ({n})"
            )
        if len(self.node_names) != n:
            raise ValueError(
                f"node_names length ({len(self.node_names)}) != "
                f"node_ids length ({n})"
            )


@dataclass
class CrossMapping:
    """Mapping between two catalogs for coherence scoring.

    Parameters
    ----------
    source_catalog : str
        Name of the source catalog.
    target_catalog : str
        Name of the target catalog.
    source_ids : np.ndarray
        (n_mappings,) source node IDs.
    target_ids : np.ndarray
        (n_mappings,) target node IDs.
    weights : np.ndarray
        (n_mappings,) confidence weights 0-1.
    """

    source_catalog: str
    target_catalog: str
    source_ids: np.ndarray
    target_ids: np.ndarray
    weights: np.ndarray

    def __post_init__(self):
        self.source_ids = np.asarray(self.source_ids, dtype=str)
        self.target_ids = np.asarray(self.target_ids, dtype=str)
        self.weights = np.asarray(self.weights, dtype=np.float32)
        n = len(self.source_ids)
        if len(self.target_ids) != n or len(self.weights) != n:
            raise ValueError("source_ids, target_ids, weights must have same length")


@dataclass
class CatalogMatch:
    """Match result for a single catalog.

    Attributes
    ----------
    catalog_name : str
        Which catalog this match is from.
    node_id : str
        Matched node identifier.
    node_name : str
        Human-readable name.
    similarity : float
        Cosine similarity to query (0-1).
    node_z : float
        Within-catalog distinctiveness z-score.
    ontological_z : float
        Does the query belong in this catalog?
    depth : int
        Depth of matched node in hierarchy.
    fit : str
        VERY_HIGH / HIGH / MODERATE / LOW / OUT_OF_DOMAIN.
    entropy : float
        Normalized entropy of match distribution.
    path_alignment : float
        Semantic alignment along path to root (0-1).
    alternatives : list[tuple[str, str, float]]
        [(id, name, similarity), ...] diverse alternatives.
    gap_detected : bool
        Whether a catalog gap was detected.
    gap_score : float
        Gap signal strength (0-1).
    """

    catalog_name: str
    node_id: str
    node_name: str
    similarity: float
    node_z: float
    ontological_z: float
    depth: int
    fit: str
    entropy: float
    path_alignment: float
    alternatives: list[tuple[str, str, float]] = field(default_factory=list)
    gap_detected: bool = False
    gap_score: float = 0.0


@dataclass
class JointMatchResult:
    """Result of matching a query across multiple catalogs.

    Attributes
    ----------
    query : str
        Query identifier (caller-assigned).
    matches : dict[str, CatalogMatch]
        catalog_name -> best match per catalog.
    coherence_score : float
        Cross-catalog consistency 0-1.
    coherence_detail : dict[str, float]
        Per-pair coherence scores.
    reranked : bool
        Whether coherence changed the result.
    reason : str
        Explanation of match decisions.
    """

    query: str
    matches: dict[str, CatalogMatch] = field(default_factory=dict)
    coherence_score: float = 1.0
    coherence_detail: dict[str, float] = field(default_factory=dict)
    reranked: bool = False
    reason: str = ""


# ── Internal fitted state ────────────────────────────────────────────────


@dataclass
class _FittedCatalog:
    """Pre-computed stats for a single catalog after fit()."""

    config: CatalogConfig
    # Ontology-wide statistics
    centroid: np.ndarray          # (d,) mean embedding
    centroid_norm: np.ndarray     # (d,) L2-normalized centroid
    tax_mean: float               # mean specificity
    tax_std: float                # std specificity
    node_z_scores: np.ndarray     # (n,) per-node z-scores
    path_alignments: np.ndarray   # (n,) per-node path alignment

    # Depth-indexed structures
    node_depths: np.ndarray       # (n,) depth of each node
    depth_masks: dict[int, np.ndarray]   # depth -> boolean mask
    id_to_idx: dict[str, int]     # node_id -> index

    # Term disambiguation
    branch_terms: dict[str, set[str]] = field(default_factory=dict)  # node_id → discriminating terms
    term_boost: float = 0.04      # additive boost alpha


# ── Utility functions ────────────────────────────────────────────────────


def _node_z_to_fit(z: float) -> str:
    """Convert node_z to fit level string."""
    if z > 1.0:
        return "VERY_HIGH"
    elif z > 0.0:
        return "HIGH"
    elif z > -1.0:
        return "MODERATE"
    elif z > -2.0:
        return "LOW"
    else:
        return "OUT_OF_DOMAIN"


def _compute_entropy(
    similarities: np.ndarray,
    temperature: float = 0.05,
    top_k: int = 10,
) -> float:
    """Normalized entropy of similarity distribution among top candidates.

    Uses softmax to convert similarities to probabilities, focusing on
    top-K candidates to capture the actual decision space.

    Returns 0 for one dominant match, 1 for uniform spread.
    """
    if len(similarities) < 2:
        return 0.0

    k = min(top_k, len(similarities))
    top_idx = np.argsort(similarities)[::-1][:k]
    top_sims = similarities[top_idx]

    shifted = top_sims - top_sims.max()
    exp_sims = np.exp(shifted / temperature)
    probs = exp_sims / exp_sims.sum()

    entropy = -np.sum(probs * np.log(probs + 1e-10))
    max_entropy = np.log(k)
    return float(entropy / max_entropy) if max_entropy > 0 else 0.0


def _compute_path_alignment(
    node_idx: int,
    fc: _FittedCatalog,
) -> float:
    """Path alignment for a single node.

    Measures cosine similarity between (parent - centroid) and
    (child - centroid) vectors, averaged along the path to root.
    """
    graph = fc.config.graph
    node_id = fc.config.node_ids[node_idx]
    embeddings = fc.config.embeddings
    centroid = fc.centroid

    alignments = []
    current_id = node_id
    visited = {current_id}

    while True:
        parents = graph.get_parents(current_id)
        if not parents:
            break
        parent_id = parents[0]
        if parent_id in visited:
            break
        visited.add(parent_id)

        if parent_id not in fc.id_to_idx:
            break

        parent_idx = fc.id_to_idx[parent_id]
        child_idx = fc.id_to_idx.get(current_id)
        if child_idx is None:
            break

        parent_vec = embeddings[parent_idx] - centroid
        child_vec = embeddings[child_idx] - centroid

        parent_mag = np.linalg.norm(parent_vec)
        child_mag = np.linalg.norm(child_vec)

        if parent_mag > 0 and child_mag > 0:
            cosine = float(np.dot(parent_vec, child_vec) / (parent_mag * child_mag))
            alignments.append(cosine)

        current_id = parent_id

    return float(np.mean(alignments)) if alignments else 1.0


# ── Term disambiguation ──────────────────────────────────────────────────


def _compute_branch_terms(
    config: CatalogConfig,
    graph: CategoryGraph,
    top_k: int = 15,
) -> dict[str, set[str]]:
    """Compute TF-IDF discriminating terms per CategoryGraph branch.

    For each internal node, treats its children as "document classes",
    tokenizes node_names of all leaf descendants per child, and finds
    top-k discriminating unigrams via TF-IDF. Results are flattened
    into {node_id: set(discriminating terms in subtree)}.
    """
    node_id_to_name: dict[str, str] = {
        str(nid): str(name)
        for nid, name in zip(config.node_ids, config.node_names)
    }

    # For each internal node, compute discriminating terms per child
    child_terms: dict[str, set[str]] = {}

    all_nodes = graph.all_nodes()
    for node_id in all_nodes:
        children = graph.get_children(node_id)
        if len(children) < 2:
            continue

        # Collect tokenized names for each child's leaf descendants
        child_docs: dict[str, list[str]] = {}
        for cid in children:
            # Get all descendants (including the child itself)
            desc = graph.get_descendants(cid)
            desc.add(cid)
            # Collect tokens from all node names in this subtree
            tokens: list[str] = []
            for d in desc:
                name = node_id_to_name.get(d, "")
                if name:
                    tokens.extend(tokenize(name))
            if tokens:
                child_docs[cid] = tokens

        if len(child_docs) < 2:
            continue

        n_children = len(child_docs)

        # Term frequency per child
        child_tf: dict[str, Counter] = {}
        for cid, tokens in child_docs.items():
            child_tf[cid] = Counter(tokens)

        # Document frequency across children
        word_df: Counter = Counter()
        for tf in child_tf.values():
            for word in tf:
                word_df[word] += 1

        # IDF: only keep terms that discriminate (not in ALL children)
        idf = {
            w: math.log(n_children / (1 + df))
            for w, df in word_df.items()
            if df < n_children
        }

        # TF-IDF per child, store top-k terms
        for cid, tf in child_tf.items():
            total = sum(tf.values())
            if total == 0:
                continue
            scores = []
            for word, count in tf.items():
                if word in idf:
                    scores.append((word, (count / total) * idf[word]))
            scores.sort(key=lambda x: -x[1])
            terms = {w for w, _ in scores[:top_k]}
            if terms:
                # Accumulate: a node can appear as child of multiple internal nodes
                if cid in child_terms:
                    child_terms[cid] |= terms
                else:
                    child_terms[cid] = terms

    # Propagate terms upward: each node's terms include all descendant terms
    result: dict[str, set[str]] = {}
    for node_id in all_nodes:
        desc = graph.get_descendants(node_id)
        accumulated = set(child_terms.get(node_id, set()))
        for d in desc:
            accumulated |= child_terms.get(d, set())
        if accumulated:
            result[node_id] = accumulated

    return result


# ── CatalogSpace ─────────────────────────────────────────────────────────


class CatalogSpace:
    """Multi-catalog matching engine with cross-catalog coherence.

    Usage::

        space = CatalogSpace()
        space.add_catalog(unspsc_config)
        space.add_catalog(broadjump_config)
        space.add_mapping(unspsc_to_broadjump)
        space.fit()

        # Single catalog
        match = space.match_single("unspsc", query_emb)

        # Joint across all catalogs
        results = space.match(query_embs, top_k=5)
    """

    def __init__(self) -> None:
        self._configs: dict[str, CatalogConfig] = {}
        self._mappings: list[CrossMapping] = []
        self._fitted: dict[str, _FittedCatalog] = {}
        self._is_fitted: bool = False

    # ── Builder methods ───────────────────────────────────────────────

    def add_catalog(self, config: CatalogConfig) -> CatalogSpace:
        """Add a catalog configuration. Returns self for chaining."""
        if self._is_fitted:
            raise RuntimeError("Cannot add catalogs after fit()")
        self._configs[config.name] = config
        return self

    def add_mapping(self, mapping: CrossMapping) -> CatalogSpace:
        """Add a cross-catalog mapping. Returns self for chaining."""
        if self._is_fitted:
            raise RuntimeError("Cannot add mappings after fit()")
        self._mappings.append(mapping)
        return self

    def fit(self) -> CatalogSpace:
        """Pre-compute statistics for all catalogs. Returns self."""
        if not self._configs:
            raise ValueError("No catalogs added")

        # Validate embedding dimensions match
        dims = {c.name: c.embeddings.shape[1] for c in self._configs.values()}
        unique_dims = set(dims.values())
        if len(unique_dims) > 1:
            raise ValueError(f"Embedding dimensions must match across catalogs: {dims}")

        for name, config in self._configs.items():
            self._fitted[name] = self._fit_single(config)

        # Build mapping lookup for coherence scoring
        self._mapping_lookup: dict[tuple[str, str], CrossMapping] = {}
        for m in self._mappings:
            self._mapping_lookup[(m.source_catalog, m.target_catalog)] = m
            # Also store reverse for bidirectional lookup
            reverse = CrossMapping(
                source_catalog=m.target_catalog,
                target_catalog=m.source_catalog,
                source_ids=m.target_ids,
                target_ids=m.source_ids,
                weights=m.weights,
            )
            self._mapping_lookup[(m.target_catalog, m.source_catalog)] = reverse

        self._is_fitted = True
        return self

    def _fit_single(self, config: CatalogConfig) -> _FittedCatalog:
        """Compute per-catalog statistics."""
        embs = config.embeddings
        n = embs.shape[0]
        graph = config.graph

        # ID -> index lookup
        id_to_idx = {str(nid): i for i, nid in enumerate(config.node_ids)}

        # Centroid
        centroid = embs.mean(axis=0)
        centroid_norm_val = np.linalg.norm(centroid)
        centroid_norm = centroid / centroid_norm_val if centroid_norm_val > 0 else centroid

        # Centroid similarities
        centroid_sims = embs @ centroid_norm

        # Best similarity to other nodes (for specificity/z-scores)
        # Use dot product against all nodes, mask self
        sim_matrix = embs @ embs.T
        np.fill_diagonal(sim_matrix, -np.inf)
        best_other = sim_matrix.max(axis=1)

        # Specificity = best_other - centroid_sim
        specificity = best_other - centroid_sims
        tax_mean = float(specificity.mean())
        tax_std = float(specificity.std())
        if tax_std == 0:
            tax_std = 1.0  # avoid division by zero

        node_z_scores = (specificity - tax_mean) / tax_std

        # Node depths
        node_depths = np.array(
            [graph.get_depth(str(nid)) for nid in config.node_ids],
            dtype=np.int32,
        )

        # Depth masks
        unique_depths = np.unique(node_depths)
        depth_masks = {int(d): node_depths == d for d in unique_depths}

        # Term disambiguation
        branch_terms = _compute_branch_terms(config, graph)

        # Create partially-filled _FittedCatalog for path alignment calc
        fc = _FittedCatalog(
            config=config,
            centroid=centroid,
            centroid_norm=centroid_norm,
            tax_mean=tax_mean,
            tax_std=tax_std,
            node_z_scores=node_z_scores,
            path_alignments=np.ones(n, dtype=np.float32),  # placeholder
            node_depths=node_depths,
            depth_masks=depth_masks,
            id_to_idx=id_to_idx,
            branch_terms=branch_terms,
        )

        # Path alignment for each node
        path_alignments = np.array(
            [_compute_path_alignment(i, fc) for i in range(n)],
            dtype=np.float32,
        )
        fc.path_alignments = path_alignments

        return fc

    # ── Single-catalog matching ───────────────────────────────────────

    def match_single(
        self,
        catalog_name: str,
        query_emb: np.ndarray,
        top_k: int = 5,
        query_text: str | None = None,
    ) -> CatalogMatch:
        """Match a single query embedding against one catalog.

        Parameters
        ----------
        catalog_name : str
            Which catalog to match against.
        query_emb : np.ndarray
            (d,) query embedding vector.
        top_k : int
            Number of top candidates for entropy/alternatives.
        query_text : str, optional
            Free-text description of the query. When provided, tokenized
            and used for term-affinity boosting in parent selection.

        Returns
        -------
        CatalogMatch
            Best match result with diagnostics.
        """
        if not self._is_fitted:
            raise RuntimeError("Call fit() before matching")
        if catalog_name not in self._fitted:
            raise KeyError(f"Unknown catalog: {catalog_name!r}")

        fc = self._fitted[catalog_name]
        query_emb = np.asarray(query_emb, dtype=np.float32)

        # Tokenize query text for term disambiguation
        query_tokens = set(tokenize(query_text)) if query_text else None

        # Compute ontological z
        ontological_z = self._compute_ontological_z(fc, query_emb)

        # Match with two-stage + within-z level selection
        result = self._match_with_levels(fc, query_emb, top_k, query_tokens=query_tokens)
        return _replace_ontological_z(result, ontological_z)

    def _compute_ontological_z(
        self,
        fc: _FittedCatalog,
        query_emb: np.ndarray,
    ) -> float:
        """Compute ontological z-score for a query against a catalog."""
        all_sims = fc.config.embeddings @ query_emb
        query_centroid_sim = float(np.dot(query_emb, fc.centroid_norm))
        query_best_sim = float(all_sims.max())
        query_specificity = query_best_sim - query_centroid_sim
        return (query_specificity - fc.tax_mean) / fc.tax_std

    def _match_with_levels(
        self,
        fc: _FittedCatalog,
        query_emb: np.ndarray,
        top_k: int = 5,
        query_tokens: set[str] | None = None,
    ) -> CatalogMatch:
        """Two-stage matching with within-z level selection.

        Iterates depths, computes within-z at each depth, selects
        the depth with sharpest discrimination, then optionally
        constrains via parent-level match (two-stage).

        When query_tokens is provided and branch_terms are available,
        an additive term-affinity boost is applied to parent scores.
        """
        config = fc.config
        graph = config.graph
        embs = config.embeddings
        max_d = graph.max_depth()

        # If flat catalog (max_depth <= 1), match all nodes directly
        if max_d <= 1:
            return self._match_flat(fc, query_emb, top_k)

        # Similarities to all nodes
        all_sims = embs @ query_emb

        # Per-depth analysis
        within_z, depth_entropy, depth_best_sim, best_depth = self._analyze_depths(
            all_sims, fc.depth_masks, max_d,
        )

        if best_depth is None:
            return self._match_flat(fc, query_emb, top_k)

        # Two-stage parent constraint
        parent_depth = best_depth - 1
        constrained_result = self._constrain_by_parent(
            all_sims, fc, best_depth, parent_depth, query_tokens,
        )

        # Determine final match
        target_mask = fc.depth_masks.get(best_depth)
        if target_mask is not None and target_mask.sum() > 0:
            target_indices = np.where(target_mask)[0]
            target_sims = all_sims[target_indices]
            unconstrained_best_local = np.argmax(target_sims)
            unconstrained_best_idx = target_indices[unconstrained_best_local]
            unconstrained_best_sim = float(target_sims[unconstrained_best_local])
        else:
            # Fallback to global best
            unconstrained_best_idx = int(np.argmax(all_sims))
            unconstrained_best_sim = float(all_sims[unconstrained_best_idx])

        # Prefer constrained (two-stage) result at deepest level
        deepest = max(fc.depth_masks.keys())
        if best_depth == deepest and constrained_result is not None:
            best_idx, best_sim = constrained_result
        else:
            best_idx = unconstrained_best_idx
            best_sim = unconstrained_best_sim

        # Entropy at the selected depth
        entropy_val = depth_entropy.get(best_depth, 0.0)

        # Gap detection
        gap_detected, gap_score = self._detect_gap(
            depth_entropy, depth_best_sim,
        )

        # Diverse alternatives
        alternatives = self._get_diverse_alternatives(
            fc, query_emb, best_depth, best_idx, top_k
        )

        node_z = float(fc.node_z_scores[best_idx])
        pa = float(fc.path_alignments[best_idx])

        return CatalogMatch(
            catalog_name=config.name,
            node_id=str(config.node_ids[best_idx]),
            node_name=str(config.node_names[best_idx]),
            similarity=round(best_sim, 4),
            node_z=round(node_z, 3),
            ontological_z=0.0,  # filled in by caller
            depth=int(fc.node_depths[best_idx]),
            fit=_node_z_to_fit(node_z),
            entropy=round(entropy_val, 4),
            path_alignment=round(pa, 4),
            alternatives=alternatives,
            gap_detected=gap_detected,
            gap_score=round(gap_score, 4),
        )

    def _analyze_depths(
        self,
        all_sims: np.ndarray,
        depth_masks: dict[int, np.ndarray],
        max_d: int,
    ) -> tuple[dict[int, float], dict[int, float], dict[int, float], int | None]:
        """Compute per-depth statistics and select the best depth.

        Parameters
        ----------
        all_sims : np.ndarray
            Cosine similarities of the query to every catalog node.
        depth_masks : dict[int, np.ndarray]
            Boolean masks keyed by depth level.
        max_d : int
            Maximum depth in the catalog hierarchy.

        Returns
        -------
        within_z : dict[int, float]
            Within-depth z-score per depth.
        depth_entropy : dict[int, float]
            Entropy of similarity distribution per depth.
        depth_best_sim : dict[int, float]
            Best similarity per depth.
        best_depth : int | None
            Depth with sharpest discrimination, or ``None`` if no valid depth.
        """
        within_z: dict[int, float] = {}
        depth_entropy: dict[int, float] = {}
        depth_best_sim: dict[int, float] = {}

        for depth in range(1, max_d + 1):
            mask = depth_masks.get(depth)
            if mask is None or mask.sum() < 2:
                continue

            depth_sims = all_sims[mask]
            std = float(depth_sims.std())
            if std > 0:
                within_z[depth] = float((depth_sims.max() - depth_sims.mean()) / std)
            else:
                within_z[depth] = 0.0

            depth_entropy[depth] = _compute_entropy(depth_sims)
            depth_best_sim[depth] = float(depth_sims.max())

        if not within_z:
            return within_z, depth_entropy, depth_best_sim, None

        # Highest within-z; tiebreaker: prefer deeper (principle of least power)
        best_depth = max(within_z, key=lambda d: (within_z[d], d))
        return within_z, depth_entropy, depth_best_sim, best_depth

    def _constrain_by_parent(
        self,
        all_sims: np.ndarray,
        fc: _FittedCatalog,
        best_depth: int,
        parent_depth: int,
        query_tokens: set[str] | None,
    ) -> tuple[int, float] | None:
        """Two-stage parent constraint: pick best child under best parent.

        Parameters
        ----------
        all_sims : np.ndarray
            Cosine similarities of the query to every catalog node.
        fc : _FittedCatalog
            Fitted catalog state.
        best_depth : int
            Selected target depth.
        parent_depth : int
            Depth of the parent level (``best_depth - 1``).
        query_tokens : set[str] | None
            Tokenised query text for term-affinity boosting.

        Returns
        -------
        tuple[int, float] | None
            ``(best_idx, best_sim)`` of the constrained match, or ``None``
            if parent constraint could not be applied.
        """
        config = fc.config
        graph = config.graph

        if parent_depth < 1 or parent_depth not in fc.depth_masks:
            return None

        parent_mask = fc.depth_masks[parent_depth]
        if parent_mask.sum() == 0:
            return None

        # Bottom-up: pick parent whose best child scores highest
        target_mask_bu = fc.depth_masks.get(best_depth)
        parent_scores: dict[str, float] = {}
        if target_mask_bu is not None:
            target_indices_bu = np.where(target_mask_bu)[0]
            for idx in target_indices_bu:
                child_id = str(config.node_ids[idx])
                child_sim = float(all_sims[idx])
                for pid in graph.get_parents(child_id):
                    if pid in fc.id_to_idx and fc.node_depths[fc.id_to_idx[pid]] == parent_depth:
                        if pid not in parent_scores or child_sim > parent_scores[pid]:
                            parent_scores[pid] = child_sim

        # Term-affinity boost
        if parent_scores and query_tokens and fc.branch_terms:
            for pid in parent_scores:
                hits = len(query_tokens & fc.branch_terms.get(pid, set()))
                if hits > 0:
                    parent_scores[pid] += fc.term_boost * (hits / len(query_tokens))

        if parent_scores:
            parent_best_id = max(parent_scores, key=lambda k: parent_scores[k])
        else:
            # Fallback: class-embedding match (no children at target depth)
            parent_sims = all_sims[parent_mask]
            parent_indices = np.where(parent_mask)[0]
            parent_best_id = str(config.node_ids[parent_indices[np.argmax(parent_sims)]])

        # Get descendants of the parent's best match at the target depth
        descendants = graph.get_descendants(parent_best_id)
        target_mask = fc.depth_masks.get(best_depth)
        if target_mask is not None:
            target_indices = np.where(target_mask)[0]

            constrained_indices = [
                idx for idx in target_indices
                if str(config.node_ids[idx]) in descendants
            ]

            if constrained_indices:
                constrained_sims = all_sims[constrained_indices]
                local_best = np.argmax(constrained_sims)
                best_idx = constrained_indices[local_best]
                return (best_idx, float(constrained_sims[local_best]))

        return None

    def _match_flat(
        self,
        fc: _FittedCatalog,
        query_emb: np.ndarray,
        top_k: int = 5,
    ) -> CatalogMatch:
        """Match against a flat (single-level) catalog."""
        config = fc.config
        all_sims = config.embeddings @ query_emb

        best_idx = int(np.argmax(all_sims))
        best_sim = float(all_sims[best_idx])

        entropy_val = _compute_entropy(all_sims)
        node_z = float(fc.node_z_scores[best_idx])

        # Alternatives: top-k excluding best
        k = min(top_k, len(all_sims) - 1)
        sorted_idx = np.argsort(all_sims)[::-1]
        alternatives = [
            (str(config.node_ids[i]), str(config.node_names[i]), round(float(all_sims[i]), 4))
            for i in sorted_idx[1:k + 1]
        ]

        return CatalogMatch(
            catalog_name=config.name,
            node_id=str(config.node_ids[best_idx]),
            node_name=str(config.node_names[best_idx]),
            similarity=round(best_sim, 4),
            node_z=round(node_z, 3),
            ontological_z=0.0,
            depth=int(fc.node_depths[best_idx]),
            fit=_node_z_to_fit(node_z),
            entropy=round(entropy_val, 4),
            path_alignment=float(fc.path_alignments[best_idx]),
            alternatives=alternatives,
            gap_detected=False,
            gap_score=0.0,
        )

    def _detect_gap(
        self,
        depth_entropy: dict[int, float],
        depth_best_sim: dict[int, float],
    ) -> tuple[bool, float]:
        """Detect catalog gap using entropy + similarity drop between depths.

        A gap occurs when parent depth has low entropy (clear match) but
        child depth has high entropy AND worse match quality.
        """
        sorted_depths = sorted(depth_entropy.keys())
        if len(sorted_depths) < 2:
            return False, 0.0

        for i in range(len(sorted_depths) - 1):
            child_depth = sorted_depths[i + 1]
            parent_depth = sorted_depths[i]

            child_entropy = depth_entropy.get(child_depth, 0.0)
            parent_entropy = depth_entropy.get(parent_depth, 0.0)
            child_sim = depth_best_sim.get(child_depth, 0.0)
            parent_sim = depth_best_sim.get(parent_depth, 0.0)

            entropy_increase = child_entropy - parent_entropy
            similarity_drop = parent_sim - child_sim

            if (
                parent_entropy < 0.5
                and child_entropy > 0.7
                and entropy_increase > 0.3
                and similarity_drop > 0.1
                and child_sim < 0.8
            ):
                gap_score = min(
                    (0.5 - parent_entropy) * 0.3
                    + (child_entropy - 0.7) * 0.3
                    + entropy_increase * 0.2
                    + similarity_drop * 1.5
                    + (0.8 - child_sim) * 0.5,
                    1.0,
                )
                return True, max(gap_score, 0.1)

        return False, 0.0

    def _get_diverse_alternatives(
        self,
        fc: _FittedCatalog,
        query_emb: np.ndarray,
        target_depth: int,
        primary_idx: int,
        n: int = 5,
    ) -> list[tuple[str, str, float]]:
        """Best match from each distinct parent group, excluding primary's parent."""
        config = fc.config
        graph = config.graph
        mask = fc.depth_masks.get(target_depth)
        if mask is None:
            return []

        target_indices = np.where(mask)[0]
        if len(target_indices) < 2:
            return []

        all_sims = config.embeddings @ query_emb

        # Find primary's parent
        primary_id = str(config.node_ids[primary_idx])
        primary_parents = graph.get_parents(primary_id)
        primary_parent = primary_parents[0] if primary_parents else None

        # Take top ~50 candidates at this depth
        target_sims = all_sims[target_indices]
        k = min(50, len(target_indices))
        top_local = np.argsort(target_sims)[::-1][:k]

        best_per_parent: dict[str, tuple[str, str, float]] = {}
        for local_idx in top_local:
            idx = target_indices[local_idx]
            node_id = str(config.node_ids[idx])
            sim = float(all_sims[idx])

            parents = graph.get_parents(node_id)
            parent_id = parents[0] if parents else "_no_parent_"

            # Skip primary's parent group
            if parent_id == primary_parent:
                continue

            if parent_id not in best_per_parent or sim > best_per_parent[parent_id][2]:
                best_per_parent[parent_id] = (
                    node_id,
                    str(config.node_names[idx]),
                    round(sim, 4),
                )

        sorted_alts = sorted(best_per_parent.values(), key=lambda x: -x[2])
        return sorted_alts[:n]

    def _get_cross_domain_affinity(
        self,
        fc: _FittedCatalog,
        query_emb: np.ndarray,
        n: int = 5,
    ) -> dict[str, float]:
        """Get affinity scores at shallowest non-root depth."""
        graph = fc.config.graph
        config = fc.config

        # Find shallowest non-root depth with nodes
        shallow_depth = 1
        nodes_at_shallow = graph.nodes_at_depth(shallow_depth)
        if not nodes_at_shallow:
            return {}

        mask = fc.depth_masks.get(shallow_depth)
        if mask is None or mask.sum() == 0:
            return {}

        indices = np.where(mask)[0]
        sims = config.embeddings[indices] @ query_emb

        top_k = min(n, len(sims))
        top_local = np.argsort(sims)[::-1][:top_k]

        affinity = {}
        for local_idx in top_local:
            idx = indices[local_idx]
            name = str(config.node_names[idx])
            affinity[name] = round(float(sims[local_idx]), 3)

        return affinity

    # ── Joint matching (Phase 2) ──────────────────────────────────────

    def match(
        self,
        query_embs: np.ndarray,
        top_k: int = 5,
        coherence_weight: float = 0.3,
        query_labels: list[str] | None = None,
        query_texts: list[str] | None = None,
    ) -> list[JointMatchResult]:
        """Match query embeddings across all catalogs with coherence.

        Parameters
        ----------
        query_embs : np.ndarray
            (n_queries, d) query embeddings.
        top_k : int
            Number of candidates per catalog for reranking.
        coherence_weight : float
            0 = independent matching, 1 = coherence dominates.
        query_labels : list[str], optional
            Human-readable labels for each query.
        query_texts : list[str], optional
            Free-text descriptions per query for term disambiguation.

        Returns
        -------
        list[JointMatchResult]
            One result per query.
        """
        if not self._is_fitted:
            raise RuntimeError("Call fit() before matching")

        query_embs = np.asarray(query_embs, dtype=np.float32)
        if query_embs.ndim == 1:
            query_embs = query_embs.reshape(1, -1)

        n_queries = query_embs.shape[0]
        if query_labels is None:
            query_labels = [f"query_{i}" for i in range(n_queries)]

        results = []
        catalog_names = list(self._fitted.keys())

        for qi in range(n_queries):
            q_emb = query_embs[qi]
            q_text = query_texts[qi] if query_texts is not None else None

            if not self._mappings or coherence_weight == 0:
                # Independent matching — no coherence reranking
                matches = {}
                for cname in catalog_names:
                    matches[cname] = self.match_single(
                        cname, q_emb, top_k, query_text=q_text
                    )

                results.append(JointMatchResult(
                    query=query_labels[qi],
                    matches=matches,
                    coherence_score=1.0,
                    coherence_detail={},
                    reranked=False,
                    reason="independent (no mappings or weight=0)",
                ))
            else:
                result = self._match_joint(
                    q_emb, query_labels[qi], catalog_names, top_k,
                    coherence_weight, query_text=q_text,
                )
                results.append(result)

        return results

    def _match_joint(
        self,
        query_emb: np.ndarray,
        query_label: str,
        catalog_names: list[str],
        top_k: int,
        coherence_weight: float,
        query_text: str | None = None,
    ) -> JointMatchResult:
        """Match with cross-catalog coherence reranking."""
        # Step 1: Get top-k candidates per catalog
        candidates: dict[str, list[tuple[int, float]]] = {}
        for cname in catalog_names:
            fc = self._fitted[cname]
            all_sims = fc.config.embeddings @ query_emb
            k = min(top_k, len(all_sims))
            top_idx = np.argsort(all_sims)[::-1][:k]
            candidates[cname] = [(int(idx), float(all_sims[idx])) for idx in top_idx]

        # Step 2: Check if independent top-1 choices are already coherent
        independent_top1 = {
            cname: cands[0] for cname, cands in candidates.items() if cands
        }

        pair_coherence, all_coherent = self._compute_pair_coherence(independent_top1)

        if all_coherent or not pair_coherence:
            # Top-1 independent choices are coherent
            matches = {}
            for cname in catalog_names:
                matches[cname] = self.match_single(
                    cname, query_emb, top_k, query_text=query_text
                )

            overall = (
                float(np.mean(list(pair_coherence.values())))
                if pair_coherence else 1.0
            )
            return JointMatchResult(
                query=query_label,
                matches=matches,
                coherence_score=round(overall, 4),
                coherence_detail=pair_coherence,
                reranked=False,
                reason="top-1 independent choices are coherent",
            )

        # Step 3: Exhaustive search over top_k^N combinations
        best_combo, _score, reason = self._rerank_joint(
            candidates, catalog_names, coherence_weight
        )

        # Build matches from best combo
        matches = {}
        for cname in catalog_names:
            idx, sim = best_combo[cname]
            fc = self._fitted[cname]
            ontological_z = self._compute_ontological_z(fc, query_emb)
            node_z = float(fc.node_z_scores[idx])
            pa = float(fc.path_alignments[idx])

            # Alternatives from match_single
            single_match = self.match_single(
                cname, query_emb, top_k, query_text=query_text
            )

            matches[cname] = CatalogMatch(
                catalog_name=cname,
                node_id=str(fc.config.node_ids[idx]),
                node_name=str(fc.config.node_names[idx]),
                similarity=round(sim, 4),
                node_z=round(node_z, 3),
                ontological_z=round(ontological_z, 3),
                depth=int(fc.node_depths[idx]),
                fit=_node_z_to_fit(node_z),
                entropy=single_match.entropy,
                path_alignment=round(pa, 4),
                alternatives=single_match.alternatives,
                gap_detected=single_match.gap_detected,
                gap_score=single_match.gap_score,
            )

        # Recompute coherence for final combo
        final_coherence, _ = self._compute_pair_coherence(best_combo)

        overall = (
            float(np.mean(list(final_coherence.values())))
            if final_coherence else 1.0
        )

        return JointMatchResult(
            query=query_label,
            matches=matches,
            coherence_score=round(overall, 4),
            coherence_detail=final_coherence,
            reranked=True,
            reason=reason,
        )

    def _compute_pair_coherence(
        self,
        catalog_choices: dict[str, tuple[int, float]],
        threshold: float = 0.5,
    ) -> tuple[dict[str, float], bool]:
        """Score pairwise coherence for a set of catalog choices.

        Parameters
        ----------
        catalog_choices : dict[str, tuple[int, float]]
            Mapping of catalog_name -> (node_idx, similarity).
        threshold : float
            Minimum score to consider a pair coherent.

        Returns
        -------
        pair_coherence : dict[str, float]
            Mapping of ``"ca-cb"`` -> coherence score for each mapped pair.
        all_coherent : bool
            True if every scored pair meets the threshold.
        """
        pair_coherence: dict[str, float] = {}
        all_coherent = True

        for (ca, cb), mapping in self._mapping_lookup.items():
            if ca not in catalog_choices or cb not in catalog_choices:
                continue
            if ca >= cb:  # avoid double-counting
                continue

            idx_a = catalog_choices[ca][0]
            idx_b = catalog_choices[cb][0]
            id_a = str(self._fitted[ca].config.node_ids[idx_a])
            id_b = str(self._fitted[cb].config.node_ids[idx_b])

            score = self._mapping_score(mapping, id_a, id_b)
            pair_coherence[f"{ca}-{cb}"] = score
            if score < threshold:
                all_coherent = False

        return pair_coherence, all_coherent

    def _rerank_joint(
        self,
        candidates: dict[str, list[tuple[int, float]]],
        catalog_names: list[str],
        coherence_weight: float,
    ) -> tuple[dict[str, tuple[int, float]], float, str]:
        """Exhaustive search over candidate combinations.

        Maximizes: product(similarity_i) * coherence_bonus^coherence_weight
        """
        # Build candidate lists per catalog
        cand_lists = [candidates[cname] for cname in catalog_names]

        best_score = -np.inf
        best_combo: dict[str, tuple[int, float]] = {}

        for combo in itertools_product(*cand_lists):
            # combo is a tuple of (idx, sim) per catalog
            sim_product = 1.0
            for _idx, sim in combo:
                sim_product *= max(sim, 1e-6)

            # Coherence bonus: for each mapped pair, check membership
            coherence_bonus = 1.0
            n_pairs = 0
            for i, ca in enumerate(catalog_names):
                for j, cb in enumerate(catalog_names):
                    if i >= j:
                        continue
                    key = (ca, cb)
                    if key not in self._mapping_lookup:
                        continue
                    mapping = self._mapping_lookup[key]
                    id_a = str(self._fitted[ca].config.node_ids[combo[i][0]])
                    id_b = str(self._fitted[cb].config.node_ids[combo[j][0]])
                    score = self._mapping_score(mapping, id_a, id_b)
                    coherence_bonus *= (1.0 + score)
                    n_pairs += 1

            total = sim_product * (coherence_bonus ** coherence_weight)

            if total > best_score:
                best_score = total
                best_combo = {
                    cname: combo[i]
                    for i, cname in enumerate(catalog_names)
                }

        n_combos = 1
        for cl in cand_lists:
            n_combos *= len(cl)
        reason = f"reranked {n_combos} combinations, coherence_weight={coherence_weight}"

        return best_combo, best_score, reason

    def _mapping_score(
        self,
        mapping: CrossMapping,
        source_id: str,
        target_id: str,
    ) -> float:
        """Check if (source_id, target_id) exists in a mapping.

        Returns the weight if found, 0.0 otherwise.
        """
        matches = (mapping.source_ids == source_id) & (mapping.target_ids == target_id)
        if matches.any():
            return float(mapping.weights[matches][0])
        return 0.0

    # ── Serialization (Phase 3) ───────────────────────────────────────

    def to_dict(self) -> dict:
        """Serialize CatalogSpace to a JSON-compatible dict.

        Includes catalog configs and mappings but not fitted state
        (call fit() after from_dict()).
        """
        catalogs = {}
        for name, config in self._configs.items():
            catalogs[name] = {
                "name": config.name,
                "graph": config.graph.to_dict(),
                "embeddings": config.embeddings.tolist(),
                "node_ids": config.node_ids.tolist(),
                "node_names": config.node_names.tolist(),
            }

        mappings = []
        for m in self._mappings:
            mappings.append({
                "source_catalog": m.source_catalog,
                "target_catalog": m.target_catalog,
                "source_ids": m.source_ids.tolist(),
                "target_ids": m.target_ids.tolist(),
                "weights": m.weights.tolist(),
            })

        return {"catalogs": catalogs, "mappings": mappings}

    @classmethod
    def from_dict(cls, data: dict) -> CatalogSpace:
        """Reconstruct CatalogSpace from to_dict() output.

        NOTE: Returns unfitted — call fit() after.
        """
        space = cls()
        for _name, cdata in data["catalogs"].items():
            config = CatalogConfig(
                name=cdata["name"],
                graph=CategoryGraph.from_dict(cdata["graph"]),
                embeddings=np.array(cdata["embeddings"], dtype=np.float32),
                node_ids=np.array(cdata["node_ids"], dtype=str),
                node_names=np.array(cdata["node_names"], dtype=str),
            )
            space.add_catalog(config)

        for mdata in data.get("mappings", []):
            mapping = CrossMapping(
                source_catalog=mdata["source_catalog"],
                target_catalog=mdata["target_catalog"],
                source_ids=np.array(mdata["source_ids"], dtype=str),
                target_ids=np.array(mdata["target_ids"], dtype=str),
                weights=np.array(mdata["weights"], dtype=np.float32),
            )
            space.add_mapping(mapping)

        return space

    def to_json(self) -> str:
        """Serialize to JSON string."""
        return json.dumps(self.to_dict())

    @classmethod
    def from_json(cls, s: str) -> CatalogSpace:
        """Reconstruct from JSON string. Returns unfitted."""
        return cls.from_dict(json.loads(s))

    # ── Catalog info ──────────────────────────────────────────────────

    @property
    def catalog_names(self) -> list[str]:
        """Names of all registered catalogs."""
        return list(self._configs.keys())

    @property
    def is_fitted(self) -> bool:
        """Whether fit() has been called."""
        return self._is_fitted

    def summary(self) -> str:
        """Human-readable summary."""
        parts = [f"CatalogSpace: {len(self._configs)} catalogs"]
        for name, config in self._configs.items():
            n = len(config.node_ids)
            d = config.embeddings.shape[1] if config.embeddings.ndim == 2 else 0
            parts.append(f"  {name}: {n} nodes, {d}d embeddings")
        if self._mappings:
            parts.append(f"  {len(self._mappings)} cross-mappings")
        parts.append(f"  fitted={self._is_fitted}")
        return "\n".join(parts)


# ── Helper on CatalogMatch for internal use ──────────────────────────────


def _replace_ontological_z(match: CatalogMatch, oz: float) -> CatalogMatch:
    """Return a new CatalogMatch with updated ontological_z and fit."""
    return CatalogMatch(
        catalog_name=match.catalog_name,
        node_id=match.node_id,
        node_name=match.node_name,
        similarity=match.similarity,
        node_z=match.node_z,
        ontological_z=round(oz, 3),
        depth=match.depth,
        fit=match.fit,
        entropy=match.entropy,
        path_alignment=match.path_alignment,
        alternatives=match.alternatives,
        gap_detected=match.gap_detected,
        gap_score=match.gap_score,
    )
