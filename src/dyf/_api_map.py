"""A grouped map of dyf's public API, and `dyf.overview()` to read it.

`from dyf import *` offers 111 names in one flat list. That is a fine *manifest* and a
poor *index*: nothing tells a reader which of them is an entry point, which are result
types that only appear as return values, and which belong together. The module docstring
covered three of the 111.

This is the same index/body split `dyf info` applies to a `.dyf` file — a cheap summary
that says what is inside so you can decide what to open — applied to the package itself.

The groups are the ones already implied by the comments in `__all__`; this makes them
readable at runtime instead of only in the source. `tests/test_api_map.py` asserts the map
covers `__all__` exactly, so a new export cannot quietly go unlisted and a removed one
cannot linger — the map is checked, not merely maintained.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class APIGroup:
    """One themed slice of the public API.

    `start_here` is the single name to read first — the entry point that makes the rest
    of the group make sense. Result types and helpers are rarely where anyone should
    start, and a flat list cannot say so.
    """

    summary: str
    start_here: str | None
    names: tuple[str, ...]


API_GROUPS: dict[str, APIGroup] = {
    "introspection": APIGroup(
        summary="This map. `dyf.overview()` for all groups, `dyf.overview('trees')` for one.",
        start_here="overview",
        names=("overview", "API_GROUPS"),
    ),
    "density": APIGroup(
        summary="Classify points as dense / bridge / orphan by LSH bucket density.",
        start_here="DensityClassifier",
        names=(
            "DensityClassifier",
            "DensityReport",
            "BridgeAnalysis",
            "DensityClassifierFull",
            "EmbedderConfig",
            "LabelerConfig",
            "list_configs",
        ),
    ),
    "trees": APIGroup(
        summary="Build, refine and cut the k-ary DYF tree. The substrate everything else routes through.",
        start_here="build_dyf_tree",
        names=(
            "build_dyf_tree",
            "refine_dyf_tree",
            "refine_clusters",
            "cut_tree_to_labels",
            "extract_boundary_persistence",
            "boundary_persistence_scores",
        ),
    ),
    "index": APIGroup(
        summary="Write and read .dyf files. Opens in ~5ms without loading data; see `dyf info`.",
        start_here="LazyIndex",
        names=(
            "LazyIndex",
            "write_lazy_index",
            "rewrite_lazy_index",
            "split_dyf3",
            "from_faiss",
            "detect_dyf_version",
            "SearchResult",
            "AdaptiveProbeConfig",
            "ExtractedData",
            "StoredFieldValue",
            "StoredFieldInput",
            "TreeNode",
        ),
    ),
    "search": APIGroup(
        summary="In-memory search over a tree. Returns the same SearchResult as LazyIndex.",
        start_here="DenseSearchIndex",
        names=("DenseSearchIndex", "flatten_tree"),
    ),
    "retrieval": APIGroup(
        summary="Bridge-anchored approximate retrieval and result re-ranking.",
        start_here="BridgeIndex",
        names=(
            "BridgeIndex",
            "SuperConnectorResult",
            "OrthogonalAnchorResult",
            "FacetDiverseResult",
            "find_super_connectors",
            "select_orthogonal_anchors",
            "diversify_by_facet",
            "get_kmeans_init",
            "rerank_standard",
            "rerank_mmr",
            "rerank_bridge_boost",
            "rerank_bridge_mmr",
        ),
    ),
    "dedup": APIGroup(
        summary="Collapse near-duplicates before indexing. Measure first — rates are corpus-dependent.",
        start_here="near_duplicate_clusters",
        names=(
            "near_duplicate_clusters",
            "dedup_for_index",
            "DedupResult",
            "DedupForIndexResult",
            "decode_members",
        ),
    ),
    "ontology": APIGroup(
        summary="Discover DAG taxonomies and layered structure from embeddings alone.",
        start_here="build_unified_ontology",
        names=(
            "build_unified_ontology",
            "build_dag_taxonomy",
            "build_rog_ontology",
            "mine_dag_chains",
            "compute_neighbor_diversity",
            "compute_hub_score",
            "DAGChain",
            "DAGMiningResult",
            "DAGTaxonomy",
            "UnifiedOntologyResult",
            "ROGLayer",
            "ROGResult",
            "HubScoreResult",
        ),
    ),
    "catalogs": APIGroup(
        summary="Match queries against declared hierarchical catalogs (UNSPSC, GMDN, ...). Independent of the ingest path.",
        start_here="CatalogSpace",
        names=(
            "CatalogSpace",
            "CatalogConfig",
            "CatalogMatch",
            "CrossMapping",
            "FittedCatalog",
            "JointMatchResult",
            "compute_similarity_entropy",
        ),
    ),
    "categorical": APIGroup(
        summary="Label hierarchies, axis diagnostics, and Fisher dimension weighting.",
        start_here="diagnose_axes",
        names=(
            "diagnose_axes",
            "embed_with_diagnostics",
            "AxisDiagnostic",
            "AxisDiagnosticsResult",
            "CategoryGraph",
            "coarsen",
            "diagnostics_to_metadata",
            "discover_categorical_columns",
            "multi_level_fisher_weights",
            "store_category_graph",
            "load_category_graphs",
            "compute_fisher_weights",
            "apply_fisher_weights",
        ),
    ),
    "labeling": APIGroup(
        summary="Derive human-readable keywords and paths for tree splits and clusters.",
        start_here="compute_split_keywords",
        names=(
            "compute_split_keywords",
            "compute_embedding_keywords",
            "compute_domain_stopwords",
            "build_tree_maps",
            "collect_descendant_indices",
            "format_split_path",
            "label_clusters_frequency",
            "assess_text_diversity",
            "TextDiversityReport",
            "tokenize",
            "build_cluster_tree_dag",
            "compute_sibling_keywords",
            "derive_path_labels",
            "format_cluster_context",
        ),
    ),
    "grouping": APIGroup(
        summary="Agglomerate tree leaves into a target number of groups.",
        start_here="agglomerate_tree_leaves",
        names=(
            "agglomerate_tree_leaves",
            "louvain_cluster_leaves",
            "merge_to_max_k",
            "LeafGroupingResult",
        ),
    ),
    "chunks": APIGroup(
        summary="Assess chunked-document corpora: redundancy, coherence, spread.",
        start_here="chunk_redundancy",
        names=(
            "chunk_redundancy",
            "cluster_quality",
            "deduplicate_chunks",
            "doc_spread",
            "neighbor_coherence",
            "DocSpread",
        ),
    ),
    "provenance": APIGroup(
        summary="Stamp and check artifact identity. NOTE: nothing in dyf writes these yet — stamping is the caller's job.",
        start_here="create_provenance",
        names=(
            "create_provenance",
            "check_compatible",
            "Provenance",
            "file_hash",
            "params_hash",
            "provenance_to_dict",
            "provenance_from_dict",
        ),
    ),
    "color": APIGroup(
        summary="Map cluster or tree structure to RGB for visualization.",
        start_here="spatial_rgb_map",
        names=("spatial_rgb_map", "spatial_color_map", "tree_rgb_map"),
    ),
}

#: Submodules that are deliberately not re-exported at top level. Import them directly.
NOT_REEXPORTED: dict[str, str] = {
    "dyf.pipeline": "DAG pipeline runner — internal/experimental.",
    "dyf.concept_graph": "Concept graph over markdown notes — dev tooling, drives `dyf concepts`.",
    "dyf.info": "Backs `dyf info`; call the CLI rather than importing.",
}


def overview(group: str | None = None, *, as_dict: bool = False):
    """Print or return a map of dyf's public API.

    Args:
        group: Show only this group. Omit for all groups.
        as_dict: Return a plain dict instead of a formatted string — for callers that
            want to parse rather than read.

    Returns:
        A formatted string, or a dict when ``as_dict`` is True.

    Raises:
        KeyError: If ``group`` is not a known group name.
    """
    if group is not None and group not in API_GROUPS:
        raise KeyError(f"unknown group {group!r}; known groups: {', '.join(sorted(API_GROUPS))}")

    selected = {group: API_GROUPS[group]} if group else API_GROUPS

    if as_dict:
        return {
            name: {
                "summary": g.summary,
                "start_here": g.start_here,
                "names": list(g.names),
            }
            for name, g in selected.items()
        }

    lines: list[str] = []
    if group is None:
        total = sum(len(g.names) for g in API_GROUPS.values())
        lines.append(f"dyf — {total} public names in {len(API_GROUPS)} groups")
        lines.append("")
    for name, g in selected.items():
        lines.append(f"{name}  ({len(g.names)})")
        lines.append(f"    {g.summary}")
        if g.start_here:
            lines.append(f"    start here: {g.start_here}")
        if group is not None:
            for n in g.names:
                lines.append(f"      {n}")
        lines.append("")
    if group is None:
        lines.append("dyf.overview('<group>') lists a group's names.")
        lines.append("Not re-exported: " + ", ".join(NOT_REEXPORTED))
    return "\n".join(lines).rstrip()
