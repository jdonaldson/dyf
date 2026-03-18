# Changelog

## 0.8.0

DYF3 chunked format, adaptive search, and embedding-optional indexes.

### Added

- **DYF3 chunked format** — new file format version with 32-byte header supporting chunked transport for CDN/GitHub Pages hosting. `split_dyf3()` splits large `.dyf` files into <95MB chunks that reassemble transparently on load. JS reader (`dyf_reader.mjs`) and viz server updated for DYF3.
- **Embedding-optional indexes** — `write_lazy_index(embeddings=None)` creates viz-only `.dyf` files without embedding vectors, preserving tree centroids for structure while dropping bulk data. `rewrite_lazy_index(drop_embeddings=True)` strips embeddings from existing files.
- **Adaptive probing** — `LazyIndex.search(nprobe="auto")` dynamically adjusts probe count based on routing margin. Confident queries (large margins) probe fewer leaves; uncertain queries (small margins) probe more. Achieves ~90% recall at ~70% of uniform 3-probe cost. Configurable via `AdaptiveProbeConfig(margin_lo, margin_hi, min_probes, max_probes)`.
- **Enrich scaffold CLI** — `dyf enrich scaffold` pre-computes LLM scaffold data (cluster structure, keyword extraction) without requiring an LLM call.
- **Routing diagnostics** — `search(return_routing=True)` now includes `min_margin` (minimum projection margin on primary path). Adaptive mode adds `adaptive_nprobe` and `nprobe_mode`.

### API Changes

- `search()` `nprobe` parameter accepts `int`, `"auto"`, or `AdaptiveProbeConfig`
- `write_lazy_index()` gains `format_version` (1/2/3) and `embedding_dim` parameters
- `rewrite_lazy_index()` gains `drop_embeddings` and `format_version` parameters
- `AdaptiveProbeConfig`, `split_dyf3` added to public API exports

## 0.7.4

- Default `num_stability_seeds` to 0 (off) for 3-4x `fit()` speedup

## 0.7.3

- Require `dyf-rs>=0.6.0` for `ComputeBackend` support

## 0.6.2

### Added

- **`CategoryGraph.lca_depth()`** — returns the depth of the lowest common ancestor between two taxonomy nodes. Enables distinguishing wrong-domain errors (LCA at root) from imprecise matches (LCA within same segment).

## 0.6.1

Term disambiguation for thin-margin catalog matching, plus public API cleanup.

### Added

- **Term disambiguation in CatalogSpace** — `_compute_branch_terms()` computes TF-IDF discriminating terms per CategoryGraph branch at `fit()` time. Optional `query_text` parameter on `match_single()` and `match()` applies an additive term-affinity boost during bottom-up parent scoring. Fixes thin-margin mismatches like "bone drill" routing to Hardware instead of Medical. No `query_text` = no boost = identical to previous behavior.
- **`tokenize()` public API** — renamed from `_tokenize` and exported from `dyf.splits`. Used internally by splits, cluster_tree, and catalog modules.
- **`compute_embedding_keywords()` export** — previously defined but not reachable from `dyf` package; now exported.

## 0.6.0

Cluster labeling is now deterministic and hierarchy-aware.

Previous releases used LLM-generated labels with heuristic keyword matching — labels were non-reproducible and couldn't leverage the DYF tree's inherent vocabulary separation. This release replaces that with a structural approach: the DYF tree's PCA-based splits naturally separate distinct vocabularies, and BIRCH clusters can be mapped back to tree nodes via item overlap. Together these produce deterministic, LLM-free cluster labels with hierarchical context.

The bottom-up parent selection fix completes this theme — even the catalog matching system now trusts commodity-level signals over abstract class embeddings when the data supports it.

### Added

- **Split-based TF-IDF keyword extraction** (`src/dyf/splits.py`) — each PCA split in the DYF tree cleanly separates vocabularies; `compute_split_keywords()` extracts discriminative terms per split side without any LLM calls. New `splits` subcommand in `dyf_enrich.py`.
- **Cluster-tree DAG** (`src/dyf/cluster_tree.py`) — connects BIRCH spatial clusters to DYF tree nodes via item overlap, enabling deterministic path labels and contrastive sibling keywords for cluster naming.
- **Cluster glyphs as separate metadata** — `annotate_cluster_names()` returns `(clean_names, glyphs_dict)` instead of prepending glyph characters to name strings. Stored in `cluster_glyphs_{k}_{dim}` metadata, exposed via MCP `get_cluster_info()`.

### Fixed

- **Bottom-up parent selection** in `CatalogSpace._match_with_levels()` — parent signal now derived from child similarities (max aggregation) instead of class-level embedding similarity. Fixes cross-segment homograph bug (e.g. "electrosurgical pencil" matching "Writing instruments" instead of "Surgical instruments").
- **Within-z depth tiebreaker** — when two depths have equal discrimination, prefer the deeper level (principle of least power).

## 0.5.0

CatalogSpace multi-catalog matching, .dyf enrichment pipeline, Fisher weighting experiments, categorical DAG module. See git log for details.

## 0.4.1

Badges, dyf.io links, gitignore cleanup.

## 0.3.0

Simplify to raw scores, remove recovery bucket logic (breaking).

## 0.2.0

Rename OutlierClassifier to DensityClassifier (breaking).

## 0.1.0

Initial release: DYF Python wrapper over dyf-rs.
