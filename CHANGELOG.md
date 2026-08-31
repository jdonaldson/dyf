# Changelog

## Unreleased

### Added

- **`dyf.dedup` — ingest-time near-duplicate detection.** `near_duplicate_clusters()`
  groups points at cosine > 0.99 using multi-table random-projection LSH, so you can index
  one representative per duplicate cluster instead of every copy. Returns a `DedupResult`
  with `.mask()` (which points to index), `.members()` (representative → the points it
  stands for), and `.member_field()` (a `stored_fields`-ready encoding of the mapping).

  **Measure your corpus before enabling this — the duplicate rate spans 0% to 88%** across
  six corpora (`benchmarks/sequence_arc/sec_dedup_corpora.py`, weighed `.dyf` files):

  | corpus | dup rate | file saving |
  |---|---|---|
  | CMU MoCap 62d (adjacent frames) | 88.3% | 77.1% |
  | SEC 10-Q 768d (legal boilerplate) | 29.4% | 25.6% |
  | news 384d | 1.0% | ~0% |
  | tweets 384d | 0.1% | ~0% |
  | arxiv 384d | 0.0% | ~0% |
  | wikipedia 384d | 0.0% | ~0% |

  Curated document collections have almost no near-duplicates; templated corpora and
  temporally oversampled ones have many. Where duplicates exist, **file saving is reliably
  ~0.87× the duplicate rate** — the shortfall is tree overhead, since every node stores a
  dim-length centroid and leaf count falls more slowly than point count.
  `near_duplicate_clusters` is itself the diagnostic at ~1s per 100k points, so measure
  first rather than enabling by default.

  Retrieval quality *improves* when scored on distinct content: measured against the top-10
  distinct clusters, the deduped index wins at every work budget tested but the largest,
  where it reaches parity. Scored against raw brute-force `recall@10` it appears to lose —
  but that metric puts a mean of 2.97 duplicate slots in every true top-10 (83% of queries
  affected), so it pays for returning redundant copies, which is what dedup exists to
  prevent.

  **No file-format change.** The representative → members mapping travels as an ordinary
  utf8 stored field, so `.dyf` schema, the generated flatbuffers, and the Rust reader are
  all untouched. `search()` already returns stored fields in `result.fields`.

  Clustering is *star*, not transitive: a point joins a representative only if it is within
  threshold of *that representative*. Transitive union-find was measured to chain a
  541-member "duplicate" cluster containing genuinely dissimilar documents, and cost 5pp of
  recall where star clustering gained 2.6pp.

  ```python
  from dyf import near_duplicate_clusters, build_dyf_tree, write_lazy_index

  result = near_duplicate_clusters(embeddings)          # cosine > 0.99
  reps = embeddings[result.mask()]
  tree = build_dyf_tree(reps, max_depth=4, num_bits=4)
  write_lazy_index(tree, reps, "index.dyf",
                   stored_fields={"dup_members": result.member_field()})
  ```

## 0.12.1

### Fixed

- **Bare `pip install dyf` was unusable** — `import dyf` raised
  `ModuleNotFoundError: No module named 'sklearn'` because three modules
  imported sklearn/scipy at module top while core dependencies list only
  numpy + dyf-rs. Those imports are now function-local (they load on first
  use of the agglomerative cut, `DensityClassifierFull` hashing, and PCA-tree
  paths). The core path — `build_dyf_tree` → `write_lazy_index` → rust-backed
  `LazyIndex.search` — now runs with no sklearn installed at all.

## 0.12.0

### DYF3 is the default file format

`write_lazy_index` now writes **DYF3** by default (was DYF1). DYF3 is the same
header-based layout with a chunk-capable header — readable by python, the
browser viewer, and the rust kernel — so default-written indexes get rust-speed
search AND stay web-deployable. DYF2 remains the explicit choice for
append-heavy workloads (rust read-write); DYF1 files remain fully readable.

- Requires **dyf-rs >= 0.10.0**, which adds read-only DYF1/DYF3 support to the
  rust kernel (mutations on header-based formats raise: their front-loaded
  index can't grow in place — convert to DYF2 first).

### Fixed

- **`LazyIndex.search` crashed on DYF1/DYF3 files** with `OSError: Invalid magic`
  when using the default `backend="rust"` (dyf-rs only reads DYF2). Since
  `write_lazy_index` defaults to `format_version=1`, any default-written index hit
  this on its first search. Non-DYF2 files now fall back to the python path, as
  PQ/overflow indexes already did. Regression introduced in 0.11.0.

### Changed

- `LazyIndex` now **capability-probes** the installed dyf-rs rather than
  hard-coding "rust = DYF2 only": with dyf-rs >= 0.10.0 (which adds DYF1/DYF3
  read support), default-format indexes are served by the rust kernel too;
  older dyf-rs (or compressed/chunked files it rejects) falls back to python
  transparently.

### CI

- Lint (`ruff check` + `ruff format --check`) actually passes again — CI had been
  red on lint since March, so tests never ran (which is how the crash above shipped).
  `ruff` is now pinned to a minor version in dev extras so formatter drift can't
  silently break CI.

## 0.11.0

### Rust is the default search backend

`LazyIndex.search` now defaults to `backend="rust"`. The Rust kernel covers the full
load-bearing query path — **fixed and adaptive `nprobe`** (`"auto"` /
`AdaptiveProbeConfig`) and **`return_routing`** — all faithful to the Python reference
(top-k overlap 1.000; `adaptive_nprobe` identical). Only PQ-compressed and overflow
indexes fall back to Python; `backend="python"` remains as an explicit override.

- Requires **dyf-rs >= 0.9.0** (extended `search_batch`: adaptive resolution + routing).
- Note: on the rust path, `routing["leaves_probed"]` is a count (the python path returns
  a list of probed batch indices) — diagnostic only.

## 0.10.0

### Dense Rust-backed multiprobe search

New `DenseSearchIndex` — a dense in-memory nearest-neighbor search over a dyf tree,
backed by the Rust kernel `dyf_rs.dense_search_batch` (requires **dyf-rs >= 0.8.0**).

- **`DenseSearchIndex(embeddings).search(queries, k=10, nprobe=256)`** routes queries
  through a `build_dyf_tree` partition via a batched, rayon-parallel Rust kernel. Top-k
  is identical to the Python multiprobe at ~100x lower per-query latency; on 8.84M
  MSMARCO it reproduces dyf-mp recall at ~9ms/query (vs ~58ms in pure Python). Accepts
  a 1D query (`(dim,)` → `(k,)`) or a batch (`(nq, dim)` → `(nq, k)`).
- **`flatten_tree`** exposed for callers who want the CSR arrays the kernel consumes.
- Dependency floor raised to `dyf-rs>=0.8.0`.

### On-disk `LazyIndex.search(backend="rust")`

`LazyIndex.search` gains a `backend=` argument. `backend="rust"` routes the on-disk
search through a preloaded Rust kernel (`dyf_rs.DyfSearcher`) — same `SearchResult`
shape, stored fields included. On the immich index (n≈35k, dim=512, f16) it is ~15×
faster with stored fields and ~100×+ on pure-retrieval indexes, warm.

- Default `backend="python"` — behavior unchanged.
- Falls back to the python path for: PQ-compressed indexes, overflow batches,
  adaptive `nprobe` (`"auto"`/`AdaptiveProbeConfig`), and `return_routing=True`.
- Uses the Rust searcher's lazy mode (instant open, bounded memory — only touched
  leaves are decoded; per-leaf contiguous scoring). Cold leaves pay a one-time
  decode on first use.
- Recall-validated, not bit-exact: top-k matches the python path on MSMARCO
  (overlap 1.000) and is within ~0.92 on some indexes; recall@10 vs exact
  brute-force is identical across backends.

## 0.9.0

### Substrate API for downstream consumers

The CatalogSpace subsystem is now a first-class public substrate API.
Downstream consumers (notably shortorder) previously had to reach into
underscore-prefixed names to use it; those reach-ins are now eliminated.

- **`CatalogSpace`, `CatalogConfig`, `CatalogMatch`, `CrossMapping`,
  `FittedCatalog`, `JointMatchResult`, `compute_similarity_entropy`**
  are now exported from the top-level `dyf` namespace.
  Previously `from dyf import CatalogSpace` raised `ImportError` despite
  the comment in `__init__.py` describing it as "available via dyf.catalog";
  it's now genuinely available from both paths.
- **`_FittedCatalog` → `FittedCatalog`** (the public class name).
  `_FittedCatalog` is retained as a backcompat alias.
- **`compute_similarity_entropy`** added as the public alias for the
  internal `_compute_entropy` (legacy name retained).
- **`CatalogSpace.get_fitted(catalog_name) -> FittedCatalog`** — public
  accessor replacing direct `._fitted[name]` reach-ins.
- **`CatalogSpace.get_lca_depth(catalog_name, a, b) -> int`** — convenience
  wrapper around `CategoryGraph.lca_depth`.
- **`CatalogSpace.get_cross_domain_affinity(catalog_name, query_emb, n=5)`**
  — public version of the prior private `_get_cross_domain_affinity`,
  consistent with the other public accessors (takes `catalog_name`).

### Dependency bump

- `dyf-rs >= 0.7.2` (was `>= 0.7.0`) — pulls in the new `.pyi` type stubs
  for `BridgeAnalysis`, `BridgePersistence`, and `MultiResolutionAnalysis`.
  Stubs document field semantics (e.g. `recovery_depth` encoding,
  `bridge_ratio` interpretation) that previously lived only in the Rust
  source.

## 0.8.1

### API Changes

- **Unified `cut_tree_to_labels` dispatcher** — replaces the two previously
  divergent cut functions (`pca_tree.cut_tree_to_labels` and
  `dyf_tree.cut_dyf_tree_to_labels`). The new top-level
  `dyf.cut_tree_to_labels(tree, n_points, n_clusters, *, max_depth=None, embeddings=None)`
  detects tree shape from its keys (`'children'` → DYF, `'left'` → PCA) and
  routes to the correct impl, raising a clear `ValueError` if the required
  kwarg for the detected shape is missing. The old per-module functions are
  now private (`_cut_pca_tree_to_labels`, `_cut_dyf_tree_to_labels`) — import
  from `dyf` or `dyf.cut` instead.

### Fixed

- **Defensive `np.asarray` on `get_bucket_ids()` returns in `dyf_tree.py`**
  — three sites (`_build_dyf_tree`, `_try_resplit`, `_resplit_ejected`) now
  wrap the Rust binding's return, so dyf-py works against any
  `dyf-rs >= 0.5.0` instead of hard-requiring the 0.6.0 numpy-return bindings.
  Prevents stale-editable-install failures that surface as
  `AttributeError: 'list' object has no attribute 'tolist'`.

## 0.8.0

DYF3 chunked format, adaptive search, embedding-optional indexes, and zero-copy numpy returns from Rust bindings.

### Added

- **DYF3 chunked format** — new file format version with 32-byte header supporting chunked transport for CDN/GitHub Pages hosting. `split_dyf3()` splits large `.dyf` files into <95MB chunks that reassemble transparently on load. JS reader (`dyf_reader.mjs`) and viz server updated for DYF3.
- **Embedding-optional indexes** — `write_lazy_index(embeddings=None)` creates viz-only `.dyf` files without embedding vectors, preserving tree centroids for structure while dropping bulk data. `rewrite_lazy_index(drop_embeddings=True)` strips embeddings from existing files.
- **Adaptive probing** — `LazyIndex.search(nprobe="auto")` dynamically adjusts probe count based on routing margin. Confident queries (large margins) probe fewer leaves; uncertain queries (small margins) probe more. Achieves ~90% recall at ~70% of uniform 3-probe cost. Configurable via `AdaptiveProbeConfig(margin_lo, margin_hi, min_probes, max_probes)`.
- **Enrich scaffold CLI** — `dyf enrich scaffold` pre-computes LLM scaffold data (cluster structure, keyword extraction) without requiring an LLM call.
- **Routing diagnostics** — `search(return_routing=True)` now includes `min_margin` (minimum projection margin on primary path). Adaptive mode adds `adaptive_nprobe` and `nprobe_mode`.

### API Changes

- **`dyf-rs` returns numpy arrays** — `get_bucket_ids()`, `get_bucket_sizes()`, `get_centroid_similarities()`, `get_isolation_scores()`, `get_stability_scores()`, `get_eigenvalues()` now return `numpy.ndarray` instead of Python lists. `get_hyperplanes()` returns a 2D numpy array. `louvain_from_centroids()` and `louvain_communities()` return numpy arrays for labels. Eliminates redundant `np.array()` wrappers at all call sites.
- Requires `dyf-rs>=0.7.0`
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
