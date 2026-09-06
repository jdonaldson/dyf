# Changelog

## 0.13.0 — 2026-09-05

Pre-v1, so the breaking changes below land in a minor bump. Two themes: **agents became a
first-class consumer** (a CLI that produced no output at all, exit codes that meant
nothing, no way to discover the API), and **return shapes were reconciled** while that is
still cheap.

The enrichment pipeline and browser tour moved out to
[dyfviz](https://github.com/jdonaldson/dyfviz).

### Removed

- **`build_pca_tree` and the `dyf.pca_tree` module.** It built a binary PC1-bisection
  tree that **nothing in the package consumed** — `write_lazy_index`, `LazyIndex`, the
  Rust kernel and every `index-*` path work on the k-ary DYF tree — while `dyf_tree`
  offered a strict superset of its public functions.

  **This fixes a live footgun.** `dyf.extract_boundary_persistence` and
  `dyf.boundary_persistence_scores` resolved to the *PCA* variants, so the most natural
  call died with a bare `KeyError: 'left'`:

  ```python
  from dyf import build_dyf_tree, extract_boundary_persistence
  extract_boundary_persistence(build_dyf_tree(embeddings, max_depth=3))  # KeyError
  ```

  That is the same failure as `KNOWN_ISSUES` #2, which had been fixed for the `cut_*`
  pair with a dispatcher but never swept across the class. Both names now resolve to the
  DYF variants and work on the tree this package actually produces.

  `cut_tree_to_labels` no longer dispatches — with one tree shape there is nothing to
  route — but keeps a shape check so a wrong dict still gets a named `ValueError`. Its
  `max_depth` argument is now ignored and accepted only for compatibility.

  ⚠ Every boundary-persistence test had been written against the PCA variants, so the
  DYF ones had **no coverage**. Ported to `tests/test_dyf_tree_boundaries.py` (15 tests)
  before the deletion, not after.

- **`dyf enrich` and `dyf tour` moved to the new `dyfviz` package.** dyf is now a library
  about indexing and search; UMAP projection, Louvain clustering, LLM labelling,
  narration, Kokoro TTS and the browser viewer live downstream in
  [dyfviz](https://github.com/jdonaldson/dyfviz). 2,875 lines across 12 modules.

  The dependency runs one way — dyfviz reads and rewrites `.dyf` files, and nothing in
  dyf imports dyfviz — so the move needed **no import rewrites**: every cross-package
  import in `enrich/` was already an absolute `from dyf.X import Y`.

  `dyf enrich ...` and `dyf tour ...` now exit 4 with a message naming the replacement
  command, rather than dumping usage. The `enrich` extra is gone; use `dyfviz[all]`.

  Two couplings survive on purpose, as format conventions rather than code dependencies:
  `LazyIndex.detect_enrichment_level()` still recognises the levels dyfviz writes, and
  `dyf info` still reports them.

### Added

- **`--dry-run` on `index-source`, `index-images` and `index-video`.** Every ingest path is
  cheap-then-expensive — scan, then parse or decode, then embed — and the embedding pass is
  the one you cannot take back. `--dry-run` runs the cheap stages and reports what the
  expensive one would cost: file and chunk counts, embedding batches, and for
  `index-source` whether the Ollama service is reachable and has the model. Add `--json`
  for a `schema_version: 0` payload.

  This encodes the *Sanity Check Before Deep Work* judgement — "don't re-embed 2.7M records
  when a regex gets 95%" — as something the tool affords rather than something a caller has
  to remember.

  Two rules it holds itself to:

  - **A dry run stays cheap.** For `index-video` the *counting itself* is the expensive
    step — scene detection is a full decode — so it reports the scene count as **unknown**
    rather than doing the work the flag exists to help you avoid. A test asserts this by
    previewing a file that is not a valid video at all: it succeeds, proving nothing tried
    to decode it.
  - **No invented time estimates.** Counts are exact and are never multiplied by a
    throughput number, because no measured one exists. A plausible fabricated duration
    would be believed, which makes it worse than an honest count. A test greps the output
    for `eta` / `estimated time` / `minutes` to keep it that way.

  `--dry-run` also works without the optional dependencies the real run needs — visible in
  the CLI audit, where both dry-runs report OK while the real invocations CLEAN-FAIL on
  missing extras.

- **`benchmarks/audit_cli_surface.py` — a fourth standing audit, covering the CLI.** The
  other three inspect the exported Python API, which is why a CLI printing zero bytes
  passed all of them. Runs a real invocation of every subcommand and classifies it
  OK / MUTE / TRACEBACK / CLEAN-FAIL / SKIP; `--selftest` validates the classifier against
  9 hand-labelled cases plus a live mute canary. It deliberately does **not** test
  `--help`: argparse prints help without touching the logger, so all 7 subcommands
  answered `--help` with rc 0 while 3 crashed on real use.

- **`dyf info <file.dyf>` — describe an index without loading it.** Reports item count,
  dimensionality, leaf/node counts, build params, stored field names, domain, enrichment
  level and provenance stages. Backed entirely by existing `LazyIndex` properties, so it
  stays cheap: **0.09 s wall clock on a 479 MB / 229,243-item index**, including
  interpreter startup. Previously the only way to learn what a `.dyf` contained was to
  write Python.

  `--json` emits a `{"schema_version": 0, ...}` envelope. **The schema is unstable before
  v1** — that is what the version stamp is for. Callers get something parseable now while
  the project stays free to change the shape.

### Changed

- **`--dedup` now works on `index-images` and `index-video`, not just `index-source`.**
  The three commands shared a byte-identical tail — normalize, build a tree with the same
  five parameters, write a `.dyf` with the same settings — copy-pasted three times, along
  with three argparse blocks carrying the same flags. `--dedup` lived in one of those
  copies, so **a capability could not spread**: adjacent video keyframes are the textbook
  case for collapsing duplicates (the table above measures 88.3% on that data) and could
  not use it.

  Now in `_ingest_common.finalize_index`. Dedup belongs in the shared tail specifically
  because it must subset embeddings and stored fields **in lockstep** — subsetting one
  without the other silently mislabels every row, which is not something three callers
  should each get right independently. `add_common_index_args` defines the shared flags
  once so the commands cannot drift apart again, which they already had.

- **Named result types for the tuples whose positions could be silently swapped.** An
  audit of all 109 exports found several multi-value returns where a positional mistake
  would type-check clean. Each also hid a conditional the tuple could not express. All
  three still unpack as their original tuples, so no caller changes.

  - **`embed_with_diagnostics` → `AxisDiagnosticsResult`.** Its 4-tuple had two elements
    of the *same type* (`list[AxisDiagnostic]`) differing only by meaning — before vs
    after restructuring. The early-exit path returned the before-diagnostics in **both**
    slots, so a caller could not distinguish "re-embedding ran and changed nothing" from
    "re-embedding never happened". New `.restructured` says which. The existing test had
    been asserting `before is after` — inferring it from *object identity*.
  - **`agglomerate_tree_leaves` / `louvain_cluster_leaves` → `LeafGroupingResult`.** Two
    identical unnamed 5-tuples, documented as "Same tuple as `agglomerate_tree_leaves`" —
    sameness asserted in prose, now enforced by a shared type. Both returned
    `(None, {}, [], None, tree)` for "fewer than two leaves"; `.ok` and `.n_groups`
    replace decoding four sentinels.
  - **`dedup_for_index` → `DedupForIndexResult`.** Was half-typed: element 3 was already
    a `DedupResult` while the first two stayed positional. New `.bookkeeping_added`
    reports whether the `orig_index`/`dup_members` fields were written — they are omitted
    when nothing collapses, and a caller previously had to probe `stored_fields` for the
    key to find out. Composes `DedupResult` rather than extending it, since
    `near_duplicate_clusters` has no embeddings to report.

  `LeafGroupingResult` and `DedupForIndexResult` deliberately define **no `__len__`**: the
  only candidate meanings are the unpacking arity or a count, and `SearchResult` had just
  demonstrated how a hard-coded arity becomes a plausible wrong number.

- **`DenseSearchIndex.search` now returns a `SearchResult`, like `LazyIndex.search`.** It
  previously returned a bare `(indices, scores)` tuple — a shape a caller cannot
  introspect or extend, and which forced you to already know the arity. The two index
  types are now interchangeable at the call site.

  **Not a breaking change**: `SearchResult` implements `__iter__`/`__getitem__`/`__len__`,
  so `indices, scores = idx.search(...)` and `I, S = idx.search(batch, ...)` keep working
  exactly as documented. `fields` is always `{}` here, since a dense index holds raw
  embeddings and has no stored fields to gather.

  `DenseSearchIndex` had **no tests at all** despite being public, exported, and shown in
  the README — one of the 31 callables `audit_public_api.py` cannot exercise without a
  fixture. Added 11, covering the return shape, tuple-unpacking compatibility, and
  behaviour (a vector's nearest neighbour is itself; scores rank descending; batched
  results match single queries; higher `nprobe` does not reduce recall).

### Fixed

- **The CLI produced no output at all.** `__init__.py` installs a `NullHandler` on the
  package logger — correct for a library — but nothing configured logging for CLI use, so
  every subcommand reporting through `logger` was silently swallowed. `dyf concepts list`
  printed **0 bytes on a graph with 100+ nodes**; `dyf concepts check` printed 0 bytes and
  exited 1. Affected `concepts` (19 logger calls), `index-source` (17), `index-images`
  (21), `index-video` (21), and most of `enrich` (102 across submodules, mixed with 25
  surviving `print` calls, so it appeared to work while its progress reporting was gone).

  Fixed with `_configure_cli_logging()` in `cli.py`, scoped to the `dyf` logger. Not
  `basicConfig`, which configures the *root* logger and turned on INFO for every
  dependency — httpx dumped every HuggingFace request during `concepts build`.

- **`Pipeline` could never see provenance on a `.dyf`, so every `.dyf` stage rebuilt.**
  `_read_provenance` looked for a `_provenance` metadata key; the enrichment stages only
  ever write `_provenance_level_1/2/3`. Verified on a real artifact: a file carrying
  `_provenance_level_1` read back as `None` with stage status `stale (no provenance)`;
  it now reads `fresh`. The rebuild-skipping the module exists for had never engaged for
  this artifact type.

  Now reads `_provenance`, falling back to the highest `_provenance_level_N` — the most
  recent stage to touch the file, so its params hash is the right one to compare against.

  The `.dyf` branch had **no test coverage**: every existing pipeline test hand-writes
  `_provenance` into a `.pkl` fixture. Added 5 tests, one end-to-end on a real `.dyf`.

- **Missing optional dependencies now produce an actionable message, not a traceback.**
  `main()` catches `ImportError` around dispatch and reports
  `pip install 'dyf[<extra>]'`, exiting 3. Before: `dyf index-images` died with a bare
  `ModuleNotFoundError: No module named 'PIL'`, and `dyf index-source` raised a perfectly
  good `Install it with: pip install "dyf[source]"` that was buried under a traceback
  because nothing caught it. Modules with their own message keep it.

- **CLI results now go to stdout, problems to stderr.** `logging.StreamHandler` defaults
  to stderr, but for these subcommands `logger.info` carries the answer, not a
  diagnostic — so `dyf info f.dyf > out.txt` would have written an empty file. Handlers
  are now split at `WARNING`.

### Changed

- **`find_super_connectors` and `BridgeIndex` now derive their bucket resolution from the
  corpus size.** `global_num_bits` and `facet_num_bits` default to `None` instead of 12 and
  10; pass integers to pin them. Behaviour on large corpora is unchanged (the derivation caps
  at 12), but see below for why the old fixed defaults were broken on small ones.

### Fixed

- **`find_super_connectors` was inert below ~8,000 points.** Local centrality is only computed
  inside buckets clearing `min_bucket_size`, but `global_num_bits=12` fixes 4,096 buckets
  regardless of `n` — so on smaller corpora no bucket ever qualified, local centrality was all
  zero, no point could be `high_global AND high_local`, and `indices` was always empty.
  Measured with the old defaults: **0 super connectors at n = 500 / 2,000 / 8,000 / 30,000 on
  isotropic data, and at 500 / 2,000 on clustered data.** The previous fix (below) was verified
  on an 8k SEC corpus that is clustered enough to sit just inside the working regime, which is
  why this survived it.

  New `_derive_num_bits(n, min_bucket_size)` targets a mean bucket occupancy of
  `2 * min_bucket_size`, capped at 12 bits. `BridgeIndex` carried the same two hardcoded
  constants and now defers identically, recording the resolved values on the instance so
  `index.global_num_bits` reads back as the value used.

  Generalises the rule from the bridge-threshold fix: **a default that fixes an absolute
  *resolution* fails across corpus size exactly as an absolute *similarity* fails across
  anisotropy.** Any constant implying a partition count must be derived from `n`.

- **`find_super_connectors` returned nothing on text embeddings.** On 8,000 SEC sections it
  produced `indices=[]` with `global_centrality` and `local_centrality` all zero — a
  documented feature yielding no output at all. Three compounding causes, all now fixed:

  1. `analyze_bridges` defines a bridge as `centroid_similarity < bridge_threshold`, and its
     **0.5 default is an absolute cosine**. Unit-norm text embeddings live in a narrow cone:
     measured on 4k SEC sections the *minimum* centroid similarity was 0.730, so **0.0% fell
     below 0.5 and zero bridges were found** — at every `num_bits` from 4 to 12. (An
     isotropic gaussian is the opposite: 92.3% below, flooding 3,693 of 4,000 points.) The
     global pass now derives its threshold from the corpus's own distribution via a new
     `bridge_percentile=10` argument.
  2. `_compute_local_centrality` had the same absolute-threshold call at the facet level.
  3. Quadrant classification tested `centrality > percentile`, but centrality is a small
     integer count and its percentile often lands *on* the modal value — on SEC the 50th
     percentile of nonzero global centrality equalled the maximum (195), so `>` selected
     nothing even after bridges were being found. Now `>=`.

  After the fix the same 8,000 sections yield **200 super connectors**. Regression tests
  assert bridges are *found* on anisotropic data; the 77 pre-existing `test_rag.py` tests all
  passed against the empty result because they check types and lengths only.

  ⚠ `DensityClassifier.analyze_bridges`'s own 0.5 default is unchanged (it lives in dyf-rs),
  so direct callers still need to pass a corpus-relative `bridge_threshold`.

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

### Testing

- **Behavioural assertions added to every test that could pass on a degenerate result.**
  `benchmarks/audit_test_assertions.py` now reports **28 of 596 shape-only (5%), and zero
  in each of the other weak categories**; its ranked list of detector/selector tests is
  empty. 35 tests were strengthened in total.

  The scanner itself grew three checks it previously lacked — vacuous comparisons
  (`assert n >= 0`, `assert len(x) <= k`), fully-guarded tests (`if result.x:` wrapping
  every assertion, so an empty result skips rather than fails), and empty-only tests — and
  lost a false-positive class of its own, in which equality against a non-literal and bare
  truthiness checks read as shape. ⚠ **Percentages reported before that correction were
  inflated**: measured over the same tree, the old classifier flagged 50 and the new one
  flags 53, with only 34 in common.

  Two tests turned out to be **dead rather than weak** — their assertions had never
  executed. `test_rog_layers_decrease_threshold` guarded its only assertion behind
  `if len(result.layers) > 1:` while `build_rog_ontology` returns exactly one layer on that
  fixture; `test_alternatives_from_different_parents` ended in `pass  # no crash` inside two
  nested conditionals, the only test in the suite asserting nothing at all.

  Two of the sixteen were hiding real defects rather than merely under-asserting — the
  `BridgeIndex` super-connector bug above, and `CatalogSpace._detect_gap` never firing
  (`KNOWN_ISSUES` #7, open). Also fixed three weak-assertion classes the scanner does not
  detect: vacuous comparisons (`assert n_components >= 0`), `if result.x:` guards that let a
  degenerate result skip rather than fail, and one test whose correct answer *is* an empty
  result but which never said so.

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
