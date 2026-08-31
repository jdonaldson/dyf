# Known Issues / TODO

Issues discovered while consuming dyf as a library. None are blocking, but
each cost some time to diagnose.

---

## Cleanup queue (opened 2026-08-31)

Ordered by leverage, not by size. Evidence for each is in the numbered issues below.

**P0 — incomplete sweep of issue 5, left in the tree by the same commit that documented it**

- [x] `find_super_connectors` global + facet passes — relative threshold (issue 5)
- [x] `_precompute_neighborhoods` (`BridgeIndex.fit`) — now relative
- [x] `select_orthogonal_anchors` — now relative, and the silent fallback logs a warning
- [x] `_find_candidate_bridges` (both single-seed and stability-seed paths) — now relative
- [x] Factored into one `_relative_bridge_threshold()` helper; all six call sites route
      through it, and `percentile=0` reproduces the old absolute floor for comparison
- [x] Verified: `select_orthogonal_anchors(use_bridges=True)` now reports
      `candidate_source='bridges'` instead of falling back to `'all'`
- [ ] ⚠ **but the outcome is unchanged on SEC** — the selected anchors are still identical
      to `use_bridges=False` (60 of 60 overlap). Plausible reason: bridges are by definition
      the points furthest from their bucket centroid, i.e. the extremes that a
      max-orthogonality greedy selects anyway, so the candidate restriction is close to a
      no-op for this selector. Worth deciding whether `use_bridges` earns its existence.
- [ ] `select_orthogonal_anchors(k=12)` returned **60** anchors — unexplained, unrelated to
      the threshold work, noticed while verifying

**P1 — shipped features that do not work**

- [ ] `nprobe="auto"` is a no-op (issue 4). Needs margin quantiles stored at build time
      *and* a probe range wider than 1–5
- [ ] `analyze_bridges`'s own `bridge_threshold=0.5` default — cross-repo, `dyf-core/dyf-rs`
- [ ] `connection_threshold=0.3` in the same function — also absolute, never audited

**P2 — sweep the rest of the pattern (see the rule at the end of issue 5)**

- [ ] `ontology.py` — 0.55, 0.45, `diversity_gap_threshold=0.02` (five sites)
- [ ] `concept_graph.py` (0.2, 0.4), `catalog.py` (0.5), `splits.py` (0.10),
      `cluster_tree.py` (`straddle_threshold=0.15`)
- [ ] `agglomerate.louvain_cluster_leaves(similarity_threshold=0.5)` — measured inert on
      SEC (cell-pair cosine baseline is 0.821, so it filters nothing) and documented as
      NetworkX-fallback-only, so lowest urgency of the class
- [ ] Eight `analyze_bridges` calls in `demo/` — not shipped, cosmetic

**P3 — the test gap that let issue 5 ship (highest leverage item here)**

- [x] Built `benchmarks/audit_test_assertions.py` — AST scan classifying every assertion in
      every `test_*` function as *shape* (isinstance / len / .shape / .dtype / hasattr) or
      *value* (anything constraining content). Flags tests whose assertions are all shape.
      Validated by construction: it flags `test_find_super_connectors_basic`, the exact test
      that passed while issue 5 shipped.
- [x] **Result: 64 of 594 tests (11%) assert shape only; 1 asserts nothing at all**
      (`test_catalog.py:347 test_alternatives_from_different_parents`). 17 of the 64 guard
      functions that detect or select, where an empty result would pass.
- [x] Strengthened `test_find_super_connectors_basic` with the two assertions that would
      have caught issue 5: `global_centrality.sum() > 0` and `(quadrant != "Regular").any()`
- [x] ⚠ Two false-positive classes fixed in the scanner first, or it would have cried wolf
      on 26 tests: `with pytest.raises(...)` (the context manager *is* the assertion) and
      `np.testing.assert_*` calls (function calls, not `ast.Assert` nodes). 26 → 1 after.
- [ ] Add behavioural assertions to the remaining 16 flagged detector/selector tests
- [ ] Consider running the audit in CI as a non-blocking report, so the count cannot grow

**P2 findings — severity downgraded, measured (`benchmarks/audit_absolute_thresholds.py`)**

The `ontology.py` constants are **inert, not destructive** — a materially different problem
from issue 5. Fraction of kNN pairs clearing each threshold:

| corpus | median kNN sim | ≥0.35 | ≥0.45 | ≥0.55 |
|---|---|---|---|---|
| SEC 768d text | 0.855 | 100% | 100% | 100% |
| CMU MoCap 62d | 0.935 | 100% | 100% | 99% |
| isotropic gaussian | 0.310 | 18% | 1% | 0% |

They admit ~everything on both real corpora, so the parameter does not discriminate — but
"admit everything" degrades to "use all neighbours", a benign default, and the builders all
produce sensible output (SEC: 275 chains, 522 taxonomy roots, 2,378 main nodes). Contrast
issue 5, where the same class of constant produced an *empty* result. So P2 is a
**misleading-knob** problem — users think they are tuning something inert — not a bug.
`build_rog_ontology` adapts its cut via `threshold_decay` + `target_coverage` and is
structurally immune; it is the model the others should follow.

⚠ The first version of that probe read `.nodes` / `.chains` off `DAGTaxonomy` and
`UnifiedOntologyResult` — neither attribute exists — and so reported 0 for three functions
that work fine. Caught before it was written up. Reading a nonexistent attribute and
reporting the default is how a probe manufactures a false positive; it is the same mistake
as the `super_connector_indices` typo earlier in this session.

**P4 — hygiene**

- [ ] `LazyIndex.search`'s `nprobe` annotated `int` while accepting `"auto"` and
      `AdaptiveProbeConfig`; type-checkers flag correct calls
- [ ] ~8 pre-existing pyright `Optional` errors in `rag.py`
- [ ] Consider splitting `DEDUP_NOTES.md` out of `SPECTRAL_NOTES.md`, whose CLOSED banner
      undersells the arc's one positive result
- [ ] Query-time dedup expansion in `LazyIndex.search` — deliberately NOT built, since
      expansion is a no-op for distinct-content retrieval. Revisit only if a
      "give me every matching id" use case appears

---

## 1. Editable-install metadata staleness vs. version constraints — FIXED

**Symptom**: After bumping `pyproject.toml` from 0.6.2 → 0.8.0, an existing
editable install kept reporting `importlib.metadata.distribution("dyf").version
== "0.6.2"` even though the *code* being executed was 0.8.0. The dependency
constraint `dyf-rs>=0.7.0` (added in 0.8.0) was therefore not re-evaluated, and
the venv kept running against a stale dyf-rs 0.5.0 wheel.

**Result**: `build_dyf_tree(...)` crashed with
`AttributeError: 'list' object has no attribute 'tolist'` at
`dyf_tree.py:104`, because dyf-rs 0.5.0's `get_bucket_ids()` returns a Python
list (numpy-array bindings landed in dyf-rs 0.6.0, commit `eb2b0a9`).

**Fix applied**: Two-layer defense:
1. `dyf_tree.py` now wraps all `clf.get_bucket_ids()` returns in `np.asarray()`,
   making the code resilient to either return type (list or ndarray).
2. `__init__.py` compares `dyf_rs.__version__` against the documented floor
   (0.7.0) at import time and emits a `RuntimeWarning` if below it — catches
   stale editable installs before they cause subtle failures.

**Remaining caveat**: Bumping `pyproject.toml` version in dyf-py still requires
rerunning `uv pip install -e .` in consumer venvs to refresh metadata. The
warning makes the failure obvious rather than silent.

---

## 2. `cut_tree_to_labels` vs. `cut_dyf_tree_to_labels` API divergence — FIXED

dyf previously exported two cut functions that were silently incompatible:

| Builder           | Tree shape          | Cut function                |
|-------------------|---------------------|-----------------------------|
| `build_pca_tree`  | binary (`left`/`right`) | `cut_tree_to_labels`     |
| `build_dyf_tree`  | n-ary (`children`)  | `cut_dyf_tree_to_labels`    |

Crossing them produced `KeyError: 'left'`. The names were easy to confuse,
and the signatures diverged in surprising ways.

**Fix applied**: Unified dispatcher `dyf.cut_tree_to_labels(tree, n_points,
n_clusters, *, max_depth=None, embeddings=None)` in `src/dyf/cut.py`. Detects
tree shape from its keys (`'children'` → DYF, `'left'` → PCA) and routes to
the correct impl. Raises a clear `ValueError` if the required kwarg for the
detected shape is missing. The old per-module functions are now private
(`_cut_pca_tree_to_labels`, `_cut_dyf_tree_to_labels`). All callsites
(tests + demos) migrated.

---

## 3. dyf-py 0.8.0 source assumes dyf-rs >= 0.6.0 unconditionally — FIXED

`src/dyf/dyf_tree.py:104` called `bucket_ids.tolist()` directly. If a stale
dyf-rs was installed (e.g. 0.5.0), the failure was at the *data* layer, not
the dependency-resolution layer.

**Fix applied**: `bucket_ids = np.asarray(clf.get_bucket_ids())` at all three
call sites (`_build_dyf_tree`, `_try_resplit`, `_resplit_ejected`). Code is now
resilient to either return type at no cost.

---

## 4. `nprobe="auto"` adaptive probing is a no-op — OPEN

**Symptom**: `AdaptiveProbeConfig`'s defaults are miscalibrated, so
`nprobe="auto"` resolves to `max_probes` for nearly every query. It behaves as a
fixed `nprobe≈5` while presenting as adaptive.

**Measured** (`benchmarks/sequence_arc/sec_adaptive_audit.py`, 100k×768 SEC
subset, 400 queries, through the real `LazyIndex.search`):

- Routing margin distribution: median **0.0083**, p10 0.0014, p90 **0.0254** —
  entirely *below* the default `margin_hi=0.1`. **0.0% of queries reach
  `margin_hi`**, so the "confident query → fewer probes" branch never fires.
  57.0% sit at/below `margin_lo=0.01` and get `max_probes`.
- Resolved nprobe distribution: `{2: 2, 3: 3, 4: 54, 5: 341}` — 85% of queries
  get exactly `max_probes=5`.
- Against a fixed-nprobe sweep on the same index, auto lands **ON** the frontier:
  recall 0.5280 at 172 candidates vs an interpolated 0.5258 (**+0.0022**). It is
  indistinguishable from `nprobe=5` (0.5305 @ 176).
- Identical on both `backend="python"` and `backend="rust"` — the kernels agree,
  so this is the shared logic, not a backend divergence.

**Root cause**: the thresholds are **absolute** margins, but `|projection|` scales
with embedding norm and hyperplane normalisation, so no single constant transfers
across corpora. Compounding it, the default range `min_probes=1 … max_probes=5`
spans recall 0.31–0.53 on this corpus, where `nprobe=128` is needed for 0.92 — so
even perfectly calibrated allocation could only move a regime nobody ships.

**Fix not applied** — needs a design decision, two parts:
1. Make thresholds *relative*: compute margin quantiles at build time, store them
   in index metadata, and interpolate on the quantile rather than a raw margin.
2. Widen the probe range, or express it as a multiplier on a caller-supplied base
   nprobe rather than absolute 1–5.

**Meanwhile**: `nprobe="auto"` is safe but pointless; pass an explicit int. Also
note `LazyIndex.search`'s `nprobe` parameter is annotated `int` while the
docstring and `_resolve_nprobe` both accept `"auto"` and `AdaptiveProbeConfig` —
the annotation is stale and type-checkers flag correct calls.

Discovered 2026-08-31 while auditing whether adaptive probing earns its
complexity. An earlier probe (`sec_adaptive_probe.py`) tested margin as a *rank
allocation* signal and also found ~0 effect, but that is a different mechanism
from the shipped one; this audit exercises the real code path.

---

## 5. Absolute cosine/margin thresholds do not transfer across corpora — PARTLY FIXED

**The pattern**, now seen twice: shipped defaults are **absolute** cosine or margin
constants, but embedding anisotropy varies enormously between corpora, so a constant that is
sensible on one is degenerate on another. Both instances were found by auditing, not by tests.

**Instance A — `analyze_bridges` (dyf-rs), and `find_super_connectors` on top of it.**
A bridge is defined as `centroid_similarity < bridge_threshold`, default **0.5**. Measured on
4,000-point samples:

| corpus | min centroid_sim | p10 | median | % below 0.5 | bridges flagged |
|---|---|---|---|---|---|
| SEC 768d, unit-norm text | **0.730** | 0.828 | 0.874 | **0.0%** | **0** |
| CMU MoCap 62d | 0.210 | 0.708 | 0.873 | 0.4% | 14 |
| isotropic gaussian 64d | −0.101 | 0.261 | 0.377 | **92.3%** | **3,693 of 4,000** |

The default therefore flags **nothing or almost everything**, never a useful regime for real
embeddings. Confirmed at every `num_bits` from 4 to 12 on SEC — zero bridges at all
granularities, so it was not a bucket-resolution mistake.

**Consequence**: `find_super_connectors` on 8,000 SEC sections returned
`indices=[]` with `global_centrality` and `local_centrality` **all zero** — a documented
feature ("10x better coverage efficiency than random anchors") producing nothing at all on
text embeddings.

**Fixed in `rag.py`** (three compounding causes, all needed):
1. Global pass now derives the threshold from the corpus's own centroid-similarity
   distribution via a new `bridge_percentile=10` parameter.
2. `_compute_local_centrality` did the same thing at the facet level; same fix.
3. Quadrant classification used `centrality > percentile`, but centrality is a small integer
   count whose percentile often lands *on* the modal value — on SEC the 50th percentile of
   nonzero global centrality **equalled the maximum (195)**, so `>` selected nothing even
   once bridges were being found. Now `>=`.

After the fix, 8,000 SEC sections yield **200 super connectors** and 748 nonzero local
centralities. Regression tests added that assert bridges are *found* on anisotropic data —
⚠ the 77 pre-existing `test_rag.py` tests all passed against the empty result, because they
assert types and array lengths and never that anything was detected.

**Still open**: the `bridge_threshold=0.5` default inside dyf-rs is unchanged, so anyone
calling `DensityClassifier.analyze_bridges(embeddings)` directly still gets zero bridges on
text. Fixing that is a dyf-core change.

**Instance B** is `nprobe="auto"` — see issue 4 above. Same root cause, absolute margin
thresholds, not yet fixed.

**Rule going forward**: a threshold on a similarity, margin, or distance should be expressed
as a percentile of the observed distribution, not as a constant.

---

## Source

Discovered 2026-04-07 while wiring `experiments/capability_dyf_router.py` in
the turnstyle project. Workaround: rebuilt dyf-rs from local source
(`maturin develop --release`) → 0.7.0, then `uv pip install -e .` to refresh
dyf-py metadata → 0.8.0. Smoke test passed afterwards.

All three issues fixed 2026-04-08.
