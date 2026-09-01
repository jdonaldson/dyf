# Known Issues / TODO

Issues discovered while consuming dyf as a library. None are blocking, but
each cost some time to diagnose.

---

## Cleanup queue (opened 2026-08-31)

Ordered by leverage, not by size. Evidence for each is in the numbered issues below.
Heading this serves: see "Heading" at the top of `CLAUDE.md` — v1 *quality*, i.e. closing the
gap between the shipped surface (109 exports / 72 callables) and the validated one.

**Standing audits** — re-run when touching the public surface; each caught a real defect:
`benchmarks/audit_public_api.py` (41 of 72 callables, 41 OK, canary reproduces issue 5),
`benchmarks/audit_test_assertions.py` (5% shape-only; vacuous / guarded / no-assert all zero;
`--selftest` validates the classifier on 34 hand-labelled cases),
`benchmarks/audit_absolute_thresholds.py` (constants that do not transfer).

⚠ **The audits are necessary but not sufficient.** Issues 6 and 7 were both found by asking
"what does this *test* actually assert?" and then measuring the payload — not by any audit.
`audit_public_api.py` passed `find_super_connectors` throughout, because its 400-point real
fixture happens to sit just inside the working regime. `benchmarks/probe_*.py` hold those
one-off measurements.

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

- [x] `find_super_connectors` / `BridgeIndex` returned nothing below ~8k points because
      `global_num_bits=12` fixes the bucket count regardless of `n` (issue 6). Now derived
      from corpus size
- [ ] `CatalogSpace._detect_gap` never fires — `gap_score` was exactly 0.0 across 16 runs
      (issue 7). Needs a real hierarchical corpus (GUDID) before redesigning; the test now
      pins the current behaviour so a fix is loud
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
- [x] **Result as first measured: 64 of 594 tests (11%) assert shape only; 1 asserts
      nothing at all.** ⚠ That 11% is now known to be inflated — see the scanner
      false-positive correction above; the real figure was ~5%.
      (`test_catalog.py:347 test_alternatives_from_different_parents`). 17 of the 64 guard
      functions that detect or select, where an empty result would pass.
- [x] Strengthened `test_find_super_connectors_basic` with the two assertions that would
      have caught issue 5: `global_centrality.sum() > 0` and `(quadrant != "Regular").any()`
- [x] ⚠ Two false-positive classes fixed in the scanner first, or it would have cried wolf
      on 26 tests: `with pytest.raises(...)` (the context manager *is* the assertion) and
      `np.testing.assert_*` calls (function calls, not `ast.Assert` nodes). 26 → 1 after.
- [x] Added behavioural assertions to the 16 flagged detector/selector tests.
      **the ranked detector list went 16 → 1** (the shape-only percentages quoted at the
      time were measured with the pre-fix classifier and were inflated). Two of the
      16 turned out to be hiding real defects rather than merely under-asserting, which is
      the argument for having done this by hand: issue 6 (`BridgeIndex` super connectors
      always empty) and issue 7 (`_detect_gap` never fires).
- [x] Three further weak-assertion classes the scanner does **not** flag, found by reading
      the tests it did flag. Worth teaching it these:
      1. **vacuous comparisons** — `assert result.n_components >= 0`, satisfied by an empty
         result (`test_mine_dag_chains_returns_chains`)
      2. **guarded blocks** — `if taxonomy.children:` / `if result.chains:` wraps the real
         assertions, so a degenerate result *skips* rather than fails. Fixed by asserting
         the guard condition before the block in four tests.
      3. **asserting the negative by accident** — `test_mine_dag_chains_basic` runs on
         isotropic noise where 0 chains is the *correct* answer; it now says so explicitly
         instead of leaving an empty result indistinguishable from a broken one.
- [x] Taught `audit_test_assertions.py` all three classes, and **corrected a
      false-positive class of its own**: equality against a non-literal
      (`r.labels[z] == z`, `spreads["A"].bucket_distribution == {0: 2, 1: 1}`) read as
      shape, as did a bare truthiness check (`assert taxonomy.children`), which is
      precisely the emptiness assertion this audit exists to prompt. Measured against the
      previous version over the same tree: **50 flagged before, 53 after, only 34 in
      common — 16 false positives and 19 misses.** ⚠ **Every count this scanner reported
      before that fix was inflated**, including the 11% and 8% figures previously written
      here. Corrected: **5% shape-only, and zero in every other weak category.**
- [x] Added `--selftest`: 34 hand-labelled cases, each a real line from `tests/`. Writing
      them caught two gaps in the new rules before they reached the report — a classifier
      announcing "N problems" is worth nothing until shown to separate the cases it claims
      to, the same lesson as `audit_public_api.py`'s canary.
- [x] Closed all 19 newly-found tests. Two were **dead, not weak** — their assertions had
      never executed: `test_rog_layers_decrease_threshold` (ROG returns one layer on its
      fixture, so `if len(layers) > 1:` never opened) and
      `test_alternatives_from_different_parents` (ended in `pass  # no crash` inside two
      nested `if`s — the only test in the suite asserting nothing, while named for a
      property it never checked; measured 15/15 alternatives from a different parent, so
      the name is assertable as written).
- [ ] Consider running the audits in CI as a non-blocking report, so the counts cannot grow

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

## 6. Fixed bucket resolution made `find_super_connectors` inert below ~8k points — FIXED

**Instance C of issue 5, with corpus SIZE as the axis that does not transfer** rather than
anisotropy. Found by probing what the *tests* were actually asserting, not by the audits.

`find_super_connectors` computes local centrality only inside buckets that clear the dense
gate, `count > max(percentile(counts, dense_percentile), min_bucket_size)`. The default
`global_num_bits=12` fixes 4096 buckets **regardless of n**, so below roughly
`min_bucket_size * 2**bits` points no bucket ever qualifies. No dense buckets → local
centrality all zero → no point can be `high_global AND high_local` → `indices` always empty.

Measured (`benchmarks/probe_superconnector_scale.py`), old default 12 bits / `min_bucket_size=20`:

| data | n | largest bucket | dense buckets | super connectors |
|---|---|---|---|---|
| isotropic | 500 | 2 | 0 | **0** |
| isotropic | 2,000 | 5 | 0 | **0** |
| isotropic | 8,000 | 11 | 0 | **0** |
| isotropic | 30,000 | 20 | 0 | **0** |
| clustered | 500 | 4 | 0 | **0** |
| clustered | 2,000 | 18 | 0 | **0** |
| clustered | 8,000 | 113 | 112 | 50 |
| clustered | 30,000 | 965 | 314 | 219 |

So it was inert below ~8k points, and on isotropic data at **every size tested**. This is why
the issue-5 fix looked complete: the 8k SEC corpus it was verified on is clustered enough to
squeak past the gate, and 229k SEC is comfortably past it.

**Fixed in `rag.py`**: `global_num_bits` and `facet_num_bits` now default to `None`, resolved
by a new `_derive_num_bits(n, min_bucket_size)` that targets a mean occupancy of
`2 * min_bucket_size` and caps at 12 — so large corpora keep their previous behaviour while
small ones stop being silently inert. `BridgeIndex` carried the same two hardcoded constants
and now defers the same way, recording the resolved values on the instance. Regression test
`test_bridge_index_derives_num_bits_from_corpus_size` asserts both the new behaviour and that
the old fixed 12/10 still produces zero on the same fixture.

**Rule, generalising issue 5**: a default that fixes an absolute *resolution* fails across
corpus size exactly as an absolute *similarity* fails across anisotropy. Any constant that
implies a count of partitions must be derived from `n`.

⚠ Also fixed a **false positive in `audit_public_api.py`** found while confirming this: its
`CONSTANT` rule ("single distinct value → no discrimination") is correct for a score array and
wrong for an index array, where a one-element selection is a legitimate result. It reported
`find_super_connectors` as CONSTANT when `indices` was `[175]` — one genuine super connector,
with all four quadrant classes populated and 40 nonzero centralities. Selection fields are now
listed in `SELECTION_FIELDS` and judged only on emptiness. The canary still has teeth.

---

## 7. `CatalogSpace._detect_gap` never fires — OPEN

`match_single().gap_detected` was **False and `gap_score` exactly 0.0 in all 16 runs** of a
fixture engineered to contain a gap, plus its control (`benchmarks/probe_catalog_gap.py`,
8 seeds × 2 conditions). The detector requires **five absolute constants simultaneously**:

```
parent_entropy < 0.5  and  child_entropy > 0.7  and  entropy_increase > 0.3
                      and  similarity_drop > 0.1  and  child_sim < 0.8
```

Measured on the engineered hierarchy (`benchmarks/probe_catalog_gap_conditions.py`), per-depth
best similarity is **0.962 / 0.885 / 0.247** — a 0.64 collapse into depth 3, versus 0.02 for a
control whose commodities sit near their parents. So the gap is large and the two conditions
that speak to it (`similarity_drop > 0.1`, `child_sim < 0.8`) both pass. It is blocked by the
entropy terms:

| pair | p_ent<0.5 | c_ent>0.7 | inc>0.3 | drop>0.1 | c_sim<0.8 |
|---|---|---|---|---|---|
| 1→2 | Y (0.00) | n (0.51) | Y (0.51) | n (0.08) | n (0.88) |
| 2→3 | **n (0.5051)** | n (0.66) | n (0.15) | Y (0.64) | Y (0.25) |

The decisive one misses by **0.005** — `parent_entropy` is 0.5051 against a required `< 0.5`.
And the requirement is close to backwards: when a child depth is uniformly *bad* (random
commodities), the best match is poor but entropy stays moderate (0.66), so demanding
`child_entropy > 0.7` rejects the clearest gaps. A conjunction of five absolute thresholds is
issue 5's bug class at its most extreme — each term multiplies the chance of never firing.

**Not fixed, deliberately.** The similarity drop alone separates the two conditions perfectly
here (0.64 vs 0.02), but redesigning the detector against **one engineered fixture** is the
mistake `SPECTRAL_NOTES.md` documents six times over. It needs a real hierarchical corpus as
ground truth — the GUDID energy-devices set (34k records with a real GMDN hierarchy) is the
obvious candidate, since a gap there is checkable by hand.

**Meanwhile** `test_gap_detected_with_engineered_data` asserts the *current* behaviour
(`gap_detected is False`, `gap_score == 0.0`) with a message telling whoever fixes it to
update the test. Previously it asserted only `isinstance(result.gap_detected, bool)`, with a
comment conceding it could not guarantee detection fired — so a feature that never fires at
all passed a test named for it firing.

---

## Source

Discovered 2026-04-07 while wiring `experiments/capability_dyf_router.py` in
the turnstyle project. Workaround: rebuilt dyf-rs from local source
(`maturin develop --release`) → 0.7.0, then `uv pip install -e .` to refresh
dyf-py metadata → 0.8.0. Smoke test passed afterwards.

All three issues fixed 2026-04-08.
