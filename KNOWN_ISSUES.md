# Known Issues / TODO

Issues discovered while consuming dyf as a **library** — the Python API, its return
shapes, and the primitives downstream projects import.

> **The other queue.** `AGENT_LEGIBILITY_TODO.md` covers the **CLI and the package's
> self-description**: output, exit codes, error messages, discoverability. The two are
> deliberately separate because they have different blind spots — the three standing
> audits look only at the exported Python API, which is why a CLI printing zero bytes
> passed all of them. Check both.

⚠ This file used to open with "None are blocking". That stopped being true when the
heading became v1 *quality*: a shipped feature that silently returns nothing is exactly
what blocks "can someone depend on this?". The P1 section below is titled *shipped features
that do not work*, which contradicted the opening line for months.

---

## Cleanup queue (opened 2026-08-31)

Ordered by leverage, not by size. Evidence for each is in the numbered issues below.
Heading this serves: see "Heading" at the top of `CLAUDE.md` — v1 *quality*, i.e. closing
the gap between the shipped surface and the validated one. (Export counts are deliberately
not quoted here; they go stale. `dyf.overview()` computes the current one.)

**Standing audits** — re-run when touching the public surface; each caught a real defect:
`benchmarks/audit_public_api.py` (auto-calls what it can; canary reproduces issue 5 — the
callables it *cannot* reach without a fixture are where this class of defect hides),
`benchmarks/audit_test_assertions.py` (shape-only / vacuous / guarded / no-assert;
`--selftest` validates the classifier on hand-labelled cases),
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
- [x] `select_orthogonal_anchors(k=12)` returned **60** anchors — explained and fixed
      2026-10-01: seeds (the 60 super connectors) were prepended and never reduced, so `k`
      was a floor. Now the `k` most spread seeds are kept (farthest-point). It hid in tests
      because the function's own `global_num_bits=12` found **zero** super connectors below
      ~8k points — issue 6's class in two call sites the sweep missed (`select_orthogonal_anchors`,
      `get_kmeans_init`); both now derive the resolution from `n`. Reviewed and left:
      `DensityClassifier.from_texts(num_bits=12)` is a classifier resolution, not the
      dense-bucket gate — a different generator

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
- [x] `DenseSearchIndex` ranks by cosine and the Python surface never says so — Euclidean
      callers got recall@15 = 0.06 with no warning (issue 8). Documented; warns once when
      row norms vary (2026-10-01)
- [ ] `auto_tune_tree_params` caps `max_depth` at 6, so `target_bucket_size` is inert above
      ~1M points — four settings, ~3,700 leaves each (issue 9). Issue 6's class, as a cap
- [x] **`louvain_communities` has no aggregation phase** (issue 10) — fixed (multilevel),
      released as dyf-rs 0.12.0 on 2026-09-30, pinned here, gallery re-rendered and live;
      brain shipped-path ARI 0.26 → 0.60 = incumbent. Open remainder: re-validate the auto-tune
      (#9) against the new optimiser
- [ ] **Every tree fit L2-normalises rows and nothing documents it** (issue 11). In 2-D the tree
      is an angular hash: circles leaf purity 0.525 = chance; a Euclidean-acting tree gets 1.000
      and connected components then recover the rings exactly. Document now; `normalize=False`
      is a mechanism decision for after the bench

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
- [x] **Result: ~5% of tests assert shape only; 1 asserted nothing at all**
      (`test_catalog.py:347 test_alternatives_from_different_parents`), and a subset of
      those guarded functions that detect or select, where an empty result would pass.

      ⚠ **First reported as "64 of 594 (11%)" — that was wrong**, from a scanner with a
      32% false-positive rate (see the correction above). The retracted numbers used to be
      written out here *above* their own retraction, so the wrong figure read first and was
      the one a skimmer carried away. Removed rather than struck through: a retracted
      number left legible is a number that gets re-cited.
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

**Superseded 2026-09-05 — the root cause is gone.** The dispatcher treated the
symptom: two tree shapes existed, so something had to route between them. Auditing
the class rather than the instance turned up **the same bug in two functions nobody
had swept**: `dyf.extract_boundary_persistence` and `dyf.boundary_persistence_scores`
resolved to the *PCA* variants, so the most natural call —

```python
from dyf import build_dyf_tree, extract_boundary_persistence
extract_boundary_persistence(build_dyf_tree(embeddings, max_depth=3))
```

— died with the identical `KeyError: 'left'` this issue is about.

`pca_tree` was then removed entirely: nothing in the package produced a PCA tree
(`write_lazy_index`, `LazyIndex` and the Rust kernel all work on the k-ary DYF tree),
and `dyf_tree` offered a strict superset of its public functions. With one tree shape
there is nothing to dispatch on; `cut.py` keeps only a shape check so a wrong dict
still gets a named error instead of a `KeyError` from deep inside the cut.

⚠ The boundary-persistence tests had **all** been written against the PCA variants, so
the DYF ones — which now own the top-level names — had no coverage at all. Ported to
`tests/test_dyf_tree_boundaries.py` (15 tests) *before* deleting, rather than after.

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

## 8. `DenseSearchIndex` ranks by cosine and does not say so — FIXED (2026-10-01)

**Fix:** the metric is named in the class docstring, `search()` and the README. One
correction to the analysis below: the kernel's `dot_normed` divides by the *row* norm as well
as normalising the query, so the ranking is cosine for any input norms — rows need not be
unit-norm, and nothing is "normalised on the way in" that changes the data. What a non-unit
corpus loses is magnitude, so the constructor now logs one warning when row norms vary by more
than 1 % (`DenseSearchIndex.norm_spread`; unit-norm float32 sits at ~1e-6, PCA coordinates
far above). Tests pin that non-unit rows rank identically to exact cosine and that the warning
fires only when norms vary. No ranking mode was added.

Original report:

The batched kernel L2-normalises the query (`dyf-rs/src/dense_search.rs`, `l2_normalize`) and
scores every candidate with `dot_normed(row, &qn)`, so results are ordered by **cosine
similarity**. Neither `DenseSearchIndex`'s docstring, `dense_search.py`, nor the README section
that introduces it names a metric. `LazyIndex` at least says "(will be L2-normalized)" on its
embeddings argument; the dense path says nothing.

Measured 2026-09-25 (`/Volumes/Models/dyf_brain_1m/rerun_2026-09-25/`): on a 20k synthetic set
with **every leaf probed**, top-10 overlap with exact cosine is 1.000, with exact dot product
0.339, with exact Euclidean 0.587. On 1.29M × 50 PCA rows — Euclidean data, the standard
scRNA-seq input — a caller who assumes dot-product ranking (the exact-Euclidean augmentation
`x → (x, -|x|²/2)`, `q → (q, 1)`) gets **recall@15 = 0.06** against true neighbours, with no
warning and `frac_missing = 0`. Confidently wrong output is the worst failure mode a search index
has.

**Fix:** state the metric in the class docstring, `search()` docstring and README; either warn
when input rows are not unit-norm or document that non-unit input is normalised on the way in.
Deciding whether to *offer* dot-product/Euclidean ranking is a separate question — do not add a
mechanism to close a documentation defect.

## 9. `auto_tune_tree_params(target_bucket_size)` is inert above ~1M points — OPEN

`docs/gallery/_gallery.py::auto_tune_tree_params` caps `max_depth` at 6 with `num_bits=2`, so the
tree can have at most 4⁶ = 4,096 leaves regardless of `n`. At 1.29M cells every leaf holds ~350
points and `min_leaf_size` (5 / 10 / 15 for targets 10 / 20 / 30) never binds:

| target_bucket_size | params | leaves | k | ARI vs scanpy |
|---|---|---|---|---|
| 10 | (2, 6, 5) | 3,846 | 104 | 0.309 |
| 20 | (2, 6, 10) | 3,759 | 106 | 0.296 |
| 30 | (2, 6, 15) | 3,684 | 114 | 0.285 |
| (default) | (3, 4, 20) | 3,691 | 115 | 0.264 |

The April 2026 note "auto target=10 best: +0.045 ARI over default" read leaf-boundary jitter as
the knob working. This is issue 6's bug class — an absolute resolution that does not scale with
`n` — on the other side: a cap instead of a floor. The docstring's validation ("beats defaults on
6-7/9 gallery datasets") was done at n ≤ ~70k, where the cap never engages.

**Fix:** derive the depth from `n / target_bucket_size` without the cap, or make the cap a
function of `n`; re-validate at 1.3M with the leaf-count actually varying. Measured in
`/Volumes/Models/dyf_brain_1m/rerun_2026-09-25/s1_gallery_results.json`.

---

## 10. `louvain_communities` is a single-level Louvain — it cannot merge, so `resolution` barely works — FIXED (dyf-rs 0.12.0, 2026-09-30)

**Fix (2026-09-27, released 2026-09-30 as dyf-rs 0.12.0; dyf pins `>=0.12.0` since `a4f6092`):**
`dyf-core/dyf-core/src/louvain.rs` is now the multilevel algorithm — local moves, collapse
communities to super-nodes (summed weights, intra weight as self-loop so degrees and `m` are
preserved), repeat until a level makes no merge. Public signature unchanged. Two tests added:
γ=0.05 must merge two 0.5-bridged triangles (the crossover is γ=0.154); an 8-triangle chain must
be 8 at γ=1 and 1 at γ=0.01. Planted partition now matches igraph exactly (k=1 at γ≤0.1,
8 blocks / ARI 1.000 at γ=1). Gallery re-rendered and live on dyf.io. Remaining follow-up is
issue 9's re-validation, which this fix changes the baseline for.

Effect with the Python side untouched (`/Volumes/Models/dyf_bench_2026-09-27/`): brain shipped
path ARI **0.264 → 0.603** (incumbent 0.608) at 4.6 s vs 34 s; on identical centroid graphs
at the shipped setting the new optimiser is within ±0.06 ARI of igraph Leiden on all seven
datasets, where the old one returned 3–20× too many communities. The remaining gap to
pynndescent + Leiden is the centroid graph, not the optimiser (systematic NMI deficit; MNIST
0.48 vs 0.87 against a leaf oracle of 0.72).

**Gallery re-rendered against the fix (2026-09-27), old → new at the shipped `resolution=1.0`:**

| page | n | k (true) | NMI | ARI |
|---|---|---|---|---|
| mnist | 70k | 86 → 12 (10) | 0.546 → 0.550 | 0.183 → **0.429** |
| cifar10-clip | 10k | 21 → 10 (10) | 0.678 → 0.690 | 0.537 → **0.601** |
| cmu-mocap | 140k | 56 → 18 (25) | 0.412 → 0.290 | 0.085 → 0.112 |
| twenty-newsgroups | 18.8k | 23 → 10 (20) | 0.497 → 0.472 | 0.350 → 0.301 |
| digits | 1.8k | 9 → 7 (10) | 0.679 → 0.627 | 0.545 → 0.475 |
| synthetic-shapes, olivetti-faces, diagnostic-stack | | unchanged | | |

Not a uniform win at small n: a correct modularity optimiser at γ=1 *under*-resolves small leaf
graphs (the resolution limit), where the old one over-split — and over-splitting happened to
score better against fine-grained labels (20 newsgroups, 25 MoCap trials). On digits, γ=2 through
the shipped `louvain_cluster_leaves` path gives k=11 / NMI 0.744 / ARI 0.630 (the gallery page
now runs this cell), above anything the old code produced; the knob works now. No single
resolution won across the eleven datasets measured (best values 0.5–4), so the default stays
1.0. ⚠ **The gallery prose is stale on mnist / cmu-mocap / digits / twenty-newsgroups /
cifar10-clip** — the "over-partitioning is hierarchy" argument on the MNIST page (81 pure
sub-clusters, a 2,325-sample 98%-sevens cluster) was the broken optimiser; the merge-walk
tables collapse to one row. Original report follows.

`dyf-core/dyf-core/src/louvain.rs` is, per its own doc comment, "Single-level Louvain community
detection", "ported from `boids-wasm/src/louvain.rs`". It runs the local node-move phase (at most
20 passes) and **never aggregates**. Without the coarsening phase, two communities can only merge
one node at a time, and moving a single node across is unfavourable even when merging the two
communities would be — that barrier is exactly what Louvain's aggregation step exists to cross.
Every consumer of `louvain_from_centroids` / `louvain_cluster_leaves` (the gallery, the 1.3M
brain, the GUDID viz, `dyfviz`) has been running on it.

**Unit-level evidence** (planted partition, n=2000, 8 blocks, p_in 0.08 / p_out 0.004, seed 0):

| resolution | dyf_rs `louvain_communities` | igraph Leiden (same graph) |
|---|---|---|
| 0.01 | k=7, ARI 0.64 | k=1 |
| 0.1 | k=7, ARI 0.64 | k=1 |
| 1.0 | **k=11, ARI 0.91** | **k=8, ARI 1.00** |
| 3.0 | k=8, ARI 1.00 | k=8, ARI 1.00 |
| 10.0 | k=239 | k=177 |

It over-splits at the standard resolution and cannot merge below 7 communities at any resolution;
a correct optimiser collapses the graph to one community by γ=0.1.

**At scale** (1.29M-cell brain, leaf-centroid cosine kNN graphs, k=10 as shipped): the graph is
**connected** (1 component, checked) yet resolution 1.0 → 0.01 moves k only 138 → 107 (3.8k
leaves) and 1289 → 974 (66k leaves) — k tracks the graph's degree, not γ. Running igraph Leiden on
the **identical graphs** (`rerun_2026-09-25/s7_*`, `s8_*`):

| tree (leaves) | leaf-majority oracle | dyf Louvain, best k≈35 | igraph Leiden, same graph, k≈35 | pynndescent + Leiden (incumbent) |
|---|---|---|---|---|
| auto10 (3.8k) | ARI 0.72 | 0.36 (k=102) | **0.50** (k=35) | 0.60 (k=34) |
| depth 8 (25k) | 0.78 | 0.15 (k=385) | **0.52** (k=32) | |
| depth 10 (66k) | 0.81 | 0.11 (k=748) | **0.52** (k=37); 0.57 at k=27 | |

Swapping only the optimiser recovers about two-thirds of the ARI gap to the incumbent on the
shipped tree, and ~85% with a deeper tree, at ~15 s total. NMI reaches 0.74 vs the incumbent's
0.82. Leaf-size edge weighting (`cos × sqrt(s_i s_j)`) was tried and does not help — dropped.

**The test that let it ship**: `test_resolution_parameter` asserts `k_high >= k_low`, which an
inert knob passes with equality. `benchmarks/audit_test_assertions.py`'s "vacuous comparison"
class, in the Rust crate where the audit does not look.

**Fix:** add the aggregation phase (collapse communities to super-nodes with summed edge weights,
recurse until no improvement) — a correction to an existing primitive, not a new mechanism. Then
assert on a planted partition that γ=1 recovers the blocks exactly and γ→0 yields k=1. Re-run
`s5`–`s8` afterwards; the gallery numbers and the brain memory note will all move.

---

## 11. Every tree fit L2-normalises rows first, and `build_dyf_tree` does not say so — OPEN

`DensityClassifier::{fit_flat, fit_raw_pca_flat, fit_with_hyperplanes_flat, fit_ensemble_flat,
fit_iterative_flat}` (`dyf-core/src/density_classifier.rs`) all copy the input into a
`normalized` buffer — each row divided by its L2 norm — before PCA, hashing and centroids. So
the tree only ever sees *directions*. `build_dyf_tree`'s docstring says "(n, d) array of
embedding vectors" and nothing about unit norm; the README never mentions it. For text/CLIP
embeddings the assumption is harmless (they live on the sphere anyway). For anything where the
norm carries information it silently throws that information away, and in low dimension it is
catastrophic: in 2-D the tree becomes an *angular hash*.

Measured 2026-09-27 on the gallery's synthetic shapes (`make_circles`, 3,000 points, centred at
the origin): **leaf purity vs ring label = 0.525 — chance for two classes**. The inner and outer
ring are the same set of unit vectors. Translate the identical data by +10 and purity is 0.79.
Emulate a Euclidean tree by appending a large constant coordinate (the row normalisation then
acts as a uniform scaling): purity **1.000** on circles and 0.998 on moons, with 160–700 leaves
where the angular hash produced 31 (most of the 256 possible buckets were empty because the
data had collapsed to one dimension). This is why the shapes page reads NMI 0.000 on circles
— it was never a clustering result, it was the input being erased.

The community stage compounds it but is a separate question, also measured: on the
Euclidean-acting leaves, **connected components of a sparse (k=5) leaf-centroid graph recover
the two circles exactly (NMI 1.000)**, while Louvain at resolution 1 on the same graph gives
0.39–0.42 because modularity chops a ring into arcs of similar size. Moons additionally need a
density-aware bridge cut (the 0.08 noise connects them at k≥5) — which is HDBSCAN's mutual
reachability, and which the leaf sizes could supply. So "Louvain over the leaves should recover
topology" is right about the *leaves* and wrong about the *objective*: connectivity does it,
modularity does not.

Related: #8 (`DenseSearchIndex` cosine, same undocumented assumption on the search side) and
the brain/TMS results, where cosine-vs-Euclidean neighbour agreement is only 59% on PCA-50
scores — the norm is being discarded there too, just less fatally in 50-D.

**A cheap regime detector exists at the Louvain phase**, measured the same day: the bottom
eigenvalues of the normalised Laplacian of the leaf graph (milliseconds at a few hundred
leaves). λ₂ ≲ 1e-3 with a ≥5× multiplicative jump low in the spectrum marks the connectivity
regime — moons λ₂ = 0.00013, λ₃/λ₂ = 10, and spectral bisection at k=2 gives NMI 0.981 / ARI
0.992 where Louvain gets 0.42; circles two exact zeros → 1.000. λ₂ ≈ 0.01–0.03 with ratios ≤ 2
marks the blob regime — digits, MNIST — where Louvain beats spectral (0.714 vs 0.694; 0.656 vs
0.591). The absolute eigengap is the wrong decoder (picks k=14 on moons); the ratio is right.
Same measurement: a Euclidean Gaussian-weighted leaf graph beats the shipped cosine one for
Louvain in the blob regime too (digits 0.714 vs 0.627; MNIST 0.656 vs 0.550).

**Shipped (same day):** the read-out. `leaf_graph_spectrum` / `LeafGraphSpectrum` on
`LeafGroupingResult.spectrum`, computed on the exact edges Louvain ran on, with a warning in the
connectivity regime; `tests/test_leaf_graph_spectrum.py` pins the decoder on graphs whose
structure is known by construction and checks the read-out reaches the caller through a real
index. It reproduces the table above on all five datasets.

**Mechanism test — `normalize=False` is falsified by the bench (same day, stages E/E2 in
`/Volumes/Models/dyf_bench_2026-09-27/`).** Emulating an un-normalised tree with the projective
lift (exactly the geometry a `normalize=False` flag would give, no format change) and running the
shipped pipeline:

| ARI vs labels | as-is | lifted tree, shipped stage | lifted tree + Euclidean stage | shipped tree + Euclidean stage |
|---|---|---|---|---|
| TMS droplet (123 types) | **0.543** | 0.234 | 0.147 | 0.527 |
| TMS FACS (120 types) | **0.618** | 0.173 | 0.065 | 0.570 |
| brain 1.29M | **0.472** | 0.372 | 0.062 | 0.390 |
| MNIST PCA-50 | 0.409 | 0.411 | 0.439 | **0.496** |
| circles | 0.000 | 0.000 | 0.214 | 0.000 |

The un-normalised *tree alone* costs 0.1–0.45 ARI on every PCA-score dataset: on scRNA PCA
scores the row norm is library size and cell-cycle amplitude — a nuisance — and normalising it
away is doing real work. A fully Euclidean mode is worse still, and does not rescue circles
either (modularity chops the rings; only a connectivity objective reached 1.000). A Euclidean
community stage on the shipped tree is a wash. So the shapes are a different regime — low-d raw
geometry where the norm *is* the structure — not evidence that the tree's geometry is wrong.

**Spectrum-shape read-outs, measured (same day):** `intrinsic_dim` (Weyl slope) shipped —
synthetic ring 0.9 / path 0.8 / 2-D grid ~2, brain clusters 1.6–4.5 with the big neuron
clusters 3–4 and the cell-cycle cluster ~2, matching the April covariance-based classes. It is
coarse (±1): column-shuffled nulls read higher-dimensional in 25/35 clusters, not all. A ring
detector (`pairing`: relative gap inside the eigenvalue pairs a cycle graph produces) separates
synthetic ring 0.14 from path 0.48 but is **falsified on real data**: against a within-cluster
column-shuffle null over all 35 brain clusters, the known cell-cycle cluster 13 scored 0.198 vs
null 0.130 ± 0.031 (less ring-like than noise), endothelium 32 likewise, endothelium 24 below
its null but 33rd of 35 in absolute terms, and the null itself spanned 0.08–0.43 — the score
tracks leaf count and graph idiosyncrasy, not cycles. Removed before shipping.
`pairing_null_brain_clusters.py`. ⚠ An earlier draft of this paragraph said "PH confirmed
cluster 13's cycle at z=+12.8" — wrong cluster. The z=+12.82 (real top-1 persistence 14.24 vs
within-cluster-shuffle 6.62 ± 0.59) is the **vascular state-cycle** spanning the endothelium
and pericyte clusters; the E18 re-test found the cell-cycle ring *not* cleanly detectable
because E18 cycling cells span lineages. So the spectrum and PH agree about cluster 13. The
vascular cycle crosses cluster boundaries, which a per-cluster spectrum cannot see by
construction — and a graph 0-Laplacian is the wrong operator for a 1-cycle in any case
(cycles live in the kernel of the Hodge 1-Laplacian on edges, which is what PH tracks across
scale).

**Decision:** document, do not add a flag. `build_dyf_tree` partitions *directions*: state it,
name where that is the right thing (embeddings; PCA scores whose norm is nuisance) and where it
is not (low-d raw features whose norm carries structure — pre-scale, or use a density/connectivity
method). Whether to offer `normalize=False` — or to centre and not normalise for non-embedding inputs —
and whether to auto-switch objective on the flag are mechanism decisions that need the
seven-dataset bench re-run under them; not made here. The gallery's "Metric: cosine" row on the index attributes the
Moons failure to the metric; the cause is one level down, in the tree's input handling.

---

## Source

Discovered 2026-04-07 while wiring `experiments/capability_dyf_router.py` in
the turnstyle project. Workaround: rebuilt dyf-rs from local source
(`maturin develop --release`) → 0.7.0, then `uv pip install -e .` to refresh
dyf-py metadata → 0.8.0. Smoke test passed afterwards.

All three issues fixed 2026-04-08.
