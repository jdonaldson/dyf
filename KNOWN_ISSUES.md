# Known Issues / TODO

Issues discovered while consuming dyf as a library. None are blocking, but
each cost some time to diagnose.

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

## Source

Discovered 2026-04-07 while wiring `experiments/capability_dyf_router.py` in
the turnstyle project. Workaround: rebuilt dyf-rs from local source
(`maturin develop --release`) → 0.7.0, then `uv pip install -e .` to refresh
dyf-py metadata → 0.8.0. Smoke test passed afterwards.

All three issues fixed 2026-04-08.
