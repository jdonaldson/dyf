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

## Source

Discovered 2026-04-07 while wiring `experiments/capability_dyf_router.py` in
the turnstyle project. Workaround: rebuilt dyf-rs from local source
(`maturin develop --release`) → 0.7.0, then `uv pip install -e .` to refresh
dyf-py metadata → 0.8.0. Smoke test passed afterwards.

All three issues fixed 2026-04-08.
