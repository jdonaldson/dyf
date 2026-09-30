"""End-to-end Parquet round trip: does an index survive export and re-import intact?

`test_parquet_io.py` covers the units. This checks the whole path at once against the
invariant that actually caught a real bug: **leaf count**. When the round trip first ran on a
50,000-product index it preserved every row and field, ran fast, produced sensible file sizes
— and turned 842 leaves into 19,974, tripling the file, with nothing raised. Rows, fields,
timings and sizes all looked fine. Only a structural comparison across the round trip exposed
it. See `PARQUET_NOTES.md` and the `build_params_inferred` fix.

Data here is synthetic but *clustered*, because isotropic gaussian noise gives the LSH tree
nothing to split on and would produce a degenerate leaf count that proves little. The fixture
yields 68 leaves.

**Verified load-bearing by mutation**: stripping `tree["build_params"]` so the writer falls
back to its defaults — the exact pre-fix behaviour — takes the round trip from 68 -> 68 leaves
to 68 -> 1,794, tripping both the shape and the size-ratio assertion. A test that cannot fail
is not evidence.

Point `DYF_E2E_INDEX` at a real `.dyf` to run the same assertions against production data.
"""

from __future__ import annotations

import os

import numpy as np
import pytest

pytest.importorskip("pyarrow")
pytest.importorskip("pyarrow.parquet")

from dyf import LazyIndex, build_dyf_tree, write_lazy_index  # noqa: E402
from dyf.parquet_io import from_parquet, to_parquet  # noqa: E402

#: Mirrors the parameters a real indexing script used (`index_products.py`), which is what
#: made the recorded-defaults bug visible: its num_bits=2 / min_leaf_size=64 were reported as
#: the library defaults 3 / 4.
BUILD = {"max_depth": 8, "num_bits": 2, "min_leaf_size": 64, "seed": 42}


def _clustered(n=4000, d=64, k=40, seed=3):
    """n points in k gaussian blobs — enough structure for the tree to actually branch."""
    rng = np.random.default_rng(seed)
    centers = rng.normal(scale=4.0, size=(k, d))
    which = rng.integers(0, k, size=n)
    return (centers[which] + rng.normal(scale=1.0, size=(n, d))).astype(np.float32)


def _assert_round_trip(tmp_path, emb, fields, quantization):
    """Export, re-import, and check the round trip preserved structure and content."""
    src = tmp_path / "src.dyf"
    tree = build_dyf_tree(emb, **BUILD)
    write_lazy_index(tree, emb, str(src), quantization=quantization, stored_fields=fields)

    with LazyIndex(str(src)) as idx:
        src_leaves = idx.num_leaves
        src_items = idx.total_items
    src_bytes = os.path.getsize(src)
    assert src_leaves > 1, "fixture must produce a branching tree or the shape check is vacuous"

    parquet = tmp_path / "x.parquet"
    exported = to_parquet(str(src), str(parquet))
    assert exported["rows"] == src_items
    assert exported["leaves"] == src_leaves

    back = tmp_path / "back.dyf"
    imported = from_parquet(str(parquet), str(back), quantization=quantization)

    # 1. shape: the invariant that caught the bug
    assert set(imported["build_params_inherited"]) == set(BUILD)
    assert imported["build_params"] == BUILD
    assert imported["leaves"] == src_leaves, (
        f"round trip reshaped the tree: {src_leaves} leaves -> {imported['leaves']}"
    )

    # 2. size did not balloon (the observed regression was ~3x)
    ratio = imported["bytes_out"] / src_bytes
    assert 0.5 < ratio < 1.5, f"file size changed {ratio:.2f}x across a round trip"

    with LazyIndex(str(back)) as idx:
        got = idx.extract_all_fields()

    # 3. every row present exactly once, and traceable to its source row
    src_idx = np.asarray(got["fields"]["source_item_index"], dtype=np.int64)
    assert sorted(src_idx.tolist()) == list(range(len(emb)))

    # 4. vectors bit-identical at storage precision. extract_all_fields sorts by the NEW
    #    index's item_index, so row i came from original row src_idx[i].
    store = {"float16": np.float16, "float32": np.float32}[quantization]
    expected = emb[src_idx].astype(store).astype(np.float32)
    assert np.abs(expected - got["embeddings"]).max() == 0.0

    # 5. stored fields travelled with their own rows, not merely survived in aggregate.
    #    got["fields"][name][i] belongs to original row src_idx[i], so reordering by src_idx
    #    must reproduce the input exactly.
    order = np.argsort(src_idx)
    for name, values in fields.items():
        got_col = np.asarray(got["fields"][name])[order]
        assert np.array_equal(got_col, np.asarray(values)), f"field {name!r} lost its row alignment"

    return {"leaves": src_leaves, "rows": src_items, "ratio": ratio}


@pytest.mark.parametrize("quantization", ["float16", "float32"])
def test_round_trip_preserves_shape_and_content(tmp_path, quantization):
    n = 4000
    emb = _clustered(n=n)
    fields = {
        "sku": [f"SKU-{i:05d}" for i in range(n)],
        "spend": np.linspace(0.0, 1000.0, n).astype(np.float64),
        "cat": np.arange(n, dtype=np.int32),
    }
    result = _assert_round_trip(tmp_path, emb, fields, quantization)
    assert result["rows"] == n


@pytest.mark.skipif(
    not os.environ.get("DYF_E2E_INDEX"),
    reason="set DYF_E2E_INDEX=/path/to/index.dyf to run the round trip against a real index",
)
def test_round_trip_on_a_real_index(tmp_path):
    """Same assertions, real embeddings. Opt-in: production indexes are not in the repo.

    Reads only the embeddings and a couple of fields from the source index, then rebuilds
    locally with this module's BUILD params — the params recorded in older `.dyf` files
    predate the build-params fix and cannot be trusted.
    """
    source = os.environ["DYF_E2E_INDEX"]
    with LazyIndex(source) as idx:
        data = idx.extract_all_fields()
    emb = data["embeddings"]
    assert emb is not None, "source index has no embeddings"

    # carry at most two fields, whatever they happen to be named
    carried = {}
    for name, values in list(data["fields"].items())[:2]:
        arr = np.asarray(values)
        if arr.dtype.kind in "OU":
            carried[name] = [("" if v is None else str(v)) for v in values]
        elif arr.dtype.kind in "if":
            carried[name] = arr.astype(np.float64 if arr.dtype.kind == "f" else np.int64)

    result = _assert_round_trip(tmp_path, emb, carried, "float16")
    assert result["rows"] == len(emb)
