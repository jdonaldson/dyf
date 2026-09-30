"""`.dyf` must record the build params it was actually built with, not this library's defaults.

Regression guard for a real defect found 2026-09-18: `_build_flatbuffer_index` fell back to
`num_bits=3, min_leaf_size=4, seed=42` whenever the caller omitted `build_params=`, and wrote
them as recorded fact. A 50,000-product index built with `num_bits=2, min_leaf_size=64`
reported 3 and 4, so rebuilding from the file's own parameters produced 19,974 leaves instead
of 842 — silently, since nothing raised.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("pyarrow")

from dyf import LazyIndex, build_dyf_tree, write_lazy_index  # noqa: E402


def _emb(n=400, d=12, seed=0):
    return np.random.default_rng(seed).normal(size=(n, d)).astype(np.float32)


def test_build_dyf_tree_records_its_own_params():
    tree = build_dyf_tree(_emb(), max_depth=5, num_bits=2, min_leaf_size=64, seed=11)
    assert tree["build_params"] == {
        "max_depth": 5,
        "num_bits": 2,
        "min_leaf_size": 64,
        "seed": 11,
        "fit_method": "raw_pca",
    }


def test_writer_records_tree_params_without_being_told(tmp_path):
    """The failing case: caller omits build_params= entirely."""
    emb = _emb()
    tree = build_dyf_tree(emb, max_depth=5, num_bits=2, min_leaf_size=64, seed=11)
    path = tmp_path / "i.dyf"
    write_lazy_index(tree, emb, str(path), quantization="float32")

    with LazyIndex(str(path)) as idx:
        bp = idx.tree_summary["build_params"]
        meta = idx._get_metadata()

    assert bp["num_bits"] == 2, "must be the build's value, not the default 3"
    assert bp["min_leaf_size"] == 64, "must be the build's value, not the default 4"
    assert bp["seed"] == 11
    assert bp["max_depth"] == 5
    assert "build_params_inferred" not in meta, "nothing was guessed, so nothing should be flagged"


def test_explicit_build_params_still_win(tmp_path):
    emb = _emb()
    tree = build_dyf_tree(emb, max_depth=5, num_bits=2, min_leaf_size=64, seed=11)
    path = tmp_path / "i.dyf"
    write_lazy_index(tree, emb, str(path), quantization="float32", build_params={"seed": 99})

    with LazyIndex(str(path)) as idx:
        bp = idx.tree_summary["build_params"]
    assert bp["seed"] == 99
    assert bp["num_bits"] == 2, "an explicit override of one key must not reset the others"


def test_guessed_params_are_flagged_in_metadata(tmp_path):
    """A tree with no recorded params still writes a value — but says which ones are guesses."""
    emb = _emb()
    tree = build_dyf_tree(emb, max_depth=4, num_bits=3, min_leaf_size=4, seed=42)
    tree.pop("build_params")  # simulate a tree from an older build path
    path = tmp_path / "i.dyf"
    write_lazy_index(tree, emb, str(path), quantization="float32")

    with LazyIndex(str(path)) as idx:
        meta = idx._get_metadata()
    flagged = set(meta["build_params_inferred"].split(","))
    assert flagged == {"max_depth", "num_bits", "min_leaf_size", "seed"}
