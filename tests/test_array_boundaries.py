"""Regression tests for the dyf-rs array-boundary contract (see dyf._arrays).

Born from a real failure: float64 embeddings hit dyf-rs's typed f32 signature,
every DensityClassifier fit threw, and _build_dyf_tree swallowed the errors
into a single-leaf tree — a degenerate all-points-in-one-cluster result that
scored trivially perfect cluster purity downstream.

Contract under test:
  1. ensure_f32 converts float64 (and lists) silently; raises a CLEAR
     TypeError naming the argument on non-convertible input.
  2. build_dyf_tree accepts float64 and produces a real (multi-cluster) tree.
  3. TypeError from the classifier is NEVER swallowed into a leaf — it
     propagates (dtype/signature errors are bugs, not data conditions).
"""

from __future__ import annotations

import numpy as np
import pytest

from dyf import cut_tree_to_labels
from dyf._arrays import ensure_f32
from dyf.dyf_tree import build_dyf_tree


def _blobs(n_blobs=6, per=60, dim=32, seed=0, dtype=np.float64):
    rng = np.random.default_rng(seed)
    centers = rng.standard_normal((n_blobs, dim)) * 6
    X = np.concatenate([c + rng.standard_normal((per, dim)) for c in centers])
    return X.astype(dtype)


def test_ensure_f32_converts_float64_and_lists():
    out = ensure_f32(np.zeros((4, 3), dtype=np.float64))
    assert out.dtype == np.float32 and out.flags["C_CONTIGUOUS"]
    out2 = ensure_f32([[1.0, 2.0], [3.0, 4.0]])
    assert out2.dtype == np.float32 and out2.shape == (2, 2)


def test_ensure_f32_no_copy_when_already_f32():
    x = np.zeros((4, 3), dtype=np.float32)
    assert ensure_f32(x) is x or np.shares_memory(ensure_f32(x), x)


def test_ensure_f32_raises_clearly_on_garbage():
    with pytest.raises(TypeError, match="centroids"):
        ensure_f32([[1, "a"], [2, "b"]], name="centroids")


def test_build_dyf_tree_accepts_float64():
    X = _blobs(dtype=np.float64)
    tree = build_dyf_tree(X, max_depth=3, num_bits=3, min_leaf_size=4)
    assert tree["children"], "root must split on well-separated blobs"
    labels = np.asarray(cut_tree_to_labels(tree, len(X), 12, embeddings=X.astype(np.float32)))
    assert len(np.unique(labels)) > 1, "float64 input must not degenerate to one cluster"


def test_classifier_typeerror_propagates(monkeypatch):
    """A TypeError inside the fit must NOT be swallowed into a leaf."""
    import dyf_rs

    class Boom:
        def __init__(self, *a, **k):
            pass

        def fit_raw_pca(self, X):
            raise TypeError("simulated dtype/signature mismatch")

        fit = fit_itq = fit_raw_pca

    monkeypatch.setattr(dyf_rs, "DensityClassifier", Boom)
    X = _blobs(dtype=np.float32)
    with pytest.raises(TypeError, match="simulated"):
        build_dyf_tree(X, max_depth=3, num_bits=3, min_leaf_size=4)
