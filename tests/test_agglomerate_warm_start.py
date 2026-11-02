"""agglomerate_tree_leaves(init_centroids=...): warm-started leaf grouping with ids that carry over."""

import os
import tempfile

import numpy as np
import pytest

try:
    import flatbuffers  # noqa: F401
    import pyarrow  # noqa: F401

    _HAS_LAZY_DEPS = True
except ImportError:
    _HAS_LAZY_DEPS = False

pytestmark = pytest.mark.skipif(not _HAS_LAZY_DEPS, reason="pyarrow and flatbuffers required")

DIM = 32


def _blobs(rng, centers, per, noise=0.1):
    X = np.concatenate([c + rng.standard_normal((per, DIM)).astype(np.float32) * noise for c in centers])
    return (X / np.linalg.norm(X, axis=1, keepdims=True)).astype(np.float32)


def _group(embeddings, **kw):
    from dyf import build_dyf_tree
    from dyf.agglomerate import agglomerate_tree_leaves
    from dyf.lazy_index import LazyIndex, write_lazy_index

    tree = build_dyf_tree(embeddings, max_depth=4, num_bits=3, min_leaf_size=4, seed=42)
    fd, path = tempfile.mkstemp(suffix=".dyf")
    os.close(fd)
    try:
        write_lazy_index(tree, embeddings, path, quantization="float32")
        coords = np.zeros((len(embeddings), 2), dtype=np.float32)
        with LazyIndex(path) as idx:
            return agglomerate_tree_leaves(idx, coords, embeddings, **kw)
    finally:
        os.unlink(path)


@pytest.fixture
def data():
    rng = np.random.default_rng(42)
    centers = rng.standard_normal((5, DIM)).astype(np.float32) * 3.0
    return rng, centers, _blobs(rng, centers, 40)


def test_cold_run_returns_centroids_without_gaps(data):
    _, _, X = data
    res = _group(X, n_groups=5)
    k = res.n_groups
    assert res.centroids.shape == (k, DIM)
    assert sorted(np.unique(res.point_labels).tolist()) == list(range(k))
    np.testing.assert_allclose(np.linalg.norm(res.centroids, axis=1), 1.0, atol=1e-5)
    assert len(res.empty_groups) == 0
    assert 0.9 < res.mean_cosine <= 1.0  # tight blobs
    labels, names, label_data, item_leaf_map, tree = res  # still unpacks as the 5-tuple
    assert labels.shape == (len(X),)


def test_default_n_groups_unchanged(data):
    _, _, X = data
    assert 1 < _group(X).n_groups <= 50


def test_own_centroids_reproduce_the_grouping(data):
    _, _, X = data
    cold = _group(X, n_groups=5)
    warm = _group(X, init_centroids=cold.centroids)
    np.testing.assert_array_equal(warm.point_labels, cold.point_labels)
    assert warm.mean_cosine == pytest.approx(cold.mean_cosine, abs=1e-5)


def test_group_id_is_the_row_of_init_centroids(data):
    _, _, X = data
    cold = _group(X, n_groups=5)
    k = len(cold.centroids)
    warm = _group(X, init_centroids=cold.centroids[::-1])
    np.testing.assert_array_equal(warm.point_labels, k - 1 - cold.point_labels)


def test_ids_carry_over_when_rows_are_added(data):
    rng, centers, X = data
    cold = _group(X, n_groups=5)
    X2 = np.concatenate([X, _blobs(rng, centers, 10)])  # more points from the same groups
    warm = _group(X2, init_centroids=cold.centroids)
    np.testing.assert_array_equal(warm.point_labels[: len(X)], cold.point_labels)
    new_truth = np.repeat(np.arange(5), 10)
    old_truth = np.repeat(np.arange(5), 40)
    # each new point joins the group its blob's old points are in
    blob_to_group = {b: np.bincount(cold.point_labels[old_truth == b]).argmax() for b in range(5)}
    np.testing.assert_array_equal(warm.point_labels[len(X) :], [blob_to_group[b] for b in new_truth])


def test_group_with_no_points_keeps_its_row_and_id(data):
    _, _, X = data
    cold = _group(X, n_groups=5)
    # a direction no point is near: orthogonal to every point's span is not available in 32-d with 200 points,
    # so use the negated mean direction, which every tight blob is far from
    far = -X.mean(0)
    far /= np.linalg.norm(far)
    init = np.vstack([cold.centroids, far])
    warm = _group(X, init_centroids=init)
    np.testing.assert_array_equal(warm.empty_groups, [len(init) - 1])
    np.testing.assert_allclose(warm.centroids[-1], far, atol=1e-6)
    np.testing.assert_array_equal(warm.point_labels, cold.point_labels)
    assert "1 groups empty" in warm.summary()


def test_bad_init_centroids_raise(data):
    _, _, X = data
    with pytest.raises(ValueError, match="dimension 16"):
        _group(X, init_centroids=np.ones((5, 16), dtype=np.float32))
    with pytest.raises(ValueError, match="conflicts"):
        _group(X, n_groups=4, init_centroids=np.ones((5, DIM), dtype=np.float32))
    bad = np.ones((5, DIM), dtype=np.float32)
    bad[2, 3] = np.nan
    with pytest.raises(ValueError, match="NaN"):
        _group(X, init_centroids=bad)
    zero = np.ones((5, DIM), dtype=np.float32)
    zero[1] = 0
    with pytest.raises(ValueError, match="zero row"):
        _group(X, init_centroids=zero)
    with pytest.raises(ValueError, match="shape"):
        _group(X, init_centroids=np.ones(DIM, dtype=np.float32))
