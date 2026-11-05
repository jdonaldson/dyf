"""compute_louvain_hierarchy: the per-community spread (cohesion) and re-cut inflow fields.

Both are read-outs of the Voronoi re-cut (KNOWN_ISSUES #14). Cohesion is the mean cosine
distance of members to their community centroid; inflow is the share of a final community whose
members carried a different Louvain label before the re-cut. Each is checked against an
independent recomputation from the other returned fields, not against a hand-typed value.
"""

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


def _hierarchy(embeddings):
    from dyf import build_dyf_tree
    from dyf.agglomerate import compute_louvain_hierarchy
    from dyf.lazy_index import LazyIndex, write_lazy_index

    tree = build_dyf_tree(embeddings, max_depth=4, num_bits=3, min_leaf_size=4, seed=42)
    fd, path = tempfile.mkstemp(suffix=".dyf")
    os.close(fd)
    try:
        write_lazy_index(tree, embeddings, path, quantization="float32")
        coords = np.zeros((len(embeddings), 2), dtype=np.float32)
        with LazyIndex(path) as idx:
            return compute_louvain_hierarchy(idx, coords, embeddings)
    finally:
        os.unlink(path)


@pytest.fixture(scope="module")
def result():
    rng = np.random.default_rng(42)
    centers = rng.standard_normal((5, DIM)).astype(np.float32) * 3.0
    X = np.concatenate([c + rng.standard_normal((40, DIM)).astype(np.float32) * 0.1 for c in centers])
    X = (X / np.linalg.norm(X, axis=1, keepdims=True)).astype(np.float32)
    h = _hierarchy(X)
    assert h is not None
    return X, h


def test_per_community_fields_cover_every_final_community(result):
    _, h = result
    ids = set(h["unique_community_ids"])
    assert ids == set(h["community_sizes"])
    assert ids == set(h["community_cohesion"])
    assert ids == set(h["community_inflow"])
    assert ids == set(np.unique(h["point_labels"]).tolist())


def test_cohesion_is_the_mean_centroid_distance_of_members(result):
    _, h = result
    labels, dist = h["point_labels"], h["centroid_dist"]
    for cid, value in h["community_cohesion"].items():
        members = labels == cid
        assert members.any()
        assert value == pytest.approx(float(dist[members].mean()), abs=1e-6)
        assert 0.0 <= value <= 2.0


def test_cohesion_is_small_on_tight_blobs(result):
    _, h = result
    # noise 0.1 on centres of norm ~17 is an angular spread well under a degree
    assert max(h["community_cohesion"].values()) < 0.05


def test_inflow_matches_the_louvain_labels_before_the_recut(result):
    X, h = result
    n = len(X)
    # Reconstruct the pre-re-cut label of every point from the leaf-level outputs
    block = np.full(n, -1, dtype=np.int32)
    for leaf_idx, items in h["leaf_item_map"].items():
        block[np.asarray(items, dtype=np.int64)] = h["leaf_to_community"][leaf_idx]
    assert (block >= 0).all(), "every point sits in some leaf"
    final = h["point_labels"]
    moved = block != final
    for cid, value in h["community_inflow"].items():
        members = final == cid
        assert value == pytest.approx(float(moved[members].mean()), abs=1e-9)
        assert 0.0 <= value <= 1.0
    # the inflow shares, weighted by size, sum to the number of points the re-cut moved
    total = sum(h["community_inflow"][c] * h["community_sizes"][c] for c in h["unique_community_ids"])
    assert total == pytest.approx(int(moved.sum()), abs=1e-6)
