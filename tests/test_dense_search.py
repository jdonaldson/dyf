"""Tests for dyf.dense_search — in-memory search over a dyf tree.

`DenseSearchIndex` is public, exported in `__all__`, and documented in the README with a
code example — and had **no tests at all** before 2026-09-05. It is one of the 31
callables `audit_public_api.py` cannot exercise automatically because it needs a fixture,
which is exactly the gap between shipped and validated surface that the project heading
names.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("dyf_rs")

from dyf import DenseSearchIndex, SearchResult  # noqa: E402


@pytest.fixture(scope="module")
def corpus():
    """Unit-norm vectors, so cosine similarity to self is the maximum."""
    rng = np.random.default_rng(0)
    X = np.ascontiguousarray(rng.standard_normal((512, 32)).astype(np.float32))
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    return X


@pytest.fixture(scope="module")
def index(corpus):
    return DenseSearchIndex(corpus)


class TestReturnShape:
    """The return type must match LazyIndex.search so the two are interchangeable."""

    def test_returns_a_search_result(self, index, corpus):
        assert isinstance(index.search(corpus[0], k=5, nprobe=16), SearchResult)

    def test_matches_lazy_index_return_type(self, index, corpus, tmp_path):
        pytest.importorskip("flatbuffers")
        from dyf import LazyIndex, build_dyf_tree, write_lazy_index

        tree = build_dyf_tree(corpus, max_depth=3, num_bits=3, min_leaf_size=4, seed=42)
        path = str(tmp_path / "x.dyf")
        write_lazy_index(tree, corpus, path)
        with LazyIndex(path) as li:
            lazy_result = li.search(corpus[0], k=5, nprobe=8)

        assert type(lazy_result) is type(index.search(corpus[0], k=5, nprobe=16))

    def test_fields_is_empty_not_missing(self, index, corpus):
        """A dense index has no stored fields; the attribute must still exist."""
        assert index.search(corpus[0], k=5, nprobe=16).fields == {}


class TestBackwardCompatibleUnpacking:
    """The README documents tuple unpacking. Changing the type must not break it."""

    def test_single_query_unpacks(self, index, corpus):
        indices, scores = index.search(corpus[0], k=5, nprobe=16)
        assert indices.shape == (5,)
        assert scores.shape == (5,)

    def test_batched_query_unpacks(self, index, corpus):
        I, S = index.search(corpus[:4], k=5, nprobe=16)
        assert I.shape == (4, 5)
        assert S.shape == (4, 5)

    def test_positional_indexing(self, index, corpus):
        result = index.search(corpus[0], k=5, nprobe=16)
        assert np.array_equal(result[0], result.indices)
        assert np.array_equal(result[1], result.scores)

    def test_len_is_the_hit_count_not_the_unpacking_arity(self, index, corpus):
        """CHANGED 2026-09-05: `__len__` returned a hard-coded 2.

        `len(result)` reported 2 on a k=10 search — a plausible wrong number, the kind
        that gets cited downstream without being questioned. Safe to change because
        unpacking goes through `__iter__`, never `__len__`, which the tests above cover.
        """
        assert len(index.search(corpus[0], k=5, nprobe=16)) == 5
        assert len(index.search(corpus[0], k=10, nprobe=64)) == 10


class TestSearchBehaviour:
    """Behaviour, not shape — these would fail on an index that returned empty results."""

    def test_finds_the_query_itself_first(self, index, corpus):
        result = index.search(corpus[7], k=5, nprobe=64)
        assert int(result.indices[0]) == 7, "a vector's nearest neighbour should be itself"

    def test_scores_are_descending(self, index, corpus):
        scores = index.search(corpus[0], k=10, nprobe=64).scores
        real = scores[np.isfinite(scores)]
        assert np.all(np.diff(real) <= 1e-6), "scores must be ranked"

    def test_self_similarity_is_near_one(self, index, corpus):
        """Unit-norm vectors: cosine with self is 1.0."""
        result = index.search(corpus[3], k=1, nprobe=64)
        assert result.scores[0] == pytest.approx(1.0, abs=1e-3)

    def test_higher_nprobe_does_not_reduce_recall(self, index, corpus):
        """More probing should find the true nearest at least as often."""
        low = index.search(corpus[:32], k=1, nprobe=1).indices[:, 0]
        high = index.search(corpus[:32], k=1, nprobe=256).indices[:, 0]
        truth = np.arange(32)
        assert (high == truth).sum() >= (low == truth).sum()

    def test_batched_matches_single(self, index, corpus):
        """A batch must give the same answer as the queries run one at a time."""
        batch = index.search(corpus[:4], k=5, nprobe=64).indices
        for i in range(4):
            single = index.search(corpus[i], k=5, nprobe=64).indices
            assert np.array_equal(batch[i], single)
