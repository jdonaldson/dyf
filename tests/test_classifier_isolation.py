"""Isolation scoring in `DensityClassifierFull` (the pure-Python classifier).

`_compute_isolation_scores` was rewritten from a per-item Python loop into one BLAS
matrix product per chunk plus two selections. A numeric hot path changed for speed, so
these tests pin it to a transcription of the original algorithm rather than to a range
check — the scores must be the *same*, not merely plausible.

The subtle part is the median. `np.median` averages the two central order statistics for
an even-length array, and the sample size is 1000. Reading a single element instead
shifts every score by ~1e-3, which is small enough to pass for float noise and is not;
that is why `test_median_matches_numpy_for_even_samples` exists separately.
"""

import numpy as np
import pytest

from dyf import DensityClassifierFull


@pytest.fixture
def embeddings():
    """Clustered, so isolation scores actually vary across items."""
    rng = np.random.default_rng(42)
    centers = rng.standard_normal((8, 32)).astype(np.float32)
    assign = rng.integers(0, 8, size=400)
    X = centers[assign] + 0.4 * rng.standard_normal((400, 32)).astype(np.float32)
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    return np.ascontiguousarray(X.astype(np.float32))


def reference_isolation(emb, seed, k, sample_size):
    """The original implementation, transcribed."""
    n = len(emb)
    rng = np.random.default_rng(seed + 12345)
    sample_indices = rng.choice(n, min(sample_size, n), replace=False)

    out = np.zeros(n, dtype=np.float32)
    for i in range(n):
        sims = emb[i] @ emb[sample_indices].T
        if i in sample_indices:
            pos = np.where(sample_indices == i)[0]
            if len(pos) > 0:
                sims[pos[0]] = -2.0
        sorted_sims = np.sort(sims)[::-1]
        out[i] = sorted_sims[:k].mean() - np.median(sorted_sims)
    return out


class TestIsolationRewrite:
    @pytest.mark.parametrize("sample_size", [64, 101, 400])
    def test_matches_the_original_algorithm(self, embeddings, sample_size):
        """Even and odd sample sizes, and a sample covering the whole corpus."""
        clf = DensityClassifierFull(embedding_dim=32, num_bits=6, seed=31, isolation_sample_size=sample_size)
        clf.fit(embeddings)

        got = clf.get_isolation_scores()
        want = reference_isolation(embeddings, seed=31, k=10, sample_size=sample_size)

        worst = np.abs(got - want).max()
        assert worst < 1e-5, f"worst disagreement {worst:.3e} at sample_size={sample_size}"

    def test_median_matches_numpy_for_even_samples(self, embeddings):
        """np.median averages the two central values when the sample size is even.

        A single-element median passes every range check and every "is it positive"
        assertion, while being wrong by ~1e-3 on every item.
        """
        clf = DensityClassifierFull(embedding_dim=32, num_bits=6, seed=31, isolation_sample_size=100)
        clf.fit(embeddings)

        want = reference_isolation(embeddings, seed=31, k=10, sample_size=100)
        assert np.abs(clf.get_isolation_scores() - want).max() < 1e-5

    def test_crosses_the_chunk_boundary(self):
        """More rows than _ISOLATION_ROW_CHUNK, so block indexing is exercised."""
        rng = np.random.default_rng(3)
        n = DensityClassifierFull._ISOLATION_ROW_CHUNK + 257
        X = rng.standard_normal((n, 16)).astype(np.float32)
        X /= np.linalg.norm(X, axis=1, keepdims=True)
        X = np.ascontiguousarray(X)

        clf = DensityClassifierFull(embedding_dim=16, num_bits=6, seed=31, isolation_sample_size=200)
        clf.fit(X)

        want = reference_isolation(X, seed=31, k=10, sample_size=200)
        worst = np.abs(clf.get_isolation_scores() - want).max()
        assert worst < 1e-5, f"worst disagreement {worst:.3e} across chunk boundary"

    def test_scores_are_not_constant(self, embeddings):
        """A constant vector would satisfy every bound check above."""
        scores = DensityClassifierFull(embedding_dim=32, num_bits=6, seed=31)
        scores.fit(embeddings)
        values = scores.get_isolation_scores()
        assert len(np.unique(values)) > len(values) // 2


class TestStabilityAbsence:
    def test_absent_rather_than_a_perfect_score(self, embeddings):
        """Under 2 seeds stability is unmeasurable; it must not report 1.0.

        1.0 is the maximum, so filling it made a skipped computation indistinguishable
        from a perfectly stable corpus.
        """
        clf = DensityClassifierFull(embedding_dim=32, num_bits=6, seed=31, num_stability_seeds=0)
        clf.fit(embeddings)

        assert clf.get_stability_scores() is None
        assert clf.report().mean_stability_score is None
        assert "not computed" in str(clf.report())

    def test_present_when_seeds_allow_it(self, embeddings):
        clf = DensityClassifierFull(embedding_dim=32, num_bits=6, seed=31, num_stability_seeds=3)
        clf.fit(embeddings)

        scores = clf.get_stability_scores()
        assert scores is not None
        assert len(scores) == len(embeddings)
        assert clf.report().mean_stability_score is not None

    def test_labels_frame_uses_null_not_a_placeholder(self, embeddings):
        clf = DensityClassifierFull(embedding_dim=32, num_bits=6, seed=31, num_stability_seeds=0)
        clf.fit(embeddings)

        df = clf.get_labels()
        assert df["stability_score"].null_count() == len(embeddings)
