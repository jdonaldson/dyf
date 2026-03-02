"""Tests for dyf.fisher dimension weighting."""

import numpy as np
import pytest

from dyf.fisher import apply_fisher_weights, compute_fisher_weights, extract_fisher_labels


class TestComputeFisherWeights:
    def test_two_clusters_discriminative_dim(self):
        """Dimension that separates two clusters should get highest weight."""
        rng = np.random.RandomState(42)
        n = 200
        d = 10
        # Dim 0 separates clusters; other dims are noise
        emb_a = rng.randn(n, d).astype(np.float32)
        emb_b = rng.randn(n, d).astype(np.float32)
        emb_a[:, 0] += 5.0  # strong separation on dim 0
        emb_b[:, 0] -= 5.0

        embeddings = np.vstack([emb_a, emb_b])
        labels = np.array(["A"] * n + ["B"] * n)

        weights = compute_fisher_weights(embeddings, labels, min_count=10)
        assert weights.shape == (d,)
        assert weights.dtype == np.float32
        # Dim 0 should have the highest weight
        assert np.argmax(weights) == 0

    def test_single_class_returns_uniform(self):
        """With only one class, weights should be uniform."""
        embeddings = np.random.randn(100, 8).astype(np.float32)
        labels = np.array(["same"] * 100)

        weights = compute_fisher_weights(embeddings, labels)
        assert weights.shape == (8,)
        # All values should be equal
        assert np.allclose(weights, weights[0])

    def test_min_count_filtering(self):
        """Classes below min_count should be excluded."""
        rng = np.random.RandomState(0)
        n_big = 100
        n_small = 5
        d = 4
        embeddings = rng.randn(n_big * 2 + n_small, d).astype(np.float32)
        labels = np.array(["A"] * n_big + ["B"] * n_big + ["C"] * n_small)

        # With high min_count, class C is excluded but A and B still survive
        weights = compute_fisher_weights(embeddings, labels, min_count=50)
        assert weights.shape == (d,)
        # Should not be uniform (A and B exist)
        assert not np.allclose(weights, weights[0])

    def test_min_count_too_high_returns_uniform(self):
        """If min_count excludes all but one class, return uniform."""
        embeddings = np.random.randn(100, 4).astype(np.float32)
        labels = np.array(["A"] * 60 + ["B"] * 40)

        weights = compute_fisher_weights(embeddings, labels, min_count=70)
        # Only class A survives (60 < 70), so uniform
        # Actually B also doesn't survive (40 < 70), so uniform
        assert np.allclose(weights, weights[0])

    def test_unit_norm(self):
        """Weights should be L2-normalized."""
        rng = np.random.RandomState(1)
        embeddings = rng.randn(200, 16).astype(np.float32)
        labels = np.array(["X"] * 100 + ["Y"] * 100)

        weights = compute_fisher_weights(embeddings, labels, min_count=10)
        assert pytest.approx(np.linalg.norm(weights), abs=1e-5) == 1.0


class TestApplyFisherWeights:
    def test_shape_dtype_preserved(self):
        """Output should have same shape and float32 dtype."""
        embeddings = np.random.randn(50, 10).astype(np.float32)
        weights = np.random.randn(10).astype(np.float32)

        result = apply_fisher_weights(embeddings, weights)
        assert result.shape == embeddings.shape
        assert result.dtype == np.float32

    def test_multiplication_correct(self):
        """Should be element-wise multiplication along dim axis."""
        embeddings = np.ones((3, 4), dtype=np.float32)
        weights = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)

        result = apply_fisher_weights(embeddings, weights)
        expected = np.array([[1, 2, 3, 4]] * 3, dtype=np.float32)
        np.testing.assert_array_equal(result, expected)


class TestExtractFisherLabels:
    def test_gmdn_first_term(self):
        """GMDN pattern: 'Forceps, bipolar, reusable' → 'forceps'."""
        raw = ["Forceps, bipolar, reusable", "Catheter, urinary, Foley"]
        labels = extract_fisher_labels(raw, mode="first_term")
        assert labels[0] == "forceps"
        assert labels[1] == "catheter"

    def test_list_input(self):
        """If elements are lists, take first element."""
        raw = [["Forceps, bipolar"], ["Catheter, urinary"]]
        labels = extract_fisher_labels(raw, mode="first_term")
        assert labels[0] == "forceps"
        assert labels[1] == "catheter"

    def test_raw_mode(self):
        """Raw mode should pass through as strings."""
        raw = ["Alpha", "Beta", "Gamma"]
        labels = extract_fisher_labels(raw, mode="raw")
        assert list(labels) == ["Alpha", "Beta", "Gamma"]

    def test_none_values(self):
        """None values should become '_unknown_'."""
        raw = [None, "Forceps, bipolar", None]
        labels = extract_fisher_labels(raw, mode="first_term")
        assert labels[0] == "_unknown_"
        assert labels[1] == "forceps"
        assert labels[2] == "_unknown_"

    def test_empty_list_element(self):
        """Empty list should become '_unknown_'."""
        raw = [[], ["Catheter"]]
        labels = extract_fisher_labels(raw, mode="first_term")
        assert labels[0] == "_unknown_"
        assert labels[1] == "catheter"

    def test_nan_values(self):
        """NaN float values should become '_unknown_'."""
        raw = [float("nan"), "Forceps, bipolar"]
        labels = extract_fisher_labels(raw, mode="first_term")
        assert labels[0] == "_unknown_"
        assert labels[1] == "forceps"
