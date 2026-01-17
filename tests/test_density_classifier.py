"""Tests for DensityClassifier (Rust core)."""

import numpy as np
import pytest

from dyf import DensityClassifier, DensityReport, check_rust_available


@pytest.fixture
def sample_embeddings():
    """Generate sample normalized embeddings."""
    rng = np.random.default_rng(42)
    embeddings = rng.standard_normal((500, 64)).astype(np.float32)
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    return embeddings


@pytest.fixture
def clustered_embeddings():
    """Generate embeddings with clear cluster structure."""
    rng = np.random.default_rng(42)
    n_per_cluster = 100
    dim = 64

    clusters = []
    for i in range(5):
        # Each cluster has a distinct center
        center = np.zeros(dim, dtype=np.float32)
        center[i * 10:(i + 1) * 10] = 1.0
        center = center / np.linalg.norm(center)

        # Add noise around center
        noise = rng.standard_normal((n_per_cluster, dim)).astype(np.float32) * 0.1
        cluster = center + noise
        cluster = cluster / np.linalg.norm(cluster, axis=1, keepdims=True)
        clusters.append(cluster)

    return np.vstack(clusters)


@pytest.mark.skipif(not check_rust_available(), reason="Rust extension not available")
class TestDensityClassifier:

    def test_init(self):
        """Test classifier initialization."""
        classifier = DensityClassifier(embedding_dim=384)
        assert not classifier.is_fitted()

    def test_init_with_params(self):
        """Test initialization with custom parameters."""
        classifier = DensityClassifier(
            embedding_dim=128,
            num_bits=12,
            seed=123,
        )
        assert not classifier.is_fitted()

    def test_fit(self, sample_embeddings):
        """Test fitting classifier."""
        classifier = DensityClassifier(embedding_dim=64, num_bits=8)
        classifier.fit(sample_embeddings)
        assert classifier.is_fitted()

    def test_fit_modifies_in_place(self, sample_embeddings):
        """Test that fit modifies classifier in place."""
        classifier = DensityClassifier(embedding_dim=64, num_bits=8)
        classifier.fit(sample_embeddings)
        assert classifier.is_fitted()

    def test_report(self, sample_embeddings):
        """Test report generation."""
        classifier = DensityClassifier(embedding_dim=64, num_bits=8)
        classifier.fit(sample_embeddings)
        report = classifier.report()

        assert isinstance(report, DensityReport)
        assert report.corpus_size == 500
        assert report.num_buckets > 0
        assert report.mean_bucket_size > 0

    def test_get_bucket_ids(self, sample_embeddings):
        """Test bucket ID retrieval."""
        classifier = DensityClassifier(embedding_dim=64, num_bits=8)
        classifier.fit(sample_embeddings)

        bucket_ids = classifier.get_bucket_ids()
        assert len(bucket_ids) == 500
        assert all(isinstance(bid, (int, np.integer)) for bid in bucket_ids)

    def test_get_bucket_sizes(self, sample_embeddings):
        """Test bucket size retrieval."""
        classifier = DensityClassifier(embedding_dim=64, num_bits=8)
        classifier.fit(sample_embeddings)

        sizes = classifier.get_bucket_sizes()
        assert len(sizes) == 500
        assert all(s >= 1 for s in sizes)

    def test_get_centroid_similarities(self, sample_embeddings):
        """Test centroid similarity retrieval."""
        classifier = DensityClassifier(embedding_dim=64, num_bits=8)
        classifier.fit(sample_embeddings)

        sims = classifier.get_centroid_similarities()
        assert len(sims) == 500
        # Most values should be in valid range (allow small floating point errors)
        assert sum(-1.1 <= s <= 1.1 for s in sims) == len(sims)

    def test_get_isolation_scores(self, sample_embeddings):
        """Test isolation score retrieval."""
        classifier = DensityClassifier(embedding_dim=64, num_bits=8)
        classifier.fit(sample_embeddings)

        scores = classifier.get_isolation_scores()
        assert len(scores) == 500

    def test_get_stability_scores(self, sample_embeddings):
        """Test stability score retrieval."""
        classifier = DensityClassifier(embedding_dim=64, num_bits=8, num_stability_seeds=3)
        classifier.fit(sample_embeddings)

        scores = classifier.get_stability_scores()
        assert len(scores) == 500
        assert all(0.0 <= s <= 1.0 for s in scores)

    def test_clustered_data_has_larger_buckets(self, clustered_embeddings, sample_embeddings):
        """Test that clustered data produces larger buckets than random data."""
        classifier_clustered = DensityClassifier(embedding_dim=64, num_bits=8)
        classifier_clustered.fit(clustered_embeddings)

        classifier_random = DensityClassifier(embedding_dim=64, num_bits=8)
        classifier_random.fit(sample_embeddings)

        report_clustered = classifier_clustered.report()
        report_random = classifier_random.report()

        # Clustered data should have larger mean bucket size
        assert report_clustered.mean_bucket_size > report_random.mean_bucket_size

    def test_reproducibility(self, sample_embeddings):
        """Test that same seed produces same results."""
        classifier1 = DensityClassifier(embedding_dim=64, num_bits=8, seed=42)
        classifier1.fit(sample_embeddings)

        classifier2 = DensityClassifier(embedding_dim=64, num_bits=8, seed=42)
        classifier2.fit(sample_embeddings)

        assert list(classifier1.get_bucket_ids()) == list(classifier2.get_bucket_ids())

    def test_different_seeds_different_results(self, sample_embeddings):
        """Test that different seeds produce different results."""
        classifier1 = DensityClassifier(embedding_dim=64, num_bits=8, seed=42)
        classifier1.fit(sample_embeddings)

        classifier2 = DensityClassifier(embedding_dim=64, num_bits=8, seed=123)
        classifier2.fit(sample_embeddings)

        # Results should be different (with high probability)
        assert list(classifier1.get_bucket_ids()) != list(classifier2.get_bucket_ids())

    def test_get_bucket_id_single(self, sample_embeddings):
        """Test getting bucket ID for a single index."""
        classifier = DensityClassifier(embedding_dim=64, num_bits=8)
        classifier.fit(sample_embeddings)

        single_id = classifier.get_bucket_id(0)
        all_ids = classifier.get_bucket_ids()

        assert single_id == all_ids[0]


@pytest.mark.skipif(not check_rust_available(), reason="Rust extension not available")
class TestDensityReport:

    def test_report_str(self, sample_embeddings):
        """Test report string representation."""
        classifier = DensityClassifier(embedding_dim=64, num_bits=8)
        classifier.fit(sample_embeddings)
        report = classifier.report()

        report_str = str(report)
        assert "Corpus:" in report_str or "items" in report_str.lower()

    def test_report_attributes(self, sample_embeddings):
        """Test report has expected attributes."""
        classifier = DensityClassifier(embedding_dim=64, num_bits=8)
        classifier.fit(sample_embeddings)
        report = classifier.report()

        assert hasattr(report, 'corpus_size')
        assert hasattr(report, 'num_buckets')
        assert hasattr(report, 'mean_bucket_size')
        assert hasattr(report, 'median_bucket_size')
        assert hasattr(report, 'max_bucket_size')
        assert hasattr(report, 'mean_centroid_similarity')
        assert hasattr(report, 'mean_isolation_score')
