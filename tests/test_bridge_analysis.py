"""Tests for BridgeAnalysis (Rust core)."""

import numpy as np
import pytest

from dyf import DensityClassifier, BridgeAnalysis, check_rust_available


@pytest.fixture
def sample_embeddings():
    """Generate sample normalized embeddings."""
    rng = np.random.default_rng(42)
    embeddings = rng.standard_normal((500, 64)).astype(np.float32)
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    return embeddings


@pytest.fixture
def clustered_embeddings():
    """Generate embeddings with clear cluster structure and bridges."""
    rng = np.random.default_rng(42)
    n_per_cluster = 80
    n_bridges = 20
    dim = 64

    clusters = []
    for i in range(5):
        center = np.zeros(dim, dtype=np.float32)
        center[i * 10:(i + 1) * 10] = 1.0
        center = center / np.linalg.norm(center)

        noise = rng.standard_normal((n_per_cluster, dim)).astype(np.float32) * 0.1
        cluster = center + noise
        cluster = cluster / np.linalg.norm(cluster, axis=1, keepdims=True)
        clusters.append(cluster)

    # Add bridge points between clusters
    for _ in range(n_bridges):
        c1, c2 = rng.choice(5, 2, replace=False)
        center1 = np.zeros(dim, dtype=np.float32)
        center1[c1 * 10:(c1 + 1) * 10] = 1.0
        center2 = np.zeros(dim, dtype=np.float32)
        center2[c2 * 10:(c2 + 1) * 10] = 1.0
        # Bridge point is between two cluster centers
        bridge = 0.5 * center1 + 0.5 * center2
        bridge = bridge / np.linalg.norm(bridge)
        bridge = bridge + rng.standard_normal(dim).astype(np.float32) * 0.05
        bridge = bridge / np.linalg.norm(bridge)
        clusters.append(bridge.reshape(1, -1))

    return np.vstack(clusters)


@pytest.fixture
def fitted_classifier(sample_embeddings):
    """Return a fitted classifier."""
    classifier = DensityClassifier(embedding_dim=64, num_bits=8)
    classifier.fit(sample_embeddings)
    return classifier


@pytest.mark.skipif(not check_rust_available(), reason="Rust extension not available")
class TestBridgeAnalysis:

    def test_analyze_bridges(self, fitted_classifier, sample_embeddings):
        """Test bridge analysis creation."""
        ba = fitted_classifier.analyze_bridges(sample_embeddings)
        assert ba is not None
        assert isinstance(ba.n_buckets, int)
        assert ba.n_buckets > 0

    def test_bridge_threshold(self, fitted_classifier, sample_embeddings):
        """Test bridge threshold property."""
        ba = fitted_classifier.analyze_bridges(sample_embeddings)
        assert isinstance(ba.bridge_threshold, float)
        assert 0.0 <= ba.bridge_threshold <= 1.0

    def test_bridge_indices(self, fitted_classifier, sample_embeddings):
        """Test bridge indices property."""
        ba = fitted_classifier.analyze_bridges(sample_embeddings)
        indices = ba.bridge_indices

        assert isinstance(indices, list)
        # All indices should be valid
        for idx in indices:
            assert 0 <= idx < 500

    def test_bucket_ids(self, fitted_classifier, sample_embeddings):
        """Test bucket IDs property (returns unique bucket IDs)."""
        ba = fitted_classifier.analyze_bridges(sample_embeddings)
        bucket_ids = ba.bucket_ids

        assert isinstance(bucket_ids, list)
        # bucket_ids returns the unique bucket IDs, not per-point
        assert len(bucket_ids) == ba.n_buckets

    def test_adjacency_matrix(self, fitted_classifier, sample_embeddings):
        """Test adjacency matrix property."""
        ba = fitted_classifier.analyze_bridges(sample_embeddings)
        adj = ba.adjacency_matrix

        assert isinstance(adj, list)
        # Should be a flattened n_buckets x n_buckets matrix
        n = ba.n_buckets
        assert len(adj) == n * n

    def test_top_connected_pairs(self, fitted_classifier, sample_embeddings):
        """Test top connected pairs method."""
        ba = fitted_classifier.analyze_bridges(sample_embeddings)
        pairs = ba.top_connected_pairs(5)

        assert isinstance(pairs, list)
        assert len(pairs) <= 5
        # Each pair should be (bucket1, bucket2, count)
        for pair in pairs:
            assert len(pair) == 3
            bucket1, bucket2, count = pair
            assert isinstance(bucket1, (int, np.integer))
            assert isinstance(bucket2, (int, np.integer))
            assert isinstance(count, (int, np.integer))

    def test_bridges_between(self, fitted_classifier, sample_embeddings):
        """Test bridges between specific buckets."""
        ba = fitted_classifier.analyze_bridges(sample_embeddings)
        bucket_ids = ba.bucket_ids
        unique_buckets = list(set(bucket_ids))

        if len(unique_buckets) >= 2:
            b1, b2 = unique_buckets[0], unique_buckets[1]
            bridges = ba.bridges_between(b1, b2)

            assert isinstance(bridges, list)
            # All returned indices should be valid
            for idx in bridges:
                assert 0 <= idx < 500

    def test_get_bridge_connections(self, fitted_classifier, sample_embeddings):
        """Test get bridge connections method."""
        ba = fitted_classifier.analyze_bridges(sample_embeddings)

        # get_bridge_connections requires a bridge_idx argument
        bridge_indices = ba.bridge_indices
        if len(bridge_indices) > 0:
            connections = ba.get_bridge_connections(bridge_indices[0])
            assert isinstance(connections, (list, tuple))

    def test_clustered_data_analysis(self, clustered_embeddings):
        """Test that clustered data can be analyzed for bridges."""
        classifier = DensityClassifier(embedding_dim=64, num_bits=8)
        classifier.fit(clustered_embeddings)

        ba = classifier.analyze_bridges(clustered_embeddings)

        # Should have valid structure regardless of bridge count
        assert ba.n_buckets > 0
        assert isinstance(ba.bridge_indices, list)
        assert isinstance(ba.adjacency_matrix, list)

    def test_n_buckets_consistent(self, fitted_classifier, sample_embeddings):
        """Test that n_buckets is consistent with adjacency matrix size."""
        ba = fitted_classifier.analyze_bridges(sample_embeddings)

        n = ba.n_buckets
        adj = ba.adjacency_matrix
        assert len(adj) == n * n


@pytest.mark.skipif(not check_rust_available(), reason="Rust extension not available")
class TestBridgeAnalysisEdgeCases:

    def test_small_dataset(self):
        """Test with very small dataset."""
        rng = np.random.default_rng(42)
        embeddings = rng.standard_normal((10, 32)).astype(np.float32)
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

        classifier = DensityClassifier(embedding_dim=32, num_bits=4)
        classifier.fit(embeddings)

        ba = classifier.analyze_bridges(embeddings)
        assert ba.n_buckets > 0

    def test_single_cluster(self):
        """Test with data that forms a single cluster."""
        rng = np.random.default_rng(42)
        # Very tight cluster - ensure float32 and contiguous
        base = rng.standard_normal((100, 32)) * 0.01
        base = base + np.ones(32)  # Shift to same region
        norms = np.linalg.norm(base, axis=1, keepdims=True)
        embeddings = (base / norms).astype(np.float32)
        embeddings = np.ascontiguousarray(embeddings)

        classifier = DensityClassifier(embedding_dim=32, num_bits=6)
        classifier.fit(embeddings)

        ba = classifier.analyze_bridges(embeddings)
        # Should have few buckets since all points are similar
        assert ba.n_buckets >= 1
