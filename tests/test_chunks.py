"""Tests for chunk analysis functions."""

import numpy as np
import pytest

from dyf.chunks import (
    DocSpread,
    chunk_redundancy,
    cluster_quality,
    deduplicate_chunks,
    doc_spread,
    neighbor_coherence,
)


@pytest.fixture
def synthetic_data():
    """Synthetic chunked data: 5 docs with varying chunk counts.

    Doc layout (15 points total):
        doc_id  bucket_id  description
        A       0          unique chunk
        A       0          redundant with A[0]
        A       1          different bucket
        B       2          unique chunk
        B       2          redundant with B[0]
        B       2          redundant with B[0]
        C       3          single-chunk doc
        D       0          unique in bucket 0 (no A siblings here? — different doc)
        D       0          redundant with D[0]
        D       4          different bucket
        D       4          redundant with D[2]
        D       4          redundant with D[2]
        E       5          all-same-bucket doc
        E       5          all-same-bucket doc
        E       5          all-same-bucket doc
    """
    bucket_ids = np.array([0, 0, 1, 2, 2, 2, 3, 0, 0, 4, 4, 4, 5, 5, 5])
    doc_ids = np.array(["A", "A", "A", "B", "B", "B", "C", "D", "D", "D", "D", "D", "E", "E", "E"])
    return bucket_ids, doc_ids


class TestChunkRedundancy:

    def test_basic(self, synthetic_data):
        bucket_ids, doc_ids = synthetic_data
        red = chunk_redundancy(bucket_ids, doc_ids)
        assert len(red) == 15

        # Doc A: bucket 0 has 2 A-chunks → each sees 1 sibling; bucket 1 has 1 → 0
        assert red[0] == 1  # A in bucket 0
        assert red[1] == 1  # A in bucket 0
        assert red[2] == 0  # A in bucket 1

        # Doc B: bucket 2 has 3 B-chunks → each sees 2 siblings
        assert red[3] == 2
        assert red[4] == 2
        assert red[5] == 2

        # Doc C: single chunk → 0
        assert red[6] == 0

        # Doc D: bucket 0 has 2 D-chunks → 1; bucket 4 has 3 D-chunks → 2
        assert red[7] == 1
        assert red[8] == 1
        assert red[9] == 2
        assert red[10] == 2
        assert red[11] == 2

        # Doc E: bucket 5 has 3 E-chunks → 2
        assert red[12] == 2
        assert red[13] == 2
        assert red[14] == 2

    def test_length_mismatch(self):
        with pytest.raises(ValueError, match="same length"):
            chunk_redundancy(np.array([1, 2]), np.array([1]))

    def test_empty(self):
        red = chunk_redundancy(np.array([]), np.array([]))
        assert len(red) == 0

    def test_all_unique(self):
        bucket_ids = np.array([0, 1, 2, 3])
        doc_ids = np.array(["a", "b", "c", "d"])
        red = chunk_redundancy(bucket_ids, doc_ids)
        np.testing.assert_array_equal(red, [0, 0, 0, 0])


class TestDeduplicateChunks:

    def test_basic(self, synthetic_data):
        bucket_ids, doc_ids = synthetic_data
        mask = deduplicate_chunks(bucket_ids, doc_ids)
        assert mask.dtype == bool
        assert len(mask) == 15

        # First occurrence of each (bucket, doc) pair should be True
        assert mask[0]   # (0, A) first
        assert not mask[1]  # (0, A) duplicate
        assert mask[2]   # (1, A) first
        assert mask[3]   # (2, B) first
        assert not mask[4]  # (2, B) duplicate
        assert not mask[5]  # (2, B) duplicate
        assert mask[6]   # (3, C) first
        assert mask[7]   # (0, D) first — different doc from A
        assert not mask[8]  # (0, D) duplicate
        assert mask[9]   # (4, D) first
        assert not mask[10] # (4, D) duplicate
        assert not mask[11] # (4, D) duplicate
        assert mask[12]  # (5, E) first
        assert not mask[13] # (5, E) duplicate
        assert not mask[14] # (5, E) duplicate

    def test_reduces_count(self, synthetic_data):
        bucket_ids, doc_ids = synthetic_data
        mask = deduplicate_chunks(bucket_ids, doc_ids)
        # 15 points → should keep fewer
        assert mask.sum() < len(mask)
        # Unique (bucket, doc) pairs: (0,A),(1,A),(2,B),(3,C),(0,D),(4,D),(5,E) = 7
        assert mask.sum() == 7

    def test_preserves_multi_bucket_docs(self, synthetic_data):
        """Multi-topic docs retain one rep per bucket."""
        bucket_ids, doc_ids = synthetic_data
        mask = deduplicate_chunks(bucket_ids, doc_ids)

        # Doc A spans buckets 0 and 1 → keeps 2 reps
        a_mask = mask[doc_ids == "A"]
        assert a_mask.sum() == 2

        # Doc D spans buckets 0 and 4 → keeps 2 reps
        d_mask = mask[doc_ids == "D"]
        assert d_mask.sum() == 2

    def test_collapses_single_bucket_docs(self, synthetic_data):
        """Single-bucket docs collapse to one rep."""
        bucket_ids, doc_ids = synthetic_data
        mask = deduplicate_chunks(bucket_ids, doc_ids)

        # Doc E: all in bucket 5 → keeps 1
        e_mask = mask[doc_ids == "E"]
        assert e_mask.sum() == 1

        # Doc C: single chunk → keeps 1
        c_mask = mask[doc_ids == "C"]
        assert c_mask.sum() == 1

    def test_length_mismatch(self):
        with pytest.raises(ValueError, match="same length"):
            deduplicate_chunks(np.array([1, 2]), np.array([1]))

    def test_empty(self):
        mask = deduplicate_chunks(np.array([]), np.array([]))
        assert len(mask) == 0


class TestDocSpread:

    def test_basic(self, synthetic_data):
        bucket_ids, doc_ids = synthetic_data
        spreads = doc_spread(bucket_ids, doc_ids)

        assert set(spreads.keys()) == {"A", "B", "C", "D", "E"}

        # Doc A: 3 chunks, 2 buckets
        assert spreads["A"].n_chunks == 3
        assert spreads["A"].n_buckets == 2
        assert spreads["A"].concentration == pytest.approx(1.5)

        # Doc B: 3 chunks, 1 bucket
        assert spreads["B"].n_chunks == 3
        assert spreads["B"].n_buckets == 1
        assert spreads["B"].concentration == pytest.approx(3.0)

        # Doc C: 1 chunk, 1 bucket
        assert spreads["C"].n_chunks == 1
        assert spreads["C"].n_buckets == 1
        assert spreads["C"].concentration == pytest.approx(1.0)

        # Doc D: 5 chunks, 2 buckets
        assert spreads["D"].n_chunks == 5
        assert spreads["D"].n_buckets == 2
        assert spreads["D"].concentration == pytest.approx(2.5)

        # Doc E: 3 chunks, 1 bucket
        assert spreads["E"].n_chunks == 3
        assert spreads["E"].n_buckets == 1
        assert spreads["E"].concentration == pytest.approx(3.0)

    def test_bucket_distribution(self, synthetic_data):
        bucket_ids, doc_ids = synthetic_data
        spreads = doc_spread(bucket_ids, doc_ids)

        assert spreads["A"].bucket_distribution == {0: 2, 1: 1}
        assert spreads["B"].bucket_distribution == {2: 3}
        assert spreads["D"].bucket_distribution == {0: 2, 4: 3}

    def test_bridge_detection(self, synthetic_data):
        """Low concentration → bridge document."""
        bucket_ids, doc_ids = synthetic_data
        spreads = doc_spread(bucket_ids, doc_ids)

        # Doc C has concentration 1.0 (lowest possible) — single chunk in single bucket
        # Doc A has 1.5 — moderate spread
        # Doc B has 3.0 — very concentrated
        bridge_docs = {k for k, v in spreads.items() if v.concentration < 2.0}
        focused_docs = {k for k, v in spreads.items() if v.concentration >= 2.0}
        assert "A" in bridge_docs  # 1.5
        assert "C" in bridge_docs  # 1.0
        assert "B" in focused_docs  # 3.0
        assert "E" in focused_docs  # 3.0

    def test_dataclass_fields(self, synthetic_data):
        bucket_ids, doc_ids = synthetic_data
        spreads = doc_spread(bucket_ids, doc_ids)
        s = spreads["A"]
        assert isinstance(s, DocSpread)
        assert isinstance(s.n_chunks, int)
        assert isinstance(s.n_buckets, int)
        assert isinstance(s.concentration, float)
        assert isinstance(s.bucket_distribution, dict)

    def test_length_mismatch(self):
        with pytest.raises(ValueError, match="same length"):
            doc_spread(np.array([1, 2]), np.array([1]))

    def test_empty(self):
        spreads = doc_spread(np.array([]), np.array([]))
        assert spreads == {}

    def test_integer_doc_ids(self):
        """Works with integer doc IDs, not just strings."""
        bucket_ids = np.array([0, 0, 1, 1])
        doc_ids = np.array([100, 100, 200, 200])
        spreads = doc_spread(bucket_ids, doc_ids)
        assert set(spreads.keys()) == {100, 200}
        assert spreads[100].n_chunks == 2
        assert spreads[200].n_chunks == 2


class TestNeighborCoherence:
    """Tests for neighbor_coherence."""

    @pytest.fixture
    def coherent_setup(self):
        """Embeddings where cluster 0 neighbors are highly similar (coherent)
        and cluster 1 neighbors are diverse (low coherence)."""
        rng = np.random.default_rng(42)
        n, d = 20, 16

        # Cluster 0: tight group — neighbors will be very similar
        emb = np.zeros((n, d))
        for i in range(10):
            emb[i] = np.array([3.0] + [0.0] * (d - 1)) + rng.normal(size=d) * 0.05

        # Cluster 1: spread out — neighbors will be diverse
        for i in range(10, 20):
            emb[i] = rng.normal(size=d) * 2.0

        # kNN indices: each point's neighbors are the other points in
        # the same cluster (simple synthetic setup)
        knn = np.zeros((n, 5), dtype=int)
        for i in range(10):
            others = [j for j in range(10) if j != i]
            knn[i] = others[:5]
        for i in range(10, 20):
            others = [j for j in range(10, 20) if j != i]
            knn[i] = others[:5]

        return emb, knn

    def test_returns_correct_length(self, coherent_setup):
        emb, knn = coherent_setup
        coh = neighbor_coherence(emb, knn)
        assert len(coh) == len(emb)

    def test_coherent_cluster_higher(self, coherent_setup):
        emb, knn = coherent_setup
        coh = neighbor_coherence(emb, knn)
        # Tight cluster should have higher coherence
        mean_tight = coh[:10].mean()
        mean_spread = coh[10:].mean()
        assert mean_tight > mean_spread

    def test_sample_k(self, coherent_setup):
        emb, knn = coherent_setup
        coh_full = neighbor_coherence(emb, knn)
        coh_sampled = neighbor_coherence(emb, knn, sample_k=3)
        # Both should be valid float arrays of same length
        assert len(coh_sampled) == len(coh_full)
        assert coh_sampled.dtype == np.float64

    def test_values_in_range(self, coherent_setup):
        emb, knn = coherent_setup
        coh = neighbor_coherence(emb, knn)
        # Cosine similarity based — should be roughly in [-1, 1]
        assert coh.min() >= -1.1
        assert coh.max() <= 1.1


class TestClusterQuality:
    """Tests for cluster_quality."""

    def test_returns_expected_keys(self):
        coherence = np.array([0.1, 0.2, 0.9, 0.85, 0.15, 0.25])
        labels = np.array([0, 0, 1, 1, 2, 2])
        result = cluster_quality(coherence, labels)
        assert 'cluster_mean_coherence' in result
        assert 'meta_clusters' in result
        assert 'threshold' in result

    def test_correct_meta_clusters(self):
        # Cluster 1 has high coherence, should be flagged
        coherence = np.array([0.1, 0.2, 0.9, 0.85, 0.15, 0.25])
        labels = np.array([0, 0, 1, 1, 2, 2])
        result = cluster_quality(coherence, labels, threshold_pct=60)
        # Cluster 1 mean = 0.875, should be above threshold
        assert 1 in result['meta_clusters']

    def test_coherence_array_length(self):
        coherence = np.array([0.5, 0.6, 0.3, 0.4])
        labels = np.array([0, 0, 1, 1])
        result = cluster_quality(coherence, labels)
        assert len(result['cluster_mean_coherence']) == 2

    def test_threshold_is_float(self):
        coherence = np.array([0.5, 0.6, 0.3, 0.4])
        labels = np.array([0, 0, 1, 1])
        result = cluster_quality(coherence, labels)
        assert isinstance(result['threshold'], float)

    def test_meta_clusters_is_set(self):
        coherence = np.array([0.5, 0.6, 0.3, 0.4])
        labels = np.array([0, 0, 1, 1])
        result = cluster_quality(coherence, labels)
        assert isinstance(result['meta_clusters'], set)
