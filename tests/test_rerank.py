"""Tests for re-ranking strategies."""

import numpy as np
import pytest

from dyf.rerank import (
    rerank_bridge_boost,
    rerank_bridge_mmr,
    rerank_mmr,
    rerank_standard,
)


@pytest.fixture
def synthetic_retrieval():
    """20 points with known embeddings and cluster assignments.

    Clusters: 4 clusters of 5 points each, well-separated in 8-D.
    """
    rng = np.random.default_rng(42)
    n, d = 20, 8

    # 4 cluster centers
    centers = np.zeros((4, d))
    centers[0, 0] = 3.0
    centers[1, 1] = 3.0
    centers[2, 2] = 3.0
    centers[3, 3] = 3.0

    embeddings = np.zeros((n, d))
    cluster_labels = np.zeros(n, dtype=int)
    for c in range(4):
        for j in range(5):
            idx = c * 5 + j
            embeddings[idx] = centers[c] + rng.normal(size=d) * 0.1
            cluster_labels[idx] = c

    # Normalize
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    emb_normed = embeddings / np.maximum(norms, 1e-10)

    # Bridge scores: points at cluster boundaries get high scores
    bridge_scores = np.zeros(n)
    # Make first point of each cluster a "bridge"
    for c in range(4):
        bridge_scores[c * 5] = 5.0

    # Query: point 0 from cluster 0
    query_idx = 0
    query_emb = emb_normed[query_idx]

    # Candidate pool: all other points
    candidate_indices = np.arange(1, n)

    # Similarities
    sims = emb_normed[candidate_indices] @ query_emb

    return {
        'emb_normed': emb_normed,
        'cluster_labels': cluster_labels,
        'bridge_scores': bridge_scores,
        'query_emb': query_emb,
        'candidate_indices': candidate_indices,
        'sims': sims,
    }


class TestRerankStandard:

    def test_returns_top_k(self, synthetic_retrieval):
        s = synthetic_retrieval
        result = rerank_standard(s['sims'], s['candidate_indices'], top_k=5)
        assert len(result) == 5

    def test_returns_highest_similarity(self, synthetic_retrieval):
        s = synthetic_retrieval
        result = rerank_standard(s['sims'], s['candidate_indices'], top_k=5)
        # All returned should be from cluster 0 (most similar to query)
        result_sims = s['emb_normed'][result] @ s['query_emb']
        # First result should have highest similarity
        all_sims = s['emb_normed'][s['candidate_indices']] @ s['query_emb']
        assert result_sims[0] >= all_sims.max() - 1e-6

    def test_top_k_larger_than_pool(self, synthetic_retrieval):
        s = synthetic_retrieval
        result = rerank_standard(s['sims'], s['candidate_indices'], top_k=100)
        assert len(result) == len(s['candidate_indices'])


class TestRerankMmr:

    def test_returns_top_k(self, synthetic_retrieval):
        s = synthetic_retrieval
        result = rerank_mmr(
            s['query_emb'], s['candidate_indices'], s['emb_normed'],
            top_k=5, lam=0.5)
        assert len(result) == 5

    def test_more_diverse_than_standard(self, synthetic_retrieval):
        s = synthetic_retrieval
        standard = rerank_standard(s['sims'], s['candidate_indices'], top_k=8)
        mmr = rerank_mmr(
            s['query_emb'], s['candidate_indices'], s['emb_normed'],
            top_k=8, lam=0.5)

        std_clusters = set(s['cluster_labels'][standard].tolist())
        mmr_clusters = set(s['cluster_labels'][mmr].tolist())
        # MMR should cover at least as many clusters
        assert len(mmr_clusters) >= len(std_clusters)

    def test_lam_1_equals_standard(self, synthetic_retrieval):
        s = synthetic_retrieval
        standard = rerank_standard(s['sims'], s['candidate_indices'], top_k=5)
        mmr = rerank_mmr(
            s['query_emb'], s['candidate_indices'], s['emb_normed'],
            top_k=5, lam=1.0)
        # With lam=1, MMR should behave like standard ranking
        np.testing.assert_array_equal(standard, mmr)


class TestRerankBridgeBoost:

    def test_returns_top_k(self, synthetic_retrieval):
        s = synthetic_retrieval
        result = rerank_bridge_boost(
            s['sims'], s['candidate_indices'], s['bridge_scores'],
            top_k=5, alpha=0.3)
        assert len(result) == 5

    def test_bridges_get_promoted(self, synthetic_retrieval):
        s = synthetic_retrieval
        standard = rerank_standard(s['sims'], s['candidate_indices'], top_k=10)
        boosted = rerank_bridge_boost(
            s['sims'], s['candidate_indices'], s['bridge_scores'],
            top_k=10, alpha=0.5)

        # Bridge points (score > 0) should appear more in boosted results
        bridge_in_std = sum(1 for i in standard if s['bridge_scores'][i] > 0)
        bridge_in_boost = sum(1 for i in boosted if s['bridge_scores'][i] > 0)
        assert bridge_in_boost >= bridge_in_std

    def test_alpha_zero_equals_standard(self, synthetic_retrieval):
        s = synthetic_retrieval
        standard = rerank_standard(s['sims'], s['candidate_indices'], top_k=5)
        boosted = rerank_bridge_boost(
            s['sims'], s['candidate_indices'], s['bridge_scores'],
            top_k=5, alpha=0.0)
        np.testing.assert_array_equal(standard, boosted)


class TestRerankBridgeMmr:

    def test_returns_top_k(self, synthetic_retrieval):
        s = synthetic_retrieval
        result = rerank_bridge_mmr(
            s['query_emb'], s['candidate_indices'], s['emb_normed'],
            s['bridge_scores'], s['cluster_labels'],
            top_k=5, lam=0.5)
        assert len(result) == 5

    def test_meta_clusters_suppressed(self, synthetic_retrieval):
        s = synthetic_retrieval
        # Without meta suppression
        result_no_meta = rerank_bridge_mmr(
            s['query_emb'], s['candidate_indices'], s['emb_normed'],
            s['bridge_scores'], s['cluster_labels'],
            top_k=8, lam=0.5)

        # With cluster 1 as meta — should reduce diversity credit for it
        result_with_meta = rerank_bridge_mmr(
            s['query_emb'], s['candidate_indices'], s['emb_normed'],
            s['bridge_scores'], s['cluster_labels'],
            top_k=8, lam=0.5, meta_clusters={1})

        # Both should return valid results
        assert len(result_no_meta) == 8
        assert len(result_with_meta) == 8

        # Meta cluster should appear less in meta-aware results
        meta_in_no = sum(1 for i in result_no_meta
                         if s['cluster_labels'][i] == 1)
        meta_in_yes = sum(1 for i in result_with_meta
                          if s['cluster_labels'][i] == 1)
        assert meta_in_yes <= meta_in_no
