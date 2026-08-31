"""Tests for BridgeIndex RAG module."""

import numpy as np
import pytest

from dyf import (
    BridgeIndex,
    FacetDiverseResult,
    OrthogonalAnchorResult,
    SuperConnectorResult,
    check_rust_available,
    diversify_by_facet,
    find_super_connectors,
    get_kmeans_init,
    select_orthogonal_anchors,
)


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
        center = np.zeros(dim, dtype=np.float32)
        center[i * 10 : (i + 1) * 10] = 1.0
        center = center / np.linalg.norm(center)

        noise = rng.standard_normal((n_per_cluster, dim)).astype(np.float32) * 0.1
        cluster = center + noise
        cluster = cluster / np.linalg.norm(cluster, axis=1, keepdims=True)
        clusters.append(cluster)

    # Add bridge points between clusters
    for _ in range(50):
        c1, c2 = rng.choice(5, 2, replace=False)
        center1 = np.zeros(dim, dtype=np.float32)
        center1[c1 * 10 : (c1 + 1) * 10] = 1.0
        center2 = np.zeros(dim, dtype=np.float32)
        center2[c2 * 10 : (c2 + 1) * 10] = 1.0
        bridge = 0.5 * center1 + 0.5 * center2
        bridge = bridge / np.linalg.norm(bridge)
        bridge = bridge + rng.standard_normal(dim).astype(np.float32) * 0.05
        bridge = bridge / np.linalg.norm(bridge)
        clusters.append(bridge.reshape(1, -1))

    return np.vstack(clusters)


@pytest.mark.skipif(not check_rust_available(), reason="Rust extension not available")
class TestSuperConnectors:
    def test_find_super_connectors_basic(self, sample_embeddings):
        """Test basic super connector finding."""
        result = find_super_connectors(sample_embeddings)

        assert isinstance(result, SuperConnectorResult)
        assert isinstance(result.indices, np.ndarray)
        assert isinstance(result.global_centrality, np.ndarray)
        assert isinstance(result.local_centrality, np.ndarray)
        assert isinstance(result.quadrant, np.ndarray)

        # Lengths should match embeddings
        assert len(result.global_centrality) == len(sample_embeddings)
        assert len(result.local_centrality) == len(sample_embeddings)
        assert len(result.quadrant) == len(sample_embeddings)

    def test_finds_bridges_on_anisotropic_embeddings(self):
        """Regression: this returned ZERO on text-like embeddings.

        `analyze_bridges` defines a bridge as centroid_similarity < bridge_threshold, whose
        default 0.5 is an ABSOLUTE cosine. Real unit-norm text embeddings live in a narrow
        cone — measured on 4k SEC sections, the MINIMUM centroid similarity was 0.730, so
        0.0% fell below 0.5 and `indices` came back empty with all-zero centrality. The older
        tests in this class only assert types and lengths, so they passed throughout.

        Uses deliberately anisotropic data (all vectors in a narrow cone) to reproduce the
        condition rather than the isotropic gaussians the other fixtures use.
        """
        rng = np.random.default_rng(7)
        dim, n = 64, 1200
        axis = np.zeros(dim, dtype=np.float32)
        axis[0] = 1.0
        # tight cone: perpendicular component small relative to the shared axis, so pairwise
        # cosine stays high the way it does for real text embeddings
        centers = axis + 0.04 * rng.standard_normal((6, dim)).astype(np.float32)
        lab = rng.integers(0, len(centers), n)
        X = centers[lab] + 0.02 * rng.standard_normal((n, dim)).astype(np.float32)
        X /= np.linalg.norm(X, axis=1, keepdims=True)

        # The regression condition, asserted as BEHAVIOUR rather than as a fixture property:
        # the old absolute threshold must find (almost) nothing here...
        old = find_super_connectors(X, min_bucket_size=10, bridge_percentile=0)
        # ...while the relative default does find bridges.
        new = find_super_connectors(X, min_bucket_size=10)
        assert new.global_centrality.sum() > old.global_centrality.sum(), (
            "a relative bridge threshold must admit more bridges than an absolute floor"
        )
        assert new.global_centrality.sum() > 0, "no global bridges found on anisotropic data"
        assert new.local_centrality.sum() > 0, "no facet-level bridges found on anisotropic data"
        assert len(new.indices) > 0, "no super connectors found on anisotropic data"

    def test_bridge_percentile_is_relative_and_monotone(self):
        """A larger bridge_percentile admits more bridges, on any corpus."""
        rng = np.random.default_rng(11)
        dim, n = 48, 900
        axis = np.zeros(dim, dtype=np.float32)
        axis[0] = 1.0
        X = axis + 0.3 * rng.standard_normal((n, dim)).astype(np.float32)
        X /= np.linalg.norm(X, axis=1, keepdims=True)

        low = find_super_connectors(X, min_bucket_size=10, bridge_percentile=5)
        high = find_super_connectors(X, min_bucket_size=10, bridge_percentile=40)
        assert (high.global_centrality > 0).sum() >= (low.global_centrality > 0).sum()

    def test_find_super_connectors_quadrants(self, clustered_embeddings):
        """Test that quadrant labels are valid."""
        result = find_super_connectors(clustered_embeddings)

        valid_quadrants = {"Regular", "Minor Bridge", "Cross-Domain", "Domain Specialist", "Super Connector"}
        for q in result.quadrant:
            assert q in valid_quadrants

    def test_super_connector_len(self, sample_embeddings):
        """Test __len__ returns index count."""
        result = find_super_connectors(sample_embeddings)
        assert len(result) == len(result.indices)

    def test_super_connector_summary(self, sample_embeddings):
        """Test summary string generation."""
        result = find_super_connectors(sample_embeddings)
        summary = result.summary()

        assert isinstance(summary, str)
        assert "Super Connector" in summary


@pytest.mark.skipif(not check_rust_available(), reason="Rust extension not available")
class TestOrthogonalAnchors:
    def test_select_orthogonal_anchors_basic(self, sample_embeddings):
        """Test basic orthogonal anchor selection."""
        result = select_orthogonal_anchors(sample_embeddings, k=50)

        assert isinstance(result, OrthogonalAnchorResult)
        assert isinstance(result.indices, np.ndarray)
        assert len(result.indices) <= 50

    def test_select_orthogonal_anchors_with_seeds(self, sample_embeddings):
        """Test selection with explicit seeds."""
        seed_indices = np.array([0, 10, 20])
        result = select_orthogonal_anchors(sample_embeddings, k=30, seed_indices=seed_indices)

        # Seeds should be in result
        for seed in seed_indices:
            assert seed in result.indices

    def test_select_orthogonal_anchors_all_points(self, sample_embeddings):
        """Test selection from all points (not just bridges)."""
        result = select_orthogonal_anchors(sample_embeddings, k=50, use_bridges=False)

        assert result.candidate_source == "all"
        assert len(result.indices) <= 50


@pytest.mark.skipif(not check_rust_available(), reason="Rust extension not available")
class TestBridgeIndex:
    def test_bridge_index_fit(self, sample_embeddings):
        """Test index fitting."""
        index = BridgeIndex(n_anchors=50, expansion_k=20)
        index.fit(sample_embeddings, verbose=False)

        assert index._fitted
        assert index._anchor_indices is not None
        assert len(index._anchor_indices) <= 50

    def test_bridge_index_query(self, sample_embeddings):
        """Test single query."""
        index = BridgeIndex(n_anchors=50, n_query_anchors=5, expansion_k=20)
        index.fit(sample_embeddings, verbose=False)

        query = sample_embeddings[0]
        indices, scores = index.query(query, k=10)

        assert len(indices) == 10
        assert len(scores) == 10
        # Scores should be sorted descending
        assert all(scores[i] >= scores[i + 1] for i in range(len(scores) - 1))

    def test_bridge_index_query_batch(self, sample_embeddings):
        """Test batch queries."""
        index = BridgeIndex(n_anchors=50, n_query_anchors=5, expansion_k=20)
        index.fit(sample_embeddings, verbose=False)

        queries = sample_embeddings[:5]
        results = index.query_batch(queries, k=10)

        assert len(results) == 5
        for indices, scores in results:
            assert len(indices) == 10
            assert len(scores) == 10

    def test_bridge_index_get_anchors(self, sample_embeddings):
        """Test anchor retrieval."""
        index = BridgeIndex(n_anchors=50)
        index.fit(sample_embeddings, verbose=False)

        anchors = index.get_anchors()
        assert isinstance(anchors, np.ndarray)
        assert len(anchors) <= 50

    def test_bridge_index_get_super_connectors(self, sample_embeddings):
        """Test super connector retrieval."""
        index = BridgeIndex(n_anchors=50)
        index.fit(sample_embeddings, verbose=False)

        sc = index.get_super_connectors()
        assert isinstance(sc, SuperConnectorResult)

    def test_bridge_index_summary(self, sample_embeddings):
        """Test summary generation."""
        index = BridgeIndex(n_anchors=50)
        index.fit(sample_embeddings, verbose=False)

        summary = index.summary()
        assert isinstance(summary, str)
        assert "BridgeIndex" in summary
        assert "Anchors" in summary

    def test_bridge_index_not_fitted_error(self):
        """Test error when querying unfitted index."""
        index = BridgeIndex()

        with pytest.raises(ValueError, match="fit"):
            index.query(np.zeros(64))

        with pytest.raises(ValueError, match="fit"):
            index.get_anchors()

    def test_bridge_index_evaluate_recall(self, clustered_embeddings):
        """Test recall evaluation."""
        index = BridgeIndex(n_anchors=100, n_query_anchors=10, expansion_k=50)
        index.fit(clustered_embeddings, verbose=False)

        metrics = index.evaluate_recall(n_queries=20, k=10)

        assert "recall" in metrics
        assert "avg_candidates" in metrics
        assert "speedup" in metrics

        # Recall should be between 0 and 1
        assert 0.0 <= metrics["recall"] <= 1.0

    def test_bridge_index_with_sparse_points(self, clustered_embeddings):
        """Test adding sparse region points."""
        index = BridgeIndex(n_anchors=100, include_sparse_points=20)
        index.fit(clustered_embeddings, verbose=False)

        # Should have anchors
        anchors = index.get_anchors()
        assert len(anchors) > 0


@pytest.mark.skipif(not check_rust_available(), reason="Rust extension not available")
class TestBridgeIndexConfiguration:
    def test_different_anchor_counts(self, sample_embeddings):
        """Test with different anchor counts."""
        for n_anchors in [10, 50, 100]:
            index = BridgeIndex(n_anchors=n_anchors, expansion_k=20)
            index.fit(sample_embeddings, verbose=False)
            assert len(index.get_anchors()) <= n_anchors

    def test_different_expansion_k(self, sample_embeddings):
        """Test with different expansion sizes."""
        for exp_k in [10, 50, 100]:
            index = BridgeIndex(n_anchors=30, expansion_k=exp_k)
            index.fit(sample_embeddings, verbose=False)

            # Query should work
            indices, scores = index.query(sample_embeddings[0], k=5)
            assert len(indices) == 5

    def test_query_anchor_override(self, sample_embeddings):
        """Test overriding query anchors at query time."""
        index = BridgeIndex(n_anchors=50, n_query_anchors=5, expansion_k=20)
        index.fit(sample_embeddings, verbose=False)

        # Default query anchors
        indices1, _ = index.query(sample_embeddings[0], k=10)

        # Override to use more anchors
        indices2, _ = index.query(sample_embeddings[0], k=10, n_query_anchors=20)

        # Both should return valid results
        assert len(indices1) == 10
        assert len(indices2) == 10


@pytest.mark.skipif(not check_rust_available(), reason="Rust extension not available")
class TestFacetDiversification:
    def test_diversify_by_facet_basic(self, clustered_embeddings):
        """Test basic facet diversification."""
        from dyf import DensityClassifier

        # Fit classifier to get bucket IDs
        clf = DensityClassifier(embedding_dim=clustered_embeddings.shape[1], num_bits=8)
        clf.fit(clustered_embeddings)
        bucket_ids = clf.get_bucket_ids()

        # Get candidates (top 100 by similarity)
        query = clustered_embeddings[0]
        sims = query @ clustered_embeddings.T
        candidates = np.argsort(sims)[-100:][::-1]

        # Diversify
        result = diversify_by_facet(query, candidates, clustered_embeddings, bucket_ids, k=10)

        assert isinstance(result, FacetDiverseResult)
        assert len(result) <= 10
        assert len(result.indices) == len(result.bucket_ids)
        assert len(result.indices) == len(result.similarities)

        # Each result should be from a different bucket
        assert len(set(result.bucket_ids)) == len(result.bucket_ids)

    def test_diversify_by_facet_empty_candidates(self, sample_embeddings):
        """Test with empty candidate list."""
        bucket_ids = np.zeros(len(sample_embeddings), dtype=np.int64)
        query = sample_embeddings[0]

        result = diversify_by_facet(query, np.array([]), sample_embeddings, bucket_ids, k=10)

        assert len(result) == 0
        assert result.buckets_covered == 0

    def test_diversify_preserves_relevance(self, clustered_embeddings):
        """Test that diversification preserves high-similarity items."""
        from dyf import DensityClassifier

        clf = DensityClassifier(embedding_dim=clustered_embeddings.shape[1], num_bits=8)
        clf.fit(clustered_embeddings)
        bucket_ids = clf.get_bucket_ids()

        query = clustered_embeddings[0]
        sims = query @ clustered_embeddings.T
        candidates = np.argsort(sims)[-100:][::-1]

        result = diversify_by_facet(query, candidates, clustered_embeddings, bucket_ids, k=10)

        # Similarities should be sorted descending (best first per bucket)
        for i in range(len(result.similarities) - 1):
            # Not strictly sorted because buckets differ, but first should be high
            pass

        # First result should be highly similar
        assert result.similarities[0] > 0.5

    def test_diversify_unique_buckets(self, clustered_embeddings):
        """Test that each result comes from a unique bucket."""
        from dyf import DensityClassifier

        clf = DensityClassifier(embedding_dim=clustered_embeddings.shape[1], num_bits=10)
        clf.fit(clustered_embeddings)
        bucket_ids = clf.get_bucket_ids()

        query = clustered_embeddings[50]
        sims = query @ clustered_embeddings.T
        candidates = np.argsort(sims)[-200:][::-1]

        result = diversify_by_facet(query, candidates, clustered_embeddings, bucket_ids, k=20)

        # All bucket IDs should be unique
        assert len(set(result.bucket_ids)) == result.buckets_covered
        assert result.buckets_covered == len(result)


@pytest.mark.skipif(not check_rust_available(), reason="Rust extension not available")
class TestKmeansInit:
    def test_get_kmeans_init_basic(self, sample_embeddings):
        """Test basic k-means initialization."""
        init = get_kmeans_init(sample_embeddings, nlist=20, verbose=False)

        assert isinstance(init, np.ndarray)
        assert init.shape == (20, sample_embeddings.shape[1])
        assert init.dtype == np.float32

    def test_get_kmeans_init_normalized(self, sample_embeddings):
        """Test that initial centroids are normalized."""
        init = get_kmeans_init(sample_embeddings, nlist=20, verbose=False)

        norms = np.linalg.norm(init, axis=1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-5)

    def test_get_kmeans_init_with_stable_bridges(self, clustered_embeddings):
        """Test k-means init with stable bridge preference."""
        init = get_kmeans_init(
            clustered_embeddings,
            nlist=30,
            use_stable_bridges=True,
            num_stability_seeds=3,
            stability_threshold=2,
            verbose=False,
        )

        assert init.shape[0] == 30

    def test_get_kmeans_init_single_seed(self, sample_embeddings):
        """Test k-means init with single seed (no stability)."""
        init = get_kmeans_init(sample_embeddings, nlist=20, use_stable_bridges=False, verbose=False)

        assert init.shape[0] == 20

    def test_get_kmeans_init_with_sklearn(self, clustered_embeddings):
        """Test that k-means init works with sklearn."""
        from sklearn.cluster import KMeans

        init = get_kmeans_init(clustered_embeddings, nlist=10, verbose=False)

        # Should work as init for sklearn KMeans
        kmeans = KMeans(n_clusters=10, init=init, n_init=1, max_iter=10)
        kmeans.fit(clustered_embeddings)

        assert len(kmeans.labels_) == len(clustered_embeddings)
        assert kmeans.cluster_centers_.shape == (10, clustered_embeddings.shape[1])

    def test_get_kmeans_init_large_nlist(self, sample_embeddings):
        """Test with nlist larger than available bridges."""
        # Request more centroids than likely bridges
        init = get_kmeans_init(sample_embeddings, nlist=200, verbose=False)

        # Should still return requested number (with padding)
        assert init.shape[0] == 200


# Import DAG mining functions
from dyf import (
    DAGChain,
    DAGMiningResult,
    compute_neighbor_diversity,
    mine_dag_chains,
)


class TestNeighborDiversity:
    def test_compute_neighbor_diversity_basic(self, sample_embeddings):
        """Test basic diversity computation."""
        diversity = compute_neighbor_diversity(sample_embeddings, k=10)

        assert isinstance(diversity, np.ndarray)
        assert len(diversity) == len(sample_embeddings)
        assert diversity.dtype == np.float64 or diversity.dtype == np.float32

    def test_compute_neighbor_diversity_range(self, sample_embeddings):
        """Test that diversity values are in valid range."""
        diversity = compute_neighbor_diversity(sample_embeddings, k=10)

        # Diversity should be between 0 and 1 (since 1 - coherence)
        assert np.all(diversity >= 0)
        assert np.all(diversity <= 1)

    def test_compute_neighbor_diversity_clustered(self, clustered_embeddings):
        """Test diversity on clustered data."""
        diversity = compute_neighbor_diversity(clustered_embeddings, k=10)

        # Points in tight clusters should have lower diversity
        # Points at boundaries should have higher diversity
        assert diversity.std() > 0  # Should have variation

    def test_compute_neighbor_diversity_with_neighbors(self, sample_embeddings):
        """Test with pre-computed neighbors."""
        from sklearn.neighbors import NearestNeighbors

        k = 10
        nn = NearestNeighbors(n_neighbors=k + 1, metric="cosine")
        nn.fit(sample_embeddings)
        _, neighbors = nn.kneighbors(sample_embeddings)

        diversity = compute_neighbor_diversity(sample_embeddings, k=k, neighbors=neighbors)

        assert len(diversity) == len(sample_embeddings)


class TestDAGMining:
    def test_mine_dag_chains_basic(self, sample_embeddings):
        """Test basic DAG mining."""
        result = mine_dag_chains(sample_embeddings, min_chain_length=3, verbose=False)

        assert isinstance(result, DAGMiningResult)
        assert isinstance(result.chains, list)
        assert isinstance(result.diversity, np.ndarray)
        assert len(result.diversity) == len(sample_embeddings)

    def test_mine_dag_chains_returns_chains(self, clustered_embeddings):
        """Test that chains are found in clustered data."""
        result = mine_dag_chains(clustered_embeddings, min_chain_length=3, verbose=False)

        # Should find some structure
        assert result.n_components >= 0
        assert len(result.parent_child_edges) >= 0

    def test_dag_chain_structure(self, clustered_embeddings):
        """Test DAGChain dataclass."""
        result = mine_dag_chains(clustered_embeddings, min_chain_length=3, verbose=False)

        if result.chains:
            chain = result.chains[0]
            assert isinstance(chain, DAGChain)
            assert isinstance(chain.indices, np.ndarray)
            assert isinstance(chain.coherence, float)
            assert isinstance(chain.diversity_range, float)
            assert len(chain) == len(chain.indices)

    def test_dag_mining_result_summary(self, clustered_embeddings):
        """Test summary method."""
        result = mine_dag_chains(clustered_embeddings, min_chain_length=3, verbose=False)

        summary = result.summary()
        assert isinstance(summary, str)
        assert "DAGMiningResult" in summary

    def test_dag_mining_result_filters(self, clustered_embeddings):
        """Test chain filtering methods."""
        result = mine_dag_chains(clustered_embeddings, min_chain_length=3, verbose=False)

        # Filter by length
        long_chains = result.get_chains_by_length(min_length=4)
        for chain in long_chains:
            assert len(chain) >= 4

        # Filter by coherence
        coherent_chains = result.get_chains_by_coherence(min_coherence=0.5)
        for chain in coherent_chains:
            assert chain.coherence >= 0.5

    def test_dag_mining_diversity_monotonic(self, clustered_embeddings):
        """Test that chains follow diversity gradient."""
        result = mine_dag_chains(clustered_embeddings, min_chain_length=3, diversity_gap_threshold=0.02, verbose=False)

        # All chains should have decreasing diversity
        for chain in result.chains:
            diversities = result.diversity[chain.indices]
            # Should be roughly monotonic (with small tolerance)
            for i in range(len(diversities) - 1):
                # Allow small violations due to floating point
                assert diversities[i] >= diversities[i + 1] - 0.05

    def test_dag_mining_parameters(self, sample_embeddings):
        """Test different parameter settings."""
        # Stricter threshold = fewer edges
        result_strict = mine_dag_chains(
            sample_embeddings, similarity_threshold=0.7, diversity_gap_threshold=0.05, verbose=False
        )

        # Looser threshold = more edges
        result_loose = mine_dag_chains(
            sample_embeddings, similarity_threshold=0.4, diversity_gap_threshold=0.01, verbose=False
        )

        # Looser should have more edges
        assert len(result_loose.parent_child_edges) >= len(result_strict.parent_child_edges)

    def test_dag_mining_empty_result(self):
        """Test with data that has no DAG structure."""
        # Random noise with no structure
        rng = np.random.default_rng(42)
        noise = rng.standard_normal((50, 32)).astype(np.float32)
        noise = noise / np.linalg.norm(noise, axis=1, keepdims=True)

        result = mine_dag_chains(
            noise,
            min_chain_length=3,
            similarity_threshold=0.9,  # Very strict
            verbose=False,
        )

        # Might have no chains, but should not error
        assert isinstance(result, DAGMiningResult)
        assert isinstance(result.chains, list)


# Import DAG taxonomy functions
from dyf import (
    DAGTaxonomy,
    ROGLayer,
    ROGResult,
    UnifiedOntologyResult,
    build_dag_taxonomy,
    build_rog_ontology,
    build_unified_ontology,
)


class TestDAGTaxonomy:
    def test_build_dag_taxonomy_basic(self, clustered_embeddings):
        """Test basic taxonomy building."""
        taxonomy = build_dag_taxonomy(clustered_embeddings, verbose=False)

        assert isinstance(taxonomy, DAGTaxonomy)
        assert taxonomy.n_nodes == len(clustered_embeddings)
        assert isinstance(taxonomy.diversity, np.ndarray)
        assert len(taxonomy.diversity) == len(clustered_embeddings)

    def test_taxonomy_structure(self, clustered_embeddings):
        """Test taxonomy has expected structure."""
        taxonomy = build_dag_taxonomy(clustered_embeddings, verbose=False)

        # Should have roots, leaves, children, parents
        assert isinstance(taxonomy.roots, list)
        assert isinstance(taxonomy.leaves, list)
        assert isinstance(taxonomy.children, dict)
        assert isinstance(taxonomy.parents, dict)

    def test_taxonomy_len(self, clustered_embeddings):
        """Test __len__ returns nodes with edges."""
        taxonomy = build_dag_taxonomy(clustered_embeddings, verbose=False)

        n_with_edges = len(set(taxonomy.children.keys()) | set(taxonomy.parents.keys()))
        assert len(taxonomy) == n_with_edges

    def test_taxonomy_summary(self, clustered_embeddings):
        """Test summary method."""
        taxonomy = build_dag_taxonomy(clustered_embeddings, verbose=False)

        summary = taxonomy.summary()
        assert isinstance(summary, str)
        assert "DAGTaxonomy" in summary
        assert "Nodes" in summary
        assert "edges" in summary

    def test_get_children_parents(self, clustered_embeddings):
        """Test getting children and parents."""
        taxonomy = build_dag_taxonomy(clustered_embeddings, verbose=False)

        # Pick a node that has children
        if taxonomy.children:
            node = list(taxonomy.children.keys())[0]
            children = taxonomy.get_children(node)
            assert isinstance(children, list)
            # All children should be valid indices
            for c in children:
                assert 0 <= c < taxonomy.n_nodes

        # Pick a node that has parents
        if taxonomy.parents:
            node = list(taxonomy.parents.keys())[0]
            parents = taxonomy.get_parents(node)
            assert isinstance(parents, list)
            for p in parents:
                assert 0 <= p < taxonomy.n_nodes

    def test_get_ancestors(self, clustered_embeddings):
        """Test getting ancestors."""
        taxonomy = build_dag_taxonomy(clustered_embeddings, verbose=False)

        if taxonomy.parents:
            # Pick a node with parents
            node = list(taxonomy.parents.keys())[0]
            ancestors = taxonomy.get_ancestors(node, max_depth=5)

            assert isinstance(ancestors, set)
            # Direct parents should be ancestors
            direct_parents = set(taxonomy.get_parents(node))
            assert direct_parents.issubset(ancestors)

    def test_get_descendants(self, clustered_embeddings):
        """Test getting descendants."""
        taxonomy = build_dag_taxonomy(clustered_embeddings, verbose=False)

        if taxonomy.children:
            # Pick a node with children
            node = list(taxonomy.children.keys())[0]
            descendants = taxonomy.get_descendants(node, max_depth=5)

            assert isinstance(descendants, set)
            # Direct children should be descendants
            direct_children = set(taxonomy.get_children(node))
            assert direct_children.issubset(descendants)

    def test_get_common_ancestors(self, clustered_embeddings):
        """Test finding common ancestors."""
        taxonomy = build_dag_taxonomy(clustered_embeddings, verbose=False)

        if len(taxonomy.parents) >= 2:
            nodes = list(taxonomy.parents.keys())[:2]
            common = taxonomy.get_common_ancestors(nodes[0], nodes[1], max_depth=5)

            assert isinstance(common, set)
            # Common ancestors should be ancestors of both
            anc_a = taxonomy.get_ancestors(nodes[0], max_depth=5)
            anc_b = taxonomy.get_ancestors(nodes[1], max_depth=5)
            assert common == (anc_a & anc_b)

    def test_get_lowest_common_ancestors(self, clustered_embeddings):
        """Test finding LCAs."""
        taxonomy = build_dag_taxonomy(clustered_embeddings, verbose=False)

        if len(taxonomy.parents) >= 2:
            nodes = list(taxonomy.parents.keys())[:2]
            lcas = taxonomy.get_lowest_common_ancestors(nodes[0], nodes[1], max_depth=5)

            assert isinstance(lcas, list)
            # LCAs should be subset of common ancestors
            common = taxonomy.get_common_ancestors(nodes[0], nodes[1], max_depth=5)
            for lca in lcas:
                assert lca in common

    def test_get_convergence_points(self, clustered_embeddings):
        """Test finding convergence points."""
        taxonomy = build_dag_taxonomy(clustered_embeddings, verbose=False)

        convergence = taxonomy.get_convergence_points(min_parents=2)

        assert isinstance(convergence, list)
        # Should be tuples of (node, count)
        for node, count in convergence:
            assert count >= 2
            assert len(taxonomy.get_parents(node)) == count

        # Should be sorted by count descending
        if len(convergence) > 1:
            for i in range(len(convergence) - 1):
                assert convergence[i][1] >= convergence[i + 1][1]

    def test_get_divergence_points(self, clustered_embeddings):
        """Test finding divergence points."""
        taxonomy = build_dag_taxonomy(clustered_embeddings, verbose=False)

        divergence = taxonomy.get_divergence_points(min_children=2)

        assert isinstance(divergence, list)
        for node, count in divergence:
            assert count >= 2
            assert len(taxonomy.get_children(node)) == count

    def test_get_diamond_patterns(self, clustered_embeddings):
        """Test finding diamond patterns."""
        taxonomy = build_dag_taxonomy(clustered_embeddings, verbose=False)

        diamonds = taxonomy.get_diamond_patterns(limit=50)

        assert isinstance(diamonds, list)
        # Each diamond is (top, left, right, bottom)
        for diamond in diamonds:
            assert len(diamond) == 4
            top, left, right, bottom = diamond

            # Verify structure: top → left, top → right, left → bottom, right → bottom
            assert left in taxonomy.get_children(top)
            assert right in taxonomy.get_children(top)
            assert bottom in taxonomy.get_children(left)
            assert bottom in taxonomy.get_children(right)

    def test_get_path(self, clustered_embeddings):
        """Test finding a path between nodes."""
        taxonomy = build_dag_taxonomy(clustered_embeddings, verbose=False)

        if taxonomy.children:
            # Find a node with children and its child
            parent = list(taxonomy.children.keys())[0]
            child = taxonomy.get_children(parent)[0]

            path = taxonomy.get_path(parent, child)

            assert path is not None
            assert path[0] == parent
            assert path[-1] == child

    def test_get_path_same_node(self, clustered_embeddings):
        """Test path from node to itself."""
        taxonomy = build_dag_taxonomy(clustered_embeddings, verbose=False)

        if taxonomy.children:
            node = list(taxonomy.children.keys())[0]
            path = taxonomy.get_path(node, node)

            assert path == [node]

    def test_get_all_paths(self, clustered_embeddings):
        """Test finding all paths between nodes."""
        taxonomy = build_dag_taxonomy(clustered_embeddings, verbose=False)

        if taxonomy.children:
            parent = list(taxonomy.children.keys())[0]
            child = taxonomy.get_children(parent)[0]

            paths = taxonomy.get_all_paths(parent, child)

            assert isinstance(paths, list)
            assert len(paths) >= 1
            for path in paths:
                assert path[0] == parent
                assert path[-1] == child

    def test_get_node_depth(self, clustered_embeddings):
        """Test computing node depth."""
        taxonomy = build_dag_taxonomy(clustered_embeddings, verbose=False)

        # Roots should have depth 0
        for root in taxonomy.roots:
            assert taxonomy.get_node_depth(root) == 0

        # Children of roots should have depth 1
        for root in taxonomy.roots:
            for child in taxonomy.get_children(root):
                depth = taxonomy.get_node_depth(child)
                assert depth >= 1

    def test_taxonomy_parameters(self, sample_embeddings):
        """Test with different parameters."""
        # Stricter similarity threshold
        tax_strict = build_dag_taxonomy(
            sample_embeddings, similarity_threshold=0.7, diversity_gap_threshold=0.05, verbose=False
        )

        # Looser thresholds
        tax_loose = build_dag_taxonomy(
            sample_embeddings, similarity_threshold=0.4, diversity_gap_threshold=0.01, verbose=False
        )

        # Looser should have more edges
        edges_strict = sum(len(v) for v in tax_strict.children.values())
        edges_loose = sum(len(v) for v in tax_loose.children.values())
        assert edges_loose >= edges_strict


class TestUnifiedOntology:
    def test_build_unified_ontology_basic(self, clustered_embeddings):
        """Test basic unified ontology building."""
        result = build_unified_ontology(clustered_embeddings, verbose=False)

        assert isinstance(result, UnifiedOntologyResult)
        assert isinstance(result.ontology, DAGTaxonomy)
        assert result.ontology.n_nodes == len(clustered_embeddings)

    def test_unified_ontology_coverage(self, clustered_embeddings):
        """Test that unified ontology has better coverage than single threshold."""
        # Single threshold
        single = build_dag_taxonomy(clustered_embeddings, similarity_threshold=0.55, verbose=False)
        single_nodes = len(set(single.children.keys()) | set(single.parents.keys()))

        # Unified
        result = build_unified_ontology(clustered_embeddings, verbose=False)
        unified_nodes = len(result.main_nodes) + len(result.outlier_nodes)

        # Unified should have >= coverage
        assert unified_nodes >= single_nodes

    def test_unified_ontology_components(self, clustered_embeddings):
        """Test that result contains correct node sets."""
        result = build_unified_ontology(clustered_embeddings, verbose=False)

        # All node sets should be disjoint
        main_set = set(result.main_nodes)
        outlier_set = set(result.outlier_nodes)
        excluded_set = set(result.excluded_nodes)

        assert len(main_set & outlier_set) == 0
        assert len(main_set & excluded_set) == 0
        assert len(outlier_set & excluded_set) == 0

        # Together they should cover all nodes
        total = len(main_set) + len(outlier_set) + len(excluded_set)
        assert total == len(clustered_embeddings)

    def test_unified_ontology_thresholds(self, clustered_embeddings):
        """Test that thresholds are stored correctly."""
        result = build_unified_ontology(
            clustered_embeddings, main_similarity_threshold=0.6, outlier_similarity_threshold=0.4, verbose=False
        )

        assert result.main_threshold == 0.6
        assert result.outlier_threshold == 0.4

    def test_unified_ontology_summary(self, clustered_embeddings):
        """Test summary method."""
        result = build_unified_ontology(clustered_embeddings, verbose=False)

        summary = result.summary()
        assert isinstance(summary, str)
        assert "UnifiedOntologyResult" in summary
        assert "Coverage" in summary
        assert "Main nodes" in summary
        assert "Outlier nodes" in summary

    def test_unified_ontology_len(self, clustered_embeddings):
        """Test __len__ returns ontology size."""
        result = build_unified_ontology(clustered_embeddings, verbose=False)

        assert len(result) == len(result.ontology)

    def test_unified_ontology_bridge_edges(self, clustered_embeddings):
        """Test that bridge edges are counted."""
        result = build_unified_ontology(clustered_embeddings, verbose=False)

        # Bridge edges should be non-negative
        assert result.bridge_edges >= 0

        # If there are outlier nodes, there might be bridge edges
        if len(result.outlier_nodes) > 0:
            # Bridge edges connect outlier to main
            pass  # Just verify it doesn't error

    def test_unified_ontology_navigation(self, clustered_embeddings):
        """Test that unified ontology supports navigation."""
        result = build_unified_ontology(clustered_embeddings, verbose=False)
        ontology = result.ontology

        # Should be able to use all DAGTaxonomy methods
        if ontology.children:
            node = list(ontology.children.keys())[0]
            children = ontology.get_children(node)
            assert isinstance(children, list)

            descendants = ontology.get_descendants(node, max_depth=3)
            assert isinstance(descendants, set)

        if ontology.parents:
            node = list(ontology.parents.keys())[0]
            parents = ontology.get_parents(node)
            assert isinstance(parents, list)

            ancestors = ontology.get_ancestors(node, max_depth=3)
            assert isinstance(ancestors, set)

    def test_unified_ontology_with_random_data(self, sample_embeddings):
        """Test with random embeddings."""
        result = build_unified_ontology(sample_embeddings, verbose=False)

        assert isinstance(result, UnifiedOntologyResult)
        # Should handle any valid embedding data
        assert result.ontology.n_nodes == len(sample_embeddings)


class TestROG:
    """Tests for Recursive Ontological Generation."""

    def test_build_rog_ontology_basic(self, clustered_embeddings):
        """Test basic ROG building."""
        result = build_rog_ontology(clustered_embeddings, verbose=False)

        assert isinstance(result, ROGResult)
        assert isinstance(result.ontology, DAGTaxonomy)
        assert result.ontology.n_nodes == len(clustered_embeddings)

    def test_rog_has_layers(self, clustered_embeddings):
        """Test that ROG produces layers."""
        result = build_rog_ontology(clustered_embeddings, verbose=False)

        assert isinstance(result.layers, list)
        assert len(result.layers) >= 1

        # Each layer should be a ROGLayer
        for layer in result.layers:
            assert isinstance(layer, ROGLayer)
            assert layer.depth >= 0
            assert layer.n_nodes > 0
            assert layer.similarity_threshold > 0

    def test_rog_coverage(self, clustered_embeddings):
        """Test that ROG achieves coverage."""
        result = build_rog_ontology(clustered_embeddings, target_coverage=0.90, verbose=False)

        # Should achieve close to target coverage
        assert result.total_coverage >= 0.80  # Allow some slack

    def test_rog_layers_decrease_threshold(self, clustered_embeddings):
        """Test that layers have decreasing thresholds."""
        result = build_rog_ontology(clustered_embeddings, verbose=False)

        if len(result.layers) > 1:
            for i in range(len(result.layers) - 1):
                assert result.layers[i].similarity_threshold >= result.layers[i + 1].similarity_threshold

    def test_rog_summary(self, clustered_embeddings):
        """Test summary method."""
        result = build_rog_ontology(clustered_embeddings, verbose=False)

        summary = result.summary()
        assert isinstance(summary, str)
        assert "ROG" in summary
        assert "coverage" in summary.lower()
        assert "Layer" in summary

    def test_rog_get_layer_for_node(self, clustered_embeddings):
        """Test get_layer_for_node method."""
        result = build_rog_ontology(clustered_embeddings, verbose=False)

        # Nodes in layers should return their depth
        for layer in result.layers:
            if len(layer.node_indices) > 0:
                node = layer.node_indices[0]
                assert result.get_layer_for_node(node) == layer.depth

        # Excluded nodes should return None
        for node in result.excluded_nodes:
            assert result.get_layer_for_node(node) is None

    def test_rog_excluded_disjoint(self, clustered_embeddings):
        """Test that excluded nodes are disjoint from layers."""
        result = build_rog_ontology(clustered_embeddings, verbose=False)

        layer_nodes = set()
        for layer in result.layers:
            layer_nodes.update(layer.node_indices)

        excluded_set = set(result.excluded_nodes)

        # Should be disjoint
        assert len(layer_nodes & excluded_set) == 0

        # Together should cover all nodes
        assert len(layer_nodes) + len(excluded_set) == len(clustered_embeddings)

    def test_rog_ontology_navigation(self, clustered_embeddings):
        """Test that ROG ontology supports navigation."""
        result = build_rog_ontology(clustered_embeddings, verbose=False)
        ontology = result.ontology

        # Should support standard DAGTaxonomy methods
        if ontology.children:
            node = list(ontology.children.keys())[0]
            children = ontology.get_children(node)
            assert isinstance(children, list)

        if ontology.parents:
            node = list(ontology.parents.keys())[0]
            ancestors = ontology.get_ancestors(node, max_depth=3)
            assert isinstance(ancestors, set)

    def test_rog_parameters(self, sample_embeddings):
        """Test ROG with different parameters."""
        # Strict (fewer layers, less coverage)
        result_strict = build_rog_ontology(
            sample_embeddings, initial_threshold=0.6, min_threshold=0.5, target_coverage=0.99, verbose=False
        )

        # Loose (more layers, more coverage)
        result_loose = build_rog_ontology(
            sample_embeddings, initial_threshold=0.5, min_threshold=0.3, target_coverage=0.99, verbose=False
        )

        # Loose should have >= coverage
        assert result_loose.total_coverage >= result_strict.total_coverage

    def test_rog_len(self, clustered_embeddings):
        """Test __len__ method."""
        result = build_rog_ontology(clustered_embeddings, verbose=False)

        assert len(result) == len(result.ontology)
