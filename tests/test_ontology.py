"""Tests for dyf.ontology — extracted helper functions."""

from collections import defaultdict

import numpy as np


class TestBuildRogLayer:
    """Tests for _build_rog_layer at various thresholds."""

    def _make_knn_data(self, n=20, k=5, seed=42):
        """Build synthetic k-NN data with known structure."""
        from dyf.ontology import _KNNData

        rng = np.random.default_rng(seed)
        # Create embeddings with two clusters
        emb = rng.standard_normal((n, 8)).astype(np.float32)
        emb /= np.linalg.norm(emb, axis=1, keepdims=True)

        # Build simple k-NN: each point's neighbors are the k nearest
        sims = emb @ emb.T
        neighbors = np.zeros((n, k), dtype=int)
        similarities = np.zeros((n, k), dtype=float)
        for i in range(n):
            order = np.argsort(-sims[i])
            neighbors[i] = order[:k]
            similarities[i] = sims[i, order[:k]]

        # Diversity: first half "general" (high), second half "specific" (low)
        diversity = np.zeros(n)
        diversity[: n // 2] = 0.5 + rng.random(n // 2) * 0.3
        diversity[n // 2 :] = 0.1 + rng.random(n - n // 2) * 0.1

        return _KNNData(neighbors=neighbors, similarities=similarities, diversity=diversity), n

    def test_basic_layer(self):
        from dyf.ontology import _build_rog_layer, _ROGState

        knn, n = self._make_knn_data()
        state = _ROGState(
            all_children=defaultdict(list),
            all_parents=defaultdict(list),
        )
        remaining = set(range(n))

        layer = _build_rog_layer(
            remaining, knn, threshold=0.0, diversity_gap_threshold=0.01, depth=0, n_points=n, state=state
        )
        # With threshold=0 (accept all), should connect some nodes
        assert layer is not None
        assert layer.n_nodes > 0
        assert layer.n_edges > 0

    def test_high_threshold_no_connections(self):
        from dyf.ontology import _build_rog_layer, _ROGState

        knn, n = self._make_knn_data()
        state = _ROGState(
            all_children=defaultdict(list),
            all_parents=defaultdict(list),
        )
        remaining = set(range(n))

        # Threshold so high that no edges qualify
        layer = _build_rog_layer(
            remaining, knn, threshold=2.0, diversity_gap_threshold=0.01, depth=0, n_points=n, state=state
        )
        assert layer is None

    def test_mutates_state(self):
        from dyf.ontology import _build_rog_layer, _ROGState

        knn, n = self._make_knn_data()
        state = _ROGState(
            all_children=defaultdict(list),
            all_parents=defaultdict(list),
        )
        remaining = set(range(n))

        _build_rog_layer(remaining, knn, threshold=0.0, diversity_gap_threshold=0.01, depth=0, n_points=n, state=state)
        # State should have accumulated edges
        total = sum(len(v) for v in state.all_children.values())
        assert total > 0

    def test_remaining_shrinks(self):
        from dyf.ontology import _build_rog_layer, _ROGState

        knn, n = self._make_knn_data()
        state = _ROGState(
            all_children=defaultdict(list),
            all_parents=defaultdict(list),
        )
        remaining = set(range(n))
        original_size = len(remaining)

        layer = _build_rog_layer(
            remaining, knn, threshold=0.0, diversity_gap_threshold=0.01, depth=0, n_points=n, state=state
        )
        if layer is not None:
            assert len(remaining) < original_size
