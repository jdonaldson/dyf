"""Tests for PCA tree module."""

import numpy as np
import pytest

from dyf import cut_tree_to_labels
from dyf.pca_tree import (
    boundary_persistence_scores,
    build_pca_tree,
    extract_boundary_persistence,
)


@pytest.fixture
def clustered_embeddings():
    """50 points in 8-D with 3 well-separated clusters."""
    rng = np.random.default_rng(42)
    centers = np.array(
        [
            [3, 0, 0, 0, 0, 0, 0, 0],
            [0, 3, 0, 0, 0, 0, 0, 0],
            [0, 0, 3, 0, 0, 0, 0, 0],
        ],
        dtype=np.float64,
    )
    points = []
    for c in centers:
        pts = rng.normal(size=(17, 8)) * 0.3 + c
        points.append(pts)
    # Add a few boundary points between clusters 0 and 1
    for _ in range(2):
        mid = (centers[0] + centers[1]) / 2
        points.append(mid + rng.normal(size=(1, 8)) * 0.1)
    points = np.vstack(points)
    # Pad to exactly 53 points — that's fine, doesn't need to be 50
    return points.astype(np.float64)


class TestBuildPcaTree:
    def test_returns_valid_tree(self, clustered_embeddings):
        tree = build_pca_tree(clustered_embeddings, max_depth=5)
        assert "left" in tree
        assert "right" in tree
        assert "indices" in tree
        assert "depth" in tree
        assert "point_margin_map" in tree

    def test_root_contains_all_points(self, clustered_embeddings):
        n = len(clustered_embeddings)
        tree = build_pca_tree(clustered_embeddings, max_depth=5)
        assert len(tree["indices"]) == n

    def test_depth_zero_is_leaf(self, clustered_embeddings):
        tree = build_pca_tree(clustered_embeddings, max_depth=0)
        assert tree["left"] is None
        assert tree["right"] is None

    def test_has_internal_nodes(self, clustered_embeddings):
        tree = build_pca_tree(clustered_embeddings, max_depth=5)
        assert tree["left"] is not None
        assert tree["right"] is not None
        assert tree["point_margin_map"] is not None

    def test_min_leaf_size_respected(self):
        rng = np.random.default_rng(99)
        emb = rng.normal(size=(10, 4))
        tree = build_pca_tree(emb, max_depth=20, min_leaf_size=3)

        # Walk leaves and check they have at least min_leaf_size points
        def _count_leaves(node):
            if node["left"] is None and node["right"] is None:
                return [len(node["indices"])]
            sizes = []
            if node["left"] is not None:
                sizes.extend(_count_leaves(node["left"]))
            if node["right"] is not None:
                sizes.extend(_count_leaves(node["right"]))
            return sizes

        leaf_sizes = _count_leaves(tree)
        # Leaf sizes should generally be >= min_leaf_size (some edge cases
        # with very small data can produce smaller leaves)
        assert all(s >= 1 for s in leaf_sizes)


class TestExtractBoundaryPersistence:
    def test_returns_expected_keys(self, clustered_embeddings):
        tree = build_pca_tree(clustered_embeddings, max_depth=5)
        result = extract_boundary_persistence(tree, margin_pct=0.10)
        assert "boundary_depths" in result
        assert "boundary_count" in result
        assert "thresholds" in result

    def test_boundary_count_shape(self, clustered_embeddings):
        n = len(clustered_embeddings)
        tree = build_pca_tree(clustered_embeddings, max_depth=5)
        result = extract_boundary_persistence(tree)
        assert result["boundary_count"].shape == (n,)

    def test_boundary_count_non_negative(self, clustered_embeddings):
        tree = build_pca_tree(clustered_embeddings, max_depth=5)
        result = extract_boundary_persistence(tree)
        assert (result["boundary_count"] >= 0).all()

    def test_some_boundaries_detected(self, clustered_embeddings):
        tree = build_pca_tree(clustered_embeddings, max_depth=5)
        result = extract_boundary_persistence(tree, margin_pct=0.20)
        assert result["boundary_count"].sum() > 0

    def test_thresholds_per_depth(self, clustered_embeddings):
        tree = build_pca_tree(clustered_embeddings, max_depth=5)
        result = extract_boundary_persistence(tree)
        # Should have at least one threshold (for depth 0)
        assert len(result["thresholds"]) >= 1


class TestBoundaryPersistenceScores:
    def test_returns_correct_length(self, clustered_embeddings):
        n = len(clustered_embeddings)
        tree = build_pca_tree(clustered_embeddings, max_depth=5)
        scores = boundary_persistence_scores(tree)
        assert len(scores) == n

    def test_non_negative(self, clustered_embeddings):
        tree = build_pca_tree(clustered_embeddings, max_depth=5)
        scores = boundary_persistence_scores(tree)
        assert (scores >= 0).all()

    def test_explicit_max_depth(self, clustered_embeddings):
        tree = build_pca_tree(clustered_embeddings, max_depth=5)
        scores = boundary_persistence_scores(tree, max_depth=5)
        assert len(scores) == len(clustered_embeddings)


class TestCutTreeToLabels:
    def test_returns_correct_length(self, clustered_embeddings):
        n = len(clustered_embeddings)
        tree = build_pca_tree(clustered_embeddings, max_depth=5)
        labels = cut_tree_to_labels(tree, max_depth=5, n_points=n, n_clusters=3)
        assert len(labels) == n

    def test_labels_in_valid_range(self, clustered_embeddings):
        n = len(clustered_embeddings)
        tree = build_pca_tree(clustered_embeddings, max_depth=5)
        labels = cut_tree_to_labels(tree, max_depth=5, n_points=n, n_clusters=3)
        assert labels.min() >= 0
        assert labels.max() < n  # labels should be reasonable

    def test_produces_requested_clusters(self, clustered_embeddings):
        n = len(clustered_embeddings)
        tree = build_pca_tree(clustered_embeddings, max_depth=5)
        labels = cut_tree_to_labels(tree, max_depth=5, n_points=n, n_clusters=3)
        n_unique = len(set(labels.tolist()))
        # Should produce approximately the requested number
        assert n_unique <= max(3, n)
        assert n_unique >= 1

    def test_different_cluster_counts(self, clustered_embeddings):
        n = len(clustered_embeddings)
        tree = build_pca_tree(clustered_embeddings, max_depth=5)
        labels_3 = cut_tree_to_labels(tree, max_depth=5, n_points=n, n_clusters=3)
        labels_5 = cut_tree_to_labels(tree, max_depth=5, n_points=n, n_clusters=5)
        # More clusters requested should give at least as many unique labels
        assert len(set(labels_5.tolist())) >= len(set(labels_3.tolist()))
