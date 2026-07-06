"""Tests for dyf_tree.py backward-compat fix and basic tree construction."""

import numpy as np
import pytest

from dyf import check_rust_available

pytestmark = pytest.mark.skipif(not check_rust_available(), reason="Rust extension not available")


def _make_clustered_embeddings(n_clusters=5, points_per_cluster=40, dim=32, seed=42):
    """Create synthetic clustered embeddings on the unit sphere."""
    rng = np.random.default_rng(seed)
    centers = rng.standard_normal((n_clusters, dim)).astype(np.float32)
    centers /= np.linalg.norm(centers, axis=1, keepdims=True)

    points = []
    for i in range(n_clusters):
        noise = rng.standard_normal((points_per_cluster, dim)).astype(np.float32) * 0.1
        cluster_pts = centers[i] + noise
        cluster_pts /= np.linalg.norm(cluster_pts, axis=1, keepdims=True)
        points.append(cluster_pts)

    return np.concatenate(points, axis=0)


class TestBuildDyfTree:
    """Verify build_dyf_tree produces a proper multi-node tree."""

    def test_tree_has_children(self):
        """Tree on clustered data should split into >1 node (not single-leaf)."""
        from dyf import build_dyf_tree

        embeddings = _make_clustered_embeddings(n_clusters=5, points_per_cluster=40, dim=32)
        tree = build_dyf_tree(embeddings, max_depth=3, num_bits=3, min_leaf_size=4, seed=42)

        assert len(tree["children"]) > 0, "Tree collapsed to single node — build_dyf_tree bug"

    def test_tree_covers_all_points(self):
        """All point indices should be present in the tree."""
        from dyf import build_dyf_tree

        n = 200
        embeddings = _make_clustered_embeddings(n_clusters=5, points_per_cluster=40, dim=32)
        tree = build_dyf_tree(embeddings, max_depth=3, num_bits=3, min_leaf_size=4, seed=42)

        assert set(tree["indices"].tolist()) == set(range(n))

    def test_raw_pca_fallback(self):
        """fit_method='raw_pca' falls back to fit() if raw_pca unavailable."""
        from dyf import build_dyf_tree

        embeddings = _make_clustered_embeddings(n_clusters=3, points_per_cluster=30, dim=16)
        # raw_pca is the default; should work without error
        tree = build_dyf_tree(embeddings, max_depth=2, num_bits=2, min_leaf_size=4, seed=42, fit_method="raw_pca")

        assert len(tree["indices"]) == 90

    def test_pca_fit_method(self):
        """fit_method='pca' (falls back to fit()) works."""
        from dyf import build_dyf_tree

        embeddings = _make_clustered_embeddings(n_clusters=3, points_per_cluster=30, dim=16)
        tree = build_dyf_tree(embeddings, max_depth=2, num_bits=2, min_leaf_size=4, seed=42, fit_method="pca")

        # Should produce a valid tree (possibly single-leaf if pca not
        # available, but no crash)
        assert len(tree["indices"]) == 90

    def test_itq_fit_method(self):
        """fit_method='itq' works."""
        from dyf import build_dyf_tree

        embeddings = _make_clustered_embeddings(n_clusters=3, points_per_cluster=30, dim=16)
        tree = build_dyf_tree(embeddings, max_depth=2, num_bits=2, min_leaf_size=4, seed=42, fit_method="itq")

        assert len(tree["indices"]) == 90

    def test_depth_and_margins(self):
        """Tree records depth and point_margin_map at internal nodes."""
        from dyf import build_dyf_tree

        embeddings = _make_clustered_embeddings(n_clusters=5, points_per_cluster=40, dim=32)
        tree = build_dyf_tree(embeddings, max_depth=3, num_bits=3, min_leaf_size=4, seed=42)

        assert tree["depth"] == 3
        # Root node (if it split) should have a margin map
        if tree["children"]:
            assert tree["point_margin_map"] is not None
            assert len(tree["point_margin_map"]) == 200
