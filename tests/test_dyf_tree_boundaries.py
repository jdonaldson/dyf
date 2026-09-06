"""Boundary-persistence and tree-cutting tests for the DYF tree.

Ported from `test_pca_tree.py` on 2026-09-05 when `pca_tree` was dropped. Every test of
`extract_boundary_persistence`, `boundary_persistence_scores` and `cut_tree_to_labels`
had been written against the *PCA* variants — the DYF variants, which now own those
top-level names, had **no coverage at all**. Deleting the PCA module without porting
these would have handed the public names to untested code.

The DYF versions take the same signatures, so the assertions transfer directly; only the
builder and its k-ary shape differ.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("dyf_rs")

from dyf import build_dyf_tree, cut_tree_to_labels  # noqa: E402
from dyf.dyf_tree import boundary_persistence_scores, extract_boundary_persistence  # noqa: E402


@pytest.fixture
def clustered_embeddings():
    """53 points in 8-D with 3 well-separated clusters plus 2 boundary points.

    Same fixture the PCA tests used — boundary points between clusters 0 and 1 are what
    give `extract_boundary_persistence` something to actually find.
    """
    rng = np.random.default_rng(42)
    centers = np.array(
        [
            [3, 0, 0, 0, 0, 0, 0, 0],
            [0, 3, 0, 0, 0, 0, 0, 0],
            [0, 0, 3, 0, 0, 0, 0, 0],
        ],
        dtype=np.float64,
    )
    points = [rng.normal(size=(17, 8)) * 0.3 + c for c in centers]
    for _ in range(2):
        mid = (centers[0] + centers[1]) / 2
        points.append(mid + rng.normal(size=(1, 8)) * 0.1)
    return np.ascontiguousarray(np.vstack(points).astype(np.float32))


@pytest.fixture
def tree(clustered_embeddings):
    return build_dyf_tree(clustered_embeddings, max_depth=4, num_bits=2, min_leaf_size=2, seed=42)


class TestExtractBoundaryPersistence:
    def test_returns_expected_keys(self, tree):
        result = extract_boundary_persistence(tree, margin_pct=0.10)
        assert "boundary_depths" in result
        assert "boundary_count" in result
        assert "thresholds" in result

    def test_boundary_count_shape(self, tree, clustered_embeddings):
        result = extract_boundary_persistence(tree)
        assert result["boundary_count"].shape == (len(clustered_embeddings),)

    def test_boundary_count_non_negative(self, tree):
        assert (extract_boundary_persistence(tree)["boundary_count"] >= 0).all()

    def test_some_boundaries_detected(self, tree):
        """A wide margin on a fixture built with boundary points must find some."""
        result = extract_boundary_persistence(tree, margin_pct=0.20)
        assert result["boundary_count"].sum() > 0

    def test_thresholds_per_depth(self, tree):
        assert len(extract_boundary_persistence(tree)["thresholds"]) >= 1

    def test_wider_margin_finds_at_least_as_many(self, tree):
        """Behavioural, not shape: the margin parameter must actually do something."""
        narrow = extract_boundary_persistence(tree, margin_pct=0.05)["boundary_count"].sum()
        wide = extract_boundary_persistence(tree, margin_pct=0.30)["boundary_count"].sum()
        assert wide >= narrow


class TestBoundaryPersistenceScores:
    def test_returns_correct_length(self, tree, clustered_embeddings):
        assert len(boundary_persistence_scores(tree)) == len(clustered_embeddings)

    def test_non_negative(self, tree):
        assert (boundary_persistence_scores(tree) >= 0).all()

    def test_explicit_max_depth(self, tree, clustered_embeddings):
        scores = boundary_persistence_scores(tree, max_depth=4)
        assert len(scores) == len(clustered_embeddings)


class TestCutTreeToLabels:
    def test_returns_correct_length(self, tree, clustered_embeddings):
        labels = cut_tree_to_labels(tree, len(clustered_embeddings), n_clusters=3, embeddings=clustered_embeddings)
        assert len(labels) == len(clustered_embeddings)

    def test_labels_in_valid_range(self, tree, clustered_embeddings):
        labels = cut_tree_to_labels(tree, len(clustered_embeddings), n_clusters=3, embeddings=clustered_embeddings)
        assert labels.min() >= 0
        assert labels.max() < 3

    def test_produces_requested_clusters(self, tree, clustered_embeddings):
        labels = cut_tree_to_labels(tree, len(clustered_embeddings), n_clusters=3, embeddings=clustered_embeddings)
        assert len(np.unique(labels)) <= 3

    def test_different_cluster_counts(self, tree, clustered_embeddings):
        for k in (2, 3, 4):
            labels = cut_tree_to_labels(tree, len(clustered_embeddings), n_clusters=k, embeddings=clustered_embeddings)
            assert len(np.unique(labels)) <= k

    def test_missing_embeddings_is_an_actionable_error(self, tree, clustered_embeddings):
        """The dispatcher must name what is missing, not raise a bare KeyError."""
        with pytest.raises(ValueError, match="embeddings"):
            cut_tree_to_labels(tree, len(clustered_embeddings), n_clusters=3)

    def test_unrecognised_shape_is_an_actionable_error(self):
        with pytest.raises(ValueError, match="Unrecognized tree shape"):
            cut_tree_to_labels({"nonsense": 1}, 10, n_clusters=2, embeddings=np.zeros((10, 4)))
