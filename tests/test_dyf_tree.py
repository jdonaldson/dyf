"""Tests for dyf.dyf_tree — extracted helper functions."""

import numpy as np


class TestEjectPeriphery:
    """Tests for _eject_periphery cluster refinement."""

    def _make_cluster_data(self, n_clusters=3, points_per=20, dim=8, seed=42):
        """Create embeddings with known cluster structure."""
        rng = np.random.default_rng(seed)
        embeddings = []
        labels = []
        for c in range(n_clusters):
            center = rng.standard_normal(dim).astype(np.float32)
            center /= np.linalg.norm(center)
            pts = center + rng.standard_normal((points_per, dim)).astype(np.float32) * 0.1
            pts /= np.linalg.norm(pts, axis=1, keepdims=True)
            embeddings.append(pts)
            labels.extend([c] * points_per)

        emb = np.concatenate(embeddings)
        labels = np.array(labels, dtype=int)

        # Compute coherence and members
        cluster_coherence = {}
        cluster_members = {}
        for c in range(n_clusters):
            members = np.where(labels == c)[0]
            cluster_members[c] = members
            subset = emb[members]
            centroid = subset.mean(axis=0)
            norm = np.linalg.norm(centroid)
            if norm > 1e-10:
                centroid /= norm
            sims = subset @ centroid
            cluster_coherence[c] = float(np.mean(sims))

        return labels, emb, cluster_coherence, cluster_members

    def test_low_coherence_ejected(self):
        from dyf.dyf_tree import _eject_periphery

        labels, emb, coherence, members = self._make_cluster_data()
        # Set threshold very high so all clusters are "low coherence"
        threshold = 1.0
        labels_out, ejected = _eject_periphery(
            labels.copy(), emb, coherence, members, threshold)
        assert len(ejected) > 0
        # Ejected points should have label -1
        for idx in ejected:
            assert labels_out[idx] == -1

    def test_high_coherence_untouched(self):
        from dyf.dyf_tree import _eject_periphery

        labels, emb, coherence, members = self._make_cluster_data()
        # Set threshold very low so nothing gets ejected
        threshold = 0.0
        original_labels = labels.copy()
        labels_out, ejected = _eject_periphery(
            labels.copy(), emb, coherence, members, threshold)
        assert len(ejected) == 0
        np.testing.assert_array_equal(labels_out, original_labels)

    def test_small_cluster_skipped(self):
        from dyf.dyf_tree import _eject_periphery

        # Create tiny clusters (< 4 members) that should be skipped
        emb = np.random.default_rng(42).standard_normal((6, 8)).astype(np.float32)
        emb /= np.linalg.norm(emb, axis=1, keepdims=True)
        labels = np.array([0, 0, 0, 1, 1, 1], dtype=int)  # 3 per cluster
        coherence = {0: 0.1, 1: 0.1}  # low coherence
        members = {0: np.array([0, 1, 2]), 1: np.array([3, 4, 5])}

        labels_out, ejected = _eject_periphery(
            labels.copy(), emb, coherence, members, threshold=0.5)
        # Clusters with < 4 members should be skipped
        assert len(ejected) == 0

    def test_ejected_labels_set_to_neg1(self):
        from dyf.dyf_tree import _eject_periphery

        labels, emb, coherence, members = self._make_cluster_data(
            n_clusters=2, points_per=30)
        threshold = 1.0  # eject from all
        labels_out, ejected = _eject_periphery(
            labels.copy(), emb, coherence, members, threshold)
        # All ejected indices should have -1
        if len(ejected) > 0:
            assert np.all(labels_out[ejected] == -1)
            # Non-ejected should retain original labels
            kept = np.setdiff1d(np.arange(len(labels)), ejected)
            assert np.all(labels_out[kept] >= 0)
