"""Tests for dyf.enrich enrichment pipeline functions.

Covers pure functions (no external services) and mocked tests for
Ollama/UMAP-dependent code paths.
"""

import json
import os
import tempfile

import numpy as np
import pytest

from dyf import check_rust_available

pytestmark = pytest.mark.skipif(not check_rust_available(), reason="Rust extension not available")

try:
    import flatbuffers  # noqa: F401
    import pyarrow  # noqa: F401

    _HAS_LAZY_DEPS = True
except ImportError:
    _HAS_LAZY_DEPS = False

lazy_deps = pytest.mark.skipif(not _HAS_LAZY_DEPS, reason="pyarrow and flatbuffers required")


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


# ── Pure function tests ──────────────────────────────────────────────────


class TestCollectDescendantIndices:
    """Test _collect_descendant_indices recursion."""

    def test_leaf_node(self):
        from dyf.enrich._tree import _collect_descendant_indices

        leaf_batches = {10: np.array([0, 1, 2])}
        children_of = {}
        result = _collect_descendant_indices(10, children_of, leaf_batches)
        np.testing.assert_array_equal(result, [0, 1, 2])

    def test_internal_node(self):
        from dyf.enrich._tree import _collect_descendant_indices

        leaf_batches = {
            2: np.array([10, 11]),
            3: np.array([20, 21, 22]),
        }
        children_of = {1: [2, 3]}
        result = _collect_descendant_indices(1, children_of, leaf_batches)
        assert set(result.tolist()) == {10, 11, 20, 21, 22}

    def test_deep_tree(self):
        from dyf.enrich._tree import _collect_descendant_indices

        leaf_batches = {
            4: np.array([100]),
            5: np.array([200, 201]),
        }
        children_of = {1: [2, 3], 2: [4], 3: [5]}
        result = _collect_descendant_indices(1, children_of, leaf_batches)
        assert set(result.tolist()) == {100, 200, 201}

    def test_empty_node(self):
        from dyf.enrich._tree import _collect_descendant_indices

        leaf_batches = {}
        children_of = {1: []}
        result = _collect_descendant_indices(1, children_of, leaf_batches)
        assert len(result) == 0


class TestMergeTinyClusters:
    """Test merge_tiny_clusters merging + relabeling."""

    def test_no_merge_needed(self):
        from dyf.enrich._cluster import merge_tiny_clusters

        n = 200
        labels = np.array([i % 4 for i in range(n)])  # 50 each
        coords = np.random.default_rng(42).standard_normal((n, 2)).astype(np.float32)
        result = merge_tiny_clusters(labels, coords, min_pct=0.01)
        assert set(result.tolist()) == {0, 1, 2, 3}

    def test_tiny_cluster_merged(self):
        from dyf.enrich._cluster import merge_tiny_clusters

        n = 200
        rng = np.random.default_rng(42)
        # Cluster 0: 97 pts, Cluster 1: 97 pts, Cluster 2: 6 pts (tiny)
        labels = np.zeros(n, dtype=int)
        labels[97:194] = 1
        labels[194:] = 2
        coords = rng.standard_normal((n, 2)).astype(np.float32)
        # Cluster 2 coords near cluster 1
        coords[194:] = coords[97:103] + 0.01

        result = merge_tiny_clusters(labels, coords, min_pct=0.05)
        unique = set(result.tolist())
        # Cluster 2 should have been merged
        assert len(unique) == 2

    def test_contiguous_relabeling(self):
        from dyf.enrich._cluster import merge_tiny_clusters

        n = 200
        rng = np.random.default_rng(42)
        # Cluster 0: 90 pts, Cluster 5: 90 pts, Cluster 10: 5 pts (tiny)
        labels = np.zeros(n, dtype=int)
        labels[90:180] = 5
        labels[180:185] = 10  # 5 pts — tiny
        labels[185:] = 5
        coords = rng.standard_normal((n, 2)).astype(np.float32)
        # Make cluster 10 near cluster 5
        coords[180:185] = coords[90:95] + 0.01
        result = merge_tiny_clusters(labels, coords, min_pct=0.05)
        # After merge, should be 0-based contiguous
        unique = sorted(set(result.tolist()))
        assert unique == list(range(len(unique)))


class TestComputeTFIDFKeywords:
    """Test TF-IDF keyword extraction."""

    def test_basic_keywords(self):
        from dyf.enrich._labeling import _compute_tfidf_keywords

        titles = [
            "cardiac pacemaker implant",
            "cardiac defibrillator lead",
            "cardiac monitor device",
            "orthopedic hip screw",
            "orthopedic knee plate",
            "orthopedic bone cement",
        ]
        labels = np.array([0, 0, 0, 1, 1, 1])
        kw = _compute_tfidf_keywords(titles, labels, n_clusters=2, top_k=5)

        # Cluster 0 keywords should include cardiac-related terms
        c0_words = [w for w, _ in kw[0]]
        c1_words = [w for w, _ in kw[1]]
        assert "cardiac" in c0_words
        assert "orthopedic" in c1_words

    def test_empty_cluster(self):
        from dyf.enrich._labeling import _compute_tfidf_keywords

        titles = ["item one", "item two"]
        labels = np.array([0, 0])
        kw = _compute_tfidf_keywords(titles, labels, n_clusters=2, top_k=5)
        # Cluster 1 has no items, should return empty
        assert kw[1] == []
        # ...but cluster 0 must still be populated, or `kw[1] == []` would also pass for a
        # function that returns nothing at all for every cluster.
        assert kw[0], "cluster 0 has items but produced no keywords"
        assert all(isinstance(w, str) and w for w, _ in kw[0])


class TestFindNearestCluster:
    """Test L2 nearest cluster centroid lookup."""

    def test_finds_nearest(self):
        from dyf.enrich._labeling import _find_nearest_cluster

        centroids = np.array(
            [
                [0.0, 0.0],
                [10.0, 0.0],
                [0.0, 10.0],
            ]
        )
        # Cluster 0 is nearest to cluster 2 (both at origin-ish)
        # Actually cluster 0 at (0,0), cluster 1 at (10,0), cluster 2 at (0,10)
        # Nearest to 0 could be 1 or 2 (both distance 10)
        nearest = _find_nearest_cluster(0, centroids)
        assert nearest in (1, 2)

    def test_asymmetric_distance(self):
        from dyf.enrich._labeling import _find_nearest_cluster

        centroids = np.array(
            [
                [0.0, 0.0],
                [1.0, 0.0],
                [100.0, 100.0],
            ]
        )
        assert _find_nearest_cluster(0, centroids) == 1
        assert _find_nearest_cluster(1, centroids) == 0


class TestSampleSpatial:
    """Test farthest-point sampling."""

    def test_returns_k_points(self):
        from dyf.enrich._labeling import _sample_spatial

        rng = np.random.default_rng(42)
        n = 100
        indices = np.arange(n)
        coords = rng.standard_normal((n, 2)).astype(np.float32)
        result = _sample_spatial(indices, coords, k=10)
        assert len(result) == 10

    def test_unique_results(self):
        from dyf.enrich._labeling import _sample_spatial

        rng = np.random.default_rng(42)
        n = 50
        indices = np.arange(n)
        coords = rng.standard_normal((n, 2)).astype(np.float32)
        result = _sample_spatial(indices, coords, k=10)
        assert len(set(result)) == 10

    def test_small_input(self):
        from dyf.enrich._labeling import _sample_spatial

        indices = np.array([5, 10, 15])
        coords = np.zeros((20, 2), dtype=np.float32)
        result = _sample_spatial(indices, coords, k=10)
        assert set(result) == {5, 10, 15}


class TestGenerateNarration:
    """Test _generate_narration output structure (no-Ollama fallback path)."""

    def _narrate(self, cluster_names, titles, labels, **kwargs):
        from dyf.enrich._narration import _generate_narration

        coords = np.zeros((len(labels), 2), dtype=np.float32)
        return _generate_narration(cluster_names, titles, labels, coords, model="nonexistent_model", **kwargs)

    def test_has_intro_outro(self):
        cluster_names = {0: "Alpha", 1: "Beta", 2: "Gamma"}
        titles = [f"item {i}" for i in range(30)]
        labels = np.array([i % 3 for i in range(30)])
        narration = self._narrate(cluster_names, titles, labels)

        assert "intro" in narration
        assert "outro" in narration

    def test_cluster_count_in_intro(self):
        cluster_names = {0: "Alpha", 1: "Beta", 2: "Gamma"}
        titles = [f"item {i}" for i in range(30)]
        labels = np.array([i % 3 for i in range(30)])
        narration = self._narrate(cluster_names, titles, labels)

        assert "three clusters" in narration["intro"]

    def test_custom_title(self):
        cluster_names = {0: "Alpha"}
        titles = ["item 0"]
        labels = np.array([0])
        narration = self._narrate(cluster_names, titles, labels, title="My Landscape")

        assert "My Landscape" in narration["intro"]

    def test_per_cluster_narration(self):
        cluster_names = {0: "Alpha", 1: "Beta"}
        titles = ["a", "b", "c", "d"]
        labels = np.array([0, 0, 1, 1])
        narration = self._narrate(cluster_names, titles, labels)

        assert 0 in narration
        assert 1 in narration
        assert "Alpha" in narration[0]
        assert "Beta" in narration[1]


class TestTransferLabelsMajorityVote:
    """Test transfer_labels_majority_vote label transfer + disambiguation."""

    def test_basic_transfer(self):
        from dyf.enrich._labeling import transfer_labels_majority_vote

        # 2D primary clusters: 3 clusters, clear separation
        labels_2d = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2])
        names_2d = {0: "Alpha", 1: "Beta", 2: "Gamma"}
        # 3D secondary clusters: same split
        labels_3d = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2])

        names_3d = transfer_labels_majority_vote(labels_2d, names_2d, labels_3d)

        assert names_3d[0] == "Alpha"
        assert names_3d[1] == "Beta"
        assert names_3d[2] == "Gamma"

    def test_different_partition(self):
        from dyf.enrich._labeling import transfer_labels_majority_vote

        # 2D: 2 clusters [0,0,0,0, 1,1,1,1]
        labels_2d = np.array([0, 0, 0, 0, 1, 1, 1, 1])
        names_2d = {0: "Left", 1: "Right"}
        # 3D: 2 clusters but split differently [0,0,0,0,0, 1,1,1]
        labels_3d = np.array([0, 0, 0, 0, 0, 1, 1, 1])

        names_3d = transfer_labels_majority_vote(labels_2d, names_2d, labels_3d)

        # 3D cluster 0 has 4 from "Left", 1 from "Right" → "Left"
        assert names_3d[0] == "Left"
        # 3D cluster 1 has 3 from "Right" → "Right"
        assert names_3d[1] == "Right"

    def test_disambiguation(self):
        from dyf.enrich._labeling import transfer_labels_majority_vote

        # Both 3D clusters map to the same 2D cluster (both majority = 0)
        labels_2d = np.array([0, 0, 0, 0, 0, 0])
        names_2d = {0: "AllSame"}
        labels_3d = np.array([0, 0, 0, 1, 1, 1])

        names_3d = transfer_labels_majority_vote(labels_2d, names_2d, labels_3d)

        # First occurrence keeps original name, second gets suffix
        assert names_3d[0] == "AllSame"
        assert names_3d[1] == "AllSame (2)"

    def test_many_collisions(self):
        from dyf.enrich._labeling import transfer_labels_majority_vote

        # 4 secondary clusters all map to same primary
        labels_2d = np.array([0] * 20)
        names_2d = {0: "Same"}
        labels_3d = np.array([i // 5 for i in range(20)])

        names_3d = transfer_labels_majority_vote(labels_2d, names_2d, labels_3d)

        # Should have "Same", "Same (2)", "Same (3)", "Same (4)"
        values = sorted(names_3d.values())
        assert "Same" in values
        assert "Same (2)" in values
        assert "Same (3)" in values
        assert "Same (4)" in values


@lazy_deps
class TestEnrichClusterDual:
    """Test enrich_cluster produces dual 2D/3D cluster fields."""

    def test_dual_cluster_fields_louvain(self):
        """Louvain mode: same labels for 2D and 3D, different centroids."""
        from unittest.mock import patch

        from dyf import build_dyf_tree
        from dyf.enrich._cluster import enrich_cluster
        from dyf.lazy_index import LazyIndex, write_lazy_index

        n = 200
        embeddings = _make_clustered_embeddings(n_clusters=5, points_per_cluster=40, dim=32)
        tree = build_dyf_tree(embeddings, max_depth=3, num_bits=3, min_leaf_size=4, seed=42)

        rng = np.random.default_rng(42)

        with tempfile.NamedTemporaryFile(suffix=".dyf", delete=False) as f:
            path = f.name
        with tempfile.NamedTemporaryFile(suffix=".dyf", delete=False) as f:
            out_path = f.name

        try:
            titles = [f"Item {i}" for i in range(n)]
            sf = {
                "title": titles,
                "umap_x": rng.standard_normal(n).astype(np.float32),
                "umap_y": rng.standard_normal(n).astype(np.float32),
                "umap_z": rng.standard_normal(n).astype(np.float32),
            }
            write_lazy_index(tree, embeddings, path, quantization="float32", stored_fields=sf)

            with patch("dyf.enrich._labeling._call_ollama", return_value="Test Label"):
                enrich_cluster(path, output_path=out_path)

            with LazyIndex(out_path) as idx:
                level = idx.detect_enrichment_level()
                assert level >= 2
                data = idx.extract_all_fields()

                # Louvain writes community_id + per-point metrics
                assert "community_id" in data["fields"]
                assert "centroid_dist" in data["fields"]
                assert "nearest_other_dist" in data["fields"]
                assert len(data["fields"]["community_id"]) == n

                # Should have dendrogram metadata
                assert "louvain_dendrogram" in data["metadata"]
                dendro = json.loads(data["metadata"]["louvain_dendrogram"])
                assert "Z" in dendro
                assert "community_names" in dendro
                assert len(dendro["community_names"]) > 0
        finally:
            for p in (path, out_path):
                if os.path.exists(p):
                    os.unlink(p)

    def test_louvain_force_rerun(self):
        """Force re-run on already-clustered file overwrites cleanly."""
        from unittest.mock import patch

        from dyf import build_dyf_tree
        from dyf.enrich._cluster import enrich_cluster
        from dyf.lazy_index import LazyIndex, write_lazy_index

        n = 200
        embeddings = _make_clustered_embeddings(n_clusters=5, points_per_cluster=40, dim=32)
        tree = build_dyf_tree(embeddings, max_depth=3, num_bits=3, min_leaf_size=4, seed=42)

        rng = np.random.default_rng(42)

        with tempfile.NamedTemporaryFile(suffix=".dyf", delete=False) as f:
            path = f.name
        with tempfile.NamedTemporaryFile(suffix=".dyf", delete=False) as f:
            out_path = f.name

        try:
            titles = [f"Item {i}" for i in range(n)]
            sf = {
                "title": titles,
                "umap_x": rng.standard_normal(n).astype(np.float32),
                "umap_y": rng.standard_normal(n).astype(np.float32),
                "umap_z": rng.standard_normal(n).astype(np.float32),
            }
            write_lazy_index(tree, embeddings, path, quantization="float32", stored_fields=sf)

            with patch("dyf.enrich._labeling._call_ollama", return_value="Test Label"):
                enrich_cluster(path, output_path=out_path)
                # Force re-run on already-clustered file
                enrich_cluster(out_path, force=True)

            with LazyIndex(out_path) as idx:
                level = idx.detect_enrichment_level()
                assert level >= 2
                data = idx.extract_all_fields()
                assert "community_id" in data["fields"]
                assert "louvain_dendrogram" in data["metadata"]
        finally:
            for p in (path, out_path):
                if os.path.exists(p):
                    os.unlink(p)


# ── Mocked tests ─────────────────────────────────────────────────────────


class TestCallOllama:
    """Test _call_ollama with mocked urllib."""

    def test_successful_call(self):
        from unittest.mock import MagicMock, patch

        from dyf.enrich._ollama import _call_ollama

        mock_resp = MagicMock()
        mock_resp.read.return_value = json.dumps({"response": "  Test Label  "}).encode()
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=False)

        with patch("dyf.enrich._ollama.urllib.request.urlopen", return_value=mock_resp):
            result = _call_ollama("test-model", "test prompt")

        assert result == "Test Label"

    def test_connection_error(self):
        from unittest.mock import patch

        from dyf.enrich._ollama import _call_ollama

        with patch("dyf.enrich._ollama.urllib.request.urlopen", side_effect=ConnectionRefusedError("no server")):
            result = _call_ollama("test-model", "test prompt")

        assert result == ""

    def test_json_parse_error(self):
        from unittest.mock import MagicMock, patch

        from dyf.enrich._ollama import _call_ollama

        mock_resp = MagicMock()
        mock_resp.read.return_value = b"not json"
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=False)

        with patch("dyf.enrich._ollama.urllib.request.urlopen", return_value=mock_resp):
            result = _call_ollama("test-model", "test prompt")

        assert result == ""


@lazy_deps
class TestEnrichProject:
    """Test enrich_project with mocked UMAP."""

    def test_adds_umap_coords(self):
        from unittest.mock import MagicMock, patch

        from dyf import build_dyf_tree
        from dyf.enrich._project import enrich_project
        from dyf.lazy_index import LazyIndex, write_lazy_index

        n = 60
        embeddings = _make_clustered_embeddings(n_clusters=3, points_per_cluster=20, dim=16)
        tree = build_dyf_tree(embeddings, max_depth=2, num_bits=2, min_leaf_size=4, seed=42)

        with tempfile.NamedTemporaryFile(suffix=".dyf", delete=False) as f:
            path = f.name
        with tempfile.NamedTemporaryFile(suffix=".dyf", delete=False) as f:
            out_path = f.name

        try:
            write_lazy_index(tree, embeddings, path, quantization="float32")

            # Mock UMAP to return random 3D coords
            rng = np.random.default_rng(99)
            fake_coords = rng.standard_normal((n, 3)).astype(np.float32)

            mock_umap = MagicMock()
            mock_umap.fit_transform.return_value = fake_coords

            with (
                patch("dyf.enrich._project.suggest_n_neighbors", return_value=15),
                patch("dyf.enrich._project.run_umap") as mock_run_umap,
                patch("dyf.enrich._project.orient_landscape", side_effect=lambda c: c),
            ):
                mock_run_umap.return_value = fake_coords
                enrich_project(path, output_path=out_path)

            with LazyIndex(out_path) as idx:
                level = idx.detect_enrichment_level()
                assert level >= 1
                data = idx.extract_all_fields()
                assert "umap_x" in data["fields"]
                assert "umap_y" in data["fields"]
                assert "umap_z" in data["fields"]
                assert len(data["fields"]["umap_x"]) == n
        finally:
            for p in (path, out_path):
                if os.path.exists(p):
                    os.unlink(p)


@lazy_deps
class TestEnrichCluster:
    """Test enrich_cluster with mocked LLM labeling."""

    def test_adds_cluster_fields(self):
        """Default Louvain mode produces cluster fields."""
        from unittest.mock import patch

        from dyf import build_dyf_tree
        from dyf.enrich._cluster import enrich_cluster
        from dyf.lazy_index import LazyIndex, write_lazy_index

        n = 200
        embeddings = _make_clustered_embeddings(n_clusters=5, points_per_cluster=40, dim=32)
        tree = build_dyf_tree(embeddings, max_depth=3, num_bits=3, min_leaf_size=4, seed=42)

        rng = np.random.default_rng(42)

        with tempfile.NamedTemporaryFile(suffix=".dyf", delete=False) as f:
            path = f.name
        with tempfile.NamedTemporaryFile(suffix=".dyf", delete=False) as f:
            out_path = f.name

        try:
            # Create Level 1 .dyf (with UMAP coords)
            titles = [f"Item {i}" for i in range(n)]
            sf = {
                "title": titles,
                "umap_x": rng.standard_normal(n).astype(np.float32),
                "umap_y": rng.standard_normal(n).astype(np.float32),
                "umap_z": rng.standard_normal(n).astype(np.float32),
            }
            write_lazy_index(tree, embeddings, path, quantization="float32", stored_fields=sf)

            # Mock _call_ollama to return a simple label
            with patch("dyf.enrich._labeling._call_ollama", return_value="Test Cluster Label"):
                enrich_cluster(path, output_path=out_path)

            with LazyIndex(out_path) as idx:
                level = idx.detect_enrichment_level()
                assert level >= 2
                data = idx.extract_all_fields()
                # Louvain writes community_id + per-point metrics
                assert "community_id" in data["fields"]
                assert "centroid_dist" in data["fields"]
                assert "nearest_other_dist" in data["fields"]
                assert len(data["fields"]["community_id"]) == n

                # Should have dendrogram metadata with community names
                assert "louvain_dendrogram" in data["metadata"]
                dendro = json.loads(data["metadata"]["louvain_dendrogram"])
                assert len(dendro["community_names"]) > 0
        finally:
            for p in (path, out_path):
                if os.path.exists(p):
                    os.unlink(p)


@lazy_deps
class TestEnrichTree:
    """Test enrich_tree with mocked LLM."""

    def test_adds_tree_labels_metadata(self):
        from unittest.mock import patch

        from dyf import build_dyf_tree
        from dyf.enrich._tree import enrich_tree
        from dyf.lazy_index import LazyIndex, write_lazy_index

        n = 200
        embeddings = _make_clustered_embeddings(n_clusters=5, points_per_cluster=40, dim=32)
        tree = build_dyf_tree(embeddings, max_depth=4, num_bits=3, min_leaf_size=4, seed=42)

        with tempfile.NamedTemporaryFile(suffix=".dyf", delete=False) as f:
            path = f.name
        with tempfile.NamedTemporaryFile(suffix=".dyf", delete=False) as f:
            out_path = f.name

        try:
            titles = [f"Item {i}" for i in range(n)]
            write_lazy_index(tree, embeddings, path, quantization="float32", stored_fields={"title": titles})

            # Mock _call_ollama to return structured response
            def mock_ollama(model, prompt, timeout=300):
                # Parse how many groups are expected from the prompt
                import re

                groups = re.findall(r"Group (\d+):", prompt)
                if groups:
                    # It's asking for labels
                    lines = []
                    for g in groups:
                        lines.append(f"Group {g}: Test Subgroup {g}")
                    lines.append("Branch: Test Branch")
                    return "\n".join(lines)
                return "Test Label"

            with patch("dyf.enrich._tree._call_ollama", side_effect=mock_ollama):
                enrich_tree(path, target_depth=3, output_path=out_path)

            with LazyIndex(out_path) as idx:
                data = idx.extract_all_fields()
                tree_key = "tree_labels_depth_3"
                assert tree_key in data["metadata"]
                tree_data = json.loads(data["metadata"][tree_key])
                assert "branch_labels" in tree_data
                assert "child_labels" in tree_data
                assert "hierarchy" in tree_data
        finally:
            for p in (path, out_path):
                if os.path.exists(p):
                    os.unlink(p)


@lazy_deps
class TestLabelTreeBottomup:
    """Test label_tree_bottomup with mocked LLM."""

    def test_returns_labels_structure(self):
        from unittest.mock import patch

        from dyf import build_dyf_tree
        from dyf.enrich._tree import label_tree_bottomup
        from dyf.lazy_index import LazyIndex, write_lazy_index

        n = 200
        embeddings = _make_clustered_embeddings(n_clusters=5, points_per_cluster=40, dim=32)
        tree = build_dyf_tree(embeddings, max_depth=4, num_bits=3, min_leaf_size=4, seed=42)

        with tempfile.NamedTemporaryFile(suffix=".dyf", delete=False) as f:
            path = f.name

        try:
            titles = [f"Item {i}" for i in range(n)]
            write_lazy_index(tree, embeddings, path, quantization="float32", stored_fields={"title": titles})

            def mock_ollama(model, prompt, timeout=300):
                import re

                groups = re.findall(r"Group (\d+):", prompt)
                if groups:
                    lines = []
                    for g in groups:
                        lines.append(f"Group {g}: Child Label {g}")
                    lines.append("Branch: Branch Label")
                    return "\n".join(lines)
                return "Fallback"

            with patch("dyf.enrich._tree._call_ollama", side_effect=mock_ollama), LazyIndex(path) as idx:
                result = label_tree_bottomup(idx, titles, target_depth=3, samples_per_child=4, min_child_size=5)

            assert "branch_labels" in result
            assert "child_labels" in result
            assert "hierarchy" in result
            # Branch labels should contain our mock label
            for v in result["branch_labels"].values():
                assert "Branch Label" in v or "Group" in v
        finally:
            if os.path.exists(path):
                os.unlink(path)
