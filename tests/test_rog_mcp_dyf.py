"""Tests for rog_mcp.py .dyf loading and query functions.

Builds enriched .dyf files programmatically (no UMAP/Ollama) and tests
the CACHE loading and query functions in rog_mcp.
"""

import json
import os
import sys
import tempfile

import numpy as np
import pytest

from dyf import check_rust_available

# Add demo/ to path so we can import rog_mcp
_demo_dir = os.path.join(os.path.dirname(__file__), '..', 'demo')
if _demo_dir not in sys.path:
    sys.path.insert(0, os.path.abspath(_demo_dir))

pytestmark = pytest.mark.skipif(
    not check_rust_available(), reason="Rust extension not available"
)

try:
    import pyarrow  # noqa: F401
    import flatbuffers  # noqa: F401
    import websockets  # noqa: F401
    _HAS_LAZY_DEPS = True
except ImportError:
    _HAS_LAZY_DEPS = False

lazy_deps = pytest.mark.skipif(
    not _HAS_LAZY_DEPS, reason="pyarrow, flatbuffers, and websockets required"
)


def _make_clustered_embeddings(n_clusters=5, points_per_cluster=40, dim=32,
                               seed=42):
    """Create synthetic clustered embeddings on the unit sphere."""
    rng = np.random.default_rng(seed)
    centers = rng.standard_normal((n_clusters, dim)).astype(np.float32)
    centers /= np.linalg.norm(centers, axis=1, keepdims=True)

    points = []
    for i in range(n_clusters):
        noise = rng.standard_normal(
            (points_per_cluster, dim)).astype(np.float32) * 0.1
        cluster_pts = centers[i] + noise
        cluster_pts /= np.linalg.norm(cluster_pts, axis=1, keepdims=True)
        points.append(cluster_pts)

    return np.concatenate(points, axis=0)


def _build_level2_dyf(n_clusters=5, points_per_cluster=40, dim=32,
                      dual_clusters=True):
    """Build a Level 2 .dyf file with UMAP coords + cluster labels.

    Args:
        dual_clusters: If True, use cluster_25_2d/cluster_25_3d naming.
            If False, use bare cluster_25 naming (backward compat).

    Returns (path, n_total) — caller must clean up the file.
    """
    from dyf import build_dyf_tree
    from dyf.lazy_index import write_lazy_index, rewrite_lazy_index

    n = n_clusters * points_per_cluster
    embeddings = _make_clustered_embeddings(
        n_clusters=n_clusters, points_per_cluster=points_per_cluster, dim=dim)
    tree = build_dyf_tree(embeddings, max_depth=3, num_bits=3,
                          min_leaf_size=4, seed=42)

    rng = np.random.default_rng(42)

    # Stored fields for Level 1+2
    titles = [f"Test Item {i}" for i in range(n)]
    umap_x = rng.standard_normal(n).astype(np.float32)
    umap_y = rng.standard_normal(n).astype(np.float32)
    umap_z = rng.standard_normal(n).astype(np.float32)

    # Assign clusters based on true cluster membership
    cluster_labels = np.array(
        [i // points_per_cluster for i in range(n)], dtype=np.int32)

    # Build cluster names and centroids
    cluster_names = {str(i): f"Cluster {i}" for i in range(n_clusters)}
    cluster_centroids = {}
    for cid in range(n_clusters):
        mask = cluster_labels == cid
        cluster_centroids[str(cid)] = [
            round(float(umap_x[mask].mean()), 4),
            round(float(umap_y[mask].mean()), 4),
            round(float(umap_z[mask].mean()), 4),
        ]

    sf = {
        'title': titles,
        'umap_x': umap_x,
        'umap_y': umap_y,
        'umap_z': umap_z,
    }
    meta = {}

    if dual_clusters:
        # New dual naming: cluster_25_2d and cluster_25_3d
        sf['cluster_25_2d'] = cluster_labels
        sf['cluster_25_3d'] = cluster_labels  # same for test simplicity
        meta['cluster_names_25_2d'] = json.dumps(cluster_names)
        meta['cluster_centroids_25_2d'] = json.dumps(cluster_centroids)
        meta['cluster_names_25_3d'] = json.dumps(cluster_names)
        meta['cluster_centroids_25_3d'] = json.dumps(cluster_centroids)
    else:
        # Bare naming (backward compat)
        sf['cluster_25'] = cluster_labels
        meta['cluster_names_25'] = json.dumps(cluster_names)
        meta['cluster_centroids_25'] = json.dumps(cluster_centroids)

    with tempfile.NamedTemporaryFile(suffix='.dyf', delete=False) as f:
        path = f.name

    write_lazy_index(tree, embeddings, path, quantization='float32',
                     stored_fields=sf, metadata=meta)

    return path, n


@lazy_deps
class TestLoadCacheFromDyf:
    """Test _load_cache_from_dyf populates CACHE correctly."""

    def test_cache_populated(self):
        import rog_mcp

        path, n = _build_level2_dyf()
        try:
            # Save and restore module globals
            old_cache = rog_mcp.CACHE
            old_dyf = rog_mcp.DYF_INDEX
            try:
                rog_mcp.CACHE = None
                rog_mcp.DYF_INDEX = None
                rog_mcp.load_cache(path)

                assert rog_mcp.CACHE is not None
                assert len(rog_mcp.CACHE['titles']) == n
                assert rog_mcp.CACHE['coords_2d'].shape == (n, 2)
            finally:
                rog_mcp.CACHE = old_cache
                rog_mcp.DYF_INDEX = old_dyf
        finally:
            if os.path.exists(path):
                os.unlink(path)

    def test_cluster_result_loaded(self):
        import rog_mcp

        path, n = _build_level2_dyf()
        try:
            old_cache = rog_mcp.CACHE
            old_dyf = rog_mcp.DYF_INDEX
            try:
                rog_mcp.CACHE = None
                rog_mcp.DYF_INDEX = None
                rog_mcp.load_cache(path)

                cr = rog_mcp.CACHE['cluster_result']
                assert 25 in cr['labels']
                assert 25 in cr['names']
                assert 25 in cr['centroids']
                assert len(cr['labels'][25]) == n
            finally:
                rog_mcp.CACHE = old_cache
                rog_mcp.DYF_INDEX = old_dyf
        finally:
            if os.path.exists(path):
                os.unlink(path)

    def test_titles_are_strings(self):
        import rog_mcp

        path, n = _build_level2_dyf()
        try:
            old_cache = rog_mcp.CACHE
            old_dyf = rog_mcp.DYF_INDEX
            try:
                rog_mcp.CACHE = None
                rog_mcp.DYF_INDEX = None
                rog_mcp.load_cache(path)

                titles = rog_mcp.CACHE['titles']
                assert isinstance(titles, list)
                assert isinstance(titles[0], str)
                assert titles[0] == "Test Item 0"
            finally:
                rog_mcp.CACHE = old_cache
                rog_mcp.DYF_INDEX = old_dyf
        finally:
            if os.path.exists(path):
                os.unlink(path)


@lazy_deps
class TestQueryFunctions:
    """Test search_points, get_cluster_info, get_neighbors on loaded CACHE."""

    @pytest.fixture(autouse=True)
    def _load_cache(self):
        """Load a Level 2 .dyf into rog_mcp.CACHE for all tests."""
        import rog_mcp

        self.path, self.n = _build_level2_dyf()
        self._old_cache = rog_mcp.CACHE
        self._old_dyf = rog_mcp.DYF_INDEX
        rog_mcp.CACHE = None
        rog_mcp.DYF_INDEX = None
        rog_mcp.load_cache(self.path)
        yield
        rog_mcp.CACHE = self._old_cache
        rog_mcp.DYF_INDEX = self._old_dyf
        if os.path.exists(self.path):
            os.unlink(self.path)

    def test_search_points_by_title(self):
        import rog_mcp

        results = rog_mcp.search_points("Test Item 1", limit=5)
        assert len(results) > 0
        assert all('title' in r for r in results)
        assert all('x' in r for r in results)
        # Should find "Test Item 1", "Test Item 10", etc.
        assert any("Test Item 1" in r['title'] for r in results)

    def test_search_points_case_insensitive(self):
        import rog_mcp

        results = rog_mcp.search_points("test item 0", limit=5)
        assert len(results) > 0

    def test_search_points_no_match(self):
        import rog_mcp

        results = rog_mcp.search_points("zzz_nonexistent_zzz", limit=5)
        assert len(results) == 0

    def test_get_cluster_info(self):
        import rog_mcp

        clusters = rog_mcp.get_cluster_info(level=25)
        assert len(clusters) > 0
        assert all('cluster_id' in c for c in clusters)
        assert all('name' in c for c in clusters)
        assert all('count' in c for c in clusters)
        # Total count should equal n
        total = sum(c['count'] for c in clusters)
        assert total == self.n

    def test_get_cluster_info_invalid_level(self):
        import rog_mcp

        result = rog_mcp.get_cluster_info(level=99)
        assert len(result) == 1
        assert 'error' in result[0]

    def test_get_neighbors(self):
        import rog_mcp

        neighbors = rog_mcp.get_neighbors(index=0, k=5)
        assert len(neighbors) == 5
        assert all('index' in n for n in neighbors)
        assert all('distance' in n for n in neighbors)
        # Distances should be sorted
        dists = [n['distance'] for n in neighbors]
        assert dists == sorted(dists)

    def test_get_neighbors_invalid_index(self):
        import rog_mcp

        result = rog_mcp.get_neighbors(index=999999, k=5)
        assert len(result) == 1
        assert 'error' in result[0]

    def test_get_cluster_members(self):
        import rog_mcp

        members = rog_mcp.get_cluster_members(cluster_id=0, level=25, limit=10)
        assert len(members) > 0
        assert len(members) <= 10
        assert all('title' in m for m in members)

    def test_get_points_in_region(self):
        import rog_mcp

        # Use a very large bounding box to ensure we get results
        result = rog_mcp.get_points_in_region(
            x_min=-1000, x_max=1000, y_min=-1000, y_max=1000, limit=10)
        assert len(result) > 0
        assert len(result) <= 10


@lazy_deps
class TestDualClusterLoading:
    """Test loading dual 2D/3D cluster fields from .dyf."""

    def test_dual_clusters_loaded(self):
        """cluster_25_2d + cluster_25_3d → labels, labels_2d, labels_3d."""
        import rog_mcp

        path, n = _build_level2_dyf(dual_clusters=True)
        try:
            old_cache = rog_mcp.CACHE
            old_dyf = rog_mcp.DYF_INDEX
            try:
                rog_mcp.CACHE = None
                rog_mcp.DYF_INDEX = None
                rog_mcp.load_cache(path)

                cr = rog_mcp.CACHE['cluster_result']
                # Default labels should be populated (from 2D)
                assert 25 in cr['labels']
                assert len(cr['labels'][25]) == n
                # 2D and 3D should both be populated
                assert 25 in cr['labels_2d']
                assert 25 in cr['labels_3d']
                assert 25 in cr['names_2d']
                assert 25 in cr['names_3d']
            finally:
                rog_mcp.CACHE = old_cache
                rog_mcp.DYF_INDEX = old_dyf
        finally:
            if os.path.exists(path):
                os.unlink(path)


@lazy_deps
class TestBackwardCompatBareClusters:
    """Test backward compat with bare cluster_25 (no _2d/_3d suffix)."""

    def test_bare_cluster_populates_both(self):
        """Bare cluster_25 → labels, labels_2d, labels_3d all populated."""
        import rog_mcp

        path, n = _build_level2_dyf(dual_clusters=False)
        try:
            old_cache = rog_mcp.CACHE
            old_dyf = rog_mcp.DYF_INDEX
            try:
                rog_mcp.CACHE = None
                rog_mcp.DYF_INDEX = None
                rog_mcp.load_cache(path)

                cr = rog_mcp.CACHE['cluster_result']
                # All three should be populated from bare cluster_25
                assert 25 in cr['labels']
                assert 25 in cr['labels_2d']
                assert 25 in cr['labels_3d']
                assert len(cr['labels'][25]) == n

                # Names should be identical across all three
                assert cr['names'][25] == cr['names_2d'][25]
                assert cr['names'][25] == cr['names_3d'][25]
            finally:
                rog_mcp.CACHE = old_cache
                rog_mcp.DYF_INDEX = old_dyf
        finally:
            if os.path.exists(path):
                os.unlink(path)

    def test_query_functions_work_with_bare(self):
        """search_points and get_cluster_info work with bare cluster naming."""
        import rog_mcp

        path, n = _build_level2_dyf(dual_clusters=False)
        try:
            old_cache = rog_mcp.CACHE
            old_dyf = rog_mcp.DYF_INDEX
            try:
                rog_mcp.CACHE = None
                rog_mcp.DYF_INDEX = None
                rog_mcp.load_cache(path)

                results = rog_mcp.search_points("Test Item 0", limit=5)
                assert len(results) > 0

                clusters = rog_mcp.get_cluster_info(level=25)
                assert len(clusters) > 0
                total = sum(c['count'] for c in clusters)
                assert total == n
            finally:
                rog_mcp.CACHE = old_cache
                rog_mcp.DYF_INDEX = old_dyf
        finally:
            if os.path.exists(path):
                os.unlink(path)


@lazy_deps
class TestTreeLabelFallback:
    """Test tree-label fallback when no BIRCH clusters exist."""

    def test_tree_labels_create_clusters(self):
        """Level 0 + tree_labels metadata → cluster assignments."""
        import rog_mcp
        from dyf import build_dyf_tree
        from dyf.lazy_index import write_lazy_index

        n = 200
        embeddings = _make_clustered_embeddings(
            n_clusters=5, points_per_cluster=40, dim=32)
        tree = build_dyf_tree(embeddings, max_depth=4, num_bits=3,
                              min_leaf_size=4, seed=42)

        rng = np.random.default_rng(42)

        # Build Level 1 .dyf (UMAP coords but NO cluster fields)
        sf = {
            'title': [f"Item {i}" for i in range(n)],
            'umap_x': rng.standard_normal(n).astype(np.float32),
            'umap_y': rng.standard_normal(n).astype(np.float32),
            'umap_z': rng.standard_normal(n).astype(np.float32),
        }

        # Build fake tree labels metadata
        # We need node IDs from the actual tree structure
        from dyf.lazy_index import LazyIndex

        with tempfile.NamedTemporaryFile(suffix='.dyf', delete=False) as f:
            path = f.name

        try:
            write_lazy_index(tree, embeddings, path, quantization='float32',
                             stored_fields=sf)

            # Get actual tree structure to build valid tree labels
            with LazyIndex(path) as idx:
                tree_struct = idx.get_tree_structure()

            # Find nodes at depth 3
            depth3_nodes = [nd for nd in tree_struct
                           if nd['depth'] == 3 and nd['num_items'] > 0]

            if not depth3_nodes:
                # Try depth 2 if tree isn't deep enough
                depth3_nodes = [nd for nd in tree_struct
                               if nd['depth'] == 2 and nd['num_items'] > 0]

            if depth3_nodes:
                # Build tree labels from actual children
                children_of = {}
                for nd in tree_struct:
                    if nd['parent_id'] is not None:
                        children_of.setdefault(
                            nd['parent_id'], []).append(nd['node_id'])

                branch_labels = {}
                child_labels = {}
                hierarchy = {}

                for node in depth3_nodes[:3]:  # just use first 3
                    nid = node['node_id']
                    branch_labels[str(nid)] = f"Branch {nid}"
                    kids = children_of.get(nid, [])
                    hierarchy[str(nid)] = kids
                    for kid in kids:
                        child_labels[str(kid)] = f"Child {kid}"

                tree_label_data = {
                    'branch_labels': branch_labels,
                    'child_labels': child_labels,
                    'hierarchy': hierarchy,
                }

                # Rewrite with tree labels metadata (no cluster fields)
                from dyf.lazy_index import rewrite_lazy_index
                rewrite_lazy_index(
                    path,
                    new_metadata={
                        'tree_labels_depth_3': json.dumps(tree_label_data)
                    })

                # Now load via rog_mcp
                old_cache = rog_mcp.CACHE
                old_dyf = rog_mcp.DYF_INDEX
                try:
                    rog_mcp.CACHE = None
                    rog_mcp.DYF_INDEX = None
                    rog_mcp.load_cache(path)

                    cr = rog_mcp.CACHE['cluster_result']
                    # Should have built cluster labels from tree
                    assert len(cr['labels']) > 0
                    # All standard levels should be populated
                    for lvl in cr['labels']:
                        labels_arr = cr['labels'][lvl]
                        assert len(labels_arr) == n
                        # No -1 labels (all assigned)
                        assert (labels_arr >= 0).all()
                finally:
                    rog_mcp.CACHE = old_cache
                    rog_mcp.DYF_INDEX = old_dyf
        finally:
            if os.path.exists(path):
                os.unlink(path)
