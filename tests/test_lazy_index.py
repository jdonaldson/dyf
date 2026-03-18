"""Tests for DYF lazy index (FlatBuffers + Arrow IPC)."""

import os
import tempfile
import time

import numpy as np
import pytest

from dyf import check_rust_available

# Skip entire module if Rust extension not available
pytestmark = pytest.mark.skipif(
    not check_rust_available(), reason="Rust extension not available"
)

try:
    import pyarrow  # noqa: F401
    import flatbuffers  # noqa: F401
    _HAS_LAZY_DEPS = True
except ImportError:
    _HAS_LAZY_DEPS = False

lazy_deps = pytest.mark.skipif(
    not _HAS_LAZY_DEPS, reason="pyarrow and flatbuffers required"
)


def _make_clustered_embeddings(n_clusters=5, points_per_cluster=40, dim=32,
                               seed=42):
    """Create synthetic clustered embeddings on the unit sphere.

    Returns:
        embeddings: (n, dim) float32 array, L2-normalized.
    """
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


@lazy_deps
class TestWriteAndLoad:
    """Round-trip: build tree -> write index -> load LazyIndex -> search."""

    @pytest.fixture
    def index_data(self):
        """Build a DYF tree and write a lazy index file."""
        from dyf import build_dyf_tree
        from dyf.lazy_index import write_lazy_index, LazyIndex

        embeddings = _make_clustered_embeddings(
            n_clusters=5, points_per_cluster=40, dim=32, seed=42)
        tree = build_dyf_tree(embeddings, max_depth=3, num_bits=3,
                              min_leaf_size=4, seed=42)

        with tempfile.NamedTemporaryFile(suffix='.dyf', delete=False) as f:
            path = f.name

        try:
            write_lazy_index(tree, embeddings, path, compression='zstd',
                             quantization='float16',
                             metadata={'test': 'true'})
            yield {
                'path': path,
                'embeddings': embeddings,
                'tree': tree,
            }
        finally:
            if os.path.exists(path):
                os.unlink(path)

    def test_file_created(self, index_data):
        """Index file is created and has non-zero size."""
        path = index_data['path']
        assert os.path.exists(path)
        assert os.path.getsize(path) > 0

    def test_magic_header(self, index_data):
        """File starts with DYF1 magic bytes."""
        with open(index_data['path'], 'rb') as f:
            magic = f.read(4)
        assert magic == b'DYF1'

    def test_tree_summary(self, index_data):
        """LazyIndex.tree_summary returns correct metadata."""
        from dyf.lazy_index import LazyIndex

        with LazyIndex(index_data['path']) as idx:
            summary = idx.tree_summary
            assert summary['embedding_dim'] == 32
            assert summary['total_items'] == 200
            assert summary['num_leaves'] > 0
            assert summary['version'] == '1.0'
            assert summary['build_params']['quantization'] == 'float16'
            assert summary['build_params']['compression'] == 'zstd'

    def test_search_returns_results(self, index_data):
        """Search returns indices and scores."""
        from dyf.lazy_index import LazyIndex

        embeddings = index_data['embeddings']
        query = embeddings[0]

        with LazyIndex(index_data['path']) as idx:
            indices, scores = idx.search(query, k=10, nprobe=3)
            assert len(indices) > 0
            assert len(indices) == len(scores)
            assert len(indices) <= 10
            # Scores should be sorted descending
            for i in range(len(scores) - 1):
                assert scores[i] >= scores[i + 1]

    def test_search_finds_query_itself(self, index_data):
        """Searching with an existing embedding should find itself."""
        from dyf.lazy_index import LazyIndex

        embeddings = index_data['embeddings']
        query = embeddings[42]

        with LazyIndex(index_data['path']) as idx:
            indices, scores = idx.search(query, k=10, nprobe=5)
            # The query itself should be in results with high similarity
            assert 42 in indices
            # The score for the query should be very high
            query_pos = np.where(indices == 42)[0][0]
            assert scores[query_pos] > 0.9

    def test_search_matches_brute_force(self, index_data):
        """LazyIndex search should find most of the true top-k."""
        from dyf.lazy_index import LazyIndex

        embeddings = index_data['embeddings']
        query = embeddings[0]

        # Brute force top-10
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        emb_n = embeddings / np.maximum(norms, 1e-10)
        qnorm = np.linalg.norm(query)
        q_n = query / max(qnorm, 1e-10)
        bf_scores = emb_n @ q_n
        bf_top = np.argsort(-bf_scores)[:10]

        with LazyIndex(index_data['path']) as idx:
            li_indices, li_scores = idx.search(query, k=10, nprobe=5)

        # At least 30% overlap with brute force (approximate + quantized)
        overlap = len(set(bf_top.tolist()) & set(li_indices.tolist()))
        assert overlap >= 3, f"Only {overlap}/10 overlap with brute force"

    def test_get_leaf(self, index_data):
        """get_leaf returns a valid Arrow RecordBatch."""
        from dyf.lazy_index import LazyIndex

        with LazyIndex(index_data['path']) as idx:
            batch = idx.get_leaf(0)
            assert batch is not None
            assert 'item_index' in batch.schema.names
            assert 'embedding' in batch.schema.names
            assert batch.num_rows > 0


@lazy_deps
class TestQuantization:
    """Test different quantization modes."""

    @pytest.fixture(params=['float32', 'float16'])
    def quant_index(self, request):
        """Build index with different quantization."""
        from dyf import build_dyf_tree
        from dyf.lazy_index import write_lazy_index

        quantization = request.param
        embeddings = _make_clustered_embeddings(
            n_clusters=3, points_per_cluster=30, dim=16, seed=42)
        tree = build_dyf_tree(embeddings, max_depth=2, num_bits=2,
                              min_leaf_size=4, seed=42)

        with tempfile.NamedTemporaryFile(suffix='.dyf', delete=False) as f:
            path = f.name

        try:
            write_lazy_index(tree, embeddings, path, compression='zstd',
                             quantization=quantization)
            yield {
                'path': path,
                'embeddings': embeddings,
                'quantization': quantization,
            }
        finally:
            if os.path.exists(path):
                os.unlink(path)

    def test_quantization_preserves_ranking(self, quant_index):
        """Quantized index should preserve similarity ranking."""
        from dyf.lazy_index import LazyIndex

        embeddings = quant_index['embeddings']
        query = embeddings[0]

        with LazyIndex(quant_index['path']) as idx:
            indices, scores = idx.search(query, k=5, nprobe=3)
            assert len(indices) > 0
            # Self should still be found
            assert 0 in indices


@lazy_deps
class TestCaching:
    """Test LRU cache behavior."""

    def test_second_search_uses_cache(self):
        """Second search on same leaves should hit cache."""
        from dyf import build_dyf_tree
        from dyf.lazy_index import write_lazy_index, LazyIndex

        embeddings = _make_clustered_embeddings(
            n_clusters=3, points_per_cluster=20, dim=16, seed=42)
        tree = build_dyf_tree(embeddings, max_depth=2, num_bits=2,
                              min_leaf_size=4, seed=42)

        with tempfile.NamedTemporaryFile(suffix='.dyf', delete=False) as f:
            path = f.name

        try:
            write_lazy_index(tree, embeddings, path)

            with LazyIndex(path) as idx:
                query = embeddings[0]
                # First search: cold cache
                t0 = time.perf_counter()
                idx.search(query, k=5, nprobe=1)
                t1 = time.perf_counter()
                cold_time = t1 - t0

                # Second search: warm cache
                t2 = time.perf_counter()
                idx.search(query, k=5, nprobe=1)
                t3 = time.perf_counter()
                warm_time = t3 - t2

                # Cache should be populated
                assert len(idx._batch_cache) > 0
        finally:
            if os.path.exists(path):
                os.unlink(path)


@lazy_deps
class TestEdgeCases:
    """Edge cases: single leaf, nprobe > num_leaves."""

    def test_single_leaf_tree(self):
        """Tree with a single leaf (no splits)."""
        from dyf import build_dyf_tree
        from dyf.lazy_index import write_lazy_index, LazyIndex

        # Small dataset that won't split
        rng = np.random.default_rng(42)
        embeddings = rng.standard_normal((10, 8)).astype(np.float32)
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings = embeddings / norms

        tree = build_dyf_tree(embeddings, max_depth=1, num_bits=2,
                              min_leaf_size=20, seed=42)

        with tempfile.NamedTemporaryFile(suffix='.dyf', delete=False) as f:
            path = f.name

        try:
            write_lazy_index(tree, embeddings, path)

            with LazyIndex(path) as idx:
                assert idx.num_leaves >= 1
                indices, scores = idx.search(embeddings[0], k=5, nprobe=1)
                assert len(indices) > 0
        finally:
            if os.path.exists(path):
                os.unlink(path)

    def test_nprobe_greater_than_leaves(self):
        """nprobe > num_leaves should still work."""
        from dyf import build_dyf_tree
        from dyf.lazy_index import write_lazy_index, LazyIndex

        embeddings = _make_clustered_embeddings(
            n_clusters=2, points_per_cluster=20, dim=8, seed=42)
        tree = build_dyf_tree(embeddings, max_depth=1, num_bits=2,
                              min_leaf_size=4, seed=42)

        with tempfile.NamedTemporaryFile(suffix='.dyf', delete=False) as f:
            path = f.name

        try:
            write_lazy_index(tree, embeddings, path)

            with LazyIndex(path) as idx:
                # nprobe=100 but likely far fewer leaves
                indices, scores = idx.search(embeddings[0], k=5, nprobe=100)
                assert len(indices) > 0
        finally:
            if os.path.exists(path):
                os.unlink(path)

    def test_no_compression(self):
        """Write and read with no compression."""
        from dyf import build_dyf_tree
        from dyf.lazy_index import write_lazy_index, LazyIndex

        embeddings = _make_clustered_embeddings(
            n_clusters=2, points_per_cluster=20, dim=8, seed=42)
        tree = build_dyf_tree(embeddings, max_depth=2, num_bits=2,
                              min_leaf_size=4, seed=42)

        with tempfile.NamedTemporaryFile(suffix='.dyf', delete=False) as f:
            path = f.name

        try:
            write_lazy_index(tree, embeddings, path, compression='none')

            with LazyIndex(path) as idx:
                indices, scores = idx.search(embeddings[0], k=5, nprobe=2)
                assert len(indices) > 0
        finally:
            if os.path.exists(path):
                os.unlink(path)

    def test_query_dimension_mismatch(self):
        """Search with wrong-dimension query should raise."""
        from dyf import build_dyf_tree
        from dyf.lazy_index import write_lazy_index, LazyIndex

        embeddings = _make_clustered_embeddings(
            n_clusters=2, points_per_cluster=20, dim=8, seed=42)
        tree = build_dyf_tree(embeddings, max_depth=2, num_bits=2,
                              min_leaf_size=4, seed=42)

        with tempfile.NamedTemporaryFile(suffix='.dyf', delete=False) as f:
            path = f.name

        try:
            write_lazy_index(tree, embeddings, path)

            with LazyIndex(path) as idx:
                bad_query = np.ones(16, dtype=np.float32)
                with pytest.raises(ValueError, match="dim=8"):
                    idx.search(bad_query, k=5)
        finally:
            if os.path.exists(path):
                os.unlink(path)


@lazy_deps
class TestMmap:
    """Verify the file is memory-mapped, not fully read."""

    def test_mmap_used(self):
        """LazyIndex should use mmap, not read full file."""
        from dyf import build_dyf_tree
        from dyf.lazy_index import write_lazy_index, LazyIndex

        embeddings = _make_clustered_embeddings(
            n_clusters=5, points_per_cluster=100, dim=64, seed=42)
        tree = build_dyf_tree(embeddings, max_depth=3, num_bits=3,
                              min_leaf_size=4, seed=42)

        with tempfile.NamedTemporaryFile(suffix='.dyf', delete=False) as f:
            path = f.name

        try:
            write_lazy_index(tree, embeddings, path)
            file_size = os.path.getsize(path)
            assert file_size > 1000  # Sanity check: file is non-trivial

            with LazyIndex(path) as idx:
                # mmap object should exist
                assert idx._mm is not None
                assert len(idx._mm) == file_size
                # Tree metadata should be accessible without decompressing leaves
                summary = idx.tree_summary
                assert summary['total_items'] == 500
        finally:
            if os.path.exists(path):
                os.unlink(path)


@lazy_deps
class TestExtractAllFields:
    """Test extract_all_fields round-trip."""

    def test_extract_embeddings(self):
        """Extracted embeddings match original (within float16 precision)."""
        from dyf import build_dyf_tree
        from dyf.lazy_index import write_lazy_index, LazyIndex

        embeddings = _make_clustered_embeddings(
            n_clusters=3, points_per_cluster=20, dim=16, seed=42)
        tree = build_dyf_tree(embeddings, max_depth=2, num_bits=2,
                              min_leaf_size=4, seed=42)

        with tempfile.NamedTemporaryFile(suffix='.dyf', delete=False) as f:
            path = f.name

        try:
            write_lazy_index(tree, embeddings, path, quantization='float16')

            with LazyIndex(path) as idx:
                data = idx.extract_all_fields()
                assert data['embeddings'].shape == embeddings.shape
                # float16 round-trip: should be close but not exact
                assert np.allclose(data['embeddings'], embeddings, atol=0.01)
                assert data['fields'] == {}
                assert 'stored_fields' not in data['metadata']
        finally:
            if os.path.exists(path):
                os.unlink(path)

    def test_extract_with_stored_fields(self):
        """Stored fields are correctly extracted and sorted by item_index."""
        from dyf import build_dyf_tree
        from dyf.lazy_index import write_lazy_index, LazyIndex

        n = 60
        embeddings = _make_clustered_embeddings(
            n_clusters=3, points_per_cluster=20, dim=16, seed=42)
        tree = build_dyf_tree(embeddings, max_depth=2, num_bits=2,
                              min_leaf_size=4, seed=42)

        titles = [f"Item {i}" for i in range(n)]
        scores = np.arange(n, dtype=np.float32) * 0.1
        labels = np.array([i % 3 for i in range(n)], dtype=np.int32)

        with tempfile.NamedTemporaryFile(suffix='.dyf', delete=False) as f:
            path = f.name

        try:
            write_lazy_index(
                tree, embeddings, path, quantization='float32',
                stored_fields={'title': titles, 'score': scores,
                               'label': labels})

            with LazyIndex(path) as idx:
                data = idx.extract_all_fields()

                # Verify stored fields
                assert set(data['fields'].keys()) == {'title', 'score', 'label'}

                # Titles should be in order
                for i in range(n):
                    assert data['fields']['title'][i] == f"Item {i}"

                # Numeric fields should round-trip exactly (float32)
                assert np.array_equal(data['fields']['score'], scores)
                assert np.array_equal(data['fields']['label'], labels)

                # Embeddings should be exact for float32
                assert np.allclose(data['embeddings'], embeddings, atol=1e-6)
        finally:
            if os.path.exists(path):
                os.unlink(path)


@lazy_deps
class TestRewriteLazyIndex:
    """Test rewrite_lazy_index for adding stored fields and metadata."""

    def test_add_stored_fields(self):
        """Adding new stored fields preserves existing data."""
        from dyf import build_dyf_tree
        from dyf.lazy_index import write_lazy_index, rewrite_lazy_index, LazyIndex

        n = 60
        embeddings = _make_clustered_embeddings(
            n_clusters=3, points_per_cluster=20, dim=16, seed=42)
        tree = build_dyf_tree(embeddings, max_depth=2, num_bits=2,
                              min_leaf_size=4, seed=42)

        titles = [f"Item {i}" for i in range(n)]

        with tempfile.NamedTemporaryFile(suffix='.dyf', delete=False) as f:
            path = f.name
        with tempfile.NamedTemporaryFile(suffix='.dyf', delete=False) as f:
            out_path = f.name

        try:
            # Write initial index with titles
            write_lazy_index(
                tree, embeddings, path, quantization='float32',
                stored_fields={'title': titles},
                metadata={'source': 'test'})

            # Add new stored fields via rewrite
            umap_x = np.random.default_rng(42).standard_normal(n).astype(np.float32)
            umap_y = np.random.default_rng(43).standard_normal(n).astype(np.float32)
            cluster_labels = np.array([i % 5 for i in range(n)], dtype=np.int32)

            rewrite_lazy_index(
                path,
                new_stored_fields={
                    'umap_x': umap_x,
                    'umap_y': umap_y,
                    'cluster_25': cluster_labels,
                },
                new_metadata={'umap_n_neighbors': '15'},
                output_path=out_path,
            )

            # Read back and verify
            with LazyIndex(out_path) as idx:
                data = idx.extract_all_fields()

                # Original title field preserved
                for i in range(n):
                    assert data['fields']['title'][i] == f"Item {i}"

                # New fields present
                assert np.allclose(data['fields']['umap_x'], umap_x, atol=1e-6)
                assert np.allclose(data['fields']['umap_y'], umap_y, atol=1e-6)
                assert np.array_equal(data['fields']['cluster_25'],
                                      cluster_labels)

                # Original metadata preserved
                assert data['metadata']['source'] == 'test'
                # New metadata added
                assert data['metadata']['umap_n_neighbors'] == '15'

                # Embeddings preserved
                assert np.allclose(data['embeddings'], embeddings, atol=1e-6)
        finally:
            for p in (path, out_path):
                if os.path.exists(p):
                    os.unlink(p)

    def test_rewrite_preserves_search(self):
        """Search quality is preserved after rewrite."""
        from dyf import build_dyf_tree
        from dyf.lazy_index import write_lazy_index, rewrite_lazy_index, LazyIndex

        n = 60
        embeddings = _make_clustered_embeddings(
            n_clusters=3, points_per_cluster=20, dim=16, seed=42)
        tree = build_dyf_tree(embeddings, max_depth=2, num_bits=2,
                              min_leaf_size=4, seed=42)

        with tempfile.NamedTemporaryFile(suffix='.dyf', delete=False) as f:
            path = f.name
        with tempfile.NamedTemporaryFile(suffix='.dyf', delete=False) as f:
            out_path = f.name

        try:
            write_lazy_index(tree, embeddings, path, quantization='float32')

            # Search before rewrite
            with LazyIndex(path) as idx:
                query = embeddings[0]
                result_before = idx.search(query, k=10, nprobe=5)

            # Rewrite with new fields
            scores = np.arange(n, dtype=np.float32)
            rewrite_lazy_index(
                path,
                new_stored_fields={'score': scores},
                output_path=out_path,
            )

            # Search after rewrite
            with LazyIndex(out_path) as idx:
                result_after = idx.search(query, k=10, nprobe=5)

                # Same indices should be found (tree structure preserved)
                overlap = len(set(result_before.indices.tolist())
                              & set(result_after.indices.tolist()))
                assert overlap >= 8, (
                    f"Only {overlap}/10 overlap after rewrite")

                # New stored field should be in search results
                assert 'score' in result_after.fields
        finally:
            for p in (path, out_path):
                if os.path.exists(p):
                    os.unlink(p)

    def test_rewrite_in_place(self):
        """Rewrite without output_path overwrites the original file."""
        from dyf import build_dyf_tree
        from dyf.lazy_index import write_lazy_index, rewrite_lazy_index, LazyIndex

        n = 60
        embeddings = _make_clustered_embeddings(
            n_clusters=3, points_per_cluster=20, dim=16, seed=42)
        tree = build_dyf_tree(embeddings, max_depth=2, num_bits=2,
                              min_leaf_size=4, seed=42)

        with tempfile.NamedTemporaryFile(suffix='.dyf', delete=False) as f:
            path = f.name

        try:
            write_lazy_index(tree, embeddings, path, quantization='float32')

            rewrite_lazy_index(
                path,
                new_metadata={'enriched': 'true'},
            )

            with LazyIndex(path) as idx:
                meta = idx._get_metadata()
                assert meta['enriched'] == 'true'
        finally:
            if os.path.exists(path):
                os.unlink(path)


    def test_rewrite_drop_fields(self):
        """drop_fields removes specified stored fields during rewrite."""
        from dyf import build_dyf_tree
        from dyf.lazy_index import write_lazy_index, rewrite_lazy_index, LazyIndex

        n = 60
        embeddings = _make_clustered_embeddings(
            n_clusters=3, points_per_cluster=20, dim=16, seed=42)
        tree = build_dyf_tree(embeddings, max_depth=2, num_bits=2,
                              min_leaf_size=4, seed=42)

        rng = np.random.default_rng(42)
        sf = {
            'field_a': rng.standard_normal(n).astype(np.float32),
            'field_b': rng.standard_normal(n).astype(np.float32),
            'field_c': rng.standard_normal(n).astype(np.float32),
        }

        with tempfile.NamedTemporaryFile(suffix='.dyf', delete=False) as f:
            path = f.name
        with tempfile.NamedTemporaryFile(suffix='.dyf', delete=False) as f:
            out_path = f.name

        try:
            write_lazy_index(tree, embeddings, path, quantization='float32',
                             stored_fields=sf)

            rewrite_lazy_index(
                path,
                drop_fields={'field_b'},
                output_path=out_path,
            )

            with LazyIndex(out_path) as idx:
                data = idx.extract_all_fields()
                assert 'field_a' in data['fields']
                assert 'field_b' not in data['fields']
                assert 'field_c' in data['fields']
                assert np.allclose(data['fields']['field_a'],
                                   sf['field_a'], atol=1e-6)
                assert np.allclose(data['fields']['field_c'],
                                   sf['field_c'], atol=1e-6)
        finally:
            for p in (path, out_path):
                if os.path.exists(p):
                    os.unlink(p)

    def test_rewrite_drop_and_add_fields(self):
        """drop_fields applies after merge so new fields can replace old."""
        from dyf import build_dyf_tree
        from dyf.lazy_index import write_lazy_index, rewrite_lazy_index, LazyIndex

        n = 60
        embeddings = _make_clustered_embeddings(
            n_clusters=3, points_per_cluster=20, dim=16, seed=42)
        tree = build_dyf_tree(embeddings, max_depth=2, num_bits=2,
                              min_leaf_size=4, seed=42)

        rng = np.random.default_rng(42)
        sf = {
            'old_field': rng.standard_normal(n).astype(np.float32),
            'keep_field': rng.standard_normal(n).astype(np.float32),
        }
        new_field = rng.standard_normal(n).astype(np.float32)

        with tempfile.NamedTemporaryFile(suffix='.dyf', delete=False) as f:
            path = f.name

        try:
            write_lazy_index(tree, embeddings, path, quantization='float32',
                             stored_fields=sf)

            rewrite_lazy_index(
                path,
                new_stored_fields={'new_field': new_field},
                drop_fields={'old_field'},
            )

            with LazyIndex(path) as idx:
                data = idx.extract_all_fields()
                assert 'old_field' not in data['fields']
                assert 'keep_field' in data['fields']
                assert 'new_field' in data['fields']
                assert np.allclose(data['fields']['new_field'],
                                   new_field, atol=1e-6)
        finally:
            if os.path.exists(path):
                os.unlink(path)


@lazy_deps
class TestDetectEnrichmentLevel:
    """Test enrichment level detection."""

    def test_level_0_base(self):
        """Base index (no enrichment) returns level 0."""
        from dyf import build_dyf_tree
        from dyf.lazy_index import write_lazy_index, LazyIndex

        embeddings = _make_clustered_embeddings(
            n_clusters=2, points_per_cluster=20, dim=8, seed=42)
        tree = build_dyf_tree(embeddings, max_depth=2, num_bits=2,
                              min_leaf_size=4, seed=42)

        with tempfile.NamedTemporaryFile(suffix='.dyf', delete=False) as f:
            path = f.name

        try:
            write_lazy_index(tree, embeddings, path)
            with LazyIndex(path) as idx:
                assert idx.detect_enrichment_level() == 0
        finally:
            if os.path.exists(path):
                os.unlink(path)

    def test_level_1_projected(self):
        """Index with UMAP coords returns level 1."""
        from dyf import build_dyf_tree
        from dyf.lazy_index import write_lazy_index, LazyIndex

        n = 40
        embeddings = _make_clustered_embeddings(
            n_clusters=2, points_per_cluster=20, dim=8, seed=42)
        tree = build_dyf_tree(embeddings, max_depth=2, num_bits=2,
                              min_leaf_size=4, seed=42)

        rng = np.random.default_rng(42)
        sf = {
            'umap_x': rng.standard_normal(n).astype(np.float32),
            'umap_y': rng.standard_normal(n).astype(np.float32),
            'umap_z': rng.standard_normal(n).astype(np.float32),
        }

        with tempfile.NamedTemporaryFile(suffix='.dyf', delete=False) as f:
            path = f.name

        try:
            write_lazy_index(tree, embeddings, path, stored_fields=sf)
            with LazyIndex(path) as idx:
                assert idx.detect_enrichment_level() == 1
        finally:
            if os.path.exists(path):
                os.unlink(path)

    def test_level_2_clustered(self):
        """Index with cluster labels returns level 2."""
        from dyf import build_dyf_tree
        from dyf.lazy_index import write_lazy_index, LazyIndex

        n = 40
        embeddings = _make_clustered_embeddings(
            n_clusters=2, points_per_cluster=20, dim=8, seed=42)
        tree = build_dyf_tree(embeddings, max_depth=2, num_bits=2,
                              min_leaf_size=4, seed=42)

        rng = np.random.default_rng(42)
        sf = {
            'umap_x': rng.standard_normal(n).astype(np.float32),
            'umap_y': rng.standard_normal(n).astype(np.float32),
            'umap_z': rng.standard_normal(n).astype(np.float32),
            'cluster_25': np.array([i % 5 for i in range(n)], dtype=np.int32),
        }

        with tempfile.NamedTemporaryFile(suffix='.dyf', delete=False) as f:
            path = f.name

        try:
            write_lazy_index(tree, embeddings, path, stored_fields=sf)
            with LazyIndex(path) as idx:
                assert idx.detect_enrichment_level() == 2
        finally:
            if os.path.exists(path):
                os.unlink(path)

    def test_level_2_community_id(self):
        """Index with community_id stored field returns level 2."""
        from dyf import build_dyf_tree
        from dyf.lazy_index import write_lazy_index, LazyIndex

        n = 40
        embeddings = _make_clustered_embeddings(
            n_clusters=2, points_per_cluster=20, dim=8, seed=42)
        tree = build_dyf_tree(embeddings, max_depth=2, num_bits=2,
                              min_leaf_size=4, seed=42)

        rng = np.random.default_rng(42)
        sf = {
            'umap_x': rng.standard_normal(n).astype(np.float32),
            'umap_y': rng.standard_normal(n).astype(np.float32),
            'umap_z': rng.standard_normal(n).astype(np.float32),
            'community_id': np.array([i % 5 for i in range(n)],
                                     dtype=np.int32),
        }

        with tempfile.NamedTemporaryFile(suffix='.dyf', delete=False) as f:
            path = f.name

        try:
            write_lazy_index(tree, embeddings, path, stored_fields=sf)
            with LazyIndex(path) as idx:
                assert idx.detect_enrichment_level() == 2
        finally:
            if os.path.exists(path):
                os.unlink(path)

    def test_level_2_louvain_metadata(self):
        """Index with louvain_dendrogram metadata returns level 2."""
        from dyf import build_dyf_tree
        from dyf.lazy_index import write_lazy_index, LazyIndex

        n = 40
        embeddings = _make_clustered_embeddings(
            n_clusters=2, points_per_cluster=20, dim=8, seed=42)
        tree = build_dyf_tree(embeddings, max_depth=2, num_bits=2,
                              min_leaf_size=4, seed=42)

        rng = np.random.default_rng(42)
        sf = {
            'umap_x': rng.standard_normal(n).astype(np.float32),
            'umap_y': rng.standard_normal(n).astype(np.float32),
            'umap_z': rng.standard_normal(n).astype(np.float32),
        }

        with tempfile.NamedTemporaryFile(suffix='.dyf', delete=False) as f:
            path = f.name

        try:
            write_lazy_index(tree, embeddings, path, stored_fields=sf,
                             metadata={'louvain_dendrogram': '{}'})
            with LazyIndex(path) as idx:
                assert idx.detect_enrichment_level() == 2
        finally:
            if os.path.exists(path):
                os.unlink(path)

    def test_level_3_viz_ready(self):
        """Index with viz metadata returns level 3."""
        from dyf import build_dyf_tree
        from dyf.lazy_index import write_lazy_index, LazyIndex

        n = 40
        embeddings = _make_clustered_embeddings(
            n_clusters=2, points_per_cluster=20, dim=8, seed=42)
        tree = build_dyf_tree(embeddings, max_depth=2, num_bits=2,
                              min_leaf_size=4, seed=42)

        rng = np.random.default_rng(42)
        sf = {
            'umap_x': rng.standard_normal(n).astype(np.float32),
            'umap_y': rng.standard_normal(n).astype(np.float32),
            'umap_z': rng.standard_normal(n).astype(np.float32),
            'cluster_25': np.array([i % 5 for i in range(n)], dtype=np.int32),
        }
        meta = {
            'edge_pairs': '[[0,1,5],[1,2,3]]',
            'tour_narration': '{"intro":"Welcome"}',
        }

        with tempfile.NamedTemporaryFile(suffix='.dyf', delete=False) as f:
            path = f.name

        try:
            write_lazy_index(tree, embeddings, path,
                             stored_fields=sf, metadata=meta)
            with LazyIndex(path) as idx:
                assert idx.detect_enrichment_level() == 3
        finally:
            if os.path.exists(path):
                os.unlink(path)


@lazy_deps
class TestAdaptiveProbing:
    """Tests for adaptive probe count based on routing margin."""

    @pytest.fixture
    def index_data(self):
        """Build a DYF tree and write a lazy index file."""
        from dyf import build_dyf_tree
        from dyf.lazy_index import write_lazy_index, LazyIndex

        embeddings = _make_clustered_embeddings(
            n_clusters=5, points_per_cluster=40, dim=32, seed=42)
        tree = build_dyf_tree(embeddings, max_depth=3, num_bits=3,
                              min_leaf_size=4, seed=42)

        with tempfile.NamedTemporaryFile(suffix='.dyf', delete=False) as f:
            path = f.name

        try:
            write_lazy_index(tree, embeddings, path, compression='zstd',
                             quantization='float16',
                             metadata={'test': 'true'})
            yield {
                'path': path,
                'embeddings': embeddings,
                'tree': tree,
            }
        finally:
            if os.path.exists(path):
                os.unlink(path)

    def test_adaptive_probes_more_for_boundary_queries(self, index_data):
        """Low-margin queries should probe more leaves than high-margin ones."""
        from dyf.lazy_index import LazyIndex

        embeddings = index_data['embeddings']

        with LazyIndex(index_data['path']) as idx:
            margins_and_probes = []
            for i in range(len(embeddings)):
                result = idx.search(embeddings[i], k=10, nprobe="auto",
                                    return_routing=True)
                margins_and_probes.append(
                    (result.routing['min_margin'],
                     len(result.routing['leaves_probed'])))

            # Split into low-margin and high-margin groups by median
            margins_and_probes.sort(key=lambda x: x[0])
            mid = len(margins_and_probes) // 2
            low_margin_probes = [p for _, p in margins_and_probes[:mid]]
            high_margin_probes = [p for _, p in margins_and_probes[mid:]]

            avg_low = sum(low_margin_probes) / len(low_margin_probes)
            avg_high = sum(high_margin_probes) / len(high_margin_probes)

            # Low-margin queries should probe at least as many leaves on average
            assert avg_low >= avg_high, (
                f"Low-margin queries should probe more: "
                f"avg_low={avg_low:.2f}, avg_high={avg_high:.2f}")

    def test_adaptive_backward_compatible(self, index_data):
        """Fixed nprobe=3 still works and routing includes min_margin."""
        from dyf.lazy_index import LazyIndex

        embeddings = index_data['embeddings']
        query = embeddings[0]

        with LazyIndex(index_data['path']) as idx:
            result = idx.search(query, k=10, nprobe=3, return_routing=True)
            assert len(result.indices) > 0
            assert 'min_margin' in result.routing
            assert 'nprobe_mode' not in result.routing
            assert 'adaptive_nprobe' not in result.routing

    def test_adaptive_config_custom(self, index_data):
        """AdaptiveProbeConfig with custom thresholds works."""
        from dyf.lazy_index import LazyIndex, AdaptiveProbeConfig

        embeddings = index_data['embeddings']
        query = embeddings[0]

        cfg = AdaptiveProbeConfig(
            margin_lo=0.005, margin_hi=0.2,
            min_probes=2, max_probes=8)

        with LazyIndex(index_data['path']) as idx:
            result = idx.search(query, k=10, nprobe=cfg, return_routing=True)
            assert len(result.indices) > 0
            nprobe_used = result.routing['adaptive_nprobe']
            assert 2 <= nprobe_used <= 8

    def test_adaptive_routing_diagnostics(self, index_data):
        """nprobe='auto' routing includes min_margin, adaptive_nprobe, nprobe_mode."""
        from dyf.lazy_index import LazyIndex

        embeddings = index_data['embeddings']
        query = embeddings[0]

        with LazyIndex(index_data['path']) as idx:
            result = idx.search(query, k=10, nprobe="auto",
                                return_routing=True)
            r = result.routing
            assert 'min_margin' in r
            assert 'adaptive_nprobe' in r
            assert r['nprobe_mode'] == 'adaptive'
            assert isinstance(r['min_margin'], float)
            assert isinstance(r['adaptive_nprobe'], int)
            assert r['adaptive_nprobe'] >= 1
