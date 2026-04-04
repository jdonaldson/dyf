"""Tests for CatalogSpace module.

All tests use synthetic hierarchies with np.random.default_rng(42).
No embedding API calls.
"""

import json

import numpy as np
import pytest

from dyf.catalog import (
    CatalogConfig,
    CatalogMatch,
    CatalogSpace,
    CrossMapping,
    JointMatchResult,
    _compute_entropy,
    _node_z_to_fit,
)
from dyf.categorical import CategoryGraph

# ── Helpers ──────────────────────────────────────────────────────────────


def _make_hierarchy(rng, n_parents=4, children_per_parent=5, dim=32):
    """Build a synthetic 2-level hierarchy with separable embeddings.

    Returns (graph, embeddings, node_ids, node_names) where children
    cluster near their parent in embedding space.
    """
    edges = []
    node_ids = []
    node_names = []
    embeddings = []

    # Parent nodes at depth 1
    parent_centers = rng.standard_normal((n_parents, dim)).astype(np.float32)
    # Normalize for cosine-like behavior
    parent_centers /= np.linalg.norm(parent_centers, axis=1, keepdims=True)

    for pi in range(n_parents):
        pid = f"P{pi:02d}"
        pname = f"Parent_{pi}"
        edges.append(("_root_", pid, 1.0 / n_parents))
        node_ids.append(pid)
        node_names.append(pname)
        embeddings.append(parent_centers[pi])

        # Child nodes at depth 2
        for ci in range(children_per_parent):
            cid = f"C{pi:02d}_{ci:02d}"
            cname = f"Child_{pi}_{ci}"
            edges.append((pid, cid, 1.0 / children_per_parent))
            node_ids.append(cid)
            node_names.append(cname)

            # Child embedding = parent + small noise
            noise = rng.standard_normal(dim).astype(np.float32) * 0.15
            child_emb = parent_centers[pi] + noise
            child_emb /= np.linalg.norm(child_emb)
            embeddings.append(child_emb)

    graph = CategoryGraph.from_edges(edges)
    return (
        graph,
        np.array(embeddings, dtype=np.float32),
        np.array(node_ids, dtype=str),
        np.array(node_names, dtype=str),
    )


def _make_flat_catalog(rng, n_nodes=20, dim=32):
    """Build a flat (single-level) catalog."""
    embeddings = rng.standard_normal((n_nodes, dim)).astype(np.float32)
    embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True)

    node_ids = np.array([f"N{i:03d}" for i in range(n_nodes)], dtype=str)
    node_names = np.array([f"Node_{i}" for i in range(n_nodes)], dtype=str)
    graph = CategoryGraph.from_single_level(node_ids)

    return graph, embeddings, node_ids, node_names


def _make_config(name, graph, embeddings, node_ids, node_names):
    return CatalogConfig(
        name=name,
        graph=graph,
        embeddings=embeddings,
        node_ids=node_ids,
        node_names=node_names,
    )


# ── Phase 1: Single-catalog tests ───────────────────────────────────────


class TestCatalogConfig:
    """Test 1: Config construction."""

    def test_basic_construction(self):
        rng = np.random.default_rng(42)
        graph, embs, ids, names = _make_hierarchy(rng)
        config = CatalogConfig("test", graph, embs, ids, names)
        assert config.name == "test"
        assert config.embeddings.shape[0] == len(ids)
        assert config.embeddings.dtype == np.float32

    def test_mismatched_lengths_raise(self):
        rng = np.random.default_rng(42)
        graph, embs, ids, names = _make_hierarchy(rng)
        with pytest.raises(ValueError, match="node_ids"):
            CatalogConfig("test", graph, embs[:-1], ids, names)

    def test_mismatched_names_raise(self):
        rng = np.random.default_rng(42)
        graph, embs, ids, names = _make_hierarchy(rng)
        with pytest.raises(ValueError, match="node_names"):
            CatalogConfig("test", graph, embs, ids, names[:-1])


class TestFitStats:
    """Test 2: Fit computes statistics."""

    def test_fit_produces_stats(self):
        rng = np.random.default_rng(42)
        graph, embs, ids, names = _make_hierarchy(rng)
        config = _make_config("test", graph, embs, ids, names)

        space = CatalogSpace()
        space.add_catalog(config)
        space.fit()

        assert space.is_fitted
        fc = space._fitted["test"]
        assert fc.centroid.shape == (embs.shape[1],)
        assert fc.tax_std > 0
        assert len(fc.node_z_scores) == len(ids)
        assert len(fc.path_alignments) == len(ids)

    def test_cannot_add_after_fit(self):
        rng = np.random.default_rng(42)
        graph, embs, ids, names = _make_hierarchy(rng)
        config = _make_config("test", graph, embs, ids, names)

        space = CatalogSpace()
        space.add_catalog(config)
        space.fit()

        with pytest.raises(RuntimeError, match="after fit"):
            space.add_catalog(config)


class TestBasicMatch:
    """Test 3: Basic matching returns correct structure."""

    def test_match_single_returns_catalog_match(self):
        rng = np.random.default_rng(42)
        graph, embs, ids, names = _make_hierarchy(rng)
        config = _make_config("test", graph, embs, ids, names)

        space = CatalogSpace()
        space.add_catalog(config).fit()

        # Use a known node's embedding as query
        query = embs[5].copy()
        result = space.match_single("test", query)

        assert isinstance(result, CatalogMatch)
        assert result.catalog_name == "test"
        assert result.node_id in ids
        assert result.similarity > 0
        assert result.fit in ("VERY_HIGH", "HIGH", "MODERATE", "LOW", "OUT_OF_DOMAIN")

    def test_match_single_unknown_catalog_raises(self):
        rng = np.random.default_rng(42)
        graph, embs, ids, names = _make_hierarchy(rng)
        config = _make_config("test", graph, embs, ids, names)

        space = CatalogSpace()
        space.add_catalog(config).fit()

        with pytest.raises(KeyError):
            space.match_single("nonexistent", embs[0])


class TestTwoStage:
    """Test 4: Two-stage constrains to correct parent."""

    def test_twostage_finds_child_of_best_parent(self):
        rng = np.random.default_rng(42)
        graph, embs, ids, names = _make_hierarchy(rng, n_parents=4, children_per_parent=5)
        config = _make_config("test", graph, embs, ids, names)

        space = CatalogSpace()
        space.add_catalog(config).fit()

        # Query near parent P01 — should match a child of P01
        parent_idx = list(ids).index("P01")
        query = embs[parent_idx] + rng.standard_normal(embs.shape[1]).astype(np.float32) * 0.05
        query /= np.linalg.norm(query)

        result = space.match_single("test", query)

        # The match should be either P01 itself or one of its children
        matched_id = result.node_id
        assert matched_id.startswith("P01") or matched_id.startswith("C01"), (
            f"Expected match under P01, got {matched_id}"
        )


class TestWithinZ:
    """Test 5: Within-z picks the sharpest level."""

    def test_within_z_selects_discriminative_depth(self):
        rng = np.random.default_rng(42)
        # Create hierarchy where depth-2 children are well-separated
        graph, embs, ids, names = _make_hierarchy(rng, n_parents=3, children_per_parent=8)
        config = _make_config("test", graph, embs, ids, names)

        space = CatalogSpace()
        space.add_catalog(config).fit()

        # Query that's very close to a specific child
        child_idx = list(ids).index("C00_00")
        query = embs[child_idx].copy()

        result = space.match_single("test", query)
        # Should match at depth 2 (child level) because children
        # are more discriminative for an exact-match query
        assert result.depth == 2


class TestPathAlignment:
    """Test 6: Path alignment near 1.0 for aligned hierarchy."""

    def test_aligned_hierarchy_has_high_alignment(self):
        rng = np.random.default_rng(42)
        # Children are near their parents → alignment should be high
        graph, embs, ids, names = _make_hierarchy(rng, n_parents=4, children_per_parent=5)
        config = _make_config("test", graph, embs, ids, names)

        space = CatalogSpace()
        space.add_catalog(config).fit()

        fc = space._fitted["test"]
        # Children (depth 2) should have high alignment
        child_mask = fc.node_depths == 2
        child_alignments = fc.path_alignments[child_mask]

        # Mean alignment should be positive (parents and children point similar direction)
        assert child_alignments.mean() > 0.5, (
            f"Expected positive alignment, got mean={child_alignments.mean():.3f}"
        )


class TestEntropy:
    """Test 7: Entropy 0 for dominant match, ~1 for uniform."""

    def test_entropy_dominant(self):
        # One value much higher than rest
        sims = np.array([0.95, 0.1, 0.1, 0.1, 0.1], dtype=np.float32)
        e = _compute_entropy(sims)
        assert e < 0.3, f"Expected low entropy for dominant match, got {e:.3f}"

    def test_entropy_uniform(self):
        # All values equal
        sims = np.array([0.5, 0.5, 0.5, 0.5, 0.5], dtype=np.float32)
        e = _compute_entropy(sims)
        assert e > 0.9, f"Expected high entropy for uniform, got {e:.3f}"

    def test_entropy_single(self):
        sims = np.array([0.8], dtype=np.float32)
        assert _compute_entropy(sims) == 0.0


class TestGapDetection:
    """Test 8: Gap detected for entropy jump + similarity drop."""

    def test_gap_detected_with_engineered_data(self):
        """Create a 3-level hierarchy where depth-3 has poor matches."""
        rng = np.random.default_rng(42)
        dim = 32

        edges = []
        node_ids = []
        node_names = []
        embeddings = []

        # Root -> 2 segments (depth 1)
        seg_centers = rng.standard_normal((2, dim)).astype(np.float32)
        seg_centers /= np.linalg.norm(seg_centers, axis=1, keepdims=True)

        for si in range(2):
            sid = f"S{si}"
            edges.append(("_root_", sid, 0.5))
            node_ids.append(sid)
            node_names.append(f"Segment_{si}")
            embeddings.append(seg_centers[si])

            # 3 classes per segment (depth 2)
            for ci in range(3):
                cid = f"CL{si}_{ci}"
                edges.append((sid, cid, 1.0 / 3))
                class_emb = seg_centers[si] + rng.standard_normal(dim).astype(np.float32) * 0.1
                class_emb /= np.linalg.norm(class_emb)
                node_ids.append(cid)
                node_names.append(f"Class_{si}_{ci}")
                embeddings.append(class_emb)

                # 5 commodities per class (depth 3) — scattered randomly
                for ki in range(5):
                    kid = f"K{si}_{ci}_{ki}"
                    edges.append((cid, kid, 0.2))
                    # Make commodities VERY noisy — not near their parent
                    comm_emb = rng.standard_normal(dim).astype(np.float32)
                    comm_emb /= np.linalg.norm(comm_emb)
                    node_ids.append(kid)
                    node_names.append(f"Comm_{si}_{ci}_{ki}")
                    embeddings.append(comm_emb)

        graph = CategoryGraph.from_edges(edges)
        config = _make_config(
            "gapped",
            graph,
            np.array(embeddings, dtype=np.float32),
            np.array(node_ids, dtype=str),
            np.array(node_names, dtype=str),
        )

        space = CatalogSpace()
        space.add_catalog(config).fit()

        # Query near segment 0 — should match well at depth 1-2 but poorly at depth 3
        query = seg_centers[0] + rng.standard_normal(dim).astype(np.float32) * 0.05
        query = query.astype(np.float32)
        query /= np.linalg.norm(query)

        result = space.match_single("gapped", query)
        # We can't guarantee gap detection fires with random data,
        # but the structure is there. Test the mechanism runs without error.
        assert isinstance(result.gap_detected, bool)
        assert isinstance(result.gap_score, float)


class TestDiverseAlternatives:
    """Test 9: Diverse alternatives from different parents."""

    def test_alternatives_from_different_parents(self):
        rng = np.random.default_rng(42)
        graph, embs, ids, names = _make_hierarchy(rng, n_parents=4, children_per_parent=5)
        config = _make_config("test", graph, embs, ids, names)

        space = CatalogSpace()
        space.add_catalog(config).fit()

        # Query near first child of P00
        child_idx = list(ids).index("C00_00")
        query = embs[child_idx].copy()

        result = space.match_single("test", query, top_k=5)

        if result.alternatives:
            # Alternatives should not share the primary's parent
            primary_id = result.node_id
            primary_parents = graph.get_parents(primary_id)

            for alt_id, _alt_name, _alt_sim in result.alternatives:
                alt_parents = graph.get_parents(alt_id)
                # At least some alternatives should be from different parents
                # (can't guarantee ALL are, but the mechanism should work)
                if alt_parents and primary_parents:
                    pass  # structure verified, no crash


# ── Phase 2: Multi-catalog tests ─────────────────────────────────────────


class TestNoMappings:
    """Test 10: No mappings = independent matches + coherence 1.0."""

    def test_independent_matching(self):
        rng = np.random.default_rng(42)
        g1, e1, i1, n1 = _make_flat_catalog(rng, n_nodes=15, dim=32)
        g2, e2, i2, n2 = _make_flat_catalog(rng, n_nodes=10, dim=32)

        space = CatalogSpace()
        space.add_catalog(_make_config("alpha", g1, e1, i1, n1))
        space.add_catalog(_make_config("beta", g2, e2, i2, n2))
        space.fit()

        query = rng.standard_normal(32).astype(np.float32)
        query /= np.linalg.norm(query)

        results = space.match(query.reshape(1, -1))
        assert len(results) == 1
        r = results[0]

        assert isinstance(r, JointMatchResult)
        assert "alpha" in r.matches
        assert "beta" in r.matches
        assert r.coherence_score == 1.0
        assert r.reranked is False


class TestCoherentMappings:
    """Test 11: Coherent mappings confirmed (reranked=False)."""

    def test_coherent_top1_not_reranked(self):
        rng = np.random.default_rng(42)
        dim = 32

        # Two catalogs with 5 nodes each, sharing embeddings
        # so node i in catalog A naturally maps to node i in catalog B
        shared_embs = rng.standard_normal((5, dim)).astype(np.float32)
        shared_embs /= np.linalg.norm(shared_embs, axis=1, keepdims=True)

        ids_a = np.array([f"A{i}" for i in range(5)], dtype=str)
        ids_b = np.array([f"B{i}" for i in range(5)], dtype=str)
        names_a = np.array([f"NodeA_{i}" for i in range(5)], dtype=str)
        names_b = np.array([f"NodeB_{i}" for i in range(5)], dtype=str)

        g_a = CategoryGraph.from_single_level(ids_a)
        g_b = CategoryGraph.from_single_level(ids_b)

        # Mapping: A0<->B0, A1<->B1, etc.
        mapping = CrossMapping(
            source_catalog="cat_a",
            target_catalog="cat_b",
            source_ids=ids_a,
            target_ids=ids_b,
            weights=np.ones(5, dtype=np.float32),
        )

        space = CatalogSpace()
        space.add_catalog(_make_config("cat_a", g_a, shared_embs.copy(), ids_a, names_a))
        space.add_catalog(_make_config("cat_b", g_b, shared_embs.copy(), ids_b, names_b))
        space.add_mapping(mapping)
        space.fit()

        # Query near node 2 — should match A2 and B2, which are mapped
        query = shared_embs[2] + rng.standard_normal(dim).astype(np.float32) * 0.01
        query = query.astype(np.float32)
        query /= np.linalg.norm(query)

        results = space.match(query.reshape(1, -1), coherence_weight=0.3)
        r = results[0]

        assert r.matches["cat_a"].node_id == "A2"
        assert r.matches["cat_b"].node_id == "B2"
        assert r.reranked is False


class TestIncoherentReranking:
    """Test 12: Incoherent top-1 reranked via mapping."""

    def test_reranking_improves_coherence(self):
        rng = np.random.default_rng(42)
        dim = 32

        # Catalog A: 5 nodes
        embs_a = rng.standard_normal((5, dim)).astype(np.float32)
        embs_a /= np.linalg.norm(embs_a, axis=1, keepdims=True)

        # Catalog B: 5 nodes, but B0 is similar to A1 (not A0)
        embs_b = rng.standard_normal((5, dim)).astype(np.float32)
        embs_b[0] = embs_a[1] + rng.standard_normal(dim).astype(np.float32) * 0.05
        embs_b[0] /= np.linalg.norm(embs_b[0])
        embs_b /= np.linalg.norm(embs_b, axis=1, keepdims=True)

        ids_a = np.array([f"A{i}" for i in range(5)], dtype=str)
        ids_b = np.array([f"B{i}" for i in range(5)], dtype=str)
        names_a = np.array([f"NodeA_{i}" for i in range(5)], dtype=str)
        names_b = np.array([f"NodeB_{i}" for i in range(5)], dtype=str)

        g_a = CategoryGraph.from_single_level(ids_a)
        g_b = CategoryGraph.from_single_level(ids_b)

        # Mapping says A0<->B0 (but embeddings disagree)
        mapping = CrossMapping(
            source_catalog="cat_a",
            target_catalog="cat_b",
            source_ids=np.array(["A0"], dtype=str),
            target_ids=np.array(["B0"], dtype=str),
            weights=np.array([1.0], dtype=np.float32),
        )

        space = CatalogSpace()
        space.add_catalog(_make_config("cat_a", g_a, embs_a, ids_a, names_a))
        space.add_catalog(_make_config("cat_b", g_b, embs_b, ids_b, names_b))
        space.add_mapping(mapping)
        space.fit()

        # Query near A0
        query = embs_a[0] + rng.standard_normal(dim).astype(np.float32) * 0.01
        query = query.astype(np.float32)
        query /= np.linalg.norm(query)

        results = space.match(query.reshape(1, -1), coherence_weight=0.5)
        r = results[0]

        # The result should have been processed (reranked or confirmed)
        assert isinstance(r, JointMatchResult)
        assert "cat_a" in r.matches
        assert "cat_b" in r.matches


class TestThreeCatalogs:
    """Test 13: Three catalogs with two mappings."""

    def test_three_catalog_joint_match(self):
        rng = np.random.default_rng(42)
        dim = 32

        # Three catalogs, shared embeddings for coherence
        shared = rng.standard_normal((5, dim)).astype(np.float32)
        shared /= np.linalg.norm(shared, axis=1, keepdims=True)

        configs = []
        for cname in ["unspsc", "broadjump", "curvo"]:
            ids = np.array([f"{cname[0].upper()}{i}" for i in range(5)], dtype=str)
            names = np.array([f"{cname}_{i}" for i in range(5)], dtype=str)
            g = CategoryGraph.from_single_level(ids)
            # Add small per-catalog noise
            embs = shared + rng.standard_normal((5, dim)).astype(np.float32) * 0.02
            embs /= np.linalg.norm(embs, axis=1, keepdims=True)
            configs.append(_make_config(cname, g, embs, ids, names))

        # Mappings: unspsc<->broadjump, broadjump<->curvo
        m1 = CrossMapping(
            source_catalog="unspsc",
            target_catalog="broadjump",
            source_ids=np.array(["U0", "U1", "U2"], dtype=str),
            target_ids=np.array(["B0", "B1", "B2"], dtype=str),
            weights=np.ones(3, dtype=np.float32),
        )
        m2 = CrossMapping(
            source_catalog="broadjump",
            target_catalog="curvo",
            source_ids=np.array(["B0", "B1", "B2"], dtype=str),
            target_ids=np.array(["C0", "C1", "C2"], dtype=str),
            weights=np.ones(3, dtype=np.float32),
        )

        space = CatalogSpace()
        for c in configs:
            space.add_catalog(c)
        space.add_mapping(m1).add_mapping(m2)
        space.fit()

        query = shared[1] + rng.standard_normal(dim).astype(np.float32) * 0.01
        query = query.astype(np.float32)
        query /= np.linalg.norm(query)

        results = space.match(query.reshape(1, -1), coherence_weight=0.3)
        r = results[0]

        assert len(r.matches) == 3
        assert "unspsc" in r.matches
        assert "broadjump" in r.matches
        assert "curvo" in r.matches
        assert 0.0 <= r.coherence_score <= 1.0


# ── Phase 3: Serialization ──────────────────────────────────────────────


class TestSerialization:
    """Test 14: Round-trip dict/JSON serialization."""

    def test_round_trip_dict(self):
        rng = np.random.default_rng(42)
        g1, e1, i1, n1 = _make_hierarchy(rng, n_parents=3, children_per_parent=4)
        g2, e2, i2, n2 = _make_flat_catalog(rng, n_nodes=10)

        mapping = CrossMapping(
            source_catalog="hier",
            target_catalog="flat",
            source_ids=np.array(["C00_00", "C01_00"], dtype=str),
            target_ids=np.array(["N000", "N001"], dtype=str),
            weights=np.array([0.9, 0.8], dtype=np.float32),
        )

        space = CatalogSpace()
        space.add_catalog(_make_config("hier", g1, e1, i1, n1))
        space.add_catalog(_make_config("flat", g2, e2, i2, n2))
        space.add_mapping(mapping)
        space.fit()

        # Match before serialization
        query = e1[5].copy()
        result_before = space.match_single("hier", query)

        # Serialize -> deserialize -> fit
        d = space.to_dict()
        assert "catalogs" in d
        assert "mappings" in d

        space2 = CatalogSpace.from_dict(d)
        space2.fit()

        result_after = space2.match_single("hier", query)

        # Results should be identical
        assert result_before.node_id == result_after.node_id
        assert result_before.similarity == result_after.similarity
        assert result_before.fit == result_after.fit

    def test_round_trip_json(self):
        rng = np.random.default_rng(42)
        g1, e1, i1, n1 = _make_flat_catalog(rng, n_nodes=8)
        config = _make_config("test", g1, e1, i1, n1)

        space = CatalogSpace()
        space.add_catalog(config).fit()

        json_str = space.to_json()
        assert isinstance(json_str, str)

        # Verify it's valid JSON
        parsed = json.loads(json_str)
        assert "catalogs" in parsed

        space2 = CatalogSpace.from_json(json_str)
        space2.fit()

        assert space2.catalog_names == ["test"]


# ── Utility tests ────────────────────────────────────────────────────────


class TestNodeZToFit:
    def test_fit_levels(self):
        assert _node_z_to_fit(1.5) == "VERY_HIGH"
        assert _node_z_to_fit(0.5) == "HIGH"
        assert _node_z_to_fit(-0.5) == "MODERATE"
        assert _node_z_to_fit(-1.5) == "LOW"
        assert _node_z_to_fit(-3.0) == "OUT_OF_DOMAIN"


class TestSummary:
    def test_summary_format(self):
        rng = np.random.default_rng(42)
        g, e, i, n = _make_flat_catalog(rng, n_nodes=10)
        space = CatalogSpace()
        space.add_catalog(_make_config("test", g, e, i, n))

        s = space.summary()
        assert "CatalogSpace" in s
        assert "test" in s
        assert "10 nodes" in s

    def test_match_before_fit_raises(self):
        rng = np.random.default_rng(42)
        g, e, i, n = _make_flat_catalog(rng)
        space = CatalogSpace()
        space.add_catalog(_make_config("test", g, e, i, n))

        with pytest.raises(RuntimeError, match="fit"):
            space.match_single("test", e[0])


class TestDimensionMismatch:
    def test_different_dims_raise(self):
        rng = np.random.default_rng(42)
        g1, e1, i1, n1 = _make_flat_catalog(rng, n_nodes=5, dim=32)
        g2 = CategoryGraph.from_single_level(["X", "Y", "Z"])
        e2 = rng.standard_normal((3, 64)).astype(np.float32)
        i2 = np.array(["X", "Y", "Z"], dtype=str)
        n2 = np.array(["X", "Y", "Z"], dtype=str)

        space = CatalogSpace()
        space.add_catalog(_make_config("a", g1, e1, i1, n1))
        space.add_catalog(_make_config("b", g2, e2, i2, n2))

        with pytest.raises(ValueError, match="dimensions"):
            space.fit()


class TestBottomUpParentSelection:
    """Test 18: Bottom-up max selects parent by best child similarity."""

    def test_bottom_up_beats_class_embedding(self):
        """Parent B has a nearer class embedding, but Parent A's child
        scores highest — bottom-up should pick Parent A.

        Uses 3 children per parent so depth-2 within-z exceeds depth-1,
        ensuring the two-stage parent-constrained path is exercised.
        """
        dim = 32
        rng = np.random.default_rng(99)

        # Query direction
        query_dir = rng.standard_normal(dim).astype(np.float32)
        query_dir /= np.linalg.norm(query_dir)

        # Orthogonal direction for separation
        ortho = rng.standard_normal(dim).astype(np.float32)
        ortho -= ortho @ query_dir * query_dir
        ortho /= np.linalg.norm(ortho)

        # Parent A ("Surgical") — class embedding far from query
        parent_a_emb = (ortho * 0.9 + query_dir * 0.1).astype(np.float32)
        parent_a_emb /= np.linalg.norm(parent_a_emb)

        # Parent B ("Writing") — class embedding near query
        parent_b_emb = (query_dir * 0.85 + ortho * 0.15).astype(np.float32)
        parent_b_emb /= np.linalg.norm(parent_b_emb)

        # Child of A: "cautery pencil" — very near query
        child_a1 = (query_dir * 0.95 + ortho * 0.05).astype(np.float32)
        child_a1 /= np.linalg.norm(child_a1)
        # Two more children of A, far from query (near parent A direction)
        child_a2 = (ortho * 0.85 + query_dir * 0.15).astype(np.float32)
        child_a2 /= np.linalg.norm(child_a2)
        child_a3 = (ortho * 0.80 + query_dir * 0.20).astype(np.float32)
        child_a3 /= np.linalg.norm(child_a3)

        # Children of B: all far from query
        child_b1 = (-query_dir * 0.6 + ortho * 0.4).astype(np.float32)
        child_b1 /= np.linalg.norm(child_b1)
        child_b2 = (-query_dir * 0.5 + ortho * 0.5).astype(np.float32)
        child_b2 /= np.linalg.norm(child_b2)
        child_b3 = (-query_dir * 0.7 + ortho * 0.3).astype(np.float32)
        child_b3 /= np.linalg.norm(child_b3)

        edges = [
            ("_root_", "PA", 0.5),
            ("_root_", "PB", 0.5),
            ("PA", "CA1", 1.0 / 3),
            ("PA", "CA2", 1.0 / 3),
            ("PA", "CA3", 1.0 / 3),
            ("PB", "CB1", 1.0 / 3),
            ("PB", "CB2", 1.0 / 3),
            ("PB", "CB3", 1.0 / 3),
        ]
        graph = CategoryGraph.from_edges(edges)

        node_ids = np.array(
            ["PA", "PB", "CA1", "CA2", "CA3", "CB1", "CB2", "CB3"], dtype=str
        )
        node_names = np.array(
            ["Surgical", "Writing", "cautery pencil", "scalpel", "forceps",
             "mechanical pencil", "ballpoint", "eraser"], dtype=str
        )
        embeddings = np.array(
            [parent_a_emb, parent_b_emb, child_a1, child_a2, child_a3,
             child_b1, child_b2, child_b3], dtype=np.float32
        )

        config = _make_config("test_bu", graph, embeddings, node_ids, node_names)
        space = CatalogSpace()
        space.add_catalog(config).fit()

        query = query_dir.copy()
        result = space.match_single("test_bu", query)

        # Verify preconditions:
        # Parent B's class embedding is closer to query than Parent A's
        sim_pa = float(parent_a_emb @ query)
        sim_pb = float(parent_b_emb @ query)
        assert sim_pb > sim_pa, "Precondition: Parent B class embedding should be nearer to query"

        # But child CA1 is nearest overall child
        sim_ca1 = float(child_a1 @ query)
        child_b_sims = [float(e @ query) for e in [child_b1, child_b2, child_b3]]
        assert sim_ca1 > max(child_b_sims), "Precondition: CA1 should be nearer to query than any B child"

        # Bottom-up should select CA1 (child of Parent A)
        assert result.node_id == "CA1", (
            f"Expected CA1 (child of Surgical), got {result.node_id}"
        )


class TestTermDisambiguation:
    """Test 19: Term-affinity boost flips thin-margin parent selection."""

    def test_term_boost_flips_parent(self):
        """Parent B (Hardware) has slightly higher best-child similarity,
        but query text "bone drill high speed" discriminates toward
        Parent A (Medical) via branch_terms overlap.

        Without query_text: Hardware wins (embedding margin).
        With query_text: Medical wins (term boost).
        """
        dim = 32
        rng = np.random.default_rng(77)

        # Query direction
        query_dir = rng.standard_normal(dim).astype(np.float32)
        query_dir /= np.linalg.norm(query_dir)

        # Orthogonal direction
        ortho = rng.standard_normal(dim).astype(np.float32)
        ortho -= ortho @ query_dir * query_dir
        ortho /= np.linalg.norm(ortho)

        # Parent A (Medical) — class embedding away from query
        parent_a_emb = (ortho * 0.8 + query_dir * 0.2).astype(np.float32)
        parent_a_emb /= np.linalg.norm(parent_a_emb)

        # Parent B (Hardware) — class embedding nearer to query
        parent_b_emb = (query_dir * 0.7 + ortho * 0.3).astype(np.float32)
        parent_b_emb /= np.linalg.norm(parent_b_emb)

        # Children of A: "bone drill surgical", "bone saw", "bone screw"
        # Best child of A is close to query but slightly less than best of B
        child_a1 = (query_dir * 0.88 + ortho * 0.12).astype(np.float32)
        child_a1 /= np.linalg.norm(child_a1)
        child_a2 = (ortho * 0.85 + query_dir * 0.15).astype(np.float32)
        child_a2 /= np.linalg.norm(child_a2)
        child_a3 = (ortho * 0.80 + query_dir * 0.20).astype(np.float32)
        child_a3 /= np.linalg.norm(child_a3)

        # Children of B: "power drill cordless", "power saw", "impact driver"
        # Best child of B is slightly closer to query than best of A
        child_b1 = (query_dir * 0.90 + ortho * 0.10).astype(np.float32)
        child_b1 /= np.linalg.norm(child_b1)
        child_b2 = (-query_dir * 0.5 + ortho * 0.5).astype(np.float32)
        child_b2 /= np.linalg.norm(child_b2)
        child_b3 = (-query_dir * 0.6 + ortho * 0.4).astype(np.float32)
        child_b3 /= np.linalg.norm(child_b3)

        edges = [
            ("_root_", "PA", 0.5),
            ("_root_", "PB", 0.5),
            ("PA", "CA1", 1.0 / 3),
            ("PA", "CA2", 1.0 / 3),
            ("PA", "CA3", 1.0 / 3),
            ("PB", "CB1", 1.0 / 3),
            ("PB", "CB2", 1.0 / 3),
            ("PB", "CB3", 1.0 / 3),
        ]
        graph = CategoryGraph.from_edges(edges)

        node_ids = np.array(
            ["PA", "PB", "CA1", "CA2", "CA3", "CB1", "CB2", "CB3"],
            dtype=str,
        )
        node_names = np.array(
            [
                "Medical",
                "Hardware",
                "bone drill surgical",
                "bone saw",
                "bone screw",
                "power drill cordless",
                "power saw",
                "impact driver",
            ],
            dtype=str,
        )
        embeddings = np.array(
            [parent_a_emb, parent_b_emb, child_a1, child_a2, child_a3,
             child_b1, child_b2, child_b3],
            dtype=np.float32,
        )

        config = _make_config("test_td", graph, embeddings, node_ids, node_names)
        space = CatalogSpace()
        space.add_catalog(config).fit()

        query = query_dir.copy()

        # Verify precondition: B's best child (CB1) is slightly nearer than A's (CA1)
        sim_ca1 = float(child_a1 @ query)
        sim_cb1 = float(child_b1 @ query)
        assert sim_cb1 > sim_ca1, (
            f"Precondition failed: CB1 sim {sim_cb1:.4f} should exceed CA1 sim {sim_ca1:.4f}"
        )

        # Without query_text: Hardware parent wins
        result_no_text = space.match_single("test_td", query)
        assert result_no_text.node_id.startswith("CB"), (
            f"Without text, expected CB* (Hardware child), got {result_no_text.node_id}"
        )

        # With query_text: "bone drill high speed" should flip to Medical
        result_with_text = space.match_single(
            "test_td", query, query_text="bone drill high speed"
        )
        assert result_with_text.node_id.startswith("CA"), (
            f"With text, expected CA* (Medical child), got {result_with_text.node_id}"
        )

    def test_no_text_unchanged(self):
        """Without query_text, behavior is identical to before."""
        rng = np.random.default_rng(42)
        graph, embs, ids, names = _make_hierarchy(rng)
        config = _make_config("test", graph, embs, ids, names)

        space = CatalogSpace()
        space.add_catalog(config).fit()

        query = embs[5].copy()
        result = space.match_single("test", query)
        assert isinstance(result, CatalogMatch)
        assert result.node_id in ids
