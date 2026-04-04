"""Tests for dyf.categorical — CategoryGraph, coarsen, multi-level Fisher,
discover_categorical_columns, embed_with_diagnostics."""

import numpy as np
import pytest

from dyf.categorical import (
    AxisDiagnostic,
    CategoryGraph,
    coarsen,
    diagnose_axes,
    diagnostics_to_metadata,
    discover_categorical_columns,
    embed_with_diagnostics,
    load_category_graphs,
    multi_level_fisher_weights,
    store_category_graph,
)

# ── CategoryGraph construction ───────────────────────────────────────────


class TestFromEdges:
    def test_simple_tree(self):
        """Three-node tree: root → A, root → B."""
        g = CategoryGraph.from_edges([
            ("root", "A", 0.6),
            ("root", "B", 0.4),
        ])
        assert set(g.roots) == {"root"}
        assert set(g.leaves) == {"A", "B"}
        assert g.get_children("root") == ["A", "B"]
        assert g.get_parents("A") == ["root"]

    def test_dag_with_diamond(self):
        """Diamond: root → A, root → B, A → C, B → C."""
        g = CategoryGraph.from_edges([
            ("root", "A", 1.0),
            ("root", "B", 1.0),
            ("A", "C", 1.0),
            ("B", "C", 1.0),
        ])
        assert set(g.roots) == {"root"}
        assert set(g.leaves) == {"C"}
        assert sorted(g.get_parents("C")) == ["A", "B"]
        assert not g.is_tree()  # C has two parents

    def test_weight_defaults_to_one(self):
        """Two-element tuples should get weight 1.0."""
        g = CategoryGraph.from_edges([("root", "A"), ("root", "B")])  # type: ignore[arg-type]
        assert len(g.children["root"]) == 2
        for _, w in g.children["root"]:
            assert w == 1.0


class TestFromLevels:
    def test_three_level_hierarchy(self):
        """Build graph from three co-occurring level arrays."""
        L0 = ["animal", "animal", "plant", "plant", "animal"]
        L1 = ["mammal", "bird", "tree", "flower", "mammal"]
        L2 = ["dog", "eagle", "oak", "rose", "cat"]
        g = CategoryGraph.from_levels([L0, L1, L2])

        # _root_ connects to L0
        assert "_root_" in g.roots
        assert set(g.get_children("_root_")) == {"animal", "plant"}

        # L0 → L1
        assert "mammal" in g.get_children("animal")
        assert "bird" in g.get_children("animal")
        assert "tree" in g.get_children("plant")

        # L1 → L2
        assert "dog" in g.get_children("mammal")
        assert "cat" in g.get_children("mammal")
        assert "eagle" in g.get_children("bird")

        assert g.max_depth() == 3  # _root_(0) → L0(1) → L1(2) → L2(3)
        assert g.is_tree()

    def test_mismatched_lengths_raises(self):
        with pytest.raises(ValueError, match="same length"):
            CategoryGraph.from_levels([["a", "b"], ["x"]])


class TestFromSingleLevel:
    def test_flat_column(self):
        """Flat column becomes depth-1 graph."""
        labels = ["cat", "dog", "cat", "bird", "dog", "dog"]
        g = CategoryGraph.from_single_level(labels)
        assert "_root_" in g.roots
        assert set(g.get_children("_root_")) == {"bird", "cat", "dog"}
        assert g.max_depth() == 1
        assert g.is_tree()


# ── Navigation ───────────────────────────────────────────────────────────


class TestNavigation:
    @pytest.fixture()
    def tree(self):
        return CategoryGraph.from_edges([
            ("root", "A", 1.0),
            ("root", "B", 1.0),
            ("A", "A1", 1.0),
            ("A", "A2", 1.0),
            ("B", "B1", 1.0),
        ])

    def test_ancestors(self, tree):
        assert tree.get_ancestors("A1") == {"A", "root"}

    def test_descendants(self, tree):
        assert tree.get_descendants("root") == {"A", "B", "A1", "A2", "B1"}

    def test_depth(self, tree):
        assert tree.get_depth("root") == 0
        assert tree.get_depth("A") == 1
        assert tree.get_depth("A1") == 2

    def test_nodes_at_depth(self, tree):
        assert sorted(tree.nodes_at_depth(1)) == ["A", "B"]
        assert sorted(tree.nodes_at_depth(2)) == ["A1", "A2", "B1"]

    def test_is_tree(self, tree):
        assert tree.is_tree()

    def test_is_tree_false_for_dag(self):
        g = CategoryGraph.from_edges([
            ("root", "A", 1.0),
            ("root", "B", 1.0),
            ("A", "C", 1.0),
            ("B", "C", 1.0),
        ])
        assert not g.is_tree()

    def test_all_nodes(self, tree):
        assert tree.all_nodes() == {"root", "A", "B", "A1", "A2", "B1"}

    def test_summary(self, tree):
        s = tree.summary()
        assert "6 nodes" in s
        assert "5 edges" in s


# ── LCA depth ────────────────────────────────────────────────────────────


class TestLCADepth:
    @pytest.fixture()
    def tree(self):
        return CategoryGraph.from_edges([
            ("root", "A", 1.0),
            ("root", "B", 1.0),
            ("A", "A1", 1.0),
            ("A", "A2", 1.0),
            ("B", "B1", 1.0),
        ])

    def test_lca_siblings(self, tree):
        """A1 and A2 share parent A at depth 1."""
        assert tree.lca_depth("A1", "A2") == 1

    def test_lca_cousins(self, tree):
        """A1 and B1 share root at depth 0."""
        assert tree.lca_depth("A1", "B1") == 0

    def test_lca_parent_child(self, tree):
        """A is ancestor of A1; LCA is A itself at depth 1."""
        assert tree.lca_depth("A", "A1") == 1

    def test_lca_same_node(self, tree):
        """Same node → own depth."""
        assert tree.lca_depth("A1", "A1") == 2

    def test_lca_root(self, tree):
        """Root and any descendant → LCA is root at depth 0."""
        assert tree.lca_depth("root", "A1") == 0

    def test_lca_unknown(self, tree):
        """Unknown node → -1."""
        assert tree.lca_depth("A1", "unknown") == -1


# ── JSON roundtrip ───────────────────────────────────────────────────────


class TestSerialization:
    def test_json_roundtrip(self):
        g = CategoryGraph.from_edges([
            ("root", "A", 0.7),
            ("root", "B", 0.3),
            ("A", "A1", 1.0),
        ])
        j = g.to_json()
        g2 = CategoryGraph.from_json(j)
        assert set(g2.roots) == set(g.roots)
        assert set(g2.leaves) == set(g.leaves)
        assert g2.get_children("root") == g.get_children("root")
        assert g2.get_children("A") == g.get_children("A")

    def test_dict_roundtrip(self):
        g = CategoryGraph.from_edges([("r", "x", 0.5), ("r", "y", 0.5)])
        d = g.to_dict()
        assert isinstance(d, dict)
        g2 = CategoryGraph.from_dict(d)
        assert set(g2.leaves) == {"x", "y"}


# ── items_at_depth ───────────────────────────────────────────────────────


class TestItemsAtDepth:
    def test_resolve_to_coarser(self):
        """Items at depth 3 should resolve to depth 1 labels."""
        g = CategoryGraph.from_edges([
            ("_root_", "animal", 1.0),
            ("_root_", "plant", 1.0),
            ("animal", "mammal", 1.0),
            ("plant", "tree", 1.0),
            ("mammal", "dog", 1.0),
            ("mammal", "cat", 1.0),
            ("tree", "oak", 1.0),
        ])
        items = np.array(["dog", "cat", "oak", "dog"])
        resolved = g.items_at_depth(1, items)
        # depth 1 = animal, plant
        assert list(resolved) == ["animal", "animal", "plant", "animal"]

    def test_depth_matches_label(self):
        """If item is already at target depth, return it unchanged."""
        g = CategoryGraph.from_edges([
            ("_root_", "A", 1.0),
            ("A", "A1", 1.0),
        ])
        items = np.array(["A", "A1"])
        resolved = g.items_at_depth(1, items)
        assert list(resolved) == ["A", "A"]

    def test_unknown_node_passthrough(self):
        """Unknown labels pass through unchanged."""
        g = CategoryGraph.from_edges([("_root_", "A", 1.0)])
        items = np.array(["A", "UNKNOWN"])
        resolved = g.items_at_depth(1, items)
        assert resolved[1] == "UNKNOWN"


# ── coarsen ──────────────────────────────────────────────────────────────


class TestCoarsen:
    def test_first_term(self):
        raw = ["Forceps, bipolar, reusable", "Catheter, urinary"]
        labels = coarsen(raw, strategy="first_term")
        assert labels[0] == "forceps"
        assert labels[1] == "catheter"

    def test_raw(self):
        raw = ["Alpha", "Beta"]
        labels = coarsen(raw, strategy="raw")
        assert list(labels) == ["Alpha", "Beta"]

    def test_prefix_n(self):
        raw = ["ABCDEF", "ABCXYZ"]
        labels = coarsen(raw, strategy="prefix_3")
        assert labels[0] == "abc"
        assert labels[1] == "abc"

    def test_callable(self):
        raw = ["hello world", "foo bar"]
        labels = coarsen(raw, strategy=lambda s: s.split()[0])
        assert labels[0] == "hello"
        assert labels[1] == "foo"

    def test_none_handling(self):
        raw = [None, "Forceps, bipolar"]
        labels = coarsen(raw, strategy="first_term")
        assert labels[0] == "_unknown_"
        assert labels[1] == "forceps"

    def test_nan_handling(self):
        raw = [float("nan"), "OK"]
        labels = coarsen(raw, strategy="first_term")
        assert labels[0] == "_unknown_"

    def test_list_element(self):
        raw = [["Forceps, bipolar"], "Catheter"]
        labels = coarsen(raw, strategy="first_term")
        assert labels[0] == "forceps"
        assert labels[1] == "catheter"

    def test_empty_list_element(self):
        raw = [[], "OK"]
        labels = coarsen(raw, strategy="first_term")
        assert labels[0] == "_unknown_"

    def test_unknown_strategy_raises(self):
        with pytest.raises(ValueError, match="Unknown coarsen strategy"):
            coarsen(["a"], strategy="bogus")


# ── multi_level_fisher_weights ───────────────────────────────────────────


class TestMultiLevelFisher:
    def test_depth1_matches_single_level(self):
        """Depth-1 graph should produce identical weights to direct Fisher."""
        from dyf.fisher import compute_fisher_weights

        rng = np.random.RandomState(42)
        d = 10
        emb_a = rng.randn(100, d).astype(np.float32)
        emb_b = rng.randn(100, d).astype(np.float32)
        emb_c = rng.randn(100, d).astype(np.float32)
        emb_a[:, 0] += 5.0
        emb_b[:, 0] -= 5.0
        embeddings = np.vstack([emb_a, emb_b, emb_c])
        labels = np.array(["A"] * 100 + ["B"] * 100 + ["C"] * 100)

        graph = CategoryGraph.from_single_level(labels)
        multi_w = multi_level_fisher_weights(
            embeddings, graph, labels, min_count=10
        )
        single_w = compute_fisher_weights(embeddings, labels, min_count=10)

        np.testing.assert_allclose(multi_w, single_w, atol=1e-5)

    def test_multi_depth_differs(self):
        """Multi-depth graph should produce different weights from leaf-only."""
        from dyf.fisher import compute_fisher_weights

        rng = np.random.RandomState(7)
        d = 8
        # 4 groups, 2 families
        groups = {
            "A1": (0, 3.0), "A2": (0, -3.0),
            "B1": (1, 3.0), "B2": (1, -3.0),
        }
        parts = []
        labels_l0, labels_l1 = [], []
        for name, (dim, shift) in groups.items():
            block = rng.randn(80, d).astype(np.float32)
            block[:, dim] += shift
            parts.append(block)
            family = "A" if name.startswith("A") else "B"
            labels_l0.extend([family] * 80)
            labels_l1.extend([name] * 80)

        embeddings = np.vstack(parts)
        item_labels = np.array(labels_l1)

        graph = CategoryGraph.from_levels(
            [np.array(labels_l0), np.array(labels_l1)]
        )

        multi_w = multi_level_fisher_weights(
            embeddings, graph, item_labels, min_count=10
        )
        single_w = compute_fisher_weights(embeddings, item_labels, min_count=10)

        # They should not be identical — multi-level mixes in coarser info
        assert not np.allclose(multi_w, single_w, atol=1e-3)
        # Both should be unit-normed
        assert pytest.approx(np.linalg.norm(multi_w), abs=1e-4) == 1.0
        assert pytest.approx(np.linalg.norm(single_w), abs=1e-4) == 1.0

    def test_no_surviving_classes_uniform(self):
        """If min_count is too high, return uniform weights."""
        rng = np.random.RandomState(0)
        embeddings = rng.randn(20, 4).astype(np.float32)
        labels = np.array(["A"] * 5 + ["B"] * 5 + ["C"] * 5 + ["D"] * 5)
        graph = CategoryGraph.from_single_level(labels)

        w = multi_level_fisher_weights(
            embeddings, graph, labels, min_count=100
        )
        # Should be uniform
        assert np.allclose(w, w[0])


# ── Metadata store/load roundtrip ────────────────────────────────────────


class TestMetadata:
    def test_store_load_roundtrip(self):
        g = CategoryGraph.from_edges([
            ("_root_", "A", 0.6),
            ("_root_", "B", 0.4),
            ("A", "A1", 1.0),
        ])
        mapping = {"0": "family", "1": "term"}
        meta = store_category_graph(g, "gmdn", mapping)
        assert "category_graphs" in meta

        loaded = load_category_graphs(meta)
        assert "gmdn" in loaded
        g2, m2 = loaded["gmdn"]
        assert set(g2.roots) == set(g.roots)
        assert set(g2.leaves) == set(g.leaves)
        assert m2 == mapping

    def test_load_empty_metadata(self):
        assert load_category_graphs({}) == {}
        assert load_category_graphs({"other_key": "val"}) == {}


# ── diagnose_axes ────────────────────────────────────────────────────────


class TestDiagnoseAxes:
    """Tests for the k-NN purity diagnostic."""

    def _make_clustered_data(self, rng, n_per_class=200, d=32, separation=5.0):
        """Generate embeddings with well-separated clusters for one axis,
        random assignments for another."""
        labels_good = np.array(["A"] * n_per_class + ["B"] * n_per_class)
        labels_random = rng.choice(["X", "Y"], size=2 * n_per_class)

        # Class A centered at +separation, class B at -separation
        emb_a = rng.standard_normal((n_per_class, d)).astype(np.float32) + separation
        emb_b = rng.standard_normal((n_per_class, d)).astype(np.float32) - separation
        embeddings = np.vstack([emb_a, emb_b])

        return embeddings, labels_good, labels_random

    def test_well_separated_axis_has_high_lift(self):
        """An axis whose classes form distinct clusters should have high lift."""
        rng = np.random.default_rng(42)
        embeddings, labels_good, labels_random = self._make_clustered_data(rng)

        diags = diagnose_axes(
            embeddings,
            {"clustered": labels_good, "random": labels_random},
            k=10,
            sample_n=0,  # no subsampling
        )
        # Find each axis
        by_name = {d.name: d for d in diags}
        assert by_name["clustered"].lift > by_name["random"].lift
        # Clustered axis should have near-perfect purity
        assert by_name["clustered"].knn_purity > 0.95
        # Random axis lift should be close to 1.0 (no better than chance)
        assert by_name["random"].lift < 2.0

    def test_sorted_by_lift_ascending(self):
        """Results should be sorted worst-first (lowest lift first)."""
        rng = np.random.default_rng(123)
        embeddings, labels_good, labels_random = self._make_clustered_data(rng)

        diags = diagnose_axes(
            embeddings,
            {"clustered": labels_good, "random": labels_random},
            k=10,
            sample_n=0,
        )
        assert diags[0].name == "random"
        assert diags[1].name == "clustered"
        assert diags[0].lift <= diags[1].lift

    def test_random_baseline_is_herfindahl(self):
        """The random baseline should match the Herfindahl index."""
        rng = np.random.default_rng(99)
        n = 300
        embeddings = rng.standard_normal((n, 16)).astype(np.float32)
        # 60% A, 40% B → Herfindahl = 0.36 + 0.16 = 0.52
        labels = np.array(["A"] * 180 + ["B"] * 120)

        diags = diagnose_axes(
            embeddings, {"test": labels}, k=10, sample_n=0,
        )
        expected_herf = (0.6**2) + (0.4**2)
        assert abs(diags[0].random_baseline - expected_herf) < 1e-6

    def test_n_classes_correct(self):
        rng = np.random.default_rng(7)
        n = 300
        embeddings = rng.standard_normal((n, 16)).astype(np.float32)
        labels = np.array(["A"] * 100 + ["B"] * 100 + ["C"] * 100)

        diags = diagnose_axes(
            embeddings, {"tri": labels}, k=10, sample_n=0,
        )
        assert diags[0].n_classes == 3

    def test_subsampling(self):
        """With sample_n < n, should still produce valid results."""
        rng = np.random.default_rng(42)
        embeddings, labels_good, _ = self._make_clustered_data(
            rng, n_per_class=500,
        )
        diags = diagnose_axes(
            embeddings,
            {"clustered": labels_good},
            k=10,
            sample_n=200,
        )
        # Should still detect the separation even with subsampling
        assert diags[0].knn_purity > 0.90

    def test_list_labels_accepted(self):
        """label_columns values can be plain lists, not just arrays."""
        rng = np.random.default_rng(42)
        n = 100
        embeddings = rng.standard_normal((n, 16)).astype(np.float32)
        labels = ["A"] * 50 + ["B"] * 50

        diags = diagnose_axes(
            embeddings, {"test": labels}, k=5, sample_n=0,
        )
        assert len(diags) == 1
        assert isinstance(diags[0], AxisDiagnostic)

    def test_repr(self):
        d = AxisDiagnostic("test", 0.95, 0.5, 1.9, 3)
        r = repr(d)
        assert "test" in r
        assert "1.9x" in r


# ── discover_categorical_columns ─────────────────────────────────────────


class TestDiscoverCategoricalColumns:
    def test_string_columns_detected(self):
        """String columns with bounded cardinality are detected."""
        import polars as pl

        df = pl.DataFrame({
            "text": ["hello world " * 10] * 100,
            "category": (["A"] * 40 + ["B"] * 30 + ["C"] * 30),
            "embedding": [[0.1, 0.2]] * 100,
        })
        result = discover_categorical_columns(df, text_col="text")
        assert "category" in result
        assert "text" not in result
        assert "embedding" not in result
        assert len(result["category"]) == 100

    def test_high_cardinality_skipped(self):
        """Columns with too many unique values are skipped."""
        import polars as pl

        df = pl.DataFrame({
            "text": [f"text_{i}" for i in range(200)],
            "id": [f"id_{i}" for i in range(200)],
            "category": (["A"] * 100 + ["B"] * 100),
            "embedding": [[0.1]] * 200,
        })
        result = discover_categorical_columns(df, text_col="text", max_cardinality=10)
        assert "id" not in result
        assert "category" in result

    def test_single_value_skipped(self):
        """Columns with only one unique value are skipped (min_cardinality=2)."""
        import polars as pl

        df = pl.DataFrame({
            "text": ["hello"] * 50,
            "constant": ["same"] * 50,
            "category": (["A"] * 25 + ["B"] * 25),
            "embedding": [[0.1]] * 50,
        })
        result = discover_categorical_columns(df, text_col="text")
        assert "constant" not in result
        assert "category" in result

    def test_list_str_columns_coarsened(self):
        """List[str] columns are coarsened with first_term strategy."""
        import polars as pl

        df = pl.DataFrame({
            "text": ["a"] * 60,
            "tags": [["Forceps, bipolar"]] * 30 + [["Catheter, urinary"]] * 30,
            "embedding": [[0.1]] * 60,
        })
        result = discover_categorical_columns(df, text_col="text")
        assert "tags" in result
        assert result["tags"][0] == "forceps"
        assert result["tags"][30] == "catheter"

    def test_free_text_skipped(self):
        """Columns with very long average strings are skipped as free text."""
        import polars as pl

        long_text = "a " * 200  # avg length > 100
        df = pl.DataFrame({
            "text": [long_text] * 50,
            "description": [long_text + str(i) for i in range(50)],
            "category": (["A"] * 25 + ["B"] * 25),
            "embedding": [[0.1]] * 50,
        })
        result = discover_categorical_columns(df, text_col="text")
        assert "description" not in result

    def test_empty_dataframe(self):
        """Empty DataFrame returns empty dict."""
        import polars as pl

        df = pl.DataFrame({"text": [], "embedding": []})
        result = discover_categorical_columns(df, text_col="text")
        assert result == {}


# ── embed_with_diagnostics ───────────────────────────────────────────────


class TestEmbedWithDiagnostics:
    def _make_data(self, rng, n=400, d=32):
        """Create embeddings that separate axis_good but not axis_weak."""
        # Two clusters separated on dim 0
        emb_a = rng.standard_normal((n // 2, d)).astype(np.float32)
        emb_b = rng.standard_normal((n // 2, d)).astype(np.float32)
        emb_a[:, 0] += 5.0
        emb_b[:, 0] -= 5.0
        embeddings = np.vstack([emb_a, emb_b])

        # axis_good aligns with the cluster structure
        axis_good = np.array(["A"] * (n // 2) + ["B"] * (n // 2))
        # axis_weak is random — not separated by embedding
        axis_weak = rng.choice(["X", "Y"], size=n)

        texts = [f"item {i}" for i in range(n)]
        return embeddings, texts, axis_good, axis_weak

    def test_no_promotion_when_all_strong(self):
        """If all axes have high lift, no re-embedding occurs."""
        rng = np.random.default_rng(42)
        embeddings, texts, axis_good, _ = self._make_data(rng)

        call_count = [0]

        def mock_embed(t):
            call_count[0] += 1
            return embeddings  # shouldn't be called

        result_emb, before, after, result_texts = embed_with_diagnostics(
            embeddings, texts,
            {"good": axis_good},
            embed_fn=mock_embed,
            lift_threshold=1.5,
        )
        # No re-embedding should happen
        assert call_count[0] == 0
        assert before is after  # same object reference
        np.testing.assert_array_equal(result_emb, embeddings)

    def test_weak_axis_triggers_reembedding(self):
        """Under-served axes trigger re-embedding with structured text."""
        rng = np.random.default_rng(42)
        embeddings, texts, axis_good, axis_weak = self._make_data(rng)

        call_count = [0]

        def mock_embed(t):
            call_count[0] += 1
            # Return embeddings that actually separate axis_weak
            n = len(t)
            new_emb = rng.standard_normal((n, 32)).astype(np.float32)
            return new_emb

        result_emb, before, after, result_texts = embed_with_diagnostics(
            embeddings, texts,
            {"good": axis_good, "weak": axis_weak},
            embed_fn=mock_embed,
            lift_threshold=3.0,
        )
        # Re-embedding should have been called
        assert call_count[0] == 1
        # Before and after should be different objects
        assert before is not after

    def test_structured_text_format(self):
        """Promoted axes should appear as 'axis: value' in text."""
        rng = np.random.default_rng(42)
        embeddings, texts, _, axis_weak = self._make_data(rng)

        def mock_embed(t):
            return rng.standard_normal((len(t), 32)).astype(np.float32)

        _, _, _, result_texts = embed_with_diagnostics(
            embeddings, texts,
            {"weak": axis_weak},
            embed_fn=mock_embed,
            lift_threshold=100.0,  # force promotion
        )
        # Check that at least some texts have the axis prefix
        has_prefix = sum(1 for t in result_texts if "weak:" in t)
        assert has_prefix > 0

    def test_prefix_applied(self):
        """The prefix is prepended to structured texts."""
        rng = np.random.default_rng(42)
        embeddings, texts, _, axis_weak = self._make_data(rng)

        def mock_embed(t):
            return rng.standard_normal((len(t), 32)).astype(np.float32)

        _, _, _, result_texts = embed_with_diagnostics(
            embeddings, texts,
            {"weak": axis_weak},
            embed_fn=mock_embed,
            lift_threshold=100.0,
            prefix="search_document: ",
        )
        for t in result_texts:
            assert t.startswith("search_document: ")

    def test_unknown_values_skipped_in_text(self):
        """_unknown_ and empty values are not included in structured text."""
        rng = np.random.default_rng(42)
        n = 100
        embeddings = rng.standard_normal((n, 16)).astype(np.float32)
        texts = [f"item {i}" for i in range(n)]
        labels = np.array(["_unknown_"] * 50 + ["valid"] * 50)

        def mock_embed(t):
            return rng.standard_normal((len(t), 16)).astype(np.float32)

        _, _, _, result_texts = embed_with_diagnostics(
            embeddings, texts,
            {"axis": labels},
            embed_fn=mock_embed,
            lift_threshold=100.0,
        )
        # First 50 items should NOT have "axis:" in text (value is _unknown_)
        for t in result_texts[:50]:
            assert "axis:" not in t
        # Last 50 should have it
        for t in result_texts[50:]:
            assert "axis: valid" in t


# ── diagnostics_to_metadata ──────────────────────────────────────────────


class TestDiagnosticsToMetadata:
    def test_roundtrip_serialization(self):
        """Diagnostics serialize to valid JSON and preserve values."""
        import json

        before = [
            AxisDiagnostic("gmdn", 0.929, 0.040, 23.0, 150),
            AxisDiagnostic("polarity", 0.978, 0.445, 2.2, 3),
        ]
        after = [
            AxisDiagnostic("gmdn", 0.935, 0.040, 23.4, 150),
            AxisDiagnostic("polarity", 0.990, 0.445, 2.2, 3),
        ]
        meta = diagnostics_to_metadata(before, after)

        assert "axis_diagnostics_before" in meta
        assert "axis_diagnostics_after" in meta

        parsed_before = json.loads(meta["axis_diagnostics_before"])
        assert len(parsed_before) == 2
        assert parsed_before[0]["name"] == "gmdn"
        assert parsed_before[0]["lift"] == 23.0

        parsed_after = json.loads(meta["axis_diagnostics_after"])
        assert parsed_after[0]["name"] == "gmdn"
        assert parsed_after[0]["lift"] == 23.4

    def test_empty_lists(self):
        """Empty diagnostic lists produce valid JSON."""
        import json

        meta = diagnostics_to_metadata([], [])
        assert json.loads(meta["axis_diagnostics_before"]) == []
        assert json.loads(meta["axis_diagnostics_after"]) == []
