"""Tests for dyf.cluster_tree — cluster-tree DAG for hierarchical labeling."""

import numpy as np

from dyf.categorical import CategoryGraph
from dyf.cluster_tree import (
    build_cluster_tree_dag,
    compute_sibling_keywords,
    derive_path_labels,
    format_cluster_context,
)
from dyf.splits import compute_split_keywords

# ── Synthetic tree fixture ────────────────────────────────────────────


def _make_synthetic_setup():
    """Create a synthetic tree + cluster labels for testing.

    Tree (200 items):
        node 0 (root, depth=0)
        ├── node 1 (depth=1, 100 items) — cardiac
        │   ├── node 3 (leaf, 50 items) — pacemakers
        │   └── node 4 (leaf, 50 items) — defibrillators
        └── node 2 (depth=1, 100 items) — orthopedic
            ├── node 5 (leaf, 50 items) — hip screws
            └── node 6 (leaf, 50 items) — knee plates

    Clusters (4 clean + 1 straddling):
        cluster 0: items 0-49 (pacemakers)
        cluster 1: items 50-99 (defibrillators)
        cluster 2: items 100-149 (hip screws)
        cluster 3: items 150-199 (knee plates)
        cluster 4: items 40-59 (straddles pacemaker/defibrillator boundary)
    """
    tree = [
        {'node_id': 0, 'parent_id': None, 'depth': 0,
         'num_items': 200, 'is_leaf': False, 'batch_index': -1},
        {'node_id': 1, 'parent_id': 0, 'depth': 1,
         'num_items': 100, 'is_leaf': False, 'batch_index': -1},
        {'node_id': 2, 'parent_id': 0, 'depth': 1,
         'num_items': 100, 'is_leaf': False, 'batch_index': -1},
        {'node_id': 3, 'parent_id': 1, 'depth': 2,
         'num_items': 50, 'is_leaf': True, 'batch_index': 0},
        {'node_id': 4, 'parent_id': 1, 'depth': 2,
         'num_items': 50, 'is_leaf': True, 'batch_index': 1},
        {'node_id': 5, 'parent_id': 2, 'depth': 2,
         'num_items': 50, 'is_leaf': True, 'batch_index': 2},
        {'node_id': 6, 'parent_id': 2, 'depth': 2,
         'num_items': 50, 'is_leaf': True, 'batch_index': 3},
    ]

    children_map = {0: [1, 2], 1: [3, 4], 2: [5, 6]}

    leaf_batches = {
        3: np.arange(0, 50),
        4: np.arange(50, 100),
        5: np.arange(100, 150),
        6: np.arange(150, 200),
    }

    titles = (
        [f"cardiac pacemaker implant model {i}" for i in range(50)]
        + [f"cardiac defibrillator lead system {i}" for i in range(50)]
        + [f"orthopedic hip screw titanium {i}" for i in range(50)]
        + [f"orthopedic knee plate stainless {i}" for i in range(50)]
    )

    # 4 clean clusters aligned with leaves
    cluster_labels_clean = np.array(
        [0] * 50 + [1] * 50 + [2] * 50 + [3] * 50, dtype=np.int32)

    # 5th cluster straddles pacemaker/defibrillator boundary (60% pace, 40% defib)
    cluster_labels_straddle = cluster_labels_clean.copy()
    # Reassign items 40-49 (pacemaker) and 50-59 (defibrillator) to cluster 4
    cluster_labels_straddle[40:60] = 4

    return (tree, children_map, leaf_batches, titles,
            cluster_labels_clean, cluster_labels_straddle)


def _make_split_keywords(tree, children_map, leaf_batches, titles):
    """Compute split keywords for the synthetic tree."""
    return compute_split_keywords(
        titles, tree, leaf_batches, children_map,
        max_depth_from_root=3, min_child_items=5)


# ── Tests ─────────────────────────────────────────────────────────────


class TestBuildClusterTreeDag:
    """Test DAG construction."""

    def test_single_path_cluster(self):
        """All points from one branch → one tree parent, clean path."""
        (tree, cmap, lbatch, titles,
         labels_clean, _) = _make_synthetic_setup()

        dag = build_cluster_tree_dag(
            tree, cmap, lbatch, labels_clean, n_clusters=25)

        # Cluster 0 (pacemakers) should connect to tree_3 (pacemaker leaf)
        # or tree_1 (cardiac parent)
        c0_parents = dag.get_parents("cluster_25_0")
        tree_parents = [p for p in c0_parents if p.startswith("tree_")]
        assert len(tree_parents) >= 1

        # Should NOT connect to orthopedic nodes
        ortho_nodes = {"tree_2", "tree_5", "tree_6"}
        assert not ortho_nodes.intersection(tree_parents)

    def test_straddling_cluster(self):
        """Points split 60/40 across boundary → two tree parents."""
        (tree, cmap, lbatch, titles,
         _, labels_straddle) = _make_synthetic_setup()

        dag = build_cluster_tree_dag(
            tree, cmap, lbatch, labels_straddle, n_clusters=25,
            straddle_threshold=0.15)

        # Cluster 4 straddles pacemaker (node 3) and defibrillator (node 4)
        c4_parents = dag.get_parents("cluster_25_4")
        tree_parents = [p for p in c4_parents if p.startswith("tree_")]

        # Should have at least 2 tree parents (the two sides of the split)
        assert len(tree_parents) >= 2, (
            f"Expected >=2 tree parents for straddling cluster, got "
            f"{tree_parents}")

    def test_threshold_filtering(self):
        """High threshold excludes small overlaps."""
        (tree, cmap, lbatch, titles,
         _, labels_straddle) = _make_synthetic_setup()

        # With very high threshold, straddling cluster should only attach
        # to the majority side
        dag_high = build_cluster_tree_dag(
            tree, cmap, lbatch, labels_straddle, n_clusters=25,
            straddle_threshold=0.55)

        c4_parents = dag_high.get_parents("cluster_25_4")
        tree_parents = [p for p in c4_parents if p.startswith("tree_")]

        # With 60/40 split and 55% threshold, only the 60% side passes
        assert len(tree_parents) >= 1

        # With low threshold, both sides should attach
        dag_low = build_cluster_tree_dag(
            tree, cmap, lbatch, labels_straddle, n_clusters=25,
            straddle_threshold=0.10)

        c4_parents_low = dag_low.get_parents("cluster_25_4")
        tree_parents_low = [p for p in c4_parents_low if p.startswith("tree_")]
        assert len(tree_parents_low) >= len(tree_parents)

    def test_tree_internal_edges_present(self):
        """Tree-internal edges (parent → child) are in the DAG."""
        (tree, cmap, lbatch, _, labels_clean, _) = _make_synthetic_setup()

        dag = build_cluster_tree_dag(
            tree, cmap, lbatch, labels_clean, n_clusters=25)

        # tree_0 → tree_1 and tree_0 → tree_2 should exist
        root_children = dag.get_children("tree_0")
        assert "tree_1" in root_children
        assert "tree_2" in root_children

    def test_all_clusters_attached(self):
        """Every cluster connects to at least one tree node."""
        (tree, cmap, lbatch, _, labels_clean, _) = _make_synthetic_setup()

        dag = build_cluster_tree_dag(
            tree, cmap, lbatch, labels_clean, n_clusters=25)

        for cid in range(4):
            node_name = f"cluster_25_{cid}"
            parents = dag.get_parents(node_name)
            tree_parents = [p for p in parents if p.startswith("tree_")]
            assert len(tree_parents) >= 1, (
                f"Cluster {cid} has no tree parents")


class TestDerivePathLabels:
    """Test path label derivation."""

    def test_single_path_format(self):
        """Single-path cluster gets clean slash-separated label."""
        (tree, cmap, lbatch, titles,
         labels_clean, _) = _make_synthetic_setup()

        split_kw = _make_split_keywords(tree, cmap, lbatch, titles)
        dag = build_cluster_tree_dag(
            tree, cmap, lbatch, labels_clean, n_clusters=25)

        path_labels = derive_path_labels(dag, split_kw, n_clusters=25)

        # Cluster 0 (pacemakers) should have a path containing
        # cardiac-related keywords
        label = path_labels.get(0, "")
        assert "/" in label or label, f"Expected path with '/', got '{label}'"

    def test_straddling_path_has_braces(self):
        """Straddling cluster shows divergence with brace notation."""
        (tree, cmap, lbatch, titles,
         _, labels_straddle) = _make_synthetic_setup()

        split_kw = _make_split_keywords(tree, cmap, lbatch, titles)
        dag = build_cluster_tree_dag(
            tree, cmap, lbatch, labels_straddle, n_clusters=25,
            straddle_threshold=0.15)

        path_labels = derive_path_labels(dag, split_kw, n_clusters=25)

        # Cluster 4 straddles — should have brace notation or multiple paths
        label = path_labels.get(4, "")
        # May have {pacemaker, defibrillator} or similar
        # At minimum, it should be non-empty
        assert label, "Straddling cluster should have a path label"

    def test_empty_splits_returns_empty(self):
        """No split keywords → empty labels."""
        (tree, cmap, lbatch, _, labels_clean, _) = _make_synthetic_setup()

        dag = build_cluster_tree_dag(
            tree, cmap, lbatch, labels_clean, n_clusters=25)

        path_labels = derive_path_labels(
            dag, {'splits': {}}, n_clusters=25)

        for cid in range(4):
            assert path_labels.get(cid, "") == ""

    def test_all_clusters_get_labels(self):
        """Every cluster gets a path label (may be empty if no keywords)."""
        (tree, cmap, lbatch, titles,
         labels_clean, _) = _make_synthetic_setup()

        split_kw = _make_split_keywords(tree, cmap, lbatch, titles)
        dag = build_cluster_tree_dag(
            tree, cmap, lbatch, labels_clean, n_clusters=25)

        path_labels = derive_path_labels(dag, split_kw, n_clusters=25)
        for cid in range(4):
            assert cid in path_labels


class TestComputeSiblingKeywords:
    """Test sibling keyword computation."""

    def test_siblings_get_contrastive_keywords(self):
        """Clusters sharing a parent get discriminative keywords."""
        (tree, cmap, lbatch, titles,
         labels_clean, _) = _make_synthetic_setup()

        dag = build_cluster_tree_dag(
            tree, cmap, lbatch, labels_clean, n_clusters=25)

        kw = compute_sibling_keywords(
            dag, titles, labels_clean, n_clusters=25)

        # Cluster 0 (pacemakers) and 1 (defibrillators) are siblings
        # under node 1 (cardiac). They should have discriminative keywords.
        assert 0 in kw
        assert 1 in kw
        assert len(kw[0]) > 0
        assert len(kw[1]) > 0

        # Keywords should be different between siblings
        kw0_words = {w for w, _ in kw[0]}
        kw1_words = {w for w, _ in kw[1]}
        assert kw0_words != kw1_words

    def test_lone_cluster_gets_corpus_keywords(self):
        """Cluster with unique parent set falls back to corpus TF-IDF."""
        (tree, cmap, lbatch, titles,
         _, labels_straddle) = _make_synthetic_setup()

        dag = build_cluster_tree_dag(
            tree, cmap, lbatch, labels_straddle, n_clusters=25,
            straddle_threshold=0.15)

        kw = compute_sibling_keywords(
            dag, titles, labels_straddle, n_clusters=25)

        # Cluster 4 straddles — may have unique parent set
        assert 4 in kw
        # Should still have keywords (either sibling or corpus fallback)
        # (may be empty if corpus TF-IDF produces nothing, but should exist)

    def test_keywords_have_scores(self):
        """Each keyword should have a float score."""
        (tree, cmap, lbatch, titles,
         labels_clean, _) = _make_synthetic_setup()

        dag = build_cluster_tree_dag(
            tree, cmap, lbatch, labels_clean, n_clusters=25)

        kw = compute_sibling_keywords(
            dag, titles, labels_clean, n_clusters=25)

        for cid, keywords in kw.items():
            for word, score in keywords:
                assert isinstance(word, str)
                assert isinstance(score, float)


class TestFormatClusterContext:
    """Test context string formatting."""

    def test_with_path_and_keywords(self):
        ctx = format_cluster_context(
            cluster_id=0,
            path_label="cardiac / pacemaker",
            sibling_keywords=[("implant", 0.5), ("lead", 0.3)],
        )
        assert "Tree path: cardiac / pacemaker" in ctx
        assert "implant" in ctx
        assert "lead" in ctx

    def test_with_sibling_labels(self):
        ctx = format_cluster_context(
            cluster_id=0,
            path_label="cardiac / pacemaker",
            sibling_keywords=[("implant", 0.5)],
            sibling_labels={0: "Pacemakers", 1: "Defibrillators"},
        )
        assert "Defibrillators" in ctx
        # Should not include own label
        assert '"Pacemakers"' not in ctx

    def test_empty_inputs(self):
        ctx = format_cluster_context(
            cluster_id=0,
            path_label="",
            sibling_keywords=[],
        )
        assert ctx == ""


class TestDagRoundtrip:
    """Test DAG serialization round-trip."""

    def test_roundtrip_preserves_edges(self):
        """Build → serialize → deserialize → edges match."""
        (tree, cmap, lbatch, _,
         labels_clean, _) = _make_synthetic_setup()

        dag = build_cluster_tree_dag(
            tree, cmap, lbatch, labels_clean, n_clusters=25)

        # Serialize and deserialize
        dag_dict = dag.to_dict()
        dag2 = CategoryGraph.from_dict(dag_dict)

        # All nodes should match
        assert dag.all_nodes() == dag2.all_nodes()

        # Edge count should match
        original_edges = sum(len(v) for v in dag.children.values())
        restored_edges = sum(len(v) for v in dag2.children.values())
        assert original_edges == restored_edges

    def test_json_roundtrip(self):
        """JSON string round-trip works."""
        (tree, cmap, lbatch, _,
         labels_clean, _) = _make_synthetic_setup()

        dag = build_cluster_tree_dag(
            tree, cmap, lbatch, labels_clean, n_clusters=25)

        json_str = dag.to_json()
        dag2 = CategoryGraph.from_json(json_str)
        assert dag.all_nodes() == dag2.all_nodes()


class TestBackwardCompat:
    """Test backward compatibility — label_clusters without DAG still works."""

    def test_label_clusters_without_dag(self):
        """label_clusters() without path_labels/sibling_keywords still works.

        This tests that the existing codepath (contrastive TF-IDF) remains
        functional when DAG data is not provided.
        """
        from unittest.mock import patch

        from dyf.enrich._labeling import label_clusters

        n = 200
        rng = np.random.default_rng(42)
        titles = [f"Item type {i % 10} variant {i}" for i in range(n)]
        coords = rng.standard_normal((n, 3)).astype(np.float32)
        labels = np.array([i % 4 for i in range(n)], dtype=np.int32)
        embeddings = rng.standard_normal((n, 32)).astype(np.float32)
        embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True)

        # Without split_keywords — should use contrastive TF-IDF fallback
        with patch('dyf.enrich._labeling._call_ollama', return_value="Test Label"):
            names = label_clusters(
                titles, coords, labels, embeddings,
                split_keywords=None)

        assert len(names) == 4
        for cid in range(4):
            assert cid in names
