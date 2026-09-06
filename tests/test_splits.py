"""Tests for dyf.splits — split-based TF-IDF keyword extraction."""

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


# ── Pure function tests ─────────────────────────────────────────────


class TestComputeDomainStopwords:
    """Test domain stop word detection."""

    def test_high_frequency_words_detected(self):
        from dyf.splits import compute_domain_stopwords

        # "device" appears in all titles → domain stop word
        titles = [
            "cardiac device implant",
            "orthopedic device screw",
            "dental device crown",
            "surgical device instrument",
            "cardiac device monitor",
        ]
        sw = compute_domain_stopwords(titles, threshold=0.5)
        assert "device" in sw

    def test_low_frequency_words_not_stopped(self):
        from dyf.splits import compute_domain_stopwords

        titles = [
            "cardiac pacemaker implant",
            "orthopedic hip screw",
            "dental crown bridge",
            "surgical instrument set",
        ]
        sw = compute_domain_stopwords(titles, threshold=0.5)
        assert "cardiac" not in sw
        assert "pacemaker" not in sw

    def test_empty_titles(self):
        from dyf.splits import compute_domain_stopwords

        sw = compute_domain_stopwords([], threshold=0.1)
        assert sw == set()

    def test_threshold_sensitivity(self):
        from dyf.splits import compute_domain_stopwords

        titles = [
            "medical device system",
            "medical device unit",
            "medical device kit",
            "surgical instrument pack",
        ]
        # "medical" appears in 3/4 = 75%, "device" in 3/4 = 75%
        sw_low = compute_domain_stopwords(titles, threshold=0.5)
        assert "medical" in sw_low
        assert "device" in sw_low

        sw_high = compute_domain_stopwords(titles, threshold=0.8)
        assert "medical" not in sw_high


class TestCollectDescendantIndices:
    """Test the public collect_descendant_indices function."""

    def test_leaf_node(self):
        from dyf.splits import collect_descendant_indices

        leaf_batches = {10: np.array([0, 1, 2])}
        children_map = {}
        result = collect_descendant_indices(10, children_map, leaf_batches)
        np.testing.assert_array_equal(result, [0, 1, 2])

    def test_internal_node(self):
        from dyf.splits import collect_descendant_indices

        leaf_batches = {
            2: np.array([10, 11]),
            3: np.array([20, 21, 22]),
        }
        children_map = {1: [2, 3]}
        result = collect_descendant_indices(1, children_map, leaf_batches)
        assert set(result.tolist()) == {10, 11, 20, 21, 22}

    def test_deep_tree(self):
        from dyf.splits import collect_descendant_indices

        leaf_batches = {
            4: np.array([100]),
            5: np.array([200, 201]),
        }
        children_map = {1: [2, 3], 2: [4], 3: [5]}
        result = collect_descendant_indices(1, children_map, leaf_batches)
        assert set(result.tolist()) == {100, 200, 201}


class TestComputeSplitKeywords:
    """Test split keyword computation with synthetic data."""

    def _make_synthetic_tree(self):
        """Create a simple synthetic tree structure for testing.

        Tree:
            node 0 (root, depth=0, 200 items)
            ├── node 1 (depth=1, 100 items) — cardiac devices
            │   ├── node 3 (leaf, depth=2, 50 items) — pacemakers
            │   └── node 4 (leaf, depth=2, 50 items) — defibrillators
            └── node 2 (depth=1, 100 items) — orthopedic devices
                ├── node 5 (leaf, depth=2, 50 items) — hip screws
                └── node 6 (leaf, depth=2, 50 items) — knee plates
        """
        tree = [
            {"node_id": 0, "parent_id": None, "depth": 0, "num_items": 200, "is_leaf": False, "batch_index": -1},
            {"node_id": 1, "parent_id": 0, "depth": 1, "num_items": 100, "is_leaf": False, "batch_index": -1},
            {"node_id": 2, "parent_id": 0, "depth": 1, "num_items": 100, "is_leaf": False, "batch_index": -1},
            {"node_id": 3, "parent_id": 1, "depth": 2, "num_items": 50, "is_leaf": True, "batch_index": 0},
            {"node_id": 4, "parent_id": 1, "depth": 2, "num_items": 50, "is_leaf": True, "batch_index": 1},
            {"node_id": 5, "parent_id": 2, "depth": 2, "num_items": 50, "is_leaf": True, "batch_index": 2},
            {"node_id": 6, "parent_id": 2, "depth": 2, "num_items": 50, "is_leaf": True, "batch_index": 3},
        ]

        children_map = {
            0: [1, 2],
            1: [3, 4],
            2: [5, 6],
        }

        leaf_batches = {
            3: np.arange(0, 50),
            4: np.arange(50, 100),
            5: np.arange(100, 150),
            6: np.arange(150, 200),
        }

        # Titles with distinct vocabulary per leaf
        titles = []
        for i in range(50):
            titles.append(f"cardiac pacemaker implant model {i}")
        for i in range(50):
            titles.append(f"cardiac defibrillator lead system {i}")
        for i in range(50):
            titles.append(f"orthopedic hip screw titanium {i}")
        for i in range(50):
            titles.append(f"orthopedic knee plate stainless {i}")

        return tree, children_map, leaf_batches, titles

    def test_produces_splits(self):
        from dyf.splits import compute_split_keywords

        tree, cmap, lbatch, titles = self._make_synthetic_tree()
        result = compute_split_keywords(titles, tree, lbatch, cmap, max_depth_from_root=3)

        assert "splits" in result
        assert "domain_stopwords" in result
        assert len(result["splits"]) > 0

    def test_root_split_discriminates(self):
        from dyf.splits import compute_split_keywords

        tree, cmap, lbatch, titles = self._make_synthetic_tree()
        result = compute_split_keywords(titles, tree, lbatch, cmap, max_depth_from_root=2)

        # Root node (0) should have a split
        root_split = result["splits"].get(0)
        assert root_split is not None
        assert len(root_split["children"]) == 2

        # Children should have discriminative keywords
        child_keywords = {}
        for cid, cinfo in root_split["children"].items():
            words = [w for w, _ in cinfo["unigrams"]]
            child_keywords[cid] = words

        # Node 1 (cardiac) and node 2 (orthopedic) should have
        # their respective keywords
        all_words = set()
        for words in child_keywords.values():
            all_words.update(words)

        # At least one child should have cardiac-related keywords
        # and the other orthopedic-related keywords
        assert "cardiac" in all_words or "pacemaker" in all_words
        assert "orthopedic" in all_words or "hip" in all_words

    def test_depth_limit(self):
        from dyf.splits import compute_split_keywords

        tree, cmap, lbatch, titles = self._make_synthetic_tree()

        # depth=1: only root split
        result_d1 = compute_split_keywords(titles, tree, lbatch, cmap, max_depth_from_root=1)
        assert len(result_d1["splits"]) == 1
        assert 0 in result_d1["splits"]

        # depth=2: root + depth-1 splits
        result_d2 = compute_split_keywords(titles, tree, lbatch, cmap, max_depth_from_root=2)
        assert len(result_d2["splits"]) >= 1

    def test_min_child_items_filter(self):
        from dyf.splits import compute_split_keywords

        tree, cmap, lbatch, titles = self._make_synthetic_tree()

        # Set min_child_items very high — should skip all splits
        result = compute_split_keywords(titles, tree, lbatch, cmap, min_child_items=1000)
        assert len(result["splits"]) == 0

    def test_domain_stopwords_applied(self):
        from dyf.splits import compute_split_keywords

        tree, cmap, lbatch, titles = self._make_synthetic_tree()

        # Without domain stopwords
        compute_split_keywords(titles, tree, lbatch, cmap, max_depth_from_root=2)

        # With "cardiac" and "orthopedic" as domain stopwords
        result_sw = compute_split_keywords(
            titles, tree, lbatch, cmap, max_depth_from_root=2, domain_stopwords={"cardiac", "orthopedic"}
        )

        # With stopwords, those words should not appear in keywords
        for split in result_sw["splits"].values():
            for cinfo in split["children"].values():
                words = [w for w, _ in cinfo["unigrams"]]
                assert "cardiac" not in words
                assert "orthopedic" not in words

    def test_bigram_check_produces_bigrams(self):
        from dyf.splits import compute_split_keywords

        tree, cmap, lbatch, titles = self._make_synthetic_tree()
        result = compute_split_keywords(titles, tree, lbatch, cmap, max_depth_from_root=2, bigram_check=True)

        # Each child should have a 'bigrams' key
        for split in result["splits"].values():
            assert "bigram_needed" in split
            for cinfo in split["children"].values():
                assert "bigrams" in cinfo

    def test_bigram_needed_false_for_clean_splits(self):
        from dyf.splits import compute_split_keywords

        tree, cmap, lbatch, titles = self._make_synthetic_tree()
        result = compute_split_keywords(titles, tree, lbatch, cmap, max_depth_from_root=2, bigram_check=True)

        # Our synthetic data has clean splits — bigram should not be needed
        for split in result["splits"].values():
            assert split["bigram_needed"] is False


class TestFormatSplitPath:
    """Test path formatting for individual items."""

    def test_returns_path_for_item(self):
        from dyf.splits import (
            compute_split_keywords,
            format_split_path,
        )

        # Build same synthetic tree
        tree = [
            {"node_id": 0, "parent_id": None, "depth": 0, "num_items": 200, "is_leaf": False, "batch_index": -1},
            {"node_id": 1, "parent_id": 0, "depth": 1, "num_items": 100, "is_leaf": False, "batch_index": -1},
            {"node_id": 2, "parent_id": 0, "depth": 1, "num_items": 100, "is_leaf": False, "batch_index": -1},
            {"node_id": 3, "parent_id": 1, "depth": 2, "num_items": 50, "is_leaf": True, "batch_index": 0},
            {"node_id": 4, "parent_id": 1, "depth": 2, "num_items": 50, "is_leaf": True, "batch_index": 1},
            {"node_id": 5, "parent_id": 2, "depth": 2, "num_items": 50, "is_leaf": True, "batch_index": 2},
            {"node_id": 6, "parent_id": 2, "depth": 2, "num_items": 50, "is_leaf": True, "batch_index": 3},
        ]
        children_map = {0: [1, 2], 1: [3, 4], 2: [5, 6]}
        leaf_batches = {
            3: np.arange(0, 50),
            4: np.arange(50, 100),
            5: np.arange(100, 150),
            6: np.arange(150, 200),
        }
        titles = []
        for i in range(50):
            titles.append(f"cardiac pacemaker implant model {i}")
        for i in range(50):
            titles.append(f"cardiac defibrillator lead system {i}")
        for i in range(50):
            titles.append(f"orthopedic hip screw titanium {i}")
        for i in range(50):
            titles.append(f"orthopedic knee plate stainless {i}")

        kw = compute_split_keywords(titles, tree, leaf_batches, children_map, max_depth_from_root=3)

        # Item 0 is in leaf 3 (cardiac pacemaker)
        path = format_split_path(0, kw, tree, leaf_batches, children_map, top_k=3)

        assert isinstance(path, list)
        assert len(path) > 0
        # Path should contain cardiac-related keywords
        path_str = " ".join(path)
        assert any(w in path_str for w in ["cardiac", "pacemaker", "implant", "defibrillator"])

    def test_returns_empty_for_missing_item(self):
        from dyf.splits import format_split_path

        tree = [{"node_id": 0, "parent_id": None, "depth": 0, "num_items": 10, "is_leaf": True, "batch_index": 0}]
        leaf_batches = {0: np.arange(10)}
        children_map = {}
        kw = {"splits": {}}

        path = format_split_path(999, kw, tree, leaf_batches, children_map)
        assert path == []


# ── Integration with LazyIndex ──────────────────────────────────────


@lazy_deps
class TestBuildTreeMaps:
    """Test build_tree_maps with a real DYF index."""

    def test_builds_maps_from_index(self):
        from dyf import build_dyf_tree
        from dyf.lazy_index import LazyIndex, write_lazy_index
        from dyf.splits import build_tree_maps

        n = 200
        dim = 32
        rng = np.random.default_rng(42)
        embeddings = rng.standard_normal((n, dim)).astype(np.float32)
        embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True)

        tree = build_dyf_tree(embeddings, max_depth=3, num_bits=3, min_leaf_size=4, seed=42)

        with tempfile.NamedTemporaryFile(suffix=".dyf", delete=False) as f:
            path = f.name

        try:
            titles = [f"Item {i}" for i in range(n)]
            write_lazy_index(tree, embeddings, path, quantization="float32", stored_fields={"title": titles})

            with LazyIndex(path) as idx:
                tree_list, cmap, lbatch = build_tree_maps(idx)

            assert len(tree_list) > 0
            assert isinstance(cmap, dict)
            assert isinstance(lbatch, dict)

            # All leaf indices should cover all items
            all_indices = set()
            for indices in lbatch.values():
                all_indices.update(indices.tolist())
            assert len(all_indices) == n
        finally:
            if os.path.exists(path):
                os.unlink(path)


@lazy_deps
class TestComputeEmbeddingKeywords:
    """Test embedding-space keyword projection."""

    def _make_synthetic_data(self):
        """Create synthetic embeddings with known split direction.

        200 items, 32-dim embeddings. Two clusters separated along dim 0.
        Cluster A (indices 0-99): positive dim 0, titles contain "cardiac", "pacemaker"
        Cluster B (indices 100-199): negative dim 0, titles contain "orthopedic", "hip"
        Hyperplane = unit vector along dim 0.
        """
        rng = np.random.default_rng(42)
        dim = 32

        # Cluster A: positive along dim 0
        emb_a = rng.standard_normal((100, dim)).astype(np.float32) * 0.1
        emb_a[:, 0] += 2.0  # strong positive signal on dim 0

        # Cluster B: negative along dim 0
        emb_b = rng.standard_normal((100, dim)).astype(np.float32) * 0.1
        emb_b[:, 0] -= 2.0  # strong negative signal on dim 0

        embeddings = np.concatenate([emb_a, emb_b])

        tree = [
            {"node_id": 0, "parent_id": None, "depth": 0, "num_items": 200, "is_leaf": False, "batch_index": -1},
            {"node_id": 1, "parent_id": 0, "depth": 1, "num_items": 100, "is_leaf": True, "batch_index": 0},
            {"node_id": 2, "parent_id": 0, "depth": 1, "num_items": 100, "is_leaf": True, "batch_index": 1},
        ]
        children_map = {0: [1, 2]}
        leaf_batches = {
            1: np.arange(0, 100),
            2: np.arange(100, 200),
        }

        # Hyperplane along dim 0
        hp = np.zeros((1, dim), dtype=np.float32)
        hp[0, 0] = 1.0
        hyperplanes = {0: hp}

        titles = []
        for i in range(100):
            titles.append(f"cardiac pacemaker implant device model {i}")
        for i in range(100):
            titles.append(f"orthopedic hip screw titanium plate {i}")

        return embeddings, tree, children_map, leaf_batches, hyperplanes, titles

    def test_embedding_keywords_basic(self):
        """Known split direction → correct keywords per side."""
        from dyf.splits import compute_embedding_keywords

        emb, tree, cmap, lbatch, hp, titles = self._make_synthetic_data()
        result = compute_embedding_keywords(
            titles, emb, tree, lbatch, cmap, hp, max_depth_from_root=2, min_child_items=10, min_term_count=3
        )

        assert 0 in result["splits"]
        root_split = result["splits"][0]

        # Collect words per child
        child_words = {}
        for cid, cinfo in root_split["children"].items():
            child_words[cid] = {w for w, _ in cinfo["unigrams"]}

        all_words = set()
        for ws in child_words.values():
            all_words.update(ws)

        # Cardiac terms should be on one side, orthopedic on the other
        cardiac_terms = {"cardiac", "pacemaker", "implant"}
        ortho_terms = {"orthopedic", "hip", "screw", "titanium", "plate"}

        # At least one child should have cardiac terms, other should have ortho
        child_list = list(child_words.values())
        assert (child_list[0] & cardiac_terms and child_list[1] & ortho_terms) or (
            child_list[1] & cardiac_terms and child_list[0] & ortho_terms
        )

    def test_embedding_keywords_format(self):
        """Output matches compute_split_keywords format."""
        from dyf.splits import compute_embedding_keywords

        emb, tree, cmap, lbatch, hp, titles = self._make_synthetic_data()
        result = compute_embedding_keywords(
            titles, emb, tree, lbatch, cmap, hp, max_depth_from_root=2, min_child_items=10, min_term_count=3
        )

        # Top-level keys
        assert "domain_stopwords" in result
        assert "splits" in result
        assert isinstance(result["domain_stopwords"], list)

        for nid, split in result["splits"].items():
            assert "depth" in split
            assert "children" in split
            for cid, cinfo in split["children"].items():
                assert "count" in cinfo
                assert "unigrams" in cinfo
                assert isinstance(cinfo["unigrams"], list)
                for item in cinfo["unigrams"]:
                    assert len(item) == 2
                    assert isinstance(item[0], str)
                    assert isinstance(item[1], float)

    def test_embedding_keywords_min_term_count(self):
        """Rare terms filtered out by min_term_count."""
        from dyf.splits import compute_embedding_keywords

        emb, tree, cmap, lbatch, hp, titles = self._make_synthetic_data()

        # Set very high min_term_count → should filter most terms
        result = compute_embedding_keywords(
            titles, emb, tree, lbatch, cmap, hp, max_depth_from_root=2, min_child_items=10, min_term_count=500
        )

        # With min_term_count=500 and only 100 items per cluster,
        # no terms can meet the threshold → no splits
        assert len(result["splits"]) == 0

    def test_embedding_keywords_domain_stopwords(self):
        """Domain stopwords excluded from results."""
        from dyf.splits import compute_embedding_keywords

        emb, tree, cmap, lbatch, hp, titles = self._make_synthetic_data()

        result = compute_embedding_keywords(
            titles,
            emb,
            tree,
            lbatch,
            cmap,
            hp,
            max_depth_from_root=2,
            min_child_items=10,
            min_term_count=3,
            domain_stopwords={"cardiac", "orthopedic"},
        )

        for split in result["splits"].values():
            for cinfo in split["children"].values():
                words = {w for w, _ in cinfo["unigrams"]}
                assert "cardiac" not in words
                assert "orthopedic" not in words


# ── Text diversity assessment tests ────────────────────────────────


class TestAssessTextDiversity:
    """Test the text diversity gate."""

    def test_mnist_like_low_diversity(self):
        """MNIST-style titles: 10 unique strings over 70K items → low diversity."""
        from dyf.splits import assess_text_diversity

        titles = [f"Digit {i % 10}" for i in range(70_000)]
        report = assess_text_diversity(titles)
        assert not report.is_diverse
        assert report.unique_token_count <= 10
        assert report.unique_title_ratio < 0.05

    def test_gudid_like_high_diversity(self):
        """Diverse product titles with many unique words → passes gate."""
        from dyf.splits import assess_text_diversity

        # Generate titles with enough unique vocabulary words
        words = [
            "cardiac",
            "pacemaker",
            "implant",
            "defibrillator",
            "stent",
            "catheter",
            "orthopedic",
            "titanium",
            "screw",
            "plate",
            "dental",
            "crown",
            "bridge",
            "ceramic",
            "porcelain",
            "surgical",
            "forceps",
            "clamp",
            "retractor",
            "scissors",
            "endoscope",
            "laparoscope",
            "arthroscope",
            "colonoscope",
            "electrode",
            "monitor",
            "sensor",
            "transducer",
            "amplifier",
            "prosthetic",
            "knee",
            "hip",
            "shoulder",
            "ankle",
            "bandage",
            "gauze",
            "dressing",
            "adhesive",
            "suture",
            "syringe",
            "needle",
            "cannula",
            "infusion",
            "tubing",
            "ventilator",
            "respirator",
            "oxygen",
            "humidifier",
            "wheelchair",
            "walker",
            "crutch",
            "brace",
            "splint",
        ]
        rng = np.random.default_rng(42)
        titles = []
        for i in range(2000):
            picked = rng.choice(words, size=4, replace=False)
            titles.append(" ".join(picked))
        report = assess_text_diversity(titles)
        assert report.is_diverse
        assert report.unique_token_count >= 50

    def test_empty_titles(self):
        from dyf.splits import assess_text_diversity

        report = assess_text_diversity([])
        assert not report.is_diverse
        assert report.reason == "empty title list"

    def test_single_repeated_title(self):
        from dyf.splits import assess_text_diversity

        titles = ["Hello World"] * 1000
        report = assess_text_diversity(titles)
        assert not report.is_diverse
        assert report.unique_title_ratio < 0.05

    def test_custom_thresholds(self):
        """Very permissive thresholds → everything passes."""
        from dyf.splits import assess_text_diversity

        titles = [f"Digit {i % 10}" for i in range(100)]
        report = assess_text_diversity(
            titles,
            min_unique_tokens=1,
            min_token_ratio=0.0,
            min_unique_title_ratio=0.0,
        )
        assert report.is_diverse

    def test_report_fields(self):
        from dyf.splits import assess_text_diversity

        titles = ["Alpha Beta Gamma"] * 50
        report = assess_text_diversity(titles)
        assert report.n_items == 50
        assert isinstance(report.unique_token_count, int)
        assert isinstance(report.token_item_ratio, float)
        assert isinstance(report.unique_title_ratio, float)
        assert isinstance(report.is_diverse, bool)
        assert isinstance(report.reason, str)


class TestLabelClustersFrequency:
    """Test frequency-based cluster labeling fallback."""

    def test_basic_labeling(self):
        from dyf.splits import label_clusters_frequency

        titles = ["cardiac pacemaker device"] * 50 + ["orthopedic hip screw"] * 50
        labels = np.array([0] * 50 + [1] * 50)
        result = label_clusters_frequency(titles, labels)

        assert isinstance(result, dict)
        assert len(result) == 2
        assert 0 in result
        assert 1 in result
        # Labels should be non-empty strings
        for cid, name in result.items():
            assert len(name) > 0

    def test_numeric_only_titles_fallback(self):
        """Titles with only numbers → falls back to raw title frequency."""
        from dyf.splits import label_clusters_frequency

        titles = [str(i % 10) for i in range(100)]
        labels = np.array([0] * 50 + [1] * 50)
        result = label_clusters_frequency(titles, labels)

        assert len(result) == 2
        # Should still produce labels (from raw titles)
        for name in result.values():
            assert len(name) > 0

    def test_dedup_suffixes(self):
        """Identical cluster vocabularies → disambiguated with (2), (3)."""
        from dyf.splits import label_clusters_frequency

        # All clusters have identical titles → same TF-IDF → same label
        titles = ["alpha beta gamma"] * 300
        labels = np.array([0] * 100 + [1] * 100 + [2] * 100)
        result = label_clusters_frequency(titles, labels)

        # At least one should have a suffix
        names = list(result.values())
        assert len(set(names)) > 1 or len(names) == 1  # either deduped or single
        # If there are duplicates, they should be disambiguated
        if len(set(names)) < len(names):
            suffixed = [n for n in names if "(" in n]
            assert len(suffixed) > 0

    def test_mnist_style(self):
        """MNIST-like: 10 digit labels, each repeated many times."""
        from dyf.splits import label_clusters_frequency

        titles = [f"Digit {i}" for i in range(10)] * 100
        labels = np.array([i // 100 for i in range(1000)])
        result = label_clusters_frequency(titles, labels)

        assert len(result) == 10
        for name in result.values():
            assert len(name) > 0


@lazy_deps
class TestLouvainClusterLeaves:
    """Test Louvain community detection on tree leaves."""

    def test_louvain_returns_correct_structure(self):
        from dyf import build_dyf_tree
        from dyf.agglomerate import louvain_cluster_leaves
        from dyf.lazy_index import LazyIndex, write_lazy_index

        n = 200
        dim = 32
        rng = np.random.default_rng(42)

        # Create 5 well-separated clusters
        centers = rng.standard_normal((5, dim)).astype(np.float32) * 3.0
        embeddings = []
        for i in range(5):
            pts = centers[i] + rng.standard_normal((40, dim)).astype(np.float32) * 0.1
            embeddings.append(pts)
        embeddings = np.concatenate(embeddings).astype(np.float32)
        embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True)

        tree = build_dyf_tree(embeddings, max_depth=4, num_bits=3, min_leaf_size=4, seed=42)

        with tempfile.NamedTemporaryFile(suffix=".dyf", delete=False) as f:
            path = f.name

        try:
            write_lazy_index(tree, embeddings, path, quantization="float32")

            # Fake UMAP coords
            coords = rng.standard_normal((n, 3)).astype(np.float32)

            with LazyIndex(path) as idx:
                result = louvain_cluster_leaves(idx, coords, embeddings)

            point_labels, lsh_names, lsh_label_data, item_leaf_map, tree_struct = result

            # Labels array shape
            assert point_labels.shape == (n,)
            assert point_labels.dtype == np.int32

            # All points assigned (no -1 labels)
            assert (point_labels >= 0).all()

            # Multiple communities found
            n_communities = len(set(point_labels.tolist()))
            assert n_communities > 1

            # Names dict matches unique labels
            assert isinstance(lsh_names, dict)
            assert len(lsh_names) == n_communities

            # Label data is a list of dicts with required keys
            assert isinstance(lsh_label_data, list)
            assert len(lsh_label_data) == n_communities
            for entry in lsh_label_data:
                assert "x" in entry
                assert "y" in entry
                assert "z" in entry
                assert "size" in entry
                assert "cid" in entry

            # item_leaf_map
            assert item_leaf_map.shape == (n,)
            assert item_leaf_map.dtype == np.int32

            # tree_struct is a list
            assert isinstance(tree_struct, list)
            assert len(tree_struct) > 0
        finally:
            if os.path.exists(path):
                os.unlink(path)

    def test_louvain_degenerate_tree(self):
        """Tree with < 2 leaves returns None tuple."""
        from dyf import build_dyf_tree
        from dyf.agglomerate import louvain_cluster_leaves
        from dyf.lazy_index import LazyIndex, write_lazy_index

        # Very small dataset → might produce single leaf
        n = 5
        dim = 8
        rng = np.random.default_rng(42)
        embeddings = rng.standard_normal((n, dim)).astype(np.float32)
        embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True)

        tree = build_dyf_tree(embeddings, max_depth=1, num_bits=1, min_leaf_size=10, seed=42)

        with tempfile.NamedTemporaryFile(suffix=".dyf", delete=False) as f:
            path = f.name

        try:
            write_lazy_index(tree, embeddings, path, quantization="float32")
            coords = rng.standard_normal((n, 3)).astype(np.float32)

            with LazyIndex(path) as idx:
                tree_struct = idx.get_tree_structure()
                n_leaves = sum(1 for nd in tree_struct if nd["is_leaf"] and nd["batch_index"] >= 0)

                result = louvain_cluster_leaves(idx, coords, embeddings)

            if n_leaves < 2:
                # Positional access still works (backward compat)...
                assert result[0] is None
                assert result[1] == {}
                assert result[2] == []
                # ...but `.ok` is what actually names the condition, instead of making
                # the caller decode four sentinel values.
                assert result.ok is False
                assert result.n_groups == 0
            else:
                # If tree happened to have 2+ leaves, just verify structure
                assert result[0] is not None
                assert result.ok is True
                assert result.n_groups > 0
        finally:
            if os.path.exists(path):
                os.unlink(path)


def _inject_hyperplanes(tree_node, dim, num_bits, rng):
    """Inject synthetic PCA hyperplanes into internal tree nodes.

    The Rust DensityClassifier doesn't expose get_hyperplanes(), so
    build_dyf_tree stores None. This helper injects synthetic hyperplanes
    for testing the FlatBuffers round-trip.
    """
    if tree_node["children"]:
        hp = rng.standard_normal((num_bits, dim)).astype(np.float32)
        hp /= np.linalg.norm(hp, axis=1, keepdims=True)
        tree_node["hyperplanes"] = hp
        for child in tree_node["children"]:
            _inject_hyperplanes(child, dim, num_bits, rng)


@lazy_deps
class TestGetSplitHyperplanes:
    """Test LazyIndex.get_split_hyperplanes()."""

    def test_returns_correct_shapes(self):
        from dyf import build_dyf_tree
        from dyf.lazy_index import LazyIndex, write_lazy_index

        n = 200
        dim = 32
        num_bits = 3
        rng = np.random.default_rng(42)
        embeddings = rng.standard_normal((n, dim)).astype(np.float32)
        embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True)

        tree = build_dyf_tree(embeddings, max_depth=3, num_bits=num_bits, min_leaf_size=4, seed=42)

        # Inject synthetic hyperplanes into internal nodes
        _inject_hyperplanes(tree, dim, num_bits, np.random.default_rng(99))

        with tempfile.NamedTemporaryFile(suffix=".dyf", delete=False) as f:
            path = f.name

        try:
            write_lazy_index(tree, embeddings, path, quantization="float32")

            with LazyIndex(path) as idx:
                hp = idx.get_split_hyperplanes()

            assert isinstance(hp, dict)
            assert len(hp) > 0  # should have internal nodes

            for nid, arr in hp.items():
                assert isinstance(arr, np.ndarray)
                assert arr.ndim == 2
                assert arr.shape[1] == dim  # (num_bits, dim)
        finally:
            if os.path.exists(path):
                os.unlink(path)

    def test_leaves_excluded(self):
        from dyf import build_dyf_tree
        from dyf.lazy_index import LazyIndex, write_lazy_index

        n = 200
        dim = 32
        num_bits = 3
        rng = np.random.default_rng(42)
        embeddings = rng.standard_normal((n, dim)).astype(np.float32)
        embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True)

        tree = build_dyf_tree(embeddings, max_depth=3, num_bits=num_bits, min_leaf_size=4, seed=42)

        # Inject synthetic hyperplanes into internal nodes
        _inject_hyperplanes(tree, dim, num_bits, np.random.default_rng(99))

        with tempfile.NamedTemporaryFile(suffix=".dyf", delete=False) as f:
            path = f.name

        try:
            write_lazy_index(tree, embeddings, path, quantization="float32")

            with LazyIndex(path) as idx:
                hp = idx.get_split_hyperplanes()
                tree_list = idx.get_tree_structure()

            # No leaf nodes should be in hyperplanes dict
            leaf_ids = {n["node_id"] for n in tree_list if n["is_leaf"]}
            for nid in hp:
                assert nid not in leaf_ids
        finally:
            if os.path.exists(path):
                os.unlink(path)


@lazy_deps
class TestComputeDepthFromRoot:
    """Tests for _compute_depth_from_root BFS."""

    def test_simple_tree(self):
        from dyf.splits import _compute_depth_from_root

        tree = [
            {"node_id": 0, "parent_id": None},
            {"node_id": 1, "parent_id": 0},
            {"node_id": 2, "parent_id": 0},
            {"node_id": 3, "parent_id": 1},
        ]
        children_map = {0: [1, 2], 1: [3]}
        depth = _compute_depth_from_root(tree, children_map)
        assert depth[0] == 0
        assert depth[1] == 1
        assert depth[2] == 1
        assert depth[3] == 2

    def test_deep_chain(self):
        from dyf.splits import _compute_depth_from_root

        tree = [{"node_id": i, "parent_id": (i - 1 if i > 0 else None)} for i in range(5)]
        children_map = {i: [i + 1] for i in range(4)}
        depth = _compute_depth_from_root(tree, children_map)
        for i in range(5):
            assert depth[i] == i

    def test_single_root(self):
        from dyf.splits import _compute_depth_from_root

        tree = [{"node_id": 0, "parent_id": None}]
        children_map = {}
        depth = _compute_depth_from_root(tree, children_map)
        assert depth == {0: 0}

    def test_unreachable_nodes(self):
        from dyf.splits import _compute_depth_from_root

        tree = [
            {"node_id": 0, "parent_id": None},
            {"node_id": 1, "parent_id": 0},
            {"node_id": 99, "parent_id": 50},  # disconnected
        ]
        children_map = {0: [1]}
        depth = _compute_depth_from_root(tree, children_map)
        assert 0 in depth
        assert 1 in depth
        assert 99 not in depth


class TestComputeChildTfidf:
    """Tests for _compute_child_tfidf TF-IDF scoring."""

    def test_distinct_vocabularies(self):
        from collections import Counter

        from dyf.splits import _compute_child_tfidf

        child_data = {
            0: {
                "count": 10,
                "word_counts": Counter({"alpha": 5, "beta": 3}),
                "total_words": 8,
                "bigram_counts": Counter(),
                "total_bigrams": 0,
            },
            1: {
                "count": 10,
                "word_counts": Counter({"gamma": 4, "delta": 2}),
                "total_words": 6,
                "bigram_counts": Counter(),
                "total_bigrams": 0,
            },
        }
        result = _compute_child_tfidf(child_data, n_children=2, top_k=5, bigram_check=False)
        words_0 = [w for w, _ in result[0]["unigrams"]]
        words_1 = [w for w, _ in result[1]["unigrams"]]
        assert "alpha" in words_0
        assert "gamma" in words_1
        assert "gamma" not in words_0

    def test_empty_child(self):
        from collections import Counter

        from dyf.splits import _compute_child_tfidf

        child_data = {
            0: {
                "count": 10,
                "word_counts": Counter({"alpha": 5}),
                "total_words": 5,
                "bigram_counts": Counter(),
                "total_bigrams": 0,
            },
            1: {"count": 0, "word_counts": Counter(), "total_words": 0, "bigram_counts": Counter(), "total_bigrams": 0},
        }
        result = _compute_child_tfidf(child_data, n_children=2, top_k=5, bigram_check=False)
        assert result[1]["unigrams"] == []

    def test_shared_vocab_excluded(self):
        from collections import Counter

        from dyf.splits import _compute_child_tfidf

        child_data = {
            0: {
                "count": 10,
                "word_counts": Counter({"common": 5, "unique_a": 3}),
                "total_words": 8,
                "bigram_counts": Counter(),
                "total_bigrams": 0,
            },
            1: {
                "count": 10,
                "word_counts": Counter({"common": 4, "unique_b": 2}),
                "total_words": 6,
                "bigram_counts": Counter(),
                "total_bigrams": 0,
            },
        }
        result = _compute_child_tfidf(child_data, n_children=2, top_k=5, bigram_check=False)
        words_0 = [w for w, _ in result[0]["unigrams"]]
        assert "common" not in words_0
        assert "unique_a" in words_0

    def test_bigram_check(self):
        from collections import Counter

        from dyf.splits import _compute_child_tfidf

        child_data = {
            0: {
                "count": 10,
                "word_counts": Counter({"alpha": 5}),
                "total_words": 5,
                "bigram_counts": Counter({"alpha beta": 3}),
                "total_bigrams": 3,
            },
            1: {
                "count": 10,
                "word_counts": Counter({"gamma": 4}),
                "total_words": 4,
                "bigram_counts": Counter({"gamma delta": 2}),
                "total_bigrams": 2,
            },
        }
        result = _compute_child_tfidf(child_data, n_children=2, top_k=5, bigram_check=True)
        assert "bigrams" in result[0]
        bigrams_0 = [bg for bg, _ in result[0]["bigrams"]]
        assert "alpha beta" in bigrams_0
