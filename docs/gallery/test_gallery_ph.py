"""Tests for the Layer 5 → Layer 7 dependency primitives in _gallery.py.

Run with: pytest docs/gallery/test_gallery_ph.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

# Make _gallery.py importable when running from repo root
sys.path.insert(0, str(Path(__file__).parent))
from _gallery import (  # noqa: E402
    DEFAULT_STRESS_MODULES,
    cycle_lineage_spans,
    qc_filter_mask,
    score_stress_modules,
    wcs_persistence_null,
)


# ---------------------------------------------------------------------------
# DEFAULT_STRESS_MODULES — sanity check the registry
# ---------------------------------------------------------------------------

class TestDefaultStressModules:
    def test_has_expected_modules(self):
        assert set(DEFAULT_STRESS_MODULES.keys()) >= {"apoptosis", "upr", "hsp"}

    def test_genes_are_strings(self):
        for name, genes in DEFAULT_STRESS_MODULES.items():
            assert isinstance(genes, list)
            assert all(isinstance(g, str) for g in genes), name
            assert len(genes) >= 3, f"{name} has too few genes"


# ---------------------------------------------------------------------------
# cycle_lineage_spans — the most important primitive to lock down
# ---------------------------------------------------------------------------

def _make_cocycle(vertex_ids: list[int]) -> np.ndarray:
    """Build a synthetic cocycle that visits the given vertex ids."""
    rows = []
    for i in range(len(vertex_ids)):
        rows.append([vertex_ids[i], vertex_ids[(i + 1) % len(vertex_ids)], 1])
    return np.array(rows, dtype=np.int64)


class TestCycleLineageSpans:
    def test_drops_single_type_cycles(self):
        """A cycle whose vertices are all type A should be filtered out."""
        cocycles = [_make_cocycle([0, 1, 2])]
        persistences = np.array([3.0])
        labels = np.array(["A", "A", "A", "B", "C"])
        result = cycle_lineage_spans(cocycles, persistences, labels,
                                       min_distinct_types=2)
        assert result == []

    def test_keeps_two_type_cycle(self):
        """A cycle visiting two types should pass with min_distinct_types=2."""
        cocycles = [_make_cocycle([0, 3, 1])]
        persistences = np.array([3.0])
        labels = np.array(["A", "A", "A", "B", "C"])
        result = cycle_lineage_spans(cocycles, persistences, labels,
                                       min_distinct_types=2)
        assert len(result) == 1
        assert result[0]["n_distinct_types"] == 2
        assert result[0]["label_counts"] == {"A": 2, "B": 1}

    def test_keeps_three_type_cycle(self):
        cocycles = [_make_cocycle([0, 3, 4])]
        persistences = np.array([2.5])
        labels = np.array(["A", "A", "A", "B", "C"])
        result = cycle_lineage_spans(cocycles, persistences, labels,
                                       min_distinct_types=2)
        assert len(result) == 1
        assert result[0]["n_distinct_types"] == 3

    def test_persistence_threshold_filters_low(self):
        """Cycles below threshold_frac × max_persistence are dropped."""
        # Two cycles: persistence 5.0 and 0.5. With threshold_frac=0.20,
        # cutoff = 1.0, so cycle 2 should be dropped.
        cocycles = [_make_cocycle([0, 3, 4]), _make_cocycle([0, 3, 4])]
        persistences = np.array([5.0, 0.5])
        labels = np.array(["A", "A", "A", "B", "C"])
        result = cycle_lineage_spans(cocycles, persistences, labels,
                                       threshold_frac=0.20)
        assert len(result) == 1
        assert result[0]["persistence"] == 5.0

    def test_rank_reflects_persistence_order(self):
        """Surviving cycles ranked by persistence descending; rank refers
        to original ordering among robust cycles."""
        cocycles = [
            _make_cocycle([0, 3]),  # rank 1 by raw persistence
            _make_cocycle([0, 4]),  # rank 2 by raw persistence
        ]
        persistences = np.array([3.0, 2.0])
        labels = np.array(["A", "A", "A", "B", "C"])
        result = cycle_lineage_spans(cocycles, persistences, labels,
                                       min_distinct_types=2)
        assert len(result) == 2
        # First in result should be the higher-persistence cycle
        assert result[0]["persistence"] > result[1]["persistence"]
        # rank field is 1-indexed and reflects sort order
        assert result[0]["rank"] == 1
        assert result[1]["rank"] == 2

    def test_empty_persistences_returns_empty(self):
        result = cycle_lineage_spans([], np.array([]), np.array(["A", "B"]))
        assert result == []

    def test_empty_cocycle_skipped(self):
        """A cocycle with shape (0, ?) should be skipped, not crash."""
        cocycles = [np.zeros((0, 3), dtype=np.int64), _make_cocycle([0, 3])]
        persistences = np.array([3.0, 2.5])
        labels = np.array(["A", "A", "A", "B", "C"])
        result = cycle_lineage_spans(cocycles, persistences, labels,
                                       min_distinct_types=2)
        # First cocycle skipped (empty), second survives
        assert len(result) == 1

    def test_min_distinct_types_one_keeps_everything_above_threshold(self):
        """With min_distinct_types=1, even single-type cycles pass."""
        cocycles = [_make_cocycle([0, 1, 2])]
        persistences = np.array([3.0])
        labels = np.array(["A", "A", "A"])
        result = cycle_lineage_spans(cocycles, persistences, labels,
                                       min_distinct_types=1)
        assert len(result) == 1


class TestWcsZFiltering:
    """The WCS-z calibration replaces the dataset-relative 20% threshold
    with a null-anchored significance gate."""

    @pytest.fixture
    def cycles_with_persistences(self):
        """3 cycles spanning ≥2 types, with widely-separated persistences."""
        cocycles = [
            _make_cocycle([0, 3, 4]),  # high persistence
            _make_cocycle([0, 3]),     # mid persistence
            _make_cocycle([0, 3]),     # low persistence
        ]
        persistences = np.array([5.0, 2.5, 1.0])
        labels = np.array(["A", "A", "A", "B", "C"])
        return cocycles, persistences, labels

    def test_wcs_z_filter_keeps_only_significant(self, cycles_with_persistences):
        """With wcs_pool centered on ~1.0±0.5, only the persistence-5 cycle
        clears z=3."""
        cocycles, persistences, labels = cycles_with_persistences
        # WCS pool: mean 1.0, std ≈ 0.5 → z thresholds: pers 5 → ~8σ,
        # pers 2.5 → ~3σ, pers 1.0 → ~0σ.
        wcs_pool = np.array([0.5, 0.7, 1.0, 1.2, 1.5, 0.8, 1.1, 1.3])
        result = cycle_lineage_spans(
            cocycles, persistences, labels,
            min_distinct_types=2,
            wcs_persistences=wcs_pool, min_z_vs_wcs=3.0,
        )
        # Persistence 5 (z ≈ 11) keeps; persistence 2.5 (z ≈ 4.4) keeps;
        # persistence 1.0 (z ≈ 0) drops.
        assert len(result) == 2
        kept_pers = sorted([r["persistence"] for r in result], reverse=True)
        assert kept_pers == [5.0, 2.5]

    def test_wcs_z_returns_z_and_p(self, cycles_with_persistences):
        cocycles, persistences, labels = cycles_with_persistences
        wcs_pool = np.array([0.5, 0.7, 1.0, 1.2, 1.5, 0.8, 1.1, 1.3])
        result = cycle_lineage_spans(
            cocycles, persistences, labels,
            min_distinct_types=2,
            wcs_persistences=wcs_pool, min_z_vs_wcs=3.0,
        )
        for r in result:
            assert "z_vs_wcs" in r
            assert "p_vs_wcs" in r
            assert r["z_vs_wcs"] >= 3.0
            # p-value: empirical fraction of WCS ≥ this persistence.
            # Both persistences (5 and 2.5) exceed every WCS pool value (max 1.5).
            assert r["p_vs_wcs"] == 0.0

    def test_wcs_z_strict_threshold_excludes_more(self, cycles_with_persistences):
        """At z=10, only the persistence-5 cycle survives."""
        cocycles, persistences, labels = cycles_with_persistences
        wcs_pool = np.array([0.5, 0.7, 1.0, 1.2, 1.5, 0.8, 1.1, 1.3])
        result = cycle_lineage_spans(
            cocycles, persistences, labels,
            min_distinct_types=2,
            wcs_persistences=wcs_pool, min_z_vs_wcs=10.0,
        )
        assert len(result) == 1
        assert result[0]["persistence"] == 5.0

    def test_wcs_z_supersedes_threshold_frac(self, cycles_with_persistences):
        """When WCS is provided, threshold_frac is ignored."""
        cocycles, persistences, labels = cycles_with_persistences
        wcs_pool = np.array([0.5, 0.7, 1.0, 1.2, 1.5, 0.8, 1.1, 1.3])
        # threshold_frac=0.99 would normally drop everything below 4.95;
        # but WCS-z mode supersedes it, so 2.5 still passes z>3.
        result = cycle_lineage_spans(
            cocycles, persistences, labels,
            min_distinct_types=2,
            threshold_frac=0.99,
            wcs_persistences=wcs_pool, min_z_vs_wcs=3.0,
        )
        kept_pers = sorted([r["persistence"] for r in result], reverse=True)
        assert kept_pers == [5.0, 2.5]

    def test_partial_wcs_args_falls_back_to_threshold_frac(
            self, cycles_with_persistences):
        """If only one of (wcs_persistences, min_z_vs_wcs) is provided,
        the function should fall back to the relative threshold."""
        cocycles, persistences, labels = cycles_with_persistences
        wcs_pool = np.array([0.5, 0.7, 1.0])
        # Only wcs_persistences, no min_z_vs_wcs: should use threshold_frac.
        result = cycle_lineage_spans(
            cocycles, persistences, labels,
            min_distinct_types=2,
            wcs_persistences=wcs_pool,  # min_z_vs_wcs not set
            threshold_frac=0.20,
        )
        # threshold = 0.20 × 5 = 1.0; persistence 5 and 2.5 pass, 1.0 doesn't.
        assert len(result) == 2
        # And no z_vs_wcs / p_vs_wcs in the output (since WCS gate inactive).
        for r in result:
            assert "z_vs_wcs" not in r

    def test_no_cycles_pass_returns_empty(self, cycles_with_persistences):
        """An impossibly strict z threshold returns empty list."""
        cocycles, persistences, labels = cycles_with_persistences
        wcs_pool = np.array([0.5, 0.7, 1.0, 1.2, 1.5])
        result = cycle_lineage_spans(
            cocycles, persistences, labels,
            min_distinct_types=2,
            wcs_persistences=wcs_pool, min_z_vs_wcs=100.0,
        )
        assert result == []


class TestWcsPersistenceNull:
    """The wcs_persistence_null helper builds the null distribution for
    cycle_lineage_spans's WCS gate."""

    def test_returns_array_of_persistences(self):
        # Synthetic 2-cluster data: two well-separated blobs in 5D.
        rng = np.random.default_rng(0)
        cluster_a = rng.normal(0, 0.3, size=(50, 5))
        cluster_b = rng.normal(5, 0.3, size=(50, 5))
        pca = np.vstack([cluster_a, cluster_b]).astype(np.float32)
        labels = np.array([0] * 50 + [1] * 50)

        wcs_pool = wcs_persistence_null(pca, labels, n_landmarks=20,
                                          n_shuffles=2, seed=42)
        # Should return some persistences (>=0), maybe empty if no Betti-1
        assert isinstance(wcs_pool, np.ndarray)
        # Persistences are nonneg by construction (death ≥ birth)
        assert (wcs_pool >= 0).all() if len(wcs_pool) > 0 else True

    def test_empty_when_landmarks_too_few(self):
        """With n_landmarks > n_cells, the function clamps and runs."""
        rng = np.random.default_rng(0)
        pca = rng.normal(size=(20, 5)).astype(np.float32)
        labels = np.zeros(20, dtype=int)
        # n_landmarks=200 will be clamped to n-1=19
        result = wcs_persistence_null(pca, labels, n_landmarks=200,
                                        n_shuffles=1, seed=0)
        assert isinstance(result, np.ndarray)


# ---------------------------------------------------------------------------
# qc_filter_mask — covers the AnnData-using path with a minimal mock
# ---------------------------------------------------------------------------

class TestQcFilterMask:
    """We use a thin pandas-DataFrame stand-in for AnnData.obs since
    qc_filter_mask only reads .obs columns. Builds a fake AnnData with just
    obs populated."""

    @pytest.fixture
    def fake_adata(self):
        """5 cells with synthetic QC fields."""
        import pandas as pd

        class FakeAnnData:
            def __init__(self, obs):
                self.obs = obs
                self.n_obs = len(obs)

        obs = pd.DataFrame({
            "n_genes":         [100,  500,  1000, 2000, 3000],
            "pct_mt":          [5.0,  3.0,   1.0,  0.5,  0.1],
            "pct_hb":          [0.0,  0.0,   0.0,  0.0, 10.0],
            "apoptosis_score": [0.0,  0.5,   0.1,  0.2,  0.3],
            "upr_score":       [0.0,  0.0,   0.0,  0.0,  0.0],
            "hsp_score":       [0.0,  0.0,   0.0,  0.0,  0.0],
        })
        return FakeAnnData(obs)

    def test_n_genes_filter(self, fake_adata):
        # Cell 0 has 100 genes, fails n_genes_min=200
        mask = qc_filter_mask(fake_adata, drop_top_pct=0,
                                pct_mt_max=None, pct_hb_max=None,
                                n_genes_min=200)
        assert mask[0] == False  # noqa: E712
        assert mask[1:].all()

    def test_pct_hb_filter(self, fake_adata):
        # Cell 4 has pct_hb=10 > 5
        mask = qc_filter_mask(fake_adata, drop_top_pct=0,
                                pct_mt_max=None, pct_hb_max=5.0,
                                n_genes_min=0)
        assert mask[4] == False  # noqa: E712
        assert mask[:4].all()

    def test_pct_mt_skipped_when_none(self, fake_adata):
        """pct_mt_max=None should skip the check entirely."""
        # Even if pct_mt is high, set max=None — should pass
        fake_adata.obs.loc[0, "pct_mt"] = 100.0
        mask = qc_filter_mask(fake_adata, drop_top_pct=0,
                                pct_mt_max=None, pct_hb_max=None,
                                n_genes_min=0)
        assert mask.all()

    def test_drop_top_pct_drops_top_score(self, fake_adata):
        """drop_top_pct=20 with n=5 → drop top quintile per score column.
        Cell 1 has highest apoptosis (0.5), should be dropped."""
        mask = qc_filter_mask(fake_adata, drop_top_pct=20,
                                pct_mt_max=None, pct_hb_max=None,
                                n_genes_min=0,
                                score_columns=["apoptosis_score"])
        assert mask[1] == False  # noqa: E712


# ---------------------------------------------------------------------------
# score_stress_modules — needs scanpy, so skip if unavailable
# ---------------------------------------------------------------------------

class TestScoreStressModules:
    def test_returns_score_columns(self):
        pytest.importorskip("scanpy")
        pytest.importorskip("anndata")
        import anndata as ad
        import scanpy as sc

        # Build a minimal AnnData with apoptosis genes detectable
        rng = np.random.default_rng(0)
        n_cells = 200
        gene_names = ["Casp3", "Bax", "Bak1", "Hspa5", "Atf4",
                       "Hspa1a", "Hsp90aa1"] + [f"GENE{i}" for i in range(50)]
        # Need >=50 genes for sc.tl.score_genes random control set
        X = rng.uniform(0.0, 5.0, size=(n_cells, len(gene_names))).astype("float32")
        a = ad.AnnData(X=X, var=__import__("pandas").DataFrame(index=gene_names))

        added = score_stress_modules(a)
        # Expect at least apoptosis, upr, hsp added (matching what's present)
        assert "apoptosis_score" in added
        assert "apoptosis_score" in a.obs.columns
        assert len(a.obs["apoptosis_score"]) == n_cells
