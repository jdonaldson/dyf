"""Tests for streaming_stats module."""

import numpy as np
import pytest

from dyf.streaming_stats import (
    ColumnProfile,
    SpaceSaving,
    TableProfile,
    WelfordStats,
)

# ---------------------------------------------------------------------------
# WelfordStats
# ---------------------------------------------------------------------------


class TestWelfordStats:
    def test_known_sequence(self):
        w = WelfordStats()
        values = [2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0]
        for v in values:
            w.update(v)

        assert w.count == 8
        assert w.mean == pytest.approx(np.mean(values))
        assert w.variance == pytest.approx(np.var(values, ddof=1))
        assert w.std == pytest.approx(np.std(values, ddof=1))
        assert w.min == 2.0
        assert w.max == 9.0

    def test_batch_matches_single(self):
        values = np.array([1.5, 3.2, 7.8, 0.1, 5.5])
        w_single = WelfordStats()
        for v in values:
            w_single.update(v)

        w_batch = WelfordStats()
        w_batch.update_batch(values)

        assert w_batch.count == w_single.count
        assert w_batch.mean == pytest.approx(w_single.mean)
        assert w_batch.variance == pytest.approx(w_single.variance)
        assert w_batch.min == w_single.min
        assert w_batch.max == w_single.max

    def test_empty(self):
        w = WelfordStats()
        assert w.count == 0
        assert w.mean == 0.0
        assert w.variance == 0.0
        assert w.std == 0.0
        assert w.min == 0.0
        assert w.max == 0.0

    def test_single_value(self):
        w = WelfordStats()
        w.update(42.0)
        assert w.count == 1
        assert w.mean == 42.0
        assert w.variance == 0.0
        assert w.min == 42.0
        assert w.max == 42.0

    def test_all_same_value(self):
        w = WelfordStats()
        for _ in range(100):
            w.update(3.14)
        assert w.mean == pytest.approx(3.14)
        assert w.variance == pytest.approx(0.0, abs=1e-10)
        assert w.min == 3.14
        assert w.max == 3.14

    def test_to_dict(self):
        w = WelfordStats()
        w.update(1.0)
        w.update(3.0)
        d = w.to_dict()
        assert d["count"] == 2
        assert d["mean"] == pytest.approx(2.0)
        assert "std" in d
        assert "min" in d
        assert "max" in d

    def test_histogram_empty(self):
        w = WelfordStats()
        h = w.histogram()
        assert len(h) == 1  # single bucket with count=0

    def test_histogram_same_values(self):
        w = WelfordStats()
        for _ in range(5):
            w.update(10.0)
        h = w.histogram()
        assert len(h) == 1
        assert h[0] == (10.0, 5)

    def test_histogram_bins(self):
        w = WelfordStats()
        for v in [0.0, 50.0, 100.0]:
            w.update(v)
        h = w.histogram(n_bins=5)
        assert len(h) == 5
        assert h[0][0] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# SpaceSaving
# ---------------------------------------------------------------------------


class TestSpaceSaving:
    def test_exact_when_within_k(self):
        ss = SpaceSaving(k=10)
        items = ["a"] * 5 + ["b"] * 3 + ["c"] * 1
        ss.update_batch(items)

        top = ss.top(3)
        assert top[0] == ("a", 5)
        assert top[1] == ("b", 3)
        assert top[2] == ("c", 1)
        assert ss.total == 9

    def test_eviction_when_over_k(self):
        ss = SpaceSaving(k=3)
        # Fill 3 slots
        for item in ["a", "b", "c"]:
            ss.update(item)
        # Now "d" should evict the min (all have count=1, one gets evicted)
        ss.update("d")

        assert ss.total == 4
        top = ss.top()
        assert len(top) == 3  # still only k items tracked

    def test_heavy_hitter_survives(self):
        """A truly frequent item should survive evictions."""
        ss = SpaceSaving(k=5)
        # "heavy" appears 50 times, rest appear once
        for _ in range(50):
            ss.update("heavy")
        for i in range(100):
            ss.update(f"rare_{i}")

        top = ss.top(1)
        assert top[0][0] == "heavy"
        assert top[0][1] >= 50  # count may be inflated but item survives

    def test_empty(self):
        ss = SpaceSaving()
        assert ss.total == 0
        assert ss.top() == []

    def test_to_dict(self):
        ss = SpaceSaving(k=5)
        ss.update_batch(["x", "y", "x"])
        d = ss.to_dict()
        assert d["total"] == 3
        assert d["tracked"] == 2
        assert len(d["top"]) == 2


# ---------------------------------------------------------------------------
# ColumnProfile
# ---------------------------------------------------------------------------


class TestColumnProfile:
    def test_numeric_auto_detection(self):
        cp = ColumnProfile("price", "float64")
        cp.update(10.5)
        cp.update(20.0)
        cp.update(None)
        text = cp.to_text()
        assert "price" in text
        assert "float64" in text
        assert "stats:" in text
        assert "nulls:" in text

    def test_string_auto_detection(self):
        cp = ColumnProfile("category", "str")
        cp.update("Electronics")
        cp.update("Clothing")
        cp.update("Electronics")
        text = cp.to_text()
        assert "category" in text
        assert "top values:" in text
        assert "unique (approx):" in text
        assert "avg length:" in text

    def test_null_counting(self):
        cp = ColumnProfile("val", "float64")
        cp.update(1.0)
        cp.update(None)
        cp.update(float("nan"))
        cp.update(2.0)
        assert cp.null_count == 2

    def test_int_includes_freq(self):
        cp = ColumnProfile("count", "int64")
        for v in [1, 2, 2, 3, 3, 3]:
            cp.update(v)
        text = cp.to_text()
        assert "top values:" in text

    def test_all_nulls(self):
        cp = ColumnProfile("empty", "float64")
        for _ in range(5):
            cp.update(None)
        assert cp.null_count == 5
        text = cp.to_text()
        assert "nulls: 100.0%" in text

    def test_to_dict_numeric(self):
        cp = ColumnProfile("x", "float64")
        cp.update(1.0)
        d = cp.to_dict()
        assert d["name"] == "x"
        assert "stats" in d

    def test_to_dict_string(self):
        cp = ColumnProfile("s", "str")
        cp.update("hello")
        d = cp.to_dict()
        assert "freq" in d
        assert "length_stats" in d


# ---------------------------------------------------------------------------
# TableProfile
# ---------------------------------------------------------------------------


class TestTableProfile:
    def test_dict_rows(self):
        tp = TableProfile(["price", "name"], ["float64", "str"])
        tp.update_row({"price": 9.99, "name": "Widget"})
        tp.update_row({"price": 19.99, "name": "Gadget"})
        text = tp.to_text()
        assert "price" in text
        assert "name" in text

    def test_list_rows(self):
        tp = TableProfile(["a", "b"], ["int64", "str"])
        tp.update_row([1, "x"])
        tp.update_row([2, "y"])
        assert tp.columns["a"]._welford.count == 2
        assert tp.columns["b"]._freq.total == 2

    def test_batch_update(self):
        tp = TableProfile(["val"], ["float64"])
        tp.update_batch({"val": [1.0, 2.0, 3.0]})
        assert tp.columns["val"]._welford.count == 3

    def test_multi_column(self):
        tp = TableProfile(
            ["id", "score", "label"],
            ["int64", "float64", "str"],
        )
        for i in range(100):
            tp.update_row({"id": i, "score": i * 0.5, "label": f"cat_{i % 5}"})

        d = tp.to_dict()
        assert len(d) == 3
        assert d["score"]["stats"]["count"] == 100
        assert d["label"]["freq"]["total"] == 100

    def test_missing_keys_become_null(self):
        tp = TableProfile(["a", "b"], ["float64", "str"])
        tp.update_row({"a": 1.0})  # "b" missing
        assert tp.columns["b"].null_count == 1

    def test_to_text_ordering(self):
        tp = TableProfile(["z_col", "a_col"], ["float64", "str"])
        tp.update_row({"z_col": 1.0, "a_col": "hi"})
        text = tp.to_text()
        # z_col should appear before a_col (insertion order, not alpha)
        assert text.index("z_col") < text.index("a_col")

    def test_to_dict(self):
        tp = TableProfile(["x"], ["float64"])
        tp.update_row({"x": 5.0})
        d = tp.to_dict()
        assert "x" in d
        assert d["x"]["stats"]["count"] == 1
