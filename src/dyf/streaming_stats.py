"""Single-pass streaming column profiling.

Provides Welford's online algorithm for numeric stats and Space-Saving
for approximate top-K frequent items. No dependencies beyond numpy.

Usage::

    profile = TableProfile(["price", "category"], ["float64", "str"])
    for row in data:
        profile.update_row(row)
    print(profile.to_text())
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np


class WelfordStats:
    """Streaming numeric stats via Welford's online algorithm."""

    __slots__ = ("_n", "_mean", "_m2", "_min", "_max")

    def __init__(self) -> None:
        self._n: int = 0
        self._mean: float = 0.0
        self._m2: float = 0.0
        self._min: float = math.inf
        self._max: float = -math.inf

    def update(self, value: float) -> None:
        self._n += 1
        delta = value - self._mean
        self._mean += delta / self._n
        delta2 = value - self._mean
        self._m2 += delta * delta2
        if value < self._min:
            self._min = value
        if value > self._max:
            self._max = value

    def update_batch(self, values: np.ndarray) -> None:
        for v in values:
            self.update(float(v))

    @property
    def count(self) -> int:
        return self._n

    @property
    def mean(self) -> float:
        return self._mean if self._n > 0 else 0.0

    @property
    def variance(self) -> float:
        if self._n < 2:
            return 0.0
        return self._m2 / (self._n - 1)

    @property
    def std(self) -> float:
        return math.sqrt(self.variance)

    @property
    def min(self) -> float:
        return self._min if self._n > 0 else 0.0

    @property
    def max(self) -> float:
        return self._max if self._n > 0 else 0.0

    def histogram(self, n_bins: int = 10) -> list[tuple[float, int]]:
        """Return bin edges and counts. Must be called after all data is seen.

        Since we only track min/max online, the caller must re-scan the data
        or this returns empty bins (edges only). For a streaming histogram,
        use ``to_dict()`` which includes min/max for downstream binning.
        """
        if self._n == 0 or self._min == self._max:
            return [(self._min, self._n)]
        step = (self._max - self._min) / n_bins
        return [(self._min + i * step, 0) for i in range(n_bins)]

    def to_dict(self) -> dict[str, Any]:
        return {
            "count": self._n,
            "mean": self.mean,
            "std": self.std,
            "min": self.min,
            "max": self.max,
            "variance": self.variance,
        }


class SpaceSaving:
    """Streaming approximate top-K frequent items (Metwally et al.)."""

    __slots__ = ("_k", "_counts", "_total")

    def __init__(self, k: int = 20) -> None:
        self._k = k
        self._counts: dict[Any, int] = {}
        self._total: int = 0

    def update(self, item: Any) -> None:
        self._total += 1
        if item in self._counts:
            self._counts[item] += 1
        elif len(self._counts) < self._k:
            self._counts[item] = 1
        else:
            # Replace the minimum-count item
            min_item = min(self._counts, key=self._counts.__getitem__)
            min_count = self._counts.pop(min_item)
            self._counts[item] = min_count + 1

    def update_batch(self, items) -> None:
        for item in items:
            self.update(item)

    @property
    def total(self) -> int:
        return self._total

    def top(self, n: int = 10) -> list[tuple[Any, int]]:
        return sorted(self._counts.items(), key=lambda x: x[1], reverse=True)[:n]

    def to_dict(self) -> dict[str, Any]:
        return {
            "total": self._total,
            "top": self.top(),
            "tracked": len(self._counts),
        }


def _format_count(n: int) -> str:
    """Format large counts with k/M suffixes."""
    if n >= 1_000_000:
        return f"{n / 1_000_000:.1f}M"
    if n >= 1_000:
        return f"{n / 1_000:.0f}k"
    return str(n)


class ColumnProfile:
    """Profile a single column, auto-detecting numeric vs categorical."""

    __slots__ = ("name", "dtype", "_is_numeric", "_welford", "_freq", "_length_stats", "null_count")

    def __init__(self, name: str, dtype: str) -> None:
        self.name = name
        self.dtype = dtype
        self._is_numeric = dtype in (
            "float32", "float64", "float", "int", "int8", "int16", "int32", "int64",
            "uint8", "uint16", "uint32", "uint64",
        )
        self._welford = WelfordStats()
        self._freq = SpaceSaving(k=20)
        self._length_stats: WelfordStats | None = WelfordStats() if not self._is_numeric else None
        self.null_count: int = 0

    def update(self, value: Any) -> None:
        if value is None or (isinstance(value, float) and math.isnan(value)):
            self.null_count += 1
            return
        if self._is_numeric:
            v = float(value)
            self._welford.update(v)
            if "int" in self.dtype:
                self._freq.update(value)
        else:
            s = str(value)
            self._freq.update(s)
            if self._length_stats is not None:
                self._length_stats.update(len(s))

    def update_batch(self, values) -> None:
        for v in values:
            self.update(v)

    def to_text(self) -> str:
        total = self._welford.count + self.null_count if self._is_numeric else self._freq.total + self.null_count
        null_pct = (self.null_count / total * 100) if total > 0 else 0.0
        lines = [f"column: {self.name} ({self.dtype})"]

        if self._is_numeric:
            w = self._welford
            lines.append(
                f"stats: n={_format_count(w.count)}, "
                f"mean={w.mean:.2f}, std={w.std:.1f}, "
                f"min={w.min}, max={w.max}"
            )
            if "int" in self.dtype and self._freq.total > 0:
                top_items = self._freq.top(5)
                top_str = ", ".join(f"{item} ({_format_count(ct)})" for item, ct in top_items)
                lines.append(f"top values: {top_str}")
        else:
            top_items = self._freq.top(5)
            if top_items:
                top_str = ", ".join(f'"{item}" ({_format_count(ct)})' for item, ct in top_items)
                lines.append(f"top values: {top_str}")
            lines.append(f"unique (approx): {self._freq.to_dict()['tracked']}")
            if self._length_stats and self._length_stats.count > 0:
                lines.append(f"avg length: {self._length_stats.mean:.1f}")

        lines.append(f"nulls: {null_pct:.1f}%")
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"name": self.name, "dtype": self.dtype, "null_count": self.null_count}
        if self._is_numeric:
            d["stats"] = self._welford.to_dict()
        else:
            d["freq"] = self._freq.to_dict()
            if self._length_stats:
                d["length_stats"] = self._length_stats.to_dict()
        return d


class TableProfile:
    """Profile all columns in a table."""

    def __init__(self, column_names: list[str], dtypes: list[str]) -> None:
        self.columns: dict[str, ColumnProfile] = {
            name: ColumnProfile(name, dtype)
            for name, dtype in zip(column_names, dtypes)
        }
        self._col_order = list(column_names)

    def update_row(self, row: dict[str, Any] | list | tuple) -> None:
        if isinstance(row, dict):
            for name, col in self.columns.items():
                col.update(row.get(name))
        else:
            for i, name in enumerate(self._col_order):
                self.columns[name].update(row[i] if i < len(row) else None)

    def update_batch(self, rows_dict: dict[str, list]) -> None:
        for name, values in rows_dict.items():
            if name in self.columns:
                self.columns[name].update_batch(values)

    def to_text(self) -> str:
        parts = []
        for name in self._col_order:
            parts.append(self.columns[name].to_text())
        return "\n\n".join(parts)

    def to_dict(self) -> dict[str, Any]:
        return {name: self.columns[name].to_dict() for name in self._col_order}
