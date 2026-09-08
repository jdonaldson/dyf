"""The one return type every retrieval entry point shares.

`LazyIndex.search`, `LazyIndex.search_ivf`, `DenseSearchIndex.search` and
`BridgeIndex.query` all return a `SearchResult`, so calling code is interchangeable
across index types. It lived in `lazy_index.py` until 0.15.0, which made the in-memory
and bridge indexes import their return type from the on-disk one — a dependency in the
wrong direction, and a misleading home for a type that has nothing to do with files.
It depends on numpy alone and is importable even when the FlatBuffers/Arrow stack is not.

Import it from the package: ``from dyf import SearchResult``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class SearchResult:
    """Search result with indices, scores, and optional stored fields.

    The standard return type of every retrieval entry point — `LazyIndex.search`,
    `LazyIndex.search_ivf`, `DenseSearchIndex.search` and `BridgeIndex.query` — so the
    same calling code works against any index.
    """

    indices: np.ndarray  # (k,) uint32
    scores: np.ndarray  # (k,) float32
    fields: dict = field(default_factory=dict)  # field_name -> (k,) values
    routing: dict | None = None  # routing diagnostics when return_routing=True

    def __iter__(self):
        """Backward-compatible unpacking: indices, scores = idx.search(...)"""
        yield self.indices
        yield self.scores

    def __getitem__(self, key):
        """`r["title"]` gets a stored field; `r[0]`/`r[1]` are indices/scores.

        The positional form exists only so tuple-style access keeps working. Prefer
        `.indices` / `.scores` — they say which one you meant.
        """
        if isinstance(key, str):
            return self.fields[key]
        return (self.indices, self.scores)[key]

    def __len__(self):
        """Number of hits — NOT the unpacking arity.

        This returned a hard-coded 2 until 2026-09-05, so `len(result)` reported 2 on a
        `k=10` search: a plausible-looking wrong number, of exactly the kind that gets
        cited downstream without being questioned. Safe to change because tuple unpacking
        goes through `__iter__`, never `__len__` — verified — so
        `indices, scores = idx.search(...)` is unaffected.
        """
        return len(self.indices)
