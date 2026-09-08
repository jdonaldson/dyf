"""SearchResult: where it lives, and the contract every index's result honours."""

from __future__ import annotations

import numpy as np


def test_canonical_import_is_the_package():
    import dyf
    from dyf.search_result import SearchResult

    assert dyf.SearchResult is SearchResult


def test_old_lazy_index_path_still_resolves_to_the_same_object():
    """lazy_index constructs SearchResult, so the pre-0.15 import keeps working as a
    side effect. Pinned so a future cleanup that breaks it does so knowingly."""
    import dyf
    from dyf.lazy_index import SearchResult

    assert SearchResult is dyf.SearchResult


def test_every_constructor_uses_the_one_type():
    import dyf
    from dyf import dense_search, lazy_index, rag

    assert dense_search.SearchResult is dyf.SearchResult
    assert lazy_index.SearchResult is dyf.SearchResult
    assert rag.SearchResult is dyf.SearchResult


def test_unpacking_len_and_getitem_contract():
    from dyf import SearchResult

    r = SearchResult(np.arange(7, dtype=np.uint32), np.linspace(1, 0, 7, dtype=np.float32), {"title": list("abcdefg")})
    indices, scores = r  # 2-tuple unpacking goes through __iter__
    assert indices is r.indices and scores is r.scores
    assert len(r) == 7  # hit count, not the unpacking arity
    assert r[0] is r.indices and r[1] is r.scores
    assert r["title"] == list("abcdefg")
