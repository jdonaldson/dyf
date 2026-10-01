"""Dense in-memory multiprobe search backed by the Rust kernel.

The required `dyf-rs` version is declared once in `pyproject.toml` and checked at import
by `dyf/__init__.py`. It is not restated here — this line used to say `>= 0.8.0` while the
requirement was `>= 0.10.0`, one of four disagreeing copies of the same number.

Builds a dyf tree over a dense embedding corpus and routes queries through it with a
batched, rayon-parallel Rust kernel (``dyf_rs.dense_search_batch``). The kernel is a
faithful port of the Python multiprobe (top-k identical) at ~100x lower per-query
latency — on 8.84M MSMARCO it reproduces dyf-mp recall at ~9ms/query vs ~58ms in pure
Python. This is the *dense in-memory* path; the on-disk LazyIndex path is unchanged.

Usage::

    idx = DenseSearchIndex(embeddings)                 # builds tree + flattens
    indices, scores = idx.search(query, k=10, nprobe=256)
    I, S = idx.search(query_batch, k=10, nprobe=256)   # batched (nq, k)
"""

from __future__ import annotations

import logging

import dyf_rs
import numpy as np

from .dyf_tree import build_dyf_tree
from .search_result import SearchResult

logger = logging.getLogger(__name__)

# Row-norm spread above which the constructor warns that cosine ranking discards magnitude.
# Spread is max/min norm - 1, so 0.01 means norms within 1% of each other: unit-norm
# embeddings with float32 rounding sit far below it; PCA coordinates sit far above.
NORM_SPREAD_WARN = 0.01


def _norm_spread(rows: np.ndarray) -> float:
    """``max/min`` row norm minus one — 0.0 for exactly equal norms, undefined (inf) if a
    row is all zeros."""
    norms = np.linalg.norm(rows, axis=1)
    lo = float(norms.min()) if norms.size else 0.0
    if lo <= 0.0:
        return float("inf")
    return float(norms.max()) / lo - 1.0


def flatten_tree(tree: dict) -> dict:
    """Flatten a ``build_dyf_tree`` dict into the CSR arrays the Rust kernel consumes.

    Pre-order node ids; CSR for children/buckets and leaf items. Does not mutate
    ``tree``. Node descent convention (validated against build_dyf_tree):
    ``bucket_id = sum_i (q . hp_i >= 0) << i`` (LSB-first), no centering.
    """
    nodes: list = []
    idmap: dict[int, int] = {}

    def walk(node):
        nid = len(nodes)
        nodes.append(node)
        idmap[id(node)] = nid
        for c in node.get("children") or []:
            walk(c)

    walk(tree)
    n = len(nodes)

    is_leaf = np.zeros(n, np.uint8)
    num_bits = np.zeros(n, np.int32)
    hp_off = np.zeros(n + 1, np.int64)
    child_off = np.zeros(n + 1, np.int64)
    leaf_off = np.zeros(n + 1, np.int64)
    hp_chunks, child_ids, child_bids, leaf_items = [], [], [], []

    for i, node in enumerate(nodes):
        children = node.get("children") or []
        if not children:
            is_leaf[i] = 1
            items = np.asarray(node["indices"], np.int64)
            leaf_items.append(items)
            leaf_off[i + 1] = leaf_off[i] + len(items)
            hp_off[i + 1] = hp_off[i]
            child_off[i + 1] = child_off[i]
        else:
            hp = np.asarray(node["hyperplanes"], np.float32)
            num_bits[i] = hp.shape[0]
            hp_chunks.append(hp.reshape(-1))
            hp_off[i + 1] = hp_off[i] + hp.size
            b2c = node["bucket_id_to_child"]
            for bid, ci in b2c.items():
                child_ids.append(idmap[id(children[ci])])
                child_bids.append(int(bid))
            child_off[i + 1] = child_off[i] + len(b2c)
            leaf_off[i + 1] = leaf_off[i]

    return dict(
        is_leaf=is_leaf,
        num_bits=num_bits,
        hp_off=hp_off,
        hp_data=(np.concatenate(hp_chunks) if hp_chunks else np.zeros(0, np.float32)),
        child_off=child_off,
        child_ids=np.asarray(child_ids, np.int64),
        child_bids=np.asarray(child_bids, np.int64),
        leaf_off=leaf_off,
        leaf_items=(np.concatenate(leaf_items) if leaf_items else np.zeros(0, np.int64)),
    )


class DenseSearchIndex:
    """Dense multiprobe search over an in-memory embedding corpus, **ranked by cosine**.

    The kernel L2-normalises the query and divides every candidate's dot product by that
    row's norm, so results are ordered by cosine similarity whatever the input norms are;
    rows do not have to be unit-norm. There is no dot-product or Euclidean mode. Routing
    through the tree is by hyperplane sign, which is scale-invariant, so it agrees with the
    ranking.

    That matters when the magnitude of a row carries information. A caller holding
    Euclidean data (PCA coordinates, the standard scRNA-seq input) who expected dot-product
    ranking — e.g. via the exact-Euclidean augmentation ``x -> (x, -|x|^2/2)`` — measured
    recall@15 = 0.06 against true neighbours here, with every slot filled and no error
    (``KNOWN_ISSUES.md`` #8). So when the row norms vary, the constructor logs a warning
    once; if your data is meant to be compared by angle, normalise it yourself and the
    warning goes away.

    Parameters
    ----------
    embeddings : (n, dim) array
        Row-vector corpus; kept resident as contiguous float32. Compared by cosine, see
        above.
    tree : dict, optional
        Prebuilt ``build_dyf_tree`` dict. If omitted, one is built from ``embeddings``.
    max_depth, num_bits, min_leaf_size : tree build params (used only when tree is None).
        Larger leaves (min_leaf_size ~128) are more latency-efficient per candidate.
    """

    def __init__(
        self, embeddings, *, tree: dict | None = None, max_depth: int = 16, num_bits: int = 3, min_leaf_size: int = 128
    ):
        self.embeddings = np.ascontiguousarray(embeddings, dtype=np.float32)
        self.norm_spread = _norm_spread(self.embeddings)
        if self.norm_spread > NORM_SPREAD_WARN:
            logger.warning(
                "DenseSearchIndex ranks by cosine, but these row norms vary "
                "(spread %.3f: max/min norm ratio - 1). Magnitude is discarded, so dot-product "
                "or Euclidean neighbours will differ from what search() returns. If angle is "
                "the intended comparison, L2-normalise the rows to silence this.",
                self.norm_spread,
            )
        self.tree = (
            tree
            if tree is not None
            else build_dyf_tree(self.embeddings, max_depth=max_depth, num_bits=num_bits, min_leaf_size=min_leaf_size)
        )
        self._flat = flatten_tree(self.tree)

    def search(self, queries, k: int = 10, nprobe: int = 256) -> SearchResult:
        """Return a :class:`SearchResult` with the top-``k`` per query.

        ``queries`` may be 1D ``(dim,)`` -> ``(k,)`` arrays, or 2D ``(nq, dim)`` ->
        ``(nq, k)`` arrays. ``nprobe`` leaves are probed (higher = more recall, more
        latency). Missing slots are padded with index -1 / score -inf.

        Returns the same type as :meth:`LazyIndex.search`, so callers can treat the two
        indexes interchangeably. This previously returned a bare ``(indices, scores)``
        tuple, which a caller could not introspect or extend — you had to already know
        the arity. ``SearchResult`` unpacks as a 2-tuple, so
        ``indices, scores = idx.search(...)`` keeps working unchanged.

        ``fields`` is always empty here: a ``DenseSearchIndex`` holds raw embeddings and
        has no stored fields to gather.
        """
        q = np.ascontiguousarray(queries, dtype=np.float32)
        single = q.ndim == 1
        if single:
            q = q[None, :]
        f = self._flat
        idx, sc = dyf_rs.dense_search_batch(
            f["is_leaf"],
            f["num_bits"],
            f["hp_off"],
            f["hp_data"],
            f["child_off"],
            f["child_ids"],
            f["child_bids"],
            f["leaf_off"],
            f["leaf_items"],
            self.embeddings,
            q,
            int(k),
            int(nprobe),
        )
        idx, sc = np.asarray(idx), np.asarray(sc)
        if single:
            idx, sc = idx[0], sc[0]
        return SearchResult(indices=idx, scores=sc)
