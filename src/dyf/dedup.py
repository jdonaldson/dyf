"""Near-duplicate detection for ingest-time deduplication.

Motivation, measured on a 229,243-section SEC 10-Q corpus
(``benchmarks/sequence_arc/sec_dedup_ingest.py``, ``sec_dedup_metric.py``):

- **29.4% of that corpus is near-duplicate content** at cosine > 0.99. Two independent
  methods agree: LSH clustering finds 32.0% transitively / 29.4% with star clustering, and
  a tree-blocked within-leaf scan finds 29.1%.
- Weighed `.dyf` files: **418.5 MB → 311.3 MB, 25.6% smaller (107 MB saved)**, with the
  ``dup_members`` field adding only 0.41 MB. Note the file saving (25.6%) is *less* than the
  point saving (29.4%): every tree node carries a dim-length centroid and leaf count only
  fell ~10%, so header overhead does not shrink proportionally. The dedup pass itself is
  ~2.5s on 229k points.
- Retrieval **improves** once measured against the right target. Raw ``recall@10`` versus
  brute-force kNN puts a mean of **2.97 duplicate slots in every true top-10** (83% of
  queries affected), so it pays for returning redundant copies. Scored on distinct content
  instead, dedup wins at every budget tested but the largest, where it reaches parity.

Only numpy is used, so this stays importable on the core dependency set.

Cosine similarity is the comparison; embeddings are unit-normalised internally, so callers
may pass raw vectors. Zero-norm rows are treated as singletons rather than matching
everything.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)

__all__ = ["DedupResult", "decode_members", "near_duplicate_clusters"]


@dataclass
class DedupResult:
    """Near-duplicate clustering of a point set.

    Attributes:
        labels: (n,) int64. ``labels[i]`` is the index of the representative point that
            stands for point ``i``. A singleton's label is itself.
        representatives: (n_clusters,) int64, sorted. The points to actually index.
        threshold: Cosine similarity above which two points were called duplicates.
        n_points: Size of the original point set.
    """

    labels: np.ndarray
    representatives: np.ndarray
    threshold: float
    n_points: int

    @property
    def n_removed(self) -> int:
        """How many points dedup eliminates."""
        return self.n_points - len(self.representatives)

    @property
    def removed_fraction(self) -> float:
        return self.n_removed / self.n_points if self.n_points else 0.0

    def mask(self) -> np.ndarray:
        """Boolean (n,) array, True for representatives — index ``embeddings[result.mask()]``."""
        m = np.zeros(self.n_points, dtype=bool)
        m[self.representatives] = True
        return m

    def members(self) -> dict[int, np.ndarray]:
        """Map representative index -> all point indices it stands for (including itself)."""
        order = np.argsort(self.labels, kind="stable")
        lab = self.labels[order]
        starts = np.flatnonzero(np.r_[True, lab[1:] != lab[:-1]])
        ends = np.r_[starts[1:], len(lab)]
        return {int(lab[s]): order[s:e] for s, e in zip(starts, ends)}

    def cluster_sizes(self) -> np.ndarray:
        """Size of each cluster, aligned with ``representatives``."""
        _, counts = np.unique(self.labels, return_counts=True)
        return counts

    def member_field(self) -> np.ndarray:
        """Extra member indices per representative, as a `stored_fields`-ready utf8 array.

        Aligned with ``representatives``, so it pairs with ``embeddings[result.mask()]``.
        Each entry is a comma-separated list of the OTHER point indices the representative
        stands for; singletons get an empty string. The representative's own index is
        omitted because `write_lazy_index` already records it as ``item_index``.

        `.dyf` stored fields support utf8 but not list types, so a packed string is used.
        That keeps this a pure convention on top of the existing format -- no schema change,
        no regenerated flatbuffers, and no change to the rust reader.
        """
        mem = self.members()
        out = []
        for r in self.representatives:
            grp = mem.get(int(r), np.array([r]))
            extras = grp[grp != r]
            out.append(",".join(str(int(x)) for x in extras))
        return np.array(out, dtype=object)


def decode_members(value: str) -> np.ndarray:
    """Inverse of the per-row encoding in :meth:`DedupResult.member_field`."""
    if not value:
        return np.zeros(0, dtype=np.int64)
    return np.fromstring(value, dtype=np.int64, sep=",")


def near_duplicate_clusters(
    embeddings,
    threshold: float = 0.99,
    n_tables: int = 4,
    n_bits: int = 12,
    seed: int = 42,
    max_bucket: int = 4000,
) -> DedupResult:
    """Cluster near-duplicate points using multi-table random-projection LSH.

    STAR clustering, not transitive: a point joins a representative only if it is itself
    within ``threshold`` of *that representative*. Transitive union-find was measured on the
    SEC corpus to chain a 541-member "duplicate" cluster -- A~B~C~...~Z where A and Z are
    not similar at all -- and cost 5pp of recall where star clustering gained 2.6pp. Star
    guarantees every member is within ``threshold`` of the vector that represents it.

    Blocking: points sharing a sign-bit code over ``n_bits`` random hyperplanes are compared
    exactly. One table misses pairs -- two vectors at cosine 0.99 agree on a single random
    bit with probability ``1 - arccos(0.99)/pi = 0.955``, hence ``0.955^12 = 0.58`` over 12
    bits -- so several independent tables are used, giving ``1 - 0.42^4 = 0.97`` expected
    pair recall at the defaults.

    Args:
        embeddings: (n, dim) array-like. Normalised internally; raw vectors are fine.
        threshold: Cosine similarity above which points are duplicates. 0.99 is
            near-identical text; loosen with care, since members inherit a representative.
        n_tables: Independent LSH tables. More tables find more pairs, linear cost.
        n_bits: Hyperplanes per table. More bits means smaller buckets: faster, less recall.
        seed: Hyperplane seed. Fixed by default so ingest is reproducible.
        max_bucket: Buckets larger than this are skipped, bounding the quadratic
            within-bucket comparison. Oversized buckets mean ``n_bits`` is too small.

    Returns:
        :class:`DedupResult`.

    Raises:
        ValueError: If ``embeddings`` is not 2-D or ``threshold`` is outside (-1, 1].
    """
    E = np.asarray(embeddings, dtype=np.float32)
    if E.ndim != 2:
        raise ValueError(f"embeddings must be 2-D, got shape {E.shape}")
    if not -1.0 < threshold <= 1.0:
        raise ValueError(f"threshold must be in (-1, 1], got {threshold}")
    n, dim = E.shape
    if n == 0:
        return DedupResult(np.zeros(0, np.int64), np.zeros(0, np.int64), threshold, 0)

    norms = np.linalg.norm(E, axis=1)
    good = norms > 1e-12
    En = np.zeros_like(E)
    En[good] = E[good] / norms[good, None]

    rng = np.random.default_rng(seed)
    assigned = np.full(n, -1, dtype=np.int64)
    # zero-norm rows have no meaningful direction; keep them as their own singletons
    for t in range(n_tables):
        H = rng.standard_normal((n_bits, dim)).astype(np.float32)
        H /= np.linalg.norm(H, axis=1, keepdims=True)
        codes = ((En @ H.T > 0).astype(np.int64) << np.arange(n_bits)).sum(1)
        order = np.argsort(codes, kind="stable")
        cs = codes[order]
        starts = np.flatnonzero(np.r_[True, cs[1:] != cs[:-1]])
        ends = np.r_[starts[1:], len(cs)]
        for s, e in zip(starts, ends):
            idx = order[s:e]
            idx = idx[(assigned[idx] < 0) & good[idx]]
            if len(idx) < 2 or len(idx) > max_bucket:
                continue
            G = En[idx] @ En[idx].T
            np.fill_diagonal(G, -2.0)
            taken = np.zeros(len(idx), dtype=bool)
            for a in range(len(idx)):
                if taken[a]:
                    continue
                grp = np.flatnonzero((G[a] > threshold) & ~taken)
                if len(grp) == 0:
                    continue
                taken[a] = True
                taken[grp] = True
                assigned[idx[a]] = idx[a]
                assigned[idx[grp]] = idx[a]
        logger.debug("dedup table %d/%d: %d points clustered", t + 1, n_tables, int((assigned >= 0).sum()))

    singles = assigned < 0
    assigned[singles] = np.flatnonzero(singles)
    reps = np.unique(assigned)
    logger.info(
        "dedup: %d representatives from %d points (%.1f%% removed) at cosine > %.3f",
        len(reps),
        n,
        100.0 * (n - len(reps)) / n,
        threshold,
    )
    return DedupResult(assigned, reps, threshold, n)
