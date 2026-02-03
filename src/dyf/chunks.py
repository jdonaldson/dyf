"""Chunk analysis for documents with multiple embedding vectors.

When documents are chunked (one document produces multiple embeddings),
these functions use LSH bucket assignments to detect redundancy and
measure document coherence/spread.
"""

from collections import Counter, defaultdict
from dataclasses import dataclass, field

import numpy as np


@dataclass
class DocSpread:
    """Per-document chunk distribution across LSH buckets.

    Attributes:
        n_chunks: Total chunks for this document.
        n_buckets: Distinct buckets touched by chunks.
        concentration: n_chunks / n_buckets ratio.
            High = topically focused, low = multi-topic bridge.
        bucket_distribution: Mapping of bucket_id to chunk count.
    """

    n_chunks: int
    n_buckets: int
    concentration: float
    bucket_distribution: dict = field(repr=False)


def chunk_redundancy(bucket_ids, doc_ids) -> np.ndarray:
    """Count same-doc siblings sharing each point's bucket.

    For each point, returns how many *other* chunks from the same document
    landed in the same bucket. 0 means unique in its bucket for that document.

    Args:
        bucket_ids: Array-like of bucket assignments per point.
        doc_ids: Array-like of document identifiers per point (same length).

    Returns:
        Integer array of length len(bucket_ids). Each value is the number
        of same-doc siblings in the same bucket (excluding self).
    """
    bucket_ids = np.asarray(bucket_ids)
    doc_ids = np.asarray(doc_ids)
    n = len(bucket_ids)
    if len(doc_ids) != n:
        raise ValueError("bucket_ids and doc_ids must have the same length")

    # Count occurrences per (bucket, doc) pair
    pair_counts: dict[tuple, int] = Counter(zip(bucket_ids.tolist(), doc_ids.tolist()))

    # Map back: each point gets count - 1 (exclude self)
    result = np.empty(n, dtype=np.int64)
    for i in range(n):
        result[i] = pair_counts[(bucket_ids[i].item(), doc_ids[i].item())] - 1
    return result


def deduplicate_chunks(bucket_ids, doc_ids) -> np.ndarray:
    """Boolean mask keeping one representative per (bucket, doc) pair.

    Multi-topic documents retain multiple representatives (one per bucket
    they touch). Single-topic documents collapse to fewer points.

    Args:
        bucket_ids: Array-like of bucket assignments per point.
        doc_ids: Array-like of document identifiers per point (same length).

    Returns:
        Boolean array of length len(bucket_ids). True for representative points.
    """
    bucket_ids = np.asarray(bucket_ids)
    doc_ids = np.asarray(doc_ids)
    n = len(bucket_ids)
    if len(doc_ids) != n:
        raise ValueError("bucket_ids and doc_ids must have the same length")

    seen: set[tuple] = set()
    mask = np.zeros(n, dtype=bool)
    for i in range(n):
        key = (bucket_ids[i].item(), doc_ids[i].item())
        if key not in seen:
            seen.add(key)
            mask[i] = True
    return mask


def doc_spread(bucket_ids, doc_ids) -> dict:
    """Per-document chunk distribution analysis.

    Articles with low concentration (high spread) are content-level bridges:
    they span multiple semantic regions by their structure.

    Args:
        bucket_ids: Array-like of bucket assignments per point.
        doc_ids: Array-like of document identifiers per point (same length).

    Returns:
        Dict mapping doc_id to DocSpread.
    """
    bucket_ids = np.asarray(bucket_ids)
    doc_ids = np.asarray(doc_ids)
    if len(bucket_ids) != len(doc_ids):
        raise ValueError("bucket_ids and doc_ids must have the same length")

    # Group bucket assignments by doc
    doc_buckets: dict = defaultdict(list)
    for i in range(len(bucket_ids)):
        doc_buckets[doc_ids[i].item()].append(bucket_ids[i].item())

    result = {}
    for doc_id, buckets in doc_buckets.items():
        dist = Counter(buckets)
        n_chunks = len(buckets)
        n_buckets = len(dist)
        result[doc_id] = DocSpread(
            n_chunks=n_chunks,
            n_buckets=n_buckets,
            concentration=n_chunks / n_buckets,
            bucket_distribution=dict(dist),
        )
    return result


def neighbor_coherence(embeddings, knn_indices, sample_k=None) -> np.ndarray:
    """Measure how coherent each point's neighborhood is.

    For each point, computes mean pairwise cosine similarity among its
    k nearest neighbors.  Topical documents have coherent neighborhoods
    (neighbors are similar to each other).  Meta/structural documents
    — date pages, event lists, disambiguation pages — pull in diverse
    neighbors that have little in common, producing low coherence.

    This is a general embedding-space signal that does not depend on
    titles, metadata, or domain-specific heuristics.

    Args:
        embeddings: (n, d) array of embedding vectors.
        knn_indices: (n, k) array of neighbor indices per point
                     (as returned by sklearn NearestNeighbors, self excluded).
        sample_k: If set, use only the first sample_k neighbors for the
                  pairwise calculation (faster for large k).

    Returns:
        Float array of length n.  Each value is the mean pairwise cosine
        similarity among that point's neighbors.  Range roughly [0, 1];
        higher values indicate meta/structural content (echo chambers of
        structurally similar documents).
    """
    embeddings = np.asarray(embeddings, dtype=np.float32)
    knn_indices = np.asarray(knn_indices)
    n = len(embeddings)

    # Normalize for cosine similarity
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    emb_normed = embeddings / np.maximum(norms, 1e-10)

    if sample_k is not None:
        knn_indices = knn_indices[:, :sample_k]

    k = knn_indices.shape[1]
    coherence = np.zeros(n, dtype=np.float64)

    for i in range(n):
        nb = knn_indices[i]
        nb_embs = emb_normed[nb]  # (k, d)
        # Mean pairwise cosine similarity: mean of upper triangle of G
        gram = nb_embs @ nb_embs.T  # (k, k)
        # Sum upper triangle, divide by k*(k-1)/2
        triu_sum = float(np.triu(gram, k=1).sum())
        n_pairs = k * (k - 1) / 2
        coherence[i] = triu_sum / n_pairs if n_pairs > 0 else 0.0

    return coherence


def cluster_quality(coherence, cluster_labels, threshold_pct=75):
    """Identify meta/structural clusters from neighbor coherence.

    Clusters whose members have high mean coherence are structural echo
    chambers — collections of documents that reference each other heavily
    (date pages, year pages, event lists) rather than covering a genuine
    topic.

    Args:
        coherence: (n,) array of per-point neighbor coherence values
                   (from neighbor_coherence).
        cluster_labels: (n,) array of cluster assignments.
        threshold_pct: Percentile of per-cluster mean coherence above
                       which a cluster is flagged as meta.  Default 75.

    Returns:
        dict with:
            cluster_mean_coherence: (n_clusters,) array of per-cluster
                                    mean coherence.
            meta_clusters: set of cluster IDs above the threshold.
            threshold: the computed coherence threshold value.
    """
    coherence = np.asarray(coherence)
    cluster_labels = np.asarray(cluster_labels)
    n_clusters = len(set(cluster_labels.tolist()))

    cluster_mean_coh = np.zeros(n_clusters)
    for c in range(n_clusters):
        mask = cluster_labels == c
        if mask.any():
            cluster_mean_coh[c] = coherence[mask].mean()

    coh_threshold = float(np.percentile(cluster_mean_coh, threshold_pct))
    meta_clusters = {
        int(c) for c in range(n_clusters)
        if cluster_mean_coh[c] > coh_threshold
    }

    return {
        'cluster_mean_coherence': cluster_mean_coh,
        'meta_clusters': meta_clusters,
        'threshold': coh_threshold,
    }
