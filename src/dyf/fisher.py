"""Fisher dimension weighting for embedding spaces.

Computes per-dimension sqrt(Fisher ratio) weights from coarse category labels
(e.g. GMDN family terms). Applying these weights before UMAP and DYF tree
building improves bucket purity by emphasizing discriminative dimensions.
"""

from __future__ import annotations

import numpy as np


def compute_fisher_weights(
    embeddings: np.ndarray,
    labels: np.ndarray,
    min_count: int = 50,
) -> np.ndarray:
    """Compute sqrt(Fisher ratio) per dimension, L2-normalized.

    Fisher ratio = between-class variance / within-class variance.
    Dimensions where classes differ most get the highest weight.

    Parameters
    ----------
    embeddings : (n, d) float32
        Embedding matrix.
    labels : (n,) str or int array
        Coarse category labels (one per row).
    min_count : int
        Classes with fewer than this many members are excluded.

    Returns
    -------
    weights : (d,) float32
        L2-normalized sqrt(Fisher ratio) weights. Uniform if <2 classes
        survive filtering.
    """
    embeddings = np.asarray(embeddings, dtype=np.float32)
    labels = np.asarray(labels)
    _, d = embeddings.shape

    # Filter to classes with enough members
    unique, counts = np.unique(labels, return_counts=True)
    keep_classes = unique[counts >= min_count]

    if len(keep_classes) < 2:
        # Not enough classes — return uniform weights
        w = np.ones(d, dtype=np.float32)
        w /= np.linalg.norm(w)
        return w

    # Mask to rows belonging to kept classes
    mask = np.isin(labels, keep_classes)
    emb_filt = embeddings[mask]
    lab_filt = labels[mask]

    # Global mean
    global_mean = emb_filt.mean(axis=0)

    # Between-class and within-class variance per dimension
    between = np.zeros(d, dtype=np.float64)
    within = np.zeros(d, dtype=np.float64)

    for cls in keep_classes:
        cls_mask = lab_filt == cls
        cls_emb = emb_filt[cls_mask]
        n_c = cls_emb.shape[0]
        cls_mean = cls_emb.mean(axis=0)

        between += n_c * (cls_mean - global_mean) ** 2
        within += ((cls_emb - cls_mean) ** 2).sum(axis=0)

    # Avoid division by zero
    within = np.maximum(within, 1e-12)
    fisher = between / within

    # sqrt(Fisher ratio), then L2-normalize
    weights = np.sqrt(fisher).astype(np.float32)
    norm = np.linalg.norm(weights)
    if norm > 0:
        weights /= norm

    return weights


def apply_fisher_weights(
    embeddings: np.ndarray,
    weights: np.ndarray,
) -> np.ndarray:
    """Multiply embeddings by per-dimension weights.

    Parameters
    ----------
    embeddings : (n, d) float32
    weights : (d,) float32

    Returns
    -------
    weighted : (n, d) float32
    """
    return (embeddings * weights[None, :]).astype(np.float32)


def extract_fisher_labels(
    raw_values: list | np.ndarray,
    mode: str = "first_term",
) -> np.ndarray:
    """Extract coarse family labels from a column of values.

    .. deprecated::
        Use ``coarsen(raw_values, strategy=mode)`` from ``dyf.categorical`` instead.

    Parameters
    ----------
    raw_values : list or array
        Column values. For GMDN-style data, each element is typically a
        string like ``"Forceps, bipolar, reusable"`` or a list of such
        strings.
    mode : str
        ``"first_term"`` — split on comma, take first token, lowercase.
        ``"raw"`` — use values as-is (converted to str).

    Returns
    -------
    labels : (n,) str array
    """
    import warnings
    warnings.warn(
        "extract_fisher_labels is deprecated, use coarsen(raw_values, strategy=mode) instead",
        DeprecationWarning,
        stacklevel=2,
    )
    from dyf.categorical import coarsen

    return coarsen(raw_values, strategy=mode)
