"""Shared helpers for the parameter-free clustering gallery.

Every notebook in the gallery imports `run_dyf` and `plot_panels` from here
so the story stays consistent: same three lines of DYF, same three-panel
figure (ground truth / DYF / comparison), same metric box.
"""

from __future__ import annotations

import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class GalleryResult:
    """What every notebook produces."""
    labels: np.ndarray          # (N,) int cluster ids from DYF
    recovered_k: int            # number of unique cluster ids
    true_k: int                 # ground-truth class count
    nmi: float                  # normalized mutual information
    ari: float                  # adjusted rand index
    umap_2d: np.ndarray         # (N, 2) 2D layout for plotting

    def save(self, path: str) -> None:
        np.savez(path, labels=self.labels, recovered_k=self.recovered_k,
                 true_k=self.true_k, nmi=self.nmi, ari=self.ari,
                 umap_2d=self.umap_2d)

    @classmethod
    def load(cls, path: str) -> "GalleryResult":
        z = np.load(path)
        return cls(
            labels=z["labels"], recovered_k=int(z["recovered_k"]),
            true_k=int(z["true_k"]), nmi=float(z["nmi"]),
            ari=float(z["ari"]), umap_2d=z["umap_2d"],
        )


def auto_tune_tree_params(n: int, target_bucket_size: int = 20) -> dict:
    """Pick tree parameters from N + a single externalized knob.

    Validated across 9 gallery datasets: beats DYF's documented defaults
    (num_bits=3, max_depth=4, min_leaf=20) on 6-7/9 by mean +0.05 ARI.

    The single knob ``target_bucket_size`` is roughly the smallest natural
    cluster you want to detect. Default 20 is reasonable; drop to 5-10 for
    high-k small-n data (Olivetti at 10 samples/class), raise to 30-50 for
    clean cluster structure with many samples per cluster.

    Returns dict ready for ``build_dyf_tree(**params)``.
    """
    import math
    target_leaves = max(4, n // target_bucket_size)
    max_depth = max(2, min(6, math.ceil(math.log(target_leaves) / math.log(4))))
    return dict(
        num_bits=2,
        max_depth=max_depth,
        min_leaf_size=max(target_bucket_size // 2, 4),
    )


def run_dyf(
    embeddings: np.ndarray,
    y_true: np.ndarray,
    *,
    max_depth: int | None = None,
    num_bits: int | None = None,
    min_leaf_size: int | None = None,
    target_bucket_size: int = 20,
    seed: int = 42,
) -> GalleryResult:
    """Run DYF clustering end-to-end with auto-tuned tree parameters.

    Tree parameters are auto-tuned via ``auto_tune_tree_params(n,
    target_bucket_size)`` by default. Explicit ``max_depth`` / ``num_bits`` /
    ``min_leaf_size`` override the auto-tune individually if supplied.

    The auto-tune was empirically validated across 9 gallery datasets to beat
    DYF's documented defaults. Pass any explicit override to opt out per-knob.
    """
    from sklearn.metrics import adjusted_mutual_info_score, adjusted_rand_score

    from dyf import build_dyf_tree, write_lazy_index, LazyIndex
    from dyf.agglomerate import louvain_cluster_leaves

    embeddings = np.ascontiguousarray(embeddings, dtype=np.float32)

    auto = auto_tune_tree_params(len(embeddings), target_bucket_size)
    final_params = {
        "max_depth": auto["max_depth"] if max_depth is None else max_depth,
        "num_bits": auto["num_bits"] if num_bits is None else num_bits,
        "min_leaf_size": auto["min_leaf_size"] if min_leaf_size is None else min_leaf_size,
    }

    tree = build_dyf_tree(
        embeddings,
        seed=seed,
        **final_params,
    )

    umap_2d = _umap(embeddings, seed=seed)

    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "gallery.dyf"
        write_lazy_index(
            tree,
            embeddings,
            str(path),
            compression="none",
            quantization="float32",
        )
        with LazyIndex(str(path)) as idx:
            point_labels, *_ = louvain_cluster_leaves(idx, umap_2d, embeddings)

    labels = np.asarray(point_labels, dtype=np.int64)
    recovered_k = int(len(np.unique(labels)))
    true_k = int(len(np.unique(y_true)))

    return GalleryResult(
        labels=labels,
        recovered_k=recovered_k,
        true_k=true_k,
        nmi=float(adjusted_mutual_info_score(y_true, labels)),
        ari=float(adjusted_rand_score(y_true, labels)),
        umap_2d=umap_2d,
    )


def run_dyf_cached(
    embeddings: np.ndarray,
    y_true: np.ndarray,
    cache_path: str,
    **kwargs: Any,
) -> GalleryResult:
    """Cached wrapper around ``run_dyf``. First call computes and writes to
    ``cache_path``; later calls load from disk. Cache is invalidated on shape
    mismatch against current inputs."""
    import os
    if os.path.exists(cache_path):
        cached = GalleryResult.load(cache_path)
        if cached.labels.shape[0] == embeddings.shape[0]:
            return cached
    result = run_dyf(embeddings, y_true, **kwargs)
    result.save(cache_path)
    return result


def run_kmeans(embeddings: np.ndarray, y_true: np.ndarray, *, seed: int = 42) -> dict[str, Any]:
    """K-means with the oracle's k. Unfair ceiling baseline."""
    from sklearn.cluster import KMeans
    from sklearn.metrics import adjusted_mutual_info_score, adjusted_rand_score

    true_k = int(len(np.unique(y_true)))
    labels = KMeans(n_clusters=true_k, n_init="auto", random_state=seed).fit_predict(embeddings)
    return {
        "labels": labels,
        "recovered_k": true_k,
        "nmi": float(adjusted_mutual_info_score(y_true, labels)),
        "ari": float(adjusted_rand_score(y_true, labels)),
    }


def run_hdbscan(embeddings: np.ndarray, y_true: np.ndarray) -> dict[str, Any] | None:
    """HDBSCAN with defaults. Parameter-light but not parameter-free."""
    try:
        import hdbscan  # type: ignore
    except ImportError:
        return None
    from sklearn.metrics import adjusted_mutual_info_score, adjusted_rand_score

    labels = hdbscan.HDBSCAN().fit_predict(embeddings)
    mask = labels >= 0
    noise_frac = float((~mask).mean())
    if mask.sum() < 2:
        return {"labels": labels, "recovered_k": 0, "nmi": 0.0, "ari": 0.0,
                "noise_frac": noise_frac}
    return {
        "labels": labels,
        "recovered_k": int(len(np.unique(labels[mask]))),
        "nmi": float(adjusted_mutual_info_score(y_true[mask], labels[mask])),
        "ari": float(adjusted_rand_score(y_true[mask], labels[mask])),
        "noise_frac": noise_frac,
    }


def _umap(embeddings: np.ndarray, *, seed: int = 42) -> np.ndarray:
    """2D UMAP layout for the figure. Fall back to PCA if UMAP isn't installed."""
    try:
        import umap  # type: ignore
        layout = umap.UMAP(n_components=2, random_state=seed).fit_transform(embeddings)
    except ImportError:
        from sklearn.decomposition import PCA
        layout = PCA(n_components=2, random_state=seed).fit_transform(embeddings)
    return np.asarray(layout, dtype=np.float32)


def plot_single(
    umap_2d: np.ndarray,
    labels: np.ndarray,
    *,
    title: str,
    cmap: str = "tab20",
):
    """One square UMAP scatter. Square so it stacks cleanly in fluid columns.

    Intended use: call once per panel (ground truth / DYF / k-means) and let
    Quarto lay them out via ``#| layout-ncol`` — vertical on narrow screens,
    side-by-side on wide ones.
    """
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(umap_2d[:, 0], umap_2d[:, 1], c=labels, cmap=cmap, s=5, alpha=0.7)
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    fig.tight_layout()
    plt.close(fig)  # prevent pyplot from also auto-displaying — Quarto uses the returned Figure
    return fig


def hierarchy_slider(
    result: "GalleryResult",
    embeddings: np.ndarray,
    y_true: np.ndarray,
    k_values: list[int],
    *,
    class_names: list[str] | None = None,
    show_ground_truth: bool = False,
    height: int = 560,
    title_prefix: str = "DYF partition at",
):
    """Interactive plotly slider that sweeps DYF's merge hierarchy.

    Pre-computes ``merge_to_max_k`` at each requested k, stacks one WebGL
    scatter trace per k on the UMAP layout, and wires a slider to toggle which
    trace is visible. NMI / ARI vs ``y_true`` appear in the title as the slider
    moves. ``k_values`` higher than ``result.recovered_k`` collapse to the raw
    partition (nothing to merge up). Uses Turbo colorscale.
    """
    import plotly.graph_objects as go
    from sklearn.metrics import adjusted_mutual_info_score, adjusted_rand_score
    from dyf.agglomerate import merge_to_max_k

    labels_i32 = result.labels.astype(np.int32)
    coords = result.umap_2d

    # Fixed palette keyed on the raw (finest) partition — every point gets a
    # stable color based on which raw cluster it belongs to. As the slider
    # moves to coarser k, we re-color each point with the palette entry of the
    # *representative* (largest) raw cluster in its merged group. Small raw
    # clusters visibly "get absorbed" into their larger siblings' color, but
    # points in the dominant cluster never change color across the whole sweep.
    raw_ids = np.unique(labels_i32)
    import plotly.colors as pc
    palette = pc.qualitative.Alphabet + pc.qualitative.Light24 + pc.qualitative.Dark24
    raw_color = {int(rid): palette[i % len(palette)] for i, rid in enumerate(raw_ids)}

    def color_points_at(merged_labels: np.ndarray) -> list[str]:
        """For each point, return the hex color of the representative raw cluster
        in its merged group at this resolution."""
        # group raw cluster -> merged cluster id
        raw_to_merged: dict[int, int] = {}
        for raw_id in raw_ids:
            m = labels_i32 == raw_id
            if m.any():
                raw_to_merged[int(raw_id)] = int(merged_labels[np.argmax(m)])
        # for each merged group, pick the raw cluster with the largest size
        merged_to_rep: dict[int, int] = {}
        for raw_id, mid in raw_to_merged.items():
            size = int((labels_i32 == raw_id).sum())
            cur = merged_to_rep.get(mid)
            if cur is None or size > int((labels_i32 == cur).sum()):
                merged_to_rep[mid] = raw_id
        rep_color = {mid: raw_color[rep] for mid, rep in merged_to_rep.items()}
        return [rep_color[int(m)] for m in merged_labels]

    # Precompute each partition + scores, de-dup by actual_k.
    # Ascending order so the slider reads left=coarse → right=fine —
    # dragging right reveals more structure, matching how people read hierarchies.
    partitions: list[dict[str, Any]] = []
    seen: set[int] = set()
    for k in sorted(set(k_values)):
        merged = merge_to_max_k(labels_i32, embeddings, max_k=k)
        actual_k = int(len(np.unique(merged)))
        if actual_k in seen:
            continue
        seen.add(actual_k)
        partitions.append({
            "k": actual_k,
            "labels": merged,
            "colors": color_points_at(merged),
            "nmi": float(adjusted_mutual_info_score(y_true, merged)),
            "ari": float(adjusted_rand_score(y_true, merged)),
        })

    if not partitions:
        raise ValueError("No partitions to display — check k_values")

    # Open on the coarsest partition — reveal complexity as the user drags right.
    default_i = 0

    # Build per-point hover text: true class name (if provided) + point index.
    if class_names is not None:
        hover_text = [f"class: {class_names[int(y)]}<br>idx: {i}"
                      for i, y in enumerate(y_true)]
    else:
        hover_text = [f"class: {int(y)}<br>idx: {i}"
                      for i, y in enumerate(y_true)]

    # Ground-truth coloring uses a separate qualitative palette — distinct from
    # the partition palette so the eye doesn't confuse the two panels.
    gt_palette = pc.qualitative.Plotly + pc.qualitative.D3 + pc.qualitative.Bold
    gt_unique = np.unique(y_true)
    gt_color_map = {int(c): gt_palette[i % len(gt_palette)]
                    for i, c in enumerate(gt_unique)}
    gt_colors = [gt_color_map[int(c)] for c in y_true]

    fig: go.Figure
    if show_ground_truth:
        from plotly.subplots import make_subplots
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=("DYF (move slider)", "Ground truth (fixed)"),
            horizontal_spacing=0.04,
            shared_xaxes=True, shared_yaxes=True,
        )  # type: ignore[assignment]
    else:
        fig = go.Figure()

    # DYF traces — one per k-resolution, only the default visible
    dyf_trace_indices: list[int] = []
    for i, p in enumerate(partitions):
        trace = go.Scattergl(
            x=coords[:, 0], y=coords[:, 1],
            mode="markers",
            marker=dict(
                color=p["colors"],  # pre-computed hex per point — stable across frames
                size=4,
                opacity=0.75,
                line=dict(width=0),
            ),
            visible=(i == default_i),
            name=f"k={p['k']}",
            text=hover_text,
            hovertemplate="%{text}<extra></extra>",
        )
        if show_ground_truth:
            fig.add_trace(trace, row=1, col=1)
        else:
            fig.add_trace(trace)
        dyf_trace_indices.append(len(fig.data) - 1)  # type: ignore[arg-type]

    # Ground-truth trace (always visible, doesn't move with slider)
    if show_ground_truth:
        fig.add_trace(
            go.Scattergl(
                x=coords[:, 0], y=coords[:, 1],
                mode="markers",
                marker=dict(color=gt_colors, size=4, opacity=0.75, line=dict(width=0)),
                text=hover_text,
                hovertemplate="%{text}<extra></extra>",
                showlegend=False,
            ),
            row=1, col=2,
        )
        gt_trace_index = len(fig.data) - 1  # type: ignore[arg-type]

    # Slider — toggles DYF trace visibility only; ground-truth trace stays on.
    steps = []
    for i, p in enumerate(partitions):
        title = (
            f"{title_prefix} k={p['k']} — "
            f"NMI={p['nmi']:.3f}, ARI={p['ari']:.3f}"
        )
        if show_ground_truth:
            visible = [False] * len(fig.data)  # type: ignore[arg-type]
            visible[dyf_trace_indices[i]] = True
            visible[gt_trace_index] = True
        else:
            visible = [j == i for j in range(len(partitions))]
        steps.append(dict(
            method="update",
            label=str(p["k"]),
            args=[{"visible": visible}, {"title.text": title}],
        ))

    p0 = partitions[default_i]
    layout_axes = dict(
        xaxis=dict(showticklabels=False, zeroline=False, showgrid=False),
        yaxis=dict(showticklabels=False, zeroline=False, showgrid=False,
                   scaleanchor="x", scaleratio=1),
    )
    if show_ground_truth:
        layout_axes = dict(
            xaxis=dict(showticklabels=False, zeroline=False, showgrid=False),
            yaxis=dict(showticklabels=False, zeroline=False, showgrid=False),
            xaxis2=dict(showticklabels=False, zeroline=False, showgrid=False),
            yaxis2=dict(showticklabels=False, zeroline=False, showgrid=False),
        )

    layout_kwargs: dict[str, Any] = dict(
        title=dict(
            text=f"{title_prefix} k={p0['k']} — "
                 f"NMI={p0['nmi']:.3f}, ARI={p0['ari']:.3f}",
            x=0.5, xanchor="center",
        ),
        sliders=[dict(
            active=default_i,
            currentvalue=dict(prefix="k = ", font=dict(size=14)),
            steps=steps,
            pad=dict(t=40, b=10),
        )],
        height=height,
        margin=dict(l=20, r=20, t=80, b=40),
        showlegend=False,
    )
    layout_kwargs.update(layout_axes)
    fig.update_layout(**layout_kwargs)
    return fig


def image_grid(
    images: np.ndarray,
    indices: np.ndarray,
    *,
    image_shape: tuple[int, int] = (28, 28),
    rows: int = 4,
    cols: int = 8,
    title: str = "",
    seed: int = 0,
):
    """Render a grid of images drawn from ``images[indices]``.

    ``images`` may be flat (N, d) — each row is reshaped to ``image_shape``.
    Sampling is random without replacement; pass ``seed`` to reproduce.
    """
    import matplotlib.pyplot as plt

    rng = np.random.default_rng(seed)
    pick = rng.choice(indices, size=min(rows * cols, len(indices)), replace=False)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 0.9, rows * 0.9))
    for ax, idx in zip(axes.flat, pick):
        ax.imshow(images[idx].reshape(image_shape), cmap="gray", vmin=0, vmax=1)
        ax.set_xticks([])
        ax.set_yticks([])
    # Blank any axes we didn't fill
    for ax in list(axes.flat)[len(pick):]:
        ax.axis("off")
    if title:
        fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    plt.close(fig)
    return fig


def merge_walk(
    dyf: GalleryResult,
    embeddings: np.ndarray,
    y_true: np.ndarray,
    targets: list[int],
) -> list[dict[str, Any]]:
    """Walk DYF's partition hierarchy to coarser resolutions.

    Calls ``dyf.agglomerate.merge_to_max_k`` at each target k, scores the
    resulting partition against ``y_true`` with NMI and ARI, and returns a
    list of rows. Targets larger than the raw recovered_k are returned as the
    raw result (nothing to merge up). Targets smaller merge to that count.
    """
    from sklearn.metrics import adjusted_mutual_info_score, adjusted_rand_score
    from dyf.agglomerate import merge_to_max_k

    labels_i32 = dyf.labels.astype(np.int32)
    rows: list[dict[str, Any]] = []
    for target in targets:
        merged = merge_to_max_k(labels_i32, embeddings, max_k=target)
        k = int(len(np.unique(merged)))
        rows.append({
            "target": target,
            "actual_k": k,
            "nmi": float(adjusted_mutual_info_score(y_true, merged)),
            "ari": float(adjusted_rand_score(y_true, merged)),
        })
    return rows


def merge_walk_table(rows: list[dict[str, Any]], raw: GalleryResult) -> str:
    """Markdown table showing the merge walk — raw DYF + merged resolutions."""
    lines = [
        "| Resolution     | Actual k | NMI   | ARI   |",
        "|----------------|---------:|------:|------:|",
        f"| **Raw DYF**    | **{raw.recovered_k}** | **{raw.nmi:.3f}** | **{raw.ari:.3f}** |",
    ]
    for r in rows:
        lines.append(
            f"| merge → {r['target']:<4d}  | {r['actual_k']:>4d}     | "
            f"{r['nmi']:.3f} | {r['ari']:.3f} |"
        )
    return "\n".join(lines)


def metrics_table(dyf: GalleryResult, kmeans: dict | None, hdbscan_: dict | None) -> str:
    """Markdown table for the bottom-of-notebook comparison."""
    lines = [
        "| Method | Recovered k | NMI | ARI | % discarded |",
        "|--------|------------:|----:|----:|------------:|",
        f"| **DYF** (parameter-free) | {dyf.recovered_k} | {dyf.nmi:.3f} | {dyf.ari:.3f} | 0% |",
    ]
    if kmeans is not None:
        lines.append(
            f"| k-means (oracle k={dyf.true_k}) | {kmeans['recovered_k']} | "
            f"{kmeans['nmi']:.3f} | {kmeans['ari']:.3f} | 0% |"
        )
    if hdbscan_ is not None:
        noise = hdbscan_.get("noise_frac", 0.0)
        lines.append(
            f"| HDBSCAN (defaults) | {hdbscan_['recovered_k']} | "
            f"{hdbscan_['nmi']:.3f} | {hdbscan_['ari']:.3f} | {noise:.0%} |"
        )
    return "\n".join(lines)


# ============================================================================
# Topology Diagnostic Stack
# ============================================================================
#
# Five layers, validated across 7 datasets (1.3M brain, Paul15 hematopoiesis,
# MoCap, MNIST, CIFAR-10, 20NG, PBMC3k). See diagnostic-stack.qmd for the full
# walkthrough; below is the substrate.
#
#   1. Geometric:      anisotropy + path/star + density + rho on centroid kNN
#   2. Spectral:       Nyström-via-bucket-centroids (cheap on full data)
#   3. Plateau:        eigenvalue distribution -> effective dim, gap_2, e12
#   4. Within-cluster: median top-1 z + sub-marker Jaccard (needs gene-level data)
#   5. Modality:       gene-panel scores (mt, Hb, IEG, ...; biology-only)
#
# This module exposes layers 1-3 (geometric primitives). Layers 4-5 belong in
# domain-specific code (scRNA-seq).


def nystrom_spectral(
    X: np.ndarray,
    bucket_labels: np.ndarray,
    *,
    n_components: int = 15,
    k_centroid_nn: int = 15,
):
    """Spectral embedding via Nyström-on-bucket-centroids.

    DYF (or k-means) gives you bucket assignments for every point. We use
    bucket centroids as Nyström landmarks: build a kNN graph on the centroids,
    compute top-k Laplacian eigenvectors there (a tiny problem), then lift
    back to per-point coordinates by hard assignment to bucket.

    On 1.29M brain cells with 1317 buckets: total pipeline runs in ~3s,
    eigendecomposition itself is sub-millisecond. Compare to direct sparse
    spectral on 1.29M cells (estimated 60-300s) — ~25-100× speedup.

    Parameters
    ----------
    X : (N, D) array
        Per-point features (e.g. PCA-50d).
    bucket_labels : (N,) integer array
        Bucket assignment per point. Buckets need not be contiguous integers.
    n_components : int
        Number of spectral components to return. Top non-trivial eigenvectors.
    k_centroid_nn : int
        kNN graph degree among centroids.

    Returns
    -------
    cell_spec : (N, n_components) array
        Spectral coordinate for each input point.
    centroid_spec : (n_buckets, n_components) array
        The underlying centroid spectral coords.
    bucket_counts : (n_buckets,) int array
        Number of points in each bucket (in centroid index order).
    """
    import scipy.sparse as sp
    from scipy.sparse.linalg import eigsh
    from sklearn.neighbors import NearestNeighbors

    unique_buckets, inverse = np.unique(bucket_labels, return_inverse=True)
    n_buckets = len(unique_buckets)
    bucket_sums = np.zeros((n_buckets, X.shape[1]), dtype=np.float64)
    bucket_counts = np.zeros(n_buckets, dtype=np.int64)
    np.add.at(bucket_sums, inverse, X)
    np.add.at(bucket_counts, inverse, 1)
    centroids = (bucket_sums / np.maximum(bucket_counts[:, None], 1)).astype(np.float32)

    nn = NearestNeighbors(n_neighbors=min(k_centroid_nn + 1, n_buckets)).fit(centroids)
    dists, indices = nn.kneighbors(centroids)
    sigma = float(np.median(dists[:, 1:]))
    rows, cols, vals = [], [], []
    for i in range(n_buckets):
        for j_pos in range(1, min(k_centroid_nn + 1, n_buckets)):
            j = int(indices[i, j_pos])
            d = float(dists[i, j_pos])
            w = np.exp(-d * d / (2 * sigma * sigma))
            rows.append(i); cols.append(j); vals.append(w)
    W = sp.csr_matrix((vals, (rows, cols)), shape=(n_buckets, n_buckets))
    W = W.maximum(W.T)
    deg = np.asarray(W.sum(axis=1)).flatten()
    D_inv_sqrt = sp.diags(1.0 / np.sqrt(np.maximum(deg, 1e-12)))
    P = D_inv_sqrt @ W @ D_inv_sqrt
    n_eig = min(n_components + 1, n_buckets - 1)
    eigvals_P, eigvecs_P = eigsh(P, k=n_eig, which="LM")
    idx_sort = np.argsort(-eigvals_P)
    eigvecs_P = eigvecs_P[:, idx_sort]
    centroid_spec = eigvecs_P[:, 1:n_eig].astype(np.float32)

    cell_spec = centroid_spec[inverse]
    return cell_spec, centroid_spec, bucket_counts


def cluster_diagnostic(
    centroid_coords: np.ndarray,
    bucket_counts: np.ndarray,
    cluster_mask_for_buckets: np.ndarray,
    *,
    min_bucket_cells: int = 20,
    min_cluster_buckets: int = 5,
    k_nn: int = 5,
) -> dict | None:
    """Compute geometric + plateau diagnostic for one cluster.

    Operates on bucket centroids (not per-cell), weighted by bucket sizes.
    Parameters are robust to choice of representation (PCA or spectral).

    Returns a dict with:
      - n_buckets  : number of buckets passing min_bucket_cells filter
      - anis       : λ_1 / Σλ on weighted covariance
      - p2, p3     : top-2 / top-3 cumulative variance fraction
      - eff70      : smallest k s.t. cumulative variance >= 70%
      - gap2       : λ_3 / λ_2 (sharpness of dim-2 plateau, low = clean 2D)
      - e12        : λ_2 / λ_1 (cycle/sin-cos signature when high)
      - density    : avg degree on bucket-centroid kNN graph
      - path       : MST diameter / (n-1)
      - star       : MST max degree / (n-1)
      - rho        : Pearson correlation of (degree, distance from cluster mean)

    None is returned when the cluster has too few buckets to characterize.
    """
    import scipy.sparse as sp
    from scipy.sparse.csgraph import minimum_spanning_tree, shortest_path
    from sklearn.neighbors import NearestNeighbors

    sub_centroids = centroid_coords[cluster_mask_for_buckets]
    sub_counts = bucket_counts[cluster_mask_for_buckets]
    keep = sub_counts >= min_bucket_cells
    sub_centroids = sub_centroids[keep]
    sub_counts = sub_counts[keep]
    n = len(sub_centroids)
    if n < min_cluster_buckets:
        return None

    weights = (sub_counts / sub_counts.sum()).astype(np.float64)
    mean = (weights[:, None] * sub_centroids).sum(axis=0)
    centered = sub_centroids - mean
    cov = (centered * weights[:, None]).T @ centered
    eigvals = np.sort(np.clip(np.linalg.eigvalsh(cov), 0, None))[::-1]
    total = eigvals.sum()
    if total <= 0:
        return None
    cumfrac = np.cumsum(eigvals) / total

    anis = float(eigvals[0] / total)
    p2 = float(cumfrac[1]) if len(cumfrac) > 1 else float(cumfrac[0])
    p3 = float(cumfrac[2]) if len(cumfrac) > 2 else float(cumfrac[-1])
    eff70 = int(np.searchsorted(cumfrac, 0.70) + 1)
    gap2 = float(eigvals[2] / eigvals[1]) if len(eigvals) >= 3 and eigvals[1] > 0 else float("nan")
    e12 = float(eigvals[1] / eigvals[0]) if eigvals[0] > 0 else 0.0

    k = min(k_nn, n - 1)
    nn = NearestNeighbors(n_neighbors=k + 1).fit(sub_centroids)
    _, idx_nn = nn.kneighbors(sub_centroids)
    rows, cols = [], []
    for i in range(n):
        for j_pos in range(1, k + 1):
            rows.append(i); cols.append(int(idx_nn[i, j_pos]))
    G = sp.csr_matrix(([1.0] * len(rows), (rows, cols)), shape=(n, n))
    G = G.maximum(G.T)
    full_deg = np.asarray((G > 0).sum(axis=1)).flatten()
    density = float(G.nnz / (2 * n))

    mst = minimum_spanning_tree(G)
    mst_sym = mst + mst.T
    d_mat = shortest_path(mst_sym, directed=False, unweighted=True)
    diam = float(np.nanmax(d_mat[np.isfinite(d_mat)]))
    deg_mst = (mst_sym > 0).sum(axis=1).A1
    path = diam / (n - 1)
    star = int(deg_mst.max()) / (n - 1)

    dist_from_mean = np.linalg.norm(centered, axis=1)
    if len(set(full_deg)) > 1 and len(set(dist_from_mean.round(4))) > 1:
        rho = float(np.corrcoef(full_deg, dist_from_mean)[0, 1])
    else:
        rho = float("nan")

    return dict(
        n_buckets=n, anis=anis, p2=p2, p3=p3, eff70=eff70,
        gap2=gap2, e12=e12, density=density, path=path, star=star, rho=rho,
        eigvals_top5=eigvals[:5].tolist(),
    )


def classify_cluster(d: dict) -> str:
    """Map a diagnostic dict to one of the topology classes.

    Class definitions (validated on brain + Paul15 + MoCap + MNIST):
      PATH       : 1D dominant axis (anis>0.65)
      CYCLE      : top-2 eigenvalues nearly equal (e12>0.80) at moderate anis
      2D-LATTICE : top-2 captures most variance, sharp gap after 2 (gap2<0.4)
      3D-LATTICE : eff_dim=3 with top-3 capturing >85%, dense topology
      TWO-HUB    : high anis with positive deg-eccen rho (hubs at extremes)
      STAR       : eff_dim>=4, sparse, deg-eccen rho strongly negative
      HUB        : MST star_score>0.55, deg-eccen rho strongly negative
      mixed      : doesn't fit any clean class

    Anisotropy gate (>0.18) is implicit: low-anis clusters with high path
    score from few k-means buckets fall into "mixed" rather than PATH.
    """
    anis = d["anis"]; e12 = d["e12"]; eff = d["eff70"]
    p2 = d["p2"]; p3 = d["p3"]; gap2 = d["gap2"]
    dens = d["density"]; rho = d["rho"]; star = d["star"]

    if 0.30 < anis < 0.60 and e12 > 0.80 and dens > 1.5:
        return "CYCLE"
    if 0.30 < anis < 0.65 and p2 > 0.65 and (np.isnan(gap2) or gap2 < 0.4) and dens > 1.5:
        return "2D-LATTICE"
    if eff >= 3 and p3 > 0.85 and dens > 1.5:
        return "3D-LATTICE"
    if anis > 0.65:
        return "PATH"
    if anis > 0.55 and not np.isnan(rho) and rho > 0.4:
        return "TWO-HUB"
    if eff >= 4 and dens < 1.5 and not np.isnan(rho) and rho < -0.3:
        return "STAR"
    if star > 0.55 and not np.isnan(rho) and rho < -0.3:
        return "HUB"
    return "mixed"


def diagnose_all_clusters(
    X: np.ndarray,
    bucket_labels: np.ndarray,
    cluster_labels: np.ndarray,
    *,
    use_spectral: bool = True,
    n_components: int = 15,
) -> list[dict]:
    """Run the full diagnostic stack across every cluster in ``cluster_labels``.

    Parameters
    ----------
    X : (N, D) array
        Per-point features.
    bucket_labels : (N,) int array
        Sub-cluster / bucket assignment (DYF leaves, k-means, etc.).
    cluster_labels : (N,) int array
        Top-level cluster assignment (e.g. scanpy Leiden, ground-truth class).
    use_spectral : bool
        If True, run Nyström spectral first and diagnose in spectral coords.
        If False, diagnose in PCA coords directly.
    n_components : int
        Spectral components if ``use_spectral=True``.

    Returns
    -------
    rows : list of dicts
        One per cluster, each containing the cluster id, the diagnostic
        signals, and the topology classification.
    """
    if use_spectral:
        _, centroid_coords, bucket_counts = nystrom_spectral(
            X, bucket_labels, n_components=n_components
        )
        unique_buckets = np.unique(bucket_labels)
    else:
        unique_buckets, inverse = np.unique(bucket_labels, return_inverse=True)
        n_buckets = len(unique_buckets)
        bsum = np.zeros((n_buckets, X.shape[1]), dtype=np.float64)
        bcnt = np.zeros(n_buckets, dtype=np.int64)
        np.add.at(bsum, inverse, X)
        np.add.at(bcnt, inverse, 1)
        centroid_coords = (bsum / np.maximum(bcnt[:, None], 1)).astype(np.float32)
        bucket_counts = bcnt

    rows = []
    for c in sorted(np.unique(cluster_labels)):
        cluster_buckets = np.unique(bucket_labels[cluster_labels == c])
        mask = np.isin(unique_buckets, cluster_buckets)
        d = cluster_diagnostic(centroid_coords, bucket_counts, mask)
        if d is None:
            continue
        d["cluster"] = c
        d["classification"] = classify_cluster(d)
        rows.append(d)
    return rows


# ─────────────────────────────────────────────────────────────────────────
# Layer 5 → Layer 7 dependency primitives
#
# The PH layer (Layer 7) requires reference-module QC (Layer 5) as a
# prerequisite, per the 2026-04-29 falsification finding. Without QC,
# stress / death / damage signatures inflate persistence values and
# create false ring signals. See
# `~/.claude/projects/.../memory/project_ph_layer_falsification.md`.
#
# These primitives are the mechanical enforcement of that dependency.
# ─────────────────────────────────────────────────────────────────────────


# Default reference modules for scRNA-seq stress/death scoring.
# Mouse symbol case. Add more or override per domain.
DEFAULT_STRESS_MODULES: dict[str, list[str]] = {
    "apoptosis": ["Casp3", "Casp7", "Casp8", "Casp9", "Bax", "Bak1",
                   "Bcl2l11", "Cycs", "Apaf1", "Diablo", "Pmaip1", "Bid"],
    "upr":       ["Hspa5", "Atf4", "Ddit3", "Xbp1", "Atf6", "Ern1", "Eif2ak3"],
    "hsp":       ["Hspa1a", "Hspa1b", "Hsp90aa1", "Hsp90ab1", "Hspb1", "Dnajb1"],
}


def score_stress_modules(
    adata,
    modules: dict[str, list[str]] | None = None,
) -> list[str]:
    """Score each cell on stress / death / damage gene modules.

    Adds one obs column per module (named ``{key}_score``). Uses scanpy's
    ``sc.tl.score_genes`` (Tirosh-style: mean expression of module minus
    mean expression of a length-matched random control set).

    Parameters
    ----------
    adata : AnnData
        Must already be normalized + log1p-transformed. Raw counts will
        give nonsensical scores.
    modules : dict[str, list[str]] | None
        Mapping ``module_name → list of gene symbols``. Defaults to
        ``DEFAULT_STRESS_MODULES`` (apoptosis, upr, hsp).

    Returns
    -------
    list[str]
        Names of the score columns added to ``adata.obs``.

    Notes
    -----
    For non-biology domains, replace ``modules`` with domain-specific
    reference panels (e.g., for text: ``{"stop_words": [...],
    "ocr_artifacts": [...]}``). The mechanism is the same: pre-defined
    feature subsets that signal known generic-state confounds.
    """
    import scanpy as sc

    modules = modules if modules is not None else DEFAULT_STRESS_MODULES
    added = []
    for name, genes in modules.items():
        present = [g for g in genes if g in adata.var_names]
        if not present:
            continue
        col = f"{name}_score"
        sc.tl.score_genes(adata, gene_list=present, score_name=col)
        added.append(col)
    return added


def qc_filter_mask(
    adata,
    *,
    score_columns: list[str] | None = None,
    drop_top_pct: float = 10.0,
    pct_mt_max: float | None = 10.0,
    pct_hb_max: float | None = 5.0,
    n_genes_min: int = 200,
) -> np.ndarray:
    """Return a boolean mask of cells passing reference-module QC.

    A cell passes if all of:
      - ``n_genes`` (or ``n_genes_by_counts``) ≥ ``n_genes_min``
      - ``pct_mt`` < ``pct_mt_max`` (skipped if column missing)
      - ``pct_hb`` < ``pct_hb_max`` (skipped if column missing)
      - For each score column: cell is NOT in the top ``drop_top_pct%``
        of that score's distribution

    Parameters
    ----------
    adata : AnnData
        Must have any score columns referenced (typically created by
        :func:`score_stress_modules`).
    score_columns : list[str] | None
        Score columns whose top-percentile cells should be dropped.
        Defaults to all ``*_score`` columns in ``adata.obs``.
    drop_top_pct : float
        Drop cells in the top this percent of EACH score (combined via
        OR — failing any one drops the cell).
    pct_mt_max, pct_hb_max : float | None
        Maximum mitochondrial / hemoglobin fractions. Set to ``None`` to
        skip the check (e.g., when mt-genes are pre-stripped, as in TMS).
    n_genes_min : int
        Minimum gene count per cell.

    Returns
    -------
    np.ndarray of bool, shape (n_cells,)
        True for cells that pass.

    Notes
    -----
    Use ``adata[qc_filter_mask(adata)]`` to obtain the QC-filtered subset
    before running PH (Layer 7). On TMS-style preprocessed data,
    ``pct_mt`` may be unavailable because mt-genes were stripped during
    upstream preprocessing — pass ``pct_mt_max=None`` in that case.
    """
    import pandas as pd

    n = adata.n_obs
    keep = np.ones(n, dtype=bool)

    # n_genes (try both common column names)
    n_genes_col = None
    for c in ("n_genes", "n_genes_by_counts"):
        if c in adata.obs.columns:
            n_genes_col = c
            break
    if n_genes_col is not None:
        keep &= np.asarray(adata.obs[n_genes_col]) >= n_genes_min

    if pct_mt_max is not None and "pct_mt" in adata.obs.columns:
        keep &= np.asarray(adata.obs["pct_mt"]) < pct_mt_max
    if pct_hb_max is not None and "pct_hb" in adata.obs.columns:
        keep &= np.asarray(adata.obs["pct_hb"]) < pct_hb_max

    if score_columns is None:
        score_columns = [c for c in adata.obs.columns
                          if c.endswith("_score") and pd.api.types.is_numeric_dtype(
                              adata.obs[c])]

    if drop_top_pct > 0:
        for c in score_columns:
            if c not in adata.obs.columns:
                continue
            vals = np.asarray(adata.obs[c], dtype=np.float64)
            thresh = np.quantile(vals, 1.0 - drop_top_pct / 100.0)
            keep &= vals < thresh

    return keep


def wcs_persistence_null(
    pca: np.ndarray,
    cluster_labels: np.ndarray,
    n_landmarks: int,
    *,
    n_shuffles: int = 5,
    seed: int = 42,
) -> np.ndarray:
    """Build a pooled within-cluster-shuffled persistence null distribution.

    Per the 2026-04-29 calibration finding, the dataset-relative
    "robust = top 20% of max persistence" threshold admits clustering-
    geometry artifacts. A WCS-calibrated significance test compares
    each real cycle's persistence to a pool of persistences from
    within-cluster-shuffled data — same cluster identities preserved,
    within-cluster joint structure broken.

    Parameters
    ----------
    pca : np.ndarray, shape (n_cells, n_pcs)
        The PCA-embedded data. Same input that produced the real
        cycles being tested.
    cluster_labels : np.ndarray, shape (n_cells,)
        Discrete labels (typically Leiden or cell-type) — the
        clustering whose internal structure WCS preserves while
        permuting feature values within each cluster.
    n_landmarks : int
        Same k-means landmark count used for the real run. Must match
        for persistences to be comparable.
    n_shuffles : int
        Number of WCS replicates. Default 5; raise to 10+ for tighter
        z-score estimates on small datasets.
    seed : int
        Base seed; replicate ``k`` uses ``seed + k + 1`` so the real
        run (which conventionally uses ``seed`` directly) is excluded.

    Returns
    -------
    np.ndarray, shape (≈ n_shuffles × n_features_per_shuffle,)
        Pooled persistence values from all WCS replicates. Use as the
        null distribution against which real persistences are compared.

    See also
    --------
    cycle_lineage_spans : pass this output as ``wcs_persistences`` and
        set ``min_z_vs_wcs=3.0`` for null-calibrated cycle filtering.
    """
    import ripser
    from sklearn.cluster import KMeans

    rng = np.random.default_rng(seed)
    pooled = []
    for k in range(n_shuffles):
        shuffled = pca.copy()
        for c in np.unique(cluster_labels):
            idx = np.where(cluster_labels == c)[0]
            for j in range(pca.shape[1]):
                shuffled[idx, j] = pca[rng.permutation(idx), j]
        n_lm = min(n_landmarks, shuffled.shape[0] - 1)
        km = KMeans(n_clusters=n_lm, n_init=3, random_state=seed + k + 1).fit(shuffled)
        result = ripser.ripser(km.cluster_centers_, maxdim=1, do_cocycles=False)
        b1 = result["dgms"][1]
        if len(b1) > 0:
            pooled.extend((b1[:, 1] - b1[:, 0]).tolist())
    return np.asarray(pooled, dtype=np.float64)


def cycle_lineage_spans(
    cocycles_1: list,
    persistences: np.ndarray,
    landmark_labels: np.ndarray,
    *,
    threshold_frac: float = 0.20,
    min_distinct_types: int = 2,
    wcs_persistences: np.ndarray | None = None,
    min_z_vs_wcs: float | None = None,
) -> list[dict[str, Any]]:
    """Filter ripser Betti-1 cycles by robustness + vertex-content + (optional) null-calibration.

    The 2026-04-29 falsification arc found that PH cycle COUNT is a
    clustering measure (≈ within-cluster shuffle null). What survives
    as a unique signal is **vertex content with ≥2 distinct dominant
    types** — these cycles trace cross-cluster lineage relationships
    (myeloid lineage, erythroblastic islands, NK/NKT pair, etc.) and are
    not reproduced by the WCS null. Single-type cycles are clustering
    artifacts and should be filtered out.

    The ``threshold_frac=0.20`` rule is dataset-relative and admits
    clustering-geometry artifacts when the data has a large persistence
    ceiling. For null-calibrated significance, pass ``wcs_persistences``
    (from :func:`wcs_persistence_null`) and ``min_z_vs_wcs`` — empirical
    default ``3.0``. This was demonstrated on Marrow res=0.5: the 20%
    rule kept 20 cycles, but only the top 7 had z > 3 vs WCS pool;
    cycles 8-20 fell into the noise band.

    Note: replication on Lung / Liver / Limb_Muscle showed the
    lineage-span signal does NOT robustly generalize across all tissues.
    Strong in actively-differentiating tissues (Marrow); null or
    reversed in mostly-terminal-identity tissues (Lung, Liver). The
    filter is appropriate; whether the resulting cycles are biologically
    meaningful is dataset-dependent.

    Parameters
    ----------
    cocycles_1 : list
        ``ripser.ripser(...)["cocycles"][1]`` — one cocycle per Betti-1
        feature. Each cocycle is an array of ``[v_i, v_j, val]`` rows.
    persistences : np.ndarray, shape (n_features,)
        Persistence value (death − birth) for each Betti-1 feature.
    landmark_labels : np.ndarray, shape (n_landmarks,)
        Label per landmark — typically the dominant cell type (mode of
        cell types of cells assigned to that landmark by k-means).
    threshold_frac : float
        Robustness cutoff: keep features with persistence > this fraction
        of the maximum. Ignored when ``wcs_persistences`` + ``min_z_vs_wcs``
        are both provided (the WCS-z filter supersedes it).
    min_distinct_types : int
        Keep only cycles whose vertex set spans at least this many
        distinct labels. ``2`` filters out single-type clustering
        artifacts.
    wcs_persistences : np.ndarray | None
        Pooled persistence values from WCS replicates. When provided
        AND ``min_z_vs_wcs`` is set, replaces the relative threshold
        with a null-calibrated z-score filter. Each cycle's
        ``z_vs_wcs`` and empirical ``p_vs_wcs`` are added to its dict.
    min_z_vs_wcs : float | None
        Minimum z-score (cycle persistence vs WCS pool) to keep a cycle.
        Empirical default 3.0; lower (e.g. 2.0) is more permissive.

    Returns
    -------
    list[dict]
        One per surviving cycle, with keys: ``rank`` (1-indexed by
        persistence), ``persistence``, ``vertices``, ``vertex_labels``,
        ``n_distinct_types``, ``label_counts``. When WCS is provided:
        also ``z_vs_wcs`` and ``p_vs_wcs`` (empirical fraction of WCS
        persistences ≥ this cycle's persistence).
    """
    if len(persistences) == 0:
        return []

    use_wcs = wcs_persistences is not None and min_z_vs_wcs is not None
    if use_wcs:
        wcs_arr = np.asarray(wcs_persistences, dtype=np.float64)
        wcs_mean = float(wcs_arr.mean()) if len(wcs_arr) > 0 else 0.0
        wcs_std = float(wcs_arr.std() + 1e-9) if len(wcs_arr) > 0 else 1.0

    if use_wcs:
        z_threshold = float(min_z_vs_wcs)  # narrow Optional[float] → float
        z_per_cycle = (persistences - wcs_mean) / wcs_std
        robust_idx = np.where(z_per_cycle >= z_threshold)[0]
    else:
        pmax = float(persistences.max())
        threshold = threshold_frac * pmax
        robust_idx = np.where(persistences > threshold)[0]
    robust_idx = robust_idx[np.argsort(-persistences[robust_idx])]

    out = []
    for rank, idx in enumerate(robust_idx):
        cc = cocycles_1[idx]
        if cc.shape[0] == 0:
            continue
        verts = np.unique(cc[:, :2].astype(int))
        labels = landmark_labels[verts]
        unique_labels, counts = np.unique(labels, return_counts=True)
        if len(unique_labels) < min_distinct_types:
            continue
        entry = {
            "rank": rank + 1,
            "persistence": float(persistences[idx]),
            "vertices": verts,
            "vertex_labels": labels,
            "n_distinct_types": int(len(unique_labels)),
            "label_counts": dict(zip(unique_labels.tolist(), counts.tolist())),
        }
        if use_wcs:
            p = float(persistences[idx])
            entry["z_vs_wcs"] = float((p - wcs_mean) / wcs_std)
            entry["p_vs_wcs"] = float((wcs_arr >= p).mean())
        out.append(entry)
    return out
