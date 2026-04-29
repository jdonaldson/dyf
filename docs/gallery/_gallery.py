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
