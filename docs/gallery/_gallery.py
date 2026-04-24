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


def run_dyf(
    embeddings: np.ndarray,
    y_true: np.ndarray,
    *,
    max_depth: int = 4,
    num_bits: int = 3,
    min_leaf_size: int = 20,
    seed: int = 42,
) -> GalleryResult:
    """Run DYF parameter-free clustering end-to-end.

    The "three lines" story: build the tree, write the index, louvain the leaves.
    Tree-build hyperparameters are intentionally set to DYF's documented defaults
    — they are the same in every gallery notebook and are not tuned per dataset.
    """
    from sklearn.metrics import adjusted_mutual_info_score, adjusted_rand_score

    from dyf import build_dyf_tree, write_lazy_index, LazyIndex
    from dyf.agglomerate import louvain_cluster_leaves

    embeddings = np.ascontiguousarray(embeddings, dtype=np.float32)

    tree = build_dyf_tree(
        embeddings,
        max_depth=max_depth,
        num_bits=num_bits,
        min_leaf_size=min_leaf_size,
        seed=seed,
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
