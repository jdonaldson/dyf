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
