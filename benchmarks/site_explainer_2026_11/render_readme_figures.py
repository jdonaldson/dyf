"""Static figures for the README, drawn from the same recorded run the homepage replays
(record_pipeline_run.py --dataset cifar100).  Nothing here computes groups: it draws the recording.

  dyf-lens-cifar100-three-levels.png   one fit at three resolutions, beside the human labels
  dyf-agreement-by-resolution.png      ARI vs resolution for both label layers, with k-means told k
  dyf-pipeline-five-steps.png          embeddings -> first cut -> leaves -> leaf graph -> groups

Light, opaque ground so the images read on GitHub in either theme and on PyPI.
Env: MPLCONFIGDIR=$TMPDIR .venv/bin/python benchmarks/site_explainer_2026_11/render_readme_figures.py
Out: assets/readme/
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.collections import LineCollection  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
RUN = Path.home() / ".cache/dyf_intent_bench/site/cifar100_clip_pipeline_run_seed42.json"
OUT = REPO / "assets/readme"
HUES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
BG, SURFACE, INK, INK2, INK3, RULE, IDLE = "#f6f5f4", "#ffffff", "#1c1b1d", "#55525a", "#8a8690", "#e2dfe3", "#b9b5bd"
plt.rcParams.update(
    {
        "font.family": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
        "font.size": 11,
        "axes.edgecolor": RULE,
        "axes.linewidth": 0.8,
        "savefig.facecolor": BG,
        "figure.facecolor": BG,
    }
)

D = json.loads(RUN.read_text())
x, y = np.array(D["x"]), np.array(D["y"])
LV, layers, M = D["levels"], D["layers"], D["meta"]
level = {lv["res"]: lv for lv in LV}
num = lambda v: f"{v:,}"  # noqa: E731


def panel(ax, colour_index, title, sub=None, s=1.6, alpha=0.85):
    ax.set_facecolor(SURFACE)
    c = np.asarray(colour_index)
    cols = np.array(HUES + [IDLE])[np.where(c < 0, 8, c)]
    ax.scatter(x, y, c=cols, s=s, alpha=alpha, linewidths=0, rasterized=True)
    ax.set(xlim=(0, 4095), ylim=(0, 4095), xticks=[], yticks=[], aspect="equal")
    ax.set_title(title, loc="left", fontsize=12.5, fontweight="bold", color=INK, pad=8)
    if sub:
        ax.text(0, -0.035, sub, transform=ax.transAxes, ha="left", va="top", fontsize=10.5, color=INK2)


def group_colours(lv):
    return np.array(lv["colours"])[np.array(lv["final"])]


def three_levels():
    picks = [0.5, layers[0]["fifty"]["best_level"], layers[1]["fifty"]["best_level"]]
    picks = [picks[0], LV[picks[1]]["res"], LV[picks[2]]["res"]]
    fig, axes = plt.subplots(1, 4, figsize=(17.6, 5.1))
    for ax, r in zip(axes[:3], picks):
        lv = level[r]
        panel(
            ax,
            group_colours(lv),
            f"resolution {r:g}: {lv['k']} groups",
            f"agreement with the {layers[0]['k']} {layers[0]['noun']} {lv['ari'][0]:.2f}, the {layers[1]['k']} {layers[1]['noun']} {lv['ari'][1]:.2f}",
        )
    la = layers[0]
    panel(
        axes[3],
        np.array(la["colours"])[np.array(la["labels"])],
        f"human labels: {la['k']} {la['noun']}",
        "never shown to dyf",
    )
    fig.suptitle(
        f"One dyf fit of {num(M['n'])} CIFAR-100 photos (CLIP embeddings), read at three levels of detail",
        x=0.012,
        y=0.985,
        ha="left",
        fontsize=15,
        fontweight="bold",
        color=INK,
    )
    fig.text(
        0.012,
        0.006,
        f"{num(len(x))} points shown, 2-D UMAP layout for display only. Colours tell neighbouring groups apart and are reused. "
        "Changing the resolution does not refit the tree.",
        fontsize=10,
        color=INK3,
    )
    fig.subplots_adjust(left=0.012, right=0.988, top=0.86, bottom=0.14, wspace=0.05)
    return fig


def agreement():
    fig, ax = plt.subplots(figsize=(10.4, 4.9))
    ax.set_facecolor(SURFACE)
    idx = np.arange(len(LV))
    top = 0.4
    for li, la in enumerate(layers):
        v = [lv["ari"][li] for lv in LV]
        ax.axhline(la["ref"]["ari_mean"], color=HUES[li], lw=1, ls=(0, (5, 4)), zorder=1)
        ax.plot(idx, v, color=HUES[li], lw=2, marker="o", ms=5, mec=SURFACE, mew=1.2, zorder=3)
        b = la["fifty"]["best_level"]
        left = b < len(LV) / 2
        ax.annotate(
            f"dyf vs the {la['k']} {la['noun']}",
            (b, v[b]),
            xytext=(10 if left else -10, 9),
            textcoords="offset points",
            ha="left" if left else "right",
            color=INK,
            fontsize=11,
            fontweight="bold",
        )
        ax.text(
            len(LV) - 0.55 if left else -0.45,
            la["ref"]["ari_mean"] + 0.006,
            f"k-means told {la['k']}: {la['ref']['ari_mean']:.2f}",
            ha="right" if left else "left",
            va="bottom",
            color=INK2,
            fontsize=10,
        )
    d = next(i for i, lv in enumerate(LV) if lv["res"] == 1.0)
    ax.axvline(d, color=INK3, lw=0.8, zorder=0)
    ax.text(d + 0.12, top - 0.008, f"default: {LV[d]['k']} groups", color=INK2, fontsize=10, va="top")
    ax.set_xticks(idx, [f"{lv['res']:g}\n{lv['k']}" for lv in LV], fontsize=9.5, color=INK2)
    ax.set_yticks(np.arange(0, top + 1e-9, 0.1))
    ax.tick_params(axis="y", colors=INK2, length=0)
    ax.tick_params(axis="x", length=0)
    ax.set(xlim=(-0.6, len(LV) - 0.4), ylim=(0, top))
    ax.grid(axis="y", color=RULE, lw=0.8)
    ax.set_axisbelow(True)
    for sp in ("top", "right", "left"):
        ax.spines[sp].set_visible(False)
    ax.set_xlabel("resolution (top) and number of groups dyf returns (bottom)", color=INK2, fontsize=10.5, labelpad=8)
    ax.set_ylabel("agreement with labels (adjusted Rand index)", color=INK2, fontsize=10.5)
    ax.set_title(
        "The two label layers are best matched at different resolutions of the same fit",
        loc="left",
        fontsize=13.5,
        fontweight="bold",
        color=INK,
        pad=12,
    )
    fig.subplots_adjust(left=0.075, right=0.985, top=0.9, bottom=0.2)
    return fig


def pipeline():
    L0 = level[1.0]
    leaf = np.array(D["cells"][-1]["labels"])
    fig, axes = plt.subplots(1, 5, figsize=(17.6, 4.25))
    cp = M["cells_per_depth"]
    panel(axes[0], np.full(len(x), -1), "embeddings", f"{num(M['n'])} unit vectors, {M['d']} dimensions", alpha=0.7)
    panel(
        axes[1],
        np.array(D["cells"][0]["colours"])[np.array(D["cells"][0]["labels"])],
        f"first cut: {cp[0]} cells",
        "sign of the top 3 principal components",
    )
    panel(
        axes[2],
        np.array(D["cells"][-1]["colours"])[leaf],
        f"{num(M['leaves'])} leaves",
        f"cut again inside each cell, {M['depth']} levels",
    )
    panel(
        axes[3], np.full(len(x), -1), "leaf graph", f"each leaf linked to its {M['leaf_k']} nearest leaves", alpha=0.18
    )
    e = np.array(D["edges"]).reshape(-1, 2)
    lx, ly = np.array(D["leaf_x"]), np.array(D["leaf_y"])
    axes[3].add_collection(
        LineCollection(
            np.stack([np.stack([lx[e[:, 0]], ly[e[:, 0]]], 1), np.stack([lx[e[:, 1]], ly[e[:, 1]]], 1)], 1),
            colors=INK2,
            linewidths=0.25,
            alpha=0.3,
            rasterized=True,
        )
    )
    axes[3].scatter(
        lx,
        ly,
        s=np.sqrt(np.array(D["leaf_size"])) * 0.9,
        c="#cf2f4f",
        linewidths=0,
        alpha=0.9,
        zorder=3,
        rasterized=True,
    )
    panel(axes[4], group_colours(L0), f"{L0['k']} groups at resolution 1", "Louvain, then nearest-centre reassignment")
    fig.suptitle(
        "The pipeline on the same run. Everything up to the leaf graph happens once; changing the resolution repeats only the last panel.",
        x=0.012,
        y=0.985,
        ha="left",
        fontsize=14,
        fontweight="bold",
        color=INK,
    )
    fig.subplots_adjust(left=0.012, right=0.988, top=0.84, bottom=0.085, wspace=0.05)
    return fig


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    for name, fn in [
        ("dyf-lens-cifar100-three-levels", three_levels),
        ("dyf-agreement-by-resolution", agreement),
        ("dyf-pipeline-five-steps", pipeline),
    ]:
        fig = fn()
        path = OUT / f"{name}.png"
        fig.savefig(path, dpi=110)
        plt.close(fig)
        print(f"wrote {path} ({path.stat().st_size / 1e3:.0f} KB)")
