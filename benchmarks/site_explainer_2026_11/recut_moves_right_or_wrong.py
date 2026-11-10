"""Of the points the re-cut moves, how many moves are right?  Read from the recorded page runs.

A point is "right" in a partition when its label equals the majority label of the group it is in
(majority over the labelled displayed points).  For each run / label layer / level:
  moved share; among moved points: wrong->right (fixed), right->wrong (broken), stayed right,
  stayed wrong; and the same split per true class, to see whether the damage is everywhere or
  concentrated in a few classes.
Data: ~/.cache/dyf_intent_bench/site/*_pipeline_run_seed42.json (12,000 displayed points each).
Env: .venv/bin/python benchmarks/site_explainer_2026_11/recut_moves_right_or_wrong.py
"""

import json
from pathlib import Path

import numpy as np

SITE = Path.home() / ".cache/dyf_intent_bench/site"
CASES = [("mnist_pca50", 0, 1.0), ("clinc150_minilm", 0, 1.0), ("clinc150_minilm", 1, 32.0)]


def right(groups, y):
    """per point: does its label equal its group's majority label?"""
    ok = np.zeros(len(y), dtype=bool)
    for g in np.unique(groups):
        m = groups == g
        ok[m] = y[m] == np.bincount(y[m]).argmax()
    return ok


for name, li, res in CASES:
    D = json.loads((SITE / f"{name}_pipeline_run_seed42.json").read_text())
    la = D["layers"][li]
    keep = ~np.array(D["unlabelled"], dtype=bool)
    y = np.array(la["labels"])[keep]
    leaf = np.array(D["cells"][-1]["labels"])[keep]
    lv = next(l for l in D["levels"] if l["res"] == res)
    block = np.array(lv["leaf_comm"])[leaf]
    final = np.array(lv["final"])[keep]
    moved = block != final
    rb, ra = right(block, y), right(final, y)
    print(
        f"\n== {name} vs {la['k']} {la['noun']} at resolution {res}: {len(y)} points, {lv['k']} groups, "
        f"ARI before {lv['ari_block'][li]:.3f} after {lv['ari'][li]:.3f}"
    )
    print(f"   right before {rb.mean():.1%}, right after {ra.mean():.1%}; moved {moved.mean():.1%}")
    n = moved.sum()
    print(
        f"   of moved: fixed {(~rb & ra & moved).sum() / n:.1%}, broken {(rb & ~ra & moved).sum() / n:.1%}, "
        f"right both {(rb & ra & moved).sum() / n:.1%}, wrong both {(~rb & ~ra & moved).sum() / n:.1%}"
    )
    print(
        f"   of unmoved: status changed anyway (group majority changed) {((rb != ra) & ~moved).sum() / (~moved).sum():.1%}"
    )
    rows = []
    for c in np.unique(y):
        m = y == c
        rows.append((la["names"][c], int(m.sum()), moved[m].mean(), rb[m].mean(), ra[m].mean()))
    rows.sort(key=lambda r: r[4] - r[3])
    show = rows if len(rows) <= 12 else rows[:6] + [("...", 0, 0, 0, 0)] + rows[-6:]
    print("   per class (sorted by change):  class, n, moved, right before -> after")
    for nm, k, mv, b, a in show:
        if nm == "...":
            print("     ...")
        else:
            print(f"     {nm:28s} {k:5d}  moved {mv:5.1%}  {b:5.1%} -> {a:5.1%}  ({a - b:+.1%})")
    d = np.array([r[4] - r[3] for r in rows])
    print(
        f"   classes helped {int((d > 0.02).sum())}, hurt {int((d < -0.02).sum())}, flat {int((np.abs(d) <= 0.02).sum())}"
    )
