"""Step 1 of the agreed plan: what would a two-line, label-free shape summary say about each
index?  Composed from this week's results (no new computation):
  dimension_profile_test_results.json     dataset mean profile per rung, synthetic dictionary
  leaf_shape_families_results.json         per-leaf family shares, re-cut moved, recurrence
  profile_stability_*_results.json         cross-seed band shares, family agreement vs chance
Rules: nothing on line 1 or 2 uses ground-truth labels (purity is deliberately absent); the
shape words come from the profile itself (coarse dimension, direction of change); the
recurrence sentence is computed from within-family vs random cosine.
Env: /Users/jdonaldson/Projects/dyf/.venv/bin/python shape_summary_probe.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

OUT = Path("/Volumes/Models/dyf_bench_2026-11-04")
ORDER = ["mnist", "fashion_mnist", "cmu_mocap", "ag_news", "tms_facs", "tms_droplet", "brain_1m"]
LABEL = {
    "mnist": "MNIST PCA-50",
    "fashion_mnist": "Fashion-MNIST PCA-50",
    "cmu_mocap": "CMU MoCap joint angles",
    "ag_news": "AG News MiniLM",
    "tms_facs": "Tabula Muris FACS PCA-50",
    "tms_droplet": "Tabula Muris droplet PCA-50",
    "brain_1m": "1.3M mouse brain PCA-50",
}
FAMILY_WORD = {
    "ball": "structureless",
    "curve": "curve-like",
    "sheet": "sheet-like",
    "blobs": "pocket",
    "mixture": "mixed-scale",
}

prof = json.loads((OUT / "dimension_profile_test_results.json").read_text())
fams = json.loads((OUT / "leaf_shape_families_results.json").read_text())
stab = json.loads((OUT / "profile_stability_light_results.json").read_text())
stab_full = OUT / "profile_stability_test_results.json"
if stab_full.exists():
    for k, v in json.loads(stab_full.read_text()).items():
        stab.setdefault(k, v)


def shape_words(p):
    """Describe the profile from its own numbers: coarse dimension and how it changes."""
    coarse = p[0]
    fine = p[-1]
    peak_i = int(np.argmax(p))
    lo = (
        "about one dimension"
        if coarse < 1.6
        else (
            "about two dimensions"
            if coarse < 3
            else (f"about {coarse:.0f} dimensions" if coarse < 8 else "high-dimensional")
        )
    )
    if coarse >= 8:
        return f"{lo} at every scale (no low-dimensional structure the tree can use)"
    if fine > coarse * 1.5 and peak_i >= len(p) - 2:
        return (
            f"{lo} at coarse scales, rising to {fine:.0f} at the finest reliable scale (a manifold with a noise floor)"
        )
    if max(p) > coarse * 1.5 and peak_i < len(p) - 2:
        return f"{lo} at the coarsest split, peaking at {max(p):.0f} at rung {peak_i + 1} where groups overlap, then {fine:.0f} within them"
    if fine < coarse * 0.7:
        return f"{lo} at coarse scales, narrowing to {fine:.1f} at the finest reliable scale"
    return f"{lo} at every reliable scale"


def pct(x, fine=False):
    return f"{100 * x:.1f}%" if fine else f"{100 * x:.0f}%"


summaries = {}
for n in ORDER:
    p = [v for v in prof[n]["mean_profile"] if v is not None and np.isfinite(v)][:6]
    F = fams[n]["families"]
    shares = {f: v["points"] for f, v in F.items()}
    manifold = shares.get("curve", 0) + shares.get("sheet", 0)
    pockets = shares.get("blobs", 0)
    pocket_moved = F["blobs"]["moved"] if "blobs" in F else None
    localized = [f for f, v in F.items() if v["within_cos"] - v["random_cos"] > 0.1 and v["points"] > 0.002]
    S = stab.get(n)
    bands = S["bands"] if S else {}
    rep, noise = bands.get("reproducible", {}).get("share"), bands.get("noise", {}).get("share")
    agree, chance = (S["family_agree"], S["family_agree_chance"]) if S else (None, None)
    line1 = (
        f"{LABEL[n]}: {shape_words(p)}. {pct(manifold)} of points sit in curve- or sheet-like branches, "
        f"{pct(shares.get('mixture', 0))} in mixed-scale branches, {pct(shares.get('ball', 0))} in structureless branches"
        + (
            f", {pct(pockets, fine=True)} in small isolated pockets ({pct(pocket_moved)} of those reassigned by the final re-cut)."
            if pockets > 0.001
            else "."
        )
    )
    if S:
        line2 = (
            f"Another tree seed reproduces the branch shape for {pct(rep)} of points beyond the random null and changes it for {pct(noise)}; "
            f"the shape family agrees across three seeds for {pct(agree)} vs {pct(chance)} by chance. "
        )
    else:
        line2 = ""
    if localized:
        line2 += (
            "Shapes recur across the embedding except "
            + " and ".join(FAMILY_WORD[f] for f in localized)
            + " branches, which sit together."
        )
    else:
        line2 += "Every shape recurs across the whole embedding."
    summaries[n] = dict(
        profile=[round(v, 2) for v in p],
        shares={k: round(v, 3) for k, v in shares.items()},
        localized=localized,
        reproducible=rep,
        noise=noise,
        family_agree=agree,
        family_agree_chance=chance,
        line1=line1,
        line2=line2,
    )
    print(f"\n{line1}\n{line2}")

(OUT / "shape_summary_probe_results.json").write_text(json.dumps(summaries, indent=1))
print("\nwrote", OUT / "shape_summary_probe_results.json")
