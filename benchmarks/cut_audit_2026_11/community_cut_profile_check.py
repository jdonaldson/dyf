"""Ground the cut-profile typology: (1) label composition of the pairs that contradict the
valley reading, (2) counts of each geometric type over the 70 scored pairs."""

import json
from pathlib import Path

import numpy as np

BENCH = Path("/Volumes/Models/dyf_bench_2026-09-27")
OUT = Path("/Volumes/Models/dyf_bench_2026-11-04")
CHECK = {
    "cmu_mocap": [(3, 4), (1, 4), (16, 1)],
    "fashion_mnist": [(8, 11), (3, 9)],
    "brain_1m": [(21, 0), (8, 0), (20, 14)],
    "tms_facs": [(19, 6), (3, 8)],
    "tms_droplet": [(26, 5), (14, 4)],
}
NAMES = {
    "fashion_mnist": ["tshirt", "trouser", "pullover", "dress", "coat", "sandal", "shirt", "sneaker", "bag", "boot"]
}


def comp(y, lab, c, raw, k=3):
    m = lab == c
    vals, cnt = np.unique(y[m], return_counts=True)
    o = np.argsort(-cnt)[:k]
    return ", ".join(f"{raw[vals[i]]}:{cnt[i] / m.sum():.0%}" for i in o)


for name, pairs in CHECK.items():
    meta = json.loads((BENCH / "datasets.json").read_text())[name]
    yraw = np.load(meta.get("y_path", BENCH / f"{name}_y.npy"), allow_pickle=True)
    raw, y = np.unique(yraw, return_inverse=True)
    if name in NAMES:
        raw = np.array(NAMES[name])
    lab = np.load(BENCH / f"sC_labels_{name}_default_3_4_20.npy")
    _, lab = np.unique(lab, return_inverse=True)
    print(f"\n== {name}")
    for c, d in pairs:
        print(f"   comm {c:3d} ({(lab == c).mean():5.1%}): {comp(y, lab, c, raw)}")
        print(f"   comm {d:3d} ({(lab == d).mean():5.1%}): {comp(y, lab, d, raw)}")

res = json.loads((OUT / "community_cut_profile_results.json").read_text())
rows = [(n, p) for n, r in res.items() if n != "pooled_auc" for p in r["pairs"]]
print(f"\n== typology over {len(rows)} ordered pairs")
no_own_mode = [(n, p) for n, p in rows if p["c_side"] >= 0.95]
valley = [(n, p) for n, p in rows if p["valley"] < 0.3]
saddle = [(n, p) for n, p in rows if 0.3 <= p["valley"] < 0.95]


def tag(n, p, arrow):
    return f"{n}:{p['c']}{arrow}{p['d']}"


print(
    f"   c has no own mode (c_side >= 0.95): {len(no_own_mode)}  [{', '.join(tag(n, p, '->') for n, p in no_own_mode)}]"
)
seen = {(n, p["c"], p["d"]) for n, p in rows}
mutual = [
    (n, p)
    for n, p in no_own_mode
    if (n, p["d"], p["c"]) in seen
    and next(q for m, q in rows if m == n and q["c"] == p["d"] and q["d"] == p["c"])["c_side"] >= 0.95
]
print(
    f"   ... of which mutual (both sides modeless = one mass cut at its mode): {len(mutual)} [{', '.join(tag(n, p, '<->') for n, p in mutual)}]"
)
print(f"   deep valley (<0.3): {len(valley)}; slices by label among them: {sum(p['slice'] for _, p in valley)}")
print(f"   saddle (0.3-0.95): {len(saddle)}; slices by label among them: {sum(p['slice'] for _, p in saddle)}")
