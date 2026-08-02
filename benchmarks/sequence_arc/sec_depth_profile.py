"""Is the delta's DEPTH PROFILE more informative than its size?

The scalar unseen-bucket rate says how much drift; it cannot say what kind. Hypothesis
(graft NOTES.md:644 -- containment at multiple resolutions): a delta confined to fine
levels means "new specifics inside known territory"; a delta reaching coarse levels
means "genuinely new subject matter".

Falsifiable prediction from sec_shift.py: the harmless split (disjoint companies) should
be fine-resolution-only, the catastrophic one (disjoint sections) should reach coarse
levels. If both look alike at every depth, the multi-resolution story is decoration.

Metrics per depth d over the frozen base partition:
  JS         Jensen-Shannon divergence (bits) between base and stream occupancy
  unseen@d   share of unseen-bucket fallbacks that fired at depth d

random is the built-in control: base and stream are the same distribution, so every
metric should sit near 0 at every depth.

GOTCHA, measured and discarded: "fraction of stream mass landing in a cell that was
EMPTY in base" is structurally zero at every depth. The tree is BUILT from base points,
so every cell it has contains base members by construction. New territory cannot show up
as an empty cell -- it only shows up as an unseen-bucket fallback (a bucket the
hyperplanes permit but no base point occupied, hence no child node). Anyone reaching for
the obvious "new cells" metric will get zeros and mistake them for "no drift".

RESULT (2026-08-01): coarse depth is the better discriminator, by a wide margin.
JS at depth 1 (16 cells): random 0.000, temporal 0.005, ticker 0.004, section 0.260.
JS at depth 4 (10.6k cells): random 0.038, temporal 0.115, ticker 0.136, section 0.585.
Section/ticker ratio falls 60x -> 4.3x going depth 1 -> 4. Mechanism: coarse cells hold
~8700 points each so sampling noise is negligible; leaf cells hold ~13 so multinomial
noise floods the signal (the random control's own JS rises to 0.038 there).
"""

import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sec_seqlib as S  # noqa: E402

TARGET_BASE = 0.61
N_SEEDS = 3
CONDS = ["random", "temporal", "ticker", "section"]


def route_paths(E, flat, max_d):
    """Route, recording the ancestor node id at every depth plus where fallbacks fire.

    A point whose leaf is shallower than d keeps that leaf as its depth-d address, so
    every depth remains a genuine partition of all points.
    """
    addr = np.full((len(E), max_d + 1), -1, np.int64)
    unseen_depth = []
    stack = [(0, np.arange(len(E), dtype=np.int64), 0)]
    while stack:
        nid, idxs, d = stack.pop()
        if not len(idxs):
            continue
        node = flat[nid]
        addr[idxs[:, None], np.arange(d, max_d + 1)[None, :]] = nid
        if node["leaf_id"] >= 0:
            continue
        H, bmap, kids = node["hp"], node["bmap"], node["children"]
        nd = min(d + 1, max_d)
        if H is None or bmap is None:
            stack.append((kids[0], idxs, nd))
            continue
        bid = ((E[idxs] @ H.T > 0).astype(np.int64) << np.arange(H.shape[0])).sum(1)
        lut = np.full(1 << H.shape[0], -1, dtype=np.int64)
        for b, c in bmap.items():
            lut[int(b)] = int(c)
        child_of = lut[bid]
        miss = child_of < 0
        if miss.any():
            unseen_depth.extend([d] * int(miss.sum()))
            kc = np.stack([flat[k]["centroid"] for k in kids])
            child_of[miss] = (E[idxs[miss]] @ kc.T).argmax(1)
        for ci, k in enumerate(kids):
            sel = child_of == ci
            if sel.any():
                stack.append((k, idxs[sel], nd))
    return addr, np.array(unseen_depth, dtype=np.int64)


def kl_bits(x, y):
    m = x > 0
    return float(np.sum(x[m] * np.log2(x[m] / y[m])))


def js_bits(a, b):
    """Jensen-Shannon divergence in bits between two count vectors."""
    p = a / max(a.sum(), 1.0)
    q = b / max(b.sum(), 1.0)
    m = 0.5 * (p + q)
    return 0.5 * kl_bits(p, m) + 0.5 * kl_bits(q, m)


def conditions(rng, T, SEC, Q, N):
    def by_group(labels):
        groups = list(dict.fromkeys(labels.tolist()))
        rng.shuffle(groups)
        counts = {g: int((labels == g).sum()) for g in groups}
        chosen, tot = set(), 0
        for g in groups:
            if tot / N >= TARGET_BASE:
                break
            chosen.add(g)
            tot += counts[g]
        return np.isin(labels, list(chosen))

    return {
        "random": rng.random(N) < TARGET_BASE,
        "temporal": Q <= "2023Q4",
        "ticker": by_group(T),
        "section": np.isin(SEC, ["risk_factors", "forward_looking"]),
    }


def run_once(seed, E, T, SEC, Q):
    rng = np.random.default_rng(seed)
    N = len(E)
    out = {}
    for name, mask in conditions(rng, T, SEC, Q, N).items():
        base_idx, stream_idx = np.where(mask)[0], np.where(~mask)[0]
        flat = S.build(E, base_idx)
        max_d = 0
        stack = [(0, 0)]
        while stack:
            nid, d = stack.pop()
            max_d = max(max_d, d)
            for c in flat[nid]["children"]:
                stack.append((c, d + 1))

        addr, useen_d = route_paths(E, flat, max_d)
        _, us_stream = route_paths(E[stream_idx], flat, max_d)

        prof = []
        for d in range(max_d + 1):
            cells, inv = np.unique(addr[:, d], return_inverse=True)
            bc = np.bincount(inv[base_idx], minlength=len(cells)).astype(float)
            sc = np.bincount(inv[stream_idx], minlength=len(cells)).astype(float)
            prof.append(
                {
                    "depth": d,
                    "n_cells": int(len(cells)),
                    "pts_per_cell": float(len(base_idx) / len(cells)),
                    "js": js_bits(bc, sc),
                }
            )
        ud = np.bincount(us_stream, minlength=max_d + 1).astype(float)
        for d in range(max_d + 1):
            prof[d]["unseen_share"] = float(ud[d] / max(ud.sum(), 1))
            prof[d]["unseen_rate"] = float(ud[d] / len(stream_idx))
        out[name] = prof
    return out


def main():
    E, D, T, SEC, Q = S.load()
    runs = []
    for s in range(N_SEEDS):
        t0 = time.time()
        runs.append(run_once(s, E, T, SEC, Q))
        print(f"seed {s} done [{time.time() - t0:.0f}s]", flush=True)
    with open(os.path.join(S.CACHE, "depth_profile.json"), "w") as f:
        json.dump(runs, f, indent=2)

    depths = range(len(runs[0]["random"]))

    def m(cond, d, key):
        return float(np.mean([r[cond][d][key] for r in runs]))

    for key, label in [
        ("js", "JS DIVERGENCE base-vs-stream occupancy (bits)"),
        ("unseen_share", "WHERE unseen-bucket fallbacks fire (share by depth)"),
    ]:
        print(f"\n=== {label} ===")
        print(f"{'depth':>6}{'cells':>9}{'pts/cell':>10}" + "".join(f"{c:>12}" for c in CONDS))
        for d in depths:
            cells = int(np.mean([r["random"][d]["n_cells"] for r in runs]))
            ppc = np.mean([r["random"][d]["pts_per_cell"] for r in runs])
            print(f"{d:>6}{cells:>9}{ppc:>10.0f}" + "".join(f"{m(c, d, key):>12.4f}" for c in CONDS))

    print("\n=== SEPARATION: JS(condition) / JS(random control) at each depth ===")
    print("How many times the noise floor each condition sits at. Higher = cleaner detector.")
    print(f"{'depth':>6}" + "".join(f"{c:>12}" for c in CONDS[1:]))
    for d in depths:
        base = m("random", d, "js")
        cells = [f"{m(c, d, 'js') / base:>11.1f}x" if base > 1e-9 else f"{'inf':>12}" for c in CONDS[1:]]
        print(f"{d:>6}" + "".join(cells))

    print("\nbreaking-vs-harmless margin (section JS / worst harmless JS):")
    for d in depths:
        worst = max(m(c, d, "js") for c in ["temporal", "ticker"])
        if worst > 1e-9:
            print(f"  depth {d}: {m('section', d, 'js') / worst:>5.1f}x")


if __name__ == "__main__":
    main()
