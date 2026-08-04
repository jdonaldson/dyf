"""Anatomy of the depth-1 JS trigger: noise floor, per-cell structure, depth profile, scale.

POSTGRES_NOTES.md establishes JS at depth 1 as the refit trigger (healthy <=0.005, collapse 0.26,
57.6x margin) and `sec_evict.py` reaches 0.0547 under composition drift. But a threshold is only
meaningful against a NOISE FLOOR, and that floor was never measured for the eviction condition.
Everything here is measured on the same corpus so the numbers are comparable.

Four questions:
  1. NOISE FLOOR  — JS between two random halves of the SAME distribution, at this window size.
     Anything below this is unreadable, so it sets the smallest usable threshold.
  2. SCALE        — base-2 JS is bounded in [0, 1] bits. Where does 0.0547 sit on that range?
  3. PER-CELL     — is JS driven by broad reshaping or by two or three cells? Decides whether a
     cheap per-cell alarm would do the same job.
  4. DEPTH        — recompute the depth-1-vs-deeper preference under EVICTION rather than under
     the withheld-section-type condition the original margin came from.

Run: /Users/jdonaldson/Projects/dyf/.direnv/python-3.12/bin/python sec_js_anatomy.py
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sec_evict as EV  # noqa: E402
import sec_seqlib as L  # noqa: E402

SECTIONS = EV.SECTIONS


def cells_at_depth(E, flat, depth):
    """Occupancy histogram over the cells reachable at `depth` levels of hyperplane splits.

    depth=1 is the root's 16 buckets. Deeper levels concatenate each node's own bucket space,
    so the histogram grows and per-cell counts shrink — which is the sampling-noise mechanism.
    """
    counts = []

    def rec(nid, idxs, d):
        node = flat[nid]
        if d == 0 or node["leaf_id"] >= 0 or node["hp"] is None:
            counts.append(len(idxs))
            return
        H, bmap, kids = node["hp"], node["bmap"], node["children"]
        proj = E[idxs] @ H.T
        bid = ((proj > 0).astype(np.int64) << np.arange(H.shape[0])).sum(1)
        lut = np.full(1 << H.shape[0], -1, dtype=np.int64)
        for b, c in bmap.items():
            lut[int(b)] = int(c)
        child_of = lut[bid]
        miss = child_of < 0
        if miss.any():
            kc = np.stack([flat[k]["centroid"] for k in kids])
            child_of[miss] = (E[idxs[miss]] @ kc.T).argmax(1)
        for ci, k in enumerate(kids):
            sel = child_of == ci
            rec(k, idxs[sel], d - 1)

    rec(0, np.arange(len(E), dtype=np.int64), depth)
    return np.array(counts, dtype=np.int64)


def main():
    E, _D, _T, S, _Q = L.load()
    idx_by_sec = {s: np.where(s == S)[0] for s in SECTIONS}

    # same fit set as sec_evict.py
    base = []
    for s in SECTIONS:
        want = int(round(EV.WINDOW * len(idx_by_sec[s]) / len(E)))
        base.extend(idx_by_sec[s][:want].tolist())
    base = np.array(sorted(base))
    used = set(base.tolist())
    pool = {s: [i for i in idx_by_sec[s].tolist() if i not in used] for s in SECTIONS}

    flat = L.build(E, base)
    fit_cells = EV.root_cells(E[base], flat)
    nz = int((fit_cells > 0).sum())
    print(f"fit set {len(base):,}  |  depth-1 cells {len(fit_cells)} ({nz} occupied)")
    print(
        f"  per-cell occupancy: min {fit_cells[fit_cells > 0].min():,}  "
        f"median {int(np.median(fit_cells[fit_cells > 0])):,}  max {fit_cells.max():,}"
    )

    # ── 1. NOISE FLOOR ───────────────────────────────────────────────────────────────────────
    # Two disjoint random halves of the fit set. Same distribution by construction, so any JS
    # here is pure sampling noise at this window size.
    print(f"\n{'=' * 88}\n=== 1. NOISE FLOOR — JS between random halves of the SAME distribution ===")
    floors = []
    for seed in range(8):
        rng = np.random.default_rng(seed)
        perm = rng.permutation(len(base))
        a, b = base[perm[: len(base) // 2]], base[perm[len(base) // 2 :]]
        floors.append(EV.js_bits(EV.root_cells(E[a], flat), EV.root_cells(E[b], flat)))
    floors = np.array(floors)
    print(f"   depth-1, n={len(base) // 2:,} per half, 8 seeds:")
    print(f"   mean {floors.mean():.6f}   max {floors.max():.6f}   sd {floors.std():.6f}")

    # half-size sensitivity: noise scales ~1/n, so a smaller live window has a higher floor
    print("\n   floor vs sample size (mean of 5 seeds):")
    for n in (2_000, 5_000, 15_000, 30_000):
        f = []
        for seed in range(5):
            rng = np.random.default_rng(100 + seed)
            pick = rng.choice(base, size=min(2 * n, len(base)), replace=False)
            f.append(EV.js_bits(EV.root_cells(E[pick[:n]], flat), EV.root_cells(E[pick[n : 2 * n]], flat)))
        print(f"     n={n:>6,}  floor {np.mean(f):.6f}")

    # ── 2. SCALE ─────────────────────────────────────────────────────────────────────────────
    print(f"\n{'=' * 88}\n=== 2. SCALE — base-2 JS is bounded [0,1] bits ===")
    worst = np.zeros_like(fit_cells)
    worst[int(np.argmin(np.where(fit_cells > 0, fit_cells, 1 << 30)))] = fit_cells.sum()
    print("   theoretical max (disjoint support)      1.000000")
    print(f"   all mass into the SMALLEST live cell    {EV.js_bits(fit_cells, worst):.6f}")
    print("   observed: fixed arm @100% turnover      0.003000")
    print(f"   observed: drift arm @100% turnover      0.054700   ({0.0547 / 1.0:.1%} of the bound)")
    print("   arc's collapse condition                0.260000")

    # ── 3/4. replay the drift schedule to get the final live window ─────────────────────────
    print(f"\n{'=' * 88}\n=== 3. PER-CELL STRUCTURE of the drift endpoint ===")
    fit_mix = {s: float((S[base] == s).mean()) for s in SECTIONS}
    live = list(base)
    pools = {s: list(pool[s]) for s in SECTIONS}
    for step in range(1, EV.STEPS + 1):
        f = step / EV.STEPS
        mix = {s: (1 - f) * fit_mix[s] + f * EV.DRIFT_TARGET[s] for s in SECTIONS}
        admitted = []
        for s in SECTIONS:
            want = int(round(EV.CHURN * mix[s]))
            admitted.extend(pools[s][:want])
            pools[s] = pools[s][want:]
        if len(admitted) < EV.CHURN:
            for s in SECTIONS:
                need = EV.CHURN - len(admitted)
                if need <= 0:
                    break
                admitted.extend(pools[s][:need])
                pools[s] = pools[s][need:]
        live = live[len(admitted) :] + admitted
    cur = np.array(live)
    cur_cells = EV.root_cells(E[cur], flat)
    js_total = EV.js_bits(fit_cells, cur_cells)
    print(f"   live {len(cur):,}   JS(fit, live) = {js_total:.6f}")

    # per-cell contribution to the JS sum
    pf = fit_cells / fit_cells.sum()
    pc = cur_cells / cur_cells.sum()
    m = 0.5 * (pf + pc)
    with np.errstate(divide="ignore", invalid="ignore"):
        contrib = 0.5 * np.where(pf > 0, pf * np.log2(pf / m), 0) + 0.5 * np.where(pc > 0, pc * np.log2(pc / m), 0)
    order = np.argsort(-np.abs(contrib))
    print(f"\n   {'cell':>5}{'fit%':>9}{'live%':>9}{'delta':>9}{'contrib':>10}{'cum%':>8}")
    cum = 0.0
    for c in order[:8]:
        cum += contrib[c]
        print(f"   {c:>5}{pf[c]:>9.4f}{pc[c]:>9.4f}{pc[c] - pf[c]:>+9.4f}{contrib[c]:>10.5f}{cum / js_total:>8.1%}")
    top3 = float(np.abs(contrib[order[:3]]).sum() / np.abs(contrib).sum())
    print(
        f"\n   top-3 cells carry {top3:.1%} of total |contribution|  "
        f"-> {'CONCENTRATED' if top3 > 0.7 else 'BROAD reshaping'}"
    )

    # ── 4. DEPTH PROFILE under eviction ─────────────────────────────────────────────────────
    print(f"\n{'=' * 88}\n=== 4. DEPTH PROFILE under eviction (margin = signal / noise) ===")
    print(f"   {'depth':>6}{'cells':>8}{'median n':>10}{'JS drift':>11}{'JS noise':>11}{'margin':>9}")
    for depth in (1, 2, 3, 4):
        fit_d = cells_at_depth(E[base], flat, depth)
        cur_d = cells_at_depth(E[cur], flat, depth)
        sig = EV.js_bits(fit_d, cur_d)
        nf = []
        for seed in range(4):
            rng = np.random.default_rng(200 + seed)
            perm = rng.permutation(len(base))
            a, b = base[perm[: len(base) // 2]], base[perm[len(base) // 2 :]]
            nf.append(EV.js_bits(cells_at_depth(E[a], flat, depth), cells_at_depth(E[b], flat, depth)))
        noise = float(np.mean(nf))
        occ = fit_d[fit_d > 0]
        print(
            f"   {depth:>6}{len(fit_d):>8}{int(np.median(occ)):>10,}{sig:>11.6f}"
            f"{noise:>11.6f}{sig / max(noise, 1e-9):>8.1f}x"
        )
    print("\n   ^ coarse wins because the noise floor collapses with per-cell occupancy,")
    print("     not because the signal is stronger there.")


if __name__ == "__main__":
    main()
