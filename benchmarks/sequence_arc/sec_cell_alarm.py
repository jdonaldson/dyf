"""Per-cell delta alarm vs JS: is the cheap legible signal actually sufficient?

`sec_js_anatomy.py` found the drift signal CONCENTRATED — top 3 of 16 depth-1 cells carry 75.8% of
JS. That raises a falsification of the JS recommendation: if a max-per-cell-delta alarm separates
the arms just as well, JS is doing no extra work as a TRIGGER and its value is diagnostic only.

Candidate detectors, all computed from the same 16 ingest counters:

  max_abs_delta   max_c |live_c - fit_c|            (share points; scale-free across cells)
  max_rel_delta   max_c |live_c - fit_c| / fit_c    (relative; sensitive on small cells)
  tv              0.5 * sum_c |live_c - fit_c|      (total variation = L1/2)
  js              Jensen-Shannon in bits

Judged on DISCRIMINATION, not raw magnitude: signal-to-noise, where noise is the same-distribution
split-half floor for that detector. A detector with a big number and a big floor is worse than a
small number with a tiny floor -- the mistake this study keeps re-learning (dirty-leaf fraction,
spread-vs-independence). Both arms have IDENTICAL turnover, so any arm separation is composition.

⚠️ The per-cell threshold problem is the real question. Depth-1 occupancy is wildly uneven
(min 106, median 1,509, max 19,614 at fit time), so ONE absolute threshold is dominated by the
big cells and one relative threshold is dominated by the small ones. That is measured here too.

Run: /Users/jdonaldson/Projects/dyf/.direnv/python-3.12/bin/python sec_cell_alarm.py
"""

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sec_evict as EV  # noqa: E402
import sec_seqlib as L  # noqa: E402

SECTIONS = EV.SECTIONS


def detectors(fit_c, live_c):
    """All four detectors from a pair of occupancy histograms."""
    pf = fit_c / max(fit_c.sum(), 1)
    pl = live_c / max(live_c.sum(), 1)
    d = np.abs(pl - pf)
    with np.errstate(divide="ignore", invalid="ignore"):
        rel = np.where(pf > 0, d / pf, 0.0)
    return {
        "max_abs_delta": float(d.max()),
        "max_rel_delta": float(rel.max()),
        "tv": float(0.5 * d.sum()),
        "js": EV.js_bits(fit_c, live_c),
    }


def replay(E, S, base, pool, arm, fit_mix):
    """Yield the live window at each step of an eviction schedule (same logic as sec_evict)."""
    live = list(base)
    pools = {s: list(pool[s]) for s in SECTIONS}
    yield 0, np.array(live)
    for step in range(1, EV.STEPS + 1):
        if arm == "fixed":
            mix = fit_mix
        else:
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
        yield step, np.array(live)


def main():
    E, _D, _T, S, _Q = L.load()
    idx_by_sec = {s: np.where(s == S)[0] for s in SECTIONS}
    base = []
    for s in SECTIONS:
        want = int(round(EV.WINDOW * len(idx_by_sec[s]) / len(E)))
        base.extend(idx_by_sec[s][:want].tolist())
    base = np.array(sorted(base))
    used = set(base.tolist())
    pool = {s: [i for i in idx_by_sec[s].tolist() if i not in used] for s in SECTIONS}

    flat = L.build(E, base)
    fit_c = EV.root_cells(E[base], flat)
    fit_mix = {s: float((S[base] == s).mean()) for s in SECTIONS}
    names = ["max_abs_delta", "max_rel_delta", "tv", "js"]

    # ── noise floors: split-half of the SAME distribution, per detector ──────────────────────
    print("=== NOISE FLOORS (split-half, same distribution, 8 seeds) ===")
    floor_samples = {n: [] for n in names}
    for seed in range(8):
        rng = np.random.default_rng(seed)
        perm = rng.permutation(len(base))
        a, b = base[perm[: len(base) // 2]], base[perm[len(base) // 2 :]]
        d = detectors(EV.root_cells(E[a], flat), EV.root_cells(E[b], flat))
        for n in names:
            floor_samples[n].append(d[n])
    floors = {n: float(np.mean(floor_samples[n])) for n in names}
    fmax = {n: float(np.max(floor_samples[n])) for n in names}
    for n in names:
        print(f"   {n:<16} mean {floors[n]:.6f}   max {fmax[n]:.6f}")

    # ── run both arms ───────────────────────────────────────────────────────────────────────
    res = {}
    for arm in ("fixed", "drift"):
        rows = []
        for step, cur in replay(E, S, base, pool, arm, fit_mix):
            d = detectors(fit_c, EV.root_cells(E[cur], flat))
            d.update({"step": step, "turnover": min(1.0, step * EV.CHURN / EV.WINDOW)})
            rows.append(d)
        res[arm] = rows

    print(f"\n{'=' * 100}")
    print("=== PER-STEP (fixed = pure turnover control, drift = turnover + composition) ===\n")
    print(f"   {'turn':>5} | " + " | ".join(f"{n.replace('_delta', ''):>21}" for n in names))
    print(f"   {'':>5} | " + " | ".join(f"{'fixed':>9}{'drift':>7}{'sep':>5}" for _ in names))
    print("   " + "-" * 96)
    for i in range(len(res["fixed"])):
        f, d = res["fixed"][i], res["drift"][i]
        cells = []
        for n in names:
            sep = d[n] / max(f[n], 1e-9)
            cells.append(f"{f[n]:>9.4f}{d[n]:>7.4f}{sep:>4.0f}x")
        print(f"   {f['turnover']:>4.0%} | " + " | ".join(cells))

    # ── verdict: signal-to-noise at the endpoint, and arm separation ─────────────────────────
    print(f"\n{'=' * 100}")
    print("=== DISCRIMINATION AT FULL TURNOVER ===\n")
    print(f"   {'detector':<16}{'floor':>10}{'fixed':>10}{'drift':>10}{'drift/floor':>13}{'drift/fixed':>13}")
    print("   " + "-" * 72)
    ranked = []
    for n in names:
        f, d = res["fixed"][-1][n], res["drift"][-1][n]
        snr = d / max(floors[n], 1e-9)
        sep = d / max(f, 1e-9)
        ranked.append((n, snr, sep))
        print(f"   {n:<16}{floors[n]:>10.6f}{f:>10.4f}{d:>10.4f}{snr:>12.0f}x{sep:>12.1f}x")

    print("\n   ranked by drift/fixed separation (the composition-specific quantity):")
    for n, snr, sep in sorted(ranked, key=lambda x: -x[2]):
        print(f"     {n:<16} {sep:>7.1f}x   (snr vs floor {snr:>6.0f}x)")

    # ── the per-cell threshold problem ──────────────────────────────────────────────────────
    print(f"\n{'=' * 100}")
    print("=== THE PER-CELL THRESHOLD PROBLEM ===\n")
    pf = fit_c / fit_c.sum()
    cur = list(replay(E, S, base, pool, "drift", fit_mix))[-1][1]
    pl = EV.root_cells(E[cur], flat) / len(cur)
    curF = list(replay(E, S, base, pool, "fixed", fit_mix))[-1][1]
    plF = EV.root_cells(E[curF], flat) / len(curF)
    print(
        f"   fit-time occupancy spans {fit_c.min():,} to {fit_c.max():,} points "
        f"({fit_c.max() / max(fit_c.min(), 1):.0f}x)"
    )
    print(f"\n   {'cell':>5}{'fit n':>9}{'fit%':>8}{'abs d':>9}{'rel d':>9}{'| fixed abs':>13}{'fixed rel':>11}")
    for c in np.argsort(-np.abs(pl - pf))[:6]:
        rel = abs(pl[c] - pf[c]) / max(pf[c], 1e-12)
        relF = abs(plF[c] - pf[c]) / max(pf[c], 1e-12)
        print(
            f"   {c:>5}{fit_c[c]:>9,}{pf[c]:>8.4f}{abs(pl[c] - pf[c]):>9.4f}{rel:>9.2f}"
            f"{abs(plF[c] - pf[c]):>13.4f}{relF:>11.2f}"
        )
    small = np.argsort(fit_c)[:4]
    print("\n   smallest cells (where a RELATIVE threshold misfires):")
    for c in small:
        rel = abs(pl[c] - pf[c]) / max(pf[c], 1e-12)
        relF = abs(plF[c] - pf[c]) / max(pf[c], 1e-12)
        print(
            f"   {c:>5}{fit_c[c]:>9,}{pf[c]:>8.4f}{abs(pl[c] - pf[c]):>9.4f}{rel:>9.2f}"
            f"{abs(plF[c] - pf[c]):>13.4f}{relF:>11.2f}"
        )
    print("\n   ^ if fixed-arm 'rel d' is comparable to drift-arm 'rel d' on small cells, a")
    print("     relative per-cell alarm FIRES ON THE HEALTHY CONTROL -- a false positive.")

    out = os.path.join(L.CACHE, "cell_alarm_results.json")
    with open(out, "w") as fh:
        json.dump({"floors": floors, "arms": res}, fh, indent=2)
    print(f"\n   written: {out}")


if __name__ == "__main__":
    main()
