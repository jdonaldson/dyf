"""Coverage decay under EVICTION: how long until a frozen basis must be reset?

Everything in POSTGRES_NOTES.md is append-only, and it says so: under append old points stay and
keep anchoring coverage, so the +64%-growth-is-free result is an OPTIMISTIC BOUND. Under eviction
coverage *decays* — the points that justified a hyperplane can leave. This probe measures that.

⚠️ DESIGN CONSTRAINT (the trap this script exists to avoid). Do NOT just slide a window over SEC
quarters. The temporal condition in `sec_shift.py` came in at +0.010 — SEC is near-stationary, so a
naive sliding window measures nothing and would report "eviction is free" for the wrong reason.
What discriminates is a **composition schedule**: the retained MIX changes over time, against a
fixed-mix control that has identical turnover. Section type is the axis known to break coverage
(withholding whole section types cost +0.119, ~8σ, vs +0.017 for disjoint companies).

So two arms, IDENTICAL eviction volume, differing only in the mix of what replaces it:

    FIXED   incoming mix == fit-time mix           (control: pure turnover, no composition change)
    DRIFT   incoming mix ramps toward one section  (turnover + composition change)

The contrast isolates composition decay from turnover per se. If DRIFT degrades and FIXED does not,
the reset trigger is about mix, not age — which is what determines the operational policy.

Tombstone semantics are what is modelled: evicted points leave the searchable set, the tree topology
and hyperplanes are untouched. Two centroid policies are measured, because it is the cheap lever:

    frozen_refresh   leaf centroids recomputed from LIVE members (a delta frame)
    frozen_stale     leaf centroids as fit (never updated)

Baseline for "if we reset right now" is a full rebuild on the current window, same search harness.

Run: /Users/jdonaldson/Projects/dyf/.direnv/python-3.12/bin/python sec_evict.py
"""

import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sec_seqlib as L  # noqa: E402

WINDOW = 60_000  # live window size, held CONSTANT (evict as many as we admit)
STEPS = 12  # 12 x 5,000 = 60,000 = one full turnover of the window
CHURN = 5_000  # points evicted (and admitted) per step
NQ = 400  # queries per measurement
K = 10
SEED = 42
OUT = os.path.join(L.CACHE, "evict_results.json")

SECTIONS = ["risk_factors", "mda", "forward_looking", "other"]
# DRIFT target: collapse toward 'mda'-heavy and starve 'risk_factors', the largest stratum.
# Ramped linearly from the fit-time mix to this over STEPS.
DRIFT_TARGET = {"risk_factors": 0.02, "mda": 0.80, "forward_looking": 0.10, "other": 0.08}


def root_cells(E, flat):
    """Occupancy over the 16 depth-1 cells — the refit trigger from POSTGRES_NOTES.md.

    Coarse wins on sampling noise: a depth-1 cell holds thousands of points, a leaf ~13.
    """
    H = flat[0]["hp"]
    if H is None:
        return np.zeros(1, dtype=np.int64)
    bid = ((E @ H.T > 0).astype(np.int64) << np.arange(H.shape[0])).sum(1)
    return np.bincount(bid, minlength=1 << H.shape[0])


def js_bits(p, q):
    """Jensen-Shannon divergence in BITS between two occupancy histograms."""
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    n = max(len(p), len(q))
    p = np.pad(p, (0, n - len(p)))
    q = np.pad(q, (0, n - len(q)))
    p = p / max(p.sum(), 1e-12)
    q = q / max(q.sum(), 1e-12)
    m = 0.5 * (p + q)

    def kl(a, b):
        mask = a > 0
        return float(np.sum(a[mask] * np.log2(a[mask] / np.maximum(b[mask], 1e-300))))

    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def measure(E, live, flat_frozen, fit_cells, C_stale, rng, probe):
    """Recall for frozen (refresh + stale centroids) and a fresh rebuild, on the SAME live set."""
    Esub = E[live]
    qi = rng.choice(len(live), size=min(NQ, len(live)), replace=False)
    Q = Esub[qi]
    truth = L.exact_knn(Q, Esub, k=K)

    # ---- frozen basis: route the live set through the ORIGINAL hyperplanes
    assign, unseen = L.route(Esub, flat_frozen)
    nl = L.n_leaves(flat_frozen)
    C_live, cnt = L.leaf_centroids(Esub, assign, nl)
    r_refresh = L.recall_at_k(L.ivf_search(Q, Esub, assign, C_live, probe, k=K), truth)
    r_stale = L.recall_at_k(L.ivf_search(Q, Esub, assign, C_stale, probe, k=K), truth)

    # ---- control: rebuild on the current window ("reset right now")
    flat_fresh = L.build(Esub)
    a_fresh = L.fresh_assign(flat_fresh, len(Esub))
    nlf = L.n_leaves(flat_fresh)
    C_fresh, _ = L.leaf_centroids(Esub, a_fresh, nlf)
    r_fresh = L.recall_at_k(L.ivf_search(Q, Esub, a_fresh, C_fresh, probe, k=K), truth)

    return {
        "r_frozen_refresh": r_refresh,
        "r_frozen_stale": r_stale,
        "r_fresh_rebuild": r_fresh,
        "gap_refresh": r_fresh - r_refresh,
        "gap_stale": r_fresh - r_stale,
        "unseen_rate": unseen / len(live),
        "js_depth1": js_bits(fit_cells, root_cells(Esub, flat_frozen)),
        "empty_leaf_frac": float((cnt == 0).mean()),
        "n_leaves_frozen": nl,
        "n_leaves_fresh": nlf,
    }


def run_arm(arm, E, S, order_pool, base, rng, probe):
    """One eviction schedule. `base` is the fit set; both arms share it and the frozen basis."""
    flat_frozen = L.build(E, base)
    # remap: flat_frozen's indices are positional within E[base]; routing only needs hyperplanes
    fit_cells = root_cells(E[base], flat_frozen)
    nl = L.n_leaves(flat_frozen)
    a_fit, _ = L.route(E[base], flat_frozen)
    C_stale, _ = L.leaf_centroids(E[base], a_fit, nl)

    fit_mix = {s: float((S[base] == s).mean()) for s in SECTIONS}
    live = list(base)
    pools = {s: list(order_pool[s]) for s in SECTIONS}
    rows = []

    r = measure(E, np.array(live), flat_frozen, fit_cells, C_stale, rng, probe)
    r.update({"arm": arm, "step": 0, "turnover": 0.0, "mix": fit_mix})
    rows.append(r)
    print(
        f"  [{arm}] step  0  turnover   0%  "
        f"frozen {r['r_frozen_refresh']:.4f}  fresh {r['r_fresh_rebuild']:.4f}  "
        f"gap {r['gap_refresh']:+.4f}  JS {r['js_depth1']:.4f}"
    )

    for step in range(1, STEPS + 1):
        # target mix for the incoming batch
        if arm == "fixed":
            mix = fit_mix
        else:
            f = step / STEPS
            mix = {s: (1 - f) * fit_mix[s] + f * DRIFT_TARGET[s] for s in SECTIONS}

        # admit CHURN points drawn to that mix
        admitted = []
        for s in SECTIONS:
            want = int(round(CHURN * mix[s]))
            take = pools[s][:want]
            pools[s] = pools[s][want:]
            admitted.extend(take)
        if len(admitted) < CHURN:  # top up from whatever pool still has points
            for s in SECTIONS:
                need = CHURN - len(admitted)
                if need <= 0:
                    break
                take = pools[s][:need]
                pools[s] = pools[s][need:]
                admitted.extend(take)

        # evict the OLDEST live points (window is date-sorted at load)
        live = live[len(admitted) :] + admitted
        cur = np.array(live)
        turn = min(1.0, step * CHURN / WINDOW)
        r = measure(E, cur, flat_frozen, fit_cells, C_stale, rng, probe)
        r.update(
            {"arm": arm, "step": step, "turnover": turn, "mix": {s: float((S[cur] == s).mean()) for s in SECTIONS}}
        )
        rows.append(r)
        print(
            f"  [{arm}] step {step:2d}  turnover {turn:4.0%}  "
            f"frozen {r['r_frozen_refresh']:.4f}  fresh {r['r_fresh_rebuild']:.4f}  "
            f"gap {r['gap_refresh']:+.4f}  JS {r['js_depth1']:.4f}  "
            f"unseen {r['unseen_rate']:.4f}"
        )
    return rows


def main():
    t0 = time.time()
    E, _D, T, S, _Q = L.load()
    print(f"corpus {len(E):,} x {E.shape[1]}  ({len(set(T))} tickers, {len(set(S))} sections)")

    # Fit set: STRATIFIED across the whole space, never a contiguous recent window
    # (that is the practical rule from the coverage findings). Take the oldest WINDOW points
    # proportionally per section so the fit mix is the corpus mix.
    idx_by_sec = {s: np.where(s == S)[0] for s in SECTIONS}
    base = []
    for s in SECTIONS:
        want = int(round(WINDOW * len(idx_by_sec[s]) / len(E)))
        base.extend(idx_by_sec[s][:want].tolist())
    base = np.array(sorted(base))
    # remaining points per section, oldest-first, are the admission pools
    used = set(base.tolist())
    pool = {s: [i for i in idx_by_sec[s].tolist() if i not in used] for s in SECTIONS}
    print(f"fit set {len(base):,}  |  pools: " + ", ".join(f"{s} {len(pool[s]):,}" for s in SECTIONS))

    # ⚠️ Probe must leave HEADROOM. At ~6% of leaves recall pins at 0.97-0.99 for both frozen and
    # fresh, so the gap is ~0 by saturation and the run would report "eviction is free" without
    # having been able to detect otherwise. Pick the probe from a sweep on the fit set: the
    # smallest probe whose rebuild recall is comfortably under ceiling.
    nl0 = L.n_leaves(L.build(E, base))
    rng0 = np.random.default_rng(SEED)
    Eb = E[base]
    qi = rng0.choice(len(base), size=min(NQ, len(base)), replace=False)
    Qb, truth_b = Eb[qi], L.exact_knn(Eb[qi], Eb, k=K)
    flat_b = L.build(E, base)
    ab = L.fresh_assign(flat_b, len(base))
    Cb, _ = L.leaf_centroids(Eb, ab, L.n_leaves(flat_b))
    probe = None
    print(f"  probe sweep ({nl0} leaves):", end=" ")
    for frac in (256, 128, 64, 32, 16, 8):
        p = max(1, nl0 // frac)
        r = L.recall_at_k(L.ivf_search(Qb, Eb, ab, Cb, p, k=K), truth_b)
        print(f"{p}:{r:.3f}", end="  ")
        if probe is None and r < 0.90:
            probe = p
    print()
    probe = probe or max(1, nl0 // 64)
    print(f"probe = {probe} leaves (headroom for degradation to be visible)\n")

    rows = []
    for arm in ("fixed", "drift"):
        print(f"=== ARM: {arm} ===")
        rows += run_arm(arm, E, S, pool, base, np.random.default_rng(SEED), probe)
        print()

    # ---- verdict: where does each arm cross a recall-gap tolerance?
    print("=" * 92)
    print("=== WHEN TO RESET ===\n")
    print(f"  {'arm':<8}{'turnover':>10}{'frozen':>10}{'fresh':>10}{'gap':>9}{'JS d1':>9}{'unseen':>9}")
    print("  " + "-" * 66)
    for r in rows:
        print(
            f"  {r['arm']:<8}{r['turnover']:>9.0%}{r['r_frozen_refresh']:>10.4f}"
            f"{r['r_fresh_rebuild']:>10.4f}{r['gap_refresh']:>+9.4f}"
            f"{r['js_depth1']:>9.4f}{r['unseen_rate']:>9.4f}"
        )

    print()
    for tol in (0.01, 0.02, 0.05):
        for arm in ("fixed", "drift"):
            hit = [r for r in rows if r["arm"] == arm and r["gap_refresh"] > tol]
            where = f"{hit[0]['turnover']:.0%} turnover (step {hit[0]['step']})" if hit else "NEVER"
            print(f"  gap > {tol:.2f}  [{arm:<5}] -> {where}")
    print()
    print("  centroid refresh value (gap_stale - gap_refresh, positive = refresh helps):")
    for arm in ("fixed", "drift"):
        a = [r for r in rows if r["arm"] == arm]
        print(
            f"    {arm:<6} mean {np.mean([r['gap_stale'] - r['gap_refresh'] for r in a]):+.4f}  "
            f"final {a[-1]['gap_stale'] - a[-1]['gap_refresh']:+.4f}"
        )

    with open(OUT, "w") as f:
        json.dump(rows, f, indent=2)
    print(f"\n  written: {OUT}   ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
