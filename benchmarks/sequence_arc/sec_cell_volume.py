"""Does per-cell VOLUME (PCA log-determinant) predict where a frozen partition fails?

Motivation: the frozen-basis arc established that the binding requirement is COVERAGE
(POSTGRES_NOTES.md), and that depth-1 JS divergence is the refit *trigger*. JS answers
"when to refit". This probe asks whether a query-free, base-only geometric quantity
answers the different question "WHERE is the frozen basis thin" -- i.e. which cells will
serve new content badly. If it works it is a sampling-target signal, complementary to JS.

The candidate is cheap: `Node.eigenvalues` is already in the .dyf schema
(src/dyf/schema/dyf_index.fbs), written from `clf.get_eigenvalues()` on that node's own
points (src/dyf/dyf_tree.py:113), and readable via LazyIndex.get_split_eigenvalues().
Ellipsoid log-volume = 0.5 * sum(log lambda_i).

MEASURED SEMANTICS of get_eigenvalues() -- re-run with `python sec_cell_volume.py --semantics`:
  - It tracks angular spread monotonically (sum_log_ev 2.55 -> 17.21 for cap spread
    0.02 -> 0.8) BUT SATURATES at the diffuse end (0.4 -> 0.8 moves it only 16.94 ->
    17.21 while mean_cos still halves). Unit-sphere data goes isotropic and the top-k
    eigenvalues hit a ceiling.
  - It is SCATTER-like, not covariance-like: sum(ev) grows with n (~n^0.86 measured).
    So log-ev is CONFOUNDED WITH OCCUPANCY and must be divided by n before use.
    A naive "logV - log n" sparsity score on raw stored ev is measuring n twice.
  - Only `num_bits` eigenvalues are stored (4 here) out of 768 -- a 4-D shadow of a
    768-D cell. Whether the shadow suffices is the main thing under test.

CRITICAL ABLATION (pre-flight #2): the competitor is not "nothing", it is the cheapest
byproduct of the existing fit. Two free baselines are included:
  neg_log_n  -- occupancy alone, no eigenvalues at all
  diffuse    -- -mean(cos to cell centroid); dyf ALREADY computes these
                (centroid_similarities -> point_margin_map, dyf_tree.py:104)
Volume only earns its keep if it beats BOTH. Every predictor is signed so that
HIGHER = MORE PREDICTED DAMAGE, and scored against the same target.

Target is the quantity that actually matters -- per-cell recall gap (fresh - frozen)
for stream-side queries routed into that cell -- not a proxy. unseen-rate is kept as a
secondary because POSTGRES_NOTES flags it as miscalibrated.

Null: cell counts are small (16 at depth 1), so every correlation is reported against a
permutation null on the same cell count. A rho without its null is unreadable here.
"""

import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sec_seqlib as S  # noqa: E402

NQ = 2000  # stream-side queries; ~125/cell at depth 1, ~8/cell at depth 2
K = 10
PROBE = 32
N_SEEDS = 5
TARGET_BASE = 0.61
DEPTHS = (1, 2)
TOPK_SPECTRUM = (4, 32)  # 4 = what .dyf actually stores; 32 = "if we stored more"
MIN_Q = {1: 20, 2: 5}  # min stream queries for a cell to enter the correlation
N_PERM = 500


# ---------------------------------------------------------------- tree with eigenvalues
def flatten_ev(tree, E):
    """S.flatten, plus per-node stored eigenvalues and depth-from-root.

    Preorder node ids, identical structure/keys to S.flatten so S.route works unchanged.
    """
    nodes = []

    def rec(t, d):
        nid = len(nodes)
        nodes.append(None)
        idxs = np.asarray(t["indices"], dtype=np.int64)
        cen = E[idxs].mean(0) if len(idxs) else np.zeros(E.shape[1], np.float32)
        cen = cen / (np.linalg.norm(cen) + 1e-12)
        kids = [rec(c, d + 1) for c in t["children"]] if t["children"] else []
        hp = t.get("hyperplanes")
        hp = np.asarray(hp, dtype=np.float32) if hp is not None and kids else None
        ev = t.get("eigenvalues")
        nodes[nid] = {
            "children": kids,
            "hp": hp,
            "bmap": t.get("bucket_id_to_child") if kids else None,
            "centroid": cen.astype(np.float32),
            "n": len(idxs),
            "indices": idxs,
            "leaf_id": -1,
            "ev": None if ev is None else np.asarray(ev, dtype=np.float64),
            "d": d,
        }
        return nid

    rec(tree, 0)
    lid = 0
    for n in nodes:
        if not n["children"]:
            n["leaf_id"] = lid
            lid += 1
    return nodes


def build_ev(E, idxs, seed):
    from dyf.dyf_tree import build_dyf_tree

    sub = E[idxs]
    tree = build_dyf_tree(sub, max_depth=S.MAX_DEPTH, num_bits=S.NUM_BITS, min_leaf_size=S.MIN_LEAF, seed=seed)
    return flatten_ev(tree, sub)


def cells_at_depth(flat, depth):
    """Return [(node_id, [leaf_ids under it])] for nodes at `depth`.

    A branch that bottoms out above `depth` contributes its own leaf as a cell, so the
    cells always partition the space (no points fall outside the cell set).
    """
    out = []

    def leaves_under(nid):
        n = flat[nid]
        if n["leaf_id"] >= 0:
            return [n["leaf_id"]]
        acc = []
        for k in n["children"]:
            acc.extend(leaves_under(k))
        return acc

    def walk(nid):
        n = flat[nid]
        if n["d"] == depth or n["leaf_id"] >= 0:
            out.append((nid, leaves_under(nid)))
            return
        for k in n["children"]:
            walk(k)

    walk(0)
    return out


# ---------------------------------------------------------------------------- predictors
def cell_predictors(Ebase_cell, ev_stored):
    """Base-only geometry for one cell. Higher = more predicted damage."""
    n = len(Ebase_cell)
    p: dict[str, float] = {"n": n, "neg_log_n": -np.log(max(n, 1))}

    cen = Ebase_cell.mean(0)
    cen = cen / (np.linalg.norm(cen) + 1e-12)
    p["diffuse"] = -float((Ebase_cell @ cen).mean())

    # true cell covariance spectrum (base points only -- no stream leakage)
    if n >= 8:
        Xc = Ebase_cell - Ebase_cell.mean(0)
        # gram trick when n < d: same nonzero spectrum, far cheaper
        if n <= Xc.shape[1]:
            M = (Xc @ Xc.T) / max(n - 1, 1)
        else:
            M = (Xc.T @ Xc) / max(n - 1, 1)
        lam = np.linalg.eigvalsh(M.astype(np.float64))[::-1]
        lam = np.clip(lam, 1e-12, None)
        for k in TOPK_SPECTRUM:
            kk = min(k, len(lam))
            ld = float(np.log(lam[:kk]).sum())
            p[f"logdet{k}"] = 0.5 * ld
            p[f"sparsity{k}"] = 0.5 * ld - np.log(max(n, 1))
    else:
        for k in TOPK_SPECTRUM:
            p[f"logdet{k}"] = np.nan
            p[f"sparsity{k}"] = np.nan

    # what is ACTUALLY in the .dyf file: num_bits scatter-like eigenvalues.
    # divide by n first -- measured to be scatter-like, so raw log-ev double-counts n.
    if ev_stored is not None and len(ev_stored) and n > 1:
        lam_s = np.clip(np.asarray(ev_stored, dtype=np.float64) / n, 1e-12, None)
        p["logdet4_stored"] = 0.5 * float(np.log(lam_s).sum())
        p["sparsity4_stored"] = p["logdet4_stored"] - np.log(n)
        p["logdet4_stored_raw"] = 0.5 * float(np.log(np.clip(ev_stored, 1e-12, None)).sum())
    else:
        p["logdet4_stored"] = np.nan
        p["sparsity4_stored"] = np.nan
        p["logdet4_stored_raw"] = np.nan
    return p


PREDICTORS = [
    "neg_log_n",
    "diffuse",
    "logdet4",
    "logdet32",
    "sparsity4",
    "sparsity32",
    "logdet4_stored",
    "sparsity4_stored",
    "logdet4_stored_raw",
]


# --------------------------------------------------------------------------------- stats
def spearman(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 5:
        return np.nan
    a, b = a[ok], b[ok]
    if len(np.unique(a)) < 3 or len(np.unique(b)) < 3:
        return np.nan
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    ra -= ra.mean()
    rb -= rb.mean()
    den = np.sqrt((ra**2).sum() * (rb**2).sum())
    return float((ra * rb).sum() / den) if den > 0 else np.nan


def perm_null(n_cells, rng, n_perm=N_PERM):
    """95th pct of |rho| under permutation, for this many cells."""
    if n_cells < 5:
        return np.nan
    x = np.arange(n_cells, dtype=float)
    rhos = [abs(spearman(x, rng.permutation(x))) for _ in range(n_perm)]
    return float(np.percentile(rhos, 95))


def per_query_recall(got, truth):
    return np.array([len(set(g.tolist()) & set(t.tolist())) / len(t) for g, t in zip(got, truth)])


# ----------------------------------------------------------------------------- main loop
def run_once(seed, E, T, SEC, Q):
    rng = np.random.default_rng(seed)
    N = len(E)
    res = {}

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

    conds = {
        "random": rng.random(N) < TARGET_BASE,
        "temporal": Q <= "2023Q4",
        "ticker": by_group(T),
        "section": np.isin(SEC, ["risk_factors", "forward_looking"]),
    }

    for name, mask in conds.items():
        t0 = time.time()
        base_pool, stream_pool = np.where(mask)[0], np.where(~mask)[0]
        qs = rng.choice(stream_pool, NQ, replace=False)
        held = np.zeros(N, bool)
        held[qs] = True
        base_idx = base_pool[~held[base_pool]]
        stream_idx = stream_pool[~held[stream_pool]]
        cur = np.concatenate([base_idx, stream_idx])
        Ecur = E[cur]

        flat = build_ev(E, base_idx, S.SEED)
        NL = S.n_leaves(flat)
        a_all, _ = S.route(E, flat)
        a_fro = a_all[cur]
        C_fro, _ = S.leaf_centroids(Ecur, a_fro, NL)

        flat_fresh = S.build(E, cur)
        a_fresh = S.fresh_assign(flat_fresh, len(cur))
        C_fresh, _ = S.leaf_centroids(Ecur, a_fresh, S.n_leaves(flat_fresh))

        QE = E[qs]
        truth = S.exact_knn(QE, Ecur, k=K)
        r_fro = per_query_recall(S.ivf_search(QE, Ecur, a_fro, C_fro, PROBE, K), truth)
        r_fresh = per_query_recall(S.ivf_search(QE, Ecur, a_fresh, C_fresh, PROBE, K), truth)
        gap_q = r_fresh - r_fro

        # route queries + stream through the FROZEN tree to get their leaf
        q_leaf, _ = S.route(QE, flat)
        s_leaf = a_all[stream_idx]

        Ebase = E[base_idx]
        cond_out: dict[str, object] = {
            "overall_gap": float(gap_q.mean()),
            "frozen_recall": float(r_fro.mean()),
        }
        stats_by_depth: dict[int, dict[str, float]] = {}

        for depth in DEPTHS:
            cells = cells_at_depth(flat, depth)
            leaf2cell = np.full(NL, -1, np.int64)
            for ci, (_, lids) in enumerate(cells):
                for lf in lids:
                    leaf2cell[lf] = ci

            rows = []
            for ci, (nid, _) in enumerate(cells):
                node = flat[nid]
                sel_q = leaf2cell[q_leaf] == ci
                nq = int(sel_q.sum())
                # base points of this cell: node["indices"] index into E[base_idx]
                bsel = node["indices"]
                nb = len(bsel)
                ns = int((leaf2cell[s_leaf] == ci).sum())
                if nb < 8:
                    continue
                p = cell_predictors(Ebase[bsel], node["ev"])
                p["nq"] = nq
                p["n_stream"] = ns
                p["flood"] = ns / max(nb, 1)
                p["cell_gap"] = float(gap_q[sel_q].mean()) if nq > 0 else np.nan
                rows.append(p)

            usable = [r for r in rows if r["nq"] >= MIN_Q[depth] and np.isfinite(r["cell_gap"])]
            nc = len(usable)
            stat: dict[str, float] = {"n_cells": len(rows), "n_usable": nc}
            if nc >= 5:
                y = [r["cell_gap"] for r in usable]
                yf = [r["flood"] for r in usable]
                for pn in PREDICTORS:
                    x = [r[pn] for r in usable]
                    stat[f"rho_gap_{pn}"] = spearman(x, y)
                    stat[f"rho_flood_{pn}"] = spearman(x, yf)
                stat["null95"] = perm_null(nc, np.random.default_rng(seed * 100 + depth))
            stats_by_depth[depth] = stat
            cond_out[f"depth{depth}"] = stat
            cond_out[f"depth{depth}_rows"] = [
                {k: (None if isinstance(v, float) and not np.isfinite(v) else v) for k, v in r.items()} for r in usable
            ]

        res[name] = cond_out
        d1 = stats_by_depth.get(1, {})
        print(
            f"  seed {seed} {name:<9} gap={cond_out['overall_gap']:+.4f} "
            f"cells={d1.get('n_usable', 0)} "
            f"rho[sparsity32]={d1.get('rho_gap_sparsity32', float('nan')):+.3f} "
            f"rho[diffuse]={d1.get('rho_gap_diffuse', float('nan')):+.3f} "
            f"rho[neg_log_n]={d1.get('rho_gap_neg_log_n', float('nan')):+.3f} "
            f"null95={d1.get('null95', float('nan')):.3f} [{time.time() - t0:.0f}s]",
            flush=True,
        )
    return res


def main():
    E, _D, T, SEC, Q = S.load()
    print(f"corpus {E.shape}  seeds={N_SEEDS}  NQ={NQ}  probe={PROBE}", flush=True)
    runs = []
    out_path = os.path.join(S.CACHE, "cell_volume_results.json")
    for s in range(N_SEEDS):
        runs.append(run_once(s, E, T, SEC, Q))
        with open(out_path, "w") as f:
            json.dump(runs, f, indent=2)
        print(f"seed {s} saved -> {out_path}", flush=True)

    print("\n" + "=" * 78)
    print("Spearman(predictor, per-cell recall gap), mean over seeds; HIGHER rho = predicts damage")
    print("=" * 78)
    for depth in DEPTHS:
        print(f"\n--- depth {depth} ---")
        hdr = f"{'predictor':<20}" + "".join(f"{c:>12}" for c in ["random", "temporal", "ticker", "section"])
        print(hdr)
        for pn in PREDICTORS:
            line = f"{pn:<20}"
            for c in ["random", "temporal", "ticker", "section"]:
                v = [r[c][f"depth{depth}"].get(f"rho_gap_{pn}", np.nan) for r in runs]
                v = [x for x in v if x is not None and np.isfinite(x)]
                line += f"{np.mean(v):>+12.3f}" if v else f"{'--':>12}"
            print(line)
        nl = [
            r[c][f"depth{depth}"].get("null95", np.nan)
            for r in runs
            for c in ["random", "temporal", "ticker", "section"]
        ]
        nl = [x for x in nl if x is not None and np.isfinite(x)]
        ncu = [
            r[c][f"depth{depth}"].get("n_usable", 0) for r in runs for c in ["random", "temporal", "ticker", "section"]
        ]
        print(f"{'[perm null95 |rho|]':<20}{np.mean(nl):>+12.3f}   (mean usable cells = {np.mean(ncu):.0f})")

    print("\nVALUE OF VOLUME = rho(sparsity32) - max(rho(neg_log_n), rho(diffuse)), per condition:")
    for depth in DEPTHS:
        for c in ["random", "temporal", "ticker", "section"]:

            def m(pn, c=c, depth=depth):
                v = [r[c][f"depth{depth}"].get(f"rho_gap_{pn}", np.nan) for r in runs]
                v = [x for x in v if x is not None and np.isfinite(x)]
                return np.mean(v) if v else np.nan

            best_free = max(m("neg_log_n"), m("diffuse"))
            print(
                f"  depth{depth} {c:<9} sparsity32={m('sparsity32'):+.3f}  best_free={best_free:+.3f}  delta={m('sparsity32') - best_free:+.3f}"
            )


# --------------------------------------------------------- eigenvalue semantics (--semantics)
def semantics():
    """What does get_eigenvalues() actually return? Test behaviour, don't assume a formula.

    Backs the two traps in the module docstring and in SPECTRAL_NOTES.md. Everything here
    is unit-normalised, matching real corpora -- so the question is not "does it scale with
    |x|" (meaningless on a sphere) but "does it track ANGULAR SPREAD, and is it a covariance
    or a scatter matrix".
    """
    from dyf_rs import DensityClassifier

    d, NB = 64, 4

    def cap(spread, n=4000, seed=0):
        r = np.random.default_rng(seed)
        axis = np.zeros(d, np.float32)
        axis[0] = 1.0
        X = axis + spread * r.standard_normal((n, d)).astype(np.float32)
        X /= np.linalg.norm(X, axis=1, keepdims=True) + 1e-12
        return X.astype(np.float32)

    def fit_ev(X, nb=NB):
        c = DensityClassifier(embedding_dim=X.shape[1], num_bits=nb, seed=42)
        c.fit_raw_pca(X)
        return np.asarray(c.get_eigenvalues(), dtype=np.float64)

    print("(a) does it track angular spread? -- and where does it saturate?")
    print(f"{'spread':>8}{'mean_cos':>10}{'sum_ev':>11}{'sum_log_ev':>12}{'true_logdet4':>14}")
    for spread in [0.02, 0.05, 0.1, 0.2, 0.4, 0.8]:
        X = cap(spread)
        ev = fit_ev(X)
        cen = X.mean(0)
        cen /= np.linalg.norm(cen)
        Xc = X - X.mean(0)
        te = np.linalg.eigvalsh(np.cov(Xc, rowvar=False))[::-1][:NB]
        print(
            f"{spread:>8.3f}{float((X @ cen).mean()):>10.4f}{ev.sum():>11.2f}"
            f"{np.log(ev).sum():>12.4f}{np.log(te).sum():>14.4f}"
        )
    print("  -> monotone in spread, but 0.4->0.8 barely moves it while mean_cos halves.")

    print("\n(b) covariance or scatter? sum(ev) vs n at FIXED spread=0.1")
    ns, sums = [500, 1000, 2000, 4000, 8000], []
    for nn in ns:
        s = fit_ev(cap(0.1, n=nn, seed=1)).sum()
        sums.append(s)
        print(f"  n={nn:>5}  sum={s:>9.2f}  sum/n={s / nn:>8.4f}")
    expo = np.polyfit(np.log(ns), np.log(sums), 1)[0]
    print(f"  -> sum(ev) ~ n^{expo:.2f}. Scatter-like: log-ev DOUBLE-COUNTS n. Divide by n.")

    print("\n(c) how many eigenvalues, and is the stored shadow enough?")
    X = cap(0.1)
    Xc = X - X.mean(0)
    lam = np.linalg.eigvalsh(np.cov(Xc, rowvar=False))[::-1]
    for nb in [2, 4, 8, 16]:
        ev = fit_ev(X, nb)
        print(f"  num_bits={nb:>3} -> len(ev)={len(ev):>3}  (of {d} dims available)")
    print(
        f"  -> only num_bits of {d} stored; top-4 vs top-32 logdet: "
        f"{np.log(lam[:4]).sum():.2f} vs {np.log(lam[:32]).sum():.2f}"
    )


# ------------------------------------------------------------- post-hoc anatomy (--anatomy)
def anatomy():
    """Read cell_volume_results.json and ground the three interpretive claims.

    Backs the tables quoted in SPECTRAL_NOTES.md: (1) every "volume" predictor is really
    occupancy, (2) damage rises with occupancy while flood falls, (3) the stored 4-D
    shadow does not track the real spectrum.
    """
    path = os.path.join(S.CACHE, "cell_volume_results.json")
    with open(path) as f:
        runs = json.load(f)
    conds = ["random", "temporal", "ticker", "section"]

    def col(rows, key):
        return np.array([np.nan if r.get(key) is None else r[key] for r in rows], float)

    print("(1) Is 'volume' just occupancy? Spearman(predictor, log n), depth 2")
    print(f"{'predictor':<22}" + "".join(f"{c:>12}" for c in conds))
    for pn in ["logdet4", "logdet32", "sparsity4", "sparsity32", "logdet4_stored", "logdet4_stored_raw", "diffuse"]:
        line = f"{pn:<22}"
        for c in conds:
            vals = [spearman(col(r[c]["depth2_rows"], pn), np.log(col(r[c]["depth2_rows"], "n"))) for r in runs]
            vals = [v for v in vals if np.isfinite(v)]
            line += f"{np.mean(vals):>+12.3f}" if vals else f"{'--':>12}"
        print(line)

    print("\n(2) Where does damage land? section condition, depth 2, binned by base occupancy")
    rows = [w for r in runs for w in r["section"]["depth2_rows"]]
    n, gap = col(rows, "n"), col(rows, "cell_gap")
    flood, ld = col(rows, "flood"), col(rows, "logdet32")
    q = np.quantile(n, [0, 0.25, 0.5, 0.75, 1.0])
    print(f"{'quartile':<16}{'n range':>16}{'mean gap':>11}{'mean flood':>12}{'mean logdet32':>15}{'cells':>7}")
    for i in range(4):
        m = (n >= q[i]) & (n <= q[i + 1] if i == 3 else n < q[i + 1])
        tag = "Q1 (smallest)" if i == 0 else "Q4 (largest)" if i == 3 else f"Q{i + 1}"
        print(
            f"{tag:<16}{f'{q[i]:.0f}-{q[i + 1]:.0f}':>16}{gap[m].mean():>+11.4f}"
            f"{flood[m].mean():>12.2f}{np.nanmean(ld[m]):>15.2f}{int(m.sum()):>7}"
        )
    print(
        f"  pooled: rho(log n, gap)={spearman(np.log(n), gap):+.3f}  "
        f"rho(log n, flood)={spearman(np.log(n), flood):+.3f}  "
        f"rho(logdet32, gap)={spearman(ld, gap):+.3f}  n_cells={len(rows)}"
    )

    print("\n(3) Does the stored 4-D shadow track the real spectrum? depth 2")
    for c in conds:
        rr = [w for r in runs for w in r[c]["depth2_rows"]]
        a, b, d32 = col(rr, "logdet4_stored"), col(rr, "logdet4"), col(rr, "logdet32")
        print(
            f"  {c:<10} rho(stored4, true4)={spearman(a, b):+.3f}   "
            f"rho(true4, true32)={spearman(b, d32):+.3f}   rho(stored4, true32)={spearman(a, d32):+.3f}"
        )


if __name__ == "__main__":
    if "--semantics" in sys.argv:
        semantics()
    elif "--anatomy" in sys.argv:
        anatomy()
    else:
        main()
