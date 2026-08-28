"""Which children are NOT self-similar to their parent, and what is in them?

sec_depth_spectra.py reported that shape is heritable along a path (parent->child
rho ~0.65) and that splits concentrate structure at every depth. Both are group MEANS.
This finds the tail: the individual splits that produce a child whose spectrum is not the
parent's shape at all.

SCORING. "Different from the parent" is not the same as "different from a random subset of
the parent" -- a child is a subset, so it differs by construction. Every child is therefore
scored against a PARENT-SUBSAMPLE NULL built from R independent same-size draws of its own
parent, giving a per-child z rather than a group mean:

    z_desc = (child_desc - mean(null_desc)) / sd(null_desc)

and a whole-shape score that does not collapse the spectrum to one number:

    div    = L1(child_spectrum, mean null spectrum)
    z_div  = (div - mean(L1(null_i, mean null))) / sd(...)

Child and null spectra are averaged over the SAME number of draws, or the child would be
the less noisy of the two and every z would be inflated.

Sign matters and the two tails are different objects:
  z_eff_rank << 0  the split isolated something far tighter than the parent -- predicted
                   to be near-duplicate/boilerplate, since eff_rank ran rho=-0.723 with
                   dup_frac in sec_cell_spectra.py
  z_eff_rank >> 0  the child is MORE diffuse than a random parent draw. Harder to explain:
                   the split gathered heterogeneous material rather than concentrating it.

Every reported outlier gets a content audit (section mix, ticker diversity, duplicate
fraction) against its parent -- a detection result without one is not a result.
"""

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sec_seqlib as S  # noqa: E402
from sec_cell_spectra import descriptors  # noqa: E402
from sec_cell_volume import flatten_ev  # noqa: E402
from sec_depth_spectra import node_depths  # noqa: E402

N_SUB = 300
D_DRAWS = 3  # draws averaged per spectrum, identical for child and null
R_NULL = 8  # null replicates per parent
DESCS = ["eff_rank", "top1", "alpha"]
SEED = 42
TOP_N = 8


def spec(X, idx, rng, n_sub=N_SUB, d=D_DRAWS):
    """Mean normalised spectrum over d draws of size n_sub from X[idx]."""
    if len(idx) < n_sub:
        return None
    acc = None
    for _ in range(d):
        s = idx[rng.choice(len(idx), n_sub, replace=False)]
        Z = X[s].astype(np.float64)
        Z -= Z.mean(0)
        lam = np.linalg.eigvalsh((Z @ Z.T) / (n_sub - 1))[::-1]
        lam = np.clip(lam[: n_sub - 1], 1e-14, None)
        lam /= lam.sum()
        acc = lam if acc is None else acc + lam
    return acc / d


def content(E, idx, SEC, T, rng):
    sec = SEC[idx]
    vals, cnt = np.unique(sec, return_counts=True)
    sm = E[rng.choice(idx, min(len(idx), 400), replace=False)]
    G = sm @ sm.T
    np.fill_diagonal(G, -1.0)
    return {
        "n": int(len(idx)),
        "sec_top": str(vals[cnt.argmax()]),
        "sec_purity": float(cnt.max() / cnt.sum()),
        "sec_mix": {str(v): int(c) for v, c in sorted(zip(vals, cnt), key=lambda x: -x[1])[:4]},
        "ticker_div": float(len(np.unique(T[idx])) / len(idx)),
        "dup_frac": float((G.max(1) > 0.99).mean()),
        "mean_cos": float(G[G > -1].mean()),
    }


def main():
    E, D, T, SEC, Q = S.load()
    rng = np.random.default_rng(SEED)
    from dyf.dyf_tree import build_dyf_tree

    flat = flatten_ev(
        build_dyf_tree(E, max_depth=S.MAX_DEPTH, num_bits=S.NUM_BITS, min_leaf_size=S.MIN_LEAF, seed=SEED), E
    )
    dep = node_depths(flat)
    print(f"tree: {len(flat)} nodes, {S.n_leaves(flat)} leaves", flush=True)

    rows = []
    parents = [i for i, nd in enumerate(flat) if nd["children"] and len(nd["indices"]) >= N_SUB * 2]
    print(f"scoring children of {len(parents)} parents (n >= {N_SUB * 2})", flush=True)
    for pi, pid in enumerate(parents):
        pidx = flat[pid]["indices"]
        nulls = [spec(E, pidx, rng) for _ in range(R_NULL)]
        nulls = [n for n in nulls if n is not None]
        if len(nulls) < 4:
            continue
        NA = np.stack(nulls)
        nmean = NA.mean(0)
        ndesc = np.array([[descriptors(n)[d] for d in DESCS] for n in nulls])
        nd_mu, nd_sd = ndesc.mean(0), ndesc.std(0) + 1e-9
        ndiv = np.abs(NA - nmean).sum(1)
        div_mu, div_sd = ndiv.mean(), ndiv.std() + 1e-9

        for kid in flat[pid]["children"]:
            kidx = flat[kid]["indices"]
            cs = spec(E, kidx, rng)
            if cs is None:
                continue
            cd = descriptors(cs)
            z = {d: float((cd[d] - nd_mu[i]) / nd_sd[i]) for i, d in enumerate(DESCS)}
            rows.append(
                {
                    "child": int(kid),
                    "parent": int(pid),
                    "depth": int(dep[kid]),
                    "n_child": int(len(kidx)),
                    "n_parent": int(len(pidx)),
                    "share": float(len(kidx) / len(pidx)),
                    "z_div": float((np.abs(cs - nmean).sum() - div_mu) / div_sd),
                    **{f"z_{d}": z[d] for d in DESCS},
                    **{d: cd[d] for d in DESCS},
                }
            )
        if (pi + 1) % 40 == 0:
            print(f"  {pi + 1}/{len(parents)} parents, {len(rows)} children scored", flush=True)

    print(f"\n{len(rows)} parent-child pairs scored")
    ze = np.array([r["z_eff_rank"] for r in rows])
    zd = np.array([r["z_div"] for r in rows])
    print("\nHow common is non-self-similarity? (z vs the parent-subsample null)")
    for lab, v in [("z_eff_rank", ze), ("z_div", zd)]:
        print(
            f"  {lab:<12} |z|>2: {100 * (np.abs(v) > 2).mean():>5.1f}%   |z|>5: {100 * (np.abs(v) > 5).mean():>5.1f}%"
            f"   |z|>10: {100 * (np.abs(v) > 10).mean():>5.1f}%   median {np.median(v):+.2f}"
        )
    print(
        f"  eff_rank z sign split: {100 * (ze < 0).mean():.0f}% negative (tighter than parent), "
        f"{100 * (ze > 0).mean():.0f}% positive (more diffuse)"
    )

    audit = {}
    for tag, key, rev in [
        ("MOST CONCENTRATING (z_eff_rank << 0)", "z_eff_rank", False),
        ("MOST DIFFUSING (z_eff_rank >> 0)", "z_eff_rank", True),
    ]:
        sel = sorted(rows, key=lambda r: r[key], reverse=rev)[:TOP_N]
        print(f"\n=== {tag} ===")
        print(f"{'child':>7}{'dep':>4}{'n':>7}{'share':>7}{'z_eff':>8}{'z_div':>8}{'eff_rank':>10}  content vs parent")
        for r in sel:
            c = content(E, flat[r["child"]]["indices"], SEC, T, rng)
            p = content(E, flat[r["parent"]]["indices"], SEC, T, rng)
            audit.setdefault(tag, []).append({**r, "child_content": c, "parent_content": p})
            print(
                f"{r['child']:>7}{r['depth']:>4}{r['n_child']:>7}{r['share']:>7.2f}"
                f"{r['z_eff_rank']:>8.1f}{r['z_div']:>8.1f}{r['eff_rank']:>10.1f}  "
                f"sec={c['sec_top']}({c['sec_purity']:.2f}) dup={c['dup_frac']:.2f} "
                f"cos={c['mean_cos']:+.2f} | parent sec={p['sec_top']}({p['sec_purity']:.2f}) "
                f"dup={p['dup_frac']:.2f}"
            )

    # is the concentrating tail the boilerplate story?
    dfs = []
    for r in sorted(rows, key=lambda r: r["z_eff_rank"])[:40]:
        dfs.append(content(E, flat[r["child"]]["indices"], SEC, T, rng)["dup_frac"])
    dfd = []
    for r in sorted(rows, key=lambda r: -r["z_eff_rank"])[:40]:
        dfd.append(content(E, flat[r["child"]]["indices"], SEC, T, rng)["dup_frac"])
    print(
        f"\nmean dup_frac: 40 most-concentrating children = {np.mean(dfs):.3f}, 40 most-diffusing = {np.mean(dfd):.3f}"
    )

    path = os.path.join(S.CACHE, "nonselfsimilar_results.json")
    with open(path, "w") as f:
        json.dump({"rows": rows, "audit": audit}, f, indent=2)
    print(f"\nsaved -> {path}")


if __name__ == "__main__":
    main()
