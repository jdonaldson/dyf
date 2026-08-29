"""Is non-self-similarity just near-duplicate content? Strip the dupes and re-measure.

sec_nonselfsimilar.py found 91.9% of children non-self-similar to a same-size draw of
their parent, with the concentrating tail carrying dup_frac 0.450 (extremes 0.52-0.76) and
the diffusing tail 0.295. If the whole effect is degenerate duplicate clusters then it is a
data-quality artifact, not a property of the partition. The only way to know is to remove
the duplicates and re-run the identical measurement.

DEDUP IS BLOCKED BY LEAF, AND THAT IS AN APPROXIMATION. Exact all-pairs at cos > 0.99 over
229k x 768 is not worth it; near-duplicates almost always route to the same leaf, so
duplicates are collapsed within each leaf of the ORIGINAL tree. Cross-leaf duplicate pairs
are missed, so the reported dedup rate is a lower bound and the ablation is conservative --
it under-removes, which biases toward the effect surviving. Stated so the result is read
with the right sign of error.

A FRESH TREE IS BUILT ON THE DEDUPED CORPUS. Reusing the original tree would leave the
partition shaped by the duplicates it was fitted on, which is the thing under test.

Read the output as: if |z| collapses toward the null, non-self-similarity was duplicates.
If it survives, the partition really does reshape distributions at every split.
"""

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sec_seqlib as S  # noqa: E402
from sec_cell_spectra import descriptors  # noqa: E402
from sec_cell_volume import flatten_ev  # noqa: E402
from sec_nonselfsimilar import DESCS, N_SUB, R_NULL, spec  # noqa: E402

SEED = 42
DUP_COS = 0.99


def dedup_by_leaf(E, flat):
    """Collapse near-duplicates within each leaf. Returns kept global indices."""
    keep = []
    removed = 0
    for nd in flat:
        if nd["leaf_id"] < 0:
            continue
        idx = nd["indices"]
        if len(idx) <= 1:
            keep.extend(idx.tolist())
            continue
        G = E[idx] @ E[idx].T
        np.fill_diagonal(G, -1.0)
        alive = np.ones(len(idx), bool)
        for i in range(len(idx)):
            if not alive[i]:
                continue
            dup = np.where((G[i] > DUP_COS) & alive)[0]
            dup = dup[dup > i]
            alive[dup] = False
            removed += len(dup)
        keep.extend(idx[alive].tolist())
    return np.array(sorted(keep), dtype=np.int64), removed


def score_tree(E, flat, rng, label):
    """Identical scoring to sec_nonselfsimilar: child vs same-size parent-subsample null."""
    rows = []
    parents = [i for i, nd in enumerate(flat) if nd["children"] and len(nd["indices"]) >= N_SUB * 2]
    for pid in parents:
        pidx = flat[pid]["indices"]
        nulls = [spec(E, pidx, rng) for _ in range(R_NULL)]
        nulls = [n for n in nulls if n is not None]
        if len(nulls) < 4:
            continue
        nd_arr = np.array([[descriptors(n)[d] for d in DESCS] for n in nulls])
        mu, sd = nd_arr.mean(0), nd_arr.std(0) + 1e-9
        for kid in flat[pid]["children"]:
            cs = spec(E, flat[kid]["indices"], rng)
            if cs is None:
                continue
            cd = descriptors(cs)
            rows.append(
                {
                    "share": float(len(flat[kid]["indices"]) / len(pidx)),
                    **{f"z_{d}": float((cd[d] - mu[i]) / sd[i]) for i, d in enumerate(DESCS)},
                }
            )
    print(f"  {label}: {len(parents)} parents, {len(rows)} children scored", flush=True)
    return rows


def summarise(rows, label):
    z = np.array([r["z_eff_rank"] for r in rows])
    return {
        "label": label,
        "n": len(rows),
        "median_z": float(np.median(z)),
        "pct_gt2": float(100 * (np.abs(z) > 2).mean()),
        "pct_gt10": float(100 * (np.abs(z) > 10).mean()),
        "pct_neg": float(100 * (z < 0).mean()),
        "mean_abs_z": float(np.abs(z).mean()),
    }


def main():
    E, *_ = S.load()
    rng = np.random.default_rng(SEED)
    from dyf.dyf_tree import build_dyf_tree

    def build(X):
        return flatten_ev(
            build_dyf_tree(X, max_depth=S.MAX_DEPTH, num_bits=S.NUM_BITS, min_leaf_size=S.MIN_LEAF, seed=SEED), X
        )

    flat0 = build(E)
    print(f"original tree: {len(flat0)} nodes, {S.n_leaves(flat0)} leaves", flush=True)
    before = score_tree(E, flat0, rng, "WITH duplicates")

    keep, removed = dedup_by_leaf(E, flat0)
    Ed = np.ascontiguousarray(E[keep])
    print(
        f"\ndedup (within-leaf, cos > {DUP_COS}): removed {removed:,} of {len(E):,} "
        f"({100 * removed / len(E):.1f}%), {len(Ed):,} remain",
        flush=True,
    )

    flat1 = build(Ed)
    print(f"deduped tree: {len(flat1)} nodes, {S.n_leaves(flat1)} leaves", flush=True)
    after = score_tree(Ed, flat1, rng, "WITHOUT duplicates")

    a, b = summarise(before, "with dupes"), summarise(after, "deduped")
    print("\n" + "=" * 78)
    print("Does non-self-similarity survive removing the duplicates?")
    print("=" * 78)
    print(
        f"{'':<14}{'children':>10}{'median z':>11}{'|z|>2 %':>10}{'|z|>10 %':>11}{'mean |z|':>10}{'% concentrating':>18}"
    )
    for s in (a, b):
        print(
            f"{s['label']:<14}{s['n']:>10}{s['median_z']:>11.2f}{s['pct_gt2']:>10.1f}"
            f"{s['pct_gt10']:>11.1f}{s['mean_abs_z']:>10.1f}{s['pct_neg']:>18.0f}"
        )
    print(
        f"\nretention of the effect: |z|>10 goes {a['pct_gt10']:.1f}% -> {b['pct_gt10']:.1f}%  "
        f"({100 * b['pct_gt10'] / max(a['pct_gt10'], 1e-9):.0f}% retained), "
        f"mean|z| {a['mean_abs_z']:.1f} -> {b['mean_abs_z']:.1f}"
    )
    print(
        "\nIf the effect were degenerate duplicates, |z| would collapse toward the null here.\n"
        "Dedup is within-leaf and therefore a LOWER bound on duplicate removal, so survival\n"
        "is the conservative direction and a collapse would be the strong result."
    )

    # ---- CONFOUND CONTROL --------------------------------------------------------
    # Dedup shrinks the corpus, so fewer nodes clear the n>=300 floor and the survivors
    # skew toward LARGER shares of their parent -- and |z| falls with share (29.4 at
    # share<0.05 down to 8.2 at share>0.35). Some of the drop could therefore be selection
    # rather than dedup. Compare within matched share bins.
    print("\n" + "=" * 78)
    print("Same comparison WITHIN MATCHED SHARE BINS (controls the selection shift)")
    print("=" * 78)
    bins = [(0.0, 0.05), (0.05, 0.15), (0.15, 0.35), (0.35, 1.01)]
    print(f"{'share bin':<14}{'n (dup)':>9}{'mean|z| dup':>13}{'n (dedup)':>11}{'mean|z| dedup':>15}{'retained':>10}")
    for lo, hi in bins:
        za = np.array([abs(r["z_eff_rank"]) for r in before if lo <= r["share"] < hi])
        zb = np.array([abs(r["z_eff_rank"]) for r in after if lo <= r["share"] < hi])
        if len(za) < 5 or len(zb) < 5:
            print(f"{f'{lo:.2f}-{hi:.2f}':<14}{len(za):>9}{'--':>13}{len(zb):>11}{'--':>15}{'--':>10}")
            continue
        print(
            f"{f'{lo:.2f}-{hi:.2f}':<14}{len(za):>9}{za.mean():>13.1f}{len(zb):>11}"
            f"{zb.mean():>15.1f}{100 * zb.mean() / za.mean():>9.0f}%"
        )
    sa = np.array([r["share"] for r in before])
    sb = np.array([r["share"] for r in after])
    print(f"\n  share distribution shifted: median {np.median(sa):.3f} -> {np.median(sb):.3f}")
    print("  If retention is similar across bins, the drop is dedup, not selection.")

    path = os.path.join(S.CACHE, "dedup_ablation_results.json")
    with open(path, "w") as f:
        json.dump({"with_dupes": a, "deduped": b, "removed": int(removed), "kept": int(len(Ed))}, f, indent=2)
    print(f"\nsaved -> {path}")


if __name__ == "__main__":
    main()
