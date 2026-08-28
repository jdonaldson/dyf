"""Does derived num_bits generalise, or is it a property of one SEC corpus?

sec_derived_bits.py / sec_hier_routing.py measured ONE corpus (229k SEC 10-Q sections,
Nomic 768d). "Deriving num_bits helps every index" is a floor-raiser claim and n=1 does
not earn it. This runs the same factorial across corpora that differ in dimension, model
and modality:

  cmu_mocap    140,837 x  62   RAW joint angles -- not embeddings at all
  wikipedia     99,921 x 384   MiniLM
  news         100,000 x 384   MiniLM
  tweets        25,342 x 384   MiniLM, short text, small N
  arxiv         99,985 x 768   Nomic (same family as SEC, different domain)

The 62-d motion-capture set is the sharpest test: the whole mechanism runs through
spectral decay, and 62 dimensions behave nothing like 768.

SIMPLIFICATION EARNED BY THE PREVIOUS RESULT. sec_derived_bits.py had to binary-search
min_leaf per condition to equalise leaf counts, because flat IVF scans every leaf centroid
so leaf count was a first-order cost. Under dyf's real hierarchical router (reimplemented
in sec_hier_routing.py) work is routing dots + members scanned, and BOTH sides of the
granularity trade are counted: finer leaves mean fewer members scanned per probe but more
routing dots. So the work metric is complete and every arm can share one min_leaf, with
only bits/offset varying. That removes ~20 tree builds.

Headline metric is the operationally meaningful one: WORK REDUCTION AT EQUAL RECALL, not
recall delta at an arbitrary budget. Recall deltas are largest at tight budgets that
correspond to recall ~0.4-0.6, which nobody ships.
"""

import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sec_seqlib as S  # noqa: E402
from sec_derived_bits import build, n_leaves  # noqa: E402
from sec_hier_routing import evaluate_hier  # noqa: E402

PAPER = os.environ.get(
    "DYF_PAPER_DATA", os.path.expanduser("~/Projects/semantic-proprioception-paper/experiments/data")
)
CORPORA = [
    ("cmu_mocap_62", "cmu_mocap_features.npy"),
    ("wikipedia_384", "wikipedia_embeddings.npy"),
    ("news_384", "news_embeddings.npy"),
    ("tweets_384", "tweets_embeddings.npy"),
    ("arxiv_768", "arxiv_nomic_embeddings.npy"),
]
MAX_N = 100000
NQ = 500
K = 10
MIN_LEAF = 16
MAX_DEPTH = 4
NUM_BITS = 4
SEED = 42
TARGET_RECALLS = (0.80, 0.90)


def load_corpus(fname, rng):
    """Unit-normalise so cosine retrieval is well defined and comparable across sets.

    Note for cmu_mocap: these are raw joint angles, not embeddings. Normalising defines a
    consistent retrieval task; it is not claimed to be the natural metric for motion.
    """
    X = np.load(os.path.join(PAPER, fname)).astype(np.float32)
    if len(X) > MAX_N:
        X = X[rng.choice(len(X), MAX_N, replace=False)]
    X = np.ascontiguousarray(X)
    X /= np.linalg.norm(X, axis=1, keepdims=True) + 1e-12
    return X


def work_at_recall(curve, target):
    """Interpolate the work needed to reach `target` recall. None if never reached."""
    pts = sorted((c["recall"], c["work"]) for c in curve)
    rs = np.array([p[0] for p in pts])
    ws = np.array([p[1] for p in pts])
    if target < rs.min() or target > rs.max():
        return None
    return float(np.exp(np.interp(target, rs, np.log(ws))))


def main():
    conds = [
        ("fixed4_origin", "fixed", "origin"),
        ("fixed4_median", "fixed", "median"),
        ("derived_origin", "derived", "origin"),
        ("derived_median", "derived", "median"),
    ]
    allout = {}
    for cname, fname in CORPORA:
        path = os.path.join(PAPER, fname)
        if not os.path.exists(path):
            print(f"SKIP {cname}: {path} not found", flush=True)
            continue
        rng = np.random.default_rng(SEED)
        E = load_corpus(fname, rng)
        qi = rng.choice(len(E), min(NQ, len(E)), replace=False)
        Qe = E[qi]
        print(f"\n=== {cname}: {E.shape} ===", flush=True)
        truth = S.exact_knn(Qe, E, k=K)

        out = {}
        for name, bm, om in conds:
            t0 = time.time()
            flat = build(E, None, MAX_DEPTH, MIN_LEAF, bm, om, NUM_BITS, np.random.default_rng(SEED))
            pts = evaluate_hier(E, flat, Qe, truth)
            out[name] = {
                "leaves": n_leaves(flat),
                "curve": [{"work": w, "recall": r, "probe": p} for w, r, p in pts],
            }
            print(
                f"  {name:<16} leaves={n_leaves(flat):>6}  "
                + "  ".join(f"{r:.3f}@{w:.0f}" for w, r, p in pts[:3])
                + f"  [{time.time() - t0:.0f}s]",
                flush=True,
            )
        allout[cname] = out

        print(f"  --- {cname}: work to reach target recall (lower is better) ---", flush=True)
        for tr in TARGET_RECALLS:
            base = work_at_recall(out["fixed4_origin"]["curve"], tr)
            row = f"    recall {tr:.2f}: "
            if base is None:
                print(row + "baseline never reaches it")
                continue
            for name, _, _ in conds:
                w = work_at_recall(out[name]["curve"], tr)
                row += f"{name.split('_')[0][0]}{name.split('_')[1][0]}={w:.0f} " if w else f"{name}=n/a "
            dm = work_at_recall(out["derived_median"]["curve"], tr)
            do = work_at_recall(out["derived_origin"]["curve"], tr)
            row += f" | speedup derived_median {base / dm:.2f}x" if dm else ""
            row += f", derived_origin {base / do:.2f}x" if do else ""
            print(row, flush=True)

    print("\n" + "=" * 84)
    print("SUMMARY: work reduction vs fixed4_origin at equal recall (>1 = better)")
    print("=" * 84)
    print(f"{'corpus':<16}{'recall':>8}" + "".join(f"{c:>17}" for c, _, _ in conds[1:]))
    for cname, out in allout.items():
        for tr in TARGET_RECALLS:
            base = work_at_recall(out["fixed4_origin"]["curve"], tr)
            if base is None:
                continue
            line = f"{cname:<16}{tr:>8.2f}"
            for name, _, _ in conds[1:]:
                w = work_at_recall(out[name]["curve"], tr)
                line += f"{base / w:>16.2f}x" if w else f"{'n/a':>17}"
            print(line)

    path = os.path.join(S.CACHE, "multicorpus_bits_results.json")
    with open(path, "w") as f:
        json.dump(allout, f, indent=2)
    print(f"\nsaved -> {path}")


if __name__ == "__main__":
    main()
