"""Content audit on the stance result -- what is the delta actually reading?

SNLI has a well-documented annotation artifact: crowdworkers writing
"definitely false" sentences insert explicit negation ("not", "nobody",
"sleeping" vs "playing"), so HYPOTHESIS-ONLY models score far above chance
without ever consulting the premise. In the balanced band, LR on the hypothesis
state alone already hit 0.783 -- that artifact is present and large.

So the honest baseline for "does the delta read stance" is NOT the 0.512
majority rate; it is the hypothesis-only score. Two questions here:

  1. How much does the delta add OVER hypothesis-only? (the real increment)
  2. Is the signal just negation-token presence? Deflate a lexical negation
     axis and see what survives -- the same treatment that dissolved Haxe.
"""

import json
import os
import re

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "snli_emb.npz")
SNLI = os.path.join(HERE, "snli_1.0", "snli_1.0_test.jsonl")

NEG = {"no", "not", "nobody", "none", "never", "nothing", "n't", "cannot", "empty", "alone", "without"}


def unit(x, axis=-1):
    n = np.linalg.norm(x, axis=axis, keepdims=True)
    return x / np.where(n > 0, n, 1)


def probe(X, y):
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score

    return float(cross_val_score(LogisticRegression(max_iter=3000), X, y, cv=4, scoring="accuracy").mean())


def main():
    z = np.load(CACHE, allow_pickle=True)
    EP, EH, Y = z["EP"], z["EH"], z["Y"]

    hyps = []
    for line in open(SNLI):
        r = json.loads(line)
        if r.get("gold_label") in ("entailment", "contradiction", "neutral"):
            hyps.append(r["sentence2"])
    hyps = np.array(hyps)
    assert len(hyps) == len(Y)

    D = EH - EP
    cos = (EP * EH).sum(1)
    ec = (Y == "entailment") | (Y == "contradiction")
    lo, hi = np.quantile(cos[ec], 0.35), np.quantile(cos[ec], 0.55)
    m = ec & (cos >= lo) & (cos < hi)
    y = Y[m]
    ent = y == "entailment"
    print(
        f"balanced band cos {lo:.3f}-{hi:.3f}  n={int(m.sum())}  "
        f"%entail {ent.mean():.2f}  majority {max(ent.mean(), 1 - ent.mean()):.3f}"
    )

    hasneg = np.array([bool(NEG & set(re.findall(r"[a-z']+", h.lower()))) for h in hyps])
    hn = hasneg[m]
    print(f"\nnegation-token rate: entailment {hn[ent].mean():.3f}  contradiction {hn[~ent].mean():.3f}")
    print(f"  negation flag ALONE as classifier: {max((hn == ~ent).mean(), (hn == ent).mean()):.3f}")

    Dm, EHm = D[m], EH[m]
    base_h = probe(EHm, y)
    base_d = probe(unit(Dm), y)
    print(f"\n{'condition':<34} {'acc':>7} {'vs hyp-only':>12}")
    print(f"  {'hypothesis-only (the artifact)':<32} {base_h:7.3f} {'--':>12}")
    print(f"  {'DELTA direction':<32} {base_d:7.3f} {base_d - base_h:+12.3f}")

    # deflate a lexical negation axis fitted on the deltas
    t = (hn.astype(float) - hn.mean()) / (hn.std() + 1e-9)
    w = Dm.T @ t / len(t)
    w /= np.linalg.norm(w)
    Dd = Dm - np.outer(Dm @ w, w)
    wh = EHm.T @ t / len(t)
    wh /= np.linalg.norm(wh)
    EHd = EHm - np.outer(EHm @ wh, wh)
    print(
        f"  {'DELTA, negation axis removed':<32} {probe(unit(Dd), y):7.3f} "
        f"{probe(unit(Dd), y) - probe(EHd, y):+12.3f}  "
        f"(vs hyp-only deflated {probe(EHd, y):.3f})"
    )

    # premise-blind control: does the delta beat hypothesis-only because it
    # actually uses the premise, or would a random premise do as well?
    rng = np.random.default_rng(0)
    perm = rng.permutation(int(m.sum()))
    D_fake = EHm - EP[m][perm]
    print(f"  {'DELTA vs RANDOM premise':<32} {probe(unit(D_fake), y):7.3f} {probe(unit(D_fake), y) - base_h:+12.3f}")


if __name__ == "__main__":
    main()
