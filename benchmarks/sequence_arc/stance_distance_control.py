"""Is the stance signal anything more than DISTANCE?

stance_gate.py found cos(premise,hyp) separates entailment from contradiction at
AUC 0.905, with ||d|| 0.658 / 0.941 / 0.783 for entail / contra / neutral. But
SNLI contradictions were written to be definitely-false, so they naturally differ
in CONTENT more than entailments do. Cosine may simply be measuring semantic
distance, with stance riding along.

Same control that flipped the music result: compare INSIDE matched distance
bands. Within a narrow cos(p,h) band, entailment and contradiction pairs are
equally far apart -- so anything that still separates them is direction, not
distance.

Also fixes an unfairness in the previous supervised comparison: cos(p,h) is a
BILINEAR feature that no linear probe on p, h, or d can express, so the probes
were handicapped against it. Here every condition is scored inside the band,
where distance is held fixed for all of them.
"""

import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "snli_emb.npz")
BITS = 6


def unit(x, axis=-1):
    n = np.linalg.norm(x, axis=axis, keepdims=True)
    return x / np.where(n > 0, n, 1)


def mi_bits(codes, y):
    cu, ci = np.unique(codes, return_inverse=True)
    yu, yi = np.unique(y, return_inverse=True)
    M = np.zeros((len(cu), len(yu)))
    np.add.at(M, (ci, yi), 1)
    P = M / M.sum()
    pi, pj = P.sum(1, keepdims=True), P.sum(0, keepdims=True)
    nz = P > 0
    return float((P[nz] * np.log2(P[nz] / (pi @ pj)[nz])).sum())


def lsh(X, b):
    Xu = unit(X)
    Xc = Xu - Xu.mean(0, keepdims=True)
    _, _, vt = np.linalg.svd(Xc, full_matrices=False)
    hp = vt[:b]
    return (((Xu @ hp.T) >= 0) @ (2 ** np.arange(b))).astype(np.int64)


def probe(X, y):
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score

    return float(cross_val_score(LogisticRegression(max_iter=3000), X, y, cv=4, scoring="accuracy").mean())


def main():
    z = np.load(CACHE, allow_pickle=True)
    EP, EH, Y = z["EP"], z["EH"], z["Y"]
    D = EH - EP
    cos = (EP * EH).sum(1)
    ec = (Y == "entailment") | (Y == "contradiction")
    _EPe, EHe, De, Ye, cose = EP[ec], EH[ec], D[ec], Y[ec], cos[ec]
    ent = Ye == "entailment"

    qs = np.quantile(cose, [0.15, 0.35, 0.55, 0.75, 0.92])
    bands = list(zip(qs[:-1], qs[1:]))
    print("Within matched cos(p,h) bands -- distance held fixed:\n")
    print(
        f"{'cos band':>14} {'n':>5} {'%ent':>6} {'majority':>9} "
        f"{'MI(dir;y)':>10} {'LR unit-d':>10} {'LR hyp':>8} {'cos AUC':>8}"
    )

    for lo, hi in bands:
        m = (cose >= lo) & (cose < hi)
        n = int(m.sum())
        if n < 300:
            continue
        y = Ye[m]
        pe = ent[m].mean()
        maj = max(pe, 1 - pe)
        mi_d = mi_bits(lsh(De[m], BITS), y)
        lr_d = probe(unit(De[m]), y)
        lr_h = probe(EHe[m], y)
        c = cose[m]
        o = np.argsort(c)
        r = np.empty(n, float)
        r[o] = np.arange(1, n + 1)
        p_, q_ = ent[m].sum(), n - ent[m].sum()
        a = (r[ent[m]].sum() - p_ * (p_ + 1) / 2) / (p_ * q_)
        print(
            f"{lo:6.3f}-{hi:6.3f} {n:5d} {pe:6.2f} {maj:9.3f} "
            f"{mi_d:10.4f} {lr_d:10.3f} {lr_h:8.3f} {max(a, 1 - a):8.3f}"
        )

    print("\nReference (no band control):")
    print(f"  MI(delta dir; y) full set        {mi_bits(lsh(De, BITS), Ye):.4f}")
    print(f"  MI(hyp state; y) full set        {mi_bits(lsh(EHe, BITS), Ye):.4f}")


if __name__ == "__main__":
    main()
