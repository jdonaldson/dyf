"""STANCE GATE: does delta-space recover stance where state-space cannot?

The claim from SEQUENCE_NOTES.md: d = emb(turn) - emb(prior_turn) cancels
common-mode TOPIC content, so "I agree, X" and "no, X is false" -- which sit
adjacent in state space because both are about X -- should separate in delta
space. If false, the whole conversation branch closes.

SNLI is the right instrument: premise/hypothesis pairs labeled
entailment / contradiction / neutral, where the SAME premise recurs across
labels, so topic is controlled by construction.

  premise    -> the prior turn (source state)
  hypothesis -> the current turn (target state)
  delta      -> emb(hyp) - emb(prem)
  label      -> the move type; entailment=agree, contradiction=disagree,
                neutral=the topic-drift control

Controls carried from the music/Haxe audits:
  * MAGNITUDE. If contradictions are simply FARTHER than entailments, then
    "delta separates stance" is trivially distance, not direction. cos(p,h)
    alone is reported as a 1-D classifier -- the cheap baseline the delta
    machinery must beat.
  * UNSUPERVISED vs SUPERVISED. sign-LSH bucket MI is what dyf would actually
    give you; logistic regression is the "a linear probe will beat you"
    ceiling from the introspection arc. Both reported.
"""

import json
import os

import numpy as np

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

HERE = os.path.dirname(os.path.abspath(__file__))
SNLI = os.path.join(HERE, "snli_1.0", "snli_1.0_test.jsonl")
CACHE = os.path.join(HERE, "snli_emb.npz")
MODEL = "BAAI/bge-base-en-v1.5"
BITS = 6


def unit(x, axis=-1):
    n = np.linalg.norm(x, axis=axis, keepdims=True)
    return x / np.where(n > 0, n, 1)


def load_pairs():
    P, H, Y = [], [], []
    for line in open(SNLI):
        r = json.loads(line)
        g = r.get("gold_label")
        if g in ("entailment", "contradiction", "neutral"):
            P.append(r["sentence1"])
            H.append(r["sentence2"])
            Y.append(g)
    return P, H, np.array(Y)


def embed():
    if os.path.exists(CACHE):
        z = np.load(CACHE, allow_pickle=True)
        return z["EP"], z["EH"], z["Y"]
    from sentence_transformers import SentenceTransformer

    P, H, Y = load_pairs()
    m = SentenceTransformer(MODEL)
    uniq = sorted(set(P) | set(H))
    print(f"pairs {len(P)}  unique sentences {len(uniq)} -> embedding...", flush=True)
    E = m.encode(uniq, batch_size=128, show_progress_bar=False, convert_to_numpy=True, normalize_embeddings=True)
    idx = {s: i for i, s in enumerate(uniq)}
    EP = E[[idx[s] for s in P]]
    EH = E[[idx[s] for s in H]]
    np.savez_compressed(CACHE, EP=EP, EH=EH, Y=Y)
    return EP, EH, Y


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


def auc(score, pos):
    o = np.argsort(score)
    r = np.empty(len(score), float)
    r[o] = np.arange(1, len(score) + 1)
    np_, nn = pos.sum(), (~pos).sum()
    return float((r[pos].sum() - np_ * (np_ + 1) / 2) / (np_ * nn))


def probe(X, y, seed=0):
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score

    lr = LogisticRegression(max_iter=2000, C=1.0)
    return float(cross_val_score(lr, X, y, cv=4, scoring="accuracy").mean())


def main():
    EP, EH, Y = embed()
    D = EH - EP
    cos = (EP * EH).sum(1)
    ent, con, neu = Y == "entailment", Y == "contradiction", Y == "neutral"
    print(f"n={len(Y)}  entail {ent.sum()}  contra {con.sum()}  neutral {neu.sum()}")

    print("\n== 1. is state space stance-blind? ==")
    for nm, m in (("entailment", ent), ("contradiction", con), ("neutral", neu)):
        print(
            f"  cos(premise,hyp)  {nm:<14} {cos[m].mean():.4f} +/- {cos[m].std():.4f}"
            f"   ||d|| {np.linalg.norm(D[m], axis=1).mean():.4f}"
        )
    ec = ent | con
    print(
        f"  --> cos alone, entail-vs-contra AUC = {auc(cos[ec], ent[ec]):.4f}"
        "   (0.5 = stance-blind; this is the cheap baseline to beat)"
    )

    print("\n== 2. UNSUPERVISED (what dyf gives you): MI(bucket; label), bits ==")
    for nm, X in (("hypothesis state", EH), ("premise state", EP), ("DELTA (direction)", D)):
        print(
            f"  {nm:<20} all-3 {mi_bits(lsh(X, BITS), Y):.4f}   entail-vs-contra {mi_bits(lsh(X[ec], BITS), Y[ec]):.4f}"
        )

    print("\n== 3. SUPERVISED ceiling: 4-fold LR accuracy ==")
    maj_all = max(np.mean(c == Y) for c in np.unique(Y))
    maj_ec = max(ent[ec].mean(), con[ec].mean())
    print(f"  {'majority baseline':<24} all-3 {maj_all:.4f}   entail-vs-contra {maj_ec:.4f}")
    for nm, X in (
        ("hypothesis state", EH),
        ("DELTA raw", D),
        ("DELTA unit-normalized", unit(D)),
        ("concat [p,h]", np.hstack([EP, EH])),
    ):
        print(f"  {nm:<24} all-3 {probe(X, Y):.4f}   entail-vs-contra {probe(X[ec], Y[ec]):.4f}")


if __name__ == "__main__":
    main()
