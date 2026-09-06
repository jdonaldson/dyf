"""Does the SNLI delta-as-relation finding transfer to real conversational turns?

SNLI result: delta = emb(hyp) - emb(premise) beat hypothesis-only by +6.7pp, and
scrambling the premise DEGRADED it below hypothesis-only -- so the delta was
reading the specific pair relation, not artifacts of the second sentence.

Transfer test on SwDA:
    prior turn -> premise      current turn -> hypothesis
    label      -> the current turn's dialogue act

Contrast chosen deliberately: `b` (backchannel, "Uh-huh") vs `aa`
(agree/accept, "Yes"). Both are short affirmative responses with overlapping
vocabulary, and which one applies depends on WHAT PRECEDED -- a genuinely
context-dependent label. Contrasts like ny/nn ("Yeah"/"No") are lexically
trivial and leave no headroom.

Same three controls as the SNLI gate:
  * current-turn-only is the honest baseline (not majority)
  * distance banding on cos(prior, current)
  * RANDOM-PRIOR scramble -- the decisive one
"""

import os

import numpy as np
import polars as pl

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

HERE = os.path.dirname(os.path.abspath(__file__))
PARSED = os.path.join(HERE, "swda_parsed.parquet")
CACHE = os.path.join(HERE, "swda_emb.npz")
MODEL = "BAAI/bge-base-en-v1.5"
PER_CLASS = 6000
BITS = 6


def unit(x, axis=-1):
    n = np.linalg.norm(x, axis=axis, keepdims=True)
    return x / np.where(n > 0, n, 1)


def build_pairs():
    df = pl.read_parquet(PARSED).sort(["conversation_no", "transcript_index"])
    df = df.with_columns(
        [
            pl.col("clean").shift(1).over("conversation_no").alias("prev_text"),
            pl.col("ntok").shift(1).over("conversation_no").alias("prev_ntok"),
            pl.col("caller").shift(1).over("conversation_no").alias("prev_caller"),
        ]
    )
    d = df.filter(
        pl.col("prev_text").is_not_null()
        & (pl.col("prev_ntok") >= 4)  # prior must carry content
        & (pl.col("caller") != pl.col("prev_caller"))  # genuine response
        & (pl.col("clean").str.len_chars() > 0)
        & pl.col("act").is_in(["b", "aa"])
    )
    parts = []
    for a in ("b", "aa"):
        s = d.filter(pl.col("act") == a)
        parts.append(s.sample(min(PER_CLASS, s.height), seed=0))
    return pl.concat(parts)


def embed(pairs):
    if os.path.exists(CACHE):
        z = np.load(CACHE, allow_pickle=True)
        return z["EPrev"], z["ECur"], z["Y"]
    from sentence_transformers import SentenceTransformer

    prev = pairs["prev_text"].to_list()
    cur = pairs["clean"].to_list()
    Y = np.array(pairs["act"].to_list())
    uniq = sorted(set(prev) | set(cur))
    print(f"pairs {len(prev)}  unique texts {len(uniq)} -> embedding...", flush=True)
    m = SentenceTransformer(MODEL)
    E = m.encode(uniq, batch_size=128, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False)
    ix = {s: i for i, s in enumerate(uniq)}
    EPrev = E[[ix[s] for s in prev]]
    ECur = E[[ix[s] for s in cur]]
    np.savez_compressed(CACHE, EPrev=EPrev, ECur=ECur, Y=Y)
    return EPrev, ECur, Y


def probe(X, y):
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score

    return float(cross_val_score(LogisticRegression(max_iter=3000), X, y, cv=4, scoring="accuracy").mean())


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
    return (((Xu @ vt[:b].T) >= 0) @ (2 ** np.arange(b))).astype(np.int64)


def main():
    pairs = build_pairs()
    print(f"pairs: {pairs.height}  {dict(zip(*np.unique(pairs['act'].to_numpy(), return_counts=True)))}")
    EPrev, ECur, Y = embed(pairs)
    D = ECur - EPrev
    cos = (EPrev * ECur).sum(1)
    maj = max(np.mean(c == Y) for c in np.unique(Y))
    print(f"majority {maj:.3f}   cos(prior,cur) mean {cos.mean():.3f}")

    print(f"\n{'condition':<32} {'acc':>7} {'vs cur-only':>12}")
    cur_only = probe(ECur, Y)
    print(f"  {'current-turn only (baseline)':<30} {cur_only:7.3f} {'--':>12}")
    for nm, X in (
        ("prior-turn only", EPrev),
        ("DELTA direction", unit(D)),
        ("concat [prior, current]", np.hstack([EPrev, ECur])),
    ):
        a = probe(X, Y)
        print(f"  {nm:<30} {a:7.3f} {a - cur_only:+12.3f}")

    rng = np.random.default_rng(0)
    perm = rng.permutation(len(Y))
    a_fake = probe(unit(ECur - EPrev[perm]), Y)
    print(f"  {'DELTA vs RANDOM prior':<30} {a_fake:7.3f} {a_fake - cur_only:+12.3f}   <- decisive control")

    print("\nunsupervised MI(bucket; act), bits:")
    print(f"  current state {mi_bits(lsh(ECur, BITS), Y):.4f}   delta {mi_bits(lsh(D, BITS), Y):.4f}")

    qs = np.quantile(cos, [0.2, 0.45, 0.7, 0.9])
    print("\nwithin matched cos(prior,current) bands:")
    print(f"{'band':>14} {'n':>5} {'maj':>6} {'cur-only':>9} {'delta':>7} {'gain':>7}")
    for lo, hi in zip(qs[:-1], qs[1:]):
        m = (cos >= lo) & (cos < hi)
        if m.sum() < 300:
            continue
        y = Y[m]
        mj = max(np.mean(y == c) for c in np.unique(y))
        c1, d1 = probe(ECur[m], y), probe(unit(D[m]), y)
        print(f"{lo:6.3f}-{hi:6.3f} {int(m.sum()):5d} {mj:6.3f} {c1:9.3f} {d1:7.3f} {d1 - c1:+7.3f}")


if __name__ == "__main__":
    main()
