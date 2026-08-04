"""How expensive is search-then-filter at realistic selectivity? The number that decides
whether dyf needs a queryable attribute facet.

Uses a REAL categorical attribute (subcat) over the real catalog embeddings, and asks: to get
k=10 neighbours that satisfy `attr == v`, how far down the similarity ranking must we go?
That depth is the over-fetch factor pushdown would eliminate.
"""

import floret  # noqa: F401  (imported for parity with the clustering pipeline)
import numpy as np
import polars as pl

RNG = np.random.default_rng(42)
K = 10

d = pl.read_parquet("/tmp/claude/dyf_clusters.parquet")
print(f"catalog rows: {d.height:,}")

# Embed names the same way the clustering pipeline does, so the geometry is comparable.
names = [str(x or "").replace("\n", " ").lower() for x in d["bj_name"].to_list()]
import tempfile

with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as f:
    for n in names:
        f.write(n + "\n")
    path = f.name
m = floret.train_unsupervised(
    input=path,
    model="skipgram",
    dim=200,
    epoch=8,
    minCount=1,
    mode="floret",
    hashCount=2,
    bucket=120000,
    verbose=0,
    thread=8,
)
E = np.array([m.get_sentence_vector(n) for n in names], dtype=np.float32)
E /= np.linalg.norm(E, axis=1, keepdims=True) + 1e-12
print(f"embeddings: {E.shape}")

attr = np.array([str(x) for x in d["subcat"].to_list()])
vals, counts = np.unique(attr, return_counts=True)
sel = counts / len(attr)

print("\nattribute selectivity (subcat), by frequency band:")
for lo, hi, label in (
    (0.10, 1.01, ">10% of rows"),
    (0.01, 0.10, "1-10%"),
    (0.001, 0.01, "0.1-1%"),
    (0.0, 0.001, "<0.1%"),
):
    pick = (sel >= lo) & (sel < hi)
    print(f"  {label:<14} {pick.sum():>5} values covering {sel[pick].sum():.1%} of rows")

print(f"\nover-fetch needed for k={K} filtered results (200 queries per band):")
print(f"  {'selectivity band':<16}{'median depth':>14}{'p90 depth':>11}{'x over k':>10}{'fails':>7}")
print("  " + "-" * 60)

for lo, hi, label in ((0.10, 1.01, ">10%"), (0.01, 0.10, "1-10%"), (0.001, 0.01, "0.1-1%"), (0.0, 0.001, "<0.1%")):
    cand_vals = vals[(sel >= lo) & (sel < hi)]
    if len(cand_vals) == 0:
        continue
    depths, fails = [], 0
    for _ in range(200):
        v = cand_vals[RNG.integers(len(cand_vals))]
        qi = int(RNG.integers(len(E)))
        sims = E @ E[qi]
        order = np.argsort(-sims)
        hits = np.flatnonzero(attr[order] == v)
        if len(hits) < K:
            fails += 1
            continue
        depths.append(int(hits[K - 1]) + 1)
    if depths:
        med, p90 = int(np.median(depths)), int(np.percentile(depths, 90))
        print(f"  {label:<16}{med:>14,}{p90:>11,}{med / K:>9.0f}x{fails:>7}")

print("\n  ^ 'depth' = how many similarity-ranked items must be scanned before k=10 satisfy the")
print("    filter. That is exactly the work a predicate-pushdown search would skip.")
