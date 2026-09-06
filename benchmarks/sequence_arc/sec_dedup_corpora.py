"""Does the dedup win generalise, or is SEC's 29% duplicate rate special?

`dyf.dedup` was justified on ONE corpus. That is the exact failure that killed derived
`num_bits` earlier in this arc: a strong SEC result that died on 4 of 5 other corpora
(`sec_multicorpus_bits.py`). The library function is generic; the value claim is not, until
measured across corpora.

Reports, per corpus, the things a user deciding whether to enable `--dedup` needs:

  dup rate        fraction of points collapsed at cosine > 0.99
  file size       WEIGHED .dyf bytes before and after -- not a vector-payload estimate.
                  The SEC file saving (25.6%) was smaller than its point saving (29.4%)
                  because every tree node stores a dim-length centroid and leaf count fell
                  only ~10%, so this ratio has to be measured per corpus rather than
                  assumed equal to the dup rate.
  cluster max     largest duplicate cluster; a huge one means the corpus has a pathological
                  boilerplate blob worth looking at directly
  dedup seconds   ingest cost

Corpora span dimension, model and modality so a null result is informative rather than
ambiguous. GUDID also has an independent LSH dedup step in its own pipeline, so its rate
doubles as a cross-check on this implementation.
"""

import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sec_seqlib as S  # noqa: E402

from dyf.dedup import near_duplicate_clusters  # noqa: E402
from dyf.dyf_tree import build_dyf_tree  # noqa: E402
from dyf.lazy_index import write_lazy_index  # noqa: E402

PAPER = os.path.expanduser("~/Projects/semantic-proprioception-paper/experiments/data")
TMP = os.environ.get("TMPDIR", "/tmp")
MAX_N = 100000
THRESHOLD = 0.99
SEED = 42


def measure(X, name, out):
    X = np.ascontiguousarray(X.astype(np.float32))
    t0 = time.time()
    r = near_duplicate_clusters(X, threshold=THRESHOLD, seed=SEED)
    dedup_s = time.time() - t0
    csz = r.cluster_sizes()

    sizes = {}
    for tag, Xi, sf in [
        ("baseline", X, None),
        (
            "dedup",
            np.ascontiguousarray(X[r.representatives]),
            {"orig_index": r.representatives.astype(np.int64), "dup_members": r.member_field()},
        ),
    ]:
        path = os.path.join(TMP, f"dedupcorp_{name}_{tag}.dyf")
        tree = build_dyf_tree(Xi, max_depth=4, num_bits=4, min_leaf_size=16, seed=SEED)
        write_lazy_index(tree, Xi, path, stored_fields=sf)
        sizes[tag] = os.path.getsize(path)
        os.unlink(path)

    saving = 1 - sizes["dedup"] / sizes["baseline"]
    rec = {
        "n": int(len(X)),
        "dim": int(X.shape[1]),
        "dup_rate": float(r.removed_fraction),
        "cluster_max": int(csz.max()),
        "baseline_mb": sizes["baseline"] / 1e6,
        "dedup_mb": sizes["dedup"] / 1e6,
        "file_saving": float(saving),
        "dedup_s": dedup_s,
    }
    out[name] = rec
    print(
        f"{name:<20}{len(X):>9,}{X.shape[1]:>6}{100 * rec['dup_rate']:>9.1f}%"
        f"{rec['baseline_mb']:>11.1f}{rec['dedup_mb']:>10.1f}{100 * saving:>10.1f}%"
        f"{rec['cluster_max']:>9}{dedup_s:>8.1f}s",
        flush=True,
    )


def main():
    rng = np.random.default_rng(SEED)
    out = {}
    print(f"dedup at cosine > {THRESHOLD}, weighed .dyf files\n")
    print(
        f"{'corpus':<20}{'points':>9}{'dim':>6}{'dup rate':>10}"
        f"{'base MB':>11}{'dedup MB':>10}{'file save':>11}{'clus max':>9}{'dedup':>9}"
    )

    E, *_ = S.load()
    measure(E, "sec_768", out)

    for name, fname in [
        ("wikipedia_384", "wikipedia_embeddings.npy"),
        ("arxiv_384", "arxiv_embeddings.npy"),
        ("news_384", "news_embeddings.npy"),
        ("tweets_384", "tweets_embeddings.npy"),
        ("cmu_mocap_62", "cmu_mocap_features.npy"),
    ]:
        p = os.path.join(PAPER, fname)
        if not os.path.exists(p):
            print(f"{name:<20}  (missing {fname})")
            continue
        X = np.load(p).astype(np.float32)
        if len(X) > MAX_N:
            X = X[rng.choice(len(X), MAX_N, replace=False)]
        measure(X, name, out)

    # GUDID — a real shipped .dyf, and its own pipeline already runs an LSH dedup
    for cand in ("data/gudid_50k_titled.dyf", "demo/gudid_50k_titled.dyf"):
        if os.path.exists(cand):
            from dyf.lazy_index import LazyIndex

            idx = LazyIndex(cand)
            try:
                # ExtractedData is a TypedDict, so subscript rather than attribute access
                emb = idx.extract_all_fields()["embeddings"]
            except Exception as e:  # noqa: BLE001
                print(f"gudid: could not extract embeddings ({type(e).__name__}: {e})")
                break
            if emb is not None:
                measure(np.asarray(emb), "gudid", out)
            break

    print("\n" + "=" * 78)
    print("Does the file saving track the duplicate rate?")
    print("=" * 78)
    print(f"{'corpus':<20}{'dup rate':>10}{'file saving':>13}{'ratio':>9}")
    for k, v in out.items():
        ratio = v["file_saving"] / v["dup_rate"] if v["dup_rate"] > 1e-9 else float("nan")
        print(f"{k:<20}{100 * v['dup_rate']:>9.1f}%{100 * v['file_saving']:>12.1f}%{ratio:>9.2f}")
    rates = [v["dup_rate"] for v in out.values()]
    print(
        f"\nduplicate rate spans {100 * min(rates):.1f}% to {100 * max(rates):.1f}% "
        f"across {len(out)} corpora -- enabling --dedup is worth it exactly where the rate is high."
    )

    path = os.path.join(S.CACHE, "dedup_corpora_results.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nsaved -> {path}")


if __name__ == "__main__":
    main()
