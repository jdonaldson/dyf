"""
Meta Document Detection: Embedding-Based Quality Signal

Validates that neighbor_coherence() catches structural/meta documents
(dates, years, lists) without using title heuristics — purely from
embedding geometry.

Usage:
    python demo/meta_detection.py demo/wiki_simple_50k.parquet --sample 10000
"""

import argparse
import re
import time
from collections import Counter, defaultdict

import numpy as np
import polars as pl
from sklearn.neighbors import NearestNeighbors

from dyf.pca_tree import build_pca_tree, cut_tree_to_labels


# Title-based meta detection (for validation only)
DATE_RE = re.compile(
    r'^(January|February|March|April|May|June|July|August|September|'
    r'October|November|December)\s+\d+$')
YEAR_RE = re.compile(r'^\d{4}s?$')
LIST_RE = re.compile(r'^List of ')


def is_meta_title(title):
    """Heuristic: does this title look structural rather than topical?"""
    return bool(DATE_RE.match(title) or YEAR_RE.match(title)
                or LIST_RE.match(title))


def print_table(title, headers, rows):
    print(f"\n{'=' * 78}")
    print(f"  {title}")
    print(f"{'=' * 78}")
    widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(str(cell)))
    fmt = "  ".join(f"{{:<{w}}}" for w in widths)
    print(fmt.format(*headers))
    print(fmt.format(*["-" * w for w in widths]))
    for row in rows:
        print(fmt.format(*row))


def main():
    parser = argparse.ArgumentParser(
        description="Validate embedding-based meta document detection")
    parser.add_argument("parquet_path", help="Path to embeddings parquet")
    parser.add_argument("--sample", type=int, default=10000)
    parser.add_argument("--max-depth", type=int, default=12)
    parser.add_argument("--margin-pct", type=float, default=0.10)
    parser.add_argument("--n-clusters", type=int, default=50)
    parser.add_argument("--knn-k", type=int, default=20,
                        help="k for neighbor coherence")
    args = parser.parse_args()

    # ── Load & dedup ─────────────────────────────────────────────────────
    print(f"Loading {args.parquet_path}...")
    df = pl.read_parquet(args.parquet_path)
    if args.sample and args.sample < len(df):
        df = df.sample(args.sample, seed=42)

    titles_all = df["title"].to_list()
    embeddings_all = np.array(df["embedding"].to_list(), dtype=np.float32)

    from dyf.chunks import deduplicate_chunks, neighbor_coherence
    from dyf_rs import DensityClassifier as RustClassifier

    clf = RustClassifier(
        embedding_dim=embeddings_all.shape[1], num_bits=12, seed=42)
    clf.fit(embeddings_all)
    bucket_ids = clf.get_bucket_ids()
    dedup_mask = deduplicate_chunks(bucket_ids, np.asarray(titles_all))

    titles = [t for t, keep in zip(titles_all, dedup_mask) if keep]
    embeddings = embeddings_all[dedup_mask]
    n = len(embeddings)
    print(f"  {len(titles_all)} -> {n} after dedup")

    # ── Title-based meta labels (ground truth for validation) ────────────
    meta_mask = np.array([is_meta_title(t) for t in titles], dtype=bool)
    n_meta = meta_mask.sum()
    print(f"\n  Title-based meta documents: {n_meta}/{n} ({100*n_meta/n:.1f}%)")

    # ── kNN (large k for multi-scale coherence) ────────────────────────
    large_k = 100
    print(f"\nComputing k={large_k} nearest neighbors...")
    t0 = time.time()
    nn = NearestNeighbors(n_neighbors=large_k + 1, metric='cosine')
    nn.fit(embeddings)
    _, knn_indices_full = nn.kneighbors(embeddings)
    knn_indices_full = knn_indices_full[:, 1:]  # drop self
    knn_indices = knn_indices_full[:, :args.knn_k]
    print(f"  Done in {time.time() - t0:.1f}s")

    # ── Multi-scale neighbor coherence ───────────────────────────────────
    k_values = [10, 20, 50, 100]
    coherence_by_k = {}
    for k in k_values:
        print(f"  Computing coherence at k={k}...")
        t0 = time.time()
        coherence_by_k[k] = neighbor_coherence(
            embeddings, knn_indices_full[:, :k])
        print(f"    Done in {time.time() - t0:.1f}s")

    coherence = coherence_by_k[args.knn_k]

    # Coherence persistence: how much does coherence drop from k=10 to k=100?
    # Meta docs stay high (huge echo chamber), topical docs drop.
    coh_drop = coherence_by_k[10] - coherence_by_k[100]
    # Negative drop = coherence INCREASES at larger k (very unusual)
    # Small drop = persistent coherence = meta echo chamber
    # Large drop = coherence fades = normal topical document

    # ── Validate: does coherence separate meta from topical? ─────────────
    print(f"\n{'=' * 78}")
    print("  Neighbor Coherence: Meta vs Topical Documents")
    print(f"{'=' * 78}")

    coh_meta = coherence[meta_mask]
    coh_topical = coherence[~meta_mask]

    print(f"\n  Topical documents (n={len(coh_topical)}):")
    print(f"    Coherence: mean={coh_topical.mean():.4f}, "
          f"median={np.median(coh_topical):.4f}, "
          f"std={coh_topical.std():.4f}")

    print(f"  Meta documents (n={len(coh_meta)}):")
    print(f"    Coherence: mean={coh_meta.mean():.4f}, "
          f"median={np.median(coh_meta):.4f}, "
          f"std={coh_meta.std():.4f}")

    # Separation
    if len(coh_meta) > 0 and len(coh_topical) > 0:
        # Cohen's d effect size
        pooled_std = np.sqrt((coh_topical.var() + coh_meta.var()) / 2)
        cohens_d = (coh_topical.mean() - coh_meta.mean()) / pooled_std
        print(f"\n  Separation (Cohen's d): {cohens_d:.2f}")
        print(f"  (>0.8 is 'large effect', >1.2 is 'very large')")

    # ── Classification performance at various thresholds ─────────────────
    print(f"\n{'=' * 78}")
    print("  Detection performance (coherence > threshold → predict meta)")
    print(f"  Meta docs have HIGH coherence: structural echo chambers")
    print(f"{'=' * 78}")

    percentiles = [70, 75, 80, 85, 90, 95]
    rows = []
    for pct in percentiles:
        threshold = np.percentile(coherence, pct)
        predicted_meta = coherence > threshold
        tp = np.sum(predicted_meta & meta_mask)
        fp = np.sum(predicted_meta & ~meta_mask)
        fn = np.sum(~predicted_meta & meta_mask)
        tn = np.sum(~predicted_meta & ~meta_mask)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = (2 * precision * recall / (precision + recall)
              if (precision + recall) > 0 else 0)
        rows.append([
            f"p{pct} ({threshold:.4f})",
            f"{int(tp + fp)}",
            f"{precision:.2f}",
            f"{recall:.2f}",
            f"{f1:.2f}",
        ])

    print_table(
        "Coherence threshold → meta classification",
        ["Threshold", "Flagged", "Precision", "Recall", "F1"],
        rows,
    )

    # ── Multi-scale coherence ────────────────────────────────────────────
    print(f"\n{'=' * 78}")
    print("  Multi-scale coherence: mean by group at each k")
    print(f"{'=' * 78}")

    print(f"\n  {'k':>5s}  {'Topical':>10s}  {'Meta':>10s}  {'Gap':>8s}")
    print(f"  {'-----':>5s}  {'----------':>10s}  {'----------':>10s}  {'--------':>8s}")
    for k in k_values:
        t_mean = coherence_by_k[k][~meta_mask].mean()
        m_mean = coherence_by_k[k][meta_mask].mean()
        print(f"  k={k:<3d}  {t_mean:.4f}      {m_mean:.4f}      {m_mean - t_mean:+.4f}")

    # ── Coherence persistence (drop from k=10 to k=100) ──────────────────
    print(f"\n{'=' * 78}")
    print("  Coherence persistence: drop from k=10 to k=100")
    print(f"  Small drop = echo chamber (meta), large drop = normal topical")
    print(f"{'=' * 78}")

    drop_meta = coh_drop[meta_mask]
    drop_topical = coh_drop[~meta_mask]

    print(f"\n  Topical: mean={drop_topical.mean():.4f}, "
          f"median={np.median(drop_topical):.4f}")
    print(f"  Meta:    mean={drop_meta.mean():.4f}, "
          f"median={np.median(drop_meta):.4f}")

    if drop_meta.std() > 0 and drop_topical.std() > 0:
        pooled = np.sqrt((drop_topical.var() + drop_meta.var()) / 2)
        d = (drop_topical.mean() - drop_meta.mean()) / pooled
        print(f"  Cohen's d: {d:.2f}")

    # Classification using persistence
    print(f"\n  Detection using persistence (drop < threshold → predict meta):")
    persist_rows = []
    for pct in [5, 10, 15, 20, 25, 30]:
        threshold = np.percentile(coh_drop, pct)
        predicted = coh_drop < threshold
        tp = np.sum(predicted & meta_mask)
        fp = np.sum(predicted & ~meta_mask)
        fn = np.sum(~predicted & meta_mask)
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
        persist_rows.append([
            f"p{pct} ({threshold:.4f})",
            f"{int(tp + fp)}", f"{prec:.2f}", f"{rec:.2f}", f"{f1:.2f}",
        ])

    print_table(
        "Persistence threshold → meta classification",
        ["Threshold", "Flagged", "Precision", "Recall", "F1"],
        persist_rows,
    )

    # ── Combined signal: high coherence @ k=100 + low drop ───────────────
    print(f"\n{'=' * 78}")
    print("  Combined signal: coherence_k100 - alpha * drop")
    print(f"{'=' * 78}")

    coh100 = coherence_by_k[100]
    # Combined: high k=100 coherence AND low drop → meta
    # Score = coh100 - 0.5 * drop  (penalize documents whose coherence fades)
    for alpha in [0.0, 0.5, 1.0, 2.0]:
        combined = coh100 + alpha * (coherence_by_k[10] - coh100)
        # equivalently: (1+alpha)*coh100 when drop=0, less when drop>0
        # Actually: combined = coh100 + alpha * (coh10 - coh100)
        #   = (1-alpha)*coh100 + alpha*coh10
        # Just blend k=10 and k=100 coherence.
        # For meta detection, we want HIGH combined.
        comb_rows = []
        for pct in [75, 80, 85, 90]:
            threshold = np.percentile(combined, pct)
            predicted = combined > threshold
            tp = np.sum(predicted & meta_mask)
            fp = np.sum(predicted & ~meta_mask)
            fn = np.sum(~predicted & meta_mask)
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0
            rec = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
            comb_rows.append([
                f"p{pct}", f"{int(tp + fp)}", f"{prec:.2f}",
                f"{rec:.2f}", f"{f1:.2f}",
            ])

        print(f"\n  alpha={alpha:.1f} → blend = "
              f"{1-alpha:.1f}*coh100 + {alpha:.1f}*coh10")
        print_table(
            f"Combined (alpha={alpha}) → meta",
            ["Pctl", "Flagged", "Prec", "Recall", "F1"],
            comb_rows,
        )

    # ── Distribution: coherence percentiles by group ─────────────────────
    print(f"\n{'=' * 78}")
    print("  Coherence distribution by group (k=20)")
    print(f"{'=' * 78}")

    for pct in [10, 25, 50, 75, 90]:
        t_val = np.percentile(coh_topical, pct) if len(coh_topical) > 0 else 0
        m_val = np.percentile(coh_meta, pct) if len(coh_meta) > 0 else 0
        print(f"  p{pct:>2d}: topical={t_val:.4f}  meta={m_val:.4f}")

    # ── Bottom-20 by coherence: what do they look like? ──────────────────
    print(f"\n{'=' * 78}")
    print("  Lowest coherence documents (most likely meta)")
    print(f"{'=' * 78}")

    order = np.argsort(coherence)
    for rank in range(20):
        idx = order[rank]
        is_m = "META" if meta_mask[idx] else "    "
        print(f"  {rank+1:>3d}. [{is_m}] coh={coherence[idx]:.4f}  "
              f"\"{titles[idx][:55]}\"")

    # ── Highest coherence: what do they look like? ───────────────────────
    print(f"\n{'=' * 78}")
    print("  Highest coherence documents (most topically focused)")
    print(f"{'=' * 78}")

    for rank in range(20):
        idx = order[-(rank + 1)]
        is_m = "META" if meta_mask[idx] else "    "
        print(f"  {rank+1:>3d}. [{is_m}] coh={coherence[idx]:.4f}  "
              f"\"{titles[idx][:55]}\"")

    # ── Cluster-level analysis ───────────────────────────────────────────
    print(f"\nCutting PCA tree to {args.n_clusters} clusters...")
    tree = build_pca_tree(embeddings, args.max_depth)
    cluster_labels = cut_tree_to_labels(
        tree, args.max_depth, n, args.n_clusters)
    n_actual = len(set(cluster_labels.tolist()))

    # Remap to contiguous
    unique_labels = sorted(set(cluster_labels.tolist()))
    label_map = {old: new for new, old in enumerate(unique_labels)}
    cluster_labels = np.array(
        [label_map[c] for c in cluster_labels], dtype=int)

    print(f"\n{'=' * 78}")
    print("  Per-cluster quality signal")
    print(f"{'=' * 78}")

    rows = []
    for c in range(n_actual):
        mask = cluster_labels == c
        c_titles = [titles[i] for i in range(n) if cluster_labels[i] == c]
        c_coherence = coherence[mask]
        c_meta_frac = meta_mask[mask].mean()
        # Sample titles for this cluster
        sample = c_titles[:3]
        sample_str = " | ".join(t[:20] for t in sample)
        rows.append((
            c_coherence.mean(), c_meta_frac, mask.sum(),
            c, sample_str
        ))

    # Sort by mean coherence ascending (worst clusters first)
    rows.sort()

    print(f"\n  {'Clst':>4s}  {'Size':>5s}  {'Coh':>6s}  {'Meta%':>6s}  Samples")
    print(f"  {'----':>4s}  {'-----':>5s}  {'------':>6s}  {'------':>6s}  -------")
    for coh_val, meta_frac, size, c, samples in rows:
        print(f"  {c:>4d}  {size:>5d}  {coh_val:.4f}  {100*meta_frac:>5.1f}%  {samples}")

    # ── Correlation: cluster mean coherence vs cluster meta fraction ─────
    cluster_coh = np.array([r[0] for r in rows])
    cluster_meta = np.array([r[1] for r in rows])
    if cluster_meta.std() > 0 and cluster_coh.std() > 0:
        corr = np.corrcoef(cluster_coh, cluster_meta)[0, 1]
        print(f"\n  Correlation(cluster_coherence, cluster_meta_fraction): {corr:.3f}")

    print()


if __name__ == "__main__":
    main()
