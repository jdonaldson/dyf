"""Benchmark: DYF LazyIndex vs FAISS vs hnswlib

Compares startup time, query latency, recall@10, build time, memory, and file size
across DYF LazyIndex, FAISS (IVFFlat, HNSW), and hnswlib on synthetic embeddings.

NOTE: DYF's tree builder uses skip_isolation=True to bypass the expensive
isolation_scores (O(n × 1000 × d)) and stability_scores computation at each
tree node, since only bucket_ids, centroid_similarities, and hyperplanes are
needed for the index.

Usage:
    pip install faiss-cpu hnswlib psutil matplotlib
    python benchmarks/bench_lazy_index.py
"""

import gc
import os
import sys
import tempfile
import time

import numpy as np
import psutil

# ---------------------------------------------------------------------------
# 1. Data generation
# ---------------------------------------------------------------------------

N_ITEMS = 100_000
EMBEDDING_DIM = 128
N_CLUSTERS = 100
N_QUERIES_KNOWN = 1000
N_QUERIES_RANDOM = 1000
K = 10
SEED = 42


def generate_clustered_data(n_items, dim, n_clusters, seed):
    """Generate clustered unit-norm embeddings on the sphere."""
    rng = np.random.default_rng(seed)

    # Random cluster centers on the unit sphere
    centers = rng.standard_normal((n_clusters, dim)).astype(np.float32)
    centers /= np.linalg.norm(centers, axis=1, keepdims=True)

    # Assign items to clusters uniformly
    assignments = rng.integers(0, n_clusters, size=n_items)

    # Generate items as center + noise, then normalize
    noise = rng.standard_normal((n_items, dim)).astype(np.float32) * 0.15
    embeddings = centers[assignments] + noise
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings /= np.maximum(norms, 1e-10)

    return embeddings


def compute_ground_truth(embeddings, queries, k):
    """Brute-force cosine top-k for each query (batched matrix multiply)."""
    # embeddings are unit-norm, so dot product = cosine similarity
    # Process in chunks to avoid OOM on large matrices
    gt_indices = np.empty((len(queries), k), dtype=np.int64)
    gt_scores = np.empty((len(queries), k), dtype=np.float32)

    chunk_size = 100
    for start in range(0, len(queries), chunk_size):
        end = min(start + chunk_size, len(queries))
        q_batch = queries[start:end]
        # (batch, dim) @ (dim, n_items) -> (batch, n_items)
        sims = q_batch @ embeddings.T
        for j in range(end - start):
            row = sims[j]
            topk = np.argpartition(-row, k)[:k]
            topk = topk[np.argsort(-row[topk])]
            gt_indices[start + j] = topk
            gt_scores[start + j] = row[topk]

    return gt_indices, gt_scores


# ---------------------------------------------------------------------------
# Timing / memory helpers
# ---------------------------------------------------------------------------

def get_rss_mb():
    gc.collect()
    return psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)


def file_size_mb(path):
    return os.path.getsize(path) / (1024 * 1024)


def compute_recall(gt_indices, result_indices, k):
    """Mean recall@k: fraction of true top-k found per query."""
    recalls = []
    for gt_row, res_row in zip(gt_indices, result_indices):
        gt_set = set(int(x) for x in gt_row[:k])
        res_set = set(int(x) for x in res_row[:k])
        recalls.append(len(gt_set & res_set) / k)
    return np.mean(recalls)


def measure_query_latency(search_fn, queries, warmup=10):
    """Measure per-query latency (p50, p95, p99) and total throughput."""
    # Warmup
    for q in queries[:warmup]:
        search_fn(q)

    latencies = []
    for q in queries:
        t0 = time.perf_counter()
        search_fn(q)
        latencies.append(time.perf_counter() - t0)

    latencies = np.array(latencies) * 1000  # ms
    total_time = latencies.sum() / 1000  # seconds
    qps = len(queries) / total_time

    return {
        'p50_ms': float(np.percentile(latencies, 50)),
        'p95_ms': float(np.percentile(latencies, 95)),
        'p99_ms': float(np.percentile(latencies, 99)),
        'qps': float(qps),
    }


# ---------------------------------------------------------------------------
# 3. DYF LazyIndex benchmark
# ---------------------------------------------------------------------------

def bench_dyf(embeddings, queries, gt_indices, tmpdir, fit_method='pca',
              num_bits=3, max_depth=3):
    """Benchmark DYF LazyIndex at multiple nprobe settings."""
    from dyf import build_dyf_tree, write_lazy_index, LazyIndex

    label = f'DYF-{fit_method.upper()}-{num_bits}b-d{max_depth}'
    results = []
    index_path = os.path.join(tmpdir, f'index_{fit_method}_{num_bits}b_d{max_depth}.dyf')

    # Build
    print(f"  Building {label} tree...", flush=True)
    t0 = time.perf_counter()
    tree = build_dyf_tree(embeddings, max_depth=max_depth, num_bits=num_bits,
                          min_leaf_size=8, seed=SEED,
                          fit_method=fit_method)
    build_tree_time = time.perf_counter() - t0
    print(f"  Tree built in {build_tree_time:.1f}s", flush=True)

    print("  Writing lazy index...", flush=True)
    t0 = time.perf_counter()
    write_lazy_index(tree, embeddings, index_path,
                     compression='zstd', quantization='float16',
                     build_params={'max_depth': max_depth, 'num_bits': num_bits,
                                   'min_leaf_size': 8, 'seed': SEED})
    write_time = time.perf_counter() - t0
    total_build = build_tree_time + write_time

    fsize = file_size_mb(index_path)

    for nprobe in [1, 3, 10, 25, 50]:
        print(f"  {label} nprobe={nprobe}...", flush=True)

        # Startup
        gc.collect()
        rss_before = get_rss_mb()
        t0 = time.perf_counter()
        idx = LazyIndex(index_path)
        startup_time = time.perf_counter() - t0
        rss_after = get_rss_mb()

        # Query latency
        def search_fn(q, _idx=idx, _nprobe=nprobe):
            return _idx.search(q, k=K, nprobe=_nprobe)

        lat = measure_query_latency(search_fn, queries)

        # Recall
        all_results = []
        for q in queries:
            indices, scores = idx.search(q, k=K, nprobe=nprobe)
            padded = np.full(K, -1, dtype=np.int64)
            padded[:len(indices)] = indices[:K]
            all_results.append(padded)
        result_indices = np.array(all_results)
        recall = compute_recall(gt_indices, result_indices, K)

        idx.close()

        results.append({
            'system': f'{label} nprobe={nprobe}',
            'build_s': total_build,
            'file_mb': fsize,
            'startup_ms': startup_time * 1000,
            'rss_delta_mb': rss_after - rss_before,
            'recall@10': recall,
            **lat,
        })

    return results


def bench_dyf_ivf(embeddings, queries, gt_indices, tmpdir, fit_method='itq',
                  num_bits=3, max_depth=3):
    """Benchmark DYF LazyIndex IVF-style search (centroid routing)."""
    from dyf import build_dyf_tree, write_lazy_index, LazyIndex

    label = f'DYF-IVF-{fit_method.upper()}-{num_bits}b-d{max_depth}'
    results = []
    index_path = os.path.join(tmpdir, f'index_ivf_{fit_method}_{num_bits}b_d{max_depth}.dyf')

    # Build
    print(f"  Building {label} tree...", flush=True)
    t0 = time.perf_counter()
    tree = build_dyf_tree(embeddings, max_depth=max_depth, num_bits=num_bits,
                          min_leaf_size=8, seed=SEED,
                          fit_method=fit_method)
    build_tree_time = time.perf_counter() - t0
    print(f"  Tree built in {build_tree_time:.1f}s", flush=True)

    print("  Writing lazy index...", flush=True)
    t0 = time.perf_counter()
    write_lazy_index(tree, embeddings, index_path,
                     compression='zstd', quantization='float16',
                     build_params={'max_depth': max_depth, 'num_bits': num_bits,
                                   'min_leaf_size': 8, 'seed': SEED})
    write_time = time.perf_counter() - t0
    total_build = build_tree_time + write_time

    fsize = file_size_mb(index_path)

    for nprobe in [1, 3, 10, 25, 50]:
        print(f"  {label} nprobe={nprobe}...", flush=True)

        gc.collect()
        rss_before = get_rss_mb()
        t0 = time.perf_counter()
        idx = LazyIndex(index_path)
        startup_time = time.perf_counter() - t0
        rss_after = get_rss_mb()

        def search_fn(q, _idx=idx, _nprobe=nprobe):
            return _idx.search_ivf(q, k=K, nprobe=_nprobe)

        lat = measure_query_latency(search_fn, queries)

        # Recall
        all_results_list = []
        for q in queries:
            indices, scores = idx.search_ivf(q, k=K, nprobe=nprobe)
            padded = np.full(K, -1, dtype=np.int64)
            padded[:len(indices)] = indices[:K]
            all_results_list.append(padded)
        result_indices = np.array(all_results_list)
        recall = compute_recall(gt_indices, result_indices, K)

        idx.close()

        results.append({
            'system': f'{label} nprobe={nprobe}',
            'build_s': total_build,
            'file_mb': fsize,
            'startup_ms': startup_time * 1000,
            'rss_delta_mb': rss_after - rss_before,
            'recall@10': recall,
            **lat,
        })

    return results


# ---------------------------------------------------------------------------
# 4. FAISS IVFFlat benchmark
# ---------------------------------------------------------------------------

def bench_faiss_ivf(embeddings, queries, gt_indices, tmpdir):
    """Benchmark FAISS IVFFlat at multiple nprobe settings."""
    import faiss

    results = []
    index_path = os.path.join(tmpdir, 'faiss_ivf.index')
    nlist = int(np.sqrt(N_ITEMS))  # sqrt(n) heuristic

    # Build
    print("  Building FAISS IVFFlat...", flush=True)
    t0 = time.perf_counter()
    quantizer = faiss.IndexFlatIP(EMBEDDING_DIM)
    index = faiss.IndexIVFFlat(quantizer, EMBEDDING_DIM, nlist,
                                faiss.METRIC_INNER_PRODUCT)
    index.train(embeddings)
    index.add(embeddings)
    build_time = time.perf_counter() - t0

    # Write
    t0 = time.perf_counter()
    faiss.write_index(index, index_path)
    write_time = time.perf_counter() - t0
    total_build = build_time + write_time
    del index
    gc.collect()

    fsize = file_size_mb(index_path)

    for nprobe in [1, 4, 16]:
        print(f"  FAISS IVF nprobe={nprobe}...", flush=True)

        gc.collect()
        rss_before = get_rss_mb()
        t0 = time.perf_counter()
        idx = faiss.read_index(index_path)
        startup_time = time.perf_counter() - t0
        rss_after = get_rss_mb()

        idx.nprobe = nprobe

        def search_fn(q, _idx=idx):
            return _idx.search(q.reshape(1, -1), K)

        lat = measure_query_latency(search_fn, queries)

        # Recall (batch)
        D, I = idx.search(queries, K)
        recall = compute_recall(gt_indices, I, K)

        del idx
        gc.collect()

        results.append({
            'system': f'FAISS IVF nprobe={nprobe}',
            'build_s': total_build,
            'file_mb': fsize,
            'startup_ms': startup_time * 1000,
            'rss_delta_mb': rss_after - rss_before,
            'recall@10': recall,
            **lat,
        })

    return results


# ---------------------------------------------------------------------------
# 5. FAISS HNSW benchmark
# ---------------------------------------------------------------------------

def bench_faiss_hnsw(embeddings, queries, gt_indices, tmpdir):
    """Benchmark FAISS HNSW at multiple efSearch settings."""
    import faiss

    results = []
    index_path = os.path.join(tmpdir, 'faiss_hnsw.index')

    # Build
    print("  Building FAISS HNSW...", flush=True)
    t0 = time.perf_counter()
    index = faiss.IndexHNSWFlat(EMBEDDING_DIM, 16, faiss.METRIC_INNER_PRODUCT)
    index.hnsw.efConstruction = 200
    index.add(embeddings)
    build_time = time.perf_counter() - t0

    t0 = time.perf_counter()
    faiss.write_index(index, index_path)
    write_time = time.perf_counter() - t0
    total_build = build_time + write_time
    del index
    gc.collect()

    fsize = file_size_mb(index_path)

    for ef_search in [32, 64, 128]:
        print(f"  FAISS HNSW efSearch={ef_search}...", flush=True)

        gc.collect()
        rss_before = get_rss_mb()
        t0 = time.perf_counter()
        idx = faiss.read_index(index_path)
        startup_time = time.perf_counter() - t0
        rss_after = get_rss_mb()

        idx.hnsw.efSearch = ef_search

        def search_fn(q, _idx=idx):
            return _idx.search(q.reshape(1, -1), K)

        lat = measure_query_latency(search_fn, queries)

        # Recall
        D, I = idx.search(queries, K)
        recall = compute_recall(gt_indices, I, K)

        del idx
        gc.collect()

        results.append({
            'system': f'FAISS HNSW ef={ef_search}',
            'build_s': total_build,
            'file_mb': fsize,
            'startup_ms': startup_time * 1000,
            'rss_delta_mb': rss_after - rss_before,
            'recall@10': recall,
            **lat,
        })

    return results


# ---------------------------------------------------------------------------
# 6. hnswlib benchmark
# ---------------------------------------------------------------------------

def bench_hnswlib(embeddings, queries, gt_indices, tmpdir):
    """Benchmark hnswlib at multiple ef settings."""
    import hnswlib

    results = []
    index_path = os.path.join(tmpdir, 'hnsw.bin')

    # Build
    print("  Building hnswlib...", flush=True)
    t0 = time.perf_counter()
    idx = hnswlib.Index(space='cosine', dim=EMBEDDING_DIM)
    idx.init_index(max_elements=N_ITEMS, ef_construction=200, M=16)
    idx.add_items(embeddings, np.arange(N_ITEMS))
    build_time = time.perf_counter() - t0

    t0 = time.perf_counter()
    idx.save_index(index_path)
    write_time = time.perf_counter() - t0
    total_build = build_time + write_time
    del idx
    gc.collect()

    fsize = file_size_mb(index_path)

    for ef in [32, 64, 128]:
        print(f"  hnswlib ef={ef}...", flush=True)

        gc.collect()
        rss_before = get_rss_mb()
        t0 = time.perf_counter()
        idx = hnswlib.Index(space='cosine', dim=EMBEDDING_DIM)
        idx.load_index(index_path, max_elements=N_ITEMS)
        startup_time = time.perf_counter() - t0
        rss_after = get_rss_mb()

        idx.set_ef(ef)

        def search_fn(q, _idx=idx):
            labels, distances = _idx.knn_query(q.reshape(1, -1), k=K)
            return labels, distances

        lat = measure_query_latency(search_fn, queries)

        # Recall — hnswlib cosine returns 1-cosine as distance
        labels, _ = idx.knn_query(queries, k=K)
        recall = compute_recall(gt_indices, labels, K)

        del idx
        gc.collect()

        results.append({
            'system': f'hnswlib ef={ef}',
            'build_s': total_build,
            'file_mb': fsize,
            'startup_ms': startup_time * 1000,
            'rss_delta_mb': rss_after - rss_before,
            'recall@10': recall,
            **lat,
        })

    return results


# ---------------------------------------------------------------------------
# 7. Summary table
# ---------------------------------------------------------------------------

def print_summary(all_results):
    """Print results as a markdown table."""
    import polars as pl

    df = pl.DataFrame(all_results)

    # Reorder columns
    cols = ['system', 'build_s', 'file_mb', 'startup_ms', 'rss_delta_mb',
            'recall@10', 'p50_ms', 'p95_ms', 'p99_ms', 'qps']
    df = df.select(cols)

    # Format for display
    fmt = df.with_columns([
        pl.col('build_s').round(2),
        pl.col('file_mb').round(1),
        pl.col('startup_ms').round(2),
        pl.col('rss_delta_mb').round(1),
        pl.col('recall@10').round(4),
        pl.col('p50_ms').round(3),
        pl.col('p95_ms').round(3),
        pl.col('p99_ms').round(3),
        pl.col('qps').round(0),
    ])

    print("\n## Benchmark Results\n")
    print(f"Dataset: {N_ITEMS:,} items, {EMBEDDING_DIM}-dim, "
          f"{N_CLUSTERS} clusters, {len(all_results[0].get('queries', [])) if False else N_QUERIES_KNOWN + N_QUERIES_RANDOM} queries\n")

    # Print as markdown table
    header = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join(["---"] * len(cols)) + " |"
    print(header)
    print(sep)
    for row in fmt.iter_rows(named=True):
        vals = [str(row[c]) for c in cols]
        print("| " + " | ".join(vals) + " |")

    print()
    return df


# ---------------------------------------------------------------------------
# 8. Plots (optional)
# ---------------------------------------------------------------------------

def plot_results(all_results, output_dir):
    """Generate recall-vs-latency and startup bar charts."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available — skipping plots.")
        return

    # Color/marker by system family
    family_style = {
        'DYF-ITQ':     {'color': '#2196F3', 'marker': 'o'},
        'DYF-IVF':     {'color': '#E91E63', 'marker': 'P'},
        'FAISS IVF':   {'color': '#FF9800', 'marker': 's'},
        'FAISS HNSW':  {'color': '#4CAF50', 'marker': '^'},
        'hnswlib':     {'color': '#9C27B0', 'marker': 'D'},
    }

    def get_family(name):
        if name.startswith('DYF-IVF'): return 'DYF-IVF'
        if name.startswith('DYF'): return 'DYF-ITQ'
        if name.startswith('FAISS IVF'): return 'FAISS IVF'
        if name.startswith('FAISS HNSW'): return 'FAISS HNSW'
        return 'hnswlib'

    # --- Recall vs Latency (p50) ---
    fig, ax = plt.subplots(figsize=(8, 5))
    for family, style in family_style.items():
        xs, ys, labels = [], [], []
        for r in all_results:
            if get_family(r['system']) == family:
                xs.append(r['p50_ms'])
                ys.append(r['recall@10'])
                labels.append(r['system'])
        if xs:
            ax.scatter(xs, ys, label=family, **style, s=80, zorder=3)
            for x, y, lab in zip(xs, ys, labels):
                ax.annotate(lab.split('=')[-1] if '=' in lab else lab,
                           (x, y), textcoords="offset points",
                           xytext=(6, 4), fontsize=7)

    ax.set_xlabel('Query Latency p50 (ms)')
    ax.set_ylabel('Recall@10')
    ax.set_title(f'Recall vs Latency — {N_ITEMS // 1000}K items, {EMBEDDING_DIM}d')
    ax.legend()
    ax.grid(True, alpha=0.3)
    path = os.path.join(output_dir, 'recall_vs_latency.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    print(f"Saved: {path}")
    plt.close(fig)

    # --- Startup Time Bar Chart ---
    fig, ax = plt.subplots(figsize=(8, 4))
    names = [r['system'] for r in all_results]
    startups = [r['startup_ms'] for r in all_results]
    colors = [family_style[get_family(n)]['color'] for n in names]

    # Deduplicate: same system family has same startup (take first)
    seen = {}
    deduped_names, deduped_startups, deduped_colors = [], [], []
    for n, s, c in zip(names, startups, colors):
        fam = get_family(n)
        if fam not in seen:
            seen[fam] = True
            deduped_names.append(fam)
            deduped_startups.append(s)
            deduped_colors.append(c)

    bars = ax.barh(deduped_names, deduped_startups, color=deduped_colors)
    ax.set_xlabel('Startup Time (ms)')
    ax.set_title('Index Startup Time (open file to first query)')
    for bar, val in zip(bars, deduped_startups):
        ax.text(bar.get_width() + max(deduped_startups) * 0.02, bar.get_y() + bar.get_height()/2,
                f'{val:.1f} ms', va='center', fontsize=9)
    ax.grid(True, alpha=0.3, axis='x')
    path = os.path.join(output_dir, 'startup_time.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    print(f"Saved: {path}")
    plt.close(fig)

    # --- Build Time Bar Chart ---
    fig, ax = plt.subplots(figsize=(8, 4))
    seen = {}
    deduped_names, deduped_builds, deduped_colors = [], [], []
    for n, c in zip(names, colors):
        fam = get_family(n)
        if fam not in seen:
            r = next(r for r in all_results if get_family(r['system']) == fam)
            seen[fam] = True
            deduped_names.append(fam)
            deduped_builds.append(r['build_s'])
            deduped_colors.append(c)

    bars = ax.barh(deduped_names, deduped_builds, color=deduped_colors)
    ax.set_xlabel('Build + Serialize Time (s)')
    ax.set_title('Index Build Time')
    for bar, val in zip(bars, deduped_builds):
        ax.text(bar.get_width() + max(deduped_builds) * 0.02, bar.get_y() + bar.get_height()/2,
                f'{val:.1f}s', va='center', fontsize=9)
    ax.grid(True, alpha=0.3, axis='x')
    path = os.path.join(output_dir, 'build_time.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    print(f"Saved: {path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print(f"=== DYF LazyIndex Benchmark ===")
    print(f"Items: {N_ITEMS:,}  Dim: {EMBEDDING_DIM}  Clusters: {N_CLUSTERS}")
    print(f"Queries: {N_QUERIES_KNOWN} known + {N_QUERIES_RANDOM} random = "
          f"{N_QUERIES_KNOWN + N_QUERIES_RANDOM}")
    print()

    # Generate data
    print("Generating clustered embeddings...", flush=True)
    embeddings = generate_clustered_data(N_ITEMS, EMBEDDING_DIM, N_CLUSTERS, SEED)

    # Sample queries: known items + random
    rng = np.random.default_rng(SEED + 1)
    known_idx = rng.choice(N_ITEMS, size=N_QUERIES_KNOWN, replace=False)
    queries_known = embeddings[known_idx].copy()
    queries_random = rng.standard_normal(
        (N_QUERIES_RANDOM, EMBEDDING_DIM)).astype(np.float32)
    queries_random /= np.linalg.norm(queries_random, axis=1, keepdims=True)
    queries = np.vstack([queries_known, queries_random])

    # Ground truth
    print("Computing brute-force ground truth...", flush=True)
    t0 = time.perf_counter()
    gt_indices, gt_scores = compute_ground_truth(embeddings, queries, K)
    gt_time = time.perf_counter() - t0
    print(f"  Ground truth computed in {gt_time:.1f}s\n")

    all_results = []

    with tempfile.TemporaryDirectory() as tmpdir:
        # DYF LSH routing (baseline)
        print("[1/5] DYF-ITQ LSH routing")
        try:
            r = bench_dyf(embeddings, queries, gt_indices, tmpdir,
                          fit_method='itq', num_bits=3, max_depth=3)
            all_results.extend(r)
        except Exception as e:
            print(f"  FAILED: {e}")

        # DYF IVF routing (centroid-based, same index format)
        print("[2/5] DYF-ITQ IVF routing")
        try:
            r = bench_dyf_ivf(embeddings, queries, gt_indices, tmpdir,
                              fit_method='itq', num_bits=3, max_depth=3)
            all_results.extend(r)
        except Exception as e:
            print(f"  FAILED: {e}")

        # FAISS IVFFlat
        print("[3/5] FAISS IVFFlat")
        try:
            faiss_ivf_results = bench_faiss_ivf(embeddings, queries, gt_indices, tmpdir)
            all_results.extend(faiss_ivf_results)
        except ImportError:
            print("  SKIPPED: faiss-cpu not installed")
        except Exception as e:
            print(f"  FAILED: {e}")

        # FAISS HNSW
        print("[4/5] FAISS HNSW")
        try:
            faiss_hnsw_results = bench_faiss_hnsw(embeddings, queries, gt_indices, tmpdir)
            all_results.extend(faiss_hnsw_results)
        except ImportError:
            print("  SKIPPED: faiss-cpu not installed")
        except Exception as e:
            print(f"  FAILED: {e}")

        # hnswlib
        print("[5/5] hnswlib")
        try:
            hnswlib_results = bench_hnswlib(embeddings, queries, gt_indices, tmpdir)
            all_results.extend(hnswlib_results)
        except ImportError:
            print("  SKIPPED: hnswlib not installed")
        except Exception as e:
            print(f"  FAILED: {e}")

    if not all_results:
        print("No benchmarks completed successfully.")
        return

    # Summary
    df = print_summary(all_results)

    # Plots
    output_dir = os.path.dirname(os.path.abspath(__file__))
    plot_results(all_results, output_dir)

    print("Done.")


if __name__ == '__main__':
    main()
