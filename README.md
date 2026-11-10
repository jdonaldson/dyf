# DYF - Density Yields Features

[![CI](https://github.com/jdonaldson/dyf/actions/workflows/ci.yml/badge.svg)](https://github.com/jdonaldson/dyf/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/dyf)](https://pypi.org/project/dyf/)
[![Python](https://img.shields.io/pypi/pyversions/dyf)](https://pypi.org/project/dyf/)

**A clustering lens for embeddings, and an index you can search.** DYF fits a tree over
your embeddings once. From that one fit you can group the corpus at any level of detail, a
few broad groups or hundreds of narrow ones, by changing a single `resolution` setting,
without fitting again. The same tree is written to a `.dyf` file that opens in ~5 ms
without loading data and is searched with a Rust kernel.

See it on real data at **[dyf.io](https://dyf.io)**: 50,000 CIFAR-100 photos, one fit,
sixteen levels of detail, with the pipeline replayed step by step.

## What it does

**Group at any level.** The tree's leaves are joined into groups at a `resolution` you
choose, and changing it repeats only the two cheapest steps: 0.13 s after a 2.2 s fit on
50,000 CLIP embeddings, where refitting k-means for 100 groups took 2.9 s. The default
gives a coarse grouping; about fifty labelled items are enough to pick the level you mean.
A refit can start from the previous groups (`init_centroids`), which keeps group ids
stable as data arrives.

**Index and search.** Build a tree, write a `.dyf`, query it. Indexes are self-describing
(`dyf info corpus.dyf`), open lazily, and carry your own stored fields alongside the
vectors. Ingest helpers turn source code, images or video into embeddings, or bring your
own array.

**Describe structure.** `DensityClassifier` labels each item by where it sits:

- **Dense**: Core items in well-populated semantic regions
- **Bridge**: Transitional items connecting different clusters
- **Orphan**: Unique items with no semantic neighbors

Use cases:
- **Clustering at a level you choose**: coarse topics or fine intents from the same fit
- **Groups that persist**: carry group ids forward as a corpus grows
- **Index building**: instant-open on-disk indexes with your own stored fields
- **Anomaly detection**: Identify orphans and bridges

> **Scope.** dyf is not competing to be the fastest ANN retriever — on the pure
> recall-vs-latency frontier, mature graph libraries (HNSW, pynndescent) win. Its strengths
> are structure discovery, hierarchy, instant-open on-disk indexes, and reaching exact
> recall on a single index by raising `nprobe`. See [Performance](#performance).
>
> It is also not the cheapest way to get one partition: a single k-means fit at a known
> `k` is faster, and at the right level dyf ties k-means on agreement with labels rather
> than beating it. What dyf adds is every level from one fit, without being told `k`. The
> homepage lists the [results that went against it](https://dyf.io/#limits).
>
> Enrichment and the browser tour — UMAP projection, clustering, LLM labelling, narration —
> live downstream in [dyfviz](https://github.com/jdonaldson/dyfviz), split out in 0.13.

## Finding your way around

There are over a hundred public names, so start with the map rather than the list:

```bash
dyf api                 # grouped, with an entry point per group
dyf api trees --json    # one group, parseable
```

```python
import dyf
print(dyf.overview())   # the same map from Python
```

## Command line

```bash
dyf index-source src/ -o code.dyf     # tree-sitter chunks, Ollama embeddings
dyf index-images ~/photos -o pics.dyf # vision embeddings + thumbnails
dyf index-video clip.mp4 -o clip.dyf  # scene detection, one keyframe per scene
dyf info code.dyf                     # describe an index without loading it
dyf api                               # map of the Python API
dyf concepts query "Debrief Pattern"  # concept graph over your markdown notes
```

Add `--dry-run` to any `index-*` command to see what it *would* do — file and chunk
counts, embedding batches, whether the embedding service is reachable — without embedding
anything or writing a file. The embedding pass is the expensive, irreversible half; this
lets you look before leaping.

> **The `index-*` commands are CLI-only.** They are deliberately not in `__all__`, so
> there is no `dyf.index_images(...)`. Import them directly
> (`from dyf.index_source import index_source`) if you need them from Python, and note
> they take a path and write a file rather than returning data.
>
> Exit codes across the CLI: `0` success, `1` a normal negative answer (nothing found,
> graph stale), `2` a bad request, `3` a missing dependency or unreachable service. The
> message names what to install or start.

## Installation

```bash
pip install dyf
```

For serialization (save/load indexes):
```bash
pip install dyf[io]
```

For full features (embedding generation, LLM labeling):
```bash
pip install dyf[full]
```

## Quick Start

### Group at Any Level

```python
from dyf import build_dyf_tree, write_lazy_index, LazyIndex, louvain_cluster_leaves

tree = build_dyf_tree(embeddings, num_bits=3, max_depth=4, min_leaf_size=20)
write_lazy_index(tree, embeddings, "items.dyf")

with LazyIndex("items.dyf") as idx:
    coarse = louvain_cluster_leaves(idx, layout_2d, embeddings, resolution=1.0)
    fine = louvain_cluster_leaves(idx, layout_2d, embeddings, resolution=32.0)
    print(coarse.n_groups, fine.n_groups)   # 22 and 151 on 23,850 intent utterances
    labels = fine.point_labels              # one group id per item
```

`layout_2d` is any (n, 2) array, for example a UMAP layout; it is used only to name groups
by position. The default `resolution=1.0` is coarse. To choose a level, label about fifty
items and keep the resolution whose groups agree with them best
(see [Resolution](#how-it-works) below).

### Discover Structure

```python
import numpy as np
from dyf import DensityClassifier

# Your embeddings (e.g., from sentence-transformers)
embeddings = np.random.randn(10000, 384).astype(np.float32)

# Find structure
classifier = DensityClassifier(embedding_dim=384)
classifier.fit(embeddings)

# What did we find?
print(classifier.report())
# Corpus: 10000 items
#   Dense: 9500 (95.0%)
#   Bridge: 450 (4.5%)
#   Orphan: 50 (0.5%)

# Get indices
bridges = classifier.get_bridge()  # Transitional items
orphans = classifier.get_orphans() # Unique items
```

### Build & Search Indexes

```python
from dyf import build_dyf_tree, write_lazy_index, LazyIndex

# Build tree from embeddings
tree = build_dyf_tree(embeddings, max_depth=4, num_bits=3, min_leaf_size=8)

# Write to disk (mmap-friendly, zero startup cost)
write_lazy_index(tree, embeddings, "index.dyf",
                 quantization="float16", compression="zstd",
                 stored_fields={"title": titles},
                 metadata={"model": "nomic-embed-text-v1.5"})

# Search — instant open, LRU-bounded leaf cache, fast Rust backend
with LazyIndex("index.dyf") as idx:
    result = idx.search(query_embedding, k=10, nprobe=256, backend="rust")
    print(result.indices, result.scores)
    print(result.fields["title"])  # stored fields returned with results
```

For a fully in-memory corpus, `DenseSearchIndex` builds the tree and searches via the
same Rust kernel (batched queries supported). Both indexes rank by **cosine similarity**
— the kernel normalises the query and divides by each row's norm, so rows need not be
unit-norm, but magnitude is discarded; `DenseSearchIndex` warns once if your row norms
vary, because Euclidean or dot-product data will get confidently wrong neighbours:

```python
from dyf import DenseSearchIndex

idx = DenseSearchIndex(embeddings)                  # builds tree + flattens

result = idx.search(query, k=10, nprobe=256)
result.indices, result.scores                       # same SearchResult as LazyIndex

indices, scores = idx.search(query, k=10, nprobe=256)   # also unpacks as a tuple
I, S = idx.search(query_batch, k=10, nprobe=256)        # batched -> (nq, k)
```

Both index types return the same `SearchResult`, so they are interchangeable at the call
site. It unpacks as a 2-tuple, so the positional form keeps working.

### Inspect an Index

`dyf info` describes a `.dyf` file without loading its data — item count, dimensionality,
build params, stored fields, and how far the enrichment pipeline has been run:

```bash
dyf info index.dyf
dyf info index.dyf --json    # machine-readable
```

It is backed by `LazyIndex`'s lazy-open path, so it stays cheap regardless of file size —
**0.09s on a 479 MB, 229k-item index.** Useful for deciding what to do with an artifact
before paying to open it.

> The `--json` payload carries `"schema_version": 0` and is **unstable before v1**; the
> field is there so callers can detect a change rather than be surprised by one.

### Shrink the Index: Dedup on Ingest

Real corpora repeat themselves. Index one representative per near-duplicate cluster and
carry the mapping as a stored field:

```python
from dyf import near_duplicate_clusters, build_dyf_tree, write_lazy_index

result = near_duplicate_clusters(embeddings)        # cosine > 0.99, ~2.5s / 229k points
print(f"{result.n_removed:,} duplicates ({result.removed_fraction:.1%})")

reps = embeddings[result.mask()]
tree = build_dyf_tree(reps, max_depth=4, num_bits=4)
write_lazy_index(tree, reps, "index.dyf",
                 stored_fields={"dup_members": result.member_field()})

# at query time, expand a hit back to the points it stands for
from dyf import decode_members
also_matched = decode_members(result_fields["dup_members"][0])
```

**Measure before enabling — the duplicate rate is wildly corpus-dependent:**

| corpus | dup rate | `.dyf` saving |
|---|---|---|
| CMU MoCap (adjacent frames) | 88.3% | **77.1%** |
| SEC 10-Q (legal boilerplate) | 29.4% | **25.6%** |
| news / tweets / arxiv / wikipedia | 0.0–1.0% | ~0% |

Curated document collections have almost no near-duplicates; templated or temporally
oversampled corpora have many. Where duplicates exist, file saving is reliably **~0.87× the
duplicate rate**. `near_duplicate_clusters` is itself the cheap diagnostic (~1s per 100k
points), so measure first. Retrieval quality also improves when scored on distinct content,
because the probe budget stops re-scanning near-identical vectors. No file-format change:
the mapping is an ordinary utf8 stored field.

### Adaptive Probing

Queries near decision boundaries automatically probe more leaves:

```python
from dyf import LazyIndex, AdaptiveProbeConfig

with LazyIndex("index.dyf") as idx:
    # Auto mode: margin-based probe count (default thresholds)
    result = idx.search(query, k=10, nprobe="auto", return_routing=True)
    print(result.routing["adaptive_nprobe"])  # how many leaves were probed

    # Custom thresholds
    cfg = AdaptiveProbeConfig(margin_lo=0.005, margin_hi=0.2,
                              min_probes=1, max_probes=8)
    result = idx.search(query, k=10, nprobe=cfg)
```

### Full-Featured Usage

```python
from dyf import DensityClassifierFull, EmbedderConfig, LabelerConfig

# From raw texts
classifier = DensityClassifierFull.from_texts(
    texts=documents,
    categories=categories,
)

# Label clusters with LLM
labels = classifier.label_buckets(**LabelerConfig.MEDIUM.as_kwargs())
print(labels['dense'][1234]['label'])  # "Machine Learning Papers"
```

## How It Works

The grouping pipeline has five steps. The [homepage](https://dyf.io/#how) replays them on a
real run, with a formula for each.

1. **Normalise.** Every vector is scaled to length 1, so only its direction is used.
2. **Tree.** Each node takes the top `num_bits` principal components of its own points and
   gives every point one bit per component: which side of the cut it is on. That makes up
   to 2^`num_bits` children. Splitting stops at `max_depth`, or when a node has fewer than
   2 × `min_leaf_size` points.
3. **Leaf graph.** Each leaf's centre is linked to its 10 most similar leaf centres.
4. **Communities.** Louvain groups the leaves at the chosen `resolution`.
5. **Reassignment.** Each point moves to the group whose centre it is most similar to, for
   up to five rounds.

Steps 1 to 3 run once. Changing `resolution` repeats only steps 4 and 5, and the tree from
step 2 is also the search index.

`DensityClassifier` is a separate entry point with its own two-stage PCA hash: items in
well-populated buckets are Dense, items in sparse buckets that share structure at a coarser
resolution are recovered as Bridges, and the rest are Orphans.

**Geometry: the tree partitions directions.** Every fit L2-normalises each row before PCA and
hashing, so points with the same direction and different norms share a leaf. That is right for
embeddings (unit-norm already) and for PCA scores whose norm is a nuisance — on scRNA-seq PCA-50
the norm is library size and cell-cycle amplitude, and an un-normalised tree measurably lost
0.1–0.45 ARI on every such dataset (`KNOWN_ISSUES.md` #11). It is wrong for low-dimensional raw
features whose norm *is* the structure: concentric rings centred at the origin are one set of
directions and cannot be separated. For that regime pre-scale the data so norm is uninformative,
or reach for a connectivity/density method; `LeafGroupingResult.spectrum` tells you which regime
the leaf graph is in.

**The communities you get are Voronoi cells, not graph communities.** Both leaf-grouping paths
(`louvain_cluster_leaves`, `agglomerate_tree_leaves`) end with spherical k-means over every
point, seeded by the leaf-level grouping: each point ends up in the community whose centroid it
is most similar to. The step moves 4–31% of points on the seven-dataset bench, makes communities
contiguous, and shifts ground-truth agreement by −0.13 (MNIST PCA-50) to +0.11 (1.3M brain) with
the sign set by the data (`KNOWN_ISSUES.md` #14). Two consequences worth knowing: the final
partition cuts about as many nearest-neighbour edges as k-means at the same k (2–4× a Leiden
partition on a point kNN graph), and a community that Louvain found as a non-convex region is
flattened into a convex cell. Use `LeafGroupingResult.centroids` to place new points by the same
rule. Two free numbers describe each cell afterwards (`compute_louvain_hierarchy`):
`community_cohesion`, its angular spread, which ranked within an index finds the least pure
quarter of communities at AUC 0.80 on six of seven bench sets, and `community_inflow`, the share
of the cell the re-cut moved in from other communities, which says whether a loose cell is a
broad region or a small seed that swallowed its neighbours. Both are ranks within one index, not
thresholds, and neither means anything on continuum data.

**Resolution is a granularity choice, and the data does not make it for you.** Louvain's
`resolution` sets how many communities come back (about 16 on 4-topic AG News and 28 on a
123-type cell atlas at the default of 1). Three label-free criteria — map-equation codelength,
edge-dropout stability, the Laplacian eigengap — all chose worse than the default on the
seven-dataset bench, and the map equation itself only trades a coarse prior for a fine one.
What works is a few labels: pick the resolution whose partition agrees best with about fifty
randomly labelled items, and you land within 0.02 ARI of the best resolution in the sweep,
closer than knowing the true number of classes would get you (`benchmarks/cut_audit_2026_11/`).
Run `compute_louvain_hierarchy` for the sweep, label fifty items, choose the level that agrees.

## Performance

Search runs on a Rust multiprobe kernel (`dyf-rs`, PyO3) — the default path for
both `LazyIndex.search` and `DenseSearchIndex`. The required version is declared in
`pyproject.toml` and checked at import; it is not repeated here, because the copy in this
sentence said `>= 0.8.0` while the requirement was `>= 0.10.0`. Results are **bit-identical** to the
pure-Python reference (`backend="python"`); the kernel handles fixed *and* adaptive
`nprobe` and `return_routing`. MSMARCO MiniLM-L6 (384d), Apple Silicon, batched unless
noted:

| path | corpus / setting | latency / query | vs pure-Python |
|------|------------------|-----------------|----------------|
| `DenseSearchIndex` (in-memory, batched) | 8.84M, nprobe=256 | ~0.5 ms | ~100× |
| `LazyIndex.search` (on-disk, batched) | up to 8.84M, nprobe=256 | ~0.9 ms | ~29× |
| `LazyIndex.search` (on-disk, single query) | nprobe=256 | ~4 ms | ~6× |
| `LazyIndex.search` (with stored fields) | immich 35K | ~2 ms | ~15× |

Speedup is largest at low `nprobe`, batched queries, and without field-gather. Lazy mode
opens in ~5 ms (vs ~0.4 s preload) and bounds memory with an LRU; only PQ-compressed and
overflow indexes fall back to Python.

> **Scope.** These are Rust-vs-pure-Python speedups — recovering the cost of the per-query
> Python loop. They are **not** a claim that dyf is the fastest ANN retriever: on the pure
> recall-vs-latency frontier, mature graph libraries (pynndescent, HNSW) are faster. dyf's
> strengths are structure discovery, hierarchy, instant-open on-disk indexes, and reaching
> exact recall on a single index by raising `nprobe`.

## API

### DensityClassifier

```python
DensityClassifier(
    embedding_dim: int,
    initial_bits: int = 14,      # LSH resolution
    recovery_bits: int = 8,      # Coarser recovery resolution
    dense_threshold: int = 10,   # Min bucket size for "dense"
    seed: int = 31
)

# Methods
classifier.fit(embeddings)
classifier.get_dense()           # Dense item indices
classifier.get_bridge()          # Bridge item indices
classifier.get_orphans()         # Orphan item indices
classifier.get_bucket_id(idx)    # Which bucket is item in?
classifier.report()              # Summary statistics
```

### LazyIndex

```python
from dyf import LazyIndex

with LazyIndex("index.dyf") as idx:
    # Search with fixed or adaptive probing
    result = idx.search(query, k=10, nprobe=3)       # fixed
    result = idx.search(query, k=10, nprobe="auto")   # adaptive

    # Inspect index structure
    idx.tree_summary          # metadata, dims, leaf count
    idx.total_items           # total indexed items
    idx.stored_field_names    # available stored fields

    # Extract all data
    data = idx.extract_all_fields()
    data['embeddings']        # (n, d) float32
    data['fields']            # {field_name: array}
```

## Documentation

- **[dyf.io](https://dyf.io)** — the lens on real data: drag the resolution, see the groups, see where dyf loses
- **[How It Works](https://dyf.io/#how)** — the pipeline step by step, replayed on a recorded run, with a formula per step
- **[Getting Started](https://dyf.io/getting-started.html)** — code recipes and examples
- **[Gallery](https://dyf.io/gallery/)** — dyf against k-means and HDBSCAN on classic datasets
- **[API Reference](https://dyf.io/reference/)** — full documentation for all classes and functions

## License

MIT
