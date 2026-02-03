# DYF Clustering Exploration Notes

## Summary

**Updated 2026-02-02**: Previous conclusion ("dyf is good at density classification, not clustering") was wrong. The DYF tree — recursive k-ary LSH splitting with agglomerative cosine merge — is the first method tested to achieve a **positive sim gap** on the wiki 50k dataset. The key was combining DYF's multi-axis PCA-on-centroids splitting with deep recursive hierarchy.

**Original date**: 2026-01-18 (single-level and two-tier experiments below).

---

## DYF Tree: Recursive LSH Clustering (2026-02-02)

### The Breakthrough

Recursive DYF tree combines:
1. **DYF's multi-axis splits**: DensityClassifier uses PCA on bucket centroids to derive hyperplanes, then hashes on all axes simultaneously (not just PC1)
2. **PCA tree's deep recursion**: Multiple levels of splitting, with per-point margins at each depth
3. **Agglomerative cosine merge**: Leaf centroids merged to target_k using cosine distance + average linkage

This is the key insight: DYF's `fit()` already does PCA on centroids — it's not random hyperplanes. The hyperplanes are data-adaptive. Using them recursively in a tree gives you multi-axis topic boundaries at every level.

### 5-Way Comparison (wiki 50k, 7104 points after dedup, 25 clusters)

| Metric | BIRCH-2D | PCA-tree | DYF-1lvl | DYF-hier | **DYF-tree** |
|--------|----------|----------|----------|----------|-------------|
| Silhouette (2D) | **0.375** | -0.032 | -0.389 | -0.698 | -0.724 |
| kNN purity (2D) | 0.930 | 0.465 | 0.332 | 0.957 | **0.984** |
| Intra-cluster sim | 0.351 | 0.341 | 0.326 | 0.391 | **0.413** |
| Inter-cluster sim | 0.727 | 0.759 | 0.639 | 0.414 | **0.309** |
| **Sim gap** | -0.376 | -0.418 | -0.312 | -0.023 | **+0.105** |
| Fragmentation | 23.0 | 154.1 | 101.3 | 24.6 | **21.1** |
| % single-blob | 0% | 0% | 10% | 12% | **36%** |

- **Sim gap**: Only DYF tree achieves positive (intra > inter). Every other method has negative sim gap.
- **kNN purity 0.984**: 2D spatial neighbors almost always share cluster labels.
- **Fragmentation 21.1**: Fewer spatial fragments than BIRCH despite clustering in high-D.

### Methods Compared

1. **BIRCH on 2D**: Clusters UMAP coordinates. Best 2D silhouette but negative sim gap — spatial clusters don't correspond to semantic topics.
2. **PCA tree (cut_tree_to_labels)**: Recursive PC1 bisection in high-D. Produces power-of-2 cluster counts (fcluster limitation). 16 clusters when targeting 25.
3. **DYF single-level**: DensityClassifier with num_bits tuned for ~target_k. Too coarse — one level of splitting can't capture topic boundaries.
4. **DYF hierarchical**: Two-tier LSH (global 8-bit + local 4-8 bit per bucket), facets merged with agglomerative. Close to zero sim gap (-0.023) but only two levels deep.
5. **DYF tree**: Recursive k-ary (num_bits=3, depth=4). Multi-axis splits at every level + cosine agglomerative merge on leaves. Positive sim gap.

### Why DYF Tree Wins

The progression tells the story:
- **PCA tree** splits on one axis (PC1) per level. Topics that don't align with the maximum variance direction get split incorrectly.
- **DYF single-level** uses multiple PCA-derived axes simultaneously but only one level — no hierarchy to refine.
- **DYF hierarchical** adds a second level but that's not enough depth.
- **DYF tree** combines both: multiple axes per level AND deep recursion. Each level re-fits DensityClassifier on the local subset, getting new PCA-on-centroids hyperplanes adapted to the local geometry.

The agglomerative cosine merge on leaf centroids is also important. PCA tree used scipy `fcluster` which cuts by linkage distance (discrete integers from tree depth), ignoring semantic similarity. Cosine agglomerative directly optimizes for the metric we care about.

### DYF's Hidden PCA

DYF's `DensityClassifier.fit()` is NOT random LSH. The actual process:
1. Random hyperplanes create bootstrap buckets
2. Compute centroids of those buckets
3. **PCA on the centroids** — principal components become final hyperplanes
4. Re-hash all points using PCA-derived hyperplanes

So DYF and PCA tree both use PCA. The difference:
- PCA tree: PCA on raw points, one axis (PC1), binary split
- DYF: PCA on bucket centroids, all num_bits axes simultaneously, k-ary split

### Parameters

```python
from dyf.dyf_tree import build_dyf_tree, cut_dyf_tree_to_labels

tree = build_dyf_tree(embeddings, max_depth=4, num_bits=3, min_leaf_size=4)
labels = cut_dyf_tree_to_labels(tree, n_points, n_clusters=25, embeddings=embeddings)
```

- `num_bits=3`: 8-way splits per level. With depth=4, up to 8^4=4096 leaves.
- `min_leaf_size=4`: Stop splitting when nodes have <8 points.
- Agglomerative merge uses `metric='cosine', linkage='average'`.

### Boundary Persistence Still Works

```python
from dyf.dyf_tree import extract_boundary_persistence, boundary_persistence_scores

bp = extract_boundary_persistence(tree, margin_pct=0.10)
scores = boundary_persistence_scores(tree)
```

Centroid similarity serves as the margin: low centroid_sim = far from bucket center = boundary point. Same threshold logic as PCA tree.

---

## What dyf IS Good For

### O(1) Density Classification
- **Dense**: Points in well-populated buckets (bucket size ≥ threshold)
- **Bridge**: Points with low similarity to their bucket centroid (< 0.5)
- **Orphan**: Points in sparse buckets that weren't recovered

This is fast and useful. Don't try to make it do clustering.

---

## What dyf is NOT Good For

### 1. Clustering Quality

| Approach | Silhouette | Notes |
|----------|------------|-------|
| Baseline k-means++ | 0.027 | Best we achieved in high-D |
| Spectral on bridge graph | -0.011 | Worse than random |
| Agglomerative on bridge graph | -0.043 | Much worse |
| Dense-core seeded k-means | 0.022 | Slightly worse |
| Multi-resolution LSH (6 bits) | -0.008 | Negative = no structure |
| K-means on UMAP 2D | **0.394** | This works, but it's UMAP not dyf |

**Key insight**: LSH buckets partition space by hyperplane cuts (angular), not by density or cluster shape. Bucket boundaries don't align with semantic cluster boundaries.

### 2. Speeding Up K-means

- Dense points = 95% of data (minimal filtering)
- Clustering bucket centroids then assigning: 70x faster but ARI = 0.199 (poor quality)
- Hybrid (centroid init + refinement): 16x faster, silhouette 0.018 vs 0.027

**Trade-off not worth it** for most cases.

### 3. Choosing k for Clustering

Multi-resolution (fewer LSH bits) gives geometric series: k = 8, 16, 32, 64, 128...

But this is just powers of 2 - **you don't need dyf for this**:
```python
k_candidates = [2**n for n in range(3, 8)]  # Same thing
```

### 4. Speeding Up UMAP

- UMAP's bottleneck is kNN graph construction
- dyf bucket-restricted search: 82% recall, ~100s
- pynndescent (UMAP's default): 99% recall, 7s

**pynndescent already uses random projection trees (LSH family) with optimized C code.** dyf doesn't help.

---

## Bridge Analysis Findings

### Bridge Classification

Bridges can be categorized:
- **Misplaced** (~10%): Should be in a different bucket (target_sim - home_sim > 0.15)
- **Meta** (~90%): Genuinely span two topics

### Correction Rules

- 52 rules generated with >70% accuracy
- Rules cluster into 5 groups by separator direction
- Most rules point toward a "high-error zone" in PCA space (PC0+, PC1-)
- **Misplaced points are NOT more similar to each other than random** - errors are scattered, not systematic

### High-Error Zone

- PC0 bin 1, PC1 bin 0: 3.91% error rate (30x higher than best region)
- Points farther from global centroid have 2.5x higher error rates
- This zone contains mixed entertainment/media topics

---

## The UMAP vs High-D Gap

| Clustering | Eval in 2D | Eval in 1024D |
|------------|------------|---------------|
| K-means on 2D UMAP | **0.394** | 0.012 |
| K-means on 1024D | -0.024 | 0.026 |

**UMAP creates structure** through its neighbor graph algorithm. The "blobs" are real in that representation but don't exist as globally separable clusters in original space.

If you want clusters matching UMAP visualization:
```python
kmeans.fit(coords_2d)  # This works (silhouette 0.39)
# NOT
kmeans.fit(embeddings)  # This doesn't (silhouette 0.03)
```

---

## Approaches That Failed

### 1. Iterative Hyperplane Refinement
Added corrective hyperplanes based on misplaced bridge directions.
- Result: More misplaced points (fragmented good buckets)
- Misplaced: 477 → 622

### 2. Prototype-Based (Voronoi) Refinement
Iteratively reassign points to nearest centroid (like k-means seeded with LSH centroids).
- Result: Misplaced → 0, coherence +9.2%
- **But loses hash property** - no longer O(1) lookup

### 3. Compound Linear Rules
Boolean combinations of hash bits + extra dot product tests.
- Result: Misplaced 477 → 417 (-12.6%)
- Still hashable but modest improvement

### 4. Bridge Flow Merging
Merge buckets with high mutual bridge flow.
- Result: 0 strong mutual pairs found
- Bridges are scattered, not systematic

---

## Practical Recommendations

### Use dyf for Cluster Confidence

The most useful finding: **centroid similarity predicts cluster quality**.

```python
# Compute similarity to bucket centroid
home_sims = [dot(point, bucket_centroid) for point in embeddings]

# High similarity = clusters cleanly
# Low similarity = ambiguous/bridge point

# Points with sim >= 0.6 (44% of data): silhouette 0.44
# Points with sim >= 0.7 (11% of data): silhouette 0.59
# All points: silhouette 0.41

# Use as confidence signal
confidence = np.where(home_sims >= 0.6, "high", "low")
```

### Use dyf for:
```python
classifier = DensityClassifier(embedding_dim=dim)
classifier.fit(embeddings)

# These are useful
dense = classifier.get_dense()      # Core items
bridges = classifier.get_bridge()   # Boundary items
orphans = classifier.get_orphans()  # Unique items
```

### For semantically meaningful clustering, use DYF tree:
```python
from dyf.dyf_tree import build_dyf_tree, cut_dyf_tree_to_labels

tree = build_dyf_tree(embeddings, max_depth=4, num_bits=3)
labels = cut_dyf_tree_to_labels(tree, len(embeddings), n_clusters=25, embeddings=embeddings)
```

### For UMAP-aligned spatial clusters, use BIRCH on 2D:
```python
reducer = umap.UMAP()
coords_2d = reducer.fit_transform(embeddings)
labels = Birch(n_clusters=None, threshold=...).fit_predict(coords_2d)
```

### Combine DYF tree + bridge detection:
```python
from dyf.dyf_tree import build_dyf_tree, cut_dyf_tree_to_labels, boundary_persistence_scores

tree = build_dyf_tree(embeddings, max_depth=4, num_bits=3)
labels = cut_dyf_tree_to_labels(tree, n, 25, embeddings)
bridge_scores = boundary_persistence_scores(tree)

# Now you know:
# - Semantically coherent cluster assignments (positive sim gap)
# - Which points are bridges (high persistence score = multi-level boundary)
# - Cluster density structure from DYF
```

---

## Clustering Algorithm Comparison

| Method | Clusters | Noise | Silhouette | Time | Notes |
|--------|----------|-------|------------|------|-------|
| K-means on 2D | 20 | 0% | 0.40 | <1s | Voronoi tessellation, doesn't follow density |
| HDBSCAN on 2D | 122 | 30% | 0.53 | 1.5s | Follows density contours |
| Spectral | - | - | - | >60s | Too slow for 50k points |
| dyf + k-means | 30 | ~10% | 0.45 | <1s | Use centroid_similarity for noise detection |

**Key insight**: HDBSCAN provides its own `probabilities_` similar to dyf's `centroid_similarity` - both indicate assignment confidence.

---

## Dataset Statistics (Wikipedia 50k)

- Points: 50,000
- Dimensions: 1,024
- Buckets: 1,523
- Avg bucket size: 32.8
- Dense points: 47,441 (94.9%)
- Bridges: 4,747 (9.5%)
- Misplaced bridges: 477 (1.0%)

---

## Key Equations

### Bridge Detection
```
point is bridge if: similarity(point, bucket_centroid) < 0.5
```

### Misplaced Detection
```
point is misplaced if:
  best_other_bucket_sim - home_bucket_sim > 0.15
```

### Multi-Resolution Buckets
```
coarse_bucket = bucket_id & ((1 << n_bits) - 1)
# n_bits=6 → ~32 buckets
# n_bits=7 → ~64 buckets
# n_bits=8 → ~128 buckets
```

---

## Files Generated During Exploration

- `/tmp/wiki_embeddings.npy` - 50k Wikipedia embeddings
- `/tmp/wiki_bucket_ids.npy` - dyf bucket assignments
- `/tmp/wiki_coords_2d.npy` - UMAP 2D coordinates
- `/tmp/wiki_texts.txt` - Article titles
- `/tmp/pca_clusters.html` - Visualization with Ollama labels
- `/tmp/compound_rules.html` - Compound linear rule results
- `/tmp/pca_rules_combined.html` - Rule cluster visualization
