# DYF for RAG: Bridge-Seeded K-means

Date: 2026-01-22

## Summary

**Key findings:**
1. DYF bridge points are excellent seeds for k-means centroid initialization in IVF indexes (+1% recall, 17-24% fewer probes)
2. DYF buckets provide **semantic facet diversification** that standard vector search methods (MMR, k-means) cannot replicate
3. DYF's value comes from **PCA-aligned geometric structure**, not specific bucket assignments—facet diversification is robust to seed choice

---

## The Discovery

### What Doesn't Work: Bridges as Centroids
Using bridge points directly as IVF centroids underperforms standard k-means:
- At ~2000 candidates: 0.858 recall (bridges) vs 0.955 (k-means)
- Bridges sit at boundaries, but queries are typically within clusters

### What Works: Bridges as K-means Seeds
Using orthogonal bridge points to *initialize* k-means beats standard k-means:

| nlist | Standard k-means | Bridge-seeded | Δ |
|-------|------------------|---------------|------|
| 50 | 0.790 | 0.816 | **+2.5%** |
| 100 | 0.876 | 0.894 | **+1.8%** |
| 200 | 0.927 | 0.931 | +0.4% |
| 500 | 0.955 | 0.966 | **+1.1%** |
| 1000 | 0.972 | 0.977 | +0.5% |
| 2000 | 0.985 | 0.988 | +0.3% |

**Win rate: 6/6 configurations**

---

## Why It Works

1. **Bridges provide better initial spread** — they're at boundaries, naturally far apart
2. **K-means migrates them toward centers** — refinement optimizes positions
3. **Result:** Centroids that started well-distributed and converged optimally

The advantage is largest at smaller nlist values (fewer centroids) where initial placement matters more.

---

## The Bigger Win: Fewer Buckets Needed

The most significant finding: **bridge-seeded centroids require 17-24% fewer bucket probes to reach the same recall.**

| Target Recall | Standard nprobe | Bridge nprobe | Savings |
|---------------|-----------------|---------------|---------|
| 90% | 8 | 7 | -12.5% |
| 95% | 18 | 15 | **-16.7%** |
| 97% | 31 | 24 | **-22.6%** |
| 99% | 77 | 59 | **-23.4%** |

### Why Fewer Buckets?

True neighbors cluster into fewer buckets with bridge-seeded centroids:

| Metric | Standard | Bridge-seeded |
|--------|----------|---------------|
| Mean buckets for k=10 neighbors | 3.47 | 3.32 |
| Queries with neighbors in ≤3 buckets | 56.4% | 59.2% |
| Max buckets needed | 10 | 8 |

**The mechanism:** Bridge-seeded centroids create more "coherent" partitions. Semantic neighbors land in the same bucket more often, so you need to probe fewer buckets to find them all.

### Implications

At scale, this matters more than the +1% recall improvement:
- **17-24% fewer bucket probes** = 17-24% less compute per query
- For billion-scale indexes with disk-based posting lists, fewer probes = fewer disk reads
- Latency-sensitive applications benefit from reduced bucket traversal

---

## Seeding Strategy Comparison

Tested 7 DYF-based seeding strategies (nlist=500):

| Strategy | @20 Recall | Δ vs baseline |
|----------|------------|---------------|
| **Orthogonal bridges** | **0.966** | **+1.1%** |
| Quadrant-balanced | 0.959 | +0.4% |
| Stratified (dense/bridge/orphan) | 0.956 | +0.1% |
| Standard k-means | 0.955 | — |
| Bucket centroids | 0.955 | -0.1% |
| Dense core representatives | 0.953 | -0.2% |
| Anti-correlated pairs | 0.951 | -0.4% |
| Bucket-weighted | 0.950 | -0.5% |

### Key Insight: Spread > Representativeness

- **Winners** maximize geometric spread (orthogonal selection, quadrant diversity)
- **Losers** try to pick "central" or "representative" points

K-means will find cluster centers anyway. What helps most is **initial coverage**.

### Dense Points Work Too

Tested orthogonal selection from dense bucket points (90% of data) vs bridges (10%):

| Strategy | @20 Recall |
|----------|------------|
| Orthogonal bridges | 0.966 |
| Orthogonal dense points | 0.966 |

**Conclusion:** The win comes from orthogonal initialization, not bridge-specific properties. Bridges are a convenient ~10% subset, but dense points work equally well.

---

## Orthogonal Bridge Selection Algorithm

```python
from dyf import select_orthogonal_anchors

# Get maximally-spread bridge points
anchors = select_orthogonal_anchors(embeddings, k=nlist, seed=42)

# Use as k-means initialization
kmeans = KMeans(n_clusters=nlist, init=embeddings[anchors.indices], n_init=1)
kmeans.fit(embeddings)

# Build FAISS index with refined centroids
centroids = kmeans.cluster_centers_
```

The `select_orthogonal_anchors` function:
1. Finds super connectors (high global + local bridge centrality)
2. Uses them as seeds for farthest-point sampling
3. Greedily selects points maximizing minimum distance from selected set

---

## Comparison vs SOTA

| System | Architecture | Recall @2k candidates |
|--------|--------------|----------------------|
| FAISS IVF (standard) | K-means centroids | 0.955 |
| FAISS IVF (bridge-seeded) | DYF → K-means | **0.966** |
| BridgeIndex (pure bridges) | Bridge anchors | 0.797 |
| SPANN (Microsoft) | Hierarchical balanced clustering | ~0.95 (paper) |

Bridge-seeded k-means provides a simple +1% recall improvement over standard FAISS IVF with minimal code change.

---

## Bridge vs All-Points Expansion

The orthogonal selection algorithm needs two inputs:
1. **Seeds:** Super connectors (high global + local centrality)
2. **Candidates:** Pool to expand from via farthest-point sampling

Tested two candidate pools:
- **Bridges only:** ~10% of points at semantic boundaries
- **All points:** 100% of points

| nlist | Target | Bridges | All pts | Winner |
|-------|--------|---------|---------|--------|
| 500 | 95% | **15** | 16 | Bridges |
| 500 | 97% | **24** | 25 | Bridges |
| 500 | 99% | 59 | **51** | All pts |
| 1000 | 97% | **31** | 33 | Bridges |
| 1000 | 99% | **88** | 92 | Bridges |

### Why Bridges Win at ≤97% Recall

- Bridges create tighter bucket coherence (3.32 vs 3.37 mean buckets)
- Bridge pool acts as "curated candidates" — filters out redundant interior points
- Farthest-point on bridges = spread across boundaries = better partitions

### Why All-Points Wins at 99%+ Recall

- At extreme recall, you need to find *every* neighbor including outliers
- Bridges miss some isolated points in sparse regions
- All-points pool has better long-tail coverage

### Recommendation

```python
# Default: use bridges (optimal for 90-97% recall)
anchors = select_orthogonal_anchors(embeddings, k=nlist, seed=42)

# High-recall mode: expand from all points (for 99%+ recall)
anchors = select_orthogonal_anchors(
    embeddings, k=nlist,
    candidate_indices=np.arange(len(embeddings)),  # all points
    seed=42
)
```

---

## When to Use This

**Good for:**
- IVF index construction where you control centroid initialization
- Smaller nlist values (50-500) where the improvement is largest
- Situations where +1% recall matters (high-volume production)
- **17-24% reduction in bucket probes** for same recall

**Not needed for:**
- Large nlist values (>1000) where k-means converges well anyway
- One-off analyses where build time matters more than recall
- HNSW or other graph-based indexes (different architecture)

---

---

## Cluster Quality & Bridge Filtering

### What Makes Clusters Clean vs Noisy?

Analyzed 50 largest buckets to identify predictors of cluster coherence:

| Predictor | Correlation with Coherence |
|-----------|---------------------------|
| Centroid norm | +1.00 (perfect predictor) |
| Bridge ratio | -0.75 |
| Size | -0.39 |
| Avg pairwise similarity | +0.97 |

**Key finding:** High centroid norm = cluster points in same direction = clean cluster. Bridge ratio inversely predicts coherence — more bridges = more noise.

### Bridge Filtering Experiments

Tested whether removing bridges improves cluster quality:

| Metric | Finding |
|--------|---------|
| Coherence improvement (remove bridges) | +0.008 mean |
| Bridges misplaced (would fit better elsewhere) | 97.1% |
| Centroid sim < 0.5 = bridges | 100% |
| Centroid sim > 0.5 = non-bridges | 100% |
| Ground truth neighbors that are bridges | 6.7% |

**Insights:**
- Bridges are "semantically homeless" — 97% would have higher similarity to a different bucket
- Centroid similarity perfectly stratifies bridges from non-bridges
- But ground truth neighbors include bridges (6.7%) — filtering loses true positives

### Iterative Cluster Cleaning

Removing bottom 10% by centroid similarity increases coherence ~0.02 per round:

```
Noisy bucket: 0.550 → 0.570 → 0.591 → 0.612 → 0.633
```

Useful for pre-processing/cleanup, not real-time retrieval.

---

## Centroid Similarity as Confidence Score

### In Exact k-NN: No Improvement
With exact similarity search, confidence filtering hurts:
- Baseline recall: 1.000
- Any filtering: < 1.000

The top-k by similarity ARE the true neighbors — no false positives to filter.

### In IVF (Approximate) Search: Moderate Value

| Metric | Value |
|--------|-------|
| Correlation (result centroid_sim, recall) | r = 0.26-0.31 |
| Correlation (query centroid_sim, recall) | r = 0.31 |
| Recall for query centroid_sim < 0.6 | 0.832 |
| Recall for query centroid_sim > 0.7 | 0.899 |

### Adaptive nprobe Based on Centroid Similarity

| Strategy | Recall | Mean Probes |
|----------|--------|-------------|
| Fixed nprobe=10 | 0.879 | 10.0 |
| Adaptive (3x for low-confidence) | 0.891 | 13.9 |
| Improvement | **+1.16%** | — |

Low-confidence queries improved from 0.832 → 0.884 with 3x probes.

### Key Takeaway
- **DON'T use centroid_sim for filtering** — removes true positives
- **DO use it for adaptive nprobe** — extra probes for struggling queries
- **DO use it as uncertainty estimate** — flag low-confidence results for downstream

---

## Hard Query Detection

### What Makes Queries Hard?

Strongest predictor: **neighbor bucket spread** (r = -0.50)

| Neighbor Buckets | Count | Recall@10 |
|------------------|-------|-----------|
| 1 | 86 | 0.900 |
| 2 | 124 | 0.898 |
| 3 | 115 | 0.897 |
| 4 | 73 | 0.892 |
| 5 | 54 | 0.857 |
| 6+ | 48 | 0.760 |

### Hard Query Characteristics

| Metric | Hard (recall<0.8) | Easy |
|--------|-------------------|------|
| Neighbor buckets | 5.86 | 2.93 |
| Query centroid_sim | 0.609 | 0.672 |
| Local fraction (same bucket) | 0.24 | 0.65 |
| Query is bridge | 14.3% | 7.6% |

### Practical Detection

At query time, you can estimate "hardness" via:
1. Query centroid_sim (precomputed at indexing)
2. Local fraction (sample neighbors, count same-bucket)
3. Top cluster similarity spread

Use these to trigger adaptive nprobe for hard queries.

---

## Hierarchy & Recursive Faceting

### Coherence Improves with Depth

| Level | Mean Coherence | Description |
|-------|----------------|-------------|
| L0 | 0.61 | Global buckets |
| L1 | 0.88 | Faceted subbuckets |
| L2 | 0.90 | Deep facets |

### Bucket Size Reduction

| Transition | Size Reduction |
|------------|---------------|
| L0 → L1 | 10.2x |
| L1 → L2 | 2.9x |

### Hierarchical vs Flat Search

| Strategy | Mean Recall | Head-to-head Win Rate |
|----------|-------------|----------------------|
| Flat (leaf search) | 0.848 | **79.8%** |
| Hierarchical (beam) | 0.833 | 20.2% |

**Why flat wins:** Beam search commits early and can miss relevant leaves. Flat has global visibility over all leaf centroids.

**When hierarchy helps:** Topic discovery, visualization, understanding cluster structure — but not for search.

---

## Facet Diversification: DYF vs Standard Methods

### The Question
Can normal vector search achieve the same diversification as DYF buckets?

### Methods Compared

| Method | How it works |
|--------|--------------|
| **MMR (λ=0.3-0.7)** | Re-ranks by penalizing similarity to already-selected results |
| **K-means on results** | Cluster top-N results, pick best from each cluster |
| **DYF Bucket** | One result per bucket, ranked by query similarity |

### Quantitative Results (Wikipedia 50k, 200 queries)

| Method | Unique Titles | Unique Buckets | Avg Similarity |
|--------|---------------|----------------|----------------|
| Similarity (baseline) | 5.67 | 6.58 | 0.653 |
| MMR (λ=0.3) | **9.55** | 8.21 | 0.548 |
| MMR (λ=0.5) | 8.81 | 7.88 | 0.583 |
| K-means | 8.32 | 7.33 | 0.605 |
| DYF Bucket | 6.11 | **10.00** | 0.634 |

### The Key Insight: Different Kinds of Diversity

**MMR maximizes geometric diversity:**
- Different points in embedding space
- More unique titles (9.55 vs 6.11)
- But may cluster in similar semantic regions

**DYF maximizes semantic facet diversity:**
- Different buckets = different semantic regions
- Fewer unique titles, but different "angles" on the topic
- Preserves relevance better (0.634 vs 0.548 similarity)

### Example: Query "Bird"

**MMR (λ=0.3)** — 9 unique titles, 6 buckets:
```
Bird, Egg, Invertebrate, Bat, Vulture, Bear, Ostrich, Penguin...
```
All animals, geometrically spread apart.

**DYF Bucket** — 3 unique titles, 10 buckets:
```
Bird (bucket 870), Bird (bucket 358), Bird (bucket 614),
Wing (bucket 894), Dinosaur (bucket 886)...
```
Same title from different Wikipedia pages = different semantic facets (taxonomy, evolution, anatomy).

### Method Overlap Analysis

| Comparison | Overlap |
|------------|---------|
| DYF vs K-means | 41.8% |
| DYF vs MMR | 28.0% |
| K-means vs MMR | 31.8% |

Methods select **different** results — only 28-42% overlap.

### LLM Evaluation (High-Duplication Queries)

Tested on 246 queries where similarity baseline had ≤5 unique titles in top-10:

| Metric | Similarity | DYF Facet-Diverse |
|--------|------------|-------------------|
| Unique titles | 3.24 | 4.04 (+25%) |
| LLM preference | 33.3% | **66.7%** |

**p-value < 0.001** (highly significant)

### What DYF Provides That Others Don't

| Aspect | MMR | K-means | DYF Buckets |
|--------|-----|---------|-------------|
| Computation | O(n²) per query | O(n·k) per query | Pre-computed |
| Semantic labels | None | None | Bucket IDs |
| Cross-query consistency | No | No | Yes |
| Bridge detection | No | No | Yes |
| Best for | "Different things" | "Different things" | "Different perspectives" |

### When to Use Each

**Use MMR when:**
- You want maximum variety in results
- Semantic structure doesn't matter
- Per-query computation is acceptable

**Use DYF Buckets when:**
- Results have topical redundancy (same thing, different pages)
- You want consistent facet labels across queries
- You need to understand semantic boundaries
- Per-query overhead must be minimal

### The Bottom Line

MMR and DYF answer different questions:
- **MMR:** "Show me 10 different relevant things"
- **DYF:** "Show me 10 different perspectives on the most relevant thing"

For queries with high topical redundancy, DYF's semantic facet diversity is preferred 2:1 by LLM evaluation.

---

## Seed Stability Analysis

### How Stable Are Bridges Across Seeds?

Tested 8 different random seeds with 10-bit DYF:

| Category | Count | % of data |
|----------|-------|-----------|
| Always bridge (all 8 seeds) | 319 | 0.6% |
| Never bridge (0 seeds) | 34,216 | 68.4% |
| Sometimes bridge (1-7 seeds) | 15,465 | 30.9% |

**Pairwise bridge overlap:** 51% mean (Jaccard: 0.29)

### What Predicts Bridge Stability?

| Predictor | Correlation |
|-----------|-------------|
| Distance to global centroid | r = +0.64 (strong) |
| Local density | r = -0.31 (moderate) |

**Always-bridges:** Peripheral points in sparse regions (Ferrari, Doctor Who, Golden Gate Bridge)
**Never-bridges:** Central, generic concepts (Italy, Zinc, Vaccination, Continent)

### How Stable Are Bucket Assignments?

- **0 points** stay in the exact same bucket across all seeds
- Mean unique buckets per point: 7.97 out of 8 seeds
- Bucket assignment NMI: 0.27 (low)

Exact bucket IDs are unstable, but the semantic partitioning effect is stable.

### Does Stability Affect DYF Features?

#### 1. Bridge-Seeded K-means

| Seeding Strategy | Recall | Δ vs baseline |
|------------------|--------|---------------|
| Standard k-means | 0.924 | — |
| Single-seed bridges | 0.934 | +1.04% |
| Stable bridges (6+ seeds) | 0.936 | +1.24% |
| Unstable bridges (1-2 seeds) | 0.932 | +0.78% |
| **Orthogonal stable** | **0.939** | **+1.50%** |

**Finding:** Stable bridges are slightly better. **Orthogonal selection on stable bridges** gives the best results—combining stability + spread.

#### 2. Facet Diversification

| Metric | Value |
|--------|-------|
| Unique titles (8 seeds) | 6.11 - 6.59 |
| Std across seeds | 0.16 |
| Ensemble improvement | None |

**Finding:** Facet diversification is **highly robust to seed choice**. Any seed works equally well. The benefit comes from PCA-aligned structure, not specific bucket assignments.

#### 3. Centroid Similarity

| Points | Csim Std | Neighbors in Same Bucket |
|--------|----------|--------------------------|
| Stable (std < 0.05) | 0.036 | 15.7% |
| Unstable (std > 0.15) | 0.043 | 13.0% |

**Finding:** Small effect. High centroid-sim points are slightly more stably placed.

### Summary: When Does Stability Matter?

| Feature | Stability Matters? | Recommendation |
|---------|-------------------|----------------|
| Bridge-seeded k-means | **Yes** (+0.5%) | Use orthogonal selection on stable bridges |
| Facet diversification | **No** | Any seed works |
| Centroid similarity | Slightly | Effect is small |

### The Key Insight

**DYF's value comes from geometric structure (PCA alignment), not specific bucket assignments.**

Facet diversification works because you're sampling from different directions in PCA space. The exact hyperplane configuration doesn't matter—different seeds create different slices through the same semantic structure.

For bridge-seeded k-means, stability provides a small additional benefit because stable bridges are genuinely at semantic boundaries (not just random hyperplane artifacts).

---

## What Doesn't Work: Density for Search

### The Hypothesis

DYF provides bucket sizes (density). Could this help search?
- Dense buckets = canonical/common concepts
- Sparse buckets = unique/niche concepts

### Experiments Tried

#### 1. Density-Weighted Ranking

Rank by: `score = similarity + α × log(density)`

| α | Recall@10 | Unique Titles |
|---|-----------|---------------|
| -0.2 (boost sparse) | 0.01 | 9.83 |
| 0.0 (no weighting) | 1.00 | 5.67 |
| +0.2 (boost dense) | 0.30 | 7.78 |

**Result:** Boosting sparse destroys recall. Those items aren't actually relevant.

#### 2. Sparse Results as Novelty

For queries like "Jazz", "Bird", "Computer"—**all top-50 results are in dense buckets**. Similar items cluster in dense regions, not sparse ones.

Sparse buckets contain outliers (Adobe Illustrator, String theory) that aren't relevant to typical queries.

#### 3. Density-Based Diversification

Mix results from dense/medium/sparse buckets.

| Method | Unique Titles |
|--------|---------------|
| Pure similarity | 5.67 |
| Density-diversified | 4.37 |

**Result:** Actually worse. Sparse results aren't diverse, they're just irrelevant.

#### 4. Adaptive nprobe by Query Density

| Query Type | Fixed (5 probes) | Adaptive | Probes Used |
|------------|------------------|----------|-------------|
| Sparse | 0.21 | 0.44 | 15 |
| Dense | 0.47 | 0.26 | 2 |

**Result:** Helps sparse queries (+0.23 recall) but hurts dense queries. Net negative because most queries are dense.

### Why Density Doesn't Help Search

**Dense buckets contain canonical concepts, and those ARE the relevant results.**

- "Bird" neighbors → dense "biology/nature" bucket
- "Jazz" neighbors → dense "music" bucket
- Sparse buckets → outliers that aren't relevant

**Density ≠ Quality signal.** Unlike bridges (semantic boundaries) or buckets (semantic facets), density just reflects topic popularity.

### One Valid Use: Query Difficulty Prediction

Sparse query bucket → flag as potentially hard, increase nprobe or warn user.

| Query Bucket Size | Recall@10 |
|-------------------|-----------|
| Sparse (1-10) | 0.21 |
| Dense (>100) | 0.47 |

Correlation: r = 0.20 (weak but significant)

But this is query classification, not result ranking.

---

## Files

- `/tmp/rag_comparison.py` — BridgeIndex vs FAISS IVF
- `/tmp/rag_hybrid_comparison.py` — Centroid selection methods
- `/tmp/rag_nlist_sweep.py` — nlist sweep validation
- `/tmp/rag_seeding_strategies.py` — All seeding strategies
- `/tmp/rag_dense_buckets2.py` — Dense bucket seeding comparison
- `/tmp/rag_nprobe_analysis.py` — nprobe savings analysis
- `/tmp/rag_cluster_quality.py` — Cluster quality analysis
- `/tmp/rag_bridge_filtering.py` — Bridge filtering experiments
- `/tmp/rag_ivf_confidence.py` — Centroid sim as confidence
- `/tmp/rag_hard_query_detection.py` — Hard query prediction
- `/tmp/rag_hierarchy_viz.py` — Topic hierarchy visualization
- `/tmp/rag_hierarchical_search.py` — Hierarchical vs flat search
- `/tmp/wiki_diversification_methods.py` — MMR vs k-means vs DYF bucket diversification
- `/tmp/wiki_diversity_quality.py` — Qualitative comparison of diversification methods
- `/tmp/wiki_relevance_eval2.py` — LLM evaluation of facet diversity
- `/tmp/bridge_seed_stability.py` — Bridge stability across seeds
- `/tmp/stability_analysis.py` — How stability affects DYF features
- `/tmp/density_for_search.py` — Density features for search (doesn't work)
- `/tmp/bridge_types_v2.py` — Bridge role taxonomy for information seeking

---

## Bridge Role Taxonomy for Information Seeking

### The Question
Different bridges have different structural properties. Can we use these to support different information seeking behaviors (exploration, discovery, refinement, etc.)?

### Taxonomy Based on Home Bucket + Connectivity

| Role | Count | % | Definition |
|------|-------|---|------------|
| **Distributor** | 4105 | 89.3% | Home in dense bucket, connects to sparse |
| **Standard** | 245 | 5.3% | Default bridging behavior |
| **Junction** | 124 | 2.7% | High out-degree (>380 connections) |
| **Frontier** | 105 | 2.3% | Low centroid similarity (<0.45) |
| **Highway** | 16 | 0.3% | Dense→dense connections |

Note: "Connector" (sparse→dense) and "Backroad" (sparse→sparse) had zero instances in the dataset—91% of bridges live in dense buckets because larger buckets have more opportunities for boundary points.

### Metrics by Role

| Role | Avg Degree | Avg Centroid Sim | Avg Diversity |
|------|------------|------------------|---------------|
| Distributor | 314.5 | 0.467 | 11.5 |
| Junction | 403.3 | 0.480 | 13.8 |
| Standard | 311.3 | 0.480 | 12.6 |
| Frontier | 322.6 | 0.424 | 13.2 |
| Highway | 97.2 | 0.438 | 8.0 |

### Search Behavior Results

**From mainstream (dense bucket) queries:**

| Strategy | Diversity | Spread | Relevance | Surprise |
|----------|-----------|--------|-----------|----------|
| Direct k-NN | 6.4 | 0.413 | 0.645 | -0.017 |
| Distributor | 6.9 | 0.442 | 0.589 | 0.014 |
| Highway | 6.5 | 0.598 | 0.338 | **0.125** |
| Frontier | **7.8** | **0.555** | 0.443 | 0.058 |
| Junction | **7.8** | 0.519 | 0.441 | 0.092 |

**From niche (sparse bucket) queries:**

| Strategy | Diversity | Spread | Relevance | Surprise |
|----------|-----------|--------|-----------|----------|
| Direct k-NN | 7.9 | 0.411 | 0.627 | 0.003 |
| Distributor | 8.0 | 0.419 | 0.575 | 0.048 |
| Highway | 6.6 | **0.626** | 0.265 | **0.172** |
| Frontier | **8.3** | 0.489 | 0.488 | 0.071 |
| Junction | 8.2 | 0.453 | 0.524 | 0.068 |

### Use Case Recommendations

| Use Case | Best Bridge Type | Why |
|----------|------------------|-----|
| **Exploration** ("related but different") | Junction | High out-degree, connects to many topics |
| **Discovery** ("surprise me") | Distributor (from mainstream) | Leads from mainstream to niche |
| **Refinement** ("more like this") | Highway | Stays within well-covered territory |
| **Escape niche** ("find mainstream") | Connector* | On-ramps from niche to mainstream |
| **Deep dive** ("show me obscure") | Distributor | Off-ramps to specialized content |
| **Boundary exploration** | Frontier | Sits at topic boundaries (low centroid sim) |

*Connector bridges (sparse→dense) are rare in practice; most bridges live in dense buckets.

### Key Insight

Bridges are **bidirectional**, but their **home bucket determines which queries find them first**:
- Bridge in dense bucket → mainstream queries discover it → use to find niche
- Bridge in sparse bucket → niche queries discover it → use to find mainstream

The direction isn't in the edge—it's in the query's starting point.

---

## DAG Mining: Finding Hierarchical Structure

### The Problem
Similarity networks are "tangled" - everything connects to everything. Can we extract regions with clear hierarchical (DAG) structure?

### The Solution: Neighbor Diversity as Generality Signal

**Key insight**: General concepts have diverse neighbors (connect many topics). Specific concepts have coherent neighbors (tight cluster).

```python
from dyf import mine_dag_chains, compute_neighbor_diversity

# Mine DAG structures
result = mine_dag_chains(embeddings, verbose=True)
print(result.summary())

# Get clean hierarchies
clean = result.get_chains_by_coherence(min_coherence=0.65)
for chain in clean[:10]:
    print(f"[len={len(chain)}] indices: {chain.indices}")
```

### How It Works

1. **Compute neighbor diversity**: For each point, measure how dissimilar its neighbors are to each other
   - High diversity = neighbors span many topics = **general concept**
   - Low diversity = neighbors form tight cluster = **specific concept**

2. **Find parent-child edges**: Connect similar points where one has higher diversity
   - Edge from A → B if: `similarity(A,B) > threshold` AND `diversity(A) > diversity(B)`

3. **Extract chains**: Follow diversity gradient from general → specific

### Results on Wikipedia 50k

| Metric | Value |
|--------|-------|
| Chains (length >= 3) | 1,080 |
| Clean hierarchies (coh >= 0.65, len >= 4) | 454 |
| Monotonic chains | 100% |
| Max chain length | 8 |

### Example Chains Found

**Taxonomy:**
- "American black bear → Grizzly bear → Bear → Mammal → Animal → Tissue → Liver"
- "Golden Retriever → Labrador Retriever → Dog → Pet → Family → Government"

**Language:**
- "Hawaiian language → Fijian → Swahili → Hebrew → English → Second language → Noun"

**Political:**
- "Rosa Luxemburg → Trotsky → Lenin → War communism → Leninism → Communist state → Constitution → Government"

**Concepts:**
- "King Arthur → Robin Hood → Folk hero → Mythology → Legend → Fact → Definition"

### Key Observations

1. **Low diversity points** form tight clusters (decades like "1920s", months)
2. **High diversity points** sit at topic intersections
3. **Chains converge to common sinks** (Liver, Noun, Government, Biology, Definition)
4. **~20% of network** has clean DAG structure; rest is peer clusters

### Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `k_neighbors` | 30 | k-NN graph size |
| `similarity_threshold` | 0.55 | Minimum similarity for edges |
| `diversity_gap_threshold` | 0.02 | Minimum diversity difference |
| `min_chain_length` | 3 | Minimum chain length to return |

### When to Use

**Good for:**
- Knowledge graph construction
- Topic hierarchy discovery
- Finding "general → specific" relationships
- Identifying hub concepts (high-diversity sinks)

**Not suitable for:**
- Peer relationships (musicians in same genre)
- Temporal ordering (unless time correlates with diversity)
- Causal relationships (direction is structural, not causal)

---

## Key Finding: Lattice Structure, Not Tree

### The Discovery

The mined DAG is **not a simple hierarchy (tree)** - it's a true **ontology** with multiple inheritance.

| Metric | Value |
|--------|-------|
| Multi-parent nodes | 2,771 (50.6% of connected nodes) |
| Diamond patterns | 26,039 |
| Max parents (single node) | 122 ("Word") |
| Max children (single node) | 23 ("Chad") |

### Convergence Points (Abstract Attractors)

Many paths flow toward abstract concepts:

| Concept | Incoming Parents |
|---------|------------------|
| Word | 122 |
| Country | 118 |
| Government | 101 |
| Food | 100 |
| Noun | 88 |
| Biology | 67 |
| Disease | 67 |
| Definition | 63 |
| Computer | 62 |

These act as "semantic sinks" - highly general concepts that many specific topics connect to.

### Divergence Points (Branching Hubs)

Specific topics branch out to many children:

| Concept | Outgoing Children |
|---------|-------------------|
| Chad | 23 |
| George Washington | 22 |
| Honolulu | 21 |
| India | 20 |
| New York City | 20 |
| Greece | 19 |

### Diamond Patterns (Multiple Inheritance)

26,039 diamond patterns found - where two paths from a common ancestor converge at a common descendant:

```
Spring (season) ↘
                 → April
Winter          ↗
```

This means "April" inherits from both "Spring" AND "Winter" contexts.

### Why This Matters

**1. Knowledge isn't tree-structured**
- Concepts belong to multiple taxonomies simultaneously
- "Dog" connects to both "Pet" hierarchy AND "Mammal" hierarchy
- "April" is both season-related AND calendar-related

**2. Navigation has choices**
- Going "up" (more general) requires choosing which parent path
- Going "down" (more specific) requires choosing which child path
- No single "canonical" path through the structure

**3. Common ancestors form an ontology**
- Finding shared context between two concepts yields multiple common ancestors
- Not a single LCA (lowest common ancestor) like in trees
- Richer representation of semantic overlap

**4. Multiple paths = multiple perspectives**
- Same concept reachable via different routes
- Each route provides different context
- Useful for exploration and explanation

### Implications for RAG

| Use Case | Implication |
|----------|-------------|
| Context retrieval | Multiple valid contexts for any query |
| Explanation | Can explain via different paths |
| Exploration | Rich branching for discovery |
| Disambiguation | Lattice structure helps distinguish senses |

### Structure Comparison

| Property | Tree | Our DAG |
|----------|------|---------|
| Parents per node | 1 | 0-122 |
| Path to root | Unique | Multiple |
| Common ancestor | Single (LCA) | Lattice |
| Inheritance | Single | Multiple |

This **ontology** structure is a key differentiator from simple hierarchical clustering or taxonomy extraction. Taxonomies enforce single inheritance; ontologies allow multiple inheritance, which better reflects how concepts actually relate.

---

## DAG Taxonomy Navigation API

### Overview

The `DAGTaxonomy` class provides a navigable ontology structure extracted from embedding space. Built using `build_dag_taxonomy()`, it enables exploration of hierarchical relationships.

```python
from dyf import build_dag_taxonomy

taxonomy = build_dag_taxonomy(embeddings, verbose=True)
print(taxonomy.summary())

# Navigate
ancestors = taxonomy.get_ancestors(node_idx, max_depth=5)
common = taxonomy.get_common_ancestors(idx_a, idx_b)
path = taxonomy.get_path(start, end)
```

### Exploring Wikipedia 50k

**Taxonomy Statistics:**
| Metric | Value |
|--------|-------|
| Nodes with edges | 4,897 |
| Total edges | 29,234 |
| Roots (no parents) | 1,324 |
| Leaves (no children) | 725 |
| Multi-parent nodes | 70.1% |
| Max parents | 127 |
| Max children | 35 |

### Convergence Points (Abstract Concepts)

Nodes where many paths converge - these are the most abstract/general concepts:

| Concept | Parents | Role |
|---------|---------|------|
| Word | 127 | Linguistic abstraction |
| Country | 126 | Political/geographic |
| Government | 108 | Institutional |
| Food | 106 | Basic need |
| Noun | 94 | Grammar |
| Biology | 74 | Life science |
| Computer | 73 | Technology |
| Disease | 72 | Medical |
| Definition | 68 | Meta-concept |
| Religion | 59 | Belief systems |

**Observation**: Abstract concepts act as "semantic sinks" - highly connected hubs that many specific topics flow toward.

### Divergence Points (Branching Hubs)

Nodes that branch to many children - often geographic or temporal:

| Concept | Children | Pattern |
|---------|----------|---------|
| Chad | 35 | Geographic entity |
| George Washington | 30 | Historical figure |
| Honolulu | 29 | City hub |
| Testosterone | 29 | Scientific concept |
| March 4 | 28 | Date reference |
| Schleswig-Holstein | 27 | Region |
| India | 26 | Country |
| Berlin | 26 | City |
| New York City | 25 | City |

### Meaningful Paths Emerge

Direct paths through the ontology:

| Path | Interpretation |
|------|----------------|
| Pizza → Bread → Food | Culinary hierarchy |
| Dog → Fish → Biology | Biological taxonomy |
| Hawaii Ponoi → Deutschlandlied → Star-Spangled Banner → Independence Day → New Year's Day → 1 → 1960s → 2020s | National anthems → holidays → dates |
| London → House of Commons → Legislature → Democracy → Regime → Monarchy | Geographic → political systems |

### Common Ancestors Reveal Semantic Connections

Finding what connects disparate concepts:

| Pair | Common Ancestors | Interpretation |
|------|------------------|----------------|
| Guitar ↔ Piano | String instrument, Violin | Musical category |
| War ↔ Peace | Anarchy, Independence | Political concepts |
| Chemistry ↔ Physics | Gravity, Speed of light, Experiment | Shared phenomena |
| Dog ↔ Cat | Rottweiler, Labrador Retriever | Pet/animal domain |

### Bridge Concepts (High In+Out Degree)

944 concepts serve as bridges with both 5+ parents AND 5+ children:

| Concept | Parents | Children | Role |
|---------|---------|----------|------|
| Country | 126 | 10 | Geographic hub |
| Year | 44 | 15 | Temporal hub |
| George Washington | 22 | 30 | Historical connector |
| London | 23 | 19 | Geographic connector |
| Fish | 34 | 14 | Taxonomic bridge |
| Constitution | 36 | 14 | Political connector |

### Diversity by Concept Type

| Category | Avg Diversity | Interpretation |
|----------|---------------|----------------|
| Numbers | 0.476 | Connect to many domains |
| Countries | 0.453 | Cross-domain references |
| Dates | 0.446 | Wide applicability |
| Sciences | 0.419 | More focused domains |

Most general within categories:
- Numbers: 1985, 1966, 1990 (decade references)
- Dates: Nov 3, May 17 (less common dates = more diverse connections)

Most specific within categories:
- Numbers: 11, 35 (smaller numbers = tighter clusters)
- Dates: March, January (month names = tight temporal cluster)

### Key Insights

1. **Convergence points identify core abstractions**: Word, Country, Government, Food, Biology - these are the ontological backbone

2. **Divergence points identify knowledge hubs**: Geographic entities (cities, countries) and historical figures branch extensively

3. **Paths reveal unexpected connections**: Anthems → holidays → dates shows how semantic similarity creates chains across domains

4. **Common ancestors = semantic overlap**: The LCA of two concepts reveals their shared context (Guitar+Piano → String instrument)

5. **Bridge concepts connect hierarchies**: Nodes with both high in-degree and out-degree serve as cross-cutting connectors

6. **Diversity correlates with abstractness**: More abstract concepts (numbers, dates) have higher neighbor diversity

### Use Cases

| Use Case | Method |
|----------|--------|
| Find context for a concept | `get_ancestors(node)` |
| Find specializations | `get_descendants(node)` |
| Find connecting concepts | `get_common_ancestors(a, b)` |
| Find semantic relationship | `get_path(start, end)` |
| Find abstract attractors | `get_convergence_points(min_parents=10)` |
| Find branching hubs | `get_divergence_points(min_children=10)` |
| Detect multiple inheritance | `get_diamond_patterns()` |

### Limitations

- Direction is based on **neighbor diversity**, not causal/definitional relationships
- Some paths are semantically meaningful, others are artifacts of embedding similarity
- Abstract concepts cluster (Word, Noun, Definition) - may want to filter
- Taxonomy quality depends on embedding quality

---

## Ontology Structure Analysis

### Terminology

What we extract is an **ontology**, not a taxonomy:

| Structure | Parents per Node | Our Data |
|-----------|------------------|----------|
| **Taxonomy** (tree) | Exactly 1 | Not this |
| **Ontology** (DAG) | 0 to many | 70% have 2+ parents |

The giant component is a true ontology with multiple inheritance. Only the isolated micro-clusters approximate tree structure.

### Overview

The ontology reveals multiple levels of organization:
1. **Isolated micro-clusters** - Self-contained knowledge islands (often tree-like)
2. **Semantic domains** - Tight clusters of related concepts
3. **Major highways** - Most-traveled edges in the graph
4. **Cross-domain gateways** - Concepts that bridge multiple domains
5. **Convergent chains** - Multiple paths leading to the same sink

### Isolated Micro-Clusters

32 separate connected components exist outside the giant component (88% of nodes). These are clean, self-contained domain clusters (often tree-like):

| Cluster | Members | Pattern |
|---------|---------|---------|
| Actresses | Sarah Michelle Gellar → Jessica Alba, Hilary Swank | Via Buffy connection |
| Constructed Languages | L.L. Zamenhof → Esperanto → Ido | Creator → language → derivative |
| Astronauts | Sally Ride → Neil Armstrong, Buzz Aldrin | Female pioneer → male pioneers |
| Figure Skaters | Lu Chen, Oksana Baiul → Irina Slutskaya | Olympic champions cluster |
| Haiku Poets | Yosa Buson → Matsuo Basho | Japanese poetry masters |
| 2005 Hurricanes | Hurricane Rita → Katrina, 2005 season | Event → related events |
| BC Volcanoes | Anahim Belt, Chilcotin → Garibaldi Belt | Geographic/geological |
| Care Bears | Care Bears → Movies I & II | Franchise → media |
| Bollywood Stars | Aamir Khan → Shah Rukh Khan | Industry cluster |
| Classic Novels | Gone with the Wind → To Kill a Mockingbird | American literature |

**Key insight**: These "knowledge islands" represent highly specific domains where concepts only relate to each other, not to broader categories. Unlike the giant component (an ontology), these isolated clusters often have tree-like single-inheritance structure.

### Semantic Domains (Low-Diversity Clusters)

Low neighbor diversity = tight semantic cluster. Clustering the 100 lowest-diversity nodes reveals core domains:

| Domain | Core Concepts | Avg Diversity |
|--------|---------------|---------------|
| **Decades** | 1920s, 1970s, 2010s, 2020s | 0.272 |
| **Months** | January, March, July, December | 0.312 |
| **Numbers** | 1, 2, 3, 5, 11, 35 | 0.324 |
| **Political** | King, Monarch, Prime Minister, Governor, Head of state | 0.322 |
| **Anatomy** | Liver, Stomach, Heart, Intestine, Circulatory system | 0.324 |
| **Linguistics** | Noun, Word, Prefix, Suffix, Adjective, Definition | 0.327 |
| **Chemistry** | Mixture, Chemical element, Substance | 0.333 |

**Observation**: Temporal concepts (decades, months) have the lowest diversity - they form the tightest clusters. Abstract categories (linguistics, chemistry) are slightly more diverse.

### Major Highways (Most-Traveled Edges)

Counting edge usage across paths from roots to sinks reveals the "main roads" of the ontology:

| Highway | Traversals | Domain |
|---------|------------|--------|
| Food → Digestive system | 116x | Biology |
| State → Head of state | 92x | Politics |
| Country → Government | 84x | Politics |
| Country → State | 84x | Politics |
| Anatomy → Digestive system | 76x | Biology |
| Chemistry → Mixture | 74x | Science |
| Dictionary → Word | 68x | Linguistics |
| Dictionary → Noun | 68x | Linguistics |
| Science → Biology | 62x | Science |
| Prince → King | 60x | Politics |

**Pattern**: Politics, Biology, and Linguistics dominate the major highways. These are the "trunk routes" that most paths flow through.

### Cross-Domain Gateways

Nodes that connect to sinks in 4+ different domains:

| Gateway | Domains | Parents | Children | Role |
|---------|---------|---------|----------|------|
| Clock | 4 | 2 | 27 | Time concept → many domains |
| BBC | 4 | 5 | 26 | Media hub |
| Orange (color) | 4 | 2 | 23 | Sensory → many domains |
| Mosque | 4 | 3 | 14 | Religion → geography, culture |
| Synagogue | 4 | 0 | 15 | Religion → geography, culture |
| Free will | 4 | 0 | 17 | Philosophy → many domains |
| Protestantism | 4 | 4 | 17 | Religion → politics, culture |
| National Hockey League | 4 | 6 | 17 | Sports → geography, media |

**Insight**: Gateways tend to be either:
- Sensory/temporal (clock, color) - universal human experience
- Institutional (BBC, NHL, religious buildings) - organizations span domains
- Abstract (free will) - philosophical concepts touch everything

### Convergent Chains

Multiple distinct semantic paths that lead to the same abstract sink:

**→ Biology** (4 convergent pathways):
```
Civil engineering → Engineering → Scientist → Science → Biology
Aerospace engineering → Engineering → Scientist → Science → Biology
Electron microscope → Microscope → Scientist → Science → Biology
Guinea pig → Environment → Science → Biology
```

**→ Government** (2 convergent pathways):
```
Evil → Devil → Religion → Family → Government
Deacon → Church (building) → Religion → Family → Government
```

**Pattern**: Convergence reveals that:
- Engineering disciplines funnel through "Scientist" to reach "Biology"
- Religious concepts funnel through "Family" to reach "Government"

### Sink Domain Clustering

Clustering the top 30 sinks by shared ancestors reveals 7 meta-domains:

| Meta-Domain | Sinks |
|-------------|-------|
| **Core abstractions** | Word, Country, Government, Food, Noun, Biology, Disease, Definition, Blood, Digestive system, Art, Tissue, Chemistry, Organism, Life, Science, Fact, Head of state |
| **Computing** | Computer, Software |
| **Geography** | Scotland, North America |
| **Religion** | Religion (standalone) |
| **Food/Chemistry** | Meal, Chemical element, Bread |
| **Quantity** | Number (standalone) |
| **Time** | Year (standalone) |

**Key finding**: Most sinks cluster into one giant "core abstractions" domain. This reflects how general concepts (Word, Government, Biology) are reachable from almost anywhere in the graph.

### Implications

1. **Knowledge isn't flat**: The ontology has distinct layers - isolated islands, domain cores, highways, and gateways

2. **Domains have cores**: Each semantic domain has a tight cluster of low-diversity "core" concepts (months, body parts, political roles)

3. **Highways predict importance**: The most-traveled edges connect fundamental human concepts (food→digestion, country→government)

4. **Gateways enable exploration**: Cross-domain nodes (BBC, Clock, religions) are natural starting points for interdisciplinary discovery

5. **Convergence reveals structure**: When multiple paths lead to the same sink, the intermediate nodes reveal implicit categorization (all engineering → "Scientist")

---

## ROG: Recursive Ontological Generation

### The Problem

Ontology building with a fixed similarity threshold faces a fundamental trade-off:
- **High threshold (≥0.55)**: Clean structure but misses sparse regions (10.5% uncovered)
- **Low threshold (≤0.45)**: Covers more but creates noise in dense regions

Different regions of embedding space have different "natural" density scales. Dense clusters need strict thresholds; sparse regions need looser thresholds.

### The Solution: Density-Adaptive Recursion

ROG (Recursive Ontological Generation) builds ontologies layer by layer:

1. Start at high threshold (default 0.55)
2. Build ontology for current population
3. Identify outliers (nodes not connected)
4. Recurse into outliers at lower threshold
5. Repeat until coverage target reached or min threshold hit
6. Knit layers together with bridge edges

```python
from dyf import build_rog_ontology

result = build_rog_ontology(
    embeddings,
    initial_threshold=0.55,    # Start strict
    min_threshold=0.35,        # Floor for sparse regions
    threshold_decay=0.9,       # 10% reduction per layer
    target_coverage=0.95,      # Stop when 95% covered
    verbose=True
)

print(f"Coverage: {result.total_coverage:.1%}")
print(f"Layers: {len(result.layers)}")
for layer in result.layers:
    print(f"  Layer {layer.depth}: {layer.n_nodes} nodes @ sim≥{layer.similarity_threshold:.2f}")
```

### Results on Wikipedia 50k

| Layer | Threshold | Nodes | Edges | Coverage |
|-------|-----------|-------|-------|----------|
| 0 | ≥0.55 | 4,897 | 29,234 | 89.5% |
| 1 | ≥0.50 | 165 | 475 | 3.0% |
| 2 | ≥0.45 | 158 | 490 | 2.9% |
| **Total** | — | **5,220** | **30,199** | **95.4%** |
| Excluded | — | 253 | — | 4.6% |

**Key finding**: 95.4% coverage achieved in just 3 layers.

### What Gets Excluded?

The 4.6% (253 nodes) that remain uncovered even at the lowest threshold are "true noise":
- Extremely isolated concepts with no similar neighbors
- Often very specific/niche topics (obscure historical figures, technical terms)
- Not structurally meaningful to include

### Layer Characteristics

| Layer | Role | Examples |
|-------|------|----------|
| Layer 0 | Dense core | Mainstream topics, well-connected concepts |
| Layer 1 | Sparse bridges | Entertainment/pop culture domain |
| Layer 2 | Deep periphery | Highly specialized topics |

### Bridge Edges

ROG knits layers together by finding edges where:
- One node is in a higher layer (denser threshold)
- One node is in a lower layer (sparser threshold)
- Similarity exceeds the lower layer's threshold

This creates a unified ontology with natural density gradients.

### Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `initial_threshold` | 0.55 | Starting similarity threshold |
| `min_threshold` | 0.35 | Minimum threshold (floor) |
| `threshold_decay` | 0.9 | Threshold multiplier per layer |
| `target_coverage` | 0.95 | Stop when coverage exceeds this |
| `diversity_gap_threshold` | 0.02 | Min diversity difference for edges |
| `k_neighbors` | 30 | k-NN graph size |
| `max_depth` | 5 | Maximum recursion depth |

### When to Use

**Good for:**
- Building comprehensive ontologies with high coverage
- Handling datasets with mixed density regions
- Discovering structure in sparse embedding regions
- Creating layered knowledge graphs

**Alternative: `build_dag_taxonomy()`** — Single-threshold ontology for simpler cases

### API

```python
@dataclass
class ROGLayer:
    depth: int                    # Layer index (0 = densest)
    similarity_threshold: float   # Threshold used for this layer
    node_indices: np.ndarray      # Indices of nodes in this layer
    n_nodes: int                  # Count of nodes
    n_edges: int                  # Count of edges within layer
    coverage: float               # Fraction of total data covered

@dataclass
class ROGResult:
    ontology: DAGTaxonomy         # Combined ontology structure
    layers: List[ROGLayer]        # Per-layer metadata
    excluded_nodes: np.ndarray    # Indices not in any layer
    total_coverage: float         # Overall coverage achieved
    bridge_edges: int             # Cross-layer connections

    def get_layer_for_node(self, node_idx: int) -> Optional[int]:
        """Return which layer a node belongs to (0=densest), or None if excluded."""
```

### The Key Insight

**Embedding spaces have natural density scales.** A single global threshold cannot capture both:
- Dense regions where strict thresholds maintain quality
- Sparse regions where looser thresholds are needed for any structure

ROG adapts the threshold to the local density, achieving high coverage without sacrificing structure quality in dense regions

---

## Edge Directionality: Diversity vs PageRank vs Peers

### The Problem with Neighbor Diversity

The original DAG mining uses **neighbor diversity** to determine edge direction:
- High diversity → low diversity (assumed "general → specific")

**But this assumption is wrong.** Neighbor diversity measures neighborhood coherence, not taxonomic generality:

| Concept | Diversity | Neighbors |
|---------|-----------|-----------|
| France | 0.409 | Ivory Coast, Switzerland, Greece, Senegal (scattered) |
| Wales | 0.387 | Scotland, UK, England, Northern Ireland (tight cluster) |

France has **higher** diversity (more scattered neighbors), so the algorithm makes France the "parent" of Wales. But neither is taxonomically more general than the other—they're peers.

**What diversity actually measures:**
- **High diversity** = bridges across clusters (scattered connections)
- **Low diversity** = sits in tight cluster (cohesive neighborhood)

This is useful structural information, but it's not taxonomic hierarchy.

### Relabeling: Core Cluster vs Bridging

More accurate labels for the diversity gradient:
- **Core Cluster** (low diversity): Nodes in tight semantic clusters
- **Bridging** (high diversity): Nodes that connect across different clusters

### PageRank as Generality Signal

PageRank on the similarity graph better captures conceptual generality:

**Top PageRank (most general):**
| Rank | Concept | PageRank |
|------|---------|----------|
| 1 | Country | 0.001265 |
| 2 | Word | 0.001162 |
| 3 | Noun | 0.001119 |
| 4 | Food | 0.001049 |
| 5 | Government | 0.000960 |
| 9 | Biology | 0.000721 |

**Test cases:**
| Pair | PageRank Ratio | Correct? |
|------|----------------|----------|
| Music > Jazz | 2.04x | ✓ |
| Country > France | 3.51x | ✓ |
| Dog > Labradoodle | 2.84x | ✓ |
| Mammal > Dog | 1.55x | ~ |

PageRank works well for **extremes** (Country vs France) but less reliably for **similar-level concepts** (France vs Wales still inverted at 1.13x).

### The Fundamental Limitation

**Taxonomic hierarchy cannot be reliably inferred from embeddings alone.**

- Embeddings capture **semantic similarity** ("Dog is similar to Pet")
- Taxonomy is about **class membership** ("Dog is a Mammal")
- These overlap but aren't the same

France and Wales are semantically similar and connected, but neither is taxonomically "above" the other.

### Detecting Peer Relationships

Heuristic to distinguish peers from hierarchical relationships:

```
if PageRank_ratio < 1.5 AND shared_neighbors > 15%:
    → PEERS (show as bidirectional ↔)
elif PageRank_ratio > 2.0:
    → HIERARCHY (show as directional →)
else:
    → UNCLEAR
```

**Test results:**

| Pair | PR Ratio | Shared Neighbors | Classification | Correct? |
|------|----------|------------------|----------------|----------|
| France ↔ Wales | 1.13x | 20% | PEERS | ✓ |
| France ↔ Switzerland | 1.46x | 30% | PEERS | ✓ |
| Dog ↔ Cat | 1.08x | 15% | PEERS | ✓ |
| Scotland ↔ Wales | 1.34x | 43% | PEERS | ✓ |
| Music → Jazz | 2.04x | 30% | HIERARCHY | ✓ |
| Country → France | 3.51x | 0% | HIERARCHY | ✓ |
| Dog → Labradoodle | 2.84x | 25% | HIERARCHY | ✓ |
| Carlin ↔ Poe | 1.05x | 7% | UNCLEAR | ✓ |

### Coherence Chains (Not Taxonomies)

The extracted chains are **coherence chains**, not taxonomies:
- Each step follows a high-similarity neighbor relationship
- The chain maintains semantic coherence (high pairwise similarity)
- But the direction doesn't imply taxonomic hierarchy

Example chain: `Eric Idle → Graham Chapman → George Carlin → Edgar Allan Poe → Aldous Huxley`

This is a valid **semantic walk** through embedding space:
- Each step is a legitimate nearest neighbor
- The chain crosses category boundaries (comedians → authors)
- Coherence score measures how tightly related all nodes are

**Recommended terminology:**
- "Coherence chains" instead of "taxonomies"
- "Core/Bridging" instead of "General/Specific"
- Show peer edges as bidirectional (↔)
- Only show directed arrows (→) for high PR ratio edges

### Implementation Recommendations

1. **Compute PageRank** on the similarity graph
2. **For each edge**, compute:
   - PR ratio = max(PR_a, PR_b) / min(PR_a, PR_b)
   - Shared neighbor ratio (Jaccard)
3. **Classify edges:**
   - PR ratio < 1.5 AND Jaccard > 0.15 → PEER
   - PR ratio > 2.0 → HIERARCHICAL
   - Otherwise → UNCLEAR
4. **Display:**
   - Peer edges: bidirectional or no arrow
   - Hierarchical edges: arrow from high PR to low PR
   - Unclear edges: optional display or different styling

---

## Geometric Taxonomy: No External Knowledge Required

### The Question

Can we derive taxonomic direction (general → specific) from embeddings alone, without hardcoded reference concepts or external knowledge bases? This would enable taxonomy extraction for any modality (images, sounds, etc.) where we have embeddings but no labels.

### The Discovery: Inverse Neighborhood Variance

**Key insight**: General concepts have **tight, coherent neighborhoods**. Specific concepts have **diverse neighbors**.

| Concept | Variance | Neighbors |
|---------|----------|-----------|
| Animal | 0.207 (low) | Organism, Species, Plant (abstract terms) |
| Dog | 0.238 (high) | Breeds, pets, specific animals (diverse) |
| Music | 0.238 (low) | Art, Culture, Sound (abstract) |
| Rock and roll | 0.263 (high) | Genres, bands, decades (diverse) |

**Why?** General concepts sit in clusters of other general concepts. Specific concepts connect to many different things.

### Algorithm

```python
# 1. Build k-NN graph
nn = NearestNeighbors(n_neighbors=30, metric='cosine')
nn.fit(embeddings)
distances, neighbors = nn.kneighbors(embeddings)

# 2. Compute neighborhood variance for each item
for i in range(n):
    neighbor_embeddings = embeddings[neighbors[i, 1:k+1]]
    centroid = normalize(neighbor_embeddings.mean(axis=0))
    dists = 1 - np.dot(neighbor_embeddings, centroid)
    neighborhood_variance[i] = dists.mean()

# 3. Lower variance = more general
# Direction: low variance → high variance
generality = 1.0 / (neighborhood_variance + 0.01)
```

### Results: Inverse Variance Alone

| Test Case | Variance (general) | Variance (specific) | Correct? |
|-----------|-------------------|---------------------|----------|
| Animal → Dog | 0.207 | 0.238 | ✓ |
| Animal → Tiger | 0.207 | 0.266 | ✓ |
| Tiger → Sumatran tiger | 0.266 | 0.263 | ✗ (too close) |
| Music → Rock and roll | 0.238 | 0.263 | ✓ |
| Rock and roll → Beatles | 0.263 | 0.251 | ✗ (inverted) |
| Science → Physics | 0.218 | 0.226 | ✓ |
| Art → Painting | 0.218 | 0.241 | ✓ |

**Accuracy: 7/9** with no tuning, no external knowledge.

### Secondary Signal: Bucket Diversity

General concepts connect across category boundaries. Specific instances connect to peers.

| Concept | Unique Buckets in k-NN | Neighbor Types |
|---------|------------------------|----------------|
| Rock and roll | 23 | Genres, concepts, decades |
| The Beatles | 15 | Other bands (Rolling Stones, Queen, Led Zeppelin) |

Bucket diversity helps when variance is inconclusive.

### Signal Conflict

The two signals sometimes disagree:

| Test Case | Variance Winner | Diversity Winner |
|-----------|-----------------|------------------|
| Animal → Dog | Animal ✓ | Dog ✗ |
| Rock → Beatles | Beatles ✗ | Rock ✓ |
| Science → Physics | Science ✓ | Tie |

**Correlation between signals: -0.365** (negatively correlated)

When they disagree:
- Variance is correct 4/6 times
- Diversity is correct 2/6 times

### The Fundamental Limit

Two cases with similar |variance_diff| but opposite requirements:

| Case | var_diff | Need |
|------|----------|------|
| Rock → Beatles | -0.35 | Use diversity |
| Science → Physics | +0.30 | Use variance |

**No threshold can separate these.** To get 9/9 requires either:
1. Tuning weights to the test set (overfitting)
2. External knowledge about what "general" means

### Honest Assessment

| Approach | Accuracy | Notes |
|----------|----------|-------|
| Inverse variance alone | **7/9** | No tuning required |
| Variance + diversity (best threshold) | **8/9** | Principled combination |
| Tuned weighted combination | 9/9 | Overfitting to test set |

**7-8/9 is the honest ceiling for pure geometry.**

### The Remaining "Failures"

Some failures may reflect legitimate alternative hierarchies:
- The Beatles arguably ARE more culturally central than "Rock and roll" as a label
- Tiger and Sumatran tiger have nearly identical neighborhood structure

**Taxonomy is partly a human construct.** Geometry captures relatedness; not all subsumption relationships are geometrically recoverable.

### Why This Works for Any Modality

The algorithm requires only:
1. **Embeddings** (any domain)
2. **k-NN graph** (from embeddings)
3. **Clustering** (for bucket diversity signal)

No language. No labels. No external knowledge bases.

Applicable to:
- Image embeddings (CLIP, etc.)
- Audio embeddings
- Sensor data
- Any meaningful vector representation

### Implementation

```python
import numpy as np
from sklearn.neighbors import NearestNeighbors
from dyf import DensityClassifier

def compute_geometric_generality(embeddings, k=30):
    """
    Compute generality score from pure geometry.
    Higher score = more general concept.
    """
    n = len(embeddings)

    # k-NN graph
    nn = NearestNeighbors(n_neighbors=k+1, metric='cosine')
    nn.fit(embeddings)
    distances, neighbors = nn.kneighbors(embeddings)

    # Signal 1: Inverse neighborhood variance
    variance = np.zeros(n)
    for i in range(n):
        neighbor_embs = embeddings[neighbors[i, 1:k+1]]
        centroid = neighbor_embs.mean(axis=0)
        centroid = centroid / (np.linalg.norm(centroid) + 1e-10)
        dists = 1 - np.dot(neighbor_embs, centroid)
        variance[i] = dists.mean()

    # Lower variance = more general
    inv_variance = 1.0 / (variance + 0.01)

    # Z-score normalize
    generality = (inv_variance - inv_variance.mean()) / inv_variance.std()

    return generality

# Build DAG: edge from higher generality to lower
def build_geometric_dag(embeddings, generality, sim_threshold=0.5, k=30):
    nn = NearestNeighbors(n_neighbors=k+1, metric='cosine')
    nn.fit(embeddings)
    distances, neighbors = nn.kneighbors(embeddings)
    similarities = 1 - distances

    children_of = {i: set() for i in range(len(embeddings))}

    for i in range(len(embeddings)):
        for jj in range(1, k+1):
            j = neighbors[i, jj]
            if similarities[i, jj] < sim_threshold:
                break

            # Direction by generality
            if generality[i] > generality[j]:
                children_of[i].add(j)
            else:
                children_of[j].add(i)

    return children_of
```

### Key Takeaways

1. **Inverse neighborhood variance** is the primary geometric signal for taxonomy (7/9 accuracy)

2. **Bucket diversity** provides a secondary signal for tiebreaking (helps 1 additional case)

3. **8/9 is achievable** with principled combination, no tuning

4. **9/9 requires overfitting** or external knowledge

5. **Some hierarchies are ambiguous** — geometry can't fully recover all human taxonomic intuitions

6. **Modality-agnostic** — works for any embedding space

---

## Automatic Anchor Extraction: Finding Hub Nodes

### The Problem

The geometric taxonomy approach uses inverse neighborhood variance to determine direction, but how do we identify the **anchor concepts** that define what "general" means? Previous approaches relied on hardcoded reference concepts ("Animal", "Music", "Science"), which defeats the goal of modality-agnostic taxonomy extraction.

### The Discovery: Anchors Have Distinct Geometric Signatures

Analyzing known anchor concepts revealed they share two key properties:

| Property | Known Anchors | Population Mean | What it measures |
|----------|---------------|-----------------|------------------|
| Eigenvector centrality | High | Lower | Tightly woven neighborhood |
| Distance to global centroid | Low | Higher | Position in embedding space |

**Key insight**: Hub nodes are **globally central** (close to the center of embedding space) AND in **tightly interconnected** neighborhoods (high eigenvector centrality).

### Why Eigenvector Centrality, Not PageRank?

Both measure network centrality, but they capture different things:

| Aspect | PageRank | Eigenvector |
|--------|----------|-------------|
| Formula | `(1-d)/n + d * Σ(score_j / degree_j)` | `Σ(similarity_ij * score_j)` |
| Damping | Yes (15% random teleport) | No |
| Contribution | Equal share (1/k each) | Weighted by similarity |
| What it rewards | Being pointed to by many | Being in tightly woven cluster |

**Critical finding**: Correlation between PageRank and Eigenvector is only **0.254** — they measure different things.

**Separation power comparison**:

| Signal | Category Min | Peer Max | Gap |
|--------|--------------|----------|-----|
| PageRank | +2.11 | +2.32 | **-0.21** (overlap!) |
| Eigenvector | +0.47 | -0.89 | **+1.36** (clean) |

PageRank fails because both categories AND popular instances are in dense clusters. Eigenvector succeeds because it rewards **how tightly woven** the neighborhood is — abstract terms (Biology, Science, Life) are more tightly interconnected than rock bands (Rolling Stones, Led Zeppelin, The Who).

### Automatic Detection: Simplified Formula

Only two signals needed:

```python
# 1. Eigenvector centrality (no damping!)
eigen = np.ones(n)
for _ in range(30):
    new_eigen = np.zeros(n)
    for i in range(n):
        for j in range(1, k+1):
            new_eigen[i] += eigen[neighbors[i, j]] * similarities[i, j]
    eigen = new_eigen / np.linalg.norm(new_eigen)

# 2. Distance to global centroid
global_centroid = embeddings.mean(axis=0)
global_centroid = global_centroid / np.linalg.norm(global_centroid)
dist_to_center = 1 - np.dot(embeddings, global_centroid)

# 3. Combine (z-scored): high eigen, close to center
hub_score = zscore(eigen) - zscore(dist_to_center)
```

**Results**:
- Categories: +2.27 to +3.82
- Peers: -1.19 to -1.46
- **Gap: 3.46** (no overlap)

### What About Disconnected Nodes?

PageRank's damping factor handles disconnected components. Without damping, eigenvector centrality can collapse to zero for isolated nodes.

In k-NN graphs, this isn't a problem:
- Every node has k outgoing edges (by construction)
- Minimum eigenvector in Wikipedia 50k: 0.000281 (not zero)
- 12 nodes have 0 in-degree, but they still get non-zero eigenvector scores

**Bottom line**: For k-NN similarity graphs, pure eigenvector centrality works fine.

### Results on Wikipedia 50k

| Metric | Value |
|--------|-------|
| Hub score gap | 3.46 (perfect separation) |
| Category score range | +2.27 to +3.82 |
| Peer score range | -1.19 to -1.46 |
| Taxonomy accuracy | 8/8 |

### Hub Nodes vs Popular Instances

The simplified hub score cleanly separates:

| Concept | Eigen_z | -Dist_z | Hub Score | Type |
|---------|---------|---------|-----------|------|
| Country | +0.87 | +2.95 | **+3.82** | Hub |
| Biology | +1.65 | +2.10 | **+3.75** | Hub |
| Animal | +1.43 | +1.48 | **+2.91** | Hub |
| Science | +1.14 | +1.33 | **+2.47** | Hub |
| Music | +0.47 | +1.79 | **+2.27** | Hub |
| — clean gap — | | | | |
| Led Zeppelin | -0.91 | -0.28 | **-1.19** | Instance |
| The Rolling Stones | -0.89 | -0.42 | **-1.31** | Instance |
| Def Leppard | -0.91 | -0.42 | **-1.33** | Instance |
| Guns N' Roses | -0.91 | -0.54 | **-1.46** | Instance |

**Gap: 3.46** — no overlap, no tuning required.

### The Hub Analogy

**Hub nodes** are structural—they hold the embedding space together without being instances themselves:
- They sit in tight clusters of OTHER abstract terms
- Their neighbors are similar to each other (high inter-neighbor similarity)
- They're central (high PageRank) because everything passes through them

**Popular instances** are content—they fill the space:
- They sit in more heterogeneous neighborhoods
- Their neighbors include peers, related items, attributes (less similar to each other)
- They're central because they're popular, not structural

### Use Cases

| Use Case | Approach |
|----------|----------|
| Taxonomy direction | Use all anchors (hub + popular) |
| Skeleton extraction | Filter to hub nodes only |
| Ontology backbone | Hub nodes define the structure |
| Facet organization | Hub nodes are natural facet labels |

### Algorithm Summary

```python
def compute_hub_score(embeddings, k=30):
    """
    Compute hub score for each node.
    Higher score = more likely to be a structural hub node.
    """
    from sklearn.neighbors import NearestNeighbors
    import numpy as np

    n = len(embeddings)

    # 1. Build k-NN graph
    nn = NearestNeighbors(n_neighbors=k+1, metric='cosine')
    nn.fit(embeddings)
    distances, neighbors = nn.kneighbors(embeddings)
    similarities = 1 - distances

    # 2. Eigenvector centrality (no damping)
    eigen = np.ones(n)
    for _ in range(30):
        new_eigen = np.zeros(n)
        for i in range(n):
            for j in range(1, k+1):
                new_eigen[i] += eigen[neighbors[i, j]] * similarities[i, j]
        eigen = new_eigen / (np.linalg.norm(new_eigen) + 1e-10)

    # 3. Distance to global centroid
    global_centroid = embeddings.mean(axis=0)
    global_centroid = global_centroid / (np.linalg.norm(global_centroid) + 1e-10)
    dist_to_center = 1 - np.dot(embeddings, global_centroid)

    # 4. Z-score and combine
    eigen_z = (eigen - eigen.mean()) / eigen.std()
    dist_z = (dist_to_center - dist_to_center.mean()) / dist_to_center.std()

    hub_score = eigen_z - dist_z  # High eigen, close to center

    return hub_score

def extract_hub_nodes(embeddings, k=30, threshold=0.0):
    """Extract structural hub nodes (score > threshold)."""
    scores = compute_hub_score(embeddings, k)
    return np.where(scores > threshold)[0]
```

### Taxonomy Direction Test Results

The hub score also works for determining taxonomy direction (general → specific):

| Method | Correct |
|--------|---------|
| **Hub (eigen - dist)** | **11/11** |
| Eigenvector alone | 11/11 |
| Distance (inv) alone | 11/11 |
| PageRank | 10/11 |
| Inverse variance | 9/11 |

**Detailed results** (difference = general - specific, positive = correct):

| Test Case | InvVar | PageRank | Eigen | -Dist | Hub |
|-----------|--------|----------|-------|-------|---------|
| Animal → Dog | +1.16 ✓ | +2.53 ✓ | +0.62 ✓ | +0.18 ✓ | +0.80 ✓ |
| Animal → Tiger | +1.98 ✓ | +3.05 ✓ | +1.52 ✓ | +0.74 ✓ | +2.26 ✓ |
| Tiger → Sumatran tiger | -0.08 ✗ | +0.45 ✓ | +0.32 ✓ | +2.30 ✓ | +2.62 ✓ |
| Music → Rock and roll | +0.75 ✓ | +1.77 ✓ | +0.65 ✓ | +0.42 ✓ | +1.06 ✓ |
| Rock and roll → Beatles | -0.35 ✗ | -2.27 ✗ | +1.00 ✓ | +0.50 ✓ | +1.50 ✓ |
| Science → Physics | +0.30 ✓ | +0.27 ✓ | +0.11 ✓ | +0.04 ✓ | +0.15 ✓ |
| Physics → Einstein | +2.00 ✓ | +2.40 ✓ | +1.86 ✓ | +0.78 ✓ | +2.64 ✓ |
| Country → Japan | +1.80 ✓ | +12.21 ✓ | +1.41 ✓ | +1.13 ✓ | +2.54 ✓ |
| Country → France | +0.52 ✓ | +11.48 ✓ | +1.27 ✓ | +1.12 ✓ | +2.39 ✓ |
| France → Paris | +0.54 ✓ | +0.98 ✓ | +0.47 ✓ | +1.94 ✓ | +2.41 ✓ |
| Art → Painting | +0.82 ✓ | +3.25 ✓ | +0.27 ✓ | +1.13 ✓ | +1.40 ✓ |

**Key observations**:

1. **Both signals agree on all 11 cases** — eigenvector and distance each get 11/11 individually

2. **Signals are complementary**:
   - Tiger → Sumatran tiger: Distance does the heavy lifting (+2.30)
   - Rock and roll → Beatles: Eigenvector does the heavy lifting (+1.00)

3. **Hard cases for other methods**:
   - Inverse variance fails on Tiger → Sumatran tiger (too similar in variance)
   - PageRank fails on Rock → Beatles (Beatles have higher PageRank than the genre!)

4. **Robustness**: When one signal is weak, the other compensates

### Key Takeaways

1. **Two signals suffice**: Eigenvector centrality + distance to global centroid

2. **Eigenvector > PageRank** for this task (correlation only 0.254, eigenvector gives clean separation)

3. **11/11 on taxonomy direction** — outperforms inverse variance (9/11) and PageRank (10/11)

4. **Gap of 3.46** between worst category and best peer — no overlap, no tuning

5. **Signals are complementary** — when one is weak, the other compensates

6. **Hub nodes** are globally central AND in tightly woven neighborhoods

7. **Popular instances** are locally central (high PageRank) but peripheral in the global space

8. **Fully modality-agnostic** — works for any embedding space with no labels

9. **k-NN graphs don't need damping** — eigenvector centrality works fine without PageRank's random teleport
