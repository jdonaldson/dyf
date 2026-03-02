# Embedding Model Progression: MiniLM → Nomic → mxbai

## Two Datasets, Two Stories

We have results from two different evaluation contexts:

1. **Energy devices subset** (8K sample, 3 low-cardinality axes: GMDN family,
   polarity, disposability) — high purity numbers, Fisher effects visible
2. **Full GUDID 50K** (5K sample, 3 high-cardinality axes: GMDN term ~1099 classes,
   product code ~896, company ~1168) — lower purity numbers, Fisher is a no-op

These tell different stories about what the models are good at.

## Results: Energy Devices Subset (Feb 2026)

8K sample, label axes with 3-40 classes. "Uniform" = no Fisher.
"Max-Fisher" = element-wise max across all axes.

| Model | Dim | Uniform | Max-Fisher | Fisher Delta |
|---|---|---|---|---|
| MiniLM-L6-v2 | 384 | 0.842 | **0.858** | **+1.6%** |
| PubMedBERT | 768 | 0.853 | 0.803 | -5.0% (hurts) |
| Nomic v1.5 | 768 | **0.934** | 0.910 | -2.4% (hurts) |
| GTE-large | 1024 | 0.915 | **0.929** | **+1.4%** |

## Results: Full GUDID 50K (Mar 2026, measured)

5K sample from 50K full GUDID, same texts embedded with all three models via
Ollama. Label axes are high-cardinality (896-1168 classes). k-NN purity (k=15)
measured in embedding space via `diagnose_axes`.

### Uniform (no Fisher)

| Model | Dim | GMDN | Product Code | Company | **AVG** |
|---|---|---|---|---|---|
| MiniLM-L6-v2 | 384 | 0.5045 | 0.3482 | 0.2428 | **0.3651** |
| Nomic v1.5 | 768 | 0.4737 | 0.3399 | 0.3027 | **0.3721** |
| mxbai-embed-large | 1024 | 0.4512 | 0.3385 | 0.2807 | **0.3568** |

### Max-Fisher (element-wise max across all axes)

| Model | Dim | GMDN | Product Code | Company | **AVG** | **Delta** |
|---|---|---|---|---|---|---|
| MiniLM-L6-v2 | 384 | 0.5034 | 0.3478 | 0.2404 | **0.3638** | -0.0013 |
| Nomic v1.5 | 768 | 0.4746 | 0.3402 | 0.3018 | **0.3722** | +0.0001 |
| mxbai-embed-large | 1024 | 0.4533 | 0.3390 | 0.2787 | **0.3570** | +0.0002 |

### Per-axis lift (uniform, ratio above random baseline)

| Model | GMDN lift | Product Code lift | Company lift | Avg lift |
|---|---|---|---|---|
| MiniLM-L6-v2 | **56.0x** | **36.5x** | 49.5x | 47.3x |
| Nomic v1.5 | 52.6x | 35.6x | **61.7x** | **50.0x** |
| mxbai-embed-large | 50.1x | 35.4x | 57.2x | 47.6x |

## Analysis

### Why the full GUDID numbers look so different

The energy devices experiment used **low-cardinality label axes** (3-40 classes)
on a **homogeneous subdomain** (energy-based devices). That's an ideal setup for
k-NN purity: with few classes, neighbors are likely to share a label.

The full GUDID experiment uses **high-cardinality axes** (896-1168 classes) on a
**heterogeneous corpus** (all medical device types). With ~1000 classes in 5000
points, the random baseline is ~0.5-1% and even 35% purity represents massive
lift over chance.

The absolute purity numbers aren't comparable across experiments. What IS
comparable: **relative model ranking and Fisher deltas**.

### Model ranking: Nomic wins, mxbai disappoints

On the full GUDID corpus:

```
Nomic v1.5     0.3721 avg purity  (best)
MiniLM-L6-v2  0.3651 avg purity  (-0.7% vs Nomic)
mxbai-embed    0.3568 avg purity  (-1.5% vs Nomic, worst)
```

**mxbai does NOT beat Nomic on this dataset.** Despite higher MTEB clustering
scores (46.71 vs 42.56), mxbai underperforms on GUDID medical device text.

Possible explanations:
- **MTEB clustering benchmarks use different domains** — mxbai's training data
  may be stronger on general text than medical device descriptions
- **Nomic's `search_document:` prefix** was used during embedding in the parquet,
  which activates task-specific behavior; mxbai was embedded without prefix
  (as recommended for non-retrieval tasks)
- **Ollama quantization** — mxbai runs as Q4_0 in Ollama (669MB for a 335M
  param model); the original float16/32 weights might score higher
- **Domain mismatch** — GUDID descriptions are highly formulaic
  ("Hip prosthesis, cemented, constrained") which may not benefit from mxbai's
  deeper architecture

### Per-axis breakdown tells the story

- **GMDN terms**: MiniLM wins (0.505), Nomic close (0.474), mxbai trails (0.451)
  - Short, technical terms favor MiniLM's efficient encoding
- **Product codes**: Near-tie across all three (~0.34)
  - Product codes are arbitrary strings, no model encodes them well
- **Company**: Nomic dominates (0.303), mxbai second (0.281), MiniLM last (0.243)
  - Company names embedded in descriptions favor Nomic's longer context

### Fisher weighting: effectively zero impact

On the full 50K GUDID with high-cardinality axes, Fisher weighting does
effectively nothing for any model:

```
MiniLM:  -0.0013  (noise)
Nomic:   +0.0001  (noise)
mxbai:   +0.0002  (noise)
```

All three models have Fisher effective_dim = full dimensionality (382/384,
768/768, 1024/1024). The Fisher ratios are completely flat — no dimensions
are discriminative for 1000-class problems. Fisher needs low-cardinality
axes (3-40 classes) to produce meaningful weight variation.

## Reconciling with Energy Devices Results

| Finding | Energy Devices (8K) | Full GUDID (5K) |
|---|---|---|
| Best model | Nomic (0.934) | Nomic (0.372) |
| MiniLM vs Nomic | Nomic +9.2% | Nomic +0.7% |
| Fisher on MiniLM | +1.6% | -0.1% (noise) |
| Fisher on Nomic | -2.4% | +0.0% (noise) |
| Label cardinality | 3-40 classes | 896-1168 classes |

**Nomic is best in both experiments.** The Fisher effects from the energy devices
experiment are real but specific to low-cardinality label axes on a focused
subdomain.

## Why MTEB Clustering Scores Don't Predict GUDID Performance

MTEB's clustering benchmarks are low-cardinality, general-domain tasks:

| MTEB Dataset | Samples | Classes | Texts/class | Domain |
|---|---|---|---|---|
| Reddit Clustering | 2,048 | 10-50 | 100-1,000 | subreddit titles |
| ArXiv Clustering S2S | 2,048 | ~129 | ~16 | paper titles |
| TwentyNewsgroups | 1,000 | 20 | ~50 | newsgroup subjects |

Our GUDID benchmarks:

| Dataset | Samples | Classes | Texts/class | Domain |
|---|---|---|---|---|
| Energy devices (Feb) | 8,000 | 3-40 | 200-2,667 | medical device descriptions |
| Full GUDID (Mar) | 5,000 | 896-1,168 | 4-5 | medical device descriptions |

The energy devices experiment (~3-40 classes) is structurally similar to MTEB
clustering — and there, model choice matters enormously (Nomic +9.2% over
MiniLM). The full GUDID experiment (~1000 classes) is **20-60x higher
cardinality** than any MTEB clustering task. At that granularity, all three
models converge to similar performance because no model was trained to
distinguish 1,000 fine-grained medical device categories.

**MTEB clustering scores predict performance on coarse topic separation
(10-50 classes), not fine-grained domain-specific categorization (1000+
classes).** Our energy devices results track MTEB; our full GUDID results don't.

## Revised Hierarchy of Impact

```
Model choice:     MiniLM → Nomic           +0.7% to +9.2%  (dataset-dependent)
Model choice:     Nomic  → mxbai           -1.5%           (mxbai LOSES on GUDID)
Fisher weighting: low-cardinality axes     +1.6% max       (energy devices only)
Fisher weighting: high-cardinality axes    ~0%             (no effect)
Taxonomy choice:  any axis combination     <0.5%           (negligible)
```

## Conclusion

**Nomic v1.5 remains the best model for this pipeline.** mxbai's MTEB clustering
advantage does not transfer to GUDID medical device text. The +4.15 MTEB
clustering delta is on general-domain benchmarks (Reddit clustering, ArXiv
classification, etc.) — not on highly formulaic medical device descriptions.

**No model change needed.** Nomic at 768d uniform is the right choice. Fisher
weighting is only useful on focused subdomains with low-cardinality label axes.

## Sources

- Energy devices findings: measured on GUDID energy devices, 8K sample, Feb 2026
- Full GUDID findings: measured on GUDID 50K, 5K sample, Mar 2026
- Embedding via Ollama: `mxbai-embed-large` (Q4_0, 669MB), `all-minilm` (45MB),
  Nomic from pre-embedded parquet (768d, sentence-transformers)
- [mxbai-embed-large-v1](https://huggingface.co/mixedbread-ai/mxbai-embed-large-v1)
- [Nomic embed paper](https://arxiv.org/html/2402.01613v2)
- [MTEB Leaderboard](https://huggingface.co/spaces/mteb/leaderboard)
- [mteb/reddit-clustering](https://huggingface.co/datasets/mteb/reddit-clustering)
- [mteb/arxiv-clustering-s2s](https://huggingface.co/datasets/mteb/arxiv-clustering-s2s)
- [mteb/twentynewsgroups-clustering](https://huggingface.co/datasets/mteb/twentynewsgroups-clustering)
