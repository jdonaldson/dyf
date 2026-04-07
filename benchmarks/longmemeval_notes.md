# LongMemEval: What Dyf Tree Routing Actually Buys You

A research note on applying dyf's recursive PCA tree to the LongMemEval
benchmark as (1) a retriever and (2) a query-level confidence gate over
flat retrieval. Four embedding conditions, honest cross-validation,
no fine-tuning.

**Status**: exploratory. Conclusions are scoped to LongMemEval's
pooled-session setup (N=19,143 unique session documents, 500 questions).
Not a product recommendation; a research finding.

## TL;DR

1. **Dyf is not a retriever at this scale.** Flat matmul at N=19K × 384d
   is sub-ms on numpy (~7 MFLOPS/query). No tree approximation can beat
   a baseline that's already free on the hardware.
2. **Dyf's routing signals can act as a modest query diagnostic** —
   predicting when flat retrieval will miss — but only on some embedders.
   ΔAUC ranges from +0.076 (MiniLM-384d) to +0.027 (BGE-768d) to 0.000
   (mpnet-768d).
3. **The real UX primitive on this benchmark isn't dyf at all.** It's
   *swap the embedder to BGE*. BGE uniformly hits 0.324 recall pooled
   (vs. MiniLM 0.266, mpnet 0.244), and its baseline top1-cosine gate
   alone hits **0.920 recall on the top-5% most confident queries** —
   matching anything dyf achieves on MiniLM, from a stronger starting
   point.
4. **A cheaper ablation — averaging the top-5 flat cosines —
   captures 63% of dyf's AUC gain on MiniLM with zero new infrastructure.**
   It's a byproduct of top-k retrieval. Use it as a middle-column
   baseline in any gate experiment.
5. **Dim-reduction is not a free lunch.** PCA-distilling MiniLM to 128d
   drops uniform recall by 17% relative (0.266 → 0.220) while raising
   baseline AUC. The "noise dims" aren't noise — they carry the residual
   signal dyf routing was exploiting.

## The Question

LongMemEval ([Wu et al. 2024](https://arxiv.org/abs/2410.10813)) is a
500-question benchmark where each question has a set of "haystack
sessions" and a set of gold session IDs containing the answer.
Mempalace reported 0.966 recall_any@5 on it using flat MiniLM retrieval
over user-turn-only session documents — reproduced to 4 decimals in
`longmemeval_flat.py`.

**But that 0.966 is a per-question tiny-corpus result.** Each question
gets ~50-200 haystack sessions of its own. When you *pool* all 19,143
unique sessions into a single persistent memory (what "persistent
memory" should actually mean), MiniLM's flat recall_any@5 drops from
0.966 to **0.266**.

That 0.266 is the real question this note investigates:
- Can dyf's tree structure improve retrieval at N=19K? (Answer: no.)
- Can dyf's routing signals *predict* which queries will miss, so we
  can abstain on them? (Answer: yes, but modestly, and not on every
  embedder.)

## Setup

All experiments live in `benchmarks/longmemeval_*.py`:

- `longmemeval_flat.py` — flat baseline (reproduces Mempalace headline)
- `longmemeval_tree.py` — dyf tree vs. flat (head-to-head retrieval)
- `longmemeval_diagnostics.py` — feature extraction + logistic regression
  analysis (in-sample AUC + suppressor-variable decomposition)
- `longmemeval_abstention.py` — 5-fold CV gate + abstention budget sweeps
  + precision@coverage + optional `--pca` distillation

Four feature conditions on the same 500 questions and 19,143 pooled
session documents:

| Condition | Embedder | Dim | Source |
|---|---|---|---|
| `LOW` | MiniLM-L6-v2 | 384 | `sentence-transformers/all-MiniLM-L6-v2` |
| `LOW + PCA` | MiniLM → PCA | 128 | scikit-learn `PCA(n_components=128)` on pool |
| `MEDIUM` | mpnet-base-v2 | 768 | `sentence-transformers/all-mpnet-base-v2` |
| `MEDIUM_BGE` | bge-base-en-v1.5 | 768 | `BAAI/bge-base-en-v1.5` |

Dyf tree: `max_depth=4`, `num_bits=3`, `fit_method=raw_pca`,
`min_leaf_size=8`, built on L2-normalized pool embeddings.

Classifier: scikit-learn `LogisticRegression(max_iter=2000, C=1.0)`
with `StandardScaler`, 5-fold `StratifiedKFold(shuffle=True, random_state=42)`
for out-of-fold predicted probabilities. All AUC numbers reported
below are out-of-fold CV, not in-sample.

Three gate variants in every run:
- **baseline** — `top1_cos` alone (1 feature)
- **top5** — `top1_cos + top5_mean_cos` (2 features, both free from top-k retrieval)
- **full** — `top1_cos + max_centroid_cos + leaf_size_np1 + min_margin + candidates_scored` (5 features, 4 from dyf routing)

## Results: Four Conditions

### Uniform recall (no abstention)

| Embedder | Dim | recall_any@5 |
|---|---:|---:|
| MiniLM-L6-v2 | 384 | 0.266 |
| MiniLM → PCA-128 | 128 | 0.220 |
| mpnet-base-v2 | 768 | 0.244 |
| **bge-base-en-v1.5** | **768** | **0.324** |

BGE is the strictly best encoder on this task. Per-type it dominates
the hard categories:

| Question type | MiniLM | mpnet | BGE | BGE rel. vs MiniLM |
|---|---:|---:|---:|---:|
| multi-session | 0.195 | 0.203 | 0.308 | +58% |
| knowledge-update | 0.295 | 0.256 | 0.397 | +35% |
| temporal-reasoning | 0.188 | 0.150 | 0.263 | +40% |

### CV gate AUC

| Embedder | baseline (top1) | top5 gate | full gate | Δ full vs base |
|---|---:|---:|---:|---:|
| MiniLM 384d | 0.638 | 0.686 | 0.714 | **+0.076** |
| MiniLM PCA-128 | 0.678 | 0.711 | 0.714 | +0.035 |
| mpnet 768d | 0.643 | 0.649 | 0.642 | −0.000 |
| BGE 768d | 0.641 | 0.644 | 0.667 | +0.027 |

### Recall on the top 5% most-confident queries

| Embedder | baseline | top5 gate | full gate |
|---|---:|---:|---:|
| MiniLM 384d | 0.680 | 0.840 | **0.920** |
| MiniLM PCA-128 | 0.800 | 0.880 | 0.800 |
| mpnet 768d | 0.640 | 0.760 | 0.760 |
| **BGE 768d** | **0.920** | 0.920 | 0.840 |

**Two ways to get 0.920 on the top 5% slice**: MiniLM + full dyf gate,
or BGE + baseline (top1_cos alone). The second is strictly better UX:
higher starting recall, zero extra infrastructure.

## Findings

### Finding 1 — Dyf is not a retriever at N=19K × 384d

`longmemeval_tree.py` pits `LazyIndex.search()` and `search_ivf()`
against flat numpy matmul on every axis (recall, latency, recall@k).
Flat wins on all of them. The regime is too small — ~7 MFLOPS/query,
sub-ms on laptop CPU — to leave headroom for approximation. A useful
rule going forward: **compute the baseline's operating cost before
designing an experiment to beat it.** If the baseline is already free
on the hardware, the experiment is asking the wrong question.

Dyf's value is as an **organizational substrate**, not an ANN index
at this scale. Where dyf earns its keep: 1M+ documents, or exploration
and visualization and routing on any corpus size.

### Finding 2 — Dyf's routing signals act as a query diagnostic, but only on some embedders

The reframe: if dyf can't retrieve at this scale, can it *predict*
when flat retrieval will miss? Yes, but modestly and not universally.

The full feature vector is:
- `top1_cos` — best flat cosine (baseline signal)
- `max_centroid_cos` — query's cosine to closest tree leaf centroid
- `leaf_size_np1` — candidates scored at nprobe=1
- `min_margin` — smallest routing decision margin at nprobe=3
- `candidates_scored` — total candidates at nprobe=3

Over MiniLM 384d, this beats `top1_cos` alone by +0.076 CV AUC
(0.638 → 0.714). Over BGE the gain shrinks to +0.027; over mpnet
it collapses to 0.000. The dyf contribution is not a universal
property — it depends on how much residual query-specificity signal
isn't already absorbed into `top1_cos` by the encoder itself.

### Finding 3 — The suppressor variable pattern

In the in-sample logistic regression, `max_centroid_cos` has a
**positive** solo coefficient (+0.14) but **flips negative** (−0.85)
when combined with `top1_cos`. Similarly, `candidates_scored` takes
a strongly negative combined coefficient.

The interpretation: dyf encodes a latent *specificity* dimension.
A query that's close to a leaf centroid but *not* to any specific
document inside that leaf is in a "generic neighborhood" — close
to a topic cluster but not a specific answer. Top-1 cosine says
"the closest doc looks good"; `max_centroid_cos` + `candidates_scored`
say "but you're in a dense and generic region, don't trust the top-1."

This is the mechanism by which dyf's routing adds a signal
orthogonal to flat cosine — when one exists.

### Finding 4 — Non-monotonic help flips location across encoders

On MiniLM, dyf helps the *tails* of the confidence distribution:
the top 5% slice (0.680 → 0.920) and the aggressive >33% abstention
regime. In the 20-25% coverage bulk, dyf actually *hurts* slightly
(baseline 0.460 → full 0.540 is the top5 gate; full dyf is 0.540
but top5 outperforms it there).

On BGE the pattern **inverts**: dyf helps the bulk (20-33% coverage)
and hurts the tails. At the top 5% slice BGE full drops to 0.840
while BGE baseline alone is already 0.920.

**The location of a feature's help is not a property of the feature —
it's an interaction with where the baseline is already calibrated.**
Never generalize "helps at coverage Y" across base models without
retesting.

### Finding 5 — A cheap ablation captures most of the complex gain

Just averaging the top-5 flat cosines — free from any top-k retrieval
pipeline — captures **63% of dyf's AUC gain on MiniLM** (+0.048 vs
dyf's +0.076). Two features beats five on most budgets in the
precision@coverage table.

Lesson for any future "does feature F help?" experiment: always
include the cheapest byproduct of existing computation as a middle
column. Three gates: baseline / cheap_alt / complex. The meaningful
delta is `complex − cheap_alt`, not `complex − baseline`.

### Finding 6 — PCA distillation is strictly worse

Hypothesis: "PCA removes noise dims, sharpens the dyf signal by
concentrating leaf structure." Tested via MiniLM 384d → PCA 128d
(78.4% variance retained).

Falsified:

```
                       MiniLM 384d    MiniLM PCA-128
uniform recall             0.266          0.220  (−17% rel)
variance retained          —              0.784
baseline CV AUC            0.638          0.678  (+0.040)
full gate CV AUC           0.714          0.714  (unchanged)
Δ full vs baseline         +0.076         +0.035  (−54%)
top 5% recall (full)       0.920          0.800  (−0.120)
top 5% recall (baseline)   0.680          0.800  (+0.120)
```

What actually happens: PCA concentrates retrieval signal into
`top1_cos` itself. Baseline AUC rises (top1 becomes more discriminative)
while dyf's marginal contribution shrinks. The "noise dims" in
MiniLM 384d aren't noise — they carry the residual information dyf
routing was exploiting. Compression eats dyf's lunch.

Meta-pattern worth reusing: **if your ranking AUC improves but the
underlying uniform recall drops, you're measuring ranking on a worse
base**. That's the hall of mirrors.

Caveat: the controlled comparison I did *not* run is random-projection
to 128d. The confound "PCA-specific structure vs. dim reduction in
general" is unresolved. If this pattern matters downstream, that's
the next experiment.

## What to Use, What to Skip

### Use
- **BGE (`bge-base-en-v1.5`) as the default encoder for pooled-session
  LongMemEval.** It's +5.8pp uniform recall over MiniLM and hits 0.920
  at top-5% on `top1_cos` alone.
- **`top1_cos + top5_mean_cos` as a cheap two-feature gate** on any
  embedder. It captures most of what a complex gate would.
- **Dyf routing diagnostics** specifically in regimes where the encoder
  has residual specificity signal not absorbed into `top1_cos` — MiniLM
  appears to be one such case; BGE less so; mpnet not at all.

### Skip
- **Dyf as an ANN retriever at N ≤ 100K.** Flat matmul on normalized
  embeddings is already free. Reframe the question before running the
  experiment.
- **PCA distillation for this task.** Strictly worse. If you need a
  smaller encoder, pick a smaller encoder — don't compress a larger one.
- **Generalizing "feature F helps at coverage Y"** across encoders
  without retesting. The location of the gain flips.
- **Committing scoping claims ("X-specific") after N=2 observations.**
  I did this twice in this arc; each time a 15-minute follow-up
  experiment falsified the scope.

## Files

All inputs, scripts, and outputs are in the repo. Everything is
reproducible from the four JSON result files.

**Scripts** (`benchmarks/`):
- `longmemeval_flat.py` — flat baseline (reproduces Mempalace headline)
- `longmemeval_tree.py` — dyf tree vs. flat retrieval head-to-head
- `longmemeval_diagnostics.py` — feature extraction + in-sample logit analysis
- `longmemeval_abstention.py` — 5-fold CV gate, abstention budgets, coverage, `--pca` flag

**Results** (`benchmarks/results/`):
- `longmemeval_flat_low_useronly.json` — user-turn-only flat baseline
- `longmemeval_flat_low_both.json` — both-role flat baseline
- `longmemeval_tree_depth{2,4}.json` — dyf tree head-to-head
- `longmemeval_tree_itq.json` — ITQ fit-method ablation
- `longmemeval_diagnostics_depth4.json` — in-sample logit
- `longmemeval_abstention.json` — MiniLM 384d gate
- `longmemeval_abstention_minilm_pca128.json` — MiniLM PCA-128 gate
- `longmemeval_abstention_medium.json` — mpnet 768d gate
- `longmemeval_abstention_bge.json` — BGE 768d gate

**Reproducing the four gate conditions**:

```bash
# MiniLM 384d (raw)
python benchmarks/longmemeval_abstention.py --model LOW \
    --out benchmarks/results/longmemeval_abstention.json

# MiniLM → PCA-128
python benchmarks/longmemeval_abstention.py --model LOW --pca 128 \
    --out benchmarks/results/longmemeval_abstention_minilm_pca128.json

# mpnet 768d
python benchmarks/longmemeval_abstention.py --model MEDIUM \
    --out benchmarks/results/longmemeval_abstention_medium.json

# BGE 768d
python benchmarks/longmemeval_abstention.py --model MEDIUM_BGE \
    --out benchmarks/results/longmemeval_abstention_bge.json
```

Each run takes ~2-3 minutes on an M-series Mac (bulk of the time is
embedding). Dataset is the LongMemEval `s_cleaned` split at
`/tmp/longmemeval-data/longmemeval_s_cleaned.json`.

## Positioning Note

After four conditions, the scoped claim on dyf's role in this
benchmark is:

> *Dyf routing adds measurable but model-specific gating value over
> flat retrieval. On MiniLM-L6-v2 it buys +0.076 CV AUC for hit/miss
> prediction. On bge-base-en-v1.5, +0.027. On mpnet-base-v2, zero.
> On PCA-distilled MiniLM, half. The practical default for
> pooled-session LongMemEval is BGE + top1_cos (already 0.920 at
> top-5%), with dyf reserved for mid-coverage windows on encoders
> where it helps.*

Dyf's value on LongMemEval is not as a retriever and not as a general
confidence layer — it's as a **specificity detector**, and that role
only shows up clearly on encoders that leave residual specificity
signal on the table.
