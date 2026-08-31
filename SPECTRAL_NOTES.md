# Spectral characterisation of dyf cells — CLOSED

⛔ **This direction is closed. Five hypotheses, five falsified.** Do not open a sixth
without choosing an outcome variable first.

| hypothesis | verdict |
|---|---|
| per-cell volume locates thin coverage | falsified — direction inverted; damage rises with occupancy |
| derive `num_bits` from spectral significance | falsified — ≤1.0× on 4 of 5 corpora |
| boilerplate pockets are shattered concepts → merge them | falsified — cross-pair duplicate rate 0.04–0.20 |
| split quality decays with depth → stopping criterion | falsified — it does not decay |
| `eff_rank` as an adaptive-probe signal | falsified — worse than uniform at every budget |

**The one durable methodological lesson**: pick the outcome variable FIRST. Everything found
by describing geometry and then hunting for a use died. The single result that got anywhere
(`eff_rank` predicts retrieval difficulty at +0.405 partialling out size, where occupancy
scores −0.002) came from correlating a descriptor directly against an outcome — and even
that failed as a lever, because `eff_rank` runs +0.67 with cell size, so the cells it flags
are precisely the expensive ones to probe. Predicted difficulty and the cost of acting on it
were the same quantity.

**What survives, and is worth keeping:**

- **29.4% of the SEC corpus is near-duplicate content.** The one actionable number, now
  shipped as `dyf.dedup` — see `src/dyf/dedup.py`, README and CHANGELOG for the *feature*;
  the sections below are the research record behind it.
- `Node.eigenvalues` is **scatter-like, not covariance-like** (∝ n^0.86), so raw log-ev is
  occupancy in disguise (ρ +0.96–0.99 with log n). The `.fbs` comment says "per-component
  PCA eigenvalues", which misleads. Divide by n before any geometric use.
- Text embedding spectra are clean **power laws** (SEC R² = 0.997 for λ ~ i^−0.89); CMU MoCap
  is exponential instead. But there is **no spatial self-similarity** — Grassberger–Procaccia
  scaling windows are 0.01–0.11 decades, and **concentration of measure**, not normalisation,
  is the binding limit. Classical fractal analysis does not apply to high-dimensional
  embeddings.

Companion files: `POSTGRES_NOTES.md` (extension design, frozen-basis/REINDEX policy, and the
shared **Scope limits**), `SEQUENCE_NOTES.md` (the earlier sequence arc).

---

## Cell volume does not say where the basis is thin — and the eigenvalues are a trap

The coverage result in `POSTGRES_NOTES.md` raises an obvious follow-on: depth-1 JS says *when* to refit, so
is there a query-free signal for *where* the frozen basis is thin — which cells will serve
new content badly, i.e. where to go get samples? The candidate is nearly free. Every
internal node already stores its own PCA spectrum (`eigenvalues` in
`src/dyf/schema/dyf_index.fbs`, written from `clf.get_eigenvalues()` on that node's own
points at `src/dyf/dyf_tree.py:113`, read back by `LazyIndex.get_split_eigenvalues()`), and
ellipsoid log-volume is `0.5 * sum(log lambda_i)`. Large volume with low occupancy should
mean a thinly-sampled region.

**Verdict: falsified, and the direction is inverted.** Measured 2026-08-28,
`sec_cell_volume.py`, 5 seeds × the same 4 conditions, 2,000 stream-side queries each,
probe 32. Target is the quantity that matters — per-cell recall gap (fresh − frozen) for
stream queries routed into that cell.

**First, two properties of `get_eigenvalues()` that any future reader of that field needs.**
Measured directly against numpy on synthetic caps before the hypothesis was tested:

- **It is scatter-like, not covariance-like.** `sum(ev)` grows as ~n^0.86 (n=500 → 20.7,
  n=8000 → 221.5 at fixed spread). So raw log-ev **is occupancy in disguise**:
  rho(sum log ev_raw, log n) = **+0.958 to +0.987** across all four conditions. It is the
  only volume-flavoured predictor that clears the significance null *in the hypothesised
  direction* — and it does so purely as a monotone function of n. (Several others clear it
  in the anti-predictive direction; see below.) Divide by n before use —
  and note that the sub-linear exponent means even that does not fully de-confound it
  (the n-corrected version still reads rho = −0.39 to −0.85 against log n).
- **It saturates at the diffuse end.** Angular spread 0.02 → 0.8 moves `sum log ev` from
  2.55 → 17.21, but 0.4 → 0.8 moves it only 16.94 → 17.21 while mean-cos-to-centroid still
  halves (0.30 → 0.15). Unit-norm data goes isotropic and the top-k eigenvalues hit a
  ceiling.
- Only `num_bits` eigenvalues are stored (4 here, of 768). They reproduce the true top-4
  spectrum faithfully (rho 0.999–1.000), but rho(true top-4, true top-32) is only
  **+0.33 to +0.57**. The stored shadow is not a usable proxy for a cell's real spectrum.

**Second, the power floor.** Permutation null on |rho| is **0.582 at depth 1** (~12 usable
cells) and **0.220 at depth 2** (~80 of 234). *No* depth-1 correlation in this probe is
interpretable, including a +0.321 apparent win for volume in the temporal condition. Note
the asymmetry with the JS trigger: depth 1 is the *best* level for a distributional
statistic (it aggregates ~8,700 points per cell into one number) and the *worst* level for
a correlation across cells (n=16 observations). Same tree level, opposite power.

**Third, the ablation.** Against the two free baselines — occupancy alone (`-log n`) and
mean-cos-to-centroid, which dyf already computes as `centroid_similarities` →
`point_margin_map` — the value of volume, `rho(sparsity32) - max(rho(neg_log_n),
rho(diffuse))`, is **≤ 0 in every condition at depth 2**: −0.016 random, −0.053 temporal,
−0.040 ticker, −0.206 section. The least n-confounded predictor (`logdet32`) carries the
least signal, which is the tell that the signal is occupancy throughout.

**Fourth, damage runs the opposite way.** Section condition (the only one that collapses),
depth 2, cells binned by base occupancy:

| base n quartile | n range | mean recall gap | mean flood (n_stream/n_base) | mean logdet32 |
|---|---|---|---|---|
| Q1 (smallest) | 13–114 | +0.0882 | 24.07 | −95.45 |
| Q2 | 114–354 | +0.0931 | 3.54 | −89.83 |
| Q3 | 354–1074 | +0.1251 | 2.24 | −92.22 |
| Q4 (largest) | 1074–6810 | +0.1571 | 0.60 | −94.90 |

rho(log n, gap) = **+0.351**, rho(log n, flood) = **−0.800**, rho(logdet32, gap) = −0.190.
Thin cells get flooded 24× over and suffer *less*; crowded cells suffer most. This is the
same inversion as the dirty-fraction trap in `POSTGRES_NOTES.md` — out-of-distribution data piles into a
corner, and the corner is not where retrieval breaks.

**Why there was nothing to detect.** `logdet32` is flat across those quartiles (−95, −90,
−92, −95) while occupancy spans 500×. dyf's PCA splits **equalise cell extent and let
count vary**, so volume is close to constant by construction and carries almost no
information about this partition. The mechanism that does explain the table is ordinary
IVF: at a fixed 32-leaf probe budget, a query in a crowded region has its true top-10
spread across more leaves than the budget covers. That is a *density* failure, not a
coverage failure — consistent with 32→128 probes recovering most of the mild-condition gap.

Caveats: a cell needed ≥5 stream queries to enter the correlation, so this says "among
cells that receive traffic", though a zero-traffic cell cannot damage measured recall by
construction. Same single corpus and embedding model as everything above. Filed alongside
the empty-cell metric in `POSTGRES_NOTES.md` "Scope limits" as a second obvious-looking metric that
measures nothing.

## What a split is doing, and deriving `num_bits` instead of setting it

Measured 2026-08-28. `sec_split_anatomy.py`, `sec_cell_spectra.py`, `sec_derived_bits.py`.

**A node's hyperplanes ARE its top-`num_bits` PCs** — |cos| = 1.000 against PC1–PC4 from
numpy. So `num_bits` is not a taste parameter, it is the answer to "how many eigenvectors
do we trust", which is estimable.

**The cut is at the origin, and that is the dominant pathology.** Routing is
`x @ H.T > 0`, but the PCs are fitted on *centred* data — nothing makes the origin pass
through the cell. Per-bit `frac>0` on a 20k subset: **0.585 / 0.118 / 0.219 / 0.239**. The
mean projection sits ~0.8 sd off the cut on PC2–PC4, so those bits are 80/20 slabs. It
worsens with PC index because sigma shrinks faster than the offset does. Across all cells,
**25% (depth 1) and 30% (depth 2) of split axes are worse than 80/20**, and effective
buckets run **7.8–8.6 of 16**. Fit method matters: `fit_raw_pca` 7.82 effective buckets
(max share 0.340), `fit` **10.26** (0.195), `fit_itq` 8.01 (0.293).

**The splits do separate real modes** — 2-component GMM, Ashman's D, against nulls that
hold the cell fixed and vary only the direction (random ambient direction; Gaussian with
the cell's covariance on its own PC1). PC1 is genuinely bimodal in **12/14 depth-1 and
69/79 depth-2 cells**, mean 3.00 and 2.22 of 4 axes.

⚠ **Null-ladder trap, recorded so nobody re-derives it.** Three nulls, three answers:
1-D standard normal (D95 ~1.7, too weak — omits that PC1 is *chosen* as max-variance);
Gaussian with the cell's covariance on its own PC1 (D95 1.6–1.8, the correct
selection-effect null — PCA does *not* manufacture modes); and a matched-size **random
corpus subset** (D95 3.5–3.9, **no cell beats it**). The third is the wrong question — it
asks "is this cell as multimodal as the whole corpus", and the answer is no *by design*
because depth 1 already separated the section types. Varying the cell contents smuggled in
a second difference.

**Spectral skew does NOT tell you what a split is doing.** rho(skew, n_bimodal) = −0.064
(depth 1) / +0.060 (depth 2); rho(top1_share, n_bimodal) = −0.143 / +0.247. Sign-unstable,
inside noise. The intuition that a dominant PC means the split is shaving derivative
variation is not supported — the waste is in the offset, not the spectrum.

### Deriving `num_bits`

`b = clip(min(significant_PCs, capacity_cap), 1, 6)` where significance is **Horn's
parallel analysis** (shuffle each column independently — kills cross-column correlation,
preserves every marginal — keep components above the permuted 95th percentile) and
`capacity_cap = floor(log2(n / (2*min_leaf)))`.

PA validated before use: returns 6 on corpus-scale subsets (var shares
0.126/0.067/0.051/0.038 vs isotropic 0.0013; observed/shuffled ratios 5.4/4.0/3.5/3.1) and
exactly **2 on a synthetic 3-cluster control**.

**Capacity binds, not signal.** Signal says ~6 nearly everywhere; what limits a node is how
many points it has to divide. Bits allocate top-heavy — `bits_hist` over 5,068 internal
nodes = `[0, 3952, 620, 260, 129, 56, 51]`, i.e. thousands of small deep nodes at 1 bit and
the large nodes near the root at 5–6. **`mean_bits` (1.40) is a misleading summary** — it
averages over the numerous deep nodes and reads like a near-binary tree when the root is
branching 64 ways.

Recall@10 at matched scan cost, **leaf counts matched to within 5%** (13,657 / 12,990 /
13,408 / 13,445, tuned via `min_leaf` = 15 / 45 / 10 / 12):

| vs `fixed4_origin` | 280 | 487 | 846 | 1470 | 2556 | 4443 candidates |
|---|---|---|---|---|---|---|
| fixed4 + median cut | +0.0865 | +0.0662 | +0.0492 | +0.0310 | +0.0163 | +0.0079 |
| **derived bits, origin cut** | **+0.1256** | +0.0953 | +0.0718 | +0.0490 | +0.0285 | +0.0128 |
| derived bits + median cut | +0.1436 | +0.1102 | +0.0797 | +0.0512 | +0.0283 | +0.0140 |

**Deriving the bits beats fixing the offset**, and the two are largely redundant (median
adds +0.018 on top of derived, but +0.087 on top of fixed). The mechanism explains why:
bit 0 is nearly centred (`frac>0` = 0.585) and the lopsidedness lives in bits 2–4, so
derived bits mostly *avoids* the bad bits rather than repairing them. Gains concentrate at
tight budgets and wash out by ~4.4k candidates.

⚠ **Measurement trap that nearly produced a wrong answer.** The first run showed the median
cut at **+0.25** recall. Artifact: `sec_seqlib.scan_cost()` counts only candidates
examined, while `ivf_search` compares each query against *every* leaf centroid. The median
arm built **43,414 leaves (5.3 points/leaf)** vs 13,327, so it reached equal recall with
8.8× fewer candidates while paying 3.3× more centroid comparisons — its *floor* cost
(43,414 dots/query) exceeded the baseline's entire budget at 0.9847 recall, and under
total-work accounting the two curves had no overlapping range at all. Any comparison
between trees of different granularity must match leaf count first.

Caveats: `min_leaf` is the knob used to match leaf counts, so it differs across arms by
construction; derived-bits builds cost ~60s vs ~4s (parallel analysis per node),
unoptimised. Deriving `num_bits` trades that parameter for continued dependence on
`min_leaf`, which now sets both the leaf floor and the capacity cap.

### Confirmed under dyf's real router (`sec_hier_routing.py`)

Everything above ran through `sec_seqlib.ivf_search` — flat IVF, scanning every leaf
centroid. **dyf does not do that.** `LazyIndex._find_candidate_leaves`
(`lazy_index.py:2136`) descends from the root, hashes the query against each node's
hyperplanes, and orders alternative buckets by margin distance —
`cost of flipping bit i = |projection[i]|` (`lazy_index.py:1590`). It never touches a leaf
centroid. So the two routers differ in **signal**, not just cost accounting, and the
flat-IVF numbers are a property of the harness.

Re-run on the same four trees, work = routing dots + members scanned, 800 queries:

| vs `fixed4_origin` | 133 | 271 | 554 | 1129 | 2303 | 4696 dots |
|---|---|---|---|---|---|---|
| fixed4 + median cut | +0.0946 | +0.0855 | +0.0617 | +0.0518 | +0.0301 | +0.0196 |
| derived bits, origin cut | +0.1261 | +0.1107 | +0.0912 | +0.0736 | +0.0361 | +0.0160 |
| derived bits + median cut | +0.1559 | +0.1280 | +0.0967 | +0.0759 | +0.0432 | +0.0248 |

- **The offset matters more under the real router, as predicted.** The median cut's benefit
  roughly doubles at mid-to-high budgets (+0.0492 → +0.0617, +0.0310 → +0.0518,
  +0.0163 → +0.0301, +0.0079 → +0.0196). Mechanism: the margin is measured from the origin,
  so on an 80/20 bit `|projection|` overstates the cost of flipping that bit and the
  alternative ordering is miscalibrated. Under flat IVF the offset only decided which
  points shared a leaf; here it corrupts routing decisions.
- **The ranking holds.** Derived bits still beats the median cut at every budget except the
  loosest (at 4,696 dots, +0.0160 vs +0.0196, where all arms have converged).
- **They are now complementary, not redundant.** Median adds ~+0.030 on top of derived here
  versus +0.018 under flat IVF, and the gap persists across the whole curve. Best arm is
  both together at every budget.
- **Imbalance costs work via size-biased leaf landing.** A query lands in leaf *i* with
  probability proportional to its size, so a skewed leaf-size distribution means the
  typical query lands in an oversized leaf. At matched leaf count (~17 points/leaf mean),
  the first probe scans **133 candidates for `fixed4_origin` vs 32 for `derived_median`** —
  4× the work from imbalance alone.

Cross-router comparison is not apples-to-apples (the flat table's x-axis omits the ~13k
centroid dots per query), so treat only the within-router columns as measurements.

### ⛔ It does not replicate. Do not build this. (`sec_multicorpus_bits.py`)

Everything above is ONE corpus. Re-run across five corpora spanning 62–768 dims and three
modalities, hierarchical router, work = routing dots + members scanned, `min_leaf=16` for
every arm. Metric is the operationally meaningful one — **work reduction at equal recall**,
since recall deltas are largest at tight budgets corresponding to recall 0.4–0.6 that
nobody ships.

| corpus | recall | fixed4+median | derived+origin | derived+median |
|---|---|---|---|---|
| cmu_mocap 62d | 0.80 | **4.55×** | (starts above) | 1.61× |
| cmu_mocap 62d | 0.90 | **4.07×** | **0.58×** | 1.44× |
| wikipedia 384d | 0.80 | never reaches | 0.92× | 0.98× |
| news 384d | 0.80 | 0.83× | 0.84× | 0.88× |
| tweets 384d | 0.80 | never reaches | 0.85× | 0.92× |
| arxiv 768d | 0.80 | 0.90× | 0.94× | 0.87× |

**Every text corpus is ≤1.0× for every intervention.** The SEC result does not generalise.
The single win is the median cut on motion capture, and there *derived bits actively hurts*
(0.58×). Neither change is a floor-raiser.

**Why derived bits fails, and it is the useful lesson: parallel analysis answers the wrong
question.** PA asks "is this principal component statistically real?"; an index needs "is
this split useful?" A split does not need significance to help — it only needs to divide
the node somewhat evenly. Deep in the tree n is small relative to d, the shuffled null is
inflated, few components pass, and the tree goes coarse: at identical `min_leaf` the
derived arms build **~4,600 leaves vs ~14,600** on every text corpus, so every probe drags
in a leaf 3× too big. Motion capture confirms it from the other side — at d=62 PA is
well-powered and *correctly* reports that motion is low-dimensional (~2–3 components), and
that correct answer still produces a worse index. **Being right about intrinsic
dimensionality is not the same as building a good tree.**

⚠ **The SEC win was granularity compensation, not bit allocation.** `sec_derived_bits.py`
binary-searched `min_leaf` per arm to equalise leaf counts, handing the derived arm a finer
floor (`min_leaf` 10 vs 15). At a shared `min_leaf` the derived arm is 3× coarser and
loses. So deriving `num_bits` requires retuning `min_leaf` to compensate — **it does not
reduce the parameter count, which was the entire motivation.**

**The median cut is corpus-specific and not predictable.** Anisotropy does not explain it:
`arxiv_768` has the *highest* mean-vector norm (0.788 vs mocap's 0.687) and a comparable
per-PC offset ratio (0.46 vs 0.43), yet gets 0.90× where mocap gets 4.07×. A diagnostic
that would have told you when to apply the fix was measured and **does not work**, so there
is no rule for switching it on.

Measurement limits: probes cap at 256, so "never reaches 0.80" means "not within 256
probes", not "impossible"; only recall 0.80 is comparable across all five corpora (the
text baselines top out at 0.80–0.88 within budget); 100k subsamples, 500 queries;
`cmu_mocap` is raw joint angles unit-normalised to define a cosine task, which is not the
natural metric for motion.

## Is the recovered structure fractal? (`sec_fractal.py`)

Asked as a *scientific* question rather than an engineering one — the earlier "does it
produce a lever" framing is a different criterion and closing the direction on it did not
answer this. Three real measurements, because `alpha` (an OLS slope of log-eigenvalue on
log-rank) was never evidence of a power law: fitting a slope to anything yields a slope.

| corpus | D2 | scaling window | D_q spread | R² power law | R² exponential | verdict |
|---|---|---|---|---|---|---|
| sec 768d | 13.30 | 0.03 dec | 2.05 | **0.997** | 0.854 | power-law spectrum |
| cmu_mocap 62d | 3.98 | 0.11 dec | 0.78 | 0.406 | **0.657** | exponential spectrum |
| wikipedia 384d | 27.21 | 0.01 dec | 3.23 | 0.963 | 0.939 | power law (marginal) |

(The `D_q spread` column is retained only because the retraction below refers to it — it is
an estimator artifact, not a multifractality measurement. `D2` and `window` come from a
detector shown below to be unreliable; treat both as indicative at best.)

- **No spatial self-similarity anywhere.** Scaling windows are 0.01–0.11 decades, i.e. none.
  The local slope of log C(r) drifts smoothly rather than plateauing (SEC: 13.4 → 12.7 →
  11.1 → 9.2 → 7.4 → 5.1 → 2.9). The correlation dimension is **scale-dependent**, so there
  is no D2 to quote honestly.
- **But the text spectra ARE power laws.** SEC R² = 0.997 for `lambda_i ~ i^-0.89` against
  0.854 exponential — a strong, clean scale-free spectrum. Wikipedia 0.963 vs 0.939 is
  marginal. MoCap is the reverse: **exponential** (0.657 vs 0.406), i.e. a characteristic
  scale. ⛔ This **falsifies the prediction written into the probe** ("SEC should show a
  scaling window and MoCap should not") — exactly backwards. Motion has the only clean
  spatial plateau (slopes 3.3–4.1 from r=0.36 to 1.02) and the *worst* power-law fit.
- ⛔ **RETRACTED: the multifractality claim does not survive a null.** D_q does decrease
  monotonically in q (SEC 8.98 → 6.93, wikipedia 21.26 → 18.03, mocap 3.32 → 2.55) — but
  **that is required by theory**, since the Rényi spectrum is non-increasing in q for *any*
  measure, so the direction carries no information. Calibrated against a Gaussian with each
  corpus's **own covariance** (preserves the power-law spectrum, homogeneous density), the
  real spread is *smaller* than the null: **0.69× (SEC), 0.70× (mocap), 0.51× (wikipedia)**.
  Homogeneous data produces MORE of the signature. The spread is an estimator artifact, and
  the tell was visible before the null — spread/D2 was near-constant (0.196 / 0.154 / 0.119),
  which is what a multiplicative estimator effect looks like. The original "multifractal"
  verdict used a hardcoded spread > 0.5 threshold with no null; do not reuse it.
- ⚠ **The scaling-window detector is biased toward saturation plateaus** and cannot be
  trusted in general: it returns D2 ≈ 0.28–0.94 on the Gaussian nulls, where the true value
  is far higher, because the flattest part of any C(r) curve is where it saturates and the
  scoring function rewards flatness. It happened to pick the right region on real data (SEC
  13.35 matches the eyeballed small-r slopes of 12.8–13.4), but that is luck, not validation.
  D_q is unaffected — it is averaged over the valid band rather than through the detector.
- **What survives is weaker and comes from elsewhere: local dimensionality IS heterogeneous
  across the map**, but that rests on the per-cell measurements (eff_rank 9.1–85.3 across
  depth-2 cells, 46.6% of variance explained by the depth-1 parent, rho(eff_rank, dup_frac)
  = −0.723), each n-matched with its own null. That is "heterogeneous local dimension", not
  "multifractal" in the scaling sense — and it cannot be upgraded to the latter, because
  without a scaling window there are no well-defined exponents to form a spectrum from.
- D2 within SEC cells: 9.67 ± 5.64 (depth 1), 11.89 ± 1.91 (depth 2) vs 13.30 corpus-wide —
  but with 0.03–0.05 decade windows these are not trustworthy numbers.

**⛔ Non-normalised embeddings do not rescue it, and the reason is fundamental.** Unit
vectors bound distances to [0,2], under one decade even in principle, so raw vectors were
the obvious fix. Source check: the MiniLM text sets and SEC filings are **already unit-norm
upstream** (norm sd = 0.000, nothing to recover); Nomic is unnormalised at ratio 1.17–1.27;
CMU MoCap is genuinely raw at ratio 3.90 (4 of 140,837 rows are all-zero).

| corpus | raw dist range | normed | raw window | normed window |
|---|---|---|---|---|
| cmu_mocap 62d | 0.90 dec | 0.75 dec | 0.09 | 0.08 |
| wikipedia_nomic | 0.16 dec | 0.14 dec | 0.02 | 0.01 |
| arxiv_nomic | 0.18 dec | 0.17 dec | 0.02 | 0.02 |
| news_nomic | 0.15 dec | 0.14 dec | 0.02 | 0.01 |

Raw buys 0.01–0.15 decades — nothing. The binding limit is **concentration of measure**, not
normalisation: distance range falls as intrinsic dimension rises (mocap D2 ≈ 3.9 → 0.90
decades; Nomic D2 ≈ 23–29 → 0.15–0.18). **Classical fractal analysis is not applicable to
high-dimensional embeddings** — the dynamic range scaling analysis needs is destroyed by
dimensionality itself. The spectral power-law result stands, since it does not depend on
distance range.

## ⛔ The eff_rank lever fails. Spectral direction closed. (`sec_adaptive_probe.py`)

`sec_cell_spectra.py`'s one surviving positive was that per-cell effective rank predicts
retrieval difficulty where occupancy fails (+0.314 raw, **+0.405** partialling log n, vs
log_n at −0.002, null95 0.186). A correlation with difficulty is not a lever. This spends a
**fixed total work budget non-uniformly across queries** — more probes where the routed cell
has high eff_rank — and asks whether recall improves at matched total work. SEC is where the
correlation is strongest, so this was the cheapest kill for the whole direction.

Rank-based allocation (percentile *u* → `base * (1 + 0.6*(2u−1))` probes), hierarchical
router, 1,500 queries, compared at interpolated equal total work:

| vs uniform | 443 | 745 | 1253 | 2107 | 3543 | 5958 |
|---|---|---|---|---|---|---|
| **eff_rank** | **−0.0241** | −0.0212 | −0.0108 | −0.0098 | −0.0086 | −0.0033 |
| log_n (negative control) | −0.0300 | −0.0285 | −0.0197 | −0.0190 | −0.0164 | −0.0088 |
| margin (incumbent signal) | +0.0015 | +0.0010 | +0.0024 | −0.0000 | −0.0029 | +0.0001 |
| combo (eff_rank + margin) | −0.0110 | −0.0100 | −0.0024 | −0.0043 | −0.0060 | −0.0013 |

**eff_rank is worse than uniform at every budget, and worse than margin** (−0.0256 …
−0.0034). The negative control behaves correctly — `log_n`, which scored ρ = −0.002 against
difficulty, is the most harmful arm — so the harness is sane and the ordering is meaningful.

**Why it fails, and it is not a tuning problem.** eff_rank correlates **+0.67 with cell
size**, so allocating extra probes to high-eff_rank cells means allocating them where leaves
are *large*, and each extra probe there costs disproportionately more work. At base=4 the
eff_rank arm spends **432 work for 0.603 recall against uniform's 399 for 0.614** — more
work, less recall. The signal identifies genuinely hard cells, but they are hard *because*
their neighbourhoods are spread out, which is exactly what makes them expensive to fix by
probing. Predicted difficulty and the cost of addressing it are confounded. Since every arm
converges to uniform as alpha → 0, the best this allocation can do is break even.

**Decision: the spectral-characterisation direction is closed.** Four hypotheses came out of
it (cell volume, derived `num_bits`, pocket merges, depth stopping criterion) and all were
falsified; the fifth — eff_rank as an adaptive-probe signal — is falsified here. Do not open
a sixth without an outcome variable chosen first.

⚠ Incidental, and **not** a claim about shipped behaviour: the margin arm shows no
measurable benefit at matched work on this corpus (+0.0024 … −0.0029). But this tests
*margin as a rank-allocation signal*, not dyf's actual `_resolve_nprobe` threshold logic, so
it is not evidence that `AdaptiveProbeConfig` fails to do its job. Testing that would need
the real code path.

## Spectral shape by depth and along paths (`sec_depth_spectra.py`)

All descriptors n-matched to 300 (effective rank tracks sample size, so a raw depth
profile is mostly a size profile). Decisive control is the **parent-subsample null**: each
child's spectrum against a random same-size draw from *its own parent*, which holds n
exactly and asks whether the SPLIT changed the shape rather than whether a smaller sample
did. SEC 768d and CMU MoCap 62d.

**(A) Shape is not uniform with depth, but the discontinuity is only at the root.** SEC
eff_rank 77.8 (root) → 65.7 (d1) → 68.6 → 69.5 → 69.8; `top1` 0.127 → 0.149 → 0.106 →
0.101 → 0.094. Below depth 1 the partition is close to **self-similar** — each level sees
structure of the same shape. MoCap: eff_rank 14.2 → 9.9 → 8.7 → 8.9 → 9.8.

**(B) ⛔ Deeper splits do NOT stop working — prediction falsified.** The expectation from
the derived-bits failure was that split quality would decay with depth and hand us a
stopping criterion. It does not. Child-minus-parent-subsample `d_eff_rank`:

| depth | SEC | MoCap |
|---|---|---|
| 1 | −11.40 ± 6.16 (ns, 15 pairs) | −4.01 ± 1.18 |
| 2 | −6.39 ± 1.66 | −3.02 ± 0.45 |
| 3 | **−15.45 ± 0.79** | −2.59 ± 0.42 |
| 4 | −13.85 ± 1.39 | −2.81 ± 0.45 |

Splits concentrate structure at *every* depth, most strongly at depth 3–4 on SEC. **This is
the strongest independent evidence for why derived `num_bits` failed**: at exactly the
depths where parallel analysis reports few significant components (n ≈ 300–500 in 768d, so
the shuffled null is inflated), the splits are demonstrably still concentrating structure
at ~19 sigma. "Not statistically significant" and "not useful for splitting" are different
properties, and PA measures the wrong one. No stopping criterion exists to be found here.

**(C) Shape is strongly heritable along a path.** Parent→child rho(eff_rank) on SEC:
**+0.618 (d1→2), +0.645 (d2→3), +0.681 (d3→4)** — coherent and *increasing* with depth.
MoCap is weaker mid-tree (+0.250, +0.181) then +0.702. So root-to-leaf paths are not
random walks through shape space; a node's spectrum substantially predicts its child's.
(Depth 0→1 reads `nan` because there is one root, hence no variance to correlate.)

**Corpus-level shape explains the MoCap divergence.** Global spectrum at matched n=300:

| corpus | eff_rank | top1 | alpha | median-cut speedup |
|---|---|---|---|---|
| cmu_mocap 62d | 13.6 | 0.217 | **2.57** | **4.07×** |
| sec 768d | 76.3 | 0.124 | 0.89 | (n/a) |
| arxiv 768d | 134.9 | 0.065 | 0.62 | 0.90× |
| news 384d | 127.2 | 0.047 | 0.51 | 0.83× |
| tweets 384d | 137.8 | 0.040 | 0.50 | (n/a) |
| wikipedia 384d | 130.4 | 0.036 | 0.46 | (n/a) |

MoCap is the only corpus with a **dominant principal axis** (alpha 2.57 vs 0.46–0.89;
eff_rank 13.6 vs 76–138) and the only one where centring the cut pays. Mechanically
coherent — with one axis carrying the variance, getting that axis's cut right is most of
the partition — and it succeeds where the anisotropy diagnostic (mean-vector norm) failed.

⚠ **But n=1 in the positive class again**, the same trap already listed in `POSTGRES_NOTES.md` Scope limits for
the section split. One steep-spectrum corpus, one payoff; "steep spectrum causes the
benefit" is not separable from "motion capture differs for some other reason". The cheap
decisive test is a **synthetic sweep with tunable alpha** (0.5 → 3.0), not another
real corpus.

Caveats: only nodes with n ≥ 300 enter, so depth 4 keeps 47 of many on SEC — survivorship
toward large nodes; two corpora.

## When children are NOT self-similar to their parent (`sec_nonselfsimilar.py`)

⚠ **Correction to (A) above.** "Close to self-similar" is a statement about the
*population* of shapes — the mean spectrum at each depth is stable. It is NOT a statement
about individual splits, which move shape enormously. Both are true, the way a gas can hold
a constant temperature while every molecule moves.

Each child scored against a null of R=8 independent same-size draws of **its own parent**
(child and null averaged over the same number of draws, or the child would be the less
noisy of the two and every z inflated). 347 pairs, SEC.

**Non-self-similarity is the rule, not the exception**: |z_eff_rank| > 2 in **91.9%** of
children, > 10 in **70.3%**, median −14.07; z_div (whole-spectrum L1) median +43.1. Note
the null sd is small — 8 draws of n=300 from one parent barely varies — so lean on the raw
descriptor differences rather than the z magnitudes. **84% of splits concentrate**
(child tighter than a parent draw), **16% diffuse**.

**The two tails are different mechanisms, and both are the user's original hypothesis
observed directly.** corr(z_eff_rank, z_top1) = **−0.764**:

| group | z_eff_rank | z_top1 | z_alpha | top1 | eff_rank | share |
|---|---|---|---|---|---|---|
| 40 most concentrating | −45.0 | **+24.9** | +22.6 | 0.134 | 59.2 | 0.064 |
| middle 40 | −14.1 | +4.0 | +8.6 | 0.102 | 68.9 | 0.158 |
| 40 most diffusing | +19.3 | **−19.3** | −20.1 | 0.087 | 77.0 | 0.197 |

- **Diffusing children lost their parent's dominant axis.** A parent holding two
  well-separated tight clusters has its spectrum dominated by the *between-cluster*
  direction — highly peaked, low eff_rank. Splitting removes that axis, so the child's
  remaining variance spreads over many small directions and the spectrum *flattens* even
  though the child is semantically **purer** (`sec_top` purity 1.00 vs a mixed parent).
  "More diffuse spectrum" is not "more heterogeneous content". This is exactly the
  "split prunes derivative variation along the dominant axis" mechanism, and it is the 16%
  tail rather than the general case.
- **Concentrating children isolate near-duplicate boilerplate.** The extreme cases pull a
  1%-share pocket of `forward_looking` text (dup_frac 0.52–0.57) out of a `risk_factors`
  parent (dup_frac 0.01) — i.e. dyf separating the standard forward-looking-statement
  disclaimer embedded inside risk-factor sections. Group means: dup_frac **0.450** for the
  40 most concentrating vs **0.295** for the 40 most diffusing.

**Non-self-similarity lives in the thin minority buckets — the origin-cut pathology decides
where it appears.** Mean |z_eff_rank| by the child's share of its parent:

| child share | 0.00–0.05 | 0.05–0.15 | 0.15–0.35 | 0.35–1.00 |
|---|---|---|---|---|
| mean \|z_eff_rank\| | **29.4** | 19.9 | 13.2 | **8.2** |
| mean z_div | 72.8 | 58.1 | 40.1 | 35.5 |

Monotone. This is what the 80/20 origin cut predicts: the majority child is "the parent
minus a thin tail" and stays parent-like, while the sliver is structurally distinct. Not a
sample-size artifact — every spectrum is computed at n=300 regardless of child size, and
children below 300 are excluded (which does mean small-share children only arise from large
parents).

### Is non-self-similarity just duplicates? Partly — ~24% of it. (`sec_dedup_ablation.py`)

Near-duplicates collapsed within each leaf at cos > 0.99, a fresh tree built on what
remains, and the identical child-vs-parent-subsample scoring re-run. Blocking by leaf misses
cross-leaf duplicate pairs, so the removal rate is a **lower bound** and the ablation
under-removes — survival is the conservative direction.

**29.1% of this corpus is near-duplicate content** (66,731 of 229,243 points).

| | children | median z | \|z\|>2 % | \|z\|>10 % | mean \|z\| | % concentrating |
|---|---|---|---|---|---|---|
| with duplicates | 347 | −14.07 | 91.9 | 70.3 | 18.7 | 84 |
| deduped | 238 | −8.35 | 90.3 | 52.1 | 13.7 | 74 |

**The phenomenon survives; its magnitude does not fully.** After removing 29% of the corpus
**90.3% of children are still non-self-similar** (vs 91.9%) — essentially unchanged — while
median z drops −14.07 → −8.35 and mean |z| 18.7 → 13.7. The concentrating share falls
84% → 74%, i.e. duplicates were specifically inflating the *concentrating* tail, exactly
where dup_frac 0.52–0.76 lived.

⚠ **Confound checked and cleared.** Dedup shrinks the corpus, so fewer nodes clear the
n ≥ 300 floor (238 vs 347 children) and survivors could skew toward larger shares — and |z|
falls with share. Within matched share bins, retention is **82% / 66% / 75% / 82%** with no
share pattern, and the share distribution barely moved (median 0.116 → 0.120). The drop is
dedup, not selection.

So: the extreme concentrating pockets *are* substantially degenerate duplicates, but
non-self-similarity as a property of dyf splits is real geometry that outlives them.

Incidental and useful: dedup removed **29% of points but only 11% of leaves**
(13,220 → 11,711), so duplicate mass sits packed *within* existing leaves rather than
spread across extra ones — points/leaf 17.3 → 13.9.

### Dedup on ingest: 29% storage, and a recall TRADE, not a free lunch (`sec_dedup_ingest.py`)

The operational test of the 29.1% duplicate finding. Outcome variables declared before the
probe was written. Retrieval task unchanged — truth is exact kNN over the **full** corpus —
with the index storing one representative per cluster plus a representative→members side
table, expanded at query time. Dedup is tree-free (multi-table random-projection LSH) so it
genuinely precedes indexing. Cross-check: LSH finds **32.0%** against the 29.1% within-leaf
estimate, two independent methods agreeing within 3pp.

| | points | leaves | vectors MB | build | cluster max |
|---|---|---|---|---|---|
| baseline | 229,243 | 13,327 | 704.2 | 3s | — |
| dedup_transitive | 155,843 (−32.0%) | 12,408 (−6.9%) | 479.3 (−31.9%) | 2s | **541** |
| dedup_star | 161,875 (−29.4%) | 11,995 (−10.0%) | 497.8 (−29.3%) | 2s | 143 |

⚠ **Transitive union-find over-merges.** Its 541-member "duplicate" cluster is a chain
A~B~C~…~Z where A and Z need not be similar, so members inherit a representative score that
is wrong for them. **Star clustering** — a point joins a representative only if within
threshold *of that representative*, no transitivity — bounds every member's error and drops
cluster max to 143. Use star; the 2.6pp less dedup is worth it.

Recall@10 vs full-corpus truth at matched total work, `_inherit` = members take the
representative's score (expansion is an array lookup, costs no dot products):

| vs baseline | 270 | 503 | 938 | 1749 | 3259 | 6072 |
|---|---|---|---|---|---|---|
| transitive_inherit | −0.004 | −0.004 | −0.006 | −0.025 | −0.039 | −0.051 |
| **star_inherit** | **+0.039** | **+0.026** | **+0.018** | −0.004 | −0.023 | −0.039 |
| star_rescore | −0.088 | −0.087 | −0.076 | −0.071 | −0.059 | −0.045 |

Work to reach a target recall: at **0.80**, star_inherit 1,398 vs baseline 1,455 (4% cheaper
*and* 29% smaller); at **0.90**, baseline 3,390 vs star_inherit 5,149; at **0.95** baseline
6,433 and star_inherit **cannot get there at all**.

⛔ **SUPERSEDED — the recall penalty below was mostly a METRIC ARTIFACT.** See
"Distinct-content recall" immediately after this section. `recall@10` against exact kNN over
the raw corpus asks "did you reproduce the brute-force list", and this corpus puts a mean of
**2.97 duplicate slots in every true top-10** (83% of queries have at least one). That metric
therefore *pays for returning redundant copies* — precisely what dedup exists to prevent.
Under a content-level metric dedup wins at nearly every budget. The numbers below are
correct as stated but answer the ANN-approximation question, not the retrieval-quality one.

**The verdict is operating-point dependent.** Below ~0.82 recall dedup is a strict win —
less storage, faster build, *better* recall, because probe budget stops being spent
re-scanning near-identical vectors. Above it dedup loses, and score-inheritance **caps out**
below 0.95 because members of one cluster cannot be ranked against each other. Re-scoring
members lifts the ceiling (reaches 0.95 at 12,238 work) but is worse everywhere at matched
work, since the expansion dots come straight off the probe budget.

The prediction going in — that dedup would improve recall by not wasting budget on duplicates
— was **half right**: correct at tight budgets, wrong at high recall, where the ranking
ceiling dominates. Obvious next tweak, untested: **cap cluster size** (collapse clusters up
to ~8 members, index larger ones in full) to keep most of the storage win while preserving
ranking inside the big boilerplate clusters that cause the ceiling.

### Distinct-content recall: dedup actually WINS (`sec_dedup_metric.py`)

The metric above was wrong for the question. `recall@10` against raw exact kNN measures
"reproduce the brute-force list", and when a corpus holds four near-identical copies of a
document the true top-10 spends four slots on the **same content**. Measured on this corpus:
**mean 2.97 duplicate slots per true top-10, 83% of queries affected.** Raw recall@10 pays
for returning redundant results, so it penalises dedup for doing its job.

Two metrics, same runs, same work axis. `distinct@10` takes ground truth as the top-10
*clusters* ranked by each cluster's best member, and predictions as the system's candidates
mapped to clusters, deduplicated in score order, first 10 distinct. **Applied symmetrically**
— the baseline is also credited only for distinct content, so it too is penalised for
spending result slots on duplicates.

| delta (dedup − baseline) | 227 | 439 | 850 | 1645 | 3184 | 6162 |
|---|---|---|---|---|---|---|
| raw@10 | +0.042 | +0.032 | +0.014 | −0.009 | −0.026 | −0.039 |
| **distinct@10** | **+0.053** | **+0.055** | **+0.042** | **+0.026** | **+0.010** | −0.006 |

**Dedup wins at every budget except the largest, where it is −0.006 (parity).** Combined with
29.4% less vector storage and half the build time, dedup on ingest is a clear win — the
earlier "trade above ~0.82 recall" was the metric, not the system.

**Why the switch fixes it, mechanically.** The failure diagnosed via dose-response was
within-cluster tie-breaking: members share an inherited score, so top-k selection picks
arbitrarily among them. Under `distinct@10` only ONE member of a cluster is ever credited, so
ties inside a cluster become irrelevant by construction. The correction is largest at the
high-budget end (−0.039 → −0.006), which is where the raw metric's duplicate reward was
biggest.

⚠ **Read the earlier dose-response table the other way round.** Baseline raw recall was
*higher* on duplicate-rich queries (0.9517 for 4–8 clusters vs 0.8531 for singletons). That
is not the baseline handling them well — those queries are easy to *score* on because the
metric pays for redundant copies. A user would rate that result worse, not better.

**Lesson, and it is the same one as the rest of this arc**: pick the outcome variable to match
the intent. Five spectral hypotheses died from measuring geometry and hunting for a use; this
one nearly died from measuring the wrong success criterion for a system whose entire purpose
is to *not* return something the metric rewarded.

### Are the pockets fragments of one concept? No. (`sec_shattered_pockets.py`)

A root-to-leaf path is an intersection of halfspaces, so the tree can hold a convex region
but must **shatter** a concept that is multi-modal in embedding space. If the boilerplate
pockets were fragments of one such concept carved out under different parents, they would
be merge candidates. Tested against the LCA-depth-conditional similarity distribution
(similarity decays with tree distance by design, so "these are similar" is not evidence —
only "similar *given* how far apart they sit" is). 347 cells, 60,031 pairs.

**Verdict: not shattered fragments, and nothing for the existing merger to catch.**

- **Cross-pair duplicate rate is 0.04–0.20** — the decisive test. Top candidates reach
  centroid cos 0.993–0.996 yet only 4–20% of A's points have a near-duplicate in B, while
  *within*-pocket dup_frac is 0.52–0.76. The duplicates live **inside** each pocket, not
  across them. High cosine, different duplicate sets.
- **No enrichment.** Pockets are 2.2% of pairs diverging at depth ≤1 and 2.3% of those also
  above the depth-conditional p95 — **1.04×**. Pockets are not special with respect to being
  shattered.
- **All 10 top candidates already exceed the Louvain link threshold**, so
  `agglomerate.louvain_cluster_leaves` would already reunite them. Zero missed.

**What they actually are: regionalised boilerplate.** Each depth-1 region carries its own
near-duplicate pocket — context-specific variants of the same legal template. The tree is
doing the right thing by isolating each region's variant; merging them would collapse a
real distinction. They are candidates for **dedup/compression, not merging** — which is the
same lever as the earlier finding that `other` consumes 18.7% of leaves for 8.1% of content.

⚠ **The similarity scale is compressed on this corpus, which breaks cosine thresholds.**
Mean centroid cos is **0.821 at LCA depth 0** (pairs diverging at the root, i.e. maximally
distant), p95 0.933; at depth 3 it is 0.982. Everything lives in a narrow cone, so centroid
cosine barely discriminates and any absolute threshold must be set relative to the corpus
baseline, not to intuition. Note `agglomerate._run_louvain_on_centroids` documents
`similarity_threshold` as "only used by the NetworkX fallback" — the default Rust path
ignores it — but on the fallback path the 0.5 default would filter nothing here.

