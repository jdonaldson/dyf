# DYF over sequences of embeddings — design notes

**Status: DESIGN + 6 EMPIRICAL RUNS. Nothing shipped.** 2026-07-31.

The question that started this: repos are sequential — can DYF detect sequential patterns?
The answer is no, not directly, and the interesting part is what you build once you accept that.
Design thread ran: order-blindness → delta-as-object → frozen basis with drift-triggered
reorientation → cheap move-hashing. Then six measurement runs against it (Haxe code, music chroma,
Haxe localized, chroma audit, order/grammar, SNLI stance, SwDA transfer) — **two of which forced
retractions of results reported earlier in the same session.**

Read **"Bottom line for the whole arc"** near the end first; it supersedes anything above it that
the runs contradicted. Probes live in `benchmarks/sequence_arc/` (19 scripts, all re-runnable).
Design sections below are unbuilt proposals; sections titled `RESULT:` are measured.

---

## THE CRUX: DYF is permutation-invariant

`build_dyf_tree(embeddings, ...)` (`src/dyf/dyf_tree.py:169`) takes the full `(n, d)` array,
fits global PCA, assigns LSH buckets, and recurses on index **sets**. `split_leaf`
(`dyf-core/dyf-core/src/tree.rs:322`) iterates `sorted_buckets` sorted by bucket id
(`tree.rs:389`), seed fixed at 42. `compute_centroid` (`tree.rs:573`) is a mean over an item
set.

**Permute the input rows and you get the same tree.** There is no place for order to enter.
Any claim that "DYF found a sequential pattern" is therefore false by construction — sequence
must be injected into the representation beforehand, or read out of the tree afterwards.

Three injection points, ranked:

1. **Position/lag features in the point** — weak. Time gets one PCA axis and stratifies the
   tree by era. High confound, low value.
2. **Embed the transition, not the state** — the point *is* a step. See below.
3. **Tree as alphabet, sequence layer on top** — leaf/community id per point turns the stream
   into a string; n-grams, transition matrices, motifs, change-points all live in that layer.

What DYF contributes over flat VQ is exactly two things, and the design should be held to them:

- **Multi-resolution motifs that nest.** Cutting at depth 2 vs depth 5 gives coarse and fine
  alphabets *that relate*. Re-running k-means at different k gives alphabets that don't.
- **`lca_depth()` as a categorical jump metric** (`src/dyf/categorical.py:111`). Cosine between
  consecutive items is one scalar; LCA depth answers whether a change stayed inside a
  subsystem-concept or crossed the root split. A 500-line diff inside one leaf is a different
  event from a 5-line diff crossing the top hyperplane.

---

## Deltas: storage loses, representation is the idea

### Storage — don't

For unit-normalized embeddings with consecutive cosine `s`, `‖d‖ = √(2(1−s))`. At `s = 0.8`
that is 0.63, so quantizing to equal absolute precision saves `log₂(1/0.63) ≈ 0.67` bits/dim —
about 8% off an int8 layout.

**That is the 6% zstd result again** (CLAUDE.md, "Compression dead ends"): float16 embeddings
are near-random, and deltas of near-random vectors are also near-random. Delta encoding
rescales entropy, it does not reduce it. The baseline here is product quantization at 8–32×;
this isn't in the race.

Two further costs specific to an index: reconstructing `e_k` is O(K) not O(1), and quantization
error accumulates along the chain. Keyframes (video I-frame/P-frame) cap both — but note the
tension: **delta encoding optimizes sequential scan; an index is a random-access structure.**

### Representation — yes

`d_i` is a **move**, not a state. The load-bearing property:

> **The delta cancels common-mode topic content.**

This is the answer to embedder stance-blindness. "I agree, X" and "no, X is false" sit adjacent
in state-space because both are *about X*; their deltas from the prior turn may point in
opposite directions once the shared topic component subtracts out. Differential measurement
rejecting common-mode signal. **Hypothesis with a clean test, not a result.**

### Two trees, not one

- **Tree A over states** → what this is about (topical retrieval; DYF as it exists)
- **Tree B over deltas** → what kind of move this is (learned transition taxonomy)

Each step carries a state-symbol and a move-symbol. Density semantics transfer well to the
second: **an orphan in delta-space is a surprise** — an unprecedented move, a topic break.
Change-point detection as a byproduct.

**Expected failure:** deltas are anchor-dependent. The same move from different starting points
gives different vectors unless moves are translation-invariant directions — the contested
word2vec-analogy assumption. Expect consistency to be **local at best**.

Mitigation is where the two-tree design earns itself rather than just being tidy: **cluster
deltas within a state-leaf.** Tree A localizes, Tree B finds move structure inside that locale
where translation-invariance is defensible. Localizing is what DYF does.

---

## Frozen basis + drift detection

### A pure rotation is a no-op

Refit PCA, rotate basis + hyperplanes + codes by the same orthogonal `R`, and the partition is
bit-identical. Orthogonal change of basis moves no point relative to any other.

**The only thing that can matter is truncation.** Keeping top-k of d, the real event is a
direction crossing the rank boundary. So:

> **Drift detector = monitor the eigenvalue gap `λ_k` vs `λ_{k+1}` on a recent window.**
> When they converge or cross, the basis is genuinely stale. Everything else is coordinate
> noise, and a detector that fires on it will thrash.

Gotcha: never compare components one-by-one across refits. Sign is arbitrary and near-degenerate
eigenvalues rotate freely within their eigenspace. Compare **subspaces** via principal angles.

### Two drifts, different costs — do not conflate

| | signature | fix | cost |
|---|---|---|---|
| **Region shift** — data moved within the same subspace | residual flat, leaf occupancy skewing / routing entropy collapsing | resplit + rebalance leaves, keep hyperplanes | local |
| **Subspace rotation** — new variance directions appeared | reconstruction residual `‖x − P_k x‖/‖x‖` rising | refit, invalidate downstream | full reindex |

Discriminator: residual rising ⇒ rotation; residual flat but occupancy skewing ⇒ region shift.
Misattribution buys a reindex where a resplit would have done.

Half of this is already instrumented — `point_margin_map` and `extract_boundary_persistence`
in `dyf_tree.py` give margin degradation per depth for free.

### The rotation chain is the artifact

Keep `R` per epoch; leave codes in their native epoch basis; rotate **lazily at query time**.
This avoids the fatal version — reorienting by re-projecting the corpus requires having kept
the original vectors, which destroys the delta-storage motive — and avoids compounding
quantization error, since nothing is ever re-quantized.

Then the payoff: **accumulated rotation `R₀→ₜ` is a sequential signal derived purely from
geometry.** Principal angles between epoch-0 and epoch-t bases measure how far the stream's
frame has travelled. Reorientation events are change-points; spans between them are a
segmentation.

DYF is permutation-invariant and cannot detect sequence — but **the maintenance schedule of a
DYF tree over a stream is order-dependent by construction.**

### Hysteresis is mandatory

Any drift detector on a noisy stream chatters and reorients constantly, inflating the rotation
chain and query cost. Required:

- **Dead band / Schmitt trigger** — `θ_high` to fire, lower `θ_low` to re-arm. One threshold
  guarantees oscillation at the boundary.
- **Minimum epoch length** — no reorientation within N samples of the last, regardless of signal.
- **Bias toward under-reacting.** Cost asymmetry favors it: reorienting late degrades index
  quality gradually and recoverably; reorienting too often compounds bookkeeping permanently.

---

## Cheap move-hashing (the practical entry point)

The DYF hashing machinery applies to deltas **unchanged**. `DensityClassifier`
(`src/dyf/classifier.py:412`, `num_bits` default 14) computes sign-based PCA-LSH at
`classifier.py:830-832` — `(embeddings @ hp.T) >= 0`, bit-packed via `powers` — exposed as
`get_bucket_ids()` / `get_bucket_sizes()` (`:1009`, `:1015`). It takes any `(n, d)` array.

Feed it deltas → each move gets a bucket id. No clustering, no kNN graph, no Louvain. Project
onto b hyperplanes and count.

Two properties worth noting:

- **Sign-LSH is scale-invariant in direction**: `sign(w·αd) = sign(w·d)` for `α > 0`. "Same move,
  bigger" hashes to the same bucket automatically; magnitude is kept separately as a scalar.
  **Verified**: `_pca_hash` (`classifier.py:803-868`) never centers the data — both stages are
  `(embeddings @ hp.T) >= 0`, sign of the projection through the origin. The property holds.

- **The existing hash suits deltas *better* than it suits states.** Its hyperplanes pass through
  the origin, but `sklearn`'s `PCA.fit` centers internally, so `hp` are variance directions about
  a non-zero mean (`classifier.py:854-858`). For states that's a known inefficiency — normalized
  embeddings occupy a narrow cone, so many origin-passing cuts land with nearly all points on one
  side and those bits go near-constant, collapsing effective bit count. **Deltas are approximately
  zero-mean by construction**, so origin-passing hyperplanes are exactly right for them. The
  geometry finally matches the code's assumption.

  Corollary — free drift statistic: if `‖mean(d)‖` is *not* ≈ 0, the sequence has a systematic
  drift direction, which is the reorientation event itself. So the same quantity both validates
  the hashing and detects drift, at O(n) cost.

  Caveat: the two-stage structure (random hash → bucket centroids → PCA **on centroids** → re-hash)
  is centroid-PCA, which per the open feature request "biases toward random hyperplane directions."
  That bias is worse at small n. For short delta sequences, consider fitting hyperplanes by raw PCA
  on the deltas instead of reusing the two-stage path.
- **Basis choice forks the meaning.** Hash deltas in the **delta**-PCA basis → move taxonomy
  (what kinds of moves exist). Hash deltas in the **state**-PCA basis → which semantic axes the
  move traversed (more interpretable, since state axes carry meaning). Both are cheap; they
  answer different questions.

### The sample-complexity rescue

Covariance-based drift detection needs `n ≫ d` (768) — impossible for a single conversation.
A **histogram over 2^b buckets** needs roughly `n ≳ 10·2^b`, i.e. `b ≈ log₂(n/10)`:

| n | usable b | buckets |
|---|---|---|
| 50 | 2 | 4 (marginal) |
| 1,000 | 6 | 64 |
| 100,000 | 13 | 8,192 |

So hashing moves the requirement from *fixed at 768* to *tunable*. Chi-square or JS divergence
between consecutive windows of bucket counts detects drift at a sample cost governed by bucket
count, not dimensionality. This does **not** make 50-turn conversations work — it moves them
from impossible to marginal-at-b=2 — but it rescues the mid-scale regime entirely.

---

## What falls out of one cheap indexing run

In rough order of what settles the question fastest:

1. **Move vocabulary + Zipf curve.** Histogram of delta-bucket ids. *The shape is the first real
   result.* Steeply Zipfian ⇒ a small recurring move set exists ⇒ taxonomy is real. Flat/uniform
   ⇒ deltas are near-random and this whole line is dead. Costs a counter.
2. **Transition matrix** `bucket(dᵢ) → bucket(dᵢ₊₁)`. Structured ⇒ a grammar of moves. This is
   where "sequential pattern" finally lives. 2^b × 2^b counts.
3. **Motifs** — top-k frequent n-grams of move codes. Literally the original question's answer.
4. **Surprise** — `−log p(bucket)` under the marginal or the transition model. Spikes are
   change-points. Cheap version of orphan-in-delta-space.
5. **Per-participant move distributions** (conversation). Do A and B use different move
   vocabularies? JS divergence over move-buckets as an agreement measure — sidesteps stance-blind
   cosine entirely.
6. **Bucket-histogram stability over time** = the drift signal from above, at histogram sample
   cost rather than covariance sample cost.

Note (1) and the delta-PCA spectrum can **disagree**, and the disagreement is informative: a
low-rank delta space with uniform bucket coverage means a continuum of moves with no discrete
vocabulary. Run both.

---

## RESULT: Haxe corpus, run 2026-07-31 — no move vocabulary

> **⚠ SUPERSEDED in part — see "Haxe re-run with localization" below.** This pass lacked
> magnitude-matching and a null-vs-null noise floor. With those controls the order effect is
> real and significant (z ≈ −22 global, −6.5 localized), not the negligible artifact called here.
> The conclusion that survives is the *magnitude* of the effect: code delta structure stays weak.

`/Users/jdonaldson/Projects/haxe/src.dyf` — 23,371 tree-sitter OCaml function chunks, Nomic 768d,
277 files with ≥2 chunks (median 37 chunks/file). Sequence = order of function definitions within
a file. Probe: `benchmarks/sequence_arc/delta_probe.py`.

**Null (this is what makes the result readable):** shuffle chunk order *within each file* and
recompute deltas. Preserves the embedding set, file grouping, file lengths, and the entire state
distribution; breaks only sequential adjacency. Any real-vs-null gap is attributable to order.

| measure | states | deltas (real) | deltas (null) |
|---|---|---|---|
| participation ratio | 81.4 | 88.5 | 90.7 ± 0.5 |
| k90 (comps to 90% var) | 209 | 235 | 223 |
| H/bits @ b=6 | 0.875 | 0.994 | 0.997 |
| H/bits @ b=10 | 0.837 | 0.981 | 0.991 |
| top-10 bucket mass @ b=10 | 0.189 | 0.051 | 0.025 |
| bigram MI @ b=6 | — | 0.634 bits | 0.707 ± 0.012 |

**Verdict: gating measurements 1 and 3 both fail.**

- **Delta space is not lower-rank than state space** — it's *higher* (PR 88.5 vs 81.4). No compact
  move subspace exists. Real deltas are marginally lower-rank than the null (88.5 vs 90.7, ~4σ),
  so order is detectable, but the effect is tiny.
- **The Zipf curve is flat.** Real-delta bucket entropy is 0.98–0.99 of maximum at every bit depth
  — essentially uniform occupancy. There is no small recurring move set.
- **Bigram MI is *below* the null** (0.634 vs 0.707). No grammar of moves. Caveat: consecutive
  deltas share an endpoint (`dᵢ = eᵢ₊₁−eᵢ`, `dᵢ₊₁ = eᵢ₊₂−eᵢ₊₁`), so both conditions are
  contaminated by telescoping anti-correlation, which is *stronger* in the null where excursions
  are larger. Read this as "no grammar signal above a confound that favors the null," not as a
  meaningful negative MI.

**Positive control passed** — the same method finds real concentration in states (H/bits 0.84,
top-10 mass 0.19 vs 0.05 for deltas). The flat delta result is not a broken measurement.

**Two predictions from this doc held:**

1. `‖mean(d)‖ = 0.0016` against mean `‖d‖ = 0.541` — deltas are zero-mean to 3 decimal places, so
   origin-passing sign-LSH hyperplanes are indeed the right geometry for them. Also confirms the
   free drift statistic reads ~0 on a non-drifting corpus, as it should.
2. The stated prior — "flat for code, moves recur in conversation, code edits mostly don't" —
   is what came back.

**What this does and doesn't falsify.** It kills the move-vocabulary hypothesis *for this corpus*.
But note the corpus is a weak instance of "sequence": the order of function definitions in a source
file is **layout, not process**. Functions are grouped by topic, not produced by a sequence of
moves. Real local coherence exists (cos between adjacent chunks = 0.84, `‖d‖/‖e‖` = 0.54) — chunks
near each other are genuinely related — but adjacency carries almost no *directional* information
beyond that. A process-ordered sequence (commit history, conversation turns, edit streams) remains
untested and is where the hypothesis should actually be judged.

---

## RESULT: music (beat-sync chroma), run 2026-07-31 — the frame is everything

Corpus: 101 × 30s iTunes previews (`cochlear/data/spotify/previews/`), beat-synchronous
`chroma_cqt` via librosa, key-normalized per track, 5,559 beats → 5,458 deltas, 12d.
Probes: `benchmarks/sequence_arc/chroma_{probe,rootrel,matched}.py`. Same null as Haxe (shuffle beat order
*within* track).

**Why music, specifically:** it is a **positive control, not another corpus**. Music theory has
catalogued the move vocabulary for centuries — progressions, cadences, voice-leading. If the
method can't find a vocabulary in chord transitions, the method is broken rather than the domain
being flat.

### It initially failed the control

Absolute-frame deltas were indistinguishable from the null: `H/b` at b=8 real 0.9837 vs null
0.9837–0.9852; top-10 mass real 0.090 vs null 0.082–0.098; bigram MI real 0.943 **below** null
1.207. Meanwhile states were sharply concentrated (`H/b` 0.586, top-10 mass 0.704) — chords are a
discrete vocabulary, as expected. So the instrument works on states and saw nothing in deltas.

### Two confounds, both mine

1. **Subtraction is the wrong operation for this space.** Chroma transitions are a **group action**
   (cyclic C12 on pitch classes), not a translation. "Up a fourth" is `chroma(F)−chroma(C)` from C
   but `chroma(G)−chroma(D)` from D — different vectors related by rotation. Plain subtraction
   entangles the move with its starting position. This is precisely the anchor-dependence failure
   predicted above, now with a concrete mechanism.
2. **Magnitude/SNR mismatch.** Adjacent beats are close (`cos = 0.938`), so real deltas are small
   and their *direction* is noise-dominated; shuffled-null deltas pair distant chords and are
   signal-dominated. Filtering each condition at its own median compares them at different absolute
   magnitudes — which manufactured an apparent null advantage.

### Fixing both reverses the result

Root-relative framing (roll source and target so the source's root sits at index 0), with **one
absolute magnitude threshold applied to both conditions**, b=6:

| `‖d‖` band | n real | H/b real | H/b null | Δ |
|---|---|---|---|---|
| 0.353–0.500 | 1091 | 0.9381 | 0.9430 ± 0.0037 | −0.005 |
| 0.500–0.666 | 819 | 0.8906 | 0.8989 ± 0.0047 | −0.008 |
| **0.666–0.869** | **546** | **0.7649** | **0.8437 ± 0.0088** | **−0.079 (~9σ)** |
| 0.869–1.302 | 272 | 0.8687 | 0.8511 ± 0.0242 | +0.018 (n.s., underpowered) |

In the absolute frame the same matched comparison gives only −0.005 → −0.025, growing
monotonically with magnitude but an order of magnitude weaker.

**Conclusion: a move vocabulary is there, and absolute differencing destroys it.** At `H/b = 0.765`
(≈24 effective buckets of 64) root-relative large transitions sit between states (0.653) and
uniform (1.0). Honest limits: the large effect lives in **one** magnitude band; the trend is not
cleanly monotone; the top band is underpowered at n=272. One corpus, 30s previews.

### ✅ AUDIT PASSED — the chroma buckets are genuinely harmonic

Probes: `benchmarks/sequence_arc/chroma_audit{,2}.py`. Same audit that dissolved Haxe: identify bucket content,
name the candidate confound, deflate it, re-test.

**Content.** In the root-relative frame the source root sits at index 0, so the target's argmax
*is* the transition interval. Buckets concentrate hard on recognizable intervals against a
baseline of unison 0.20 / P5 0.14 / P4 0.12:

| bucket | n | dominant interval | enrichment |
|---|---|---|---|
| 8 | 84 | **P4 0.87** | ×7.1 |
| 10 | 95 | **P5 0.77** | ×5.4 |
| 12 | 82 | M2 0.68 | ×6.5 |
| 34 | 97 | m6 0.38 | ×6.7 |
| 44 | 78 | M2 0.51 / m2 0.38 | ×4.9 / ×7.7 |

Perfect fourths and fifths — the circle-of-fifths backbone of Western harmony — falling out of an
unsupervised 6-bit sign hash.

**Confound.** Per-frame loudness is already removed (frames are unit-normalized), so the analogue
of Haxe's identifier-length axis is frame *peakiness*: clear chord = low-entropy chroma, percussive
or transitional beat = diffuse. The correlation is **stronger than Haxe's**: r = 0.79 (vs 0.52).

**But deflating it changes almost nothing:**

| | MI(bucket; interval) |
|---|---|
| entropy axis intact | 1.634 bits |
| entropy axis removed | **1.550 bits (95% retained)** |
| entropy axis *alone* (octiles) | 0.274 bits |

Post-deflation buckets are still cleanly harmonic — P5 ×5.1 (n=182), M2 ×7.5, m7 ×7.9, P4 ×3.2.
The confounding axis carries only 0.27 bits of interval information on its own; it is not the
carrier. **This is the exact opposite of the Haxe outcome**, where deflating one axis took the
effect from z −15.5 to z −0.54.

**One thing that does not survive: the sequential claim.** Deflating entropy drops the real-vs-null
concentration gap from z −2.43 to z −0.48. On reflection the null is the problem — shuffling
*within track* preserves the track's key and chord inventory, so random pairs draw intervals from
the same restricted vocabulary. That null cannot separate "a vocabulary exists" from "temporal
order matters." The vocabulary is real; the evidence that adjacency adds information *beyond the
track's inventory* is weak. A null that breaks order while preserving inventory more sharply
(e.g. resample transitions from the track's own interval marginal) is the missing experiment.

### What this validates

**The two-tree design's core claim.** The notes argued: *"cluster deltas within a state-leaf —
Tree A localizes, Tree B finds move structure inside that locale, where translation-invariance is
defensible."* Root-relative framing **is** that localization, done by hand with domain knowledge
(pitch-class root). It is the difference between seeing nothing and a 9σ effect. The generic,
domain-free version of that maneuver is exactly the two-tree architecture.

It also confirms the flagged liability: sign-LSH scale-invariance hurts on densely-sampled signals,
since near-zero deltas hash by their noise direction. The effect only appears once small deltas are
excluded. **Video would suffer this far worse** — at 30fps adjacent frames are near-identical, so
frame-level deltas would be almost entirely noise. Any video attempt must be shot- or
scene-synchronous, the analogue of beat-sync here.

### Re-reading the Haxe negative

Haxe was run in the **absolute frame only** — the condition that also showed nothing on music,
where a vocabulary provably exists. So the Haxe result is weaker evidence than it looked: it
falsifies absolute-frame differencing on code, not the move-vocabulary hypothesis for code. Re-run
with per-leaf localization before concluding anything about code.

---

## RESULT: Haxe re-run with localization, 2026-07-31 — mechanism confirmed, payoff small

Probe: `benchmarks/sequence_arc/haxe_localized2.py`. Adds the three controls the music run proved necessary:
**magnitude banding** on one absolute threshold, **n-matching** per (region, band) — since fitting
a b-bit basis on fewer points trivially concentrates the histogram — and a **null-vs-null noise
floor**, where each null seed is scored against the mean of the others exactly as `real` is.
Localization = fit the LSH basis separately within each *source-state region* of the dyf tree,
i.e. literally "cluster deltas within a state-leaf." b=4, 6 null seeds, 23,086 deltas.

| granularity | condition | real-vs-null ΔH/b | null-vs-null floor | z |
|---|---|---|---|---|
| height 3 (16 regions) | global basis | −0.0052 | ±0.0002 | −22.6 |
| height 3 | **per-region basis** | **−0.0110** | ±0.0017 | −6.5 |
| height 2 (158 regions) | global basis | −0.0052 | ±0.0001 | −39.5 |
| height 2 | **per-region basis** | **−0.0140** | ±0.0021 | −6.6 |

**Read effect size, not z.** The global condition pools everything into 3 high-n cells, so its
variance is tiny and its z is huge despite the *smaller* effect. Per-region splits into ~11
lower-n cells, raising the floor. Effect size is the comparable quantity across conditions; z only
says "not zero."

**Two conclusions, pulling opposite ways:**

1. **Localization works, on a second and very different domain.** Per-region basis fitting roughly
   doubles-to-triples the effect (−0.005 → −0.011/−0.014), and finer regions help more. Combined
   with the chroma root-relative result, the two-tree design's central claim — *localize with Tree
   A, then find moves inside the locale with Tree B* — now has support in both a 12d group-structured
   signal and a 768d text-embedding space. That is the load-bearing architectural claim of this
   document, and it survived contact with data twice.
2. **Code delta structure is weak regardless.** −0.014 moves H/b from ≈0.96 to ≈0.95: about 14
   effective buckets of 16. Music root-relative reached H/b 0.765 (≈24 of 64). The code effect is
   ~5× smaller and nowhere near a usable move vocabulary. Real, reproducible, and not worth
   building a product on.

**Also falsified:** the Householder "source-aligned frame" heuristic (rotate each delta so its
source points at a fixed reference) went the *wrong* direction globally (+0.004). Expected in
hindsight — a Householder reflection pins one axis and leaves 767 arbitrary, so it isn't a
canonical frame the way chroma's root-roll is. Localization has to come from the partition, not
from a hand-built rotation.

### RESOLVED: the localized Haxe effect is a chunking artifact — conclusion 1 above is RETRACTED

Probes: `benchmarks/sequence_arc/haxe_buckets{,2}.py`, `haxe_deflate.py`.

Token-overlap enrichment came back **flat** — every top bucket sits at its region's baseline
(tok_jaccard 0.264–0.299 vs baseline 0.286; `same_module` is degenerate at 1.00 because pairs are
within-file). So shared vocabulary is not what sorts the buckets.

But the example pairs showed a different axis the overlap test never measured:

```
abstractCast.e1   ->  abstractCast.find_array_write_access_raise
analyzer.Ssa.e    ->  analyzer.Ssa.insert_phi
abstract.m        ->  abstract.anonymous_101
```

Sources are short local bindings, targets are long top-level functions. Quantified:
**corr(delta projected on its best length-axis, `len(tgt)−len(src)`) = 0.52**, and buckets separate
by it — region 111 spans −2.3 to +6.4 mean length change against a +3.0 baseline.

Deflating that single direction out of 768 and re-running the full comparison:

| condition | length axis intact | length axis removed |
|---|---|---|
| global basis | −0.0049 (z −12.2) | −0.0027 (z −6.4) |
| **per-region basis** | **−0.0148 (z −15.5)** | **−0.0013 (z −0.54)** |

**The entire per-region localization gain was riding on the identifier-length axis.** Remove one
direction and it collapses into the noise floor. tree-sitter chunked both short `let e = ...`
bindings and long function definitions; consecutive chunks tend to run short→long; Nomic weights
the title heavily; the buckets sorted on that. It is a property of the *chunker*, not of code.

What survives: a small global residual (−0.0027, z −6.4) that is **not** helped by localization
(−0.0013, n.s.). Whatever it is, it isn't locally structured — which is the opposite of what the
two-tree design predicts.

**Retraction.** The previous conclusion — "localization works on a second and very different
domain, the two-tree claim survived contact with data twice" — was wrong. It survived contact
with data *once* (music). On Haxe the localization gain dissolved under an artifact audit.

---

## RESULT: does ORDER carry information beyond inventory? 2026-07-31 — mostly no

Probes: `benchmarks/sequence_arc/chroma_order.py`, `chroma_grammar.py`. This is the test every earlier null
failed to pose: beat-shuffling changes *which* transitions exist, conflating "a vocabulary exists"
with "order matters."

**Two nulls, and the choice decides the answer.**

- *Permutation null* — permute the transition sequence within track. Preserves each track's move
  multiset exactly. Against this, the progression shows lag-1 MI 0.3208 vs 0.0691 ± 0.0080,
  **z = +31.5**. Looks like overwhelming grammar.
- *Walk null* — resample each track's **root sequence** i.i.d. from that track's own root
  distribution, then derive intervals. Preserves chord inventory, its frequencies, **and
  walk-consistency**, destroying only ordering preference.

| lag | MI real | perm null | z | **walk null** | **z** |
|---|---|---|---|---|---|
| 1 | 0.3208 | 0.0691 ± 0.0080 | +31.5 | **0.3224 ± 0.0265** | **−0.06** |
| 2 | 0.1017 | 0.0656 ± 0.0051 | +7.1 | 0.0623 ± 0.0056 | **+6.97** |
| 3 | 0.0721 | 0.0663 ± 0.0067 | +0.9 | 0.0565 ± 0.0049 | +3.20 |

**The entire lag-1 effect is walk-consistency, not grammar.** z = −0.06. Real intervals chain
through shared roots — interval *i* ends on the root interval *i+1* starts from — so consecutive
intervals are dependent with zero harmonic syntax. The permutation null breaks that chaining and
so scores the mechanical artifact as signal.

**The bigram content confirms it mechanistically.** Every top over-represented transition is an
interval followed by its own inverse:

| bigram | n | lift | note |
|---|---|---|---|
| M7 → m2 | 57 | ×4.18 | +11 then +1 ≡ 0 |
| M3 → m6 | 47 | ×3.68 | +4 then +8 ≡ 0 |
| M6 → m3 | 58 | ×3.48 | +9 then +3 ≡ 0 |
| P4 → P5 | 159 | ×2.41 | +5 then +7 ≡ 0 |

That is **oscillation — leave a chord and come straight back** — exactly what an i.i.d. walk on a
peaked root distribution produces. Not progression syntax.

**What genuinely survives: lag 2–3.** Excess MI +0.039 (z +6.97) at lag 2 and +0.016 (z +3.20) at
lag 3 against the walk null. An i.i.d. walk cannot produce periodicity, so this is real repetition
structure — 4-bar loops and repeated chord cycles. Modest in size but not attributable to any
confound identified.

**Separately: the dyf hash does not carry the grammar.** With b=6 bucket symbols the excess is
*flat* across lags 1–4 (+0.109, +0.068, +0.044, +0.068) rather than decaying — the signature of
local homogeneity, not sequential structure. Interval symbols decay; bucket symbols don't. Whatever
sequential information exists, the sign-LSH code is not the representation that preserves it.

---

## RESULT: stance gate (SNLI), 2026-07-31 — PASSES, with the increment properly measured

Probes: `benchmarks/sequence_arc/stance_{gate,distance_control,content_audit}.py`. Corpus: SNLI test, 9,824
labeled pairs, `BAAI/bge-base-en-v1.5` (cached locally; huggingface.co is blocked in-sandbox,
`nlp.stanford.edu` is not). SNLI is the right instrument because the **same premise recurs across
labels**, so topic is controlled by construction: premise = prior turn, hypothesis = current turn,
delta = the move, label = agree / disagree / topic-drift-control.

**Prediction that was wrong: state space is not stance-blind.** cos(premise, hypothesis) separates
entailment from contradiction at **AUC 0.905** (means 0.771 / 0.546 / 0.683 for
entail / contra / neutral). I had argued embedders put "X is true" and "X is false" next to each
other. They do not, on this data.

**But most of that is distance, not stance.** SNLI contradictions were written to be definitely
false, so they differ in content more. Inside matched cos bands the residual cosine AUC collapses
to 0.56–0.71. So the cheap scalar is largely measuring semantic distance.

**Inside the distance-matched, class-balanced band** (cos 0.601–0.703, n=1321, 49% entailment):

| condition | accuracy | vs. correct baseline |
|---|---|---|
| majority | 0.512 | — |
| **hypothesis-only** (the known SNLI artifact) | **0.783** | ← *this* is the baseline |
| hypothesis state (full) | 0.783 | — |
| **delta direction** (unit-normalized) | **0.849** | **+6.7pp** |
| delta, negation axis deflated | 0.844 | +8.4pp (vs 0.760 deflated hyp-only) |
| **delta vs. RANDOM premise** | **0.728** | **−5.4pp** |

Unsupervised, the dyf-relevant number: MI(delta-direction bucket; label) = **0.2895 bits** in this
band vs **0.0528** for hypothesis state — 5.5×, and *higher* than the full-set value (0.1479),
i.e. controlling for distance sharpens it rather than removing it.

**Three controls, all passed:**

1. **Right baseline.** SNLI's hypothesis-only artifact is real and large (0.783). Scored against
   majority the delta looks like 0.85-vs-0.51; scored honestly the increment is **+6.7pp**.
2. **Negation is not the carrier.** Negation-token rate is 0.016 (entail) vs 0.044 (contra), the
   flag alone classifies at 0.503 (chance), and deflating a lexical negation axis costs 0.005.
   Unlike Haxe's identifier-length axis, this confound is simply absent.
3. **Premise-blind control — the decisive one.** Pairing each hypothesis with a *random* premise
   drops accuracy to 0.728, **below** hypothesis-only. If the delta's advantage came from the
   hypothesis alone it would survive premise scrambling. It doesn't; it degrades. The delta is
   genuinely reading the specific premise→hypothesis relation.

**Verdict: the common-mode cancellation mechanism is real.** The delta reads the *relation* rather
than the topic, and that is what state space does not give you. This is the first clean positive of
the arc, and it is the only significant result here that survived a content audit unchanged.

**Limits, stated plainly.** SNLI pairs are crowdworker-written sentence pairs, not conversational
turns — transfer to real dialogue is untested and is exactly what SwDA (on GitHub, reachable) would
test. The unsupervised hash captures 0.29 bits where the supervised probe reaches 0.849 accuracy,
so the introspection arc's standing verdict holds: **a probe beats the unsupervised structure.** One
embedder, one dataset.

---

## RESULT: SwDA transfer test, 2026-07-31 — does NOT transfer to real dialogue

Probe: `benchmarks/sequence_arc/swda_{parse,relation}.py`. Corpus: Switchboard Dialog Act Corpus
(`cgpotts/swda`, GitHub — reachable in-sandbox), 223,606 utterances / 1,155 conversations, parsed
with disfluency cleanup and DAMSL tag normalization.

**Contrast chosen deliberately.** 35.7% of SwDA utterances are ≤2 tokens and the agreement acts are
near-universally single words — `aa` "Yes.", `ny` "Yeah.", `nn` "No.", `b` "Uh-huh." So `ny`/`nn`
is lexically trivial with no headroom. `b` (backchannel) vs `aa` (agree/accept) is the honest test:
overlapping vocabulary, and which applies depends on what preceded. 6,000 per class, prior turn
required to be ≥4 tokens and from the *other* speaker.

| condition | accuracy | vs current-only |
|---|---|---|
| majority | 0.500 | — |
| **current-turn only** (baseline) | **0.808** | — |
| prior-turn only | 0.660 | −0.147 |
| **DELTA direction** | **0.812** | **+0.005** |
| concat [prior, current] | 0.822 | +0.014 |
| **DELTA vs RANDOM prior** | **0.795** | −0.013 |

Unsupervised: MI(bucket; act) = 0.2784 for delta vs **0.2833 for current state** — the delta is
*slightly worse*. On SNLI it was 5.5× better.

**Compare the two datasets on the decisive control.** SNLI: delta +6.7pp over the honest baseline,
random-premise −5.4pp below it — a 12pp spread proving the delta used the *specific* pair. SwDA:
delta +0.5pp, random-prior only 1.7pp below the real delta. The prior turn is contributing almost
nothing.

**Also note the reversal:** `concat [prior, current]` (0.822) **beats** delta (0.812) here, whereas
on SNLI delta beat concat (0.767 vs 0.760). Where context does help, plain concatenation exploits
it better than subtraction. That is evidence against *subtraction* specifically, not just against
context being useful.

### Diagnosis: common-mode cancellation requires common mode

The mechanism was: the delta cancels shared topic content, leaving the relation. That presupposes
the two states *share* content. Mean cos(source, target):

| corpus | cos | delta mechanism |
|---|---|---|
| chroma, adjacent beats | 0.938 | works (root-relative) |
| SNLI premise/hypothesis | 0.65–0.77 | works (+6.7pp) |
| Haxe adjacent chunks | 0.838 | artifact only |
| **SwDA prior→response** | **0.480** | **fails** |

SNLI hypotheses are rewritings of their premise — subtraction genuinely cancels shared content. A
backchannel shares almost nothing with the statement it answers, so there is no common mode to
remove and the difference is dominated by two unrelated things rather than their relation.

Weak supporting evidence within this run: the delta gain rises across cos bands
(+0.000 → +0.019 → +0.017 for bands 0.418–0.464, 0.464–0.510, 0.510–0.574). Suggestive of the
"needs high cos" story, but small and non-monotone at the top — a hypothesis, not a finding.

**Honest limitation of my design choice:** current-turn-only at 0.808 left only ~19pp of headroom.
I predicted `b` vs `aa` would be strongly context-dependent; it is more lexically determined than
that. A contrast with genuine ambiguity might behave differently, and this run does not rule that
out.

---

## Bottom line for the whole arc

1. **DYF cannot detect sequence.** Permutation-invariant by construction; verified in code.
2. **Localized delta hashing recovers a real transition vocabulary — where one exists.** On music
   it finds P4/P5/M2 at ×5–7 enrichment, unsupervised, and this survives deflating a strong (r=0.79)
   confound with 95% of interval MI retained. On code the equivalent effect was **entirely** an
   identifier-length chunking artifact (z −15.5 → −0.54 after deflating one axis).
3. **Order itself has not demonstrated much value.** The lag-1 signal that looked like grammar is
   walk-consistency (z −0.06 against the right null). A small real periodicity effect survives at
   lag 2–3.
3b. **The delta-as-relation mechanism is real but narrowly scoped.** It works on SNLI (+6.7pp over
   the hypothesis-only artifact, premise-scramble at −5.4pp) and fails to transfer to real dialogue
   turns (SwDA: +0.5pp, scramble −1.3pp, and plain concatenation beats subtraction). It is a claim
   about *pairs*, not sequences — and it holds only where the two states **share content**:
   cos ≈ 0.94 (chroma) and 0.65–0.77 (SNLI) work, cos ≈ 0.48 (dialogue prior→response) does not.
   **Common-mode cancellation requires common mode.** That is the most transferable single sentence
   in this document.
4. **Three nulls were needed to get here, and the first two both gave wrong answers.** Beat-shuffle
   conflated vocabulary with order; sequence-permutation scored walk-consistency as grammar. The
   null design — not the metric, not the model — determined every conclusion in this document.
5. **Every significant result that was not audited for content turned out to be an artifact.**
   Inspecting what is *in* the buckets caught what no significance test did.

---

## Gating measurements (~30 lines; 1–3 run on Haxe + music, see above)

1. **PCA spectrum of deltas vs states.** Is delta-space sharply lower-rank? Decides compression
   *and* whether a move taxonomy can exist.
2. **`‖d‖/‖e‖` distribution.** How close are consecutive items really.
3. **Delta-bucket Zipf curve** (above, #1). Cheapest of the three; possibly decisive alone.
4. **Direction consistency across anchors.** Is delta-PCA top-k stable across state-regions, or
   does each region carry its own move basis? Decides global vs per-leaf Tree B.
5. **Stance test.** Embed matched agree/disagree pairs; compare cosine in state-space vs
   delta-space against a topic-matched control. Falsifies or supports the common-mode-cancellation
   claim directly.

**Required middle column** (pre-flight #2): for storage it's PQ; for semantics it's DYF-on-states.
The claim to defend is `delta − states`, never `delta − nothing`.

---

## What breaks

- **`n < d` for single conversations.** 50 turns cannot support a 768d covariance estimate; every
  "drift" measured is estimation noise. Hashing partially rescues this (above); PCA-based drift
  detection does not work per-conversation, period.
- **Small-n generally.** Prior finding: DYF's wins are at ≥100k–1M; at small n it was confidence
  calibration, not a retriever (LongMemEval). Prefer per-file-version / per-hunk / per-turn
  granularity over per-commit / per-conversation — more sequential *and* more samples.
- **Repos are DAGs, not chains.** Merges force a linearization choice (first-parent? topo?
  author-date?) that sits as a confound under every motif found. Per-entity histories are the
  genuinely sequential object.
- **Diff/delta embeddings may encode file type, not change kind.** Same shape as the `size1`
  grab-bag. Look at raw values before believing any facet.
- **The baseline for drift detection is a cron job.** Adaptive reorientation only beats "rebuild
  nightly" when rebuilds are expensive *and* drift is bursty. If drift is steady, use a timer.
  Measure the drift time-series before building a controller for it.
- **If labels exist, expect a linear probe to win.** The introspection arc's verdict: supervised
  probes beat DYF's unsupervised structure on every task tried, and its one success (obfuscated-harm,
  AUC 0.92) was a flat difference-of-means probe. DYF earns a place here only if the object of
  interest is the *structure* — whose partition, breaking at what depth — not a binary call.

---

## Where this came from

Started from "repos are by nature sequential — can DYF detect sequential patterns?", moved
through conversation/agreement as a better-posed instance of the same shape (true chain rather
than DAG; agreement supplies the labels the repo case lacked), then to delta-as-object, then to
the frozen-basis question.

Connects to `POSTGRES_NOTES.md`, which flagged as an open policy question: *"an incrementally
maintained index needs a policy for when the global fit goes stale — periodic REINDEX, or accept
drift and measure it."* The `λ_k`/`λ_{k+1}` boundary monitor plus a lazily-applied rotation chain
is a third answer, and a better one than either.
