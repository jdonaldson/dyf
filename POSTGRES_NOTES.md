# DYF as a PostgreSQL extension — design notes

**Status: DESIGN. Nothing built.** 2026-07-31.

Long-range goal: Postgres as the single datastore, with DYF supplying the *topology* layer
(dense / bridge / orphan) that no existing extension provides. LibreChat then becomes one
consumer of a Postgres+DYF stack rather than a thing to be rearchitected — see
"Where this came from" at the end.

---

## The differentiator: topology, not ANN

pgvector already does nearest-neighbour well (HNSW/IVFFlat in Postgres pages). **Do not
rebuild that.** DYF's reason to exist in Postgres is the query nobody can currently write:

```sql
-- items with no semantic neighbours
SELECT * FROM items WHERE dyf_class(embedding) = 'orphan';

-- items bridging two concept regions
SELECT * FROM dyf_bridges('items_idx');

-- semantic path between two points
SELECT * FROM dyf_path_between($1::vector, $2::vector);
```

Density structure, bridges, orphans, and navigable paths are the product. If a design decision
trades topology fidelity for ANN speed, it is the wrong trade — pgvector wins that race and
should.

---

## THE CRUX: Postgres indexes live in Postgres pages

A Postgres index access method must store its data in the buffer manager and emit WAL records.
That is not ceremony — it is what buys crash recovery, streaming replication, and PITR.
pgvector puts HNSW in Postgres pages for exactly this reason.

**An extension that mmaps its own `.dyf` file beside the heap gets none of that**: it does not
replicate, does not survive PITR, and can desync from its table after a crash. So "wrap
dyf-core in `pgrx` and call it an index" is not a viable shape for a real IndexAM.

### The signature that makes it concrete

`dyf-core/src/tree.rs:258`:

```rust
/// `embeddings` must contain ALL items (including the new one at index `idx`).
pub fn insert(&mut self, idx: u32, embeddings: &[f32])
```

**The tree does not own the vectors.** It holds `u32` indices into an external contiguous
array, and every `insert` needs that whole array to recompute the leaf centroid
(`compute_centroid(embeddings, dim, &items)`).

In Postgres the vectors live in the heap and arrive **one tuple at a time** via `aminsert`.
There is no "all embeddings" array to pass in. So centroid maintenance must become
**incremental** (running mean or sum+count per leaf) rather than recompute-from-all. That is an
algorithmic change to the kernel, not glue code.

Related: hyperplanes come from a **global PCA fit at build time** (`build_dyf_tree(...,
fit_method="raw_pca")`). Splits are local (`split_leaf`), but the top-level projection is
global. An incrementally-maintained index needs a policy for when the global fit goes stale —
periodic REINDEX, or accept drift and measure it.

**That policy question is now measured** — see "When does the global fit actually go stale?"
below. Short answer: far less often than expected, and the trigger is coverage, not age.

---

## Two tiers. Do Tier 1 first.

### Tier 1 — functions over a prebuilt index (weeks)

Ship DYF classification as UDFs and table-returning functions reading a `.dyf` file built
offline. **No `ambuild`/`aminsert`/`ambulkdelete`, no page layout, no WAL.**

- `pgrx` is mature and dyf-core is already Rust — real leverage.
- Rebuild on a schedule, which is how DYF is used today anyway.
- Delivers the entire novelty above.
- Honest limits to document: not crash-consistent with the table, not replicated, index goes
  stale between rebuilds, and the file must be present on every replica that serves the
  functions.

This is the version worth building. It answers "is topology-in-SQL useful?" without paying for
an IndexAM.

### Tier 2 — a real index access method (months)

Only if query-planner integration and incremental maintenance are actually required. Requires
reimplementing the tree on Postgres pages with WAL records, plus the incremental-centroid work
above. This competes with pgvector on its home turf and should not start until Tier 1 has
proven demand.

---

## What DYF2 already gives us (corrected 2026-07-31)

Earlier note in this line of thinking claimed `.dyf` was immutable. **Wrong** — that is true of
DYF1/DYF3 only, whose front-loaded header cannot grow in place (hence the CHANGELOG's "convert
to DYF2 first"). DYF2 is read-write:

| capability | location |
|---|---|
| `insert` (routes to leaf, splits when overfull) | `dyf-core/src/tree.rs:258` |
| `remove` (finds leaf, merges when underfull) | `dyf-core/src/tree.rs:284` |
| `append_items` | `dyf-core/src/format.rs:447` |
| `append_field_layer` | `dyf-core/src/format.rs:402` |

So the write path exists in the kernel. Three things still need establishing before relying on
it for anything transactional:

1. **~~Does `remove` reclaim or tombstone?~~ Answered 2026-08-02, and it disqualifies
   `remove` for any frozen-basis index — see "Eviction: tombstone only" below.**
2. **Durability on the append path.** `append_items` returns `io::Result`, i.e. file I/O — not
   necessarily a journal with fsync ordering and crash recovery. Fine for a rebuildable index;
   **not** fine for a log of record.
3. **Concurrency.** `&mut self` on `insert`/`append_items` means single-writer. Compatible with
   one-writer/many-mmap-readers; rules out arbitrary concurrent writers.

### Eviction: tombstone only. `remove` is off-limits. (decided 2026-08-02)

**Invariant: a frozen-basis index must never call `Tree::remove`.** Not a preference —
`remove` destroys the basis it depends on.

`tree.rs:310` calls `merge_children(parent)` whenever a leaf falls below `min_leaf_size`,
and `merge_children` (`tree.rs:417`) is not a local repair:

```rust
let all_items = self.collect_items(parent_idx);  // ALL descendants
self.nodes[parent_idx].items = Some(all_items);
self.nodes[parent_idx].hyperplanes = None;       // <- basis destroyed
self.nodes[parent_idx].children.clear();
self.nodes[parent_idx].num_bits = 0;
// Note: old child nodes become orphans in the nodes vec.
```

One underfull leaf collapses its whole parent subtree — up to 16 siblings at
`num_bits=4` — and nulls the hyperplanes. They are *gone*, not bypassed: those addresses
cannot be recovered without a refit. Since address stability is the entire reason to
freeze (see below), a single expiry can undo it for a large region.

Three further hazards in the same path:

- **Orphan nodes leak.** The comment says `compact()` "would" reclaim them. `compact`
  exists (`format.rs:525`, exposed on `DyfFile`) but whether it reclaims *tree* nodes as
  opposed to file space is untested.
- **Silent no-op removals.** `remove` locates its target via `find_leaf(emb)` and then
  searches that leaf for the index. After any topology change a point may no longer route
  to the leaf holding it, and removal returns `false` with no error. Compounds with the
  merging above.
- **`swap_remove` reorders survivors**, so a leaf that lost one point serialises to
  different bytes than a filtered version would — byte-identity dedup breaks more often
  than the removal count suggests.

**The policy: tombstone in the leaf batch, skip at search, reclaim at scheduled
compaction** — the shape the incremental-updates design already assumed. Topology
untouched, hyperplanes intact, addresses stable. Costs retained space and scanning dead
rows. Two cheap companions: a `remove`-without-merge mode if in-place shrinking is ever
wanted (the partition tolerates severe imbalance — max leaf grew 947→1545 with no recall
cost), and **ranking only occupied cells at probe time** so drained leaves do not consume
probe budget.

Not urgent in practice: tree `insert`/`remove` are not exposed to Python in `dyf_rs`
0.9.0 (`DyfFile` offers `append_items`, `append_field_layer`, `compact`, `update_tree`
and no removal), so this is a constraint on whoever builds the write path, not a live bug.

**Scope warning for everything measured below: it is all append-only, and eviction makes
drift strictly worse.** Under append, old points stay and keep anchoring coverage. Under
eviction, coverage *decays* — the fit retains cells for regions the data has left while
the data moves into regions the fit never saw. The "+64% growth costs ~0.01 recall" result
is an **optimistic bound** and does not transfer to a sliding window. Expect uniform
eviction to be nearly free (it is subsampling, structurally the `random` control) and
evict-oldest-under-drift to be the damaging case. Untested; the design that would
discriminate is a window sliding over a *composition schedule*, with a fixed-mix window as
the control — sliding over SEC quarters alone would measure that SEC does not drift.

---

## When does the global fit actually go stale? (measured 2026-08-01)

Answers the REINDEX-policy question raised above. Corpus: `sec10quant/data/filings.dyf`,
229,243 SEC 10-Q sections, 768d, real filing dates, 501 companies, 4 section types.
Probes in `benchmarks/sequence_arc/sec_*.py`; all re-runnable (`sec_extract.py` first).

**Why freeze at all, given rebuilds are cheap?** Not storage (~2.3×, modelled) and not
compute (seconds). The reason is **address stability**: a rebuild renumbers everything
downstream. The concrete instance already in this repo is
`demo/energy_label_cache.json` — cluster index → *hand-curated* name ("ELMED European
Bipolars", "Laparoscopic Instruments"), under three separate clusterings. A refit points
every curated label at the wrong cluster, and that content cannot be regenerated
automatically. `community_id` is likewise a persisted stored field
(`enrich/_cluster.py:264`) read by the viz pipeline and the browser reader
(`gudid-explorer/js/dyf_reader.mjs:535`). Read everything below as serving address
stability; the storage and compute numbers are secondary and mostly negative.

### The frozen partition is exact — no recentering

`DensityClassifier.fit_with_hyperplanes(embeddings, hyperplanes)` does bucketing only.
Verified at 100% on all four checks (`sec_frozen_probe.py`): re-deriving buckets from
hyperplanes alone, adding 20k unrelated points, reordering the array, and routing a
100-point subset all leave assignments unchanged. Routing reduces exactly to

```
bucket_id = sum_i( (x @ H.T > 0)_i << i )      # LSB-first sign bits
```

So an index can be **re-used as the foundation for its own successor** with no kernel
change. Routing cost measured at ~358k points/sec/core in numpy.

### Growth is free; only *missing coverage* breaks it

Same-parameter control, base ≈ 61% / stream ≈ 39%, varying only how the split is drawn.
Recall@10 against exact kNN, queries drawn from the **stream** side (content the
partition never saw), mean ± sd over 4 seeds (`sec_shift.py`):

| split | dirty% | unseen% | frozen R@10 | gap @32 | gap @128 |
|---|---|---|---|---|---|
| random (control) | 84.6% | 0.69% | 0.9002 | −0.002 ± 0.004 | +0.000 ± 0.002 |
| temporal (2 yr, +64% growth) | 71.3% | 1.17% | 0.8958 | +0.010 ± 0.007 | +0.004 ± 0.004 |
| disjoint companies | 68.2% | 1.79% | 0.8785 | +0.017 ± 0.015 | +0.006 ± 0.005 |
| disjoint sections | 43.6% | 9.55% | 0.7782 | **+0.119 ± 0.010** | +0.062 ± 0.007 |

The first three are **not distinguishable** — the noise floor on a 500-query draw is
±0.015, and all three sit inside it. A single-seed run invented an ordering among them
that did not survive replication; don't reintroduce one.

Splitting 501 companies disjointly is a real distribution shift by any normal definition
and costs nothing measurable — because every *region* of the space is still covered.
Splitting by section type removes whole regions, and that collapses the partition.

**The response is a knee, not a slope — and not literally a cliff either.** Sweeping
shift continuously (`sec_dose.py`) by
oversampling two section types with weight *w*:

| base purity | 59.1% | 68.6% | 74.2% | 81.2% | 87.9% | 93.5% | 100% |
|---|---|---|---|---|---|---|---|
| gap @32 | +0.007 | +0.018 | +0.002 | +0.008 | +0.009 | −0.004 | **+0.122** |

Pearson r = +0.97 but **rank correlation = +0.18** — the correlation is one point of
leverage. Heavy sampling *bias* is harmless up to 93.5% purity; only categorical absence
matters. That sweep *looks* like a cliff because it has no samples between 93.5% and
100%; bracketing that gap with base size held fixed (`sec_cliff.py`) resolves it into a
steep but continuous knee — flat above ~5% representation, rising sharply below it:

| points from missing region | 0 | 47 | 188 | 938 | 1,876 | 4,691 |
|---|---|---|---|---|---|---|
| unseen% | 9.34 | 6.56 | 8.04 | 5.33 | 3.46 | 1.97 |
| gap @32 | +0.132 | +0.094 | +0.081 | +0.057 | +0.020 | **+0.001** |

**~2–5% representation fully restores quality.** 4,691 points — 3.5% of a 134k base.

### The delta's depth profile beats its size as a trigger

The scalar drift rate says *how much*; the depth at which the delta appears says *what
kind*. Jensen-Shannon divergence between base and stream occupancy, computed at each
level of the frozen tree, mean over 3 seeds (`sec_depth_profile.py`):

| depth | cells | pts/cell | random | temporal | ticker | **section** |
|---|---|---|---|---|---|---|
| 1 | 16 | 8,742 | 0.0000 | 0.0045 | 0.0043 | **0.2596** |
| 2 | 234 | 598 | 0.0008 | 0.0156 | 0.0125 | 0.4094 |
| 3 | 2,358 | 59 | 0.0082 | 0.0486 | 0.0489 | 0.5085 |
| 4 | 10,619 | 13 | 0.0383 | 0.1151 | 0.1360 | 0.5848 |

Margin between the split that broke the index and the worst harmless one:
**57.6× at depth 1, 26.3× at depth 2, 10.4× at depth 3, 4.3× at depth 4.**

**Coarse levels are the better detector, and the reason is sampling noise.** A depth-1
cell holds ~8,700 points, so its occupancy is measured almost exactly and the random
control sits at 0.0000. A leaf holds ~13, so multinomial noise alone drives the control
to 0.0383 — swamping the distinction between harmless and damaging drift.

Two properties make depth-1 JS the right refit trigger:

- **It fires on exactly the drift that breaks things.** Section (recall gap +0.119) reads
  0.2596; all three harmless conditions read ≤0.0045. Not "a drift detector" — a detector
  for *damaging* drift specifically.
- **It is better calibrated than the scalar.** Unseen-rate ranks ticker (1.79%) above
  temporal (1.17%), implying an ordering that the recall measurements say does not exist.
  Depth-1 JS puts them at 0.0043 and 0.0045 — correctly identical.

Use the whole profile, not depth 1 alone: drift that redistributes mass *within* coarse
cells is invisible at depth 1 and shows only at depth 3–4, where the margin is 4×. The
ticker split is exactly that case. So depth 1 answers "is this structural?" and the deep
levels answer "is anything happening at all?" — at much lower confidence.

Where unseen-bucket fallbacks fire shifts shallower under real drift too (depth-2 share
14–18% harmless vs 46% for section), but at only ~3× separation it is the weaker signal.

### REINDEX policy this implies

1. **Do not REINDEX on a schedule.** Age is the wrong trigger; two years and +64% growth
   cost ~0.01 recall, recoverable by probing harder.
2. **Trigger on depth-1 JS divergence** between the fit-time occupancy and the current
   occupancy — 16 counters, computed at ingest, no ground truth, no queries. Healthy
   ≤0.005 bits, collapse at 0.26 bits on this corpus; anywhere in 0.02–0.10 is a wide
   dead zone to place a hysteresis band in. Keep the **unseen-bucket rate** (0.7–1.8%
   healthy, 7–10% collapsed) as a secondary confirm: it is knife-edge sensitive, moving
   7.3→9.6% across seeds on the same split, so it should never be the sole trigger.
3. **Fit the base for coverage, not proportion.** The fit sample must *contain* every
   region; it need not represent them proportionally. For a time-partitioned table this
   means stratifying the fit set across the whole space rather than fitting on a recent
   window — a contiguous-window fit is exactly the failing condition above.
4. **Escalate before rebuilding.** Mild degradation is fixable with probe budget
   (disjoint-companies recovers from +0.017 to +0.006 going 32→128). Reserve REINDEX for
   the coverage cliff, which probing does *not* fix (+0.119 → +0.062 only).

### Delta-encoded snapshot sequences — real but modest, and the metric is a trap

If snapshots must be addressable (time-travel queries, reproducible eval sets), each can
be stored as a delta against its predecessor: hyperplanes shared verbatim, plus the leaf
batches and centroids whose membership changed.

Header is 6–30% of a `.dyf`, but it is ~75% **centroids** and only ~25% hyperplanes —
every node carries a dim-length centroid, only internal nodes carry hyperplanes, and
leaves outnumber internals ~12:1 (`dyf_header_compose.py`). So sharing hyperplanes is
free but small; the bulk dedupes only for unchanged leaves.

**Modelled** — not measured — over 10 quarterly snapshots (`sec_granularity.py`). No
delta-encoded file was ever written or weighed. The inputs are measured (dirty fractions
per quarter, 1,898 B/row from the real file, the 1.09 dirty-node multiplier from
`sec_sequence.py`'s 47.5%/43.3%); the totals are arithmetic on top of them, and they
assume **zero format overhead** for generation ids and manifests: 10 independent snapshots = **4,457 MB**; delta-encoded at the
search-validated granularity = **1,903 MB (2.34×)**; at the byte-optimal granularity =
1,495 MB (2.98×), though that config is ~3 pts/leaf and was never validated for search.

Two cautions:

- **The saving is arithmetic, not semantics.** A big quarter dirties 44% of leaves; a
  permutation null says random arrival would dirty ~5,400 vs the observed ~4,900 — only
  ~10% better. Temporal clustering contributes almost nothing.
- **Dirty-fraction is anti-correlated with index health.** It is 84.6% when the partition
  is healthy and 43.6% when it has collapsed, because out-of-distribution data piles into
  a corner of a partition that does not fit it. **Optimizing for delta size selects for a
  broken index.** Use unseen-bucket rate for health; use dirty-fraction only for capacity
  planning.

Tree construction is 3–4s for 229k points, so none of this is about saving compute. (That
figure is `build_dyf_tree` only — a full REINDEX also writes ~500MB of Arrow batches,
unmeasured here. The conclusion is insensitive to it; the number should not be quoted as
total rebuild time.)

### Cell volume does not say where the basis is thin — and the eigenvalues are a trap

The coverage result above raises an obvious follow-on: depth-1 JS says *when* to refit, so
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
same inversion as the dirty-fraction trap above — out-of-distribution data piles into a
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
the empty-cell metric in "Scope limits" as a second obvious-looking coverage metric that
measures nothing.

### What a split is doing, and deriving `num_bits` instead of setting it

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

#### Deriving `num_bits`

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

#### Confirmed under dyf's real router (`sec_hier_routing.py`)

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

#### ⛔ It does not replicate. Do not build this. (`sec_multicorpus_bits.py`)

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

### Spectral shape by depth and along paths (`sec_depth_spectra.py`)

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

⚠ **But n=1 in the positive class again**, the same trap already listed in Scope limits for
the section split. One steep-spectrum corpus, one payoff; "steep spectrum causes the
benefit" is not separable from "motion capture differs for some other reason". The cheap
decisive test is a **synthetic sweep with tunable alpha** (0.5 → 3.0), not another
real corpus.

Caveats: only nodes with n ≥ 300 enter, so depth 4 keeps 47 of many on SEC — survivorship
toward large nodes; two corpora.

### When children are NOT self-similar to their parent (`sec_nonselfsimilar.py`)

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

#### Is non-self-similarity just duplicates? Partly — ~24% of it. (`sec_dedup_ablation.py`)

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

#### Are the pockets fragments of one concept? No. (`sec_shattered_pockets.py`)

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

### Scope limits

- One corpus, one embedding model (768d), one tree shape (`max_depth=4, min_leaf=16`,
  ~11k leaves). Thresholds are candidates, not calibrations.
- **n=1 in the positive class.** Exactly one condition (disjoint sections) ever broke the
  index. Every claim about the depth-1 JS trigger — the 57.6× margin, "selective for
  damaging drift" — rests on that single contrast. One breaking case is an existence
  proof, not a detector; a second independent breaking condition is the cheapest thing
  that would firm this up.
- **All recall numbers come from a bespoke harness, not from dyf's own search path.**
  Routing is the numpy reimplementation in `sec_seqlib.route()` (verified equivalent to
  `fit_with_hyperplanes` at 100%, and reproducing build-time membership at 100%), and
  retrieval is `sec_seqlib.ivf_search()` — probe-top-N-centroids then exact scan. It is
  *not* `dyf.LazyIndex.search()`. Frozen-vs-fresh comparisons are fair because both sides
  use the identical harness, which is all the conclusions require, but absolute figures
  like "R@10 = 0.88 at probe 32" are properties of this harness and say nothing about
  dyf's shipped search performance.
- **Append-only.** Eviction/TTL was not tested and dirties leaves differently. See
  "Eviction: tombstone only" above — the results here are an optimistic bound, because
  under eviction the fit's coverage decays instead of persisting.
- The section split is a split by document *type* and is probably an upper bound on
  real-world shift severity.
- One real file (`haxe/src.dyf`) stores **zero hyperplanes** across 191 internal nodes.
  Such a file cannot serve as a frozen foundation; whichever build path drops them needs
  tracing before this is relied on.
- The depth↔semantic-scale mapping rests on **two contrasting shifts** (one coarse, one
  fine), not a survey. It is a working hypothesis with two confirming instances.
- Measured-and-discarded metric, documented so nobody re-derives it: "share of stream
  landing in cells that were empty at fit time" is **structurally zero at every depth**.
  The tree is built from base points, so every cell it has contains base members by
  construction. New territory can only surface as an unseen-bucket fallback, never as an
  empty cell. The obvious metric returns zeros that look like "no drift". Per-cell volume
  is the second metric of this genre — see "Cell volume does not say where the basis is
  thin" above.

### Relation to graft

`graft/NOTES.md:644` ran the complementary half on 2026-07-30: flat 12-bit `_pca_hash`
addresses, 540 source chunks, corpus **edited then refit** — asking what survives a refit.
This work freezes the basis and **appends** — asking whether a refit is needed at all.
The mapping is graft's "coarse prefix" ≙ shallow tree levels; both are sign-packed PCA
projections, one flat and one recursive.

Two results transfer back:

- graft's "you can freeze the top for free" (variance 0.203 vs 0.206 at 540 docs) now
  holds at 229k docs under a different metric (recall). Its hedge that a partial-refit
  cost gap might open on a larger corpus **resolves negatively** — tree construction is
  3–4s at 229k (file write excluded). The win is address stability, never wall-clock.
- graft only ever perturbed by *editing* existing content, which is the sampling-bias
  regime — harmless here too. It never withheld a region, so it could not have found the
  coverage cliff.

graft's caveat #1 stands unchanged: a fitted basis is corpus-relative and needs both
sides to share a fit, which is exactly the coordination git tree hashes do not require.
Everything here lives inside a coordinated system.

---

## Naming

`pg_dyf` or `dyfpg`, matching `pgvector` / `pg_documentdb` convention.

---

## Where this came from

Traced back through: (a) wanting to replace LibreChat's MongoDB, (b) discovering LibreChat
already runs pgvector as a separate service for RAG while Mongo holds entities, (c) sizing an
upstream abstraction layer at **350-400 files / 4-7 months / permanently-diverged fork**, and
(d) noticing that Postgres-as-single-store with DYF for topology makes the LibreChat question
secondary rather than central.

The LibreChat survey that produced (c) is in memory under `librechat_mongo_to_postgres_survey`.
Its most reusable finding for this project: **embeddings never touch Mongo** — the RAG boundary
is already an HTTP hop to a separate pgvector container, so consolidating onto Postgres is a
simplification there, not a migration.
