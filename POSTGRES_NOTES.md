# DYF as a PostgreSQL extension — design notes

**Status: DESIGN. Nothing built.** 2026-07-31.

Long-range goal: Postgres as the single datastore, with DYF supplying the *topology* layer
(dense / bridge / orphan) that no existing extension provides. LibreChat then becomes one
consumer of a Postgres+DYF stack rather than a thing to be rearchitected — see
"Where this came from" at the end.

---

## Findings index

Operational conclusions, most actionable first. Detail follows in the sections below.

| finding | where |
|---|---|
| **Do not REINDEX on a schedule.** Age is the wrong trigger; +64% growth over 2 years costs ~0.01 recall | "REINDEX policy this implies" |
| **Trigger on depth-1 JS divergence** — 16 counters at ingest, no queries, no ground truth. Healthy ≤0.005 bits, collapse at 0.26 | "The delta's depth profile beats its size" |
| **The requirement is COVERAGE, not stationarity.** Disjoint companies cost nothing; withholding whole section types collapses the index. ~2–5% representation of a region restores it | "Growth is free; only *missing coverage* breaks it" |
| **`fit_with_hyperplanes` is an exact frozen partition** — routing reduces to LSB-first sign bits, 100% reproducible | "The frozen partition is exact" |
| ⚠ **Never call `Tree::remove` on a frozen index** — it nulls hyperplanes for a whole subtree. Tombstone instead | "Eviction: tombstone only" |
| ⚠ **Dirty-leaf fraction is ANTI-correlated with health** (84.6% healthy, 43.6% collapsed). Optimising delta size selects for a broken index | "Delta-encoded snapshot sequences" |
| **Value prop is address stability**, not storage or compute. Tree build is 3–4s at 229k | "Why freeze at all" |
| **29.4% of this corpus is near-duplicate content** → shipped as `dyf.dedup` | `SPECTRAL_NOTES.md` |
| ⛔ **Spectral characterisation is a closed direction** — five hypotheses, five falsified | `SPECTRAL_NOTES.md` |

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
  thin" in `SPECTRAL_NOTES.md`.

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
