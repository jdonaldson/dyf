# DYF and Parquet: interop at the dataset boundary, not inside the file

Date: 2026-09-18
Status: Tier 0 + Tier 1 implemented in `src/dyf/parquet_io.py`; Tier 2/3 not built (and Tier 2 is recommended against)

## Verdict up front

**Do the round trip (`to_parquet` / `from_parquet`). Do NOT make Parquet the in-file payload
encoding.** The motivating ask was "add Parquet to dyf" so that a polars-centric pipeline can hold
embeddings beside business columns. That is satisfied entirely by an import/export boundary. Putting
Parquet *inside* `.dyf` looks cheap — and structurally it is — but it is the wrong granularity and
would most likely make files bigger, not smaller. Reasoning in Tier 2 below.

## The finding that makes import/export cheap

`format.rs` is **already encoding-agnostic**. It stores opaque byte ranges and hands back raw bytes;
nothing in the format layer parses Arrow.

- `BatchDescriptor { offset: u64, length: u64, num_rows: u32 }` (`format.rs:65`) records **no
  encoding** — just where the bytes are and how many rows they cover.
- The writer takes pre-encoded bytes: `create(path, tree, leaf_batches: &[Vec<u8>], …)`
  (`format.rs:126`), `append_items(&mut self, batch_data: &[Vec<u8>], num_rows: &[u32])`
  (`format.rs:447`), `append_field_layer(…)` (`format.rs:402`).
- The readers return raw bytes: `read_base_batch(usize) -> io::Result<Vec<u8>>` (`format.rs:613`),
  `read_overflow_batch` (`:618`), `read_field_layer_batch` (`:623`).
- `FieldDescriptor { name, arrow_type: String }` (`format.rs:84`) already carries a per-field type
  tag as a string, and `BuildParams` (`format.rs:91`) already has both `quantization: String` and
  `compression: String`.
- `metadata: HashMap<String, String>` with `update_metadata(&[(&str, &str)])` (`format.rs:480`)
  gives a free-form, post-hoc-writable key/value space with no FlatBuffers schema change.
- `dyf-rs/Cargo.toml` already depends on `arrow = { version = "58", features = ["pyarrow", "ipc",
  "ipc_compression"] }` — pyarrow interop and IPC compression are in place.

So encode/decode already lives *above* the format module. That is the property to preserve.

---

## Tier 0 — `to_parquet()`: export a Parquet dataset (recommended, do first)

Read the batches, concatenate, write a Parquet dataset that polars and DuckDB open natively.
**No format change at all.** This is the deliverable that satisfies the original ask.

```python
idx.to_parquet("out/", partition_by="leaf")   # or a single file
```

Design points:

- **Preserve leaf locality — but as sort order, NOT as Hive partitions.** `.dyf` batches are
  one-per-leaf, so the layout is already grouped by semantic similarity, and that locality is worth
  keeping: a predicate correlated with semantics (same subcategory, manufacturer, cohort) then
  touches fewer row groups. ⚠️ **Do not `partition_by="leaf"`** — the measured leaf counts above mean
  `gudid_viz.dyf` would emit **5,938 directories averaging 8 rows each**, a textbook small-files
  problem that costs more than the pruning earns. Instead: emit a `leaf_id` column, write rows **in
  leaf order** (which is free — concatenating batches in batch-index order is already sorted), and
  let Parquet row-group min/max statistics on `leaf_id` do the pruning. Same benefit, one file.
  Hive partitioning stays available behind an explicit opt-in for the rare wide-leaf index.
- **Emit the embedding column as `FixedSizeList<float32, dim>`**, not `List`. Verified: polars
  1.39.3 restores this as `pl.Array(Float32, shape=(dim,))` through a Parquet round trip, so no
  import-side cast is needed. Cast half-float storage up to float32 on export — Parquet has no
  half-precision logical type.
- **Carry the row index.** Emit dyf's `u32` item index as an explicit column so a polars result can
  be joined back to the index without relying on row order.
- **Include provenance**: source `.dyf` path, its `BuildParams`, `embedding_dim`, `total_items`,
  and a content hash, written to Parquet key/value metadata. Without this an exported dataset
  cannot be tied back to the index that produced it.

Cost: small. Nothing in `dyf-core` changes.

## Tier 1 — `from_parquet()`: build a `.dyf` over a Parquet dataset (recommended, second)

The other half of the round trip, and what lets polars be the front end.

```python
idx = dyf.from_parquet("features/", embedding_col="emb", carry=["sku", "unspsc_code", "spend"])
```

- Read the embedding column through Arrow **zero-copy** into the contiguous `&[f32]` the tree
  already wants. `LshTree::insert(&mut self, idx: u32, embeddings: &[f32])` (`tree.rs:258`) documents *"`embeddings`
  must contain ALL items"* — a build-time constraint that Parquet satisfies naturally, since a
  column scan yields the whole column contiguously. This is the same property that made the
  Postgres `aminsert` path hard (one tuple at a time) and makes Parquet easy.
- **Carry non-embedding columns through as field layers**, not as a side table. `append_field_layer`
  is the existing mechanism and it means business columns live in the index.
- Row order: the `.dyf` item index is assigned by scan order. Record the scan order (sorted file
  list + row offsets) in `metadata` so a rebuild is reproducible.

### Why field layers matter more than Parquet does

`append_field_layer` lets you add columns to an existing `.dyf` **without a rebuild**. That is the
schema-evolution story a feature store needs: add a new attribute, or add a second embedding
model's vectors as a new layer and A/B them against the first. A flat Parquet dataset requires
rewriting files to do the same. This capability is already in the format and is under-exploited.

Caveat to document: the tree's hyperplanes come from a **global PCA fit at build time**, so appended
data drifts from the projection. Any append path needs a staleness policy — e.g. rebuild when
appended rows exceed X% of `total_items`. `fragmentation()` (`format.rs:512`) and `compact()`
(`format.rs:525`) already exist as the primitives to hang that on.

## Tier 2 — Parquet as the in-file payload encoding (NOT recommended)

Structurally almost free, and still the wrong call.

**How it would work** (recorded because it's cheap to describe and someone will suggest it):
`BatchDescriptor` needs no change. Add `payload_encoding = "arrow_ipc" | "parquet"` to `metadata`
— no FlatBuffers bump — and switch the codec in `dyf-rs`. The 4 reserved header flag bytes
(`format.rs:147`, written as zeros) are a tempting alternative but are **never validated on read**,
so an old reader would proceed and then fail confusingly while IPC-decoding Parquet bytes. Metadata
is the honest place, and readers must **refuse on an unrecognized encoding** rather than guess.

**Why not:**

1. **Granularity mismatch.** Parquet's advantages — row-group min/max statistics, dictionary and RLE
   encoding — are *per row group*. A dyf leaf batch is far below row-group scale: demos run
   **measured p50 is 5 rows/leaf on `gudid_viz.dyf` and 34 on a 50k product index** (see Resolved
   below). Default Parquet row groups are ~1M rows. Every leaf batch becomes a single tiny row group
   where statistics buy nothing and dictionaries have no repetition to exploit.
2. **Per-batch overhead goes the wrong way.** Each Parquet batch carries its own schema and footer
   (~1 KB+). Measured leaves hold a **median of 5 rows** (`gudid_viz.dyf`, 5,938 leaves), so the
   footer would rival or exceed the payload, 5,938 times over. Outcome is not "plausibly larger" —
   it is larger.
3. **The compression win is probably already banked.** `ipc_compression` is enabled in
   `dyf-rs/Cargo.toml` today.
4. **External tools still can't read it.** Batches sit inside a `.dyf` container at offsets relative
   to `data_origin`; polars cannot open them regardless of encoding. The interop goal is served by
   Tier 0, not by this.

**If anyone wants to overturn this, run the ablation first** — and note the middle column, which is
the one that matters:

| arm | what it measures |
|---|---|
| baseline: Arrow IPC, uncompressed | current worst case |
| **cheap_alt: Arrow IPC + zstd** (already available) | **what compression alone buys** |
| complex: Parquet per leaf batch | the proposal |

The number that decides it is `complex − cheap_alt`, not `complex − baseline`. Measure file size,
build wall time, and batch read latency on a real index (the 362,953-product catalog build is the
right fixture).

## Tier 3 — `.dyf` referencing external Parquet (only if genuinely needed)

`.dyf` becomes a pure index; vectors and columns live in Parquet that polars/DuckDB read directly.
Maximum interop, and it reintroduces exactly the failure mode `POSTGRES_NOTES.md` rejected for an
mmap'd `.dyf` beside the heap: the index does not co-move with the data, so a rewritten Parquet
file **silently invalidates every `u32` in the index**.

Mandatory guard if this is ever built — a fingerprint in `metadata`, checked on open, hard failure
on mismatch, never a warning:

- `row_count`
- schema hash (field names + arrow types, ordered)
- content hash of the embedding column bytes
- the resolved file list and row offsets that defined scan order

Silent misalignment is the dangerous shape: the index still answers, it just answers about the wrong
rows. Prefer Tier 0/1 unless something concrete forces this.

## Non-goals

- **ANN.** Unchanged from `POSTGRES_NOTES.md`: pgvector does ANN well; the differentiator is
  topology (`dyf_class`, bridges, paths) and the facet diversification result in `RAG_NOTES.md`.
  Nothing here should trade topology fidelity for retrieval speed.
- **A transactional store.** `append_items` returns `io::Result` — file I/O, not a journal with
  fsync ordering — and the tree is single-writer (`&mut self`). The right posture is **Parquet as
  source of truth, `.dyf` as a derived, always-rebuildable index**. That posture also dissolves the
  durability objection rather than solving it.
- **`cut_tree_to_labels`.** Agglomerative merge is ~O(n²) in leaf count and did not return in 6
  minutes at k=1,000 on 362,953 items. Use the tree's own leaves.

## Resolved 2026-09-18 (were open)

1. **polars preserves `pl.Array` through Parquet.** Measured on polars 1.39.3:
   `Array(Float32, shape=(3,))` round-trips identically. No cast needed on import; drop the
   defensive re-cast the Tier 0 note originally called for.
2. **Leaf-size distribution measured on two production indexes** — this is much more lopsided than
   inferred, and it settles Tier 2 outright:

   | index | items | dim | leaves | rows/leaf p50 | mean | max |
   |---|---:|---:|---:|---:|---:|---:|
   | `gudid_viz.dyf` | 50,000 | 768 | **5,938** | **5** | 8.4 | 196 |
   | `products_…_n50000` (fastembed) | 50,000 | 384 | 842 | **34** | 59.4 | 4,420 |

   A median leaf holds **5 to 34 rows** against a default Parquet row group of ~1M. Confirmed, not
   inferred.

## Built 2026-09-18

`src/dyf/parquet_io.py` (Tier 0 + Tier 1), `tests/test_parquet_io.py` (13 tests),
`tests/test_build_params_recorded.py` (4), `tests/test_parquet_roundtrip_e2e.py` (2 + 1
env-gated on `DYF_E2E_INDEX`), `benchmarks/parquet_roundtrip_smoke.py`. Exported via
`dyf.to_parquet` / `dyf.from_parquet`. Full suite 718 passed / 17 skipped, ruff clean.

**A defect fell out of building it, upstream of Parquet entirely.** `_build_flatbuffer_index`
wrote `num_bits=3, min_leaf_size=4, seed=42` into the file whenever a caller omitted
`build_params=` — presenting this library's defaults as recorded fact. `index_products.py:272`
built its 50k index with `num_bits=2, min_leaf_size=64`; the file reported 3 and 4. So the first
round trip inherited the recorded params faithfully and *still* produced **19,974 leaves against
842**, tripling the file, with nothing raised. Fixed at the generator: `build_dyf_tree` now
returns `tree["build_params"]`, the writer prefers explicit → tree-recorded → last-resort, and
anything guessed is named in a `build_params_inferred` metadata key. Round trip is now
shape-preserving — 50,000 × 384 real embeddings, 839 → 839 leaves, 1.01x size, vectors
bit-identical at float16 storage precision.

Worth noting what caught it: the leaf count. Sizes and timings all looked reasonable, and the
export/import both "worked". Only comparing a structural invariant across the round trip exposed
it — which is the same lesson as the fingerprint guard Tier 3 would need.

That invariant is now the e2e test, and it was **mutation-checked**: stripping
`tree["build_params"]` to restore the pre-fix behaviour takes the fixture from 68 -> 68 leaves to
68 -> 1,794, tripping both the shape and size-ratio assertions. The real-index arm is gated on
`DYF_E2E_INDEX` rather than hardcoding a path — the index used during development lives in a
private tree, which has no business being referenced from this repo.

## Still open

1. Whether `append_field_layer` batch bytes are ever decoded inside `dyf-core` or only passed
   through. If only passed through, Tier 1's carry-columns work is pure binding-layer code.
2. Does `remove` (`tree.rs:284`) reclaim space or tombstone? Open from `POSTGRES_NOTES.md`; affects
   whether a Parquet-backed rebuild can be incremental.
3. Whether `append_field_layer` batch bytes are ever decoded inside `dyf-core` or only passed
   through. If only passed through, Tier 1's carry-columns work is pure binding-layer code.
4. Does `remove` (`tree.rs:284`) reclaim space or tombstone? Open from `POSTGRES_NOTES.md`; affects
   whether a Parquet-backed rebuild can be incremental.
