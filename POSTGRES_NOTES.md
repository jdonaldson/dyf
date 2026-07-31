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

1. **Does `remove` reclaim or tombstone?** It `swap_remove`s from the leaf's item list and
   updates counts/centroid — an index-structure update. Whether the underlying batch space is
   reclaimed is unverified. Matters for any TTL-style expiry.
2. **Durability on the append path.** `append_items` returns `io::Result`, i.e. file I/O — not
   necessarily a journal with fsync ordering and crash recovery. Fine for a rebuildable index;
   **not** fine for a log of record.
3. **Concurrency.** `&mut self` on `insert`/`append_items` means single-writer. Compatible with
   one-writer/many-mmap-readers; rules out arbitrary concurrent writers.

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
