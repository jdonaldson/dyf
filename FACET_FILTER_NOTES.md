# Feature idea: queryable attribute facets (predicate pushdown into search)

**Status: not built, not scheduled.** Written up 2026-08-04 while consuming dyf from the Curvo
enrichment MCP server, because the cost of *not* having it got measured incidentally. Recording the
measurement so the decision can be made on numbers rather than intuition later.

Related: [[dyf_postgres_extension_direction]] — predicate pushdown is precisely the thing a
Postgres index gives you that a standalone `.dyf` does not, so this note and that direction are two
answers to one question.

## What exists today

Attributes CAN live alongside the vectors:

- `DyfFile.append_field_layer(name, fields, batch_data)` writes a named field layer.
- `LazyIndex` exposes `stored_field_names`, `has_stored_fields`,
  `get_stored_fields(item_indices)`, `extract_all_fields()`.
- `discover_categorical_columns` and `diversify_by_facet` exist for post-hoc faceting.

So storage is solved. **Query is not:** `search(query, k, nprobe, return_routing, backend)` and
`search_ivf(query, k, nprobe)` take no predicate argument, and `get_stored_fields` is a *retrieval*
call — you pass indices you already have. There is no `where attr == v`.

Access pattern is therefore **search-then-filter**, never filter-then-search.

## The measured cost

Real categorical attribute (`subcat`) over 40,000 real catalog embeddings (floret, 200d, unit-norm).
For each query: how deep into the similarity ranking must we scan before `k=10` results satisfy
`attr == v`? That depth is the over-fetch a pushdown search would avoid.

| selectivity of the filter | median depth | p90 depth | over-fetch vs k=10 | queries that never reached k=10 |
|---|---:|---:|---:|---:|
| 1–10% of rows | 2,470 | 11,171 | **247×** | 0/200 |
| 0.1–1% | 12,011 | 31,397 | **1,201×** | 0/200 |
| <0.1% | 19,779 | 36,347 | 1,978× | **186/200** |

Attribute distribution on the same data — most filters land in the expensive bands:

| band | distinct values | share of rows |
|---|---:|---:|
| 1–10% | 12 | 22.0% |
| 0.1–1% | 179 | 46.5% |
| <0.1% | 3,077 | 31.5% |

**The failure mode at high selectivity is the real argument, not the latency.** Below 0.1%
selectivity, 93% of queries cannot assemble `k=10` from the whole ranking — so a search-then-filter
implementation silently returns fewer results than asked for. Quiet under-return is worse than slow.

Harness: `/tmp/claude/facet_cost.py` (regenerable; needs `dyf_clusters.parquet` + floret).

## Sketch of a design, if it is ever wanted

The tree already partitions the data, so each node can carry a summary of the attribute values
beneath it, and search prunes any subtree whose summary cannot satisfy the predicate:

- **low-cardinality categoricals** — a bitset or small value-set per node. Cheap, and this is the
  case that matters (`subcat` has 3,268 distinct values across 40k rows).
- **high-cardinality fields** — do NOT do this. A per-node summary over something like
  `product_id` approaches the size of the data and prunes nothing. Bound the feature to declared
  low-cardinality field layers and reject the rest at build time.
- **numeric ranges** — a per-node min/max is the obvious analogue, untested here.

## Why it was NOT built for the case that surfaced it

Honest scoping, so this does not read as a blocked dependency:

1. The consumer (`product_next` in `curvo/shortorder/mcp/`) does not call `search()` at all — it
   groups coded siblings with Polars `group_by`. Pushdown would speed up a call it never makes.
2. The bug that prompted the investigation (a pharma product offered device codes) was a **routing**
   error, fixed with a structured catalog column (`ProductTypeID`). A facet filter would have let
   the wrong instrument run with a filter on it, which is still the wrong instrument.
3. Field layers already cover the storage half, and `get_stored_fields` retrieves fine for known
   indices.

So: a real capability gap with a real measurement behind it, and no current consumer blocked on it.
Revisit when something actually needs filtered ANN retrieval.
