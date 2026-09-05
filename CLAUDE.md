# DYF Project Notes

## Heading (asserted 2026-08-31, consumers amended 2026-09-05)

**Destination: dyf's primitives become trustworthy enough that downstream consumers can
rely on them. Moving toward v1 *quality*, not more capability.**

Score work against that. The test is "can someone depend on this?", not "is this
interesting?".

**Consumers (amended 2026-09-05).** `sec10quant` and `shortorder` via the Python API —
and **agents, as a first-class consumer, via both the CLI and the Python API.** Agents
are a target, not a side effect of having shipped a CLI.

This widens *which surface* the heading covers, not what counts as done on it. Before the
amendment the dependency surface was the exported Python API, which is why the three
standing audits look only there — and why a CLI that printed nothing at all survived
(`AGENT_LEGIBILITY_TODO.md`). The CLI and the module-to-module contracts are now in
scope. The bar is unchanged: can someone depend on it?

⚠ **The risk this amendment carries.** A bearing check found this project drifting
because no destination was stated, and "add a consumer class mid-flight" is structurally
the same move that produced the drift. The guard: agent work earns its place only where
it makes an *existing* surface dependable. Parseable output, loud failure, honest
docs — yes. New mechanisms to serve agents — no, same as before.

**Why this heading, and why now.** A bearing check found the project *drifting* — real
progress, but effort allocated opportunistically because no destination was stated, which is
also why "is this good bang for buck?" had no crisp answer. The evidence:

- **109 public exports, 72 callables, 30 modules, 26 test files.** The validated surface is
  materially smaller than the shipped surface.
- **Two of the handful of features inspected by hand were silently broken** — both with
  performance claims in their docstrings (`KNOWN_ISSUES` #4, #5). That is a poor base rate
  for the ~70 not inspected.
- **Tests that pass on a degenerate result**: measured 5% shape-only, plus vacuous,
  fully-guarded and no-assert categories — all now closed.
  `benchmarks/audit_test_assertions.py` measures this and carries a `--selftest`.
- **12+ absolute cosine/margin constants** are measured inert on real corpora. A v1 would
  freeze those into the API.

Pre-v1 is the only cheap moment to fix any of that (see "Backward Compatibility" in
`~/Projects/CLAUDE.md` — break freely before v1).

### What this heading implies

- **In scope**: making existing primitives correct, measured, and honestly documented;
  closing the gap between shipped and validated surface; deleting or relativising knobs that
  do nothing.
- **In scope as of 2026-09-05**: giving the CLI and the Python API a contract an agent can
  rely on — machine-readable output, meaningful exit codes, errors that say what to do,
  and a cheap way to ask what an artifact contains before paying to open it. This is the
  same "can someone depend on this?" test applied to a consumer that cannot squint at a
  blank terminal and guess.
- **Out of scope until the floor is solid**: new mechanisms in dyf. `SPECTRAL_NOTES.md`
  records six consecutive falsified hypotheses of that shape; the only thing that shipped
  from that arc (`dyf.dedup`) started from a *measurement of the data*, not from a mechanism.
- **The one rule that keeps paying**: pick the outcome variable before building the probe.
  And measure against a null or an incumbent, never against intuition about what a number
  should look like.

### Standing audits

Re-run these when touching the public surface; each exists because it caught a real defect:

| command | catches |
|---|---|
| `python benchmarks/audit_public_api.py` | exported callables returning empty/all-zero/constant (has a canary that reproduces #5) |
| `python benchmarks/audit_test_assertions.py` | tests asserting only types and lengths |
| `python benchmarks/audit_absolute_thresholds.py` | absolute cosine constants that do not transfer across corpora |

Open queue with priorities: `KNOWN_ISSUES.md`.

## Release Workflow

When adding features or making API changes:
1. **CHANGELOG.md** — add entry under current version (or new `## Unreleased` section)
2. **README.md** — update if the feature touches Quick Start, API examples, or installation

## Visualization Development Pattern

The pydeck visualizations use a WebSocket bridge for live control:

1. **Start the viz server**: `python demo/viz_server.py --port 8766 --dir demo --watch`
2. **Open viz through server**: `http://localhost:8766/rog_3d_birch_clusters.html` (not as a file)
3. **Control via MCP tools**: `draw_circle`, `highlight_points`, `zoom_to_cluster`, etc.

The `--watch` flag enables hot-reload when HTML files change.

Static HTML files (`file://`) won't receive WebSocket commands — must use the HTTP server.

## Sanity Check Before Deep Work

Before embarking on lengthy computation (full-corpus embeddings, model retraining, multi-hour pipelines):
- **Validate the problem exists first** — check a quick sample, eyeball the data, run a 5-minute smoke test
- **Check if a simpler approach works** — structured metadata filtering (GMDN codes, product codes) often beats embeddings for thematic extraction
- **Don't solve non-existent problems** — e.g., don't re-embed 2.7M records when a regex on category labels gets you 95% of the way there in 30 seconds
- **Measure before optimizing** — e.g., zstd compression on .dyf files only saved 6% (float16 embeddings are near-random), but cost browser compatibility. Always check actual impact before assuming an optimization matters.
- When in doubt, spend 10 minutes on a sanity check before spending 2 hours on a solution


## Embedding Visualization

On macOS (Apple Silicon), use **[mlx-vis](https://github.com/hanxiao/mlx-vis)** (`pip install mlx-vis`) for UMAP/t-SNE/PaCMAP. Pure MLX on Metal GPU — **9-28x faster** than `umap-learn` (70K points: 5s vs 50s). GPU-rendered scatter plots and animations, no matplotlib.

```python
from mlx_vis import UMAP
Y = UMAP(n_components=2, n_neighbors=15).fit_transform(X)
```

## DAG-Oriented Task Flow

Prefer DAG (directed acyclic graph) oriented pipelines wherever possible:
- **Render loops, MCP server pipelines, preprocessing steps** — model these as DAGs with explicit dependencies
- **Create data structures that track provenance** — each output should know its inputs and the transform that produced it
- **Enable interrogation of the DAG** — code should support questions like:
  - Can we re-run just this step, or do we need upstream work?
  - What's the cost of re-running from this point vs. from scratch?
  - Is a cached intermediate still valid, or have its inputs changed?
- **Practical application**: `dyfviz.py` and `rog_preprocess.py` currently run monolithic pipelines; prefer checkpointed stages where each stage reads/writes a known artifact and can be skipped if the artifact is fresh
- Think `make`-style: each target depends on prerequisites, only rebuild what's stale

