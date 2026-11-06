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

- **The validated surface is materially smaller than the shipped surface.** `dyf.overview()`
  reports the current export count and `benchmarks/audit_public_api.py` reports how many it
  can actually exercise; the gap between them is the point. (Counts are deliberately not
  written here — every hand-typed total in this repo went stale, including the ones that
  used to sit on this line.)
- **Two of the handful of features inspected by hand were silently broken** — both with
  performance claims in their docstrings (`KNOWN_ISSUES` #4, #5). That is a poor base rate
  for the ones not inspected.
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
| `python benchmarks/audit_cli_surface.py` | subcommands that exit 0 saying nothing, or fail with a traceback instead of a message (has a `--selftest` and a live mute canary) |

⚠ The first three inspect the **exported Python API only**. That is why a CLI printing
zero bytes passed all three; `cli.py` is not in the public surface. The fourth exists to
cover it, and deliberately does **not** test `--help` — argparse prints help itself
without touching the logger, so every subcommand answers `--help` with rc 0 even when its
real execution path is mute. It runs real invocations instead.

Open queue with priorities: `KNOWN_ISSUES.md`.

### Measuring stick for descriptive output (asserted 2026-11-06)

dyf has never won on retrieval accuracy and this is not the stick it is scored on. The wins on
record are continuity, stability and legibility over time: group ids that carry over between
windows (71% vs 36%), drift detection (15× separation), dedup (29% of SEC), resolution from
fifty labels better than knowing k. The shape read-outs from the November cut audit are the
same kind of thing — descriptions that reproduce across trees — and are scored on three sticks,
each fixed before the probe and each with a null:

- **Decision value**: does the read-out change what a consumer does, and is the change right?
  First instance: inspecting "pocket" branches catches 1.8–2.5× the label errors per item of a
  random sample on five of seven bench sets (1.0–1.2× on MoCap, AG News) and 1.5–6.4× the
  points the final re-cut moves.
- **Agent legibility**: an agent answering fixed questions about an unfamiliar index (what kind
  of data, how many natural groups, which regions are unreliable, which resolution) with and
  without the summary, scored on correctness and tool calls spent.
- **Lead time on drift**: a shape change per window vs the Jensen–Shannon occupancy trigger,
  scored as lead time and false alarms on sec10quant's 52 quarterly windows.

Retrieval recall and query-confidence AUC stay the stick for search, and shape lost there
(LongMemEval, 2026-11-06). No composite "insight score"; a description is valued by whether a
reader acts on it, so a human read sits in the loop at least once per read-out.

### Closed: where the leaf-grouping quality goes (2026-11-04/05)

Two days of measurement on the seven-dataset bench (`benchmarks/cut_audit_2026_11/`, memory
`project_community_conductance_bench_2026_11`, `project_hash_native_composition_2026_11`,
`project_leaf_graph_construction_ceiling_2026_11`) settled four design questions that keep
coming back. Numbers are in `KNOWN_ISSUES.md` #9, #11, #14; the conclusions:

- **The atoms are cones and that is fine.** Any linear hash gives convex cells; shape has to come
  from composition. Centring the threshold (`sign(w·(x−μ))`) makes purer leaves but moves the
  shipped output by ≤ 0.03 — not worth a format change while the re-cut below stays (#11).
- **The leaf graph's edges are not the bottleneck.** The full point kNN graph coarsened to leaves
  adds at most +0.06 ARI over the shipped centroid-cosine kNN at matched k; shared-neighbour,
  spread-aware and hash-derived graphs are a wash or worse (#9).
- **The shipped partition is spherical k-means seeded by Louvain** (`_reassign_points`): a Voronoi
  re-cut that moves 4–31% of points, helps single-cell, hurts MNIST, nets ~0 — documented, kept,
  and the leaf-level / straggler-only alternatives lose to it (#14).
- **What is left** is the objective and community count modularity returns (leaf oracle vs block
  labels: 0.1–0.7 ARI, mostly k) and the atoms' purity (leaf oracle 0.72 vs Leiden-on-points
  0.885 on MNIST). The map equation (Infomap) was tried as the objective: it is a *fine-k*
  prior, 49–90 modules, +0.06 on the 120-type atlases (mostly the k; Louvain at resolution 2–3
  gets most of it) and −0.2 to −0.3 on 10-class images and 4-class text — not a default. Three
  label-free resolution criteria (map-equation codelength, edge-dropout stability, Laplacian
  eigengap) all lose to resolution 1: mean regret 0.18 / 0.13 / 0.18 vs 0.08, and knowing k
  would only reach 0.03. Prediction strength through the frozen tree never clears 0.7 and picks
  the coarsest level. **What works: about fifty randomly labelled items pick the resolution
  within 0.02 of the best in the sweep, better than knowing k** — that is the consumer advice
  (README "How It Works"). If a nested multi-resolution profile ever ships, build it as the
  resolution sweep snapped bottom-up into exact nesting (the sweep already nests 92–99% of
  points; snapping costs ≤0.02 ARI at the useful level on five of seven sets, 0.04–0.06 on
  Fashion and brain), not by merging up from a fine partition (0.06–0.13 lost on images) or
  by recursive Louvain inside communities (skips every intermediate k); per-community
  persistence does not predict purity. "Is it a continuum" is answerable only per boundary at
  a level: the cut profile of touching sibling pairs sorts class boundaries from continuum
  slices on the cell atlases (AUC 0.66–0.97) and brain (0.61–0.78), not on images (≈0.5) —
  annotate splits with it on atlas-shaped data, never as a dataset verdict. Per point, the
  box-counting local dimension read off a deep tree's path (log node size vs log 1/radius)
  is the one label-free continuousness measure with signal: it predicts which points the
  re-cut moves (AUC ≈ 0.6–0.7, stable over three seeds) and anticorrelates with leaf purity
  on every dataset; a longer binary ladder is no better and margin ratios carry nothing — a
  moderate signal, not built. Read per rung and calibrated on synthetic families, the
  dimension *profile* is interpretable as a description (dip = clusters separate at that
  scale, spike = a hyperplane sliced a homogeneous region, low-then-rising = noisy manifold):
  atlases and brain are 2–3-D sheets plus noise, MiniLM text has no low-d structure at any
  scale, MoCap is ≈2 at fine scales, images bump where classes mix — but the shape adds
  nothing over the mean as a per-point score. The profile is a *leaf* property (all points in
  a leaf share the path); per-leaf nearest-family labels are class-aligned above a shuffle
  null on every dataset, curve/sheet leaves are the purest, the "dip" shape marks small
  impure pockets the re-cut moves, and the same shapes recur across the whole embedding
  except a few localized ones that name real structures. Clustering leaves on shape does not
  rediscover the tree (AMI with the top-level branch 0.02–0.10 on rung-free metrics), the
  same shapes recur on most top-level branches, and branches sharing a shape are not related
  by class, community or position — a common shape is local vocabulary, a single-branch shape
  identifies a region. The shape is data, not tree: across
  three seeds a point's profile reproduces far better than random on every dataset; the
  unstable minority is misclassified only on Fashion and droplet. The shipped bridge,
  connector and boundary flags do not coincide with the unstable points (AUC 0.42–0.59), and
  `bridge_persistence` took 37 minutes at 140k points (KNOWN_ISSUES #15). Branch shape does
  not calibrate retrieval: on LongMemEval it adds −0.005 CV AUC to the routing features, and
  top-1 plus mean top-5 cosine (0.669) beats top-1 plus every dyf routing feature (0.613) —
  query confidence should come from those two cosines, not from dyf. Reading the shape per
  *community* (a deep tree per shipped community) instead of per leaf does not make it a
  reliability signal either: for "which communities are impure" the free numbers — radius,
  conductance, size — reach leave-one-dataset-out AUC 0.79, the per-community shape 0.54, the
  global-leaf shape restricted to the community 0.36, and adding shape to the free numbers
  lowers them; the community-level profile reads differently (brain's hub becomes one coarse
  axis then noise) but ranks nothing better. Shape stays a description; reliability comes from
  radius and φ. Nothing in the graph-construction or objective
  direction is worth another probe without a consumer whose data sits in the regime where it moved.

### Parked: sequence key / leaf Markov state model (2026-10-01)

One day of pre-registered probes asked whether dyf can read non-linear structure (cycles,
gradients, direction). Result: the tree does not destroy cycles, it pools over them — leaves are
phase slices, so centroid-level topology sees only polygons. The structure lives in the *order* of
points, which dyf does not model. A sequence layer (two fields `seq_id`/`seq_pos`, a tree-node
symbol string per sequence, a transition count matrix) recovered the MoCap gait period from the
complex eigenvalue of the leaf transition matrix (65 vs 59 frames; dance null), found shared
transition poses via egress entropy, and read direction via entry/exit mutual information. On
sec10quant the embedding carries no fiscal cycle (probe at chance), so no consumer has data that
needs it yet.

**Justin's call: not compelling enough to pursue now.** Do not build it into dyf or start a
package without a consumer question. If it is picked up, it is a downstream package (dyfviz
pattern), dyf core unchanged. Everything needed to resume is in
`/Volumes/Models/dyf_bench_2026-10-01/` (`leaf_msm_*`, `sequence_key_layer_*`, the plans, and
`nonlinear-structure-under-the-dyf-tree.qmd`) and in memory `project_leaf_graph_cycles_probe_2026_10`.

## Release Workflow

When adding features or making API changes:
1. **CHANGELOG.md** — add entry under current version (or new `## Unreleased` section)
2. **README.md** — update if the feature touches Quick Start, API examples, or installation

## Visualization: it lives in dyfviz

The browser viewer (`dyf_viewer.html`, `dyf_tree_view.html`, `pydeck_overlay.js`,
`js/dyf_reader.mjs`, `viz_server.py`) is canonical in `dyfviz/src/dyfviz/assets/` since
2026-11-06; `dyfviz tour <file.dyf>` serves it with the WebSocket bridge. The copies that used
to sit in `demo/`, the `rog_mcp.py` live-control MCP server and the `.mcp.json` entry for it
(which pointed at a `.dyf` that no longer existed) were removed in the same cleanup. `demo/`
now holds only what `demo/README.md` documents, the data builders, and `puppet/`; one-off
experiments moved to `benchmarks/demo_archive_2026/`. Any viewer change goes to dyfviz, and its
`tests/test_viewer_assets.py` checks the packaged viewer is complete.

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
- **Practical application**: the `dyfviz` enrichment ladder is the live instance (each level reads what the one below wrote; `demo/dyfviz.py` and `rog_preprocess.py`, the monolithic predecessors, were removed 2026-11-06); prefer checkpointed stages where each stage reads/writes a known artifact and can be skipped if the artifact is fresh
- Think `make`-style: each target depends on prerequisites, only rebuild what's stale

