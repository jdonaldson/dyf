# DYF Project Notes

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

## 🔄 RESUME CONTEXT - DELETE AFTER READING

### Current Status
Parameter-free clustering gallery scaffolded on the docs site. One notebook (Digits) is live and verified end-to-end; six others are draft stubs with metadata, ready to be filled in. Site SCSS tweaked (fluid figure grid + removed isoline code-block background).

### What Was Built
- `docs/gallery/index.qmd` — landing page with grid listing (true-k / recovered-k / NMI per card).
- `docs/gallery/_gallery.py` — shared helpers: `run_dyf`, `run_kmeans`, `run_hdbscan`, `plot_single`, `metrics_table`. Writes a temp .dyf to satisfy `louvain_cluster_leaves`'s LazyIndex requirement.
- `docs/gallery/digits.qmd` — working first notebook. DYF recovered k=10 exactly, NMI 0.696 vs. oracle k-means 0.773. HDBSCAN column surfaces `% discarded` because defaults drop most points as noise.
- `docs/gallery/{mnist,fashion-mnist,twenty-newsgroups,olivetti-faces,cmu-mocap,wine-iris}.qmd` — draft stubs with metadata only.
- `docs/_quarto.yml` — added `Gallery` nav entry.
- `docs/custom.scss` + `docs/custom-dark.scss` — added `.gallery-fluid` CSS grid (`auto-fit, minmax(320px, 1fr))`); removed isoline `background: url('hero-bg-light.png')` from `div.sourceCode` in both themes.

### Blocked On
- User feedback on the rendered digits page: leading-notebook ordering, HDBSCAN framing (column vs footnote), figure size.
- Decide next notebook to write. MNIST is the obvious extension (same domain, 40× bigger), but 20 Newsgroups could land stronger wins (oracle k-means is known to be weak on text).

### Next Action When User Returns
- Take ordering / framing feedback.
- Start on MNIST (or whichever notebook the user prioritizes): follow the digits template — load, `run_dyf`, `run_kmeans`, `run_hdbscan`, three `plot_single` calls inside `::: {.gallery-fluid}`, `metrics_table` footer.

### Context to Remember
- `louvain_cluster_leaves` requires a `LazyIndex` handle → shared helper writes a temp .dyf. A `louvain_from_tree(tree, embeddings)` helper would remove the detour and is worth a future feature request.
- `quarto render A.qmd B.qmd` on multiple gallery files in one invocation fails with `withBinaryFile: does not exist` on the listing page. Rendering solo works. Work around, don't diagnose unless it recurs off-gallery.
- Matplotlib default figsize (15, 5) overflows Quarto's ~700px body column. Use square `figsize=(5, 5)` panels and let the CSS grid tile them.
- Fluid layout pattern: `.gallery-fluid { display: grid; grid-template-columns: repeat(auto-fit, minmax(320px, 1fr)); gap: 1.25rem; }` — reusable for any multi-figure Quarto page.

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

