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

### Current Status (2026-04-28)
Consolidation pass after the audio cross-domain validation. The
diagnostic stack now has **7 layers** and is documented as a
synthesis (`memory/synthesis_diagnostic_stack.md`). **Read that first**
on resume — it's the readable overview. `spectral_dyf_composition.md`
remains the detailed chronological narrative.

The 7 layers: (1) Geometric (anisotropy + path/star + density + ρ),
(2) Spectral preprocessing (Nyström-via-DYF, ~3s on 1.3M),
(3) Eigenvalue plateau (effective dim, gap_2, e12, cycle_alt),
(4) Within-cluster discriminability (effect-size + feature Jaccard +
identity alignment), (5) Reference-module scoring (mt/Hb/IEG/...),
(6) Edge classification (chord taxonomy + transitions + SCC),
(7) Persistent homology (Vietoris-Rips Betti-1 on bucket centroids,
ripser, 0.4s on 1317 brain centroids).

Outputs: 6-class node taxonomy, 5-mode chord taxonomy, PH cycle ranking.

### Validated Across 8 Datasets
- **Brain 1.3M**: cell cycle confirmed (PH rank 7); **vascular
  compartment discovered** (endothelium 24+32, pericytes 26, smooth
  muscle 33); **~30/35 clusters annotated** via targeted markers
- **Paul15 hematopoiesis**: GMP and MEP branching points as PH cycles
- **PBMC3k**: small-n MST artifact identified; anisotropy gate >0.18
- **MoCap**: sit-down cycle, walk path, dance/sway 2D-lattice
- **MNIST**: spectral preprocessing unfolds curved digits (Δanis +0.5)
- **CIFAR raw pixels**: correctly identified "no structure"
- **20NG MiniLM**: effective dim correlates with topic semantic breadth
- **Audio (synthesized music)**: chromatic key cycle (12-fold) detected
  as rank-1 PH feature spanning all 4 styles × all 12 keys × all 4 tempos

### What's Built and Where
- **Memory**: `synthesis_diagnostic_stack.md` (overview, ~200 lines),
  `spectral_dyf_composition.md` (detailed narrative, ~1700 lines)
- **Gallery notebook**: `docs/gallery/diagnostic-stack.qmd` — live
  visualizations + 7-layer table + brain/audio results, rendered to
  `docs/_site/gallery/diagnostic-stack.html`
- **Primitives**: `docs/gallery/_gallery.py` exposes `nystrom_spectral`,
  `cluster_diagnostic`, `classify_cluster`, `diagnose_all_clusters`
- **Per-experiment scripts**: `/tmp/*.py` — informal, reproducible
- **Cached features**: `dyf/.embeddings_cache/` (twenty_ng_minilm.npz,
  audio_synth_features.npz); `/Volumes/Models/dyf_brain_1m/` (brain)

### Downstream Application: sec10quant (2026-04-28)
Applied PH-on-DYF (Layer 7) to sec10quant's 27-feature SEC 10-Q text
signals. Walk-forward 2017-2025: pooled Q5−Q1 = +2.34%/qtr (default
config), recommended config 300 landmarks/0.05 threshold/decile bins
lifts to **+4.50%/qtr with 9/9 positive years**. Multivariate OLS:
PH coefficient survives all controls (composite, bullish, bearish,
n_sentences, year FE) at +0.96%/σ, t=8.6. Audit found
`composite_score` is empirically inverted (high bullish-tone language
→ underperforms by 2.67%/qtr) — language-hype reversal. 2022 weakness
diagnosed as direction-flip (51.7% of cycles flipped vs 14-19% in
neighbors), not detection failure (topology coverage unchanged).
Full memo: [project_sec10quant_ph_alpha.md](~/.claude/projects/-Users-jdonaldson-Projects-dyf/memory/project_sec10quant_ph_alpha.md)
Scripts: `/tmp/sec10q_*.py` (7 informal scripts, reproducible).

### Open Threads (in priority order if resumed)
1. **Real music dataset** (FMA-small, GTZAN) — synthesized was clean;
   real audio might have different topology
2. **Spatial transcriptomics** (Visium HD / Xenium) — natural biology
   extension, biggest scientific impact
3. **Adult mouse brain** vs E18 — predict mostly homogeneous, few
   cycles
4. **Cancer scRNA-seq atlas** — clinical impact
5. **Formal SCC-condensation DAG output** — discussed but not built;
   Tarjan + condensation + super-nodes annotated by internal topology
6. **CellRank/PAGA/scvelo benchmark comparison** — direct head-to-head
7. **Per-domain threshold calibration** — anisotropy gate 0.18, PH
   threshold 20% are anecdotal cross-domain

### sec10quant follow-ups (if returning to that downstream)
1. **Regime-conditional directions** — store two cycle directions
   keyed by VIX/yield-curve/growth-value regime; gate at scoring time.
   Real fix for unfixable Q1 2022 (rate-shock peak Q5−Q1 = −2.76%).
2. **Sector / size-tilt audit** — check if +4.50% spread comes from
   concentrations; if so, sector-neutralize.
3. **Live integration** — `compute_ph_signal.py` updating
   `prototype_signals.parquet` with `ph_score` column. Use recommended
   config (300 landmarks / 0.05 threshold / decile L/S).
4. **Cross-asset replication** — apply same pipeline to 10-K filings
   to test 10-Q specificity.

### Reference Pointers
- All experiments use `/Volumes/Models/dyf_brain_1m/pca50.npy` + `scanpy_leiden_labels.npy` + `dyf_leiden_labels.npy`
- Marker probes use h5 with `min_genes=200 + min_cells=3 + pct_mt<15` → 1,290,055 cells (truncate to 1,290,029)
- 20NG MiniLM cached at `dyf/.embeddings_cache/twenty_ng_minilm.npz`
- Brain spectral diagnostic results: `/tmp/nystrom_spectral_brain.py`, `/tmp/plateau_diagnostic.py`
- All findings consolidated in `~/.claude/projects/.../memory/spectral_dyf_composition.md` (~1100 lines)

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

