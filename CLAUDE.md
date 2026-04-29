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

### ⚠ Layer 7 (PH) falsified + salvaged 2026-04-28/29

**Adult mouse brain falsification arc** — TMS Droplet didn't contain
brain; pivoted to multi-tissue panel + cell-cycle gene module test +
QC-filtering test. Falsification + salvage via QC dependency.

**Falsified — cycle COUNT is not topology:**
Real cycle count ≈ within-cluster-shuffle null (the methodologically
correct null). PH count measures clustering, not topology beyond it.
Original column-shuffle finding (real << shuffled, z = −5 avg) was
artifact of wrong null — column shuffle destroyed BOTH clustering and
topology, making the test trivial.

**Survives in three refined claims:**
1. **Vertex content with ≥2 dominant cell types = lineage relationships**
   (Marrow myeloid lineage, Spleen erythroblastic islands, NK/NKT pair).
   50/50 split between single-type clustering artifacts and real lineage
   spans. Filter for ≥2 types downstream.
2. **Pooled cycling cells produce a detectable cell-cycle ring** (after
   filtering by Tirosh S/G2M gene module score). Persistence ~3 after
   QC. Single-cluster too small (promonocyte n=231, 99.6% cycling →
   0 cycles).
3. **CRITICAL: Layer 5 (reference modules) MUST precede Layer 7 (PH).**
   Stress/death/UPR/HSP signatures inflate ring persistence (7.49 raw →
   3.16 after removing top 10% stressed cells). Stress-only cells alone
   produce no cycles — they sit at G2/M ring periphery and inflate its
   diameter. The diagnostic stack's stated architecture is empirically
   required, not optional.

**Memo:** [project_ph_layer_falsification.md](~/.claude/projects/-Users-jdonaldson-Projects-dyf/memory/project_ph_layer_falsification.md) (317 lines, full investigation).

**Synthesis updated:** [synthesis_diagnostic_stack.md](~/.claude/projects/-Users-jdonaldson-Projects-dyf/memory/synthesis_diagnostic_stack.md) Layer 7 caveat block reflects the salvage path + QC dependency.

**Cross-tissue replication (2026-04-29):**
The lineage-span signal is **tissue-specific**, correlating with active
ongoing lineage flow rather than cell-type count:

| Tissue | Δh (real − WCS) | Lineage flow status |
|---|---:|---|
| Marrow | +0.36 ★ | hematopoiesis lifelong |
| **Brain_Non-Myeloid 3mo** | **+0.46 ★** | **neurogenesis + OPC→oligo + endothelial** |
| Limb_Muscle | +0.17 ◐ | satellite cells (small) |
| Liver | −0.10 ✗ | terminal hepatocytes |
| Lung | −0.35 ◐ | terminal alveolar/airway (REVERSED) |

Adult brain rank-2 cycle (the partial rescue of the original
"vascular compartment" claim) spans 6 types: astrocyte + brain pericyte +
endothelial + oligodendrocyte + OPC + neuronal stem cell. The cycle
visits the vascular compartment as part of a broader glial+vascular+
stem-cell loop. Adult brain showed **no cell-cycle ring** even in the
cycling subset — too few cells and too diverse identities.

**Shipped this arc:**
- `score_stress_modules`, `qc_filter_mask`, `cycle_lineage_spans` in
  `docs/gallery/_gallery.py` (mechanically encode Layer 5 → Layer 7)
- 15-test pytest suite at `docs/gallery/test_gallery_ph.py`
- Worked-example code listing in `diagnostic-stack.qmd` (rendered)
- Universal lesson: null-design rule promoted to `~/.claude/CLAUDE.md`
  Signal Pipeline Auditing section

**Open threads (in priority order):**
1. **TCC-gated E18 re-validation** — apply new framework. Predict:
   detectable cell-cycle ring (E18 has 70% cycling progenitors,
   should pool cleanly across cell types unlike adult brain's 23%);
   stronger lineage-span signal than adult brain's +0.46.
   **Still blocked on TCC.**
2. **Cross-cycle cluster comparison** for the brain rank-2 6-type
   cycle — does it trace known glial-vascular-stem-cell developmental
   relationships, or generic geometric loops? Inspect cocycle reps.
3. **Other dyf-core threads** (real music, spatial transcriptomics,
   SCC-DAG, threshold calibration, CellRank/PAGA benchmark) — PH
   falsification arc is at clean closure.

**Scripts:** `~/Projects/dyf/data/adult_brain/` (gitignored) — TMS
Droplet h5ad (8.2 GB), TMS FACS h5ad (4.8 GB, has brain), and ten
analysis scripts spanning the multi-test arc.

**TCC issue blocking E18 re-validation:** ghostty's FDA grant doesn't
flow through to subprocesses writing to `/Volumes/Models`. Sandbox
config update + full session restart didn't fix it. To resolve:
remove ghostty from FDA list, quit, re-add (TCC keys by code-signing
identity; updates can break grants).

### Recent commits (2026-04-28)
4 commits landing the diagnostic-stack consolidation:
- `761e347` docs(session): resume context — sec10quant downstream
- `0d3c385` docs(gallery): index — listing + parameter-free honesty
- `65518ed` feat(gallery): diagnostic-stack notebook — 7-layer stack
- `75d8c04` feat(gallery): _gallery.py primitives — auto-tune + diagnostic

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

