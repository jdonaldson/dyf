# DYF Project Notes

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
- When in doubt, spend 10 minutes on a sanity check before spending 2 hours on a solution

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

## 🔄 RESUME CONTEXT - DELETE AFTER READING

**⚠️ INSTRUCTIONS FOR CLAUDE**:
- Read this section when starting a new session
- Complete any pending tasks listed here
- Delete this entire section after processing
- Continue with the conversation

### Current Status
dyfviz.py cleanup complete (4 tasks). Embedding model benchmark complete. File triage partially done.

### What Was Built

**dyfviz.py cleanup (3 commits: 54b920b, 3bc224a, 48bea0c):**
1. Removed dead `golden_ratio_color_map` from dyfviz.py
2. Extracted 2,920-line JS blob from Python f-string → `demo/pydeck_overlay.js` (dyfviz.py shrank from ~6K to ~3K lines)
3. Fixed 4 Pyright warnings (`.group()` on None, `min()` overload)
4. Created `src/dyf/colors.py` with spatial color functions; `dyf_enrich.py` stores color maps in .dyf metadata; `dyfviz.py` loads pre-stored maps with on-the-fly fallback

**Embedding model benchmark:**
- `embedding_model_comparison.md` — full analysis: Nomic v1.5 wins on GUDID (0.3721 avg purity), mxbai loses (-1.5%), Fisher is zero-effect at high cardinality
- `/tmp/embed_purity_benchmark.py` — benchmark script (not committed)
- `demo/purity_benchmark_results.json` — raw results (committed)
- Embedding caches in demo/ (`mxbai_5k_cache.npy`, `minilm_5k_cache.npy`) — not committed, should be gitignored

**Source modules and tests committed (3bc224a):**
- `src/dyf/fisher.py`, `pipeline.py`, `provenance.py`
- 7 test files in `tests/`

**Demo scripts committed (48bea0c):**
- `cifar_embeddings.py`, `patch_overlay.py`, `pca_tree_knn_umap.py`, `pubmed_embeddings.py`, `rog_demo.py`

**Trashed:** `demo/rog_3d.py` (pickle-dependent), `demo/wiki_viz_original.py` (pre-dyfviz era)

### Remaining Uncommitted/Untracked
- **Modified but uncommitted**: `.claude/settings.local.json`, `.mcp.json`, `CLAUDE.md`, `demo/build_dyf_indexes.py`, `demo/energy_label_cache.json`, `demo/rog_mcp.py`, `demo/rog_preprocess.py`, `src/dyf/__init__.py`, `src/dyf/lazy_index.py`
- **Untracked exploration scripts** (left intentionally): `demo/clustering_experiments.py`, `demo/dyf_tree_sankey.py`, `demo/pca_tree_multi_address.py`, `demo/wiki_umap_birch.py`, `demo/wiki_umap_birch_eval.py`
- **Untracked infra**: `.envrc`, `pyrightconfig.json`, `uv.lock`, `docs/.gitignore`, `benchmarks/`
- **Embedding caches** (`demo/mxbai_5k_cache.npy`, `demo/minilm_5k_cache.npy`) — should be gitignored

### Context to Remember
- `__init__.py` was modified externally to add CatalogSpace imports — do not revert
- CatalogSpace module (commit 8c2a167) is complete; next phase would be shortorder integration at `/Users/jdonaldson/Projects/work/curvo/shortorder/`
- Pre-commit hook rejects `Co-Authored-By:.*[Cc]laude` — omit from commit messages
- 7 pre-existing Pyright warnings remain in dyfviz.py (not from our changes)
