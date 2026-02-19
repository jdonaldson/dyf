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