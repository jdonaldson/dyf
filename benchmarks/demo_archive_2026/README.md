# Archived demo experiments (moved 2026-11-06)

One-off research scripts that lived in `demo/` from January to April 2026. They still import
against the current library (checked by resolving every `dyf`/`dyf_rs` import on
2026-11-06) but none is documented in `demo/README.md`, none is run by anything, and `demo/`
now holds only the documented demos, the data builders and `puppet/`. Moved rather than
deleted so the paths stay visible beside the other probes; ruff-formatted on the move, no
behaviour change.

| script | what it explored |
|---|---|
| `clustering_experiments.py` | DensityClassifier Gini / similarity sweeps |
| `pca_tree_knn_umap.py`, `pca_tree_multi_address.py` | PCA-tree kNN + UMAP layouts; multi-address routing (the second imports the first) |
| `edge_animation_test.py` | edge-bundle animation prototype (writes an HTML next to itself) |
| `dyf_tree_sankey.py` | Sankey view of tree cuts via `dyf_rs.DensityClassifier` |
| `krapivin_explainer.py` | Bokeh explainer app |
| `wiki_umap_birch.py`, `wiki_umap_birch_eval.py` | BIRCH-era Wikipedia clustering and its evaluation |

Removed outright in the same cleanup (git history has them): six scripts importing the deleted
`dyf.pca_tree` module, the pre-split `dyfviz.py` monolith with `patch_overlay.py` and the tour
narration JSONs (superseded by the `dyfviz` package), the five viewer files now canonical in
`dyfviz/src/dyfviz/assets/`, and the ROG live-control trio (`rog_mcp.py`, `rog_demo.py`,
`rog_preprocess.py`) whose MCP entry pointed at a `.dyf` that no longer existed.
