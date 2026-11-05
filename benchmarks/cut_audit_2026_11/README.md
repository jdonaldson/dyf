# Cut audit, 2026-11-04/05

One-off probes behind `KNOWN_ISSUES.md` issue 14 and memory
`project_community_conductance_bench_2026_11`. Copied verbatim from
`/Volumes/Models/dyf_bench_2026-11-04/`; the scripts hard-code that directory and read the
seven-dataset bench in `/Volumes/Models/dyf_bench_2026-09-27/` (datasets, September dyf and
Leiden labels). The cached point kNN graphs (`knn15_<name>.npy`, up to 77 MB) are not copied.

Question answered, in order:

| script | question | result file |
|---|---|---|
| `wedge_radius_shapes_probe.py`, `wedge_radius_bench.py`, `wedge_radius_kmatched_tms.py` | can the row norm inside each leaf recover rings? | circles yes, real data no (`wedge_radius_*results.json`) |
| `community_conductance_bench.py`, `community_slice_diagnostic.py` | does dyf's final partition cut dense structure more than Leiden / k-means? | like k-means, 2–4× Leiden |
| `community_contact_proxies.py` | can anything free reproduce kNN contact? | centroid tie, ρ 0.71–0.95 |
| `community_cut_profile*.py`, `cut_profile_calibration.py` | density profile across a cut: valley / saddle / shoulder, calibrated on Gaussians | strict shoulders 0–4% over all pairs |
| `reassignment_effect.py` | what does `_reassign_points` do? | spherical k-means; moves 4–31% of points, ARI −0.13…+0.11, sign stable over seeds 42/1/2 |
| `leaf_reassignment_test.py` | would leaf-level or straggler-only reassignment do better? | no: leaf-level loses to the shipped point step everywhere it helps and cuts more edges; straggler arm moves zero leaves by construction |
| `hash_leaf_graph_test.py` | can the leaf graph come from the hash (boundary mass across each hyperplane) instead of centroid cosine? | no: loses at matched k on every dataset, −0.18 ARI on MNIST; AG News gain was community count |
| `centred_hash_tree_test.py`, `centred_hash_tree_kmatched.py` | does centring the hash threshold, `sign(w·(x−μ))`, help? | purer leaves, block ARI +0.05 on three sets at matched k, shipped output within ±0.03; shapes unchanged; not worth the format change |
| `leaf_graph_construction_test.py` | how much can any leaf graph buy? centroid cosine vs SNN vs spread-aware vs the full point kNN coarsened to leaves (ceiling) vs a 20k sample of it | ceiling adds +0.01…+0.06 at matched k; SNN/spread a wash; sampled counts fragment Louvain; the edges are not the bottleneck |
| `map_equation_test.py` (needs `uv run --with infomap`) | is the map equation (Infomap) a better objective than modularity on the leaf graph? | it is a fine-k prior: 49–90 modules; +0.06 shipped ARI on the 120-type atlases (mostly k), −0.2…−0.3 on 10-class images and 4-class text; not a default |
| `resolution_criterion_test.py` (needs `uv run --with infomap`) | can a label-free criterion pick Louvain's resolution? codelength minimum, edge-dropout stability, Laplacian eigengap | all worse than the default: mean regret 0.18 / 0.13 / 0.18 vs 0.08 for resolution 1; codelength is monotone, eigengap always says k 2–4 |
| `resolution_labels_and_prediction_strength.py` | how many labels pick the resolution, and does prediction strength through the frozen tree work label-free? | 20 labels beat the default, 50 beat knowing k (regret 0.016 vs 0.034); prediction strength never clears 0.7 and picks the coarsest level |
| `nested_louvain_test.py`, `nested_sweep_snap_test.py` | what does an exactly nested resolution profile cost? sweep snapped bottom-up vs intersected vs merged up from a fine base (centroid linkage, graph cut) vs recursive top-down | the sweep already nests 92–99%; snapping costs ≤0.02 ARI at the best level on 5/7 (fashion 0.04, brain 0.06); merging up loses 0.06–0.13 on images; top-down skips intermediate k; persistence does not predict purity |
| `per_split_valley_validation.py` | on the snapped profile, does the cut profile of each touching sibling pair sort class boundaries from continuum slices? | 1,113 pairs: AUC 0.66–0.97 on the cell atlases, 0.61–0.78 on brain, ≈0.5 on images; pooled 0.63 (valley) / 0.70 (min side); informative only where classes are density gaps |
| `point_continuousness_probe.py` | a per-point, multi-scale continuousness from the tree path on a depth-8 tree: box-counting local dimension, depth-wise margin ratios, largest radius step | local dimension predicts which points the re-cut moves (AUC 0.59–0.78, ≥0.65 on 4/7) and anticorrelates with leaf purity on all 7; margins null; step works on the atlases only; one seed |

Run with the project venv: `.venv/bin/python benchmarks/cut_audit_2026_11/<script>.py`.
