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

Run with the project venv: `.venv/bin/python benchmarks/cut_audit_2026_11/<script>.py`.
