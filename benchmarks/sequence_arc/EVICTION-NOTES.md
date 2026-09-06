# Eviction, coverage decay, and why the trigger is JS

Extends `POSTGRES_NOTES.md`, whose every measurement is **append-only** and says so: under append,
old points stay and keep anchoring coverage, so "+64% growth costs ~0.01 recall" is an *optimistic
bound*. Under eviction, coverage can decay — the points that justified a hyperplane can leave. The
tombstone-only policy was decided without measuring that decay. These three probes measure it.

Scripts: `sec_evict.py` (decay), `sec_js_anatomy.py` (what JS is made of), `sec_cell_alarm.py`
(the falsification test). Corpus: 229,243 SEC 10-Q sections, 768d, same as the rest of the arc.

## The design constraint, first

**Do not slide a window over SEC quarters.** `sec_shift.py` measured the temporal condition at
+0.010 — SEC is near-stationary, so a naive sliding window measures nothing and would report
"eviction is free" for the wrong reason. What discriminates is a **composition schedule** with a
fixed-mix control at *identical turnover*:

| arm | eviction volume | incoming mix |
|---|---|---|
| `fixed` | 5,000/step | == fit-time mix (pure turnover control) |
| `drift` | 5,000/step | ramps toward 80% `mda`, starving `risk_factors` to 2% |

60,000-point live window, 12 steps = one complete turnover. Section type is the axis known to break
coverage (withholding whole section types cost +0.119, ~8σ; disjoint companies only +0.017).

## Result 1 — recall did not decay, but this is a floor, not a green light

| | fixed | drift |
|---|---|---|
| recall gap vs rebuild @100% turnover | +0.0042 | −0.0007 |
| gap ever > 0.01 | **never** | **never** |
| JS depth-1 @100% | 0.0030 | **0.0547** |
| unseen-bucket rate @100% | 0.0147 | 0.0139 |

The gap oscillates in ±0.006 and flips sign — frozen beats rebuild as often as it loses. That is
noise around zero, not decay.

⚠️ **The `NEVER` row is a non-result.** Recall sat at 0.97 for all 26 measurements, so the
experiment had no room to show degradation. The probe was set at 100 leaves (0.902 on a fit-set
sweep) specifically to leave headroom and *still* saturated once the window went live — calibrate
the probe against a **drifted** window, not the fit set. Defensible claim: **≤1 window turnover at
≤0.055 JS costs nothing measurable at this probe depth.** Not "eviction is free."

Also: the drift I induced reaches only **5.5% of the JS bound** (0.0547 of 1.0 bits) against the
arc's collapse condition at 0.26. I never drove the index near broken.

**Centroid refresh is the real lever here** — recomputing leaf centroids from live members beats
stale centroids by **+0.048 (fixed) / +0.061 (drift)** at full turnover, an order of magnitude
larger than any gap being hunted. Under eviction, refresh the delta frame.

## Result 2 — unseen-rate is a turnover odometer; JS is composition-aware

The two metrics answer different questions, and only one is usable as a trigger.

| | vs turnover | vs mix-distance | arm separation @100% |
|---|---|---|---|
| unseen-bucket rate | **+0.9999** | +0.9737 | **0.95×** (drift reads *lower*) |
| JS depth-1 | +0.8739 | **+0.9584** | **18.2×** |

Unseen-rate's arm ratio stays in 0.95–1.01× across **all 13 steps** — it never separates. It counts
fallback events, which scale with arrival volume regardless of kind. JS compares *distributional
shape* against fit-time occupancy, and turnover alone does not change shape.

Note JS also correlates +0.87 with plain turnover, so it is not a pure composition signal. What
makes it usable is that its **magnitude** stays trivial under pure turnover (0.0030, ~14× floor) and
goes 18× larger under composition change, while unseen-rate's magnitude is indifferent.

⚠️ **Neither metric predicted recall here** (JS vs gap: −0.098) — because nothing lost recall. JS is
calibrated to *composition*, not yet to *retrieval quality*. That link is unproven.

## Result 3 — JS anatomy: floor, scale, concentration, depth

**Noise floor** (split-half of the same distribution, 8 seeds), the thing a threshold needs:

| window | floor |
|---|---|
| 2,000 | 0.002853 |
| 5,000 | 0.001125 |
| 15,000 | 0.000366 |
| 30,000 | 0.000155 |

Scales ~1/n. **A JS threshold without a stated window size is meaningless** — at a 2,000-point
window the floor (0.0029) swallows the fixed arm's healthy reading (0.0030). At 60k the drift
endpoint sits **255× above the floor**; the fixed control sits ~14× above it, so fixed's 0.0030 is
*real small drift from turnover*, not noise.

**Depth profile under eviction** — the arc's depth-1 preference, reproduced on a new condition and
with signal and noise measured *separately*:

| depth | cells | median n | JS drift | JS noise | margin |
|---|---|---|---|---|---|
| **1** | 16 | 1,509 | 0.0547 | 0.000192 | **285.3×** |
| 2 | 231 | 65 | 0.0664 | 0.002948 | 22.5× |
| 3 | 1,981 | 9 | 0.0931 | 0.027055 | 3.4× |
| 4 | 6,402 | 5 | 0.1393 | 0.092312 | 1.5× |

**Signal is monotonically stronger at depth; the margin collapses anyway** because noise grows
faster. Coarse wins on sampling noise, not signal strength — now demonstrated rather than inferred.

## Result 4 — the falsification that failed: per-cell delta is NOT enough

JS is **concentrated**: top 3 of 16 cells carry 75.8% of it (cell 8 +17.7pp, cell 9 −14.2pp — a mass
transfer between two regions, not a reshaping). That suggests a cheap max-per-cell-delta alarm might
do the same job. **It does not.** All four detectors from the same 16 counters:

| detector | floor | fixed | drift | drift/floor | **drift/fixed** |
|---|---|---|---|---|---|
| **js** | 0.000214 | 0.0030 | 0.0547 | **255×** | **18.2×** |
| max_abs_delta | 0.004888 | 0.0213 | 0.1771 | 36× | 8.3× |
| tv (L1/2) | 0.010533 | 0.0484 | 0.2541 | 24× | 5.2× |
| max_rel_delta | 0.280515 | 0.4801 | 1.0094 | 4× | 2.1× |

Three reasons concentration did not imply redundancy:

1. **Max-abs reads one cell and discards fifteen.** Even at 75.8% concentration, the residual 24% is
   real signal *absent from the fixed arm*, so including it improves the ratio. One cell captures
   most of the magnitude; the tail carries most of the discrimination.
2. **The floors diverge more than the signals do** — max-abs's floor is **23× higher** than JS's. A
   single cell's share is one noisy bin; JS averages 16 against a midpoint. Max-abs pays twice.
   Same shape as the dirty-leaf-fraction trap: the bigger number is the worse detector.
3. **Relative delta is actively dangerous.** Cell 14 holds 106 points; it reads `rel d = 1.01` on
   drift vs **0.39 on the healthy control**. Depth-1 occupancy spans 106→19,614 (**185×**), so no
   single relative threshold works across cells — any threshold catching drift fires on turnover.

## The policy

- **Trigger on JS at depth 1.** 16 integer counters at ingest, compared to a stored fit-time
  histogram. No queries, no ground truth. Threshold in 0.01–0.05 **at a 60k window** — restate it
  against the floor for any other window size.
- **Diagnose with per-cell deltas.** Once fired, rank cells by contribution: "cell 8 went 19.8% →
  37.5%" names *where* the drift is and is directly actionable. This is where the user's intuition
  holds — but it is the diagnostic half, not the trigger.
- **Refresh leaf centroids from live members** under eviction; worth more than everything else here.
- **Never call `Tree::remove`** (unchanged, `POSTGRES_NOTES.md`). Not exported to Python today;
  keep it that way and add `tombstone()` instead.

Choosing JS as the trigger also survives a case this data lacks: if drift were **broad** rather than
concentrated, max-abs would degrade further while JS would hold. It costs nothing and is robust to a
drift shape we have not tested.

## What is still owed

1. **No recall linkage.** The threshold that matters is where recall breaks; drift never got past
   5.5% of the JS bound. Push toward 0.26 with a probe low enough (~0.75–0.85 baseline) to see loss.
2. **Only one drift shape** — proportions of existing strata. Retiring a *category* creates a
   genuinely absent region, closer to the section-withholding condition that cost +0.119.
3. **Tombstone/JS interaction unmodelled.** A tombstoned point still occupies its cell unless the
   counter excludes it, so a tombstone-heavy index may drift without JS noticing. Decide whether
   occupancy counts live members only.
4. This is an **independent second breaking condition** for the arc's n=1 weak spot — different
   mechanism, same verdict. Weaker than the original (18× vs 57.6×, no recall anchor) but real.
