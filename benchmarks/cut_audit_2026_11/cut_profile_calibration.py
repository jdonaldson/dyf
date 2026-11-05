"""Calibrate the cut-profile thresholds on known geometry (d=50 isotropic Gaussians,
n=20k per side unless stated).  Same profile() as community_cut_profile_full.py.
  two Gaussians at separation s*sigma, equal sizes and 1:4 sizes -> valley(s), c_side(s)
  one Gaussian cut at its mode (t=0)         -> expect both sides shoulder (c_side ~ 1)
  one Gaussian, flank cut at +1 sigma        -> expect the flank side shoulder, bulk not
  uniform line segment + Gaussian noise, cut mid -> continuum: valley ~ 1, both sides ~ 1
Three seeds each; reports mean (min-max).
"""

import sys

import numpy as np

sys.path.insert(0, "/Volumes/Models/dyf_bench_2026-11-04")
from community_cut_profile_ownmode import profile2  # noqa: E402

D, N = 50, 20000


def both_sides(X, lab):
    a = profile2(X, lab, 0, 1)
    b = profile2(X, lab, 1, 0)
    return a["valley"], a["c_side"], b["c_side"], a["own_offset"], b["own_offset"]


def rep(tag, rows):
    a = np.array(rows)
    m, lo, hi = a.mean(0), a.min(0), a.max(0)
    print(
        f"   {tag:28s} valley={m[0]:.2f} ({lo[0]:.2f}-{hi[0]:.2f})  "
        f"c_side={m[1]:.2f} ({lo[1]:.2f}-{hi[1]:.2f})  d_side={m[2]:.2f} ({lo[2]:.2f}-{hi[2]:.2f})  "
        f"own_offset c={m[3]:.2f} d={m[4]:.2f}"
    )


print("== two Gaussians, cut at min-misclassification")
for ratio in (1, 4):
    for s in (0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 6.0):
        rows = []
        for seed in range(3):
            rng = np.random.default_rng(seed)
            A = rng.normal(size=(N, D)).astype(np.float32)
            B = rng.normal(size=(N * ratio, D)).astype(np.float32)
            B[:, 0] += s
            X = np.vstack([A, B])
            lab = np.r_[np.zeros(N, int), np.ones(N * ratio, int)]
            rows.append(both_sides(X, lab))
        rep(f"sizes 1:{ratio} sep={s}sigma", rows)

print("== one Gaussian cut at its mode (slice through the mode)")
rows = []
for seed in range(3):
    X = np.random.default_rng(seed).normal(size=(2 * N, D)).astype(np.float32)
    rows.append(both_sides(X, (X[:, 0] > 0).astype(int)))
rep("mode cut", rows)

print("== one Gaussian, flank cut at +1 sigma (comm 0 = flank, comm 1 = bulk)")
rows = []
for seed in range(3):
    X = np.random.default_rng(seed).normal(size=(2 * N, D)).astype(np.float32)
    rows.append(both_sides(X, (X[:, 0] <= 1.0).astype(int)))
rep("flank cut", rows)

print("== continuum: uniform on a segment of length 10 sigma + isotropic noise, cut mid")
rows = []
for seed in range(3):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(2 * N, D)).astype(np.float32)
    X[:, 0] += rng.uniform(0, 10, size=2 * N)
    rows.append(both_sides(X, (X[:, 0] > 5).astype(int)))
rep("continuum mid cut", rows)
