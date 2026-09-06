"""Leaf granularity trades header size against delta locality.

Finer leaves  -> fewer dirty per step (better dedup) but far more centroids (bigger header).
Coarser leaves -> tiny header but nearly every batch dirties.

Bytes are modelled from filings.dyf measurements: 1898 B/row of Arrow payload
((total - fb) / rows), centroids on every node, hyperplanes on internal nodes only.

Result (2026-08-01): best naive storage (10 independent snapshots) = 4457 MB; best
delta-encoded = 1495 MB (2.98x), or 1903 MB (2.34x) at the granularity whose search
quality was actually validated in sec_probe_sweep.py.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sec_seqlib as S  # noqa: E402

DIM = 768
EMB_B = DIM * 2  # float16
META_B = 362  # measured: (502606080 - 67553712) / 229243 - 1536
F32 = 4
N_SNAPSHOTS = 10
CONFIGS = [(3, 4), (3, 16), (4, 64), (4, 32), (4, 16), (4, 4), (5, 16), (5, 4), (6, 4)]


def main():
    E, D, T, SEC, Q = S.load()
    base_idx = np.where(Q <= "2023Q4")[0]
    steps = [q for q in sorted(set(Q.tolist())) if q > "2023Q4"]
    step_idx = {q: np.where(np.equal(Q, q))[0] for q in steps}

    print(
        f"{'cfg':<22}{'leaves':>8}{'pts/lf':>8}{'hdr MB':>9}{'batch MB':>10}"
        f"{'full MB':>9}{'avg dirty%':>11}{'seq MB':>9}"
    )
    print("-" * 86)
    rows = []
    for md, ml in CONFIGS:
        flat = S.build(E, base_idx, max_depth=md, min_leaf=ml)
        NL = S.n_leaves(flat)
        n_int = len(flat) - NL
        nb = max((n["hp"].shape[0] for n in flat if n["hp"] is not None), default=4)
        a_all, _ = S.route(E, flat)

        cen_B = len(flat) * DIM * F32
        hp_B = n_int * nb * DIM * F32
        batch_B = len(E) * (EMB_B + META_B)
        full_B = cen_B + hp_B + batch_B

        seq_B = full_B
        dirties = []
        for qtr in steps:
            frac = len(np.unique(a_all[step_idx[qtr]])) / NL
            dirties.append(frac)
            # delta frame: dirty batches + dirty centroids; hyperplanes shared = free
            seq_B += frac * batch_B + min(1.0, frac * 1.09) * cen_B
        rows.append((f"depth={md},minleaf={ml}", NL, full_B, seq_B))
        print(
            f"{f'depth={md},minleaf={ml}':<22}{NL:>8}{len(base_idx) / NL:>8.1f}"
            f"{(cen_B + hp_B) / 1e6:>9.1f}{batch_B / 1e6:>10.1f}{full_B / 1e6:>9.1f}"
            f"{100 * np.mean(dirties):>10.1f}%{seq_B / 1e6:>9.1f}",
            flush=True,
        )

    naive = N_SNAPSHOTS * min(r[2] for r in rows)
    best = min(rows, key=lambda r: r[3])
    print(f"\nbest naive ({N_SNAPSHOTS} independent snapshots): {naive / 1e6:.0f} MB")
    print(f"best delta-encoded: {best[3] / 1e6:.0f} MB via {best[0]} ({best[1]} leaves)  -> {naive / best[3]:.2f}x")
    print(
        "NOTE: the byte-optimal config is ~3 pts/leaf and was NOT validated for search "
        "quality. The validated config is depth=4,minleaf=16."
    )


if __name__ == "__main__":
    main()
