"""For the highest-conductance dyf communities: where do their out-edges go, and is the
destination the same ground-truth class?  A sliced region = out-edges concentrate on
ONE neighbour community and both sides share a majority label.  Uses the cached kNN
graphs from community_conductance_bench.py.
"""

import json
import sys
from pathlib import Path

import numpy as np

BENCH = Path("/Volumes/Models/dyf_bench_2026-09-27")
OUT = Path("/Volumes/Models/dyf_bench_2026-11-04")
KNN = 15
ORDER = ["mnist", "fashion_mnist", "cmu_mocap", "ag_news", "tms_facs", "tms_droplet", "brain_1m"]
TOP = 3


def main(names):
    out = {}
    for name in names:
        meta = json.loads((BENCH / "datasets.json").read_text())[name]
        y = np.load(meta.get("y_path", BENCH / f"{name}_y.npy"), allow_pickle=True)
        _, y = np.unique(y, return_inverse=True)
        J = np.load(OUT / f"knn{KNN}_{name}.npy")
        lab = np.load(BENCH / f"sC_labels_{name}_default_3_4_20.npy")
        _, lab = np.unique(lab, return_inverse=True)
        n, k = J.shape
        src = np.repeat(lab, k)
        dst = lab[J.ravel()]
        C = lab.max() + 1
        flow = np.zeros((C, C), dtype=np.int64)
        np.add.at(flow, (src, dst), 1)
        size = np.bincount(lab, minlength=C)
        out_edges = flow.sum(1) - np.diag(flow)
        phi = out_edges / np.maximum(np.minimum(k * size, k * n - k * size), 1)
        rows = []
        print(f"\n== {name}: dyf k={C}")
        for c in np.argsort(-phi)[:TOP]:
            dests = flow[c].copy()
            dests[c] = 0
            d = int(np.argmax(dests))
            share = dests[d] / max(out_edges[c], 1)
            maj_c = np.bincount(y[lab == c]).argmax()
            maj_d = np.bincount(y[lab == d]).argmax()
            pur_c = np.mean(y[lab == c] == maj_c)
            pur_d = np.mean(y[lab == d] == maj_d)
            # how much of the true class maj_c sits in c vs d
            cls = y == maj_c
            in_c, in_d = np.mean(lab[cls] == c), np.mean(lab[cls] == d)
            row = dict(
                community=int(c),
                size_frac=float(size[c] / n),
                phi=float(phi[c]),
                dest=int(d),
                dest_share=float(share),
                maj_label=int(maj_c),
                purity=float(pur_c),
                dest_maj_label=int(maj_d),
                dest_purity=float(pur_d),
                same_label=bool(maj_c == maj_d),
                class_in_c=float(in_c),
                class_in_dest=float(in_d),
            )
            rows.append(row)
            print(
                f"   comm {c:3d} ({size[c] / n:5.1%} of pts) phi={phi[c]:.3f}: {share:.0%} of out-edges -> comm {d:3d}; "
                f"majority label {maj_c} (purity {pur_c:.2f}) vs dest label {maj_d} (purity {pur_d:.2f}) "
                f"{'SAME' if maj_c == maj_d else 'different'}; class {maj_c} split {in_c:.0%} here / {in_d:.0%} there"
            )
        out[name] = rows
    (OUT / "community_slice_diagnostic_results.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main(sys.argv[1:] or ORDER)
