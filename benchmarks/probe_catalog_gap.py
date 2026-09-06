"""Does gap detection actually fire on test_catalog.py's engineered hierarchy?

`test_gap_detected_with_engineered_data` builds a 3-level catalog whose depth-3
commodities are deliberately random (not near their parents), then asserts only
`isinstance(result.gap_detected, bool)` -- with a comment conceding it cannot
guarantee the gap fires. If the engineered data does reliably trigger detection,
the test should assert it; if it does not, either the fixture or the detector is
the problem, and the test name is a lie either way.

Runs the fixture over several seeds to separate "works" from "works at seed 42".
"""

from __future__ import annotations

import sys

import numpy as np

sys.path.insert(0, "tests")
from test_catalog import _make_config  # noqa: E402

from dyf import CatalogSpace  # noqa: E402
from dyf.categorical import CategoryGraph  # noqa: E402


def build(seed: int, noise: float):
    """Replicate the fixture; `noise` scales how far commodities sit from parents."""
    rng = np.random.default_rng(seed)
    dim = 32
    edges, node_ids, node_names, embeddings = [], [], [], []

    seg_centers = rng.standard_normal((2, dim)).astype(np.float32)
    seg_centers /= np.linalg.norm(seg_centers, axis=1, keepdims=True)

    for si in range(2):
        sid = f"S{si}"
        edges.append(("_root_", sid, 0.5))
        node_ids.append(sid)
        node_names.append(f"Segment_{si}")
        embeddings.append(seg_centers[si])
        for ci in range(3):
            cid = f"CL{si}_{ci}"
            edges.append((sid, cid, 1.0 / 3))
            ce = seg_centers[si] + rng.standard_normal(dim).astype(np.float32) * 0.1
            ce /= np.linalg.norm(ce)
            node_ids.append(cid)
            node_names.append(f"Class_{si}_{ci}")
            embeddings.append(ce)
            for ki in range(5):
                kid = f"K{si}_{ci}_{ki}"
                edges.append((cid, kid, 0.2))
                if noise >= 1.0:
                    ke = rng.standard_normal(dim).astype(np.float32)
                else:
                    ke = ce + rng.standard_normal(dim).astype(np.float32) * noise
                ke /= np.linalg.norm(ke)
                node_ids.append(kid)
                node_names.append(f"Comm_{si}_{ci}_{ki}")
                embeddings.append(ke)

    graph = CategoryGraph.from_edges(edges)
    config = _make_config(
        "gapped",
        graph,
        np.array(embeddings, dtype=np.float32),
        np.array(node_ids, dtype=str),
        np.array(node_names, dtype=str),
    )
    space = CatalogSpace()
    space.add_catalog(config).fit()

    query = seg_centers[0] + rng.standard_normal(dim).astype(np.float32) * 0.05
    query = (query / np.linalg.norm(query)).astype(np.float32)
    return space.match_single("gapped", query)


def main() -> None:
    print("Engineered gap (commodities random, noise=1.0) — the fixture as written")
    print(f"{'seed':>5} {'gap_detected':>13} {'gap_score':>10} {'depth_sims'}")
    fired = 0
    for seed in (42, 0, 1, 7, 123, 2024, 31337, 99):
        r = build(seed, 1.0)
        sims = getattr(r, "depth_similarities", None)
        s = " ".join(f"{v:.3f}" for v in np.asarray(sims)) if sims is not None else "n/a"
        fired += bool(r.gap_detected)
        print(f"{seed:>5} {str(r.gap_detected):>13} {r.gap_score:>10.4f} {s}")
    print(f"  fired {fired}/8 seeds\n")

    print("Control: commodities NEAR their parents (noise=0.1) — gap should NOT fire")
    print(f"{'seed':>5} {'gap_detected':>13} {'gap_score':>10}")
    fired_ctrl = 0
    for seed in (42, 0, 1, 7, 123, 2024, 31337, 99):
        r = build(seed, 0.1)
        fired_ctrl += bool(r.gap_detected)
        print(f"{seed:>5} {str(r.gap_detected):>13} {r.gap_score:>10.4f}")
    print(f"  fired {fired_ctrl}/8 seeds")

    print("\nVerdict:")
    if fired == 8 and fired_ctrl == 0:
        print("  detector SEPARATES the two conditions -> assert gap_detected is True")
    elif fired >= 6 and fired_ctrl <= 2:
        print("  mostly separates -> assert on gap_score ordering, not the boolean")
    else:
        print("  does NOT separate -> the test name is unsupportable; assert the")
        print("  score ordering only, or fix the detector. Record which.")


if __name__ == "__main__":
    main()
