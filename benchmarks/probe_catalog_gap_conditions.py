"""Which of `_detect_gap`'s five absolute conditions actually blocks?

`CatalogSpace._detect_gap` fires only when FIVE absolute constants hold at once:

    parent_entropy   < 0.5
    child_entropy    > 0.7
    entropy_increase > 0.3
    similarity_drop  > 0.1
    child_sim        < 0.8

A conjunction of five absolute thresholds on quantities whose scale is corpus
dependent is the KNOWN_ISSUES #5 bug class at its most extreme -- each condition
multiplies the chance of never firing. This prints the observed per-depth entropy
and similarity so the blocking condition is named rather than guessed.
"""

from __future__ import annotations

import sys

sys.path.insert(0, "benchmarks")
from probe_catalog_gap import build  # noqa: E402


def main() -> None:
    import dyf.catalog as cat

    captured: list[tuple[dict, dict]] = []
    original = cat.CatalogSpace._detect_gap

    def spy(self, depth_entropy, depth_best_sim):
        captured.append((dict(depth_entropy), dict(depth_best_sim)))
        return original(self, depth_entropy, depth_best_sim)

    cat.CatalogSpace._detect_gap = spy
    try:
        for label, noise in (("engineered gap (noise=1.0)", 1.0), ("control (noise=0.1)", 0.1)):
            captured.clear()
            build(42, noise)
            print(f"\n{label}")
            if not captured:
                print("  _detect_gap was never called")
                continue
            ent, sim = captured[-1]
            depths = sorted(ent)
            print(f"  depths present: {depths}")
            print(f"  {'depth':>6} {'entropy':>9} {'best_sim':>9}")
            for d in depths:
                print(f"  {d:>6} {ent[d]:>9.4f} {sim.get(d, float('nan')):>9.4f}")

            if len(depths) < 2:
                print("  -> fewer than 2 depths, _detect_gap returns early")
                continue
            print(
                f"\n  {'pair':>8} {'p_ent<0.5':>10} {'c_ent>0.7':>10} {'inc>0.3':>9} {'drop>0.1':>9} {'c_sim<0.8':>10}"
            )
            for i in range(len(depths) - 1):
                p, c = depths[i], depths[i + 1]
                pe, ce = ent[p], ent[c]
                ps, cs = sim.get(p, 0.0), sim.get(c, 0.0)
                conds = {
                    "p_ent<0.5": (pe < 0.5, pe),
                    "c_ent>0.7": (ce > 0.7, ce),
                    "inc>0.3": (ce - pe > 0.3, ce - pe),
                    "drop>0.1": (ps - cs > 0.1, ps - cs),
                    "c_sim<0.8": (cs < 0.8, cs),
                }
                cells = " ".join(f"{('Y' if ok else 'n') + f'({v:.2f})':>10}" for ok, v in conds.values())
                print(f"  {f'{p}->{c}':>8} {cells}")
                blockers = [k for k, (ok, _) in conds.items() if not ok]
                print(f"           blocked by: {blockers or 'nothing — would fire'}")
    finally:
        cat.CatalogSpace._detect_gap = original


if __name__ == "__main__":
    main()
