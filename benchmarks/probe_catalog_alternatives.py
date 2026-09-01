"""Do `match_single` alternatives actually come from different parents?

`test_alternatives_from_different_parents` is the one test in the suite that
asserts NOTHING — its body ends in `pass  # structure verified, no crash`, inside
two nested `if`s. It is named for a property it never checks.

Before asserting that property, measure it: diversification could be real, absent,
or partial, and each calls for a different assertion.
"""

from __future__ import annotations

import sys

import numpy as np

sys.path.insert(0, "tests")
from test_catalog import _make_config, _make_hierarchy  # noqa: E402

from dyf import CatalogSpace  # noqa: E402


def main() -> None:
    print(f"{'seed':>5} {'primary':>10} {'n_alts':>7} {'alt parents differ':>19}  alternatives")
    diff_total = same_total = 0
    for seed in (42, 0, 1, 7, 123):
        rng = np.random.default_rng(seed)
        graph, embs, ids, names = _make_hierarchy(rng, n_parents=4, children_per_parent=5)
        space = CatalogSpace()
        space.add_catalog(_make_config("test", graph, embs, ids, names)).fit()

        child_idx = list(ids).index("C00_00")
        result = space.match_single("test", embs[child_idx].copy(), top_k=5)

        primary_parents = set(graph.get_parents(result.node_id))
        differ = same = 0
        alt_ids = []
        for alt_id, _n, _s in result.alternatives:
            alt_ids.append(alt_id)
            if set(graph.get_parents(alt_id)) & primary_parents:
                same += 1
            else:
                differ += 1
        diff_total += differ
        same_total += same
        print(
            f"{seed:>5} {result.node_id:>10} {len(result.alternatives):>7} "
            f"{f'{differ} differ / {same} same':>19}  {alt_ids}"
        )

    print(f"\ntotals: {diff_total} from a different parent, {same_total} from the same parent")
    if same_total == 0 and diff_total > 0:
        print("-> diversification is STRICT: assert every alternative has a different parent")
    elif diff_total > 0:
        print("-> diversification is PARTIAL: assert at least one differs, not all")
    else:
        print("-> NO diversification: the test name describes behaviour that does not happen")


if __name__ == "__main__":
    main()
