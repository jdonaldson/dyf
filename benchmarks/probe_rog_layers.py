"""What makes build_rog_ontology produce more than one layer?

`test_rog_layers_decrease_threshold` guards its only assertion behind
`if len(result.layers) > 1:`, and on its fixture ROG returns exactly ONE layer —
coverage 0.960 clears the default target_coverage=0.95 on the first pass, so
recursion never happens and the test has never run its own body.

One layer is CORRECT behaviour there. The test needs a condition that forces
recursion. Find the cheapest one, and check the thresholds actually decrease.

Also dumps the real field names on UnifiedOntologyResult, because reading a
nonexistent attribute and believing the getattr default is a recurring way to
manufacture a false positive in this repo.
"""

from __future__ import annotations

import numpy as np
from probe_guarded_tests import clustered_embeddings, sample_embeddings


def main() -> None:
    from dyf import build_rog_ontology, build_unified_ontology

    clus = clustered_embeddings()
    samp = sample_embeddings()

    print("build_rog_ontology — conditions that force more than one layer")
    print(f"{'fixture':<12} {'target_cov':>10} {'layers':>7} {'coverage':>9}  thresholds")
    conditions = [
        ("clustered", clus, {}),
        ("clustered", clus, {"target_coverage": 0.99}),
        ("clustered", clus, {"target_coverage": 0.999}),
        ("clustered", clus, {"target_coverage": 1.0}),
        ("sample", samp, {}),
        ("sample", samp, {"target_coverage": 0.99}),
        ("sample", samp, {"target_coverage": 1.0}),
    ]
    for label, emb, kw in conditions:
        r = build_rog_ontology(emb, verbose=False, **kw)
        th = [round(x.similarity_threshold, 4) for x in r.layers]
        tc = kw.get("target_coverage", 0.95)
        print(f"{label:<12} {tc:>10} {len(r.layers):>7} {r.total_coverage:>9.4f}  {th}")

    print("\nUnifiedOntologyResult — actual fields (not guessed)")
    uni = build_unified_ontology(clus, verbose=False)
    for k, v in vars(uni).items():
        if isinstance(v, np.ndarray):
            desc = f"ndarray len={len(v)}"
        elif isinstance(v, (list, tuple, set, dict)):
            desc = f"{type(v).__name__} len={len(v)}"
        else:
            desc = repr(v)[:60]
        print(f"  {k:<24} {desc}")


if __name__ == "__main__":
    main()
