"""Do the `if` guards in the 11 all-guarded tests actually open?

`audit_test_assertions.py` now flags tests whose every assertion sits behind an
`if`, because a degenerate result then SKIPS rather than fails. Before replacing
each guard with an assertion, check whether it opens on its fixture:

  opens  -> the guard is dead weight; assert the condition and delete the `if`
  closed -> the test has never run its own body, and the subject may be broken

Last session's measurements say the taxonomy fixture yields 365 children keys and
398 parents keys, so most should open. `test_rog_layers_decrease_threshold` is the
one to watch: build_rog_ontology produced exactly ONE layer on this fixture, and a
guard like `if len(layers) > 1` would never open.
"""

from __future__ import annotations

import numpy as np


def sample_embeddings() -> np.ndarray:
    rng = np.random.default_rng(42)
    e = rng.standard_normal((500, 64)).astype(np.float32)
    return e / np.linalg.norm(e, axis=1, keepdims=True)


def clustered_embeddings() -> np.ndarray:
    rng = np.random.default_rng(42)
    dim = 64
    clusters = []
    for i in range(5):
        c = np.zeros(dim, dtype=np.float32)
        c[i * 10 : (i + 1) * 10] = 1.0
        c /= np.linalg.norm(c)
        pts = c + rng.standard_normal((100, dim)).astype(np.float32) * 0.1
        clusters.append(pts / np.linalg.norm(pts, axis=1, keepdims=True))
    for _ in range(50):
        c1, c2 = rng.choice(5, 2, replace=False)
        a = np.zeros(dim, dtype=np.float32)
        a[c1 * 10 : (c1 + 1) * 10] = 1.0
        b = np.zeros(dim, dtype=np.float32)
        b[c2 * 10 : (c2 + 1) * 10] = 1.0
        br = 0.5 * a + 0.5 * b
        br /= np.linalg.norm(br)
        br = br + rng.standard_normal(dim).astype(np.float32) * 0.05
        clusters.append((br / np.linalg.norm(br)).reshape(1, -1))
    return np.vstack(clusters)


def say(test: str, guard: str, opens: bool, detail: str) -> None:
    print(f"  [{'OPENS' if opens else 'CLOSED'}] {test:<46} {guard:<28} {detail}")


def main() -> None:
    from dyf import build_dag_taxonomy, build_rog_ontology, build_unified_ontology

    clus = clustered_embeddings()

    print("tests/test_rag.py — DAGTaxonomy on clustered_embeddings")
    tax = build_dag_taxonomy(clus, verbose=False)
    say("test_get_children_parents", "if taxonomy.children:", bool(tax.children), f"{len(tax.children)} keys")
    say("test_get_ancestors", "if taxonomy.parents:", bool(tax.parents), f"{len(tax.parents)} keys")
    say("test_get_descendants", "if taxonomy.children:", bool(tax.children), f"{len(tax.children)} keys")
    say(
        "test_get_lowest_common_ancestors",
        "if len(parents) >= 2:",
        len(tax.parents) >= 2,
        f"{len(tax.parents)} keys",
    )
    say("test_get_path / test_get_all_paths", "if taxonomy.children:", bool(tax.children), f"{len(tax.children)} keys")

    print("\ntests/test_rag.py — ROG / unified ontology")
    rog = build_rog_ontology(clus, verbose=False)
    say(
        "test_rog_layers_decrease_threshold",
        "if len(layers) > 1:",
        len(rog.layers) > 1,
        f"layers={len(rog.layers)} thresholds={[round(x.similarity_threshold, 3) for x in rog.layers]}",
    )
    uni = build_unified_ontology(clus, verbose=False)
    n_bridge = len(np.asarray(getattr(uni, "bridge_nodes", [])))
    print(f"  (test_unified_ontology_bridge_edges is vacuous, not guarded: bridge_nodes={n_bridge})")


if __name__ == "__main__":
    main()
