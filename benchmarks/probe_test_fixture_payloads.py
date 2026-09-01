"""What do the flagged test fixtures actually produce?

The test-assertion audit flags 16 tests as shape-only. Before adding behavioural
assertions, measure whether the functions under test return anything at all on
those fixtures. Two possible outcomes, and they call for different work:

- non-empty  -> the tests are simply weak; add assertions (hygiene)
- empty      -> the tests were hiding a #5-class bug (a real defect)

Reproduces the exact fixtures from tests/test_rag.py.
"""

from __future__ import annotations

import numpy as np


def sample_embeddings() -> np.ndarray:
    rng = np.random.default_rng(42)
    e = rng.standard_normal((500, 64)).astype(np.float32)
    return e / np.linalg.norm(e, axis=1, keepdims=True)


def clustered_embeddings() -> np.ndarray:
    rng = np.random.default_rng(42)
    n_per_cluster, dim = 100, 64
    clusters = []
    for i in range(5):
        center = np.zeros(dim, dtype=np.float32)
        center[i * 10 : (i + 1) * 10] = 1.0
        center = center / np.linalg.norm(center)
        noise = rng.standard_normal((n_per_cluster, dim)).astype(np.float32) * 0.1
        cluster = center + noise
        clusters.append(cluster / np.linalg.norm(cluster, axis=1, keepdims=True))
    for _ in range(50):
        c1, c2 = rng.choice(5, 2, replace=False)
        a = np.zeros(dim, dtype=np.float32)
        a[c1 * 10 : (c1 + 1) * 10] = 1.0
        b = np.zeros(dim, dtype=np.float32)
        b[c2 * 10 : (c2 + 1) * 10] = 1.0
        bridge = 0.5 * a + 0.5 * b
        bridge = bridge / np.linalg.norm(bridge)
        bridge = bridge + rng.standard_normal(dim).astype(np.float32) * 0.05
        clusters.append((bridge / np.linalg.norm(bridge)).reshape(1, -1))
    return np.vstack(clusters)


def report(label: str, verdict: str, detail: str) -> None:
    mark = {"OK": " ok ", "EMPTY": "EMPTY", "PARTIAL": "part"}[verdict]
    print(f"  [{mark}] {label:44s} {detail}")


def main() -> None:
    from dyf import (
        BridgeIndex,
        build_dag_taxonomy,
        build_rog_ontology,
        build_unified_ontology,
        compute_neighbor_diversity,
        get_kmeans_init,
        mine_dag_chains,
    )

    samp = sample_embeddings()
    clus = clustered_embeddings()
    print(f"fixtures: sample={samp.shape} clustered={clus.shape}\n")

    print("BridgeIndex.get_super_connectors  (test_rag.py:230, on sample_embeddings)")
    idx = BridgeIndex(n_anchors=50)
    idx.fit(samp, verbose=False)
    sc = idx.get_super_connectors()
    n_sc = len(np.asarray(sc.indices))
    gc = np.asarray(sc.global_centrality)
    report(
        "indices",
        "OK" if n_sc else "EMPTY",
        f"n={n_sc}, centrality sum={gc.sum()}, nonzero={int((gc > 0).sum())}",
    )
    quad = getattr(sc, "quadrant", None)
    if quad is not None:
        vals, counts = np.unique(np.asarray(quad), return_counts=True)
        report("quadrant", "OK" if len(vals) > 1 else "EMPTY", str(dict(zip(vals, counts.tolist()))))

    print("\nget_kmeans_init  (test_rag.py:397 sample, :431 clustered)")
    for name, emb, nlist in (("sample nlist=20", samp, 20), ("clustered nlist=10", clus, 10)):
        init = get_kmeans_init(emb, nlist=nlist, verbose=False)
        n_uniq = len(np.unique(init, axis=0))
        report(
            name,
            "OK" if n_uniq == nlist else "PARTIAL",
            f"unique rows={n_uniq}/{nlist}, all-zero rows={int((np.abs(init).sum(1) == 0).sum())}",
        )

    print("\ncompute_neighbor_diversity  (test_rag.py:463, :487)")
    div = compute_neighbor_diversity(samp, k=10)
    report(
        "sample k=10",
        "OK" if div.std() > 0 else "EMPTY",
        f"min={div.min():.4f} med={np.median(div):.4f} max={div.max():.4f} std={div.std():.4f}",
    )

    print("\nmine_dag_chains  (test_rag.py:502, on sample_embeddings)")
    for name, emb in (("sample", samp), ("clustered", clus)):
        res = mine_dag_chains(emb, min_chain_length=3, verbose=False)
        report(
            name,
            "OK" if len(res.chains) else "EMPTY",
            f"chains={len(res.chains)} edges={len(res.parent_child_edges)} components={res.n_components}",
        )

    print("\nbuild_dag_taxonomy  (test_rag.py:612, :697, :786 — all on clustered)")
    tax = build_dag_taxonomy(clus, verbose=False)
    n_edges = sum(len(v) for v in tax.children.values())
    report(
        "children/parents",
        "OK" if tax.children and tax.parents else "EMPTY",
        f"children keys={len(tax.children)} parents keys={len(tax.parents)} edges={n_edges}",
    )
    report(
        "roots/leaves",
        "OK" if tax.roots and tax.leaves else "EMPTY",
        f"roots={len(tax.roots)} leaves={len(tax.leaves)}",
    )
    # the two guarded tests: do their guards even open?
    if len(tax.parents) >= 2:
        nodes = list(tax.parents.keys())[:2]
        common = tax.get_common_ancestors(nodes[0], nodes[1], max_depth=5)
        report("get_common_ancestors guard opens", "OK", f"|common|={len(common)}")
    else:
        report("get_common_ancestors guard opens", "EMPTY", "guard never opens -> test is a no-op")
    if tax.children:
        node = list(tax.children.keys())[0]
        report("get_path(node,node) guard opens", "OK", f"path={tax.get_path(node, node)}")
    else:
        report("get_path guard opens", "EMPTY", "guard never opens -> test is a no-op")

    print("\nbuild_unified_ontology  (test_rag.py:845, on clustered)")
    uni = build_unified_ontology(clus, verbose=False)
    n_main, n_out = len(np.asarray(uni.main_nodes)), len(np.asarray(uni.outlier_nodes))
    report(
        "node sets",
        "OK" if (n_main or n_out) else "EMPTY",
        f"main={n_main} outlier={n_out} edges={sum(len(v) for v in uni.ontology.children.values())}",
    )

    print("\nbuild_rog_ontology  (test_rag.py:955, on clustered)")
    rog = build_rog_ontology(clus, verbose=False)
    report(
        "layers/coverage",
        "OK" if rog.layers else "EMPTY",
        f"layers={len(rog.layers)} coverage={rog.total_coverage:.3f} "
        f"edges={sum(len(v) for v in rog.ontology.children.values())}",
    )


if __name__ == "__main__":
    main()
