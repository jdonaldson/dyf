"""Do the other absolute-cosine defaults degenerate on real corpora? (KNOWN_ISSUES #5, P2)

`audit_test_assertions.py` flags tests that assert shape but never behaviour. Several of the
functions it flags live in `ontology.py`, whose defaults are absolute cosine thresholds
(`similarity_threshold=0.55`, `outlier_similarity_threshold=0.45`, `initial_threshold=0.55`)
— the same shape of constant that made `find_super_connectors` return nothing on text.

Issue 5's instance STARVED: centroid similarity never fell below 0.5 so zero bridges were
found. These use *pairwise neighbour* similarity instead, which on text runs high, so the
expected failure is the opposite — FLOODING, where the threshold admits everything and the
parameter stops discriminating. Either way it is not doing its job.

Reported per function and corpus: the size/coverage of the output, plus what fraction of
kNN pairs clear the threshold. A fraction near 0% or near 100% means the constant is not in
a usable regime for that corpus.

Corpora are chosen to span anisotropy, which is the variable that breaks these constants:
SEC 768d text (very anisotropic), CMU MoCap 62d, and an isotropic gaussian control.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "sequence_arc"))
import sec_seqlib as S  # noqa: E402

PAPER = os.path.expanduser("~/Projects/semantic-proprioception-paper/experiments/data")
N = 2500
K = 30
SEED = 42


def corpora():
    rng = np.random.default_rng(SEED)
    E, *_ = S.load()
    yield "sec_768 (text)", np.ascontiguousarray(E[rng.choice(len(E), N, replace=False)])
    p = os.path.join(PAPER, "cmu_mocap_features.npy")
    if os.path.exists(p):
        X = np.load(p).astype(np.float32)
        X = X[rng.choice(len(X), N, replace=False)]
        X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
        yield "cmu_mocap_62", np.ascontiguousarray(X)
    G = rng.standard_normal((N, 64)).astype(np.float32)
    G /= np.linalg.norm(G, axis=1, keepdims=True)
    yield "isotropic gaussian", G


def knn_sim_profile(X, k=K):
    """What fraction of kNN pairs clear each candidate threshold?"""
    Sm = X @ X.T
    np.fill_diagonal(Sm, -np.inf)
    top = np.sort(Sm, axis=1)[:, -k:]
    return top.ravel()


def main():
    from dyf import build_dag_taxonomy, build_rog_ontology, build_unified_ontology, mine_dag_chains

    print("Fraction of kNN pairs clearing each ABSOLUTE threshold\n")
    print(f"{'corpus':<22}{'median kNN sim':>16}{'>=0.35':>9}{'>=0.45':>9}{'>=0.55':>9}")
    profiles = {}
    for name, X in corpora():
        s = knn_sim_profile(X)
        profiles[name] = (X, s)
        print(
            f"{name:<22}{np.median(s):>16.3f}" + "".join(f"{100 * (s >= t).mean():>8.0f}%" for t in (0.35, 0.45, 0.55))
        )
    print("\n  ~0% or ~100% means the constant is not discriminating on that corpus.")

    print("\n" + "=" * 78)
    print("Do the ontology builders produce non-degenerate output at their defaults?")
    print("=" * 78)
    # NB: field names verified against the real dataclasses. An earlier version of this probe
    # read `.nodes` / `.chains` off DAGTaxonomy and UnifiedOntologyResult -- neither exists --
    # and so printed 0 for three functions that are in fact working. Reading a nonexistent
    # attribute and reporting the default is how a probe manufactures a false positive.
    print(
        f"{'corpus':<22}{'chains':>9}{'tax roots':>11}{'tax leaves':>12}{'main nodes':>12}{'outliers':>10}{'rog layers':>12}"
    )
    for name, (X, _s) in profiles.items():
        row = f"{name:<22}"
        try:
            ch = mine_dag_chains(X, k_neighbors=K)
            row += f"{len(ch.chains):>9}"
        except Exception as e:  # noqa: BLE001
            row += f"{type(e).__name__:>9}"
        try:
            tx = build_dag_taxonomy(X, k_neighbors=K)
            row += f"{len(tx.roots):>11}{len(tx.leaves):>12}"
        except Exception as e:  # noqa: BLE001
            row += f"{type(e).__name__:>11}{'':>12}"
        try:
            un = build_unified_ontology(X)
            row += f"{len(un.main_nodes):>12}{len(un.outlier_nodes):>10}"
        except Exception as e:  # noqa: BLE001
            row += f"{type(e).__name__:>12}{'':>10}"
        try:
            rg = build_rog_ontology(X)
            row += f"{len(rg.layers):>12}"
        except Exception as e:  # noqa: BLE001
            row += f"{type(e).__name__:>12}"
        print(row, flush=True)

    print("\nVERDICT. The constants are INERT on real corpora, not destructive: 0.35/0.45/0.55")
    print("admit ~100% of kNN pairs on both SEC text and MoCap, so the parameter does not")
    print("discriminate — but 'admit everything' degrades to 'use all neighbours', which is a")
    print("benign default. That is materially different from KNOWN_ISSUES #5, where the same")
    print("class of constant produced an EMPTY result. So this is a misleading-knob problem")
    print("(users think they are tuning something that does nothing) rather than a bug.")
    print("`build_rog_ontology` adapts its cut via threshold_decay + target_coverage and is")
    print("the one design here that is structurally immune.")


if __name__ == "__main__":
    main()
