"""The leaf-graph spectrum read-out (KNOWN_ISSUES #11 addendum).

Two regimes, measured 2026-09-27 on moons / circles / blobs / digits / MNIST:
  connectivity — the leaf graph is disconnected or one weak cut from it
                 (lambda_2 ~ 1e-4, a >= 5x multiplicative jump low in the spectrum);
                 modularity chops such graphs into arcs, bisection / components recover them.
  blob         — lambda_2 ~ 1e-2 with ratios <= 2; modularity is the right objective.
These tests pin the decoder on graphs whose structure is known by construction, then check
the read-out reaches the caller through louvain_cluster_leaves on a real index.
"""

import numpy as np
import pytest

from dyf.agglomerate import LeafGraphSpectrum, leaf_graph_spectrum


def _clique_edges(nodes, w=1.0):
    return [(i, j, w) for i in nodes for j in nodes if i != j]


def test_two_disjoint_cliques_are_two_components():
    edges = _clique_edges(range(0, 8)) + _clique_edges(range(8, 16))
    s = leaf_graph_spectrum(edges, 16)
    assert isinstance(s, LeafGraphSpectrum)
    assert s.n_components == 2
    assert s.k_gap == 2
    assert s.regime == "connectivity"
    assert s.eigenvalues[1] < 1e-6  # lambda_2 exactly zero for a disconnected graph


def test_weak_bridge_is_connectivity_regime_with_k_two():
    # Two 12-cliques joined by one edge a hundred times weaker than the clique edges:
    # connected (lambda_2 > 0) but one cut from splitting.
    edges = _clique_edges(range(0, 12)) + _clique_edges(range(12, 24)) + [(11, 12, 0.01), (12, 11, 0.01)]
    s = leaf_graph_spectrum(edges, 24)
    assert s.n_components == 1
    assert 0 < s.lambda2 < 1e-3
    assert s.k_gap == 2
    assert s.max_ratio >= 5
    assert s.regime == "connectivity"


def test_single_dense_blob_is_blob_regime():
    # A random geometric graph on one Gaussian blob: no cut is much weaker than another.
    rng = np.random.default_rng(0)
    P = rng.standard_normal((200, 3))
    d2 = ((P[:, None, :] - P[None, :, :]) ** 2).sum(-1)
    k = 10
    nbr = np.argsort(d2, axis=1)[:, 1 : k + 1]
    sig = np.median(np.sqrt(np.take_along_axis(d2, nbr, 1)))
    edges = [(i, int(j), float(np.exp(-d2[i, j] / (2 * sig**2)))) for i in range(200) for j in nbr[i]]
    s = leaf_graph_spectrum(edges, 200)
    assert s.n_components == 1
    assert s.lambda2 > 1e-3
    assert s.max_ratio < 5
    assert s.regime == "blob"


def test_degenerate_graphs_do_not_raise():
    s = leaf_graph_spectrum([], 1)
    assert s.regime == "degenerate" and s.n_components == 1
    s = leaf_graph_spectrum([(0, 1, 1.0), (1, 0, 1.0)], 2)
    assert s.n_components == 1
    assert s.eigenvalues.shape[0] <= 2


def test_readout_reaches_the_caller_through_louvain_cluster_leaves(tmp_path):
    """Three far-apart Gaussian blobs, well away from the origin so the tree's row
    normalisation does not fold them together: the leaf graph must come out disconnected
    (or one weak cut from it) and the result must carry the spectrum."""
    from dyf import build_dyf_tree, write_lazy_index
    from dyf.agglomerate import louvain_cluster_leaves
    from dyf.lazy_index import LazyIndex

    rng = np.random.default_rng(42)
    centers = np.array([[30.0, 0.0, 0.0], [0.0, 30.0, 0.0], [0.0, 0.0, 30.0]], dtype=np.float32)
    X = np.vstack([c + rng.standard_normal((400, 3)).astype(np.float32) for c in centers])
    y = np.repeat(np.arange(3), 400)

    tree = build_dyf_tree(X, max_depth=5, num_bits=2, min_leaf_size=5, seed=42)
    path = str(tmp_path / "blobs.dyf")
    write_lazy_index(tree, X, path, compression="none", quantization="float32")
    with LazyIndex(path) as idx:
        res = louvain_cluster_leaves(idx, X[:, :2], X)

    assert res.ok
    assert res.spectrum is not None
    assert res.spectrum.regime == "connectivity"
    assert res.spectrum.n_components == 3
    assert res.spectrum.k_gap == 3
    # Modularity may split a blob internally (that is the point of the read-out), but no
    # community may straddle two blobs: every community is pure in the blob label.
    for c in np.unique(res.point_labels):
        assert len(np.unique(y[res.point_labels == c])) == 1
    assert "connectivity" in res.summary()


@pytest.mark.parametrize("bad", [-1, 0])
def test_n_leaves_must_be_positive(bad):
    with pytest.raises(ValueError):
        leaf_graph_spectrum([(0, 0, 1.0)], bad)
