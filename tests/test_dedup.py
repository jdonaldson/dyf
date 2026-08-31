"""Tests for ingest-time near-duplicate detection."""

import numpy as np
import pytest

from dyf.dedup import DedupResult, decode_members, near_duplicate_clusters


def _clustered(n_clusters=20, per=4, dim=32, jitter=0.0, seed=0):
    """n_clusters groups of `per` near-identical unit vectors, plus optional jitter."""
    rng = np.random.default_rng(seed)
    centers = rng.standard_normal((n_clusters, dim)).astype(np.float32)
    centers /= np.linalg.norm(centers, axis=1, keepdims=True)
    X = np.repeat(centers, per, axis=0)
    if jitter:
        X = X + rng.standard_normal(X.shape).astype(np.float32) * jitter
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    return X.astype(np.float32)


def test_exact_duplicates_collapse():
    X = _clustered(n_clusters=20, per=4, jitter=0.0)
    r = near_duplicate_clusters(X, n_tables=6)
    assert len(r.representatives) == 20
    assert r.n_removed == 60
    assert r.removed_fraction == pytest.approx(0.75)


def test_distinct_points_are_untouched():
    rng = np.random.default_rng(1)
    X = rng.standard_normal((200, 32)).astype(np.float32)
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    r = near_duplicate_clusters(X)
    assert len(r.representatives) == 200
    assert r.n_removed == 0


def test_star_property_every_member_near_its_representative():
    """The star guarantee: members sit within threshold OF THE REPRESENTATIVE, not a chain."""
    X = _clustered(n_clusters=15, per=5, jitter=0.02, seed=3)
    thresh = 0.99
    r = near_duplicate_clusters(X, threshold=thresh, n_tables=6)
    Xn = X / np.linalg.norm(X, axis=1, keepdims=True)
    for rep, grp in r.members().items():
        sims = Xn[grp] @ Xn[rep]
        others = sims[grp != rep]
        if len(others):
            assert others.min() > thresh, f"member of {rep} only reached cos {others.min():.4f}"


def test_mask_and_representatives_agree():
    X = _clustered(n_clusters=10, per=3)
    r = near_duplicate_clusters(X, n_tables=6)
    m = r.mask()
    assert m.sum() == len(r.representatives)
    assert np.array_equal(np.flatnonzero(m), r.representatives)


def test_members_partition_the_input():
    X = _clustered(n_clusters=12, per=3, jitter=0.01)
    r = near_duplicate_clusters(X, n_tables=6)
    seen = np.concatenate(list(r.members().values()))
    assert np.array_equal(np.sort(seen), np.arange(len(X)))


def test_member_field_roundtrip_excludes_representative():
    X = _clustered(n_clusters=8, per=3)
    r = near_duplicate_clusters(X, n_tables=6)
    field = r.member_field()
    assert len(field) == len(r.representatives)
    mem = r.members()
    for rep, encoded in zip(r.representatives, field):
        decoded = decode_members(encoded)
        expected = mem[int(rep)]
        expected = expected[expected != rep]
        assert np.array_equal(np.sort(decoded), np.sort(expected))
        assert int(rep) not in decoded.tolist()


def test_member_field_is_a_valid_stored_field_type():
    """utf8 is what `.dyf` stored fields support; lists are not."""
    from dyf.lazy_index import _infer_arrow_type

    X = _clustered(n_clusters=5, per=2)
    r = near_duplicate_clusters(X, n_tables=6)
    _, tname = _infer_arrow_type(r.member_field())
    assert tname == "utf8"


def test_threshold_controls_aggressiveness():
    X = _clustered(n_clusters=10, per=4, jitter=0.05, seed=7)
    loose = near_duplicate_clusters(X, threshold=0.90, n_tables=6)
    tight = near_duplicate_clusters(X, threshold=0.999, n_tables=6)
    assert loose.n_removed >= tight.n_removed


def test_zero_norm_rows_are_singletons_not_universal_matches():
    X = _clustered(n_clusters=5, per=2)
    X = np.vstack([X, np.zeros((3, X.shape[1]), np.float32)])
    r = near_duplicate_clusters(X, n_tables=6)
    zero_ids = list(range(len(X) - 3, len(X)))
    for z in zero_ids:
        assert r.labels[z] == z, "a zero-norm row must not be absorbed into a cluster"


def test_empty_and_single_input():
    empty = near_duplicate_clusters(np.zeros((0, 8), np.float32))
    assert empty.n_points == 0 and empty.n_removed == 0
    one = near_duplicate_clusters(np.ones((1, 8), np.float32))
    assert len(one.representatives) == 1


def test_raw_unnormalised_input_is_handled():
    X = _clustered(n_clusters=10, per=3) * np.linspace(1, 50, 30)[:, None]
    r = near_duplicate_clusters(X, n_tables=6)
    # scaling does not change direction, so the same 10 clusters must be found
    assert len(r.representatives) == 10


def test_reproducible_for_a_fixed_seed():
    X = _clustered(n_clusters=12, per=3, jitter=0.01)
    a = near_duplicate_clusters(X, seed=5)
    b = near_duplicate_clusters(X, seed=5)
    assert np.array_equal(a.labels, b.labels)


def test_validation():
    with pytest.raises(ValueError, match="2-D"):
        near_duplicate_clusters(np.zeros(10, np.float32))
    with pytest.raises(ValueError, match="threshold"):
        near_duplicate_clusters(np.zeros((4, 4), np.float32), threshold=1.5)


def test_cluster_sizes_sum_to_n():
    X = _clustered(n_clusters=9, per=4, jitter=0.01)
    r = near_duplicate_clusters(X, n_tables=6)
    assert r.cluster_sizes().sum() == len(X)
    assert len(r.cluster_sizes()) == len(r.representatives)


def test_dataclass_is_constructible_directly():
    r = DedupResult(np.array([0, 0, 2]), np.array([0, 2]), 0.99, 3)
    assert r.n_removed == 1
    assert r.members()[0].tolist() == [0, 1]


def test_end_to_end_dedup_then_index_roundtrip():
    """The whole point of member_field(): survive write -> read with NO format change.

    Builds a tree over representatives only, stores the member lists as an ordinary utf8
    stored field, reads them back through LazyIndex, and checks the decoded members
    reconstruct the original point set exactly.
    """
    import os
    import tempfile

    pytest.importorskip("dyf_rs")
    from dyf.dyf_tree import build_dyf_tree
    from dyf.lazy_index import LazyIndex, write_lazy_index

    X = _clustered(n_clusters=25, per=4, dim=32, jitter=0.001, seed=11)
    r = near_duplicate_clusters(X, n_tables=6)
    assert r.n_removed > 0, "fixture must actually contain duplicates"

    reps = r.representatives
    Xr = np.ascontiguousarray(X[reps])
    tree = build_dyf_tree(Xr, max_depth=3, num_bits=3, min_leaf_size=2, seed=42)

    with tempfile.NamedTemporaryFile(suffix=".dyf", delete=False) as f:
        path = f.name
    try:
        write_lazy_index(
            tree,
            Xr,
            path,
            stored_fields={
                "dup_members": r.member_field(),
                "orig_index": reps.astype(np.int64),
            },
        )
        idx = LazyIndex(path)
        assert "dup_members" in idx.stored_field_names

        res = idx.search(Xr[0], k=len(reps), nprobe=len(reps))
        got_orig = res.fields["orig_index"]
        got_members = res.fields["dup_members"]
        assert len(got_orig) > 0

        # every returned row reconstructs its own cluster
        truth = r.members()
        for orig, enc in zip(got_orig, got_members):
            expanded = np.concatenate([[int(orig)], decode_members(enc)])
            assert np.array_equal(np.sort(expanded), np.sort(truth[int(orig)]))
    finally:
        if os.path.exists(path):
            os.unlink(path)


def test_full_recovery_of_original_point_set():
    """Representatives plus decoded members must cover every original point exactly once."""
    X = _clustered(n_clusters=18, per=5, jitter=0.002, seed=13)
    r = near_duplicate_clusters(X, n_tables=6)
    recovered = []
    for rep, enc in zip(r.representatives, r.member_field()):
        recovered.append(int(rep))
        recovered.extend(decode_members(enc).tolist())
    assert np.array_equal(np.sort(np.array(recovered)), np.arange(len(X)))
