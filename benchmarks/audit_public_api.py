"""Smoke-test the public API: does every exported callable produce non-degenerate output?

Motivation. Two of the handful of features inspected by hand this session were silently
broken on text embeddings — `find_super_connectors` returned `indices=[]` with all-zero
centrality (KNOWN_ISSUES #5) and `nprobe="auto"` resolves to a constant (#4). Both carried
performance claims in their docstrings, and both passed their tests, because those tests
assert types and array lengths. With 109 exports and 72 callables, finding two by hand says
nothing reassuring about the rest.

This enumerates the public surface and, for every callable it can construct arguments for,
checks the output is not degenerate:

  EMPTY      None, or a zero-length array/list/dict
  ALLZERO    numeric output that is entirely zero
  CONSTANT   numeric output with a single distinct value (no discrimination)
  OK         anything else

ANISOTROPIC FIXTURE BY DEFAULT. The failure mode is specific to embeddings that live in a
narrow cone, which is what real text embeddings do (measured: maximally-distant dyf cells on
SEC still sit at mean cosine 0.821). Isotropic gaussians — what most of the existing test
fixtures use — do not reproduce it. `--isotropic` runs the contrasting fixture so the
difference is visible.

Functions whose arguments cannot be auto-constructed are reported as SKIP with their
signature, so the unaudited fraction is explicit and countable rather than silently omitted.

Usage:
    python benchmarks/audit_public_api.py                # anisotropic (text-like)
    python benchmarks/audit_public_api.py --isotropic     # contrast fixture
    python benchmarks/audit_public_api.py --verbose       # include SKIP signatures
"""

import inspect
import os
import sys
import traceback
import warnings

import numpy as np

warnings.filterwarnings("ignore")

N = 400
DIM = 32
SEED = 42

# first-parameter names we know how to supply an embedding matrix for
EMB_PARAMS = {"embeddings", "embedding", "X", "vectors", "coords", "data"}
# params we can fill from the fixture when they have no default
KNOWN_FILLS = {
    "k": 5,
    "n_clusters": 4,
    "nlist": 8,
    "max_k": 4,
    "n_groups": 4,
    "num_bits": 4,
    "max_depth": 3,
    "min_leaf_size": 5,
    "dim": DIM,
    "embedding_dim": DIM,
    "n_components": 2,
    "resolution": 1.0,
    "seed": SEED,
    "verbose": False,
}


def load_real(n=N):
    """Prefer a REAL corpus. Synthetic anisotropy proved hard to calibrate honestly.

    Three attempts at a synthetic cone each failed differently: too tight collapsed to 1 LSH
    bucket (false positives from `diversify_by_facet`/`chunk_redundancy`), too wide became
    isotropic at cosine 0.102 (no longer reproducing the failure mode), and adding hierarchy
    tightened it again to 3 buckets and 5 false positives. Anisotropy, bucket occupancy and
    hierarchy are three separate properties and hand-tuning all three is a losing game.

    Real embeddings have them by construction, so use them when present and treat synthetic as
    a clearly-labelled fallback whose degenerate verdicts must be checked against real data
    before being believed.
    """
    try:
        sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "sequence_arc"))
        import sec_seqlib as S

        E, *_ = S.load()
        rng = np.random.default_rng(SEED)
        return np.ascontiguousarray(E[rng.choice(len(E), min(n, len(E)), replace=False)])
    except Exception:  # noqa: BLE001
        return None


def make_fixture(isotropic: bool):
    """Anisotropic (text-like) by default: a cone, CALIBRATED against real text.

    ⚠ Calibration matters and a first attempt got it wrong. A cone of
    ``axis + 0.05*randn`` centres with ``0.03*randn`` jitter gives median pairwise cosine
    0.924 — plausible-looking — but collapses all 400 points into a SINGLE LSH bucket. Two
    functions then reported degenerate output (`diversify_by_facet` found 1 facet,
    `chunk_redundancy` was constant) and both were *correct given the fixture*: there really
    was only one bucket to diversify across. A fixture more extreme than reality manufactures
    false positives.

    Widening it too far is the opposite error: 0.45/0.25 gives 27 buckets but median cosine
    0.102, i.e. isotropic, which no longer reproduces the failure mode at all.

    CALIBRATION TARGET, measured on the same N and num_bits: real SEC 768d text gives median
    all-pairs cosine **0.700** with **9 of 32** buckets occupied. A sweep of cone widths puts
    ``centres 0.10 / jitter 0.07`` at cosine 0.695 with 17 buckets — the closest match, and
    what is used below. (An earlier note quoted 0.855 for SEC; that is the median over kNN
    pairs, a different and much higher statistic than all-pairs.)
    """
    rng = np.random.default_rng(SEED)
    if isotropic:
        X = rng.standard_normal((N, DIM)).astype(np.float32)
    else:
        # HIERARCHICAL, not a flat blob. A flat 8-centre cone at the right anisotropy still
        # made `mine_dag_chains` return EMPTY, because there was no nested structure for chain
        # mining to find — while real SEC yields 46-66 chains at this same n=400. Anisotropy
        # and hierarchy are separate fixture properties and both have to be present, or
        # structure-mining functions read as broken when the fixture is simply featureless.
        axis = np.zeros(DIM, dtype=np.float32)
        axis[0] = 1.0
        supers = axis + 0.10 * rng.standard_normal((3, DIM)).astype(np.float32)
        subs = np.repeat(supers, 3, axis=0) + 0.05 * rng.standard_normal((9, DIM)).astype(np.float32)
        lab = rng.integers(0, len(subs), N)
        X = subs[lab] + 0.03 * rng.standard_normal((N, DIM)).astype(np.float32)
    X /= np.linalg.norm(X, axis=1, keepdims=True) + 1e-12
    return np.ascontiguousarray(X)


#: Result fields that ARE the answer. If one of these is degenerate the result is degenerate,
#: however healthy the rest of the object looks.
PAYLOAD_FIELDS = {
    "indices",
    "index",
    "selected",
    "labels",
    "point_labels",
    "chains",
    "roots",
    "leaves",
    "nodes",
    "main_nodes",
    "bridge_indices",
    "scores",
    "clusters",
    "communities",
    "layers",
    "representatives",
    "keywords",
    "anchors",
}

#: Payload fields that are a SELECTION (which items) rather than a SCORE (how much per item).
#: For these, only EMPTY is degenerate: an index array of length 1 has a single distinct value
#: by definition, so the CONSTANT rule misfires on it. Measured false positive —
#: `find_super_connectors` on the 400-point SEC fixture returns `indices=[175]` with all four
#: quadrant classes populated and 40 nonzero centralities, and was reported CONSTANT.
#: `labels` / `point_labels` / `scores` stay under the CONSTANT rule, where a single distinct
#: value genuinely means no discrimination (one cluster, or a flat score).
SELECTION_FIELDS = {
    "indices",
    "index",
    "selected",
    "roots",
    "leaves",
    "nodes",
    "main_nodes",
    "bridge_indices",
    "representatives",
    "anchors",
}


def classify(out) -> str:
    """EMPTY / ALLZERO / CONSTANT / OK for an arbitrary return value.

    ⚠ The first version judged a dataclass OK if ANY field was OK, and the canary at the end
    of `main` caught that: `find_super_connectors` in its pre-fix state has an empty `indices`
    but a full 400-element `quadrant` of "Regular" strings, so the healthy field masked the
    empty payload and the harness could not see the very bug it was built for. Payload fields
    are therefore judged first, and a degenerate payload decides the verdict.
    """
    if out is None:
        return "EMPTY"
    # dataclass-ish results: the payload field decides, others are context
    if hasattr(out, "__dataclass_fields__") or (hasattr(out, "__dict__") and not isinstance(out, type)):
        members = vars(out)
        payload = {
            k: v for k, v in members.items() if k in PAYLOAD_FIELDS and isinstance(v, (np.ndarray, list, tuple, dict))
        }
        for name, v in payload.items():
            k = classify(v)
            if name in SELECTION_FIELDS and k in ("ALLZERO", "CONSTANT"):
                # a selection is only degenerate when it selects nothing
                k = "OK"
            if k != "OK":
                return k
        if payload:
            return "OK"
        fields = [v for v in members.values() if isinstance(v, (np.ndarray, list, tuple, dict))]
        if fields:
            kinds = {classify(f) for f in fields}
            if kinds == {"EMPTY"}:
                return "EMPTY"
            if "OK" in kinds:
                return "OK"
            return sorted(kinds)[0]
    if isinstance(out, (list, tuple, dict, set)):
        if len(out) == 0:
            return "EMPTY"
        if isinstance(out, (list, tuple)) and all(isinstance(o, (int, float, np.number)) for o in out):
            return classify(np.asarray(out))
        return "OK"
    if isinstance(out, np.ndarray):
        if out.size == 0:
            return "EMPTY"
        if out.dtype.kind in "fiub":
            if out.dtype.kind != "b" and not np.any(out):
                return "ALLZERO"
            if len(np.unique(out)) == 1:
                return "CONSTANT"
        return "OK"
    if isinstance(out, (int, float, np.number)):
        return "OK"
    return "OK"


def make_bundle(X):
    """Derived fixtures so functions taking trees/labels/bucket-ids can be called too.

    Auto-detection alone reached only 13 of 72 callables, because most of the surface takes a
    tree, cluster labels, bucket ids or a kNN matrix rather than raw embeddings. Building
    those once lifts coverage substantially; anything still unreachable is reported as SKIP so
    the unaudited fraction stays visible.
    """
    from dyf_rs import DensityClassifier

    from dyf import build_dyf_tree, cut_tree_to_labels

    n = len(X)
    tree = build_dyf_tree(X, max_depth=3, num_bits=3, min_leaf_size=5, seed=SEED)
    labels = np.asarray(cut_tree_to_labels(tree, n, 4, embeddings=X))
    clf = DensityClassifier(embedding_dim=X.shape[1], num_bits=5, seed=SEED)
    clf.fit(X)
    buckets = np.asarray(clf.get_bucket_ids())
    coh = np.asarray(clf.get_centroid_similarities(), dtype=np.float32)
    knn = np.argsort(-(X @ X.T), axis=1)[:, 1:11]
    cand = np.arange(min(40, n))
    # `sims` must be ALIGNED WITH `candidate_indices`, not full-length: rerank_standard does
    # candidate_indices[argsort(-sims)], so a length-N sims indexes a length-40 array and
    # raises IndexError. Another fixture bug that looked like a library bug.
    sims = (X[cand] @ X[0]).astype(np.float32)
    from dyf import build_pca_tree

    return {
        "X": X,
        "n": n,
        "tree": tree,
        # boundary-persistence functions want a BINARY pca tree (keys 'left'/'right'),
        # not a k-ary dyf tree — passing the latter raised KeyError: 'left'
        "pca_tree": build_pca_tree(X, max_depth=4, min_leaf_size=5),
        "labels": labels,
        "buckets": buckets,
        "doc_ids": np.arange(n) // 4,
        "coh": coh,
        "sims": sims,
        "knn": knn,
        "cand": cand,
        "titles": [f"cluster {labels[i]} item {i} sample text" for i in range(n)],
        "weights": np.linspace(0.5, 1.5, X.shape[1]).astype(np.float32),
    }


#: Explicit argument builders for callables that auto-detection cannot reach.
REGISTRY: dict = {
    "cut_tree_to_labels": lambda f: dict(tree=f["tree"], n_points=f["n"], n_clusters=4, embeddings=f["X"]),
    "refine_clusters": lambda f: dict(labels=f["labels"], embeddings=f["X"]),
    "refine_dyf_tree": lambda f: dict(tree=f["tree"], embeddings=f["X"]),
    "merge_to_max_k": lambda f: dict(point_labels=f["labels"], embeddings=f["X"], max_k=3),
    "flatten_tree": lambda f: dict(tree=f["tree"]),
    "extract_boundary_persistence": lambda f: dict(tree=f["pca_tree"]),
    "boundary_persistence_scores": lambda f: dict(tree=f["pca_tree"]),
    "neighbor_coherence": lambda f: dict(embeddings=f["X"], knn_indices=f["knn"]),
    "chunk_redundancy": lambda f: dict(bucket_ids=f["buckets"], doc_ids=f["doc_ids"]),
    "deduplicate_chunks": lambda f: dict(bucket_ids=f["buckets"], doc_ids=f["doc_ids"]),
    "doc_spread": lambda f: dict(bucket_ids=f["buckets"], doc_ids=f["doc_ids"]),
    "cluster_quality": lambda f: dict(coherence=f["coh"], cluster_labels=f["labels"]),
    "compute_fisher_weights": lambda f: dict(embeddings=f["X"], labels=f["labels"], min_count=2),
    "apply_fisher_weights": lambda f: dict(embeddings=f["X"], weights=f["weights"]),
    "spatial_color_map": lambda f: dict(labels=f["labels"], embeddings=f["X"]),
    "spatial_rgb_map": lambda f: dict(labels=f["labels"], embeddings=f["X"]),
    "compute_similarity_entropy": lambda f: dict(similarities=f["X"] @ f["X"][:10].T),
    "rerank_standard": lambda f: dict(sims=f["sims"], candidate_indices=f["cand"], top_k=5),
    "rerank_mmr": lambda f: dict(query_emb=f["X"][0], candidate_indices=f["cand"], embeddings_normed=f["X"], top_k=5),
    "rerank_bridge_boost": lambda f: dict(sims=f["sims"], candidate_indices=f["cand"], bridge_scores=f["coh"], top_k=5),
    "rerank_bridge_mmr": lambda f: dict(
        query_emb=f["X"][0],
        candidate_indices=f["cand"],
        embeddings_normed=f["X"],
        bridge_scores=f["coh"],
        cluster_labels=f["labels"],
        top_k=5,
    ),
    "diversify_by_facet": lambda f: dict(
        query=f["X"][0], candidate_indices=f["cand"], embeddings=f["X"], bucket_ids=f["buckets"]
    ),
    "label_clusters_frequency": lambda f: dict(titles=f["titles"], labels=f["labels"]),
    "compute_domain_stopwords": lambda f: dict(titles=f["titles"]),
    "assess_text_diversity": lambda f: dict(titles=f["titles"]),
    "tokenize": lambda f: dict(text=f["titles"][0]),
    "coarsen": lambda f: dict(values=[t.split()[0] for t in f["titles"]]),
    "diagnose_axes": lambda f: dict(embeddings=f["X"], label_columns={"cluster": f["labels"].astype(str)}, k=5),
    "compute_hub_score": lambda f: dict(embeddings=f["X"], k=10),
}


def build_args(fn, X):
    """Return kwargs for `fn`, or None if it cannot be auto-called."""
    try:
        sig = inspect.signature(fn)
    except (TypeError, ValueError):
        return None
    params = list(sig.parameters.values())
    if not params:
        return None
    kwargs, saw_emb = {}, False
    for i, p in enumerate(params):
        if p.kind in (p.VAR_POSITIONAL, p.VAR_KEYWORD):
            continue
        if i == 0 and p.name in EMB_PARAMS:
            kwargs[p.name] = X
            saw_emb = True
            continue
        if p.default is not p.empty:
            continue
        if p.name in KNOWN_FILLS:
            kwargs[p.name] = KNOWN_FILLS[p.name]
            continue
        return None  # a required arg we cannot invent
    return kwargs if saw_emb else None


def main() -> int:
    import dyf

    isotropic = "--isotropic" in sys.argv
    verbose = "--verbose" in sys.argv
    synthetic = "--synthetic" in sys.argv
    X, source = None, ""
    if not isotropic and not synthetic:
        X = load_real()
        source = "REAL (SEC 10-Q)"
    if X is None:
        X = make_fixture(isotropic)
        source = "SYNTHETIC isotropic" if isotropic else "SYNTHETIC anisotropic"
    Sm = X @ X.T
    np.fill_diagonal(Sm, np.nan)
    print(f"fixture: {source} {X.shape}, median pairwise cosine {np.nanmedian(Sm):.3f}")
    if source.startswith("SYNTHETIC"):
        print("  ⚠ synthetic fixtures are calibration-sensitive; check any degenerate verdict")
        print("    against a real corpus before believing it (see load_real's docstring)")
    print()

    fx = make_bundle(X)
    n_buckets = len(np.unique(fx["buckets"]))
    print(f"fixture LSH occupancy: {n_buckets} buckets, {len(np.unique(fx['labels']))} cut labels")
    if not isotropic and n_buckets < 2:
        print("  !! fixture collapses to one bucket — any 'degenerate' verdict below is the")
        print("     FIXTURE's fault, not the library's. Widen the cone before trusting output.")
    results, skipped = [], []
    for name in sorted(dyf.__all__):
        obj = getattr(dyf, name, None)
        if obj is None or inspect.isclass(obj) or not callable(obj):
            continue
        if name in REGISTRY:
            try:
                kwargs = REGISTRY[name](fx)
            except Exception as e:  # noqa: BLE001
                results.append((name, "ERROR", f"fixture: {type(e).__name__}: {str(e)[:50]}"))
                continue
        else:
            kwargs = build_args(obj, X)
        if kwargs is None:
            try:
                sig = str(inspect.signature(obj))
            except (TypeError, ValueError):
                sig = "(?)"
            skipped.append((name, sig))
            continue
        try:
            out = obj(**kwargs)
            results.append((name, classify(out), ""))
        except Exception as e:  # noqa: BLE001
            tb = traceback.format_exc(limit=1).strip().splitlines()[-1]
            results.append((name, "ERROR", f"{type(e).__name__}: {str(e)[:60]}"))
            del tb

    order = {"EMPTY": 0, "ALLZERO": 1, "CONSTANT": 2, "ERROR": 3, "OK": 4}
    results.sort(key=lambda r: (order.get(r[1], 9), r[0]))

    print(f"{'function':<34}{'verdict':<10}detail")
    for name, verdict, detail in results:
        mark = "  " if verdict == "OK" else "! "
        print(f"{mark}{name:<32}{verdict:<10}{detail}")

    counts = {}
    for _, v, _ in results:
        counts[v] = counts.get(v, 0) + 1
    print(f"\n{len(results)} auto-called, {len(skipped)} not auto-callable")
    print("  " + "   ".join(f"{k}={v}" for k, v in sorted(counts.items(), key=lambda kv: order.get(kv[0], 9))))
    suspicious = sum(counts.get(k, 0) for k in ("EMPTY", "ALLZERO", "CONSTANT"))
    print(f"\n{suspicious} produced degenerate output on this fixture — each is a candidate for the")
    print("KNOWN_ISSUES #5 class (a shipped function that silently returns nothing).")

    # CANARY. A smoke test reporting "0 problems" proves nothing unless it can detect a
    # problem, so reproduce the known one: find_super_connectors at bridge_percentile=0 is the
    # pre-fix absolute-floor behaviour that returned an empty result on text embeddings. If
    # the classifier does not flag that, the whole report above is untrustworthy.
    from dyf import find_super_connectors

    pre_fix = find_super_connectors(X, min_bucket_size=10, bridge_percentile=0)
    post_fix = find_super_connectors(X, min_bucket_size=10)
    v_pre, v_post = classify(pre_fix), classify(post_fix)
    print(f"\ncanary (KNOWN_ISSUES #5): pre-fix behaviour -> {v_pre}, current -> {v_post}")
    if v_pre == "OK":
        print("  !! the canary was NOT detected — this harness cannot see the bug it was built")
        print("     for, so treat the verdicts above as unvalidated.")
    else:
        print("  harness has teeth: it flags the known regression and clears the fix.")

    if verbose and skipped:
        print(f"\nNot auto-callable ({len(skipped)}) — the unaudited fraction:")
        for name, sig in skipped:
            print(f"  {name}{sig[:100]}")
    elif skipped:
        print(f"\n(--verbose lists the {len(skipped)} that need hand-written fixtures)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
