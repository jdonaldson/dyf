"""Find tests that assert SHAPE but never BEHAVIOUR.

Motivation, from a real escape: `find_super_connectors` shipped returning `indices=[]` with
all-zero centrality on text embeddings, and all 77 `test_rag.py` tests passed throughout.
They asserted `isinstance(result.indices, np.ndarray)` and
`len(result.global_centrality) == len(embeddings)` — both true of an empty result. See
KNOWN_ISSUES #5.

That test shape is the *generator* of the bug class: any detector, selector, or filter whose
tests only check types and lengths can silently start returning nothing.

CLASSIFICATION. Each `assert` in each `test_*` function is labelled:

  shape     isinstance / type() / hasattr / `.shape` / `.dtype` / `is not None`, or a
            `len(a) == len(b)` that only ties an output's length to an input's
  value      anything that constrains CONTENT: a comparison against a literal, `> 0`,
            `.sum()`, `.any()`, `.all()`, `in`, `==` against an expected value, approx

A test whose assertions are *all* shape is reported. That is not automatically a bug — some
tests legitimately only check plumbing — so output is ranked by whether the function under
test sounds like something that can return empty (find/select/detect/compute/build...).

Usage:
    python benchmarks/audit_test_assertions.py            # summary + ranked list
    python benchmarks/audit_test_assertions.py --all      # every shape-only test
"""

import ast
import pathlib
import sys
from collections import defaultdict

TESTS = pathlib.Path(__file__).resolve().parent.parent / "tests"

SHAPE_FUNCS = {"isinstance", "type", "hasattr", "len", "callable", "id"}
SHAPE_ATTRS = {"shape", "dtype", "ndim", "size", "nbytes"}
VALUE_FUNCS = {
    "sum",
    "any",
    "all",
    "min",
    "max",
    "mean",
    "count",
    "std",
    "argmax",
    "argmin",
    "nonzero",
    "unique",
    "allclose",
    "approx",
    "isclose",
    "array_equal",
    "sorted",
    "set",
}
# functions whose whole job is to return something; empty output is a plausible silent failure
DETECTOR_HINTS = (
    "find",
    "select",
    "detect",
    "compute",
    "build",
    "search",
    "cluster",
    "extract",
    "mine",
    "discover",
    "label",
    "diversify",
    "rerank",
    "get",
)


def _is_len_call(node) -> bool:
    return isinstance(node, ast.Call) and getattr(node.func, "id", "") == "len"


def _is_shape_expr(node) -> bool:
    """`len(x)`, `x.shape`, `x.dtype`, `x.ndim` — an expression that measures, not inspects."""
    return _is_len_call(node) or (isinstance(node, ast.Attribute) and node.attr in SHAPE_ATTRS)


def _literal(node):
    """The literal number behind a Constant, or None. Bools are not numbers here."""
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
        return node.value
    return None


def _classify_compare(n: ast.Compare) -> str | None:
    """'vacuous' for a comparison every degenerate result satisfies; else 'value' or None.

    BLIND SPOT 1. These read as content assertions and are counted as such, but constrain
    nothing. Both shapes were live in this repo:

        assert result.n_components >= 0        # counts are non-negative by construction
        assert len(result.indices) <= 50       # an upper bound; an empty result passes

    A strict `> 0` is NOT vacuous — that is a genuine emptiness check — so the distinction
    is exactly `>=` versus `>`.
    """
    left = n.left
    for op, comp in zip(n.ops, n.comparators):
        lit = _literal(comp)
        if isinstance(op, ast.GtE) and lit == 0:
            return "vacuous"
        if isinstance(op, (ast.LtE, ast.Lt)) and _is_len_call(left) and lit is not None:
            return "vacuous"
        if isinstance(op, (ast.In, ast.NotIn, ast.Lt, ast.Gt, ast.LtE, ast.GtE)):
            # len(x) > 0 is a genuine emptiness check
            return "value"
        if isinstance(op, (ast.Eq, ast.NotEq)):
            # An equality tying one MEASURED SIZE to another constrains no content:
            #   len(out) == len(inp)          a length echo
            #   result.n_nodes == len(inp)    an attribute echoing the input length
            # Both are satisfied by a completely empty result of the right length. But an
            # equality against a literal pins an exact count (`len(x) == 10`), and one
            # between two ordinary expressions pins content (`path == [node]`).
            # `x.shape == (20, 64)` / `x.dtype == np.float32` pin a measurement, not content,
            # however literal the right-hand side looks.
            if isinstance(left, ast.Attribute) and left.attr in SHAPE_ATTRS:
                left = comp
                continue
            if isinstance(comp, ast.Attribute) and comp.attr in SHAPE_ATTRS:
                left = comp
                continue
            # `len(out) == len(inp)` and `r.n_nodes == len(inp)` are size echoes: satisfied by
            # a completely empty result of the right length. `len(x) == 10` is different — it
            # pins an exact count, so it does prove non-emptiness.
            if any((_is_shape_expr(left), _is_shape_expr(comp))) and _literal(left) is None and _literal(comp) is None:
                left = comp
                continue
            return "value"
        if isinstance(comp, ast.Constant) and not isinstance(comp.value, bool):
            return "value"
        left = comp
    return None


def classify_assert(node: ast.Assert) -> str:
    """'value' if any sub-expression constrains content, 'vacuous' if it only appears to,
    else 'shape'."""
    verdict = "shape"
    for n in ast.walk(node.test):
        if isinstance(n, ast.Call):
            fn = n.func
            name = getattr(fn, "id", None) or getattr(fn, "attr", None)
            if name in VALUE_FUNCS:
                return "value"
            if name not in SHAPE_FUNCS and name is not None:
                # a call we do not recognise as shape-ish -- treat as value-bearing
                verdict = "value"
        elif isinstance(n, ast.Attribute) and n.attr in VALUE_FUNCS:
            return "value"
        elif isinstance(n, ast.Compare):
            v = _classify_compare(n)
            if v is not None:
                return v
        elif isinstance(n, ast.Subscript):
            verdict = "value" if verdict == "value" else verdict
    return verdict


def implicit_assertions(node: ast.FunctionDef) -> int:
    """Count assertions that are not `assert` statements.

    Two blind spots that made the first version of this scanner cry wolf on 26 tests:
    `with pytest.raises(...)` — where the context manager *is* the assertion — and
    `np.testing.assert_*` / `assert_frame_equal`-style calls, which are function calls rather
    than `ast.Assert` nodes. Both constrain behaviour, so both count.
    """
    n = 0
    for sub in ast.walk(node):
        if isinstance(sub, (ast.With, ast.AsyncWith)):
            for item in sub.items:
                call = item.context_expr
                if isinstance(call, ast.Call):
                    name = getattr(call.func, "attr", None) or getattr(call.func, "id", None)
                    if name in {"raises", "warns", "deprecated_call"}:
                        n += 1
        elif isinstance(sub, ast.Call):
            name = getattr(sub.func, "attr", None) or getattr(sub.func, "id", None)
            if isinstance(name, str) and name.startswith("assert_"):
                n += 1
    return n


def guarded_asserts(node: ast.FunctionDef) -> tuple[int, int]:
    """(assertions sitting inside a conditional, assertions total).

    BLIND SPOT 2. `if result.chains:` / `if taxonomy.children:` wrapping the real assertions
    means a degenerate result SKIPS the test rather than failing it — the strongest possible
    assertions are worth nothing behind a guard the empty case never opens. Four tests here
    were written that way.

    A guard is sometimes legitimate (an optional dependency, a platform check), so this is
    reported separately rather than folded into shape-only. What is never legitimate is
    EVERY assertion in a test being guarded: such a test cannot fail on an empty result.
    Fix is one line — assert the guard condition immediately before the block.
    """
    total = guarded = 0

    def walk(body, inside: bool) -> None:
        nonlocal total, guarded
        for stmt in body:
            if isinstance(stmt, ast.Assert):
                total += 1
                guarded += int(inside)
                continue
            if isinstance(stmt, ast.If):
                # `if True:` hides nothing
                const = isinstance(stmt.test, ast.Constant)
                walk(stmt.body, inside or not const)
                walk(stmt.orelse, inside)
                continue
            for field in ("body", "orelse", "finalbody"):
                sub = getattr(stmt, field, None)
                if isinstance(sub, list):
                    walk(sub, inside)

    walk(node.body, False)
    return guarded, total


#: A test whose NAME says the empty result is the expected one has stated its intent already.
EMPTY_INTENT = ("empty", "no_match", "no_result", "none", "missing", "skipped", "absent", "not_found")


def _declares_empty(test_name: str) -> bool:
    return any(h in test_name.lower() for h in EMPTY_INTENT)


def asserts_only_emptiness(asserts: list[ast.Assert]) -> bool:
    """Every assertion in the test says the result is EMPTY.

    BLIND SPOT 3. Sometimes correct — a detector run on pure noise SHOULD find nothing, and
    saying so explicitly is better than not testing it. But an empty-only test is
    indistinguishable from one whose subject is simply broken, so the intent has to be
    stated. Flagged to force that: either add an assertion that the function works on data
    that does have structure, or say in the test why empty is the right answer.
    """
    if not asserts:
        return False
    for a in asserts:
        t = a.test
        if not (isinstance(t, ast.Compare) and len(t.ops) == 1 and isinstance(t.ops[0], ast.Eq)):
            return False
        comp = t.comparators[0]
        if isinstance(comp, (ast.List, ast.Set, ast.Tuple)) and not comp.elts:
            continue
        if isinstance(comp, ast.Dict) and not comp.keys:
            continue
        if _literal(comp) == 0 and _is_len_call(t.left):
            continue
        return False
    return True


#: Hand-labelled cases. A classifier reporting "N problems" is worth nothing until it is shown
#: to separate the cases it claims to — this repo learned that from `audit_public_api.py`,
#: whose canary caught a bug in the harness itself. Every entry here is a real line, or the
#: minimal form of one, from `tests/`.
SELFTEST_ASSERTS = [
    # --- shape: true of a completely empty result of the right length -------------------
    ("assert isinstance(r.indices, np.ndarray)", "shape"),
    ("assert len(r.centrality) == len(emb)", "shape"),
    ("assert r.n_nodes == len(emb)", "shape"),
    ("assert r.ontology.n_nodes == len(clustered)", "shape"),
    ("assert isinstance(summary, str)", "shape"),
    ("assert init.shape == (20, emb.shape[1])", "shape"),
    ("assert init.dtype == np.float32", "shape"),
    ("assert result.buckets_covered == len(result)", "shape"),
    # --- vacuous: reads as a content check, constrains nothing --------------------------
    ("assert r.n_components >= 0", "vacuous"),
    ("assert len(r.parent_child_edges) >= 0", "vacuous"),
    ("assert len(r.indices) <= 50", "vacuous"),
    ("assert layer.depth >= 0", "vacuous"),
    # --- value: genuinely constrains content --------------------------------------------
    ("assert len(r.indices) > 0", "value"),
    ("assert r.centrality.sum() > 0", "value"),
    ("assert (r.quadrant != 'Regular').any()", "value"),
    ("assert path == [node]", "value"),
    ("assert single_id == all_ids[0]", "value"),
    ("assert len(indices) == 10", "value"),
    ("assert e < 0.3", "value"),
    ("assert q in valid_quadrants", "value"),
    ("assert 0 <= c < taxonomy.n_nodes", "value"),
    ("assert np.all(d >= 0)", "value"),
    ("assert diversity.std() > 0", "value"),
    ("assert result.meta_clusters == {0}", "value"),
]

SELFTEST_FUNCS = [
    (
        "def test_guarded():\n"
        "    r = f()\n"
        "    if r.chains:\n"
        "        assert len(r.chains) > 0\n"
        "        assert r.chains[0].coherence > 0\n",
        "all-guarded",
    ),
    (
        "def test_partly_guarded():\n"
        "    r = f()\n"
        "    assert len(r.chains) > 0\n"
        "    if r.chains:\n"
        "        assert r.chains[0].coherence > 0\n",
        None,
    ),
    ("def test_thing():\n    kw = f()\n    assert kw[1] == []\n", "empty-only"),
    ("def test_empty_cluster():\n    kw = f()\n    assert kw[1] == []\n", None),  # name states intent
    (
        "def test_thing():\n    kw = f()\n    assert kw[1] == []\n    assert kw[0]\n",
        None,  # also asserts the positive case
    ),
]


def selftest() -> int:
    """Validate the classifier against hand-labelled cases. Returns exit code."""
    bad = 0
    print("assertion-level")
    for src, want in SELFTEST_ASSERTS:
        node = ast.parse(src).body[0]
        assert isinstance(node, ast.Assert), src
        got = classify_assert(node)
        ok = got == want
        bad += not ok
        print(f"  [{'ok' if ok else 'FAIL'}] want={want:<8} got={got:<8} {src}")

    print("\ntest-level")
    for src, want in SELFTEST_FUNCS:
        fn = ast.parse(src).body[0]
        asserts = [x for x in ast.walk(fn) if isinstance(x, ast.Assert)]
        n_guarded, n_total = guarded_asserts(fn)
        if n_total and n_guarded == n_total:
            got = "all-guarded"
        elif asserts_only_emptiness(asserts) and not _declares_empty(fn.name):
            got = "empty-only"
        else:
            got = None
        ok = got == want
        bad += not ok
        first = src.strip().splitlines()[0]
        print(f"  [{'ok' if ok else 'FAIL'}] want={str(want):<12} got={str(got):<12} {first}")

    print(f"\n{len(SELFTEST_ASSERTS) + len(SELFTEST_FUNCS) - bad} passed, {bad} failed")
    if bad:
        print("The classifier does not separate the cases it claims to. Fix before believing")
        print("any count it prints.")
    return 1 if bad else 0


def main() -> int:
    if "--selftest" in sys.argv:
        return selftest()
    show_all = "--all" in sys.argv
    per_file = defaultdict(list)
    totals = {
        "tests": 0,
        "shape_only": 0,
        "no_assert": 0,
        "implicit_only": 0,
        "vacuous": 0,
        "guarded": 0,
        "empty_only": 0,
    }

    for path in sorted(TESTS.glob("test_*.py")):
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if not isinstance(node, ast.FunctionDef) or not node.name.startswith("test"):
                continue
            asserts = [n for n in ast.walk(node) if isinstance(n, ast.Assert)]
            implicit = implicit_assertions(node)
            totals["tests"] += 1
            if not asserts:
                if implicit:
                    # pytest.raises / np.testing.assert_* — genuinely behavioural
                    totals["implicit_only"] += 1
                else:
                    totals["no_assert"] += 1
                    per_file[path.name].append((node.name, node.lineno, 0, "NO ASSERT"))
                continue
            kinds = {classify_assert(a) for a in asserts}
            n_guarded, n_total = guarded_asserts(node)

            # Priority order: report the weakest thing true of the test, once.
            kind = None
            if kinds == {"shape"} and implicit == 0:
                kind = "shape-only"
                totals["shape_only"] += 1
            elif kinds <= {"shape", "vacuous"} and implicit == 0:
                kind = "vacuous-only"
                totals["vacuous"] += 1
            elif implicit == 0 and n_total and n_guarded == n_total:
                kind = "all-guarded"
                totals["guarded"] += 1
            elif asserts_only_emptiness(asserts) and not _declares_empty(node.name):
                kind = "empty-only"
                totals["empty_only"] += 1
            if kind:
                per_file[path.name].append((node.name, node.lineno, len(asserts), kind))

    n = max(totals["tests"], 1)
    weak = totals["shape_only"] + totals["vacuous"] + totals["guarded"] + totals["no_assert"]
    print(f"{totals['tests']} test functions scanned")
    print(f"  {totals['shape_only']} assert SHAPE only  ({100 * totals['shape_only'] / n:.0f}%)")
    print(f"  {totals['vacuous']} assert only VACUOUSLY  (x >= 0, len(x) <= k — an empty result passes)")
    print(f"  {totals['guarded']} have EVERY assertion behind an `if` guard (empty result skips, not fails)")
    print(f"  {totals['no_assert']} assert NOTHING at all")
    print(f"  {weak} weak in total  ({100 * weak / n:.0f}%)")
    print(f"  {totals['empty_only']} assert only that the result is EMPTY (correct sometimes — say so)")
    print(f"  {totals['implicit_only']} assert only via pytest.raises / assert_* (fine, counted as behavioural)")

    def risky(name: str) -> bool:
        return any(h in name.lower() for h in DETECTOR_HINTS)

    print("\nRanked: shape-only tests of functions that return a detection/selection")
    print("(an empty or all-zero result would pass these)\n")
    print(f"{'file':<28}{'line':>6}  test")
    shown = 0
    for fname, items in sorted(per_file.items()):
        for tname, lineno, n_asserts, kind in items:
            if not show_all and not risky(tname):
                continue
            print(f"{fname:<28}{lineno:>6}  {tname}  [{n_asserts} asserts, {kind}]")
            shown += 1
    if not shown:
        print("  (none)")
    print(f"\n{shown} shown. Re-run with --all to list every shape-only test.")
    print("\nA shape-only test is not automatically wrong — but for anything that DETECTS or")
    print("SELECTS, add an assertion that it found something, not merely that it returned an")
    print("array of the right length.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
