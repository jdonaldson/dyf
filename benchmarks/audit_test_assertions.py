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


def classify_assert(node: ast.Assert) -> str:
    """'value' if any sub-expression constrains content, else 'shape'."""
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
            for op, comp in zip(n.ops, n.comparators):
                if isinstance(op, (ast.In, ast.NotIn, ast.Lt, ast.Gt, ast.LtE, ast.GtE)):
                    # len(x) > 0 is a genuine emptiness check
                    return "value"
                if isinstance(comp, ast.Constant) and not isinstance(comp.value, bool):
                    left = n.left
                    is_len = isinstance(left, ast.Call) and getattr(left.func, "id", "") == "len"
                    if not is_len:
                        return "value"
                    # len(x) == <literal> pins an exact count: weakly behavioural
                    return "value"
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


def main() -> int:
    show_all = "--all" in sys.argv
    per_file = defaultdict(list)
    totals = {"tests": 0, "shape_only": 0, "no_assert": 0, "implicit_only": 0}

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
            if kinds == {"shape"} and implicit == 0:
                totals["shape_only"] += 1
                per_file[path.name].append((node.name, node.lineno, len(asserts), "shape-only"))

    print(f"{totals['tests']} test functions scanned")
    print(f"  {totals['shape_only']} assert SHAPE only  ({100 * totals['shape_only'] / max(totals['tests'], 1):.0f}%)")
    print(f"  {totals['no_assert']} assert NOTHING at all")
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
