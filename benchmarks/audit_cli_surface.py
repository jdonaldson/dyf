#!/usr/bin/env python3
"""Audit the CLI surface: does every subcommand actually say something?

The fourth standing audit. The other three inspect the exported Python API —
`audit_public_api.py`, `audit_test_assertions.py`, `audit_absolute_thresholds.py` — and
all three passed cleanly while `dyf concepts list` printed **zero bytes** on a graph with
100+ nodes. `cli.py` is not in the public API, so nothing looked at it.

**Why this audit does not test `--help` on subcommands.** It would not have caught the
bug. argparse prints help itself, writing straight to stdout without ever touching the
logger, so every subcommand answers `--help` with rc 0 even when its real execution path
is mute. Measured during this audit's construction: all 7 subcommands returned rc 0 and
non-empty output for `--help`, while 3 of them raised an unhandled traceback on a real
invocation. A help-only smoke test is blind to the failure it appears to check for.

⚠ **That reasoning holds only where argparse is doing the parsing, and it was
over-generalised into "do not test --help" — which cost a release.** The *top level* has
no argparse: `main()` hand-rolls its usage text, and in 0.13.0 it routed `--help`, `-h`,
an unknown command and no command at all into one usage dump with one exit code (1). So
`dyf --help` — the first thing any caller runs — exited 1 on the release whose stated
premise was that agents can trust these exit codes. Caught by installing the published
wheel and running it, not by any check in the repo.

`check_top_level_contract()` below now covers that layer, and `tests/test_cli_toplevel.py`
covers it in CI. The transferable lesson: an exemption earns its scope from the mechanism
that justifies it. "argparse handles help" justifies skipping *argparse* commands, and
nothing more.

So each case here runs a **real invocation** and classifies what came back:

    OK          exit 0 with output — the command answered
    MUTE        exit 0 with NO output — the bug class this audit exists for
    TRACEBACK   nonzero exit with a Python traceback — a failure nobody can act on
    CLEAN-FAIL  nonzero exit with a message and no traceback — acceptable
    SKIP        no invocation could be constructed here

MUTE and TRACEBACK are the two verdicts that fail the run. CLEAN-FAIL is a pass: a
command that cannot run because an optional dependency is absent is expected to say so.

Run `--selftest` to check the classifier against hand-labelled cases plus a live canary
that is deliberately mute. That habit comes from `audit_test_assertions.py`, whose
scanner turned out to have a 32% false-positive rate and had inflated every number it had
ever reported — a classifier is itself a derived result and needs its own ground truth.

Usage:
    python benchmarks/audit_cli_surface.py
    python benchmarks/audit_cli_surface.py --selftest
    python benchmarks/audit_cli_surface.py --json
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
TIMEOUT = 120

OK = "OK"
MUTE = "MUTE"
TRACEBACK = "TRACEBACK"
CLEAN_FAIL = "CLEAN-FAIL"
SKIP = "SKIP"

FAILING_VERDICTS = {MUTE, TRACEBACK}


@dataclass
class Result:
    name: str
    argv: list[str]
    verdict: str
    returncode: int | None = None
    stdout_bytes: int = 0
    stderr_bytes: int = 0
    note: str = ""


def classify(returncode: int, stdout: str, stderr: str) -> tuple[str, str]:
    """Classify one invocation. Returns (verdict, note).

    Kept pure and separate from the running so `--selftest` can exercise it directly.
    """
    combined = stdout + stderr
    has_traceback = "Traceback (most recent call last)" in combined

    if returncode == 0:
        if stdout.strip():
            return OK, ""
        if stderr.strip():
            # Exited successfully but spoke only on stderr — a caller redirecting
            # stdout gets an empty file.
            return MUTE, "exit 0, but output went to stderr only"
        return MUTE, "exit 0 with no output at all"

    if has_traceback:
        return TRACEBACK, "unhandled exception — no actionable message"
    if combined.strip():
        return CLEAN_FAIL, "failed with a message"
    return MUTE, f"exit {returncode} with no output at all"


@dataclass
class Fixtures:
    """Throwaway inputs the real invocations need."""

    tmpdir: str
    dyf_path: str | None = None
    srcdir: str | None = None
    notes: list[str] = field(default_factory=list)

    @classmethod
    def build(cls) -> Fixtures:
        tmp = tempfile.mkdtemp(prefix="dyf-cli-audit-")
        fx = cls(tmpdir=tmp)

        srcdir = Path(tmp, "src")
        srcdir.mkdir()
        Path(srcdir, "sample.py").write_text("def hello():\n    return 1\n")
        fx.srcdir = str(srcdir)

        try:
            import numpy as np

            from dyf.dyf_tree import build_dyf_tree
            from dyf.lazy_index import write_lazy_index

            rng = np.random.default_rng(0)
            X = np.ascontiguousarray(rng.standard_normal((48, 16)).astype(np.float32))
            tree = build_dyf_tree(X, max_depth=3, num_bits=3, min_leaf_size=2, seed=42)
            path = str(Path(tmp, "fixture.dyf"))
            write_lazy_index(
                tree,
                X,
                path,
                stored_fields={"title": [f"item-{i}" for i in range(len(X))]},
                metadata={"domain": "cli_audit_fixture"},
            )
            fx.dyf_path = path
        except Exception as exc:  # noqa: BLE001 — audit must survive any fixture failure
            fx.notes.append(f"could not build a .dyf fixture ({type(exc).__name__}: {exc})")

        return fx


def build_cases(fx: Fixtures) -> list[tuple[str, list[str] | None]]:
    """The real invocation for each subcommand, or None if it cannot be constructed."""
    out_dyf = str(Path(fx.tmpdir, "out.dyf"))
    return [
        ("info", ["info", fx.dyf_path] if fx.dyf_path else None),
        ("info --json", ["info", fx.dyf_path, "--json"] if fx.dyf_path else None),
        ("api", ["api"]),
        ("api --json", ["api", "--json"]),
        ("api <group>", ["api", "trees"]),
        ("concepts check", ["concepts", "check"]),
        ("concepts list", ["concepts", "list"]),
        ("concepts query", ["concepts", "query", "critical rules"]),
        # --dry-run must work without the optional deps the real run needs; that is much
        # of its value, so it is audited separately from the real invocation.
        ("index-source --dry-run", ["index-source", fx.srcdir, "-o", out_dyf, "--dry-run"]),
        ("index-images --dry-run", ["index-images", fx.srcdir, "-o", out_dyf, "--dry-run"]),
        ("index-source", ["index-source", fx.srcdir, "-o", out_dyf]),
        ("index-images", ["index-images", fx.srcdir, "-o", out_dyf]),
        ("index-video", ["index-video", str(Path(fx.tmpdir, "absent.mp4")), "-o", out_dyf]),
        # `enrich` and `tour` moved to dyfviz on 2026-09-05. dyf keeps a redirect for
        # them, and that redirect is worth auditing: it is what an old command line or a
        # stale Makefile target hits.
        ("enrich (moved)", ["enrich", "all", "x.dyf"]),
        ("tour (moved)", ["tour", "x.dyf"]),
    ]


def run_case(name: str, argv: list[str] | None) -> Result:
    if argv is None:
        return Result(name=name, argv=[], verdict=SKIP, note="no non-interactive invocation")
    try:
        proc = subprocess.run(
            [sys.executable, "-m", "dyf.cli", *argv],
            capture_output=True,
            text=True,
            timeout=TIMEOUT,
            cwd=REPO,
        )
    except subprocess.TimeoutExpired:
        return Result(name=name, argv=argv, verdict=SKIP, note=f"timed out after {TIMEOUT}s")

    verdict, note = classify(proc.returncode, proc.stdout, proc.stderr)
    return Result(
        name=name,
        argv=argv,
        verdict=verdict,
        returncode=proc.returncode,
        stdout_bytes=len(proc.stdout),
        stderr_bytes=len(proc.stderr),
        note=note,
    )


def check_dispatch_matches_help() -> list[str]:
    """Every advertised command should be wired, and vice versa.

    A command listed in the usage text but missing from the dispatch chain silently
    falls through to the usage message — it looks like a typo rather than a bug.

    Reads the **rendered** usage text rather than scraping `print(...)` calls out of the
    source. The scraping version silently stopped working the moment the usage text was
    refactored from a run of `print()` calls into a list — it then found zero advertised
    commands and would have reported every command as unadvertised. Rendering the real
    thing has no such coupling to how the text happens to be emitted.
    """
    import io

    from dyf.cli import _print_usage

    cli_src = Path(REPO, "src", "dyf", "cli.py").read_text()

    dispatched = set()
    for line in cli_src.splitlines():
        stripped = line.strip()
        if stripped.startswith(("if cmd ==", "elif cmd ==")):
            dispatched.add(stripped.split('"')[1])

    buf = io.StringIO()
    _print_usage(buf)
    advertised = set()
    for line in buf.getvalue().splitlines():
        # Command rows are indented two spaces under "Commands:"; prose lines are not.
        if line.startswith("  ") and not line.startswith("   "):
            token = line.strip().split()[0] if line.strip() else ""
            if token:
                advertised.add(token)

    problems = []
    if not advertised:
        problems.append("parsed no commands out of the usage text — this check is broken")
    for cmd in sorted(advertised - dispatched):
        problems.append(f"advertised in help but not dispatched: {cmd}")
    for cmd in sorted(dispatched - advertised):
        problems.append(f"dispatched but not advertised in help: {cmd}")
    return problems


def check_top_level_contract() -> list[str]:
    """The four top-level argv cases, which have no argparse behind them.

    `main()` hand-rolls this layer, so the reasoning that lets this audit skip `--help`
    for subcommands does not reach it. In 0.13.0 all four cases below shared one usage
    dump and one exit code, so `dyf --help` exited 1.

    Asserted here as a *contract*, not as "whatever it does today":

        dyf --help        rc 0, usage on stdout   (a question that was answered)
        dyf              rc 2, usage on stderr    (a usage error, not an answer)
        dyf <unknown>     rc 2, names the word     (distinguishable from the above)
        dyf enrich        rc 4, names dyfviz       (moved, not mistyped)
    """
    problems: list[str] = []

    def run(*args):
        return subprocess.run(
            [sys.executable, "-m", "dyf.cli", *args],
            capture_output=True,
            text=True,
            timeout=60,
        )

    for flag in ("-h", "--help", "help"):
        r = run(flag)
        if r.returncode != 0:
            problems.append(f"`dyf {flag}` exited {r.returncode}, expected 0")
        if "Usage: dyf" not in r.stdout:
            problems.append(f"`dyf {flag}` did not put usage on stdout")

    r = run()
    if r.returncode != EXPECTED_USAGE_RC:
        problems.append(f"bare `dyf` exited {r.returncode}, expected {EXPECTED_USAGE_RC}")
    if "Usage: dyf" not in r.stderr:
        problems.append("bare `dyf` did not put usage on stderr")

    r = run("frobnicate")
    if r.returncode != EXPECTED_USAGE_RC:
        problems.append(f"`dyf frobnicate` exited {r.returncode}, expected {EXPECTED_USAGE_RC}")
    if "frobnicate" not in r.stderr:
        problems.append("an unknown command is not named in the error")

    r = run("enrich")
    if r.returncode != EXPECTED_MOVED_RC:
        problems.append(f"`dyf enrich` exited {r.returncode}, expected {EXPECTED_MOVED_RC}")
    if "dyfviz" not in r.stderr:
        problems.append("the moved-command redirect does not name dyfviz")

    return problems


#: A malformed request. Mirrors `dyf._ingest_errors.EXIT_BAD_REQUEST`, restated rather
#: than imported so the audit fails loudly if the contract is quietly renumbered.
EXPECTED_USAGE_RC = 2
#: A command that moved to dyfviz — deliberately distinct from "unknown".
EXPECTED_MOVED_RC = 4


# ---------------------------------------------------------------------------
# selftest
# ---------------------------------------------------------------------------

# (returncode, stdout, stderr) -> expected verdict. Hand-labelled.
SELFTEST_CASES = [
    ((0, "items 42\n", ""), OK, "normal success"),
    ((0, "", ""), MUTE, "the original bug: exit 0, nothing printed"),
    ((0, "", "done\n"), MUTE, "spoke only on stderr — redirect gets an empty file"),
    ((0, "   \n", ""), MUTE, "whitespace is not output"),
    ((1, "", "Traceback (most recent call last):\n  File x\n"), TRACEBACK, "unhandled"),
    ((1, "", "no such file: /x\n"), CLEAN_FAIL, "actionable failure"),
    ((2, "", "usage: dyf info\n"), CLEAN_FAIL, "argparse usage error"),
    ((1, "", ""), MUTE, "failed silently — worst case for a caller"),
    ((1, "partial\n", "Traceback (most recent call last):\n"), TRACEBACK, "output then crash"),
]


def run_selftest() -> int:
    failures = 0
    print("Classifier cases")
    for (rc, out, err), expected, label in SELFTEST_CASES:
        got, _ = classify(rc, out, err)
        ok = got == expected
        failures += not ok
        print(f"  {'pass' if ok else 'FAIL'}  {expected:<10} {label}" + ("" if ok else f"  (got {got})"))

    # Live canary: a subprocess that exits 0 saying nothing must be caught. This is the
    # exact shape of the shipped bug, so if the audit cannot flag it the audit is inert.
    print("\nLive canary (a deliberately mute command)")
    proc = subprocess.run(
        [sys.executable, "-c", "import sys; sys.exit(0)"],
        capture_output=True,
        text=True,
    )
    verdict, note = classify(proc.returncode, proc.stdout, proc.stderr)
    ok = verdict == MUTE
    failures += not ok
    print(f"  {'pass' if ok else 'FAIL'}  detected as {verdict} ({note})")

    print(f"\n{len(SELFTEST_CASES) + 1} checks, {failures} failed")
    return 1 if failures else 0


# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--selftest", action="store_true", help="check the classifier against known cases")
    parser.add_argument("--json", action="store_true", dest="as_json")
    args = parser.parse_args()

    if args.selftest:
        return run_selftest()

    fx = Fixtures.build()
    for note in fx.notes:
        print(f"note: {note}\n")

    results = [run_case(name, argv) for name, argv in build_cases(fx)]
    dispatch_problems = check_dispatch_matches_help()
    top_level_problems = check_top_level_contract()

    if args.as_json:
        print(
            json.dumps(
                {
                    "results": [r.__dict__ for r in results],
                    "dispatch_problems": dispatch_problems,
                    "top_level_problems": top_level_problems,
                },
                indent=2,
            )
        )
    else:
        print(f"{'command':<18} {'verdict':<11} {'rc':<5} {'stdout':<8} {'stderr':<8} note")
        print("-" * 88)
        for r in results:
            rc = "-" if r.returncode is None else str(r.returncode)
            print(f"{r.name:<18} {r.verdict:<11} {rc:<5} {r.stdout_bytes:<8} {r.stderr_bytes:<8} {r.note}")

        print("\nDispatch table vs advertised help")
        if dispatch_problems:
            for problem in dispatch_problems:
                print(f"  {problem}")
        else:
            print("  consistent")

        print("\nTop-level contract (--help / no command / unknown / moved)")
        if top_level_problems:
            for problem in top_level_problems:
                print(f"  {problem}")
        else:
            print("  holds")

        counts: dict[str, int] = {}
        for r in results:
            counts[r.verdict] = counts.get(r.verdict, 0) + 1
        print("\n" + "  ".join(f"{k}={v}" for k, v in sorted(counts.items())))

    failing = [r for r in results if r.verdict in FAILING_VERDICTS]
    if failing or dispatch_problems or top_level_problems:
        if not args.as_json:
            print(
                f"\n{len(failing)} command(s) mute or crashing; "
                f"{len(dispatch_problems)} dispatch problem(s); "
                f"{len(top_level_problems)} top-level contract problem(s)"
            )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
