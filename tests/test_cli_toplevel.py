"""Tests for `dyf`'s top-level argv handling.

This file exists because 0.13.0 shipped with `dyf --help` exiting 1. The release's whole
premise was that an agent can depend on the CLI's exit codes, and the very first thing
any caller runs returned failure.

Nothing caught it. There were no CLI tests, and `benchmarks/audit_cli_surface.py`
documents a deliberate decision not to test `--help` — sound for subcommands, where
argparse answers help itself without touching the logger, so a help smoke test cannot
tell a working command from a mute one. The flaw was generalising that to the top level,
where there is no argparse at all and help is hand-rolled.

So these tests cover exactly the layer the audit does not: the four cases `main()` used
to collapse into a single usage dump with a single exit code.
"""

import subprocess
import sys

import pytest


def run_cli(*args: str) -> subprocess.CompletedProcess:
    """Invoke the CLI the way a caller does — a real process, real streams.

    In-process (`main()` under `pytest.raises(SystemExit)`) would not catch a stream
    mistake, because pytest's capture replaces sys.stdout/sys.stderr before the handlers
    are built.
    """
    return subprocess.run(
        [sys.executable, "-m", "dyf.cli", *args],
        capture_output=True,
        text=True,
    )


class TestHelp:
    @pytest.mark.parametrize("flag", ["-h", "--help", "help"])
    def test_help_succeeds(self, flag):
        """Asking for help is a successful request, not a usage error."""
        r = run_cli(flag)
        assert r.returncode == 0, f"`dyf {flag}` exited {r.returncode}: {r.stderr}"

    @pytest.mark.parametrize("flag", ["-h", "--help", "help"])
    def test_help_answers_on_stdout(self, flag):
        """Help is the answer to the question asked, so it belongs on stdout.

        A caller doing `dyf --help > cheatsheet.txt` must not get an empty file.
        """
        r = run_cli(flag)
        assert "Usage: dyf" in r.stdout
        assert r.stderr == ""

    def test_help_lists_every_dispatchable_command(self):
        """Guards against a command being added to dispatch but not to the usage text."""
        from dyf.cli import COMMAND_EXTRAS

        out = run_cli("--help").stdout
        missing = [c for c in COMMAND_EXTRAS if c not in out]
        assert not missing, f"dispatchable but absent from --help: {missing}"

    def test_help_points_at_per_command_help(self):
        """The top-level list is not enough to actually run anything."""
        assert "--help" in run_cli("--help").stdout


class TestUsageErrors:
    def test_no_command_is_a_usage_error(self):
        r = run_cli()
        assert r.returncode == 2

    def test_no_command_writes_usage_to_stderr(self):
        """stdout is reserved for answers; there is no answer here."""
        r = run_cli()
        assert r.stdout == ""
        assert "Usage: dyf" in r.stderr

    def test_unknown_command_is_a_usage_error(self):
        r = run_cli("frobnicate")
        assert r.returncode == 2

    def test_unknown_command_names_the_offending_word(self):
        """Dumping usage alone makes the caller diff it against what they typed."""
        r = run_cli("frobnicate")
        assert "frobnicate" in r.stderr

    def test_unknown_command_is_distinguishable_from_no_command(self):
        """Both are rc 2, so the *message* has to carry the difference."""
        assert "unknown command" not in run_cli().stderr
        assert "unknown command" in run_cli("frobnicate").stderr


class TestMovedCommands:
    """The moved-command redirect must stay distinguishable from a typo.

    `dyf enrich` is not an unknown command — it is a command that exists somewhere else,
    and it has its own exit code so a script can tell "you need to install dyfviz" from
    "you misspelled something".
    """

    @pytest.mark.parametrize("cmd", ["enrich", "tour"])
    def test_moved_command_has_its_own_exit_code(self, cmd):
        r = run_cli(cmd)
        assert r.returncode == 4

    @pytest.mark.parametrize("cmd", ["enrich", "tour"])
    def test_moved_command_names_the_replacement(self, cmd):
        r = run_cli(cmd)
        assert "dyfviz" in r.stderr

    def test_moved_command_is_not_reported_as_unknown(self):
        assert "unknown command" not in run_cli("enrich").stderr


class TestExitCodesAreDistinct:
    """The contract is only useful if the codes actually differ from each other."""

    def test_help_success_differs_from_usage_error(self):
        assert run_cli("--help").returncode != run_cli("frobnicate").returncode

    def test_moved_differs_from_unknown(self):
        assert run_cli("enrich").returncode != run_cli("frobnicate").returncode
