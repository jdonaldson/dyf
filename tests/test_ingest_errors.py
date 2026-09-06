"""The `dyf index-*` failure contract.

Three things are asserted, each of which was broken before 2026-09-05:

1. Ingest failures **raise** rather than calling `sys.exit`. The index modules used to
   call `sys.exit(1)` from inside library functions; `SystemExit` derives from
   `BaseException`, so it bypasses normal handling and kills the interpreter of anyone
   who imported them.
2. Exit codes **distinguish** the failures. Everything used to be 1, so a caller could
   not tell a wrong path from an empty corpus from a dead embedding service.
3. A missing *service* fails **before** the expensive work, with a message rather than a
   traceback. Measured before: a 69-line `requests.ConnectionError` that never mentioned
   Ollama by name, raised only after the whole source tree had been parsed.
"""

from __future__ import annotations

import subprocess
import sys

import pytest

from dyf._ingest_errors import (
    EXIT_BAD_REQUEST,
    EXIT_EMPTY,
    EXIT_UNAVAILABLE,
    BadIngestRequest,
    EmbeddingServiceError,
    EmptyIngestError,
    IngestError,
)

UNREACHABLE = "http://localhost:59999/api/embed"


def _run(*args):
    return subprocess.run(
        [sys.executable, "-m", "dyf.cli", *args],
        capture_output=True,
        text=True,
        timeout=120,
    )


class TestErrorTypes:
    def test_all_are_ordinary_exceptions(self):
        """Not SystemExit — a library must not terminate its caller's interpreter."""
        for cls in (IngestError, EmptyIngestError, BadIngestRequest, EmbeddingServiceError):
            assert issubclass(cls, Exception)
            assert not issubclass(cls, BaseException) or issubclass(cls, Exception)
            assert not issubclass(cls, SystemExit)

    def test_exit_codes_are_distinct(self):
        codes = {
            EmptyIngestError.exit_code,
            BadIngestRequest.exit_code,
            EmbeddingServiceError.exit_code,
        }
        assert codes == {EXIT_EMPTY, EXIT_BAD_REQUEST, EXIT_UNAVAILABLE}
        assert len(codes) == 3, "a caller must be able to tell these apart"

    def test_all_inherit_from_ingest_error(self):
        """`main` catches the base class, so a new subclass is handled automatically."""
        for cls in (EmptyIngestError, BadIngestRequest, EmbeddingServiceError):
            assert issubclass(cls, IngestError)


class TestEmbeddingPreflight:
    def test_unreachable_service_raises_with_a_remedy(self):
        from dyf.index_source import check_embedding_service

        with pytest.raises(EmbeddingServiceError) as exc:
            check_embedding_service(ollama_url=UNREACHABLE, timeout=2.0)

        message = str(exc.value)
        assert UNREACHABLE in message, "must name the URL it tried"
        assert "ollama serve" in message, "must say how to fix it"

    def test_embed_batch_wraps_connection_failure(self):
        """Even a mid-run failure must not surface as a bare requests traceback."""
        from dyf.index_source import embed_batch

        with pytest.raises(EmbeddingServiceError) as exc:
            embed_batch(["hello"], ollama_url=UNREACHABLE)
        assert "0/1" in str(exc.value), "should report how far it got"


class TestCliExitCodes:
    def test_bad_directory_exits_2(self, tmp_path):
        for cmd in ("index-source", "index-images"):
            p = _run(cmd, str(tmp_path / "absent"), "-o", str(tmp_path / "x.dyf"))
            assert p.returncode == EXIT_BAD_REQUEST, f"{cmd} should exit {EXIT_BAD_REQUEST}"
            assert "not a directory" in p.stderr
            assert "Traceback" not in p.stderr

    def test_missing_video_file_exits_2(self, tmp_path):
        p = _run("index-video", str(tmp_path / "absent.mp4"), "-o", str(tmp_path / "x.dyf"))
        assert p.returncode == EXIT_BAD_REQUEST
        assert "not a file" in p.stderr

    def test_unreachable_service_exits_3_without_a_traceback(self, tmp_path):
        """The regression test for the 69-line ConnectionError."""
        src = tmp_path / "src"
        src.mkdir()
        p = _run(
            "index-source",
            str(src),
            "-o",
            str(tmp_path / "x.dyf"),
            "--ollama-url",
            UNREACHABLE,
        )
        assert p.returncode == EXIT_UNAVAILABLE
        assert "Traceback" not in p.stderr
        assert "cannot reach the embedding service" in p.stderr
        assert "ollama serve" in p.stderr

    def test_errors_go_to_stderr_not_stdout(self, tmp_path):
        p = _run("index-source", str(tmp_path / "absent"), "-o", str(tmp_path / "x.dyf"))
        assert "not a directory" in p.stderr
        assert "not a directory" not in p.stdout
