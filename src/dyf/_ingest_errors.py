"""Failure types shared by the `dyf index-*` commands, and the exit-code contract.

These exist so ingest failures are *reportable* rather than fatal. The three index
modules previously called `sys.exit(1)` from inside library functions — `SystemExit`
derives from `BaseException`, so it bypasses normal handling and terminates the
interpreter of anyone who imported them, which is not a library's decision to make.
Every distinct failure also collapsed to exit code 1, so a caller could not tell
"you pointed me at the wrong directory" from "the corpus was empty" from "the embedding
service is down".

Exit codes, matching `dyf concepts`:

    0  success
    1  a normal negative answer — nothing to index (EmptyIngestError)
    2  the request was wrong — bad path, bad argument (BadIngestRequest)
    3  a dependency or service is unavailable (EmbeddingServiceError, ImportError)

Every message is meant to be shown to the caller verbatim, and should say what to do
rather than only what went wrong.
"""

from __future__ import annotations

#: Nothing to index. A negative answer, not a malfunction — grep's convention.
EXIT_EMPTY = 1
#: The request itself was wrong: a path that is not a directory, an unusable argument.
EXIT_BAD_REQUEST = 2
#: A dependency or external service is missing or unreachable.
EXIT_UNAVAILABLE = 3


class IngestError(Exception):
    """Base for ingest failures whose message is fit to show a caller as-is."""

    exit_code = EXIT_BAD_REQUEST


class EmptyIngestError(IngestError):
    """The source exists and is readable, but yielded nothing indexable.

    Distinct from a bad request: the caller did nothing wrong, there is simply no output.
    """

    exit_code = EXIT_EMPTY


class BadIngestRequest(IngestError):
    """The inputs were unusable — a missing directory, an out-of-range argument."""

    exit_code = EXIT_BAD_REQUEST


class EmbeddingServiceError(IngestError):
    """An embedding backend is unreachable or cannot serve the requested model.

    A *service* dependency rather than an import one, so `cli.main`'s ImportError handler
    does not cover it — hence its own type and exit code.
    """

    exit_code = EXIT_UNAVAILABLE
