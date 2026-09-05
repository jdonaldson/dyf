"""Tests for `dyf info` — the describe-without-loading command.

These are also the first tests of the CLI surface at all. Their absence is why a CLI
that printed literally nothing shipped and survived three standing audits: every audit
looks at the exported Python API, and `cli.py` is not in it. See
AGENT_LEGIBILITY_TODO.md.

The assertions here deliberately check *behaviour an agent depends on* — non-empty
output, results on stdout, distinguishable exit codes — rather than shapes and lengths.
A test asserting only `isinstance(result, dict)` would pass against the empty output
this command exists to prevent.
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("flatbuffers")


@pytest.fixture(scope="module")
def dyf_path():
    """A small but real .dyf file with stored fields."""
    pytest.importorskip("dyf_rs")
    from dyf.dyf_tree import build_dyf_tree
    from dyf.lazy_index import write_lazy_index

    rng = np.random.default_rng(7)
    X = np.ascontiguousarray(rng.standard_normal((64, 16)).astype(np.float32))
    tree = build_dyf_tree(X, max_depth=3, num_bits=3, min_leaf_size=2, seed=42)

    with tempfile.NamedTemporaryFile(suffix=".dyf", delete=False) as f:
        path = f.name
    write_lazy_index(
        tree,
        X,
        path,
        stored_fields={"title": [f"item-{i}" for i in range(len(X))]},
        metadata={"domain": "test_corpus"},
    )
    yield path
    Path(path).unlink(missing_ok=True)


def _run(*args):
    """Invoke the CLI in a subprocess so stream routing is actually exercised."""
    return subprocess.run(
        [sys.executable, "-m", "dyf.cli", *args],
        capture_output=True,
        text=True,
    )


class TestCollectInfo:
    def test_reports_the_real_item_count_and_dim(self, dyf_path):
        from dyf.info import collect_info

        info = collect_info(dyf_path)
        assert info["total_items"] == 64
        assert info["embedding_dim"] == 16

    def test_reports_stored_fields_by_name(self, dyf_path):
        from dyf.info import collect_info

        assert collect_info(dyf_path)["stored_fields"] == ["title"]

    def test_reports_domain_and_enrichment_level(self, dyf_path):
        from dyf.info import collect_info

        info = collect_info(dyf_path)
        assert info["domain"] == "test_corpus"
        # A freshly written index has had no enrichment stage run against it.
        assert info["enrichment_level"] == 0
        assert "base" in info["enrichment_label"]

    def test_does_not_load_embeddings(self, dyf_path):
        """The whole value of this command is that it stays cheap on huge files."""
        from dyf.info import collect_info

        info = collect_info(dyf_path)
        assert "embeddings" not in info
        # Payload must stay small enough to hand to a caller verbatim.
        assert len(json.dumps(info, default=str)) < 8_000

    def test_bulk_metadata_is_summarized_not_inlined(self, tmp_path):
        """A describe command must not inline megabytes of base64 audio.

        The original fixture carried no bulk metadata, so the size assertion above
        passed vacuously while `tour_audio` — the largest payload the pipeline writes —
        was absent from the exclusion list entirely. This builds the case that failed.
        """
        pytest.importorskip("dyf_rs")
        from dyf.dyf_tree import build_dyf_tree
        from dyf.info import collect_info
        from dyf.lazy_index import write_lazy_index

        rng = np.random.default_rng(3)
        X = np.ascontiguousarray(rng.standard_normal((32, 8)).astype(np.float32))
        tree = build_dyf_tree(X, max_depth=2, num_bits=2, min_leaf_size=2, seed=42)
        path = str(tmp_path / "with-audio.dyf")

        fake_audio = "A" * 200_000  # stands in for base64 WAV
        write_lazy_index(
            tree,
            X,
            path,
            metadata={"domain": "d", "tour_audio": fake_audio, "some_future_blob": "B" * 50_000},
        )

        info = collect_info(path)
        payload = json.dumps(info, default=str)

        assert fake_audio not in payload, "tour_audio was inlined into the summary"
        assert len(payload) < 8_000, f"summary ballooned to {len(payload)} bytes"
        # It should still be *reported*, just as a size rather than a value.
        assert info["bulk_metadata_bytes"]["tour_audio"] == len(fake_audio)
        assert "tour_audio" in info["metadata_keys"]
        # And the cap must catch a key nobody thought to list.
        assert "some_future_blob" in info["bulk_metadata_bytes"]


class TestCliContract:
    def test_human_output_is_not_empty(self, dyf_path):
        """The regression test for the mute-CLI bug."""
        p = _run("info", dyf_path)
        assert p.returncode == 0
        assert len(p.stdout.strip()) > 0, "info printed nothing at all"
        assert "64" in p.stdout

    def test_results_go_to_stdout_not_stderr(self, dyf_path):
        """`dyf info f.dyf > out.txt` must produce a non-empty file."""
        p = _run("info", dyf_path)
        assert len(p.stdout) > 0
        assert p.stderr == "", f"results leaked to stderr: {p.stderr[:200]!r}"

    def test_json_mode_parses_and_is_versioned(self, dyf_path):
        p = _run("info", dyf_path, "--json")
        assert p.returncode == 0
        payload = json.loads(p.stdout)
        assert payload["schema_version"] == 0
        assert payload["total_items"] == 64

    def test_missing_file_exits_2_and_says_so_on_stderr(self):
        p = _run("info", "/definitely/not/here.dyf")
        assert p.returncode == 2
        assert p.stdout == "", "error path must not write to stdout"
        assert "no such file" in p.stderr

    def test_directory_argument_is_rejected(self, tmp_path):
        p = _run("info", str(tmp_path))
        assert p.returncode == 2
        assert "directory" in p.stderr

    def test_unreadable_file_exits_1_not_a_traceback(self, tmp_path):
        junk = tmp_path / "not-an-index.dyf"
        junk.write_bytes(b"this is not a flatbuffer")
        p = _run("info", str(junk))
        assert p.returncode == 1
        assert "Traceback" not in p.stderr
        assert "could not read" in p.stderr
