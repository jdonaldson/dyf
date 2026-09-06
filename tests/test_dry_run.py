"""`--dry-run` previews for the `dyf index-*` commands.

Two properties are asserted, and they are the whole point:

**A dry run stays cheap.** It must not do the expensive work it exists to help you avoid.
For `index-video` that means it must *not* run scene detection — a full decode pass — even
though that is the only way to learn the item count. Reporting "unknown" is the correct
answer there, and a preview that costs as much as the run would be worthless.

**No invented time estimates.** Counts are exact; nothing is multiplied by a throughput
number, because no measured one exists. A plausible fabricated duration would be believed,
which makes it worse than an honest count.
"""

from __future__ import annotations

import base64
import json
import subprocess
import sys

import pytest

# 1x1 PNG — enough for a file-count preview, which never decodes.
TINY_PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
)
UNREACHABLE = "http://localhost:59999/api/embed"


def _run(*args):
    return subprocess.run(
        [sys.executable, "-m", "dyf.cli", *args],
        capture_output=True,
        text=True,
        timeout=120,
    )


class TestPreviewObject:
    def test_batches_is_ceil_division(self):
        from dyf._preview import batches_for

        assert batches_for(0, 16) == 0
        assert batches_for(1, 16) == 1
        assert batches_for(16, 16) == 1
        assert batches_for(17, 16) == 2

    def test_batches_is_none_when_count_is_unknown(self):
        """An unknown count must not silently become 0 batches."""
        from dyf._preview import batches_for

        assert batches_for(None, 16) is None

    def test_render_says_nothing_happened(self):
        from dyf._preview import IngestPreview

        text = IngestPreview(command="index-source", source="/s", output="/o.dyf", model="m", batch_size=8).render()
        assert "DRY RUN" in text
        assert "no file was written" in text


class TestIndexSourceDryRun:
    def test_reports_file_count_without_indexing(self, tmp_path):
        src = tmp_path / "src"
        src.mkdir()
        (src / "a.py").write_text("def alpha():\n    return 1\n")
        (src / "b.py").write_text("def beta():\n    return 2\n")
        out = tmp_path / "out.dyf"

        p = _run("index-source", str(src), "-o", str(out), "--dry-run", "--ollama-url", UNREACHABLE)
        assert p.returncode == 0, "a preview is not a verdict — it should not fail"
        assert not out.exists(), "dry run must not write the index"

    def test_json_is_parseable_and_flagged(self, tmp_path):
        src = tmp_path / "src"
        src.mkdir()
        (src / "a.py").write_text("def alpha():\n    return 1\n")

        p = _run(
            "index-source",
            str(src),
            "-o",
            str(tmp_path / "o.dyf"),
            "--dry-run",
            "--json",
            "--ollama-url",
            UNREACHABLE,
        )
        payload = json.loads(p.stdout)
        assert payload["dry_run"] is True
        assert payload["schema_version"] == 0
        assert payload["counts"]["files"] == 1

    def test_unreachable_service_is_reported_not_raised(self, tmp_path):
        """The preview must surface a dead service as information, not as a failure."""
        src = tmp_path / "src"
        src.mkdir()
        (src / "a.py").write_text("def alpha():\n    return 1\n")

        p = _run(
            "index-source",
            str(src),
            "-o",
            str(tmp_path / "o.dyf"),
            "--dry-run",
            "--json",
            "--ollama-url",
            UNREACHABLE,
        )
        assert p.returncode == 0
        payload = json.loads(p.stdout)
        assert "UNAVAILABLE" in payload["service"]
        assert any("cannot reach" in n for n in payload["notes"])

    def test_no_fabricated_duration(self, tmp_path):
        """Counts only. A plausible invented time estimate would be believed."""
        src = tmp_path / "src"
        src.mkdir()
        (src / "a.py").write_text("def alpha():\n    return 1\n")

        p = _run(
            "index-source",
            str(src),
            "-o",
            str(tmp_path / "o.dyf"),
            "--dry-run",
            "--ollama-url",
            UNREACHABLE,
        )
        combined = (p.stdout + p.stderr).lower()
        for unit in ("minutes", "seconds remaining", "eta", "estimated time"):
            assert unit not in combined, f"preview invented a duration: {unit}"


class TestIndexImagesDryRun:
    def test_counts_files_without_loading_a_model(self, tmp_path):
        pics = tmp_path / "pics"
        pics.mkdir()
        for name in ("a.png", "b.png", "c.jpg"):
            (pics / name).write_bytes(TINY_PNG)
        out = tmp_path / "pics.dyf"

        p = _run("index-images", str(pics), "-o", str(out), "--dry-run", "--json")
        assert p.returncode == 0, "must not need torch/transformers to preview"
        payload = json.loads(p.stdout)
        assert payload["counts"]["images"] == 3
        assert payload["embedding_batches"] == 1
        assert not out.exists()

    def test_says_the_usable_count_is_not_yet_known(self, tmp_path):
        """Files matching is not the same as files decoding; the preview must not conflate them."""
        pics = tmp_path / "pics"
        pics.mkdir()
        (pics / "a.png").write_bytes(TINY_PNG)

        p = _run("index-images", str(pics), "-o", str(tmp_path / "o.dyf"), "--dry-run", "--json")
        payload = json.loads(p.stdout)
        assert any("fail to decode" in n for n in payload["notes"])


class TestIndexVideoDryRun:
    def test_does_not_run_scene_detection(self, tmp_path):
        """The load-bearing assertion of this whole feature.

        Scene detection is a full decode. A preview that ran it would cost what the run
        costs, so `scenes` must come back unknown rather than counted — and crucially,
        this must hold on a file that is not even a valid video, proving nothing tried to
        decode it.
        """
        clip = tmp_path / "clip.mp4"
        clip.write_bytes(b"this is not a video")

        p = _run("index-video", str(clip), "-o", str(tmp_path / "v.dyf"), "--dry-run", "--json")
        assert p.returncode == 0, "must not fail on an undecodable file — it never opened it"
        payload = json.loads(p.stdout)
        assert payload["counts"]["scenes"] is None
        assert payload["embedding_batches"] is None
        assert any("not previewable" in n for n in payload["notes"])

    def test_names_the_parameter_that_decides_the_count(self, tmp_path):
        clip = tmp_path / "clip.mp4"
        clip.write_bytes(b"not a video")

        p = _run("index-video", str(clip), "-o", str(tmp_path / "v.dyf"), "--dry-run", "--json")
        payload = json.loads(p.stdout)
        assert any("threshold" in n for n in payload["notes"])


@pytest.mark.parametrize("cmd", ["index-source", "index-images"])
def test_dry_run_writes_nothing_for_any_command(cmd, tmp_path):
    src = tmp_path / "in"
    src.mkdir()
    out = tmp_path / "out.dyf"
    _run(cmd, str(src), "-o", str(out), "--dry-run", "--ollama-url", UNREACHABLE)
    assert not out.exists()
