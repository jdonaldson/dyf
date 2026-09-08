"""Tests for image indexing into .dyf files."""

import base64
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

PIL = pytest.importorskip("PIL", reason="requires Pillow")

from dyf.index_images import make_thumbnail, scan_images

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_pil_image(width=64, height=64, mode="RGB"):
    """Create a simple PIL Image for testing."""
    from PIL import Image

    return Image.new(mode, (width, height), color=(128, 64, 32))


def _make_image_file(tmp_path: Path, name: str, width=64, height=64) -> Path:
    """Write a real image file to tmp_path."""
    from PIL import Image

    img = Image.new("RGB", (width, height), color=(100, 150, 200))
    p = tmp_path / name
    p.parent.mkdir(parents=True, exist_ok=True)
    img.save(p)
    return p


# ---------------------------------------------------------------------------
# Tests: scan_images
# ---------------------------------------------------------------------------


class TestScanImages:
    def test_finds_common_formats(self, tmp_path):
        _make_image_file(tmp_path, "photo.jpg")
        _make_image_file(tmp_path, "icon.png")
        _make_image_file(tmp_path, "banner.webp")

        results = scan_images(tmp_path)
        names = {p.name for p in results}
        assert "photo.jpg" in names
        assert "icon.png" in names
        assert "banner.webp" in names

    def test_recursive(self, tmp_path):
        _make_image_file(tmp_path, "sub/deep/photo.jpg")
        results = scan_images(tmp_path)
        assert len(results) == 1
        assert results[0].name == "photo.jpg"

    def test_ignores_non_images(self, tmp_path):
        (tmp_path / "readme.txt").write_text("hello")
        (tmp_path / "data.csv").write_text("a,b\n1,2")
        _make_image_file(tmp_path, "photo.jpg")

        results = scan_images(tmp_path)
        assert len(results) == 1

    def test_empty_dir(self, tmp_path):
        results = scan_images(tmp_path)
        assert results == []

    def test_case_insensitive_extensions(self, tmp_path):
        _make_image_file(tmp_path, "photo.JPG")
        _make_image_file(tmp_path, "icon.PNG")
        results = scan_images(tmp_path)
        assert len(results) == 2


# ---------------------------------------------------------------------------
# Tests: make_thumbnail
# ---------------------------------------------------------------------------


class TestMakeThumbnail:
    def test_returns_data_uri(self):
        img = _make_pil_image(256, 256)
        result = make_thumbnail(img)
        assert result.startswith("data:image/webp;base64,")

    def test_valid_base64(self):
        img = _make_pil_image(256, 256)
        result = make_thumbnail(img)
        b64_part = result.split(",", 1)[1]
        decoded = base64.b64decode(b64_part)
        assert len(decoded) > 0

    def test_small_image_not_upscaled(self):
        img = _make_pil_image(32, 32)
        result = make_thumbnail(img, max_size=128)
        assert result.startswith("data:image/webp;base64,")

    def test_non_square_image(self):
        img = _make_pil_image(400, 100)
        result = make_thumbnail(img, max_size=128)
        assert result.startswith("data:image/webp;base64,")

    def test_rgba_image(self):
        img = _make_pil_image(64, 64, mode="RGBA")
        result = make_thumbnail(img)
        assert result.startswith("data:image/webp;base64,")


# ---------------------------------------------------------------------------
# Tests: index_images end-to-end (mocked model)
# ---------------------------------------------------------------------------


class TestIndexImagesE2E:
    @patch("dyf.index_images.load_vision_model")
    @patch("dyf.index_images.embed_images")
    def test_creates_dyf_file(self, mock_embed, mock_load_model, tmp_path):
        from dyf.index_images import index_images

        # Create test images
        _make_image_file(tmp_path / "input", "a.jpg", 100, 80)
        _make_image_file(tmp_path / "input", "b.png", 60, 60)
        _make_image_file(tmp_path / "input", "c.webp", 200, 150)

        # Mock model
        mock_load_model.return_value = (MagicMock(), MagicMock(), "cpu")

        # Mock embeddings: 3 images, 768d
        mock_embed.return_value = np.random.randn(3, 768).astype(np.float32)

        output = tmp_path / "test.dyf"
        index_images(
            source_dir=tmp_path / "input",
            output=output,
            min_leaf_size=1,
        )

        assert output.exists()
        assert output.stat().st_size > 0

    @patch("dyf.index_images.load_vision_model")
    @patch("dyf.index_images.embed_images")
    def test_stored_fields(self, mock_embed, mock_load_model, tmp_path):
        from dyf.index_images import index_images
        from dyf.lazy_index import LazyIndex

        _make_image_file(tmp_path / "input", "photo.jpg", 100, 80)
        _make_image_file(tmp_path / "input", "icon.png", 60, 60)

        mock_load_model.return_value = (MagicMock(), MagicMock(), "cpu")
        mock_embed.return_value = np.random.randn(2, 768).astype(np.float32)

        output = tmp_path / "test.dyf"
        index_images(
            source_dir=tmp_path / "input",
            output=output,
            min_leaf_size=1,
        )

        idx = LazyIndex(str(output))
        data = idx.extract_all_fields()
        fields = data["fields"]

        assert "title" in fields
        assert "thumbnail" in fields
        assert "file" in fields
        assert "width" in fields
        assert "height" in fields

        # Check thumbnail is a data URI
        for thumb in fields["thumbnail"]:
            assert thumb.startswith("data:image/webp;base64,")

    @patch("dyf.index_images.load_vision_model")
    @patch("dyf.index_images.embed_images")
    def test_corrupt_image_skipped(self, mock_embed, mock_load_model, tmp_path):
        from dyf.index_images import index_images

        # Create one valid and one corrupt image
        _make_image_file(tmp_path / "input", "good.jpg", 64, 64)
        corrupt = tmp_path / "input" / "bad.jpg"
        corrupt.write_bytes(b"not an image at all")

        mock_load_model.return_value = (MagicMock(), MagicMock(), "cpu")
        mock_embed.return_value = np.random.randn(1, 768).astype(np.float32)

        output = tmp_path / "test.dyf"
        index_images(
            source_dir=tmp_path / "input",
            output=output,
            min_leaf_size=1,
        )

        assert output.exists()

    @patch("dyf.index_images.load_vision_model")
    @patch("dyf.index_images.embed_images")
    def test_metadata(self, mock_embed, mock_load_model, tmp_path):
        from dyf.index_images import index_images
        from dyf.lazy_index import LazyIndex

        _make_image_file(tmp_path / "input", "photo.jpg", 64, 64)

        mock_load_model.return_value = (MagicMock(), MagicMock(), "cpu")
        mock_embed.return_value = np.random.randn(1, 768).astype(np.float32)

        output = tmp_path / "test.dyf"
        index_images(
            source_dir=tmp_path / "input",
            output=output,
            min_leaf_size=1,
        )

        idx = LazyIndex(str(output))
        data = idx.extract_all_fields()
        meta = data["metadata"]
        assert meta.get("domain") == "images"
        assert meta.get("thumbnail_format") == "webp"


# ---------------------------------------------------------------------------
# Model fetch failure: transformers downloads the model on first use, and an offline
# or unwritable cache must be a legible exit-3 error, not an OSError traceback.
# ---------------------------------------------------------------------------


def test_model_fetch_failure_is_legible(monkeypatch):
    transformers = pytest.importorskip("transformers", reason="requires dyf[vision]")
    from dyf._ingest_errors import EXIT_UNAVAILABLE, ModelUnavailableError
    from dyf.index_images import load_vision_model

    def _offline(*a, **kw):
        raise OSError("We couldn't connect to 'https://huggingface.co' to load the files")

    monkeypatch.setattr(transformers.AutoProcessor, "from_pretrained", _offline)
    with pytest.raises(ModelUnavailableError) as info:
        load_vision_model("org/some-model", device="cpu")
    msg = str(info.value)
    assert info.value.exit_code == EXIT_UNAVAILABLE
    assert "org/some-model" in msg
    assert "couldn't connect" in msg
    assert "HF_HOME" in msg


def test_cli_model_failure_exits_3_without_traceback(tmp_path, monkeypatch, caplog):
    from dyf import index_images as mod
    from dyf._ingest_errors import EXIT_UNAVAILABLE, ModelUnavailableError

    def _unavailable(model_name, device=None):
        raise ModelUnavailableError(f"cannot load the vision model {model_name!r}: offline")

    monkeypatch.setattr(mod, "load_vision_model", _unavailable)
    PIL.Image.new("RGB", (8, 8)).save(tmp_path / "a.png")
    rc = mod.main([str(tmp_path), "-o", str(tmp_path / "out.dyf")])
    assert rc == EXIT_UNAVAILABLE
    assert "cannot load the vision model" in caplog.text
    assert "Traceback" not in caplog.text


def test_dry_run_notes_uncached_model(tmp_path, monkeypatch):
    pytest.importorskip("huggingface_hub")
    import huggingface_hub

    from dyf import index_images as mod

    monkeypatch.setattr(huggingface_hub, "try_to_load_from_cache", lambda *a, **kw: None)
    PIL.Image.new("RGB", (8, 8)).save(tmp_path / "a.png")
    preview = mod.preview_images(source_dir=tmp_path, output=tmp_path / "out.dyf", model="org/never-cached")
    assert any("not in the local Hugging Face cache" in n for n in preview.notes)


def test_dry_run_silent_when_model_cached(tmp_path, monkeypatch):
    pytest.importorskip("huggingface_hub")
    import huggingface_hub

    from dyf import index_images as mod

    monkeypatch.setattr(huggingface_hub, "try_to_load_from_cache", lambda *a, **kw: "/cache/config.json")
    PIL.Image.new("RGB", (8, 8)).save(tmp_path / "a.png")
    preview = mod.preview_images(source_dir=tmp_path, output=tmp_path / "out.dyf", model="org/cached")
    assert not any("Hugging Face cache" in n for n in preview.notes)
