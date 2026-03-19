"""Tests for video keyframe indexing into .dyf files."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from dyf.index_video import _format_timestamp


# ---------------------------------------------------------------------------
# Tests: _format_timestamp
# ---------------------------------------------------------------------------

class TestFormatTimestamp:
    def test_zero(self):
        assert _format_timestamp(0) == "0:00"

    def test_seconds_only(self):
        assert _format_timestamp(5) == "0:05"
        assert _format_timestamp(30) == "0:30"
        assert _format_timestamp(59) == "0:59"

    def test_minutes_and_seconds(self):
        assert _format_timestamp(60) == "1:00"
        assert _format_timestamp(90) == "1:30"
        assert _format_timestamp(125) == "2:05"

    def test_hours(self):
        assert _format_timestamp(3600) == "1:00:00"
        assert _format_timestamp(3661) == "1:01:01"
        assert _format_timestamp(7200) == "2:00:00"

    def test_fractional_seconds(self):
        assert _format_timestamp(1.5) == "0:01"
        assert _format_timestamp(59.9) == "0:59"

    def test_negative_clamped(self):
        assert _format_timestamp(-5) == "0:00"


# ---------------------------------------------------------------------------
# Tests: detect_scenes (mocked)
# ---------------------------------------------------------------------------

class TestDetectScenes:
    def test_multiple_scenes(self):
        from dyf.index_video import detect_scenes

        # Mock scene list with 3 scenes
        mock_start1, mock_end1 = MagicMock(), MagicMock()
        mock_start1.get_seconds.return_value = 0.0
        mock_end1.get_seconds.return_value = 10.0

        mock_start2, mock_end2 = MagicMock(), MagicMock()
        mock_start2.get_seconds.return_value = 10.0
        mock_end2.get_seconds.return_value = 25.0

        mock_start3, mock_end3 = MagicMock(), MagicMock()
        mock_start3.get_seconds.return_value = 25.0
        mock_end3.get_seconds.return_value = 40.0

        mock_sm = MagicMock()
        mock_sm.get_scene_list.return_value = [
            (mock_start1, mock_end1),
            (mock_start2, mock_end2),
            (mock_start3, mock_end3),
        ]

        mock_video = MagicMock()
        mock_video.frame_rate = 30.0
        mock_video.duration.get_frames.return_value = 1200

        mock_open_video = MagicMock(return_value=mock_video)
        mock_scene_manager_cls = MagicMock(return_value=mock_sm)
        mock_content_detector_cls = MagicMock()

        with patch("dyf.index_video._load_scenedetect",
                   return_value=(mock_open_video, mock_scene_manager_cls, mock_content_detector_cls)):
            scenes = detect_scenes(Path("test.mp4"))

        assert len(scenes) == 3
        assert scenes[0]["scene_id"] == 0
        assert scenes[0]["start_time"] == 0.0
        assert scenes[0]["end_time"] == 10.0
        assert scenes[0]["keyframe_time"] == 5.0

    def test_single_scene_fallback(self):
        from dyf.index_video import detect_scenes

        mock_sm = MagicMock()
        mock_sm.get_scene_list.return_value = []  # No scene cuts

        mock_video = MagicMock()
        mock_video.frame_rate = 30.0
        mock_video.duration.get_frames.return_value = 600  # 20 seconds

        mock_open_video = MagicMock(return_value=mock_video)
        mock_scene_manager_cls = MagicMock(return_value=mock_sm)
        mock_content_detector_cls = MagicMock()

        with patch("dyf.index_video._load_scenedetect",
                   return_value=(mock_open_video, mock_scene_manager_cls, mock_content_detector_cls)):
            scenes = detect_scenes(Path("test.mp4"))

        # Should fall back to 5s uniform sampling: 0-5, 5-10, 10-15, 15-20
        assert len(scenes) == 4
        assert scenes[0]["keyframe_time"] == 2.5
        assert scenes[-1]["end_time"] == 20.0


# ---------------------------------------------------------------------------
# Tests: index_video end-to-end (mocked)
# ---------------------------------------------------------------------------

try:
    import PIL  # noqa: F401
    _has_pil = True
except ImportError:
    _has_pil = False


@pytest.mark.skipif(not _has_pil, reason="requires Pillow")
class TestIndexVideoE2E:
    @patch("dyf.index_video.detect_scenes")
    @patch("dyf.index_video.extract_keyframes")
    @patch("dyf.index_video.load_vision_model")
    @patch("dyf.index_video.embed_images")
    @patch("dyf.index_video.make_thumbnail")
    def test_creates_dyf_file(
        self, mock_thumb, mock_embed, mock_load_model,
        mock_extract, mock_detect, tmp_path
    ):
        from dyf.index_video import index_video
        from PIL import Image

        # Mock scenes
        mock_detect.return_value = [
            {"scene_id": 0, "start_time": 0.0, "end_time": 5.0,
             "duration": 5.0, "keyframe_time": 2.5},
            {"scene_id": 1, "start_time": 5.0, "end_time": 12.0,
             "duration": 7.0, "keyframe_time": 8.5},
        ]

        # Mock keyframe extraction
        mock_extract.return_value = [
            Image.new("RGB", (320, 240)),
            Image.new("RGB", (320, 240)),
        ]

        mock_load_model.return_value = (MagicMock(), MagicMock(), "cpu")
        mock_embed.return_value = np.random.randn(2, 768).astype(np.float32)
        mock_thumb.return_value = "data:image/webp;base64,AAAA"

        # Create a dummy video file so path.is_file() works
        video = tmp_path / "test.mp4"
        video.write_bytes(b"fake video")
        output = tmp_path / "test.dyf"

        index_video(video_path=video, output=output, min_leaf_size=1)

        assert output.exists()
        assert output.stat().st_size > 0

    @patch("dyf.index_video.detect_scenes")
    @patch("dyf.index_video.extract_keyframes")
    @patch("dyf.index_video.load_vision_model")
    @patch("dyf.index_video.embed_images")
    @patch("dyf.index_video.make_thumbnail")
    def test_stored_fields(
        self, mock_thumb, mock_embed, mock_load_model,
        mock_extract, mock_detect, tmp_path
    ):
        from dyf.index_video import index_video
        from dyf.lazy_index import LazyIndex
        from PIL import Image

        mock_detect.return_value = [
            {"scene_id": 0, "start_time": 0.0, "end_time": 5.0,
             "duration": 5.0, "keyframe_time": 2.5},
        ]
        mock_extract.return_value = [Image.new("RGB", (320, 240))]
        mock_load_model.return_value = (MagicMock(), MagicMock(), "cpu")
        mock_embed.return_value = np.random.randn(1, 768).astype(np.float32)
        mock_thumb.return_value = "data:image/webp;base64,BBBB"

        video = tmp_path / "test.mp4"
        video.write_bytes(b"fake")
        output = tmp_path / "test.dyf"

        index_video(video_path=video, output=output, min_leaf_size=1)

        idx = LazyIndex(str(output))
        data = idx.extract_all_fields()
        fields = data["fields"]

        assert "title" in fields
        assert "thumbnail" in fields
        assert "file" in fields
        assert "timestamp" in fields
        assert "scene_id" in fields
        assert "duration" in fields

        assert fields["title"][0].startswith("Scene 0 at")
        assert fields["thumbnail"][0] == "data:image/webp;base64,BBBB"

    @patch("dyf.index_video.detect_scenes")
    @patch("dyf.index_video.extract_keyframes")
    @patch("dyf.index_video.load_vision_model")
    @patch("dyf.index_video.embed_images")
    @patch("dyf.index_video.make_thumbnail")
    def test_metadata(
        self, mock_thumb, mock_embed, mock_load_model,
        mock_extract, mock_detect, tmp_path
    ):
        from dyf.index_video import index_video
        from dyf.lazy_index import LazyIndex
        from PIL import Image

        mock_detect.return_value = [
            {"scene_id": 0, "start_time": 0.0, "end_time": 5.0,
             "duration": 5.0, "keyframe_time": 2.5},
        ]
        mock_extract.return_value = [Image.new("RGB", (320, 240))]
        mock_load_model.return_value = (MagicMock(), MagicMock(), "cpu")
        mock_embed.return_value = np.random.randn(1, 768).astype(np.float32)
        mock_thumb.return_value = "data:image/webp;base64,CCCC"

        video = tmp_path / "clip.mp4"
        video.write_bytes(b"fake")
        output = tmp_path / "test.dyf"

        index_video(video_path=video, output=output, min_leaf_size=1)

        idx = LazyIndex(str(output))
        data = idx.extract_all_fields()
        meta = data["metadata"]
        assert meta.get("domain") == "video"
        assert meta.get("thumbnail_format") == "webp"
        assert meta.get("source_video") == "clip.mp4"
