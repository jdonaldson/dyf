"""
Index video keyframes into a .dyf file.

Uses PySceneDetect for scene boundary detection, extracts keyframes at scene
midpoints, then reuses vision functions from index_images.

Usage (via CLI):
    dyf index-video movie.mp4 -o scenes.dyf
    dyf index-video clip.webm -o clip.dyf --threshold 20

Requires: pip install "dyf[video]"

⚠ **Video is indexed as still images, and the `title` field is a timestamp** like
`"Scene 3 at 1:24"`. There is no audio track, no motion or temporal embedding — one
keyframe per scene goes through the *image* model. Anything downstream that reads `title`
is reading timestamps, which carry no semantics at all, so LLM labelling and keywording
degrade further here than they do for images.

Exit codes (see `_ingest_errors`): 0 ok, 1 nothing to index, 2 bad request,
3 dependency or service unavailable.
"""

import argparse
import logging
import time
from pathlib import Path

logger = logging.getLogger(__name__)

from ._ingest_common import add_common_index_args, finalize_index
from ._ingest_errors import BadIngestRequest, EmptyIngestError, IngestError
from ._preview import IngestPreview
from .index_images import DEFAULT_MODEL, embed_images, load_vision_model, make_thumbnail


def _format_timestamp(seconds: float) -> str:
    """Format seconds as M:SS or H:MM:SS."""
    seconds = max(0.0, seconds)
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    if h > 0:
        return f"{h}:{m:02d}:{s:02d}"
    return f"{m}:{s:02d}"


def _load_scenedetect():
    """Lazy-load scenedetect and return (open_video, SceneManager, ContentDetector)."""
    try:
        from scenedetect import SceneManager, open_video
        from scenedetect.detectors import ContentDetector

        return open_video, SceneManager, ContentDetector
    except ImportError:
        raise ImportError('scenedetect is required for video indexing.\nInstall it with: pip install "dyf[video]"')


def detect_scenes(
    video_path: Path,
    threshold: float = 27.0,
    min_scene_len: int = 15,
) -> list[dict]:
    """Detect scene boundaries in a video using PySceneDetect.

    Returns list of dicts with scene_id, start_time, end_time, duration, keyframe_time.
    Falls back to uniform 5s sampling if only 1 scene detected.
    """
    open_video, SceneManager, ContentDetector = _load_scenedetect()

    video = open_video(str(video_path))
    scene_manager = SceneManager()
    scene_manager.add_detector(
        ContentDetector(
            threshold=threshold,
            min_scene_len=min_scene_len,
        )
    )
    scene_manager.detect_scenes(video)
    scene_list = scene_manager.get_scene_list()

    # Get video duration from the video backend
    fps = video.frame_rate
    total_frames = video.duration.get_frames()
    video_duration = total_frames / fps if fps > 0 else 0

    if len(scene_list) <= 1:
        # Fallback: uniform sampling every 5 seconds
        interval = 5.0
        scenes = []
        t = 0.0
        scene_id = 0
        while t < video_duration:
            end_t = min(t + interval, video_duration)
            scenes.append(
                {
                    "scene_id": scene_id,
                    "start_time": t,
                    "end_time": end_t,
                    "duration": end_t - t,
                    "keyframe_time": (t + end_t) / 2,
                }
            )
            scene_id += 1
            t = end_t
        return scenes

    scenes = []
    for i, (start, end) in enumerate(scene_list):
        start_sec = start.get_seconds()
        end_sec = end.get_seconds()
        scenes.append(
            {
                "scene_id": i,
                "start_time": start_sec,
                "end_time": end_sec,
                "duration": end_sec - start_sec,
                "keyframe_time": (start_sec + end_sec) / 2,
            }
        )
    return scenes


def extract_keyframes(video_path: Path, scenes: list[dict]) -> list:
    """Extract keyframe images from video at scene midpoints.

    Returns list of PIL Images.
    """
    try:
        import cv2
        from PIL import Image
    except ImportError:
        raise ImportError('opencv-python and Pillow are required: pip install "dyf[video]"')

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    images = []

    for scene in scenes:
        frame_num = int(scene["keyframe_time"] * fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        ret, frame = cap.read()
        if ret:
            # OpenCV BGR -> RGB -> PIL
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            images.append(Image.fromarray(rgb))
        else:
            # Fallback: try start of scene
            frame_num = int(scene["start_time"] * fps)
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
            ret, frame = cap.read()
            if ret:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                images.append(Image.fromarray(rgb))
            else:
                images.append(None)

    cap.release()
    return images


def preview_video(
    video_path: Path,
    output: Path,
    model: str = DEFAULT_MODEL,
    threshold: float = 27.0,
    batch_size: int = 16,
) -> IngestPreview:
    """Report what `index_video` would do, without decoding the video.

    Unlike source and images, here the *counting itself* is the expensive step: scene
    detection is a full decode pass, and the scene count determines everything downstream.
    So this reports the count as unknown rather than running the work a dry run exists to
    help you avoid — a preview that costs as much as the run is not a preview.

    What it can say cheaply — file size, and that the scene threshold is the parameter
    that will decide the item count — is what a caller actually needs to choose between
    running it and reconsidering.
    """
    size_mb = video_path.stat().st_size / 1_048_576

    return IngestPreview(
        command="index-video",
        source=str(video_path),
        output=str(output),
        model=model,
        batch_size=batch_size,
        counts={"megabytes": int(size_mb), "scenes": None, "keyframes": None},
        batches=None,
        notes=[
            "scene count is not previewable: detecting scenes is a full decode pass, "
            "which is the expensive work this flag exists to let you avoid",
            f"one keyframe is embedded per scene; --threshold {threshold} decides how many (lower finds more scenes)",
        ],
    )


def index_video(
    video_path: Path,
    output: Path,
    model: str = DEFAULT_MODEL,
    threshold: float = 27.0,
    min_scene_len: int = 15,
    max_depth: int = 4,
    num_bits: int = 4,
    min_leaf_size: int = 5,
    seed: int = 42,
    batch_size: int = 16,
    dedup: float | None = None,
) -> None:
    """Index video keyframes into a .dyf file.

    Args:
        dedup: Cosine threshold for collapsing near-duplicate keyframes before indexing.
            Reached video for the first time in 0.13 -- it had existed only in
            `index_source` because the shared tail was copy-pasted rather than shared.
            This is the textbook case for it: the README measures 88.3% duplicates on
            adjacent-frame data, and a static shot yields near-identical keyframes.
    """
    logger.info("Indexing video")
    logger.info(f"  Video:  {video_path}")
    logger.info(f"  Output: {output}")
    logger.info(f"  Model:  {model}")

    # Detect scenes
    logger.info(f"Detecting scenes (threshold={threshold})...")
    t0 = time.time()
    scenes = detect_scenes(video_path, threshold=threshold, min_scene_len=min_scene_len)
    logger.info(f"  {len(scenes)} scenes in {time.time() - t0:.1f}s")

    # Extract keyframes
    logger.info("Extracting keyframes...")
    t0 = time.time()
    raw_images = extract_keyframes(video_path, scenes)
    logger.info(f"  Extracted in {time.time() - t0:.1f}s")

    # Filter out failed extractions
    valid = [(s, img) for s, img in zip(scenes, raw_images) if img is not None]
    if not valid:
        raise EmptyIngestError(
            f"detected {len(scenes)} scene(s) in {video_path.name}, but no keyframe could "
            f"be extracted from any of them.\n"
            f"  The file may be truncated, or in a codec OpenCV cannot decode."
        )

    scenes_valid, images = zip(*valid)
    scenes_valid = list(scenes_valid)
    images = list(images)
    logger.info(f"  {len(images)} valid keyframes")

    # Load vision model
    logger.info("Loading vision model...")
    t0 = time.time()
    processor, vision_model, device = load_vision_model(model)
    logger.info(f"  Model loaded on {device} in {time.time() - t0:.1f}s")

    # Generate thumbnails
    logger.info("Generating thumbnails...")
    thumbnails = [make_thumbnail(img) for img in images]

    # Embed
    logger.info("Embedding keyframes...")
    t0 = time.time()
    embeddings = embed_images(images, processor, vision_model, device, batch_size)
    logger.info(f"  {embeddings.shape} in {time.time() - t0:.1f}s")

    # Build stored fields
    titles = [f"Scene {s['scene_id']} at {_format_timestamp(s['keyframe_time'])}" for s in scenes_valid]
    files = [str(video_path.name)] * len(scenes_valid)
    timestamps = [s["keyframe_time"] for s in scenes_valid]
    scene_ids = [s["scene_id"] for s in scenes_valid]
    durations = [s["duration"] for s in scenes_valid]

    finalize_index(
        embeddings,
        output,
        stored_fields={
            "title": titles,
            "thumbnail": thumbnails,
            "file": files,
            "timestamp": timestamps,
            "scene_id": scene_ids,
            "duration": durations,
        },
        metadata={
            "embedding_model": model,
            "domain": "video",
            "thumbnail_size": "128x128",
            "thumbnail_format": "webp",
            "scene_threshold": str(threshold),
            "source_video": video_path.name,
        },
        max_depth=max_depth,
        num_bits=num_bits,
        min_leaf_size=min_leaf_size,
        seed=seed,
        dedup=dedup,
    )
    logger.info(f"Done. {len(images)} keyframes indexed.")


def main(argv: list[str] | None = None) -> int:
    """CLI entry point for `dyf index-video`."""
    parser = argparse.ArgumentParser(
        prog="dyf index-video",
        description="Index video keyframes into a .dyf file using scene detection + vision embeddings",
    )
    parser.add_argument(
        "video_file",
        type=Path,
        help="Video file to index",
    )
    add_common_index_args(parser, default_model=DEFAULT_MODEL)
    parser.add_argument(
        "--threshold",
        type=float,
        default=27.0,
        help="Scene detection threshold (default: 27.0)",
    )
    parser.add_argument(
        "--min-scene-len",
        type=int,
        default=15,
        help="Minimum scene length in frames (default: 15)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Embedding batch size (default: 16)",
    )

    args = parser.parse_args(argv)

    video_path = args.video_file.resolve()
    if not video_path.is_file():
        logger.error("not a file: %s", video_path)
        return BadIngestRequest.exit_code

    output = args.output
    if output is None:
        output = Path(f"{video_path.stem}.dyf")

    if args.dry_run:
        return preview_video(
            video_path=video_path,
            output=output.resolve(),
            model=args.model,
            threshold=args.threshold,
            batch_size=args.batch_size,
        ).emit(args.as_json, logger)

    try:
        index_video(
            video_path=video_path,
            output=output.resolve(),
            model=args.model,
            threshold=args.threshold,
            min_scene_len=args.min_scene_len,
            max_depth=args.max_depth,
            num_bits=args.num_bits,
            min_leaf_size=args.min_leaf_size,
            seed=args.seed,
            batch_size=args.batch_size,
            dedup=args.dedup,
        )
    except IngestError as exc:
        for line in str(exc).splitlines():
            logger.error("%s", line)
        return exc.exit_code
    return 0
