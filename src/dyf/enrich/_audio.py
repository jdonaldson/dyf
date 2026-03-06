"""Generate Kokoro TTS audio for tour narration and store in .dyf metadata."""

import base64
import io
import json

from dyf.lazy_index import LazyIndex, rewrite_lazy_index


def generate_tour_audio(dyf_path, voice="bf_emma", speed=1.0, output_path=None):
    """Read tour_narration from .dyf, generate Kokoro TTS, store as tour_audio.

    Args:
        dyf_path: Path to .dyf file (must have tour_narration metadata).
        voice: Kokoro voice ID (default: bf_emma).
        speed: Playback speed multiplier (default: 1.0).
        output_path: Output path (default: overwrite input).
    """
    try:
        import soundfile as sf
        from kokoro import KPipeline
    except ImportError as e:
        print(f"Error: Kokoro TTS not available ({e})")
        print("  Install with: pip install kokoro soundfile")
        raise SystemExit(1)

    idx = LazyIndex(dyf_path)
    meta = idx._get_metadata()

    narration_json = meta.get("tour_narration")
    if not narration_json:
        print(f"Error: No tour_narration metadata in {dyf_path}")
        print("  Run 'dyf enrich viz' first to generate narration text.")
        raise SystemExit(1)

    narration = json.loads(narration_json)
    print(f"\n=== Generating tour audio ===")
    print(f"  Voice: {voice}, Speed: {speed}")
    print(f"  Narration keys: {len(narration)} ({', '.join(sorted(narration.keys()))})")

    # Initialize Kokoro pipeline
    lang_code = 'b' if voice.startswith('b') else 'a'
    pipeline = KPipeline(lang_code=lang_code)

    audio_data = {}
    done = 0
    for cid, text in narration.items():
        try:
            for _, _, audio in pipeline(text, voice=voice, speed=speed):
                if audio is None:
                    continue
                duration_ms = int(len(audio) / 24000 * 1000)
                buf = io.BytesIO()
                sf.write(buf, audio, 24000, format='WAV')
                buf.seek(0)
                audio_data[str(cid)] = {
                    "data": base64.b64encode(buf.read()).decode('ascii'),
                    "duration": duration_ms,
                }
                break  # Only need first chunk
        except Exception as e:
            print(f"  [TTS] Failed for key '{cid}': {e}")

        done += 1
        if done % 5 == 0 or done == len(narration):
            print(f"  Rendered {done}/{len(narration)} clips...")

    print(f"  Total audio entries: {len(audio_data)}")
    total_bytes = sum(len(v["data"]) for v in audio_data.values())
    print(f"  Total base64 size: {total_bytes / 1024 / 1024:.1f} MB")

    out = output_path or dyf_path
    print(f"  Writing tour_audio to: {out}")
    rewrite_lazy_index(dyf_path, new_metadata={
        "tour_audio": json.dumps(audio_data),
    }, output_path=out)
    print("  Done.")
