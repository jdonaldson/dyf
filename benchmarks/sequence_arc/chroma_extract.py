"""Extract beat-synchronous, key-normalized chroma sequences from Spotify previews.

Beat-sync is essential: at the default 23ms hop consecutive chroma frames are
near-identical, so frame-level deltas measure nothing but noise. One frame per
beat makes consecutive frames genuinely different (chord changes).

Key-normalization matters too: ii-V-I in C and in F are the *same move* but
different chroma deltas. Without rotating each track to a common tonic, every
progression appears in up to 12 rotated variants, diluting any vocabulary 12x.
"""

import glob
import os
import sys
from concurrent.futures import ThreadPoolExecutor

import numpy as np

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "chroma_beats.npz")
PREVIEWS = "/Users/jdonaldson/Projects/cochlear/data/spotify/previews/*.m4a"


def one(path):
    import warnings

    warnings.filterwarnings("ignore")
    import librosa

    try:
        y, sr = librosa.load(path, sr=22050, mono=True)
        if len(y) < sr * 5:
            return None
        _, beats = librosa.beat.beat_track(y=y, sr=sr)
        if len(beats) < 8:
            return None
        c = librosa.feature.chroma_cqt(y=y, sr=sr)
        cs = librosa.util.sync(c, beats, aggregate=np.median)  # (12, n_beats)
        # key-normalize: rotate so the strongest average pitch class sits at 0
        tonic = int(np.argmax(cs.mean(axis=1)))
        cs = np.roll(cs, -tonic, axis=0)
        return os.path.basename(path), cs.T.astype(np.float32)  # (n_beats, 12)
    except Exception as e:  # noqa: BLE001
        print(f"  FAIL {os.path.basename(path)}: {e}", flush=True)
        return None


def main():
    files = sorted(glob.glob(PREVIEWS))
    print(f"tracks: {len(files)}", flush=True)
    results = []
    with ThreadPoolExecutor(max_workers=8) as ex:
        for i, r in enumerate(ex.map(one, files), 1):
            if r is not None:
                results.append(r)
            if i % 10 == 0:
                print(f"  [{i}/{len(files)}] ok={len(results)}", flush=True)

    names = [n for n, _ in results]
    seqs = [s for _, s in results]
    lens = np.array([len(s) for s in seqs])
    print(f"\nextracted {len(seqs)} tracks | beats/track: median {np.median(lens):.0f} total {lens.sum()}", flush=True)
    np.savez_compressed(
        OUT,
        names=np.array(names),
        lengths=lens,
        stacked=np.concatenate(seqs, axis=0),
    )
    print(f"wrote {OUT}", flush=True)


if __name__ == "__main__":
    sys.exit(main())
