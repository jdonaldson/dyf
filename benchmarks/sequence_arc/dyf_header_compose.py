"""What fraction of a .dyf file is tree header, and what is the header made of?

Decides whether sharing hyperplanes across a sequence of dyfs is worth anything.
Answer (2026-08-01): header is 6-30% of the file, but it is ~75% CENTROIDS and only
~25% hyperplanes -- because every node carries a dim-length centroid while only
internal nodes carry hyperplanes, and leaves outnumber internals ~12:1.

So freezing hyperplanes is free but small; the centroid bulk dedupes only for leaves
whose membership is unchanged.

Usage: python dyf_header_compose.py [file.dyf ...]
"""

import os
import struct
import sys

DEFAULTS = [
    os.path.expanduser("~/Projects/gudid-explorer/data/gudid_viz.dyf"),
    os.path.expanduser("~/Projects/haxe/src.dyf"),
    os.path.expanduser("~/Projects/sec10quant/data/filings.dyf"),
]


def header_size(path):
    """FlatBuffers section size for header-based formats (DYF1/DYF3). None for DYF2."""
    with open(path, "rb") as f:
        magic = f.read(4)
        if magic in (b"DYF1", b"DYF3"):
            return magic.decode(), struct.unpack("<Q", f.read(8))[0]
    return magic.decode(errors="replace"), None


def main(paths):
    from dyf.lazy_index import LazyIndex

    for p in paths:
        if not os.path.exists(p):
            print(f"skip (missing): {p}")
            continue
        total = os.path.getsize(p)
        fmt, fb = header_size(p)
        fb_txt = f"{fb:,} ({100 * fb / total:.1f}%)" if fb else "n/a (footer-based)"

        idx = LazyIndex(p)
        d = idx._index
        hp = cen = ev = internal = leaves = 0
        for i in range(d.NodesLength()):
            nd = d.Nodes(i)
            hp += nd.HyperplanesLength()
            cen += nd.CentroidLength()
            ev += nd.EigenvaluesLength()
            if nd.ChildrenLength() > 0:
                internal += 1
            else:
                leaves += 1
        flt = hp + cen + ev
        print(f"\n{os.path.basename(p)}  [{fmt}] dim={d.EmbeddingDim()}")
        print(f"  total {total:,}   fb section {fb_txt}")
        print(f"  nodes {d.NodesLength():,} (internal {internal:,}, leaf {leaves:,})")
        if flt:
            print(f"  hyperplanes {hp * 4:>12,} B  {100 * hp / flt:>5.1f}% of float payload")
            print(f"  centroids   {cen * 4:>12,} B  {100 * cen / flt:>5.1f}%")
            print(f"  eigenvalues {ev * 4:>12,} B  {100 * ev / flt:>5.1f}%")
        if hp == 0 and internal:
            print(
                f"  !! {internal} internal nodes store ZERO hyperplanes -- "
                "this file cannot serve as a frozen foundation"
            )


if __name__ == "__main__":
    main(sys.argv[1:] or DEFAULTS)
