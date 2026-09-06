"""Extract embeddings + date/ticker/section from filings.dyf into an npz cache.

Every other sec_*.py script reads this cache. Run once.
"""

import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sec_seqlib as S  # noqa: E402

from dyf.lazy_index import LazyIndex  # noqa: E402


def main():
    if os.path.exists(S.NPZ):
        print(f"cached: {S.NPZ}")
        return
    os.makedirs(S.CACHE, exist_ok=True)
    idx = LazyIndex(S.FILINGS_DYF)
    n_batches = idx._index.BatchesLength()
    print(f"{S.FILINGS_DYF}: batches={n_batches} total={idx._index.TotalItems()}")

    embs, dates, tickers, items, secs = [], [], [], [], []
    t0 = time.time()
    for bi in range(n_batches):
        b = idx.get_leaf(bi)
        e = b.column("embedding").to_numpy(zero_copy_only=False)
        e = np.stack(e) if e.dtype == object else e.reshape(b.num_rows, -1)
        embs.append(e.astype(np.float16))
        dates.append(np.asarray(b.column("filing_date").to_pylist()))
        tickers.append(np.asarray(b.column("ticker").to_pylist()))
        secs.append(np.asarray(b.column("section").to_pylist()))
        items.append(b.column("item_index").to_numpy(zero_copy_only=False))
        if bi % 4000 == 0:
            print(f"  [{bi}/{n_batches}] {time.time() - t0:.0f}s", flush=True)

    E = np.concatenate(embs)
    np.savez(
        S.NPZ,
        embeddings=E,
        dates=np.concatenate(dates),
        tickers=np.concatenate(tickers),
        item_index=np.concatenate(items),
        sections=np.concatenate(secs),
    )
    print(f"saved {E.shape} -> {S.NPZ} in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
