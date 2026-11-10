"""Record one real dyf run for the explainer page's replay widgets (CLINC150 or MNIST).

The page replays this file; nothing on the page reimplements dyf.  One run, seed 42, the
shipped path: build_dyf_tree (3,4,20) -> leaf-centroid kNN graph (k=10) -> Louvain at a
resolution sweep -> _reassign_points (the re-cut).  Fitted on every row, scored on the
labelled rows against each label layer the dataset has:
  clinc150  23,850 requests, MiniLM 384-d (or --embedding); layers: 10 domains, 150 intents
  mnist     70,000 digit images, PCA to 50-d from 784 pixels; layer: 10 digits
  cifar100  50,000 photos (train split), CLIP ViT-B/32 512-d; layers: 20 superclasses, 100 classes

Recorded:
  tree     for every tree depth, the cell each displayed point falls in
  leaves   leaf id per point, leaf 2-D position, the leaf graph's edges
  levels   per resolution: community per leaf (before the re-cut), final label per point
           (after), k, ARI vs each layer before and after, share of points the re-cut moved
  layers   labels, names, k-means told the true k (k-means++, one init, 3 seeds), and the
           fifty-label pick (200 draws: which level it lands on, regret)
  items    what a point is: its text, or a 14x14 thumbnail in a PNG sprite
  layout   UMAP 2-D of the embeddings, for display only (dyf never sees it)
Colours are a map colouring (8 hues, neighbours in the picture differ, a group keeps its
parent's hue where it can); they carry no meaning beyond telling neighbours apart.

Env: .venv/bin/python benchmarks/site_explainer_2026_11/record_pipeline_run.py --dataset mnist
     (--no-layout prints the sweep table only, to judge a dataset before building a page on it)
Out: ~/.cache/dyf_intent_bench/site/<dataset>_<embedding>_pipeline_run_seed42.json
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import time
from pathlib import Path

import numpy as np
import polars as pl
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score as ari
from sklearn.neighbors import NearestNeighbors

from dyf import build_dyf_tree
from dyf.agglomerate import _reassign_points

DATA = Path.home() / ".cache/dyf_intent_bench"
OUT = DATA / "site"
EMB = {
    "minilm": "",
    "bge_small": "_bge_small",
    "bge_base": "_bge_base",
    "mpnet": "_mpnet",
    "qwen3_plain": "_qwen3_plain",
    "qwen3_intent": "_qwen3_intent",
    "pca50": "",
}
TREE = dict(num_bits=3, max_depth=4, min_leaf_size=20)
RESOLUTIONS = [0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0, 16.0, 24.0, 32.0, 48.0, 64.0]
SEED = 42
N_DISPLAY = 12000
N_HUES = 8
TILE = 14


def log(msg):
    print(msg, flush=True)


def load(dataset, emb):
    """-> X, labelled mask, layers [dict(key, noun, singular, y, names)], meta frame, description."""
    meta = pl.read_parquet(DATA / f"{dataset}_meta.parquet")
    X = np.load(DATA / f"{dataset}_X{EMB[emb]}.npy").astype(np.float32)
    labelled = ~meta["is_oos"].to_numpy()

    def layer(key, noun, singular, col):
        names, y = np.unique(meta[col].to_numpy(), return_inverse=True)
        return dict(key=key, noun=noun, singular=singular, y=y, names=[str(s) for s in names])

    if dataset == "clinc150":
        layers = [layer("domain", "domains", "domain", "domain"), layer("intent", "intents", "intent", "intent_name")]
        desc = dict(
            title="CLINC150 (plus)",
            items="requests",
            item="request",
            kind="text",
            embedding=f"{dict(minilm='MiniLM').get(emb, emb)} embeddings, {X.shape[1]} dimensions",
        )
    elif dataset == "mnist":
        layers = [layer("digit", "digits", "digit", "intent_name")]
        desc = dict(
            title="MNIST",
            items="handwritten digits",
            item="digit image",
            kind="image",
            embedding=f"784 pixels reduced to {X.shape[1]} dimensions with PCA",
        )
    else:
        raise SystemExit(f"unknown dataset {dataset}")
    for la in layers:
        la["k"] = int(len(np.unique(la["y"][labelled])))
    return X, labelled, layers, meta, desc


def load_cifar100():
    """CIFAR-100 train split, CLIP ViT-B/32 image embeddings (demo/cifar_embeddings.py -> demo/cifar100.npz)."""
    d = np.load(Path(__file__).resolve().parents[2] / "demo/cifar100.npz")
    X = d["embeddings"].astype(np.float32)
    images = d["images"]

    def layer(key, noun, singular, y, names):
        return dict(
            key=key,
            noun=noun,
            singular=singular,
            y=y.astype(np.int64),
            names=[str(s) for s in names],
            k=int(len(names)),
        )

    layers = [
        layer("superclass", "superclasses", "superclass", d["coarse"], d["coarse_names"]),
        layer("class", "classes", "class", d["fine"], d["fine_names"]),
    ]

    def sprite(disp, tile=24):
        from PIL import Image

        cols = int(np.ceil(np.sqrt(len(disp))))
        sheet = Image.new("RGB", (cols * tile, int(np.ceil(len(disp) / cols)) * tile), (0, 0, 0))
        for j, i in enumerate(disp):
            sheet.paste(
                Image.fromarray(images[int(i)]).resize((tile, tile), Image.LANCZOS),
                ((j % cols) * tile, (j // cols) * tile),
            )
        buf = io.BytesIO()
        sheet.save(buf, format="JPEG", quality=80)
        return dict(tile=tile, cols=cols, sprite="data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode())

    desc = dict(
        title="CIFAR-100",
        items="photos",
        item="photo",
        kind="image",
        sprite=sprite,
        embedding=f"CLIP image embeddings, {X.shape[1]} dimensions",
    )
    return X, np.ones(len(X), dtype=bool), layers, None, desc


def mnist_sprite(disp):
    """PNG sprite of the displayed images, TILE px each, in display order.  Row order of the
    embeddings is sorted(parquet names): test, then train (prepare_coarse_datasets.images_pca50)."""
    from PIL import Image

    d = DATA / "raw/mnist/mnist"
    df = pl.concat([pl.read_parquet(p) for p in sorted(d.glob("*.parquet"))])
    cols = int(np.ceil(np.sqrt(len(disp))))
    rows = int(np.ceil(len(disp) / cols))
    sheet = Image.new("L", (cols * TILE, rows * TILE), 0)
    imgs = df["image"]
    for j, i in enumerate(disp):
        im = Image.open(io.BytesIO(imgs[int(i)]["bytes"])).resize((TILE, TILE), Image.LANCZOS)
        sheet.paste(im, ((j % cols) * TILE, (j // cols) * TILE))
    buf = io.BytesIO()
    sheet.save(buf, format="PNG", optimize=True)
    return dict(tile=TILE, cols=cols, sprite="data:image/png;base64," + base64.b64encode(buf.getvalue()).decode())


def tree_cells(tree, n):
    """cells[:, d-1] = id of the depth-d cell holding each point; a shallow leaf stays one cell below."""

    def height(node):
        ch = node.get("children") or []
        return 0 if not ch else 1 + max(height(c) for c in ch)

    D = height(tree)
    cells = np.full((n, D), -1, dtype=np.int64)
    counters = [0] * D

    def walk(node, d):
        idx = np.asarray(node["indices"])
        ch = node.get("children") or []
        if d >= 1:
            cells[idx, d - 1] = counters[d - 1]
            counters[d - 1] += 1
        if not ch:
            for dd in range(d + 1, D + 1):
                cells[idx, dd - 1] = counters[dd - 1]
                counters[dd - 1] += 1
            return
        for c in ch:
            walk(c, d + 1)

    walk(tree, 0)
    assert (cells >= 0).all()
    return cells


def group_positions(labels, xy, n_groups):
    pos = np.zeros((n_groups, 2))
    size = np.bincount(labels, minlength=n_groups)
    np.add.at(pos, labels, xy)
    return pos / np.maximum(size, 1)[:, None], size


def map_colour(labels, xy, n_groups, parent_labels=None, parent_colours=None):
    """Greedy map colouring: a group avoids the hues of its 6 nearest groups in the picture and
    keeps the hue of the parent group it overlaps most when that is free."""
    pos, size = group_positions(labels, xy, n_groups)
    live = np.flatnonzero(size > 0)
    col = np.full(n_groups, -1, dtype=np.int64)
    if len(live) == 0:
        return col
    kk = min(7, len(live))
    _, nb = NearestNeighbors(n_neighbors=kk).fit(pos[live]).kneighbors(pos[live])
    nbrs = {int(live[i]): set(int(live[j]) for j in nb[i][1:]) for i in range(len(live))}
    for g, hs in list(nbrs.items()):
        for h in hs:
            nbrs[h].add(g)
    pref = {}
    if parent_labels is not None and parent_colours is not None:
        for g in live:
            pl_ = parent_labels[labels == g]
            pref[int(g)] = int(parent_colours[np.bincount(pl_).argmax()])
    counts = np.zeros(N_HUES, dtype=np.int64)
    for g in sorted((int(g) for g in live), key=lambda g: -size[g]):
        used = {int(col[h]) for h in nbrs[g] if col[h] >= 0}
        p = pref.get(g, -1)
        if p >= 0 and p not in used:
            c = p
        else:
            free = [c for c in range(N_HUES) if c not in used] or list(range(N_HUES))
            c = min(free, key=lambda c: (counts[c], c))
        col[g] = c
        counts[c] += 1
    return col


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="clinc150", choices=["clinc150", "mnist", "cifar100"])
    ap.add_argument("--embedding", default=None)
    ap.add_argument("--no-layout", action="store_true")
    a = ap.parse_args()
    emb = a.embedding or dict(mnist="pca50", cifar100="clip").get(a.dataset, "minilm")
    X, labelled, layers, meta, desc = load_cifar100() if a.dataset == "cifar100" else load(a.dataset, emb)
    Xn = X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-12)
    n = len(X)
    log(
        f"{a.dataset} / {emb}: n={n} d={X.shape[1]} labelled={int(labelled.sum())} layers={[(la['key'], la['k']) for la in layers]}"
    )

    t0 = time.perf_counter()
    tree = build_dyf_tree(X, seed=SEED, **TREE)
    t_tree = time.perf_counter() - t0
    cells = tree_cells(tree, n)
    D = cells.shape[1]
    leaf = cells[:, D - 1]
    L = int(leaf.max() + 1)
    cent = np.zeros((L, X.shape[1]), dtype=np.float64)
    np.add.at(cent, leaf, Xn)
    cent = (cent / np.maximum(np.linalg.norm(cent, axis=1, keepdims=True), 1e-12)).astype(np.float32)
    log(f"tree: depth {D}, cells per depth {[int(cells[:, d].max() + 1) for d in range(D)]}, {L} leaves, {t_tree:.2f}s")

    from dyf_rs import build_knn_graph, louvain_communities

    t0 = time.perf_counter()
    adjacency = build_knn_graph(np.ascontiguousarray(cent), k=10)
    edges = [(i, int(j), float(w)) for i, nbrs in enumerate(adjacency) for j, w in nbrs]
    t_graph = time.perf_counter() - t0
    und = sorted({(min(i, j), max(i, j)) for i, j, _ in edges if i != j})
    log(f"leaf graph: {len(edges)} directed edges, {len(und)} undirected, {t_graph:.2f}s")

    levels = []
    for r in RESOLUTIONS:
        t1 = time.perf_counter()
        lab, _ = louvain_communities(L, edges, resolution=r)
        leaf_comm = np.asarray(lab).astype(np.int32)
        t_louv = time.perf_counter() - t1
        block = leaf_comm[leaf].astype(np.int32)
        t2 = time.perf_counter()
        final = np.asarray(_reassign_points(block.copy(), X))
        t_recut = time.perf_counter() - t2
        lv = dict(
            res=r,
            leaf_comm=leaf_comm,
            block=block,
            final=final,
            n_block=int(len(np.unique(block))),
            k=int(len(np.unique(final))),
            ari=[float(ari(la["y"][labelled], final[labelled])) for la in layers],
            ari_block=[float(ari(la["y"][labelled], block[labelled])) for la in layers],
            moved=float((block != final).mean()),
            t_louvain=t_louv,
            t_recut=t_recut,
        )
        levels.append(lv)
        log(
            f"  res {r:5.2f}: k={lv['k']:4d} (block {lv['n_block']:4d}) "
            + " ".join(
                f"ARI {la['key']} {v:.3f} (before re-cut {b:.3f})"
                for la, v, b in zip(layers, lv["ari"], lv["ari_block"])
            )
            + f" moved {lv['moved']:.1%} louvain {t_louv:.2f}s re-cut {t_recut:.2f}s"
        )

    rng = np.random.default_rng(SEED)
    pool = np.flatnonzero(labelled)
    for li, la in enumerate(layers):
        y = la["y"]
        vals, secs = [], []
        for s in (42, 1, 2):  # timed: the page compares a k-means refit per k with dyf's cost of changing level
            t0 = time.perf_counter()
            pred = KMeans(n_clusters=la["k"], n_init=1, random_state=s).fit_predict(Xn)
            secs.append(time.perf_counter() - t0)
            vals.append(float(ari(y[labelled], pred[labelled])))
        la["ref"] = dict(
            k=la["k"],
            ari_mean=float(np.mean(vals)),
            ari_min=float(min(vals)),
            ari_max=float(max(vals)),
            seconds_mean=float(np.mean(secs)),
        )
        full = np.array([lv["ari"][li] for lv in levels])
        picks = np.array(
            [
                int(np.argmax([ari(y[S], lv["final"][S]) for lv in levels]))
                for S in (rng.choice(pool, 50, replace=False) for _ in range(200))
            ]
        )
        la["fifty"] = dict(
            best_level=int(full.argmax()),
            best_ari=float(full.max()),
            pick_counts=np.bincount(picks, minlength=len(levels)).tolist(),
            regret_mean=float((full.max() - full[picks]).mean()),
            regret_p90=float(np.quantile(full.max() - full[picks], 0.9)),
            default_regret=float(full.max() - full[RESOLUTIONS.index(1.0)]),
        )
        log(
            f"  {la['key']}: k-means told k={la['k']} ARI {np.mean(vals):.3f} ({min(vals):.3f}-{max(vals):.3f}); "
            f"best level res {RESOLUTIONS[full.argmax()]} ARI {full.max():.3f}; fifty labels mean regret {la['fifty']['regret_mean']:.3f} "
            f"p90 {la['fifty']['regret_p90']:.3f}; default regret {la['fifty']['default_regret']:.3f}; picks {la['fifty']['pick_counts']}"
        )
    if a.no_layout:
        return

    OUT.mkdir(parents=True, exist_ok=True)
    lay = OUT / f"umap2d_{a.dataset}_{emb}_seed{SEED}.npy"
    if lay.exists():
        xy = np.load(lay)
    else:
        import umap

        t0 = time.perf_counter()
        xy = np.asarray(
            umap.UMAP(n_components=2, n_neighbors=15, min_dist=0.15, metric="cosine", random_state=SEED).fit_transform(
                Xn
            )
        )
        np.save(lay, xy.astype(np.float32))
        log(f"UMAP 2-d: {time.perf_counter() - t0:.1f}s -> {lay}")
    xy = xy.astype(np.float64)
    lo, hi = np.quantile(xy, 0.001, axis=0), np.quantile(xy, 0.999, axis=0)
    span = (hi - lo).max()  # one scale for both axes, so the layout is not stretched
    mid = (hi + lo) / 2
    q = np.clip(np.round((xy - mid) / span * 4000 + 2048), 0, 4095).astype(int)

    depth_cols, prev_l, prev_c = [], None, None
    for d in range(D):
        c = map_colour(cells[:, d], xy, int(cells[:, d].max() + 1), prev_l, prev_c)
        depth_cols.append(c)
        prev_l, prev_c = cells[:, d], c[cells[:, d]]
    prev_l, prev_c = None, None
    for lv in levels:
        ng = int(max(lv["final"].max(), lv["block"].max()) + 1)
        c = map_colour(lv["final"], xy, ng, prev_l, prev_c)
        missing = np.flatnonzero(c < 0)  # communities the re-cut emptied still colour the "before" view
        if len(missing):
            c[missing] = map_colour(lv["block"], xy, ng)[missing]
        lv["colours"] = c
        prev_l, prev_c = lv["final"], c[lv["final"]]
    prev_l, prev_c = None, None
    for la in layers:
        la["colours"] = map_colour(la["y"], xy, len(la["names"]), prev_l, prev_c)
        prev_l, prev_c = la["y"], la["colours"][la["y"]]

    disp = np.sort(np.random.default_rng(0).choice(n, min(N_DISPLAY, n), replace=False))
    leaf_pos, leaf_size = group_positions(leaf, q.astype(np.float64), L)
    items = dict(kind=desc["kind"])
    if desc["kind"] == "text":
        items["text"] = meta["text"].to_numpy()[disp].tolist()
    else:
        items.update(desc["sprite"](disp) if "sprite" in desc else mnist_sprite(disp))
    r1 = levels[RESOLUTIONS.index(1.0)]
    out = dict(
        meta=dict(
            dataset=desc["title"],
            key=a.dataset,
            items=desc["items"],
            item=desc["item"],
            embedding=desc["embedding"],
            n=int(n),
            n_labelled=int(labelled.sum()),
            d=int(X.shape[1]),
            n_display=int(len(disp)),
            seed=SEED,
            tree=TREE,
            leaf_k=10,
            leaves=L,
            depth=D,
            cells_per_depth=[int(cells[:, d].max() + 1) for d in range(D)],
            seconds=dict(tree=t_tree, graph=t_graph, louvain_res1=r1["t_louvain"], recut_res1=r1["t_recut"]),
        ),
        x=q[disp, 0].tolist(),
        y=q[disp, 1].tolist(),
        unlabelled=(~labelled[disp]).astype(int).tolist(),
        items=items,
        layers=[
            dict(
                key=la["key"],
                noun=la["noun"],
                singular=la["singular"],
                k=la["k"],
                names=la["names"],
                labels=la["y"][disp].tolist(),
                colours=la["colours"].tolist(),
                ref=la["ref"],
                fifty=la["fifty"],
            )
            for la in layers
        ],
        cells=[dict(labels=cells[disp, d].tolist(), colours=depth_cols[d].tolist()) for d in range(D)],
        leaf_x=np.round(leaf_pos[:, 0]).astype(int).tolist(),
        leaf_y=np.round(leaf_pos[:, 1]).astype(int).tolist(),
        leaf_size=leaf_size.tolist(),
        edges=[v for e in und for v in e],
        levels=[
            dict(
                res=lv["res"],
                k=lv["k"],
                n_block=lv["n_block"],
                ari=[round(v, 4) for v in lv["ari"]],
                ari_block=[round(v, 4) for v in lv["ari_block"]],
                moved=round(lv["moved"], 4),
                leaf_comm=lv["leaf_comm"].tolist(),
                final=lv["final"][disp].tolist(),
                colours=lv["colours"].tolist(),
            )
            for lv in levels
        ],
    )
    path = OUT / f"{a.dataset}_{emb}_pipeline_run_seed{SEED}.json"
    path.write_text(json.dumps(out, separators=(",", ":")))
    log(f"wrote {path} ({path.stat().st_size / 1e6:.2f} MB)")


if __name__ == "__main__":
    main()
