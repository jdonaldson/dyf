"""CIFAR-100 dyf image-scatter demo (interactive, open dataset).

Pipeline: local CIFAR-100 -> CLIP embeddings -> dyf tree (cut to clusters) -> UMAP ->
an interactive deck.gl scatter where every image sits at its UMAP coordinate, outlined
in its dyf-cluster color, with hover tooltips (fine class + cluster).

Outputs (written next to this script, in demo/):
  cifar100.npz       cached embeddings + thumbnails + labels (regenerate-skippable)
  cifar_atlas.png    texture atlas of cluster-outlined 32px thumbnails
  cifar_deck.html    the viewer (references cifar_atlas.png -> serve over http)

Run:
  pip install "dyf[vision]" mlx-vis matplotlib            # (umap-learn works instead of mlx-vis)
  # ensure the CIFAR-100 pickle exists (one-time, ~178MB):
  python -c "import torchvision; torchvision.datasets.CIFAR100('~/.cache/torchvision', download=True)"
  python demo/cifar_deck_demo.py
  python demo/viz_server.py --dir demo                    # then open cifar_deck.html
Because deck.gl fetches the atlas, the page MUST be served over http(s) (a file:// open
is blocked by CORS) — use viz_server.py or `python -m http.server`.
"""
import json
import os
import pickle
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CIFAR = os.path.expanduser("~/.cache/torchvision/cifar-100-python")
NPZ = os.path.join(HERE, "cifar100.npz")
ATLAS = os.path.join(HERE, "cifar_atlas.png")
HTML = os.path.join(HERE, "cifar_deck.html")
CLIP_MODEL = "openai/clip-vit-base-patch32"
N_SCATTER = 10000   # subset rendered for smooth interactivity
K = 20              # dyf clusters (matches CIFAR-100's 20 coarse superclasses)


def _unpickle(f):
    with open(f, "rb") as fo:
        return pickle.load(fo, encoding="bytes")


def embed_cifar():
    """Load CIFAR-100 (local pickle) and CLIP-embed all 50k train images -> NPZ cache."""
    import torch
    from PIL import Image
    from transformers import CLIPModel, CLIPProcessor

    tr, meta = _unpickle(f"{CIFAR}/train"), _unpickle(f"{CIFAR}/meta")
    imgs = tr[b"data"].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)  # (N,32,32,3) uint8
    fine = np.asarray(tr[b"fine_labels"], np.int16)
    coarse = np.asarray(tr[b"coarse_labels"], np.int16)
    fine_names = [n.decode() for n in meta[b"fine_label_names"]]
    coarse_names = [n.decode() for n in meta[b"coarse_label_names"]]

    device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    model = CLIPModel.from_pretrained(CLIP_MODEL).to(device).eval()
    proc = CLIPProcessor.from_pretrained(CLIP_MODEL)
    t0, embs, B = time.perf_counter(), [], 256
    with torch.no_grad():
        for s in range(0, len(imgs), B):
            batch = [Image.fromarray(x) for x in imgs[s:s + B]]
            pv = proc(images=batch, return_tensors="pt").to(device)["pixel_values"]
            f = model.visual_projection(model.vision_model(pixel_values=pv).pooler_output)
            embs.append(f.cpu().numpy())
    E = np.concatenate(embs).astype(np.float32)
    print(f"embedded {E.shape} in {time.perf_counter()-t0:.0f}s")
    np.savez(NPZ, embeddings=E, images=imgs, fine=fine, coarse=coarse,
             fine_names=np.array(fine_names), coarse_names=np.array(coarse_names))


def _umap(X):
    try:
        from mlx_vis import UMAP            # fast on Apple Silicon
        return np.asarray(UMAP(n_components=2, n_neighbors=15).fit_transform(X))
    except ImportError:
        from umap import UMAP
        return np.asarray(UMAP(n_components=2, n_neighbors=15).fit_transform(X))


def build_viz():
    import matplotlib
    from PIL import Image
    from sklearn.metrics import normalized_mutual_info_score as nmi

    import dyf

    D = np.load(NPZ, allow_pickle=True)
    X = np.ascontiguousarray(D["embeddings"], np.float32)
    imgs, fine, coarse = D["images"], D["fine"], D["coarse"]
    fine_names = list(D["fine_names"])

    # dyf structure + how well it recovers the true coarse superclasses
    tree = dyf.build_dyf_tree(X, max_depth=10, num_bits=3, min_leaf_size=16)
    labels = np.asarray(dyf.cut_tree_to_labels(tree, len(X), K, embeddings=X))
    print(f"NMI(dyf clusters, true coarse classes) = {nmi(coarse, labels):.3f}")

    rng = np.random.default_rng(0)
    sub = rng.choice(len(X), min(N_SCATTER, len(X)), replace=False)
    lab = labels[sub]
    Y = _umap(X[sub])

    cmap = matplotlib.colormaps["tab20"]
    palette = [tuple(int(255 * c) for c in cmap(i % 20)[:3]) for i in range(K)]

    CELL, IMG, BORDER = 40, 32, 4
    cols = int(np.ceil(np.sqrt(len(sub))))
    rows = int(np.ceil(len(sub) / cols))
    atlas = Image.new("RGB", (cols * CELL, rows * CELL), (0, 0, 0))
    mapping = {}
    for k, i in enumerate(sub):
        cx, cy = (k % cols) * CELL, (k // cols) * CELL
        cell = Image.new("RGB", (CELL, CELL), palette[lab[k]])
        cell.paste(Image.fromarray(imgs[i]).resize((IMG, IMG)), (BORDER, BORDER))
        atlas.paste(cell, (cx, cy))
        mapping[str(k)] = {"x": cx, "y": cy, "width": CELL, "height": CELL, "mask": False}
    atlas.save(ATLAS)

    Yn = (Y - Y.mean(0)) / (Y.std(0) + 1e-9) * 100
    data = [{"position": [float(Yn[k, 0]), float(Yn[k, 1])], "icon": str(k),
             "name": f"{fine_names[fine[sub[k]]]}  ·  dyf cluster {int(lab[k])}"}
            for k in range(len(sub))]
    html = _HTML.replace("__DATA__", json.dumps(data)).replace("__MAPPING__", json.dumps(mapping))
    with open(HTML, "w") as f:
        f.write(html)
    print(f"atlas {cols*CELL}x{rows*CELL}, {len(sub)} icons -> {HTML}")


_HTML = """<!doctype html><html><head><meta charset="utf-8">
<title>CIFAR-100 - dyf image scatter</title>
<style>html,body,#c{margin:0;width:100%;height:100%;background:#0b0b10;overflow:hidden}
#cap{position:fixed;top:10px;left:12px;color:#ccc;font:13px system-ui;
 background:rgba(0,0,0,.5);padding:6px 10px;border-radius:6px;z-index:5}#cap b{color:#fff}</style>
<script src="https://unpkg.com/deck.gl@9.0.0/dist.min.js"></script></head><body>
<div id="cap"><b>CIFAR-100</b> - images at UMAP coords, outline = dyf cluster &middot; scroll zoom / drag pan / hover</div>
<div id="c"></div><script>
const DATA = __DATA__, MAPPING = __MAPPING__;
const {DeckGL, IconLayer, OrthographicView} = deck;
new DeckGL({container:"c", views:[new OrthographicView({})],
  initialViewState:{target:[0,0,0], zoom:2}, controller:true,
  getTooltip: ({object}) => object && {text: object.name},
  layers:[ new IconLayer({id:"imgs", data:DATA, iconAtlas:"cifar_atlas.png",
    iconMapping:MAPPING, getIcon:d=>d.icon, getPosition:d=>d.position,
    getSize:1.6, sizeUnits:"common", pickable:true, alphaCutoff:-1}) ]});
</script></body></html>"""


if __name__ == "__main__":
    if not os.path.exists(NPZ):
        embed_cifar()
    else:
        print(f"using cached {NPZ}")
    build_viz()
