"""CIFAR-100 dyf image-scatter demo (interactive, open dataset).

Pipeline: local CIFAR-100 -> CLIP embeddings -> dyf tree -> UMAP -> an interactive deck.gl
scatter where every image sits at its UMAP coordinate, backed by a square outline colored
by its cluster. Two toolbar controls:
  * facet depth slider  — re-cuts the clustering at different granularities (few -> many)
  * method toggle       — recolors outlines by dyf clusters vs k-means clusters at that depth
so you can eyeball where dyf and k-means agree/disagree. An NMI table (dyf vs k-means vs the
true coarse classes, per depth) is printed at build time.

Outputs (in demo/): cifar100.npz (cache), cifar_atlas.png (thumbnails), cifar_deck.html (viewer).

Run:
  pip install "dyf[vision]" mlx-vis matplotlib scikit-learn   # umap-learn works instead of mlx-vis
  python -c "import torchvision; torchvision.datasets.CIFAR100('~/.cache/torchvision', download=True)"
  python demo/cifar_deck_demo.py
  python demo/viz_server.py --dir demo                        # then open cifar_deck.html
deck.gl fetches the atlas, so the page MUST be served over http(s) (file:// is CORS-blocked).
"""
import base64
import io
import json
import os
import pickle
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CIFAR = os.path.expanduser("~/.cache/torchvision/cifar-100-python")
NPZ = os.path.join(HERE, "cifar100.npz")
ATLAS = os.path.join(HERE, "cifar_atlas.jpg")  # JPEG: 10k photo thumbs compress ~10x vs PNG
HTML = os.path.join(HERE, "cifar_deck.html")
CLIP_MODEL = "openai/clip-vit-base-patch32"
N_SCATTER = 10000                            # subset rendered for smooth interactivity
DEPTHS = [2, 4, 8, 16, 32, 64, 128]          # facet granularities (n_clusters) for the slider


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
    from PIL import Image
    from sklearn.cluster import MiniBatchKMeans
    from sklearn.metrics import normalized_mutual_info_score as nmi

    import dyf

    D = np.load(NPZ, allow_pickle=True)
    X = np.ascontiguousarray(D["embeddings"], np.float32)
    imgs, fine, coarse = D["images"], D["fine"], D["coarse"]
    fine_names = list(D["fine_names"])

    # cluster the full set with BOTH methods at every facet depth
    tree = dyf.build_dyf_tree(X, max_depth=10, num_bits=3, min_leaf_size=16)
    dyf_labels, km_labels = {}, {}
    for g in DEPTHS:
        dyf_labels[g] = np.asarray(dyf.cut_tree_to_labels(tree, len(X), g, embeddings=X))
        km_labels[g] = MiniBatchKMeans(n_clusters=g, random_state=0, n_init="auto").fit_predict(X)

    print(f"\n{'facet':>6} {'dyf~coarse':>11} {'kmeans~coarse':>14} {'dyf~kmeans':>11}")
    for g in DEPTHS:
        print(f"{g:>6} {nmi(coarse, dyf_labels[g]):>11.3f} {nmi(coarse, km_labels[g]):>14.3f} "
              f"{nmi(dyf_labels[g], km_labels[g]):>11.3f}")

    rng = np.random.default_rng(0)
    sub = rng.choice(len(X), min(N_SCATTER, len(X)), replace=False)
    Y = _umap(X[sub])

    # borderless thumbnail atlas (outline is a separate deck.gl layer)
    CELL = 32
    cols = int(np.ceil(np.sqrt(len(sub))))
    rows = int(np.ceil(len(sub) / cols))
    atlas = Image.new("RGB", (cols * CELL, rows * CELL), (0, 0, 0))
    mapping = {}
    for k, i in enumerate(sub):
        cx, cy = (k % cols) * CELL, (k // cols) * CELL
        atlas.paste(Image.fromarray(imgs[i]).resize((CELL, CELL)), (cx, cy))
        mapping[str(k)] = {"x": cx, "y": cy, "width": CELL, "height": CELL, "mask": False}
    atlas.save(ATLAS, quality=82)

    # 1x1 white square -> tintable backing icon (mask:true colors it by getColor)
    buf = io.BytesIO(); Image.new("RGBA", (1, 1), (255, 255, 255, 255)).save(buf, "PNG")
    white = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()

    Yn = (Y - Y.mean(0)) / (Y.std(0) + 1e-9) * 100
    data = [{"position": [float(Yn[k, 0]), float(Yn[k, 1])], "icon": str(k),
             "dyf": [int(dyf_labels[g][sub[k]]) for g in DEPTHS],
             "km": [int(km_labels[g][sub[k]]) for g in DEPTHS],
             "name": fine_names[fine[sub[k]]]}
            for k in range(len(sub))]
    html = (_HTML.replace("__DMAX__", str(len(DEPTHS) - 1))
                 .replace("__DATA__", json.dumps(data))
                 .replace("__MAPPING__", json.dumps(mapping))
                 .replace("__DEPTHS__", json.dumps(DEPTHS))
                 .replace("__WHITE__", white))
    with open(HTML, "w") as f:
        f.write(html)
    print(f"\natlas {cols*CELL}x{rows*CELL}, {len(sub)} icons, {len(DEPTHS)} depths, 2 methods -> {HTML}")


_HTML = """<!doctype html><html><head><meta charset="utf-8">
<title>CIFAR-100 - dyf vs k-means image scatter</title>
<style>html,body,#c{margin:0;width:100%;height:100%;background:#0b0b10;overflow:hidden}
#bar{position:fixed;top:10px;left:12px;color:#ddd;font:13px system-ui;z-index:5;
 background:rgba(0,0,0,.6);padding:8px 12px;border-radius:8px;display:flex;gap:12px;align-items:center}
#bar b{color:#fff}#bar input[type=range]{vertical-align:middle}#dval{min-width:58px;color:#7fd1ff}
.seg{display:inline-flex;border:1px solid #555;border-radius:6px;overflow:hidden}
.seg button{background:#1a1a22;color:#bbb;border:0;padding:3px 10px;cursor:pointer;font:13px system-ui}
.seg button.on{background:#2d6cdf;color:#fff}
#note{position:fixed;top:48px;left:12px;color:#9aa;font:11px system-ui;z-index:5;
 background:rgba(0,0,0,.5);padding:4px 9px;border-radius:6px}</style>
<script src="https://unpkg.com/deck.gl@9.0.0/dist.min.js"></script></head><body>
<div id="bar"><b>CIFAR-100</b>
 &middot; granularity (k): <input id="d" type="range" min="0" max="__DMAX__" value="3" step="1"> <span id="dval"></span>
 &middot; outline: <span class="seg"><button id="m_dyf" class="on">dyf</button><button id="m_km">k-means</button></span>
 &middot; <span class="seg"><button id="img_t" class="on">images</button></span>
 &middot; scroll/drag/hover</div>
<div id="note">same k, two clusterings &mdash; dyf = nested tree cuts (one fit &rarr; all k) &middot; k-means = independent fit per k</div>
<div id="c"></div><script>
const DATA = __DATA__, MAPPING = __MAPPING__, DEPTHS = __DEPTHS__, WHITE = "__WHITE__";
const {DeckGL, IconLayer, OrthographicView} = deck;
let di = 3, method = "dyf", showImg = true;
function hsl(label){ // golden-angle hue -> distinct colors at any cluster count
  const h=(label*137.508)%360, s=0.62, l=0.55, a=s*Math.min(l,1-l);
  const f=n=>{const k=(n+h/30)%12; return Math.round(255*(l-a*Math.max(-1,Math.min(k-3,9-k,1))));};
  return [f(0),f(8),f(4)];
}
const deckgl = new DeckGL({container:"c", views:[new OrthographicView({})],
  initialViewState:{target:[0,0,0], zoom:2}, controller:true,
  getTooltip: ({object}) => object && {text: object.name
    +"\\ndyf cluster "+object.dyf[di]+"  |  k-means "+object.km[di]+"   ("+DEPTHS[di]+"-way)"}});
function render(){
  const layers=[
    new IconLayer({id:"outline", data:DATA, iconAtlas:WHITE,
      iconMapping:{sq:{x:0,y:0,width:1,height:1,mask:true}}, getIcon:()=>"sq",
      getPosition:d=>d.position, getColor:d=>hsl(d[method][di]), getSize:showImg?2.4:2.1,
      sizeUnits:"common", pickable:true, updateTriggers:{getColor:[di,method], getSize:showImg}})];
  if(showImg) layers.push(
    new IconLayer({id:"imgs", data:DATA, iconAtlas:"cifar_atlas.jpg",
      iconMapping:MAPPING, getIcon:d=>d.icon, getPosition:d=>d.position,
      getSize:1.9, sizeUnits:"common", pickable:true, alphaCutoff:-1}));
  deckgl.setProps({layers});
  document.getElementById("dval").textContent = DEPTHS[di]+"-way";
}
document.getElementById("d").addEventListener("input", e=>{di=+e.target.value; render();});
function setM(m){method=m; document.getElementById("m_dyf").classList.toggle("on",m==="dyf");
  document.getElementById("m_km").classList.toggle("on",m==="km"); render();}
document.getElementById("m_dyf").onclick=()=>setM("dyf");
document.getElementById("m_km").onclick=()=>setM("km");
document.getElementById("img_t").onclick=()=>{showImg=!showImg;
  document.getElementById("img_t").classList.toggle("on",showImg); render();};
render();
</script></body></html>"""


if __name__ == "__main__":
    if not os.path.exists(NPZ):
        embed_cifar()
    else:
        print(f"using cached {NPZ}")
    build_viz()
