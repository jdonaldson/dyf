# DYF Demos

## CIFAR-100 Interactive Image Scatter (`cifar_deck_demo.py`)

An open-dataset demo of dyf structure discovery: 50k CIFAR-100 images → CLIP embeddings →
dyf tree → UMAP → an interactive **deck.gl** scatter where every image sits at its UMAP
coordinate, backed by a square **outline colored by its cluster**, with hover tooltips.

Toolbar:
- **granularity (k)** slider — re-cuts the clustering at 2…128 clusters.
- **dyf / k-means** toggle — recolors outlines by either method *at the same k*, a fair
  matched-k comparison. dyf gives **nested** tree cuts (one fit → all k); k-means is an
  **independent** fit per k.
- **images** on/off — hide the photos to read the cluster coloring cleanly.

Honest results (printed at build time, `NMI` vs the 20 true coarse classes): both land
around **0.45**, a real *partial* recovery — dyf edges k-means at coarse k (≤16), k-means
edges dyf at fine k (≥32); their agreement rises with k. CIFAR-100 does not cluster
trivially in CLIP space.

### Requirements

```bash
pip install "dyf[vision]" mlx-vis matplotlib scikit-learn   # umap-learn works in place of mlx-vis
# one-time CIFAR-100 fetch (~178MB) if not already cached:
python -c "import torchvision; torchvision.datasets.CIFAR100('~/.cache/torchvision', download=True)"
```

### Usage

```bash
python demo/cifar_deck_demo.py        # embeds (cached to demo/cifar100.npz), builds the viewer
python demo/viz_server.py --dir demo  # serve, then open http://localhost:<port>/cifar_deck.html
```

deck.gl fetches the texture atlas, so the page **must be served over http(s)** — opening
`cifar_deck.html` as a `file://` is blocked by CORS. Generated artifacts (`cifar100.npz`,
`cifar_atlas.jpg`, `cifar_deck.html`) are gitignored. A built copy is published to the docs
gallery at [dyf.io/gallery/cifar100-deck](https://dyf.io/gallery/cifar100-deck.html).

## Wikipedia Knowledge Graph Visualization

Interactive visualization of Wikipedia article embeddings showing clusters and density-based bridges.

### Requirements

```bash
pip install dyf umap-learn plotly scikit-learn requests polars
```

### Usage

```bash
# Basic visualization
python wiki_visualization.py embeddings.parquet -o wiki_graph.html

# With Ollama cluster labeling
python wiki_visualization.py embeddings.parquet --label-with-ollama --ollama-model gemma2:9b

# Custom settings
python wiki_visualization.py embeddings.parquet \
    --n-clusters 15 \
    --title "My Knowledge Graph" \
    --seed 123
```

### Input Format

Expects a parquet file with columns:
- `embedding`: List of floats (the embedding vector)
- `title` or `text`: String (used for hover labels)
- `category` (optional): String (category labels)

### Output

Generates an interactive HTML file with:
- 2D UMAP projection of embeddings
- Clusters colored by k-means grouping
- Edge-bundled bridges showing density connections between clusters
- Scroll to zoom, click+drag to pan
- Hover for article titles

