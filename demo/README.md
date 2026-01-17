# DYF Demos

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

### Live Demo

See the visualization in action: https://jdonaldson.github.io/dyf/
