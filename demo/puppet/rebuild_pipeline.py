"""
Full pipeline: re-cluster at higher resolution, export, reassign, build skeleton, label.
"""
import sys, json, collections
import numpy as np
from scipy.spatial import cKDTree
import heapq

sys.path.insert(0, "/Users/jdonaldson/Projects/dyf/src")
from dyf.lazy_index import LazyIndex
from dyf.agglomerate import compute_louvain_hierarchy

RESOLUTION = 2.0

# --- Step 1: Re-cluster ---
print(f"=== Step 1: Louvain at resolution={RESOLUTION} ===")
idx = LazyIndex("/tmp/3dgs_position_color_enriched.dyf")
data = idx.extract_all_fields()
fields = data["fields"]

x = np.array(fields["umap_x"], dtype=np.float32)
y = np.array(fields["umap_y"], dtype=np.float32)
z = np.array(fields["umap_z"], dtype=np.float32)
r = np.array(fields["r"], dtype=np.int32)
g = np.array(fields["g"], dtype=np.int32)
b = np.array(fields["b"], dtype=np.int32)
positions = np.column_stack([x, y, z])
n = len(x)

embeddings = data.get("embeddings")
result = compute_louvain_hierarchy(idx, positions, embeddings, resolution=RESOLUTION)
cid_orig = np.array(result["point_labels"], dtype=np.int32)
communities = sorted(set(cid_orig))
print(f"  {len(communities)} communities, {n} points")

# --- Step 2: Spatial reassignment ---
print(f"\n=== Step 2: Spatial reassignment ===")
centroids = np.array([positions[cid_orig == c].mean(axis=0) for c in communities])
tree = cKDTree(centroids)
_, nearest = tree.query(positions)
new_cid = np.array([communities[i] for i in nearest], dtype=np.int32)

changed = (new_cid != cid_orig).sum()
print(f"  {changed}/{n} points reassigned ({100*changed/n:.1f}%)")

for iteration in range(10):
    new_centroids = np.array([positions[new_cid == c].mean(axis=0) if (new_cid == c).any()
                              else centroids[i] for i, c in enumerate(communities)])
    tree = cKDTree(new_centroids)
    _, nearest = tree.query(positions)
    newer_cid = np.array([communities[i] for i in nearest], dtype=np.int32)
    delta = (newer_cid != new_cid).sum()
    new_cid = newer_cid
    centroids = new_centroids
    print(f"    Iteration {iteration+1}: {delta} changed")
    if delta == 0:
        break

# Recompute communities list (some might be empty now)
communities = sorted(np.unique(new_cid))
print(f"  {len(communities)} communities after reassignment")

# --- Step 3: Export binary files ---
print(f"\n=== Step 3: Export binary files ===")
meta_communities = {}
for c in communities:
    mask = new_cid == c
    pts = positions[mask]
    meta_communities[int(c)] = {
        "centroid": [float(pts[:,0].mean()), float(pts[:,1].mean()), float(pts[:,2].mean())],
        "color": [int(r[mask].mean()), int(g[mask].mean()), int(b[mask].mean())],
        "count": int(mask.sum()),
    }

with open("/tmp/puppet/puppet_meta.json", "w") as f:
    json.dump({"n": n, "communities": meta_communities}, f)
positions.astype(np.float32).tofile("/tmp/puppet/puppet_positions.bin")
np.column_stack([r, g, b]).astype(np.int32).tofile("/tmp/puppet/puppet_colors.bin")
new_cid.astype(np.int32).tofile("/tmp/puppet/puppet_communities.bin")
print("  Wrote binary files")

# --- Step 4: Build skeleton (z-score overlap MST) ---
print(f"\n=== Step 4: Build skeleton ===")
com_stats = {}
for c in communities:
    mask = new_cid == c
    pts = positions[mask]
    com_stats[c] = {
        "mean": pts.mean(axis=0),
        "std": pts.std(axis=0),
        "count": int(mask.sum()),
    }

# Pairwise z-score overlap
edges = []
for i, ca in enumerate(communities):
    for cb in communities[i+1:]:
        sa, sb = com_stats[ca], com_stats[cb]
        delta = np.abs(sa["mean"] - sb["mean"])
        pooled_std = np.sqrt(sa["std"]**2 + sb["std"]**2)
        pooled_std = np.maximum(pooled_std, 1e-6)
        z_scores = delta / pooled_std
        weight = float(z_scores.max())

        wa = 1.0 / (np.linalg.norm(sa["std"]) + 1e-6)
        wb = 1.0 / (np.linalg.norm(sb["std"]) + 1e-6)
        joint = ((sa["mean"] * wb + sb["mean"] * wa) / (wa + wb)).tolist()
        edges.append((weight, ca, cb, joint))

# Find root: most central community
overall_centroid = positions.mean(axis=0)
root_cid = min(communities, key=lambda c: np.linalg.norm(com_stats[c]["mean"] - overall_centroid))

# Build adjacency
adj_map = {}
for weight, ca, cb, joint in edges:
    adj_map.setdefault(ca, []).append((weight, cb, joint))
    adj_map.setdefault(cb, []).append((weight, ca, joint))

# Prim's MST
visited = {root_cid}
skeleton = {}
skeleton[root_cid] = {"parent": None, "joint": None, "children": [],
                       "centroid": com_stats[root_cid]["mean"].tolist(),
                       "count": com_stats[root_cid]["count"]}

heap = []
for weight, neighbor, joint in adj_map.get(root_cid, []):
    heapq.heappush(heap, (weight, neighbor, root_cid, joint))

while heap:
    weight, node, parent, joint = heapq.heappop(heap)
    if node in visited:
        continue
    visited.add(node)
    skeleton[node] = {
        "parent": parent, "joint": joint, "children": [],
        "centroid": com_stats[node]["mean"].tolist(),
        "count": com_stats[node]["count"]
    }
    skeleton[parent]["children"].append(node)
    for w, neighbor, j in adj_map.get(node, []):
        if neighbor not in visited:
            heapq.heappush(heap, (w, neighbor, node, j))

# Print tree
def print_tree(cid, depth=0):
    info = skeleton[cid]
    col = meta_communities[cid]["color"]
    print(f"{'  '*depth}[{cid:2d}] n={info['count']:4d}  rgb=({col[0]:3d},{col[1]:3d},{col[2]:3d})")
    for child in info["children"]:
        print_tree(child, depth + 1)

print(f"  Root: {root_cid}")
print_tree(root_cid)

# Export skeleton
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if hasattr(obj, 'item'): return obj.item()
        return super().default(obj)

skeleton_export = {}
for cid_key, info in skeleton.items():
    skeleton_export[str(cid_key)] = info

output = {"root": int(root_cid), "skeleton": skeleton_export}
with open("/tmp/puppet/skeleton.json", "w") as f:
    json.dump(output, f, indent=2, cls=NumpyEncoder)
print(f"\n  Wrote skeleton.json")

# Summary
print(f"\n=== Summary ===")
print(f"  {len(communities)} communities, {n} points")
print(f"  Resolution: {RESOLUTION}")
for c in communities:
    col = meta_communities[c]["color"]
    cnt = meta_communities[c]["count"]
    print(f"  [{c:2d}] n={cnt:5d}  rgb=({col[0]:3d},{col[1]:3d},{col[2]:3d})")
