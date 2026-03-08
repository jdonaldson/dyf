"""
Automatic skeleton labeling v4.
Strategy:
  1. Find feet (symmetric terminal pair at one vertical extreme)
  2. Trace feet to LCA → hips
  3. Identify leg subtrees
  4. Everything else = upper body
  5. Find spine path: from hips, walk upward toward the topmost community
  6. Branches off the spine = arms (by lateral displacement)
  7. Remaining spine terminals = head
"""
import json
import numpy as np

positions = np.fromfile("/tmp/puppet/puppet_positions.bin", dtype=np.float32).reshape(-1, 3)
cid = np.fromfile("/tmp/puppet/puppet_communities.bin", dtype=np.int32)

with open("/tmp/puppet/skeleton.json") as f:
    skel_data = json.load(f)
skeleton = skel_data["skeleton"]
root = skel_data["root"]

with open("/tmp/puppet/puppet_meta.json") as f:
    meta = json.load(f)

com_centroids = {}
for cid_str, info in meta["communities"].items():
    com_centroids[int(cid_str)] = np.array(info["centroid"])

# --- Vertical axis from PCA ---
centered = positions - positions.mean(axis=0)
cov = np.cov(centered.T)
eigenvalues, eigenvectors = np.linalg.eigh(cov)
vertical = eigenvectors[:, np.argmax(eigenvalues)]

v_proj = {c: float(np.dot(com_centroids[c], vertical)) for c in com_centroids}

# Lateral axis
lat_raw = np.cross(vertical, [0, 0, 1])
if np.linalg.norm(lat_raw) < 0.1:
    lat_raw = np.cross(vertical, [0, 1, 0])
lateral = lat_raw / np.linalg.norm(lat_raw)
l_proj = {c: float(np.dot(com_centroids[c], lateral)) for c in com_centroids}

# --- Helpers ---
def get_terminals():
    return [int(k) for k, v in skeleton.items() if not v["children"]]

def ancestors(cid_key):
    path = [cid_key]
    node = skeleton[str(cid_key)]
    while node["parent"] is not None:
        path.append(node["parent"])
        node = skeleton[str(node["parent"])]
    return path

def get_subtree(cid_key):
    result = [cid_key]
    for child in skeleton[str(cid_key)]["children"]:
        result.extend(get_subtree(child))
    return result

def check_extreme_pair(t1, t2):
    """Score a pair: must be at similar vertical level with lateral separation."""
    model_height = max(v_proj.values()) - min(v_proj.values())
    lat_sep = abs(l_proj[t1] - l_proj[t2])
    vert_diff = abs(v_proj[t1] - v_proj[t2])
    # Pair must be within 10% of model height vertically
    if vert_diff > model_height * 0.15:
        return -1
    return lat_sep

# --- Step 1: Find feet ---
terminals = get_terminals()
sorted_terms = sorted(terminals, key=lambda c: v_proj[c])

print(f"Terminals ({len(terminals)}): {sorted_terms}")
for t in sorted_terms:
    info = meta["communities"][str(t)]
    print(f"  [{t:2d}] v={v_proj[t]:+.3f}  l={l_proj[t]:+.3f}  n={info['count']:4d}  rgb=({info['color'][0]:3d},{info['color'][1]:3d},{info['color'][2]:3d})")

# Check the 2 most extreme terminals at each end
bottom_score = check_extreme_pair(sorted_terms[0], sorted_terms[1]) if len(sorted_terms) >= 2 else -1
top_score = check_extreme_pair(sorted_terms[-1], sorted_terms[-2]) if len(sorted_terms) >= 2 else -1

print(f"\nBottom pair: ({sorted_terms[0]},{sorted_terms[1]}) score={bottom_score:.3f}")
print(f"Top pair:    ({sorted_terms[-1]},{sorted_terms[-2]}) score={top_score:.3f}")

if top_score > bottom_score:
    feet = (sorted_terms[-1], sorted_terms[-2])
    vertical = -vertical
    v_proj = {c: -v for c, v in v_proj.items()}
    print("Flipped vertical (feet at top)")
else:
    feet = (sorted_terms[0], sorted_terms[1])

if top_score < 0 and bottom_score < 0:
    print("WARNING: No clear symmetric pair found at either extreme")
    feet = (sorted_terms[0], sorted_terms[1])

left_foot = feet[0] if l_proj[feet[0]] > l_proj[feet[1]] else feet[1]
right_foot = feet[1] if left_foot == feet[0] else feet[0]
print(f"Feet: L=[{left_foot}] R=[{right_foot}]")

# --- Step 2: Find hips (LCA of feet) ---
left_path = ancestors(left_foot)
right_path = ancestors(right_foot)
left_set = set(left_path)
hips = None
for node in right_path:
    if node in left_set:
        hips = node
        break

left_leg = left_path[:left_path.index(hips)]
right_leg = right_path[:right_path.index(hips)]
leg_cids = set(left_leg + right_leg)

print(f"Hips: [{hips}]")
print(f"Left leg:  {left_leg}")
print(f"Right leg: {right_leg}")

# --- Step 3: Label legs ---
labels = {}
labels[hips] = "hips"

leg_names_up = ["foot", "shin", "knee", "thigh"]
for chain, side in [(left_leg, "L"), (right_leg, "R")]:
    for i, c in enumerate(chain):
        if i < len(leg_names_up):
            labels[c] = f"{side}_{leg_names_up[i]}"
        else:
            labels[c] = f"{side}_leg_{i}"

# --- Step 4: Find spine path ---
# The spine goes from hips upward toward the topmost community.
# Walk the tree: from hips, at each node, pick the child whose subtree
# contains the topmost (highest v_proj) community. That child is "spine."
# Other children are lateral branches (arms).

all_cids = set(int(k) for k in skeleton.keys())
upper_cids = all_cids - leg_cids - {hips}

# Find the topmost community (highest v_proj, excluding legs)
top_community = max(upper_cids, key=lambda c: v_proj[c])
print(f"\nTopmost community: [{top_community}] v={v_proj[top_community]:+.3f}")

# Trace from topmost community down to hips (or root)
top_path = ancestors(top_community)
# Find where this path intersects with hips' ancestor path
hips_path = ancestors(hips)

# The spine path is from hips upward through the tree toward top_community.
# It might go through the root.
spine_path_set = set(top_path) & set(hips_path)  # common ancestors
# But we also need the path FROM hips TO top_community through the tree.

# Better: find LCA of hips and top_community
hips_set = set(hips_path)
lca_top = None
for node in top_path:
    if node in hips_set:
        lca_top = node
        break

# Spine = path from hips up to LCA, then down to top_community
spine_up = hips_path[:hips_path.index(lca_top) + 1]  # hips → LCA
spine_down = top_path[:top_path.index(lca_top)]  # top → LCA (reversed, excluding LCA)
spine_down.reverse()  # LCA → top (but LCA already in spine_up)

spine_path = spine_up + spine_down  # hips, ..., LCA, ..., top
spine_set = set(spine_path)

print(f"Spine path: {spine_path}")

# --- Step 5: Find arm branches (off the spine, not legs) ---
arm_branches = []
for spine_node in spine_path:
    children = skeleton[str(spine_node)]["children"]
    for child in children:
        if child not in spine_set and child not in leg_cids and child != hips:
            subtree = get_subtree(child)
            # Compute average lateral displacement of subtree
            avg_lat = np.mean([l_proj[c] for c in subtree])
            arm_branches.append((child, subtree, avg_lat))

print(f"\nArm branches off spine:")
for child, subtree, avg_lat in arm_branches:
    print(f"  [{child}] subtree={subtree} avg_lateral={avg_lat:+.3f}")

# Classify: positive lateral = left arm, negative = right arm
left_arms = sorted([b for b in arm_branches if b[2] > 0.05], key=lambda b: b[2], reverse=True)
right_arms = sorted([b for b in arm_branches if b[2] < -0.05], key=lambda b: b[2])
midline_branches = [b for b in arm_branches if abs(b[2]) <= 0.05]

print(f"\nLeft arm branches: {[b[0] for b in left_arms]}")
print(f"Right arm branches: {[b[0] for b in right_arms]}")
print(f"Midline branches: {[b[0] for b in midline_branches]}")

# Label arm chains
arm_names = ["upper_arm", "forearm", "hand", "fingers"]
for branches, side in [(left_arms, "L"), (right_arms, "R")]:
    # Merge all subtree nodes for this side, sort by distance from spine
    all_arm_cids = []
    for _, subtree, _ in branches:
        all_arm_cids.extend(subtree)
    # Sort by distance from the nearest spine node (inner to outer)
    spine_centroids = np.array([com_centroids[c] for c in spine_path])
    def dist_to_spine(c):
        return min(np.linalg.norm(com_centroids[c] - sc) for sc in spine_centroids)
    all_arm_cids.sort(key=dist_to_spine)
    for i, c in enumerate(all_arm_cids):
        if i < len(arm_names):
            labels[c] = f"{side}_{arm_names[i]}"
        else:
            labels[c] = f"{side}_arm_{i}"

# Add midline branches to spine
for _, subtree, _ in midline_branches:
    spine_path.extend(subtree)
    spine_set.update(subtree)

# --- Step 6: Label spine (everything on spine path that isn't hips) ---
spine_non_hips = [c for c in spine_path if c != hips and c not in labels]
spine_sorted = sorted(spine_non_hips, key=lambda c: v_proj[c])

# Name from bottom to top
spine_names = ["lower_torso", "torso", "upper_torso", "chest", "neck", "face",
               "forehead", "hat_brim", "hat_top", "hat_peak"]
for i, c in enumerate(spine_sorted):
    if i < len(spine_names):
        labels[c] = spine_names[i]
    else:
        labels[c] = f"spine_{i}"

# --- Print results ---
print(f"\n{'='*70}")
print("LABELED SKELETON")
print(f"{'='*70}")

def print_labeled_tree(cid_key, depth=0):
    label = labels.get(cid_key, f"???_{cid_key}")
    info = meta["communities"][str(cid_key)]
    col = info["color"]
    n = info["count"]
    print(f"{'  '*depth}[{cid_key:2d}] {label:<16} n={n:4d}  "
          f"v={v_proj[cid_key]:+.3f}  l={l_proj[cid_key]:+.3f}  "
          f"rgb=({col[0]:3d},{col[1]:3d},{col[2]:3d})")
    for child in skeleton[str(cid_key)]["children"]:
        print_labeled_tree(child, depth + 1)

print_labeled_tree(root)

# Validate
print(f"\n--- Validation ---")
all_labels = list(labels.values())
for expected in ["L_foot", "R_foot", "L_upper_arm", "R_upper_arm",
                 "hips", "face", "torso", "chest"]:
    found = expected in all_labels
    cids = [c for c, l in labels.items() if l == expected]
    pos_str = ""
    if cids:
        c = cids[0]
        col = meta["communities"][str(c)]["color"]
        pos_str = f" [{c}] rgb=({col[0]:3d},{col[1]:3d},{col[2]:3d})"
    print(f"  {expected:<16} {'FOUND' if found else 'MISSING'}{pos_str}")

unlabeled = [int(k) for k in skeleton.keys() if int(k) not in labels]
if unlabeled:
    print(f"\n  UNLABELED: {unlabeled}")

leg_count = sum(1 for l in all_labels if "foot" in l)
arm_labels = [l for l in all_labels if "upper_arm" in l or "hand" in l or "forearm" in l]
print(f"\n  Legs: {leg_count}, Arm parts: {len(arm_labels)}")
if leg_count == 2 and len(arm_labels) >= 2:
    print("  => HUMANOID")

# Save
for cid_key in skeleton:
    skeleton[cid_key]["label"] = labels.get(int(cid_key), f"part_{cid_key}")
skel_data["labels"] = {str(c): l for c, l in labels.items()}

with open("/tmp/puppet/skeleton.json", "w") as f:
    json.dump(skel_data, f, indent=2)
print(f"\nSaved to /tmp/puppet/skeleton.json")
