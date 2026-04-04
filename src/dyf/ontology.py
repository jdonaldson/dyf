"""
DAG taxonomy and ontology extraction from embedding spaces.

Discovers hierarchical structure using neighbor diversity as a generality signal:
points with diverse neighbors (connecting many topics) are "general", while points
with coherent neighbors (tight clusters) are "specific".

Key algorithms:
- mine_dag_chains: Extract directed chains from general → specific concepts
- build_dag_taxonomy: Build a navigable DAG lattice (multiple parents allowed)
- build_unified_ontology: Two-tier ontology covering dense and sparse regions
- build_rog_ontology: Recursive Ontological Generation with adaptive thresholds

Example:
    >>> from dyf import build_dag_taxonomy
    >>> taxonomy = build_dag_taxonomy(embeddings, verbose=True)
    >>> print(taxonomy.summary())
    >>> ancestors = taxonomy.get_ancestors(node_idx)

    >>> from dyf import build_rog_ontology
    >>> result = build_rog_ontology(embeddings, verbose=True)
    >>> print(result.summary())
"""

import logging
from collections import defaultdict
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)


# ============================================================
# DAG Mining
# ============================================================


@dataclass
class DAGChain:
    """
    A single DAG chain representing a hierarchical path.

    Chains flow from general concepts (high neighbor diversity) to
    specific concepts (low neighbor diversity).

    Attributes:
        indices: Point indices in chain order (general → specific)
        coherence: Average pairwise similarity within chain (higher = tighter)
        diversity_range: Difference between first and last diversity scores
    """
    indices: np.ndarray
    coherence: float
    diversity_range: float

    def __len__(self) -> int:
        return len(self.indices)


@dataclass
class DAGMiningResult:
    """
    Result of DAG mining on an embedding space.

    DAG mining finds hierarchical structures by using neighbor diversity
    as a generality signal. Points with diverse neighbors (connecting
    many topics) are considered "general", while points with coherent
    neighbors (tight clusters) are "specific".

    Attributes:
        chains: List of extracted DAG chains
        diversity: Neighbor diversity score for each point
        parent_child_edges: List of (parent, child, similarity, gap) tuples
        n_components: Number of connected components in the DAG
    """
    chains: list[DAGChain]
    diversity: np.ndarray
    parent_child_edges: list[tuple[int, int, float, float]]
    n_components: int

    def __len__(self) -> int:
        return len(self.chains)

    def summary(self) -> str:
        """Return summary statistics."""
        if not self.chains:
            return "DAGMiningResult: No chains found"

        lengths = [len(c) for c in self.chains]
        coherences = [c.coherence for c in self.chains]

        return (
            f"DAGMiningResult:\n"
            f"  Chains: {len(self.chains)}\n"
            f"  Length: min={min(lengths)}, max={max(lengths)}, "
            f"avg={np.mean(lengths):.1f}\n"
            f"  Coherence: min={min(coherences):.2f}, max={max(coherences):.2f}, "
            f"avg={np.mean(coherences):.2f}\n"
            f"  Components: {self.n_components}\n"
            f"  Parent-child edges: {len(self.parent_child_edges)}"
        )

    def get_chains_by_length(self, min_length: int = 3) -> list[DAGChain]:
        """Return chains with at least min_length nodes."""
        return [c for c in self.chains if len(c) >= min_length]

    def get_chains_by_coherence(self, min_coherence: float = 0.65) -> list[DAGChain]:
        """Return chains with at least min_coherence."""
        return [c for c in self.chains if c.coherence >= min_coherence]


def compute_neighbor_diversity(
    embeddings: np.ndarray,
    k: int = 15,
    neighbors: np.ndarray | None = None,
) -> np.ndarray:
    """
    Compute neighbor diversity for each point.

    Diversity measures how dissimilar a point's neighbors are to each other.
    High diversity indicates a "general" concept that connects multiple topics.
    Low diversity indicates a "specific" concept within a tight cluster.

    Args:
        embeddings: Normalized embeddings (n, d)
        k: Number of neighbors to consider
        neighbors: Pre-computed k-NN indices (n, k+1). If None, computed internally.

    Returns:
        Diversity scores for each point (n,). Higher = more general.

    Example:
        >>> diversity = compute_neighbor_diversity(embeddings, k=15)
        >>> general_idx = np.argsort(diversity)[-10:]  # Most general concepts
        >>> specific_idx = np.argsort(diversity)[:10]  # Most specific concepts
    """
    n_points, dim = embeddings.shape

    # Compute k-NN if not provided
    if neighbors is None:
        from sklearn.neighbors import NearestNeighbors
        nn = NearestNeighbors(n_neighbors=k + 1, metric='cosine')
        nn.fit(embeddings)
        _, neighbors = nn.kneighbors(embeddings)

    diversity = np.zeros(n_points)

    for i in range(n_points):
        # Get neighbor embeddings (exclude self)
        neighb_idx = neighbors[i, 1:k + 1]
        neighb_emb = embeddings[neighb_idx]

        # Compute pairwise similarity among neighbors
        pairwise = neighb_emb @ neighb_emb.T

        # Get off-diagonal (neighbor-to-neighbor similarity)
        mask = ~np.eye(len(neighb_idx), dtype=bool)
        neighbor_coherence = pairwise[mask].mean()

        # Diversity = 1 - coherence (higher diversity = more general)
        diversity[i] = 1 - neighbor_coherence

    return diversity


@dataclass
class HubScoreResult:
    """
    Result of hub score computation.

    Hub nodes are structural centers that hold the embedding space together.
    They are characterized by:
    - High eigenvector centrality (in tightly woven neighborhoods)
    - Low distance to global centroid (central position in space)

    Unlike PageRank, which rewards being pointed to by many nodes (both categories
    AND popular instances qualify), eigenvector centrality rewards being in a
    tightly interconnected cluster. Abstract concepts cluster more tightly than
    popular instances.

    Attributes:
        scores: Hub score for each point (higher = more structural/general)
        eigenvector: Eigenvector centrality for each point
        distance_to_center: Distance to global centroid for each point
        eigenvector_z: Z-scored eigenvector centrality
        distance_z: Z-scored distance to center
    """
    scores: np.ndarray
    eigenvector: np.ndarray
    distance_to_center: np.ndarray
    eigenvector_z: np.ndarray
    distance_z: np.ndarray

    def get_hub_nodes(self, threshold: float = 0.0) -> np.ndarray:
        """Return indices of nodes with hub score above threshold."""
        return np.where(self.scores > threshold)[0]

    def get_top_hubs(self, k: int = 100) -> np.ndarray:
        """Return indices of top k hub nodes."""
        return np.argsort(self.scores)[-k:][::-1]


def compute_hub_score(
    embeddings: np.ndarray,
    k: int = 30,
    n_iterations: int = 30,
    neighbors: np.ndarray | None = None,
    similarities: np.ndarray | None = None,
) -> HubScoreResult:
    """
    Compute hub score for each point in the embedding space.

    Hub score identifies structural hub nodes that hold the embedding space
    together, as opposed to popular instances that fill it. The score combines:
    - Eigenvector centrality: rewards being in a tightly woven neighborhood
    - Distance to global centroid: rewards central position in the space

    General/structural concepts score high on both signals; specific instances
    score low on both.

    Args:
        embeddings: Normalized embeddings (n, d)
        k: Number of neighbors for k-NN graph
        n_iterations: Iterations for eigenvector power method
        neighbors: Pre-computed k-NN indices (n, k+1). If None, computed internally.
        similarities: Pre-computed similarities (n, k+1). If None, computed internally.

    Returns:
        HubScoreResult with scores and component signals.

    Example:
        >>> result = compute_hub_score(embeddings)
        >>> hub_nodes = result.get_hub_nodes(threshold=1.0)
        >>> print(f"Found {len(hub_nodes)} hub nodes")
        >>>
        >>> # Use for taxonomy direction
        >>> if result.scores[idx_animal] > result.scores[idx_dog]:
        ...     print("Animal is more general than Dog")
    """
    n_points, dim = embeddings.shape

    # Normalize embeddings
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = embeddings / np.where(norms > 0, norms, 1)

    # Compute k-NN if not provided
    if neighbors is None or similarities is None:
        from sklearn.neighbors import NearestNeighbors
        nn = NearestNeighbors(n_neighbors=k + 1, metric='cosine')
        nn.fit(embeddings)
        distances, neighbors = nn.kneighbors(embeddings)
        similarities = 1 - distances

    # 1. Eigenvector centrality (no damping - key difference from PageRank)
    eigen = np.ones(n_points)
    for _ in range(n_iterations):
        new_eigen = np.zeros(n_points)
        for i in range(n_points):
            for j in range(1, k + 1):
                if j < neighbors.shape[1]:
                    new_eigen[i] += eigen[neighbors[i, j]] * similarities[i, j]
        norm = np.linalg.norm(new_eigen)
        if norm > 0:
            eigen = new_eigen / norm
        else:
            break

    # 2. Distance to global centroid
    global_centroid = embeddings.mean(axis=0)
    centroid_norm = np.linalg.norm(global_centroid)
    if centroid_norm > 0:
        global_centroid = global_centroid / centroid_norm
    distance_to_center = 1 - np.dot(embeddings, global_centroid)

    # 3. Z-score normalize
    eigen_z = (eigen - eigen.mean()) / (eigen.std() + 1e-10)
    dist_z = (distance_to_center - distance_to_center.mean()) / (distance_to_center.std() + 1e-10)

    # 4. Combine: high eigenvector, close to center (low distance)
    hub_score = eigen_z - dist_z

    return HubScoreResult(
        scores=hub_score,
        eigenvector=eigen,
        distance_to_center=distance_to_center,
        eigenvector_z=eigen_z,
        distance_z=dist_z,
    )


def _prepare_knn_graph(
    embeddings: np.ndarray,
    k_neighbors: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    L2-normalize embeddings and build a k-NN graph.

    Shared setup used by mine_dag_chains, build_dag_taxonomy, and
    build_rog_ontology.

    Args:
        embeddings: Raw embeddings (n, d).
        k_neighbors: Number of neighbors (the graph will query k_neighbors + 1
            to account for self-neighbor).

    Returns:
        Tuple of (embeddings_normed, neighbors, similarities) where
        embeddings_normed is float32 L2-normed, neighbors is (n, k+1) indices,
        and similarities is 1 - cosine_distance.
    """
    from sklearn.neighbors import NearestNeighbors

    embeddings = np.array(embeddings, dtype=np.float32)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = embeddings / np.where(norms > 0, norms, 1)

    nn = NearestNeighbors(n_neighbors=k_neighbors + 1, metric='cosine')
    nn.fit(embeddings)
    distances, neighbors = nn.kneighbors(embeddings)
    similarities = 1 - distances

    return embeddings, neighbors, similarities


def _find_parent_child_edges(
    n_points: int,
    neighbors: np.ndarray,
    similarities: np.ndarray,
    diversity: np.ndarray,
    similarity_threshold: float,
    diversity_gap_threshold: float,
) -> list[tuple[int, int, float, float]]:
    """
    Discover directed parent-child edges from the k-NN graph.

    An edge ``(parent, child, sim, gap)`` is emitted when two neighbors
    exceed *similarity_threshold* and their diversity difference exceeds
    *diversity_gap_threshold*.  The node with higher diversity is the parent.
    """
    parent_child_edges: list[tuple[int, int, float, float]] = []

    for i in range(n_points):
        for j_idx, j in enumerate(neighbors[i, 1:16]):
            sim = similarities[i, j_idx + 1]
            if sim < similarity_threshold:
                continue

            div_i = diversity[i]
            div_j = diversity[j]
            gap = div_i - div_j

            # Parent has higher diversity (more general)
            if gap > diversity_gap_threshold:
                parent_child_edges.append((i, j, float(sim), float(gap)))
            elif gap < -diversity_gap_threshold:
                parent_child_edges.append((j, i, float(sim), float(-gap)))

    return parent_child_edges


def _extract_chains_from_component(
    component_nodes: set[int],
    children: dict[int, list[tuple[int, float, float]]],
    parents: dict[int, list[tuple[int, float, float]]],
    diversity: np.ndarray,
    min_chain_length: int,
) -> list[list[int]]:
    """
    Extract maximal chains from a connected component.

    Finds root nodes (no parents within the component) and greedily
    follows the highest-similarity child edge to build chains.
    """
    # Find roots (nodes with no parents in this component)
    roots = [
        n for n in component_nodes
        if n not in parents or all(p not in component_nodes for p, _, _ in parents[n])
    ]

    if not roots:
        # No clear root - pick highest diversity node
        roots = [max(component_nodes, key=lambda x: diversity[x])]

    all_chains: list[list[int]] = []
    for root in roots:
        chain = [root]
        current = root
        visited = {root}

        while True:
            if current not in children:
                break
            valid = [
                (c, s, g) for c, s, g in children[current]
                if c in component_nodes and c not in visited
            ]
            if not valid:
                break
            # Pick by highest similarity
            valid.sort(key=lambda x: x[1], reverse=True)
            next_node = valid[0][0]
            chain.append(next_node)
            visited.add(next_node)
            current = next_node

        if len(chain) >= min_chain_length:
            all_chains.append(chain)

    return all_chains


def mine_dag_chains(
    embeddings: np.ndarray,
    k_neighbors: int = 30,
    similarity_threshold: float = 0.55,
    diversity_gap_threshold: float = 0.02,
    min_chain_length: int = 3,
    verbose: bool = False,
) -> DAGMiningResult:
    """
    Mine DAG (directed acyclic graph) structures from embedding space.

    Finds hierarchical chains by using neighbor diversity as a generality
    signal. Points with diverse neighbors connect many topics (general),
    while points with coherent neighbors form tight clusters (specific).

    Chains flow from general → specific, following the diversity gradient.

    Args:
        embeddings: Embeddings to analyze (n, d). Will be L2-normalized.
        k_neighbors: Number of neighbors for k-NN graph and diversity computation.
        similarity_threshold: Minimum similarity for parent-child edges.
        diversity_gap_threshold: Minimum diversity difference for directed edge.
        min_chain_length: Minimum chain length to return.
        verbose: Print progress information.

    Returns:
        DAGMiningResult with chains, diversity scores, and edge information.

    Example:
        >>> from dyf import mine_dag_chains
        >>>
        >>> result = mine_dag_chains(embeddings, verbose=True)
        >>> print(result.summary())
        >>>
        >>> # Get clean hierarchies
        >>> clean = result.get_chains_by_coherence(min_coherence=0.65)
        >>> for chain in clean[:10]:
        ...     print(f"[len={len(chain)}] {chain.indices}")

    Notes:
        - Low diversity points (decades, months) have highly coherent neighbors
        - High diversity points sit at intersections of multiple topics
        - Chains often converge to common "sinks" (abstract concepts)
        - ~100% of extracted chains follow monotonic diversity gradient
    """
    # Normalize + build k-NN graph
    embeddings, neighbors, similarities = _prepare_knn_graph(embeddings, k_neighbors)
    n_points, dim = embeddings.shape

    if verbose:
        logger.info("Mining DAG chains: %s points, dim=%d", f"{n_points:,}", dim)
        logger.info("Building k-NN graph...")

    # Compute neighbor diversity
    if verbose:
        logger.info("Computing neighbor diversity...")
    diversity = compute_neighbor_diversity(embeddings, k=15, neighbors=neighbors)

    if verbose:
        logger.debug("Diversity range: %.3f to %.3f", diversity.min(), diversity.max())

    # Find parent-child pairs based on diversity gradient
    if verbose:
        logger.info("Finding parent-child pairs...")

    parent_child_edges = _find_parent_child_edges(
        n_points, neighbors, similarities, diversity,
        similarity_threshold, diversity_gap_threshold,
    )

    if verbose:
        logger.debug("Found %d parent-child pairs", len(parent_child_edges))

    # Build adjacency lists
    children = defaultdict(list)
    parents = defaultdict(list)

    for parent, child, sim, gap in parent_child_edges:
        children[parent].append((child, sim, gap))
        parents[child].append((parent, sim, gap))

    # Find connected components using union-find
    parent_map = list(range(n_points))

    def find(x):
        if parent_map[x] != x:
            parent_map[x] = find(parent_map[x])
        return parent_map[x]

    def union(x, y):
        px, py = find(x), find(y)
        if px != py:
            parent_map[px] = py

    for parent, child, _, _ in parent_child_edges:
        union(parent, child)

    components = defaultdict(set)
    for i in range(n_points):
        if i in children or i in parents:
            components[find(i)].add(i)

    components = {k: v for k, v in components.items() if len(v) >= min_chain_length}
    n_components = len(components)

    if verbose:
        logger.debug("Found %d non-trivial components", n_components)

    # Extract chains from components
    if verbose:
        logger.info("Extracting chains...")

    chains = []
    for comp_nodes in components.values():
        comp_chains = _extract_chains_from_component(
            comp_nodes, children, parents, diversity, min_chain_length,
        )
        for chain in comp_chains:
            # Compute chain metrics
            chain_emb = embeddings[chain]
            coherence = float((chain_emb @ chain_emb.T).mean())
            div_range = float(diversity[chain[0]] - diversity[chain[-1]])

            chains.append(DAGChain(
                indices=np.array(chain),
                coherence=coherence,
                diversity_range=div_range
            ))

    # Sort by quality (coherence * length)
    chains.sort(key=lambda x: x.coherence * len(x), reverse=True)

    if verbose:
        logger.info("Extracted %d chains (length >= %d)", len(chains), min_chain_length)
        if chains:
            clean = [c for c in chains if c.coherence >= 0.65 and len(c) >= 4]
            logger.debug("Clean hierarchies (coh >= 0.65, len >= 4): %d", len(clean))

    return DAGMiningResult(
        chains=chains,
        diversity=diversity,
        parent_child_edges=parent_child_edges,
        n_components=n_components
    )


# ============================================================
# DAG Taxonomy
# ============================================================


@dataclass
class DAGTaxonomy:
    """
    A navigable DAG taxonomy extracted from embedding space.

    Provides lattice operations for exploring hierarchical relationships:
    - Find ancestors/descendants of a concept
    - Find common ancestors between concepts
    - Find paths through the taxonomy
    - Identify convergence points (abstract attractors)

    The taxonomy is a true lattice (not a tree) - nodes can have multiple
    parents, enabling multiple inheritance and diamond patterns.

    Attributes:
        n_nodes: Total number of nodes in the taxonomy
        diversity: Neighbor diversity score for each node
        children: Dict mapping node → list of (child, similarity, gap)
        parents: Dict mapping node → list of (parent, similarity, gap)
        roots: Nodes with no parents (most general concepts)
        leaves: Nodes with no children (most specific concepts)

    Example:
        >>> from dyf import build_dag_taxonomy
        >>>
        >>> taxonomy = build_dag_taxonomy(embeddings, verbose=True)
        >>> print(taxonomy.summary())
        >>>
        >>> # Find ancestors of a concept
        >>> ancestors = taxonomy.get_ancestors(node_idx, max_depth=5)
        >>>
        >>> # Find common ancestors between two concepts
        >>> common = taxonomy.get_common_ancestors(idx_a, idx_b)
        >>>
        >>> # Get convergence points (highly connected abstract concepts)
        >>> hubs = taxonomy.get_convergence_points(min_parents=10)
    """
    n_nodes: int
    diversity: np.ndarray
    children: dict[int, list[tuple[int, float, float]]]
    parents: dict[int, list[tuple[int, float, float]]]
    roots: list[int]
    leaves: list[int]

    def __len__(self) -> int:
        """Number of nodes with edges."""
        return len(set(self.children.keys()) | set(self.parents.keys()))

    def summary(self) -> str:
        """Return summary statistics."""
        n_edges = sum(len(v) for v in self.children.values())
        n_multi_parent = sum(1 for v in self.parents.values() if len(v) > 1)
        max_parents = max((len(v) for v in self.parents.values()), default=0)
        max_children = max((len(v) for v in self.children.values()), default=0)

        return (
            f"DAGTaxonomy:\n"
            f"  Nodes with edges: {len(self)}\n"
            f"  Total edges: {n_edges}\n"
            f"  Roots (no parents): {len(self.roots)}\n"
            f"  Leaves (no children): {len(self.leaves)}\n"
            f"  Multi-parent nodes: {n_multi_parent} ({100*n_multi_parent/max(len(self),1):.1f}%)\n"
            f"  Max parents: {max_parents}\n"
            f"  Max children: {max_children}"
        )

    def get_children(self, node: int) -> list[int]:
        """Get direct children of a node."""
        return [c for c, _, _ in self.children.get(node, [])]

    def get_parents(self, node: int) -> list[int]:
        """Get direct parents of a node."""
        return [p for p, _, _ in self.parents.get(node, [])]

    def get_ancestors(self, node: int, max_depth: int = 10) -> set:
        """
        Get all ancestors of a node (transitive closure upward).

        Args:
            node: Starting node index
            max_depth: Maximum depth to traverse

        Returns:
            Set of all ancestor node indices
        """
        ancestors = set()
        frontier = {node}
        depth = 0

        while frontier and depth < max_depth:
            next_frontier = set()
            for n in frontier:
                for parent, _, _ in self.parents.get(n, []):
                    if parent not in ancestors:
                        ancestors.add(parent)
                        next_frontier.add(parent)
            frontier = next_frontier
            depth += 1

        return ancestors

    def get_descendants(self, node: int, max_depth: int = 10) -> set:
        """
        Get all descendants of a node (transitive closure downward).

        Args:
            node: Starting node index
            max_depth: Maximum depth to traverse

        Returns:
            Set of all descendant node indices
        """
        descendants = set()
        frontier = {node}
        depth = 0

        while frontier and depth < max_depth:
            next_frontier = set()
            for n in frontier:
                for child, _, _ in self.children.get(n, []):
                    if child not in descendants:
                        descendants.add(child)
                        next_frontier.add(child)
            frontier = next_frontier
            depth += 1

        return descendants

    def get_common_ancestors(self, node_a: int, node_b: int, max_depth: int = 10) -> set:
        """
        Find common ancestors of two nodes.

        In a lattice (vs tree), there can be multiple common ancestors,
        not just a single LCA.

        Args:
            node_a: First node index
            node_b: Second node index
            max_depth: Maximum depth to search

        Returns:
            Set of common ancestor indices
        """
        ancestors_a = self.get_ancestors(node_a, max_depth)
        ancestors_b = self.get_ancestors(node_b, max_depth)
        return ancestors_a & ancestors_b

    def get_lowest_common_ancestors(self, node_a: int, node_b: int, max_depth: int = 10) -> list[int]:
        """
        Find lowest common ancestors (LCAs) of two nodes.

        Returns ancestors that are common to both nodes but have no
        descendants that are also common ancestors.

        Args:
            node_a: First node index
            node_b: Second node index
            max_depth: Maximum depth to search

        Returns:
            List of LCA indices (may be multiple in a lattice)
        """
        common = self.get_common_ancestors(node_a, node_b, max_depth)
        if not common:
            return []

        # Filter to those with no common descendants
        lcas = []
        for ancestor in common:
            descendants = self.get_descendants(ancestor, max_depth)
            # If none of its descendants are also common ancestors, it's an LCA
            if not (descendants & common):
                lcas.append(ancestor)

        return lcas

    def get_convergence_points(self, min_parents: int = 5) -> list[tuple[int, int]]:
        """
        Find convergence points (nodes with many incoming edges).

        These are abstract concepts where many paths converge.

        Args:
            min_parents: Minimum number of parents to qualify

        Returns:
            List of (node_idx, parent_count) tuples, sorted by count descending
        """
        convergence = [
            (node, len(parents))
            for node, parents in self.parents.items()
            if len(parents) >= min_parents
        ]
        convergence.sort(key=lambda x: x[1], reverse=True)
        return convergence

    def get_divergence_points(self, min_children: int = 5) -> list[tuple[int, int]]:
        """
        Find divergence points (nodes with many outgoing edges).

        These are branching hubs where paths diverge.

        Args:
            min_children: Minimum number of children to qualify

        Returns:
            List of (node_idx, child_count) tuples, sorted by count descending
        """
        divergence = [
            (node, len(children))
            for node, children in self.children.items()
            if len(children) >= min_children
        ]
        divergence.sort(key=lambda x: x[1], reverse=True)
        return divergence

    def get_diamond_patterns(self, limit: int = 100) -> list[tuple[int, int, int, int]]:
        """
        Find diamond patterns (A → B, A → C, B → D, C → D).

        Diamonds indicate multiple inheritance / non-tree structure.

        Args:
            limit: Maximum number of diamonds to return

        Returns:
            List of (top, left, right, bottom) tuples
        """
        diamonds = []

        for d in self.parents:  # potential bottom of diamond
            d_parents = self.get_parents(d)
            if len(d_parents) < 2:
                continue

            # Check if any two parents share a common parent (top of diamond)
            for i, p1 in enumerate(d_parents):
                for p2 in d_parents[i + 1:]:
                    common_grandparents = set(self.get_parents(p1)) & set(self.get_parents(p2))
                    for a in common_grandparents:
                        diamonds.append((a, p1, p2, d))
                        if len(diamonds) >= limit:
                            return diamonds

        return diamonds

    def get_path(self, start: int, end: int, max_depth: int = 10) -> list[int] | None:
        """
        Find a path from start to end (following parent→child direction).

        Args:
            start: Starting node (more general)
            end: Ending node (more specific)
            max_depth: Maximum path length

        Returns:
            List of node indices from start to end, or None if no path
        """
        if start == end:
            return [start]

        # BFS from start
        queue = [(start, [start])]
        visited = {start}

        while queue:
            current, path = queue.pop(0)
            if len(path) > max_depth:
                continue

            for child, _, _ in self.children.get(current, []):
                if child == end:
                    return path + [child]
                if child not in visited:
                    visited.add(child)
                    queue.append((child, path + [child]))

        return None

    def get_all_paths(self, start: int, end: int, max_depth: int = 10) -> list[list[int]]:
        """
        Find all paths from start to end.

        In a lattice, there can be multiple paths between nodes.

        Args:
            start: Starting node (more general)
            end: Ending node (more specific)
            max_depth: Maximum path length

        Returns:
            List of paths, where each path is a list of node indices
        """
        if start == end:
            return [[start]]

        paths = []

        def dfs(current: int, path: list[int]):
            if len(path) > max_depth:
                return
            if current == end:
                paths.append(path.copy())
                return

            for child, _, _ in self.children.get(current, []):
                if child not in path:  # avoid cycles (shouldn't happen in DAG)
                    path.append(child)
                    dfs(child, path)
                    path.pop()

        dfs(start, [start])
        return paths

    def get_node_depth(self, node: int) -> int:
        """
        Get the depth of a node (distance from nearest root).

        Args:
            node: Node index

        Returns:
            Minimum distance to any root, or -1 if not connected
        """
        if node in self.roots:
            return 0

        depth = 0
        frontier = {node}
        visited = {node}

        while frontier:
            depth += 1
            if depth > self.n_nodes:  # safety limit
                return -1

            next_frontier = set()
            for n in frontier:
                for parent, _, _ in self.parents.get(n, []):
                    if parent in self.roots:
                        return depth
                    if parent not in visited:
                        visited.add(parent)
                        next_frontier.add(parent)
            frontier = next_frontier

        return -1  # not connected to any root


def build_dag_taxonomy(
    embeddings: np.ndarray,
    k_neighbors: int = 30,
    similarity_threshold: float = 0.55,
    diversity_gap_threshold: float = 0.02,
    verbose: bool = False,
) -> DAGTaxonomy:
    """
    Build a navigable DAG taxonomy from embedding space.

    Creates a lattice structure where:
    - Nodes with high neighbor diversity are "general" (parents)
    - Nodes with low neighbor diversity are "specific" (children)
    - Multiple parents are allowed (non-tree structure)

    Args:
        embeddings: Embeddings to analyze (n, d). Will be L2-normalized.
        k_neighbors: Number of neighbors for k-NN graph.
        similarity_threshold: Minimum similarity for parent-child edges.
        diversity_gap_threshold: Minimum diversity difference for directed edge.
        verbose: Print progress information.

    Returns:
        DAGTaxonomy with navigation methods.

    Example:
        >>> from dyf import build_dag_taxonomy
        >>>
        >>> taxonomy = build_dag_taxonomy(embeddings)
        >>>
        >>> # Navigate the taxonomy
        >>> ancestors = taxonomy.get_ancestors(query_idx)
        >>> common = taxonomy.get_common_ancestors(idx_a, idx_b)
        >>> hubs = taxonomy.get_convergence_points(min_parents=10)
    """
    # Normalize + build k-NN graph
    embeddings, neighbors, similarities = _prepare_knn_graph(embeddings, k_neighbors)
    n_points, dim = embeddings.shape

    if verbose:
        logger.info("Building DAG taxonomy: %s points, dim=%d", f"{n_points:,}", dim)
        logger.info("Building k-NN graph...")

    # Compute neighbor diversity
    if verbose:
        logger.info("Computing neighbor diversity...")
    diversity = compute_neighbor_diversity(embeddings, k=15, neighbors=neighbors)

    # Find parent-child pairs
    if verbose:
        logger.info("Finding parent-child relationships...")

    children: dict[int, list[tuple[int, float, float]]] = defaultdict(list)
    parents: dict[int, list[tuple[int, float, float]]] = defaultdict(list)

    for i in range(n_points):
        for j_idx, j in enumerate(neighbors[i, 1:16]):
            sim = similarities[i, j_idx + 1]
            if sim < similarity_threshold:
                continue

            div_i = diversity[i]
            div_j = diversity[j]
            gap = div_i - div_j

            # Parent has higher diversity (more general)
            if gap > diversity_gap_threshold:
                children[i].append((j, float(sim), float(gap)))
                parents[j].append((i, float(sim), float(gap)))
            elif gap < -diversity_gap_threshold:
                children[j].append((i, float(sim), float(-gap)))
                parents[i].append((j, float(sim), float(-gap)))

    # Convert to regular dicts
    children = dict(children)
    parents = dict(parents)

    # Find roots (no parents) and leaves (no children)
    all_nodes = set(children.keys()) | set(parents.keys())
    roots = [n for n in all_nodes if n not in parents]
    leaves = [n for n in all_nodes if n not in children]

    if verbose:
        n_edges = sum(len(v) for v in children.values())
        n_multi = sum(1 for v in parents.values() if len(v) > 1)
        logger.debug("Nodes: %d, Edges: %d", len(all_nodes), n_edges)
        logger.debug("Roots: %d, Leaves: %d", len(roots), len(leaves))
        logger.debug("Multi-parent nodes: %d (%.1f%%)", n_multi, 100 * n_multi / max(len(all_nodes), 1))

    return DAGTaxonomy(
        n_nodes=n_points,
        diversity=diversity,
        children=children,
        parents=parents,
        roots=roots,
        leaves=leaves,
    )


@dataclass
class UnifiedOntologyResult:
    """
    Result of building a unified ontology from embedding space.

    The unified ontology combines:
    1. Main ontology (high similarity threshold) for dense regions
    2. Outlier ontology (lower threshold) for sparse regions
    3. Bridge edges connecting the two

    Attributes:
        ontology: The unified DAGTaxonomy
        main_nodes: Indices of nodes from main ontology
        outlier_nodes: Indices of nodes from outlier ontology
        excluded_nodes: Indices of double outliers (excluded)
        bridge_edges: Number of edges connecting main and outlier components
        main_threshold: Similarity threshold used for main ontology
        outlier_threshold: Similarity threshold used for outlier ontology
    """
    ontology: DAGTaxonomy
    main_nodes: np.ndarray
    outlier_nodes: np.ndarray
    excluded_nodes: np.ndarray
    bridge_edges: int
    main_threshold: float
    outlier_threshold: float

    def __len__(self) -> int:
        return len(self.ontology)

    def summary(self) -> str:
        """Return summary statistics."""
        total = self.ontology.n_nodes
        covered = len(self.main_nodes) + len(self.outlier_nodes)
        n_edges = sum(len(v) for v in self.ontology.children.values())

        return (
            f"UnifiedOntologyResult:\n"
            f"  Coverage: {covered}/{total} ({100*covered/total:.1f}%)\n"
            f"  Main nodes (sim≥{self.main_threshold}): {len(self.main_nodes)}\n"
            f"  Outlier nodes (sim≥{self.outlier_threshold}): {len(self.outlier_nodes)}\n"
            f"  Bridge edges: {self.bridge_edges}\n"
            f"  Excluded (double outliers): {len(self.excluded_nodes)}\n"
            f"  Total edges: {n_edges}\n"
            f"  Multi-parent: {sum(1 for v in self.ontology.parents.values() if len(v) > 1)}"
        )


def _merge_ontology_edges(
    main_ont: DAGTaxonomy,
    outlier_ont: DAGTaxonomy | None,
    outlier_to_global: dict[int, int],
) -> tuple[dict[int, list[tuple[int, float, float]]], dict[int, list[tuple[int, float, float]]]]:
    """
    Merge edges from main and outlier ontologies into unified dicts.

    Copies all edges from *main_ont* directly, then remaps outlier-local
    indices to global indices and adds those edges as well.

    Returns:
        (children, parents) defaultdicts with merged edges.
    """
    children: dict[int, list[tuple[int, float, float]]] = defaultdict(list)
    parents: dict[int, list[tuple[int, float, float]]] = defaultdict(list)

    # Add main ontology edges
    for parent, child_list in main_ont.children.items():
        for child, sim, gap in child_list:
            children[parent].append((child, sim, gap))
            parents[child].append((parent, sim, gap))

    # Add outlier ontology edges (remapped to global indices)
    if outlier_ont is not None:
        for parent, child_list in outlier_ont.children.items():
            global_parent = outlier_to_global[parent]
            for child, sim, gap in child_list:
                global_child = outlier_to_global[child]
                children[global_parent].append((global_child, sim, gap))
                parents[global_child].append((global_parent, sim, gap))

    return children, parents


def build_unified_ontology(
    embeddings: np.ndarray,
    main_similarity_threshold: float = 0.55,
    outlier_similarity_threshold: float = 0.45,
    diversity_gap_threshold: float = 0.02,
    outlier_diversity_gap: float = 0.015,
    k_neighbors: int = 30,
    verbose: bool = False,
) -> UnifiedOntologyResult:
    """
    Build a unified ontology that covers both dense and sparse embedding regions.

    Uses a two-tier approach:
    1. Build main ontology with high similarity threshold (dense regions)
    2. Build outlier ontology with lower threshold (sparse regions)
    3. Add bridge edges to connect them
    4. Exclude "double outliers" that don't fit either

    This achieves ~96% coverage vs ~89% for single-threshold approach.

    Args:
        embeddings: Embeddings to analyze (n, d). Will be L2-normalized.
        main_similarity_threshold: Similarity threshold for main ontology (default 0.55).
        outlier_similarity_threshold: Similarity threshold for outlier ontology (default 0.45).
        diversity_gap_threshold: Diversity gap for main ontology edges (default 0.02).
        outlier_diversity_gap: Diversity gap for outlier ontology edges (default 0.015).
        k_neighbors: Number of neighbors for k-NN graph.
        verbose: Print progress information.

    Returns:
        UnifiedOntologyResult with combined ontology and metadata.

    Example:
        >>> from dyf import build_unified_ontology
        >>>
        >>> result = build_unified_ontology(embeddings, verbose=True)
        >>> print(result.summary())
        >>>
        >>> # Access the unified ontology
        >>> ontology = result.ontology
        >>> ancestors = ontology.get_ancestors(node_idx)
        >>>
        >>> # Check which tier a node came from
        >>> if node_idx in result.main_nodes:
        ...     print("From main ontology")
        >>> elif node_idx in result.outlier_nodes:
        ...     print("From outlier ontology")
    """
    # Normalize + build k-NN graph (reused for bridge edges later)
    embeddings, neighbors, similarities = _prepare_knn_graph(embeddings, k_neighbors)
    n_points, dim = embeddings.shape

    if verbose:
        logger.info("Building unified ontology: %s points, dim=%d", f"{n_points:,}", dim)

    # Step 1: Build main ontology
    if verbose:
        logger.info("1. Building main ontology (sim>=%.2f)...", main_similarity_threshold)

    main_ont = build_dag_taxonomy(
        embeddings,
        k_neighbors=k_neighbors,
        similarity_threshold=main_similarity_threshold,
        diversity_gap_threshold=diversity_gap_threshold,
        verbose=False,
    )

    main_connected = set(main_ont.children.keys()) | set(main_ont.parents.keys())
    main_outlier_indices = list(set(range(n_points)) - main_connected)

    if verbose:
        logger.debug("Main: %d nodes, %d edges", len(main_connected), sum(len(v) for v in main_ont.children.values()))
        logger.debug("Outliers: %d nodes", len(main_outlier_indices))

    # Step 2: Build outlier ontology
    if verbose:
        logger.info("2. Building outlier ontology (sim>=%.2f)...", outlier_similarity_threshold)

    if len(main_outlier_indices) > 0:
        outlier_embeddings = embeddings[main_outlier_indices]
        outlier_ont = build_dag_taxonomy(
            outlier_embeddings,
            k_neighbors=min(k_neighbors, len(main_outlier_indices) - 1),
            similarity_threshold=outlier_similarity_threshold,
            diversity_gap_threshold=outlier_diversity_gap,
            verbose=False,
        )

        outlier_connected = set(outlier_ont.children.keys()) | set(outlier_ont.parents.keys())
        double_outlier_local = set(range(len(main_outlier_indices))) - outlier_connected

        # Map outlier indices to global indices
        outlier_to_global = {i: main_outlier_indices[i] for i in range(len(main_outlier_indices))}

        if verbose:
            logger.debug("Outlier ontology: %d nodes, %d edges", len(outlier_connected), sum(len(v) for v in outlier_ont.children.values()))
            logger.debug("Double outliers: %d", len(double_outlier_local))
    else:
        outlier_ont = None
        outlier_connected = set()
        double_outlier_local = set()
        outlier_to_global = {}

    # Step 3: Build unified structure
    if verbose:
        logger.info("3. Merging into unified ontology...")

    children, parents = _merge_ontology_edges(main_ont, outlier_ont, outlier_to_global)

    # Step 4: Add bridge edges
    if verbose:
        logger.info("4. Adding bridge edges...")

    bridge_edges = 0
    if len(outlier_connected) > 0:
        # Reuse k-NN graph built at the top of the function
        for outlier_local_idx in outlier_connected:
            global_idx = outlier_to_global[outlier_local_idx]
            for j in range(1, min(16, neighbors.shape[1])):
                neighbor = neighbors[global_idx, j]
                sim = similarities[global_idx, j]
                if neighbor in main_connected and sim >= outlier_similarity_threshold:
                    div_diff = main_ont.diversity[global_idx] - main_ont.diversity[neighbor]
                    if abs(div_diff) >= outlier_diversity_gap:
                        if div_diff > 0:  # outlier is parent
                            children[global_idx].append((neighbor, float(sim), float(div_diff)))
                            parents[neighbor].append((global_idx, float(sim), float(div_diff)))
                        else:  # main is parent
                            children[neighbor].append((global_idx, float(sim), float(-div_diff)))
                            parents[global_idx].append((neighbor, float(sim), float(-div_diff)))
                        bridge_edges += 1

    if verbose:
        logger.debug("Bridge edges added: %d", bridge_edges)

    # Convert to regular dicts
    children = dict(children)
    parents = dict(parents)

    # Find roots and leaves
    all_nodes = set(children.keys()) | set(parents.keys())
    roots = [n for n in all_nodes if n not in parents]
    leaves = [n for n in all_nodes if n not in children]

    # Create unified ontology
    unified = DAGTaxonomy(
        n_nodes=n_points,
        diversity=main_ont.diversity,
        children=children,
        parents=parents,
        roots=roots,
        leaves=leaves,
    )

    # Compute result arrays
    main_nodes = np.array(sorted(main_connected), dtype=np.int64)
    outlier_nodes = np.array([outlier_to_global[i] for i in outlier_connected], dtype=np.int64)
    excluded_nodes = np.array([main_outlier_indices[i] for i in double_outlier_local], dtype=np.int64)

    if verbose:
        n_edges = sum(len(v) for v in children.values())
        n_multi = sum(1 for v in parents.values() if len(v) > 1)
        outlier_edges = sum(len(v) for v in outlier_ont.children.values()) if outlier_ont else 0
        logger.info("Unified ontology complete:")
        logger.info("  Coverage: %d/%d (%.1f%%)", len(all_nodes), n_points, 100 * len(all_nodes) / n_points)
        logger.info("  Edges: %d (main=%d, outlier=%d, bridge=%d)", n_edges, sum(len(v) for v in main_ont.children.values()), outlier_edges, bridge_edges)
        logger.info("  Multi-parent: %d (%.1f%%)", n_multi, 100 * n_multi / max(len(all_nodes), 1))
        logger.info("  Excluded: %d double outliers", len(excluded_nodes))

    return UnifiedOntologyResult(
        ontology=unified,
        main_nodes=main_nodes,
        outlier_nodes=outlier_nodes,
        excluded_nodes=excluded_nodes,
        bridge_edges=bridge_edges,
        main_threshold=main_similarity_threshold,
        outlier_threshold=outlier_similarity_threshold,
    )


# ============================================================
# ROG - Recursive Ontological Generation
# ============================================================


@dataclass
class ROGLayer:
    """
    A single layer in the ROG hierarchy.

    Attributes:
        depth: Recursion depth (0 = root layer)
        similarity_threshold: Threshold used for this layer
        node_indices: Global indices of nodes in this layer
        n_nodes: Number of nodes
        n_edges: Number of edges
        coverage: Fraction of input covered by this layer
    """
    depth: int
    similarity_threshold: float
    node_indices: np.ndarray
    n_nodes: int
    n_edges: int
    coverage: float


@dataclass
class ROGResult:
    """
    Result of Recursive Ontological Generation.

    ROG applies density-adaptive thresholding recursively:
    1. Build ontology at high threshold (dense regions)
    2. Recurse into outliers with lower threshold
    3. Repeat until coverage target or min threshold reached
    4. Knit layers together with bridge edges

    Attributes:
        ontology: The unified DAGTaxonomy
        layers: List of ROGLayer describing each recursion level
        excluded_nodes: Indices that couldn't be connected at any threshold
        total_coverage: Fraction of input nodes in ontology
        bridge_edges: Edges connecting different layers
    """
    ontology: DAGTaxonomy
    layers: list[ROGLayer]
    excluded_nodes: np.ndarray
    total_coverage: float
    bridge_edges: int

    def __len__(self) -> int:
        return len(self.ontology)

    def summary(self) -> str:
        """Return summary statistics."""
        lines = [
            "ROGResult (Recursive Ontological Generation):",
            f"  Total coverage: {self.total_coverage*100:.1f}%",
            f"  Layers: {len(self.layers)}",
            f"  Bridge edges: {self.bridge_edges}",
            f"  Excluded: {len(self.excluded_nodes)}",
            "",
            "  Layer breakdown:",
        ]
        for layer in self.layers:
            lines.append(
                f"    Depth {layer.depth}: {layer.n_nodes} nodes, "
                f"{layer.n_edges} edges (sim≥{layer.similarity_threshold:.2f})"
            )
        return "\n".join(lines)

    def get_layer_for_node(self, node_idx: int) -> int | None:
        """Return which layer a node belongs to, or None if excluded."""
        for layer in self.layers:
            if node_idx in set(layer.node_indices):
                return layer.depth
        return None


def _build_rog_layer(
    remaining_indices: set[int],
    neighbors: np.ndarray,
    similarities: np.ndarray,
    diversity: np.ndarray,
    threshold: float,
    diversity_gap_threshold: float,
    depth: int,
    n_points: int,
    all_children: dict[int, list[tuple[int, float, float]]],
    all_parents: dict[int, list[tuple[int, float, float]]],
    verbose: bool = False,
) -> ROGLayer | None:
    """
    Build one layer of the ROG hierarchy at the given threshold.

    Finds edges among remaining (unconnected) nodes, merges them into the
    global children/parents dicts, and returns the layer metadata.  Returns
    ``None`` when no nodes are connected at the current threshold.

    Side effects: mutates *all_children*, *all_parents*, and
    *remaining_indices* in place.
    """
    remaining_list = list(remaining_indices)
    layer_children: dict[int, list[tuple[int, float, float]]] = defaultdict(list)
    layer_parents: dict[int, list[tuple[int, float, float]]] = defaultdict(list)

    remaining_set = set(remaining_list)
    for i in remaining_list:
        for j_idx in range(1, min(16, neighbors.shape[1])):
            j = neighbors[i, j_idx]
            if j not in remaining_set:
                continue

            sim = similarities[i, j_idx]
            if sim < threshold:
                continue

            div_i = diversity[i]
            div_j = diversity[j]
            gap = div_i - div_j

            # Adjust gap threshold for deeper layers
            gap_thresh = diversity_gap_threshold * (0.9 ** depth)

            if gap > gap_thresh:
                layer_children[i].append((j, float(sim), float(gap)))
                layer_parents[j].append((i, float(sim), float(gap)))
            elif gap < -gap_thresh:
                layer_children[j].append((i, float(sim), float(-gap)))
                layer_parents[i].append((j, float(sim), float(-gap)))

    # Identify connected nodes in this layer
    connected = set(layer_children.keys()) | set(layer_parents.keys())
    n_edges = sum(len(v) for v in layer_children.values())

    if len(connected) == 0:
        return None

    # Add to global structure
    for parent, children_list in layer_children.items():
        all_children[parent].extend(children_list)
    for child, parents_list in layer_parents.items():
        all_parents[child].extend(parents_list)

    layer = ROGLayer(
        depth=depth,
        similarity_threshold=threshold,
        node_indices=np.array(sorted(connected), dtype=np.int64),
        n_nodes=len(connected),
        n_edges=n_edges,
        coverage=len(connected) / n_points,
    )

    if verbose:
        total_connected = len(set(all_children.keys()) | set(all_parents.keys()))
        logger.debug("Connected: %d, edges: %d", len(connected), n_edges)
        logger.debug("Total coverage: %d/%d (%.1f%%)", total_connected, n_points, 100 * total_connected / n_points)

    # Update remaining
    remaining_indices -= connected

    return layer


def _add_bridge_edges(
    all_children: dict[int, list[tuple[int, float, float]]],
    all_parents: dict[int, list[tuple[int, float, float]]],
    neighbors: np.ndarray,
    similarities: np.ndarray,
    diversity: np.ndarray,
    min_threshold: float,
    diversity_gap_threshold: float,
) -> int:
    """
    Add cross-layer bridge edges to a ROG or unified ontology.

    Scans the k-NN graph for pairs of already-connected nodes that lack
    a direct edge and adds one when the diversity gap is sufficient.

    Returns the number of bridge edges added.  Mutates *all_children* and
    *all_parents* in place.
    """
    bridge_edges = 0
    all_connected = set(all_children.keys()) | set(all_parents.keys())

    for i in all_connected:
        for j_idx in range(1, min(16, neighbors.shape[1])):
            j = neighbors[i, j_idx]
            if j not in all_connected or j == i:
                continue

            # Check if edge already exists
            existing_children = {c for c, _, _ in all_children.get(i, [])}
            existing_parents = {p for p, _, _ in all_parents.get(i, [])}
            if j in existing_children or j in existing_parents:
                continue

            sim = similarities[i, j_idx]
            if sim < min_threshold:
                continue

            div_i = diversity[i]
            div_j = diversity[j]
            gap = div_i - div_j

            if abs(gap) > diversity_gap_threshold * 0.5:  # Looser for bridges
                if gap > 0:
                    all_children[i].append((j, float(sim), float(gap)))
                    all_parents[j].append((i, float(sim), float(gap)))
                else:
                    all_children[j].append((i, float(sim), float(-gap)))
                    all_parents[i].append((j, float(sim), float(-gap)))
                bridge_edges += 1

    return bridge_edges


def build_rog_ontology(
    embeddings: np.ndarray,
    initial_threshold: float = 0.55,
    min_threshold: float = 0.35,
    threshold_decay: float = 0.9,
    target_coverage: float = 0.95,
    diversity_gap_threshold: float = 0.02,
    k_neighbors: int = 30,
    max_depth: int = 5,
    verbose: bool = False,
) -> ROGResult:
    """
    Recursive Ontological Generation (ROG).

    Recursively builds ontology layers at decreasing similarity thresholds,
    then knits them together. Achieves higher coverage than single-threshold
    approaches by adapting to local density.

    Algorithm:
        1. Build ontology at initial_threshold
        2. Identify outliers (unconnected nodes)
        3. If coverage < target and threshold > min:
           - Recurse on outliers with threshold * decay
        4. Knit all layers with bridge edges
        5. Return unified ontology with layer metadata

    Args:
        embeddings: Embeddings to analyze (n, d). Will be L2-normalized.
        initial_threshold: Starting similarity threshold (default 0.55).
        min_threshold: Minimum threshold to try (default 0.35).
        threshold_decay: Multiply threshold by this each recursion (default 0.9).
        target_coverage: Stop recursing when this coverage reached (default 0.95).
        diversity_gap_threshold: Minimum diversity gap for edges (default 0.02).
        k_neighbors: Number of neighbors for k-NN graph.
        max_depth: Maximum recursion depth (default 5).
        verbose: Print progress information.

    Returns:
        ROGResult with unified ontology and layer metadata.

    Example:
        >>> from dyf import build_rog_ontology
        >>>
        >>> result = build_rog_ontology(embeddings, verbose=True)
        >>> print(result.summary())
        >>>
        >>> # Access unified ontology
        >>> ontology = result.ontology
        >>> ancestors = ontology.get_ancestors(node_idx)
        >>>
        >>> # Check which layer a node came from
        >>> layer = result.get_layer_for_node(node_idx)
        >>> if layer is not None:
        ...     print(f"Node from layer {layer}")
    """
    # Normalize + build k-NN graph once for the full dataset
    embeddings, neighbors, similarities = _prepare_knn_graph(embeddings, k_neighbors)
    n_points, dim = embeddings.shape

    if verbose:
        logger.info("ROG: Recursive Ontological Generation")
        logger.info("  Points: %s, dim=%d", f"{n_points:,}", dim)
        logger.info("  Target coverage: %.0f%%", target_coverage * 100)
        logger.info("  Threshold range: %.2f -> %.2f", initial_threshold, min_threshold)

    # Compute diversity once for full dataset
    diversity = compute_neighbor_diversity(embeddings, k=15, neighbors=neighbors)

    # Recursive layer building
    layers: list[ROGLayer] = []
    all_children: dict[int, list[tuple[int, float, float]]] = defaultdict(list)
    all_parents: dict[int, list[tuple[int, float, float]]] = defaultdict(list)

    remaining_indices = set(range(n_points))
    threshold = initial_threshold
    depth = 0

    while (
        len(remaining_indices) / n_points > (1 - target_coverage)
        and threshold >= min_threshold
        and depth < max_depth
        and len(remaining_indices) > 10
    ):
        if verbose:
            logger.info("Layer %d: threshold=%.3f, candidates=%d", depth, threshold, len(remaining_indices))

        layer = _build_rog_layer(
            remaining_indices, neighbors, similarities, diversity,
            threshold, diversity_gap_threshold, depth, n_points,
            all_children, all_parents, verbose=verbose,
        )
        if layer is not None:
            layers.append(layer)

        # Decay threshold for next iteration
        threshold *= threshold_decay
        depth += 1

    # Add bridge edges between layers
    if verbose:
        logger.info("Adding bridge edges...")

    bridge_edges = _add_bridge_edges(
        all_children, all_parents, neighbors, similarities,
        diversity, min_threshold, diversity_gap_threshold,
    )

    if verbose:
        logger.debug("Bridge edges: %d", bridge_edges)

    # Build final ontology
    children = dict(all_children)
    parents = dict(all_parents)

    all_nodes = set(children.keys()) | set(parents.keys())
    roots = [n for n in all_nodes if n not in parents]
    leaves = [n for n in all_nodes if n not in children]

    ontology = DAGTaxonomy(
        n_nodes=n_points,
        diversity=diversity,
        children=children,
        parents=parents,
        roots=roots,
        leaves=leaves,
    )

    excluded = np.array(sorted(remaining_indices), dtype=np.int64)
    total_coverage = len(all_nodes) / n_points

    if verbose:
        n_multi = sum(1 for v in parents.values() if len(v) > 1)
        logger.info("ROG complete:")
        logger.info("  Layers: %d", len(layers))
        logger.info("  Coverage: %d/%d (%.1f%%)", len(all_nodes), n_points, total_coverage * 100)
        logger.info("  Excluded: %d", len(excluded))
        logger.info("  Multi-parent: %d (%.1f%%)", n_multi, 100 * n_multi / max(len(all_nodes), 1))

    return ROGResult(
        ontology=ontology,
        layers=layers,
        excluded_nodes=excluded,
        total_coverage=total_coverage,
        bridge_edges=bridge_edges,
    )
