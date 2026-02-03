"""Re-ranking strategies for retrieval result diversification.

All functions operate on a pre-computed candidate pool (indices + similarities)
and return a re-ordered subset.  Pure numpy — no external dependencies beyond
numpy.

Strategies:
    rerank_standard      — top-k by similarity (baseline)
    rerank_mmr           — Maximal Marginal Relevance
    rerank_bridge_boost  — blend similarity with bridge score
    rerank_bridge_mmr    — MMR with bridge-weighted diversity
"""

import numpy as np


def rerank_standard(sims, candidate_indices, top_k):
    """Take the top-k most similar candidates.

    Args:
        sims: 1-D array of similarity scores for each candidate.
        candidate_indices: 1-D array of point indices corresponding to sims.
        top_k: Number of results to return.

    Returns:
        np.ndarray of selected point indices, length <= top_k.
    """
    order = np.argsort(-sims)[:top_k]
    return candidate_indices[order]


def rerank_mmr(query_emb, candidate_indices, embeddings_normed, top_k,
               lam=0.5):
    """Maximal Marginal Relevance re-ranking.

    Iteratively selects candidates that maximize:
        lam * sim(query, doc) - (1-lam) * max_sim(doc, already_selected)

    Args:
        query_emb: (d,) normalized query embedding.
        candidate_indices: 1-D array of candidate point indices.
        embeddings_normed: (n, d) L2-normalized embedding matrix.
        top_k: Number of results to return.
        lam: Relevance-diversity tradeoff (1 = pure relevance).

    Returns:
        np.ndarray of selected point indices.
    """
    cand_embs = embeddings_normed[candidate_indices]
    sims_to_query = cand_embs @ query_emb

    n_cand = len(candidate_indices)
    selected = []
    selected_embs = []
    remaining = set(range(n_cand))

    for _ in range(top_k):
        best_score = -np.inf
        best_idx = -1

        for idx in remaining:
            relevance = sims_to_query[idx]

            if selected_embs:
                sel_arr = np.array(selected_embs)
                max_sim_to_selected = float(np.max(sel_arr @ cand_embs[idx]))
            else:
                max_sim_to_selected = 0.0

            score = lam * relevance - (1 - lam) * max_sim_to_selected

            if score > best_score:
                best_score = score
                best_idx = idx

        if best_idx < 0:
            break

        selected.append(best_idx)
        selected_embs.append(cand_embs[best_idx])
        remaining.discard(best_idx)

    return candidate_indices[np.array(selected)]


def rerank_bridge_boost(sims, candidate_indices, bridge_scores, top_k,
                        alpha=0.3):
    """Re-rank by blending similarity with normalized bridge score.

    score = (1-alpha) * sim_normalized + alpha * bridge_normalized

    Args:
        sims: 1-D array of similarity scores for each candidate.
        candidate_indices: 1-D array of candidate point indices.
        bridge_scores: Array of bridge scores indexed by point index.
        top_k: Number of results to return.
        alpha: Weight for bridge score (0 = pure similarity).

    Returns:
        np.ndarray of selected point indices.
    """
    sim_min, sim_max = sims.min(), sims.max()
    if sim_max > sim_min:
        sim_norm = (sims - sim_min) / (sim_max - sim_min)
    else:
        sim_norm = np.ones_like(sims)

    bs = bridge_scores[candidate_indices]
    bs_max = bs.max()
    if bs_max > 0:
        bs_norm = bs / bs_max
    else:
        bs_norm = np.zeros_like(bs)

    combined = (1 - alpha) * sim_norm + alpha * bs_norm
    order = np.argsort(-combined)[:top_k]
    return candidate_indices[order]


def rerank_bridge_mmr(query_emb, candidate_indices, embeddings_normed,
                      bridge_scores, cluster_labels, top_k, lam=0.5,
                      meta_clusters=None):
    """MMR with bridge-weighted diversity and optional meta-cluster awareness.

    Instead of max cosine similarity to selected set, penalizes candidates
    that share a cluster with already-selected results.  Bridge points get
    a diversity bonus because they span clusters.

    If meta_clusters is provided (set of cluster IDs), those clusters don't
    count as "new coverage" — the algorithm won't waste diversity slots
    chasing structural/meta clusters (dates, years, event lists).

    Args:
        query_emb: (d,) normalized query embedding.
        candidate_indices: 1-D array of candidate point indices.
        embeddings_normed: (n, d) L2-normalized embedding matrix.
        bridge_scores: Array of bridge scores indexed by point index.
        cluster_labels: Array of cluster labels indexed by point index.
        top_k: Number of results to return.
        lam: Relevance-diversity tradeoff.
        meta_clusters: Optional set of cluster IDs to suppress for
                       diversity credit.

    Returns:
        np.ndarray of selected point indices.
    """
    cand_embs = embeddings_normed[candidate_indices]
    sims_to_query = cand_embs @ query_emb
    cand_clusters = cluster_labels[candidate_indices]
    cand_bridge = bridge_scores[candidate_indices]
    meta_set = meta_clusters if meta_clusters is not None else set()

    n_cand = len(candidate_indices)
    selected = []
    selected_clusters = set()
    remaining = set(range(n_cand))

    for _ in range(top_k):
        best_score = -np.inf
        best_idx = -1

        for idx in remaining:
            relevance = float(sims_to_query[idx])

            c = cand_clusters[idx]

            if c in meta_set:
                cluster_overlap = 1.0
            else:
                cluster_overlap = 1.0 if c in selected_clusters else 0.0

            bridge_bonus = float(cand_bridge[idx]) / max(cand_bridge.max(), 1)

            diversity = (1 - cluster_overlap) + 0.3 * bridge_bonus

            score = lam * relevance + (1 - lam) * diversity

            if score > best_score:
                best_score = score
                best_idx = idx

        if best_idx < 0:
            break

        selected.append(best_idx)
        selected_clusters.add(cand_clusters[best_idx])
        remaining.discard(best_idx)

    return candidate_indices[np.array(selected)]
