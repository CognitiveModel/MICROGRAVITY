import numpy as np
from typing import List, Tuple, Any

def cosine_similarity(v1: np.ndarray, v2: np.ndarray) -> float:
    """Calculates cosine similarity between two vectors."""
    dot_product = np.dot(v1, v2)
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)
    if norm_v1 == 0 or norm_v2 == 0:
        return 0.0
    return float(dot_product / (norm_v1 * norm_v2))

def apply_mmr(
    query_embedding: np.ndarray,
    result_embeddings: List[np.ndarray],
    results: List[Any],
    lambda_mult: float = 0.5,
    top_k: int = 5
) -> List[Any]:
    """
    Maximal Marginal Relevance (MMR) implementation.
    Maximizes relevance to the query while penalizing redundancy among selected documents.
    
    score = (lambda_mult * sim(doc, query)) - ((1 - lambda_mult) * max_sim(doc, already_selected))
    
    Args:
        query_embedding: The embedding of the search query.
        result_embeddings: List of embeddings for the candidate documents.
        results: List of the actual document/result objects corresponding to the embeddings.
        lambda_mult: Value between 0 and 1. 
                     1 = Maximum relevance (standard semantic search).
                     0 = Maximum diversity (ignores query relevance).
        top_k: Number of diverse results to return.
    """
    if not results or not result_embeddings:
        return []

    # Calculate similarities with query
    doc_query_sims = [cosine_similarity(query_embedding, doc_emb) for doc_emb in result_embeddings]
    
    # Initialize unselected vs selected lists
    unselected_idx = list(range(len(results)))
    selected_idx = []
    
    # The first document is always the one most similar to the query
    best_idx = int(np.argmax(doc_query_sims))
    selected_idx.append(best_idx)
    unselected_idx.remove(best_idx)

    # Convert embeddings to numpy array for fast broadcasting if needed, 
    # but simple loop is fine given typically small candidate sets (<100).
    while len(selected_idx) < top_k and unselected_idx:
        best_score = -float('inf')
        best_candidate = -1
        
        for idx in unselected_idx:
            # Relevance to query
            relevance = doc_query_sims[idx]
            
            # Redundancy: max similarity to any document ALREADY selected
            redundancy = max([cosine_similarity(result_embeddings[idx], result_embeddings[sel_idx]) 
                              for sel_idx in selected_idx])
            
            # MMR Equation
            mmr_score = (lambda_mult * relevance) - ((1.0 - lambda_mult) * redundancy)
            
            if mmr_score > best_score:
                best_score = mmr_score
                best_candidate = idx
                
        selected_idx.append(best_candidate)
        unselected_idx.remove(best_candidate)

    return [results[i] for i in selected_idx]

def cluster_functional_blocks(
    embeddings: List[np.ndarray], 
    results: List[Any], 
    distance_threshold: float = 0.3
) -> List[List[Any]]:
    """
    Groups code blocks into similar functional clusters using a fast agglomerative approach.
    
    Args:
        embeddings: List of numpy arrays representing code chunks.
        results: The actual code chunks/objects.
        distance_threshold: Cutoff. 0.0 means identical, 1.0 means orthogonal. 
                           Elements closer than this are clustered together.
                           Cosine distance = 1 - cosine_similarity
    
    Returns:
        A list of clusters, where each cluster is a list of result objects.
    """
    if not results:
        return []
        
    n = len(embeddings)
    
    # Calculate all pairwise distances (1 - cosine similarity)
    # distance[i][j]
    distances = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i == j:
                distances[i][j] = 0.0
            else:
                distances[i][j] = 1.0 - cosine_similarity(embeddings[i], embeddings[j])
                
    # Initialize clusters
    clusters = [[i] for i in range(n)]
    
    # Simple hierarchical merging:
    while True:
        # Find the two closest clusters
        min_dist = float('inf')
        merge_pair = None
        
        for i in range(len(clusters)):
            for j in range(i + 1, len(clusters)):
                # Calculate linkage distance (Complete Linkage: max distance between elements)
                cluster_dist = max(distances[a][b] for a in clusters[i] for b in clusters[j])
                
                if cluster_dist < min_dist:
                    min_dist = cluster_dist
                    merge_pair = (i, j)
                    
        # If the closest pair is still further than threshold, stop merging
        if merge_pair is None or min_dist > distance_threshold:
            break
            
        # Merge the pair
        i, j = merge_pair
        clusters[i].extend(clusters[j])
        del clusters[j]
        
    # Map back to result objects
    final_clusters = []
    for cluster_indices in clusters:
        final_clusters.append([results[i] for i in cluster_indices])
        
    # Sort clusters by size (largest first)
    final_clusters.sort(key=len, reverse=True)
    
    return final_clusters
