import numpy as np
import sys
import os

# Ensure the package is in the path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from coding_agent.storage.advanced_retrieval import apply_mmr, cluster_functional_blocks, cosine_similarity

def test_mmr_diversification():
    print("--- Testing Maximum Marginal Relevance (MMR) ---")
    
    # Mock embeddings (3D vectors for simplicity)
    query = np.array([1.0, 0.0, 0.0])
    
    # doc1 is identical to query
    # doc2 is very similar to doc1 (redundant)
    # doc3 is less similar to query, but completely orthogonal to doc1/doc2 (diverse)
    doc1 = np.array([1.0, 0.1, 0.0])
    doc2 = np.array([0.9, 0.1, 0.0])
    doc3 = np.array([0.5, 0.0, 0.8])
    
    embeddings = [doc1, doc2, doc3]
    results = [
        {"id": "doc1", "desc": "Highly relevant, identical to query"},
        {"id": "doc2", "desc": "Highly relevant, but redundant (similar to doc1)"},
        {"id": "doc3", "desc": "Less relevant, but highly diverse"}
    ]
    
    # 1. Standard search (lambda = 1.0, no diversity penalty)
    standard_results = apply_mmr(query, embeddings, results, lambda_mult=1.0, top_k=2)
    print(f"Standard Search (top 2): {[r['id'] for r in standard_results]}")
    # Expect doc1, doc2 because they are most similar mathematically
    assert [r['id'] for r in standard_results] == ["doc1", "doc2"], "Standard search failed."
    
    # 2. MMR Search (lambda = 0.5, balances relevance and diversity)
    mmr_results = apply_mmr(query, embeddings, results, lambda_mult=0.5, top_k=2)
    print(f"MMR Search (top 2): {[r['id'] for r in mmr_results]}")
    # Expect doc1, doc3 because doc2 gets penalized for being too similar to doc1
    assert [r['id'] for r in mmr_results] == ["doc1", "doc3"], "MMR failed to diversify results."
    
    print("✅ MMR Diversification passed.")
    return True

def test_functional_clustering():
    print("\n--- Testing Functional Similarity Clustering ---")
    
    # Mocking semantic spaces
    # Topic A: Authentication (docs 0, 1, 2)
    auth_docs = [
        np.array([0.9, 0.1, 0.0]),
        np.array([0.8, 0.2, 0.0]),
        np.array([0.95, 0.0, 0.05])
    ]
    
    # Topic B: Database (docs 3, 4)
    db_docs = [
        np.array([0.0, 0.9, 0.1]),
        np.array([0.05, 0.95, 0.0])
    ]
    
    # Topic C: UI (doc 5)
    ui_doc = [
        np.array([0.0, 0.0, 1.0])
    ]
    
    embeddings = auth_docs + db_docs + ui_doc
    results = [f"Auth_Script_{i}" for i in range(3)] + [f"DB_Script_{i}" for i in range(2)] + ["UI_Script_0"]
    
    clusters = cluster_functional_blocks(embeddings, results, distance_threshold=0.3)
    
    print(f"Identified {len(clusters)} conceptual clusters:")
    for i, cluster in enumerate(clusters):
        print(f"  Cluster {i + 1} ({len(cluster)} elements): {cluster}")
        
    # We expect exactly 3 clusters: one for Auth, one for DB, one for UI.
    assert len(clusters) == 3, f"Expected 3 clusters, got {len(clusters)}"
    
    # Assert contents (they are sorted by size descending)
    assert len(clusters[0]) == 3, "Auth cluster incorrect size"
    assert "Auth_" in clusters[0][0], "Cluster 0 is not Auth"
    
    assert len(clusters[1]) == 2, "DB cluster incorrect size"
    assert "DB_" in clusters[1][0], "Cluster 1 is not DB"
    
    assert len(clusters[2]) == 1, "UI cluster incorrect size"
    assert "UI_" in clusters[2][0], "Cluster 2 is not UI"
    
    print("✅ Functional Clustering passed.")
    return True

if __name__ == "__main__":
    try:
        test_mmr_diversification()
        test_functional_clustering()
        print("\n🎉 All Advanced Retrieval Algorithm Tests Passed.")
    except AssertionError as e:
        print(f"\n❌ Test Failed: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        sys.exit(1)
