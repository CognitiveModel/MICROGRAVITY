import sys
import os
import json
from typing import List, Any

# Ensure import paths work
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from coding_agent.storage.advanced_retrieval import apply_mmr

class ExploratoryPatternSearcher:
    """
    Dedicated search engine for the IntrospectionAgent to conduct exploratory 
    searches across codebases. It uses Maximum Marginal Relevance (MMR) 
    to intentionally seek out DIVERSE artifacts that match a proposed 
    design pattern or architectural hypothesis.
    """
    
    def __init__(self, search_engine, embedding_calculator):
        """
        Args:
            search_engine: An instance of the underlying vector database searcher 
                           (e.g., HybridSearchEngine or KnowledgeStore)
            embedding_calculator: Function that computes the embedding format 
                                  required for MMR operations.
        """
        self.search_engine = search_engine
        self.embed = embedding_calculator

    def test_design_hypothesis(self, pattern_description: str, hypothesis_tags: str = "", limit: int = 20, return_k: int = 5, lambda_mult: float = 0.4) -> List[Any]:
        """
        Tests a design hypothesis by fetching a wide pool of matches and filtering 
        for mathematical diversity using MMR.
        
        Args:
            pattern_description: Semantic description of the architecture/pattern.
            hypothesis_tags: Keyword filters (e.g. 'language:python metadata:factory').
            limit: Fetch a large pool initially.
            return_k: How many highly-diverse examples to return representing the pattern.
            lambda_mult: < 0.5 favors extreme diversity (exploratory spread).
        """
        print(f"🕵️ Exploratory Search: Testing Design Pattern -> '{pattern_description}'")
        
        # 1. Fetch a large pool of strictly semantic/hybrid matches
        # We assume the underlying search_engine has a .search() method 
        # (compatible with CocoIndex or our internal storage)
        try:
            pool_results = self.search_engine.search(
                vector_query=pattern_description,
                keyword_query=hypothesis_tags,
                top_k=limit
            )
        except AttributeError:
            # Fallback if the engine API differs
            pool_results = self.search_engine.retrieve(query=pattern_description, top_k=limit)
            
        if not pool_results:
            print("No baseline artifacts found for hypothesis.")
            return []

        # 2. Extract texts to compute local MMR embeddings
        # Assuming result objects have a 'code' or 'content' attribute
        
        pool_embeddings = []
        valid_results = []
        for res in pool_results:
            content = getattr(res, 'code', getattr(res, 'content', str(res)))
            try:
                emb = self.embed(content)
                # handle formats if needed (e.g. CocoIndex returns lists or arrays)
                import numpy as np
                emb_array = np.array(emb)
                pool_embeddings.append(emb_array)
                valid_results.append(res)
            except Exception as e:
                pass # Skip items that fail to embed
                
        # Embed the query itself
        query_emb = np.array(self.embed(pattern_description))
        
        # 3. Apply MMR to find diverse structural examples of the pattern
        print(f"Applying Maximum Marginal Relevance (λ={lambda_mult}) to {len(valid_results)} candidates...")
        diverse_examples = apply_mmr(
            query_embedding=query_emb,
            result_embeddings=pool_embeddings,
            results=valid_results,
            lambda_mult=lambda_mult,
            top_k=return_k
        )
        
        print(f"Found {len(diverse_examples)} mathematically diverse artifacts verifying the hypothesis.")
        return diverse_examples
