"""
Keyword Extractor Subagent (OpenCodeWiki Analogue)

Distills document essence, user intent, and case studies into dense 
keyword arrays. These keywords are used to populate CocoIndex 
metadata attributes for hybrid search optimization.
"""

from typing import List, Dict, Any
import os
from coding_agent.utils.gemini_client import init_gemini, get_gemini_response

class KeywordExtractorSubagent:
    def __init__(self):
        self.gemini_model = init_gemini()

    def extract_keywords(self, content: str, max_keywords: int = 15) -> List[str]:
        """
        Extracts high-density keywords describing the 'Case Constitution' and essence.
        """
        prompt = f"""
CONTENT:
{content}

--- KEYWORD EXTRACTION TASK ---
Extract the top {max_keywords} keywords or short phrases that describe the 'Essence' and 'Constitution' of the content above. 
Focus on:
1. Core Utility / Domain (e.g., Cryptography, Cloud Scaling)
2. Mathematical Foundations (e.g., Linear Algebra, Big-O, Poisson Distribution)
3. Technical Stack (e.g., KuzuDB, React, PyTorch)
4. Strategic Objective (e.g., Monetization, Security Hardening)

Output strictly as a comma-separated list of keywords.
"""
        response = get_gemini_response(self.gemini_model, prompt)
        # Ensure keywords are typed as List[str] to satisfy slicer expectations
        keywords_raw: List[str] = [k.strip() for k in response.split(",")]
        return keywords_raw[:max_keywords]
