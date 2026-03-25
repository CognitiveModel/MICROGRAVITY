"""
Partitioned Case Study database

Stores real-world case studies, justifications, and calculations.
Partitions data by domain to ensure scalability and relevance.
"""

import os
import json
import logging
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)

class CaseStudyDB:
    def __init__(self, base_dir: str = "case_studies"):
        self.base_dir = base_dir
        if not os.path.exists(self.base_dir):
            os.makedirs(self.base_dir)

    def _get_partition_path(self, domain: str) -> str:
        return os.path.join(self.base_dir, f"{domain.lower()}_cases.json")

    def save_case(self, domain: str, case_data: Dict[str, Any]):
        path: str = str(self._get_partition_path(domain))
        data: List[Any] = []
        if os.path.exists(path):
            try:
                with open(path, "r", encoding='utf-8') as f:
                    data = json.load(f)
            except Exception as e:
                logger.error(f"Error reading partition {domain}: {e}")
        
        data.append(case_data)
        
        with open(path, "w", encoding='utf-8') as f:
            json.dump(data, f, indent=2)

    def get_cases(self, domain: str) -> List[Dict[str, Any]]:
        path = self._get_partition_path(domain)
        if os.path.exists(path):
            with open(path, "r", encoding='utf-8') as f:
                return json.load(f)
        return []

    def retrieve_cases(self, domain: str, query_keywords: List[str]) -> List[Dict[str, Any]]:
        """
        Retrieves cases within a domain that match any of the provided essence keywords.
        """
        cases = self.get_cases(domain)
        results = []
        for case in cases:
            case_keywords = case.get("keywords", [])
            # Intersect keywords
            if any(k.lower() in [ck.lower() for ck in case_keywords] for k in query_keywords):
                results.append(case)
        return results

    def list_partitions(self) -> List[str]:
        return [f.replace("_cases.json", "") for f in os.listdir(self.base_dir) if f.endswith("_cases.json")]
