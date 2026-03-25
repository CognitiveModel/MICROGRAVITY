import logging
from typing import Dict, Any, List

logger = logging.getLogger(__name__)

class DatabaseRouter:
    """
    Dynamically routes and mounts specific knowledge namespaces or databases
    based on the inferred domain of the autonomous task.
    """
    def __init__(self):
        self.namespace_map = {
            "Medical": ["medical_index", "pubmed_cache"],
            "Legal": ["case_law_sql", "contract_templates"],
            "Financial": ["market_data_stream", "sec_filings_vector"],
            "Coding": ["github_repos", "stackoverflow_cache", "local_codebase"]
        }
    
    def get_mounted_namespaces(self, domain: str) -> List[str]:
        """Returns the specific databases to mount for the context payload."""
        namespaces = self.namespace_map.get(domain, ["general_knowledge"])
        print(f"[DB_ROUTER] Mapping Domain '{domain}' -> Mounting Namespaces: {namespaces}")
        return namespaces

    def build_context_payload(self, domain: str, raw_knowledge: Dict[str, Any]) -> str:
        """Filters retrieved knowledge strictly through the mounted namespaces."""
        mounted = self.get_mounted_namespaces(domain)
        payload = f"--- AUTHORIZED DB MOUNTS ({', '.join(mounted)}) ---\n"
        # In a real execution environment, we would pass 'mounted' as the `where` filter during query.
        # Here we simulate the formatting of strictly namespaced context.
        return payload + "Retrieved parameters strictly filtered by domain router."
