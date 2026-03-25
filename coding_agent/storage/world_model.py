"""
World Model Graph (Lightweight KuzuDB Simulation)

Manages the directional graph of Domain -> Task -> Strategy -> Outcome.
Allows the swarm to search for "Structural Congruence" analogies.
"""

import json
import logging
import uuid
import os
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)

WORLD_DB = "world_graph.json"

class GraphNode:
    def __init__(self, type_: str, value: str, properties: Dict = None):
        self.id = str(uuid.uuid4())
        self.type = type_ # DOMAIN, TASK, STRATEGY, OUTCOME
        self.value = value
        self.properties = properties or {}

class WorldModel:
    """Manages Relational Wisdom mapped across execution flows."""
    
    def __init__(self, db_path: str = WORLD_DB):
        self.db_path = db_path
        self.nodes: Dict[str, dict] = {}
        self.edges: List[Dict] = []
        self._load()

    def _load(self):
        if os.path.exists(self.db_path):
            try:
                with open(self.db_path, "r") as f:
                    data = json.load(f)
                    self.nodes = data.get("nodes", {})
                    self.edges = data.get("edges", [])
            except Exception as e:
                logger.error(f"Error loading WorldModel: {e}")
                self.nodes = {}
                self.edges = []

    def _save(self):
         data = {
             "nodes": self.nodes,
             "edges": self.edges
         }
         with open(self.db_path, "w") as f:
             json.dump(data, f, indent=4)

    def record_run(self, domain: str, task: str, strategy: str, is_success: bool):
        """Records a full execution trace into the graph."""
        d_id = self._get_or_create_node("DOMAIN", domain)
        t_id = self._get_or_create_node("TASK", task)
        s_id = self._get_or_create_node("STRATEGY", strategy)
        
        outcome_val = "SUCCESS" if is_success else "FAILURE"
        o_id = self._get_or_create_node("OUTCOME", outcome_val)

        # Edges
        self._add_edge(d_id, t_id, "HAS_TASK")
        self._add_edge(t_id, s_id, "EMPLOYS")
        self._add_edge(s_id, o_id, "RESULTS_IN")
        
        self._save()
        logger.info(f"Recorded WorldModel trace: {domain} -> {outcome_val}")

    def _get_or_create_node(self, n_type: str, val: str) -> str:
        for nid, ndata in self.nodes.items():
            if ndata["type"] == n_type and ndata["value"] == val:
                return nid
        
        node = GraphNode(n_type, val)
        self.nodes[node.id] = {"type": node.type, "value": node.value, "properties": node.properties}
        return node.id

    def _add_edge(self, src: str, dst: str, rel: str):
        edge = {"src": src, "dst": dst, "rel": rel}
        if edge not in self.edges:
            self.edges.append(edge)

    def find_analogous_strategies(self, new_task_keywords: List[str]) -> List[str]:
        """Finds successful strategies for tasks with similar keywords (Structural Congruence)."""
        # Very simplified graph traversal to find successful strategies
        successful_strategies = []
        for edge in self.edges:
            if edge["rel"] == "RESULTS_IN" and self.nodes[edge["dst"]]["value"] == "SUCCESS":
                s_id = edge["src"]
                
                # Check what task employed this strategy
                for p_edge in self.edges:
                    if p_edge["rel"] == "EMPLOYS" and p_edge["dst"] == s_id:
                        t_id = p_edge["src"]
                        task_val = self.nodes[t_id]["value"].lower()
                        
                        # Compare structural similarity via simple keyword matching
                        matches = sum(1 for kw in new_task_keywords if kw.lower() in task_val)
                        if matches > 0:
                            successful_strategies.append(self.nodes[s_id]["value"])
                            
        return list(set(successful_strategies))
