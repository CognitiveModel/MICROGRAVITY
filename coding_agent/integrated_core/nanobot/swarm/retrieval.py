"""Adaptive Retrieval Surfaces."""

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional

from loguru import logger # type: ignore

from .graph_store import GraphStore # type: ignore
from .lmdb_store import LMDBStore # type: ignore
from .vector_store import VectorStore # type: ignore


@dataclass
class RetrievalSurface:
    """A parameterized query interface tuned to a specific domain or intent. Swarm Architecture §17."""
    id: str
    name: str
    type: str     # Reflex, Associative, Contextual, Temporal, Identity, Wisdom, Cross-Memory, Objective
    scope: str    # e.g., 'coding', 'personal_facts', 'recent_chat', 'global', 'session'
    maturity: str = "RAW" # RAW -> CALIBRATING -> TUNED -> SPECIALIZED -> GENERALIZED -> ARCHIVED
    parameters: dict[str, Any] = field(default_factory=dict)
    hit_rate: float = 0.0
    total_queries: int = 0
    created_at: datetime = field(default_factory=datetime.utcnow)


class SurfaceManager:
    """
    Manages custom-tuned retrieval strategies.
    Changes vector k-values, similarity thresholds, and prefix filters based on the `Surface`.
    """

    def __init__(self, lmdb: LMDBStore, vector: VectorStore, graph: GraphStore):
        self.lmdb = lmdb
        self.vector = vector
        self.graph = graph
        self._seed_default_surfaces()

    def _seed_default_surfaces(self) -> None:
        """Seed the minimum required surfaces if they don't exist."""
        if not list(self.lmdb.prefix_scan("retrieval_surface:")):
            # Original default surfaces
            self.create_surface("Default Reflex", "Reflex", "global", {"k": 1, "labels": []})
            self.create_surface("Deep Context", "Contextual", "global", {"k": 10, "labels": []})
            self.create_surface("Temporal Recent", "Temporal", "session", {"k": 5, "prefix": "vh:"})
            
            # Phase 4 advanced surfaces
            self.create_surface("Identity Core", "Identity", "global", {"k": 3, "use_graph": True, "node": "Soul"})
            self.create_surface("Wisdom Traversal", "Wisdom", "global", {"k": 5, "use_graph": True, "node": "Wisdom_Hop"})
            self.create_surface("Action Context", "Action-Outcome", "global", {"k": 5, "prefix": "sprocess:"})
            self.create_surface("Cross-Memory Fusion", "Cross-Memory", "global", {"k": 7, "fusion": True})
            self.create_surface("Objective Alignment", "Objective", "session", {"k": 3, "prefix": "objective:"})

    def create_surface(self, name: str, surface_type: str, scope: str, parameters: dict[str, Any]) -> RetrievalSurface:
        """Create a new parameterized retrieval surface."""
        s_id = f"surf_{uuid.uuid4().hex[:6]}" # type: ignore
        surface = RetrievalSurface(
            id=s_id,
            name=name,
            type=surface_type,
            scope=scope,
            parameters=parameters
        )
        self._save(surface)
        logger.debug("Created Retrieval Surface: {} ({})", name, surface_type)
        return surface

    def _save(self, surf: RetrievalSurface) -> None:
        self.lmdb.put(f"retrieval_surface:{surf.type.lower()}:{surf.scope}:{surf.id}", {
            "id": surf.id,
            "name": surf.name,
            "type": surf.type,
            "scope": surf.scope,
            "maturity": surf.maturity,
            "parameters": surf.parameters,
            "hit_rate": surf.hit_rate,
            "total_queries": surf.total_queries,
            "created_at": surf.created_at.isoformat() # type: ignore
        })

    def get_surface(self, name: str) -> Optional[RetrievalSurface]:
        """Find surface by exact name."""
        for key, val in self.lmdb.prefix_scan("retrieval_surface:"):
            if val.get("name") == name:
                return RetrievalSurface(
                    id=val["id"],
                    name=val["name"],
                    type=val["type"],
                    scope=val["scope"],
                    maturity=val["maturity"],
                    parameters=val["parameters"],
                    hit_rate=val["hit_rate"],
                    total_queries=val["total_queries"],
                    created_at=datetime.fromisoformat(val["created_at"])
                )
        return None

    def query(self, surface_name: str, query_text: str) -> list[str]:
        """Execute a retrieval operation using the configured surface parameters."""
        surf = self.get_surface(surface_name)
        if not surf:
            logger.warning("Surface '{}' not found, falling back to default vector search", surface_name)
            return self.vector.search_longterm(query_text, n_results=3)
            
        if surf.maturity == "ARCHIVED":
            logger.debug("Surface '{}' is archived, skipping", surface_name) # type: ignore
            return []

        surf.total_queries += 1 # type: ignore
        
        # Apply surface parameters to query
        k = surf.parameters.get("k", 3)
        labels = surf.parameters.get("labels", [])
        prefix = surf.parameters.get("prefix", "vm:")
        use_graph = surf.parameters.get("use_graph", False)
        fusion = surf.parameters.get("fusion", False)
        
        results: list[str] = []

        try:
            # Cross-Memory Fusion (combines paths)
            if fusion:
                v_res = self.vector.search_longterm(query_text, n_results=k // 2)
                h_res = self.vector.search_history(query_text, n_results=k // 2)
                results.extend(v_res)
                results.extend(h_res)
            # Graph-based surfaces (Identity / Wisdom)
            elif use_graph:
                node_type = surf.parameters.get("node", "Wisdom_Hop")
                # MVP placeholder: fall back to vector search on graph embeddings if graph query isn't fully robust
                # In full implementation, this calls `self.graph.search_nodes(label=node_type, query=query_text)`
                results = self.vector.search_longterm(f"{node_type}: {query_text}", n_results=k) # type: ignore
            # Prefix-based LMDB lookups + Vector search
            elif prefix == "vh:":
                results = self.vector.search_history(query_text, n_results=k, labels=labels if labels else None) # type: ignore
            else:
                results = self.vector.search_longterm(query_text, n_results=k, labels=labels if labels else None) # type: ignore
                # Quick LMDB prefix scan enrichment for Action-Outcome or Objectives
                if prefix in ["sprocess:", "objective:"]:
                    for key, val in self.lmdb.prefix_scan(prefix):
                        if query_text.lower() in str(val).lower():
                            results.append(str(val))
                            if len(results) >= k * 2:
                                break

            # Standardize and deduplicate results
            results = list(dict.fromkeys(results))[:k] # type: ignore

            # Update metrics
            if results:
                # simple running average approximation
                surf.hit_rate = ((surf.hit_rate * (surf.total_queries - 1)) + 1.0) / surf.total_queries
                self._evaluate_maturity(surf)
            else:
                surf.hit_rate = ((surf.hit_rate * (surf.total_queries - 1)) + 0.0) / surf.total_queries

        except Exception as e:
            logger.error("Error executing surface query {}: {}", surface_name, e)
            
        self._save(surf)
        return results
        
    def _evaluate_maturity(self, surf: RetrievalSurface) -> None:
        """Evaluate and manage the lifecycle state of a surface."""
        if surf.total_queries > 50 and surf.hit_rate > 0.95 and surf.maturity == "SPECIALIZED":
            surf.maturity = "GENERALIZED"
            logger.info("Surface {} promoted to GENERALIZED", surf.name)
        elif surf.total_queries > 30 and surf.hit_rate > 0.9 and surf.maturity == "TUNED":
            surf.maturity = "SPECIALIZED"
            logger.info("Surface {} promoted to SPECIALIZED", surf.name)
        elif surf.total_queries > 20 and surf.hit_rate > 0.85 and surf.maturity == "CALIBRATING":
            surf.maturity = "TUNED"
            logger.info("Surface {} promoted to TUNED", surf.name)
        elif surf.total_queries > 10 and surf.hit_rate > 0.8 and surf.maturity == "RAW":
            surf.maturity = "CALIBRATING"
            logger.info("Surface {} promoted to CALIBRATING", surf.name)
            
        # Demote if terrible performance over time
        if surf.total_queries > 100 and surf.hit_rate < 0.2 and surf.maturity != "ARCHIVED":
            surf.maturity = "ARCHIVED"
            logger.warning("Surface {} demoted to ARCHIVED due to low hit rate", surf.name)

    def combo_query(self, surface_names: list[str], query_text: str) -> list[str]:
        """Execute a cross-surface fusion query for deep context retrieval."""
        combined_results = []
        for name in surface_names:
            combined_results.extend(self.query(name, query_text)) # type: ignore
            
        # Basic deduplication while preserving ordering (highest relevance first)
        return list(dict.fromkeys(combined_results)) # type: ignore
