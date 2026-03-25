"""Trigger Engine for autonomous awakenings."""

import uuid
from datetime import datetime

from loguru import logger # type: ignore

from .graph_store import GraphStore # type: ignore
from .lmdb_store import LMDBStore # type: ignore
from .models.wisdom import TriggerType # type: ignore


class TriggerEngine:
    """
    Manages active triggers and evaluates streaming context to "awaken"
    or interrupt processes automatically.
    Implements Swarm Architecture §10.
    """

    def __init__(self, lmdb: LMDBStore, graph: GraphStore):
        self.lmdb = lmdb
        self.graph = graph
        self._cache_triggers()

    def _cache_triggers(self) -> None:
        """Load all active triggers into fast memory."""
        self.active_triggers = [] # type: ignore
        for key, value in self.lmdb.prefix_scan("trigger:def:"):
            if value.get("status") == "ACTIVE": # type: ignore
                 self.active_triggers.append(value) # type: ignore
        logger.debug("Loaded {} active triggers into fast memory.", len(self.active_triggers)) # type: ignore

    def register_trigger(self, trigger_type: TriggerType, condition: str, success_rate: float = 1.0) -> str:
        """Register a new active trigger condition into LMDB and KuzuDB."""
        t_id = f"trg_{uuid.uuid4().hex[:8]}" # type: ignore
        
        # 1. Fast State storage
        def_payload = {
            "id": t_id,
            "type": trigger_type.value,
            "condition": condition,
            "status": "ACTIVE",
            "success_rate": success_rate,
            "created_at": datetime.utcnow().isoformat()
        }
        self.lmdb.put(f"trigger:def:{t_id}", def_payload)
        
        # 2. Graph topology storage
        self.graph.create_node(
            label="Trigger_Definition",
            node_id=t_id,
            props={
                "type": trigger_type.value,
                "condition": condition[:150] # type: ignore
            }
        )
        
        self.active_triggers.append(def_payload) # type: ignore
        logger.info("Registered new trigger [{}] {}", trigger_type.name, condition[:50]) # type: ignore
        return t_id

    def evaluate_triggers(self, context_str: str, source: str = "Live Stream") -> list[str]:
        """
        Evaluate context against active triggers using fast heuristic matching pre-LLM.
        Returns a list of fired trigger IDs.
        """
        fired = []
        lower_context = context_str.lower()
        
        for trg in self.active_triggers: # type: ignore
            # MVP logic: simple substring matching for condition.
            # In advanced implementations, this would map to AST or Regex or Vector query.
            cond = trg["condition"].lower() # type: ignore
            if cond in lower_context:
                fired.append(trg["id"]) # type: ignore
                self.fire_trigger(trg["id"], source, context_str) # type: ignore
                
        return fired

    def fire_trigger(self, trigger_id: str, source: str, context: str) -> None:
        """Handle the state mutation when a trigger conditions is met."""
        logger.warning("🔔 TRIGGER FIRED: {} from {}", trigger_id, source) # type: ignore
        
        # Log to LMDB fast state
        self.lmdb.put(f"trigger:fired:{trigger_id}:{datetime.utcnow().timestamp()}", { # type: ignore
            "id": trigger_id,
            "source": source,
            "context_sample": context[:100], # type: ignore
            "timestamp": datetime.utcnow().isoformat()
        })
        
        # In a full system, this would drop a high-priority message onto the MessageBus
        # or immediately pause current subagents.
