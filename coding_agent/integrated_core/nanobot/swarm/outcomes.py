"""Outcome Cardinality and tracking."""

import uuid
from datetime import datetime

from loguru import logger # type: ignore

from .graph_store import GraphStore # type: ignore
from .lmdb_store import LMDBStore # type: ignore
from .models.wisdom import OutcomeMode # type: ignore


class OutcomeTracker:
    """
    Tracks the 'Outcome Cardinality' of tasks (Singular, Dual, Triad, Multi).
    Maps synergies and tradeoffs between outcomes.
    Implements Swarm Architecture §8.
    """

    def __init__(self, lmdb: LMDBStore, graph: GraphStore):
        self.lmdb = lmdb
        self.graph = graph

    def determine_mode(self, intended_outcomes: list[str]) -> OutcomeMode:
        """Heuristically assign a mode based on the number of concurrent goals."""
        count = len(intended_outcomes)
        if count == 0:
            return OutcomeMode.OPEN
        if count == 1:
            return OutcomeMode.SINGULAR
        if count == 2:
            return OutcomeMode.DUAL
        if count == 3:
            return OutcomeMode.TRIAD
        return OutcomeMode.MULTI

    def register_task_outcomes(self, task_id: str, intended_outcomes: list[str]) -> list[str]:
        """Record the intended outcomes for an active task before execution."""
        mode = self.determine_mode(intended_outcomes)
        
        self.lmdb.put(f"sprocess:outcome_cardinality:{task_id}", {
            "mode": mode.value,
            "intents": intended_outcomes,
            "registered_at": datetime.utcnow().isoformat()
        })
        logger.debug("Task {} registered as outcome mode {}", task_id, mode.name)
        
        # Serialize intended Outcome nodes to Kuzu
        outcome_ids = []
        for intent in intended_outcomes:
            o_id = f"out_{uuid.uuid4().hex[:8]}" # type: ignore
            self.graph.create_node(
                label="Outcome",
                node_id=o_id,
                props={
                    "mode": mode.value,
                    "summary": intent[:200], # type: ignore
                    "status": "INTENDED"
                }
            )
            outcome_ids.append(o_id)
            
        return outcome_ids

    def record_actual_outcome(self, action_id: str, outcome_id: str, success: bool, details: str) -> None:
        """Link an executed action to its mapped outcome, updating status."""
        
        # 1. We assume the Action node was already created by the SwarmEngine wrapper.
        # Now we link ACTION -[RESULTED_IN]-> OUTCOME
        self.graph.create_edge("RESULTED_IN", action_id, outcome_id, props={"success": success, "details": details[:100]}) # type: ignore
        
        status = "ACHIEVED" if success else "FAILED"
        
        # Read existing node to preserve other props, then replace.
        # For MVP, we pass the updated status directly to create_node which acts as partial upsert depending on backend backend.
        try:
            self.graph.create_node(
                label="Outcome",
                node_id=outcome_id,
                props={
                    "status": status,
                    "updated_at": datetime.utcnow().isoformat()
                }
            )
            logger.debug("Outcome {} marked as {}", outcome_id, status)
        except Exception as e:
            logger.error("Failed to update outcome {}: {}", outcome_id, e)

    def link_synergy(self, outcome_a: str, outcome_b: str, details: str = "") -> None:
        """Create a synergy relationship between two outcomes."""
        self.graph.create_edge("OUTCOME_SYNERGY", outcome_a, outcome_b, props={"details": details})
        logger.debug("Linked synergy betweeen outcomes {} and {}", outcome_a, outcome_b)

    def link_tradeoff(self, outcome_a: str, outcome_b: str, magnitude: float) -> None:
        """Create a tradeoff relationship between two outcomes."""
        self.graph.create_edge("OUTCOME_TRADEOFF", outcome_a, outcome_b, props={"magnitude": magnitude})
        logger.debug("Linked tradeoff (mag={}) between {} and {}", magnitude, outcome_a, outcome_b)
