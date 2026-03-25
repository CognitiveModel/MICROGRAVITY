"""Cognitive Tandem Mode Manager."""

from enum import Enum
from datetime import datetime
from loguru import logger # type: ignore
from .lmdb_store import LMDBStore # type: ignore


class CognitiveMode(str, Enum):
    """The 6 cognitive modes of the Swarm Architecture §16."""
    REFLEX = "REFLEX"           # <20ms, LMDB lookups, static routing
    SCANNING = "SCANNING"         # <50ms, VLM visual diffs, simple heuristics
    DELIBERATION = "DELIBERATION" # <3000ms, Standard LLM prompt parsing
    CREATIVE = "CREATIVE"         # <10s, Generative divergence, high temperature
    MONITORING = "MONITORING"     # Background cron, anomaly detection
    RECOVERY = "RECOVERY"         # Deep system introspection after critical failure


class CognitiveController:
    """
    Manages transitions between cognitive modes to save resources
    and match reasoning depth to task difficulty.
    """

    def __init__(self, lmdb: LMDBStore):
        self.lmdb = lmdb
    
    def get_mode(self, task_id: str) -> CognitiveMode:
        """Get the current active mode for a task."""
        state = self.lmdb.get(f"sprocess:cognitive_mode:{task_id}")
        if state:
            return CognitiveMode(state["mode"]) # type: ignore
        return CognitiveMode.REFLEX

    def set_mode(self, task_id: str, new_mode: CognitiveMode, reason: str = "") -> None:
        """Force a transition to a new cognitive mode."""
        current = self.get_mode(task_id)
        if current == new_mode:
            return
            
        self.lmdb.put(f"sprocess:cognitive_mode:{task_id}", {
            "mode": new_mode.value,
            "transitioned_at": datetime.utcnow().isoformat(),
            "reason": reason
        })
        logger.info("Cognitive Shift: {} -> {} ({})", current.name, new_mode.name, reason)

    def evaluate_transition(self, task_id: str, recent_errors: int, resource_pressure: float) -> None:
        """
        Evaluate if a mode transition is necessary based on environmental clues.
        resource_pressure is 0.0 (abundant) to 1.0 (exhausted).
        """
        current = self.get_mode(task_id)
        
        # Heuristic rules from Swarm Blueprint
        
        if current == CognitiveMode.REFLEX and recent_errors > 0:
            self.set_mode(task_id, CognitiveMode.DELIBERATION, "Reflex failure detected")
            return
            
        if current == CognitiveMode.DELIBERATION and recent_errors > 2:
            self.set_mode(task_id, CognitiveMode.RECOVERY, "Deliberation failed repeatedly")
            return
            
        if current in (CognitiveMode.CREATIVE, CognitiveMode.DELIBERATION) and resource_pressure > 0.8:
            self.set_mode(task_id, CognitiveMode.REFLEX, "Resource preservation logic engaged")
            return
