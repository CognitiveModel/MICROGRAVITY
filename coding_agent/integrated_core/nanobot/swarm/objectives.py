"""Objective Memory & Identity Association."""

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

from loguru import logger # type: ignore

from .lmdb_store import LMDBStore # type: ignore


@dataclass
class Objective:
    """A tracked goal across the Swarm Architecture."""
    id: str
    tier: str  # 'short', 'medium', 'long'
    description: str
    status: str = "PENDING"  # PENDING, IN_PROGRESS, COMPLETED, FAILED, DEFERRED, BLOCKED
    execution_mode: str = "SERIAL"  # SERIAL, PARALLEL, BACKGROUND
    interference_scope: list[str] = field(default_factory=list)  # filesystem, database, mcp_server, etc.
    resource_estimate: dict = field(default_factory=lambda: {"tokens": 4096, "compute_ms": 30000})
    session_id: Optional[str] = None
    parent_id: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    identity_association: Optional[str] = None # Link to soul:capability or soul:role


class ObjectiveTracker:
    """
    Manages the 3-tier objective hierarchy (short/medium/long term)
    and links goals to the continuous Soul Identity.
    Implements Swarm Architecture §18.
    """

    def __init__(self, lmdb: LMDBStore):
        self.lmdb = lmdb

    def _get_key(self, objective: Objective) -> str:
        if objective.tier == "short":
            return f"objective:short:{objective.session_id}:{objective.id}"
        elif objective.tier == "medium":
            return f"objective:medium:{objective.session_id}:{objective.id}"
        else:
            return f"objective:long:{objective.id}"

    def register_objective(self, tier: str, description: str, session_id: str | None = None, parent_id: str | None = None) -> Objective:
        """Create a new objective."""
        if tier not in ("short", "medium", "long"):
            raise ValueError(f"Invalid objective tier: {tier}")
            
        obj_id = f"obj_{uuid.uuid4().hex[:8]}" # type: ignore
        obj = Objective(
            id=obj_id,
            tier=tier,
            description=description,
            session_id=session_id,
            parent_id=parent_id
        )
        
        # Heuristic identity association based on description content
        if "learn" in description.lower() or "understand" in description.lower():
            obj.identity_association = "soul:capability:analysis"
        elif "build" in description.lower() or "create" in description.lower():
            obj.identity_association = "soul:capability:engineering"
        
        self._save(obj)
        logger.info("Registered [{}] Objective: {}", tier.upper(), description[:50]) # type: ignore
        return obj

    def update_status(self, obj_id: str, new_status: str) -> None:
        """Transition an objective's state."""
        obj = self.get_objective(obj_id)
        if obj:
            obj.status = new_status
            obj.updated_at = datetime.utcnow()
            self._save(obj)
            logger.debug("Objective {} status updated to {}", obj_id, new_status)

    def _save(self, obj: Objective) -> None:
        """Persist to LMDB."""
        data = {
            "id": obj.id,
            "tier": obj.tier,
            "description": obj.description,
            "status": obj.status,
            "execution_mode": obj.execution_mode,
            "interference_scope": obj.interference_scope,
            "resource_estimate": obj.resource_estimate,
            "session_id": obj.session_id,
            "parent_id": obj.parent_id,
            "created_at": obj.created_at.isoformat(),
            "updated_at": obj.updated_at.isoformat(),
            "identity_association": obj.identity_association
        }
        self.lmdb.put(self._get_key(obj), data)

    def get_objective(self, obj_id: str) -> Optional[Objective]:
        """Fetch objective details across all tiers (expensive if ID not known, but fast for LMDB iteration)."""
        # Simplest way is to scan all objectives for the ID since we don't have secondary indices
        for key, val in self.lmdb.prefix_scan("objective:"):
            if val.get("id") == obj_id:
                return Objective(
                    id=val["id"],
                    tier=val["tier"],
                    description=val["description"],
                    status=val["status"],
                    session_id=val.get("session_id"),
                    parent_id=val.get("parent_id"),
                    created_at=datetime.fromisoformat(val["created_at"]),
                    updated_at=datetime.fromisoformat(val["updated_at"]),
                    identity_association=val.get("identity_association")
                )
        return None

    def build_objective_context(self, session_id: str | None = None) -> str:
        """
        Produce a summary string of active objectives for LLM injection pre-generation.
        """
        active = []
        for key, val in self.lmdb.prefix_scan("objective:"):
            if val.get("status") in ("PENDING", "IN_PROGRESS"):
                # Filter to global long-term or matching session
                if val.get("tier") == "long" or val.get("session_id") == session_id:
                    active.append(f"- [{val['tier'].upper()}] {val['description']}")
                    
        if not active:
            return "No active formal objectives."
            
        return "Active Objectives:\n" + "\n".join(active)

    def prune_stale(self) -> int:
        """Archive FAILED/DEFERRED objectives."""
        pruned = 0
        for key, val in self.lmdb.prefix_scan("objective:"):
            if val.get("status") in ("FAILED", "DEFERRED"):
                updated = datetime.fromisoformat(val.get("updated_at", datetime.utcnow().isoformat()))
                age = (datetime.utcnow() - updated).total_seconds()
                if age > 86400:  # >24h old
                    self.lmdb.delete(key)
                    pruned += 1 # type: ignore
        if pruned:
            logger.info("ObjectiveTracker: pruned {} stale objectives", pruned)
        return pruned
