"""Resource Governance."""

from dataclasses import dataclass
from datetime import datetime

from loguru import logger # type: ignore

from .lmdb_store import LMDBStore # type: ignore


@dataclass
class ResourceBudget:
    """A limit envelope for a given operation. Swarm Architecture §19."""
    session_id: str
    max_tokens: int
    max_compute_ms: int
    remaining_tokens: int
    consumed_tokens: int = 0
    start_time: datetime = datetime.utcnow()


class ResourceManager:
    """Manages system resource allocation and usage tracking across active sessions."""

    def __init__(self, lmdb: LMDBStore):
        self.lmdb = lmdb

    def allocate_budget(self, session_id: str, tokens: int, compute_ms: int) -> ResourceBudget:
        """Create a new budget for a session."""
        budget = ResourceBudget(
            session_id=session_id,
            max_tokens=tokens,
            max_compute_ms=compute_ms,
            remaining_tokens=tokens
        )
        self._save(budget)
        logger.debug("Allocated budget for {}: {} tokens, {}ms", session_id, tokens, compute_ms) # type: ignore
        return budget

    def _save(self, budget: ResourceBudget) -> None:
        self.lmdb.put(f"resource:budget:{budget.session_id}", {
            "session_id": budget.session_id,
            "max_tokens": budget.max_tokens,
            "max_compute_ms": budget.max_compute_ms,
            "consumed_tokens": budget.consumed_tokens,
            "remaining_tokens": budget.remaining_tokens,
            "start_time": budget.start_time.isoformat()
        })

    def get_budget(self, session_id: str) -> ResourceBudget | None:
        """Retrieve the active budget."""
        data = self.lmdb.get(f"resource:budget:{session_id}")
        if data:
            return ResourceBudget(
                session_id=data["session_id"],
                max_tokens=data["max_tokens"],
                max_compute_ms=data["max_compute_ms"],
                consumed_tokens=data["consumed_tokens"],
                remaining_tokens=data["remaining_tokens"],
                start_time=datetime.fromisoformat(data["start_time"])
            )
        return None

    def consume(self, session_id: str, tokens: int) -> bool:
        """Record token usage against the budget. Returns False if budget exhausted."""
        budget = self.get_budget(session_id)
        if not budget:
            return True # No budget enforcement if not explicitly allocated
            
        budget.consumed_tokens += tokens
        budget.remaining_tokens -= tokens
        
        self._save(budget)
        self.lmdb.put(f"resource:utilization:{session_id}:{datetime.utcnow().timestamp()}", { # type: ignore
            "tokens_consumed": tokens,
            "total_consumed": budget.consumed_tokens
        })
        
        if budget.remaining_tokens <= 0:
            logger.warning("Resource Exhaustion: Session {} has depleted token budget.", session_id) # type: ignore
            return False
        return True
