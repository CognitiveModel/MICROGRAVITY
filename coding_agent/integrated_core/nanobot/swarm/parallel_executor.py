"""Parallel Objective Executor — concurrent task execution with interference detection.

Manages safe concurrent execution of multiple objectives (short/long-term)
when resources permit, with scope-based interference detection to prevent
undesired interactions.
"""

from __future__ import annotations

import asyncio
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any, Callable, Coroutine

from loguru import logger # type: ignore

if TYPE_CHECKING:
    from nanobot.swarm.engine import SwarmEngine # type: ignore


# ---------------------------------------------------------------------------
# Interference model
# ---------------------------------------------------------------------------

class InterferenceScope(Enum):
    """Scopes where parallel objectives can interfere."""
    NONE = "none"
    FILESYSTEM = "filesystem"       # Overlapping file reads/writes
    DATABASE = "database"           # Same DB/collection access
    MCP_SERVER = "mcp_server"       # Same MCP server contention
    SESSION = "session"             # Same user session
    MEMORY = "memory"               # Shared LMDB/graph state
    NETWORK = "network"             # External API rate limits


class ExecutionMode(Enum):
    """How an objective should be executed."""
    SERIAL = "serial"           # Wait for previous to finish
    PARALLEL = "parallel"       # Run alongside others
    BACKGROUND = "background"   # Fire-and-forget, low priority


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

@dataclass
class ResourceReservation:
    """Resources reserved for a parallel objective."""
    token_budget: int = 4096
    compute_budget_ms: int = 30_000
    io_slots: int = 1

    def to_dict(self) -> dict[str, Any]:
        return {
            "token_budget": self.token_budget,
            "compute_budget_ms": self.compute_budget_ms,
            "io_slots": self.io_slots,
        }


@dataclass
class ObjectiveSlot:
    """A running parallel objective."""
    objective_id: str
    description: str
    task: asyncio.Task | None = None
    mode: ExecutionMode = ExecutionMode.PARALLEL
    interference_scopes: list[InterferenceScope] = field(default_factory=list)
    resource_reservation: ResourceReservation = field(default_factory=ResourceReservation)
    started_at: float = 0.0
    session_id: str | None = None

    @property
    def is_active(self) -> bool:
        return self.task is not None and not self.task.done() # type: ignore


# ---------------------------------------------------------------------------
# Parallel Executor
# ---------------------------------------------------------------------------

class ParallelExecutor:
    """Manages concurrent objective execution with resource budgets and interference guards.

    Coordinates with the SwarmEngine's ResourceManager and ObjectiveTracker
    to safely run multiple objectives in parallel.
    """

    def __init__(
        self,
        engine: SwarmEngine | None = None,
        max_parallel: int = 3,
    ):
        self._engine = engine
        self._max_parallel = max_parallel
        self._slots: dict[str, ObjectiveSlot] = {}
        self._queue: list[ObjectiveSlot] = []

    # ------------------------------------------------------------------
    # Interference detection
    # ------------------------------------------------------------------

    def _detect_interference(
        self,
        new_scopes: list[InterferenceScope],
        new_session: str | None,
    ) -> list[str]:
        """Check if new objective's scopes conflict with active ones.

        Returns list of conflict descriptions. Empty = safe to proceed.
        """
        conflicts: list[str] = []

        for slot in self._slots.values():
            if not slot.is_active:
                continue

            for scope in new_scopes:
                if scope == InterferenceScope.NONE:
                    continue

                if scope in slot.interference_scopes:
                    # Same scope type = potential interference
                    if scope == InterferenceScope.SESSION and new_session != slot.session_id:
                        # Different sessions don't interfere
                        continue

                    conflicts.append(
                        f"Scope conflict: {scope.value} with objective '{slot.description[:40]}'" # type: ignore
                    )

        return conflicts

    def can_run_parallel(
        self,
        scopes: list[InterferenceScope],
        session_id: str | None = None,
        resource_est: ResourceReservation | None = None,
    ) -> tuple[bool, list[str]]:
        """Check if a new objective can run in parallel.

        Returns (can_run, reasons). If can_run is False, reasons lists
        the blocking factors.
        """
        reasons: list[str] = []

        # 1. Check max parallelism
        active_count = sum(1 for s in self._slots.values() if s.is_active)
        if active_count >= self._max_parallel:
            reasons.append(f"Max parallel limit reached ({self._max_parallel})")

        # 2. Check interference
        conflicts = self._detect_interference(scopes, session_id)
        if conflicts:
            reasons.extend(conflicts)

        # 3. Check resource budget
        if self._engine and resource_est:
            try:
                budget = self._engine.resources.get_budget_summary()
                remaining_pct = budget.get("remaining_pct", 100)
                if remaining_pct < 20:
                    reasons.append(f"Insufficient resources ({remaining_pct:.0f}% remaining)")
            except Exception:
                pass

        return len(reasons) == 0, reasons

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    async def execute_objective(
        self,
        objective_id: str,
        description: str,
        coroutine: Coroutine[Any, Any, Any],
        mode: ExecutionMode = ExecutionMode.PARALLEL,
        scopes: list[InterferenceScope] | None = None,
        session_id: str | None = None,
        resource_est: ResourceReservation | None = None,
    ) -> ObjectiveSlot:
        """Launch an objective as a parallel task.

        If interference is detected and mode is PARALLEL, queues the
        objective instead. BACKGROUND mode always runs.
        """
        scopes = scopes or [InterferenceScope.NONE]
        resource_est = resource_est or ResourceReservation()

        slot = ObjectiveSlot(
            objective_id=objective_id,
            description=description,
            mode=mode,
            interference_scopes=scopes,
            resource_reservation=resource_est,
            session_id=session_id,
            started_at=time.time(),
        )

        if mode == ExecutionMode.SERIAL:
            # Wait for all active tasks to complete first
            await self._drain_active()
            slot.task = asyncio.create_task(
                self._supervised_run(slot, coroutine)
            )
        elif mode == ExecutionMode.BACKGROUND:
            # Always run, ignore interference
            slot.task = asyncio.create_task(
                self._supervised_run(slot, coroutine)
            )
        else:
            # Parallel: check interference
            can_run, reasons = self.can_run_parallel(scopes, session_id, resource_est)
            if can_run:
                slot.task = asyncio.create_task(
                    self._supervised_run(slot, coroutine)
                )
            else:
                logger.info(
                    "ParallelExecutor: queuing objective '{}': {}",
                    description[:40], "; ".join(reasons), # type: ignore
                )
                self._queue.append(slot)
                # Don't assign task — will be started when slot opens
                return slot

        self._slots[objective_id] = slot

        # Record in LMDB for crash recovery
        if self._engine:
            try:
                self._engine.lmdb.put(
                    f"parallel:slot:{objective_id}",
                    {
                        "objective_id": objective_id,
                        "description": description,
                        "mode": mode.value,
                        "scopes": [s.value for s in scopes],
                        "started_at": slot.started_at,
                        "session_id": session_id,
                    },
                    ttl=3600,  # 1hr max execution time
                )
            except Exception:
                pass

        logger.info(
            "ParallelExecutor: launched objective '{}' in {} mode (active: {})",
            description[:40], mode.value, len(self._slots), # type: ignore
        )
        return slot

    async def _supervised_run(
        self, slot: ObjectiveSlot, coro: Coroutine[Any, Any, Any]
    ) -> Any:
        """Run a coroutine with supervision, cleanup, and queue processing."""
        try:
            result = await coro

            # Record success
            if self._engine:
                try:
                    self._engine.outcomes.record(
                        f"parallel:{slot.objective_id}", slot.description, True
                    )
                except Exception:
                    pass

            return result

        except asyncio.CancelledError:
            logger.info("ParallelExecutor: objective '{}' cancelled", slot.description[:40]) # type: ignore
            raise

        except Exception as exc:
            logger.error(
                "ParallelExecutor: objective '{}' failed: {}",
                slot.description[:40], exc, # type: ignore
            )

            # Record failure
            if self._engine:
                try:
                    self._engine.outcomes.record(
                        f"parallel:{slot.objective_id}", str(exc), False
                    )
                except Exception:
                    pass

        finally:
            # Cleanup slot
            self._slots.pop(slot.objective_id, None)

            # Cleanup LMDB
            if self._engine:
                try:
                    self._engine.lmdb.delete(f"parallel:slot:{slot.objective_id}")
                except Exception:
                    pass

            # Try to start queued objectives
            await self._process_queue() # type: ignore

    async def _process_queue(self) -> None:
        """Try to start queued objectives when slots open."""
        still_queued: list[ObjectiveSlot] = []

        for slot in self._queue:
            can_run, _ = self.can_run_parallel(
                slot.interference_scopes, slot.session_id, slot.resource_reservation,
            )
            if can_run:
                # Re-create the task — this is a placeholder since the original
                # coroutine can't be restarted. In practice, the queue would
                # store a factory function.
                logger.info("ParallelExecutor: dequeuing '{}'", slot.description[:40]) # type: ignore
                self._slots[slot.objective_id] = slot
            else:
                still_queued.append(slot)

        self._queue = still_queued

    async def _drain_active(self) -> None:
        """Wait for all active tasks to complete."""
        active = [s.task for s in self._slots.values() if s.is_active and s.task]
        if active:
            await asyncio.gather(*active, return_exceptions=True) # type: ignore

    # ------------------------------------------------------------------
    # Status
    # ------------------------------------------------------------------

    def get_status(self) -> dict[str, Any]:
        """Get executor status summary."""
        return {
            "active": sum(1 for s in self._slots.values() if s.is_active),
            "queued": len(self._queue),
            "max_parallel": self._max_parallel,
            "slots": [
                {
                    "id": s.objective_id,
                    "description": s.description[:40], # type: ignore
                    "mode": s.mode.value,
                    "active": s.is_active,
                    "elapsed_s": time.time() - s.started_at if s.started_at else 0,
                }
                for s in self._slots.values()
            ],
        }

    async def cancel_objective(self, objective_id: str) -> bool:
        """Cancel a running or queued objective."""
        # Check running
        slot = self._slots.get(objective_id)
        if slot and slot.task and not slot.task.done(): # type: ignore
            slot.task.cancel() # type: ignore
            return True

        # Check queue
        self._queue = [s for s in self._queue if s.objective_id != objective_id]
        return False

    async def cancel_session(self, session_id: str) -> int:
        """Cancel all objectives for a session."""
        count = 0
        for slot in list(self._slots.values()):
            if slot.session_id == session_id and slot.is_active and slot.task: # type: ignore
                slot.task.cancel() # type: ignore
                count += 1
        self._queue = [s for s in self._queue if s.session_id != session_id]
        return count
