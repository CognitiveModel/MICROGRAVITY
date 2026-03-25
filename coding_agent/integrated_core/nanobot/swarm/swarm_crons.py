"""Swarm Cron Bridge — registers swarm maintenance jobs as CronService entries.

Maps the tick rates from swarm.md §21 to croniter expressions and hooks
into the SwarmEngine for execution.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any

from loguru import logger # type: ignore

if TYPE_CHECKING:
    from nanobot.cron.service import CronService # type: ignore
    from nanobot.cron.types import CronSchedule # type: ignore
    from nanobot.swarm.engine import SwarmEngine # type: ignore


# Maintenance job definitions matching swarm.md §21
_SWARM_CRON_DEFS: list[dict[str, Any]] = [
    {
        "name": "swarm:lmdb_pruner",
        "description": "Scan sprocess: namespace, delete expired TTL entries",
        "schedule_expr": "* * * * *",       # Every minute (downscaled from 1s for safety)
        "handler": "run_pruner",
    },
    {
        "name": "swarm:kuzu_consolidator",
        "description": "Merge >0.98 cosine-similar Wisdom_Hop nodes",
        "schedule_expr": "0 3 * * *",       # Daily at 3 AM
        "handler": "run_consolidator",
    },
    {
        "name": "swarm:exposure_harvester",
        "description": "Parse exposure_logs for Discernment_Point candidates",
        "schedule_expr": "0 * * * *",       # Hourly
        "handler": "run_harvester",
    },
    {
        "name": "swarm:trigger_decay",
        "description": "Decay success_rate for unfired learned triggers",
        "schedule_expr": "0 */6 * * *",     # Every 6 hours
        "handler": "run_decay",
    },
    {
        "name": "swarm:conviction_auditor",
        "description": "Scan Wisdom_Hop nodes, fire STALE_CONVICTION if needed",
        "schedule_expr": "0 */12 * * *",    # Every 12 hours
        "handler": "run_auditor",
    },
    {
        "name": "swarm:budget_reconciler",
        "description": "Reconcile estimated vs actual token/CPU usage",
        "schedule_expr": "*/5 * * * *",     # Every 5 minutes
        "handler": "run_budget_reconcile",
    },
    {
        "name": "swarm:objective_pruner",
        "description": "Archive FAILED/DEFERRED objectives",
        "schedule_expr": "*/30 * * * *",    # Every 30 minutes
        "handler": "run_objective_prune",
    },
    {
        "name": "swarm:surface_maturity",
        "description": "Evaluate retrieval surface hit rates, promote/demote/retire",
        "schedule_expr": "0 * * * *",       # Hourly
        "handler": "run_surface_maturity",
    },
]


class SwarmCronBridge:
    """Bridges the SwarmEngine maintenance subsystem into the CronService.

    Registers maintenance cron jobs matching swarm.md §21 schedules and
    routes execution through the SwarmEngine's maintenance modules.
    """

    def __init__(self, engine: SwarmEngine):
        self._engine = engine
        self._registered = False

    def register_jobs(self, cron_service: CronService) -> int:
        """Register all swarm maintenance jobs into the CronService.

        Returns the number of jobs registered.
        """
        if self._registered:
            logger.debug("SwarmCronBridge: jobs already registered")
            return 0

        from nanobot.cron.types import CronSchedule # type: ignore

        count = 0
        existing_names = {j.name for j in cron_service.list_jobs(include_disabled=True)}

        for defn in _SWARM_CRON_DEFS:
            if defn["name"] in existing_names:
                logger.debug("SwarmCronBridge: '{}' already exists, skipping", defn["name"])
                continue

            try:
                cron_service.add_job( # type: ignore
                    name=defn["name"], # type: ignore
                    schedule=CronSchedule(kind="cron", expr=defn["schedule_expr"]), # type: ignore
                    message=defn["description"], # type: ignore
                    deliver=False, # type: ignore
                )
                count += 1 # type: ignore
                logger.info("SwarmCronBridge: registered '{}'", defn["name"])
            except Exception as exc:
                logger.warning("SwarmCronBridge: failed to register '{}': {}", defn["name"], exc)

        self._registered = True
        logger.info("SwarmCronBridge: registered {} maintenance jobs", count)
        return count

    async def execute_handler(self, handler_name: str) -> str:
        """Execute a swarm maintenance handler by name.

        Called by the CronService when a swarm job fires.
        """
        try:
            if handler_name == "run_pruner":
                from nanobot.swarm.maintenance.pruner import LMDBPruner # type: ignore
                pruner = LMDBPruner(self._engine.lmdb) # type: ignore
                count = pruner.prune() # type: ignore
                return f"Pruned {count} expired LMDB entries"

            elif handler_name == "run_consolidator":
                from nanobot.swarm.maintenance.consolidator import KuzuConsolidator # type: ignore
                consolidator = KuzuConsolidator(self._engine.graph) # type: ignore
                count = consolidator.consolidate() # type: ignore
                return f"Consolidated {count} similar nodes"

            elif handler_name == "run_harvester":
                from nanobot.swarm.maintenance.harvester import ExposureHarvester # type: ignore
                harvester = ExposureHarvester(self._engine.blobs, self._engine.graph) # type: ignore
                count = harvester.harvest() # type: ignore
                return f"Harvested {count} discernment points"

            elif handler_name == "run_decay":
                from nanobot.swarm.maintenance.decay import TriggerDecay # type: ignore
                decay = TriggerDecay(self._engine.lmdb) # type: ignore
                count = decay.decay() # type: ignore
                return f"Decayed {count} stale triggers"

            elif handler_name == "run_auditor":
                from nanobot.swarm.maintenance.auditor import ConvictionAuditor # type: ignore
                auditor = ConvictionAuditor(self._engine.graph, self._engine.lmdb) # type: ignore
                count = auditor.audit() # type: ignore
                return f"Audited {count} conviction nodes"

            elif handler_name == "run_budget_reconcile":
                self._engine.resources.reconcile()
                return "Budget reconciled"

            elif handler_name == "run_objective_prune":
                self._engine.objectives.prune_stale()
                return "Stale objectives pruned"

            elif handler_name == "run_surface_maturity":
                self._engine.retrieval.evaluate_surfaces()
                return "Surface maturity evaluated"

            else:
                return f"Unknown handler: {handler_name}"

        except Exception as exc:
            logger.error("SwarmCronBridge: handler '{}' failed: {}", handler_name, exc)
            return f"Handler failed: {exc}"
