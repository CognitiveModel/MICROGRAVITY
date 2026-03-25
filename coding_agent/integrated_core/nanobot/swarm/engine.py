"""Swarm Engine orchestrator."""

import uuid
from pathlib import Path
from typing import Any

from loguru import logger # type: ignore

from .blob_store import BlobStore # type: ignore
from .cognitive import CognitiveController # type: ignore
from .context_classifier import ContextClassifier # type: ignore
from .discernment import DiscernmentHarvester # type: ignore
from .graph_store import GraphStore # type: ignore
from .interrupts import InterruptMonitor # type: ignore
from .lmdb_store import LMDBStore # type: ignore
from .objectives import ObjectiveTracker # type: ignore
from .outcomes import OutcomeTracker # type: ignore
from .parallel_executor import ParallelExecutor # type: ignore
from .pipeline import WisdomPipeline # type: ignore
from .resources import ResourceManager # type: ignore
from .retrieval import SurfaceManager # type: ignore
from .scenario_triggers import ScenarioRegistry # type: ignore
from .soul import SoulController # type: ignore
from .triggers import TriggerEngine # type: ignore
from .vector_store import VectorStore # type: ignore
from .experiential_bridge import ExperientialBridge # type: ignore


class SwarmEngine:
    """
    Top-level orchestrator for the Swarm Architecture.
    Initializes all subsystems and provides hooks for the AgentLoop.
    """

    def __init__(self, workspace: Path):
        self.workspace = workspace
        
        # 1. Core Data Layer
        self.lmdb = LMDBStore(workspace)
        self.graph = GraphStore(workspace)
        self.blobs = BlobStore(workspace)
        self.vector = VectorStore(workspace)
        
        # 2. Controllers & Managers
        self.soul = SoulController(self.lmdb)
        self.cognitive = CognitiveController(self.lmdb)
        self.resources = ResourceManager(self.lmdb)
        self.objectives = ObjectiveTracker(self.lmdb)
        self.outcomes = OutcomeTracker(self.lmdb, self.graph)
        self.surfaces = SurfaceManager(self.lmdb, self.vector, self.graph)
        
        # 3. Wisdom Pipeline Engines
        self.discernment = DiscernmentHarvester(self.lmdb, self.graph, self.blobs)
        self.triggers = TriggerEngine(self.lmdb, self.graph)
        self.interrupts = InterruptMonitor(self.lmdb, self.graph, self.blobs)
        self.pipeline = WisdomPipeline(self.lmdb, self.graph)
        self.experiential = ExperientialBridge(self)
        
        # 4. Phase 3: Scenario Triggers, Parallel Execution, Context Classification
        self.scenarios = ScenarioRegistry(lmdb=self.lmdb)
        self.parallel = ParallelExecutor(engine=self, max_parallel=3)
        self.classifier = ContextClassifier(engine=self, max_context_tokens=4000)
        
        # 5. Phase 5: Cognitive Evaluation
        self.evaluator: Any | None = None
        
        logger.info("SwarmEngine Initialized (with Phase 3 modules).")

    def pre_llm_inject(self, task_id: str, context: str) -> str:
        """
        Hook called right before the LLM prompt is finalized.
        Evaluates triggers, matches scenarios, injects active objectives.
        Returns context extensions to add to the system prompt.
        """
        extensions = []
        
        # 1. Check Triggers
        fired = self.triggers.evaluate_triggers(context, source="pre_llm_eval")
        if fired:
             extensions.append(f"System Note: Triggers {fired} activated. Escalate focus on this input.")
             
        # 2. Inject Active Objectives
        obj_context = self.objectives.build_objective_context()
        if obj_context != "No active formal objectives.":
            extensions.append(obj_context)
        
        # 3. Scenario matching (NL → MCP action plans)
        try:
            match = self.scenarios.match(context)
            if match and match.confidence >= match.scenario.confidence_threshold:
                plan = self.scenarios.build_action_plan(match)
                extensions.append(f"\n{plan}")
                logger.info("Swarm: scenario '{}' matched (conf={:.0%})", match.scenario.intent, match.confidence) # type: ignore
        except Exception as exc:
            logger.debug("Swarm: scenario matching failed: {}", exc)
            
        # 4. Inject Experiential / Worldly Context
        try:
             # Just use the raw task intent as a heuristic task text
             task_intent = context[:200] # type: ignore
             shared_ctx = self.experiential.get_shared_context(task_intent)
             if shared_ctx:
                  extensions.append(shared_ctx)
        except Exception as exc:
             logger.debug("Swarm: failed to inject experiential context: {}", exc)
            
        return "\n\n".join(extensions) if extensions else ""

    def post_llm_process(self, task_id: str, response: str) -> None:
        """
        Hook called immediately after an LLM text response.
        Scans for interrupts (shallow reasoning/circular logic).
        """
        # Interrupt scan
        interrupt = self.interrupts.scan_chunk(response, task_id)
        if interrupt:
            logger.warning("Swarm Engine caught interrupt in LLM response: {}", interrupt.type.name)
            # Future: return control signal to agent loop to discard/retry
            
        # Hook cognitive evaluation based on task metrics (using placeholder metrics for now)
        try:
            self.cognitive.evaluate_transition(task_id, recent_errors=0, resource_pressure=0.2)
        except Exception as e:
            logger.debug("Cognitive transition evaluation failed: {}", e)

    def trigger_evaluation(
        self,
        provider: Any,
        task_id: str,
        task_intent: str,
        plan: Any,
        result: str,
        api_keys: Any = None,
        web_proxy: str | None = None
    ) -> None:
        """Fire-and-forget background evaluation of a completed subagent plan."""
        import asyncio
        from .evaluator import CognitiveEvaluator # type: ignore
        if not self.evaluator:
            brave_api_key = getattr(api_keys, "brave_search", None) if api_keys else None
            self.evaluator = CognitiveEvaluator(
                provider=provider,
                lmdb=self.lmdb,
                graph=self.graph,
                brave_api_key=brave_api_key,
                web_proxy=web_proxy
            )
            
        asyncio.create_task(
            self.evaluator.evaluate_completed_objective(task_id, task_intent, plan, result) # type: ignore
        )

    def record_action_outcome(self, task_id: str, action_name: str, payload: dict[str, Any], results: str, success: bool = True) -> None:
        """
        Hook called after a tool executes, feeding the experiential ledger.
        """
        try:
            # Create Action node
            a_id = f"act_{uuid.uuid4().hex[:8]}" # type: ignore
            import json
            self.graph.create_node(
                label="Action",
                node_id=a_id,
                props={
                    "tool_name": action_name,
                    "payload": json.dumps(payload)[:200] # type: ignore
                }
            )
            
            # Create generic outcome node for tool run
            o_id = f"out_{uuid.uuid4().hex[:8]}" # type: ignore
            self.graph.create_node(
                label="Outcome",
                node_id=o_id,
                props={
                    "mode": "SINGULAR",
                    "summary": results[:200], # type: ignore
                    "status": "ACHIEVED" if success else "FAILED"
                }
            )
            
            # Link them
            self.graph.create_edge("RESULTED_IN", a_id, o_id, props={"success": success})
            
            if not success:
                self.interrupts.record_error(task_id)
            else:
                self.interrupts.clear_errors(task_id)
                
        except Exception as e:
            logger.warning("Failed to record action outcome in Swarm topology: {}", e)

    def shutdown(self):
        """Cleanup resources."""
        if hasattr(self.lmdb, 'close'): self.lmdb.close()
        if hasattr(self.graph, 'close'): self.graph.close()
