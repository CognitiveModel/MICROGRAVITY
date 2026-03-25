"""
MICROGRAVITY CORE — Seeking Operator (Central Executive Brain)

The orchestrator that interprets user intent, decomposes it into a DAG,
selects agents via Mental Association, manages the execution lifecycle,
and routes signals through the Event Bus.

This is the "brain" of the swarm — it does NOT execute domain tasks itself;
it delegates to specialized Seeker agents.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime
from typing import Any, Optional

from agents.base_seeker import AgentResult
from core.context_manager import ContextManager
from core.dag_planner import DAGPlanner
from core.event_bus import EventBus
from core.tool_registry import ToolRegistry
from integrations.llm_router import LLMRouter
from memory.execution_ledger import ExecutionLedger
from memory.scratchpad import Scratchpad
from models.abstraction_sop import AbstractionSOPEngine
from models.capability_tracker import (
    CapabilityScope,
    CognitiveFaculty,
    ControlDecisionKind,
    ControlStructureAuditor,
)
from models.judgment import JudgmentEngine
from models.schemas import (
    DAGNode,
    DAGNodeStatus,
    EventType,
    ExecutionDAG,
    LedgerEntry,
    LLMRequest,
    ResourceBlueprint,
    StrategyMode,
    SwarmEvent,
)
from models.self_model import SelfModel
from models.world_model import WorldModel

logger = logging.getLogger(__name__)


class SeekingOperator:
    """
    Central Executive of the MICROGRAVITY CORE swarm.
    
    Responsibilities:
    1. Intent Interpretation — understand what the user wants
    2. Necessity Assessment — scope resources before execution
    3. DAG Planning — decompose objective into agent tasks
    4. Agent Selection — match tasks to the best available agents
    5. Execution Orchestration — run the DAG with lifecycle management
    6. State Management — coordinate scratchpad, ledger, and event bus
    7. Meta-Cognitive Handoff — trigger self-improvement after execution
    """

    def __init__(
        self,
        llm_router: LLMRouter,
        event_bus: EventBus,
        tool_registry: ToolRegistry,
        scratchpad: Scratchpad,
        ledger: ExecutionLedger,
        self_model: SelfModel,
        world_model: WorldModel,
    ):
        self._llm = llm_router
        self._event_bus = event_bus
        self._tool_registry = tool_registry
        self._scratchpad = scratchpad
        self._ledger = ledger
        self._self_model = self_model
        self._world_model = world_model

        # Sub-components
        self._judgment_engine = JudgmentEngine(llm_call_fn=self._llm.call)
        self._context_manager = ContextManager(self._self_model, self._world_model)
        self._dag_planner = DAGPlanner(
            llm_call_fn=self._llm.call,
            available_agents=self._self_model.capabilities.registered_agents,
        )
        self._auditor = ControlStructureAuditor()
        self._sop_engine = AbstractionSOPEngine()

        # Agent registry (name → agent instance)
        self._agents: dict[str, Any] = {}

        # Active execution state
        self._active_dag: Optional[ExecutionDAG] = None
        self._paused = False

        # Register event handlers
        self._setup_event_handlers()

    def register_agent(self, agent) -> None:
        """Register a Seeker agent with the Operator."""
        self._agents[agent.name] = agent
        self._self_model.register_agent(agent.name)
        logger.info(f"Agent registered: {agent.name} ({agent.agent_type})")

    # ── Main Orchestration Loop ───────────────────────────────

    async def orchestrate(self, objective: str, context: dict[str, Any] | None = None) -> dict:
        """
        Main entry point: orchestrate the execution of a user objective.
        
        Steps:
        1. Select strategy mode
        2. Plan the DAG
        3. Execute nodes (respecting dependencies)
        4. Handle failures (feedback → alternative → HITL)
        5. Report results and trigger meta-cognition
        """
        logger.info(f"═══ ORCHESTRATION START: {objective} ═══")
        start_time = datetime.utcnow()

        # 1. Select Strategy
        strategy = self._select_strategy(objective)
        logger.info(f"Strategy selected: {strategy.value}")

        # Audit: strategy selection
        self._auditor.record_simple(
            ControlDecisionKind.STRATEGY_SELECTION,
            agent_name="operator",
            to_state=strategy.value,
            reason=f"Selected for objective: {objective[:80]}",
            faculties=[CognitiveFaculty.PLANNING, CognitiveFaculty.RISK_ASSESSMENT,
                       CognitiveFaculty.SCOPING],
        )

        # 2. Plan DAG
        dag = await self._dag_planner.plan_dynamic(
            objective, context=context, strategy=strategy,
        )
        self._active_dag = dag
        logger.info(f"DAG planned: {len(dag.nodes)} nodes")

        # 3. Execute DAG
        results = await self._execute_dag(dag, context or {})

        # 4. Post-execution
        success = all(r.get("success", False) for r in results)
        total_tokens = sum(r.get("tokens_used", 0) for r in results)
        total_cost = self._llm.total_cost

        # Update Self-Model
        self._self_model.record_session(total_tokens, total_cost, success)

        # Emit completion event
        await self._event_bus.emit_simple(
            EventType.TASK_COMPLETED if success else EventType.TASK_FAILED,
            source="operator",
            payload={"objective": objective, "success": success, "node_count": len(dag.nodes)},
        )

        summary = {
            "objective": objective,
            "success": success,
            "strategy": strategy.value,
            "nodes_total": len(dag.nodes),
            "nodes_completed": sum(1 for n in dag.nodes if n.status == DAGNodeStatus.COMPLETED),
            "nodes_failed": sum(1 for n in dag.nodes if n.status == DAGNodeStatus.FAILED),
            "results": results,
            "total_tokens": total_tokens,
            "total_cost_usd": round(total_cost, 4),
            "duration_seconds": (datetime.utcnow() - start_time).total_seconds(),
            "llm_usage": self._llm.get_usage_summary(),
            "control_audit": self._auditor.get_audit_summary(),
            "cognitive_profile": self._self_model.get_cognitive_tracker().get_all_cluster_scores(),
        }

        logger.info(f"═══ ORCHESTRATION {'SUCCESS' if success else 'FAILED'}: {objective} ═══")
        self._active_dag = None
        return summary

    async def _execute_dag(self, dag: ExecutionDAG, context: dict[str, Any]) -> list[dict]:
        """Execute all nodes in a DAG, respecting dependencies and allowing parallelism."""
        results = []
        shared_context = dict(context)
        shared_context["prior_results"] = []

        while not dag.is_complete() and not self._paused:
            ready_nodes = dag.get_ready_nodes()

            if not ready_nodes:
                # Check if we're stuck (all remaining nodes have unresolvable dependencies)
                pending = [n for n in dag.nodes if n.status == DAGNodeStatus.PENDING]
                if pending:
                    logger.error(f"DAG stuck: {len(pending)} pending nodes with unresolvable deps")
                    for n in pending:
                        n.status = DAGNodeStatus.FAILED
                        n.error = "Unresolvable dependencies"
                break

            # Execute ready nodes (potentially in parallel)
            if len(ready_nodes) == 1:
                result = await self._execute_node(ready_nodes[0], shared_context)
                results.append(result)
                if result.get("success"):
                    shared_context["prior_results"].append(result)
            else:
                # Parallel execution of independent nodes
                tasks = [self._execute_node(node, dict(shared_context)) for node in ready_nodes]
                parallel_results = await asyncio.gather(*tasks)
                for result in parallel_results:
                    results.append(result)
                    if result.get("success"):
                        shared_context["prior_results"].append(result)

        return results

    async def _execute_node(self, node: DAGNode, context: dict[str, Any]) -> dict:
        """Execute a single DAG node through its assigned agent."""
        node.status = DAGNodeStatus.RUNNING
        node.started_at = datetime.utcnow()

        agent_name = node.assigned_agent
        agent = self._agents.get(agent_name) if agent_name else None

        if not agent:
            # Try to find the best agent via Mental Association (keyword matching fallback)
            agent = self._select_agent_for_task(node.task_description)

        if not agent:
            node.status = DAGNodeStatus.FAILED
            node.error = f"No agent available for task: {node.task_description}"
            logger.error(node.error)
            return {"success": False, "error": node.error, "node_id": node.id}

        logger.info(f"Executing node '{node.id}' via agent '{agent.name}'")

        # Classify entity & inject treatment SOP into context
        entity_type, sop = self._sop_engine.get_sop_for_text(
            node.task_description, context.get("objective", "")
        )
        treatment_abstraction = self._sop_engine.calibrate_abstraction(
            entity_type,
            context_signals={
                "task_phase": "implementing",
                "uncertainty": 0.5,
                "time_pressure": 0.3,
            },
        )
        treatment_prompt = self._sop_engine.to_treatment_prompt(entity_type, treatment_abstraction)
        # Inject treatment protocol into the context passed to agent
        context["treatment_sop"] = treatment_prompt
        context["entity_type"] = entity_type.value

        # Classify judgment for this decision
        judgment = await self._judgment_engine.classify(
            f"Executing task: {node.task_description}",
            strategy_mode=self._active_dag.strategy_mode if self._active_dag else StrategyMode.BALANCED,
        )
        node.judgment_used = judgment.judgment_type

        # Execute
        agent_result: AgentResult = await agent.execute(node, context)

        # Record in ledger
        ledger_entry = LedgerEntry(
            dag_id=self._active_dag.id if self._active_dag else "unknown",
            node_id=node.id,
            agent_name=agent.name,
            hypothesis=node.hypothesis,
            actual_output=agent_result.output,
            hypothesis_match=agent_result.success,
            judgment=judgment,
            tokens_used=agent_result.tokens_used,
            cost_usd=agent_result.cost_usd,
            duration_ms=agent_result.duration_ms,
            error=agent_result.error,
        )
        await self._ledger.append(ledger_entry)

        # Audit: agent execution decision
        self._auditor.record_simple(
            ControlDecisionKind.AGENT_SELECTION,
            agent_name=agent.name,
            dag_id=self._active_dag.id if self._active_dag else None,
            node_id=node.id,
            reason=f"Assigned via mental association for: {node.task_description[:60]}",
            faculties=[CognitiveFaculty.ANALYTICAL, CognitiveFaculty.PRIORITIZATION],
        )

        # Record SOP application for learning
        sop_quality = 1.0 if agent_result.success else 0.3
        self._sop_engine.record_sop_application(
            entity_type, sop_quality,
            learned_nuance=agent_result.error if not agent_result.success else "",
        )

        # Track cognitive faculties used by this execution
        task_faculties = self._infer_faculties_for_task(node.task_description)
        outcome_score = 1.0 if agent_result.success else 0.2
        scope = self._infer_scope_for_task(node.task_description)
        self._self_model.record_faculty_use(
            task_faculties, outcome_score, scope=scope, agent_name=agent.name
        )

        # Update other-agent capability awareness
        for faculty in task_faculties:
            tracker = self._self_model.get_cognitive_tracker()
            current = tracker.other_agents_capabilities.get(agent.name, {}).get(faculty.value, 0.5)
            alpha = 0.15
            updated = alpha * outcome_score + (1 - alpha) * current
            tracker.register_other_agent_capability(agent.name, faculty, updated)

        # Update tool usage
        if node.assigned_tools:
            for tool_name in node.assigned_tools:
                self._tool_registry.record_usage(tool_name, success=agent_result.success)

        # Update node status
        if agent_result.success:
            node.status = DAGNodeStatus.COMPLETED
            node.result = agent_result.output
        else:
            node.status = DAGNodeStatus.FAILED
            node.error = agent_result.error

        node.completed_at = datetime.utcnow()

        return agent_result.to_dict()

    # ── Strategy Selection ────────────────────────────────────

    def _select_strategy(self, objective: str) -> StrategyMode:
        """Select strategy mode based on task, world, and self context."""
        obj_lower = objective.lower()

        # High-risk keywords → strict
        if any(kw in obj_lower for kw in ["deploy", "production", "database", "delete", "payment"]):
            return StrategyMode.STRICT_PRINCIPLED

        # Research / exploration keywords → exploratory
        if any(kw in obj_lower for kw in ["explore", "research", "investigate", "try", "experiment"]):
            return StrategyMode.EXPLORATORY

        # Self-improvement objectives → opportunistic
        if self._self_model.intents.has_self_improvement_objective():
            return StrategyMode.OPPORTUNISTIC

        # High competence + high trust → guided
        avg_competence = sum(self._self_model.positioning.competence_scores.values()) / max(
            1, len(self._self_model.positioning.competence_scores)
        )
        if avg_competence > 0.8 and self._world_model.user.trust_level > 0.8:
            return StrategyMode.GUIDED_PRINCIPLED

        return StrategyMode.BALANCED

    def _select_agent_for_task(self, task_description: str):
        """Simple keyword-based agent selection (Mental Association fallback)."""
        task_lower = task_description.lower()

        keyword_map = {
            "system": ["shell", "command", "terminal", "file", "folder", "os", "process"],
            "coding": ["code", "write", "implement", "function", "class", "refactor", "program"],
            "qa": ["test", "debug", "error", "troubleshoot", "fix", "bug", "unittest"],
            "devops": ["deploy", "docker", "k8s", "kubernetes", "ci/cd", "infrastructure", "monitor"],
            "api": ["api", "http", "request", "endpoint", "webhook", "scrape", "fetch"],
        }

        best_match = None
        best_score = 0

        for agent_key, keywords in keyword_map.items():
            score = sum(1 for kw in keywords if kw in task_lower)
            if score > best_score:
                best_score = score
                best_match = agent_key

        if best_match:
            # Find an agent matching this category
            for agent in self._agents.values():
                if best_match in agent.agent_type.lower():
                    return agent

        # Return the first available agent as ultimate fallback
        return next(iter(self._agents.values()), None)

    # ── Event Handlers ────────────────────────────────────────

    def _setup_event_handlers(self) -> None:
        """Register handlers for lifecycle events."""
        self._event_bus.subscribe(EventType.PAUSE, self._handle_pause)
        self._event_bus.subscribe(EventType.RESUME, self._handle_resume)
        self._event_bus.subscribe(EventType.KILL, self._handle_kill)
        self._event_bus.subscribe(EventType.HUMAN_OVERRIDE_RECEIVED, self._handle_human_override)

    async def _handle_pause(self, event: SwarmEvent) -> None:
        self._paused = True
        self._auditor.record_simple(
            ControlDecisionKind.FSM_TRANSITION, agent_name="operator",
            from_state="running", to_state="paused", triggered_by="event",
        )
        logger.info("Operator PAUSED by event")

    async def _handle_resume(self, event: SwarmEvent) -> None:
        self._paused = False
        self._auditor.record_simple(
            ControlDecisionKind.FSM_TRANSITION, agent_name="operator",
            from_state="paused", to_state="running", triggered_by="event",
        )
        logger.info("Operator RESUMED by event")

    async def _handle_kill(self, event: SwarmEvent) -> None:
        self._paused = True
        self._auditor.record_simple(
            ControlDecisionKind.FSM_TRANSITION, agent_name="operator",
            from_state="running", to_state="killed", triggered_by="event",
        )
        if self._active_dag:
            for node in self._active_dag.nodes:
                if node.status in (DAGNodeStatus.PENDING, DAGNodeStatus.RUNNING):
                    node.status = DAGNodeStatus.SKIPPED
        logger.info("Operator KILLED — all pending nodes skipped")

    async def _handle_human_override(self, event: SwarmEvent) -> None:
        self._auditor.record_simple(
            ControlDecisionKind.ESCALATION, agent_name="operator",
            reason=f"Human override: {event.payload}", triggered_by="human",
        )
        logger.info(f"Human override received: {event.payload}")
        self._paused = False

    # ── Cognitive Faculty Inference ────────────────────────────

    @staticmethod
    def _infer_faculties_for_task(task: str) -> list[CognitiveFaculty]:
        """Heuristically infer which cognitive faculties a task engages."""
        task_lower = task.lower()
        faculties = []

        faculty_keywords = {
            CognitiveFaculty.ANALYTICAL: ["analyze", "compare", "evaluate", "assess", "examine"],
            CognitiveFaculty.PLANNING: ["plan", "design", "architect", "organize", "structure"],
            CognitiveFaculty.ABSTRACTION: ["abstract", "generalize", "pattern", "conceptual"],
            CognitiveFaculty.CREATIVITY: ["creative", "novel", "innovative", "generate", "invent"],
            CognitiveFaculty.CODING_PROFICIENCY: ["code", "implement", "write", "function", "class"],
            CognitiveFaculty.DEBUGGING: ["debug", "fix", "error", "troubleshoot", "bug"],
            CognitiveFaculty.TESTING_QA: ["test", "verify", "validate", "qa", "quality"],
            CognitiveFaculty.SYSTEM_ADMINISTRATION: ["shell", "command", "terminal", "os", "system"],
            CognitiveFaculty.API_DESIGN: ["api", "endpoint", "rest", "request", "http"],
            CognitiveFaculty.ARCHITECTURE_DESIGN: ["architecture", "design", "structure", "pattern"],
            CognitiveFaculty.HYPOTHESIS_GENERATION: ["hypothesis", "predict", "expect", "assume"],
            CognitiveFaculty.RISK_ASSESSMENT: ["risk", "danger", "safe", "security", "threat"],
            CognitiveFaculty.DECOMPOSITION: ["decompose", "break down", "split", "divide"],
            CognitiveFaculty.OBJECTIVITY: ["objective", "fact", "data", "evidence", "measure"],
            CognitiveFaculty.PRACTICALITY: ["practical", "feasible", "realistic", "actionable"],
            CognitiveFaculty.OPTIMIZATION: ["optimize", "performance", "speed", "efficient"],
            CognitiveFaculty.SCOPING: ["scope", "boundary", "limit", "constraint", "requirement"],
        }

        for faculty, keywords in faculty_keywords.items():
            if any(kw in task_lower for kw in keywords):
                faculties.append(faculty)

        # Always include at least one faculty
        if not faculties:
            faculties = [CognitiveFaculty.ANALYTICAL]

        return faculties

    @staticmethod
    def _infer_scope_for_task(task: str) -> CapabilityScope | None:
        """Heuristically infer the capability scope for a task."""
        task_lower = task.lower()
        scope_keywords = {
            CapabilityScope.CODE_GENERATION: ["code", "implement", "write", "function"],
            CapabilityScope.ARCHITECTURE_DESIGN: ["architecture", "design", "plan"],
            CapabilityScope.ERROR_DEBUGGING: ["debug", "error", "fix", "bug"],
            CapabilityScope.API_INTEGRATION: ["api", "endpoint", "integration"],
            CapabilityScope.DOCUMENTATION: ["document", "readme", "explain"],
            CapabilityScope.SECURITY_ANALYSIS: ["security", "vulnerability", "threat"],
            CapabilityScope.PERFORMANCE_OPTIMIZATION: ["optimize", "performance", "speed"],
            CapabilityScope.HYPOTHESIS_TESTING: ["test", "verify", "hypothesis"],
            CapabilityScope.RESEARCH_AND_EXPLORATION: ["research", "explore", "investigate"],
        }

        for scope, keywords in scope_keywords.items():
            if any(kw in task_lower for kw in keywords):
                return scope
        return None
