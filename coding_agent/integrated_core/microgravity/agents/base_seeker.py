"""
MICROGRAVITY CORE — Base Seeker Agent (FSM-based)

Every Seeker inherits from this class. It implements a Finite State Machine with
four operational modes: OPERATIONAL → FEEDBACK_PROCESSING → ALTERNATIVE_DECISION → SCHEDULING.

The FSM governs how the agent responds to errors, unexpected outputs, and mode transitions.
Each Seeker overrides `_execute_operational()` with its domain-specific logic.
"""

from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from typing import Any, Optional

from models.schemas import (
    AgentMode,
    AgentState,
    DAGNode,
    JudgmentRecord,
    JudgmentType,
    LedgerEntry,
    LLMRequest,
    LLMResponse,
    SwarmEvent,
    EventType,
)

logger = logging.getLogger(__name__)


class BaseSeekerAgent(ABC):
    """
    FSM-based Seeker Agent — the atomic unit of execution in the swarm.
    
    Lifecycle:
    1. IDLE → receives a task
    2. OPERATIONAL → executes domain-specific logic
    3. If unexpected output → FEEDBACK_PROCESSING (self-correction loop)
    4. If correction fails 3x → ALTERNATIVE_DECISION (strategy pivot)
    5. SCHEDULING → can set up temporal triggers before returning
    
    Subclasses MUST implement:
    - `_execute_operational(node, context)` — the core domain logic
    - `agent_type` property — unique identifier string
    - `system_prompt` property — the agent's persona prompt
    """

    def __init__(
        self,
        name: str,
        llm_call_fn=None,
        event_bus=None,
        tool_registry=None,
        max_retries: int = 3,
    ):
        self.name = name
        self._llm_call = llm_call_fn
        self._event_bus = event_bus
        self._tool_registry = tool_registry
        self._max_retries = max_retries

        # FSM State
        self._mode = AgentMode.IDLE
        self._state = AgentState(agent_name=name, agent_type=self.agent_type)

    # ── Abstract Methods (subclass MUST implement) ────────────

    @property
    @abstractmethod
    def agent_type(self) -> str:
        """Unique type identifier (e.g., 'system_seeker')."""
        ...

    @property
    @abstractmethod
    def system_prompt(self) -> str:
        """The agent's persona system prompt."""
        ...

    @abstractmethod
    async def _execute_operational(self, node: DAGNode, context: dict[str, Any]) -> Any:
        """
        Core domain-specific execution logic.
        
        Args:
            node: The DAG node assigned to this agent
            context: Shared context (scratchpad, prior results, world model summary, etc.)
        
        Returns:
            The result of the execution (any serializable value)
        
        Raises:
            Exception on failure — triggers FEEDBACK_PROCESSING mode
        """
        ...

    # ── Public Interface ──────────────────────────────────────

    async def execute(self, node: DAGNode, context: dict[str, Any]) -> AgentResult:
        """
        Main entry point: execute a DAG node through the FSM lifecycle.
        
        Returns an AgentResult with the outcome, judgment used, tokens consumed, etc.
        """
        start_time = time.time()
        self._mode = AgentMode.OPERATIONAL
        self._state.current_task_id = node.id
        self._state.mode = self._mode

        result = AgentResult(agent_name=self.name, node_id=node.id)

        try:
            # Phase 1: OPERATIONAL — attempt primary execution
            output = await self._execute_operational(node, context)
            result.output = output
            result.success = True
            self._state.consecutive_failures = 0
            logger.info(f"[{self.name}] OPERATIONAL success for node {node.id}")

        except Exception as e:
            logger.warning(f"[{self.name}] OPERATIONAL failed: {e}")

            # Phase 2: FEEDBACK_PROCESSING — self-correction loop
            self._mode = AgentMode.FEEDBACK_PROCESSING
            self._state.mode = self._mode
            correction = await self._feedback_processing(node, context, str(e))

            if correction is not None:
                result.output = correction
                result.success = True
                result.used_feedback = True
                logger.info(f"[{self.name}] FEEDBACK_PROCESSING succeeded")
            else:
                # Phase 3: ALTERNATIVE_DECISION — strategy pivot
                self._mode = AgentMode.ALTERNATIVE_DECISION
                self._state.mode = self._mode
                alt_result = await self._alternative_decision(node, context, str(e))

                if alt_result is not None:
                    result.output = alt_result
                    result.success = True
                    result.used_alternative = True
                    logger.info(f"[{self.name}] ALTERNATIVE_DECISION succeeded")
                else:
                    result.success = False
                    result.error = str(e)
                    self._state.consecutive_failures += 1

                    # Circuit breaker: emit HITL event if too many failures
                    if self._state.consecutive_failures >= self._max_retries:
                        if self._event_bus:
                            await self._event_bus.emit_simple(
                                EventType.HUMAN_OVERRIDE_REQUIRED,
                                source=self.name,
                                payload={
                                    "node_id": node.id,
                                    "consecutive_failures": self._state.consecutive_failures,
                                    "last_error": str(e),
                                },
                            )

        # Record timing
        result.duration_ms = int((time.time() - start_time) * 1000)
        result.tokens_used = self._state.total_tokens_used

        # Return to IDLE
        self._mode = AgentMode.IDLE
        self._state.mode = self._mode

        return result

    # ── FSM: Feedback Processing ──────────────────────────────

    async def _feedback_processing(
        self, node: DAGNode, context: dict[str, Any], error: str
    ) -> Optional[Any]:
        """
        Self-correction loop: analyze the error, adjust the approach, retry.
        Uses the LLM to reason about what went wrong and how to fix it.
        """
        if not self._llm_call:
            return None

        logger.info(f"[{self.name}] Entering FEEDBACK_PROCESSING for error: {error}")

        for attempt in range(self._max_retries):
            try:
                correction_prompt = (
                    f"Your previous attempt to execute this task failed.\n\n"
                    f"Task: {node.task_description}\n"
                    f"Error: {error}\n"
                    f"Attempt: {attempt + 1}/{self._max_retries}\n\n"
                    f"Analyze what went wrong and provide a corrected approach. "
                    f"Focus on the specific error and adjust your strategy."
                )

                request = LLMRequest(
                    messages=[
                        {"role": "system", "content": self.system_prompt},
                        {"role": "user", "content": correction_prompt},
                    ],
                    model_tier="executive",
                    temperature=0.2,
                )

                response: LLMResponse = await self._llm_call(request)
                self._state.total_tokens_used += response.tokens_input + response.tokens_output

                # Try executing with the corrected approach
                context["correction_hint"] = response.content
                output = await self._execute_operational(node, context)
                return output

            except Exception as retry_error:
                error = str(retry_error)
                logger.debug(f"[{self.name}] Feedback retry {attempt + 1} failed: {retry_error}")

        return None

    # ── FSM: Alternative Decision ─────────────────────────────

    async def _alternative_decision(
        self, node: DAGNode, context: dict[str, Any], error: str
    ) -> Optional[Any]:
        """
        Strategy pivot: try a completely different approach.
        This might involve changing tools, reframing the task, or delegating.
        """
        if not self._llm_call:
            return None

        logger.info(f"[{self.name}] Entering ALTERNATIVE_DECISION")

        try:
            alt_prompt = (
                f"Your primary approach and all correction attempts have FAILED.\n\n"
                f"Task: {node.task_description}\n"
                f"Final Error: {error}\n\n"
                f"You MUST try a COMPLETELY DIFFERENT strategy. Consider:\n"
                f"- Using different tools or methods\n"
                f"- Breaking the task into smaller sub-steps\n"
                f"- Working around the constraint entirely\n"
                f"- Simplifying the goal to a partial solution\n\n"
                f"Provide an alternative approach and execute it."
            )

            request = LLMRequest(
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": alt_prompt},
                ],
                model_tier="sovereign",  # Escalate to highest tier for creative problem-solving
                temperature=0.5,         # Higher temperature for creative alternatives
            )

            response: LLMResponse = await self._llm_call(request)
            self._state.total_tokens_used += response.tokens_input + response.tokens_output

            # The alternative strategy itself is the result
            return {
                "alternative_strategy": response.content,
                "original_error": error,
                "mode": "alternative_decision",
            }
        except Exception as e:
            logger.error(f"[{self.name}] ALTERNATIVE_DECISION failed: {e}")
            return None

    # ── Utility Methods ───────────────────────────────────────

    async def call_llm(self, messages: list[dict], tier: str = "executive",
                       temperature: float = 0.3, max_tokens: int = 4096) -> LLMResponse:
        """Convenience method for subclasses to call the LLM."""
        request = LLMRequest(
            messages=messages,
            model_tier=tier,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        response = await self._llm_call(request)
        self._state.total_tokens_used += response.tokens_input + response.tokens_output
        self._state.total_cost_usd += response.cost_usd
        return response

    def get_state(self) -> AgentState:
        """Return the current serializable state."""
        return self._state.model_copy()

    def load_state(self, state: AgentState) -> None:
        """Restore from a serialized state (for pause/resume)."""
        self._state = state
        self._mode = state.mode


# ═══════════════════════════════════════════════════════════════
#  Agent Result
# ═══════════════════════════════════════════════════════════════

class AgentResult:
    """Result of an agent's execution cycle."""
    def __init__(self, agent_name: str, node_id: str):
        self.agent_name = agent_name
        self.node_id = node_id
        self.success: bool = False
        self.output: Any = None
        self.error: Optional[str] = None
        self.used_feedback: bool = False
        self.used_alternative: bool = False
        self.tokens_used: int = 0
        self.cost_usd: float = 0.0
        self.duration_ms: int = 0
        self.judgment: Optional[JudgmentRecord] = None

    def to_dict(self) -> dict:
        return {
            "agent_name": self.agent_name,
            "node_id": self.node_id,
            "success": self.success,
            "output": self.output,
            "error": self.error,
            "used_feedback": self.used_feedback,
            "used_alternative": self.used_alternative,
            "tokens_used": self.tokens_used,
            "duration_ms": self.duration_ms,
        }
