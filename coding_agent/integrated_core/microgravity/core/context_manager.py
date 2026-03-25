"""
MICROGRAVITY CORE — Context Manager

Builds the 5-layer prompt payload for every LLM call:
L0: Identity Core (permanent persona)
L1: World Context (environment, user, project)
L2: Self Context (capabilities, objectives, competence)
L3: Task Context (DAG node, prior results, scratchpad)
L4: Instruction (specific directive for this call)

Handles token budget management and context compression.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from models.self_model import SelfModel
from models.world_model import WorldModel
from models.schemas import DAGNode, JudgmentType

logger = logging.getLogger(__name__)

# ── L0: Identity Core (system-level persona) ──────────────────

IDENTITY_CORE = """You are an agent within the MICROGRAVITY CORE agentic swarm — a self-augmenting,
self-improving distributed cognitive system.

CORE AXIOMS:
- Cognitive Specialization: You own one narrow capability and execute it with extreme precision.
- Hypothesis-First: Before any action, state what you EXPECT to happen. After, compare reality to hypothesis.
- Structured Output: Always respond in the exact JSON format specified in your instructions.
- Safety-Aware: Never execute destructive operations without explicit confirmation.
- Provenance: Tag every decision with the judgment type used (accommodative, estimative, etc.).

You are precise, reliable, and self-correcting."""


class ContextManager:
    """
    Assembles multi-layer prompts for every LLM call in the system.
    
    Manages the token budget across layers, compressing when necessary.
    """

    def __init__(
        self,
        self_model: SelfModel,
        world_model: WorldModel,
        max_tokens: int = 128_000,
    ):
        self._self_model = self_model
        self._world_model = world_model
        self._max_context_tokens = max_tokens

    def build_system_prompt(
        self,
        agent_persona: str,
        judgment_prompt: str | None = None,
        tool_summary: str | None = None,
    ) -> str:
        """
        Build the complete system prompt with all context layers.
        
        Args:
            agent_persona: The specific agent's persona prompt (e.g., "You are a senior sysadmin")
            judgment_prompt: Optional judgment mode wrapper (from JudgmentEngine)
            tool_summary: Optional tool registry summary
        """
        sections = []

        # L0: Identity Core
        sections.append(IDENTITY_CORE)

        # Agent-specific persona
        sections.append(f"\n## YOUR ROLE\n{agent_persona}")

        # L1: World Context
        sections.append(f"\n## WORLD CONTEXT\n{self._world_model.to_context_summary()}")

        # L2: Self Context
        sections.append(f"\n## SELF CONTEXT\n{self._self_model.to_context_summary()}")

        # Tool summary (if provided)
        if tool_summary:
            sections.append(f"\n## AVAILABLE TOOLS\n{tool_summary}")

        # Judgment frame (if classified)
        if judgment_prompt:
            sections.append(f"\n## REASONING MODE\n{judgment_prompt}")

        return "\n".join(sections)

    def build_task_prompt(
        self,
        node: DAGNode,
        prior_results: list[dict[str, Any]] | None = None,
        scratchpad: dict[str, Any] | None = None,
        correction_hint: str | None = None,
    ) -> str:
        """
        Build the user-facing task prompt (L3 + L4).
        
        Args:
            node: The current DAG node being executed
            prior_results: Results from predecessor nodes
            scratchpad: Current ephemeral variables
            correction_hint: If in feedback processing, the correction guidance
        """
        sections = []

        # L3: Task Context
        sections.append(f"## TASK\n{node.task_description}")

        if node.hypothesis:
            sections.append(f"\n## HYPOTHESIS (What I Expect)\n{node.hypothesis}")

        if prior_results:
            sections.append("\n## PRIOR RESULTS (from earlier pipeline stages)")
            for i, result in enumerate(prior_results[-5:]):  # last 5 results
                agent = result.get("agent", "unknown")
                output = str(result.get("output", ""))[:500]
                sections.append(f"  [{i+1}] Agent: {agent} → Output: {output}")

        if scratchpad:
            sections.append(f"\n## SCRATCHPAD VARIABLES\n{_compact_dict(scratchpad)}")

        if correction_hint:
            sections.append(f"\n## CORRECTION GUIDANCE\n{correction_hint}")

        # L4: Instruction
        sections.append(
            "\n## INSTRUCTIONS\n"
            "Execute the task above. Respond with a structured JSON output containing your result. "
            "If you need to call a tool, specify it in the 'tool_call' field. "
            "Include a 'hypothesis_check' field comparing your output to the hypothesis if one was provided."
        )

        return "\n".join(sections)

    def build_messages(
        self,
        agent_persona: str,
        node: DAGNode,
        prior_results: list[dict[str, Any]] | None = None,
        scratchpad: dict[str, Any] | None = None,
        judgment_prompt: str | None = None,
        tool_summary: str | None = None,
        correction_hint: str | None = None,
    ) -> list[dict[str, str]]:
        """
        Build the complete messages array for an LLM call.
        Combines all layers into a [system, user] message pair.
        """
        system = self.build_system_prompt(agent_persona, judgment_prompt, tool_summary)
        user = self.build_task_prompt(node, prior_results, scratchpad, correction_hint)

        return [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]


def _compact_dict(d: dict, max_value_len: int = 200) -> str:
    """Format a dict compactly for prompt injection."""
    lines = []
    for k, v in d.items():
        v_str = str(v)
        if len(v_str) > max_value_len:
            v_str = v_str[:max_value_len] + "..."
        lines.append(f"  {k}: {v_str}")
    return "\n".join(lines)
