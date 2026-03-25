"""Dynamic Template Engine — replaces static MD templates with swarm-powered generation.

Builds system prompts dynamically from the swarm's live state:
  - SoulController → identity, traits
  - ObjectiveTracker → active objectives
  - CognitiveController → current mode
  - ToolRegistry → available tools
  - ResourceManager → current budget
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from loguru import logger # type: ignore

if TYPE_CHECKING:
    from nanobot.swarm.engine import SwarmEngine # type: ignore


class SwarmTemplateEngine:
    """Generates system prompts dynamically from the swarm's live state.

    Replaces the static SOUL.md, TOOLS.md, AGENTS.md, USER.md, HEARTBEAT.md
    template files with context-aware prompt construction.
    """

    def __init__(
        self,
        engine: SwarmEngine | None = None,
    ):
        self._engine = engine

    # ------------------------------------------------------------------
    # Core prompt builder
    # ------------------------------------------------------------------

    def get_system_prompt(self, session_context: dict[str, Any] | None = None) -> str:
        """Build a dynamic system prompt from all swarm subsystems.

        Args:
            session_context: Optional dict with session-specific context
                (e.g. session_id, channel, user info).

        Returns:
            A fully constructed system prompt string.
        """
        sections: list[str] = []

        # 1. Soul Identity (replaces SOUL.md)
        sections.append(self._build_soul_section())

        # 2. Active Objectives (replaces HEARTBEAT.md role)
        sections.append(self._build_objectives_section(session_context))

        # 3. Cognitive Mode
        sections.append(self._build_cognitive_section())

        # 4. Available Tools (replaces TOOLS.md)
        sections.append(self._build_tools_section(session_context))

        # 5. Agent Capabilities (replaces AGENTS.md)
        sections.append(self._build_agents_section())

        # 6. User Preferences (replaces USER.md)
        sections.append(self._build_user_section(session_context))

        # 7. Resource Budget
        sections.append(self._build_resource_section())

        # Filter empty sections and join
        return "\n\n".join(s for s in sections if s.strip())

    # ------------------------------------------------------------------
    # Section builders
    # ------------------------------------------------------------------

    def _build_soul_section(self) -> str:
        """Build the identity section from SoulController."""
        if self._engine:
            try:
                identity = self._engine.soul.get_identity()
                traits = self._engine.soul.get_traits()

                lines = ["## Identity"]
                if identity:
                    lines.append(f"You are **{identity.get('name', 'nanobot')}**, "
                                 f"{identity.get('role', 'an AI assistant')}.")

                if traits:
                    active_traits = [t for t in traits if t.get("active", False)]
                    if active_traits:
                        lines.append("\n### Core Traits")
                        for t in active_traits:
                            trait_name = t.get("name", "unknown")
                            lines.append(f"- **{trait_name}**")

                return "\n".join(lines)
            except Exception as exc:
                logger.debug("TemplateEngine: soul section failed: {}", exc)

        return ""

    def _build_objectives_section(self, ctx: dict[str, Any] | None) -> str:
        """Build active objectives from ObjectiveTracker."""
        if self._engine:
            try:
                session_id = (ctx or {}).get("session_id", "default")
                obj_context = self._engine.objectives.build_objective_context(session_id)
                if obj_context:
                    return f"## Active Objectives\n{obj_context}"
            except Exception as exc:
                logger.debug("TemplateEngine: objectives section failed: {}", exc)

        return ""

    def _build_cognitive_section(self) -> str:
        """Build current cognitive mode info."""
        if self._engine:
            try:
                mode = self._engine.cognitive.get_current_mode()
                return (f"## Cognitive Mode\n"
                        f"Current mode: **{mode}**")
            except Exception as exc:
                logger.debug("TemplateEngine: cognitive section failed: {}", exc)
        return ""

    def _build_tools_section(self, ctx: dict[str, Any] | None) -> str:
        """Build available tools listing."""
        if self._engine and hasattr(self._engine, "mcp_router"):
            try:
                task_hint = (ctx or {}).get("task_hint", "")
                if task_hint:
                    routed_tools = self._engine.mcp_router.get_tools_for_context(task_hint)
                    if routed_tools:
                        tool_names = [t.name for t in routed_tools] # type: ignore
                        return (f"## Routed Tools\n"
                                f"Based on current context, prioritize: "
                                f"{', '.join(tool_names[:10])}") # type: ignore
            except Exception as exc:
                logger.debug("TemplateEngine: tools section failed: {}", exc)

        return ""

    def _build_agents_section(self) -> str:
        """Build agent capabilities listing."""
        if self._engine:
            try:
                caps = self._engine.soul.get_capabilities()
                if caps:
                    lines = ["## Agent Capabilities"]
                    active_caps = [c for c in caps if c.get("status") == "active"] # type: ignore
                    for cap in active_caps[:15]: # type: ignore
                        name = cap.get("name", "unknown") # type: ignore
                        confidence = cap.get("confidence", 0.0) # type: ignore
                        lines.append(f"- {name} (confidence: {confidence:.0%})") # type: ignore
                    return "\n".join(lines)
            except Exception as exc:
                logger.debug("TemplateEngine: agents section failed: {}", exc)

        return ""

    def _build_user_section(self, ctx: dict[str, Any] | None) -> str:
        """Build user preferences from LMDB."""
        if self._engine:
            try:
                user_prefs = self._engine.lmdb.get("soul:user:preferences")
                if user_prefs and isinstance(user_prefs, dict):
                    lines = ["## User Preferences"]
                    for k, v in user_prefs.items():
                        lines.append(f"- **{k}**: {v}")
                    return "\n".join(lines)
            except Exception as exc:
                logger.debug("TemplateEngine: user section failed: {}", exc)

        return ""

    def _build_resource_section(self) -> str:
        """Build resource budget summary."""
        if self._engine:
            try:
                budget = self._engine.resources.get_budget_summary() # type: ignore
                if budget:
                    remaining_pct = budget.get("remaining_pct", 100) # type: ignore
                    mode = "normal"
                    if remaining_pct < 20: # type: ignore
                        mode = "EMERGENCY — reflex/process-replay only"
                    elif remaining_pct < 50: # type: ignore
                        mode = "rationing — prefer Tiers 0-3"
                    return (f"## Resource Budget\n"
                            f"Remaining: {remaining_pct:.0f}% | Mode: {mode}") # type: ignore
            except Exception as exc:
                logger.debug("TemplateEngine: resource section failed: {}", exc)
        return ""

