"""
MICROGRAVITY CORE — Global Tool Registry

Centralized, searchable catalog of every capability (native functions, API endpoints,
MCP server methods, scripts, Process IPs) at the swarm's disposal.

Each tool is wrapped in a ToolDescriptor with: purpose, how-to, when-to, dependencies,
cost estimates, reliability scores, and embedded vectors for semantic matching.
"""

from __future__ import annotations

import logging
from typing import Optional

from models.schemas import ToolDescriptor, ToolMasteryLevel, ToolSource

logger = logging.getLogger(__name__)


class ToolRegistry:
    """
    Global catalog of all tools known to the swarm.
    
    Supports:
    - Registration & deregistration of tools
    - Lookup by name, category, or source
    - Semantic search via embedding vectors (when LanceDB is connected)
    - Mastery level progression tracking
    - Usage statistics
    """

    def __init__(self):
        self._tools: dict[str, ToolDescriptor] = {}

    def register(self, tool: ToolDescriptor) -> None:
        """Register a tool in the catalog."""
        self._tools[tool.name] = tool
        logger.info(f"Tool registered: {tool.name} [{tool.source.value}] — {tool.category}")

    def deregister(self, name: str) -> bool:
        """Remove a tool from the catalog."""
        if name in self._tools:
            del self._tools[name]
            logger.info(f"Tool deregistered: {name}")
            return True
        return False

    def get(self, name: str) -> Optional[ToolDescriptor]:
        """Get a tool by exact name."""
        return self._tools.get(name)

    def exists(self, name: str) -> bool:
        return name in self._tools

    def list_all(self) -> list[ToolDescriptor]:
        """Return all registered tools."""
        return list(self._tools.values())

    def list_by_category(self, category: str) -> list[ToolDescriptor]:
        """Filter tools by category."""
        return [t for t in self._tools.values() if t.category == category]

    def list_by_source(self, source: ToolSource) -> list[ToolDescriptor]:
        """Filter tools by source type."""
        return [t for t in self._tools.values() if t.source == source]

    def list_by_mastery(self, level: ToolMasteryLevel) -> list[ToolDescriptor]:
        """Filter tools by mastery level."""
        return [t for t in self._tools.values() if t.mastery_level == level]

    def search_by_keywords(self, keywords: list[str]) -> list[ToolDescriptor]:
        """Simple keyword search across tool name, description, and when_to_use."""
        results = []
        for tool in self._tools.values():
            searchable = f"{tool.name} {tool.description} {tool.when_to_use}".lower()
            if any(kw.lower() in searchable for kw in keywords):
                results.append(tool)
        return results

    def record_usage(self, name: str, success: bool = True) -> None:
        """Track that a tool was used and optionally adjust reliability."""
        tool = self._tools.get(name)
        if not tool:
            return
        tool.usage_count += 1
        from datetime import datetime
        tool.last_used = datetime.utcnow()

        # Update reliability score (exponential moving average)
        outcome = 1.0 if success else 0.0
        alpha = 0.1
        tool.reliability_score = alpha * outcome + (1 - alpha) * tool.reliability_score

        # Auto-upgrade mastery level based on usage
        if tool.mastery_level == ToolMasteryLevel.DISCOVERY and tool.usage_count >= 1:
            tool.mastery_level = ToolMasteryLevel.ACQUAINTANCE
        elif tool.mastery_level == ToolMasteryLevel.ACQUAINTANCE and tool.usage_count >= 5:
            tool.mastery_level = ToolMasteryLevel.PROFICIENCY
        elif tool.mastery_level == ToolMasteryLevel.PROFICIENCY and tool.usage_count >= 20:
            if tool.reliability_score >= 0.8:
                tool.mastery_level = ToolMasteryLevel.INTERNALIZATION

    def get_categories(self) -> list[str]:
        """Return all unique tool categories."""
        return list(set(t.category for t in self._tools.values()))

    def get_summary_for_prompt(self, max_tools: int = 30) -> str:
        """
        Generate a compressed tool catalog summary for LLM prompt injection.
        Prioritizes: high-reliability, recently-used, highest-mastery tools.
        """
        sorted_tools = sorted(
            self._tools.values(),
            key=lambda t: (
                t.mastery_level.value,    # Higher mastery first
                t.reliability_score,       # More reliable first
                t.usage_count,             # More used first
            ),
            reverse=True,
        )[:max_tools]

        lines = ["[TOOL REGISTRY]"]
        for t in sorted_tools:
            lines.append(
                f"  • {t.name} ({t.source.value}/{t.category}) — {t.description[:80]} "
                f"[reliability={t.reliability_score:.0%}, mastery={t.mastery_level.value}]"
            )
        return "\n".join(lines)

    @property
    def count(self) -> int:
        return len(self._tools)

    def get_stats(self) -> dict:
        """Return registry statistics."""
        return {
            "total_tools": self.count,
            "by_source": {s.value: len(self.list_by_source(s)) for s in ToolSource},
            "by_mastery": {m.value: len(self.list_by_mastery(m)) for m in ToolMasteryLevel},
            "categories": self.get_categories(),
        }
