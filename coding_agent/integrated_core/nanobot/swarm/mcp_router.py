"""Multi-MCP Decision Router — domain-based MCP server selection.

Provides a provisional routing layer on top of the existing ToolRegistry so
the swarm can choose which MCP server to call based on the current task
context.  Uses LMDB for persisting learned routing preferences.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from loguru import logger # type: ignore

if TYPE_CHECKING:
    from nanobot.agent.tools.registry import ToolRegistry # type: ignore
    from nanobot.swarm.lmdb_store import LMDBStore # type: ignore


# ---------------------------------------------------------------------------
# Domain → MCP server mapping rules
# ---------------------------------------------------------------------------

@dataclass
class RoutingRule:
    """A single domain→server routing rule."""
    domain: str                  # e.g. "firebase", "database", "search"
    server_pattern: str          # MCP server name prefix or exact match
    keywords: list[str] = field(default_factory=list)
    priority: int = 0           # Higher = preferred when multiple match


# Default rules — expanded at runtime via LMDB learned routes
_DEFAULT_RULES: list[RoutingRule] = [
    RoutingRule(
        domain="firebase",
        server_pattern="firebase",
        keywords=["firebase", "firestore", "hosting", "deploy", "auth", "realtime database"],
        priority=10,
    ),
    RoutingRule(
        domain="database",
        server_pattern="database",
        keywords=["sql", "postgres", "mysql", "sqlite", "query", "migration", "schema"],
        priority=5,
    ),
    RoutingRule(
        domain="search",
        server_pattern="search",
        keywords=["search", "find", "lookup", "query web", "google"],
        priority=5,
    ),
    RoutingRule(
        domain="filesystem",
        server_pattern="filesystem",
        keywords=["file", "directory", "path", "read", "write", "delete file"],
        priority=3,
    ),
    RoutingRule(
        domain="git",
        server_pattern="git",
        keywords=["git", "commit", "branch", "merge", "pull request", "push"],
        priority=5,
    ),
]


class MCPRouter:
    """Routes task contexts to the best-fit MCP server.

    Combines static domain rules with LMDB-persisted learned routing
    preferences to determine which MCP server should receive a given
    task context.
    """

    def __init__(
        self,
        registry: ToolRegistry | None = None,
        lmdb: LMDBStore | None = None,
    ):
        self._registry = registry
        self._lmdb = lmdb
        self._rules: list[RoutingRule] = list(_DEFAULT_RULES)
        self._server_names: set[str] = set()
        self._load_learned_routes()

    # ------------------------------------------------------------------
    # Initialization helpers
    # ------------------------------------------------------------------

    def register_server(self, server_name: str) -> None:
        """Record a connected MCP server name for routing decisions."""
        self._server_names.add(server_name)

    def _load_learned_routes(self) -> None:
        """Load LMDB-persisted routing preferences."""
        if not self._lmdb:
            return
        try:
            for key, val in self._lmdb.prefix_scan("trigger:mcp_route:"):
                if isinstance(val, dict):
                    self._rules.append(RoutingRule(
                        domain=val.get("domain", "unknown"),
                        server_pattern=val.get("server_pattern", ""),
                        keywords=val.get("keywords", []),
                        priority=val.get("priority", 1),
                    ))
        except Exception as exc:
            logger.debug("MCPRouter: could not load learned routes: {}", exc)

    # ------------------------------------------------------------------
    # Core routing
    # ------------------------------------------------------------------

    def route(self, task_context: str) -> str | None:
        """Return the best MCP server name for the given task context.

        Returns ``None`` when no rule matches — caller should broadcast to
        all servers.
        """
        task_lower = task_context.lower()
        scored: list[tuple[int, RoutingRule]] = []

        for rule in self._rules:
            score = 0
            for kw in rule.keywords:
                if kw in task_lower:
                    score += rule.priority # type: ignore
            if score > 0:
                scored.append((score, rule))

        if not scored:
            return None

        scored.sort(key=lambda t: t[0], reverse=True)
        best_rule = scored[0][1]

        # Find a connected server that matches the pattern
        for sn in self._server_names:
            if best_rule.server_pattern in sn:
                return sn

        # Fallback: return the pattern itself — caller can try a fuzzy match
        return best_rule.server_pattern

    def get_tools_for_context(
        self, context: str
    ) -> list[Any]:
        """Return tools filtered to the routed MCP server."""
        if not self._registry:
            return []

        server_name = self.route(context)
        if not server_name:
            # No specific route — return all MCP tools
            return [
                t for t in self._registry.all_tools()
                if hasattr(t, "server_name")
            ]

        return [
            t for t in self._registry.all_tools()
            if hasattr(t, "server_name") and server_name in t.server_name # type: ignore
        ]

    # ------------------------------------------------------------------
    # Learning
    # ------------------------------------------------------------------

    def record_route_outcome(
        self,
        context: str,
        server_name: str,
        success: bool,
    ) -> None:
        """Record whether a routing decision was successful for learning."""
        if not self._lmdb:
            return

        route_key = f"trigger:mcp_route:{server_name}"
        existing = self._lmdb.get(route_key)

        if existing and isinstance(existing, dict):
            total = existing.get("total", 0) + 1
            wins = existing.get("wins", 0) + (1 if success else 0)
            existing["total"] = total
            existing["wins"] = wins
            existing["success_rate"] = wins / total if total > 0 else 0.0
            # Extract new keyword from context if successful
            if success:
                words = set(re.findall(r"\b\w{4,}\b", context.lower()))
                kws = set(existing.get("keywords", []))
                new_kws = words - kws
                if new_kws:
                    existing["keywords"] = list(kws | new_kws)[:20]  # type: ignore # Cap at 20
            self._lmdb.put(route_key, existing)
        else:
            self._lmdb.put(route_key, {
                "domain": server_name,
                "server_pattern": server_name,
                "keywords": list(
                    set(re.findall(r"\b\w{4,}\b", context.lower()))
                )[:10], # type: ignore
                "priority": 3,
                "total": 1,
                "wins": 1 if success else 0,
                "success_rate": 1.0 if success else 0.0,
            })

        logger.debug(
            "MCPRouter: recorded route outcome for '{}' → {} (success={})",
            context[:40], server_name, success, # type: ignore
        )

    # ------------------------------------------------------------------
    # Discovery & auto-registration
    # ------------------------------------------------------------------

    def discover_server_capabilities(self, server_name: str) -> list[str]:
        """Query tool list from a connected MCP server for scenario bootstrapping."""
        if not self._registry:
            return []

        tool_names = []
        for tool in self._registry.all_tools():
            if hasattr(tool, "server_name") and server_name in getattr(tool, "server_name", ""):
                tool_names.append(tool.name) # type: ignore
        return tool_names

    def auto_register_scenarios(self, scenario_registry: Any) -> int:
        """Auto-creates basic scenarios from discovered tool descriptions.

        Iterates over all connected servers and their tools, delegating
        to ScenarioRegistry.auto_register_from_tools().
        """
        if not self._registry:
            return 0

        total = 0
        for server_name in self._server_names:
            tools = [
                t for t in self._registry.all_tools()
                if hasattr(t, "server_name") and server_name in getattr(t, "server_name", "")
            ]
            if tools:
                count = scenario_registry.auto_register_from_tools(server_name, tools)
                total += count # type: ignore

        return total

