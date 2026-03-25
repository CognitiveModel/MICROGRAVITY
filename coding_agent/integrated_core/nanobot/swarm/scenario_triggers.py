"""NL→MCP Scenario Triggers — natural language to MCP action sequence mapping.

Recognizes user intent from natural language and composes multi-step MCP
action plans with preemptive validation.  Scenarios are stored in LMDB
and learned from execution outcomes.
"""

from __future__ import annotations

import re
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any

from loguru import logger # type: ignore

if TYPE_CHECKING:
    from nanobot.agent.tools.registry import ToolRegistry # type: ignore
    from nanobot.swarm.lmdb_store import LMDBStore # type: ignore


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

class StepStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class ActionStep:
    """A single step in a scenario action sequence."""
    server: str                     # MCP server name or "builtin"
    tool_name: str                  # Tool to call
    param_template: dict[str, str]  # Params with {{variable}} placeholders
    depends_on: list[int] = field(default_factory=list)  # Step indices this depends on
    description: str = ""
    optional: bool = False          # If True, failure doesn't abort sequence
    status: StepStatus = StepStatus.PENDING


@dataclass
class PreemptiveCheck:
    """A precondition to validate before scenario execution."""
    check_type: str          # "server_connected", "file_exists", "tool_available", "custom"
    target: str              # What to check (server name, file path, tool name)
    message: str = ""        # Human-readable failure message
    blocking: bool = True    # If True, blocks execution on failure


@dataclass
class Scenario:
    """A natural language → MCP action sequence mapping."""
    id: str
    intent: str                          # Semantic label ("deploy_firebase", "git_push", etc.)
    patterns: list[str]                  # Regex patterns to match
    keywords: list[str]                  # Quick keyword triggers
    steps: list[ActionStep]
    preemptive_checks: list[PreemptiveCheck] = field(default_factory=list)
    confidence_threshold: float = 0.6    # Min confidence to auto-suggest
    success_count: int = 0
    failure_count: int = 0
    last_used: float = 0.0

    @property
    def success_rate(self) -> float:
        total = self.success_count + self.failure_count
        return self.success_count / total if total > 0 else 0.5


@dataclass
class ScenarioMatch:
    """Result of matching user input against scenarios."""
    scenario: Scenario
    confidence: float
    matched_pattern: str
    extracted_params: dict[str, str] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Built-in scenario library
# ---------------------------------------------------------------------------

def _builtin_scenarios() -> list[Scenario]:
    """Pre-defined scenarios for common MCP operations."""
    return [
        # Firebase scenarios
        Scenario(
            id="firebase_deploy",
            intent="deploy_firebase",
            patterns=[
                r"deploy\s+(to\s+)?firebase(\s+hosting)?",
                r"push\s+(to\s+)?firebase",
                r"firebase\s+deploy",
            ],
            keywords=["deploy firebase", "firebase hosting deploy", "push firebase"],
            steps=[
                ActionStep(
                    server="firebase", tool_name="firebase_get_project",
                    param_template={},
                    description="Verify active Firebase project",
                ),
                ActionStep(
                    server="firebase", tool_name="firebase_get_environment",
                    param_template={},
                    depends_on=[0],
                    description="Check Firebase environment configuration",
                ),
            ],
            preemptive_checks=[
                PreemptiveCheck("server_connected", "firebase", "Firebase MCP not connected"),
            ],
        ),
        Scenario(
            id="firebase_init_firestore",
            intent="init_firestore",
            patterns=[
                r"(set\s*up|init|create)\s+(a\s+)?firestore(\s+database)?",
                r"firestore\s+(setup|init)",
            ],
            keywords=["setup firestore", "init firestore", "create firestore"],
            steps=[
                ActionStep(
                    server="firebase", tool_name="firebase_init",
                    param_template={"features": '{"firestore": {}}'},
                    description="Initialize Firestore in project", # type: ignore
                ),
                ActionStep(
                    server="firebase", tool_name="firebase_get_security_rules",
                    param_template={"type": "firestore"},
                    depends_on=[0],
                    description="Retrieve default security rules", # type: ignore
                ),
            ],
            preemptive_checks=[
                PreemptiveCheck("server_connected", "firebase", "Firebase MCP not connected"),
            ],
        ),
        # Git scenarios
        Scenario(
            id="git_push_changes",
            intent="git_push",
            patterns=[
                r"push\s+(my\s+)?(code|changes|commits?)",
                r"git\s+push",
                r"commit\s+and\s+push",
            ],
            keywords=["push code", "git push", "commit and push"],
            steps=[
                ActionStep(
                    server="builtin", tool_name="exec",
                    param_template={"command": "git status --porcelain"},
                    description="Check for uncommitted changes", # type: ignore
                ),
                ActionStep(
                    server="builtin", tool_name="exec",
                    param_template={"command": "git add -A"},
                    depends_on=[0],
                    description="Stage all changes", # type: ignore
                ),
                ActionStep(
                    server="builtin", tool_name="exec",
                    param_template={"command": 'git commit -m "{{commit_message}}"'},
                    depends_on=[1],
                    description="Commit with message", # type: ignore
                ),
                ActionStep(
                    server="builtin", tool_name="exec",
                    param_template={"command": "git push"},
                    depends_on=[2],
                    description="Push to remote", # type: ignore
                ),
            ],
        ),
        # File organization
        Scenario(
            id="search_codebase",
            intent="search_code",
            patterns=[
                r"(search|find|grep|look\s+for)\s+(.+)\s+in\s+(the\s+)?(code|codebase|files|project)",
                r"where\s+is\s+(.+)\s+(defined|used|imported)",
            ],
            keywords=["search code", "find in codebase", "grep project"],
            steps=[
                ActionStep(
                    server="builtin", tool_name="exec",
                    param_template={"command": 'grep -rn "{{search_term}}" --include="*.py" .'},
                    description="Search Python files for term", # type: ignore
                ),
            ],
            confidence_threshold=0.5,
        ),
        # Database scenarios
        Scenario(
            id="create_database",
            intent="create_db",
            patterns=[
                r"(create|set\s*up|init)\s+(a\s+)?(database|db|postgres|mysql|sqlite)",
            ],
            keywords=["create database", "setup db", "init postgres"],
            steps=[
                ActionStep(
                    server="database", tool_name="create_database",
                    param_template={"name": "{{db_name}}"},
                    description="Create new database", # type: ignore
                ),
            ],
            preemptive_checks=[
                PreemptiveCheck("server_connected", "database", "Database MCP not connected"),
            ],
            confidence_threshold=0.7,
        ),
    ]


# ---------------------------------------------------------------------------
# Scenario Registry
# ---------------------------------------------------------------------------

class ScenarioRegistry:
    """Manages NL→MCP scenario mappings with learning and preemptive planning.

    Matches user input against registered scenarios, validates preconditions,
    and builds action plans for the agent loop to execute.
    """

    def __init__(
        self,
        lmdb: LMDBStore | None = None,
        registry: ToolRegistry | None = None,
    ):
        self._lmdb = lmdb
        self._registry = registry
        self._scenarios: dict[str, Scenario] = {}
        self._load_builtins()
        self._load_learned()

    def _load_builtins(self) -> None:
        """Load built-in scenario library."""
        for s in _builtin_scenarios():
            self._scenarios[s.id] = s

    def _load_learned(self) -> None:
        """Load LMDB-persisted learned scenarios."""
        if not self._lmdb:
            return
        try:
            for key, val in self._lmdb.prefix_scan("scenario:def:"):
                if isinstance(val, dict):
                    steps = []
                    for step_data in val.get("steps", []):
                        steps.append(ActionStep(
                            server=step_data.get("server", "builtin"),
                            tool_name=step_data.get("tool_name", ""),
                            param_template=step_data.get("param_template", {}),
                            depends_on=step_data.get("depends_on", []),
                            description=step_data.get("description", ""),
                            optional=step_data.get("optional", False),
                        ))
                    scenario = Scenario(
                        id=val.get("id", str(key)), # type: ignore
                        intent=val.get("intent", "unknown"),
                        patterns=val.get("patterns", []),
                        keywords=val.get("keywords", []),
                        steps=steps,
                        confidence_threshold=val.get("confidence_threshold", 0.6),
                        success_count=val.get("success_count", 0),
                        failure_count=val.get("failure_count", 0),
                    )
                    self._scenarios[scenario.id] = scenario
        except Exception as exc:
            logger.debug("ScenarioRegistry: error loading learned scenarios: {}", exc) # type: ignore

    # ------------------------------------------------------------------
    # Matching
    # ------------------------------------------------------------------

    def match(self, user_input: str) -> ScenarioMatch | None:
        """Match user input against all registered scenarios.

        Returns the best matching scenario above confidence threshold,
        or None if no match.
        """
        input_lower = user_input.lower().strip()
        best_match: ScenarioMatch | None = None
        best_score = 0.0

        for scenario in self._scenarios.values():
            score, pattern, params = self._score_scenario(input_lower, scenario)
            if score > best_score and score >= scenario.confidence_threshold:
                best_score = score
                best_match = ScenarioMatch(
                    scenario=scenario,
                    confidence=score,
                    matched_pattern=pattern,
                    extracted_params=params,
                )

        return best_match

    def _score_scenario(
        self, input_lower: str, scenario: Scenario
    ) -> tuple[float, str, dict[str, str]]:
        """Score how well user input matches a scenario."""
        best_score = 0.0
        best_pattern = ""
        params: dict[str, str] = {}

        # 1. Regex pattern matching (highest signal)
        for pattern in scenario.patterns:
            try:
                m = re.search(pattern, input_lower)
                if m:
                    score = 0.9  # High confidence for regex match
                    # Boost by success_rate history
                    score *= (0.5 + 0.5 * scenario.success_rate)
                    if score > best_score:
                        best_score = score
                        best_pattern = pattern
                        params = {f"group_{i}": g for i, g in enumerate(m.groups()) if g}
            except re.error:
                continue

        # 2. Keyword matching (supplementary)
        if best_score < 0.5:
            kw_hits = sum(1 for kw in scenario.keywords if kw in input_lower)
            if kw_hits > 0:
                kw_score = min(0.7, 0.3 + 0.15 * kw_hits)
                kw_score *= (0.5 + 0.5 * scenario.success_rate)
                if kw_score > best_score:
                    best_score = kw_score
                    best_pattern = f"keyword:{scenario.keywords[0]}"

        return best_score, best_pattern, params

    # ------------------------------------------------------------------
    # Preemptive validation
    # ------------------------------------------------------------------

    def validate_preconditions(
        self, match: ScenarioMatch, connected_servers: set[str]
    ) -> list[str]:
        """Check scenario preconditions. Returns list of failure messages."""
        failures: list[str] = []

        for check in match.scenario.preemptive_checks:
            if check.check_type == "server_connected":
                if not any(check.target in s for s in connected_servers):
                    if check.blocking:
                        failures.append(check.message or f"Server '{check.target}' not connected")

            elif check.check_type == "tool_available":
                if self._registry and not self._registry.get(check.target):
                    if check.blocking:
                        failures.append(check.message or f"Tool '{check.target}' not available")

        return failures

    # ------------------------------------------------------------------
    # Plan building
    # ------------------------------------------------------------------

    def build_action_plan(self, match: ScenarioMatch) -> str:
        """Build a human-readable action plan for LLM injection."""
        scenario = match.scenario
        lines = [
            f"📋 **Detected Intent**: {scenario.intent} (confidence: {match.confidence:.0%})",
            f"**Suggested Action Plan** ({len(scenario.steps)} steps):",
        ]

        for i, step in enumerate(scenario.steps):
            deps = f" (after step {', '.join(str(d+1) for d in step.depends_on)})" if step.depends_on else ""
            opt = " [optional]" if step.optional else ""
            # Resolve param templates
            resolved_params = {}
            for k, v in step.param_template.items():
                for pname, pval in match.extracted_params.items():
                    v = v.replace(f"{{{{{pname}}}}}", pval)
                resolved_params[k] = v

            param_str = ", ".join(f"{k}={v}" for k, v in resolved_params.items()) if resolved_params else "no params"
            lines.append(f"  {i+1}. `{step.tool_name}` ({param_str}){deps}{opt}") # type: ignore
            if step.description:
                lines.append(f"     → {step.description}")

        lines.append(f"\nScenario success rate: {scenario.success_rate:.0%} ({scenario.success_count + scenario.failure_count} runs)")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Learning
    # ------------------------------------------------------------------

    def record_outcome(self, scenario_id: str, success: bool) -> None:
        """Record scenario execution outcome for learning."""
        scenario = self._scenarios.get(scenario_id)
        if not scenario:
            return

        if success:
            scenario.success_count += 1
        else:
            scenario.failure_count += 1
        scenario.last_used = time.time()

        if self._lmdb:
            self._lmdb.put(f"scenario:outcome:{scenario_id}:{int(time.time())}", {
                "success": success,
                "timestamp": time.time(),
            })

        logger.debug(
            "ScenarioRegistry: recorded outcome for '{}': success={} (rate: {:.0%})",
            scenario_id, success, scenario.success_rate,
        )

    # ------------------------------------------------------------------
    # Dynamic scenario registration
    # ------------------------------------------------------------------

    def register_scenario(self, scenario: Scenario) -> None:
        """Register a new or updated scenario."""
        self._scenarios[scenario.id] = scenario

        if self._lmdb:
            self._lmdb.put(f"scenario:def:{scenario.id}", {
                "id": scenario.id,
                "intent": scenario.intent,
                "patterns": scenario.patterns,
                "keywords": scenario.keywords,
                "steps": [
                    {
                        "server": s.server,
                        "tool_name": s.tool_name,
                        "param_template": s.param_template,
                        "depends_on": s.depends_on,
                        "description": s.description,
                        "optional": s.optional,
                    }
                    for s in scenario.steps
                ],
                "confidence_threshold": scenario.confidence_threshold,
                "success_count": scenario.success_count,
                "failure_count": scenario.failure_count,
            })

    def auto_register_from_tools(self, server_name: str, tools: list[Any]) -> int:
        """Auto-generate basic scenarios from discovered MCP tool definitions.

        Creates simple 1-step scenarios for each tool based on its name
        and description.
        """
        count = 0
        for tool in tools:
            tool_name = getattr(tool, "name", str(tool))
            description = getattr(tool, "description", "")
            if not description:
                continue

            scenario_id = f"auto_{server_name}_{tool_name}"
            if scenario_id in self._scenarios:
                continue

            # Extract action verbs from description for keyword generation
            desc_lower = description.lower() # type: ignore
            keywords = []
            for verb in ["create", "delete", "list", "get", "update", "search", "deploy", "init"]:
                if verb in desc_lower: # type: ignore
                    keywords.append(f"{verb} {server_name}")

            if not keywords:
                keywords = [f"{server_name} {tool_name.split('_')[-1]}"] # type: ignore

            scenario = Scenario(
                id=scenario_id,
                intent=f"auto_{tool_name}",
                patterns=[],
                keywords=keywords,
                steps=[
                    ActionStep(
                        server=server_name,
                        tool_name=tool_name,
                        param_template={},
                        description=description[:100], # type: ignore
                    ),
                ],
                confidence_threshold=0.7,  # Higher threshold for auto-generated
            )
            self.register_scenario(scenario) # type: ignore
            count += 1

        logger.info("ScenarioRegistry: auto-registered {} scenarios from '{}' tools", count, server_name)
        return count

    @property
    def scenario_count(self) -> int:
        return len(self._scenarios)

    def list_scenarios(self) -> list[dict[str, Any]]:
        """List all registered scenarios."""
        return [
            {
                "id": s.id,
                "intent": s.intent,
                "keywords": s.keywords[:3], # type: ignore
                "steps": len(s.steps),
                "success_rate": f"{s.success_rate:.0%}",
            }
            for s in sorted(self._scenarios.values(), key=lambda x: x.success_count, reverse=True)
        ]
