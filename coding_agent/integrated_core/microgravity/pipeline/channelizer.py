"""
MICROGRAVITY CORE — Channelization Pipeline

The strict middleware funnel between raw LLM output and actual tool/command execution.

Steps:
1. Analytical Scripts (regex enforcement, phrase/word detection, safety filtering)
2. Hypothesis Comparison (expected vs actual output validation)
3. Output Channelization (route output by type: code, command, data, error)
4. Schema Validation (Pydantic enforcement of structured output contracts)
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Optional

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════
#  Analytical Scripts — Regex & Phrase Detection
# ═══════════════════════════════════════════════════════════════

@dataclass
class AnalyticalRule:
    """A single analytical filter rule."""
    name: str
    pattern: str         # regex pattern
    action: str          # "block" | "warn" | "flag" | "rewrite"
    message: str = ""
    replacement: str = ""  # for "rewrite" action


DEFAULT_SAFETY_RULES: list[AnalyticalRule] = [
    AnalyticalRule(
        name="dangerous_rm",
        pattern=r"rm\s+(-rf?|--recursive)\s+[/\\]",
        action="block",
        message="Blocked: Recursive delete on root-level path",
    ),
    AnalyticalRule(
        name="format_disk",
        pattern=r"(format\s+[a-zA-Z]:|mkfs|fdisk)",
        action="block",
        message="Blocked: Disk formatting command detected",
    ),
    AnalyticalRule(
        name="env_var_leak",
        pattern=r"(API_KEY|SECRET|PASSWORD|TOKEN)\s*=\s*['\"][^'\"]{8,}",
        action="flag",
        message="Warning: Potential secret in output",
    ),
    AnalyticalRule(
        name="hallucination_marker",
        pattern=r"(As an AI language model|I cannot|I don't have access to|I apologize)",
        action="flag",
        message="Hallucination marker detected — agent may be confused about capabilities",
    ),
    AnalyticalRule(
        name="infinite_loop_indicator",
        pattern=r"(while\s+True|for\s+;;\s*|for\s+\(\s*;\s*;\s*\))",
        action="warn",
        message="Warning: Potential infinite loop pattern detected",
    ),
]


class AnalyticalScripts:
    """
    Regex enforcement, phrase detection, and safety filtering layer.
    
    Runs BEFORE any LLM output reaches a tool or the execution environment.
    """

    def __init__(self, custom_rules: list[AnalyticalRule] | None = None):
        self._rules = DEFAULT_SAFETY_RULES.copy()
        if custom_rules:
            self._rules.extend(custom_rules)

    def analyze(self, text: str) -> AnalysisResult:
        """Run all analytical rules against the given text."""
        result = AnalysisResult()

        for rule in self._rules:
            matches = re.findall(rule.pattern, text, re.IGNORECASE)
            if matches:
                finding = AnalysisFinding(
                    rule_name=rule.name,
                    action=rule.action,
                    message=rule.message,
                    matches=[str(m) for m in matches],
                )
                result.findings.append(finding)

                if rule.action == "block":
                    result.blocked = True
                    result.block_reason = rule.message
                    logger.warning(f"BLOCKED by {rule.name}: {rule.message}")
                elif rule.action == "warn":
                    logger.warning(f"WARN from {rule.name}: {rule.message}")
                elif rule.action == "flag":
                    logger.info(f"FLAG from {rule.name}: {rule.message}")
                elif rule.action == "rewrite":
                    text = re.sub(rule.pattern, rule.replacement, text, flags=re.IGNORECASE)
                    result.rewritten = True

        result.processed_text = text
        return result

    def add_rule(self, rule: AnalyticalRule) -> None:
        self._rules.append(rule)

    def remove_rule(self, name: str) -> bool:
        before = len(self._rules)
        self._rules = [r for r in self._rules if r.name != name]
        return len(self._rules) < before


@dataclass
class AnalysisFinding:
    rule_name: str
    action: str
    message: str
    matches: list[str] = field(default_factory=list)


@dataclass
class AnalysisResult:
    blocked: bool = False
    block_reason: str = ""
    rewritten: bool = False
    processed_text: str = ""
    findings: list[AnalysisFinding] = field(default_factory=list)


# ═══════════════════════════════════════════════════════════════
#  Hypothesis Engine — Expected vs Actual Comparison
# ═══════════════════════════════════════════════════════════════

class HypothesisEngine:
    """
    Before a tool is executed, the agent writes a hypothesis about expected output.
    After execution, this engine compares reality to hypothesis.
    
    Comparison modes:
    - Structural: Does the output have the expected shape (keys, types)?
    - Content: Does the output contain expected keywords?
    - Semantic: Is the output's meaning aligned? (uses LLM if available)
    """

    def __init__(self, llm_call_fn=None):
        self._llm_call = llm_call_fn

    def compare_structural(self, hypothesis: dict, actual: Any) -> HypothesisResult:
        """Compare JSON structure: key presence and value types."""
        if not isinstance(actual, dict):
            return HypothesisResult(
                match=False,
                score=0.0,
                explanation=f"Expected dict, got {type(actual).__name__}",
            )

        expected_keys = set(hypothesis.get("expected_keys", []))
        actual_keys = set(actual.keys())
        overlap = expected_keys & actual_keys
        score = len(overlap) / len(expected_keys) if expected_keys else 1.0

        missing = expected_keys - actual_keys
        extra = actual_keys - expected_keys

        return HypothesisResult(
            match=score >= 0.8,
            score=score,
            explanation=f"Key overlap: {len(overlap)}/{len(expected_keys)}. "
                        f"Missing: {missing or 'none'}. Extra: {extra or 'none'}.",
        )

    def compare_content(self, hypothesis: str, actual: str) -> HypothesisResult:
        """Check if expected keywords/phrases appear in the output."""
        if not hypothesis or not actual:
            return HypothesisResult(match=True, score=1.0, explanation="No hypothesis to compare")

        # Extract keywords from hypothesis
        keywords = [w.strip().lower() for w in hypothesis.split(",") if w.strip()]
        if not keywords:
            keywords = hypothesis.lower().split()

        actual_lower = actual.lower()
        found = [kw for kw in keywords if kw in actual_lower]
        score = len(found) / len(keywords) if keywords else 1.0

        return HypothesisResult(
            match=score >= 0.6,
            score=score,
            explanation=f"Keywords found: {len(found)}/{len(keywords)} — {found}",
        )

    async def compare_semantic(self, hypothesis: str, actual: str) -> HypothesisResult:
        """Use an LLM to semantically compare hypothesis vs actual."""
        if not self._llm_call:
            return self.compare_content(hypothesis, actual)

        from models.schemas import LLMRequest
        request = LLMRequest(
            messages=[
                {"role": "system", "content": (
                    "You compare an expected hypothesis with actual output. "
                    "Respond with JSON: {\"match\": bool, \"score\": float 0-1, \"explanation\": str}"
                )},
                {"role": "user", "content": (
                    f"HYPOTHESIS: {hypothesis}\n\nACTUAL OUTPUT: {actual[:2000]}"
                )},
            ],
            model_tier="tactical",
            temperature=0.0,
            max_tokens=200,
        )

        try:
            response = await self._llm_call(request)
            result = json.loads(response.content)
            return HypothesisResult(
                match=result.get("match", False),
                score=float(result.get("score", 0.0)),
                explanation=result.get("explanation", ""),
            )
        except Exception as e:
            logger.warning(f"Semantic comparison failed: {e}")
            return self.compare_content(hypothesis, actual)


@dataclass
class HypothesisResult:
    match: bool
    score: float
    explanation: str = ""


# ═══════════════════════════════════════════════════════════════
#  Output Channelizer — Route by Content Type
# ═══════════════════════════════════════════════════════════════

class OutputChannelizer:
    """
    Routes LLM output to the appropriate execution channel based on content analysis.
    
    Channels:
    - code: Detected programming language → sandbox / file write
    - command: Shell command → terminal executor
    - data: Structured data → storage / API call
    - text: Natural language → user response / agent message
    - error: Error report → feedback processing mode
    """

    @staticmethod
    def classify(text: str) -> str:
        """Classify the type of output content."""
        # Check for code blocks
        if re.search(r"```\w*\n", text):
            return "code"

        # Check for shell commands
        shell_patterns = [
            r"^\s*(npm|pip|docker|kubectl|git|curl|wget|mkdir|ls|cd|cat|echo)\s",
            r"^\s*\$\s+",
            r"^\s*(sudo|chmod|chown|apt|brew|yarn|npx)\s",
        ]
        for pattern in shell_patterns:
            if re.search(pattern, text, re.MULTILINE):
                return "command"

        # Check for structured data
        try:
            json.loads(text)
            return "data"
        except (json.JSONDecodeError, ValueError):
            pass

        # Check for error indicators
        error_patterns = [r"error:", r"exception:", r"traceback", r"failed:", r"fatal:"]
        if any(re.search(p, text, re.IGNORECASE) for p in error_patterns):
            return "error"

        return "text"

    @staticmethod
    def extract_code_blocks(text: str) -> list[dict[str, str]]:
        """Extract all code blocks with their language tags."""
        pattern = r"```(\w*)\n(.*?)```"
        matches = re.findall(pattern, text, re.DOTALL)
        return [{"language": lang or "text", "code": code.strip()} for lang, code in matches]

    @staticmethod
    def extract_commands(text: str) -> list[str]:
        """Extract shell commands from text."""
        commands = []
        for line in text.splitlines():
            stripped = line.strip()
            # Lines starting with $ prompt
            if stripped.startswith("$ "):
                commands.append(stripped[2:])
            # Lines that look like commands
            elif re.match(r"^(npm|pip|docker|kubectl|git|curl|wget|mkdir)\s", stripped):
                commands.append(stripped)
        return commands
