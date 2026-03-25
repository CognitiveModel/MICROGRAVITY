"""
LoopSentinel — Sub-objective tracking and behavioral loop detection.

Monitors the agentic step history for patterns indicating the agent is
stuck in a loop:
  - Repetition: Same (action, target) attempted ≥ N times in a window.
  - Oscillation: Agent alternates between 2 targets (A→B→A→B).
  - Progress Stall: No successful action in last M steps.
  - Coordinate Drift: Clicking the same target at ≤20px-different coords.

Severity tiers:
  MILD      — 3 repeats or stall=4  → inject warning
  MODERATE  — 5 repeats or stall=6  → trigger interrupt routine
  CRITICAL  — 7+ repeats or stall=8 → force goal_failed
"""

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
from collections import Counter, defaultdict
from enum import Enum


class LoopSeverity(str, Enum):
    NONE = "NONE"
    MILD = "MILD"
    MODERATE = "MODERATE"
    CRITICAL = "CRITICAL"


@dataclass
class LoopDiagnosis:
    """Result of a loop analysis pass."""
    severity: LoopSeverity
    loop_type: str            # REPETITION | OSCILLATION | STALL | COORD_DRIFT | NONE
    description: str          # Human-readable description
    offending_target: str     # The target that is looping
    offending_action: str     # The action type
    repeat_count: int         # How many times repeated
    evidence: List[str]       # Step numbers / details
    suggested_strategies: List[str]  # Alternative approaches to try

    @property
    def should_interrupt(self) -> bool:
        return self.severity in (LoopSeverity.MODERATE, LoopSeverity.CRITICAL)

    @property
    def should_fail(self) -> bool:
        return self.severity == LoopSeverity.CRITICAL


class LoopSentinel:
    """Monitors step history for behavioral loops and triggers interrupts."""

    # ── Thresholds ──
    WINDOW_SIZE = 10          # How many recent steps to analyze
    MILD_REPEAT = 3
    MODERATE_REPEAT = 5
    CRITICAL_REPEAT = 7
    STALL_MILD = 4
    STALL_MODERATE = 6
    STALL_CRITICAL = 8
    COORD_DRIFT_PX = 25       # Max pixel distance to consider "same location"

    def __init__(self):
        # Rolling counters per sub-objective key: (action_type, normalized_target)
        self._sub_objective_counts: Dict[str, int] = defaultdict(int)
        self._sub_objective_first_seen: Dict[str, float] = {}
        self._interrupt_history: List[Dict[str, Any]] = []
        self._total_interrupts: int = 0

    # ═══════════════════════  Main Analysis  ═══════════════════════

    def analyze(self, step_history: List[Dict[str, Any]]) -> LoopDiagnosis:
        """
        Analyzes the recent step history for loop patterns.
        Returns a LoopDiagnosis with severity and recommendations.
        """
        if len(step_history) < 3:
            return self._no_loop()

        window = list(step_history[-self.WINDOW_SIZE:])  # type: ignore

        # Run detectors in priority order
        diagnosis = self._detect_repetition_loop(window)
        if diagnosis.severity != LoopSeverity.NONE:
            return diagnosis

        diagnosis = self._detect_oscillation_loop(window)
        if diagnosis.severity != LoopSeverity.NONE:
            return diagnosis

        diagnosis = self._detect_coordinate_drift(window)
        if diagnosis.severity != LoopSeverity.NONE:
            return diagnosis

        diagnosis = self._detect_progress_stall(window)
        if diagnosis.severity != LoopSeverity.NONE:
            return diagnosis

        return self._no_loop()

    # ═══════════════════════  Detectors  ═══════════════════════

    def _detect_repetition_loop(self, window: List[Dict]) -> LoopDiagnosis:
        """Detects repeated (action, target) pairs."""
        key_counts: Dict[str, int] = Counter()
        key_steps: Dict[str, List[int]] = defaultdict(list)

        for step in window:
            key = self._make_sub_objective_key(step)
            key_counts[key] += 1
            key_steps[key].append(step.get("step_num", 0))

        if not key_counts:
            return self._no_loop()

        most_common_key, count = key_counts.most_common(1)[0]

        if count >= self.CRITICAL_REPEAT:
            severity = LoopSeverity.CRITICAL
        elif count >= self.MODERATE_REPEAT:
            severity = LoopSeverity.MODERATE
        elif count >= self.MILD_REPEAT:
            severity = LoopSeverity.MILD
        else:
            return self._no_loop()

        # Parse the key back into action/target
        parts = most_common_key.split("::", 1)
        action_type = parts[0] if len(parts) > 0 else "unknown"
        target = parts[1] if len(parts) > 1 else ""

        # Update tracking
        self._sub_objective_counts[most_common_key] = count

        return LoopDiagnosis(
            severity=severity,
            loop_type="REPETITION",
            description=(
                f"The agent has attempted '{action_type}' on '{target}' "
                f"{count} times in the last {len(window)} steps."
            ),
            offending_target=target,
            offending_action=action_type,
            repeat_count=count,
            evidence=[f"Steps: {key_steps[most_common_key]}"],
            suggested_strategies=self._get_alternative_strategies(action_type, target),
        )

    def _detect_oscillation_loop(self, window: List[Dict]) -> LoopDiagnosis:
        """Detects A→B→A→B alternation patterns."""
        if len(window) < 4:
            return self._no_loop()

        targets = [self._normalize_target(s.get("target", "")) for s in window if s.get("target")]
        if len(targets) < 4:
            return self._no_loop()

        # Check last 6 targets for alternation
        recent = list(targets[-6:])  # type: ignore
        if len(recent) >= 4:
            # Check if it alternates between exactly 2 values
            unique = list(set(recent))
            if len(unique) == 2:
                # Verify it's actually alternating (not just 2 different targets)
                alternating = True
                for i in range(2, len(recent)):
                    if recent[i] != recent[i - 2]:
                        alternating = False
                        break

                if alternating:
                    osc_count = len(recent) // 2
                    severity = LoopSeverity.MODERATE if osc_count >= 3 else LoopSeverity.MILD
                    return LoopDiagnosis(
                        severity=severity,
                        loop_type="OSCILLATION",
                        description=(
                            f"The agent is oscillating between '{unique[0]}' and "
                            f"'{unique[1]}' ({osc_count} cycles)."
                        ),
                        offending_target=unique[0],
                        offending_action="click",
                        repeat_count=osc_count,
                        evidence=[f"Pattern: {' → '.join(recent)}"],
                        suggested_strategies=[
                            "STOP interacting with both targets. Try a completely different UI path.",
                            "Use a keyboard shortcut to bypass both elements.",
                            "Scroll or resize the window to reveal different options.",
                            "Try right-clicking for a context menu alternative.",
                        ],
                    )

        return self._no_loop()

    def _detect_coordinate_drift(self, window: List[Dict]) -> LoopDiagnosis:
        """Detects clicking the same target at slightly different coordinates repeatedly."""
        if len(window) < 3:
            return self._no_loop()

        # Group clicks by normalized target
        target_coords: Dict[str, List[Tuple[int, int, int]]] = defaultdict(list)
        for step in window:
            action_type = step.get("action_type", "")
            if action_type in ("click", "double_click") and not step.get("success"):
                target = self._normalize_target(step.get("target", ""))
                coords = step.get("resolved_coords") or {}
                x = coords.get("x")
                y = coords.get("y")
                if x is not None and y is not None:
                    target_coords[target].append((x, y, step.get("step_num", 0)))

        for target, coord_list in target_coords.items():
            if len(coord_list) < 3:
                continue

            # Check if all coordinates are within COORD_DRIFT_PX of each other
            xs = [c[0] for c in coord_list]
            ys = [c[1] for c in coord_list]
            x_spread = max(xs) - min(xs)
            y_spread = max(ys) - min(ys)

            if x_spread <= self.COORD_DRIFT_PX * 2 and y_spread <= self.COORD_DRIFT_PX * 2:
                count = len(coord_list)
                if count >= 5:
                    severity = LoopSeverity.MODERATE
                else:
                    severity = LoopSeverity.MILD

                step_nums = [c[2] for c in coord_list]
                return LoopDiagnosis(
                    severity=severity,
                    loop_type="COORD_DRIFT",
                    description=(
                        f"The agent has clicked on '{target}' {count} times "
                        f"within a {x_spread}×{y_spread}px region, all failing."
                    ),
                    offending_target=target,
                    offending_action="click",
                    repeat_count=count,
                    evidence=[
                        f"Steps: {step_nums}",
                        f"Coord spread: {x_spread}×{y_spread}px",
                    ],
                    suggested_strategies=[
                        "The element may not be clickable at this location. Try a different approach.",
                        "The element might be behind a modal/overlay. Check for popups.",
                        "Try using keyboard navigation (Tab/Enter) to reach this element.",
                        "Request a closeup zoom to verify the element's exact boundary.",
                    ],
                )

        return self._no_loop()

    def _detect_progress_stall(self, window: List[Dict]) -> LoopDiagnosis:
        """Detects when no actions have succeeded in M consecutive steps."""
        consecutive_failures = 0
        for step in reversed(window):
            if step.get("success"):
                break
            consecutive_failures += 1

        if consecutive_failures >= self.STALL_CRITICAL:
            severity = LoopSeverity.CRITICAL
        elif consecutive_failures >= self.STALL_MODERATE:
            severity = LoopSeverity.MODERATE
        elif consecutive_failures >= self.STALL_MILD:
            severity = LoopSeverity.MILD
        else:
            return self._no_loop()

        # Collect the failed actions for context
        failed_actions = []
        for step in list(window[-consecutive_failures:]):  # type: ignore
            action_type = step.get("action_type", "unknown")
            target = step.get("target", "unknown")
            failed_actions.append(
                f"Step {step.get('step_num', '?')}: {action_type} on '{target}'"
            )

        return LoopDiagnosis(
            severity=severity,
            loop_type="STALL",
            description=(
                f"No successful action in the last {consecutive_failures} steps. "
                f"The agent appears completely stuck."
            ),
            offending_target="multiple",
            offending_action="various",
            repeat_count=consecutive_failures,
            evidence=failed_actions,
            suggested_strategies=[
                "STOP the current approach entirely. Re-analyze the screen from scratch.",
                "Press Escape to dismiss any hidden overlays or modals.",
                "Try using the Windows search (Win+S) to find and open the target application.",
                "Use keyboard shortcuts (Ctrl+L, Alt+D) to navigate directly.",
                "Consider if the goal is actually achievable with the current application state.",
            ],
        )

    # ═══════════════════════  Interrupt Recording  ═══════════════════════

    def record_interrupt(self, diagnosis: LoopDiagnosis, recovery_action: str,
                         recovery_success: bool):
        """Records an interrupt event for statistics and learning."""
        self._total_interrupts += 1
        self._interrupt_history.append({
            "timestamp": time.time(),
            "severity": diagnosis.severity.value,
            "loop_type": diagnosis.loop_type,
            "target": diagnosis.offending_target,
            "repeat_count": diagnosis.repeat_count,
            "recovery_action": recovery_action,
            "recovery_success": recovery_success,
        })

    def get_interrupt_stats(self) -> Dict[str, Any]:
        """Returns interrupt statistics for the current session."""
        if not self._interrupt_history:
            return {"total_interrupts": 0}

        by_type = Counter(i["loop_type"] for i in self._interrupt_history)
        by_severity = Counter(i["severity"] for i in self._interrupt_history)
        recovery_rate = (
            sum(1 for i in self._interrupt_history if i["recovery_success"])
            / len(self._interrupt_history)
        )

        return {
            "total_interrupts": self._total_interrupts,
            "by_type": dict(by_type),
            "by_severity": dict(by_severity),
            "recovery_rate": round(recovery_rate, 2),  # type: ignore
        }

    # ═══════════════════════  Helpers  ═══════════════════════

    def _make_sub_objective_key(self, step: Dict) -> str:
        """Creates a normalized key for a step's sub-objective."""
        action = step.get("action_type", "unknown")
        target = self._normalize_target(step.get("target", ""))
        return f"{action}::{target}"

    def _normalize_target(self, target: str) -> str:
        """Normalizes a target description for comparison."""
        if not target:
            return ""
        # Lowercase, strip extra whitespace, remove quotes
        t = target.lower().strip().replace("'", "").replace('"', '')
        # Remove common suffixes that don't change the semantic target
        for suffix in [" button", " icon", " field", " input", " link", " tab"]:
            if t.endswith(suffix):  # type: ignore
                t = t[:-len(suffix)].strip()  # type: ignore
        return t

    def _get_alternative_strategies(self, action: str, target: str) -> List[str]:
        """Returns context-appropriate alternative strategies."""
        strategies = [
            "Try a COMPLETELY DIFFERENT approach to achieve this sub-objective.",
            "Use keyboard shortcuts instead of mouse clicks.",
        ]

        target_lower = target.lower()
        if "login" in target_lower or "sign in" in target_lower:
            strategies.extend([
                "If the login form is inside a dialog, make sure you're clicking INSIDE the dialog.",
                "Try pressing Tab to move between fields and Enter to submit.",
                "Check if there's a 'Sign in with Google' or SSO alternative.",
            ])
        elif "search" in target_lower:
            strategies.extend([
                "Try clicking the address/URL bar and typing the search query directly.",
                "Use Ctrl+F or Ctrl+K for in-page or app search.",
                "Try navigating to the search URL directly.",
            ])
        elif "close" in target_lower or "dismiss" in target_lower:
            strategies.extend([
                "Press Escape to dismiss dialogs/modals.",
                "Try clicking outside the modal to close it.",
                "Use Alt+F4 to close the window entirely.",
            ])
        else:
            strategies.extend([
                "Scroll the page to reveal the element if it's off-screen.",
                "Try right-clicking for a context menu with alternative options.",
                "Resize or reposition the window to reveal obscured UI elements.",
            ])

        return strategies

    def _no_loop(self) -> LoopDiagnosis:
        """Returns a clean no-loop diagnosis."""
        return LoopDiagnosis(
            severity=LoopSeverity.NONE,
            loop_type="NONE",
            description="",
            offending_target="",
            offending_action="",
            repeat_count=0,
            evidence=[],
            suggested_strategies=[],
        )


# ═══════════════════════  Recovery Root Causes  ═══════════════════════

class LoopRootCause(str, Enum):
    STALE_COORDS = "STALE_COORDS"           # Memorised coordinates no longer valid
    WINDOW_RESIZED = "WINDOW_RESIZED"       # Target window geometry changed
    ELEMENT_OCCLUDED = "ELEMENT_OCCLUDED"   # Target behind overlay/modal
    WRONG_INSTANCE = "WRONG_INSTANCE"       # Clicking duplicate label on wrong element
    APP_STATE_CHANGED = "APP_STATE_CHANGED" # App navigated away from expected state
    UNRESPONSIVE_TARGET = "UNRESPONSIVE_TARGET"  # Element exists but doesn't react
    UNKNOWN = "UNKNOWN"


@dataclass
class RecoveryPlan:
    """Concrete recovery plan generated by the recovery engine."""
    root_cause: LoopRootCause
    recovery_actions: List[Dict[str, Any]]   # Actions to execute for recovery
    context_injection: str                    # Text to inject into planner context
    should_normalize_window: bool             # Whether to auto-maximize/restore window
    should_invalidate_coords: bool            # Whether to clear cached coords
    should_try_alternative_route: bool        # Whether planner should abandon current approach
    past_mitigations: List[str]              # What worked before for similar situations
    confidence: float                         # 0.0-1.0 confidence in the recovery plan


class LoopRecoveryEngine:
    """
    Autonomous recovery module that wraps loop diagnosis with deeper
    root-cause analysis, generates concrete recovery plans, and records
    incidents for future mitigation.
    """

    def __init__(self, sentinel: LoopSentinel, experiential_memory=None):
        self.sentinel = sentinel
        self.memory = experiential_memory
        self._incident_log: List[Dict[str, Any]] = []
        # Maps (app_class, loop_type) → list of recovery strategies that worked
        self._successful_mitigations: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

    def diagnose_and_recover(
        self,
        diagnosis: LoopDiagnosis,
        step_history: List[Dict[str, Any]],
        window_state: Optional[Dict[str, Any]] = None,
        app_class: str = "",
    ) -> RecoveryPlan:
        """
        Performs deep root-cause analysis on a loop diagnosis and generates
        a concrete recovery plan.

        Args:
            diagnosis: The LoopDiagnosis from LoopSentinel.analyze()
            step_history: Full step history from the planner
            window_state: Current window geometry {rect, is_maximized, title, hwnd}
            app_class: Current application class name
        """
        root_cause = self._classify_root_cause(diagnosis, step_history, window_state)
        recovery_actions = self._generate_recovery_actions(root_cause, diagnosis, window_state)
        context_injection = self._build_recovery_context(root_cause, diagnosis, app_class)
        past = self._recall_past_mitigations(app_class, diagnosis.loop_type)

        plan = RecoveryPlan(
            root_cause=root_cause,
            recovery_actions=recovery_actions,
            context_injection=context_injection,
            should_normalize_window=(root_cause in (
                LoopRootCause.WINDOW_RESIZED,
                LoopRootCause.STALE_COORDS,
                LoopRootCause.ELEMENT_OCCLUDED,
            )),
            should_invalidate_coords=(root_cause in (
                LoopRootCause.STALE_COORDS,
                LoopRootCause.WINDOW_RESIZED,
            )),
            should_try_alternative_route=(root_cause in (
                LoopRootCause.WRONG_INSTANCE,
                LoopRootCause.APP_STATE_CHANGED,
                LoopRootCause.UNRESPONSIVE_TARGET,
            ) or diagnosis.severity == LoopSeverity.CRITICAL),
            past_mitigations=past,
            confidence=self._estimate_confidence(root_cause, diagnosis),
        )

        # Record the incident
        self._record_incident(diagnosis, plan, app_class)

        print(f"[LoopRecoveryEngine] Root cause: {root_cause.value} | "
              f"Recovery: {'normalize_window' if plan.should_normalize_window else ''} "
              f"{'invalidate_coords' if plan.should_invalidate_coords else ''} "
              f"{'alt_route' if plan.should_try_alternative_route else ''} | "
              f"Confidence: {plan.confidence:.2f}")

        return plan

    def record_recovery_outcome(self, plan: RecoveryPlan, success: bool, app_class: str = ""):
        """Records whether a recovery plan actually worked, for future learning."""
        mitigation_key = f"{app_class}::{plan.root_cause.value}"
        outcome = {
            "root_cause": plan.root_cause.value,
            "recovery_actions": [a.get("action", "") for a in plan.recovery_actions],
            "success": success,
            "timestamp": time.time(),
        }
        if success:
            self._successful_mitigations[mitigation_key].append(outcome)

        # Update the sentinel's interrupt record
        if self._incident_log:
            self._incident_log[-1]["recovery_success"] = success

        # Persist to experiential memory
        if self.memory and hasattr(self.memory, 'record_loop_incident'):
            try:
                self.memory.record_loop_incident(
                    app_class=app_class,
                    loop_type=self._incident_log[-1].get("loop_type", "") if self._incident_log else "",
                    offending_target=self._incident_log[-1].get("target", "") if self._incident_log else "",
                    repeat_count=self._incident_log[-1].get("repeat_count", 0) if self._incident_log else 0,
                    root_cause=plan.root_cause.value,
                    recovery_outcome="SUCCESS" if success else "FAILED",
                    context_summary=plan.context_injection[:200],
                )
            except Exception as e:
                print(f"[LoopRecoveryEngine] Failed to persist incident: {e}")

    # ═══════════════════════  Root Cause Classification  ═══════════════════════

    def _classify_root_cause(
        self,
        diagnosis: LoopDiagnosis,
        step_history: List[Dict[str, Any]],
        window_state: Optional[Dict[str, Any]],
    ) -> LoopRootCause:
        """Classifies the deeper root cause behind a loop detection."""

        # Rule 1: COORD_DRIFT with window that isn't maximized → WINDOW_RESIZED
        if diagnosis.loop_type == "COORD_DRIFT":
            if window_state and not window_state.get("is_maximized", True):
                return LoopRootCause.WINDOW_RESIZED
            return LoopRootCause.STALE_COORDS

        # Rule 2: REPETITION on same target → look at failure patterns
        if diagnosis.loop_type == "REPETITION":
            target = diagnosis.offending_target
            # Check if the failures mention "WRONG_TARGET" or "WRONG_INSTANCE"
            recent_failures = [s for s in step_history[-10:]
                               if not s.get("success") and
                               self.sentinel._normalize_target(s.get("target", "")) ==
                               self.sentinel._normalize_target(target)]

            wrong_instance_count = sum(
                1 for f in recent_failures
                if f.get("failure_info", {}).get("root_cause") in ("WRONG_TARGET", "WRONG_INSTANCE")
            )
            no_effect_count = sum(
                1 for f in recent_failures
                if f.get("failure_info", {}).get("root_cause") == "NO_EFFECT"
            )

            if wrong_instance_count >= 2:
                return LoopRootCause.WRONG_INSTANCE
            elif no_effect_count >= 2:
                # Element is there but not responding to clicks
                if window_state and not window_state.get("is_maximized", True):
                    return LoopRootCause.WINDOW_RESIZED
                return LoopRootCause.UNRESPONSIVE_TARGET

        # Rule 3: OSCILLATION → likely confused between two similar elements
        if diagnosis.loop_type == "OSCILLATION":
            return LoopRootCause.WRONG_INSTANCE

        # Rule 4: STALL → check for window state issues first, then app state
        if diagnosis.loop_type == "STALL":
            if window_state and not window_state.get("is_maximized", True):
                return LoopRootCause.WINDOW_RESIZED
            # Check if recent failures have diverse targets (app state changed)
            recent_targets = set(
                self.sentinel._normalize_target(s.get("target", ""))
                for s in step_history[-8:] if not s.get("success")
            )
            if len(recent_targets) >= 3:
                return LoopRootCause.APP_STATE_CHANGED

        return LoopRootCause.UNKNOWN

    # ═══════════════════════  Recovery Action Generation  ═══════════════════════

    def _generate_recovery_actions(
        self,
        root_cause: LoopRootCause,
        diagnosis: LoopDiagnosis,
        window_state: Optional[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Generates concrete actions the agent should execute for recovery."""
        actions: List[Dict[str, Any]] = []

        if root_cause == LoopRootCause.WINDOW_RESIZED:
            # Step 1: Maximize the window
            if window_state and window_state.get("hwnd"):
                actions.append({
                    "action": "maximize",
                    "app_window": window_state.get("title", ""),
                    "hwnd": window_state["hwnd"],
                    "reasoning": "Maximizing window to normalize coordinate space after resize detection",
                })
            # Step 2: Wait for UI to settle
            actions.append({"action": "wait", "duration": 1.0, "reason": "Waiting for window resize to settle"})

        elif root_cause == LoopRootCause.STALE_COORDS:
            # Request a fresh full-screen observation
            actions.append({
                "action": "wait", "duration": 0.5,
                "reason": "Clearing stale coordinate cache — next observation will re-resolve all targets",
            })

        elif root_cause == LoopRootCause.ELEMENT_OCCLUDED:
            # Try pressing Escape to dismiss overlays
            actions.append({
                "action": "press", "key": "escape",
                "reasoning": "Pressing Escape to dismiss potential overlay blocking the target element",
            })
            actions.append({"action": "wait", "duration": 0.5, "reason": "Waiting after Escape"})

        elif root_cause == LoopRootCause.WRONG_INSTANCE:
            # The planner should be told to look for the target at a DIFFERENT location
            # No direct action, but strong context injection
            pass

        elif root_cause == LoopRootCause.APP_STATE_CHANGED:
            # Press Escape, then try keyboard navigation
            actions.append({
                "action": "press", "key": "escape",
                "reasoning": "Resetting potential modal/overlay state",
            })

        elif root_cause == LoopRootCause.UNRESPONSIVE_TARGET:
            # Try Tab+Enter as alternative to clicking
            actions.append({
                "action": "press", "key": "tab",
                "reasoning": "Using keyboard navigation instead of mouse for unresponsive target",
            })

        return actions

    # ═══════════════════════  Context Building  ═══════════════════════

    def _build_recovery_context(
        self,
        root_cause: LoopRootCause,
        diagnosis: LoopDiagnosis,
        app_class: str,
    ) -> str:
        """Builds rich context injection string for the planner."""
        parts = [
            f"\n> [!CAUTION] LOOP RECOVERY ENGINE — ROOT CAUSE: {root_cause.value}",
            f"> Loop type: {diagnosis.loop_type} | Severity: {diagnosis.severity.value} | "
            f"Repeats: {diagnosis.repeat_count}",
            f"> Description: {diagnosis.description}",
        ]

        if root_cause == LoopRootCause.WINDOW_RESIZED:
            parts.extend([
                "> DIAGNOSIS: The target window has been resized or dragged from its expected position.",
                "> Memorised/cached coordinates are INVALID. The window is being auto-maximized.",
                "> You MUST re-observe the screen and resolve coordinates from the CURRENT layout.",
                "> Do NOT use any previously memorised coordinates or hint_coords from failed steps.",
            ])
        elif root_cause == LoopRootCause.STALE_COORDS:
            parts.extend([
                "> DIAGNOSIS: Cached coordinates are stale. The UI layout has shifted.",
                "> All presumption-based fast coordinates have been invalidated.",
                "> Re-resolve the target by looking at the CURRENT screenshot carefully.",
            ])
        elif root_cause == LoopRootCause.ELEMENT_OCCLUDED:
            parts.extend([
                "> DIAGNOSIS: The target element is likely behind a modal, dialog, or overlay.",
                "> Escape has been pressed to attempt dismissal.",
                "> Look for the target element INSIDE any visible dialog, not behind it.",
            ])
        elif root_cause == LoopRootCause.WRONG_INSTANCE:
            parts.extend([
                "> DIAGNOSIS: Multiple elements share the same label. You are clicking the WRONG one.",
                "> STOP clicking the same element. Use spatial context: 'the X button INSIDE the dialog'",
                "> vs 'the X button on the page behind the dialog'.",
                "> Consider using keyboard navigation (Tab, Enter) instead of clicking.",
            ])
        elif root_cause == LoopRootCause.APP_STATE_CHANGED:
            parts.extend([
                "> DIAGNOSIS: The application has changed state (navigated away, new page loaded, etc.).",
                "> The elements you were targeting may no longer exist on this screen.",
                "> Re-analyze the ENTIRE screen from scratch and replan the approach.",
            ])
        elif root_cause == LoopRootCause.UNRESPONSIVE_TARGET:
            parts.extend([
                "> DIAGNOSIS: The target element exists but does not respond to mouse clicks.",
                "> Possible reasons: element is disabled, requires hover first, or needs keyboard focus.",
                "> Switching to keyboard navigation (Tab to focus, Enter to activate).",
            ])

        # Inject past mitigations if available
        past = self._recall_past_mitigations(app_class, diagnosis.loop_type)
        if past:
            parts.append("> ")
            parts.append("> PAST SUCCESSFUL RECOVERY (from similar loops on this app):")
            for m in past[:3]:
                parts.append(f">   ✓ {m}")

        return "\n".join(parts)

    # ═══════════════════════  Memory & Learning  ═══════════════════════

    def _recall_past_mitigations(self, app_class: str, loop_type: str) -> List[str]:
        """Recalls what worked before for similar loop situations."""
        mitigations = []
        # Check local cache
        for key, outcomes in self._successful_mitigations.items():
            if app_class and app_class in key:
                for o in outcomes[-3:]:
                    mitigations.append(
                        f"{o['root_cause']}: {', '.join(o['recovery_actions'])} → SUCCESS"
                    )

        # Check experiential memory for historical loop incidents
        if self.memory and hasattr(self.memory, 'recall_loop_incidents'):
            try:
                past_incidents = self.memory.recall_loop_incidents(app_class)
                for inc in (past_incidents or [])[-3:]:
                    if inc.get("recovery_outcome") == "SUCCESS":
                        mitigations.append(
                            f"Historical: {inc.get('root_cause', '?')} on "
                            f"'{inc.get('offending_target', '?')}' → "
                            f"recovered via {inc.get('recovery_action', '?')}"
                        )
            except Exception:
                pass

        return mitigations

    def _record_incident(self, diagnosis: LoopDiagnosis, plan: RecoveryPlan, app_class: str):
        """Records an incident in the internal log."""
        self._incident_log.append({
            "timestamp": time.time(),
            "severity": diagnosis.severity.value,
            "loop_type": diagnosis.loop_type,
            "target": diagnosis.offending_target,
            "repeat_count": diagnosis.repeat_count,
            "root_cause": plan.root_cause.value,
            "recovery_actions": [a.get("action", "") for a in plan.recovery_actions],
            "app_class": app_class,
            "recovery_success": None,  # Updated later via record_recovery_outcome
        })

    def _estimate_confidence(self, root_cause: LoopRootCause, diagnosis: LoopDiagnosis) -> float:
        """Estimates confidence in the recovery plan based on root cause specificity."""
        base_confidence = {
            LoopRootCause.WINDOW_RESIZED: 0.90,    # Very detectable
            LoopRootCause.STALE_COORDS: 0.80,
            LoopRootCause.ELEMENT_OCCLUDED: 0.70,
            LoopRootCause.WRONG_INSTANCE: 0.75,
            LoopRootCause.APP_STATE_CHANGED: 0.60,
            LoopRootCause.UNRESPONSIVE_TARGET: 0.55,
            LoopRootCause.UNKNOWN: 0.30,
        }
        conf = base_confidence.get(root_cause, 0.30)
        # Lower confidence for higher severity (more stuck = less certain about fix)
        if diagnosis.severity == LoopSeverity.CRITICAL:
            conf *= 0.8
        return round(conf, 2)

    def get_incident_summary(self) -> Dict[str, Any]:
        """Returns a summary of all incidents for diagnostics."""
        if not self._incident_log:
            return {"total_incidents": 0}

        by_cause = Counter(i["root_cause"] for i in self._incident_log)
        recovery_rate = sum(
            1 for i in self._incident_log if i.get("recovery_success")
        ) / max(1, len(self._incident_log))

        return {
            "total_incidents": len(self._incident_log),
            "by_root_cause": dict(by_cause),
            "recovery_rate": round(recovery_rate, 2),
        }
