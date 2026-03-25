"""
PostponedJudgement — Defers success/failure evaluation for multi-step
sequences where intermediate steps can't be judged in isolation.

Core Idea:
  - Opening a menu is NOT a failure if the next step is to select an item
  - Typing text is NOT a success until the form is submitted
  - Scrolling is never judged immediately — its purpose depends on the next action

When objective completes:
  SUCCESS -> all deferred steps -> SEMI_SUCCESS (they contributed)
  FAILED  -> all deferred steps -> FAILED (they didn't help)
"""

import time
from enum import Enum
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any


class DeferralReason(Enum):
    INTERMEDIATE_STEP = "INTERMEDIATE_STEP"
    MENU_OPENED = "MENU_OPENED"
    DIALOG_OPENED = "DIALOG_OPENED"
    TEXT_ENTERED = "TEXT_ENTERED"
    SCROLL_ACTION = "SCROLL_ACTION"
    NAVIGATION_IN_PROGRESS = "NAVIGATION_IN_PROGRESS"
    MULTI_STEP_SEQUENCE = "MULTI_STEP_SEQUENCE"


@dataclass
class DeferredJudgement:
    """A step whose outcome evaluation is postponed."""
    step_id: str
    action_type: str
    target_label: str
    deferral_reason: str
    deferred_at: float
    context: Dict[str, Any] = field(default_factory=dict)
    resolved: bool = False
    final_outcome: Optional[str] = None
    resolved_at: float = 0.0


# ─────────────────  Deferral Heuristics  ─────────────────

# Actions that should typically be deferred
_ALWAYS_DEFER = {"scroll", "wait"}

# Actions deferred if they're NOT the last step
_DEFER_IF_INTERMEDIATE = {"type", "hotkey", "press"}

# Actions deferred based on visual outcome
_DEFER_ON_PARTIAL_CHANGE = {"click", "double_click"}


class PostponedJudgement:
    """Manages deferred evaluations for multi-step action sequences."""

    def __init__(self):
        self._pending: Dict[str, DeferredJudgement] = {}  # step_id -> DeferredJudgement
        self._resolved: List[DeferredJudgement] = []
        print("[PostponedJudgement] Initialized")

    # ═══════════════════════  Should Defer?  ═══════════════════════

    def should_defer(
        self,
        action_type: str,
        step_index: int,
        total_steps_estimate: int,
        visual_change_pct: float = 0.0,
        dialog_opened: bool = False,
        menu_opened: bool = False,
    ) -> tuple:
        """
        Determines if evaluation should be deferred based on heuristics.
        Returns: (should_defer: bool, reason: str)
        """
        is_last_step = step_index >= total_steps_estimate - 1

        # Always defer scrolls and waits
        if action_type in _ALWAYS_DEFER:
            return True, DeferralReason.SCROLL_ACTION.value

        # Never defer the final step
        if is_last_step:
            return False, ""

        # Defer intermediate typing/hotkeys
        if action_type in _DEFER_IF_INTERMEDIATE:
            return True, DeferralReason.INTERMEDIATE_STEP.value

        # Defer clicks that opened menus/dialogs (partial progress)
        if dialog_opened:
            return True, DeferralReason.DIALOG_OPENED.value
        if menu_opened:
            return True, DeferralReason.MENU_OPENED.value

        # Defer clicks with minimal visual change (might be intermediate)
        if action_type in _DEFER_ON_PARTIAL_CHANGE:
            if 1.0 < visual_change_pct < 30.0:
                return True, DeferralReason.NAVIGATION_IN_PROGRESS.value

        return False, ""

    # ═══════════════════════  Defer / Resolve  ═══════════════════════

    def defer_judgement(
        self,
        step_id: str,
        action_type: str,
        target_label: str,
        reason: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> DeferredJudgement:
        """Marks a step as 'judgement deferred'."""
        dj = DeferredJudgement(
            step_id=step_id,
            action_type=action_type,
            target_label=target_label,
            deferral_reason=reason,
            deferred_at=time.time(),
            context=context or {},
        )
        self._pending[step_id] = dj

        print(f"[PostponedJudgement] Deferred: {action_type} -> '{target_label}' "
              f"reason={reason}")

        return dj

    def resolve_deferred(self, step_id: str, final_outcome: str):
        """Retroactively classifies a deferred step."""
        dj = self._pending.pop(step_id, None)
        if not dj:
            return

        dj.resolved = True
        dj.final_outcome = final_outcome
        dj.resolved_at = time.time()
        self._resolved.append(dj)
        
        # Memory capping (skip if it gets too large)
        if len(self._resolved) > 500:
            self._resolved = self._resolved[-500:]  # type: ignore

        print(f"[PostponedJudgement] Resolved '{dj.target_label}': {final_outcome} "
              f"(was deferred {time.time() - dj.deferred_at:.1f}s ago)")

    def auto_resolve_by_objective(self, objective_outcome: str):
        """
        When objective completes, resolve all pending deferred steps:
          objective SUCCESS -> deferred steps -> SEMI_SUCCESS
          objective FAILED  -> deferred steps -> FAILED
        """
        resolution = "SEMI_SUCCESS" if objective_outcome == "SUCCESS" else "FAILED"

        pending_ids = list(self._pending.keys())
        for step_id in pending_ids:
            self.resolve_deferred(step_id, resolution)

        if pending_ids:
            print(f"[PostponedJudgement] Auto-resolved {len(pending_ids)} deferred steps -> {resolution}")

    # ═══════════════════════  Queries  ═══════════════════════

    def get_pending_judgements(self) -> List[DeferredJudgement]:
        """Returns all unresolved deferred evaluations."""
        return list(self._pending.values())

    def get_pending_count(self) -> int:
        """Returns count of pending deferred evaluations."""
        return len(self._pending)

    def get_resolved_summary(self) -> Dict[str, int]:
        """Returns summary of resolved deferred evaluations."""
        summary = {"SEMI_SUCCESS": 0, "FAILED": 0, "SUCCESS": 0}
        for dj in self._resolved:
            outcome = dj.final_outcome or "FAILED"
            summary[outcome] = summary.get(outcome, 0) + 1
        return summary

    def get_deferral_stats(self) -> Dict[str, Any]:
        """Returns overall deferral statistics."""
        total_deferred = len(self._resolved) + len(self._pending)
        return {
            "total_deferred": total_deferred,
            "pending": len(self._pending),
            "resolved": len(self._resolved),
            "resolution_breakdown": self.get_resolved_summary(),
        }
