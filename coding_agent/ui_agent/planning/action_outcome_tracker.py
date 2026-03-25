"""
ActionOutcomeTracker — 3-tier action outcome tracking with detailed
categorization for objective and step-level fulfillment.

Outcome Levels:
  SUCCESS       — Goal fully achieved
  SEMI_SUCCESS  — Partial progress, intermediate result
  FAILED        — No progress or regression

Tracks per-target, per-app, per-site success rates to feed the
PresumptionEngine with weighted confidence data.
"""

import json
import os
import time
from enum import Enum
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any, Tuple
from collections import defaultdict


# ──────────────────────────  Enums  ──────────────────────────

class OutcomeLevel(Enum):
    SUCCESS = "SUCCESS"
    SEMI_SUCCESS = "SEMI_SUCCESS"
    FAILED = "FAILED"


class FailureCategory(Enum):
    NO_EFFECT = "NO_EFFECT"
    WRONG_TARGET = "WRONG_TARGET"
    WRONG_INSTANCE = "WRONG_INSTANCE"
    UNEXPECTED_STATE = "UNEXPECTED_STATE"
    TIMEOUT = "TIMEOUT"
    ELEMENT_NOT_FOUND = "ELEMENT_NOT_FOUND"
    STALE_COORDINATES = "STALE_COORDINATES"
    PERMISSION_DENIED = "PERMISSION_DENIED"
    TRANSIENT_ERROR = "TRANSIENT_ERROR"
    MISCLICK_ADJACENT = "MISCLICK_ADJACENT"
    WINDOW_BOUND_MISMATCH = "WINDOW_BOUND_MISMATCH"


class SemiSuccessCategory(Enum):
    PARTIAL_NAVIGATION = "PARTIAL_NAVIGATION"
    MENU_OPENED = "MENU_OPENED"
    DIALOG_TRIGGERED = "DIALOG_TRIGGERED"
    INPUT_ACCEPTED = "INPUT_ACCEPTED"
    STATE_CHANGED = "STATE_CHANGED"


# ──────────────────────────  Data Structures  ──────────────────────────

@dataclass
class OutcomeRecord:
    """A single action outcome record."""
    timestamp: float
    action_type: str
    target_label: str
    app_class: str
    app_instance: str
    site: str
    outcome: str                        # OutcomeLevel value
    category: str                       # FailureCategory or SemiSuccessCategory value
    coords: Optional[List[int]] = None
    latency_ms: float = 0.0
    resolution_tier: str = ""
    cv_confidence: float = 0.0
    edge_cid: str = ""
    context: Dict[str, Any] = field(default_factory=dict)
    deferred: bool = False
    deferred_final_outcome: Optional[str] = None


@dataclass
class ObjectiveOutcome:
    """Outcome of a full objective/task."""
    objective_id: str
    objective: str
    app_class: str
    app_instance: str
    outcome: str                        # OutcomeLevel value
    steps: List[OutcomeRecord] = field(default_factory=list)
    total_steps: int = 0
    successful_steps: int = 0
    semi_steps: int = 0
    failed_steps: int = 0
    total_duration_ms: float = 0.0
    timestamp: float = 0.0


# ──────────────────────────  ActionOutcomeTracker  ──────────────────────────

class ActionOutcomeTracker:
    """Tracks action outcomes with 3-tier classification and detailed stats."""

    def __init__(self, storage_dir: str = "outcome_logs"):
        self.storage_dir = storage_dir
        os.makedirs(storage_dir, exist_ok=True)

        self._step_log: List[OutcomeRecord] = []
        self._objective_log: List[ObjectiveOutcome] = []

        # Running stats: {(target_label, app_class): {success: N, semi: N, fail: N}}
        self._target_stats: Dict[Tuple[str, str], Dict[str, int]] = {}
        # Action pattern stats: {(action_type, app_class): {success: N, ...}}
        self._action_stats: Dict[Tuple[str, str], Dict[str, int]] = {}

        self._load()
        print(f"[ActionOutcomeTracker] Loaded {len(self._step_log)} step records, "
              f"{len(self._objective_log)} objectives")

    # ═══════════════════════  Step-Level Recording  ═══════════════════════

    def record_step_outcome(
        self,
        action_type: str,
        target_label: str,
        app_class: str,
        app_instance: str,
        site: str,
        outcome: OutcomeLevel,
        category: str = "",
        coords: Optional[List[int]] = None,
        latency_ms: float = 0.0,
        resolution_tier: str = "",
        cv_confidence: float = 0.0,
        edge_cid: str = "",
        context: Optional[Dict[str, Any]] = None,
        deferred: bool = False,
    ) -> OutcomeRecord:
        """Records a single step outcome with full metadata."""
        record = OutcomeRecord(
            timestamp=time.time(),
            action_type=action_type,
            target_label=target_label,
            app_class=app_class,
            app_instance=app_instance,
            site=site,
            outcome=str(outcome.value), # type: ignore
            category=category,
            coords=coords,
            latency_ms=latency_ms,
            resolution_tier=resolution_tier,
            cv_confidence=cv_confidence,
            edge_cid=edge_cid,
            context=context or {},
            deferred=deferred,
        )

        self._step_log.append(record)

        # Update running stats
        key = (target_label.lower(), app_class)
        if key not in self._target_stats:
            self._target_stats[key] = {"success": 0, "semi_success": 0, "failed": 0}
        
        outcome_val = str(outcome.value).lower()
        if outcome_val in self._target_stats[key]:
            self._target_stats[key][outcome_val] += 1
        
        akey = (action_type, app_class)
        if akey not in self._action_stats:
            self._action_stats[akey] = {"success": 0, "semi_success": 0, "failed": 0}
        if outcome_val in self._action_stats[akey]:
            self._action_stats[akey][outcome_val] += 1

        status_abbr = {"SUCCESS": "[OK]", "SEMI_SUCCESS": "[~]", "FAILED": "[X]"}
        icon = status_abbr.get(str(outcome.value), "[?]")
        print(f"[OutcomeTracker] {icon} {str(outcome.value)}: {action_type} -> '{target_label}' "
              f"[{app_class}/{app_instance}] cat={category}")

        self._save()
        return record

    # ═══════════════════════  Objective-Level Recording  ═══════════════════════

    def record_objective_outcome(
        self,
        objective: str,
        app_class: str,
        app_instance: str,
        steps: List[OutcomeRecord],
        overall_outcome: OutcomeLevel,
    ) -> ObjectiveOutcome:
        """Records the outcome of a full objective/task."""
        obj = ObjectiveOutcome(
            objective_id=f"obj_{int(time.time())}_{len(self._objective_log)}",
            objective=objective,
            app_class=app_class,
            app_instance=app_instance,
            outcome=str(overall_outcome.value) if hasattr(overall_outcome, "value") else str(overall_outcome), # type: ignore
            steps=steps,
            total_steps=len(steps),
            successful_steps=sum(1 for s in steps if s.outcome == "SUCCESS"),
            semi_steps=sum(1 for s in steps if s.outcome == "SEMI_SUCCESS"),
            failed_steps=sum(1 for s in steps if s.outcome == "FAILED"),
            total_duration_ms=sum(s.latency_ms for s in steps),
            timestamp=time.time(),
        )

        self._objective_log.append(obj)
        print(f"[OutcomeTracker] Objective '{(objective or '')[:40]}': {overall_outcome.value} " # type: ignore
              f"({obj.successful_steps} OK, {obj.semi_steps} SEMI, {obj.failed_steps} FAIL)")

        self._save()
        return obj

    # ═══════════════════════  Stats Queries  ═══════════════════════

    def get_target_success_rate(self, target_label: str, app_class: str = "") -> Dict:
        """Returns success rate for a specific target element."""
        key = (target_label.lower(), app_class)
        stats = self._target_stats.get(key, {"success": 0, "semi_success": 0, "failed": 0})
        total = sum(stats.values())
        return {
            **stats,
            "total": total,
            "success_rate": round(stats["success"] / max(total, 1), 3), # type: ignore
            "reliability": round((stats["success"] + 0.5 * stats["semi_success"]) / max(total, 1), 3), # type: ignore
        }

    def get_action_pattern_stats(self, action_type: str, app_class: str = "") -> Dict:
        """Returns stats for a specific action pattern."""
        key = (action_type, app_class)
        stats = self._action_stats.get(key, {"success": 0, "semi_success": 0, "failed": 0})
        total = sum(stats.values())
        return {**stats, "total": total,
                "success_rate": round(stats["success"] / max(total, 1), 3)} # type: ignore

    def get_failure_patterns(self, app_class: str = "", top_k: int = 5) -> List[Dict]:
        """Returns most common failure patterns."""
        fail_counts: Dict[str, int] = {}
        for rec in self._step_log:
            if rec.outcome == "FAILED" and (not app_class or rec.app_class == app_class):
                pattern = f"{rec.action_type}->{rec.target_label}({rec.category})"
                fail_counts[pattern] = fail_counts.get(pattern, 0) + 1

        sorted_fails = sorted(fail_counts.items(), key=lambda x: x[1], reverse=True)
        return [{"pattern": p, "count": c} for p, c in sorted_fails[:top_k]] # type: ignore

    def get_improvement_suggestions(self, target_label: str, app_class: str = "") -> List[str]:
        """Data-driven suggestions to improve success on a target."""
        suggestions = []
        stats = self.get_target_success_rate(target_label, app_class)

        if stats["success_rate"] < 0.5 and stats["total"] > 3:
            suggestions.append(f"Low success rate ({stats['success_rate']:.0%}) — consider using closeup zoom")

        # Check for common failure categories
        recent_fails = [r for r in self._step_log[-20:] # type: ignore
                       if (r.target_label or "").lower() == (target_label or "").lower()
                       and r.outcome == "FAILED"]
        categories = [r.category for r in recent_fails]

        if categories.count("WRONG_INSTANCE") >= 2:
            suggestions.append("Multiple WRONG_INSTANCE failures — disambiguate with spatial context")
        if categories.count("STALE_COORDINATES") >= 2:
            suggestions.append("Coordinates going stale — re-detect before each attempt")
        if categories.count("NO_EFFECT") >= 2:
            suggestions.append("Clicks having no effect — try edge correlation CID lookup")

        return suggestions

    # ═══════════════════════  Persistence  ═══════════════════════

    def _save(self):
        """Persists outcome logs to disk."""
        data = {
            "step_log": [asdict(r) for r in self._step_log[-500:]], # type: ignore
            "objective_log": [asdict(o) for o in self._objective_log[-100:]], # type: ignore
        }
        path = os.path.join(self.storage_dir, "outcomes.json")
        try:
            with open(path, "w") as f:
                json.dump(data, f, indent=2, default=str)
        except Exception as e:
            print(f"[OutcomeTracker] Save failed: {e}")

    def _load(self):
        """Loads persisted outcome logs."""
        path = os.path.join(self.storage_dir, "outcomes.json")
        if not os.path.exists(path):
            return
        try:
            with open(path, "r") as f:
                data = json.load(f)
            
            step_log_data = data.get("step_log", [])
            for rec in step_log_data:
                # Explicit construction to avoid unpacking lints
                outcome_str = str(rec.get("outcome", "FAILED"))
                target_label = str(rec.get("target_label", ""))
                app_class = str(rec.get("app_class", ""))
                
                record = OutcomeRecord(
                    timestamp=float(rec.get("timestamp", 0.0)),
                    action_type=str(rec.get("action_type", "")),
                    target_label=target_label,
                    app_class=app_class,
                    app_instance=str(rec.get("app_instance", "")),
                    site=str(rec.get("site", "")),
                    outcome=outcome_str,
                    category=str(rec.get("category", "")),
                    coords=rec.get("coords"),
                    latency_ms=float(rec.get("latency_ms", 0.0)),
                    resolution_tier=str(rec.get("resolution_tier", "")),
                    cv_confidence=float(rec.get("cv_confidence", 0.0)),
                    edge_cid=str(rec.get("edge_cid", "")),
                    context=rec.get("context", {}),
                    deferred=bool(rec.get("deferred", False)),
                    deferred_final_outcome=rec.get("deferred_final_outcome")
                )
                self._step_log.append(record)
                
                key = (target_label.lower(), app_class)
                if key not in self._target_stats:
                    self._target_stats[key] = {"success": 0, "semi_success": 0, "failed": 0}
                self._target_stats[key][outcome_str.lower()] += 1
                
            obj_log_data = data.get("objective_log", [])
            for obj in obj_log_data:
                steps_data = obj.get("steps", [])
                steps = []
                for s in steps_data:
                    steps.append(OutcomeRecord(
                        timestamp=float(s.get("timestamp", 0.0)),
                        action_type=str(s.get("action_type", "")),
                        target_label=str(s.get("target_label", "")),
                        app_class=str(s.get("app_class", "")),
                        app_instance=str(s.get("app_instance", "")),
                        site=str(s.get("site", "")),
                        outcome=str(s.get("outcome", "FAILED")),
                        category=str(s.get("category", "")),
                        coords=s.get("coords"),
                        latency_ms=float(s.get("latency_ms", 0.0)),
                        context=s.get("context", {}),
                        deferred=bool(s.get("deferred", False))
                    ))
                
                objective_record = ObjectiveOutcome(
                    objective_id=str(obj.get("objective_id", "")),
                    objective=str(obj.get("objective", "")),
                    app_class=str(obj.get("app_class", "")),
                    app_instance=str(obj.get("app_instance", "")),
                    outcome=str(obj.get("outcome", "FAILED")),
                    steps=steps,
                    total_steps=int(obj.get("total_steps", 0)),
                    successful_steps=int(obj.get("successful_steps", 0)),
                    semi_steps=int(obj.get("semi_steps", 0)),
                    failed_steps=int(obj.get("failed_steps", 0)),
                    total_duration_ms=float(obj.get("total_duration_ms", 0.0)),
                    timestamp=float(obj.get("timestamp", 0.0))
                )
                self._objective_log.append(objective_record)
        except Exception as e:
            print(f"[OutcomeTracker] Load failed: {e}")
