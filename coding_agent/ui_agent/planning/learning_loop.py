import time
from typing import Dict, Any, Optional, List

try:
    from coding_agent.ui_agent.planning.action_outcome_tracker import ActionOutcomeTracker, OutcomeLevel, FailureCategory, SemiSuccessCategory  # type: ignore
    from coding_agent.ui_agent.planning.presumption_engine import PresumptionEngine  # type: ignore
    from coding_agent.ui_agent.planning.postponed_judgement import PostponedJudgement  # type: ignore
    from coding_agent.ui_agent.perception.anticipatory_observer import AnticipatoryObserver  # type: ignore
except ImportError:
    ActionOutcomeTracker = None
    PresumptionEngine = None
    PostponedJudgement = None
    AnticipatoryObserver = None


class LearningLoop:
    """
    Evaluates action success with 3-tier outcomes (SUCCESS / SEMI_SUCCESS / FAILED),
    performs structured failure analysis, builds presumptions from successes,
    and supports postponed judgement for multi-step sequences.
    """
    def __init__(self, vision_analyzer, action_predictor, experiential_memory=None,
                 cv_pipeline=None, outcome_tracker=None, presumption_engine=None,
                 postponed_judgement=None, vlm_bridge_callback=None):
        self.vision = vision_analyzer
        self.predictor = action_predictor
        self.memory = experiential_memory
        self.cv = cv_pipeline
        self.action_history: List[Dict[str, Any]] = []
        self._current_episode_steps: List[Dict[str, Any]] = []
        self._step_counter = 0
        self.vlm_bridge_callback = vlm_bridge_callback
        self.loop_incidents: List[Dict[str, Any]] = [] # Added type hint for loop_incidents

        # New Phase 13 modules
        self.outcome_tracker = outcome_tracker
        self.presumption_engine = presumption_engine
        self.postponed = postponed_judgement
        
        if AnticipatoryObserver:
            self.anticipatory_observer = AnticipatoryObserver(memory_agent=self.memory, cv_pipeline=self.cv)
        else:
            self.anticipatory_observer = None

    def evaluate_action_success(self, action: Dict[str, Any], state_before: str, state_after: str,
                                 app_class: str = "", app_instance: str = "",
                                 site: str = "", task: str = "",
                                 total_steps_estimate: int = 10) -> bool:
        """
        3-tier evaluation: SUCCESS / SEMI_SUCCESS / FAILED.
        Feeds structured outcomes into ActionOutcomeTracker and PresumptionEngine.
        Returns True for SUCCESS/SEMI_SUCCESS, False for FAILED (backward-compatible).
        """
        print(f"[LearningLoop] Evaluating: {action.get('action', 'unknown')} -> '{action.get('target', '')}'")

        # --- NEW: Anticipatory Fast Validation ---
        fast_success = False
        bridge_insight = None
        expected_obs = action.get("expected_observable")
        
        visual_change_pct = 0.0
        dialog_opened = False
        menu_opened = False
        
        if expected_obs and self.anticipatory_observer and self.cv:
            print(f"[LearningLoop] Found expected_observable: {expected_obs}")
            try:
                import cv2  # type: ignore
                import numpy as np  # type: ignore
                fb = cv2.imread(state_before) if isinstance(state_before, str) else None
                fa = cv2.imread(state_after) if isinstance(state_after, str) else None
                
                if fb is not None and fa is not None:
                    res = self.anticipatory_observer.verify_observable(expected_obs, fb, fa, action)
                    
                    if res.get("success"):
                        print(f"[LearningLoop] Fast check PASSED ({res.get('method')}).")
                        fast_success = True
                    elif res.get("request_fidelity") == "HIGH" and self.vlm_bridge_callback:
                        print(f"[LearningLoop] Upgrade requested: {res.get('reason')}")
                        bridge_insight = self.vlm_bridge_callback(res.get("reason"), fa if isinstance(state_after, str) else None)
                        if bridge_insight:
                            # Heuristic: if bridge gives a definitive "Yes" or similar
                            if "success" in bridge_insight.lower() or "completed" in bridge_insight.lower():
                                fast_success = True
                                consequence_reason = f"VLM Bridge Insight: {bridge_insight}"
            except Exception as e:
                print(f"[LearningLoop] Anticipatory observer error: {e}")

        if fast_success:
            success = True
            if not locals().get("consequence_reason"):
                consequence_reason = "Anticipatory visual check passed instantly."
            visual_change_pct = 10.0 # Pass threshold
        else:
            # Fallback to standard VLM
            success, consequence_reason = self.vision.visual_diff(state_before, state_after, action_context=action)
            # ... (CV change detection logic remains the same)
            if self.cv:
                try:
                    import cv2  # type: ignore
                    import numpy as np  # type: ignore
                    fb = cv2.imread(state_before) if isinstance(state_before, str) else None
                    fa = cv2.imread(state_after) if isinstance(state_after, str) else None
                    if fb is not None and fa is not None:
                        diff = cv2.absdiff(
                            cv2.cvtColor(fb, cv2.COLOR_BGR2GRAY),
                            cv2.cvtColor(fa, cv2.COLOR_BGR2GRAY)
                        )
                        visual_change_pct = float((np.sum(diff > 20) / diff.size) * 100)
                        h, w = diff.shape
                        center_region = diff[h//4:3*h//4, w//4:3*w//4]
                        center_change = float((np.sum(center_region > 30) / center_region.size) * 100)
                        if center_change > 20 and visual_change_pct < 60:
                            dialog_opened = True
                except Exception:
                    pass

        outcome = OutcomeLevel.FAILED if OutcomeLevel else None
        category = ""

        # Classify outcome
        if OutcomeLevel:
            if success and visual_change_pct > 5:
                outcome = OutcomeLevel.SUCCESS
                category = ""
            elif not success and visual_change_pct < 1.0:
                outcome = OutcomeLevel.FAILED
                category = FailureCategory.NO_EFFECT.value if FailureCategory else "NO_EFFECT"
            elif visual_change_pct >= 1.0 and visual_change_pct < 30.0:
                outcome = OutcomeLevel.SEMI_SUCCESS
                if dialog_opened:
                    category = SemiSuccessCategory.DIALOG_TRIGGERED.value if SemiSuccessCategory else "DIALOG_TRIGGERED"
                elif action.get('action') == 'type':
                    category = SemiSuccessCategory.INPUT_ACCEPTED.value if SemiSuccessCategory else "INPUT_ACCEPTED"
                else:
                    category = SemiSuccessCategory.STATE_CHANGED.value if SemiSuccessCategory else "STATE_CHANGED"
            elif not success:
                outcome = OutcomeLevel.FAILED
                failure_info = self._analyze_failure(action, state_before, state_after)
                category = failure_info.get("root_cause", "UNKNOWN")
                action["failure_info"] = failure_info
            else:
                outcome = OutcomeLevel.SUCCESS
                category = ""

        outcome_label = outcome.value if outcome else ("SUCCESS" if success else "FAILED")
        status_abbr = {"SUCCESS": "[OK]", "SEMI_SUCCESS": "[~]", "FAILED": "[X]"}
        print(f"[LearningLoop] {status_abbr.get(outcome_label, '[?]')} {outcome_label} "
              f"(change={visual_change_pct:.1f}%, cat={category})")

        # Check postponed judgement
        self._step_counter += 1
        step_id = f"step_{self._step_counter}_{int(time.time())}"

        if self.postponed and outcome:
            should_defer, defer_reason = self.postponed.should_defer(
                action.get('action', ''), self._step_counter,
                total_steps_estimate, visual_change_pct,
                dialog_opened, menu_opened,
            )
            if should_defer:
                self.postponed.defer_judgement(step_id, action.get('action', ''),
                                               action.get('target', ''), defer_reason)

        # Record to ActionOutcomeTracker
        if self.outcome_tracker and outcome:
            # The following lines seem to be misplaced or part of a different context
            # best_id = max(hyp_ids, key=lambda hid: self.hypotheses.get(hid, Hypothesis("","","","")).confidence)
            # self.promote_to_global(best_id)
            # promoted_count += 1
            self.outcome_tracker.record_step_outcome(
                action_type=action.get("action", ""),
                target_label=action.get("target", ""),
                app_class=app_class,
                app_instance=app_instance,
                site=site,
                outcome=outcome,
                category=category,
                coords=action.get("resolved_coords") or action.get("hint_coords"),
                resolution_tier=action.get("resolution_tier", ""),
                cv_confidence=action.get("cv_confidence", 0.0),
                edge_cid=action.get("edge_cid", ""),
                deferred=bool(self.postponed and self.postponed._pending.get(step_id)),
            )

        # Build presumption on SUCCESS
        if self.presumption_engine and outcome == OutcomeLevel.SUCCESS:
            coords = action.get("resolved_coords") or action.get("hint_coords")
            
            coord_tuple = None
            if isinstance(coords, dict) and "x" in coords and "y" in coords:
                coord_tuple = (coords["x"], coords["y"])
            elif isinstance(coords, (list, tuple)) and len(coords) >= 2:
                coord_tuple = tuple(coords[:2])
                
            if coord_tuple:
                self.presumption_engine.build_presumption(
                    target_label=action.get("target", ""),
                    coords=coord_tuple,
                    element_type=action.get("element_type", "UNKNOWN"),
                    app_class=app_class,
                    app_instance=app_instance,
                    site=site,
                    edge_cid=action.get("edge_cid", ""),
                )

        # Record outcome to predictor (backward compat)
        self.predictor.record_outcome(action, success)

        # Build step record
        import time as _time
        step_record = {
            "action": action.get("action", ""),
            "target": action.get("target", ""),
            "method": action.get("method", ""),
            "success": success,
            "outcome": outcome_label,
            "category": category,
            "visual_change_pct": round(visual_change_pct, 2),  # type: ignore
            "timestamp": _time.time(),
        }
        if not success and "failure_info" in action:
            step_record["failure_info"] = action["failure_info"]
        self._current_episode_steps.append(step_record)

        self.action_history.append({
            "action": action, "success": success,
            "outcome": outcome_label,
            "state_before": state_before, "state_after": state_after
        })

        return success or (outcome == OutcomeLevel.SEMI_SUCCESS if outcome else False)

    def finalize_episode(self, task: str, app_name: str, app_class: str, overall_success: bool,
                         failure_reason: str = ""):
        """Finalizes the current episode, resolves deferred judgements, and records everything."""
        # Resolve deferred judgements
        if self.postponed:
            obj_outcome = "SUCCESS" if overall_success else "FAILED"
            self.postponed.auto_resolve_by_objective(obj_outcome)

        if self.memory and self._current_episode_steps:
            self.memory.record_episode(
                task=task, app_name=app_name, app_class=app_class,
                steps=self._current_episode_steps,
                success=overall_success, failure_reason=failure_reason,
            )
            # Phase 13.5: Also record in outcome tracker
            if self.outcome_tracker:
                obj_outcome = OutcomeLevel.SUCCESS if overall_success else OutcomeLevel.FAILED
                self.outcome_tracker.record_objective_outcome(
                    objective=task,
                    app_class=app_class,
                    app_instance="", # Instance not strictly required for aggregate obj stats
                    steps=[], # We don't necessarily need to duplicate the full step list here for aggregation
                    overall_outcome=obj_outcome
                )
            
            print(f"[LearningLoop] Episode recorded: {'SUCCESS' if overall_success else 'FAILED'} "
                  f"({len(self._current_episode_steps)} steps)")

        self._current_episode_steps = []
        self._step_counter = 0

    def _analyze_failure(self, action: Dict[str, Any], state_before: str, state_after: str) -> Dict[str, Any]:
        """Structured failure root-cause analysis with wrong-instance detection."""
        failure_info = {
            "root_cause": "UNKNOWN",
            "details": "",
        }

        # --- NEW: Wrong-Instance Detection ---
        # Check if we've failed on the same TARGET LABEL before at similar coordinates
        current_target = action.get("target", "")
        current_coords = action.get("resolved_coords", action.get("hint_coords"))
        
        if current_target:
            same_label_failures = 0
            history_slice = list(self.action_history)[-6:]  # type: ignore
            for h in history_slice:
                if (not h["success"] and 
                    h["action"].get("target", "") == current_target):
                    same_label_failures = int(same_label_failures) + 1  # type: ignore
            
            if int(same_label_failures) >= 2:  # type: ignore
                failure_info["root_cause"] = "WRONG_INSTANCE"
                failure_info["details"] = (
                    f"You have failed {int(same_label_failures) + 1} times on '{current_target}'. "  # type: ignore
                    f"You are likely clicking the WRONG INSTANCE of this element. "
                    f"There may be multiple elements with the label '{current_target}' on screen "
                    f"(e.g., one on the page and one inside a dialog/modal). "
                    f"STRATEGY: Look for '{current_target}' at a COMPLETELY DIFFERENT location "
                    f"on screen, or use 'press Enter' as an alternative to clicking submit buttons."
                )
                return failure_info

        # If we have CV pipeline and frames, do visual analysis
        if self.cv:
            try:
                import cv2  # type: ignore
                import numpy as np  # type: ignore
                
                frame_before = cv2.imread(state_before) if isinstance(state_before, str) else None
                frame_after = cv2.imread(state_after) if isinstance(state_after, str) else None
                
                if frame_before is not None and frame_after is not None:
                    # Check if screens are nearly identical (action had no effect)
                    diff = cv2.absdiff(
                        cv2.cvtColor(frame_before, cv2.COLOR_BGR2GRAY),
                        cv2.cvtColor(frame_after, cv2.COLOR_BGR2GRAY)
                    )
                    change_pct = (np.sum(diff > 20) / diff.size) * 100
                    
                    if change_pct < 1.0:
                        failure_info["root_cause"] = "NO_EFFECT"
                        failure_info["details"] = f"Screen unchanged ({change_pct:.1f}% change). Click missed target. STRATEGY: Use 'request_closeup_zoom' next time for better precision."
                    elif change_pct > 50:
                        failure_info["root_cause"] = "UNEXPECTED_STATE_CHANGE"
                        failure_info["details"] = f"Major screen change ({change_pct:.1f}%). Possible dialog/error appeared."
                    else:
                        failure_info["root_cause"] = "WRONG_TARGET"
                        failure_info["details"] = f"Partial change ({change_pct:.1f}%). Clicked wrong element. STRATEGY: Verify target, or use 'request_closeup_zoom'."
            except Exception as e:
                failure_info["details"] = f"CV analysis failed: {e}"

        return failure_info

    def get_recent_failures(self, n: int = 5) -> List[Dict[str, Any]]:
        """Returns the most recent failure records."""
        failures = [h for h in self.action_history if not h["success"]]
        return failures[max(0, len(failures) - n):]  # type: ignore

