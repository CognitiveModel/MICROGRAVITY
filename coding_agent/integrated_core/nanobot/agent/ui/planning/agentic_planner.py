"""
AgenticPlanner - Step-by-step agentic UI automation with dynamic Gemini feedback.

Instead of planning all steps upfront, this planner operates in a true
observe → decide → act → verify loop where Gemini sees the screen after
every step and dynamically decides what to do next.
"""

import os
import json
import time
from typing import Dict, Any, Optional, List
from PIL import Image # type: ignore
from google import genai # type: ignore
from google.genai import types # type: ignore
from nanobot.agent.ui.planning.loop_sentinel import LoopSentinel, LoopSeverity, LoopRecoveryEngine # type: ignore


class AgenticPlanner:
    """
    Replaces the static GoalManager with a step-by-step agentic loop.
    
    Each iteration:
      1. OBSERVE: Capture current screen state
      2. DECIDE: Send screenshot + context to Gemini → get ONE next action
      3. ACT: Execute that action (done by UIAgent)
      4. VERIFY: Capture after-state, send verification prompt
      5. RECORD: Update step history with result
      6. LOOP: If goal not complete, repeat
    """

    # Dynamic prompt that steers Gemini to think step-by-step
    SYSTEM_PROMPT = """You are an autonomous UI automation agent controlling a Windows desktop.
You can see the screen and must decide the SINGLE NEXT ACTION to perform.

Available actions:
- {"action": "click", "target": "<description>", "hint_coords": [y_center_norm, x_center_norm], "needs_zoom": false}
- {"action": "double_click", "target": "<description>", "hint_coords": [y_norm, x_norm]}
- {"action": "type", "text": "<text to type>", "clear_first": true}  # clear_first=true sends Ctrl+A before typing to replace existing text
- {"action": "press", "key": "<key name like enter, tab, escape, f5>"}
- {"action": "hotkey", "keys": ["ctrl", "t"]}  # e.g. New Tab
- {"action": "hotkey", "keys": ["ctrl", "w"]}  # e.g. Close Tab
- {"action": "hotkey", "keys": ["ctrl", "tab"]} # e.g. Switch Tab
- {"action": "scroll", "direction": "up|down", "amount": 300}
- {"action": "right_click", "target": "<description>", "hint_coords": [y_norm, x_norm], "needs_zoom": false}  # Right-click to open context menus (e.g. on taskbar icons)
- {"action": "minimize_window", "target": "<window title keyword>"}  # Minimize a window by title match
- {"action": "focus_window", "hwnd": <HWND_INT>}
- {"action": "remember_window", "hwnd": <HWND_INT>, "tag": "<custom label, e.g. 'Work Account'>"} # Persistently tag a window so you remember which is which
- {"action": "request_closeup_zoom", "target": "<description of tiny element>"}
- {"action": "delegate_to_browser_tool", "objective": "<description of what the browser tool should do>"}
- {"action": "wait", "duration": 1.5, "reason": "<why waiting>"}
- {"action": "ask_swarm", "question": "<specific question to the main Swarm agent about missing context (e.g., credentials, target info)>"}

### COMPOUND ACTIONS (for batching related steps):
- {"action": "compound", "reasoning": "Filling in the login form", "steps": [
    {"action": "click", "target": "username input field", "hint_coords": [y, x]},
    {"action": "type", "text": "myusername", "clear_first": true},
    {"action": "press", "key": "tab"},
    {"action": "type", "text": "mypassword", "clear_first": true},
    {"action": "press", "key": "enter"}
  ]}
Use compound actions for FORM FILLING, LOGIN FLOWS, and SEQUENTIAL INTERACTIONS where intermediate verification is unnecessary.
Verification only runs ONCE at the END of a compound action, NOT between sub-steps.

### CV Analysis Tools (use these to inspect UI elements before acting):
- {"action": "cv_template_match", "target_label": "<element name>", "threshold": 0.80}
- {"action": "cv_snip_element", "element_label": "<name>", "bbox": [x,y,w,h]}
- {"action": "cv_fingerprint_compare", "element_label": "<name>"}
- {"action": "cv_stability_check", "region": [x,y,w,h]}
- {"action": "cv_embedding_search", "target_description": "<what element looks like>"}
- {"action": "request_closeup_zoom", "target": "<description of tiny element>"}

### Edge Correlation Tools (use for complex UIs or learning new apps):
- {"action": "cv_edge_detect", "annotate": true}
- {"action": "cv_structural_map"}
- {"action": "cv_find_by_cid", "cid": "<CID_xxxxxxxxxxxx>"}

### CRITICAL: Element Disambiguation
When multiple elements on screen share the SAME LABEL (e.g., two buttons both labeled "Log In"):
1. **Always specify spatial context** in your target description: e.g. "Log In button inside the login dialog" vs "Log In button on the top navigation bar".
2. **After opening a dialog/modal**, the submit/confirm button is INSIDE the dialog, NOT behind it on the page.
3. **Never reuse coordinates** from an element you interacted with BEFORE a dialog appeared — dialog elements have DIFFERENT coordinates.
4. If a click on a button "opens a dialog" instead of "submitting a form", you clicked the WRONG instance. Look for the button INSIDE the dialog.

### CRITICAL: Modal/Dialog Focus
When a popup, dialog, or overlay is visible on screen:
- ONLY interact with elements INSIDE the dialog/modal.
- Elements behind the dialog are NOT clickable and will cause failures.
- Look for submit buttons, close buttons, and input fields WITHIN the dialog boundary.
- If you need to dismiss the dialog, click its close button or press Escape.

### CRITICAL: Form Filling Intelligence
When you encounter a LOGIN FORM, SIGNUP FORM, or any multi-field form:
1. **ALWAYS use a compound action** to fill the entire form at once.
2. **ALWAYS set clear_first: true** when typing into input fields to replace any existing text.
3. **Use Tab to move between fields** instead of clicking the next field separately.
4. **If the submit button is not visible**, scroll down slightly FIRST, then submit.
5. **NEVER type both username and password into the same field**. Use Tab to switch fields.
6. **Do NOT individually verify each click/type step** — chain them in a compound action.

### Taskbar Navigation & Multi-Instance Strategy:
1. **Taskbar Location**: The Windows taskbar is typically at the BOTTOM of the screen. Check ENVIRONMENT AWARENESS for exact taskbar rect.
2. **Right-Click Context Menus**: To open a NEW window of an app that's already running, RIGHT-CLICK its taskbar icon and select "New window" from the context menu. This is the PREFERRED way to open additional instances.
3. **Multi-Account Intelligence**: When a browser window is already open:
   a. READ the window title — it usually shows the logged-in account/profile name (e.g. "Gmail - user@gmail.com - Google Chrome")
   b. If the title shows a DIFFERENT account than needed, DO NOT use that window
   c. Instead: MINIMIZE that window (using `minimize_window`), then right-click the taskbar icon and select "New window"
   d. If the window title matches the needed account, FOCUS it and proceed in-place
   e. If dealing with multiple identical windows (e.g. two "Notepad" windows), use `"action": "remember_window"` to tag an `hwnd` with a custom label (e.g. "Draft") so you don't confuse them later.
4. **Icon Identification REQUIRED Keyword**: You MUST include the exact word "taskbar" in your `target` description when clicking these icons (e.g., `"target": "Chrome icon on the taskbar"`).
5. **Context Menu Navigation**: After right-clicking a taskbar icon, a small popup menu appears. Click the desired option (e.g. "New window").
6. **Collapsing Windows**: If a window is blocking your view, use `minimize_window` to collapse it before proceeding.

### Tab Management & Multi-Window Philosophy:
1. **In-Place Execution**: If the application needed for the goal is ALREADY OPEN and visible on screen, DO NOT try to open it again.
2. **Multi-Instance Handling**: Use `focus_window` with the specific `hwnd` from ENVIRONMENT AWARENESS.
3. **Tab Discovery**: Use `ctrl+tab` or click on tab titles to scan all open options.
4. **Proactive Discovery**: Mention interesting features or options in your reasoning.
5. **Tab Lifecycle**: Open new tabs (`ctrl+t`) or close unnecessary ones (`ctrl+w`).

### Advanced Automation Artifacts (for complex gestures):
- {"action": "click_and_drag", "start": [x1, y1], "end": [x2, y2], "duration_ms": 500, "curve": "linear|arc|bezier"}
- {"action": "drag_to_resize", "edge": "left|right|top|bottom|top-right|bottom-left", "delta_px": 200}
- {"action": "nested_menu_navigate", "menu_path": ["File", "Export", "PDF"], "root_coords": [x, y]}
- {"action": "scroll_until_visible", "target": "<element description>", "direction": "down|up", "max_scrolls": 10}
- {"action": "variable_rate_drag", "start": [x1, y1], "end": [x2, y2], "profile": "ease_in|ease_out|ease_in_out", "total_ms": 800}
Use these for complex gestures like resizing windows, drag-and-drop, navigating nested menus, or scrolling to find elements.

### Interaction Patterns (IP) & Learned Knowledge: Tiered insights from experiential memory (SIMPLE, COMPOUND, COMPLEX, NAVIGATION).
- Structural Patterns: Identified recurring UI clusters (Modals, Grids, Sidebars) that may be templateable artifacts.
- Navigation Routes: Learned paths to sub-sections or specific app states.
1. **Follow Proven Processes:** If the 'LEARNED KNOWLEDGE' section suggests a 'Reusable process' for your current task, try to follow its sequence of steps. They are proven to work.
2. **Observe Hypotheses:** Pay attention to 'Known behaviors' (e.g. IF I click X, THEN Y happens). Use these to predict the outcome of your actions.
3. **Handle Nuances:** If 'Critical nuances' are listed (e.g. "Dialog X takes 5s to load"), adjust your timing/strategy accordingly.

### Structural UI Artifacts
If "Recurring UI Patterns" are detected:
1. Treat them as high-level "Modals" or "Panels".
2. If a pattern matches a known "Modal", prioritize interacting with its internal elements (Submit, Cancel, Close).
3. Recognize "ADAPTIVE" patterns might change layout based on state; use "DYNAMIC" matchers for these.

### Spatial Anchoring & Relational Targeting:
1. **Adjacent Icons (CRITICAL FOR TASKBAR):** If two icons look similar and are close together (e.g., Edge vs Chrome on taskbar), ALWAYS use a relational anchor. NEVER just say "Chrome icon". You MUST say "The Chrome icon precisely to the right of the Firefox icon on the taskbar" or "The left-most Chrome icon on the taskbar".
2. **Hover Disambiguation:** If unsure which grouped icon is which, use the `hover` action first to reveal the preview peek, then visually verify before clicking.
3. **Relative Search:** Use relative terms like "top-left", "right-most", "inside active window", "in the system tray".
4. **Disambiguation:** If a previous click landed on the wrong app (e.g., clicked Edge instead of Chrome), re-target using a unique visual neighbor: "Click the Chrome icon located exactly between X and Y".

### Precision Targeting (Zoom & CV):
1. **Small Targets:** If the target element is SMALL (window buttons, tiny icons, browser tabs, checkboxes), ALWAYS set `"needs_zoom": true`. This triggers a high-precision two-pass CV analysis.
2. **CV Fast-Match:** You can now proactively use `cv_template_match` if you see a 'VLM Insight' from a previous step that suggests a strong CV candidate.
3. **Speculative Snipping:** Successful clicks are now automatically saved as templates. If you interact with an element successfully, its location will be matched nearly instantly in future steps.

Coordinate format: hint_coords are [y, x] normalized to 0-1000 range.
If the target element is SMALL (tabs, icons, small buttons), set "needs_zoom": true.
If a click fails, the UI Agent will automatically attempt a high-precision Zoom Retry. Avoid repetitive clicks on the same pixel if it failed; instead, describe the target differently or request an explicit zoom.

### Anticipatory Observables (Fast Feedback)
To bypass slow visual verification, predict the immediate visual change your action will cause using `expected_observable`:
- `{"type": "color_shift", "target_region": "local"}`: For button clicks that cause hover/active color changes.
- `{"type": "structural_change", "threshold": 0.15}`: For navigation clicks that load a new page.
- `{"type": "element_appears", "template_label": "..."}`: For actions that open a specific known modal or dropdown.

Your response MUST be a single JSON object with these fields:
{
  "reasoning": "<1-2 sentence explanation>",
  "action": "<action type>",
  "target": "<element description if click>",
  "hint_coords": [y_norm, x_norm],
  "needs_zoom": false,
  "expected_observable": null,
  "goal_complete": false,
  ... other action-specific fields
}

> [!CAUTION] STRICT GOAL VALIDATION
> You must ONLY set `"goal_complete": true` if you visually verify the EXACT task request has been fulfilled on screen.
> Do NOT assume an action (like "clicking calculator icon") succeeded automatically. Wait for the next frame to verify.

If you believe the goal cannot be achieved after multiple adaptive attempts, set "goal_failed": true with a "failure_reason".
"""

    MAX_STEPS = 30  # Safety limit to prevent infinite loops
    MAX_RETRIES_PER_STEP = 2

    def __init__(self, model_name: str = "models/gemini-2.5-flash", situational_awareness=None,
                 experiential_memory=None, decision_manager=None,
                 presumption_engine=None, outcome_tracker=None):
        from nanobot.config.loader import load_config # type: ignore
        self.model_name = model_name
        config = load_config()
        api_key = config.providers.gemini.api_key
        if not api_key:
            raise ValueError("GEMINI_API_KEY not configured in config.json")
        self.client = genai.Client(api_key=api_key) # type: ignore
        
        self.goal: Optional[str] = None
        self.step_history: List[Dict[str, Any]] = []
        self.current_step: int = 0
        self._is_complete: bool = False
        self._is_failed: bool = False
        self._failure_reason: str = ""
        self.last_action_coords: Optional[Dict[str, Any]] = None
        self.consecutive_failures: int = 0
        self.tab_inventory: Dict[str, str] = {}
        self.window_memory: Dict[int, str] = {}  # Store custom tags for specific hwnds
        
        # Loop Sentinel — sub-objective tracking and loop detection
        self.sentinel = LoopSentinel()
        
        # Loop Recovery Engine — autonomous root-cause analysis and recovery
        self.recovery_engine = LoopRecoveryEngine(
            sentinel=self.sentinel,
            experiential_memory=experiential_memory,
        )
        self._window_state: dict = None  # type: ignore  # Injected by UIAgent before each step
        self._last_recovery_plan = None  # Track for outcome recording
        
        # Awareness stack hooks
        self.situational_awareness = situational_awareness  # SituationalAwareness instance
        self.experiential_memory = experiential_memory      # ExperientialMemory instance
        self.decision_manager = decision_manager            # DecisionManager instance
        self.presumptions = presumption_engine              # PresumptionEngine instance
        self.outcomes = outcome_tracker                     # ActionOutcomeTracker instance
        self.hitl_interrupts: List[Dict[str, Any]] = []  # Store asynchronous HITL feedback

    def set_goal(self, goal_description: str):
        """Sets the high-level goal for the agent."""
        self.goal = goal_description
        self.step_history = []
        self.current_step = 0
        self._is_complete = False
        self._is_failed = False
        self.hitl_interrupts = []  # Clear interrupts on new goal
        print(f"[AgenticPlanner] Goal set: {goal_description}")

    def inject_hitl_interrupt(self, analysis_data: Dict[str, Any]):
        """Called asynchronously when the user provides mid-task feedback/correction."""
        self.hitl_interrupts.append(analysis_data)
        print(f"[AgenticPlanner] Received HITL interrupt: {analysis_data.get('category')}")

    def is_complete(self) -> bool:
        """Returns True if the goal is achieved or max steps reached."""
        if self.current_step >= self.MAX_STEPS:
            print(f"[AgenticPlanner] Safety limit reached ({self.MAX_STEPS} steps). Stopping.")
            return True
        return self._is_complete or self._is_failed

    def get_status(self) -> str:
        """Returns a human-readable status string."""
        if self._is_complete:
            return "COMPLETE"
        elif self._is_failed:
            return f"FAILED: {self._failure_reason}"
        else:
            return f"Step {self.current_step}/{self.MAX_STEPS}"

    def _build_dynamic_context(self) -> str:
        """
        Builds a dynamic context prompt that includes:
        - The original goal
        - World Model context (from SituationalAwareness)
        - Experiential memory context (hypotheses, processes, nuances)
        - History of completed steps with success/failure
        - Current step number
        """
        context_parts = [f"## GOAL\n{self.goal}\n"]
        
        # Inject World Model context from SituationalAwareness
        if self.situational_awareness:
            try:
                world_context = self.situational_awareness.get_context_for_planner()
                if world_context:
                    context_parts.append("## ENVIRONMENT AWARENESS")
                    context_parts.append(world_context)
                    
                    # --- Phase 10: Inject Explicit Window Boundaries & State ---
                    if hasattr(self.situational_awareness, 'get_current_context'):
                        ctx = self.situational_awareness.get_current_context()
                        if ctx:
                            context_parts.append(f"### Foreground Window State")
                            context_parts.append(f"- Active App: {ctx.get('app_name', 'Unknown')}")
                            context_parts.append(f"- Window Title: {ctx.get('window_title', 'Unknown')}")
                            context_parts.append(f"- Window Rect (Global): {ctx.get('rect', 'Unknown')}")
                            context_parts.append(f"- Is Maximized: {ctx.get('is_maximized', 'Unknown')}")
                    
                    # --- Inject Detected Elements ---
                    model = self.situational_awareness.get_world_model()
                    if model and model.element_map:
                        context_parts.append("\n### Detected Elements (Pixel Coordinates)")
                        for i, el in enumerate(model.element_map[:15]): # type: ignore
                            context_parts.append(f"- [{i}] {el['type']} '{el.get('label', '')}': [x: {el['x']}, y: {el['y']}, w: {el['w']}, h: {el['h']}]")
                    
                    # --- Inject Remembered Windows ---
                    if self.window_memory:
                        context_parts.append("\n### Remembered Windows (Custom Tags)")
                        for hwnd, tag in self.window_memory.items():
                            context_parts.append(f"- hwnd={hwnd}: '{tag}'")
                    
                    context_parts.append("")
            except Exception as e:
                print(f"[AgenticPlanner] World context error: {e}")
        
        # Inject Experiential Memory context
        if self.experiential_memory:
            try:
                # Determine current app class from world model
                app_class = ""
                if self.situational_awareness:
                    m = self.situational_awareness.get_world_model()
                    app_class = m.foreground_state.get("class_name", "")
                
                memory_context = self.experiential_memory.get_context_for_planner(
                    app_class=app_class, current_task=self.goal or ""
                )
                if memory_context:
                    context_parts.append("## LEARNED KNOWLEDGE")
                    context_parts.append(memory_context)
                    context_parts.append("")
            except Exception as e:
                print(f"[AgenticPlanner] Memory context error: {e}")

        # Inject Fast Action Candidates (Presumptions)
        app_class = ""
        if self.situational_awareness:
            ctx = self.situational_awareness.get_current_context()
            app_class = ctx.get("app_class", "")

        if self.presumptions and app_class:
            try:
                candidates = self.presumptions.get_fast_action_candidates(app_class, min_weight=0.7)
                if candidates:
                    context_parts.append("## HIGH-CONFIDENCE PRESUMPTIONS (Fast Actions)")
                    context_parts.append("These elements have verified, stable locations. You can target them by label directly:")
                    for c in candidates[:5]: # type: ignore
                        context_parts.append(f"  - '{c['label']}' ({c['type']}) is typically at {c['location']} "
                                             f"(coords: {c['coords'][0]:.2f}, {c['coords'][1]:.2f}) [conf={c['confidence']:.2f}]") # type: ignore
                    context_parts.append("")
            except Exception as e:
                print(f"[AgenticPlanner] Presumption context error: {e}")
        if self.step_history:
            context_parts.append("## COMPLETED STEPS")
            # Explicit slicing to avoid linter confusion
            history_slice = self.step_history[max(0, len(self.step_history)-8):] # type: ignore
            for i, step in enumerate(history_slice, 1):
                status = "[OK] SUCCESS" if step.get("success") else "[X] FAILED"
                action_desc = step.get("action_type", "unknown")
                target = step.get("target", "")
                reasoning = step.get("reasoning", "")
                verification = step.get("verification_note", "")
                
                step_line = f"  Step {step['step_num']}: [{status}] {action_desc}"
                if target:
                    step_line += f" on '{target}'"
                if reasoning:
                    step_line += f" -> {reasoning}"
                if verification:
                    step_line += f" | Verify: {verification}"
                # Phase 8: Show VLM parallel feedback and CV confidence if available
                vlm_fb = step.get("vlm_feedback")
                if vlm_fb:
                    step_line += f"\n    > VLM Insight: {vlm_fb[:150]}" # type: ignore
                cv_conf = step.get("cv_confidence")
                if cv_conf:
                    step_line += f" | CV Match: {cv_conf:.2f}" # type: ignore
                context_parts.append(step_line)
            context_parts.append("")
            
        if self.tab_inventory:
            context_parts.append("## TAB INVENTORY (Discovered so far)")
            for tab, summary in self.tab_inventory.items():
                context_parts.append(f"  - {tab}: {summary}")
            context_parts.append("")
        
        # Inject HITL Interrupts
        if getattr(self, "hitl_interrupts", []):
            context_parts.append("## [CRITICAL] HUMAN-IN-THE-LOOP FEEDBACK")
            for interrupt in self.hitl_interrupts:
                cat = interrupt.get("category", "")
                urg = interrupt.get("urgency", "NORMAL")
                intent = interrupt.get("extracted_intent", interrupt.get("raw_message", ""))
                context_parts.append(f"> [{urg}] {cat}: {intent}")
            context_parts.append("> Adjust your next step to IMMEDATELY address this human feedback.\n")
            # Clear or consume them so they don't persist forever, or just keep them for history.
            # We will clear them since the LLM reads them once and adjusts course.
            self.hitl_interrupts.clear()

        context_parts.append(f"## CURRENT STATE")
        context_parts.append(f"Step number: {self.current_step + 1}")
        context_parts.append(f"Look at the CURRENT screenshot and decide the SINGLE NEXT ACTION.")
        
        if self.step_history and not self.step_history[-1].get("success"):
            last_step = self.step_history[-1]
            context_parts.append("\n> [!WARNING] The PREVIOUS step FAILED. Adapt your approach.")
            context_parts.append(f"> Failed action: {last_step.get('action_type')} on '{last_step.get('target', '')}'")
            
            coords = last_step.get("resolved_coords")
            if coords:
                context_parts.append(f"> FAILED at screen coordinates: x={coords.get('x')}, y={coords.get('y')}")
                context_parts.append("> If the target was slightly missed, adjust your hint_coords accordingly.")
            
            # Phase 5: Deep Root Cause Analysis and Memory Tree branching
            fi = last_step.get("failure_info", {})
            root_cause = fi.get("root_cause")
            if root_cause:
                context_parts.append(f"> RCA Category: {root_cause}")
                if root_cause == "WRONG_TARGET":
                    context_parts.append("> ANALYSIS: You interacted with the wrong element. Do NOT click the same location/icon again. Try searching or looking for a different visual cue.")
                elif root_cause == "TARGET_MISSING":
                    context_parts.append("> ANALYSIS: The expected result didn't appear. You may need to improvise by opening a menu, searching, or verifying the app state.")
            
            if fi.get("suggested_recovery"):
                context_parts.append(f"> Suggested Recovery: {fi['suggested_recovery']}")
                
            if fi.get("consequence_reason"):
                context_parts.append(f"> Consequence Analysis: {fi['consequence_reason']}")
            
            # Inject history-driven improvement suggestions
            if self.outcomes:
                try:
                    suggs = self.outcomes.get_improvement_suggestions(last_step.get('target', ''), app_class)
                    if suggs:
                        context_parts.append("> EXPERIENTIAL SUGGESTIONS:")
                        for s in suggs:
                            context_parts.append(f">   - {s}")
                except Exception:
                    pass
            
            context_parts.append(f"> Improvise a new plan. Re-examine the screen state and try an alternative approach.\n")
        
        return "\n".join(context_parts)

    def decide_next_step(self, screenshot_path: str) -> Optional[Dict[str, Any]]:
        """
        Core agentic decision function.
        
        Takes a screenshot of the current screen state, builds a dynamic
        context prompt, and queries Gemini for the single next action.
        
        Returns an action dict or None if goal is complete.
        """
        if self.is_complete():
            return None
        
        self.current_step += 1
        print(f"\n[AgenticPlanner] ====== Step {self.current_step} ======")
        
        # Build dynamic context
        context = self._build_dynamic_context()
        
        # ── Loop Sentinel Analysis + Recovery Engine ──
        diagnosis = self.sentinel.analyze(self.step_history)
        
        # If the previous recovery plan succeeded, record it
        if self._last_recovery_plan:
            last_success = self.step_history[-1].get('success', False) if self.step_history else False
            self.recovery_engine.record_recovery_outcome(
                self._last_recovery_plan, last_success,
                app_class=self._get_current_app_class(),
            )
            self._last_recovery_plan = None
        
        if diagnosis.should_fail:
            # CRITICAL: Use recovery engine for deep analysis before failing
            print(f"[AgenticPlanner] CRITICAL LOOP — attempting final recovery: {diagnosis.description}")
            recovery_plan = self.recovery_engine.diagnose_and_recover(
                diagnosis=diagnosis,
                step_history=self.step_history,
                window_state=self._window_state,
                app_class=self._get_current_app_class(),
            )
            
            # If recovery engine is confident, try recovery instead of failing
            if recovery_plan.confidence >= 0.7 and recovery_plan.recovery_actions:
                print(f"[AgenticPlanner] Recovery engine is {recovery_plan.confidence:.0%} confident. "
                      f"Root cause: {recovery_plan.root_cause.value}. Attempting recovery...")
                context += recovery_plan.context_injection + "\n"
                self._last_recovery_plan = recovery_plan
                self.sentinel.record_interrupt(diagnosis, "RECOVERY_ENGINE", True)
                # Don't fail yet — let the recovery play out next iteration
            else:
                # Actually fail
                self._is_failed = True
                self._failure_reason = f"Loop detected ({diagnosis.loop_type}): {diagnosis.description}"
                self.recovery_engine.record_recovery_outcome(recovery_plan, False, self._get_current_app_class())
                self.sentinel.record_interrupt(diagnosis, "FORCED_FAIL", False)
                return None
        
        elif diagnosis.should_interrupt:
            # MODERATE: Use recovery engine for root-cause analysis and course correction
            print(f"[AgenticPlanner] LOOP INTERRUPT ({diagnosis.severity.value}): {diagnosis.description}")
            recovery_plan = self.recovery_engine.diagnose_and_recover(
                diagnosis=diagnosis,
                step_history=self.step_history,
                window_state=self._window_state,
                app_class=self._get_current_app_class(),
            )
            
            # Inject rich recovery context (with root cause and past mitigations)
            context += recovery_plan.context_injection + "\n"
            
            # Also inject general alternative strategies
            context += "> \n> ### MANDATORY ALTERNATIVE ROUTE EXPLORATION\n"
            context += "> You MUST abandon the current approach and try something fundamentally different:\n"
            for i, strategy in enumerate(diagnosis.suggested_strategies, 1):
                context += f">   {i}. {strategy}\n"
            context += "> \n> Additional alternatives to consider:\n"
            context += ">   - Keyboard shortcuts (Ctrl+L for address bar, Alt+D, F6, Tab, Enter)\n"
            context += ">   - Right-click context menus for hidden options\n"
            context += ">   - Application menus (File → Open, Edit → Find)\n"
            context += ">   - URL bar direct navigation (type the URL)\n"
            context += ">   - OS-level workarounds (Win+R, Win+S)\n"
            context += ">   - Scroll or resize the window to reveal different elements\n"
            context += ">   - Use an automation artifact (click_and_drag, nested_menu_navigate, scroll_until_visible)\n"
            context += ">   - Use create_cv_script to author a custom visual check\n"
            
            self._last_recovery_plan = recovery_plan
            self.sentinel.record_interrupt(diagnosis, "RECOVERY_ENGINE", True)
        
        elif diagnosis.severity == LoopSeverity.MILD:
            # MILD severity: inject warning
            print(f"[AgenticPlanner] Loop warning (MILD): {diagnosis.description}")
            context = str(context) + f"\n> [!WARNING] MILD LOOP WARNING: {diagnosis.description}\n" # type: ignore
            context += "> Consider trying a different approach before this escalates.\n"
        
        # Enhanced stuck detection: build a blacklist of failed targets with their coordinates
        failed_targets_blacklist = []
        blacklist_slice = list(self.step_history[max(0, len(self.step_history)-6):]) # type: ignore
        for step in blacklist_slice:
            if not step.get("success") and step.get("target"):
                coords = step.get("resolved_coords") or {}
                entry = f"'{step['target']}'"
                if coords:
                    entry += f" at approx ({coords.get('x', '?')}, {coords.get('y', '?')})" # type: ignore
                failed_targets_blacklist.append(entry)
        
        if failed_targets_blacklist:
            context = str(context) + "\n> [!CAUTION] BLACKLISTED TARGETS (recently failed — DO NOT click these again at the same coordinates):\n" # type: ignore
            for t in failed_targets_blacklist:
                context += f">   ✗ {t}\n"
            context += "> If you need to click an element with the same label, it MUST be at DIFFERENT coordinates (e.g. inside a dialog instead of on the page behind it).\n"

        print(f"[AgenticPlanner] Context:\n{context}")
        
        # Load screenshot
        try:
            img = Image.open(screenshot_path)
        except Exception as e:
            print(f"[AgenticPlanner] Failed to load screenshot: {e}")
            return {"action": "wait", "duration": 1.0, "reasoning": "Screenshot capture failed, retrying"}
        
        # Query Gemini with screenshot + dynamic context
        try:
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=[
                    self.SYSTEM_PROMPT,
                    context,
                    "Current screen state:",
                    img
                ],
                config=types.GenerateContentConfig(
                    temperature=0.2,
                    response_mime_type="application/json"
                )
            )
            
            raw_text = response.text.strip()
            print(f"[AgenticPlanner] Gemini response: {raw_text}")
            
            # Parse the JSON response
            decision = json.loads(raw_text)
            if not decision:
                return {"action": "wait", "duration": 1.0, "reasoning": "Gemini returned a null decision"}
        except json.JSONDecodeError as e:
            print(f"[AgenticPlanner] JSON parse error: {e}. Raw: {raw_text}")
            # Try to extract JSON from markdown wrappers
            if "```json" in raw_text:
                json_block = raw_text.split("```json")[1].split("```")[0].strip()
                decision = json.loads(json_block)
            else:
                return {"action": "wait", "duration": 1.0, "reasoning": "Failed to parse Gemini response"}
        except Exception as e:
            print(f"[AgenticPlanner] Gemini query failed: {e}")
            return {"action": "wait", "duration": 1.0, "reasoning": f"Gemini API error: {e}"}
        
        # Check for goal completion
        if decision.get("goal_complete"):
            print(f"[AgenticPlanner] [OK] Goal COMPLETE: {decision.get('reasoning', '')}")
            self._is_complete = True
            return None
        
        if decision.get("goal_failed"):
            self._is_failed = True
            self._failure_reason = decision.get("failure_reason", "Unknown")
            print(f"[AgenticPlanner] [X] Goal FAILED: {self._failure_reason}")
            return None
        
        # Log the decision
        print(f"[AgenticPlanner] Decision: {decision.get('action')} — {decision.get('reasoning', '')}")
        
        return decision

    def verify_step(self, action: Dict[str, Any], after_screenshot_path: str) -> bool:
        """
        After executing an action, verify whether it succeeded by
        sending the after-screenshot to Gemini with context.
        
        This is the verification half of the agentic loop.
        """
        target = action.get("target", action.get("text", ""))
        action_type = action.get("action", "unknown")
        reasoning = action.get("reasoning", "")
        
        # Skip verification for wait/press/hotkey (hard to visually verify)
        if action_type in ["wait", "press", "hotkey"]:
            self._record_step(action, success=True, verification_note="Skipped (non-visual action)")
            return True
        
        try:
            img = Image.open(after_screenshot_path)
        except Exception as e:
            print(f"[AgenticPlanner] Failed to load after-screenshot: {e}")
            self._record_step(action, success=True, verification_note="Screenshot unavailable")
            return True
        
        verify_prompt = f"""You are an expert UI verification agent.
You just executed this action:
- Action: {action_type}
- Target: {target}
- INTENDED CONSEQUENCE: {reasoning}

Look at the CURRENT screen state (after the action).
Does it show data/UI changes consistent with the action reaching its target?

CHECK FOR:
1. **Visual Cues**: Did a new window appear? Did a menu open? Did text appear in a box?
2. **State Match**: If clicking an icon, is that app now in the foreground?
3. **Negative Cues**: Did a 'Not Found' or 'Error' popup appear instead?

Respond in JSON ONLY with this structure:
{{
  "observation": "<precise description of what changed on screen>",
  "success": true/false,
  "confidence": 0.0-1.0,
  "analysis": "<detailed technical explanation of why it succeeded or failed based on visual evidence>",
  "root_cause": "<if success=false, pick one: WRONG_TARGET, TARGET_MISSING, STATE_CHANGED_UNEXPECTEDLY, NO_VISUAL_CHANGE, ERROR_POPUP, UNKNOWN>",
  "suggested_recovery": "<if success=false, what should the agent try next?>",
  "discovered_tabs": [{{"title": "Tab Title", "summary": "Brief summary of content"}}],
  "note": "<brief summary for agent history>"
}}
"""
        
        try:
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=[verify_prompt, img],
                config=types.GenerateContentConfig(
                    temperature=0.1,
                    response_mime_type="application/json"
                )
            )
            
            result = json.loads(response.text.strip())
            success = result.get("success", True)
            note = result.get("note", result.get("analysis", ""))
            observation = result.get("observation", "")
            
            if not success:
                # Inject RCA directly into the action's failure_info dict for _record_step
                if "failure_info" not in action:
                    action["failure_info"] = {}
                action["failure_info"]["root_cause"] = result.get("root_cause", "UNKNOWN")
                action["failure_info"]["suggested_recovery"] = result.get("suggested_recovery", "")
            
            print(f"[AgenticPlanner] Verification: {'[OK] SUCCESS' if success else '[X] FAILED'} ({result.get('root_cause', 'N/A')}) -- {note}")
            
            self._record_step(action, success=success, 
                            verification_note=f"{note} | Screen: {observation}",
                            discovered_tabs=result.get("discovered_tabs", []))
            return success
            
        except Exception as e:
            print(f"[AgenticPlanner] Verification query failed: {e}. Assuming success.")
            self._record_step(action, success=True, verification_note=f"Verification error: {e}")
            return True

    def _record_step(self, action: Dict[str, Any], success: bool, verification_note: str = "", discovered_tabs: List[Dict[str, str]] = []):
        """Records a completed step in the history for context building."""
        step_record = {
            "step_num": self.current_step,
            "action_type": action.get("action", "unknown"),
            "target": action.get("target", action.get("text", "")),
            "reasoning": action.get("reasoning", ""),
            "success": success,
            "verification_note": verification_note,
            "resolved_coords": action.get("resolved_coords"),
            "failure_info": action.get("failure_info", {}),
            "timestamp": time.time()
        }
        self.step_history.append(step_record)
        
        # New: Register tabs
        if discovered_tabs:
            for tab_info in discovered_tabs:
                title = tab_info.get("title")
                if title:
                    self.tab_inventory[title] = tab_info.get("summary", "No summary")
        
        # New: Auto-register tabs if reasoning mention discovery
        reasoning = str(step_record.get("reasoning", "") or "").lower()
        if "tab" in reasoning or "window" in reasoning:
            # We don't have the exact tab name here, but the LLM will provide it 
            # in the next turn's reasoning after seeing the verified screen.
            pass
            
        if not success:
            self.consecutive_failures += 1
        else:
            self.consecutive_failures = 0

    def get_step_summary(self) -> str:
        """
        Returns a text summary of the last completed step.
        This is sent as a text input to the Live API to steer its understanding.
        """
        if not self.step_history:
            return f"Starting task: {self.goal}"
        
        last = self.step_history[-1]
        status = "SUCCESS" if last["success"] else "FAILED"
        summary = (
            f"Step {last['step_num']} completed: {last['action_type']}"
        )
        if last.get("target"):
            summary += f" on '{last['target']}'"
        summary += f". Result: {status}."
        if last.get("verification_note"):
            summary += f" {last['verification_note']}"
        
        return summary

    def _get_current_app_class(self) -> str:
        """Returns the current app class from situational awareness, or empty string."""
        if self.situational_awareness:
            try:
                m = self.situational_awareness.get_world_model()
                return m.foreground_state.get("class_name", "")
            except Exception:
                pass
        return ""

    def get_full_history_summary(self) -> str:
        """Returns a complete summary of all steps for final reporting."""
        lines = [f"Task: {self.goal}", f"Status: {self.get_status()}", ""]
        for step in self.step_history:
            status = "[OK]" if step["success"] else "[X]"
            lines.append(f"  {status} Step {step['step_num']}: {step['action_type']} on '{step.get('target', '')}'")
            if step.get("reasoning"):
                lines.append(f"    Reason: {step['reasoning']}")
        return "\n".join(lines)
