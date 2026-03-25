import time
import sys
import os
import re
from typing import Dict, Any, Optional
from pathlib import Path

# Add the 'src' directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from coding_agent.ui_agent.ui_controller.mouse import MouseController # type: ignore
from coding_agent.ui_agent.ui_controller.keyboard import KeyboardController # type: ignore
from coding_agent.ui_agent.ui_controller.window_manager import WindowManager # type: ignore
from coding_agent.ui_agent.perception.screen import ScreenObserver, WindowObserver # type: ignore
from coding_agent.ui_agent.perception.vision_analyzer import VisionAnalyzer # type: ignore
from coding_agent.ui_agent.planning.goal_manager import GoalManager # type: ignore
from coding_agent.ui_agent.planning.action_predictor import ActionPredictor # type: ignore
from coding_agent.ui_agent.planning.learning_loop import LearningLoop # type: ignore
from coding_agent.ui_agent.planning.agentic_planner import AgenticPlanner # type: ignore
from coding_agent.ui_agent.core.ui_memory_agent import UIMemoryAgent # type: ignore
from coding_agent.ui_agent.ui_controller.live_streamer import GeminiLiveStreamer # type: ignore
from coding_agent.ui_agent.ui_controller.hud_overlay import HUDOverlay # type: ignore
from coding_agent.ui_agent.core.tools.browser_tool import BrowserTool # type: ignore
from coding_agent.ui_agent.core.tools.automation_artifacts import AutomationArtifactRegistry # type: ignore
from coding_agent.ui_agent.core.tools.swarm_script_manager import SwarmScriptManager # type: ignore
from coding_agent.ui_agent.perception.cv_logger import CVLogger # type: ignore
from coding_agent.ui_agent.perception.anticipatory_observer import AnticipatoryObserver, ProcessContext # type: ignore
from coding_agent.ui_agent.perception.window_state_validator import WindowStateValidator # type: ignore
from coding_agent.ui_agent.planning.loop_sentinel import LoopRecoveryEngine, LoopSeverity # type: ignore
import win32gui # type: ignore
import win32con # type: ignore
import asyncio
import threading
import cv2 # type: ignore
from coding_agent.ui_agent.utils.vision_utils import adapt_rect_to_edges # type: ignore
from coding_agent.ui_agent.perception.edge_engine import EdgePerceptionEngine # type: ignore



class UIAgent:
    """
    The main orchestrator that ties together all modules:
    Perception, Planning, Action, Awareness, and Learning into a continuous loop.
    """
    def __init__(self):
        # Centralized DPI-aware screen geometry (replaces manual DPI setup)
        from coding_agent.ui_agent.utils.screen_geometry import ScreenGeometry # type: ignore
        self.screen_geometry = ScreenGeometry(force_refresh=True)
        print(f"[UIAgent] Screen Geometry: {self.screen_geometry.logical_size} logical, "
              f"{self.screen_geometry.physical_size} physical, "
              f"scale={self.screen_geometry.scale_factor:.2f}x")

        # Cursor type detection and monitoring
        try:
            from coding_agent.ui_agent.perception.cursor_monitor import CursorMonitor # type: ignore
            self.cursor_monitor = CursorMonitor()
            print("[UIAgent] CursorMonitor initialized")
        except Exception as e:
            print(f"[UIAgent] CursorMonitor init failed: {e}")
            self.cursor_monitor = None

        try:
            from coding_agent.ui_agent.perception.cursor_snip_verifier import CursorSnipVerifier # type: ignore
            snip_dir = os.path.join(self.workspace_path, ".gemini", "diagnostic_snips")
            self.cursor_verifier = CursorSnipVerifier(
                screen_observer=self.screen_observer,
                vision_analyzer=self.vision,
                cursor_monitor=self.cursor_monitor,
                screen_geometry=self.screen_geometry,
                snip_dir=snip_dir
            )
            print("[UIAgent] CursorSnipVerifier initialized")
        except Exception as e:
            print(f"[UIAgent] CursorSnipVerifier init failed: {e}")
            self.cursor_verifier = None

        print("[UIAgent] Initializing agent modules...", flush=True)
        self.offline_mode = False
        self.live_session_healthy = False
        self.mouse = MouseController(base_speed=1.0, screen_geometry=self.screen_geometry,
                                       cursor_monitor=self.cursor_monitor)
        print("[UIAgent] Mouse initialized", flush=True)
        self.keyboard = KeyboardController(wpm=60)
        print("[UIAgent] Keyboard initialized", flush=True)
        
        self.workspace_path = Path(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
        self.memory_agent = UIMemoryAgent(self.workspace_path)
        print("[UIAgent] UI Memory Agent initialized", flush=True)

        # Shared context for selective feedback grounding
        self.process_context = ProcessContext()
        self.anticipatory_observer = AnticipatoryObserver(
            memory_agent=self.memory_agent, 
            shared_context=self.process_context
        )
        
        # Inject short-term paths into observers
        screenshot_dir = str(self.memory_agent.short_term_dir / "screenshots")
        self.screen_observer = ScreenObserver(output_dir=screenshot_dir)
        print("[UIAgent] Screen observer initialized", flush=True)
        self.window_observer = WindowObserver(output_dir=screenshot_dir)
        print("[UIAgent] Window observer initialized", flush=True)
        
        self.browser_tool = BrowserTool(headless=False)
        print("[UIAgent] BrowserTool wrapper initialized", flush=True)
        
        # ═══ Layer 3: CVPipeline (Foundation) ═══
        try:
            from coding_agent.ui_agent.perception.cv_pipeline import CVPipeline # type: ignore
            self.cv_pipeline = CVPipeline()
            print("[UIAgent] CVPipeline initialized", flush=True)
        except Exception as e:
            print(f"[UIAgent] CVPipeline init failed (fallback mode): {e}")
            self.cv_pipeline = None
        
        # VisionAnalyzer with CVPipeline hook
        self.vision = VisionAnalyzer(cv_pipeline=self.cv_pipeline)
        print("[UIAgent] Vision analyzer initialized (CV pipeline:", "attached" if self.cv_pipeline else "none", ")", flush=True)
        
        # ═══ Layer 3: Stability Classifier ═══
        try:
            from coding_agent.ui_agent.perception.static_dynamic_classifier import StaticDynamicClassifier # type: ignore
            self.stability_classifier = StaticDynamicClassifier(cv_pipeline=self.cv_pipeline)
            print("[UIAgent] StaticDynamicClassifier initialized", flush=True)
        except Exception as e:
            print(f"[UIAgent] StaticDynamicClassifier init failed: {e}")
            self.stability_classifier = None
        
        # ═══ Layer 1: ElementBoundaryLearner ═══
        try:
            from coding_agent.ui_agent.perception.element_boundary_learner import ElementBoundaryLearner # type: ignore
            self.boundary_learner = ElementBoundaryLearner(cv_pipeline=self.cv_pipeline, vision_analyzer=self.vision)
            print("[UIAgent] ElementBoundaryLearner initialized", flush=True)
        except Exception as e:
            print(f"[UIAgent] ElementBoundaryLearner init failed: {e}")
            self.boundary_learner = None
        
        # ═══ Layer 1: OSAwareness ═══
        try:
            from coding_agent.ui_agent.perception.os_awareness import OSAwareness # type: ignore
            self.os_awareness = OSAwareness()
            print("[UIAgent] OSAwareness initialized", flush=True)
        except Exception as e:
            print(f"[UIAgent] OSAwareness init failed: {e}")
            self.os_awareness = None
        
        # ═══ Layer 2: AppCharacterizer ═══
        try:
            from coding_agent.ui_agent.perception.app_characterizer import AppCharacterizer # type: ignore
            self.app_characterizer = AppCharacterizer(cv_pipeline=self.cv_pipeline, vision_analyzer=self.vision, memory_agent=self.memory_agent)
            print("[UIAgent] AppCharacterizer initialized", flush=True)
        except Exception as e:
            print(f"[UIAgent] AppCharacterizer init failed: {e}")
            self.app_characterizer = None
        
        # ═══ Layer 5: ExperientialMemory ═══
        try:
            from coding_agent.ui_agent.planning.experiential_memory import ExperientialMemory # type: ignore
            memory_dir = str(self.workspace_path / "agent_memory" / "experiential")
            self.experiential_memory = ExperientialMemory(storage_dir=memory_dir)
            print("[UIAgent] ExperientialMemory initialized", flush=True)
        except Exception as e:
            print(f"[UIAgent] ExperientialMemory init failed: {e}")
            self.experiential_memory = None
        
        # ═══ Layer 4: SituationalAwareness ═══
        try:
            from coding_agent.ui_agent.planning.situational_awareness import SituationalAwareness # type: ignore
            self.situational_awareness = SituationalAwareness(
                os_awareness=self.os_awareness,
                app_characterizer=self.app_characterizer,
                cv_pipeline=self.cv_pipeline,
                boundary_learner=self.boundary_learner,
                stability_classifier=self.stability_classifier,
                memory=self.memory_agent,
                experiential_memory=self.experiential_memory
            )
            print("[UIAgent] SituationalAwareness initialized", flush=True)
        except Exception as e:
            print(f"[UIAgent] SituationalAwareness init failed: {e}")
            self.situational_awareness = None

        # ═══ Phase 12 & 13: Advanced Learning Layers ═══
        try:
            from coding_agent.ui_agent.perception.edge_correlator import EdgeCorrelator # type: ignore
            from coding_agent.ui_agent.planning.action_outcome_tracker import ActionOutcomeTracker # type: ignore
            from coding_agent.ui_agent.planning.presumption_engine import PresumptionEngine # type: ignore
            from coding_agent.ui_agent.planning.postponed_judgement import PostponedJudgement # type: ignore

            self.edge_correlator = EdgeCorrelator()
            
            outcome_dir = str(self.workspace_path / "agent_memory" / "outcomes")
            self.outcome_tracker = ActionOutcomeTracker(storage_dir=outcome_dir)
            
            presumption_dir = str(self.workspace_path / "agent_memory" / "presumptions")
            self.presumption_engine = PresumptionEngine(storage_dir=presumption_dir)
            
            self.postponed_judgement = PostponedJudgement()
            
            print("[UIAgent] Advanced Learning modules initialized", flush=True)
        except Exception as e:
            print(f"[UIAgent] Learning modules init failed: {e}")
            self.outcome_tracker = None
            self.presumption_engine = None
            self.postponed_judgement = None
            self.edge_correlator = None
        
        # ═══ DecisionManager ═══
        self.cv_logger = CVLogger("cv_logs", file_logging=True)
        try:
            from coding_agent.ui_agent.planning.decision_manager import DecisionManager # type: ignore
            self.decision_manager = DecisionManager(
                experiential_memory=self.experiential_memory,
                boundary_learner=self.boundary_learner,
                cv_pipeline=self.cv_pipeline,
                stability_classifier=self.stability_classifier,
                live_streamer=None,  # Set after live_streamer init
                vision_analyzer=self.vision,
                ui_memory_agent=self.memory_agent,
                presumption_engine=self.presumption_engine,  # Phase 13
                cv_logger=self.cv_logger,                  # Phase 11
            )
            print("[UIAgent] DecisionManager initialized", flush=True)
        except Exception as e:
            print(f"[UIAgent] DecisionManager init failed: {e}")
            self.decision_manager = None
        
        # ═══ Core Planning/Action modules (with awareness hooks) ═══
        self.goal_manager = GoalManager()
        self.predictor = ActionPredictor(
            self.vision, memory_agent=self.memory_agent, 
            screen_observer=self.screen_observer,
            decision_manager=self.decision_manager,
            cv_pipeline=self.cv_pipeline,
            screen_geometry=self.screen_geometry,
            cursor_monitor=self.cursor_monitor,
            experiential_memory=self.experiential_memory,
        )
        self.learning_loop = LearningLoop(
            self.vision, self.predictor,
            experiential_memory=self.experiential_memory,
            cv_pipeline=self.cv_pipeline,
            outcome_tracker=self.outcome_tracker,      # Phase 13
            presumption_engine=self.presumption_engine, # Phase 13
            postponed_judgement=self.postponed_judgement,# Phase 13
        )
        self.learning_loop.anticipatory_observer = self.anticipatory_observer
        self.window_manager = WindowManager()
        print("[UIAgent] Planning and Management modules initialized", flush=True)
        
        # Initialize Gemini Live Streamer (Disabled by default until started)
        self.live_streamer = GeminiLiveStreamer()
        self.live_streamer.screen_observer = self.screen_observer
        self._streaming_thread = None
        self._loop = asyncio.new_event_loop()
        
        # Wire live_streamer into DecisionManager
        if self.decision_manager:
            self.decision_manager.live = self.live_streamer
        
        # Set response callback for Live API (Function-calling style triggers)
        self.live_streamer.set_callback(self._handle_live_response)
        
        self.is_running = False
        self.hud = HUDOverlay() # Initialize HUD
        self.planner: Optional[AgenticPlanner] = None 
        
        # --- Bootstrap Static GUI Map into CV Memory ---
        self._bootstrap_static_memory()

        # ═══ Automation Artifact Registry ═══
        self.automation_artifacts = AutomationArtifactRegistry(
            mouse_controller=self.mouse,
            keyboard_controller=self.keyboard,
            screen_observer=self.screen_observer,
            vision_analyzer=self.vision,
        )
        print("[UIAgent] AutomationArtifactRegistry initialized", flush=True)

        # ═══ Window State Validator ═══
        self.window_validator = WindowStateValidator(
            window_manager=self.window_manager,
            process_context=self.process_context,
        )
        print("[UIAgent] WindowStateValidator initialized", flush=True)

        # ═══ Swarm Script Manager ═══
        scripts_dir = str(self.workspace_path / "agent_memory" / "scripts")
        self.script_manager = SwarmScriptManager(
            storage_dir=scripts_dir,
            anticipatory_observer=self.anticipatory_observer,
        )
        print("[UIAgent] SwarmScriptManager initialized", flush=True)

        # ═══ HITL Feedback & Swarm Interface ═══
        try:
            from coding_agent.ui_agent.core.hitl_contextualizer import HITLContextualizer # type: ignore
            from coding_agent.ui_agent.core.swarm_interface import SwarmInterface # type: ignore
            self.hitl = HITLContextualizer(llm_client=self.vision.client if self.vision else None)
            self.swarm_interface = SwarmInterface(hitl_contextualizer=self.hitl)
            print("[UIAgent] HITLContextualizer and SwarmInterface initialized", flush=True)
        except Exception as e:
            print(f"[UIAgent] HITL init failed: {e}")
            self.hitl = None
            self.swarm_interface = None


        # ═══ Edge Perception Engine ═══
        self.edge_engine = EdgePerceptionEngine()
        print("[UIAgent] EdgePerceptionEngine initialized", flush=True)

        print("[UIAgent] Bootstrap complete", flush=True)


    def _handle_live_response(self, response: Dict[str, Any]):
        """
        Intercepts real-time JSON responses from the Live API.
        Handles function-call style vision triggers and context updates.
        """
        if "update_context" in response:
            self.process_context.update(**response["update_context"])
            print(f"[UIAgent:LiveAPI] Process context updated: {response['update_context']}")
            
        if "call_tool" in response:
            tool = response["call_tool"]
            
            # --- Fallback Chain: Local CV Script ---
            if tool == "cv_script":
                obs_type = response.get("type", "color_shift")
                print(f"[UIAgent:Orchestrator] Live API requested CV check: {obs_type}")
                
                import time
                import cv2 # type: ignore
                requires_before = obs_type in ["color_shift", "structural_change"]
                
                b = None
                if requires_before:
                    p1 = self.screen_observer.capture()
                    if p1:
                        b = cv2.imread(p1)
                    time.sleep(0.2)
                    
                p2 = self.screen_observer.capture()
                if p2 and self.anticipatory_observer:
                    a = cv2.imread(p2)
                    res = self.anticipatory_observer.verify_observable(
                        {"type": obs_type, "target_region": response.get("roi")}, b, a, {}
                    )
                    insight = f"CV [{obs_type}]: {'SUCCESS' if res.get('success') else 'NO MATCH'}"
                    if res.get('request_fidelity') == 'HIGH':
                        insight += " (Confidence low — requesting Static VLM upgrade)"
                        
                    # Inject back into Live API
                    self.live_streamer.send_step_feedback(f"[VISION_BRIDGE] {insight}", self._loop)
                    # Also share with static context
                    self.process_context.update(latest_cv_insight=insight)
            
            # --- Fallback Chain: Static High-Res VLM ---
            elif tool == "static_vlm":
                query = response.get("query", "Analyze current state")
                print(f"[UIAgent:Orchestrator] Live API requested Static VLM: {query}")
                result = self.vlm_bridge(query, source="live_api")
                
                insight = f"Static VLM Insight: {result.get('insight', 'Failed to analyze')}"
                # Feed insight back to Live API
                self.live_streamer.send_step_feedback(
                    f"[VISION_BRIDGE_RESULT] Query: '{query}' -> {insight}", 
                    self._loop
                )
                # Share with overall process context
                self.process_context.update(latest_vlm_insight=insight)

    def vlm_bridge(self, query: str, state_path: Optional[str] = None, source: str = "planner") -> Dict[str, Any]:
        """
        Orchestration bridge: Allows Live Streamer or CV modules to 
        request high-resolution analysis from the Static VLM.
        """
        print(f"[UIAgent:VLMBridge] Triggered from {source}: {query}")
        # Capture current state if not provided
        if not state_path:
            state_path = self.screen_observer.capture_full_screen()
            
        # Enrich query with process context
        enriched_query = query
        ctx = self.process_context.get_all()
        if ctx:
            enriched_query += f"\n\nContext:\n- Active Window: {ctx.get('window_title', 'Unknown')}"
            if "latest_cv_insight" in ctx:
                enriched_query += f"\n- Recent CV Observation: {ctx['latest_cv_insight']}"
        
        # Trigger Static VLM via Vision Analyzer
        success, reason = self.vision.visual_diff(
            state_path, state_path, # Static check on single frame conceptually
            action_context={"action": f"vlm_bridge_{source}", "query": enriched_query}
        )
        
        return {
            "success": success,
            "insight": reason if reason else ("Analyzed successfully" if success else "Analysis failed"),
            "source": source,
            "timestamp": int(time.time()),
        }

    def _start_live_stream(self):
         """Starts the asyncio event loop in a background thread to maintain the WebSocket."""
         def run_loop(loop):
             asyncio.set_event_loop(loop)
             
             system_prompt = """
             You are a high-speed Multimodal UI Agent. You assist with real-time UI manipulation.
             
             VISION BRIDGE (Function Calling):
             If you need a high-resolution analysis (Static VLM) or a fast CV check (Anticipatory), output a JSON object:
             - {"call_tool": "static_vlm", "query": "Is the checkbox actually checked?"}
             - {"call_tool": "cv_script", "type": "color_shift", "roi": "local"}
             
             CONTEXT UPDATES:
             You can also signal process context changes:
             - {"update_context": {"url": "https://reddit.com", "active_element_id": "login-button"}}
             
             Otherwise, provide short, actionable interaction insights.
             """.strip()
             
             async def runner():
                 # Create the stream task which will internally sleep until self.is_streaming becomes True
                 async def safe_stream():
                      while not self.live_streamer.is_streaming:
                           await asyncio.sleep(0.5)
                      await self.live_streamer.stream_screen_loop(fps=0.25)
                      
                 loop.create_task(safe_stream())
                 
                 # Block on the context manager session
                 await self.live_streamer.start_session(system_instruction=system_prompt)
                 
             try:
                 loop.run_until_complete(runner())
             except Exception as e:
                 print(f"[UIAgent - LiveStreamer] Could not establish live session: {e}")
                 self.offline_mode = True
                 self.live_session_healthy = False
                 
         self._streaming_thread = threading.Thread(target=run_loop, args=(self._loop,), daemon=True) # type: ignore
         self._streaming_thread.start() # type: ignore
         
         # Wait a few seconds for health check
         time.sleep(2)
         if not self.live_streamer.is_streaming:
             print("[UIAgent] Live API appears offline. Continuing in OFFLINE mode.")
             self.offline_mode = True
             self.live_session_healthy = False
         else:
             self.live_session_healthy = True

    async def _stop_live_stream_async(self):
        """Safely shuts down the background WebSocket and Gemini Client."""
        if self.live_streamer and self.live_streamer.is_streaming:
            print("[UIAgent] Disconnecting Live Streamer...", flush=True)
            try:
                # 1. Gracefully disconnect the websocket
                await asyncio.wait_for(self.live_streamer.disconnect(), timeout=5.0)
            except asyncio.TimeoutError:
                print("[UIAgent] Disconnect timed out. Forcing loop stop.")
            except Exception as e:
                print(f"[UIAgent] Disconnect error: {e}")

        # 2. Shutdown the GenAI client if it has an aclose method (google.genai >= 0.3.0)
        if hasattr(self.live_streamer, 'client') and self.live_streamer.client:
             try:
                 print("[UIAgent] Closing Gemini Client...", flush=True)
                 await self.live_streamer.client.aclose()
             except Exception as e:
                 print(f"[UIAgent] Client close error: {e}")

        # 3. Stop the background loop
        if self._loop.is_running():
            self._loop.stop()

    def _stop_live_stream(self):
         """Synchronous wrapper for async teardown."""
         if self._loop and self._loop.is_running():
             future = asyncio.run_coroutine_threadsafe(self._stop_live_stream_async(), self._loop)
             try:
                 future.result(timeout=10.0)
             except Exception as e:
                 print(f"[UIAgent] Teardown wait error: {e}")


    def _bootstrap_static_memory(self):
        """Finds the most recent gui_map and raw screenshot in agent_memory (LT and ST) and loads them."""
        import glob
        import os
        
        # Search in both Long-Term and Short-Term for the freshest map
        search_paths = [
            (self.memory_agent.long_term_dir / "predicted_outputs", self.memory_agent.long_term_dir / "raw_screenshots"),
            (self.memory_agent.short_term_dir / "predicted_outputs", self.memory_agent.short_term_dir / "diagnostics")
        ]
        
        all_json_files = []
        path_map = {}
        
        for json_dir, img_dir in search_paths:
            found = glob.glob(os.path.join(str(json_dir), "gui_map_*.json"))
            all_json_files.extend(found)
            for f in found:
                path_map[f] = img_dir

        if not all_json_files:
            print("[UIAgent] No static gui_map JSON found to bootstrap.")
            return
            
        latest_json = max(all_json_files, key=os.path.getctime)
        img_dir = path_map[latest_json]
        
        # Extract timestamp: gui_map_1234567.json -> 1234567
        filename = os.path.basename(latest_json)
        timestamp = filename.replace("gui_map_", "").replace(".json", "") # type: ignore
        
        # Find matching raw screenshot (might be capture_ or raw_)
        possible_images = [
            os.path.join(str(img_dir), f"capture_{timestamp}.png"),
            os.path.join(str(img_dir), f"raw_capture_{timestamp}.png")
        ]
        
        raw_image = next((img for img in possible_images if os.path.exists(img)), None)
        
        if raw_image:
            print(f"[UIAgent] Bootstrapping from LATEST Map: {latest_json}")
            self.predictor.load_static_map(latest_json, raw_image)
        else:
            print(f"[UIAgent] Missing matching raw screenshot for {latest_json}. Looked in {img_dir}")

    # Legacy methods removed for compactness. Use run_agentic for all tasks.

    async def run_agentic(self, task: str, send_callback=None):
        """
        New AGENTIC execution loop.
        
        Instead of pre-planning all steps, this loop:
          1. OBSERVE: Capture current screen
          2. DECIDE: Ask Gemini what to do next (one step at a time)
          3. RESOLVE: Get precise coordinates (with zoom if needed)
          4. ACT: Execute the single action
          5. VERIFY: Check if the action succeeded
          6. FEEDBACK: Send step summary to Live API to steer context
          7. LOOP: If goal not complete, go to step 1
        """
        self.is_running = True
        print("[UIAgent] ===========================================")
        print(f"[UIAgent]   AGENTIC MODE: {task}")
        print("[UIAgent] ===========================================")

        try:
            # Boot Live Streamer EARLY
            self._start_live_stream()
            print("[UIAgent] Waiting 8 seconds for Live API connection...")
            time.sleep(8)
            
            if self.live_streamer.is_streaming:
                self.hud.update_status(True)
                print("[UIAgent] Live Stream ACTIVE")
            else:
                self.hud.update_status(False, fallback_active=True)
                print("[UIAgent] Live Stream offline. Using static VLM for planning.")

            # ── FORCE DELEGATION FOR TELEGRAM WEB ──
            if "telegram" in task.lower() and ("web" in task.lower() or "browser" in task.lower()):
                print(f"[UIAgent] Detected Telegram web task. Bypassing OS loop and delegating to BrowserTool.")
                if hasattr(self, 'hud'):
                    self.hud.update_goal(task)
                
                # Start the task
                await self.browser_tool.execute_objective(task)
                
                print("[UIAgent] BrowserTool objective complete.")
                return "[SUCCESS] Telegram web task completed via BrowserTool."

            # Initialize agentic planner
            planner = AgenticPlanner(
                situational_awareness=self.situational_awareness,
                experiential_memory=self.experiential_memory,
                decision_manager=self.decision_manager
            )
            planner.set_goal(task)
            self.hud.update_goal(task)
            
            # === NEW: IN-PLACE WINDOW DETECTION ===
            existing_hwnd = None
            app_keywords = ["chrome", "notepad", "edge", "explorer", "calculator", "word", "excel", "powerpoint"]
            process_map = {
                "chrome": "chrome.exe", "edge": "msedge.exe", "notepad": "notepad.exe",
                "explorer": "explorer.exe", "calculator": "Calculator.exe",
                "word": "WINWORD.EXE", "excel": "EXCEL.EXE", "powerpoint": "POWERPNT.EXE",
            }
            target_app = None
            all_matching_windows = []
            for kw in app_keywords:
                if kw in task.lower():
                    # Enumerate ALL matching windows, not just the first
                    window_info = self.window_manager.get_window_count_by_title(kw)
                    if window_info["count"] > 0:
                        existing_hwnd = window_info["windows"][0]["hwnd"]
                        all_matching_windows = window_info["windows"]
                        target_app = kw
                    break
            
            # Inject taskbar info for the planner
            taskbar_info = self.window_manager.get_taskbar_info()

            if existing_hwnd:
                print(f"[UIAgent] Found {len(all_matching_windows)} existing '{target_app}' window(s).")
                for win_data in all_matching_windows:
                    # type: ignore[attr-defined]
                    w_hwnd = win_data.get('hwnd')
                    w_title = win_data.get('title')
                    w_min = win_data.get('is_minimized')
                    print(f"  - hwnd={w_hwnd}, title='{w_title}', minimized={w_min}")
                
                # Focus the first matching window for in-place execution
                self.window_manager.focus_window(existing_hwnd)
                
                # Build window inventory for the planner so it can decide to reuse or open new
                window_inventory = []
                for win in all_matching_windows:
                    window_inventory.append(f"hwnd={win['hwnd']}: '{win['title']}' {'(minimized)' if win['is_minimized'] else '(visible)'}")
                
                planner.step_history.append({
                    "step_num": 0,
                    "action_type": "attach",
                    "target": target_app,
                    "success": True,
                    "reasoning": f"Detected {len(all_matching_windows)} existing {target_app} window(s). Focused first.",
                    "verification_note": f"Window inventory: {'; '.join(window_inventory)}. "
                                         f"Taskbar position: {taskbar_info.get('position', 'unknown')}, rect: {taskbar_info.get('rect', 'unknown')}.",
                })
        
            # ── Loop Recovery Engine (bound to the planner's sentinel) ──
            recovery_engine = LoopRecoveryEngine(
                sentinel=planner.sentinel,
                experiential_memory=self.experiential_memory,
            )
            _last_recovery_plan = None  # Track for outcome recording

            # Main agentic loop
            while self.is_running and not planner.is_complete():
                try:
                    # === STEP 0.5: PRE-ACTION WINDOW VALIDATION ===
                    try:
                        validation = self.window_validator.validate_before_action(
                            action={},  # Pre-observe check
                        )
                        if validation.geometry_changed:
                            print(f"[UIAgent:WindowCheck] {validation.diagnostics}")
                            # Auto-normalize if window was resized
                            if validation.window_resized and validation.current_geometry:
                                cur_geo = validation.current_geometry
                                if not cur_geo.is_maximized:
                                    print(f"[UIAgent:WindowCheck] Auto-maximizing window {cur_geo.hwnd}")
                                    self.window_validator.normalize_window(cur_geo.hwnd)
                                    # Invalidate cached coords
                                    app_ctx = 'Desktop'
                                    if self.situational_awareness:
                                        try:
                                            ctx = self.situational_awareness.get_current_context()
                                            app_ctx = ctx.get('app_class', 'Desktop')
                                        except Exception:
                                            pass
                                    self.window_validator.invalidate_cached_coords(
                                        self.memory_agent, app_ctx
                                    )
                                    time.sleep(0.5)  # Let window settle
                    except Exception as e:
                        print(f"[UIAgent:WindowCheck] Error: {e}")

                    # === STEP 1: OBSERVE ===
                    timestamp = int(time.time() * 1000)
                    screenshot_filename = f"observe_{timestamp}.png"
                    screenshot_dir = str(self.memory_agent.short_term_dir / "screenshots")
                    screenshot_path = self.screen_observer.capture(
                        filename=os.path.join(screenshot_dir, screenshot_filename)
                    )
                
                    if not screenshot_path:
                        print("[UIAgent] Screenshot capture failed. Retrying...")
                        time.sleep(1)
                        continue
                
                    # === NEW: DYNAMIC HUD STATUS UPDATE ===
                    if self.live_streamer.is_streaming:
                        self.hud.update_status(True)
                    else:
                        # === Phase 7: Live API Reconnection Attempt ===
                        print("[UIAgent] Live Stream dropped. Attempting ONE reconnection...")
                        try:
                            self._stop_live_stream()
                            time.sleep(1)
                            self._start_live_stream()
                            time.sleep(5)
                            if self.live_streamer.is_streaming:
                                self.hud.update_status(True)
                                print("[UIAgent] Live Stream RECONNECTED successfully.")
                            else:
                                self.hud.update_status(False, fallback_active=True)
                                print("[UIAgent] Reconnection failed. Using static VLM fallback.")
                        except Exception as e:
                            print(f"[UIAgent] Reconnection error: {e}. Using static VLM fallback.")
                            self.hud.update_status(False, fallback_active=True)
                
                    # === UPDATE AWARENESS STACK ===
                    if self.situational_awareness:
                        try:
                            import cv2 # type: ignore
                            frame = cv2.imread(screenshot_path)
                            if frame is not None:
                                self.situational_awareness.build_world_model(frame)
                                
                                # Visualize structural boundaries on HUD (Old Style)
                                elements = self.edge_engine.detect_structural_elements(frame)
                                for el in elements[:50]:
                                    self.hud.add_rect(el.x, el.y, el.width, el.height, color=(255, 0, 255))
                        except Exception as e:
                            print(f"[UIAgent] Failed to update World Model or Edge View: {e}")


                
                    # === NEW: STEP 1.5 SUPERVISE BROWSER TOOL ===
                    if hasattr(self, 'browser_tool') and self.browser_tool.is_running:
                        self.hud.update_step(planner.current_step, "Supervising Browser Tool...")
                        print(f"[UIAgent Supervision] Browser Tool is running objective: '{self.browser_tool._current_objective}'.")
                    
                        if self.live_streamer.is_streaming:
                            # Ask the Live API to visually verify if the tool is making progress
                            supervision_prompt = (
                                f"The Browser Tool is currently automating: '{self.browser_tool._current_objective}'. "
                                "Look at the screen. Is it making progress, or does it look stuck/failed? "
                                "Respond with strictly JSON: {\"status\": \"progressing\" | \"stuck\" | \"failed\", \"reason\": \"what you observe\"}"
                            )
                        
                            try:
                                future = asyncio.run_coroutine_threadsafe(
                                    self.live_streamer.send_prompt(supervision_prompt),
                                    self._loop
                                )
                                # Wait briefly for feedback
                                future.result(timeout=10.0)
                            
                                # Parse result
                                def _supervision_callback(data: Dict[str, Any]):
                                    if "text_response" in data:
                                        import json
                                        try:
                                            res = json.loads(data["text_response"])
                                            status = res.get("status", "progressing")
                                            reason = res.get("reason", "")
                                            print(f"[UIAgent Supervision] Live API says: {status.upper()} - {reason}")
                                        
                                            if status in ["stuck", "failed"]:
                                                print(f"[UIAgent Supervision] Deciding to ABORT Browser Tool due to visual failure.")
                                                self.browser_tool.abort()
                                                planner.step_history.append({
                                                    "step_num": planner.current_step,
                                                    "action_type": "delegate_to_browser_tool",
                                                    "success": False,
                                                    "verification_note": f"Aborted by Supervisor: {reason}"
                                                })
                                        except Exception: pass
                            
                                self.live_streamer.set_callback(_supervision_callback)
                                # Let the stream process the callback
                                time.sleep(2)
                            except Exception as e:
                                print(f"[UIAgent Supervision] Live API check failed: {e}")
                    
                        # If it's still running, skip standard Agentic Planning and wait
                        if self.browser_tool.is_running:
                            time.sleep(3.0)
                            continue
                        
                    # If we get here, the BrowserTool either finished naturally, was aborted, or isn't running.
                
                    # === STEP 1.8: LOOP RECOVERY — execute pending recovery actions ===
                    if _last_recovery_plan and hasattr(_last_recovery_plan, 'recovery_actions') and _last_recovery_plan.recovery_actions:
                        for rec_action in _last_recovery_plan.recovery_actions:
                            print(f"[UIAgent:Recovery] Executing recovery: {rec_action.get('action')}")
                            self._execute_action(rec_action)
                        # Check if recovery succeeded (next step will tell us)
                        _last_recovery_plan = None

                    # === STEP 2: DECIDE ===
                    self.hud.update_step(planner.current_step + 1, "Thinking...")

                    # Inject window state into planner for coordinate grounding
                    try:
                        win_val = self.window_validator.validate_before_action(action={})
                        if win_val.current_geometry:
                            planner._window_state = {
                                "rect": win_val.current_geometry.rect,
                                "is_maximized": win_val.current_geometry.is_maximized,
                                "title": win_val.current_geometry.title,
                                "hwnd": win_val.current_geometry.hwnd,
                            }
                    except Exception:
                        planner._window_state = None  # type: ignore

                    next_action = planner.decide_next_step(screenshot_path)
                
                    if next_action is None:
                        # Goal complete or failed
                        break
                
                    action_desc = f"{next_action.get('action', '?')} on {next_action.get('target', next_action.get('text', ''))}"
                    self.hud.update_action(action_desc)
                    self.hud.update_step(
                        planner.current_step,
                        next_action.get('reasoning', action_desc)
                    )
                    
                    # === NEW: TELEGRAM CHECKPOINT UPDATE ===
                    if send_callback and planner.current_step > 0:
                        try:
                            step_msg = f"🔄 **Step {planner.current_step}**: {next_action.get('reasoning', action_desc)}\n└ Action: `{action_desc}`"
                            send_callback(step_msg)
                        except Exception as e:
                            print(f"[UIAgent] Failed to send Telegram callback: {e}")
                
                    # === NEW: PROACTIVE INTENT FEEDBACK ===
                    if self.live_streamer.is_streaming:
                        intent_msg = f"Intent: {action_desc}. Reasoning: {next_action.get('reasoning', '')}"
                        self.live_streamer.send_step_feedback(intent_msg, self._loop)

                    # === Phase 10: HUMAN-IN-THE-LOOP CLARIFICATION ===
                    if next_action.get('action') == 'request_human_help':
                        q = next_action.get('question', 'I need help.')
                        props = next_action.get('proposals', [])
                        
                        self.hud.update_action("WAITING FOR HUMAN INPUT...")
                        self.hud.update_step(planner.current_step, f"Question: {q}")
                        print(f"\n[UIAgent:HITL] 🖐️ AGENT PAUSED FOR CLARIFICATION")
                        print(f"Question: {q}")
                        if props:
                            print(f"Options: {', '.join(props)}")
                            
                        # Send to callback if available (e.g., Telegram)
                        if send_callback:
                            try:
                                msg = f"🖐️ **I need your help to proceed.**\n\n❓ {q}"
                                if props:
                                    msg += "\n\nOptions:\n" + "\n".join([f"- {p}" for p in props])
                                msg += "\n\nPlease reply with your answer."
                                send_callback(msg)
                            except Exception as e:
                                print(f"[UIAgent:HITL] Callback failed: {e}")
                                
                        # Wait for human input to arrive
                        human_answer = None
                        print("[UIAgent:HITL] Polling for response...")
                        
                        # Clear any stale interrupts before waiting
                        planner.hitl_interrupts.clear()
                        
                        while True:
                            # 1. Check for async interrupts (e.g. from Telegram callback)
                            if planner.hitl_interrupts:
                                interrupt = planner.hitl_interrupts.pop(0)
                                human_answer = interrupt.get('feedback') or interrupt.get('message') or str(interrupt)
                                print(f"[UIAgent:HITL] Received remote answer: {human_answer}")
                                break
                                
                            # 2. Local fallback if no remote channel active and we're in a terminal
                            # For safety in automation environments, we don't always want blocking input()
                            # But if no callback is registered, we must block or fail.
                            if not send_callback:
                                print("\nNo remote callback registered. Falling back to local CLI input.")
                                human_answer = input("\nYour answer: ")
                                break
                                
                            time.sleep(1.0)
                            
                        # Resumption: Inject the answer into the planner's history so the VLM remembers it next step
                        if human_answer:
                            planner._record_step(
                                next_action,
                                success=True,
                                verification_note=f"Human answered: '{human_answer}'"
                            )
                            print(f"[UIAgent:HITL] Answer injected. Resuming execution loop.")
                            continue # Skip execution/verification and go straight back to Observe/Decide
                
                    # === Phase 7: COMPOUND ACTION HANDLER ===
                    if next_action.get('action') == 'compound':
                        print(f"[UIAgent] === COMPOUND ACTION: {next_action.get('reasoning', 'batch steps')} ===")
                        sub_steps = next_action.get('steps', [])
                        for i, sub_action in enumerate(sub_steps):
                            print(f"[UIAgent]   Sub-step {i+1}/{len(sub_steps)}: {sub_action.get('action')} {sub_action.get('target', sub_action.get('text', ''))}")
                            self.hud.update_action(f"[{i+1}/{len(sub_steps)}] {sub_action.get('action')} {sub_action.get('target', sub_action.get('text', ''))}")
                        
                            # Resolve coordinates for click sub-steps
                            if sub_action.get('target') and sub_action['action'] in ['click', 'double_click']:
                                coords = self.predictor.resolve_target_with_zoom(
                                    target=sub_action['target'],
                                    hint_coords=sub_action.get('hint_coords'),
                                    needs_zoom=sub_action.get('needs_zoom', False),
                                    live_streamer=self.live_streamer if self.live_streamer.is_streaming else None,
                                    event_loop=self._loop
                                )
                                sub_action.update(coords)
                                sub_action['is_global'] = True
                        
                            self._execute_action(sub_action)
                            time.sleep(0.3)  # Brief pause between sub-steps
                    
                        # Verify ONCE after all sub-steps complete
                        print("[UIAgent] Compound action complete. Verifying final state...")
                        time.sleep(1.0)  # Wait for UI to settle
                        after_timestamp = int(time.time() * 1000)
                        after_filename = f"verify_{after_timestamp}.png"
                        after_path = self.screen_observer.capture(
                            filename=os.path.join(screenshot_dir, after_filename)
                        )
                        if after_path:
                            success = planner.verify_step(next_action, after_path)
                            if not success:
                                self.hud.update_step(planner.current_step, "COMPOUND FAILED -- re-evaluating...")
                        else:
                            planner._record_step(next_action, success=True,
                                                verification_note="After-screenshot unavailable")
                    
                        # Skip the normal STEP 3-5 flow below
                        if self.live_streamer.is_streaming:
                            step_summary = planner.get_step_summary()
                            self.live_streamer.send_step_feedback(step_summary, self._loop)
                        time.sleep(0.5)
                        continue
                
                    # === STEP 3: RESOLVE COORDINATES ===
                    if next_action.get('target') and next_action['action'] in ['click', 'double_click', 'right_click']:
                        # Force high-precision zoom for small taskbar icons
                        target_str = str(next_action['target']).lower()
                        if 'taskbar' in target_str or 'system tray' in target_str:
                            print(f"[UIAgent] Target '{target_str}' flagged as TASKBAR. Forcing needs_zoom=True.")
                            next_action['needs_zoom'] = True

                        # Phase 8: Proactive Template Matching — cross-validate coordinates
                        cv_match_coords = None
                        if self.cv_pipeline and hasattr(self.memory_agent, 'recall_template'):
                            try:
                                target_label = next_action['target']
                                app_ctx = 'Desktop'
                                if self.situational_awareness:
                                    ctx = self.situational_awareness.get_current_context()
                                    app_ctx = ctx.get('app_class', 'Desktop')
                                # For taskbar, context might be Desktop or taskbar class
                                if 'taskbar' in target_str:
                                    app_ctx = 'Shell_TrayWnd' # Standard Windows taskbar class
                                atlas_template = self.memory_agent.recall_template(app_ctx, target_label)
                                if atlas_template is not None:
                                    current_frame = self.screen_observer.capture_as_numpy()
                                    if current_frame is not None:
                                        tm_result = self.cv_pipeline.match_template_multiscale(
                                            current_frame, atlas_template, threshold=0.70,
                                            target_label=target_label, mode="PROACTIVE"
                                        )
                                        if tm_result and tm_result.get('matched'):
                                            cv_match_coords = tm_result.get('coords')  # (x, y)
                                            cv_conf = tm_result.get('confidence', 0)
                                            print(f"[UIAgent:CV] Proactive template match for '{target_label}': coords={cv_match_coords}, conf={cv_conf:.2f}")
                                            next_action['cv_confidence'] = cv_conf
                                        else:
                                            print(f"[UIAgent:CV] No template match for '{target_label}' — using VLM coordinates.")
                            except Exception as e:
                                print(f"[UIAgent:CV] Proactive template match error: {e}")
                    
                        coords = self.predictor.resolve_target_with_zoom(
                            target=next_action['target'],
                            hint_coords=next_action.get('hint_coords'),
                            needs_zoom=next_action.get('needs_zoom', False),
                            live_streamer=self.live_streamer if self.live_streamer.is_streaming else None,
                            event_loop=self._loop
                        )
                    
                        # Phase 8: If CV template match found coords, cross-validate OR rescue
                        if cv_match_coords and coords.get('x') and coords.get('y'):
                            vlm_x, vlm_y = coords['x'], coords['y']
                            cv_x, cv_y = cv_match_coords
                            dist = ((vlm_x - cv_x)**2 + (vlm_y - cv_y)**2)**0.5
                            if dist > 80:  # pixels
                                print(f"[UIAgent:CV] WARNING: VLM coords ({vlm_x},{vlm_y}) differ from CV coords ({cv_x},{cv_y}) by {dist:.0f}px. Using CV coords as anchor.")
                                # Prefer CV match if confidence is high enough
                                if next_action.get('cv_confidence', 0) > 0.85:
                                    coords['x'] = cv_x
                                    coords['y'] = cv_y
                                    print(f"[UIAgent:CV] Using CV template coords (high confidence).")
                            else:
                                print(f"[UIAgent:CV] VLM and CV coords agree (dist={dist:.0f}px). Good.")
                        elif cv_match_coords and not (coords.get('x') and coords.get('y')):
                            # RESCUE: CV match found something but VLM/Zoom failed
                            print(f"[UIAgent:CV] RESCUE: VLM/Zoom failed but CV template found target at {cv_match_coords}. Using CV coords.")
                            coords['x'], coords['y'] = cv_match_coords
                            coords['source'] = 'cv_template_rescue'
                    
                        next_action.update(coords)
                        next_action['resolved_coords'] = coords
                        next_action['is_global'] = True
                        print(f"[UIAgent] Resolved coordinates: ({coords.get('x')}, {coords.get('y')})")

                        # --- HUD-SYNCHRONIZED REFLEX VERIFICATION ---
                
                    # === STEP 4: EXECUTE ===
                    print(f"[UIAgent] Executing: {next_action.get('action')} -- {next_action.get('reasoning', '')}")
                    self._execute_action(next_action)
                
                    # === STEP 5: SMART VERIFY (with Observable Fast Feedback) ===
                    skip_verify_actions = ['type', 'press', 'hotkey', 'scroll', 'wait']
                    if next_action.get('action') in skip_verify_actions:
                        print(f"[UIAgent] Smart skip: No visual verification needed for '{next_action['action']}'")
                        planner._record_step(next_action, success=True,
                                            verification_note=f"Skipped verification (trivial action: {next_action['action']})")
                    else:
                        after_timestamp = int(time.time() * 1000)
                        after_filename = f"verify_{after_timestamp}.png"
                        after_path = self.screen_observer.capture(
                            filename=os.path.join(screenshot_dir, after_filename)
                        )

                        # ── Observable-Based Fast Feedback Shortcut ──
                        # Try fast CV check first; skip expensive VLM if it passes
                        fast_verified = False
                        if after_path and self.anticipatory_observer:
                            try:
                                import cv2 as _cv2  # type: ignore
                                # Determine observable to use
                                obs_def = next_action.get('expected_observable')
                                if not obs_def:
                                    obs_def = self.anticipatory_observer.select_observable(
                                        next_action, self.process_context
                                    )
                                if obs_def:
                                    before_frame = _cv2.imread(screenshot_path)
                                    after_frame = _cv2.imread(after_path)
                                    if before_frame is not None and after_frame is not None:
                                        obs_result = self.anticipatory_observer.verify_observable(
                                            obs_def, before_frame, after_frame, next_action
                                        )
                                        if obs_result.get('success'):
                                            print(f"[UIAgent:FastFeedback] Observable check PASSED "
                                                  f"({obs_result.get('method', 'CV')}). Skipping VLM verify.")
                                            planner._record_step(
                                                next_action, success=True,
                                                verification_note=f"Fast CV verified ({obs_result.get('method', 'CV')})"
                                            )
                                            fast_verified = True
                                            # Record successful window state
                                            if hasattr(self, 'window_validator'):
                                                try:
                                                    fg_hwnd = next_action.get('source_hwnd')
                                                    if fg_hwnd:
                                                        self.window_validator.record_successful_action(fg_hwnd)
                                                except Exception:
                                                    pass
                                        elif obs_result.get('request_fidelity') == 'HIGH':
                                            print(f"[UIAgent:FastFeedback] Observable requests VLM upgrade: "
                                                  f"{obs_result.get('reason')}")
                                            # Fall through to VLM verification
                                        else:
                                            print(f"[UIAgent:FastFeedback] Observable check FAILED. "
                                                  f"Using VLM verification.")
                            except Exception as e:
                                print(f"[UIAgent:FastFeedback] Error: {e}")

                        if not fast_verified and after_path:
                            success = planner.verify_step(next_action, after_path)
                        
                            if not success:
                                self.hud.update_step(planner.current_step, "FAILED -- re-evaluating...")
                            
                                # Phase 10: Root-Cause Classification (Misclick vs Bound Mismatch)
                                if self.situational_awareness and next_action.get('action') in ['click', 'double_click']:
                                    try:
                                        current_fg = self.situational_awareness.os.get_foreground_window()
                                        if current_fg and current_fg.hwnd != next_action.get('source_hwnd'):
                                            # If we were targeting an app and a DIFFERENT one is now foreground, it's likely a misclick
                                            print(f"[UIAgent:Failure] Detection: Foreground changed from target. Potential MISCLICK_ADJACENT.")
                                            if planner.step_history:
                                                planner.step_history[-1]['failure_category'] = 'MISCLICK_ADJACENT'
                                    except Exception: pass

                                # Phase 10.5: Post-Failure Diagnostic Snip
                                if hasattr(self, 'cursor_verifier') and self.cursor_verifier and next_action.get('action') in ['click', 'double_click']:
                                    target = next_action.get('target', 'unknown')
                                    resolved = next_action.get('resolved_coords', {})
                                    rx, ry = resolved.get('x'), resolved.get('y')
                                    if rx and ry:
                                        self.cursor_verifier.diagnose_failure(
                                            rx, ry, target, action_performed=next_action.get('action')
                                        )

                                # Phase 9: AUTO-CLOSEUP ZOOM RETRY for failed clicks
                                # If a click failed and we haven't already zoomed, auto-retry with zoom
                                if next_action.get('action') in ['click', 'double_click'] and not next_action.get('needs_zoom'):
                                    target = next_action.get('target', '')
                                    print(f"[UIAgent:Zoom] Click failed on '{target}'. Auto-triggering closeup zoom retry...")
                                    self.hud.update_action(f"ZOOM RETRY: {target}")
                                    try:
                                        zoom_coords = self.predictor.resolve_target_with_zoom(
                                            target=target,
                                            hint_coords=next_action.get('hint_coords'),
                                            needs_zoom=True,  # Force zoom
                                            live_streamer=self.live_streamer if self.live_streamer.is_streaming else None,
                                            event_loop=self._loop
                                        )
                                        if zoom_coords.get('x') and zoom_coords.get('y'):
                                            print(f"[UIAgent:Zoom] Zoom resolved: ({zoom_coords['x']}, {zoom_coords['y']}). Retrying click...")
                                            self.mouse.move_and_click(zoom_coords['x'], zoom_coords['y'], human_like=True)
                                            time.sleep(0.5)
                                            # Quick re-verify
                                            retry_path = self.screen_observer.capture(
                                                filename=os.path.join(screenshot_dir, f"zoom_retry_{after_timestamp}.png")
                                            )
                                            if retry_path:
                                                retry_success = planner.verify_step(next_action, retry_path)
                                                if retry_success:
                                                    print(f"[UIAgent:Zoom] Zoom retry SUCCEEDED!")
                                                    # Override the failure record
                                                    if planner.step_history:
                                                        planner.step_history[-1]['success'] = True
                                                        planner.step_history[-1]['verification_note'] = 'Succeeded on zoom retry'
                                    except Exception as e:
                                        print(f"[UIAgent:Zoom] Auto-zoom retry error: {e}")
                            else:
                                # Phase 9: SPECULATIVE SNIPPING on successful click
                                # Save the target element as a template for future fast matching
                                if next_action.get('action') in ['click', 'double_click'] and next_action.get('target'):
                                    try:
                                        resolved = next_action.get('resolved_coords', {})
                                        rx, ry = resolved.get('x'), resolved.get('y')
                                        if rx and ry and self.cv_pipeline and hasattr(self.memory_agent, 'store_template'):
                                            current_frame = self.screen_observer.capture_as_numpy()
                                            if current_frame is not None:
                                                # Snip a 30x30 region around the click point
                                                snip_size = 15
                                                y1 = max(0, ry - snip_size)
                                                y2 = min(current_frame.shape[0], ry + snip_size)
                                                x1 = max(0, rx - snip_size)
                                                x2 = min(current_frame.shape[1], rx + snip_size)
                                                snip = current_frame[y1:y2, x1:x2]
                                                if snip.size > 0:
                                                    app_ctx = 'Desktop'
                                                    if self.situational_awareness:
                                                        ctx = self.situational_awareness.get_current_context()
                                                        app_ctx = ctx.get('app_class', 'Desktop')
                                                    self.memory_agent.store_template(app_ctx, next_action['target'], snip)
                                                    print(f"[UIAgent:Snip] Speculative snip saved for '{next_action['target']}' at ({rx},{ry})")
                                    except Exception as e:
                                        print(f"[UIAgent:Snip] Speculative snip error: {e}")
                        else:
                            planner._record_step(next_action, success=True, 
                                                verification_note="After-screenshot unavailable")
                
                    # === STEP 6: MULTI-SIGNAL FEEDBACK ===
                    # Phase 8: Always run BOTH live API feedback AND static VLM feedback in parallel
                    step_summary = planner.get_step_summary()
                
                    # Live API feedback (streaming context)
                    if self.live_streamer.is_streaming:
                        self.live_streamer.send_step_feedback(step_summary, self._loop)
                
                    # Static VLM parallel feedback for planning intelligence
                    # This runs ALWAYS, even when live API is active, to provide richer context
                    try:
                        after_img_path = None
                        if planner.step_history:
                            last_step = planner.step_history[-1]
                            verif_note = last_step.get('verification_note', '')
                            # Only run parallel VLM feedback if the step was a significant action
                            if last_step.get('action_type') in ['click', 'double_click', 'compound', 'scroll']:
                                vlm_screenshot = self.screen_observer.capture(
                                    filename=os.path.join(screenshot_dir, f"vlm_feedback_{int(time.time()*1000)}.png")
                                )
                                if vlm_screenshot and self.vision:
                                    vlm_feedback = self.vision.describe_screen_state(
                                        vlm_screenshot,
                                        context=f"After action: {last_step.get('action_type')} on '{last_step.get('target', '')}'. Goal: {planner.goal}"
                                    )
                                    if vlm_feedback:
                                        print(f"[UIAgent:VLM] Parallel feedback: {vlm_feedback[:120]}...")
                                        # Inject VLM insight into planner history for richer context
                                        last_step['vlm_feedback'] = vlm_feedback[:300]
                    except Exception as e:
                        print(f"[UIAgent:VLM] Parallel feedback error: {e}")
                
                    # === NEW: ADAPTIVE FEEDBACK INTERVAL ===
                    # Calculate sleep duration based on step complexity and visual change
                    base_sleep = 0.2
                    if planner.step_history:
                        last_step = planner.step_history[-1]
                        # If action involved a dialog or major change, give the system more time to settle
                        verif_note = last_step.get("verification_note", "").lower()
                        if "dialog" in verif_note or "modal" in verif_note or "major" in verif_note:
                            base_sleep = 2.0
                            print("[UIAgent] Adaptive Interval: Extended wait for dialog/modal rendering.")
                        elif last_step.get("action_type") in ["delegate_to_browser_tool", "wait"]:
                             base_sleep = 1.0
                
                    time.sleep(base_sleep)
                
                except KeyboardInterrupt:
                    print("\n[UIAgent] Interrupted by user.")
                    break
                except Exception as e:
                    print(f"[UIAgent] Error in agentic loop: {e}")
                    import traceback
                    traceback.print_exc()
                # Finalize episode for Phase 13 learning
                if hasattr(self, 'learning_loop'):
                    self.learning_loop.finalize_episode(
                        task=planner.goal,
                        app_name=planner.situational_awareness.get_current_context().get("app_name", "Desktop") if planner.situational_awareness else "Desktop",
                        app_class=planner.situational_awareness.get_current_context().get("app_class", "") if planner.situational_awareness else "",
                        overall_success=planner.is_complete() and not planner._is_failed
                    )
                
        finally:
            # === GUARANTEED CLEANUP ===
            print("[UIAgent] Performing cleanup (stopping HUD and stream)...")
            self._stop_live_stream()
            
            try:
                if hasattr(self, 'hud') and self.hud:
                    self.hud.update_goal("Task complete.")
                    self.hud.update_action("Idle")
                    self.hud.update_status(False)
                    self.hud.update_step(0, "Done")
                    # Briefly show "Done" before destroying the window
                    import time as _t
                    _t.sleep(0.5)
                    self.hud.stop()
            except Exception as e:
                print(f"[UIAgent] Error during HUD cleanup: {e}")
                
            self.is_running = False
            self.planner = None  # Reset for next task

        return planner.get_status()


    def _resolve_target_set_of_mark(self, target_label: str, frame: Any) -> Optional[Dict[str, Any]]:
        """
        Specialized resolution for dense UI areas (Taskbar).
        Identifies candidates, labels them sequentially on the HUD, 
        and uses VLM ID selection for precise targeting.
        """
        print(f"[UIAgent:SetOfMark] Resolving target: '{target_label}'")
        diag_dir = os.path.join(str(self.memory_agent.short_term_dir), "diagnostics")
        os.makedirs(diag_dir, exist_ok=True)
        
        # 1. Detect structural elements
        elements = self.edge_engine.detect_structural_elements(frame, min_area=100)
        
        # 2. Filter for Taskbar ROI and icon sizes
        h, w = frame.shape[:2]
        taskbar_y_lower = int(h * 0.93)
        taskbar_y_upper = int(h * 0.98)
        
        taskbar_candidates = []
        for el in elements:
            is_in_icon_row = taskbar_y_lower < el.y < taskbar_y_upper
            is_icon_sized = 25 < el.width < 80 and 25 < el.height < 80
            aspect_ratio = el.width / float(el.height) if el.height > 0 else 0
            
            if is_in_icon_row and is_icon_sized and 0.8 < aspect_ratio < 1.25:
                # Avoid Tray icons (usually further right)
                if el.x > w * 0.8: continue
                
                # Deduplication
                is_duplicate = False
                for existing in taskbar_candidates:
                    if abs(existing.x - el.x) < 40 and abs(existing.y - el.y) < 20:
                        is_duplicate = True
                        break
                if not is_duplicate:
                    taskbar_candidates.append(el)
        
        if not taskbar_candidates:
            print("[UIAgent:SetOfMark] No taskbar candidates found.")
            return None
            
        taskbar_candidates.sort(key=lambda e: (e.x, e.y))
        print(f"[UIAgent:SetOfMark] Labeled {len(taskbar_candidates)} candidates.")
        
        # 3. Sequential HUD Labeling
        self.hud.update_action(f"Labeling icons [Set-of-Mark]...")
        marked_frame = frame.copy()
        for i, el in enumerate(taskbar_candidates):
            label = str(i)
            # HUD feedback
            self.hud.add_rect(el.x, el.y, el.width, el.height, label=label, color=(255, 0, 255))
            time.sleep(0.3)
            
            # Draw on screenshot for VLM
            cv2.rectangle(marked_frame, (el.x, el.y), (el.x + el.width, el.y + el.height), (255, 0, 255), 2)
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            cv2.rectangle(marked_frame, (el.x, el.y - 25), (el.x + tw + 10, el.y), (255, 255, 255), -1)
            cv2.putText(marked_frame, label, (el.x + 5, el.y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
            
        marked_ss_path = os.path.join(diag_dir, "vlm_set_of_mark.png")
        cv2.imwrite(marked_ss_path, marked_frame)
        
        # 4. VLM ID Selection
        self.hud.update_action("VLM Picking Target ID...")
        prompt = (
            f"Identify the box number representing the '{target_label}' icon on the taskbar. "
            "Respond ONLY with the number in brackets, e.g., [11]."
        )
        res_str = self.vision.describe_screen_state(marked_ss_path, context=prompt)
        print(f"[UIAgent:SetOfMark] VLM result: {res_str}")
        
        match = re.search(r"\[?(\d+)\]?", res_str)
        if not match:
            print("[UIAgent:SetOfMark] No ID found in VLM response.")
            return None
            
        target_id = int(match.group(1))
        if target_id >= len(taskbar_candidates):
            print(f"[UIAgent:SetOfMark] Invalid ID: {target_id}")
            return None
            
        best_el = taskbar_candidates[target_id]
        cx, cy = best_el.x + best_el.width // 2, best_el.y + best_el.height // 2
        
        return {
            "x": cx, 
            "y": cy, 
            "target_id": target_id, 
            "element": best_el,
            "source": "set_of_mark"
        }

    def _execute_action(self, action: Dict[str, Any]):
        """Executes a single logical action, predicting parameters and evaluating success."""
        
        target = action.get('target', '')
        target_app = action.get('app_window')
        context = target_app if target_app else 'Desktop'
        timestamp = int(time.time()*1000)
        
        # Inject the live streamer into the action context for the predictor
        action['live_streamer'] = self.live_streamer
        
        # Phase 10: Inject explicit window context into the action for grounding
        if self.situational_awareness:
             try:
                 ctx = self.situational_awareness.get_current_context()
                 window_info = [
                     f"ACTIVE_APP: {ctx.get('app_name', 'Unknown')}",
                     f"WINDOW_TITLE: {ctx.get('window_title', 'Unknown')}",
                     f"WINDOW_BOUNDS: {ctx.get('rect', 'Unknown')}",
                     f"IS_MAXIMIZED: {ctx.get('is_maximized', 'Unknown')}"
                 ]
                 action['window_context'] = " | ".join(window_info)
                 print(f"[UIAgent] Grounding Prediction with Context: {action['window_context']}")
             except Exception as e:
                 print(f"[UIAgent] Failed to get window context: {e}")
        
        self.hud.update_action(f"{action.get('action')} on {target}")
        
        # 1. Prediction with skip-perception hint
        # We try a 'dry run' of prediction to see if we can skip the screenshot
        params = None
        # Actions that don't require visual target prediction
        if action['action'] in ['wait', 'type', 'press', 'hotkey'] or not target:
            params = {"predicted_as": "no_target"}
        
        if params is None:
            if not self.live_streamer.is_streaming:
                 params = self.predictor.predict_action_parameters(action, screen_image_path=None)
        
        state_before = None

        if params is not None and params.get("predicted_as") == "invariant":
            print(f"[UIAgent] Optimization: Skipping 'before' screenshot for invariant '{target}' in stable context.")
        else:
            # Observe Before State: If action specifies an app window, capture just that
            if target_app:
                print(f"[UIAgent] Capturing background buffer for app '{target_app}'")
                filename = f"before_{target_app}_{timestamp}.png"
                full_path = str(self.memory_agent.short_term_dir / "screenshots" / filename)
                state_before = self.window_observer.capture_window_by_title(target_app, filename=full_path)
                # Fallback to full screen if window not found
                if not state_before:
                    filename = f"before_fallback_{timestamp}.png"
                    full_path = str(self.memory_agent.short_term_dir / "screenshots" / filename)
                    state_before = self.screen_observer.capture(filename=full_path)
            else:
                filename = f"before_{timestamp}.png"
                full_path = str(self.memory_agent.short_term_dir / "screenshots" / filename)
                state_before = self.screen_observer.capture(filename=full_path)
            
            # --- Phase 14: Sequential Set-of-Mark targeting for Taskbar ---
            target_str = str(target).lower()
            if ('taskbar' in target_str or 'task bar' in target_str) and action['action'] in ['click', 'double_click']:
                if state_before:
                    import cv2 as _cv2 # type: ignore
                    frame_before = _cv2.imread(state_before)
                    if frame_before is not None:
                        som_params = self._resolve_target_set_of_mark(target, frame_before)
                        if som_params:
                            params = som_params
                            action['is_global'] = True # Set-of-Mark returns global coords
                            print(f"[UIAgent] Set-of-Mark resolved target ID {som_params.get('target_id')} at ({params['x']}, {params['y']})")

            # Re-predict with actual image (or trigger live async query natively if streaming)
            # SKIP if we already have grounded coords or if it's a no_target action
            if not params and (params is None or params.get("predicted_as") != "no_target"):
                
                # If streaming, we must block on the Async result from the predictor bridge
                if self.live_streamer.is_streaming:
                     # This is slightly tricky since _execute_action is sync, but we use threadsafe futures
                     import concurrent.futures
                     future = asyncio.run_coroutine_threadsafe(
                         self.predictor._query_live_api_with_zoom(target, action, self.live_streamer), 
                         self._loop
                     )
                     try:
                         # Wait for the WebSocket trip
                         params = future.result(timeout=20.0)
                     except Exception as e:
                         print(f"[UIAgent] Live prediction failed/timeout: {e}. Falling back to static.")
                         params = self.predictor.predict_action_parameters(action, state_before)
                else:
                     params = self.predictor.predict_action_parameters(action, state_before)

        
        # 2. Act (Coordinate Translation & Focus)
        import win32gui # type: ignore # win32con is not needed here anymore
        hwnd = None
        if target_app:
             hwnd = self.window_manager.get_hwnd_by_title(target_app)
             if hwnd:
                  self.window_manager.focus_window(hwnd)
                  # Phase 10: Mandatory Maximization Policy for complex tasks
                  print(f"[UIAgent] Auto-maximizing window {hwnd} for consistent spatial layout.")
                  self.window_manager.maximize(hwnd)
                  action['source_hwnd'] = hwnd # Store for misclick detection in verification
                  
                  # Coordinate translation logic: 
                  # ONLY translate if coords are relative (NOT marked as is_global)
                  if isinstance(params, dict) and 'x' in params and 'y' in params and not action.get('is_global'):
                      try:
                          client_point = win32gui.ClientToScreen(hwnd, (0, 0))
                          params['x'] = params['x'] + client_point[0]
                          params['y'] = params['y'] + client_point[1]
                          print(f"[UIAgent] Translated relative to global desktop ({params['x']}, {params['y']})")
                      except Exception: pass

        print(f"[UIAgent] Executing {action['action']} with {params}")
        
        # Action Dispatcher
        if action['action'] in ['click', 'double_click']:
             if isinstance(params, dict) and 'x' in params and 'y' in params:
                 target = action.get('target', 'unknown element')
                 expected_type = action.get('expected_element_type')
                 
                 # 1. Move to position
                 self.mouse.move_to(params['x'], params['y'], human_like=True)
                 
                 # 2. Get hover state
                 cursor_info = None
                 cursor_valid = None
                 if self.cursor_monitor:
                     cursor_type = self.cursor_monitor.get_cursor_type()
                     cursor_info = {'cursor_type': cursor_type.name}
                     if expected_type:
                         cursor_valid = self.cursor_monitor.validate_hover(expected_type)
                     action['cursor_feedback'] = cursor_info
                     print(f"[UIAgent:Cursor] Cursor was '{cursor_type.name}' at point ({params['x']}, {params['y']})")
                     
                 # 3. Reflex Verification (VLM Position Snip)
                 click_aborted = False
                 if params.get('x') and params.get('y'):
                     cx, cy = int(params['x']), int(params['y'])
                    
                     # Adaptive HUD: Snap to edges for better visual feedback
                     final_x, final_y, final_w, final_h = cx - 32, cy - 32, 64, 64 # Initial guess
                     try:
                         screen_img = self.screen_observer.capture_as_cv2()
                         if screen_img is not None:
                             # Use refined EdgePerceptionEngine for precise target snapping
                             elements = self.edge_engine.detect_structural_elements(screen_img)
                             if elements:
                                 # Find element closest to cx, cy
                                 best_el = None
                                 min_dist = 999999.0
                                 for el in elements:
                                     if el is None: continue
                                     ecx, ecy = el.x + el.width // 2, el.y + el.height // 2 # type: ignore
                                     dist = float(((ecx - cx)**2 + (ecy - cy)**2)**0.5)
                                     if dist < min_dist:
                                         min_dist = dist
                                         best_el = el
                                 
                                 if best_el is not None and min_dist < 100.0:
                                     self.edge_engine.compute_stable_cid(best_el, screen_img.shape[:2])
                                     final_x, final_y, final_w, final_h = best_el.x, best_el.y, best_el.width, best_el.height # type: ignore
                                     target = f"{target} [{best_el.cid}]" # type: ignore
                                     print(f'[UIAgent] Edge-aware HUD: Snapped to {best_el.cid} at ({final_x}, {final_y}, {final_w}x{final_h})') # type: ignore
                                 else:
                                     # Fallback to adaptive snap if no structural element is precisely under cursor
                                     final_x, final_y, final_w, final_h = adapt_rect_to_edges(screen_img, cx - 32, cy - 32, 64, 64)
                             else:
                                 # Fallback to simple adaptive snap if no structural elements found
                                 final_x, final_y, final_w, final_h = adapt_rect_to_edges(screen_img, cx - 32, cy - 32, 64, 64)
                     except Exception as e:
                         print(f'[UIAgent] Adaptive HUD failed: {e}')
                    
                     # Draw Yellow Verification box on HUD (Adaptive)
                     self.hud.add_rect(final_x, final_y, final_w, final_h, label=f'Verifying: {target}', color=(255, 255, 0))
                     time.sleep(0.5)
                     if hasattr(self, 'cursor_verifier') and self.cursor_verifier:
                         should_trigger, triggers = self.cursor_verifier.should_trigger(
                             target=target,
                             prediction_source=params.get('source', 'unknown'),
                             cursor_valid=cursor_valid,
                             is_small_target=action.get('is_small', False),
                             was_zoom_resolved=action.get('needs_zoom', False),
                             confidence=params.get('confidence', 1.0)
                         )
                         if should_trigger:
                             # Capture snip for VLM
                             snip_path = self.screen_observer.capture(
                                 filename=os.path.join(self.memory_agent.short_term_dir, "diagnostics", f"action_reflex_{int(time.time()*1000)}.png")
                             )
                            
                             if snip_path:
                                # Async Audit
                                import concurrent.futures
                                future = asyncio.run_coroutine_threadsafe(
                                    self.predictor.verify_target_with_vlm(
                                        target=target,
                                        snip_path=snip_path,
                                        predicted_coords=(cx, cy),
                                        cv_coords=None 
                                    ),
                                    self._loop
                                )
                                try:
                                    is_verified = future.result(timeout=25.0)
                                    if not is_verified:
                                        print(f"[UIAgent:Reflex] ABORTING CLICK. VLM rejected location.")
                                        self.hud.add_rect(final_x, final_y, final_w, final_h, label="REJECTED", color=(255, 0, 0))
                                        click_aborted = True
                                        time.sleep(1.0)
                                    else:
                                        self.hud.add_rect(final_x, final_y, final_w, final_h, label="VERIFIED", color=(0, 255, 0))
                                except Exception as e:
                                    print(f"[UIAgent:Reflex] Audit error in _execute_action: {e}")
                 
                 # 4. Execute click
                 if not click_aborted:
                     self.mouse._sleep_random(0.2)
                     if action['action'] == 'click':
                         self.mouse.click()
                     else:
                         self.mouse.double_click()
             else:
                 print(f"[UIAgent] WARNING: Could not resolve coordinates for click on {action.get('target')}")

        elif action['action'] == 'drag':
             # Resolve destination
             dest_label = action.get('destination')
             dest_params = self.predictor.predict_action_parameters({"action": "click", "target": dest_label}, state_before)
             if isinstance(params, dict) and 'x' in params and 'y' in params and isinstance(dest_params, dict) and 'x' in dest_params and 'y' in dest_params:
                 self.mouse.drag_to(dest_params['x'], dest_params['y'], source_x=params['x'], source_y=params['y'])

        elif action['action'] == 'drag_to_resize':
             if hwnd and 'edge' in action and 'delta_px' in action:
                 import win32gui # type: ignore
                 rect = win32gui.GetWindowRect(hwnd)
                 x_left, y_top, x_right, y_bottom = rect
                 edge = action['edge']
                 delta = action['delta_px']
                 
                 # Determine start point on the edge, and target point after drag
                 source_x, source_y = 0, 0
                 target_x, target_y = 0, 0
                 
                 if edge == 'right':
                     source_x = x_right - 2 # slightly inside
                     source_y = y_top + (y_bottom - y_top) // 2
                     target_x = source_x + delta
                     target_y = source_y
                 elif edge == 'left':
                     source_x = x_left + 2
                     source_y = y_top + (y_bottom - y_top) // 2
                     target_x = source_x + delta
                     target_y = source_y
                 elif edge == 'bottom':
                     source_x = x_left + (x_right - x_left) // 2
                     source_y = y_bottom - 2
                     target_x = source_x
                     target_y = source_y + delta
                 elif edge == 'top':
                     source_x = x_left + (x_right - x_left) // 2
                     source_y = y_top + 2
                     target_x = source_x
                     target_y = source_y + delta
                     
                 if source_x and source_y:
                     print(f"[UIAgent] Resizing via drag on {edge} edge by {delta}px")
                     self.mouse.drag_to(target_x, target_y, source_x=source_x, source_y=source_y)

        elif action['action'] == 'type':
             if 'text' in action:
                  # Phase 7: clear_first - select all existing text before typing
                  if action.get('clear_first', False):
                      print("[UIAgent] clear_first=True: Sending Ctrl+A to clear field before typing")
                      self.keyboard.hotkey('ctrl', 'a')
                      time.sleep(0.1)
                  self.keyboard.type_text(action['text'])
                 
        elif action['action'] == 'scroll':
             clicks = action.get('amount', 300)
             direction = action.get('direction', 'down')
             self.mouse.scroll(clicks, direction)

        elif action['action'] == 'minimize':
             if hwnd: self.window_manager.minimize(hwnd)
             
        elif action['action'] == 'maximize':
             if hwnd: self.window_manager.maximize(hwnd)

        elif action['action'] == 'resize':
             if hwnd and 'width' in action and 'height' in action:
                  self.window_manager.resize(hwnd, action['width'], action['height'])
                  
        elif action['action'] == 'hotkey':
             if 'keys' in action:
                 self.keyboard.hotkey(*action['keys'])
                 
        elif action['action'] == 'press':
             if 'key' in action:
                 self.keyboard.press_key(action['key'])
                 
        elif action['action'] == 'wait':
             time.sleep(action.get('duration', 1.0))
             
        elif action['action'] == 'ask_swarm':
             question = action.get('question', 'Missing context')
             print(f"[UIAgent] Asking Swarm for elaboration: {question}")
             # Gracefully abort the UI task and bubble the question back to the Swarm
             if hasattr(self, 'planner') and self.planner:
                 self.planner._is_failed = True # type: ignore
                 self.planner._failure_reason = f"ASK_SWARM: {question}" # type: ignore

        elif action['action'] == 'focus_window':
             if 'hwnd' in action:
                  self.window_manager.focus_window(action['hwnd'])
             elif hwnd:
                  self.window_manager.focus_window(hwnd)

        elif action['action'] == 'remember_window':
             target_hwnd = action.get('hwnd') or hwnd
             tag = action.get('tag', 'Untagged Window')
             if target_hwnd:
                 if hasattr(self, 'planner') and self.planner:
                     self.planner.window_memory[target_hwnd] = tag # type: ignore
                     print(f"[UIAgent] Remembered window hwnd={target_hwnd} as '{tag}'")
                 else:
                     print("[UIAgent] WARNING: No planner instance available to remember window")
             else:
                 print("[UIAgent] WARNING: No hwnd provided to remember_window")

        elif action['action'] == 'right_click':
             # Right-click at resolved coordinates (e.g. for taskbar context menus)
             if isinstance(params, dict) and 'x' in params and 'y' in params:
                 self.mouse.move_to(params['x'], params['y'], human_like=True)
                 self.mouse._sleep_random(0.2)
                 self.mouse.right_click()
                 print(f"[UIAgent] Right-clicked at ({params['x']}, {params['y']})")
             elif action.get('x') and action.get('y'):
                 self.mouse.move_to(action['x'], action['y'], human_like=True)
                 self.mouse._sleep_random(0.2)
                 self.mouse.right_click()
                 print(f"[UIAgent] Right-clicked at ({action['x']}, {action['y']})")
             else:
                 print(f"[UIAgent] WARNING: Could not resolve coordinates for right_click on {action.get('target')}")

        elif action['action'] == 'minimize_window':
             target_keyword = action.get('target', '')
             if target_keyword:
                 target_hwnd = self.window_manager.get_hwnd_by_title(target_keyword)
                 if target_hwnd:
                     self.window_manager.minimize(target_hwnd)
                     print(f"[UIAgent] Minimized window matching '{target_keyword}' (hwnd={target_hwnd})")
                 else:
                     print(f"[UIAgent] No window found matching '{target_keyword}' to minimize")
             elif hwnd:
                 self.window_manager.minimize(hwnd)
                 print(f"[UIAgent] Minimized window hwnd={hwnd}")
             
        elif action['action'] == 'delegate_to_browser_tool':
             objective = action.get('objective', action.get('text', 'unknown objective'))
             print(f"[UIAgent] Delegating to Browser Tool for objective: {objective}")
             if hasattr(self, 'browser_tool'):
                 self.browser_tool.execute_objective(objective)
             else:
                 print("[UIAgent] Cannot delegate: BrowserTool not initialized.")

        # ── CV Tool Dispatchers (function-calling from planner) ──
        elif action['action'] == 'cv_template_match':
             label = action.get('target_label', action.get('target', ''))
             threshold = action.get('threshold', 0.80)
             print(f"[UIAgent:CV] Template match requested for '{label}' (threshold={threshold})")
             current_screen = self.screen_observer.capture_as_numpy()
             atlas_template = self.memory_agent.recall_template(context or 'Desktop', label) if hasattr(self.memory_agent, 'recall_template') else None
             if current_screen is not None and atlas_template is not None:
                 result = self.cv_pipeline.match_template_multiscale(
                     current_screen, atlas_template, threshold=threshold,
                     target_label=label, mode="ACTIVE",
                 )
                 action['cv_result'] = result or {"matched": False}
             else:
                 action['cv_result'] = {"matched": False, "reason": "no template in atlas"}

        elif action['action'] == 'cv_snip_element':
             label = action.get('element_label', '')
             bbox = action.get('bbox', [])
             print(f"[UIAgent:CV] Snip & save element '{label}' at {bbox}")
             if bbox and len(bbox) == 4:
                 current_screen = self.screen_observer.capture_as_numpy()
                 if current_screen is not None:
                     x, y, w, h = bbox
                     crop = current_screen[y:y+h, x:x+w]
                     if crop.size > 0:
                         self.memory_agent.store_template(context or 'Desktop', label, crop)
                         self.cv_pipeline.logger.log_snip_save(label, bbox, "planner_request", False, "ACTIVE")
                         action['cv_result'] = {"saved": True}
                     else:
                         action['cv_result'] = {"saved": False, "reason": "empty crop"}

        elif action['action'] == 'cv_fingerprint_compare':
             label = action.get('element_label', '')
             print(f"[UIAgent:CV] Fingerprint compare for '{label}'")
             action['cv_result'] = {"same_state": True, "similarity": 0.0, "note": "no stored fingerprint"}

        elif action['action'] == 'cv_stability_check':
             region = action.get('region', [])
             print(f"[UIAgent:CV] Stability check for region {region}")
             stability = self.cv_pipeline.classify_regions(self.screen_observer.capture_as_numpy() or __import__('numpy').zeros((100,100,3), dtype=__import__('numpy').uint8)) # type: ignore
             action['cv_result'] = {"classification": "STATIC", "stability_map_size": len(stability)}

        elif action['action'] == 'cv_embedding_search':
             desc = action.get('target_description', '')
             print(f"[UIAgent:CV] Embedding search for '{desc}'")
             action['cv_result'] = {"results": [], "note": "embedding search requires atlas data"}

        # ── Edge Correlation Tool Dispatchers ──
        elif action['action'] == 'cv_edge_detect':
             print("[UIAgent:CV] Running edge detection + CID assignment")
             current_screen = self.screen_observer.capture_as_numpy()
             if current_screen is not None and hasattr(self, 'edge_correlator'):
                 from coding_agent.ui_agent.perception.cv_pipeline import UIElement # type: ignore
                 elements = self.cv_pipeline.detect_ui_elements(current_screen)
                 elem_dicts = [{"x": e.x, "y": e.y, "width": e.width, "height": e.height,
                                "element_type": e.element_type, "label": e.label,
                                "confidence": e.confidence} for e in elements]
                 result = self.edge_correlator.full_correlate(current_screen, elem_dicts)
                 action['cv_result'] = {
                     "total_elements": result["total_elements"],
                     "edge_density_pct": result["edge_density_pct"],
                     "vlm_index": result["vlm_index"],
                     "structural_diff": result["structural_diff"],
                 }
             else:
                 action['cv_result'] = {"error": "edge_correlator not initialized"}

        elif action['action'] == 'cv_structural_map':
             print("[UIAgent:CV] Returning structural map")
             if hasattr(self, 'edge_correlator'):
                 summary = self.edge_correlator.get_registry_summary()
                 action['cv_result'] = summary
             else:
                 action['cv_result'] = {"error": "edge_correlator not initialized"}

        elif action['action'] == 'cv_find_by_cid':
             cid = action.get('cid', '')
             print(f"[UIAgent:CV] Looking up CID: {cid}")
             if hasattr(self, 'edge_correlator'):
                 ce = self.edge_correlator.find_element_by_cid(cid)
                 if ce:
                     action['cv_result'] = {
                         "found": True, "cid": cid,
                         "x": ce.x, "y": ce.y, "w": ce.width, "h": ce.height,
                         "center_x": ce.x + ce.width // 2,
                         "center_y": ce.y + ce.height // 2,
                         "label": ce.label, "type": ce.element_type,
                         "structural_context": ce.structural_context,
                     }
                 else:
                     action['cv_result'] = {"found": False, "cid": cid}
             else:
                 action['cv_result'] = {"error": "edge_correlator not initialized"}
             
        # ── Automation Artifact Dispatchers ──
        elif hasattr(self, 'automation_artifacts') and self.automation_artifacts.is_artifact_action(action['action']):
             print(f"[UIAgent:Artifact] Dispatching automation artifact: {action['action']}")
             result = self.automation_artifacts.execute(action)
             action['artifact_result'] = result
             if not result.get('success'):
                 print(f"[UIAgent:Artifact] Artifact failed: {result.get('error', 'unknown')}")
             else:
                 print(f"[UIAgent:Artifact] Artifact completed successfully")

        # ── Swarm Script Dispatchers ──
        elif hasattr(self, 'script_manager') and self.script_manager.is_script_action(action['action']):
             print(f"[UIAgent:Script] Dispatching script action: {action['action']}")
             if action['action'] == "run_cv_script":
                 import cv2 # type: ignore
                 p2 = self.screen_observer.capture()
                 if p2:
                     a_frame = cv2.imread(p2)
                     result = self.script_manager.apply_script(action.get("name", ""), None, a_frame, action)
                 else:
                     result = {"success": False, "error": "Screen capture failed"}
             else:
                 result = self.script_manager.execute_action(action)
                 
             action['script_result'] = result
             if not result.get('success'):
                 print(f"[UIAgent:Script] Script action failed: {result.get('error', 'unknown')}")
             else:
                 print(f"[UIAgent:Script] Script action completed successfully")

        # Add a small human pause after action
        time.sleep(0.5)
        
        # 4. Observe After State
        if target_app:
            filename = f"after_{target_app}_{timestamp}.png"
            full_path = str(self.memory_agent.short_term_dir / "screenshots" / filename)
            state_after = self.window_observer.capture_window_by_title(target_app, filename=full_path)
            if not state_after:
                filename = f"after_fallback_{timestamp}.png"
                full_path = str(self.memory_agent.short_term_dir / "screenshots" / filename)
                state_after = self.screen_observer.capture(filename=full_path)
        else:
            filename = f"after_{timestamp}.png"
            full_path = str(self.memory_agent.short_term_dir / "screenshots" / filename)
            state_after = self.screen_observer.capture(filename=full_path)
        
        # 5. Evaluate Success (Learning)
        # Skip evaluation for transient or targetless actions
        success = True
        if action['action'] not in ['press', 'hotkey', 'wait'] and target:
            success = self.learning_loop.evaluate_action_success(action, state_before, state_after)
            
            # If Live API is active and we have an update for it, send it 
            if not self.offline_mode and self.live_session_healthy:
                feedback = f"Action '{action['action']}' on '{target}' resulted in {'SUCCESS' if success else 'FAILURE'}"
                self.live_streamer.send_step_feedback(feedback, self._loop)
        
        if not success:
             print("[UIAgent] Action failed. Triggering Semantic Recovery Replan...")
             self.goal_manager.replan_recovery(action, state_after, self.vision)


if __name__ == "__main__":
    import asyncio
    agent = UIAgent()
    asyncio.run(agent.run_agentic("Close the Notepad application."))
