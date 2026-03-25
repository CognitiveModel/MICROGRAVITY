"""Bridge connecting Swarm Engine learning with UI Agent experiential memory."""

import time
from dataclasses import dataclass
from typing import Optional, Dict, Any, List
from threading import Lock

from loguru import logger # type: ignore


@dataclass
class TaskContext:
    """Classifies the deep intent, impact, and domain of a user task."""
    intent: str      # TEST, WORK, EXPLORATION, MAINTENANCE
    impact: str      # TRIVIAL, MODERATE, SIGNIFICANT, CRITICAL
    domain: str      # UI_AUTOMATION, INFORMATION, DEVELOPMENT, COMMUNICATION
    is_capability_test: bool
    description: str


class TaskContextClassifier:
    """Heuristic classifier to understand *why* a task is being requested."""
    
    # Keywords indicating the user is just testing the agent's capabilities
    TEST_KEYWORDS = [
        "can you", "try to", "test", "demonstrate", "show me", "prove",
        "are you able to", "let's see if", "example of"
    ]
    
    # Keywords indicating real-world work with tangible outcomes
    WORK_KEYWORDS = [
        "send", "deploy", "delete", "buy", "purchase", "submit", "save",
        "update production", "commit to main"
    ]
    
    def classify(self, task_text: str, conversation_history: str = "") -> TaskContext:
        """Classifies the intent and worldly importance of the task."""
        text_lower = task_text.lower()
        
        # 1. Determine Intent (Is it a test?)
        is_test = any(kw in text_lower for kw in self.TEST_KEYWORDS)
        if "test" in conversation_history.lower()[-200:]: # Look at recent history # type: ignore
            is_test = True
            
        intent = "TEST" if is_test else "WORK"
        if "research" in text_lower or "find" in text_lower or "what is" in text_lower:
            intent = "EXPLORATION"
            
        # 2. Determine Impact / Tangible Outcome
        impact = "TRIVIAL"
        if any(kw in text_lower for kw in self.WORK_KEYWORDS):
            impact = "SIGNIFICANT"
            if "production" in text_lower or "money" in text_lower or "delete" in text_lower:
                impact = "CRITICAL"
        elif not is_test and len(text_lower) > 50:
             impact = "MODERATE"
             
        # 3. Determine Domain
        domain = "GENERAL"
        if "window" in text_lower or "click" in text_lower or "browser" in text_lower or "open" in text_lower:
            domain = "UI_AUTOMATION"
        elif "code" in text_lower or "script" in text_lower or "git" in text_lower:
            domain = "DEVELOPMENT"
        elif "message" in text_lower or "email" in text_lower or "slack" in text_lower:
            domain = "COMMUNICATION"
        else:
            domain = "INFORMATION"
            
        desc = f"Intent: {intent} {'(Capability Test)' if is_test else '(Active Work)'} | Impact: {impact} | Domain: {domain}"
        
        return TaskContext(
            intent=intent,
            impact=impact,
            domain=domain,
            is_capability_test=is_test,
            description=desc
        )


class ExperientialBridge:
    """
    Connects the Swarm's OutcomeTracker (Graph) with UI Agent's ExperientialMemory.
    Provides a shared context pool so UI learning helps the Swarm, and vice versa.
    """
    
    def __init__(self, swarm_engine):
        self.engine = swarm_engine
        self.classifier = TaskContextClassifier()
        self._ui_memory = None
        self._lock = Lock()
        logger.info("ExperientialBridge initialized.")
        
    def _ensure_ui_memory(self):
        """Lazily load the UI experiential memory (heavy disk I/O)."""
        if self._ui_memory is None:
            with self._lock:
                if self._ui_memory is None:
                    try:
                        from nanobot.agent.ui.planning.experiential_memory import ExperientialMemory # type: ignore
                        # Use a shared memory path across UI Agent and Swarm
                        import os
                        from pathlib import Path
                        mem_dir = str(self.engine.workspace / "agent_memory" / "experiential")
                        self._ui_memory = ExperientialMemory(storage_dir=mem_dir)
                    except ImportError as e:
                        logger.warning(f"Could not load UI ExperientialMemory bridge: {e}")
                        
        return self._ui_memory

    def classify_task(self, task_text: str) -> TaskContext:
        """Expose classifier to the rest of the swarm."""
        return self.classifier.classify(task_text)

    def record_ui_outcome(self, task_id: str, task_text: str, result: str, success: bool, steps_summary: Optional[List[Dict[Any, Any]]] = None): # type: ignore
        """
        Records a completed UI task into both the Swarm Graph and the UI Experiential Memory.
        """
        # 1. Record in Swarm Topology
        try:
             # This creates an Action node and an Outcome node in Kuzu/SQLite
             self.engine.record_action_outcome(
                 task_id=task_id,
                 action_name="ui_action",
                 payload={"task": task_text},
                 results=result,
                 success=success
             )
        except Exception as e:
             logger.error(f"Failed bridging UI outcome to Swarm Graph: {e}")
             
        # 2. Record in UI Experiential Memory (if we have detailed steps)
        # Usually the UIAgent records its own episodes natively, but if the tool fails early,
        # we can record the failure nuance here.
        ui_mem = self._ensure_ui_memory()
        if ui_mem and not success:
            app_class = "Desktop" # Default if unknown
            if "chrome" in task_text.lower() or "browser" in task_text.lower():
                app_class = "BROWSER"
                
            ui_mem.record_nuance(
                app_class=app_class,
                element_id="unknown_target",
                nuance_type="PLATFORM_QUIRK",
                severity="IMPORTANT",
                description=f"Swarm-directed UI task failed: {result}",
                trigger_condition=task_text
            )

    def get_shared_context(self, task_text: str, app_class: str = "Desktop") -> str:
        """
        Builds a unified context string merging Swarm Objectives and UI Learning.
        Injected into both the UI Agent's Planner prompt and Swarm's System Prompt.
        """
        context_parts = []
        
        # 1. Task Classification (Worldly Context)
        ctx = self.classify_task(task_text)
        context_parts.append("=== TASK WORLDLY CONTEXT ===")
        context_parts.append(ctx.description)
        if ctx.is_capability_test:
            context_parts.append("WARNING: This appears to be a capability test. Focus on demonstrating functionality safely.")
        if ctx.impact in ["SIGNIFICANT", "CRITICAL"]:
            context_parts.append("CRITICAL: This task has high real-world impact. Verify outcomes carefully.")

        # 2. UI Experiential Knowledge
        ui_mem = self._ensure_ui_memory()
        if ui_mem:
            try:
                # Use UI Agent's hierarchical context builder
                ui_insights = ui_mem.get_context_for_planner(app_class=app_class, current_task=task_text)
                if ui_insights:
                    context_parts.append("\n=== SHARED EXPERIENTIAL LEARNING ===")
                    context_parts.append(ui_insights)
            except Exception as e:
                logger.debug(f"Failed to fetch UI experiential context: {e}")
                
        # 3. Swarm Graph Outcomes (Placeholder for future graph queries)
        # e.g., self.engine.graph.execute("MATCH (a:Action)-[r:RESULTED_IN]->(o:Outcome) WHERE o.status='FAILED' RETURN a")

        return "\n".join(context_parts)
