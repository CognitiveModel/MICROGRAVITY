"""UI automation tool — exposes the UIAgent to the LLM as a callable tool."""

import asyncio
import platform
import threading
from typing import Any

from nanobot.agent.tools.base import Tool  # type: ignore


class UITool(Tool):
    """
    Tool that delegates desktop/browser UI tasks to the UIAgent.

    The UIAgent runs in a background thread (it uses blocking win32 APIs)
    and reports back when the task completes or fails.
    """

    def __init__(self, engine=None, send_callback=None):
        self._agent = None  # Lazy-initialized on first call
        self._lock = threading.Lock()
        self.engine = engine
        self.send_callback = send_callback

    def _ensure_agent(self):
        """Lazily initialize the UIAgent (heavy startup, Windows-only)."""
        if self._agent is not None:
            return
        with self._lock:
            if self._agent is not None:
                return
            from nanobot.agent.ui.core.ui_agent import UIAgent  # type: ignore
            self._agent = UIAgent()

    @property
    def name(self) -> str:
        return "ui_action"

    @property
    def description(self) -> str:
        return (
            "Perform a desktop UI automation task such as opening applications, "
            "clicking buttons, typing text, browsing the web, or interacting with "
            "any visible window on the user's screen. "
            "Use this tool ONLY when the user EXPLICITLY asks you to interact with their desktop, "
            "open a browser, navigate to a website, fill forms, or control a GUI application. "
            "DO NOT use this tool for conversational responses, greetings, or answering informational questions. "
            "The task should be described in natural language."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "task": {
                    "type": "string",
                    "description": (
                        "A natural-language description of the UI task to perform, "
                        "e.g. 'Open Chrome and search for clawhub' or "
                        "'Click the Submit button in the current window'."
                    ),
                },
            },
            "required": ["task"],
        }

    async def execute(self, task: str, **kwargs: Any) -> str:
        """Run the UIAgent in a background thread and return the result."""
        if platform.system() != "Windows":
            return "Error: UI automation is only available on Windows."

        try:
            self._ensure_agent()
        except Exception as e:
            return f"Error: Failed to initialize UI agent: {e}"

        loop = asyncio.get_event_loop()
        try:
            # 1. Get Shared Context from Bridge
            shared_ctx = ""
            if self.engine and hasattr(self.engine, 'experiential'):
                shared_ctx = self.engine.experiential.get_shared_context(task)
                
            # Create a thread-safe callback wrapper if we have a send_callback
            def safe_callback(msg: str):
                if self.send_callback:
                    # Run the async callback in the background
                    asyncio.run_coroutine_threadsafe(
                        self.send_callback(msg, kwargs.get("channel"), kwargs.get("chat_id")), 
                        loop
                    )
                
            # 2. Run the task
            result = await loop.run_in_executor(
                None, self._run_task, task, shared_ctx, safe_callback
            )
            
            # 3. Record Outcome via Bridge
            if self.engine and hasattr(self.engine, 'experiential'):
                success = "failed" not in result.lower()
                self.engine.experiential.record_ui_outcome(
                    task_id=kwargs.get("task_id", f"ui_{int(asyncio.get_event_loop().time())}"),
                    task_text=task,
                    result=result,
                    success=success
                )
                
            return result
        except Exception as e:
            return f"Error: UI task failed: {e}"

    def _run_task(self, task: str, shared_context: str = "", send_callback = None) -> str:
        """Blocking wrapper that runs the agentic UI loop."""
        agent = self._agent
        if agent is None:
            return "Error: UI agent not initialized."

        try:
            # Injecting shared context into the UI Agent's active planner
            if shared_context and hasattr(agent, 'planner'):
                # We append the context to the objective so the planner sees it
                task = f"{task}\n\n{shared_context}"
                
            if send_callback:
                send_callback(f"🖥️ **UI Agent Started**: `{task.splitlines()[0]}`\nI'll keep you updated on my progress.")
                
            agent.run_agentic(task, send_callback=send_callback)
            
            # Collect summary from the planner if available
            if hasattr(agent, 'planner') and agent.planner and hasattr(agent.planner, 'step_history'):
                steps: list[dict[str, Any]] = agent.planner.step_history  # type: ignore
                if steps:
                    summary_lines = []
                    for step in steps[-5:]:  # type: ignore
                        status = "✓" if step.get('success') else "✗"
                        action = step.get('action_type', step.get('action', '?'))
                        note = step.get('verification_note', step.get('reasoning', ''))
                        summary_lines.append(f"  {status} {action}: {note}")
                    
                    final_msg = f"✅ **UI Task Completed!** Here are the last few actions:\n" + "\n".join(summary_lines)
                    if send_callback:
                        send_callback(final_msg)
                    return final_msg
            return f"UI task completed: '{task}'"
        except Exception as e:
            err_msg = f"❌ **UI Task Failed**: {e}"
            if send_callback:
                send_callback(err_msg)
            return err_msg
        finally:
            # Destroy the agent so the next task gets a fresh instance
            # (HUD window is destroyed by stop(), needs recreation)
            self._agent = None

        return ""
