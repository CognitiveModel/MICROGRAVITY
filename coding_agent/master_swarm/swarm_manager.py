"""
Swarm Manager - Ported from Nanobot 0

This module manages background subagent execution, spawning parallel planners
and executors natively within the coding_agent infrastructure.
"""

import asyncio
import json
import uuid
import logging
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Optional

from coding_agent.utils.gemini_client import get_gemini_response
from coding_agent.core.swarm_factory import SwarmFactory
from coding_agent.master_swarm.coordinator import SwarmCoordinator

logger = logging.getLogger(__name__)

class PlanStatus(str, Enum):
    DRAFT = "DRAFT"
    ACTIVE = "ACTIVE"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"

@dataclass
class PlanStep:
    description: str
    status: str = "PENDING"
    tool_hint: str = ""
    depends_on: list[str] = field(default_factory=list)

@dataclass
class SwarmPlan:
    task: str
    priority: int = 1
    status: PlanStatus = PlanStatus.DRAFT
    steps: list[PlanStep] = field(default_factory=list)
    progress_pct: float = 0.0

class SubagentManager:
    """Manages background subagent execution, ported from nanobot 0."""

    def __init__(self, gemini_model, workspace: Path, coordinator: SwarmCoordinator = None):
        self.gemini_model = gemini_model
        self.workspace = workspace
        self.coordinator = coordinator or SwarmCoordinator()

    def spawn_sync(self, task: str, label: Optional[str] = None) -> str:
        """Helper to spawn from a fully synchronous parent like IntrospectionAgent."""
        # For simplicity in this adaptation, we fire and forget in a separate thread/loop 
        # or abstract it if the parent environment isn't async-aware yet.
        # This function acts as the bridge.
        import threading
        
        def run_in_thread():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            msg, task_obj = loop.run_until_complete(self.spawn(task, label))
            loop.run_until_complete(task_obj)
            loop.close()

            
        thread = threading.Thread(target=run_in_thread, daemon=True)
        thread.start()
        return f"Spawned background swarm agent for task: {task[:30]}..."

    async def spawn(self, task: str, label: Optional[str] = None) -> tuple[str, asyncio.Task[Any]]:
        """Spawn a subagent to execute a task in the background."""
        task_id = str(uuid.uuid4())[:8]
        display_label = label or task[:30] + ("..." if len(task) > 30 else "")

        bg_task = asyncio.create_task(self._run_subagent(task_id, task, display_label))
        self.coordinator.register_system_task(task_id, bg_task)

        def _cleanup(_: asyncio.Task) -> None:
            self.coordinator.remove_system_task(task_id)

        bg_task.add_done_callback(_cleanup)
        logger.info(f"Spawned subagent [{task_id}]: {display_label}")
        return f"Subagent [{display_label}] started (id: {task_id}).", bg_task


    async def _run_subagent(self, task_id: str, task: str, label: str) -> None:
        """Execute the subagent task and announce the result."""
        try:
            msg = f"\n[SWARM] Subagent [{task_id}] starting task: {label}"
            try:
                print(msg)
            except UnicodeEncodeError:
                print(msg.encode('ascii', 'ignore').decode('ascii'))

            # 1. Synthesize Plan before execution
            plan = await self._create_plan(task)
            plan.status = PlanStatus.ACTIVE

            loop = asyncio.get_event_loop()
            
            # --- LIVE SNAPSHOT RE-ANALYSIS LOOP ---
            max_steps = 3
            step = 0
            final_result = "Task terminated without final result."
            
            while plan.status != PlanStatus.COMPLETED and step < max_steps:
                step += 1
                prompt = f"Executing Swarm Plan for task: {task}\nPlan outline:\n"
                for i, s in enumerate(plan.steps):
                    prompt += f"{i}. {s.description}\n"
                prompt += f"Iteration {step}/{max_steps}. Observe the ongoing environment snapshot. Provide the outcome or indicate if more steps are needed."

                # Full tool usage porting
                if "ui" in task.lower() or "screen" in task.lower() or "click" in task.lower():
                    print(f"[SWARM] UI-bound task detected. Delegating to UIAgent...")
                    ui_agent = SwarmFactory.create_custom_agent(
                        "coding_agent.ui_agent.core.ui_agent",
                        "UIAgent"
                    )
                    final_result = await ui_agent.run_agentic(task)
                    plan.status = PlanStatus.COMPLETED
                else:
                    final_result = await loop.run_in_executor(
                        None, get_gemini_response, self.gemini_model, prompt
                    )
                    
                    if "FINAL" in final_result or step == max_steps:
                        plan.status = PlanStatus.COMPLETED
                    else:
                        print(f"[SWARM SNAPSHOT] Re-evaluating plan based on partial execution step {step}...")
                
            plan.status = PlanStatus.COMPLETED
            msg = f"\n[SWARM] Subagent [{task_id}] completed successfully: \n{final_result}"
            try:
                print(msg)
            except UnicodeEncodeError:
                print(msg.encode('ascii', 'ignore').decode('ascii'))
                
        except asyncio.CancelledError:
            if 'plan' in locals():
                plan.status = PlanStatus.CANCELLED # type: ignore
            msg = f"\n[SWARM PREEMPTION] Subagent [{task_id}] was gracefully CANCELLED by system."
            try:
                print(msg)
            except UnicodeEncodeError:
                pass
            raise  # Re-raise for asyncio to clean up

        except Exception as e:
            # We don't have 'plan' yet if _create_plan failed, but we can try to set it if it exists
            if 'plan' in locals():
                plan.status = PlanStatus.FAILED # type: ignore
            msg = f"\n[SWARM ERROR] Subagent [{task_id}] failed: {e}"
            try:
                print(msg)
            except UnicodeEncodeError:
                print(msg.encode('ascii', 'ignore').decode('ascii'))

    async def _create_plan(self, task: str) -> SwarmPlan:
        """Pre-execution LLM call to break the task down into a structured plan."""
        prompt = f"""You are a planning module for a subagent.
Task: {task}
Create a JSON execution plan with a list of 'steps'. Each step has 'description'.
Return ONLY valid JSON in this format:
{{
  "priority": 1,
  "steps": [
    {{"description": "Read the codebase structure"}}
  ]
}}"""
        loop = asyncio.get_event_loop()
        content = await loop.run_in_executor(None, get_gemini_response, self.gemini_model, prompt)
        
        try:
            if "```json" in content:
                content = content.split("```json")[-1].split("```")[0].strip()
            data = json.loads(content)
            steps = [PlanStep(description=s.get("description", "")) for s in data.get("steps", [])]
            return SwarmPlan(task=task, priority=data.get("priority", 1), steps=steps)
        except Exception as e:
             return SwarmPlan(task=task, steps=[PlanStep(description=task)])
