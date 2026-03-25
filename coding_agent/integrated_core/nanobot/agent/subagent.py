"""Subagent manager for background task execution."""

import asyncio
import json
import uuid
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

from loguru import logger  # type: ignore

from nanobot.agent.tools.base import Tool  # type: ignore
from nanobot.agent.tools.filesystem import EditFileTool, ListDirTool, ReadFileTool, WriteFileTool  # type: ignore
from nanobot.agent.tools.registry import ToolRegistry  # type: ignore
from nanobot.agent.tools.shell import ExecTool  # type: ignore
from nanobot.agent.tools.web import WebFetchTool, WebSearchTool  # type: ignore
from nanobot.bus.events import InboundMessage  # type: ignore
from nanobot.bus.queue import MessageBus  # type: ignore
from nanobot.config.schema import ExecToolConfig  # type: ignore
from nanobot.providers.base import LLMProvider  # type: ignore
from nanobot.utils.helpers import build_assistant_message  # type: ignore


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
class Plan:
    task: str
    priority: int = 1
    status: PlanStatus = PlanStatus.DRAFT
    steps: list[PlanStep] = field(default_factory=list)
    progress_pct: float = 0.0

class UpdatePlanStepTool(Tool):
    """Tool for the subagent to report progress on its active plan."""
    name = "update_plan_step"
    description = "Mark a step in your execution plan as complete and log progress."
    parameters = {
        "type": "object",
        "properties": {
            "step_index": {"type": "integer", "description": "The 0-based index of the step completed"},
            "status": {"type": "string", "enum": ["COMPLETED", "FAILED", "SKIPPED"]}
        },
        "required": ["step_index", "status"]
    }

    def __init__(self, plan: Plan):
        self.plan = plan

    async def execute(self, step_index: int, status: str) -> str:
        if 0 <= step_index < len(self.plan.steps):
            self.plan.steps[step_index].status = status
            completed_count = sum(1 for s in self.plan.steps if s.status in ("COMPLETED", "SKIPPED"))
            self.plan.progress_pct = (completed_count / len(self.plan.steps)) * 100
            return f"Step {step_index} marked as {status}. Total progress: {self.plan.progress_pct:.0f}%"
        return f"Error: step_index {step_index} out of bounds."



class SubagentManager:
    """Manages background subagent execution."""

    def __init__(
        self,
        provider: LLMProvider,
        workspace: Path,
        bus: MessageBus,
        model: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        reasoning_effort: str | None = None,
        brave_api_key: str | None = None,
        web_proxy: str | None = None,
        exec_config: "ExecToolConfig | None" = None,
        restrict_to_workspace: bool = False,
        swarm_engine: Any = None,
    ):
        from nanobot.config.schema import ExecToolConfig  # type: ignore
        self.provider = provider
        self.workspace = workspace
        self.bus = bus
        self.model = model or provider.get_default_model()
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.reasoning_effort = reasoning_effort
        self.brave_api_key = brave_api_key
        self.web_proxy = web_proxy
        self.exec_config = exec_config or ExecToolConfig()
        self.restrict_to_workspace = restrict_to_workspace
        self.swarm = swarm_engine
        self._running_tasks: dict[str, asyncio.Task[None]] = {}
        self._session_tasks: dict[str, set[str]] = {}  # session_key -> {task_id, ...}

    async def spawn(
        self,
        task: str,
        label: str | None = None,
        origin_channel: str = "cli",
        origin_chat_id: str = "direct",
        session_key: str | None = None,
    ) -> str:
        """Spawn a subagent to execute a task in the background."""
        task_id = str(uuid.uuid4())[:8]  # type: ignore
        display_label = label or task[:30] + ("..." if len(task) > 30 else "")  # type: ignore
        origin = {"channel": origin_channel, "chat_id": origin_chat_id}

        bg_task = asyncio.create_task(
            self._run_subagent(task_id, task, display_label, origin)
        )
        self._running_tasks[task_id] = bg_task
        if session_key:
            self._session_tasks.setdefault(session_key, set()).add(task_id)

        def _cleanup(_: asyncio.Task) -> None:
            self._running_tasks.pop(task_id, None)
            if session_key and (ids := self._session_tasks.get(session_key)):
                ids.discard(task_id)
                if not ids:
                    del self._session_tasks[session_key]  # type: ignore

        bg_task.add_done_callback(_cleanup)

        logger.info("Spawned subagent [{}]: {}", task_id, display_label)
        return f"Subagent [{display_label}] started (id: {task_id}). I'll notify you when it completes."

    async def _run_subagent(
        self,
        task_id: str,
        task: str,
        label: str,
        origin: dict[str, str],
    ) -> None:
        """Execute the subagent task and announce the result."""
        logger.info("Subagent [{}] starting task: {}", task_id, label)

        # 1. Synthesize Plan before execution
        plan = await self._create_plan(task)
        plan.status = PlanStatus.ACTIVE

        # 2. Integrate with Swarm Objective Tracker if available
        obj_id = None
        if self.swarm:
            try:
                # We defer loading these enum types to runtime to avoid circular imports during subagent init
                from nanobot.swarm.objectives import ExecutionMode, InterferenceScope  # type: ignore
                
                # Assume default minimal interference scope (FILESYSTEM) 
                # This ensures the ParallelExecutor respects the running subagent during scheduling
                obj_id = self.swarm.objectives.track_objective(
                    intent=task,
                    execution_mode=ExecutionMode.BACKGROUND,
                    interference_scope=[InterferenceScope.FILESYSTEM],
                    resource_estimate=0.5
                )
            except Exception as e:
                logger.debug("Failed to track objective for subagent task {}: {}", task_id, e)

        try:
            # Build subagent tools (no message tool, no spawn tool)
            tools = ToolRegistry()
            allowed_dir = self.workspace if self.restrict_to_workspace else None
            tools.register(ReadFileTool(workspace=self.workspace, allowed_dir=allowed_dir))
            tools.register(WriteFileTool(workspace=self.workspace, allowed_dir=allowed_dir))
            tools.register(EditFileTool(workspace=self.workspace, allowed_dir=allowed_dir))
            tools.register(ListDirTool(workspace=self.workspace, allowed_dir=allowed_dir))
            tools.register(ExecTool(
                working_dir=str(self.workspace),
                timeout=self.exec_config.timeout,
                restrict_to_workspace=self.restrict_to_workspace,
                path_append=self.exec_config.path_append,
            ))
            tools.register(WebSearchTool(api_key=self.brave_api_key, proxy=self.web_proxy))
            tools.register(WebFetchTool(proxy=self.web_proxy))
            tools.register(UpdatePlanStepTool(plan))
            
            system_prompt = self._build_subagent_prompt(plan)
            messages: list[dict[str, Any]] = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": task},
            ]

            # Run agent loop (limited iterations)
            max_iterations = 15
            iteration = 0
            final_result: str | None = None

            while iteration < max_iterations:
                iteration += 1

                response = await self.provider.chat_with_retry(
                    messages=messages,
                    tools=tools.get_definitions(),
                    model=self.model,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    reasoning_effort=self.reasoning_effort,
                )

                if response.has_tool_calls:
                    tool_call_dicts = [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {
                                "name": tc.name,
                                "arguments": json.dumps(tc.arguments, ensure_ascii=False),
                            },
                        }
                        for tc in response.tool_calls
                    ]
                    messages.append(build_assistant_message(
                        response.content or "",
                        tool_calls=tool_call_dicts,
                        reasoning_content=response.reasoning_content,
                        thinking_blocks=response.thinking_blocks,
                    ))

                    # Execute tools
                    for tool_call in response.tool_calls:
                        args_str = json.dumps(tool_call.arguments, ensure_ascii=False)
                        logger.debug("Subagent [{}] executing: {} with arguments: {}", task_id, tool_call.name, args_str)
                        result = await tools.execute(tool_call.name, tool_call.arguments)
                        messages.append({
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "name": tool_call.name,
                            "content": result,
                        })
                else:
                    final_result = response.content
                    break

            if final_result is None:
                final_result = "Task completed but no final response was generated."
                
            plan.status = PlanStatus.COMPLETED

            logger.info("Subagent [{}] completed successfully", task_id)
            await self._announce_result(task_id, label, task, final_result, origin, "ok", plan)  # type: ignore

            if self.swarm:
                if obj_id:
                    self.swarm.objectives.log_progress(obj_id, 100, finalize=True)
                # Phase 5: Trigger background cognitive evaluation
                from nanobot.config.loader import load_config  # type: ignore
                config = load_config()
                self.swarm.trigger_evaluation(
                    provider=self.provider,
                    task_id=task_id,
                    task_intent=task,
                    plan=plan,
                    result=final_result,
                    api_keys=config.api_keys,
                    web_proxy=self.web_proxy
                )

        except Exception as e:
            error_msg = f"Error: {str(e)}"
            plan.status = PlanStatus.FAILED
            logger.error("Subagent [{}] failed: {}", task_id, e)
            await self._announce_result(task_id, label, task, error_msg, origin, "error", plan)
            if self.swarm and obj_id:
                self.swarm.objectives.log_progress(obj_id, plan.progress_pct, finalize=True)

    async def _create_plan(self, task: str) -> Plan:
        """Pre-execution LLM call to break the task down into a structured plan, enhanced with past insights."""
        
        past_insights = ""
        if self.swarm:
            # Query the LMDB for past planning insights relevant to this task intent
            # MVP prefix scan (In prod: vector search against sprocess:insight: or query graph)
            insights = []
            for key, val in self.swarm.lmdb.prefix_scan("sprocess:insight:"):
                # Very basic keyword matching for MVP retrieval
                # A full implementation would use self.swarm.surfaces.query("Cross-Memory", task)
                if any(word.lower() in str(val.get("intent", "")).lower() for word in task.split() if len(word) > 4):
                    insights.append(val.get("content", ""))
            
            if insights:
                past_insights = "\n## Lessons Learned from Past Executions:\n- " + "\n- ".join(insights[:3]) + "\n\nCRITICAL: Incorporate these lessons into your plan!"  # type: ignore

        prompt = f"""You are a planning module for a subagent.
Task: {task}
{past_insights}

Create a JSON execution plan with a list of 'steps'. Each step has 'description', and optional 'tool_hint' and 'depends_on' (list of integer zero-indexed dependencies).
Return ONLY valid JSON in this format:
{{
  "priority": 1,
  "steps": [
    {{"description": "Read the codebase structure", "tool_hint": "list_dir"}},
    {{"description": "Update target file", "tool_hint": "edit_file", "depends_on": [0]}}
  ]
}}"""
        try:
            response = await self.provider.chat_with_retry(
                messages=[{"role": "user", "content": prompt}],
                model=self.model,
                temperature=0.1,
                max_tokens=1000
            )
            content = response.content or "{}"
            
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                content = content.split("```")[1].split("```")[0].strip()
                
            data = json.loads(content)
            
            steps = []
            for s in data.get("steps", []):
                steps.append(PlanStep(
                    description=s.get("description", ""),
                    tool_hint=s.get("tool_hint", ""),
                    depends_on=[str(x) for x in s.get("depends_on", [])]
                ))
            
            return Plan(
                task=task,
                priority=data.get("priority", 1),
                steps=steps
            )
        except Exception as e:
            logger.warning("Failed to parse subagent plan (fallbacking to monolithic plan): {}", e)
            return Plan(task=task, steps=[PlanStep(description=task)])

    async def _announce_result(
        self,
        task_id: str,
        label: str,
        task: str,
        result: str,
        origin: dict[str, str],
        status: str,
        plan: Plan,
    ) -> None:
        """Announce the subagent result to the main agent via the message bus."""
        status_text = "completed successfully" if status == "ok" else "failed"

        announce_content = f"""[Subagent '{label}' {status_text}]

Task: {task}
Progress: {plan.progress_pct:.0f}% ({sum(1 for s in plan.steps if s.status == "COMPLETED")}/{len(plan.steps)} steps)

Result:
{result}

Summarize this naturally for the user. Keep it brief (1-2 sentences). Do not mention technical details like "subagent" or task IDs."""

        # Inject as system message to trigger main agent
        msg = InboundMessage(
            channel="system",
            sender_id="subagent",
            chat_id=f"{origin['channel']}:{origin['chat_id']}",
            content=announce_content,
        )

        await self.bus.publish_inbound(msg)
        logger.debug("Subagent [{}] announced result to {}:{}", task_id, origin['channel'], origin['chat_id'])
    
    def _build_subagent_prompt(self, plan: Plan) -> str:
        """Build a focused system prompt for the subagent."""
        from nanobot.agent.context import ContextBuilder  # type: ignore
        from nanobot.agent.skills import SkillsLoader  # type: ignore

        time_ctx = ContextBuilder._build_runtime_context(None, None)
        parts = [f"""# Subagent

{time_ctx}

You are a subagent spawned by the main agent to complete a specific task.
Stay focused on the assigned task. Your final response will be reported back to the main agent.

## Execution Plan
This is your auto-generated plan for the task. Use `update_plan_step` to track progress!
"""]
        
        for i, s in enumerate(plan.steps):
            parts.append(f"{i}. [{s.status}] {s.description}")
            if s.tool_hint:
                parts.append(f"   Potential Tool: `{s.tool_hint}`")

        parts.append(f"""
## Workspace
{self.workspace}""")

        skills_summary = SkillsLoader(self.workspace).build_skills_summary()
        if skills_summary:
            parts.append(f"## Skills\n\nRead SKILL.md with read_file to use a skill.\n\n{skills_summary}")

        return "\n\n".join(parts)
    
    async def cancel_by_session(self, session_key: str) -> int:
        """Cancel all subagents for the given session. Returns count cancelled."""
        tasks = [self._running_tasks[tid] for tid in self._session_tasks.get(session_key, [])  # type: ignore
                 if tid in self._running_tasks and not self._running_tasks[tid].done()]  # type: ignore
        for t in tasks:
            t.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        return len(tasks)

    def get_running_count(self) -> int:
        """Return the number of currently running subagents."""
        return len(self._running_tasks)
