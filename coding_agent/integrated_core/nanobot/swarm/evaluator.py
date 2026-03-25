"""Self-capability evaluation and web-grounded planning improvement."""

import asyncio
import json
import uuid
from dataclasses import asdict
from datetime import datetime
from typing import Any, TYPE_CHECKING

from loguru import logger # type: ignore

from nanobot.agent.tools.web import WebSearchTool # type: ignore
from nanobot.providers.base import LLMProvider # type: ignore
from nanobot.swarm.graph_store import GraphStore # type: ignore
from nanobot.swarm.lmdb_store import LMDBStore # type: ignore

if TYPE_CHECKING:
    from nanobot.agent.subagent import Plan # type: ignore


class CognitiveEvaluator:
    """
    Evaluates completed plans against web-grounded best practices
    and creates permanent Planning_Insight hops to improve future execution.
    Swarm Architecture § 22.
    """

    def __init__(
        self,
        provider: LLMProvider,
        lmdb: LMDBStore,
        graph: GraphStore,
        brave_api_key: str | None = None,
        web_proxy: str | None = None,
    ):
        self.provider = provider
        self.lmdb = lmdb
        self.graph = graph
        self.brave_api_key = brave_api_key
        self.web_proxy = web_proxy
        self.model = provider.get_default_model()

    async def evaluate_completed_objective(
        self,
        task_id: str,
        task_intent: str,
        plan: "Plan",
        result: str,
    ) -> None:
        """
        Background process: Critique a completed plan against web literature
        and store actionable insights for future similar tasks.
        """
        logger.info("CognitiveEvaluator starting review for task {}", task_id)
        try:
            # 1. Gather Web Context
            search_query = f"best practices for {task_intent}"
            web_context = await self._fetch_web_context(search_query)

            # 2. Generate Critique
            insight = await self._generate_critique(task_intent, plan, result, web_context)

            # 3. Store Insight if valuable
            if insight.get("actionable_insight"):
                self._store_insight(task_id, task_intent, insight)

            logger.info("CognitiveEvaluator completed review for task {}", task_id)

        except Exception as e:
            logger.error("Cognitive evaluation failed for task {}: {}", task_id, e)

    async def _fetch_web_context(self, query: str) -> str:
        """Fetch real-world best practices using Brave Search."""
        if not self.brave_api_key:
            return "No web search available. Proceeding with intrinsic knowledge critique."

        tool = WebSearchTool(api_key=self.brave_api_key, proxy=self.web_proxy)
        try:
            # Execute tool and grab the raw string result
            result_str = await tool.execute(query=query)
            return result_str[:2000]  # type: ignore # Cap context size
        except Exception as e:
            logger.warning("CognitiveEvaluator web search failed: {}", e)
            return f"Web search failed: {e}"

    async def _generate_critique(
        self,
        task_intent: str,
        plan: "Plan",
        result: str,
        web_context: str,
    ) -> dict[str, Any]:
        """Use LLM to compare actual execution against web best practices."""
        
        prompt = f"""You are a Cognitive Evaluator for an autonomous AI agent.
Your job is to critique the agent's execution of a task and generate a permanent 'Planning Insight' that will be injected into future prompts to prevent repeated mistakes or improve efficiency.

Task Intent: {task_intent}

Executed Plan:
{json.dumps([asdict(s) for s in plan.steps], indent=2)}

Final Result:
{result[:500]}... # type: ignore

Web Research / Best Practices for this task:
{web_context}

Critique the plan. Did the agent miss standard steps? Was error handling adequate? Did it approach the problem backward?
Return ONLY a JSON object:
{{
  "critique_summary": "Short summary of what went right/wrong",
  "actionable_insight": "A concrete directive for the agent next time it faces a similar task (e.g., 'Always verify virtualenv exists before pip installing')",
  "sophistication_score": 1-10 (How profound is this insight?)
}}
"""
        messages = [{"role": "user", "content": prompt}]
        response = await self.provider.chat_with_retry(
            messages=messages,
            model=self.model,
            temperature=0.2,
            max_tokens=500
        )
        content = response.content or "{}"

        # Clean JSON markdown
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0].strip()
        elif "```" in content:
            content = content.split("```")[1].split("```")[0].strip()

        try:
            return json.loads(content)
        except json.JSONDecodeError:
            logger.warning("Failed to parse cognitive critique. Content: {}", content)
            return {}

    def _store_insight(self, task_id: str, intent: str, insight: dict[str, Any]) -> None:
        """Store the insight as a high-sophistication Wisdom_Hop in LMDB and Graph."""
        insight_id = f"pi_{uuid.uuid4().hex[:8]}" # type: ignore
        timestamp = datetime.utcnow().isoformat()
        
        # 1. State Store (LMDB) - using similar schema to pipeline hops but clearly marked
        self.lmdb.put(f"sprocess:insight:{insight_id}", {
            "id": insight_id,
            "task_id": task_id,
            "intent": intent,
            "type": "PLANNING_INSIGHT",
            "content": insight["actionable_insight"],
            "critique": insight.get("critique_summary", ""),
            "sophistication": insight.get("sophistication_score", 5),
            "timestamp": timestamp
        })
        
        # 2. Graph Store
        try:
            self.graph.create_node(
                label="Planning_Insight",
                node_id=insight_id,
                props={
                    "intent": intent,
                    "insight": insight["actionable_insight"],
                    "timestamp": timestamp
                }
            )
        except Exception as e:
            logger.debug("Graph storage for insight failed: {}", e)
