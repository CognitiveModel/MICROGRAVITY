"""
MICROGRAVITY CORE — DAG Planner

Constructs and manages Directed Acyclic Graphs (DAGs) of execution tasks.
Supports both fixed (pre-defined) plans and dynamic (LLM-generated) plans.

The Operator uses this to decompose user objectives into executable sequences
of agent tasks, with dependency tracking and parallel execution support.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Optional

from models.schemas import (
    DAGNode,
    DAGNodeStatus,
    ExecutionDAG,
    LLMRequest,
    StrategyMode,
)

logger = logging.getLogger(__name__)


class DAGPlanner:
    """
    Plans, validates, and manages execution DAGs.
    
    Two planning modes:
    - Fixed: Build a DAG from a predefined template
    - Dynamic: Use an LLM to decompose an objective into a DAG
    """

    def __init__(self, llm_call_fn=None, available_agents: list[str] | None = None):
        self._llm_call = llm_call_fn
        self._available_agents = available_agents or []

    async def plan_dynamic(
        self,
        objective: str,
        context: dict[str, Any] | None = None,
        strategy: StrategyMode = StrategyMode.BALANCED,
        max_nodes: int = 10,
    ) -> ExecutionDAG:
        """
        Use an LLM to decompose an objective into an execution DAG.
        
        Returns an ExecutionDAG with nodes, dependencies, and agent assignments.
        """
        if not self._llm_call:
            # Fallback: create a single-node DAG
            return self._single_node_dag(objective, strategy)

        agents_list = ", ".join(self._available_agents) or "SystemSeeker, CodingSeeker, QASeeker, DevOpsSeeker, APISeeker"

        system_prompt = f"""You are the DAG Planner for the MICROGRAVITY CORE agentic swarm.

Decompose the user's objective into a Directed Acyclic Graph (DAG) of sub-tasks.

Available agents: [{agents_list}]

Rules:
- Each node must have a clear, specific task description
- Assign the most appropriate agent to each node
- Define dependencies (which nodes must complete before this one starts)
- Keep the DAG as lean as possible (max {max_nodes} nodes)
- Nodes without dependencies can run in parallel

Respond with ONLY valid JSON in this format:
{{
  "nodes": [
    {{
      "id": "step_1",
      "task_description": "...",
      "assigned_agent": "SystemSeeker",
      "dependencies": []
    }},
    {{
      "id": "step_2",
      "task_description": "...",
      "assigned_agent": "CodingSeeker",
      "dependencies": ["step_1"]
    }}
  ]
}}"""

        user_message = f"Objective: {objective}"
        if context:
            user_message += f"\n\nContext: {json.dumps(context, default=str)[:2000]}"

        request = LLMRequest(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
            model_tier="executive",
            temperature=0.2,
            max_tokens=2048,
        )

        try:
            response = await self._llm_call(request)
            dag_data = json.loads(response.content)

            dag = ExecutionDAG(
                objective=objective,
                strategy_mode=strategy,
            )

            for node_data in dag_data.get("nodes", []):
                node = DAGNode(
                    id=node_data["id"],
                    task_description=node_data["task_description"],
                    assigned_agent=node_data.get("assigned_agent"),
                    dependencies=node_data.get("dependencies", []),
                )
                dag.nodes.append(node)

            # Validate the DAG
            if self._validate_dag(dag):
                logger.info(f"Dynamic DAG created: {len(dag.nodes)} nodes for '{objective}'")
                return dag
            else:
                logger.warning("DAG validation failed, falling back to single node")
                return self._single_node_dag(objective, strategy)

        except Exception as e:
            logger.error(f"Dynamic planning failed: {e}")
            return self._single_node_dag(objective, strategy)

    def plan_fixed(self, template: list[dict[str, Any]], objective: str) -> ExecutionDAG:
        """
        Build a DAG from a predefined template.
        
        Template format:
        [
            {"id": "step_1", "task": "...", "agent": "SystemSeeker", "deps": []},
            {"id": "step_2", "task": "...", "agent": "CodingSeeker", "deps": ["step_1"]},
        ]
        """
        dag = ExecutionDAG(objective=objective, strategy_mode=StrategyMode.STRICT_PRINCIPLED)

        for item in template:
            node = DAGNode(
                id=item["id"],
                task_description=item["task"],
                assigned_agent=item.get("agent"),
                dependencies=item.get("deps", []),
            )
            dag.nodes.append(node)

        logger.info(f"Fixed DAG created: {len(dag.nodes)} nodes for '{objective}'")
        return dag

    async def replan_from_checkpoint(
        self,
        original_dag: ExecutionDAG,
        failed_node_id: str,
        error: str,
    ) -> ExecutionDAG:
        """
        Re-plan from a failed checkpoint — keep completed nodes, re-route from failure.
        """
        # Find remaining nodes (those that haven't completed)
        completed_ids = {n.id for n in original_dag.nodes if n.status == DAGNodeStatus.COMPLETED}
        remaining_tasks = [
            n.task_description for n in original_dag.nodes if n.id not in completed_ids
        ]

        if not remaining_tasks:
            return original_dag

        new_objective = (
            f"REPLANNING after failure in node '{failed_node_id}' (error: {error}). "
            f"Remaining tasks to accomplish: {remaining_tasks}"
        )

        new_dag = await self.plan_dynamic(
            new_objective,
            context={"completed_nodes": list(completed_ids), "failure_error": error},
            strategy=original_dag.strategy_mode,
        )

        # Carry forward completed nodes
        for node in original_dag.nodes:
            if node.status == DAGNodeStatus.COMPLETED:
                new_dag.nodes.insert(0, node)

        return new_dag

    def _single_node_dag(self, objective: str, strategy: StrategyMode) -> ExecutionDAG:
        """Fallback: single-node DAG when dynamic planning fails."""
        dag = ExecutionDAG(objective=objective, strategy_mode=strategy)
        dag.nodes.append(DAGNode(
            id="single_step",
            task_description=objective,
        ))
        return dag

    @staticmethod
    def _validate_dag(dag: ExecutionDAG) -> bool:
        """Validate that the DAG is well-formed (no cycles, valid dependencies)."""
        node_ids = {n.id for n in dag.nodes}

        # Check all dependencies reference valid nodes
        for node in dag.nodes:
            for dep in node.dependencies:
                if dep not in node_ids:
                    logger.warning(f"Invalid dependency: {node.id} → {dep}")
                    return False

        # Check for cycles (topological sort)
        visited = set()
        in_stack = set()

        def has_cycle(node_id: str) -> bool:
            if node_id in in_stack:
                return True
            if node_id in visited:
                return False
            visited.add(node_id)
            in_stack.add(node_id)
            node = next((n for n in dag.nodes if n.id == node_id), None)
            if node:
                for dep in node.dependencies:
                    if has_cycle(dep):
                        return True
            in_stack.remove(node_id)
            return False

        for node in dag.nodes:
            if has_cycle(node.id):
                logger.warning(f"Cycle detected in DAG involving node {node.id}")
                return False

        return True
