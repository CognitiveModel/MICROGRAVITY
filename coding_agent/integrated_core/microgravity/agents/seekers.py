"""
MICROGRAVITY CORE — System Seeker Agent

Specialized for OS-level operations: shell commands, filesystem access,
process management, and environment interactions.
"""

from __future__ import annotations

import json
from typing import Any

from agents.base_seeker import BaseSeekerAgent
from models.schemas import DAGNode


class SystemSeeker(BaseSeekerAgent):
    """
    Domain: Shell commands, filesystem, OS utilities, process management.
    Uses LLM to reason about what commands to run, then validates the plan
    through the Channelization Pipeline before execution.
    """

    @property
    def agent_type(self) -> str:
        return "system_seeker"

    @property
    def system_prompt(self) -> str:
        return (
            "You are a senior Linux/Windows Systems Administrator working within an AI swarm. "
            "Your expertise is in shell commands, filesystem operations, process management, "
            "and OS-level troubleshooting.\n\n"
            "RULES:\n"
            "- Always state what command you intend to run BEFORE running it.\n"
            "- For each command, write a HYPOTHESIS of expected output.\n"
            "- NEVER run destructive commands (rm -rf /, format, etc.) without explicit override.\n"
            "- Prefer read-only operations when possible.\n"
            "- Report results in structured JSON.\n\n"
            "Respond with JSON: {\"command\": \"...\", \"hypothesis\": \"...\", \"reasoning\": \"...\"}"
        )

    async def _execute_operational(self, node: DAGNode, context: dict[str, Any]) -> Any:
        """Generate and validate system commands for the given task."""
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": (
                f"Task: {node.task_description}\n\n"
                f"Context: {json.dumps(context.get('prior_results', [])[-3:], default=str)[:1000]}\n\n"
                f"Generate the command(s) needed to accomplish this task. "
                f"Include your hypothesis of expected output."
            )},
        ]

        response = await self.call_llm(messages, tier="executive")
        return {"agent": self.name, "type": self.agent_type, "plan": response.content}


class CodingSeeker(BaseSeekerAgent):
    """
    Domain: Code generation, architecture planning, refactoring, feature implementation.
    """

    @property
    def agent_type(self) -> str:
        return "coding_seeker"

    @property
    def system_prompt(self) -> str:
        return (
            "You are a senior Software Engineer within an AI swarm. "
            "Your expertise is in writing clean, efficient, well-documented code across "
            "multiple languages (Python, TypeScript, Rust, etc.).\n\n"
            "RULES:\n"
            "- Write production-quality code with proper error handling.\n"
            "- Follow language-specific best practices and idioms.\n"
            "- Include docstrings and type hints.\n"
            "- Structure code for reusability (the swarm may internalize it as a Process IP).\n"
            "- Respond with code blocks and a brief explanation.\n\n"
            "Respond with JSON: {\"code\": \"...\", \"language\": \"...\", \"explanation\": \"...\"}"
        )

    async def _execute_operational(self, node: DAGNode, context: dict[str, Any]) -> Any:
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": (
                f"Task: {node.task_description}\n\n"
                f"Context: {json.dumps(context.get('prior_results', [])[-3:], default=str)[:1500]}"
            )},
        ]

        response = await self.call_llm(messages, tier="executive")
        return {"agent": self.name, "type": self.agent_type, "output": response.content}


class QASeeker(BaseSeekerAgent):
    """
    Domain: Testing, error analysis, troubleshooting, A/B testing, quality assurance.
    """

    @property
    def agent_type(self) -> str:
        return "qa_seeker"

    @property
    def system_prompt(self) -> str:
        return (
            "You are a senior QA Engineer within an AI swarm. "
            "Your expertise is in testing strategies, error analysis, troubleshooting, "
            "and quality assurance.\n\n"
            "RULES:\n"
            "- Think adversarially: what could go wrong?\n"
            "- Write comprehensive test cases covering edge cases.\n"
            "- When troubleshooting, systematically narrow down the root cause.\n"
            "- Propose both immediate fixes and long-term preventive measures.\n\n"
            "Respond with JSON: {\"analysis\": \"...\", \"test_cases\": [...], \"recommendation\": \"...\"}"
        )

    async def _execute_operational(self, node: DAGNode, context: dict[str, Any]) -> Any:
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": (
                f"Task: {node.task_description}\n\n"
                f"Context: {json.dumps(context.get('prior_results', [])[-3:], default=str)[:1500]}"
            )},
        ]

        response = await self.call_llm(messages, tier="executive")
        return {"agent": self.name, "type": self.agent_type, "output": response.content}


class DevOpsSeeker(BaseSeekerAgent):
    """
    Domain: Infrastructure, deployment, monitoring, CI/CD, resource tracking.
    """

    @property
    def agent_type(self) -> str:
        return "devops_seeker"

    @property
    def system_prompt(self) -> str:
        return (
            "You are a senior DevOps Engineer within an AI swarm. "
            "Your expertise is in containerization, deployment pipelines, monitoring, "
            "resource optimization, and infrastructure management.\n\n"
            "RULES:\n"
            "- Prioritize reproducibility (Dockerfiles, IaC).\n"
            "- Consider cost implications of infrastructure decisions.\n"
            "- Monitor for resource leaks and unnecessary processes.\n"
            "- Design for scalability and fault tolerance.\n\n"
            "Respond with JSON: {\"plan\": \"...\", \"resources\": [...], \"estimated_cost\": \"...\"}"
        )

    async def _execute_operational(self, node: DAGNode, context: dict[str, Any]) -> Any:
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": (
                f"Task: {node.task_description}\n\n"
                f"Context: {json.dumps(context.get('prior_results', [])[-3:], default=str)[:1500]}"
            )},
        ]

        response = await self.call_llm(messages, tier="executive")
        return {"agent": self.name, "type": self.agent_type, "output": response.content}


class APISeeker(BaseSeekerAgent):
    """
    Domain: External API interactions, web scraping, network operations.
    """

    @property
    def agent_type(self) -> str:
        return "api_seeker"

    @property
    def system_prompt(self) -> str:
        return (
            "You are an API Integration Specialist within an AI swarm. "
            "Your expertise is in RESTful APIs, GraphQL, webhooks, web scraping, "
            "and network protocol management.\n\n"
            "RULES:\n"
            "- Always handle authentication securely (never log secrets).\n"
            "- Implement proper error handling and retry logic.\n"
            "- Respect rate limits and API usage policies.\n"
            "- Validate response schemas before processing.\n\n"
            "Respond with JSON: {\"api_plan\": \"...\", \"endpoint\": \"...\", \"expected_response\": \"...\"}"
        )

    async def _execute_operational(self, node: DAGNode, context: dict[str, Any]) -> Any:
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": (
                f"Task: {node.task_description}\n\n"
                f"Context: {json.dumps(context.get('prior_results', [])[-3:], default=str)[:1500]}"
            )},
        ]

        response = await self.call_llm(messages, tier="executive")
        return {"agent": self.name, "type": self.agent_type, "output": response.content}
