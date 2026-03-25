"""
HITLContextualizer — Human-in-the-Loop Feedback & Interrupt Ledger

This module acts as an intelligent interceptor for asynchronous human messages.
When a user sends a mid-task message (e.g., "Wait, click the OTHER button", "Open Excel too", "It's taking too long"), 
this engine uses an LLM to categorize the intent and correlate it to the active task.

Categories:
1. NEW_TASK: A completely new objective to add to the queue.
2. FEEDBACK_CORRECTION: Advice or correction for the *currently active* task.
3. INTERRUPT_STOP: A safety stop or emergency halt command.
4. NOISE_SPECULATIVE: Chitchat, rhetorical questions, or non-actionable speculation.

It maintains a ledger of active tasks to provide context to the LLM during categorization.
"""

import json
import time
from enum import Enum
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field

class MessageCategory(str, Enum):
    NEW_TASK = "NEW_TASK"
    FEEDBACK_CORRECTION = "FEEDBACK_CORRECTION"
    INTERRUPT_STOP = "INTERRUPT_STOP"
    NOISE_SPECULATIVE = "NOISE_SPECULATIVE"

@dataclass
class TaskLedgerEntry:
    task_id: str
    description: str
    status: str = "PENDING"  # PENDING, ACTIVE, COMPLETED, FAILED
    started_at: float = 0.0
    recent_actions: List[str] = field(default_factory=list)
    observables: List[str] = field(default_factory=list)
    resources: List[str] = field(default_factory=list)

class HITLContextualizer:
    def __init__(self, llm_client=None):
        self.llm_client = llm_client  # Expects a Gemini client or similar with generate_content
        self.ledger: Dict[str, TaskLedgerEntry] = {}
        self.active_task_id: Optional[str] = None
        self.message_history: List[Dict[str, Any]] = []

    def register_task(self, task_id: str, description: str):
        """Registers a newly received task in the ledger."""
        self.ledger[task_id] = TaskLedgerEntry(task_id=task_id, description=description)

    def mark_task_active(self, task_id: str):
        """Marks a task as currently being executed by the UI Agent."""
        if task_id in self.ledger:
            self.ledger[task_id].status = "ACTIVE"
            self.ledger[task_id].started_at = time.time()
            self.active_task_id = task_id

    def update_task_context(self, task_id: str, action: str = None, observable: str = None, resource: str = None):
        """Live updates from the agentic planner about what it is currently doing."""
        if task_id in self.ledger:
            entry = self.ledger[task_id]
            if action:
                entry.recent_actions.append(action)
                if len(entry.recent_actions) > 5:
                    entry.recent_actions.pop(0)
            if observable and observable not in entry.observables:
                entry.observables.append(observable)
            if resource and resource not in entry.resources:
                entry.resources.append(resource)

    def mark_task_complete(self, task_id: str, success: bool):
        if task_id in self.ledger:
            self.ledger[task_id].status = "COMPLETED" if success else "FAILED"
            if self.active_task_id == task_id:
                self.active_task_id = None

    def process_message(self, human_message: str, current_screen_summary: str = "") -> Dict[str, Any]:
        """
        Takes raw unstructured human text, prompts the LLM with the current Task Ledger, 
        and returns a structured categorization.
        """
        self.message_history.append({"timestamp": time.time(), "sender": "user", "text": human_message})
        
        if not self.llm_client:
            # Fallback heuristic if no LLM
            return self._heuristic_categorize(human_message)

        prompt = self._build_categorization_prompt(human_message, current_screen_summary)
        
        try:
            # Assuming Google GenAI client syntax
            response = self.llm_client.models.generate_content(
                model='gemini-2.5-pro',
                contents=prompt,
                config={
                    "response_mime_type": "application/json",
                    "system_instruction": "You are the HITL Contextualizer Engine for an autonomous UI Agent.",
                }
            )
            data = json.loads(response.text)
            
            category = MessageCategory(data.get("category", "NOISE_SPECULATIVE"))
            correlated_task = data.get("correlated_task_id")
            
            # If it's feedback but the LLM failed to correlate, default to active task
            if category == MessageCategory.FEEDBACK_CORRECTION and not correlated_task:
                correlated_task = self.active_task_id
                
            result = {
                "category": category.value,
                "correlated_task_id": correlated_task,
                "extracted_intent": data.get("extracted_intent", human_message),
                "urgency": data.get("urgency", "NORMAL"),
                "raw_message": human_message
            }
            
            self.message_history.append({"timestamp": time.time(), "sender": "system_categorized", "data": result})
            return result
            
        except Exception as e:
            print(f"[HITLContextualizer] LLM classification failed: {e}. Using fallback.")
            return self._heuristic_categorize(human_message)

    def _build_categorization_prompt(self, message: str, screen_summary: str) -> str:
        prompt = "Analyze the following asynchronous human message sent to the UI Agent.\n\n"
        prompt += "CURRENT LEDGER OF TASKS:\n"
        
        if not self.ledger:
            prompt += "- (Idle, no tasks currently in ledger)\n"
        else:
            for tid, t in self.ledger.items():
                active_str = " (ACTIVE)" if t.status == "ACTIVE" else ""
                prompt += f"[{tid}]{active_str} {t.status} - Goal: {t.description}\n"
                if t.status == "ACTIVE":
                    prompt += f"  Recent actions: {', '.join(t.recent_actions)}\n"
                    prompt += f"  Observing: {', '.join(t.observables)}\n"
        
        if screen_summary:
             prompt += f"\nCURRENT ON-SCREEN CONTEXT:\n{screen_summary}\n"
             
        prompt += f"\nNEW HUMAN MESSAGE:\n\"{message}\"\n\n"
        
        prompt += "CATEGORIES:\n"
        prompt += "1. NEW_TASK: A distinct new objective.\n"
        prompt += "2. FEEDBACK_CORRECTION: Advice, guidance, or correction clearly related to the ACTIVE task (e.g., 'click the blue one', 'scroll down more').\n"
        prompt += "3. INTERRUPT_STOP: A safety stop command (e.g., 'stop', 'cancel', 'wait don't do that').\n"
        prompt += "4. NOISE_SPECULATIVE: Rhetorical questions, chitchat, or observations not requiring urgent course correction.\n\n"
        
        prompt += "Respond strictly as a JSON object with this schema:\n"
        prompt += "{\n"
        prompt += "  \"category\": \"NEW_TASK\" | \"FEEDBACK_CORRECTION\" | \"INTERRUPT_STOP\" | \"NOISE_SPECULATIVE\",\n"
        prompt += "  \"correlated_task_id\": \"<task_id_from_ledger>\" (or null if NEW_TASK or NOISE),\n"
        prompt += "  \"extracted_intent\": \"Clear, concise summary of what the user actually wants the agent to do\",\n"
        prompt += "  \"urgency\": \"NORMAL\" | \"HIGH\" | \"CRITICAL\"\n"
        prompt += "}\n"
        
        return prompt

    def _heuristic_categorize(self, message: str) -> Dict[str, Any]:
        """Simple keyword-based fallback if LLM is unavailable."""
        msg = message.lower()
        
        if any(w in msg for w in ["stop", "cancel", "halt", "wait", "abort", "don't"]):
            cat = MessageCategory.INTERRUPT_STOP
            urgency = "CRITICAL"
        elif any(w in msg for w in ["no", "click", "scroll", "instead", "other", "wrong", "try"]):
            cat = MessageCategory.FEEDBACK_CORRECTION
            urgency = "HIGH"
        elif any(w in msg for w in ["open", "go to", "create", "new", "do "]):
            cat = MessageCategory.NEW_TASK
            urgency = "NORMAL"
        else:
            cat = MessageCategory.NOISE_SPECULATIVE
            urgency = "NORMAL"
            
        return {
            "category": cat.value,
            "correlated_task_id": self.active_task_id if cat == MessageCategory.FEEDBACK_CORRECTION else None,
            "extracted_intent": message,
            "urgency": urgency,
            "raw_message": message
        }
