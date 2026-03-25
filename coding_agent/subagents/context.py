"""
Hyper-Active Context Engineering Subagent

Retrieves, prioritizes, and maps important context scopes.
Explicitly models the situational picture for the Introspection Agent,
stating exactly WHY certain code artifacts are relevant in a given scenario.
"""

import os
from coding_agent.utils.gemini_client import init_gemini, get_gemini_response
from typing import List, Dict

class ContextSubagent:
    def __init__(self):
        self.gemini_model = init_gemini()

    def engineer_context(self, task: str, files_available: List[str]) -> str:
        """
        Actively models the situation and retrieves relevant context with justifications.
        """
        prompt = f"""
TASK: {task}
FILES IN WORKSPACE: {files_available}

--- CONTEXT ENGINEERING TASK ---
You are a situational modeler. Your goal is to:
1. IDENTIFY the most critical context files for this task.
2. JUSTIFY exactly why each file is important for this specific execution scope.
3. MODEL THE SITUATION: Describe the architectural intersection where this task lives.

Output a structured "Situational Context Model" including the Importance List.
"""
        response = get_gemini_response(self.gemini_model, prompt)
        return response
