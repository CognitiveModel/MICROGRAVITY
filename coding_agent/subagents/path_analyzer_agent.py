"""
PathAnalyzerAgent - Specialized subagent to analyze failed/recovered episodes.
Distills raw event logs into concise ErrorRecord objects for the experiential memory.
"""

import json
import time
from typing import List, Dict, Any, Optional
from google import genai # type: ignore
from google.genai import types # type: ignore
from coding_agent.utils import config
from coding_agent.ui_agent.planning.task_schema import ErrorRecord

class PathAnalyzerAgent:
    def __init__(self, model_name: str = "models/gemini-2.5-flash"):
        self.model_name = model_name
        api_key = config.GEMINI_API_KEY
        if not api_key:
            raise ValueError("GEMINI_API_KEY not configured")
        self.client = genai.Client(api_key=api_key)

    def analyze_episode(self, episode_steps: List[Dict[str, Any]]) -> Optional[ErrorRecord]:
        """
        Takes raw episode steps, identifies where the failure occurred and how the agent recovered,
        and returns a highly concise ErrorRecord.
        Returns None if there were no failures.
        """
        failed_steps_log = []
        recovery_steps_log = []
        has_failed = False
        
        for step in episode_steps:
            action_desc = f"{step.get('action') or step.get('action_type', '')} on '{step.get('target', '')}'"
            if not step.get("success", False):
                has_failed = True
                failed_steps_log.append(action_desc)
            elif has_failed:
                # Steps taken after the failure are recovery steps
                recovery_steps_log.append(action_desc)
                
        if not has_failed:
            return None # No error to track
            
        prompt = f"""You are the Path Analyzer subagent.
An agent attempted to perform a UI task but made a mistake, then recovered and succeeded.
Extract a highly concise Error Record so the agent avoids this wrong path next time.

FAILED STEPS:
{failed_steps_log}

RECOVERY STEPS:
{recovery_steps_log}

Respond in JSON ONLY with this exact structure:
{{
  "description": "<1 line description of the mistake/wrong path taken>",
  "recovery_summary": "<1 line describing the heuristic/steps used to find the right path and recover>"
}}"""
        
        try:
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=[prompt],
                config=types.GenerateContentConfig(
                    temperature=0.1,
                    response_mime_type="application/json"
                )
            )
            raw_text = response.text.strip()
            data = json.loads(raw_text)
            
            error_id = f"err_{int(time.time())}"
            return ErrorRecord(
                error_id=error_id,
                description=data.get("description", "Unknown error path"),
                recovery_summary=data.get("recovery_summary", "Unknown recovery"),
                failed_steps=failed_steps_log
            )
        except Exception as e:
            print(f"[PathAnalyzerAgent] Failed to analyze: {e}")
            return ErrorRecord(
                error_id=f"err_{int(time.time())}",
                description="Failed to parse error via LLM",
                recovery_summary="Fallback recovery",
                failed_steps=failed_steps_log
            )
