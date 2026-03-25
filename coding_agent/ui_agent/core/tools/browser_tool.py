import asyncio
import time
import os
import json
import base64
import io
from typing import Dict, Any, Optional
from playwright.async_api import async_playwright
try:
    from PIL import Image
except ImportError:
    # Fallback if PIL is not available (though it should be for vision)
    Image = None

# Try relative and absolute imports for the coding_agent structure
try:
    from coding_agent.utils.gemini_client import init_gemini, get_gemini_response
except ImportError:
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
    from coding_agent.utils.gemini_client import init_gemini, get_gemini_response

class BrowserTool:
    """
    A functional Playwright-based browser automation tool.
    Uses Gemini to decide steps for a given objective.
    """
    def __init__(self, headless: bool = False):
        self.headless = headless
        self.is_running = False
        self._current_objective = ""
        self._start_time: float = 0
        self._task: Optional[asyncio.Task[Any]] = None

        self.model = init_gemini()
        
    async def _execute_logic(self, objective: str):
        """Internal execution loop using Playwright and Gemini."""
        print(f"[BrowserTool] Starting objective: '{objective}' (Headless: {self.headless})")
        self.is_running = True
        self._current_objective = objective
        self._start_time = time.time()
        
        async with async_playwright() as p:
            # Use a temporary user data dir to keep sessions across runs
            base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
            user_data_dir = os.path.join(base_dir, ".gemini", "browser_session")
            os.makedirs(user_data_dir, exist_ok=True)
            
            try:
                context = await p.chromium.launch_persistent_context(
                    user_data_dir=user_data_dir,
                    headless=self.headless,
                    viewport={'width': 1280, 'height': 720}
                )
                page = context.pages[0] if context.pages else await context.new_page()
                
                history = []
                max_steps = 15
                
                for step_num in range(1, max_steps + 1):
                    if not self.is_running: break
                    
                    # Capture state
                    url = page.url
                    title = await page.title()
                    
                    # Take a screenshot for Gemini Vision
                    screenshot_bytes = await page.screenshot()
                    screenshot_img = None
                    if Image:
                        screenshot_img = Image.open(io.BytesIO(screenshot_bytes))
                    
                    prompt = f"""You are a browser automation agent.
Objective: {objective}
Current URL: {url}
Page Title: {title}
Step: {step_num}/{max_steps}

History:
{json.dumps(history[-5:], indent=2)}

Available actions (JSON format):
{{"action": "goto", "url": "..."}}
{{"action": "click", "selector": "..."}}
{{"action": "fill", "selector": "...", "text": "..."}}
{{"action": "press", "selector": "...", "key": "Enter|Tab|Escape"}}
{{"action": "wait", "ms": 2000}}
{{"action": "complete", "reason": "..."}}
{{"action": "fail", "reason": "..."}}

Selectors should be Playwright-compatible (e.g. 'text=Login', '#submit', '[aria-label="Search"]').

Return ONLY the JSON for the NEXT action.
"""
                    # Use multimodal input if screenshot available
                    contents = [prompt]
                    if screenshot_img:
                        contents.append(screenshot_img)
                        
                    raw_response = get_gemini_response(self.model, contents)
                    
                    try:
                        # Clean markdown if present
                        processed_response = raw_response.strip()
                        if "```json" in processed_response:
                            processed_response = processed_response.split("```json")[-1].split("```")[0].strip()
                        elif "```" in processed_response:
                            processed_response = processed_response.split("```")[-1].split("```")[0].strip()
                        
                        decision = json.loads(processed_response)
                    except Exception as e:
                        print(f"[BrowserTool] Failed to parse decision: {raw_response}")
                        history.append({"step": step_num, "error": f"Parse error: {e}"})
                        continue
                        
                    action = decision.get("action")
                    params = decision
                    
                    print(f"[BrowserTool] Step {step_num}: {action} - {decision.get('reasoning', '')}")
                    history.append({"step": step_num, "decision": decision})
                    
                    if action == "goto":
                        await page.goto(params.get("url"))
                    elif action == "click":
                         # Add a small wait before click for stability
                         await page.wait_for_selector(params.get("selector"), timeout=5000)
                         await page.click(params.get("selector"))
                    elif action == "fill":
                         await page.wait_for_selector(params.get("selector"), timeout=5000)
                         await page.fill(params.get("selector"), params.get("text"))
                    elif action == "press":
                         await page.press(params.get("selector", "body"), params.get("key"))
                    elif action == "wait":
                         await asyncio.sleep(params.get("ms", 2000) / 1000)
                    elif action == "complete":
                        print(f"[BrowserTool] Success: {params.get('reason')}")
                        self.is_running = False
                        break
                    elif action == "fail":
                        print(f"[BrowserTool] Failed: {params.get('reason')}")
                        self.is_running = False
                        break
                    
                    await asyncio.sleep(1)
                    
            except Exception as e:
                print(f"[BrowserTool] Error during execution: {e}")
            finally:
                self.is_running = False
                if 'context' in locals():
                    await context.close()
                print("[BrowserTool] Browser instance closed.")

    def execute_objective(self, objective: str) -> asyncio.Task[Any]:
        """
        Spawns the browser automation task in the background.
        Returns the asyncio Task object so the supervisor can monitor/cancel it.
        """
        if self.is_running and self._task is not None:
            print("[BrowserTool] Tool is already running. Returning current task.")
            return self._task
            
        loop = asyncio.get_event_loop()
        self._task = loop.create_task(self._execute_logic(objective))
        return self._task


    def abort(self):
        """Cancels the current execution if it is running."""
        if self.is_running and self._task and not self._task.done():
            print(f"[BrowserTool] Aborting current objective: '{self._current_objective}'")
            self._task.cancel()
            self.is_running = False
        else:
            print("[BrowserTool] Nothing to abort.")

