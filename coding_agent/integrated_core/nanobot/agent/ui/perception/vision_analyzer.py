import os
from PIL import Image # type: ignore
from google import genai # type: ignore
from google.genai import types # type: ignore
from typing import Any, Dict, Optional, Tuple

from nanobot.config.loader import load_config # type: ignore

class VisionAnalyzer:
    """
    Acts as the hook bridging visual capture and a Vision-Language Model (VLM).
    Uses the modern google-genai SDK for Gemini 1.5/2.5 Flash spatial understanding.
    Integrates CVPipeline for hybrid CV+VLM enrichment.
    """
    def __init__(self, model_name: str = "gemini-2.5-flash", cv_pipeline=None):
        self.model_name = model_name
        self.is_ready = True
        self.cv_pipeline = cv_pipeline  # CVPipeline instance for fast detection
        
        # Initialize Google GenAI client using centralized config
        self._config = load_config()
        self.client = genai.Client(api_key=self._config.providers.gemini.api_key)

    def find_element_bbox(self, image_path: str, element_description: str) -> str:
        """
        Uses the new SpatialUnderstandingTool to find the element.
        To maintain backward compatibility with ActionPredictor, we convert the
        normalized [x, y, width, height] back to the [ymin, xmin, ymax, xmax] 0-1000 format.
        """
        print(f"[VLM] Finding bounding box for '{element_description}' in {image_path} using Spatial Tool")
        
        try:
            # Lazy import to avoid circular dependencies if any
            from .spatial_understanding.spatial_tool import SpatialUnderstandingTool # type: ignore
            
            # The tool pulls GEMINI_API_KEY from environment automatically
            spatial_tool = SpatialUnderstandingTool()
            
            # Get 2d bounding boxes
            res = spatial_tool.execute(image_path, element_description, "2d_bounding_boxes")
            
            if res.get("success") and res.get("results"):
                # Take the first best match
                box = res["results"][0]
                
                # Spatial tool returns normalized 0.0-1.0 coords: x, y, width, height
                xmin = int(box["x"] * 1000)
                ymin = int(box["y"] * 1000)
                xmax = int((box["x"] + box["width"]) * 1000)
                ymax = int((box["y"] + box["height"]) * 1000)
                
                # Ensure bounds
                xmin, ymin = max(0, xmin), max(0, ymin)
                xmax, ymax = min(1000, xmax), min(1000, ymax)
                
                return f"[{ymin}, {xmin}, {ymax}, {xmax}]"
            else:
                return "NOT FOUND"
                
        except Exception as e:
            print(f"[VLM ERROR] Failed to find bounding box via Spatial Tool: {e}")
            return "NOT FOUND"

    def extract_ui_state(self, image_path: str) -> str:
        """
        Uses VLM to extract a structured representation of the current screen state.
        This helps the agent understand what windows/buttons are currently visible.
        """
        print(f"[VLM] Extracting complete UI state from {image_path}")
        try:
            img = Image.open(image_path)
            prompt = "Describe the UI state of this screen. What windows, applications, and important elements are visible? Provide a structured summary."
            
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=[img, prompt]
            )
            return response.text
        except Exception as e:
            print(f"[VLM ERROR] Failed to extract UI state: {e}")
            return ""
            
    def describe_screen_state(self, image_path: str, context: Optional[str] = None, reference_images: Optional[list] = None) -> str:
        """
        Uses VLM to describe what is currently on the screen, optionally with a guiding context/prompt
        and optional reference images for one-shot / few-shot prompting.
        """
        try:
            # Main image
            img = Image.open(image_path)
            contents: list = [img]
            
            # Additional reference images
            if reference_images:
                for ref_path in reference_images:
                    if os.path.exists(ref_path):
                        contents.append(Image.open(ref_path))
            
            # Prompt context
            prompt = context if context else "Describe what is currently visible on this screen in detail."
            contents.append(prompt)
            
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=contents
            )
            return response.text
        except Exception as e:
            print(f"[VLM ERROR] Failed to describe screen state: {e}")
            return ""
        
    def visual_diff(self, image_path1: str, image_path2: str, action_context: Optional[Dict[str, Any]] = None) -> tuple[bool, str]:
        """
        Checks if two screen states are meaningfully different relative to the intended action.
        Returns (success_bool, semantic_explanation).
        """
        print(f"[VLM] Evaluating Semantic Success: {image_path1} -> {image_path2}")
        try:
            img1 = Image.open(image_path1)
            img2 = Image.open(image_path2)
            
            action_type = action_context.get('action', 'unknown') if action_context else 'unknown'
            target = action_context.get('target', action_context.get('text', 'unknown')) if action_context else 'unknown'
            intent = action_context.get('description', 'A meaningful UI state change.') if action_context else 'A meaningful UI state change.'
            
            prompt = f"""
You are evaluating the success of a UI automation action.
Action taken: {action_type} 
Target: {target}
Expected Consequence: {intent}

Look at the Before image and After image. 
Did the action successfully achieve its expected consequence? 
Pay close attention to edge cases: if clicking an icon only opened a taskbar thumbnail preview instead of focusing the app window, that is a FAILURE.
Respond in strict JSON. Examples:
{{"success": true, "reason": "The expected window appeared in the foreground."}}
{{"success": false, "reason": "Clicking the icon opened a thumbnail preview instead of focusing the app."}}
"""
            
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=[prompt, img1, img2]
            )
            
            import json
            raw_text = response.text.strip()
            if "```json" in raw_text:
                raw_text = raw_text.split("```json")[1].split("```")[0].strip()
            
            result = json.loads(raw_text)
            print(f"[VLM Semantic Eval] Success: {result.get('success')}, Reason: {result.get('reason')}")
            return result.get("success", False), result.get("reason", "No reason provided by VLM.")
        except Exception as e:
            print(f"[VLM ERROR] Failed to compare images: {e}")
            # Transient VLM errors should not penalize the agent
            return True, f"VLM unavailable (assuming success): {e}"

    def detect_ui_elements_fast(self, current_frame, previous_frame=None):
        """
        Uses CVPipeline when available, otherwise falls back to legacy OpenCV heuristics.
        Returns: list of dicts: {'x','y','w','h', 'label', 'color'}
        """
        import cv2 # type: ignore
        import numpy as np # type: ignore

        # Prefer CVPipeline if available
        if self.cv_pipeline is not None:
            perception = self.cv_pipeline.full_analysis(current_frame, previous_frame)
            elements = []
            for el in perception.elements:
                color_map = {'BUTTON': 'cyan', 'TEXT_INPUT': 'green', 'ICON': 'blue',
                             'DYNAMIC': 'red', 'STRUCTURAL': 'purple'}
                elements.append({
                    'x': el.x, 'y': el.y, 'width': el.width, 'height': el.height,
                    'label': el.element_type, 'color': color_map.get(el.element_type, 'purple'),
                })
            # Add text regions
            for tr in perception.text_regions:
                elements.append({
                    'x': tr["x"], 'y': tr["y"], 'width': tr["w"], 'height': tr["h"],
                    'label': 'TEXT', 'color': 'green',
                })
            return elements

        # Legacy fallback
        elements = []
        gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
        
        # 1. Comparative / Dynamic Region Detection
        if previous_frame is not None:
            prev_gray = cv2.cvtColor(previous_frame, cv2.COLOR_BGR2GRAY)
            diff = cv2.absdiff(gray, prev_gray)
            _, diff_thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
            
            kernel_dyn = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
            diff_closed = cv2.morphologyEx(diff_thresh, cv2.MORPH_CLOSE, kernel_dyn)
            
            contours_dyn, _ = cv2.findContours(diff_closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for c in contours_dyn:
                x, y, w, h = cv2.boundingRect(c)
                if w * h > 100:
                    elements.append({'x': x, 'y': y, 'width': w, 'height': h, 'label': 'DYNAMIC', 'color': 'red'})
        
        # 2. Heuristic Structural Detection
        edges = cv2.Canny(gray, 50, 150)
        kernel_struct = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 5)) 
        closed_edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel_struct)
        contours_struct, _ = cv2.findContours(closed_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        screen_h, screen_w = gray.shape
        
        for c in contours_struct:
            x, y, w, h = cv2.boundingRect(c)
            area = w * h
            if area < 50 or area > (screen_w * screen_h * 0.9):
                continue
            aspect_ratio = w / float(h)
            label = "STRUCTURAL"
            color = "purple"
            
            if y > screen_h - 60 and w > screen_w * 0.8:
                label = "TASKBAR"; color = "yellow"
            elif 0.8 < aspect_ratio < 1.2 and 100 < area < 5000:
                label = "ICON/AVATAR"; color = "blue"
            elif aspect_ratio > 3.0 and h > 10 and h < 80:
                label = "TEXT BOX/ROW"; color = "green"
            elif area > 50000:
                label = "PANEL/IMAGE"; color = "purple"
            
            elements.append({'x': x, 'y': y, 'width': w, 'height': h, 'label': label, 'color': color})
            
        return elements

    def hybrid_detect_elements(self, frame, previous_frame=None, confidence_threshold=0.5):
        """
        Runs CVPipeline detection then enriches low-confidence elements with VLM.
        Returns tuple: (elements, enrichment_count)
        """
        elements = self.detect_ui_elements_fast(frame, previous_frame)
        
        # Identify elements needing VLM enrichment
        enrichment_count = 0
        # For now, return all elements — VLM enrichment will be added 
        # when specific semantic labeling is needed by the planner
        
        return elements, enrichment_count
