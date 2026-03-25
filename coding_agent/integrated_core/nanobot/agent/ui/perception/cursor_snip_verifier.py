"""
CursorSnipVerifier — Reflex-like visual position verification via VLM.

Captures a small region around the cursor position and sends it to the VLM
to verify that the cursor is pointing at the intended target. Fires reflexively
— only when triggered by specific conditions, not on every action.

Trigger conditions (any one is enough):
  1. CursorMonitor reports unexpected cursor type for the target element
  2. Coordinates were resolved via fallback (not CV cache hit)
  3. Previous action on the same target failed
  4. Target was zoom-resolved (small/hard-to-find element)
  5. Confidence from predictor is below threshold

Usage:
    verifier = CursorSnipVerifier(screen_observer, vision_analyzer, cursor_monitor)
    
    # Before clicking — verify position reflexively
    result = verifier.verify_before_click(
        x=500, y=300,
        target_description="Submit button",
        trigger_reasons=["cursor_mismatch"]
    )
    if result.verified:
        mouse.click()
    else:
        # Adjust coordinates or retry with zoom
        
    # After failed action — diagnostic snip
    diagnosis = verifier.diagnose_failure(
        x=500, y=300, 
        target_description="Submit button",
        action_performed="click"
    )
"""

import os
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional, List, Dict, Any, Tuple
from pathlib import Path


# ──────────────────────────  Trigger Reasons  ──────────────────────────

class VerifyTrigger(Enum):
    """Reasons why the reflex verification was triggered."""
    CURSOR_MISMATCH = auto()       # Cursor type doesn't match expectations
    FALLBACK_RESOLUTION = auto()   # Coords came from fallback, not CV cache
    PREVIOUS_FAILURE = auto()      # Same target failed in a previous attempt
    SMALL_TARGET = auto()          # Target was identified as small/precise
    ZOOM_RESOLVED = auto()         # Target required zoom to resolve
    LOW_CONFIDENCE = auto()        # Predictor confidence below threshold
    MANUAL_REQUEST = auto()        # Explicitly requested by caller
    POST_FAILURE_DIAGNOSTIC = auto()  # Diagnostic after a failed action


@dataclass
class VerifyResult:
    """Result of a cursor position verification."""
    verified: bool                         # True if VLM confirms correct position
    confidence: float = 0.0               # VLM's confidence (0.0-1.0)
    target_description: str = ""          # What we were looking for
    vlm_response: str = ""                # Raw VLM response text
    actual_element: str = ""              # What VLM thinks is under cursor
    cursor_type: str = ""                 # Cursor type at verification time
    position: Tuple[int, int] = (0, 0)    # Cursor position during verification
    snip_path: str = ""                   # Path to saved snip image
    trigger_reasons: List[str] = field(default_factory=list)
    timestamp: float = 0.0
    correction_suggestion: Optional[Dict[str, int]] = None  # Suggested offset {dx, dy}


class CursorSnipVerifier:
    """
    Reflex-like visual position verification via VLM.
    
    Captures a small region around the cursor, marks the cursor position,
    and sends it to the VLM to confirm the cursor is on the intended target.
    
    Designed to fire like a reflex — fast, automatic, and only when needed.
    """
    
    # Snip region sizes (half-width/height from cursor)
    SNIP_SMALL = 40       # 80×80 px region — for close-range precision
    SNIP_MEDIUM = 80      # 160×160 px region — default context
    SNIP_LARGE = 150      # 300×300 px region — for large/spread-out elements
    
    # Cooldown: minimum seconds between reflex verifications
    COOLDOWN_SECONDS = 2.0
    
    # Maximum verifications per task (to avoid infinite loops)
    MAX_VERIFICATIONS_PER_TASK = 15
    
    def __init__(self, screen_observer=None, vision_analyzer=None, 
                 cursor_monitor=None, screen_geometry=None,
                 snip_dir: Optional[str] = None):
        """
        Args:
            screen_observer: ScreenObserver for capturing screen regions
            vision_analyzer: VisionAnalyzer for VLM queries
            cursor_monitor: CursorMonitor for cursor type context
            screen_geometry: ScreenGeometry for coordinate bounds
            snip_dir: Directory to save diagnostic snip images
        """
        self.screen_observer = screen_observer
        self.vision = vision_analyzer
        self.cursor_monitor = cursor_monitor
        self.screen_geometry = screen_geometry
        self.snip_dir = snip_dir
        
        # Rate limiting
        self._last_verify_time: float = 0.0
        self._verify_count: int = 0
        
        # Failure memory: tracks targets that have failed before
        self._failure_history: Dict[str, int] = {}  # target -> failure count
        
        if snip_dir:
            os.makedirs(snip_dir, exist_ok=True)
        
        print("[CursorSnipVerifier] Initialized")
    
    # ═══════════════════════  Smart Triggering  ═══════════════════════
    
    def should_trigger(self, target: str, prediction_source: str = "",
                       cursor_valid: Optional[bool] = None,
                       is_small_target: bool = False,
                       was_zoom_resolved: bool = False,
                       confidence: float = 1.0) -> Tuple[bool, List[VerifyTrigger]]:
        """
        Decides whether to trigger the reflex verification.
        Returns (should_trigger, list_of_trigger_reasons).
        
        This is the brain of the reflex system — it avoids firing too
        frequently while ensuring critical verifications happen.
        """
        triggers: List[VerifyTrigger] = []
        
        # Rate limiting: don't fire too often
        elapsed = time.time() - self._last_verify_time
        if elapsed < self.COOLDOWN_SECONDS:
            return False, []
        
        # Budget limiting: don't exceed max per task
        if self._verify_count >= self.MAX_VERIFICATIONS_PER_TASK:
            return False, []
        
        # Trigger 1: Cursor type mismatch
        if cursor_valid is False:
            triggers.append(VerifyTrigger.CURSOR_MISMATCH)
        
        # Trigger 2: Fallback resolution (not from CV cache)
        if prediction_source in ("vlm_fallback", "static_vlm", "hint_coords_direct",
                                  "fallback", "live_api_fallback"):
            triggers.append(VerifyTrigger.FALLBACK_RESOLUTION)
        
        # Trigger 3: Target has failed before
        target_key = target.lower().strip()
        if target_key in self._failure_history and self._failure_history[target_key] > 0:
            triggers.append(VerifyTrigger.PREVIOUS_FAILURE)
        
        # Trigger 4: Small target
        if is_small_target:
            triggers.append(VerifyTrigger.SMALL_TARGET)
        
        # Trigger 5: Zoom-resolved (implies difficulty)
        if was_zoom_resolved:
            triggers.append(VerifyTrigger.ZOOM_RESOLVED)
        
        # Trigger 6: Low confidence
        if confidence < 0.7:
            triggers.append(VerifyTrigger.LOW_CONFIDENCE)
        
        should_fire = len(triggers) > 0
        return should_fire, triggers
    
    def record_failure(self, target: str):
        """Records a target failure for future trigger decisions."""
        key = target.lower().strip()
        self._failure_history[key] = self._failure_history.get(key, 0) + 1
    
    def record_success(self, target: str):
        """Records a target success, reducing future trigger sensitivity."""
        key = target.lower().strip()
        if key in self._failure_history:
            self._failure_history[key] = max(0, self._failure_history[key] - 1)
    
    def reset_task_budget(self):
        """Resets the per-task verification budget (call at task start)."""
        self._verify_count = 0
        self._failure_history.clear()
    
    # ═══════════════════════  Core Verification  ═══════════════════════
    
    def verify_before_click(self, x: int, y: int, target_description: str,
                            trigger_reasons: Optional[List[VerifyTrigger]] = None,
                            snip_radius: Optional[int] = None) -> VerifyResult:
        """
        PRE-CLICK REFLEX: Snips the area around (x, y), sends to VLM to confirm
        the cursor is pointing at the intended target.
        
        Call this AFTER move_to() but BEFORE click(). The mouse should already
        be at position (x, y).
        
        Args:
            x, y: Current cursor position (should match mouse position)
            target_description: What the cursor should be pointing at
            trigger_reasons: Why this verification was triggered
            snip_radius: Half-size of the snip region (auto-selected if None)
            
        Returns:
            VerifyResult with verification outcome
        """
        # Rate limit
        self._last_verify_time = time.time()
        self._verify_count += 1
        
        radius = snip_radius or self.SNIP_MEDIUM
        
        print(f"[CursorSnipVerifier] PRE-CLICK VERIFY at ({x}, {y}) for '{target_description}' "
              f"(triggers: {[t.name for t in (trigger_reasons or [])]})")
        
        # 1. Capture snip around cursor
        snip_img, snip_path = self._capture_cursor_snip(x, y, radius, label="pre_click")
        if snip_img is None:
            return VerifyResult(
                verified=True,  # Fail-open: don't block if capture fails
                target_description=target_description,
                vlm_response="Snip capture failed — proceeding without verification",
                position=(x, y),
                trigger_reasons=[t.name for t in (trigger_reasons or [])],
                timestamp=time.time(),
            )
        
        # 2. Get cursor context
        cursor_type_str = ""
        if self.cursor_monitor:
            cursor_type_str = self.cursor_monitor.get_cursor_type().name
        
        # 3. Query VLM
        prompt = (
            f"The red crosshair/dot marks the exact cursor position in this cropped screenshot region. "
            f"The intended target is: '{target_description}'.\n\n"
            f"Question: Is the cursor (marked position) pointing at or very near "
            f"'{target_description}'?\n\n"
            f"Respond in JSON: "
            f"{{\"verified\": true/false, \"confidence\": 0.0-1.0, "
            f"\"actual_element\": \"what is actually under the cursor\", "
            f"\"correction\": \"none\" or \"move_left/right/up/down by N pixels\"}}"
        )
        
        vlm_response = self._query_vlm_with_snip(snip_path, prompt)
        
        # 4. Parse result
        result = self._parse_verify_response(vlm_response, target_description,
                                              cursor_type_str, (x, y), snip_path,
                                              trigger_reasons)
        
        if result.verified:
            print(f"[CursorSnipVerifier] ✓ VERIFIED: cursor is on '{target_description}' "
                  f"(confidence: {result.confidence:.1%})")
        else:
            print(f"[CursorSnipVerifier] ✗ MISMATCH: cursor is on '{result.actual_element}', "
                  f"not '{target_description}' (confidence: {result.confidence:.1%})")
            if result.correction_suggestion:
                print(f"[CursorSnipVerifier] Suggested correction: {result.correction_suggestion}")
        
        return result
    
    def diagnose_failure(self, x: int, y: int, target_description: str,
                         action_performed: str = "click") -> VerifyResult:
        """
        POST-FAILURE DIAGNOSTIC: Called after an action fails. Captures the cursor
        area BEFORE the mouse moves away to understand what went wrong.
        
        This is more detailed than pre-click verification — it asks the VLM
        to explain what element is under the cursor and why the action might
        have failed.
        
        Args:
            x, y: Position where the action was performed
            target_description: Intended target element
            action_performed: What action was attempted (click, double_click, etc.)
            
        Returns:
            VerifyResult with diagnostic information
        """
        self._last_verify_time = time.time()
        self._verify_count += 1
        
        print(f"[CursorSnipVerifier] POST-FAILURE DIAGNOSTIC at ({x}, {y}) "
              f"for '{action_performed}' on '{target_description}'")
        
        # Record failure for future trigger decisions
        self.record_failure(target_description)
        
        # Capture with larger radius for more context
        snip_img, snip_path = self._capture_cursor_snip(x, y, self.SNIP_LARGE, 
                                                         label="post_fail")
        if snip_img is None:
            return VerifyResult(
                verified=False,
                target_description=target_description,
                vlm_response="Diagnostic snip capture failed",
                position=(x, y),
                trigger_reasons=[VerifyTrigger.POST_FAILURE_DIAGNOSTIC.name],
                timestamp=time.time(),
            )
        
        cursor_type_str = ""
        if self.cursor_monitor:
            cursor_type_str = self.cursor_monitor.get_cursor_type().name
        
        prompt = (
            f"A '{action_performed}' action was attempted at the red crosshair/dot "
            f"position in this screenshot region. The intended target was: "
            f"'{target_description}'. The action FAILED.\n\n"
            f"Current cursor type: {cursor_type_str or 'unknown'}\n\n"
            f"Analyze the area and answer:\n"
            f"1. What UI element is actually at the marked position?\n"
            f"2. Why might the action have failed?\n"
            f"3. Where should the cursor be moved to hit '{target_description}'?\n\n"
            f"Respond in JSON: "
            f"{{\"verified\": false, \"actual_element\": \"...\", "
            f"\"failure_reason\": \"...\", "
            f"\"correction\": \"move_left/right/up/down by N pixels\" or \"target_not_visible\"}}"
        )
        
        vlm_response = self._query_vlm_with_snip(snip_path, prompt)
        
        result = self._parse_verify_response(vlm_response, target_description,
                                              cursor_type_str, (x, y), snip_path,
                                              [VerifyTrigger.POST_FAILURE_DIAGNOSTIC])
        
        print(f"[CursorSnipVerifier] Diagnostic: actual='{result.actual_element}', "
              f"correction={result.correction_suggestion}")
        
        return result
    
    # ═══════════════════════  Snip Capture  ═══════════════════════
    
    def _capture_cursor_snip(self, x: int, y: int, radius: int,
                              label: str = "snip") -> Tuple[Any, str]:
        """
        Captures a small region around (x, y) and draws a crosshair marker
        at the cursor position for the VLM to see.
        
        Returns:
            (PIL.Image or None, saved_file_path or "")
        """
        if not self.screen_observer:
            return None, ""
        
        try:
            from PIL import Image, ImageDraw  # type: ignore
            
            # Calculate snip bounds (in screen coordinates)
            screen_w, screen_h = (1920, 1080)
            if self.screen_geometry:
                screen_w, screen_h = self.screen_geometry.logical_size
            
            left = max(0, x - radius)
            top = max(0, y - radius)
            width = min(radius * 2, screen_w - left)
            height = min(radius * 2, screen_h - top)
            
            if width <= 0 or height <= 0:
                return None, ""
            
            # Capture the region
            snip_img = self.screen_observer.capture_as_pil(
                region=(left, top, width, height)
            )
            
            # Draw crosshair at cursor position within the snip
            cursor_in_snip_x = x - left
            cursor_in_snip_y = y - top
            
            draw = ImageDraw.Draw(snip_img)
            marker_size = 8
            color = (255, 0, 0)  # Red crosshair
            
            # Crosshair lines
            draw.line([(cursor_in_snip_x - marker_size, cursor_in_snip_y),
                       (cursor_in_snip_x + marker_size, cursor_in_snip_y)],
                      fill=color, width=2)
            draw.line([(cursor_in_snip_x, cursor_in_snip_y - marker_size),
                       (cursor_in_snip_x, cursor_in_snip_y + marker_size)],
                      fill=color, width=2)
            # Small circle
            draw.ellipse([(cursor_in_snip_x - 3, cursor_in_snip_y - 3),
                          (cursor_in_snip_x + 3, cursor_in_snip_y + 3)],
                         outline=color, width=2)
            
            # Save snip
            snip_path = ""
            if self.snip_dir:
                timestamp = int(time.time() * 1000)
                filename = f"{label}_{timestamp}_{x}_{y}.png"
                snip_path = os.path.join(self.snip_dir, filename)
                snip_img.save(snip_path)
            
            return snip_img, snip_path
            
        except Exception as e:
            print(f"[CursorSnipVerifier] Snip capture error: {e}")
            return None, ""
    
    # ═══════════════════════  VLM Query  ═══════════════════════
    
    def _query_vlm_with_snip(self, snip_path: str, prompt: str) -> str:
        """Sends the snip image to the VLM with the given prompt."""
        if not self.vision or not snip_path:
            return ""
        
        try:
            # Use the vision analyzer's existing infrastructure
            response = self.vision.analyze_image_with_prompt(snip_path, prompt)
            return response if response else ""
        except AttributeError:
            # Fallback: try describe_screen_state if analyze_image_with_prompt not available
            try:
                response = self.vision.describe_screen_state(snip_path, context=prompt)
                return response if response else ""
            except Exception as e:
                print(f"[CursorSnipVerifier] VLM query error: {e}")
                return ""
        except Exception as e:
            print(f"[CursorSnipVerifier] VLM query error: {e}")
            return ""
    
    # ═══════════════════════  Response Parsing  ═══════════════════════
    
    def _parse_verify_response(self, vlm_response: str, target: str,
                                cursor_type: str, position: Tuple[int, int],
                                snip_path: str,
                                triggers: Optional[List[VerifyTrigger]] = None) -> VerifyResult:
        """Parses the VLM response into a VerifyResult."""
        import json
        
        verified = True  # Default fail-open
        confidence = 0.5
        actual_element = ""
        correction = None
        
        if vlm_response:
            try:
                # Try to extract JSON from the response
                json_str = vlm_response
                # Handle markdown code blocks
                if "```" in json_str:
                    lines = json_str.split("```")
                    for block in lines:
                        block = block.strip()
                        if block.startswith("json"):
                            block = block[4:].strip()
                        if block.startswith("{"):
                            json_str = block
                            break
                
                # Find the JSON object
                start = json_str.find("{")
                end = json_str.rfind("}") + 1
                if start >= 0 and end > start:
                    data = json.loads(json_str[start:end])
                    verified = data.get("verified", True)
                    confidence = float(data.get("confidence", 0.5))
                    actual_element = data.get("actual_element", "")
                    
                    # Parse correction suggestion
                    correction_str = data.get("correction", "none")
                    if correction_str and correction_str != "none":
                        correction = self._parse_correction(correction_str)
                        
            except (json.JSONDecodeError, ValueError, KeyError) as e:
                print(f"[CursorSnipVerifier] Parse error: {e}")
                # Heuristic fallback: check for keywords
                lower = vlm_response.lower()
                if "not" in lower and ("pointing" in lower or "at" in lower or "on" in lower):
                    verified = False
                elif "yes" in lower or "correct" in lower:
                    verified = True
        
        return VerifyResult(
            verified=verified,
            confidence=confidence,
            target_description=target,
            vlm_response=vlm_response[:500] if vlm_response else "",
            actual_element=actual_element,
            cursor_type=cursor_type,
            position=position,
            snip_path=snip_path,
            trigger_reasons=[t.name for t in (triggers or [])],
            timestamp=time.time(),
            correction_suggestion=correction,
        )
    
    def _parse_correction(self, correction_str: str) -> Optional[Dict[str, int]]:
        """Parses a correction string like 'move_right by 15 pixels' into {dx, dy}."""
        try:
            import re
            lower = correction_str.lower()
            
            # Extract pixel amount
            nums = re.findall(r'(\d+)', lower)
            amount = int(nums[0]) if nums else 5  # Default 5px
            
            dx, dy = 0, 0
            if "left" in lower:
                dx = -amount
            elif "right" in lower:
                dx = amount
            if "up" in lower:
                dy = -amount
            elif "down" in lower:
                dy = amount
            
            if dx != 0 or dy != 0:
                return {"dx": dx, "dy": dy}
        except Exception:
            pass
        
        return None
