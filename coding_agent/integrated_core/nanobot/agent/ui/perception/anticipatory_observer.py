import cv2 # type: ignore
import numpy as np # type: ignore
from typing import Dict, Any, Optional

try:
    import pytesseract # type: ignore
    HAS_PYTESSERACT = True
except ImportError:
    HAS_PYTESSERACT = False


class ProcessContext:
    """
    Shared context between vision modules (Live Streamer, Static VLM, CV) 
    to ensure grounding across different fidelity levels.
    """
    def __init__(self):
        self.active_hwnd = None
        self.window_title = None
        self.url = None
        self.active_element_id = None
        self.rect = None # (left, top, right, bottom)
        self.timestamp = 0

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if hasattr(self, k):
                setattr(self, k, v)
        self.timestamp = 0 # Invalidate or update timestamp

class AnticipatoryObserver:
    """
    Feedback orchestrator that selectively runs fast OpenCV checks or 
    triggers high-resolution VLM 'insights' based on action complexity.
    """
    
    def __init__(self, memory_agent=None, cv_pipeline=None, shared_context=None):
        self.memory_agent = memory_agent
        self.cv_pipeline = cv_pipeline
        self.shared_context = shared_context or ProcessContext()
        self._custom_observables: Dict[str, Dict[str, Any]] = {}  # name → {code, target_pattern, ...}

    def verify_observable(self, observable_def: Dict[str, Any], 
                          state_before: np.ndarray, 
                          state_after: np.ndarray, 
                          action: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evaluates the expected observable. Can return a boolean success
        or a request for high-fidelity VLM analysis.
        """
        if state_after is None:
            return {"success": False, "reason": "Missing after frame"}
            
        obs_type = observable_def.get("type", "")
        
        # Check before frame requirement
        requires_before = obs_type in ["color_shift", "structural_change", "frame_transition"]
        if requires_before and state_before is None:
            return {"success": False, "reason": "Missing before frame for anticipatory check"}
        # Selective judgment: Should we bypass VLM?
        is_fast_candidate = obs_type in ["color_shift", "structural_change", "frame_transition"]
        
        print(f"[AnticipatoryObserver] Evaluating '{obs_type}' (Fast candidate: {is_fast_candidate})")
        
        try:
            # Optionally detect and log peripheral changes if we have a before frame
            peripheral = []
            if requires_before and state_before is not None and state_after is not None:
                peripheral = self.detect_change_regions(state_before, state_after)

            if obs_type == "color_shift":
                success = self._check_color_shift(observable_def, state_before, state_after, action)
                return {"success": success, "method": "CV_FAST", "peripheral_changes": peripheral}
                
            elif obs_type == "structural_change":
                success = self._check_structural_change(observable_def, state_before, state_after)
                return {"success": success, "method": "CV_FAST", "peripheral_changes": peripheral}
                
            elif obs_type == "frame_transition":
                success = self._check_frame_transition(observable_def, state_before, state_after)
                return {"success": success, "method": "CV_FAST", "peripheral_changes": peripheral}
                
            elif obs_type == "element_appears":
                success = self._check_element_appears(observable_def, state_after, action)
                return {"success": success, "method": "CV_TEMPLATE"}
                
            elif obs_type == "text_appears":
                success = self._check_text_appears(observable_def, state_after, action)
                return {"success": success, "method": "CV_OCR"}
                
            elif obs_type == "text_match":
                success = self._check_text_match(observable_def, state_before, state_after, action)
                return {"success": success, "method": "CV_OCR"}
                
            elif obs_type == "vlm_insight":
                # This explicitly requests a bridge to the High-Resolution VLM
                return {
                    "request_fidelity": "HIGH",
                    "reason": observable_def.get("query", "Verify state change"),
                    "context": vars(self.shared_context)
                }
                
            elif obs_type == "custom_script":
                success = self._execute_custom_script(observable_def, state_before, state_after, action)
                return {"success": success, "method": "CUSTOM_CV"}
                
            else:
                return {"success": False, "reason": f"Unknown type: {obs_type}"}
        except Exception as e:
            print(f"[AnticipatoryObserver] Error: {e}")
            return {"success": False, "error": str(e)}

    def _get_local_roi(self, image: np.ndarray, action: Dict[str, Any], size: int = 50) -> Optional[np.ndarray]:
        # ... (Method logic remains the same)
        x = action.get('resolved_x', action.get('x'))
        y = action.get('resolved_y', action.get('y'))
        
        if x is None or y is None:
            return None
            
        h, w = image.shape[:2]
        half = size // 2
        
        x1 = max(0, x - half) # type: ignore
        y1 = max(0, y - half) # type: ignore
        x2 = min(w, x + half) # type: ignore
        y2 = min(h, y + half) # type: ignore
        
        if x2 > x1 and y2 > y1:
            return image[y1:y2, x1:x2]
        return None

    def _check_color_shift(self, obs_def: Dict[str, Any], before: np.ndarray, after: np.ndarray, action: Dict[str, Any]) -> bool:
        region = obs_def.get("target_region", "local")
        threshold = obs_def.get("threshold", 15.0)
        
        if region == "local":
            roi_before = self._get_local_roi(before, action, size=obs_def.get("roi_size", 60))
            roi_after = self._get_local_roi(after, action, size=obs_def.get("roi_size", 60))
        else:
            roi_before, roi_after = before, after
            
        if roi_before is None or roi_after is None: return False
        return np.linalg.norm(np.mean(roi_before, axis=(0,1)) - np.mean(roi_after, axis=(0,1))) >= threshold

    def _check_structural_change(self, obs_def: Dict[str, Any], before: np.ndarray, after: np.ndarray) -> bool:
        threshold = obs_def.get("threshold", 0.15)
        diff = cv2.absdiff(before, after)
        _, thresh = cv2.threshold(cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY), 30, 255, cv2.THRESH_BINARY)
        return (cv2.countNonZero(thresh) / (before.shape[0] * before.shape[1])) >= threshold # type: ignore

    def _check_element_appears(self, obs_def: Dict[str, Any], after: np.ndarray, action: Dict[str, Any]) -> bool:
        label = obs_def.get("template_label")
        if not label or not self.memory_agent: return False
        atlas_data = self.memory_agent.recall_element(action.get("app_window", "Desktop"), label)
        if not atlas_data or not atlas_data.get("template_path"): return False
        template = cv2.imread(atlas_data["template_path"])
        if template is None: return False
        _, max_val, _, _ = cv2.minMaxLoc(cv2.matchTemplate(after, template, cv2.TM_CCOEFF_NORMED))
        return max_val >= obs_def.get("threshold", 0.85)

    def _check_text_appears(self, obs_def: Dict[str, Any], after: np.ndarray, action: Dict[str, Any]) -> bool:
        if not HAS_PYTESSERACT:
            print("[AnticipatoryObserver] PyTesseract is missing. Cannot check text.")
            return False
            
        expected_text = obs_def.get("expected_text", "").lower()
        if not expected_text:
            return False
            
        # Optional: constrain search space to local ROI
        region = obs_def.get("target_region", "local")
        if region == "local":
            roi_after = self._get_local_roi(after, action, size=obs_def.get("roi_size", 150))
            if roi_after is None: roi_after = after
        else:
            roi_after = after
            
        text = pytesseract.image_to_string(roi_after).lower()
        return expected_text in text

    def _check_text_match(self, obs_def: Dict[str, Any], before: np.ndarray, after: np.ndarray, action: Dict[str, Any]) -> bool:
        """Checks if text changed completely or matches a new exact string."""
        if not HAS_PYTESSERACT: return False
        
        region = obs_def.get("target_region", "local")
        roi_after = self._get_local_roi(after, action, size=obs_def.get("roi_size", 150)) if region == "local" else after
        roi_before = self._get_local_roi(before, action, size=obs_def.get("roi_size", 150)) if region == "local" else before
        
        if roi_after is None or roi_before is None: return False
        
        text_before = pytesseract.image_to_string(roi_before).strip()
        text_after = pytesseract.image_to_string(roi_after).strip()
        
        # Did the text meaningfully change?
        return text_before != text_after


    def _check_frame_transition(self, obs_def: Dict[str, Any], before: np.ndarray, after: np.ndarray) -> bool:
        """Detects major scene changes using HSV histogram comparison."""
        threshold = obs_def.get("threshold", 0.5) # Correlation threshold (lower means more different)
        
        hsv_before = cv2.cvtColor(before, cv2.COLOR_BGR2HSV)
        hsv_after = cv2.cvtColor(after, cv2.COLOR_BGR2HSV)
        
        hist_before = cv2.calcHist([hsv_before], [0, 1, 2], None, [50, 50, 50], [0, 180, 0, 256, 0, 256])
        cv2.normalize(hist_before, hist_before, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
        
        hist_after = cv2.calcHist([hsv_after], [0, 1, 2], None, [50, 50, 50], [0, 180, 0, 256, 0, 256])
        cv2.normalize(hist_after, hist_after, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
        
        similarity = cv2.compareHist(hist_before, hist_after, cv2.HISTCMP_CORREL)
        print(f"[AnticipatoryObserver] Frame transition similarity: {similarity:.3f} (threshold: {threshold})")
        
        return similarity < threshold

    def detect_change_regions(self, before: np.ndarray, after: np.ndarray, min_area: int = 200) -> list:
        """Finds bounding boxes of all changed regions between two frames."""
        try:
            diff = cv2.absdiff(before, after)
            gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
            _, thresh = cv2.threshold(gray, 30, 255, cv2.THRESH_BINARY)
            
            kernel = np.ones((5,5),np.uint8)
            thresh = cv2.dilate(thresh, kernel, iterations=1)
            
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            regions = []
            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area >= min_area:
                    x, y, w, h = cv2.boundingRect(cnt)
                    regions.append({'bbox': [x, y, w, h], 'area': float(area)})
                    
            regions.sort(key=lambda r: r['area'], reverse=True)
            return regions
        except Exception as e:
            print(f"[AnticipatoryObserver] Error detecting regions: {e}")
            return []

    def _execute_custom_script(self, obs_def: Dict[str, Any], before: np.ndarray, after: np.ndarray, action: Dict[str, Any]) -> bool:
        code = obs_def.get("script_code")
        if not code:
            # Check if it's a named script from the registry
            script_name = obs_def.get("script_name")
            if script_name and script_name in self._custom_observables:
                code = self._custom_observables[script_name].get("code")
            if not code:
                return False
        safe_globals = {"cv2": cv2, "np": np, "before": before, "after": after, "action": action, "result": False}
        try:
            exec(code, safe_globals)
            return bool(safe_globals.get("result", False))
        except Exception: return False

    # ═══════════════════════  Observable Selection  ═══════════════════════

    def select_observable(self, action: Dict[str, Any], process_context=None) -> Optional[Dict[str, Any]]:
        """
        Auto-selects the appropriate observable type based on action semantics.
        Returns an observable definition dict, or None if VLM should be used.

        Heuristic rules:
          - click on buttons/links → color_shift (local)
          - navigation clicks → structural_change (global, threshold 0.15)
          - type/press actions → None (keyboard actions need VLM)
          - scroll → structural_change (threshold 0.05)
          - click on known template → element_appears (check via template match)
          - custom script available → custom_script
        """
        action_type = action.get("action", "")
        target = (action.get("target", "") or "").lower()

        # Pure keyboard logic (hotkey, wait) are hard to CV verify
        if action_type in ("hotkey", "wait"):
            return None

        # Check for explicit expected_observable from the planner
        explicit = action.get("expected_observable")
        if explicit:
            return explicit

        # Check for a custom script matching this action/target
        for name, meta in self._custom_observables.items():
            if target and meta.get("target_pattern") and meta["target_pattern"] in target:
                return {"type": "custom_script", "script_name": name}

        # Auto-selection based on action type and target
        if action_type in ("type", "press"):
            # Check if typing produces expected text logic:
            typed_text = action.get("text", "")
            if typed_text and len(typed_text) > 2:
                # Expect typed text to appear in a local ROI around the active element
                return {"type": "text_appears", "expected_text": typed_text, "target_region": "local", "roi_size": 200}
            return None

        if action_type in ("click", "double_click"):
            # Reading / text selection clicks → structural change
            text_keywords = ["read", "select text", "highlight"]
            if any(kw in target for kw in text_keywords):
                return {"type": "structural_change", "threshold": 0.05}
                
            # Major Navigation acts → frame transition
            major_nav = ["wait for load", "login", "submit", "navigate", "open page", "search"]
            if any(kw in target for kw in major_nav):
                return {"type": "frame_transition", "threshold": 0.70}

            # Minor navigation-like targets → structural change
            nav_keywords = ["menu", "tab", "link", "url"]
            if any(kw in target for kw in nav_keywords):
                return {"type": "structural_change", "threshold": 0.10}
            # Button/icon clicks → local color shift
            return {"type": "color_shift", "target_region": "local", "roi_size": 60}

        if action_type == "scroll":
            return {"type": "structural_change", "threshold": 0.05}

        if action_type in ("drag", "click_and_drag", "variable_rate_drag", "drag_to_resize"):
            return {"type": "structural_change", "threshold": 0.08}

        return None

    def extract_text_roi(self, image: np.ndarray, region: list = None) -> str:
        """Public API to grab text from an arbitrary bounding box [x,y,w,h]."""
        if not HAS_PYTESSERACT:
            return ""
        if region and len(region) == 4:
            x, y, w, h = region
            roi = image[y:y+h, x:x+w]
            if roi.size > 0:
                return pytesseract.image_to_string(roi).strip()
        else:
            return pytesseract.image_to_string(image).strip()
        return ""

    def register_custom_observable(self, name: str, code: str, metadata: Dict[str, Any] = None):
        """
        Registers a custom observable script that can be selected by the
        auto-selector or referenced by name.
        """
        entry = {
            "code": code,
            "name": name,
            "target_pattern": (metadata or {}).get("target_pattern", ""),
            "description": (metadata or {}).get("description", ""),
            "registered_at": 0,  # Will use time if needed
        }
        self._custom_observables[name] = entry
        print(f"[AnticipatoryObserver] Registered custom observable: '{name}'")

    def multi_region_check(
        self,
        before: np.ndarray,
        after: np.ndarray,
        regions: list,
        threshold: float = 10.0,
    ) -> Dict[str, Any]:
        """
        Monitors multiple screen regions for changes simultaneously.
        Each region is a dict: {"name": str, "bbox": [x, y, w, h]}
        Returns per-region change detection results.
        """
        results = {}
        for region in regions:
            name = region.get("name", "unknown")
            bbox = region.get("bbox", [])
            if len(bbox) != 4:
                results[name] = {"changed": False, "error": "invalid bbox"}
                continue

            x, y, w, h = bbox
            roi_before = before[y:y+h, x:x+w]
            roi_after = after[y:y+h, x:x+w]

            if roi_before.size == 0 or roi_after.size == 0:
                results[name] = {"changed": False, "error": "empty ROI"}
                continue

            mean_diff = float(np.linalg.norm(
                np.mean(roi_before, axis=(0, 1)) - np.mean(roi_after, axis=(0, 1))
            ))
            results[name] = {
                "changed": mean_diff >= threshold,
                "magnitude": round(mean_diff, 2),
            }

        any_changed = any(r.get("changed", False) for r in results.values())
        return {"regions": results, "any_changed": any_changed}


