"""
AutomationArtifacts — Registry of complex gesture recipes as reusable tools.

These artifacts encapsulate multi-step or parameterized UI manipulations
that the AgenticPlanner can invoke by name. Each artifact handles its own
coordinate resolution, timing, and error recovery.

Artifacts:
  click_and_drag        — Click-hold-drag with configurable speed and curve
  drag_to_resize        — Detect edge of window/panel, drag to resize
  nested_menu_navigate  — Navigate multi-level menus by label path
  scroll_until_visible  — Scroll until a target element is found
  variable_rate_drag    — Drag with acceleration/deceleration
"""

import time
import math
from typing import Dict, Any, Optional, List, Tuple, Callable


class AutomationArtifactRegistry:
    """Registry and dispatcher for complex UI automation artifacts."""

    def __init__(self, mouse_controller=None, keyboard_controller=None,
                 screen_observer=None, vision_analyzer=None):
        self.mouse = mouse_controller
        self.keyboard = keyboard_controller
        self.screen = screen_observer
        self.vision = vision_analyzer

        # Registry of artifact names → handler methods
        self._artifacts: Dict[str, Callable] = {
            "click_and_drag": self._click_and_drag,
            "drag_to_resize": self._drag_to_resize,
            "nested_menu_navigate": self._nested_menu_navigate,
            "scroll_until_visible": self._scroll_until_visible,
            "variable_rate_drag": self._variable_rate_drag,
        }

    def get_available_artifacts(self) -> List[Dict[str, str]]:
        """Returns metadata for all registered artifacts (for system prompt injection)."""
        return [
            {
                "name": "click_and_drag",
                "schema": '{"action": "click_and_drag", "start": [x1, y1], "end": [x2, y2], "duration_ms": 500, "curve": "linear|arc|bezier"}',
                "description": "Click at start, hold, drag to end with configurable speed and trajectory curve.",
            },
            {
                "name": "drag_to_resize",
                "schema": '{"action": "drag_to_resize", "edge": "left|right|top|bottom|top-left|top-right|bottom-left|bottom-right", "target_window": "<description>", "delta_px": 200}',
                "description": "Detect the edge/corner of a window or panel and drag to resize by delta_px pixels.",
            },
            {
                "name": "nested_menu_navigate",
                "schema": '{"action": "nested_menu_navigate", "menu_path": ["File", "Export", "PDF"], "root_coords": [x, y]}',
                "description": "Navigate a multi-level dropdown/context menu by clicking each label in sequence.",
            },
            {
                "name": "scroll_until_visible",
                "schema": '{"action": "scroll_until_visible", "target": "<description of element>", "direction": "down|up", "max_scrolls": 10}',
                "description": "Scroll in a direction until the target element is visually detected by VLM.",
            },
            {
                "name": "variable_rate_drag",
                "schema": '{"action": "variable_rate_drag", "start": [x1, y1], "end": [x2, y2], "profile": "ease_in|ease_out|ease_in_out|constant", "total_ms": 800}',
                "description": "Drag with variable speed profile. ease_in starts slow, ease_out ends slow.",
            },
        ]

    def execute(self, action: Dict[str, Any]) -> Dict[str, Any]:
        """Dispatches an artifact action to its handler. Returns result dict."""
        artifact_name = action.get("action", "")
        handler = self._artifacts.get(artifact_name)

        if handler is None:
            return {"success": False, "error": f"Unknown artifact: {artifact_name}"}

        try:
            return handler(action)
        except Exception as e:
            print(f"[AutomationArtifact:{artifact_name}] Error: {e}")
            return {"success": False, "error": str(e)}

    def is_artifact_action(self, action_name: str) -> bool:
        """Returns True if the action name is a registered artifact."""
        return action_name in self._artifacts

    # ═══════════════════════  Artifact Implementations  ═══════════════════════

    def _click_and_drag(self, action: Dict) -> Dict[str, Any]:
        """Click at start position, hold, drag to end position."""
        if not self.mouse:
            return {"success": False, "error": "Mouse controller not available"}

        start = action.get("start", [0, 0])
        end = action.get("end", [0, 0])
        duration_ms = action.get("duration_ms", 500)
        curve = action.get("curve", "linear")

        x1, y1 = int(start[0]), int(start[1])
        x2, y2 = int(end[0]), int(end[1])
        steps = max(10, duration_ms // 16)  # ~60fps

        print(f"[AutomationArtifact:click_and_drag] ({x1},{y1}) → ({x2},{y2}) in {duration_ms}ms, curve={curve}")

        # Move to start and press
        self.mouse.move_to(x1, y1)
        time.sleep(0.05)
        self.mouse.press(x1, y1)

        # Interpolate path
        for i in range(1, steps + 1):
            t = i / steps

            if curve == "arc":
                # Arc: add a perpendicular offset peaking at midpoint
                arc_height = abs(x2 - x1 + y2 - y1) * 0.15
                perp_offset = arc_height * math.sin(t * math.pi)
                dx = x2 - x1
                dy = y2 - y1
                length = max(1, math.sqrt(dx * dx + dy * dy))
                nx, ny = -dy / length, dx / length  # perpendicular unit vector
                ix = x1 + dx * t + nx * perp_offset
                iy = y1 + dy * t + ny * perp_offset
            elif curve == "bezier":
                # Quadratic bezier with control point above midpoint
                cx = (x1 + x2) / 2
                cy = min(y1, y2) - abs(y2 - y1) * 0.3
                ix = (1 - t) ** 2 * x1 + 2 * (1 - t) * t * cx + t ** 2 * x2
                iy = (1 - t) ** 2 * y1 + 2 * (1 - t) * t * cy + t ** 2 * y2
            else:  # linear
                ix = x1 + (x2 - x1) * t
                iy = y1 + (y2 - y1) * t

            self.mouse.move_to(int(ix), int(iy))
            time.sleep(duration_ms / steps / 1000.0)

        # Release
        self.mouse.release(x2, y2)
        print(f"[AutomationArtifact:click_and_drag] Complete.")
        return {"success": True, "start": [x1, y1], "end": [x2, y2]}

    def _drag_to_resize(self, action: Dict) -> Dict[str, Any]:
        """Detect an edge of a window and drag to resize it."""
        if not self.mouse:
            return {"success": False, "error": "Mouse controller not available"}

        edge = action.get("edge", "right")
        delta_px = action.get("delta_px", 100)
        target_rect = action.get("window_rect")  # [x, y, w, h] if provided

        if not target_rect:
            # Try to get the foreground window rect
            try:
                import win32gui
                hwnd = win32gui.GetForegroundWindow()
                rect = win32gui.GetWindowRect(hwnd)
                target_rect = [rect[0], rect[1], rect[2] - rect[0], rect[3] - rect[1]]
            except Exception:
                return {"success": False, "error": "Cannot determine window rect"}

        x, y, w, h = target_rect

        # Calculate edge midpoints
        edge_coords = {
            "left": (x + 2, y + h // 2),
            "right": (x + w - 2, y + h // 2),
            "top": (x + w // 2, y + 2),
            "bottom": (x + w // 2, y + h - 2),
            "top-left": (x + 2, y + 2),
            "top-right": (x + w - 2, y + 2),
            "bottom-left": (x + 2, y + h - 2),
            "bottom-right": (x + w - 2, y + h - 2),
        }

        start = edge_coords.get(edge)
        if not start:
            return {"success": False, "error": f"Unknown edge: {edge}"}

        # Calculate drag direction
        dx, dy = 0, 0
        if "right" in edge:
            dx = delta_px
        elif "left" in edge:
            dx = -delta_px
        if "bottom" in edge:
            dy = delta_px
        elif "top" in edge:
            dy = -delta_px

        end = (start[0] + dx, start[1] + dy)

        print(f"[AutomationArtifact:drag_to_resize] Edge '{edge}' ({start[0]},{start[1]}) → ({end[0]},{end[1]})")

        # Execute the drag using click_and_drag logic
        return self._click_and_drag({
            "start": list(start),
            "end": list(end),
            "duration_ms": 400,
            "curve": "linear",
        })

    def _nested_menu_navigate(self, action: Dict) -> Dict[str, Any]:
        """Navigate a multi-level dropdown menu by clicking each label in sequence."""
        menu_path = action.get("menu_path", [])
        root_coords = action.get("root_coords")

        if not menu_path:
            return {"success": False, "error": "Empty menu_path"}
        if not self.mouse:
            return {"success": False, "error": "Mouse controller not available"}

        print(f"[AutomationArtifact:nested_menu_navigate] Path: {' → '.join(menu_path)}")

        # Click root menu item
        if root_coords:
            self.mouse.move_and_click(int(root_coords[0]), int(root_coords[1]), human_like=True)
            time.sleep(0.5)  # Wait for menu to appear

        # For each subsequent level, find and click the label
        for i, label in enumerate(menu_path):
            if i == 0 and root_coords:
                continue  # Already clicked root

            time.sleep(0.3)  # Wait for sub-menu animation

            # Use VLM to find the menu item
            if self.vision and self.screen:
                try:
                    screenshot_path = self.screen.capture()
                    if screenshot_path:
                        bbox = self.vision.find_element_bbox(screenshot_path, label)
                        if bbox:
                            # Click the center of the found element
                            cx = (bbox[0] + bbox[2]) // 2
                            cy = (bbox[1] + bbox[3]) // 2
                            self.mouse.move_and_click(cx, cy, human_like=True)
                            print(f"[AutomationArtifact:nested_menu] Clicked '{label}' at ({cx},{cy})")
                            continue
                except Exception as e:
                    print(f"[AutomationArtifact:nested_menu] VLM lookup failed for '{label}': {e}")

            # Fallback: type the first letter to jump to the item, then press Enter
            if self.keyboard:
                self.keyboard.type_key(label[0])
                time.sleep(0.2)
                self.keyboard.type_key("enter")

        print(f"[AutomationArtifact:nested_menu_navigate] Complete.")
        return {"success": True, "path_completed": menu_path}

    def _scroll_until_visible(self, action: Dict) -> Dict[str, Any]:
        """Scroll until a target element becomes visible on screen."""
        target_desc = action.get("target", "")
        direction = action.get("direction", "down")
        max_scrolls = action.get("max_scrolls", 10)

        if not target_desc:
            return {"success": False, "error": "No target description provided"}
        if not self.mouse or not self.vision or not self.screen:
            return {"success": False, "error": "Required controllers not available"}

        print(f"[AutomationArtifact:scroll_until_visible] Looking for '{target_desc}', direction={direction}")

        scroll_amount = 300 if direction == "down" else -300

        for i in range(max_scrolls):
            # Check if element is visible
            try:
                screenshot_path = self.screen.capture()
                if screenshot_path:
                    bbox = self.vision.find_element_bbox(screenshot_path, target_desc)
                    if bbox:
                        cx = (bbox[0] + bbox[2]) // 2
                        cy = (bbox[1] + bbox[3]) // 2
                        print(f"[AutomationArtifact:scroll_until_visible] Found '{target_desc}' at ({cx},{cy}) after {i} scrolls")
                        return {
                            "success": True,
                            "scrolls_needed": i,
                            "element_coords": [cx, cy],
                            "element_bbox": list(bbox),
                        }
            except Exception as e:
                print(f"[AutomationArtifact:scroll_until_visible] VLM check error: {e}")

            # Scroll
            self.mouse.scroll(scroll_amount)
            time.sleep(0.8)  # Wait for scroll animation and page load

        print(f"[AutomationArtifact:scroll_until_visible] Element not found after {max_scrolls} scrolls")
        return {"success": False, "error": f"Element '{target_desc}' not found after {max_scrolls} scrolls"}

    def _variable_rate_drag(self, action: Dict) -> Dict[str, Any]:
        """Drag with variable speed profile (ease in/out)."""
        if not self.mouse:
            return {"success": False, "error": "Mouse controller not available"}

        start = action.get("start", [0, 0])
        end = action.get("end", [0, 0])
        profile = action.get("profile", "ease_in_out")
        total_ms = action.get("total_ms", 800)

        x1, y1 = int(start[0]), int(start[1])
        x2, y2 = int(end[0]), int(end[1])
        num_steps = max(20, total_ms // 16)

        print(f"[AutomationArtifact:variable_rate_drag] ({x1},{y1}) → ({x2},{y2}), profile={profile}")

        self.mouse.move_to(x1, y1)
        time.sleep(0.05)
        self.mouse.press(x1, y1)

        for i in range(1, num_steps + 1):
            raw_t = i / num_steps

            # Apply easing function
            if profile == "ease_in":
                t = raw_t * raw_t  # Quadratic ease in (starts slow)
            elif profile == "ease_out":
                t = 1 - (1 - raw_t) ** 2  # Quadratic ease out (ends slow)
            elif profile == "ease_in_out":
                if raw_t < 0.5:
                    t = 2 * raw_t * raw_t
                else:
                    t = 1 - (-2 * raw_t + 2) ** 2 / 2
            else:  # constant
                t = raw_t

            ix = x1 + (x2 - x1) * t
            iy = y1 + (y2 - y1) * t
            self.mouse.move_to(int(ix), int(iy))

            # Variable sleep based on how "fast" this segment should feel
            # Invert the derivative for sleep time
            time.sleep(total_ms / num_steps / 1000.0)

        self.mouse.release(x2, y2)
        print(f"[AutomationArtifact:variable_rate_drag] Complete.")
        return {"success": True, "profile": profile}
