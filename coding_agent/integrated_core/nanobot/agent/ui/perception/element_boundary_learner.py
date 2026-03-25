"""
ElementBoundaryLearner — Discovers exact pixel boundaries of all UI chrome elements.

Uses a 3-pass hybrid approach:
  Pass 1: Win32 API probing (precise, fast) — standard chrome buttons, menus, scrollbars
  Pass 2: OpenCV contour analysis (medium) — custom-drawn UI elements  
  Pass 3: VLM fallback (slow) — semantic labeling for ambiguous elements

Each learned boundary persists in the UI Atlas for cross-session reuse.
"""

import cv2 # type: ignore
import numpy as np # type: ignore
import ctypes # type: ignore
import ctypes.wintypes # type: ignore
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any

try:
    import win32gui # type: ignore
    import win32con # type: ignore
    import win32api # type: ignore
except ImportError:
    win32gui = None
    win32con = None
    win32api = None


# ──────────────────────────  Data Structures  ──────────────────────────

@dataclass
class ElementBoundary:
    """Exact pixel boundary information for a single UI element."""
    element_id: str             # 'close_button', 'menu_File', 'toolbar_btn_3', etc.
    element_type: str           # BUTTON | ICON | MENU_ITEM | INPUT_BOX | SCROLLBAR | TAB | ...
    rect: Tuple[int, int, int, int]  # (x1, y1, x2, y2) exact pixel coordinates
    center: Tuple[int, int]     # (cx, cy) best click target
    relative_position: str      # TOP_RIGHT, TOP_LEFT, BOTTOM_CENTER, etc.
    size_px: Tuple[int, int]    # (width, height)
    detection_method: str       # WIN32_API | CV_CONTOUR | CV_TEMPLATE | VLM | HYBRID
    confidence: float           # 0.0-1.0
    interaction_type: str       # CLICK | DOUBLE_CLICK | RIGHT_CLICK | DRAG | TYPE | HOVER
    semantic_label: str = ""    # VLM-provided label if available
    template_path: str = ""     # Stored template image path for future matching
    last_verified: float = 0.0
    verification_count: int = 0


# ──────────────────────────  Learner  ──────────────────────────

class ElementBoundaryLearner:
    """Learns exact pixel boundaries of UI elements using Win32 API + CV + VLM."""

    def __init__(self, cv_pipeline=None, vision_analyzer=None):
        self.cv = cv_pipeline
        self.vision = vision_analyzer
        self.learned_boundaries: Dict[str, Dict[str, ElementBoundary]] = {}  # {app_class: {element_id: boundary}}

    # ═══════════════════════  Pass 1: Win32 API Detection  ═══════════════════════

    def detect_window_chrome(self, hwnd: int) -> Dict[str, ElementBoundary]:
        """
        Uses Win32 API to find exact boundaries of standard window chrome:
        title bar, close/min/max buttons, menu bar, scrollbars.
        """
        if win32gui is None:
            return {}

        result = {}

        try:
            # Window outer + client rects
            outer = win32gui.GetWindowRect(hwnd)  # (left, top, right, bottom)
            client_origin = win32gui.ClientToScreen(hwnd, (0, 0))
            client_rect = win32gui.GetClientRect(hwnd)  # (0, 0, width, height)
        except Exception as e:
            print(f"[ElementBoundaryLearner] Win32 rect query failed: {e}")
            return {}

        # === Title bar region ===
        try:
            get_metrics = getattr(win32api, "GetSystemMetrics")
            sm_cycaption = getattr(win32con, "SM_CYCAPTION")
            sm_cxframe = getattr(win32con, "SM_CXFRAME")
            sm_cxpadded = getattr(win32con, "SM_CXPADDEDBORDER")
            sm_cyframe = getattr(win32con, "SM_CYFRAME")
            caption_height = int(get_metrics(sm_cycaption))
            border_x = int(get_metrics(sm_cxframe)) + int(get_metrics(sm_cxpadded))
            border_y = int(get_metrics(sm_cyframe)) + int(get_metrics(sm_cxpadded))
        except Exception:
            caption_height = 30
            border_x = 8
            border_y = 8

        title_bar_rect = (outer[0], outer[1], outer[2], outer[1] + border_y + caption_height)
        result["title_bar"] = ElementBoundary(
            element_id="title_bar",
            element_type="TITLE_BAR",
            rect=title_bar_rect,
            center=((title_bar_rect[0] + title_bar_rect[2]) // 2, (title_bar_rect[1] + title_bar_rect[3]) // 2),
            relative_position="TOP",
            size_px=(title_bar_rect[2] - title_bar_rect[0], title_bar_rect[3] - title_bar_rect[1]),
            detection_method="WIN32_API",
            confidence=0.95,
            interaction_type="DRAG",
        )

        # === Caption buttons (Close / Maximize / Minimize) via DWM ===
        try:
            caption_buttons = self._get_caption_button_bounds(hwnd, title_bar_rect)
            result.update(caption_buttons)
        except Exception as e:
            print(f"[ElementBoundaryLearner] Caption button detection fallback: {e}")
            # Fallback: estimate from title bar right edge
            btn_w = 46  # Standard Windows button width
            btn_h = caption_height
            right_edge = outer[2] - border_x
            top_edge = outer[1] + border_y

            for i, (btn_id, label) in enumerate([("close_button", "Close"), ("maximize_button", "Maximize"), ("minimize_button", "Minimize")]):
                bx2 = right_edge - (i * btn_w)
                bx1 = bx2 - btn_w
                rect = (bx1, top_edge, bx2, top_edge + btn_h)
                result[btn_id] = ElementBoundary(
                    element_id=btn_id,
                    element_type="BUTTON",
                    rect=rect,
                    center=((bx1 + bx2) // 2, (top_edge + top_edge + btn_h) // 2),
                    relative_position="TOP_RIGHT",
                    size_px=(btn_w, btn_h),
                    detection_method="WIN32_API",
                    confidence=0.80,
                    interaction_type="CLICK",
                    semantic_label=label,
                )

        # === Menu bar ===
        try:
            menu = win32gui.GetMenu(hwnd)
            if menu:
                menu_info = self._get_menu_bar_info(hwnd, outer)
                result.update(menu_info)
        except Exception:
            pass

        # === Scrollbars ===
        try:
            scrollbar_info = self._get_scrollbar_info(hwnd, outer, client_origin, client_rect)
            result.update(scrollbar_info)
        except Exception:
            pass

        # === System icon (top-left) ===
        icon_size = 16
        try:
            get_metrics = getattr(win32api, "GetSystemMetrics")
            sm_cxsmicon = getattr(win32con, "SM_CXSMICON")
            icon_size = int(get_metrics(sm_cxsmicon))
        except Exception:
            pass
        icon_rect = (outer[0] + border_x, outer[1] + border_y, outer[0] + border_x + icon_size, outer[1] + border_y + icon_size)
        result["system_icon"] = ElementBoundary(
            element_id="system_icon",
            element_type="ICON",
            rect=icon_rect,
            center=((icon_rect[0] + icon_rect[2]) // 2, (icon_rect[1] + icon_rect[3]) // 2),
            relative_position="TOP_LEFT",
            size_px=(icon_size, icon_size),
            detection_method="WIN32_API",
            confidence=0.85,
            interaction_type="CLICK",
            semantic_label="System menu icon",
        )

        return result

    def _get_caption_button_bounds(self, hwnd: int, title_bar_rect: tuple) -> Dict[str, ElementBoundary]:
        """Attempts to get caption button bounds via DwmGetWindowAttribute."""
        result = {}

        # Try DWM API for precise button area
        try:
            windll = getattr(ctypes, "windll", None)
            if not windll:
                raise Exception("No windll")
            dwmapi = windll.dwmapi

            class RECT(ctypes.Structure):
                _fields_ = [("left", ctypes.c_long), ("top", ctypes.c_long),
                            ("right", ctypes.c_long), ("bottom", ctypes.c_long)]

            # DWMWA_CAPTION_BUTTON_BOUNDS = 5
            rect = RECT()
            hr = dwmapi.DwmGetWindowAttribute(hwnd, 5, ctypes.byref(rect), ctypes.sizeof(rect))

            if hr == 0:
                # The returned rect covers all three buttons
                total_w = rect.right - rect.left
                btn_w = total_w // 3
                btn_h = rect.bottom - rect.top

                buttons = [
                    ("minimize_button", "Minimize", rect.left),
                    ("maximize_button", "Maximize", rect.left + btn_w),
                    ("close_button", "Close", rect.left + 2 * btn_w),
                ]

                for btn_id, label, bx in buttons:
                    btn_rect = (bx, rect.top, bx + btn_w, rect.bottom)
                    result[btn_id] = ElementBoundary(
                        element_id=btn_id,
                        element_type="BUTTON",
                        rect=btn_rect,
                        center=((btn_rect[0] + btn_rect[2]) // 2, (btn_rect[1] + btn_rect[3]) // 2),
                        relative_position="TOP_RIGHT",
                        size_px=(btn_w, btn_h),
                        detection_method="WIN32_API",
                        confidence=0.95,
                        interaction_type="CLICK",
                        semantic_label=label,
                    )
        except Exception:
            pass

        return result

    def _get_menu_bar_info(self, hwnd: int, outer_rect: tuple) -> Dict[str, ElementBoundary]:
        """Gets menu bar and individual menu item info."""
        result = {}

        try:
            class MENUBARINFO(ctypes.Structure):
                _fields_ = [
                    ("cbSize", ctypes.wintypes.DWORD),
                    ("rcBar", ctypes.wintypes.RECT),
                    ("hMenu", ctypes.wintypes.HMENU),
                    ("hwndMenu", ctypes.wintypes.HWND),
                    ("fBarFocused", ctypes.c_int, 1),
                    ("fFocused", ctypes.c_int, 1),
                ]

            mbi = MENUBARINFO()
            setattr(mbi, "cbSize", ctypes.sizeof(MENUBARINFO)) # type: ignore
            # OBJID_MENU = -3 (system constant)
            windll = getattr(ctypes, "windll", None)
            if windll:
                success = windll.user32.GetMenuBarInfo(hwnd, 0xFFFFFFFD, 0, ctypes.byref(mbi))

            if success:
                bar_rect = (mbi.rcBar.left, mbi.rcBar.top, mbi.rcBar.right, mbi.rcBar.bottom)
                result["menu_bar"] = ElementBoundary(
                    element_id="menu_bar",
                    element_type="MENU_BAR",
                    rect=bar_rect,
                    center=((bar_rect[0] + bar_rect[2]) // 2, (bar_rect[1] + bar_rect[3]) // 2),
                    relative_position="TOP",
                    size_px=(bar_rect[2] - bar_rect[0], bar_rect[3] - bar_rect[1]),
                    detection_method="WIN32_API",
                    confidence=0.90,
                    interaction_type="CLICK",
                )
        except Exception:
            pass

        return result

    def _get_scrollbar_info(self, hwnd: int, outer_rect: tuple, client_origin: tuple, client_rect: tuple) -> Dict[str, ElementBoundary]:
        """Gets vertical and horizontal scrollbar positions."""
        result = {}

        try:
            class SCROLLBARINFO(ctypes.Structure):
                _fields_ = [
                    ("cbSize", ctypes.wintypes.DWORD),
                    ("rcScrollBar", ctypes.wintypes.RECT),
                    ("dxyLineButton", ctypes.c_int),
                    ("xyThumbTop", ctypes.c_int),
                    ("xyThumbBottom", ctypes.c_int),
                    ("reserved", ctypes.c_int),
                    ("rgstate", ctypes.wintypes.DWORD * 6), # type: ignore
                ]

            # Vertical scrollbar (OBJID_VSCROLL = -5)
            sbi_v = SCROLLBARINFO()
            setattr(sbi_v, "cbSize", ctypes.sizeof(SCROLLBARINFO)) # type: ignore
            windll = getattr(ctypes, "windll", None)
            if windll and windll.user32.GetScrollBarInfo(hwnd, 0xFFFFFFFB, ctypes.byref(sbi_v)):
                v_rect = (sbi_v.rcScrollBar.left, sbi_v.rcScrollBar.top, sbi_v.rcScrollBar.right, sbi_v.rcScrollBar.bottom)
                if v_rect[2] - v_rect[0] > 0 and v_rect[3] - v_rect[1] > 0:
                    result["scrollbar_v"] = ElementBoundary(
                        element_id="scrollbar_v",
                        element_type="SCROLLBAR",
                        rect=v_rect,
                        center=((v_rect[0] + v_rect[2]) // 2, (v_rect[1] + v_rect[3]) // 2),
                        relative_position="RIGHT",
                        size_px=(v_rect[2] - v_rect[0], v_rect[3] - v_rect[1]),
                        detection_method="WIN32_API",
                        confidence=0.90,
                        interaction_type="DRAG",
                    )

            # Horizontal scrollbar (OBJID_HSCROLL = -6)
            sbi_h = SCROLLBARINFO()
            setattr(sbi_h, "cbSize", ctypes.sizeof(SCROLLBARINFO)) # type: ignore
            if windll and windll.user32.GetScrollBarInfo(hwnd, 0xFFFFFFFA, ctypes.byref(sbi_h)):
                h_rect = (sbi_h.rcScrollBar.left, sbi_h.rcScrollBar.top, sbi_h.rcScrollBar.right, sbi_h.rcScrollBar.bottom)
                if h_rect[2] - h_rect[0] > 0 and h_rect[3] - h_rect[1] > 0:
                    result["scrollbar_h"] = ElementBoundary(
                        element_id="scrollbar_h",
                        element_type="SCROLLBAR",
                        rect=h_rect,
                        center=((h_rect[0] + h_rect[2]) // 2, (h_rect[1] + h_rect[3]) // 2),
                        relative_position="BOTTOM",
                        size_px=(h_rect[2] - h_rect[0], h_rect[3] - h_rect[1]),
                        detection_method="WIN32_API",
                        confidence=0.90,
                        interaction_type="DRAG",
                    )
        except Exception:
            pass

        return result

    # ═══════════════════════  Pass 2: CV-Based Detection  ═══════════════════════

    def detect_all_elements_cv(self, frame: np.ndarray, window_rect: Tuple[int, int, int, int]) -> List[ElementBoundary]:
        """
        For custom-drawn UIs (Electron apps, games, etc.) where Win32 APIs
        don't expose internal elements. Uses CVPipeline for detection.
        """
        if self.cv is None or frame is None:
            return []

        elements = self.cv.detect_ui_elements(frame)
        boundaries = []

        wx, wy = window_rect[0], window_rect[1]
        fh, fw = frame.shape[:2]

        for i, el in enumerate(elements):
            # Convert to global coordinates
            gx1 = wx + el.x
            gy1 = wy + el.y
            gx2 = gx1 + el.width
            gy2 = gy1 + el.height

            # Determine relative position
            rel_pos = self._classify_position(el.x, el.y, el.width, el.height, fw, fh)

            # Infer interaction type from element type
            interaction = self._infer_interaction(el.element_type)

            boundaries.append(ElementBoundary(
                element_id=f"cv_{el.element_type.lower()}_{i}",
                element_type=el.element_type,
                rect=(gx1, gy1, gx2, gy2),
                center=((gx1 + gx2) // 2, (gy1 + gy2) // 2),
                relative_position=rel_pos,
                size_px=(el.width, el.height),
                detection_method="CV_CONTOUR",
                confidence=el.confidence * 0.8,  # Slightly lower than Win32 API
                interaction_type=interaction,
            ))

        return boundaries

    def detect_title_bar_buttons(self, frame: np.ndarray, window_rect: Tuple[int, int, int, int]) -> Dict[str, ElementBoundary]:
        """
        Specialized detector for close/minimize/maximize buttons
        in the top-right corner of a window using CV.
        """
        if frame is None:
            return {}

        fh, fw = frame.shape[:2]
        result = {}

        # Crop top-right corner (where buttons typically live)
        crop_w = min(200, fw // 3)
        crop_h = min(50, fh // 8)
        crop = frame[0:crop_h, fw - crop_w:fw]

        if crop.size == 0:
            return {}

        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if len(crop.shape) == 3 else crop

        # Edge detection to find button boundaries
        edges = cv2.Canny(gray, 50, 150)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Filter for button-shaped contours (rectangular, similar size, in a row)
        button_candidates = []
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            aspect = w / max(h, 1)
            if 15 < w < 60 and 15 < h < 45 and 0.8 < aspect < 3.0:
                button_candidates.append((x, y, w, h))

        # Sort right-to-left (close → maximize → minimize)
        button_candidates.sort(key=lambda b: b[0], reverse=True)

        wx, wy = window_rect[0], window_rect[1]
        names = [("close_button", "Close"), ("maximize_button", "Maximize"), ("minimize_button", "Minimize")]

        for i, btn in enumerate(button_candidates[:3]): # type: ignore
            bx, by, bw, bh = btn
            gx = wx + (fw - crop_w) + bx
            gy = wy + by

            btn_id, label = names[i] if i < len(names) else (f"btn_{i}", f"Button {i}")
            rect = (gx, gy, gx + bw, gy + bh)
            result[btn_id] = ElementBoundary(
                element_id=btn_id,
                element_type="BUTTON",
                rect=rect,
                center=((rect[0] + rect[2]) // 2, (rect[1] + rect[3]) // 2),
                relative_position="TOP_RIGHT",
                size_px=(bw, bh),
                detection_method="CV_CONTOUR",
                confidence=0.70,
                interaction_type="CLICK",
                semantic_label=label,
            )

        return result

    def detect_menu_items(self, frame: np.ndarray, menu_bar_rect: Tuple[int, int, int, int]) -> List[ElementBoundary]:
        """Detects individual menu items within a known menu bar."""
        if self.cv is None or frame is None:
            return []

        x1, y1, x2, y2 = menu_bar_rect
        h_frame, w_frame = frame.shape[:2]

        # Convert to local frame coordinates if needed
        crop = frame[max(0, y1):min(y2, h_frame), max(0, x1):min(x2, w_frame)]
        if crop.size == 0:
            return []

        # Use MSER text detection to find menu labels
        text_regions = self.cv.detect_text_regions(crop, merge_distance=5)

        items = []
        for j, tr in enumerate(text_regions):
            gx = x1 + tr["x"]
            gy = y1 + tr["y"]
            rect = (gx, gy, gx + tr["w"], gy + tr["h"])
            items.append(ElementBoundary(
                element_id=f"menu_item_{j}",
                element_type="MENU_ITEM",
                rect=rect,
                center=((rect[0] + rect[2]) // 2, (rect[1] + rect[3]) // 2),
                relative_position="TOP",
                size_px=(tr["w"], tr["h"]),
                detection_method="CV_CONTOUR",
                confidence=0.65,
                interaction_type="CLICK",
            ))

        return items

    def detect_scrollbars(self, frame: np.ndarray, window_rect: Tuple[int, int, int, int]) -> Dict[str, ElementBoundary]:
        """Detects vertical + horizontal scrollbars via CV."""
        if frame is None:
            return {}

        fh, fw = frame.shape[:2]
        result = {}
        wx, wy = window_rect[0], window_rect[1]

        # Vertical scrollbar: thin strip at right edge
        strip_w = 25
        if fw > strip_w:
            v_strip = frame[:, fw - strip_w:fw]
            gray = cv2.cvtColor(v_strip, cv2.COLOR_BGR2GRAY) if len(v_strip.shape) == 3 else v_strip
            # Look for a consistent vertical strip with a thumb (darker rectangle)
            edges = cv2.Canny(gray, 30, 100)
            if np.sum(edges > 0) > fh * 0.3:  # Enough edge content for a scrollbar
                rect = (wx + fw - strip_w, wy, wx + fw, wy + fh)
                result["scrollbar_v"] = ElementBoundary(
                    element_id="scrollbar_v",
                    element_type="SCROLLBAR",
                    rect=rect,
                    center=((rect[0] + rect[2]) // 2, (rect[1] + rect[3]) // 2),
                    relative_position="RIGHT",
                    size_px=(strip_w, fh),
                    detection_method="CV_CONTOUR",
                    confidence=0.60,
                    interaction_type="DRAG",
                )

        # Horizontal scrollbar: thin strip at bottom edge
        strip_h = 25
        if fh > strip_h:
            h_strip = frame[fh - strip_h:fh, :]
            gray = cv2.cvtColor(h_strip, cv2.COLOR_BGR2GRAY) if len(h_strip.shape) == 3 else h_strip
            edges = cv2.Canny(gray, 30, 100)
            if np.sum(edges > 0) > fw * 0.3:
                rect = (wx, wy + fh - strip_h, wx + fw, wy + fh)
                result["scrollbar_h"] = ElementBoundary(
                    element_id="scrollbar_h",
                    element_type="SCROLLBAR",
                    rect=rect,
                    center=((rect[0] + rect[2]) // 2, (rect[1] + rect[3]) // 2),
                    relative_position="BOTTOM",
                    size_px=(fw, strip_h),
                    detection_method="CV_CONTOUR",
                    confidence=0.60,
                    interaction_type="DRAG",
                )

        return result

    # ═══════════════════════  Hybrid: Full Learning  ═══════════════════════

    def learn_all_boundaries(self, hwnd: int, screenshot_path: Optional[str] = None) -> Dict[str, ElementBoundary]:
        """
        Full boundary learning for a window. Three-pass approach:
          Pass 1 (Win32 API): Standard chrome boundaries — fast and precise
          Pass 2 (CV): Custom/non-standard elements — medium speed
          Pass 3 (VLM): Label ambiguous elements semantically — slow but accurate

        Results are merged with Win32 API having highest priority.
        """
        import time
        t0 = time.perf_counter()
        result: Dict[str, ElementBoundary] = {}

        # Pass 1: Win32 API
        api_elements = self.detect_window_chrome(hwnd)
        result.update(api_elements)
        print(f"[ElementBoundaryLearner] Pass 1 (Win32): Found {len(api_elements)} chrome elements")

        # Get window rect for CV passes
        if win32gui:
            try:
                outer = win32gui.GetWindowRect(hwnd)
                window_rect = outer
            except Exception:
                window_rect = (0, 0, 1920, 1080)
        else:
            window_rect = (0, 0, 1920, 1080)

        # Pass 2: CV-based detection
        if self.cv and screenshot_path:
            try:
                frame = cv2.imread(screenshot_path)
                if frame is not None:
                    # Title bar buttons (if not found by API)
                    if "close_button" not in result:
                        cv_buttons = self.detect_title_bar_buttons(frame, window_rect)
                        result.update(cv_buttons)

                    # Scrollbars (if not found by API)
                    if "scrollbar_v" not in result:
                        cv_scrollbars = self.detect_scrollbars(frame, window_rect)
                        result.update(cv_scrollbars)

                    # General element detection
                    cv_elements = self.detect_all_elements_cv(frame, window_rect)
                    for el in cv_elements:
                        if el.element_id not in result:
                            result[el.element_id] = el

                    print(f"[ElementBoundaryLearner] Pass 2 (CV): Added {len(cv_elements)} elements")
            except Exception as e:
                print(f"[ElementBoundaryLearner] CV pass failed: {e}")

        # Store in learned boundaries
        app_class = ""
        if win32gui:
            try:
                app_class = win32gui.GetClassName(hwnd)
            except Exception:
                pass
        self.learned_boundaries[app_class] = result

        elapsed = (time.perf_counter() - t0) * 1000
        print(f"[ElementBoundaryLearner] Total: {len(result)} elements learned in {elapsed:.1f}ms")

        return result

    def verify_boundaries(self, hwnd: int, stored_boundaries: Dict[str, ElementBoundary]) -> Dict[str, Any]:
        """
        Re-checks stored boundaries after window resize/move.
        Returns: {still_valid: [], shifted: [(old, new)], disappeared: []}
        """
        current = self.detect_window_chrome(hwnd)

        still_valid = []
        shifted = []
        disappeared = []

        for elem_id, old_boundary in stored_boundaries.items():
            if elem_id in current:
                new_boundary = current[elem_id]
                if old_boundary.rect == new_boundary.rect:
                    still_valid.append(elem_id)
                else:
                    shifted.append({
                        "element_id": elem_id,
                        "old_rect": old_boundary.rect,
                        "new_rect": new_boundary.rect,
                        "old_center": old_boundary.center,
                        "new_center": new_boundary.center,
                    })
            else:
                disappeared.append(elem_id)

        return {
            "still_valid": still_valid,
            "shifted": shifted,
            "disappeared": disappeared,
        }

    def get_clickable_center(self, app_class: str, element_id: str) -> Optional[Tuple[int, int]]:
        """Returns the safest click point for a learned element."""
        boundaries = self.learned_boundaries.get(app_class, {})
        boundary = boundaries.get(element_id)
        if boundary:
            return boundary.center
        return None

    # ═══════════════════════  Helpers  ═══════════════════════

    def _classify_position(self, x: int, y: int, w: int, h: int, frame_w: int, frame_h: int) -> str:
        """Classifies an element's relative position within the window."""
        cx = x + w // 2
        cy = y + h // 2
        x_third = frame_w / 3
        y_third = frame_h / 3

        if cy < y_third:
            v = "TOP"
        elif cy > 2 * y_third:
            v = "BOTTOM"
        else:
            v = "CENTER"

        if cx < x_third:
            h_pos = "LEFT"
        elif cx > 2 * x_third:
            h_pos = "RIGHT"
        else:
            h_pos = "CENTER"

        if v == "CENTER" and h_pos == "CENTER":
            return "CENTER"
        if h_pos == "CENTER":
            return v
        if v == "CENTER":
            return h_pos
        return f"{v}_{h_pos}"

    def _infer_interaction(self, element_type: str) -> str:
        """Infers the most likely interaction type for an element type."""
        mapping = {
            "BUTTON": "CLICK",
            "ICON": "CLICK",
            "TEXT_INPUT": "TYPE",
            "MENU_ITEM": "CLICK",
            "TAB": "CLICK",
            "SCROLLBAR": "DRAG",
            "CHECKBOX": "CLICK",
            "DROPDOWN": "CLICK",
            "SLIDER": "DRAG",
            "PANEL": "HOVER",
            "STRUCTURAL": "HOVER",
        }
        return mapping.get(element_type, "CLICK")
