"""
CursorMonitor — Win32-based cursor type detection and monitoring engine.

Provides real-time cursor state awareness for the nanobot UI agent:
  - Detects current cursor type (arrow, hand, I-beam, resize, crosshair, etc.)
  - Monitors cursor type transitions for hover/click validation
  - Enables recalibration: "did the mouse land on the right element type?"

Usage:
    monitor = CursorMonitor()
    
    # Instant cursor type query
    cursor_type = monitor.get_cursor_type()
    # -> CursorType.HAND (hovering over a link)
    
    # Validate hover target
    is_correct = monitor.validate_hover("link")  # expects HAND cursor
    
    # Wait for cursor to change (useful after mouse.move_to)
    new_type = monitor.wait_for_cursor_change(timeout=2.0)
    
    # Start background monitoring with callback
    monitor.start_monitoring(callback=on_cursor_change)
"""

import ctypes
import ctypes.wintypes
import time
import threading
from enum import Enum, auto
from dataclasses import dataclass
from typing import Optional, Callable, Dict, Tuple, List


# ──────────────────────────  Cursor Type Enum  ──────────────────────────

class CursorType(Enum):
    """Semantic cursor type classification."""
    UNKNOWN = auto()
    ARROW = auto()           # Default pointer — general navigation
    HAND = auto()            # Link/clickable element cursor
    IBEAM = auto()           # Text input cursor (I-beam)
    CROSSHAIR = auto()       # Precision select / drawing tool
    WAIT = auto()            # Busy/hourglass — system processing
    BUSY_ARROW = auto()      # Arrow with hourglass — working in background
    RESIZE_NS = auto()       # North-South resize (↕)
    RESIZE_EW = auto()       # East-West resize (↔)
    RESIZE_NESW = auto()     # NE-SW diagonal resize (⤢)
    RESIZE_NWSE = auto()     # NW-SE diagonal resize (⤡)
    MOVE = auto()            # Four-directional move/drag
    NOT_ALLOWED = auto()     # Prohibited / no-drop
    HELP = auto()            # Help cursor (arrow with ?)
    UP_ARROW = auto()        # Up-arrow select cursor
    PEN = auto()             # Pen/handwriting cursor
    CUSTOM = auto()          # Application-defined custom cursor


# ──────────────────────────  Win32 Cursor Constants  ──────────────────────────

# Standard cursor resource IDs (from WinUser.h)
# These are the resource IDs used by LoadCursor(NULL, IDC_xxx)
_IDC_ARROW = 32512
_IDC_IBEAM = 32513
_IDC_WAIT = 32514
_IDC_CROSS = 32515
_IDC_UPARROW = 32516
_IDC_SIZENWSE = 32642
_IDC_SIZENESW = 32643
_IDC_SIZEWE = 32644
_IDC_SIZENS = 32645
_IDC_SIZEALL = 32646
_IDC_NO = 32648
_IDC_HAND = 32649
_IDC_APPSTARTING = 32650
_IDC_HELP = 32651
_IDC_PEN = 32631


@dataclass
class CursorState:
    """Complete cursor state at a point in time."""
    cursor_type: CursorType
    position: Tuple[int, int]       # (x, y) in screen coordinates
    handle: int                      # Raw Win32 cursor handle
    timestamp: float                 # time.time()
    is_visible: bool = True


# ──────────────────────────  Win32 Structures  ──────────────────────────

class CURSORINFO(ctypes.Structure):
    _fields_ = [
        ("cbSize", ctypes.c_uint),
        ("flags", ctypes.c_uint),
        ("hCursor", ctypes.c_void_p),
        ("ptScreenPos", ctypes.wintypes.POINT),
    ]


class ICONINFO(ctypes.Structure):
    _fields_ = [
        ("fIcon", ctypes.wintypes.BOOL),
        ("xHotspot", ctypes.wintypes.DWORD),
        ("yHotspot", ctypes.wintypes.DWORD),
        ("hbmMask", ctypes.wintypes.HBITMAP),
        ("hbmColor", ctypes.wintypes.HBITMAP),
    ]


# ──────────────────────────  CursorMonitor  ──────────────────────────

class CursorMonitor:
    """
    Win32-based cursor type detection and monitoring engine.
    
    Works by comparing the current cursor handle against known system cursor
    handles loaded at initialization time. This approach is robust against
    theme changes and works across all Windows versions.
    """
    
    def __init__(self):
        # Load all standard system cursor handles at init
        self._system_cursors: Dict[int, CursorType] = {}
        self._load_system_cursors()
        
        # Monitoring state
        self._monitoring = False
        self._monitor_thread: Optional[threading.Thread] = None
        self._on_change_callback: Optional[Callable[[CursorState, CursorState], None]] = None
        self._last_state: Optional[CursorState] = None
        
        # History buffer for pattern detection
        self._history: List[CursorState] = []
        self._max_history = 50
        
        # Expected cursor types per UI element category
        self._element_cursor_map: Dict[str, List[CursorType]] = {
            # Clickable elements
            "link": [CursorType.HAND],
            "button": [CursorType.HAND, CursorType.ARROW],
            "hyperlink": [CursorType.HAND],
            "clickable": [CursorType.HAND, CursorType.ARROW],
            "menu": [CursorType.ARROW, CursorType.HAND],
            "menu_item": [CursorType.ARROW, CursorType.HAND],
            "tab": [CursorType.HAND, CursorType.ARROW],
            "icon": [CursorType.HAND, CursorType.ARROW],
            "checkbox": [CursorType.HAND, CursorType.ARROW],
            "radio": [CursorType.HAND, CursorType.ARROW],
            "dropdown": [CursorType.HAND, CursorType.ARROW],
            "taskbar_icon": [CursorType.HAND, CursorType.ARROW],
            "taskbar_item": [CursorType.HAND, CursorType.ARROW],
            
            # Text elements
            "text_field": [CursorType.IBEAM],
            "text_input": [CursorType.IBEAM],
            "text_area": [CursorType.IBEAM],
            "input": [CursorType.IBEAM],
            "search": [CursorType.IBEAM],
            "address_bar": [CursorType.IBEAM],
            "url_bar": [CursorType.IBEAM],
            "search_bar": [CursorType.IBEAM],
            "text_box": [CursorType.IBEAM],
            "editable": [CursorType.IBEAM],
            
            # Resize elements
            "resize_handle": [CursorType.RESIZE_NS, CursorType.RESIZE_EW, 
                              CursorType.RESIZE_NESW, CursorType.RESIZE_NWSE],
            "window_edge": [CursorType.RESIZE_NS, CursorType.RESIZE_EW,
                            CursorType.RESIZE_NESW, CursorType.RESIZE_NWSE],
            "splitter": [CursorType.RESIZE_EW, CursorType.RESIZE_NS],
            "scrollbar": [CursorType.ARROW],
            
            # Drag elements
            "drag_handle": [CursorType.MOVE],
            "title_bar": [CursorType.ARROW, CursorType.MOVE],
            "draggable": [CursorType.MOVE, CursorType.ARROW],
            "slider": [CursorType.ARROW, CursorType.HAND],
            
            # Drawing/selection
            "canvas": [CursorType.CROSSHAIR, CursorType.ARROW],
            "drawing_area": [CursorType.CROSSHAIR, CursorType.PEN],
            "selection_area": [CursorType.CROSSHAIR],
        }
        
        print("[CursorMonitor] Initialized with "
              f"{len(self._system_cursors)} system cursor mappings")
    
    # ═══════════════════════  System Cursor Loading  ═══════════════════════
    
    def _load_system_cursors(self):
        """Loads all standard Windows cursor handles for comparison."""
        cursor_map = {
            _IDC_ARROW: CursorType.ARROW,
            _IDC_IBEAM: CursorType.IBEAM,
            _IDC_WAIT: CursorType.WAIT,
            _IDC_CROSS: CursorType.CROSSHAIR,
            _IDC_UPARROW: CursorType.UP_ARROW,
            _IDC_SIZENWSE: CursorType.RESIZE_NWSE,
            _IDC_SIZENESW: CursorType.RESIZE_NESW,
            _IDC_SIZEWE: CursorType.RESIZE_EW,
            _IDC_SIZENS: CursorType.RESIZE_NS,
            _IDC_SIZEALL: CursorType.MOVE,
            _IDC_NO: CursorType.NOT_ALLOWED,
            _IDC_HAND: CursorType.HAND,
            _IDC_APPSTARTING: CursorType.BUSY_ARROW,
            _IDC_HELP: CursorType.HELP,
        }
        
        load_cursor = ctypes.windll.user32.LoadCursorW  # type: ignore
        load_cursor.argtypes = [ctypes.wintypes.HINSTANCE, ctypes.c_void_p]
        load_cursor.restype = ctypes.wintypes.HANDLE # HCURSOR fallback
        
        for idc_id, cursor_type in cursor_map.items():
            try:
                handle = load_cursor(None, ctypes.c_void_p(idc_id))
                if handle:
                    self._system_cursors[handle] = cursor_type
            except Exception:
                pass
    
    # ═══════════════════════  Core Detection  ═══════════════════════
    
    def get_cursor_state(self) -> CursorState:
        """Returns the complete current cursor state."""
        ci = CURSORINFO()
        ci.cbSize = ctypes.sizeof(CURSORINFO) # type: ignore
        
        cursor_type = CursorType.UNKNOWN
        position = (0, 0)
        handle = 0
        is_visible = True
        
        try:
            if ctypes.windll.user32.GetCursorInfo(ctypes.byref(ci)):  # type: ignore
                handle = ci.hCursor or 0  # type: ignore
                position = (ci.ptScreenPos.x, ci.ptScreenPos.y)
                is_visible = bool(ci.flags & 0x00000001)  # CURSOR_SHOWING
                
                # Match against known system cursors
                if handle in self._system_cursors:
                    cursor_type = self._system_cursors[handle]
                elif handle:
                    # Unknown cursor — likely application-specific
                    cursor_type = self._classify_custom_cursor(handle)
        except Exception as e:
            print(f"[CursorMonitor] GetCursorInfo failed: {e}")
        
        state = CursorState(
            cursor_type=cursor_type,
            position=position,
            handle=handle,
            timestamp=time.time(),
            is_visible=is_visible,
        )
        
        return state
    
    def get_cursor_type(self) -> CursorType:
        """Returns just the current cursor type (convenience method)."""
        return self.get_cursor_state().cursor_type
    
    def get_cursor_position(self) -> Tuple[int, int]:
        """Returns the current cursor position in screen coordinates."""
        return self.get_cursor_state().position
    
    def _classify_custom_cursor(self, handle: int) -> CursorType:
        """
        Attempts to classify a non-standard cursor by analyzing its ICONINFO.
        Uses hotspot position and bitmap dimensions as heuristics.
        """
        try:
            icon_info = ICONINFO()
            if ctypes.windll.user32.GetIconInfo(handle, ctypes.byref(icon_info)):  # type: ignore
                hotspot_x = icon_info.xHotspot
                hotspot_y = icon_info.yHotspot
                
                # Clean up bitmaps to prevent resource leak
                if icon_info.hbmMask:
                    ctypes.windll.gdi32.DeleteObject(icon_info.hbmMask)  # type: ignore
                if icon_info.hbmColor:
                    ctypes.windll.gdi32.DeleteObject(icon_info.hbmColor)  # type: ignore
                
                # Heuristic classification by hotspot position
                # I-beam cursors typically have hotspot roughly centered horizontally
                # Hand cursors have hotspot near the index finger tip
                # Most arrows have hotspot at top-left
                
                if hotspot_x == 0 and hotspot_y == 0:
                    return CursorType.ARROW  # Top-left hotspot → likely arrow variant
                elif hotspot_x > 5 and hotspot_y > 15:
                    return CursorType.HAND  # Deep hotspot → likely hand variant
                elif hotspot_x > 3 and hotspot_y < 5:
                    return CursorType.IBEAM  # Centered-X, top-Y → likely text variant
                
        except Exception:
            pass
        
        return CursorType.CUSTOM
    
    # ═══════════════════════  Hover Validation  ═══════════════════════
    
    def validate_hover(self, element_type: str) -> bool:
        """
        Validates that the current cursor type matches the expected cursor 
        for the given element type.
        
        Args:
            element_type: Semantic element type (e.g., "link", "text_field", "resize_handle")
            
        Returns:
            True if cursor type matches expectations for this element type
        """
        current = self.get_cursor_type()
        element_key = element_type.lower().replace(" ", "_").replace("-", "_")
        
        expected_types = self._element_cursor_map.get(element_key)
        if expected_types is None:
            # Unknown element type — can't validate, assume OK
            return True
        
        is_valid = current in expected_types
        if not is_valid:
            print(f"[CursorMonitor] Hover validation FAILED: cursor is {current.name} "
                  f"but expected {[t.name for t in expected_types]} for '{element_type}'")
        else:
            print(f"[CursorMonitor] Hover validation OK: {current.name} for '{element_type}'")
        
        return is_valid
    
    def infer_element_type(self) -> str:
        """
        Infers the type of UI element currently under the cursor based on cursor type.
        
        Returns:
            Inferred element category string
        """
        cursor = self.get_cursor_type()
        
        inference_map = {
            CursorType.ARROW: "general",
            CursorType.HAND: "clickable",
            CursorType.IBEAM: "text_input",
            CursorType.CROSSHAIR: "selection_area",
            CursorType.WAIT: "loading",
            CursorType.BUSY_ARROW: "background_loading",
            CursorType.RESIZE_NS: "vertical_resize",
            CursorType.RESIZE_EW: "horizontal_resize",
            CursorType.RESIZE_NESW: "diagonal_resize",
            CursorType.RESIZE_NWSE: "diagonal_resize",
            CursorType.MOVE: "draggable",
            CursorType.NOT_ALLOWED: "disabled",
            CursorType.HELP: "help_area",
            CursorType.PEN: "drawing_area",
        }
        
        return inference_map.get(cursor, "unknown")
    
    # ═══════════════════════  Change Detection  ═══════════════════════
    
    def wait_for_cursor_change(self, timeout: float = 2.0, 
                                poll_interval: float = 0.05) -> Optional[CursorState]:
        """
        Blocks until the cursor type changes from its current type or timeout expires.
        
        Args:
            timeout: Maximum wait time in seconds
            poll_interval: How often to check (seconds)
            
        Returns:
            New CursorState if cursor type changed, None if timed out
        """
        initial = self.get_cursor_state()
        deadline = time.time() + timeout
        
        while time.time() < deadline:
            current = self.get_cursor_state()
            if current.cursor_type != initial.cursor_type:
                return current
            time.sleep(poll_interval)
        
        return None
    
    def wait_for_specific_cursor(self, expected: CursorType, timeout: float = 3.0,
                                  poll_interval: float = 0.05) -> bool:
        """
        Blocks until the cursor becomes the expected type or timeout expires.
        
        Args:
            expected: The CursorType to wait for
            timeout: Maximum wait time in seconds
            
        Returns:
            True if the expected cursor type was observed
        """
        deadline = time.time() + timeout
        
        while time.time() < deadline:
            if self.get_cursor_type() == expected:
                return True
            time.sleep(poll_interval)
        
        return False
    
    # ═══════════════════════  Background Monitoring  ═══════════════════════
    
    def start_monitoring(self, callback: Callable[[CursorState, CursorState], None],
                         poll_interval: float = 0.1):
        """
        Starts background monitoring. Calls callback(old_state, new_state) 
        whenever cursor type changes.
        """
        if self._monitoring:
            self.stop_monitoring()
        
        self._on_change_callback = callback
        self._monitoring = True
        self._last_state = self.get_cursor_state()
        
        thread = threading.Thread(
            target=self._monitor_loop,
            args=(poll_interval,),
            daemon=True,
            name="CursorMonitor"
        )
        self._monitor_thread = thread
        thread.start()
        print("[CursorMonitor] Background monitoring started")
    
    def stop_monitoring(self):
        """Stops background monitoring."""
        self._monitoring = False
        thread = self._monitor_thread
        if thread and thread.is_alive():
            thread.join(timeout=2.0)
        self._monitor_thread = None
        print("[CursorMonitor] Background monitoring stopped")
    
    def _monitor_loop(self, poll_interval: float):
        """Background thread: polls cursor state and fires callbacks on change."""
        while self._monitoring:
            try:
                current = self.get_cursor_state()
                
                if (self._last_state and 
                    current.cursor_type != getattr(self._last_state, 'cursor_type', None)):
                    # Cursor type changed!
                    old_state = self._last_state
                    
                    # Record in history
                    self._history.append(current)
                    if len(self._history) > self._max_history:
                        self._history = self._history[-self._max_history:] # type: ignore
                    
                    # Fire callback
                    callback = self._on_change_callback
                    if callback and old_state and current:
                        try:
                            callback(old_state, current)
                        except Exception as e:
                            print(f"[CursorMonitor] Callback error: {e}")
                
                self._last_state = current
                
            except Exception as e:
                print(f"[CursorMonitor] Monitor loop error: {e}")
            
            time.sleep(poll_interval)
    
    # ═══════════════════════  History & Patterns  ═══════════════════════
    
    def get_recent_transitions(self, count: int = 10) -> List[CursorState]:
        """Returns the most recent cursor type transitions."""
        return self._history[-count:] # type: ignore
    
    def was_recently(self, cursor_type: CursorType, within_seconds: float = 2.0) -> bool:
        """Checks if the cursor was recently a specific type."""
        cutoff = time.time() - within_seconds
        return any(
            s.cursor_type == cursor_type and s.timestamp >= cutoff 
            for s in self._history
        )
    
    def get_cursor_summary(self) -> Dict[str, str]:
        """Returns a human-readable summary of the current cursor state."""
        state = self.get_cursor_state()
        return {
            "type": str(state.cursor_type.name),
            "position": f"({state.position[0]}, {state.position[1]})",
            "visible": str(state.is_visible),
            "inferred_element": str(self.infer_element_type()),
        }
