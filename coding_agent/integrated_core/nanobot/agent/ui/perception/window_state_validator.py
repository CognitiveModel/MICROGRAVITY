"""
WindowStateValidator — Pre-action window geometry validation and coordinate grounding.

Prevents the "resized window stale coordinates" bug by checking window geometry
before every click action. If the window has been resized or moved, the validator
can re-normalize it (maximize/restore), invalidate cached coordinates, and
transform old coordinates to the new geometry.

Process constants tracked:
  - Expected vs actual window rect
  - Maximized state
  - DPI scaling factor
  - Window drift history (for variance detection)
"""

import time
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List, Tuple
from collections import deque

try:
    import win32gui  # type: ignore
    import win32con  # type: ignore
    HAS_WIN32 = True
except ImportError:
    HAS_WIN32 = False


@dataclass
class WindowGeometry:
    """Snapshot of a window's geometry at a point in time."""
    hwnd: int
    title: str
    rect: Tuple[int, int, int, int]  # (left, top, right, bottom)
    is_maximized: bool
    is_visible: bool
    timestamp: float

    @property
    def width(self) -> int:
        return self.rect[2] - self.rect[0]

    @property
    def height(self) -> int:
        return self.rect[3] - self.rect[1]

    @property
    def center(self) -> Tuple[int, int]:
        return (
            (self.rect[0] + self.rect[2]) // 2,
            (self.rect[1] + self.rect[3]) // 2,
        )


@dataclass
class ValidationResult:
    """Result of pre-action window state validation."""
    is_valid: bool                     # True if coordinates are trustworthy
    geometry_changed: bool             # True if window geometry has changed
    window_resized: bool               # True specifically if size changed
    window_moved: bool                 # True specifically if position changed
    is_maximized: bool
    is_visible: bool
    current_geometry: Optional[WindowGeometry]
    expected_geometry: Optional[WindowGeometry]
    coord_transform: Optional[Dict[str, float]]  # {scale_x, scale_y, offset_x, offset_y}
    diagnostics: str                   # Human-readable diagnostic


class WindowStateValidator:
    """
    Validates window geometry before action execution to prevent stale
    coordinate bugs. Tracks geometry history for variance analysis.
    """

    # Thresholds
    POSITION_DRIFT_PX = 10      # Ignore position changes < 10px (sub-pixel rounding)
    SIZE_CHANGE_PCT = 5.0       # Flag size changes > 5%
    HISTORY_SIZE = 20           # Keep last 20 geometry snapshots

    def __init__(self, window_manager=None, process_context=None):
        self.window_manager = window_manager
        self.process_context = process_context
        # Geometry history per hwnd
        self._geometry_history: Dict[int, deque] = {}
        # Last known "good" geometry per hwnd (when action succeeded)
        self._expected_geometry: Dict[int, WindowGeometry] = {}

    def capture_geometry(self, hwnd: Optional[int] = None, title: Optional[str] = None) -> Optional[WindowGeometry]:
        """Captures current window geometry. Can find window by hwnd or title."""
        if not HAS_WIN32:
            return None

        try:
            if hwnd is None and title:
                hwnd = self._find_hwnd_by_title(title)
            if hwnd is None:
                hwnd = win32gui.GetForegroundWindow()
            if not hwnd or not win32gui.IsWindow(hwnd):
                return None

            rect = win32gui.GetWindowRect(hwnd)
            title_str = win32gui.GetWindowText(hwnd) or ""
            is_visible = win32gui.IsWindowVisible(hwnd)

            # Check if maximized
            placement = win32gui.GetWindowPlacement(hwnd)
            is_maximized = (placement[1] == win32con.SW_SHOWMAXIMIZED) if placement else False

            geo = WindowGeometry(
                hwnd=hwnd,
                title=title_str,
                rect=rect,
                is_maximized=is_maximized,
                is_visible=bool(is_visible),
                timestamp=time.time(),
            )

            # Record to history
            if hwnd not in self._geometry_history:
                self._geometry_history[hwnd] = deque(maxlen=self.HISTORY_SIZE)
            self._geometry_history[hwnd].append(geo)

            return geo

        except Exception as e:
            print(f"[WindowStateValidator] Capture error: {e}")
            return None

    def validate_before_action(
        self,
        action: Dict[str, Any],
        hwnd: Optional[int] = None,
        title: Optional[str] = None,
    ) -> ValidationResult:
        """
        Validates window state before executing an action.
        Compares current geometry against expected (last successful) geometry.
        """
        current = self.capture_geometry(hwnd=hwnd, title=title)

        if current is None:
            return ValidationResult(
                is_valid=False,
                geometry_changed=False,
                window_resized=False,
                window_moved=False,
                is_maximized=False,
                is_visible=False,
                current_geometry=None,
                expected_geometry=None,
                coord_transform=None,
                diagnostics="Cannot capture window geometry (window not found or win32 unavailable)",
            )

        # Get expected geometry
        expected = self._expected_geometry.get(current.hwnd)

        if expected is None:
            # First time seeing this window — accept as-is
            self._expected_geometry[current.hwnd] = current
            return ValidationResult(
                is_valid=True,
                geometry_changed=False,
                window_resized=False,
                window_moved=False,
                is_maximized=current.is_maximized,
                is_visible=current.is_visible,
                current_geometry=current,
                expected_geometry=None,
                coord_transform=None,
                diagnostics="First observation of this window — accepting current geometry as baseline",
            )

        # Compare geometries
        window_resized = False
        window_moved = False
        diagnostics_parts = []

        # Check size change
        width_change = abs(current.width - expected.width)
        height_change = abs(current.height - expected.height)
        if expected.width > 0:
            width_pct = (width_change / expected.width) * 100
        else:
            width_pct = 0
        if expected.height > 0:
            height_pct = (height_change / expected.height) * 100
        else:
            height_pct = 0

        if width_pct > self.SIZE_CHANGE_PCT or height_pct > self.SIZE_CHANGE_PCT:
            window_resized = True
            diagnostics_parts.append(
                f"Window RESIZED: {expected.width}x{expected.height} → "
                f"{current.width}x{current.height} "
                f"(Δ{width_pct:.1f}% W, Δ{height_pct:.1f}% H)"
            )

        # Check position drift
        dx = abs(current.rect[0] - expected.rect[0])
        dy = abs(current.rect[1] - expected.rect[1])
        if dx > self.POSITION_DRIFT_PX or dy > self.POSITION_DRIFT_PX:
            window_moved = True
            diagnostics_parts.append(
                f"Window MOVED: ({expected.rect[0]},{expected.rect[1]}) → "
                f"({current.rect[0]},{current.rect[1]}) (Δ{dx}px, Δ{dy}px)"
            )

        # Check maximized state change
        if current.is_maximized != expected.is_maximized:
            diagnostics_parts.append(
                f"Maximized state changed: {expected.is_maximized} → {current.is_maximized}"
            )

        geometry_changed = window_resized or window_moved
        is_valid = not geometry_changed

        # Compute coordinate transform if geometry changed
        coord_transform = None
        if geometry_changed and expected.width > 0 and expected.height > 0:
            coord_transform = {
                "scale_x": current.width / expected.width,
                "scale_y": current.height / expected.height,
                "offset_x": current.rect[0] - expected.rect[0],
                "offset_y": current.rect[1] - expected.rect[1],
            }

        if not diagnostics_parts:
            diagnostics_parts.append("Window geometry stable — coordinates are trustworthy")

        return ValidationResult(
            is_valid=is_valid,
            geometry_changed=geometry_changed,
            window_resized=window_resized,
            window_moved=window_moved,
            is_maximized=current.is_maximized,
            is_visible=current.is_visible,
            current_geometry=current,
            expected_geometry=expected,
            coord_transform=coord_transform,
            diagnostics=" | ".join(diagnostics_parts),
        )

    def normalize_window(self, hwnd: int) -> bool:
        """
        Normalizes a window by maximizing it to ensure consistent coordinate space.
        Updates the expected geometry to the new maximized state.
        """
        if not HAS_WIN32:
            return False

        try:
            # Bring to foreground and maximize
            win32gui.SetForegroundWindow(hwnd)
            win32gui.ShowWindow(hwnd, win32con.SW_SHOWMAXIMIZED)
            time.sleep(0.5)  # Wait for animation

            # Capture new geometry as the expected baseline
            new_geo = self.capture_geometry(hwnd=hwnd)
            if new_geo:
                self._expected_geometry[hwnd] = new_geo
                print(f"[WindowStateValidator] Window {hwnd} normalized to "
                      f"{new_geo.width}x{new_geo.height} (maximized)")
                return True

        except Exception as e:
            print(f"[WindowStateValidator] Normalize error: {e}")

        return False

    def invalidate_cached_coords(self, memory_agent=None, app_class: str = ""):
        """
        Invalidates cached presumption coordinates for an app class when
        window geometry has changed, preventing stale coordinate reuse.
        """
        if memory_agent and hasattr(memory_agent, 'invalidate_presumptions'):
            try:
                memory_agent.invalidate_presumptions(app_class)
                print(f"[WindowStateValidator] Invalidated cached coords for '{app_class}'")
            except Exception as e:
                print(f"[WindowStateValidator] Invalidation error: {e}")

        # Also invalidate presumption engine if available
        if memory_agent and hasattr(memory_agent, 'presumption_engine'):
            try:
                pe = memory_agent.presumption_engine
                if pe and hasattr(pe, 'invalidate_app'):
                    pe.invalidate_app(app_class)
            except Exception:
                pass

    def transform_coords(
        self,
        old_x: int, old_y: int,
        old_rect: Tuple[int, int, int, int],
        new_rect: Tuple[int, int, int, int],
    ) -> Tuple[int, int]:
        """
        Transforms coordinates from old window geometry to new geometry.
        Useful for re-mapping memorised coordinates after a window resize.
        """
        old_w = old_rect[2] - old_rect[0]
        old_h = old_rect[3] - old_rect[1]
        new_w = new_rect[2] - new_rect[0]
        new_h = new_rect[3] - new_rect[1]

        if old_w <= 0 or old_h <= 0:
            return (old_x, old_y)

        # Convert to relative position within old window
        rel_x = (old_x - old_rect[0]) / old_w
        rel_y = (old_y - old_rect[1]) / old_h

        # Map to new window
        new_x = int(new_rect[0] + rel_x * new_w)
        new_y = int(new_rect[1] + rel_y * new_h)

        return (new_x, new_y)

    def record_successful_action(self, hwnd: int):
        """
        Called after a successful action to update the expected geometry
        baseline. This ensures that valid window positions are remembered.
        """
        current = self.capture_geometry(hwnd=hwnd)
        if current:
            self._expected_geometry[hwnd] = current

    def get_geometry_variance(self, hwnd: int) -> Dict[str, Any]:
        """
        Analyzes geometry variance for a window over its history.
        Returns statistics about how much the window moves/resizes.
        """
        history = self._geometry_history.get(hwnd)
        if not history or len(history) < 2:
            return {"invariant": True, "samples": 0}

        widths = [g.width for g in history]
        heights = [g.height for g in history]
        x_positions = [g.rect[0] for g in history]
        y_positions = [g.rect[1] for g in history]

        width_variance = max(widths) - min(widths)
        height_variance = max(heights) - min(heights)
        x_variance = max(x_positions) - min(x_positions)
        y_variance = max(y_positions) - min(y_positions)

        is_invariant = (
            width_variance < self.POSITION_DRIFT_PX and
            height_variance < self.POSITION_DRIFT_PX
        )

        return {
            "invariant": is_invariant,
            "samples": len(history),
            "width_variance": width_variance,
            "height_variance": height_variance,
            "x_variance": x_variance,
            "y_variance": y_variance,
            "typically_maximized": sum(1 for g in history if g.is_maximized) > len(history) // 2,
        }

    def _find_hwnd_by_title(self, title: str) -> Optional[int]:
        """Finds a window handle by partial title match."""
        if not HAS_WIN32:
            return None

        result = [None]
        title_lower = title.lower()

        def callback(hwnd, _):
            if win32gui.IsWindowVisible(hwnd):
                window_text = win32gui.GetWindowText(hwnd).lower()
                if title_lower in window_text:
                    result[0] = hwnd
                    return False  # Stop enumeration
            return True

        try:
            win32gui.EnumWindows(callback, None)
        except Exception:
            pass

        return result[0]

    def update_process_context(self, geometry: WindowGeometry):
        """Updates the shared process context with current window state."""
        if self.process_context:
            self.process_context.update(
                active_hwnd=geometry.hwnd,
                window_title=geometry.title,
                rect=geometry.rect,
            )
