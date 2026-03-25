"""
OSAwareness — OS-level window and process enumeration with state tracking.

Provides the foundational reality model:
  - Window State Ledger: exact boundaries, display state, snap/anchor detection
  - Process Ledger: per-process resource monitoring (CPU, RAM)
  - Z-Order tracking: front-to-back window ordering
  - Overlap detection: finds overlapping windows with coverage percentages
  - Change tracking: diffs between scans to detect opened/closed/moved windows
"""

import time
import ctypes
import ctypes.wintypes
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any

try:
    import win32gui # type: ignore
    import win32con # type: ignore
    import win32api # type: ignore
    import win32process # type: ignore
except ImportError:
    win32gui = None
    win32con = None
    win32api = None
    win32process = None

try:
    import psutil # type: ignore
except ImportError:
    psutil = None


# ──────────────────────────  Data Structures  ──────────────────────────

@dataclass
class WindowState:
    """Complete state record for a single OS window."""
    hwnd: int
    title: str
    class_name: str
    process_name: str
    pid: int

    # === Exact Boundaries ===
    outer_rect: Tuple[int, int, int, int] = (0, 0, 0, 0)   # (left, top, right, bottom)
    client_rect: Tuple[int, int, int, int] = (0, 0, 0, 0)   # (left, top, right, bottom) in screen coords
    title_bar_rect: Tuple[int, int, int, int] = (0, 0, 0, 0)
    border_width: int = 0
    title_bar_height: int = 0

    # === Display State ===
    state: str = "NORMAL"           # NORMAL | MINIMIZED | MAXIMIZED | FULLSCREEN | HIDDEN
    is_visible: bool = True
    is_foreground: bool = False
    z_order: int = -1               # 0 = topmost
    opacity: float = 1.0

    # === Snap / Anchor ===
    snap_position: str = "NONE"     # NONE | LEFT_HALF | RIGHT_HALF | TOP_LEFT | TOP_RIGHT | ...
    is_anchored: bool = False
    anchor_coords: Optional[Tuple[int, int, int, int]] = None

    # === Size Analysis ===
    width: int = 0
    height: int = 0
    screen_coverage_pct: float = 0.0
    size_category: str = "MEDIUM"   # TINY | SMALL | MEDIUM | LARGE | MAXIMIZED | FULLSCREEN

    # === Chrome Boundaries (populated by ElementBoundaryLearner) ===
    chrome: Dict[str, Any] = field(default_factory=dict)

    # === Tracking ===
    first_seen: float = 0.0
    last_updated: float = 0.0
    position_history: List[Tuple[int, int, int, int, float]] = field(default_factory=list)  # (x,y,w,h,timestamp)


@dataclass
class ProcessInfo:
    """Resource info for a running process."""
    pid: int
    name: str
    exe_path: str = ""
    cpu_percent: float = 0.0
    memory_mb: float = 0.0
    status: str = "running"
    window_hwnds: List[int] = field(default_factory=list)


# ──────────────────────────  OSAwareness  ──────────────────────────

class OSAwareness:
    """Enumerates all OS windows and processes, tracks exact boundaries,
    state (min/max/snap/anchor), and resource usage."""

    def __init__(self):
        self.window_ledger: Dict[int, WindowState] = {}
        self.process_ledger: Dict[int, ProcessInfo] = {}
        self.foreground_hwnd: Optional[int] = None
        self.z_order_stack: List[int] = []

        # Screen dimensions for snap detection
        self._screen_w = win32api.GetSystemMetrics(0) if win32api else 1920
        self._screen_h = win32api.GetSystemMetrics(1) if win32api else 1080

        # System metrics cache
        try:
            self._caption_h = win32api.GetSystemMetrics(win32con.SM_CYCAPTION) if win32api else 30 # type: ignore
            self._border_x = (win32api.GetSystemMetrics(win32con.SM_CXFRAME) + # type: ignore
                              win32api.GetSystemMetrics(win32con.SM_CXPADDEDBORDER)) if win32api else 8 # type: ignore
            self._border_y = (win32api.GetSystemMetrics(win32con.SM_CYFRAME) + # type: ignore
                              win32api.GetSystemMetrics(win32con.SM_CXPADDEDBORDER)) if win32api else 8 # type: ignore

        except Exception:
            self._caption_h = 30
            self._border_x = 8
            self._border_y = 8

        print("[OSAwareness] Initialized")

    # ═══════════════════════  Window Enumeration  ═══════════════════════

    def scan_all_windows(self) -> Dict[int, WindowState]:
        """Enumerates all visible windows with full state information."""
        if win32gui is None:
            return {}

        now = time.time()
        old_ledger: Dict[int, WindowState] = self.window_ledger.copy()
        new_ledger: Dict[int, WindowState] = {}
        z_order = []

        fg_hwnd = win32gui.GetForegroundWindow()
        self.foreground_hwnd = fg_hwnd

        def enum_callback(hwnd, _):
            if not win32gui.IsWindowVisible(hwnd): # type: ignore
                return True
            title = win32gui.GetWindowText(hwnd) # type: ignore
            if not title or title in ("", "Program Manager"):
                return True

            try:
                class_name = win32gui.GetClassName(hwnd) # type: ignore

            except Exception:
                class_name = ""

            # Get process info
            pid = 0
            process_name = ""
            try:
                _, pid = win32process.GetWindowThreadProcessId(hwnd) # type: ignore
                if psutil:
                    try:
                        proc = psutil.Process(pid)
                        process_name = proc.name()
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        process_name = "unknown"
            except Exception:
                pass

            # Get window rect
            try:
                outer = win32gui.GetWindowRect(hwnd) # type: ignore
            except Exception:
                outer = (0, 0, 0, 0)

            # Get client rect in screen coordinates
            try:
                client_origin = win32gui.ClientToScreen(hwnd, (0, 0)) # type: ignore
                c_rect = win32gui.GetClientRect(hwnd) # type: ignore
                client_rect = (client_origin[0], client_origin[1],
                               client_origin[0] + c_rect[2], client_origin[1] + c_rect[3])
            except Exception:
                client_rect = outer

            # Determine window state
            state = "NORMAL"
            is_minimized = win32gui.IsIconic(hwnd) # type: ignore
            if is_minimized:
                state = "MINIMIZED"
            elif self._is_zoomed(hwnd):
                state = "MAXIMIZED"

            w = outer[2] - outer[0]
            h = outer[3] - outer[1]

            # Fullscreen check
            if (not is_minimized and w >= self._screen_w - 5 and h >= self._screen_h - 5):
                state = "FULLSCREEN"

            # Title bar and border
            title_bar_rect = (outer[0], outer[1], outer[2], outer[1] + self._border_y + self._caption_h)

            # Coverage
            screen_area = self._screen_w * self._screen_h
            coverage = (w * h) / max(screen_area, 1) * 100

            # Size category
            size_cat = self._classify_size(w, h, coverage, state)

            # Snap detection
            snap_pos = self._detect_snap_position(outer)

            # Check if anchored (same position as previously known)
            is_anchored = False
            anchor_coords = None
            if hwnd in old_ledger:
                old = old_ledger[hwnd]
                if old.outer_rect == outer and old.state == state:
                    is_anchored = True
                    anchor_coords = outer

            # Position history
            pos_history = []
            if hwnd in old_ledger:
                pos_history = list(old_ledger[hwnd].position_history)
            pos_history.append((outer[0], outer[1], w, h, now))
            pos_history = list(pos_history)[-5:] # type: ignore  # Keep last 5

            # Chrome (preserved from old if available)
            chrome = old_ledger[hwnd].chrome if hwnd in old_ledger else {}

            ws = WindowState(
                hwnd=hwnd,
                title=title,
                class_name=class_name,
                process_name=process_name,
                pid=pid,
                outer_rect=outer,
                client_rect=client_rect,
                title_bar_rect=title_bar_rect,
                border_width=self._border_x,
                title_bar_height=self._caption_h,
                state=state,
                is_visible=True,
                is_foreground=(hwnd == fg_hwnd),
                z_order=len(z_order),
                snap_position=snap_pos,
                is_anchored=is_anchored,
                anchor_coords=anchor_coords,
                width=w,
                height=h,
                screen_coverage_pct=float(f"{coverage:.1f}"), # type: ignore
                size_category=size_cat,
                chrome=chrome,
                first_seen=old_ledger[hwnd].first_seen if hwnd in old_ledger else now,
                last_updated=now,
                position_history=pos_history,
            )

            new_ledger[hwnd] = ws
            z_order.append(hwnd)
            return True

        win32gui.EnumWindows(enum_callback, None)

        self.window_ledger = new_ledger
        self.z_order_stack = z_order

        return new_ledger

    # ═══════════════════════  Process Enumeration  ═══════════════════════

    def scan_processes(self) -> Dict[int, ProcessInfo]:
        """Enumerates running processes with resource info."""
        if psutil is None:
            return {}

        result: Dict[int, ProcessInfo] = {}

        # Build pid → hwnd mapping
        pid_hwnds: Dict[int, List[int]] = {}
        for hwnd, ws in self.window_ledger.items():
            pid_hwnds.setdefault(ws.pid, []).append(hwnd)

        for proc in psutil.process_iter(["pid", "name", "exe", "cpu_percent", "memory_info", "status"]):
            try:
                info = proc.info
                pid = info["pid"]
                mem_mb = info["memory_info"].rss / (1024 * 1024) if info.get("memory_info") else 0
                result[pid] = ProcessInfo(
                    pid=pid,
                    name=info.get("name", ""),
                    exe_path=info.get("exe", "") or "",
                    cpu_percent=info.get("cpu_percent", 0) or 0,
                    memory_mb=round(mem_mb, 1),
                    status=info.get("status", "unknown"),
                    window_hwnds=pid_hwnds.get(pid, []),
                )
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                continue

        self.process_ledger = result
        return result

    # ═══════════════════════  State Queries  ═══════════════════════

    def get_foreground_window(self) -> Optional[WindowState]:
        """Returns the currently focused window's full WindowState."""
        if self.foreground_hwnd and self.foreground_hwnd in self.window_ledger:
            return self.window_ledger[self.foreground_hwnd] # type: ignore
        return None

    def get_z_order(self) -> List[WindowState]:
        """Returns windows ordered by Z-order (front to back)."""
        return [self.window_ledger[h] for h in self.z_order_stack if h in self.window_ledger]

    def get_minimized_windows(self) -> List[WindowState]:
        """Returns list of all currently minimized windows."""
        return [ws for ws in self.window_ledger.values() if ws.state == "MINIMIZED"]

    def get_maximized_windows(self) -> List[WindowState]:
        """Returns list of all currently maximized windows."""
        return [ws for ws in self.window_ledger.values() if ws.state == "MAXIMIZED"]

    def get_snapped_windows(self) -> List[WindowState]:
        """Returns list of windows snapped to screen edges/quadrants."""
        return [ws for ws in self.window_ledger.values() if ws.snap_position != "NONE"]

    def get_anchored_windows(self) -> List[WindowState]:
        """Returns windows that maintain their position across scans."""
        return [ws for ws in self.window_ledger.values() if ws.is_anchored]

    # ═══════════════════════  Overlap Detection  ═══════════════════════

    def detect_overlapping_windows(self) -> List[Dict[str, Any]]:
        """Returns pairs of windows whose rects overlap, with overlap percentage."""
        visible = [ws for ws in self.window_ledger.values()
                    if ws.is_visible and ws.state not in ("MINIMIZED", "HIDDEN")]

        overlaps = []
        for i, w1 in enumerate(visible):
            visible_slice = visible[i + 1:] # type: ignore
            for w2 in visible_slice:
                pct = self._overlap_percentage(w1.outer_rect, w2.outer_rect)
                if pct > 0:
                    overlaps.append({
                        "window_a": {"hwnd": w1.hwnd, "title": w1.title},
                        "window_b": {"hwnd": w2.hwnd, "title": w2.title},
                        "overlap_pct": float(f"{pct:.1f}"), # type: ignore
                        "a_z_order": w1.z_order,
                        "b_z_order": w2.z_order,
                    })

        return sorted(overlaps, key=lambda x: x["overlap_pct"], reverse=True)

    # ═══════════════════════  Resource Summary  ═══════════════════════

    def get_resource_summary(self) -> Dict[str, Any]:
        """Returns system-wide and per-app resource usage."""
        summary = {
            "total_cpu": 0.0,
            "total_ram_mb": 0.0,
            "per_app": {},
        }

        if psutil:
            try:
                summary["total_cpu"] = psutil.cpu_percent(interval=0)
                mem = psutil.virtual_memory()
                summary["total_ram_mb"] = round(mem.used / (1024 * 1024), 1)
            except Exception:
                pass

        # Aggregate by process name
        for proc in self.process_ledger.values():
            name = proc.name
            if name not in summary["per_app"]: # type: ignore
                summary["per_app"][name] = {"cpu": 0, "ram_mb": 0, "window_count": 0} # type: ignore
            summary["per_app"][name]["cpu"] += proc.cpu_percent # type: ignore
            summary["per_app"][name]["ram_mb"] += proc.memory_mb # type: ignore
            summary["per_app"][name]["window_count"] += len(proc.window_hwnds) # type: ignore

        return summary

    # ═══════════════════════  Window Size Analysis  ═══════════════════════

    def analyse_window_size(self, hwnd: int) -> Dict[str, Any]:
        """Full size analysis of a window."""
        ws = self.window_ledger.get(hwnd)
        if not ws:
            return {}

        return {
            "outer_rect": ws.outer_rect,
            "client_rect": ws.client_rect,
            "title_bar_height": ws.title_bar_height,
            "border_width": ws.border_width,
            "width": ws.width,
            "height": ws.height,
            "total_area_px": ws.width * ws.height,
            "screen_coverage_pct": ws.screen_coverage_pct,
            "is_fullscreen": ws.state == "FULLSCREEN",
            "size_category": ws.size_category,
        }

    # ═══════════════════════  Change Tracking  ═══════════════════════

    def track_window_changes(self, prev_ledger: Dict[int, WindowState]) -> Dict[str, List]:
        """Diffs current vs previous ledger."""
        current_hwnds = set(self.window_ledger.keys())
        prev_hwnds = set(prev_ledger.keys())

        opened = [self.window_ledger[h] for h in current_hwnds - prev_hwnds]
        closed = [prev_ledger[h] for h in prev_hwnds - current_hwnds]

        moved = []
        resized = []
        state_changed = []

        for h in current_hwnds & prev_hwnds:
            curr = self.window_ledger[h]
            prev = prev_ledger[h]

            if curr.outer_rect != prev.outer_rect:
                if curr.width != prev.width or curr.height != prev.height:
                    resized.append({"hwnd": h, "title": curr.title,
                                    "old_size": (prev.width, prev.height),
                                    "new_size": (curr.width, curr.height)})
                else:
                    moved.append({"hwnd": h, "title": curr.title,
                                  "old_pos": prev.outer_rect[:2],
                                  "new_pos": curr.outer_rect[:2]})

            if curr.state != prev.state:
                state_changed.append({"hwnd": h, "title": curr.title,
                                      "old_state": prev.state, "new_state": curr.state})

        return {
            "opened": opened,
            "closed": closed,
            "moved": moved,
            "resized": resized,
            "state_changed": state_changed,
        }

    # ═══════════════════════  Helpers  ═══════════════════════

    def _is_zoomed(self, hwnd: int) -> bool:
        """Checks if a window is maximized."""
        try:
            return bool(ctypes.windll.user32.IsZoomed(hwnd)) # type: ignore
        except Exception:
            return False

    def _detect_snap_position(self, rect: Tuple[int, int, int, int]) -> str:
        """Detects snap position by comparing window rect to screen quadrants."""
        left, top, right, bottom = rect
        w = right - left
        h = bottom - top
        tol = 15  # Pixel tolerance

        sw, sh = self._screen_w, self._screen_h
        half_w = sw // 2
        half_h = sh // 2

        left_edge = abs(left) < tol
        right_edge = abs(right - sw) < tol
        top_edge = abs(top) < tol
        bottom_edge = abs(bottom - sh) < tol or abs(bottom - (sh - 40)) < tol  # Taskbar offset

        half_width = abs(w - half_w) < tol * 2
        half_height = abs(h - half_h) < tol * 2
        full_width = abs(w - sw) < tol * 2
        full_height = abs(h - sh) < tol * 2 or abs(h - (sh - 40)) < tol * 2

        # Full screen
        if full_width and full_height:
            return "FULL"

        # Left half
        if left_edge and half_width and (top_edge or full_height):
            return "LEFT_HALF"

        # Right half
        if right_edge and half_width and (top_edge or full_height):
            return "RIGHT_HALF"

        # Quadrants
        if left_edge and top_edge and half_width and half_height:
            return "TOP_LEFT"
        if right_edge and top_edge and half_width and half_height:
            return "TOP_RIGHT"
        if left_edge and bottom_edge and half_width and half_height:
            return "BOTTOM_LEFT"
        if right_edge and bottom_edge and half_width and half_height:
            return "BOTTOM_RIGHT"

        return "NONE"

    def _classify_size(self, w: int, h: int, coverage: float, state: str) -> str:
        """Classifies window size category."""
        if state == "FULLSCREEN":
            return "FULLSCREEN"
        if state == "MAXIMIZED":
            return "MAXIMIZED"
        if coverage > 70:
            return "LARGE"
        if coverage > 30:
            return "MEDIUM"
        if coverage > 10:
            return "SMALL"
        return "TINY"

    def _overlap_percentage(self, rect_a: tuple, rect_b: tuple) -> float:
        """Computes overlap percentage between two rectangles."""
        x_overlap = max(0, min(rect_a[2], rect_b[2]) - max(rect_a[0], rect_b[0]))
        y_overlap = max(0, min(rect_a[3], rect_b[3]) - max(rect_a[1], rect_b[1]))
        inter = x_overlap * y_overlap

        area_a = (rect_a[2] - rect_a[0]) * (rect_a[3] - rect_a[1])
        area_b = (rect_b[2] - rect_b[0]) * (rect_b[3] - rect_b[1])
        smaller = min(area_a, area_b)

        return (inter / max(smaller, 1)) * 100

    def get_summary(self) -> str:
        """Concise text summary for planner context."""
        total = len(self.window_ledger)
        mins = len(self.get_minimized_windows())
        maxs = len(self.get_maximized_windows())
        snaps = len(self.get_snapped_windows())
        fg = self.get_foreground_window()
        fg_title = fg.title[:40] if fg else "None" # type: ignore
        overlaps = len(self.detect_overlapping_windows())
        taskbar = self.get_taskbar_info()
        tb_str = f"Taskbar: {taskbar.get('position', 'unknown')}" if taskbar.get("available") else ""

        return (f"OS: {total} windows ({mins} min, {maxs} max, {snaps} snapped), "
                f"{overlaps} overlapping pairs. Foreground: '{fg_title}'. {tb_str}")

    def get_taskbar_info(self) -> Dict[str, Any]:
        """Returns taskbar position and rect."""
        try:
            hwnd = win32gui.FindWindow("Shell_TrayWnd", None) # type: ignore
            if hwnd:
                rect = win32gui.GetWindowRect(hwnd) # type: ignore
                left, top, right, bottom = rect
                pos = "bottom"
                if top > self._screen_h * 0.5: pos = "bottom"
                elif bottom < self._screen_h * 0.5: pos = "top"
                elif left > self._screen_w * 0.5: pos = "right"
                else: pos = "left"
                return {"available": True, "rect": rect, "position": pos}
        except Exception:
            pass
        return {"available": False}
