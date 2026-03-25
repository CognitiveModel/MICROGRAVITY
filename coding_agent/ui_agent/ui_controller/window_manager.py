import win32gui # type: ignore
import win32con # type: ignore
import win32api # type: ignore
import ctypes
import time
from typing import Optional, List, Tuple, Dict, Any

class WindowManager:
    """
    Full window operations engine: move, resize, minimize, maximize, close,
    snap, anchor detection, Z-order tracking, and smart window switching.
    """
    def __init__(self):
        self._screen_w = win32api.GetSystemMetrics(0)
        self._screen_h = win32api.GetSystemMetrics(1)
        self._anchor_memory: Dict[int, Tuple[int, int, int, int]] = {}  # hwnd → last known "home" rect

    def get_hwnd_by_title(self, partial_title: str) -> Optional[int]:
        """Finds a window handle by partial title match."""
        hwnd_list = []
        def enum_handler(hwnd, lparam):
            if win32gui.IsWindowVisible(hwnd):
                title = win32gui.GetWindowText(hwnd)
                if partial_title.lower() in title.lower():
                    hwnd_list.append(hwnd)
            return True
        
        win32gui.EnumWindows(enum_handler, None)
        return hwnd_list[0] if hwnd_list else None

    def minimize(self, hwnd: int) -> bool:
        """Minimizes the specified window (Shrink)."""
        try:
            win32gui.ShowWindow(hwnd, win32con.SW_MINIMIZE)
            return True
        except Exception as e:
            print(f"[WindowManager] Error minimizing window {hwnd}: {e}")
            return False

    def maximize(self, hwnd: int) -> bool:
        """Maximizes the specified window (Expand)."""
        try:
            win32gui.ShowWindow(hwnd, win32con.SW_MAXIMIZE)
            return True
        except Exception as e:
            print(f"[WindowManager] Error maximizing window {hwnd}: {e}")
            return False

    def restore(self, hwnd: int) -> bool:
        """Restores a minimized/maximized window to its normal state."""
        try:
            win32gui.ShowWindow(hwnd, win32con.SW_RESTORE)
            return True
        except Exception as e:
            print(f"[WindowManager] Error restoring window {hwnd}: {e}")
            return False

    def resize(self, hwnd: int, width: int, height: int, x: Optional[int] = None, y: Optional[int] = None) -> bool:
        """Resizes the specified window. Optionally moves it."""
        try:
            curr_rect = win32gui.GetWindowRect(hwnd)
            new_x = x if x is not None else curr_rect[0]
            new_y = y if y is not None else curr_rect[1]
            win32gui.MoveWindow(hwnd, new_x, new_y, width, height, True)
            return True
        except Exception as e:
            print(f"[WindowManager] Error resizing window {hwnd}: {e}")
            return False

    def get_window_rect(self, hwnd: int) -> Optional[Tuple[int, int, int, int]]:
        """Returns the window bounds as (x, y, width, height)."""
        try:
            rect = win32gui.GetWindowRect(hwnd)
            return (rect[0], rect[1], rect[2] - rect[0], rect[3] - rect[1])
        except Exception as e:
            print(f"[WindowManager] Error getting rect for window {hwnd}: {e}")
            return None

    def focus_window(self, hwnd: int) -> bool:
        """Brings the window to the foreground."""
        try:
            if win32gui.IsIconic(hwnd):
                win32gui.ShowWindow(hwnd, win32con.SW_RESTORE)
            win32gui.SetForegroundWindow(hwnd)
            time.sleep(0.3)  # Human-like focus delay
            return True
        except Exception as e:
            print(f"[WindowManager] Error focusing window {hwnd}: {e}")
            return False

    # ═══════════════════════  NEW: Extended Operations  ═══════════════════════

    def close_window(self, hwnd: int) -> bool:
        """Sends WM_CLOSE to gracefully close a window."""
        try:
            win32gui.PostMessage(hwnd, win32con.WM_CLOSE, 0, 0)
            return True
        except Exception as e:
            print(f"[WindowManager] Error closing window {hwnd}: {e}")
            return False

    def get_all_visible_windows(self) -> List[Dict[str, Any]]:
        """Returns list of all visible windows with basic info."""
        windows = []
        def enum_handler(hwnd, _):
            if win32gui.IsWindowVisible(hwnd):
                title = win32gui.GetWindowText(hwnd)
                if title and title != "Program Manager":
                    try:
                        rect = win32gui.GetWindowRect(hwnd)
                        windows.append({
                            "hwnd": hwnd,
                            "title": title,
                            "rect": rect,
                            "is_minimized": bool(win32gui.IsIconic(hwnd)),
                            "is_maximized": bool(ctypes.windll.user32.IsZoomed(hwnd)), # type: ignore
                            "class_name": win32gui.GetClassName(hwnd),
                        })
                    except Exception:
                        pass
            return True
        win32gui.EnumWindows(enum_handler, None)
        return windows

    def get_z_order(self) -> List[int]:
        """Returns hwnds ordered by Z-order (front to back)."""
        z_order = []
        def enum_handler(hwnd, _):
            if win32gui.IsWindowVisible(hwnd):
                title = win32gui.GetWindowText(hwnd)
                if title and title != "Program Manager":
                    z_order.append(hwnd)
            return True
        win32gui.EnumWindows(enum_handler, None)
        return z_order

    def get_window_state(self, hwnd: int) -> str:
        """Returns NORMAL | MINIMIZED | MAXIMIZED | FULLSCREEN | HIDDEN."""
        try:
            if not win32gui.IsWindowVisible(hwnd):
                return "HIDDEN"
            if win32gui.IsIconic(hwnd):
                return "MINIMIZED"
            if ctypes.windll.user32.IsZoomed(hwnd): # type: ignore
                return "MAXIMIZED"
            rect = win32gui.GetWindowRect(hwnd)
            w, h = rect[2] - rect[0], rect[3] - rect[1]
            if w >= self._screen_w - 5 and h >= self._screen_h - 5:
                return "FULLSCREEN"
            return "NORMAL"
        except Exception:
            return "UNKNOWN"

    def analyse_window_size(self, hwnd: int) -> Dict[str, Any]:
        """Returns full size analysis with DWM-aware bounds."""
        try:
            outer = win32gui.GetWindowRect(hwnd)
            client_origin = win32gui.ClientToScreen(hwnd, (0, 0))
            client_rect = win32gui.GetClientRect(hwnd)

            caption_h = win32api.GetSystemMetrics(win32con.SM_CYCAPTION)
            border_x = win32api.GetSystemMetrics(win32con.SM_CXFRAME) + win32api.GetSystemMetrics(win32con.SM_CXPADDEDBORDER)

            w = outer[2] - outer[0]
            h = outer[3] - outer[1]
            coverage = (w * h) / (self._screen_w * self._screen_h) * 100

            state = self.get_window_state(hwnd)
            if state == "FULLSCREEN":
                cat = "FULLSCREEN"
            elif state == "MAXIMIZED":
                cat = "MAXIMIZED"
            elif coverage > 70:
                cat = "LARGE"
            elif coverage > 30:
                cat = "MEDIUM"
            elif coverage > 10:
                cat = "SMALL"
            else:
                cat = "TINY"

            return {
                "outer_rect": outer,
                "client_rect": (client_origin[0], client_origin[1],
                                client_origin[0] + client_rect[2], client_origin[1] + client_rect[3]),
                "title_bar_height": caption_h,
                "border_width": border_x,
                "width": w, "height": h,
                "screen_coverage_pct": round(coverage, 1),
                "size_category": cat,
            }
        except Exception as e:
            print(f"[WindowManager] Error analysing window {hwnd}: {e}")
            return {}

    def is_snapped(self, hwnd: int) -> str:
        """Returns snap position or NONE."""
        try:
            rect = win32gui.GetWindowRect(hwnd)
        except Exception:
            return "NONE"

        left, top, right, bottom = rect
        w, h = right - left, bottom - top
        sw, sh = self._screen_w, self._screen_h
        tol = 15

        half_w = sw // 2
        left_edge = abs(left) < tol
        right_edge = abs(right - sw) < tol
        top_edge = abs(top) < tol
        half_width = abs(w - half_w) < tol * 2

        if left_edge and half_width: return "LEFT_HALF"
        if right_edge and half_width: return "RIGHT_HALF"
        if left_edge and top_edge and abs(w - half_w) < tol * 2 and abs(h - sh // 2) < tol * 2: return "TOP_LEFT"
        if right_edge and top_edge and abs(w - half_w) < tol * 2 and abs(h - sh // 2) < tol * 2: return "TOP_RIGHT"
        return "NONE"

    def snap_window(self, hwnd: int, position: str) -> bool:
        """Snaps window to position by moving/resizing it."""
        try:
            sw, sh = self._screen_w, self._screen_h
            positions = {
                "LEFT_HALF": (0, 0, sw // 2, sh),
                "RIGHT_HALF": (sw // 2, 0, sw // 2, sh),
                "TOP_LEFT": (0, 0, sw // 2, sh // 2),
                "TOP_RIGHT": (sw // 2, 0, sw // 2, sh // 2),
                "BOTTOM_LEFT": (0, sh // 2, sw // 2, sh // 2),
                "BOTTOM_RIGHT": (sw // 2, sh // 2, sw // 2, sh // 2),
            }
            if position not in positions:
                return False
            x, y, w, h = positions[position]
            if win32gui.IsIconic(hwnd):
                win32gui.ShowWindow(hwnd, win32con.SW_RESTORE)
            win32gui.MoveWindow(hwnd, x, y, w, h, True)
            return True
        except Exception as e:
            print(f"[WindowManager] Snap failed: {e}")
            return False

    def is_anchored(self, hwnd: int) -> bool:
        """Checks if window position matches its last-known 'home' position."""
        if hwnd not in self._anchor_memory:
            try:
                self._anchor_memory[hwnd] = win32gui.GetWindowRect(hwnd)
            except Exception:
                pass
            return False
        try:
            current = win32gui.GetWindowRect(hwnd)
            return current == self._anchor_memory[hwnd]
        except Exception:
            return False

    def switch_to_window_smart(self, target_hwnd: int, window_ledger: dict = None) -> List[str]: # type: ignore
        """Intelligently switches to a window considering overlaps and state."""
        actions_taken = []

        if win32gui.GetForegroundWindow() == target_hwnd:
            return ["Already foreground"]

        state = self.get_window_state(target_hwnd)

        if state == "MINIMIZED":
            self.restore(target_hwnd)
            actions_taken.append("Restored from minimized")
            time.sleep(0.3)

        self.focus_window(target_hwnd)
        actions_taken.append("Focused window")

        return actions_taken

    def cascade_windows(self, hwnds: List[int]) -> bool:
        """Arranges windows in cascade layout."""
        try:
            offset = 30
            for i, hwnd in enumerate(hwnds):
                x = offset * i
                y = offset * i
                win32gui.MoveWindow(hwnd, x, y, 800, 600, True)
            return True
        except Exception as e:
            print(f"[WindowManager] Cascade failed: {e}")
            return False

    def tile_windows(self, hwnds: List[int], layout: str = 'horizontal') -> bool:
        """Tiles given windows side-by-side or stacked."""
        try:
            n = len(hwnds)
            if n == 0:
                return False
            sw, sh = self._screen_w, self._screen_h
            for i, hwnd in enumerate(hwnds):
                if win32gui.IsIconic(hwnd):
                    win32gui.ShowWindow(hwnd, win32con.SW_RESTORE)
                if layout == 'horizontal':
                    w = sw // n
                    win32gui.MoveWindow(hwnd, i * w, 0, w, sh, True)
                else:
                    h = sh // n
                    win32gui.MoveWindow(hwnd, 0, i * h, sw, h, True)
            return True
        except Exception as e:
            print(f"[WindowManager] Tile failed: {e}")
            return False

    # ═══════════════════════  Taskbar Awareness  ═══════════════════════

    def get_taskbar_rect(self) -> Optional[Tuple[int, int, int, int]]:
        """Returns the taskbar bounding box as (left, top, right, bottom).
        
        Uses the Shell_TrayWnd class to find the Windows taskbar.
        Also returns position info (bottom, top, left, right) based on geometry.
        """
        try:
            taskbar_hwnd = win32gui.FindWindow("Shell_TrayWnd", None)
            if taskbar_hwnd:
                rect = win32gui.GetWindowRect(taskbar_hwnd)
                return rect  # (left, top, right, bottom)
        except Exception as e:
            print(f"[WindowManager] Error getting taskbar rect: {e}")
        return None

    def get_taskbar_info(self) -> Dict[str, Any]:
        """Returns taskbar position, dimensions, and orientation."""
        rect = self.get_taskbar_rect()
        if not rect:
            return {"available": False}
        
        left, top, right, bottom = rect
        width = right - left
        height = bottom - top

        # Determine taskbar position based on geometry
        if top > self._screen_h * 0.5:
            position = "bottom"
        elif bottom < self._screen_h * 0.5:
            position = "top"
        elif left > self._screen_w * 0.5:
            position = "right"
        else:
            position = "left"

        return {
            "available": True,
            "rect": rect,
            "position": position,
            "width": width,
            "height": height,
            "center_y": top + height // 2,
        }

    def get_all_windows_for_process(self, process_name: str) -> List[Dict[str, Any]]:
        """Returns all visible windows belonging to a given process name (e.g. 'chrome.exe').
        
        Useful for multi-instance detection: enumerate all Chrome windows to check
        which accounts are logged in, then decide whether to reuse or open new.
        """
        try:
            import win32process as _w32p  # type: ignore
            import psutil as _psutil  # type: ignore
        except ImportError:
            return []

        results: List[Dict[str, Any]] = []
        process_name_lower = process_name.lower()

        def enum_handler(hwnd, _):
            if not win32gui.IsWindowVisible(hwnd):
                return True
            title = win32gui.GetWindowText(hwnd)
            if not title:
                return True
            try:
                _, pid = _w32p.GetWindowThreadProcessId(hwnd)
                proc = _psutil.Process(pid)
                if process_name_lower in proc.name().lower():
                    rect = win32gui.GetWindowRect(hwnd)
                    results.append({
                        "hwnd": hwnd,
                        "title": title,
                        "pid": pid,
                        "process_name": proc.name(),
                        "is_minimized": bool(win32gui.IsIconic(hwnd)),
                        "is_maximized": bool(ctypes.windll.user32.IsZoomed(hwnd)),  # type: ignore
                        "rect": rect,
                    })
            except Exception:
                pass
            return True

        win32gui.EnumWindows(enum_handler, None)
        return results

    def get_window_count_by_title(self, partial_title: str) -> Dict[str, Any]:
        """Returns count and list of all visible windows matching a partial title.
        
        Example: get_window_count_by_title("chrome") returns all Chrome windows
        with their full titles (which often include account/profile info).
        """
        matches: List[Dict[str, Any]] = []

        def enum_handler(hwnd, _):
            if not win32gui.IsWindowVisible(hwnd):
                return True
            title = win32gui.GetWindowText(hwnd)
            if title and partial_title.lower() in title.lower():
                matches.append({
                    "hwnd": hwnd,
                    "title": title,
                    "is_minimized": bool(win32gui.IsIconic(hwnd)),
                })
            return True

        win32gui.EnumWindows(enum_handler, None)
        return {
            "count": len(matches),
            "windows": matches,
        }
