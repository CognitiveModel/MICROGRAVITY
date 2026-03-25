"""
ScreenGeometry — Centralized, DPI-aware screen geometry provider.

Single source of truth for all coordinate calculations in the nanobot UI agent.
Handles the mismatch between:
  - Physical pixels (what mss captures, what the display actually has)
  - Logical pixels (what pyautogui operates in, what Windows APIs report when DPI-aware)
  - Normalized coordinates (0-1000 range from VLM responses)

Usage:
    geo = ScreenGeometry()
    print(geo.logical_size)          # (1920, 1080) — what pyautogui sees
    print(geo.physical_size)         # (2880, 1620) — actual pixel count on a 150% scaled display
    print(geo.scale_factor)          # 1.5
    
    # Convert VLM normalized coords (0-1000) -> pyautogui-compatible coords
    x, y = geo.normalized_to_screen(500, 500)  # center of screen
    
    # Convert screenshot image pixel coords -> pyautogui-compatible coords  
    x, y = geo.image_to_screen(img_x, img_y, img_w, img_h)
"""

import ctypes
import ctypes.wintypes
from dataclasses import dataclass
from typing import Tuple, Optional, Dict, Any
import threading


@dataclass
class MonitorInfo:
    """Per-monitor geometry info."""
    handle: int
    name: str
    left: int
    top: int
    right: int
    bottom: int
    physical_width: int
    physical_height: int
    logical_width: int
    logical_height: int
    scale_factor: float
    is_primary: bool


class ScreenGeometry:
    """
    Centralized screen geometry provider with DPI-awareness.
    
    Resolves the coordinate space differences between mss (physical pixels),
    pyautogui (logical pixels), and VLM responses (normalized 0-1000).
    """
    
    # DPI awareness levels
    DPI_UNAWARE = 0
    SYSTEM_DPI_AWARE = 1
    PER_MONITOR_DPI_AWARE = 2
    
    def __init__(self, force_refresh: bool = True):
        self._lock = threading.Lock()
        self._monitors: list[MonitorInfo] = []
        self._primary: Optional[MonitorInfo] = None
        
        # Cached values for fast access
        self._logical_w: int = 0
        self._logical_h: int = 0
        self._physical_w: int = 0
        self._physical_h: int = 0
        self._scale: float = 1.0
        self._dpi: int = 96
        
        # Set DPI awareness FIRST, before any queries
        self._set_dpi_awareness()
        
        if force_refresh:
            self.refresh()
    
    # ═══════════════════════  DPI Setup  ═══════════════════════
    
    def _set_dpi_awareness(self):
        """
        Sets process DPI awareness to Per-Monitor V2 (best accuracy).
        Falls back gracefully through levels if the API isn't available.
        """
        try:
            # Try Per-Monitor V2 first (Windows 10 1703+)
            # SetProcessDpiAwarenessContext with DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2
            awareness_context = ctypes.c_void_p(-4)  # DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2
            result = ctypes.windll.user32.SetProcessDpiAwarenessContext(awareness_context)  # type: ignore
            if result:
                print("[ScreenGeometry] DPI: Per-Monitor V2 (best)")
                return
        except (AttributeError, OSError):
            pass
        
        try:
            # Fallback: Per-Monitor V1 (Windows 8.1+)
            ctypes.windll.shcore.SetProcessDpiAwareness(self.PER_MONITOR_DPI_AWARE)  # type: ignore
            print("[ScreenGeometry] DPI: Per-Monitor V1")
            return
        except (AttributeError, OSError):
            pass
        
        try:
            # Fallback: System DPI aware
            ctypes.windll.user32.SetProcessDPIAware()  # type: ignore
            print("[ScreenGeometry] DPI: System-level (fallback)")
        except (AttributeError, OSError):
            print("[ScreenGeometry] WARNING: Could not set DPI awareness")
    
    # ═══════════════════════  Refresh / Query  ═══════════════════════
    
    def refresh(self):
        """Re-queries all monitor geometry from the OS. Thread-safe."""
        with self._lock:
            self._query_geometry()
    
    def _query_geometry(self):
        """Internal: queries physical/logical resolution and DPI scale."""
        try:
            # --- Method 1: Direct DPI query via shcore (most accurate) ---
            try:
                # Get primary monitor handle
                hmon = ctypes.windll.user32.MonitorFromPoint(  # type: ignore
                    ctypes.wintypes.POINT(0, 0),
                    1  # MONITOR_DEFAULTTOPRIMARY
                )
                
                # Get DPI for this monitor
                dpi_x = ctypes.c_uint()
                dpi_y = ctypes.c_uint()
                ctypes.windll.shcore.GetDpiForMonitor(  # type: ignore
                    hmon, 0,  # MDT_EFFECTIVE_DPI
                    ctypes.byref(dpi_x), ctypes.byref(dpi_y)
                )
                self._dpi = dpi_x.value
                self._scale = dpi_x.value / 96.0
            except (AttributeError, OSError):
                # Fallback: query DPI from DC
                try:
                    hdc = ctypes.windll.user32.GetDC(0)  # type: ignore
                    self._dpi = ctypes.windll.gdi32.GetDeviceCaps(hdc, 88)  # LOGPIXELSX  # type: ignore
                    ctypes.windll.user32.ReleaseDC(0, hdc)  # type: ignore
                    self._scale = self._dpi / 96.0
                except Exception:
                    self._dpi = 96
                    self._scale = 1.0
            
            # --- Logical resolution (what pyautogui sees when DPI-aware) ---
            # SM_CXSCREEN / SM_CYSCREEN returns logical pixels when DPI-aware
            self._logical_w = ctypes.windll.user32.GetSystemMetrics(0)  # type: ignore
            self._logical_h = ctypes.windll.user32.GetSystemMetrics(1)  # type: ignore
            
            # --- Physical resolution (actual display pixels) ---
            # When we're Per-Monitor DPI aware, GetSystemMetrics already gives
            # physical pixels. To get the true physical resolution regardless
            # of DPI awareness, we use EnumDisplaySettings.
            try:
                devmode = ctypes.wintypes.DEV_MODE()  # type: ignore
                devmode.dmSize = ctypes.sizeof(devmode)
                # ENUM_CURRENT_SETTINGS = -1
                if ctypes.windll.user32.EnumDisplaySettingsW(None, -1, ctypes.byref(devmode)):  # type: ignore
                    self._physical_w = devmode.dmPelsWidth  # type: ignore
                    self._physical_h = devmode.dmPelsHeight  # type: ignore
                else:
                    # Fallback: assume physical = logical * scale
                    self._physical_w = int(self._logical_w * self._scale)
                    self._physical_h = int(self._logical_h * self._scale)
            except Exception:
                self._physical_w = int(self._logical_w * self._scale)
                self._physical_h = int(self._logical_h * self._scale)
            
            # Recalculate scale from actual physical/logical if both are valid
            if self._logical_w > 0 and self._physical_w > 0:
                computed_scale = self._physical_w / self._logical_w
                # Only override if significantly different (accounts for rounding)
                if abs(computed_scale - self._scale) > 0.05:
                    self._scale = computed_scale
            
            # Build primary monitor info
            self._primary = MonitorInfo(
                handle=0,
                name="PRIMARY",
                left=0, top=0,
                right=self._logical_w, bottom=self._logical_h,
                physical_width=self._physical_w,
                physical_height=self._physical_h,
                logical_width=self._logical_w,
                logical_height=self._logical_h,
                scale_factor=self._scale,
                is_primary=True,
            )
            
            print(f"[ScreenGeometry] Logical: {self._logical_w}×{self._logical_h}, "
                  f"Physical: {self._physical_w}×{self._physical_h}, "
                  f"Scale: {self._scale:.2f}x, DPI: {self._dpi}")
            
        except Exception as e:
            print(f"[ScreenGeometry] Failed to query geometry: {e}")
            # Absolute fallback
            self._logical_w = 1920
            self._logical_h = 1080
            self._physical_w = 1920
            self._physical_h = 1080
            self._scale = 1.0
            self._dpi = 96
    
    # ═══════════════════════  Properties  ═══════════════════════
    
    @property
    def logical_size(self) -> Tuple[int, int]:
        """Screen size in logical pixels (pyautogui coordinate space)."""
        return (self._logical_w, self._logical_h)
    
    @property
    def physical_size(self) -> Tuple[int, int]:
        """Screen size in physical pixels (mss capture space)."""
        return (self._physical_w, self._physical_h)
    
    @property
    def scale_factor(self) -> float:
        """DPI scale factor (1.0 = 100%, 1.25 = 125%, 1.5 = 150%, etc.)."""
        return self._scale
    
    @property
    def dpi(self) -> int:
        """Effective DPI (96 = 100%, 120 = 125%, 144 = 150%)."""
        return self._dpi
    
    @property
    def primary_monitor(self) -> Optional[MonitorInfo]:
        """Primary monitor info."""
        return self._primary
    
    # ═══════════════════════  Coordinate Conversions  ═══════════════════════
    
    def normalized_to_screen(self, nx: float, ny: float) -> Tuple[int, int]:
        """
        Converts VLM normalized coordinates (0-1000 range) to pyautogui-compatible
        logical screen coordinates.
        
        Args:
            nx: X coordinate in 0-1000 normalized space
            ny: Y coordinate in 0-1000 normalized space
            
        Returns:
            (x, y) in logical pixel coordinates suitable for pyautogui
        """
        x = int((nx / 1000.0) * self._logical_w)
        y = int((ny / 1000.0) * self._logical_h)
        return (x, y)
    
    def image_to_screen(self, img_x: int, img_y: int, 
                        img_width: int, img_height: int) -> Tuple[int, int]:
        """
        Converts pixel coordinates from a screenshot image to pyautogui-compatible
        logical screen coordinates.
        
        Handles the case where mss captures at physical resolution but pyautogui
        operates at logical resolution.
        
        Args:
            img_x: X coordinate within the screenshot image
            img_y: Y coordinate within the screenshot image
            img_width: Total width of the screenshot image
            img_height: Total height of the screenshot image
            
        Returns:
            (x, y) in logical pixel coordinates suitable for pyautogui
        """
        # The screenshot image is in physical pixel space.
        # Convert to normalized (0.0-1.0) first, then to logical pixels.
        if img_width <= 0 or img_height <= 0:
            return (img_x, img_y)
        
        norm_x = img_x / img_width
        norm_y = img_y / img_height
        
        screen_x = int(norm_x * self._logical_w)
        screen_y = int(norm_y * self._logical_h)
        
        return (screen_x, screen_y)
    
    def physical_to_logical(self, px: int, py: int) -> Tuple[int, int]:
        """Converts physical pixel coordinates to logical coordinates."""
        if self._scale <= 0:
            return (px, py)
        return (int(px / self._scale), int(py / self._scale))
    
    def logical_to_physical(self, lx: int, ly: int) -> Tuple[int, int]:
        """Converts logical coordinates to physical pixel coordinates."""
        return (int(lx * self._scale), int(ly * self._scale))
    
    def screen_to_normalized(self, x: int, y: int) -> Tuple[float, float]:
        """
        Converts pyautogui screen coordinates to VLM normalized (0-1000) space.
        Inverse of normalized_to_screen().
        """
        if self._logical_w <= 0 or self._logical_h <= 0:
            return (0.0, 0.0)
        nx = (x / self._logical_w) * 1000.0
        ny = (y / self._logical_h) * 1000.0
        return (nx, ny)
    
    def clamp_to_screen(self, x: int, y: int) -> Tuple[int, int]:
        """Clamps coordinates to within the logical screen bounds."""
        cx = max(0, min(x, self._logical_w - 1))
        cy = max(0, min(y, self._logical_h - 1))
        return (cx, cy)
    
    def is_within_screen(self, x: int, y: int, margin: int = 0) -> bool:
        """Checks if coordinates are within the logical screen bounds."""
        return (margin <= x < self._logical_w - margin and 
                margin <= y < self._logical_h - margin)
    
    # ═══════════════════════  Diagnostics  ═══════════════════════
    
    def get_diagnostics(self) -> Dict[str, Any]:
        """Returns a diagnostic summary for troubleshooting coordinate issues."""
        center_x, center_y = self.normalized_to_screen(500, 500)
        return {
            "logical_size": self.logical_size,
            "physical_size": self.physical_size,
            "scale_factor": round(self._scale, 3),
            "dpi": self._dpi,
            "center_point_logical": (center_x, center_y),
            "center_point_physical": self.logical_to_physical(center_x, center_y),
            "is_scaled": self._scale != 1.0,
        }
