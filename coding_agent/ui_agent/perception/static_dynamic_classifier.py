"""
StaticDynamicClassifier — Temporal stability analysis for UI regions.

Captures N consecutive frames, computes per-region stability,
and classifies regions as STATIC / DYNAMIC / TRANSIENT.
Feeds into SituationalAwareness and ActionPredictor for optimized decision routing.
"""

import time
import numpy as np # type: ignore
import cv2 # type: ignore
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Callable, Tuple


@dataclass
class RegionStability:
    """Stability metadata for a single screen region."""
    bbox: Tuple[int, int, int, int]   # (x1, y1, x2, y2)
    stability_class: str               # STATIC | DYNAMIC | TRANSIENT
    confidence: float                  # 0.0-1.0
    avg_diff: float                    # Mean pixel change across frames
    frames_unchanged: int              # Consecutive frames with no change
    region_key: str                    # Grid identifier e.g. "r_3_7"


class StaticDynamicClassifier:
    """Classifies UI regions as static or dynamic based on temporal frame analysis."""

    def __init__(
        self,
        cv_pipeline,
        grid_rows: int = 8,
        grid_cols: int = 12,
        stability_threshold: int = 3,
        change_epsilon: float = 2.0,
        transient_epsilon: float = 3.0,
    ):
        """
        Args:
            cv_pipeline: CVPipeline instance for frame processing utilities.
            grid_rows: Number of vertical grid divisions.
            grid_cols: Number of horizontal grid divisions.
            stability_threshold: Frames unchanged before classifying as STATIC.
            change_epsilon: Max avg pixel diff to consider region unchanged.
            transient_epsilon: Max diff between first and last frame for TRANSIENT.
        """
        self.cv = cv_pipeline
        self.grid_rows = grid_rows
        self.grid_cols = grid_cols
        self.stability_threshold = stability_threshold
        self.change_epsilon = change_epsilon
        self.transient_epsilon = transient_epsilon

        self._frames: List[np.ndarray] = []
        self._stability_map: Dict[str, RegionStability] = {}
        self._calibrated = False
        self._frame_shape: Optional[Tuple[int, int]] = None

    # ───────────────────  Calibration  ───────────────────

    def calibrate(
        self,
        capture_func: Callable[[], Optional[np.ndarray]],
        num_frames: int = 5,
        interval: float = 0.5,
    ) -> Dict[str, RegionStability]:
        """
        Captures N frames using the provided capture function and builds
        the initial stability map.

        Args:
            capture_func: Callable that returns a BGR numpy frame (e.g. ScreenObserver.capture_as_cv2).
            num_frames: Number of frames to capture for calibration.
            interval: Seconds between captures.

        Returns:
            The computed stability map.
        """
        print(f"[StaticDynamicClassifier] Calibrating with {num_frames} frames @ {interval}s interval...")
        self._frames.clear()

        for i in range(num_frames):
            frame = capture_func()
            if frame is not None:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
                self._frames.append(gray)
                if self._frame_shape is None:
                    self._frame_shape = gray.shape[:2]
            if i < num_frames - 1:
                time.sleep(interval)

        if len(self._frames) < 2:
            print("[StaticDynamicClassifier] Warning: insufficient frames for calibration")
            return {}

        self._compute_stability_map()
        self._calibrated = True

        static_count = sum(1 for r in self._stability_map.values() if r.stability_class == "STATIC")
        dynamic_count = sum(1 for r in self._stability_map.values() if r.stability_class == "DYNAMIC")
        transient_count = sum(1 for r in self._stability_map.values() if r.stability_class == "TRANSIENT")

        print(f"[StaticDynamicClassifier] Calibrated: {static_count} STATIC, {dynamic_count} DYNAMIC, {transient_count} TRANSIENT regions")
        return self._stability_map

    # ───────────────────  Stability Map  ───────────────────

    def _compute_stability_map(self):
        """Computes stability classification for each grid cell using frame history."""
        shape = self._frame_shape
        if not self._frames or shape is None:
            return

        h = shape[0] # type: ignore
        w = shape[1] # type: ignore
        cell_h = h // max(1, self.grid_rows) # type: ignore
        cell_w = w // max(1, self.grid_cols) # type: ignore

        self._stability_map.clear()

        for gy in range(self.grid_rows):
            for gx in range(self.grid_cols):
                y1 = gy * cell_h # type: ignore
                y2 = min((gy + 1) * cell_h, h) # type: ignore
                x1 = gx * cell_w # type: ignore
                x2 = min((gx + 1) * cell_w, w) # type: ignore
                region_key = f"r_{gy}_{gx}"

                # Compute pairwise diffs
                diffs = [] # type: ignore
                frames_unchanged = 0

                for i in range(1, len(self._frames)):
                    r_curr = self._frames[i][y1:y2, x1:x2]
                    r_prev = self._frames[i - 1][y1:y2, x1:x2]
                    diff_val = float(cv2.absdiff(r_curr, r_prev).mean())
                    diffs.append(diff_val)
                    if diff_val < self.change_epsilon:
                        frames_unchanged += 1 # type: ignore

                avg_diff = float(np.mean(diffs)) if diffs else 0.0

                # Classify
                if frames_unchanged >= self.stability_threshold - 1: # type: ignore
                    cls = "STATIC"
                    conf = min(1.0, frames_unchanged / max(len(diffs), 1)) # type: ignore
                elif len(self._frames) >= 3:
                    # Check transient: first ≈ last but middle differs
                    r_first = self._frames[0][y1:y2, x1:x2]
                    r_last = self._frames[-1][y1:y2, x1:x2]
                    bounce_diff = float(cv2.absdiff(r_first, r_last).mean())
                    if bounce_diff < self.transient_epsilon and avg_diff > self.change_epsilon:
                        cls = "TRANSIENT"
                        conf = 0.6
                    else:
                        cls = "DYNAMIC"
                        conf = min(1.0, avg_diff / 50.0)
                else:
                    cls = "DYNAMIC"
                    conf = 0.5

                self._stability_map[region_key] = RegionStability(
                    bbox=(x1, y1, x2, y2),
                    stability_class=cls,
                    confidence=float(f"{conf:.3f}"), # type: ignore
                    avg_diff=float(f"{avg_diff:.2f}"), # type: ignore
                    frames_unchanged=frames_unchanged,
                    region_key=region_key,
                )

    # ───────────────────  Queries  ───────────────────

    def get_stability_map(self) -> Dict[str, RegionStability]:
        """Returns the full stability map. Call calibrate() first."""
        return dict(self._stability_map)

    def is_region_static(self, bbox: Tuple[int, int, int, int]) -> bool:
        """
        Quick check: is the given bounding box within a STATIC region?
        Checks all grid cells that overlap with the bbox.
        """
        shape = self._frame_shape
        if not self._stability_map or shape is None:
            return False

        h = shape[0] # type: ignore
        w = shape[1] # type: ignore
        cell_h = h // max(1, self.grid_rows) # type: ignore
        cell_w = w // max(1, self.grid_cols) # type: ignore

        x1, y1, x2, y2 = bbox
        # Find overlapping grid cells
        gy_start = max(0, y1 // cell_h)
        gy_end = min(self.grid_rows - 1, y2 // cell_h)
        gx_start = max(0, x1 // cell_w)
        gx_end = min(self.grid_cols - 1, x2 // cell_w)

        for gy in range(gy_start, gy_end + 1):
            for gx in range(gx_start, gx_end + 1):
                key = f"r_{gy}_{gx}"
                region = self._stability_map.get(key)
                if region and region.stability_class != "STATIC":
                    return False

        return True

    def get_region_class(self, x: int, y: int) -> str:
        """Returns the stability class for a specific screen point."""
        shape = self._frame_shape
        if not self._stability_map or shape is None:
            return "UNKNOWN"

        h = shape[0] # type: ignore
        w = shape[1] # type: ignore
        cell_h = h // max(1, self.grid_rows) # type: ignore
        cell_w = w // max(1, self.grid_cols) # type: ignore

        gy = min(y // cell_h, self.grid_rows - 1)
        gx = min(x // cell_w, self.grid_cols - 1)

        key = f"r_{gy}_{gx}"
        region = self._stability_map.get(key)
        return region.stability_class if region else "UNKNOWN"

    # ───────────────────  Incremental Update  ───────────────────

    def update(self, new_frame: np.ndarray):
        """
        Incrementally updates the stability map with a new frame.
        Maintains a rolling buffer and recomputes classifications.
        """
        gray = cv2.cvtColor(new_frame, cv2.COLOR_BGR2GRAY) if len(new_frame.shape) == 3 else new_frame

        self._frames.append(gray)
        # Keep rolling window
        max_frames = max(self.stability_threshold + 2, 5)
        if len(self._frames) > max_frames:
            self._frames.pop(0)

        if self._frame_shape is None:
            self._frame_shape = gray.shape[:2]

        if len(self._frames) >= 2:
            self._compute_stability_map()

    def reset(self):
        """Resets the classifier (e.g. after window switch)."""
        self._frames.clear()
        self._stability_map.clear()
        self._calibrated = False
        self._frame_shape = None
        print("[StaticDynamicClassifier] Reset")

    @property
    def is_calibrated(self) -> bool:
        return self._calibrated

    def get_summary(self) -> str:
        """Returns a concise text summary for injection into planner context."""
        if not self._stability_map:
            return "Stability: Not calibrated"

        static = sum(1 for r in self._stability_map.values() if r.stability_class == "STATIC")
        dynamic = sum(1 for r in self._stability_map.values() if r.stability_class == "DYNAMIC")
        transient = sum(1 for r in self._stability_map.values() if r.stability_class == "TRANSIENT")
        total = static + dynamic + transient

        return f"Stability: {static}/{total} STATIC, {dynamic} DYNAMIC, {transient} TRANSIENT"
