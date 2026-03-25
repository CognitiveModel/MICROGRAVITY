"""
CVPipeline — Central OpenCV processing pipeline for the UI Agent.

Provides 6 analysis modes + embedding-based similarity search:
  1. Multi-Scale Template Matching (3 scales)
  2. Feature-Based Matching (ORB + BFMatcher)
  3. Contour-Based UI Element Detection (adaptive threshold + NMS)
  4. Text Region Detection (MSER)
  5. Color Histogram Fingerprinting
  6. Frame Differencing Engine (rolling 3-frame buffer)
  + Embedding-Based Similarity Search (60-dim vectors)
"""

import cv2 # type: ignore
import numpy as np # type: ignore
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict, Any
from coding_agent.ui_agent.perception.cv_logger import CVLogger # type: ignore
from coding_agent.ui_agent.perception.edge_engine import EdgePerceptionEngine # type: ignore


# ──────────────────────────  Data Structures  ──────────────────────────

@dataclass
class UIElement:
    """A single detected UI element with classification metadata."""
    x: int
    y: int
    width: int
    height: int
    element_type: str       # BUTTON, ICON, TEXT_INPUT, STRUCTURAL, PANEL, TEXT_REGION, DYNAMIC, etc.
    label: str              # Human-readable label
    confidence: float = 1.0
    color: str = "#00ff00"  # Display color for HUD overlay
    detection_method: str = "cv_contour"
    fingerprint: Optional[np.ndarray] = field(default=None, repr=False)
    embedding: Optional[np.ndarray] = field(default=None, repr=False)

    def center(self) -> Tuple[int, int]:
        return (self.x + self.width // 2, self.y + self.height // 2)

    def rect(self) -> Tuple[int, int, int, int]:
        return (self.x, self.y, self.x + self.width, self.y + self.height)

    def area(self) -> int:
        return self.width * self.height


@dataclass
class PerceptionResult:
    """Unified output from a full CVPipeline analysis pass."""
    elements: List[UIElement]
    text_regions: List[Dict[str, Any]]
    stability_classes: Dict[str, str]   # region_hash → STATIC|DYNAMIC|TRANSIENT
    dynamic_regions: List[Dict[str, Any]]
    frame_shape: Tuple[int, int]        # (height, width)
    analysis_time_ms: float = 0.0


# ──────────────────────────  Pipeline  ──────────────────────────

class CVPipeline:
    """Central OpenCV processing pipeline for the UI Agent."""

    # ── Construction ──

    def __init__(self, config: Optional[Dict] = None, cv_logger: Optional[CVLogger] = None):
        config = config or {}

        # Structured logger
        self.logger = cv_logger or CVLogger()

        # ORB feature detector
        self.orb = cv2.ORB_create(nfeatures=config.get("orb_features", 500))
        self.bf_matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

        # MSER for text regions
        self.mser = cv2.MSER_create(
            delta=config.get("mser_delta", 5),
            min_area=config.get("mser_min_area", 60),
            max_area=config.get("mser_max_area", 14400),
        )

        # Multi-scale factors
        self.scales = config.get("scales", [0.8, 1.0, 1.2])

        # Rolling frame buffer for temporal analysis
        self._frame_buffer: List[np.ndarray] = []
        self._max_buffer = config.get("frame_buffer_size", 3)

        # Embedding dimension breakdown
        self._orb_hist_dim = 32
        self._color_hist_dim = 24  # 8 bins × 3 channels
        self._shape_dim = 4        # aspect_ratio, area_ratio, corner_count, edge_density
        self.embedding_dim = self._orb_hist_dim + self._color_hist_dim + self._shape_dim

        # Structural Edge Engine
        self.edge_engine = EdgePerceptionEngine()

        print("[CVPipeline] Initialized with ORB, MSER, multi-scale, embedding support + CVLogger + EdgeEngine")

    # ═══════════════════════  1. Multi-Scale Template Matching  ═══════════════════════

    def match_template_multiscale(
        self,
        screen: np.ndarray,
        template: np.ndarray,
        threshold: float = 0.80,
        target_label: str = "unknown",
        mode: str = "PASSIVE",
    ) -> Optional[Dict[str, Any]]:
        """
        Searches for *template* in *screen* at multiple scales.
        Returns {x, y, w, h, confidence, scale} or None.
        """
        import time as _time
        _t0 = _time.perf_counter()

        if screen is None or template is None:
            return None

        screen_gray = cv2.cvtColor(screen, cv2.COLOR_BGR2GRAY) if len(screen.shape) == 3 else screen # type: ignore
        tpl_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY) if len(template.shape) == 3 else template # type: ignore
        th, tw = tpl_gray.shape[:2] # type: ignore

        best = None

        for scale in self.scales:
            # Resize template to simulate zoom differences
            new_w = max(1, int(tw * scale))
            new_h = max(1, int(th * scale))
            if new_w > screen_gray.shape[1] or new_h > screen_gray.shape[0]: # type: ignore
                continue

            resized = cv2.resize(tpl_gray, (new_w, new_h), interpolation=cv2.INTER_AREA)

            try:
                result = cv2.matchTemplate(screen_gray, resized, cv2.TM_CCOEFF_NORMED)
            except cv2.error:
                continue

            _, max_val, _, max_loc = cv2.minMaxLoc(result)

            if float(max_val) >= threshold and (best is None or float(max_val) > float(best["confidence"])): # type: ignore
                best = {
                    "x": max_loc[0],
                    "y": max_loc[1],
                    "w": new_w,
                    "h": new_h,
                    "confidence": float(f"{float(max_val):.4f}"), # type: ignore
                    "scale": scale,
                }

        _elapsed = (_time.perf_counter() - _t0) * 1000
        self.logger.log_template_match(
            target=target_label,
            scale=best["scale"] if best else 0, # type: ignore
            confidence=best["confidence"] if best else 0, # type: ignore
            coords=[best["x"], best["y"]] if best else None, # type: ignore
            matched=best is not None,
            latency_ms=_elapsed,
            mode=mode,
        )
        return best

    # ═══════════════════════  2. Feature-Based Matching (ORB)  ═══════════════════════

    def match_features_orb(
        self,
        screen: np.ndarray,
        template: np.ndarray,
        min_matches: int = 10,
        target_label: str = "unknown",
        mode: str = "PASSIVE",
    ) -> Optional[Dict[str, Any]]:
        """
        ORB keypoint matching.  Returns {center_x, center_y, confidence, num_matches,
        homography} or None.
        """
        import time as _time
        _t0 = _time.perf_counter()

        if screen is None or template is None:
            return None

        screen_gray = cv2.cvtColor(screen, cv2.COLOR_BGR2GRAY) if len(screen.shape) == 3 else screen
        tpl_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY) if len(template.shape) == 3 else template

        kp1, des1 = self.orb.detectAndCompute(tpl_gray, None)
        kp2, des2 = self.orb.detectAndCompute(screen_gray, None)

        num_kp = len(kp2) if kp2 else 0

        if des1 is None or des2 is None or len(des1) < 4 or len(des2) < 4:
            _elapsed = (_time.perf_counter() - _t0) * 1000
            self.logger.log_orb_match(target_label, num_kp, 0, 0.0, False, _elapsed, mode)
            return None

        matches = self.bf_matcher.match(des1, des2)
        matches = sorted(matches, key=lambda m: m.distance)

        if len(matches) < min_matches:
            _elapsed = (_time.perf_counter() - _t0) * 1000
            self.logger.log_orb_match(target_label, num_kp, len(matches), 0.0, False, _elapsed, mode)
            return None

        # Compute homography for robust center estimation
        src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2) # type: ignore
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2) # type: ignore

        H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        if H is None:
            _elapsed = (_time.perf_counter() - _t0) * 1000
            self.logger.log_orb_match(target_label, num_kp, len(matches), 0.0, False, _elapsed, mode)
            return None

        inliers = int(mask.sum()) if mask is not None else len(matches) # type: ignore

        # Project template center through homography
        th, tw = tpl_gray.shape[:2]
        tpl_center = np.float32([[tw / 2, th / 2]]).reshape(-1, 1, 2)
        projected = cv2.perspectiveTransform(tpl_center, H)
        cx, cy = projected[0][0]

        confidence = inliers / max(len(matches), 1)

        _elapsed = (_time.perf_counter() - _t0) * 1000
        self.logger.log_orb_match(
            target_label, num_kp, len(matches),
            round(float(confidence), 4), True, _elapsed, mode, # type: ignore
        )

        return {
            "center_x": int(cx),
            "center_y": int(cy),
            "confidence": float(f"{float(confidence):.4f}"), # type: ignore
            "num_matches": len(matches),
            "inliers": inliers,
            "homography": H,
        }

    # ═══════════════════════  3. Contour-Based UI Element Detection  ═══════════════════════

    def detect_ui_elements(
        self,
        frame: np.ndarray,
        previous_frame: Optional[np.ndarray] = None,
        min_area: int = 200,
        max_area_ratio: float = 0.5,
        include_structural: bool = True,
    ) -> List[UIElement]:
        """
        Detects rectangular UI elements using adaptive thresholding,
        contour analysis, and NMS.  Classifies by aspect ratio heuristics.
        If include_structural is True, also runs the EdgePerceptionEngine.
        """
        if frame is None:
            return []

        h, w = frame.shape[:2]
        max_area = int(h * w * max_area_ratio)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame

        # Adaptive thresholding
        binary = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 15, 4
        )

        # Morphological close to merge nearby edges
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=1)

        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        raw_elements: List[UIElement] = []

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < min_area or area > max_area:
                continue

            x, y, cw, ch = cv2.boundingRect(cnt)
            aspect = cw / max(ch, 1)

            # Classify by shape heuristics
            if 1.5 <= aspect <= 6.0 and ch < 60:
                etype, label, color = "BUTTON", "BUTTON", "#00ff88"
            elif 0.7 <= aspect <= 1.3 and cw < 80:
                etype, label, color = "ICON", "ICON", "#ffaa00"
            elif aspect > 6.0 and ch < 50:
                etype, label, color = "TEXT_INPUT", "TEXT_FIELD", "#00ccff"
            elif area > (h * w * 0.05):
                etype, label, color = "PANEL", "PANEL", "#444444"
            else:
                etype, label, color = "STRUCTURAL", "STRUCTURAL", "#888888"

            raw_elements.append(UIElement(
                x=x, y=y, width=cw, height=ch,
                element_type=etype, label=label, color=color,
                detection_method="cv_contour",
            ))

        # Add structural boundaries from EdgePerceptionEngine
        if include_structural:
            struct_elements = self.edge_engine.detect_structural_elements(frame, min_area=min_area)
            for se in struct_elements:
                raw_elements.append(UIElement(
                    x=se.x, y=se.y, width=se.width, height=se.height,
                    element_type="STRUCTURAL", label="BOUNDARY", color="#ff00ff",
                    confidence=se.confidence,
                    detection_method="edge_structural"
                ))

        # Non-Maximum Suppression
        elements = self._nms(raw_elements, iou_threshold=0.45)

        # Dynamic region detection (frame differencing)
        if previous_frame is not None:
            dynamic = self._detect_dynamic_regions(frame, previous_frame)
            elements.extend(dynamic)

        return elements

    def _nms(self, elements: List[UIElement], iou_threshold: float = 0.45) -> List[UIElement]:
        """Non-Maximum Suppression on detected elements."""
        if not elements:
            return []

        boxes = np.array([[e.x, e.y, e.x + e.width, e.y + e.height] for e in elements], dtype=np.float32)
        scores = np.array([e.confidence for e in elements], dtype=np.float32)

        x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        areas = (x2 - x1) * (y2 - y1)
        order = scores.argsort()[::-1]

        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)

            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            inter = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
            iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)

            inds = np.where(iou <= iou_threshold)[0]
            order = order[inds + 1]

        return [elements[i] for i in keep]

    def _detect_dynamic_regions(
        self,
        current: np.ndarray,
        previous: np.ndarray,
        change_threshold: int = 30,
        min_area: int = 400,
    ) -> List[UIElement]:
        """Detects regions that changed between two frames."""
        gray_curr = cv2.cvtColor(current, cv2.COLOR_BGR2GRAY) if len(current.shape) == 3 else current
        gray_prev = cv2.cvtColor(previous, cv2.COLOR_BGR2GRAY) if len(previous.shape) == 3 else previous

        if gray_curr.shape != gray_prev.shape:
            return []

        diff = cv2.absdiff(gray_curr, gray_prev)
        _, thresh = cv2.threshold(diff, change_threshold, 255, cv2.THRESH_BINARY)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)

        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        regions = []
        for cnt in contours:
            if cv2.contourArea(cnt) < min_area:
                continue
            x, y, w, h = cv2.boundingRect(cnt)
            regions.append(UIElement(
                x=x, y=y, width=w, height=h,
                element_type="DYNAMIC", label="DYNAMIC_REGION",
                color="#ff0000", detection_method="frame_diff",
            ))
        return regions

    # ═══════════════════════  4. Text Region Detection (MSER)  ═══════════════════════

    def detect_text_regions(self, frame: np.ndarray, merge_distance: int = 15) -> List[Dict[str, Any]]:
        """
        Detects text-containing bounding boxes using MSER.
        Returns list of {x, y, w, h, area, confidence}.
        """
        if frame is None:
            return []

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame

        try:
            regions, _ = self.mser.detectRegions(gray)
        except cv2.error:
            return []

        if not regions:
            return []

        # Convert individual MSER regions to bounding boxes
        raw_boxes = []
        for region in regions:
            x, y, w, h = cv2.boundingRect(region)
            # Text regions are typically small height, moderate width
            if 5 < h < 60 and w > 5:
                raw_boxes.append([x, y, x + w, y + h])

        if not raw_boxes:
            return []

        # Merge nearby boxes (text characters into words/lines)
        merged = self._merge_text_boxes(np.array(raw_boxes), merge_distance)

        results = []
        for box in merged:
            x1, y1, x2, y2 = box
            results.append({
                "x": int(x1), "y": int(y1),
                "w": int(x2 - x1), "h": int(y2 - y1),
                "area": int((x2 - x1) * (y2 - y1)),
                "confidence": 0.7,
            })
        return results

    def _merge_text_boxes(self, boxes: np.ndarray, distance: int) -> List[List[int]]:
        """Merges nearby text bounding boxes into word/line regions."""
        if len(boxes) == 0:
            return []

        # Sort by x coordinate
        boxes = boxes[boxes[:, 0].argsort()]
        merged = [boxes[0].tolist()]

        for box in boxes[1:]:
            last = merged[-1]
            # Check if horizontally close and vertically overlapping
            if (box[0] <= last[2] + distance and
                abs(box[1] - last[1]) < distance and
                abs(box[3] - last[3]) < distance):
                # Merge
                last[0] = min(last[0], box[0])
                last[1] = min(last[1], box[1])
                last[2] = max(last[2], box[2])
                last[3] = max(last[3], box[3])
            else:
                merged.append(box.tolist())

        return merged

    # ═══════════════════════  5. Color Histogram Fingerprinting  ═══════════════════════

    def fingerprint_element(self, element_crop: np.ndarray, bins: int = 8) -> Optional[np.ndarray]:
        """
        Computes a compact color histogram signature for an element crop.
        Returns a normalized 24-dim vector (8 bins × 3 channels).
        """
        if element_crop is None or element_crop.size == 0:
            return None

        if len(element_crop.shape) < 3:
            element_crop = cv2.cvtColor(element_crop, cv2.COLOR_GRAY2BGR)

        hist_b = cv2.calcHist([element_crop], [0], None, [bins], [0, 256]).flatten()
        hist_g = cv2.calcHist([element_crop], [1], None, [bins], [0, 256]).flatten()
        hist_r = cv2.calcHist([element_crop], [2], None, [bins], [0, 256]).flatten()

        combined = np.concatenate([hist_b, hist_g, hist_r])
        norm = np.linalg.norm(combined)
        return combined / norm if norm > 0 else combined

    def compare_fingerprints(self, fp1: np.ndarray, fp2: np.ndarray) -> float:
        """Cosine similarity between two fingerprints. Returns 0.0-1.0."""
        if fp1 is None or fp2 is None:
            return 0.0
        dot = np.dot(fp1, fp2)
        norm = np.linalg.norm(fp1) * np.linalg.norm(fp2)
        return float(dot / norm) if norm > 0 else 0.0

    # ═══════════════════════  6. Frame Differencing Engine  ═══════════════════════

    def classify_regions(self, current_frame: np.ndarray) -> Dict[str, str]:
        """
        Using rolling frame buffer, classifies screen regions as:
        - STATIC:    No change across N frames
        - DYNAMIC:   Changed between frames
        - TRANSIENT: Changed then returned to original (animation/blink)

        Returns {region_hash: classification_string}.
        """
        gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY) if len(current_frame.shape) == 3 else current_frame

        # Update buffer
        self._frame_buffer.append(gray.copy())
        if len(self._frame_buffer) > self._max_buffer:
            self._frame_buffer.pop(0)

        if len(self._frame_buffer) < 2:
            return {}

        classifications = {}

        # Divide screen into a grid of regions
        h, w = gray.shape
        grid_h, grid_w = 8, 12
        cell_h, cell_w = h // grid_h, w // grid_w

        for gy in range(grid_h):
            for gx in range(grid_w):
                y1, y2 = gy * cell_h, (gy + 1) * cell_h
                x1, x2 = gx * cell_w, (gx + 1) * cell_w
                region_key = f"r_{gy}_{gx}"

                # Compare across buffer
                diffs = []
                for i in range(1, len(self._frame_buffer)):
                    r_curr = self._frame_buffer[i][y1:y2, x1:x2]
                    r_prev = self._frame_buffer[i - 1][y1:y2, x1:x2]
                    diff_val = float(cv2.absdiff(r_curr, r_prev).mean())
                    diffs.append(diff_val)

                avg_diff = np.mean(diffs)

                if avg_diff < 2.0:
                    classifications[region_key] = "STATIC"
                elif len(self._frame_buffer) >= 3:
                    # Check for transient: first and last frame similar, but middle different
                    r_first = self._frame_buffer[0][y1:y2, x1:x2]
                    r_last = self._frame_buffer[-1][y1:y2, x1:x2]
                    bounce_diff = float(cv2.absdiff(r_first, r_last).mean())
                    if bounce_diff < 3.0 and avg_diff > 5.0:
                        classifications[region_key] = "TRANSIENT"
                    else:
                        classifications[region_key] = "DYNAMIC"
                else:
                    classifications[region_key] = "DYNAMIC"

        return classifications

    def clear_frame_buffer(self):
        """Clears the rolling frame buffer (e.g. after window switch)."""
        self._frame_buffer.clear()

    # ═══════════════════════  Embedding-Based Similarity Search  ═══════════════════════

    def build_element_embedding(self, element_crop: np.ndarray, frame: Optional[np.ndarray] = None) -> Optional[np.ndarray]:
        """
        Computes a 60-dim feature vector for a UI element:
          - ORB descriptor histogram (32-dim)
          - Color histogram (24-dim)
          - Shape features (4-dim: aspect_ratio, area_ratio, corner_count, edge_density)
        """
        if element_crop is None or element_crop.size == 0:
            return None

        h, w = element_crop.shape[:2]
        if h < 5 or w < 5:
            return None

        # 1. ORB descriptor histogram (32-dim)
        gray = cv2.cvtColor(element_crop, cv2.COLOR_BGR2GRAY) if len(element_crop.shape) == 3 else element_crop
        _, des = self.orb.detectAndCompute(gray, None)
        if des is not None and len(des) > 0: # type: ignore
            # Each ORB descriptor is 32 bytes; compute a histogram of byte values
            orb_hist = np.zeros(self._orb_hist_dim, dtype=np.float32)
            for d in des:
                bin_idx = np.clip(d[:self._orb_hist_dim], 0, 255)
                for val in bin_idx:
                    orb_hist[val % self._orb_hist_dim] += 1
            orb_norm = np.linalg.norm(orb_hist)
            orb_hist = orb_hist / orb_norm if orb_norm > 0 else orb_hist
        else:
            orb_hist = np.zeros(self._orb_hist_dim, dtype=np.float32)

        # 2. Color histogram (24-dim)
        color_hist = self.fingerprint_element(element_crop, bins=8)
        if color_hist is None:
            color_hist = np.zeros(self._color_hist_dim, dtype=np.float32)

        # 3. Shape features (4-dim)
        aspect_ratio = w / max(h, 1)
        area_ratio = (w * h) / max(1, frame.shape[0] * frame.shape[1]) if frame is not None else 0.0
        # Corner count (Harris)
        corners = cv2.cornerHarris(gray, 2, 3, 0.04)
        corner_count = float(np.sum(corners > 0.01 * corners.max())) / max(w * h, 1)
        # Edge density
        edges = cv2.Canny(gray, 50, 150)
        edge_density = float(np.sum(edges > 0)) / max(w * h, 1)

        shape_features = np.array([aspect_ratio, area_ratio, corner_count, edge_density], dtype=np.float32)

        embedding = np.concatenate([orb_hist, color_hist, shape_features]) # type: ignore
        return embedding

    def build_element_embeddings(self, elements: List[UIElement], frame: np.ndarray) -> Dict[str, np.ndarray]:
        """Computes embeddings for a list of detected UIElements."""
        embeddings = {}
        for i, el in enumerate(elements):
            crop = frame[el.y:el.y + el.height, el.x:el.x + el.width]
            emb = self.build_element_embedding(crop, frame)
            if emb is not None:
                key = f"{el.element_type}_{i}_{el.x}_{el.y}"
                embeddings[key] = emb
                el.embedding = emb
        return embeddings

    def similarity_search(
        self,
        query_embedding: np.ndarray,
        candidate_embeddings: Dict[str, np.ndarray],
        top_k: int = 5,
    ) -> List[Tuple[str, float]]:
        """
        Cosine similarity search through candidate embeddings.
        Returns top-K (key, score) pairs sorted by descending similarity.
        """
        if query_embedding is None or not candidate_embeddings:
            return []

        q_norm = np.linalg.norm(query_embedding)
        if q_norm == 0:
            return []
        q = query_embedding / q_norm

        results = []
        for key, emb in candidate_embeddings.items():
            e_norm = np.linalg.norm(emb)
            if e_norm == 0:
                continue
            score = float(np.dot(q, emb / e_norm))
            results.append((key, score))

        results.sort(key=lambda x: x[1], reverse=True)
        return list(results)[:top_k] # type: ignore

    def find_typical_positions(self, elements_history: List[UIElement]) -> Dict[str, Dict]:
        """
        Analyzes stored element positions across sessions.
        Returns statistical typical position for each element type.
        """
        from collections import defaultdict
        positions = defaultdict(list)
        for el in elements_history:
            positions[el.element_type].append((el.x, el.y, el.width, el.height))

        result = {}
        for etype, pos_list in positions.items():
            xs = [p[0] for p in pos_list]
            ys = [p[1] for p in pos_list]
            ws = [p[2] for p in pos_list]
            hs = [p[3] for p in pos_list]
            result[etype] = {
                "mean_x": int(np.mean(xs)),
                "mean_y": int(np.mean(ys)),
                "mean_w": int(np.mean(ws)),
                "mean_h": int(np.mean(hs)),
                "std_x": float(np.std(xs)),
                "std_y": float(np.std(ys)),
                "count": len(pos_list),
            }
        return result

    # ═══════════════════════  Full Analysis  ═══════════════════════

    def full_analysis(
        self,
        current_frame: np.ndarray,
        previous_frame: Optional[np.ndarray] = None,
    ) -> PerceptionResult:
        """
        Runs all detectors and returns a unified PerceptionResult.
        Intended to be called once per observation cycle.
        """
        import time
        t0 = time.perf_counter()

        # 1. UI Element detection
        elements = self.detect_ui_elements(current_frame, previous_frame)

        # 2. Text regions
        t_text = time.perf_counter()
        text_regions = self.detect_text_regions(current_frame)
        text_elapsed = (time.perf_counter() - t_text) * 1000
        total_text_area = sum(r.get("area", 0) for r in text_regions)
        frame_area = current_frame.shape[0] * current_frame.shape[1]
        self.logger.log_text_region(
            num_regions=len(text_regions),
            total_area_pct=(total_text_area / max(frame_area, 1)) * 100,
            latency_ms=text_elapsed,
        )

        # 3. Stability classification
        t_stab = time.perf_counter()
        stability = self.classify_regions(current_frame)
        stab_elapsed = (time.perf_counter() - t_stab) * 1000
        n_static = sum(1 for v in stability.values() if v == "STATIC")
        n_dynamic = sum(1 for v in stability.values() if v == "DYNAMIC")
        n_transient = sum(1 for v in stability.values() if v == "TRANSIENT")
        self.logger.log_stability_scan(n_static, n_dynamic, n_transient, stab_elapsed)

        # 4. Extract dynamic regions from stability
        dynamic = [
            {"region": k, "class": v}
            for k, v in stability.items()
            if v in ("DYNAMIC", "TRANSIENT")
        ]

        elapsed = (time.perf_counter() - t0) * 1000
        print(f"[CVPipeline] full_analysis: {len(elements)} elements, "
              f"{len(text_regions)} text regions, "
              f"{n_static}S/{n_dynamic}D/{n_transient}T stability, "
              f"{elapsed:.1f}ms total")

        return PerceptionResult(
            elements=elements,
            text_regions=text_regions,
            stability_classes=stability,
            dynamic_regions=dynamic,
            frame_shape=(current_frame.shape[0], current_frame.shape[1]), # type: ignore
            analysis_time_ms=float(f"{elapsed:.2f}"), # type: ignore
        )
