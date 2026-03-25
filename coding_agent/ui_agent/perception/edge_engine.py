"""
EdgePerceptionEngine — Advanced UI structural analysis.
Consolidates multi-threshold Canny fusion, morphological hull expansion,
and simplified polygonal boundary detection.
"""

import cv2 # type: ignore
import numpy as np # type: ignore
import hashlib
import time
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass

@dataclass
class StructuralElement:
    """A detected structural UI boundary."""
    x: int
    y: int
    width: int
    height: int
    confidence: float
    hull_complexity: int  # Number of vertices in simplified polygon
    edge_density: float
    cid: str = ""

class EdgePerceptionEngine:
    """
    Unified engine for structural UI perception through edge analysis.
    """

    def __init__(self, sensitivity_levels: Optional[List[Tuple[int, int]]] = None):
        # Default sensitivities: (low, high) thresholds for Canny
        self.sensitivities = sensitivity_levels or [(30, 100), (50, 150), (80, 200)]
        print(f"[EdgePerceptionEngine] Initialized with {len(self.sensitivities)} sensitivity levels.")

    def detect_structural_elements(self, frame: np.ndarray, 
                                   min_area: int = 25,
                                   max_area_ratio: float = 0.8) -> List[StructuralElement]:
        """
        Detects UI boundaries using multi-threshold fusion and morphological hulls.
        """
        if frame is None:
            return []

        h, w = frame.shape[:2]
        max_area = int(h * w * max_area_ratio)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)

        # 1. Multi-threshold Fusion
        # We run Canny at multiple levels and OR them to catch both subtle and strong edges
        fused_edges = np.zeros_like(blurred)
        for low, high in self.sensitivities:
            layer = cv2.Canny(blurred, low, high)
            fused_edges = cv2.bitwise_or(fused_edges, layer)

        # 2. Morphological Hull Expansion
        # Consolidate fragmented edges into solid blocks for better contouring
        kernel = np.ones((3, 3), np.uint8)
        dilated = cv2.dilate(fused_edges, kernel, iterations=1)
        closed = cv2.morphologyEx(dilated, cv2.MORPH_CLOSE, kernel, iterations=1)

        # 3. Contour Detection & Poly Approximation
        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        elements = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < min_area or area > max_area:
                continue

            # Simplify polygon to find rectangles/primitives
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.01 * peri, True)

            
            x, y, cw, ch = cv2.boundingRect(cnt)
            
            # Compute edge density within the specific bounding box on the original fused map
            crop = fused_edges[y:y+ch, x:x+cw]
            density = float(np.sum(crop > 0)) / max(crop.size, 1)

            elements.append(StructuralElement(
                x=x, y=y, width=cw, height=ch,
                confidence=min(1.0, area / (cw * ch + 1e-6)), # Shape solidity
                hull_complexity=len(approx),
                edge_density=round(float(density), 4) # type: ignore
            ))

        # Sort by area descending
        elements.sort(key=lambda e: e.width * e.height, reverse=True)
        return elements

    def compute_stable_cid(self, element: StructuralElement, frame_shape: Tuple[int, int]) -> str:
        """
        Generates a Correlational ID (CID) stable across subtle shifts.
        """
        fh, fw = frame_shape
        # Quantize position to 2% grid
        qx = int(element.x / max(fw, 1) * 50)
        qy = int(element.y / max(fh, 1) * 50)
        # Quantize dimensions to 5px buckets
        qw = int(element.width / 5) * 5
        qh = int(element.height / 5) * 5
        
        raw = f"{qx}_{qy}_{qw}_{qh}_{element.hull_complexity}_{round(float(element.edge_density), 2)}" # type: ignore
        digest = hashlib.sha256(raw.encode()).hexdigest()
        cid = "CID_" + digest[:12] # type: ignore
        element.cid = cid
        return cid

    def compute_gradient_magnitude(self, frame: np.ndarray) -> np.ndarray:
        """
        Computes a grayscale gradient magnitude map (Sobel-based).
        White on lines, black elsewhere.
        """
        if frame is None:
            return np.zeros((1, 1), dtype=np.uint8)
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)
        
        # Sobel gradients
        grad_x = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)
        
        mag = np.sqrt(np.square(grad_x) + np.square(grad_y))
        
        # Normalize to 0-255
        mag_norm = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U) # type: ignore
        return mag_norm

    def visualize_intermediate(self, frame: np.ndarray) -> Dict[str, np.ndarray]:
        """Returns internal states for debugging (fused_edges, dilated, final)."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)
        
        fused_edges = np.zeros_like(blurred)
        for low, high in self.sensitivities:
            layer = cv2.Canny(blurred, low, high)
            fused_edges = cv2.bitwise_or(fused_edges, layer)

        kernel = np.ones((3, 3), np.uint8)
        dilated = cv2.dilate(fused_edges, kernel, iterations=2)
        
        return {
            "fused": fused_edges,
            "morph": dilated
        }
