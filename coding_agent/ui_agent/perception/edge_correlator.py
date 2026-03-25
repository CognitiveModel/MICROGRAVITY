"""
EdgeCorrelator — Structural edge detection, positional correlation,
and deterministic Correlational ID (CID) assignment for UI elements.

Capabilities:
  1. Canny edge maps → structural boundary signatures
  2. Edge signatures per element (gradient angles, density, corners, lines)
  3. Positional correlation graph (left_of, right_of, above, below, adjacent)
  4. Correlational IDs — deterministic hashes stable across frames
  5. VLM pre/post-indexing with CID annotations
  6. Structural diff between frames
"""

import hashlib
import time
import numpy as np # type: ignore
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple, Any, Set
from coding_agent.ui_agent.perception.edge_engine import EdgePerceptionEngine, StructuralElement # type: ignore

try:
    import cv2 # type: ignore
except ImportError:
    cv2 = None


# ──────────────────────────  Data Structures  ──────────────────────────

@dataclass
class EdgeSignature:
    """Compact structural fingerprint for a single UI element."""
    dominant_angles: List[float]    # Top 4 gradient orientations (0-180°)
    edge_density: float             # % of pixels that are edges
    corner_count: int               # Harris corner count
    line_segments: int              # Hough line segment count
    signature_hash: str = ""        # Compact hash for fast comparison

    def compute_hash(self) -> str:
        """Deterministic hash from structural features."""
        bucket_angles = [int(a / 15) * 15 for a in self.dominant_angles[:4]] # type: ignore
        density_q = round(float(self.edge_density), 2) # type: ignore
        corner_q = min(self.corner_count, 50)
        raw = f"{bucket_angles}|{density_q}|{corner_q}|{self.line_segments}"
        self.signature_hash = hashlib.sha256(raw.encode()).hexdigest()[:8] # type: ignore
        return self.signature_hash


@dataclass
class CorrelatedElement:
    """A UI element enriched with edge correlation and positional context."""
    x: int
    y: int
    width: int
    height: int
    element_type: str
    label: str
    cid: str = ""                              # Correlational ID
    edge_sig: Optional[EdgeSignature] = None
    neighbors: Dict[str, str] = field(default_factory=dict)   # {relation: neighbor_cid}
    structural_context: str = ""               # "button left_of text_input, below toolbar"
    confidence: float = 0.0


# ──────────────────────────  EdgeCorrelator  ──────────────────────────

class EdgeCorrelator:
    """
    Structural edge analysis, positional correlation, and CID assignment.

    Workflow:
      1. compute_edge_map(frame) → edge density map
      2. extract_edge_signature(crop) → per-element structural fingerprint
      3. build_positional_graph(elements) → spatial adjacency graph
      4. assign_correlational_ids(elements, graph) → deterministic CIDs
      5. pre_index_for_vlm / post_index_from_vlm → annotated VLM maps
    """

    def __init__(self, canny_low: int = 50, canny_high: int = 150,
                 adjacency_threshold: int = 60):
        self.adjacency_threshold = adjacency_threshold

        # Initialize the new unified engine
        self.engine = EdgePerceptionEngine(
            sensitivity_levels=[(canny_low, canny_high), (30, 80), (80, 200)]
        )

        # CID registry: {cid: CorrelatedElement}
        self._registry: Dict[str, CorrelatedElement] = {}
        # Frame-level cache for structural diff
        self._last_cid_set: set = set()

        print("[EdgeCorrelator] Initialized.")

    # ═══════════════════════  1. Edge Map  ═══════════════════════

    def compute_edge_map(self, frame: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Runs Canny edge detection + Sobel gradient computation.
        Returns: (edge_binary, gradient_magnitude, gradient_direction)
        """
        if frame is None:
            empty = np.zeros((1, 1), dtype=np.uint8)
            return empty, empty.astype(np.float32), empty.astype(np.float32)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame # type: ignore

        # Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (3, 3), 0) # type: ignore

        # Canny edges
        edges = cv2.Canny(blurred, self.canny_low, self.canny_high) # type: ignore

        # Sobel gradients for orientation analysis
        grad_x = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3) # type: ignore
        grad_y = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3) # type: ignore

        magnitude = np.sqrt(np.square(grad_x) + np.square(grad_y)) # type: ignore
        direction = np.degrees(np.arctan2(grad_y, grad_x)) % 180  # 0-180°

        return edges, magnitude.astype(np.float32), direction.astype(np.float32) # type: ignore

    # ═══════════════════════  2. Edge Signature  ═══════════════════════

    def extract_edge_signature(self, crop: np.ndarray) -> EdgeSignature:
        """
        Computes a structural fingerprint for a single element crop.
        """
        if crop is None or crop.size == 0:
            return EdgeSignature([], 0.0, 0, 0)

        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if len(crop.shape) == 3 else crop # type: ignore

        # Edge density
        edges = cv2.Canny(gray, self.canny_low, self.canny_high) # type: ignore
        total_pixels = max(edges.size, 1) # type: ignore
        edge_density = float(np.sum(edges > 0)) / total_pixels # type: ignore

        # Dominant gradient angles
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3) # type: ignore
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3) # type: ignore
        mag = np.sqrt(np.square(grad_x) + np.square(grad_y)) # type: ignore
        angles = np.degrees(np.arctan2(grad_y, grad_x)) % 180 # type: ignore

        # Weight angles by gradient magnitude → histogram
        angle_hist, _ = np.histogram(angles[mag > 20], bins=12, range=(0, 180), # type: ignore
                                      weights=mag[mag > 20] if np.any(mag > 20) else None)
        top_bins = np.argsort(angle_hist)[-4:][::-1] # type: ignore
        dominant_angles = [float(b * 15 + 7.5) for b in top_bins]

        # Harris corners
        try:
            corners = cv2.cornerHarris(gray.astype(np.float32), blockSize=2, ksize=3, k=0.04) # type: ignore
            corner_count = int(np.sum(corners > 0.01 * corners.max())) if corners.max() > 0 else 0 # type: ignore
        except Exception:
            corner_count = 0

        # Hough line segments
        try:
            lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=20, # type: ignore
                                     minLineLength=10, maxLineGap=5)
            line_count = len(lines) if lines is not None else 0 # type: ignore
        except Exception:
            line_count = 0

        sig = EdgeSignature(
            dominant_angles=dominant_angles,
            edge_density=round(float(edge_density), 4), # type: ignore
            corner_count=corner_count,
            line_segments=line_count,
        )
        sig.compute_hash()
        return sig

    # ═══════════════════════  3. Positional Graph  ═══════════════════════

    def build_positional_graph(self, elements: List[Dict]) -> Dict[int, Dict[str, List[int]]]:
        """
        Builds spatial adjacency graph between detected elements.
        Returns: {elem_index: {"left_of": [indices], "right_of": [...], "above": [...], "below": [...]}}
        """
        graph: Dict[int, Dict[str, List[int]]] = {}
        thresh = self.adjacency_threshold

        for i, a in enumerate(elements):
            graph[i] = {"left_of": [], "right_of": [], "above": [], "below": [],
                        "inside": [], "adjacent": []}

            ax1, ay1 = a["x"], a["y"]
            ax2, ay2 = ax1 + a["width"], ay1 + a["height"]
            acx, acy = (ax1 + ax2) // 2, (ay1 + ay2) // 2

            for j, b in enumerate(elements):
                if i == j:
                    continue

                bx1, by1 = b["x"], b["y"]
                bx2, by2 = bx1 + b["width"], by1 + b["height"]
                bcx, bcy = (bx1 + bx2) // 2, (by1 + by2) // 2

                # Containment
                if ax1 >= bx1 and ay1 >= by1 and ax2 <= bx2 and ay2 <= by2:
                    graph[i]["inside"].append(j) # type: ignore
                    continue

                # Horizontal relationship
                dx = bcx - acx
                dy = bcy - acy

                # Check adjacency (close enough to be neighbors)
                gap_x = max(0, max(bx1 - ax2, ax1 - bx2))
                gap_y = max(0, max(by1 - ay2, ay1 - by2))

                if gap_x < thresh and gap_y < thresh:
                    graph[i]["adjacent"].append(j) # type: ignore

                    # Determine primary direction
                    if abs(dx) > abs(dy):
                        if dx > 0:
                            graph[i]["left_of"].append(j) # type: ignore # A is left_of B
                        else:
                            graph[i]["right_of"].append(j) # type: ignore
                    else:
                        if dy > 0:
                            graph[i]["above"].append(j) # type: ignore # A is above B
                        else:
                            graph[i]["below"].append(j) # type: ignore

        return graph

    # ═══════════════════════  4. Correlational IDs  ═══════════════════════

    def assign_correlational_ids(
        self,
        elements: List[Dict],
        frame: np.ndarray,
        graph: Optional[Dict] = None,
    ) -> List[CorrelatedElement]:
        """
        Assigns deterministic Correlational IDs (CIDs) to each element.

        CID = sha256(edge_sig_hash + element_type + neighbor_types + relative_position)[:12]
        """
        if graph is None:
            graph = self.build_positional_graph(elements)

        correlated: List[CorrelatedElement] = []

        for i, elem in enumerate(elements):
            # Extract crop for edge signature
            x, y, w, h = elem["x"], elem["y"], elem["width"], elem["height"]
            crop = frame[y:y+h, x:x+w] if frame is not None and y+h <= frame.shape[0] and x+w <= frame.shape[1] else None # type: ignore

            edge_sig = self.extract_edge_signature(crop) if crop is not None and crop.size > 0 else EdgeSignature([], 0.0, 0, 0)

            # Collect neighbor types for CID computation
            neighbors = graph.get(i, {}) # type: ignore
            neighbor_types = []
            for rel in ["left_of", "right_of", "above", "below"]:
                for idx in neighbors.get(rel, [])[:2]:  # type: ignore # Max 2 per direction
                    if idx < len(elements):
                        neighbor_types.append(f"{rel}:{elements[idx].get('element_type', 'UNK')}") # type: ignore
            neighbor_types.sort()

            # Relative position in frame (quantized quadrant)
            frame_h, frame_w = frame.shape[:2] if frame is not None else (1, 1) # type: ignore
            qx = int((x + w / 2) / max(frame_w, 1) * 4)  # 0-3
            qy = int((y + h / 2) / max(frame_h, 1) * 4)   # 0-3

            # Build CID
            cid_raw = (
                f"{edge_sig.signature_hash}|"
                f"{elem.get('element_type', 'UNK')}|"
                f"{'_'.join(neighbor_types)}|"
                f"Q{qx}{qy}|"
                f"{min(edge_sig.corner_count, 20)}"
            )
            cid = "CID_" + hashlib.sha256(cid_raw.encode()).hexdigest()[:12] # type: ignore

            # Build structural context string
            ctx_parts = []
            for rel in ["left_of", "right_of", "above", "below", "inside"]:
                for idx in neighbors.get(rel, [])[:1]: # type: ignore
                    if idx < len(elements):
                        ctx_parts.append(f"{rel} {elements[idx].get('label', 'element')}") # type: ignore
            structural_ctx = ", ".join(ctx_parts) if ctx_parts else "standalone"

            # Build neighbor CID map (placeholder — fill after all CIDs assigned)
            neighbor_cids = {}

            ce = CorrelatedElement(
                x=x, y=y, width=w, height=h,
                element_type=elem.get("element_type", "UNKNOWN"),
                label=elem.get("label", ""),
                cid=cid,
                edge_sig=edge_sig,
                neighbors=neighbor_cids,
                structural_context=structural_ctx,
                confidence=elem.get("confidence", 0.0),
            )
            correlated.append(ce)
            self._registry[cid] = ce
            
            # Memory capping for registry (prevent bloat in long sessions)
            if len(self._registry) > 2000:
                # Remove oldest 500 entries
                old_keys = list(self._registry.keys())[:500] # type: ignore
                for k in old_keys:
                    self._registry.pop(k, None)

        # Second pass: fill neighbor CID references
        for i, ce in enumerate(correlated):
            for rel in ["left_of", "right_of", "above", "below", "inside", "adjacent"]:
                for idx in graph.get(i, {}).get(rel, [])[:2]: # type: ignore
                    if idx < len(correlated):
                        ce.neighbors[f"{rel}_{idx}"] = correlated[idx].cid # type: ignore

        # Update diff set
        new_cids = {ce.cid for ce in correlated}
        self._last_cid_set = new_cids

        return correlated

    # ═══════════════════════  5. VLM Pre/Post Indexing  ═══════════════════════

    def pre_index_for_vlm(self, correlated_elements: List[CorrelatedElement]) -> str:
        """
        Generates a compact text index for VLM consumption.
        VLM can reference elements by CID instead of describing positions.
        """
        lines = ["## Detected UI Elements (indexed by CID)\n"]
        for ce in correlated_elements:
            lines.append(
                f"- {ce.cid}: {ce.element_type} \"{ce.label}\" "
                f"at ({ce.x},{ce.y},{ce.width}x{ce.height}) "
                f"[{ce.structural_context}] "
                f"edges={ce.edge_sig.edge_density:.2f} corners={ce.edge_sig.corner_count}" # type: ignore
            )
        return "\n".join(lines)

    def post_index_from_vlm(self, vlm_response: str,
                             correlated_elements: List[CorrelatedElement]) -> List[Dict]:
        """
        Maps CIDs mentioned in VLM response back to element coordinates.
        Returns: [{cid, x, y, w, h, label, vlm_context}]
        """
        results = []
        for ce in correlated_elements:
            if ce.cid in vlm_response:
                results.append({
                    "cid": ce.cid,
                    "x": ce.x, "y": ce.y,
                    "w": ce.width, "h": ce.height,
                    "center_x": ce.x + ce.width // 2,
                    "center_y": ce.y + ce.height // 2,
                    "label": ce.label,
                    "element_type": ce.element_type,
                })
        return results

    # ═══════════════════════  6. CID Lookup  ═══════════════════════

    def find_element_by_cid(self, cid: str) -> Optional[CorrelatedElement]:
        """Retrieves a correlated element by its CID from the registry."""
        return self._registry.get(cid)

    def find_elements_by_type(self, element_type: str) -> List[CorrelatedElement]:
        """Finds all registered elements of a specific type."""
        return [ce for ce in self._registry.values() if ce.element_type == element_type]

    def find_neighbors(self, cid: str) -> Dict[str, CorrelatedElement]:
        """Returns all neighbors of a CID-identified element."""
        ce = self._registry.get(cid)
        if not ce:
            return {}
        result = {}
        for rel, ncid in ce.neighbors.items():
            neighbor = self._registry.get(ncid)
            if neighbor:
                result[rel] = neighbor
        return result

    # ═══════════════════════  7. Structural Diff  ═══════════════════════

    def diff_structural_layout(
        self,
        cids_before: set,
        cids_after: set,
    ) -> Dict[str, Any]:
        """
        Computes structural differences between two frames.
        Returns: {added: [...], removed: [...], stable: [...], change_ratio}
        """
        added = cids_after - cids_before
        removed = cids_before - cids_after
        stable = cids_before & cids_after
        total = len(cids_before | cids_after) or 1

        return {
            "added": list(added),
            "removed": list(removed),
            "stable": list(stable),
            "num_added": len(added),
            "num_removed": len(removed),
            "num_stable": len(stable),
            "change_ratio": round(float((len(added) + len(removed)) / total), 3), # type: ignore
        }

    # ═══════════════════════  8. Full Analysis  ═══════════════════════

    def detect_and_correlate(self, frame: np.ndarray) -> Dict[str, Any]:
        """
        New unified detection mode: Uses EdgePerceptionEngine to find boundaries
        then correlates them.
        """
        t0 = time.perf_counter()
        
        # 1. Structural detect
        struct_elements = self.engine.detect_structural_elements(frame)
        
        # 2. Map to Dict for build_positional_graph
        elements_for_graph = []
        for se in struct_elements:
            elements_for_graph.append({
                "x": se.x, "y": se.y, 
                "width": se.width, "height": se.height,
                "element_type": "STRUCTURAL",
                "label": "boundary",
                "confidence": se.confidence
            })
            
        # 3. Build graph
        graph = self.build_positional_graph(elements_for_graph)
        
        # 4. Assign CIDs (this also populates registry)
        old_cids = self._last_cid_set.copy()
        correlated = self.assign_correlational_ids(elements_for_graph, frame, graph)
        
        # 5. Diff
        new_cids = {ce.cid for ce in correlated}
        diff = self.diff_structural_layout(old_cids, new_cids)
        
        elapsed = (time.perf_counter() - t0) * 1000
        
        return {
            "correlated_elements": correlated,
            "structural_diff": diff,
            "processing_ms": round(float(elapsed), 2) # type: ignore
        }

    def full_correlate(self, frame: np.ndarray,
                       elements: List[Dict]) -> Dict[str, Any]:
        """
        Runs the full edge correlation pipeline:
        edge_map → signatures → graph → CIDs → VLM index.
        Returns a complete structural analysis dict.
        """
        t0 = time.perf_counter()

        # 1. Edge map
        edges, magnitude, direction = self.compute_edge_map(frame)
        edge_pixel_pct = float(np.sum(edges > 0)) / max(edges.size, 1) * 100

        # 2. Positional graph
        graph = self.build_positional_graph(elements)

        # 3. Assign CIDs
        old_cids = self._last_cid_set.copy()
        correlated = self.assign_correlational_ids(elements, frame, graph)

        # 4. VLM index
        vlm_index = self.pre_index_for_vlm(correlated)

        # 5. Structural diff
        new_cids = {ce.cid for ce in correlated}
        diff = self.diff_structural_layout(old_cids, new_cids)

        elapsed = (time.perf_counter() - t0) * 1000

        return {
            "correlated_elements": correlated,
            "total_elements": len(correlated),
            "edge_density_pct": round(float(edge_pixel_pct), 2), # type: ignore
            "positional_graph_size": len(graph),
            "vlm_index": vlm_index,
            "structural_diff": diff,
            "processing_ms": round(float(elapsed), 2), # type: ignore
        }

    def get_registry_summary(self) -> Dict[str, Any]:
        """Returns summary statistics of the CID registry."""
        type_counts: Dict[str, int] = {}
        for ce in self._registry.values():
            type_counts[ce.element_type] = type_counts.get(ce.element_type, 0) + 1

        return {
            "total_registered": len(self._registry),
            "type_distribution": type_counts,
            "cid_list": list(self._registry.keys()),
        }
