import cv2 # type: ignore
import numpy as np # type: ignore
from typing import Tuple, Optional

def adapt_rect_to_edges(image: np.ndarray, x: int, y: int, w: int, h: int) -> Tuple[int, int, int, int]:
    """
    Refines a bounding box [x, y, w, h] to snap to the closest salient edges 
    detected in the image. Useful for making HUD boxes fit icons/buttons perfectly.
    """
    try:
        if image is None or image.size == 0:
            return x, y, w, h

        img_h, img_w = image.shape[:2]
        
        # 1. Expand slightly for context then crop
        margin = 20
        x1 = max(0, x - margin)
        y1 = max(0, y - margin)
        x2 = min(img_w, x + w + margin)
        y2 = min(img_h, y + h + margin)
        
        if x2 <= x1 or y2 <= y1:
            return x, y, w, h
            
        roi = image[y1:y2, x1:x2]
        
        # 2. Edge Detection
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)
        # Higher sensitivity (30, 100) to catch subtle icons/borders
        edges = cv2.Canny(blurred, 30, 100)
        
        # MORPHOLOGICAL GROUPING: Dilation merges nearby edges (e.g., fragments of an icon)
        kernel = np.ones((3,3), np.uint8)
        dilated = cv2.dilate(edges, kernel, iterations=1)
        
        # 3. Find Contours
        # Using RETR_EXTERNAL on dilated map for clean outer boundaries of icons/buttons
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return x, y, w, h
            
        # 4. Find the contour closest to the original center
        target_cx = x + w // 2 - x1
        target_cy = y + h // 2 - y1
        best_bbox = None
        min_dist = float('inf')
        
        for cnt in contours:
            bx_val, by_val, bw_val, bh_val = cv2.boundingRect(cnt)
            # Permissive noise filter: 3x3 (to catch Minimize/Ellipsis buttons)
            if bw_val < 3 or bh_val < 3 or bw_val > (w + margin*2) * 0.9:
                continue
                
            cx, cy = bx_val + bw_val // 2, by_val + bh_val // 2
            dist = ((cx - target_cx)**2 + (cy - target_cy)**2)**0.5
            
            # Weighted toward center and area consistency
            if dist < min_dist:
                min_dist = dist
                best_bbox = (bx_val, by_val, bw_val, bh_val)
                
        if best_bbox is not None:
            # Map back to global coordinates
            final_bx, final_by, final_bw, final_bh = best_bbox # type: ignore
            return (final_bx + x1, final_by + y1, final_bw, final_bh)
            
        return x, y, w, h
    except Exception as e:
        print(f"[VisionUtils] Error in adapt_rect_to_edges: {e}")
        return x, y, w, h
