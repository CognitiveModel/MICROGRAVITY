from typing import Tuple, Dict

class ROIManager:
    """
    Handles calculation of Regions of Interest (ROI) and coordinate 
    mapping between cropped/magnified views and global screen space.
    """
    
    @staticmethod
    def calculate_roi(center_x: int, center_y: int, zoom_factor: float, screen_size: Tuple[int, int]) -> Tuple[int, int, int, int]:
        """
        Calculates the bounding box (x1, y1, x2, y2) for a magnified ROI.
        """
        screen_w, screen_h = screen_size
        
        # Width and height of the crop area
        crop_w = int(screen_w / zoom_factor)
        crop_h = int(screen_h / zoom_factor)
        
        # Initial top-left based on center
        x1 = center_x - (crop_w // 2)
        y1 = center_y - (crop_h // 2)
        
        # Clamp to screen boundaries
        x1 = max(0, min(x1, screen_w - crop_w))
        y1 = max(0, min(y1, screen_h - crop_h))
        
        x2 = x1 + crop_w
        y2 = y1 + crop_h
        
        return (x1, y1, x2, y2)

    @staticmethod
    def map_to_global(nx: float, ny: float, roi: Tuple[int, int, int, int], screen_size: Tuple[int, int]) -> Tuple[int, int]:
        """
        Maps normalized coordinates (0.0-1.0) from a cropped ROI back to global pixel coordinates.
        nx, ny: normalized coordinates within the ROI [0.0, 1.0]
        roi: (x1, y1, x2, y2) global pixel bounds of the crop
        """
        x1, y1, x2, y2 = roi
        roi_w = x2 - x1
        roi_h = y2 - y1
        
        global_x = x1 + int(nx * roi_w)
        global_y = y1 + int(ny * roi_h)
        
        return (global_x, global_y)

    @staticmethod
    def normalize_to_roi(gx: int, gy: int, roi: Tuple[int, int, int, int]) -> Tuple[float, float]:
        """
        Maps global pixels to normalized (0.0-1.0) coordinates within an ROI.
        """
        x1, y1, x2, y2 = roi
        roi_w = x2 - x1
        roi_h = y2 - y1
        
        nx = (gx - x1) / roi_w
        ny = (gy - y1) / roi_h
        
        return (nx, ny)
