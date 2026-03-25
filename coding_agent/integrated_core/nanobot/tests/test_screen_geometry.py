"""
Tests for ScreenGeometry — centralized DPI-aware screen geometry provider.
"""
import pytest
import sys
import os

# Add project root to sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestScreenGeometry:
    """Test suite for ScreenGeometry coordinate math and DPI handling."""

    @pytest.fixture(autouse=True)
    def setup(self):
        from nanobot.agent.ui.utils.screen_geometry import ScreenGeometry
        self.geo = ScreenGeometry(force_refresh=True)

    def test_logical_size_is_valid(self):
        """Logical screen size should be positive integers."""
        w, h = self.geo.logical_size
        assert w > 0, f"Logical width should be > 0, got {w}"
        assert h > 0, f"Logical height should be > 0, got {h}"
        assert isinstance(w, int)
        assert isinstance(h, int)

    def test_physical_size_is_valid(self):
        """Physical screen size should be >= logical size."""
        pw, ph = self.geo.physical_size
        lw, lh = self.geo.logical_size
        assert pw > 0, f"Physical width should be > 0, got {pw}"
        assert ph > 0, f"Physical height should be > 0, got {ph}"
        assert pw >= lw, f"Physical width {pw} should be >= logical width {lw}"
        assert ph >= lh, f"Physical height {ph} should be >= logical height {lh}"

    def test_scale_factor_reasonable(self):
        """Scale factor should be >= 1.0 and <= 4.0 (no fractional downscaling)."""
        scale = self.geo.scale_factor
        assert scale >= 1.0, f"Scale factor {scale} should be >= 1.0"
        assert scale <= 4.0, f"Scale factor {scale} should be <= 4.0"

    def test_dpi_reasonable(self):
        """DPI should be >= 96 (standard) and <= 384 (4x scaling)."""
        dpi = self.geo.dpi
        assert dpi >= 96, f"DPI {dpi} should be >= 96"
        assert dpi <= 384, f"DPI {dpi} should be <= 384"

    def test_normalized_to_screen_center(self):
        """Normalized (500, 500) should map to approximately the center of the screen."""
        cx, cy = self.geo.normalized_to_screen(500, 500)
        lw, lh = self.geo.logical_size
        expected_cx = lw // 2
        expected_cy = lh // 2
        assert abs(cx - expected_cx) <= 5, f"Center X {cx} should be ~{expected_cx}"
        assert abs(cy - expected_cy) <= 5, f"Center Y {cy} should be ~{expected_cy}"

    def test_normalized_to_screen_corners(self):
        """Normalized (0,0) should map to (0,0) and (1000,1000) to screen max."""
        x0, y0 = self.geo.normalized_to_screen(0, 0)
        assert x0 == 0 and y0 == 0

        xmax, ymax = self.geo.normalized_to_screen(1000, 1000)
        lw, lh = self.geo.logical_size
        assert xmax == lw, f"Max X should be {lw}, got {xmax}"
        assert ymax == lh, f"Max Y should be {lh}, got {ymax}"

    def test_screen_to_normalized_roundtrip(self):
        """Converting screen->normalized->screen should round-trip within ±1px."""
        lw, lh = self.geo.logical_size
        test_x, test_y = lw // 3, lh // 4  # arbitrary test point
        
        nx, ny = self.geo.screen_to_normalized(test_x, test_y)
        rx, ry = self.geo.normalized_to_screen(nx, ny)
        
        assert abs(rx - test_x) <= 1, f"Round-trip X: {test_x} -> {nx} -> {rx}"
        assert abs(ry - test_y) <= 1, f"Round-trip Y: {test_y} -> {ny} -> {ry}"

    def test_image_to_screen_identity(self):
        """When image size matches logical size, image_to_screen should be identity."""
        lw, lh = self.geo.logical_size
        test_x, test_y = 100, 200
        sx, sy = self.geo.image_to_screen(test_x, test_y, lw, lh)
        assert sx == test_x, f"Expected {test_x}, got {sx}"
        assert sy == test_y, f"Expected {test_y}, got {sy}"

    def test_image_to_screen_scaled(self):
        """When image is 2x logical size (physical pixels), coords should be halved."""
        lw, lh = self.geo.logical_size
        pw, ph = lw * 2, lh * 2  # Simulate 200% DPI screenshot
        # Point at image center (pw//2, ph//2) -> should map to screen center (lw//2, lh//2)
        sx, sy = self.geo.image_to_screen(pw // 2, ph // 2, pw, ph)
        assert abs(sx - lw // 2) <= 1, f"Expected ~{lw // 2}, got {sx}"
        assert abs(sy - lh // 2) <= 1, f"Expected ~{lh // 2}, got {sy}"

    def test_physical_to_logical_roundtrip(self):
        """physical_to_logical(logical_to_physical(x,y)) should ≈ (x,y)."""
        test_x, test_y = 500, 300
        px, py = self.geo.logical_to_physical(test_x, test_y)
        rx, ry = self.geo.physical_to_logical(px, py)
        assert abs(rx - test_x) <= 1, f"Round-trip X: {test_x} -> {px} -> {rx}"
        assert abs(ry - test_y) <= 1, f"Round-trip Y: {test_y} -> {py} -> {ry}"

    def test_clamp_to_screen(self):
        """Clamping should keep coords within [0, size-1]."""
        lw, lh = self.geo.logical_size
        
        # In-bounds should stay unchanged
        assert self.geo.clamp_to_screen(100, 200) == (100, 200)
        
        # Out-of-bounds should clamp
        assert self.geo.clamp_to_screen(-10, -20) == (0, 0)
        assert self.geo.clamp_to_screen(lw + 100, lh + 100) == (lw - 1, lh - 1)

    def test_is_within_screen(self):
        """Boundary checking should work correctly."""
        lw, lh = self.geo.logical_size
        
        assert self.geo.is_within_screen(100, 100) is True
        assert self.geo.is_within_screen(-1, 100) is False
        assert self.geo.is_within_screen(lw, 100) is False
        assert self.geo.is_within_screen(100, lh) is False

    def test_diagnostics(self):
        """Diagnostics should return a valid dict with expected keys."""
        diag = self.geo.get_diagnostics()
        assert isinstance(diag, dict)
        assert "logical_size" in diag
        assert "physical_size" in diag
        assert "scale_factor" in diag
        assert "dpi" in diag
        assert "center_point_logical" in diag

    def test_consistency_across_calls(self):
        """Multiple calls should return identical values (no jitter)."""
        s1 = self.geo.logical_size
        s2 = self.geo.logical_size
        assert s1 == s2

        p1 = self.geo.physical_size
        p2 = self.geo.physical_size
        assert p1 == p2

        f1 = self.geo.scale_factor
        f2 = self.geo.scale_factor
        assert f1 == f2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
