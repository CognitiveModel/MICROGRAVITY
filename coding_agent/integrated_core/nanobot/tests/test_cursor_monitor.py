"""
Tests for CursorMonitor — Win32-based cursor type detection.
"""
import pytest
import sys
import os
import time

# Add project root to sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestCursorMonitor:
    """Test suite for CursorMonitor cursor type detection."""

    @pytest.fixture(autouse=True)
    def setup(self):
        from nanobot.agent.ui.perception.cursor_monitor import CursorMonitor, CursorType
        self.monitor = CursorMonitor()
        self.CursorType = CursorType

    def test_get_cursor_type_returns_valid_enum(self):
        """get_cursor_type() should return a valid CursorType enum value."""
        cursor = self.monitor.get_cursor_type()
        assert isinstance(cursor, self.CursorType), f"Expected CursorType enum, got {type(cursor)}"

    def test_get_cursor_state_returns_complete(self):
        """get_cursor_state() should return all fields populated."""
        state = self.monitor.get_cursor_state()
        assert state.cursor_type is not None
        assert isinstance(state.position, tuple)
        assert len(state.position) == 2
        assert isinstance(state.position[0], int)
        assert isinstance(state.position[1], int)
        assert isinstance(state.timestamp, float)
        assert state.timestamp > 0

    def test_get_cursor_position(self):
        """get_cursor_position() should return a tuple of two ints."""
        pos = self.monitor.get_cursor_position()
        assert isinstance(pos, tuple)
        assert len(pos) == 2
        assert isinstance(pos[0], int)
        assert isinstance(pos[1], int)

    def test_system_cursors_loaded(self):
        """System cursor handles should be loaded at init time."""
        assert len(self.monitor._system_cursors) > 0, \
            "No system cursor handles loaded"
        # At minimum, ARROW should be loaded
        arrow_found = any(
            ct == self.CursorType.ARROW 
            for ct in self.monitor._system_cursors.values()
        )
        assert arrow_found, "ARROW cursor should be in system cursor map"

    def test_infer_element_type(self):
        """infer_element_type() should return a non-empty string."""
        element_type = self.monitor.infer_element_type()
        assert isinstance(element_type, str)
        assert len(element_type) > 0

    def test_validate_hover_unknown_type(self):
        """Validating an unknown element type should return True (benefit of doubt)."""
        # This is a made-up element type that doesn't exist in the map
        result = self.monitor.validate_hover("nonexistent_element_xyz_123")
        assert result is True, "Unknown element types should pass validation"

    def test_cursor_summary(self):
        """get_cursor_summary() should return a dict with expected keys."""
        summary = self.monitor.get_cursor_summary()
        assert isinstance(summary, dict)
        assert "type" in summary
        assert "position" in summary
        assert "visible" in summary
        assert "inferred_element" in summary

    def test_was_recently_returns_bool(self):
        """was_recently() should return a boolean."""
        result = self.monitor.was_recently(self.CursorType.ARROW)
        assert isinstance(result, bool)

    def test_consistency(self):
        """Multiple rapid calls should return consistent cursor type (no jitter)."""
        types = [self.monitor.get_cursor_type() for _ in range(10)]
        # In a stable desktop scenario, cursor shouldn't change during rapid queries
        unique_types = set(types)
        assert len(unique_types) <= 2, \
            f"Cursor type jittered across {len(unique_types)} types during rapid queries: {unique_types}"

    def test_element_cursor_map_coverage(self):
        """The element-cursor map should cover common UI element types."""
        essential_types = [
            "link", "button", "text_field", "text_input", 
            "resize_handle", "drag_handle", "clickable"
        ]
        for elem_type in essential_types:
            assert elem_type in self.monitor._element_cursor_map, \
                f"Element type '{elem_type}' missing from cursor map"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
