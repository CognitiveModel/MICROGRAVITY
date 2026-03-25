import os
import sys
import time
import asyncio
from unittest.mock import MagicMock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nanobot.agent.ui.perception.screen import ScreenObserver
from nanobot.agent.ui.perception.cursor_monitor import CursorMonitor
from nanobot.agent.ui.utils.screen_geometry import ScreenGeometry
from nanobot.agent.ui.perception.cursor_snip_verifier import CursorSnipVerifier, VerifyTrigger

def test_verifier_snip():
    print("--- Testing CursorSnipVerifier ---")
    
    # Setup dependencies
    output_dir = os.path.join(os.path.dirname(__file__), "test_output")
    os.makedirs(output_dir, exist_ok=True)
    
    screen_obs = ScreenObserver(output_dir=output_dir)
    cursor_mon = CursorMonitor()
    screen_geo = ScreenGeometry(force_refresh=True)
    
    # Mock Vision Analyzer since we don't want to make real API calls in a quick unit test
    mock_vision = MagicMock()
    mock_vision.analyze_image_with_prompt.return_value = '{"verified": true, "confidence": 0.9, "actual_element": "test button", "correction": "none"}'
    
    snip_dir = os.path.join(output_dir, "snips")
    verifier = CursorSnipVerifier(
        screen_observer=screen_obs,
        vision_analyzer=mock_vision,
        cursor_monitor=cursor_mon,
        screen_geometry=screen_geo,
        snip_dir=snip_dir
    )
    
    print("Dependencies initialized.")
    
    # 1. Test Should Trigger logic
    should, triggers = verifier.should_trigger(
        target="submit button",
        prediction_source="vlm_fallback", # Should trigger FALLBACK_RESOLUTION
        cursor_valid=False,               # Should trigger CURSOR_MISMATCH
        is_small_target=True              # Should trigger SMALL_TARGET
    )
    print(f"Should Trigger: {should}")
    print(f"Triggers: {[t.name for t in triggers]}")
    assert should is True
    assert VerifyTrigger.CURSOR_MISMATCH in triggers
    assert VerifyTrigger.FALLBACK_RESOLUTION in triggers
    assert VerifyTrigger.SMALL_TARGET in triggers
    
    screen_w, screen_h = screen_geo.logical_size
    test_x = screen_w // 2
    test_y = screen_h // 2
    
    # 2. Reset cooldown for test
    verifier._last_verify_time = 0
    
    # 3. Test Pre-Click Verify
    print(f"Testing Snip at ({test_x}, {test_y})...")
    result = verifier.verify_before_click(
        x=test_x, y=test_y,
        target_description="Center of screen",
        trigger_reasons=triggers
    )
    
    print(f"Result Verified: {result.verified}")
    print(f"Result Confidence: {result.confidence}")
    print(f"Result Element: {result.actual_element}")
    print(f"Snip Path: {result.snip_path}")
    
    assert result.verified is True
    assert result.confidence == 0.9
    assert result.actual_element == "test button"
    assert os.path.exists(result.snip_path)
    
    print("\n--- Test Passed Successfully ---")

if __name__ == "__main__":
    test_verifier_snip()
