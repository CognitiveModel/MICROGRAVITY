import sys
import os
import asyncio
import threading
import time

# Add project root to path
sys.path.append(r'c:\Users\HP\coding agent final')

from coding_agent.ui_agent.core.ui_agent import UIAgent

async def test_ui_agent_cleanup():
    print("Initializing UIAgent...")
    agent = UIAgent()
    
    print("Starting Live Stream...")
    # This usually starts a background thread and loop
    agent._start_live_stream()
    
    # Wait for session to establish (or fail if no key, but we want to test the teardown)
    time.sleep(5)
    
    print("Stopping Live Stream...")
    # This should now properly await the internal async teardown
    agent._stop_live_stream()
    
    print("Verifying if background loop is stopped...")
    if not agent._loop.is_running():
        print("PASS: Background loop stopped successfully.")
    else:
        print("FAIL: Background loop is still running.")

    # Check for any "Task was destroyed" warnings in a real run is harder, 
    # but the fact that we await the result in _stop_live_stream (result=10s timeout)
    # is the fix.

if __name__ == "__main__":
    # We run this in a way that allows us to see the output
    try:
        asyncio.run(test_ui_agent_cleanup())
    except Exception as e:
        print(f"Error during test: {e}")
