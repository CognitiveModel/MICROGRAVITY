"""
Live Telegram & UI Agent Test Script

This script initializes the agent, starts the Telegram gateway, 
and waits for a "UI task" command. It also simulates a UI task 
to verify outbound Telegram messaging.
"""

import asyncio
import logging
from coding_agent.core.agent import IntrospectionAgent
from coding_agent.core.bus import Message

logging.basicConfig(level=logging.INFO)

async def test_live_ui_telegram():
    print("--- [STARTING TELEGRAM & UI TEST] ---", flush=True)
    agent = IntrospectionAgent()
    
    # Wait for gateway to warm up
    print("Waiting for Gateway to initialize...", flush=True)
    await asyncio.sleep(5)
    
    # Subscribe to outbound messages to see responses
    async def monitor_outbound():
        print("[MONITOR] Subscribing to OUTBOUND messages...", flush=True)
        async for msg in agent.bus.subscribe(direction="outbound"):
            print(f"[MONITOR] Outbound Detected: {msg.content}", flush=True)

    asyncio.create_task(monitor_outbound())
    
    # Simulate an inbound UI task from a user (Mocked chat_id)
    # In a real test, the user would send this via Telegram
    TEST_CHAT_ID = "840748154" # Example chat id, or whatever the user has
    
    print(f"Injecting mock UI task for chat_id: {TEST_CHAT_ID}", flush=True)
    msg = Message(
        content="Perform a UI scan and find the 'Submit' button on screen.",
        sender_id="user_123",
        channel="telegram",
        metadata={"chat_id": TEST_CHAT_ID}
    )
    
    # Publish to inbound bus
    await agent.bus.publish(msg, direction="inbound")
    print("[TEST] Message Published to Inbound Bus.", flush=True)
    
    # Keep running to allow processing and Telegram outbound
    print("Running for 30 seconds to allow Telegram communication...")
    try:
        await asyncio.sleep(30)
    except KeyboardInterrupt:
        pass
    
    print("--- [TEST COMPLETE] ---")

if __name__ == "__main__":
    asyncio.run(test_live_ui_telegram())
