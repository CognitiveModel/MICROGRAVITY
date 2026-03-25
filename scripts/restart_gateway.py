"""
Gateway Restart Verification Script
"""

import asyncio
from coding_agent.core.agent import IntrospectionAgent

async def test_restart():
    print("Initializing Agent...")
    agent = IntrospectionAgent()
    
    # Wait a bit
    await asyncio.sleep(1)
    
    # Trigger restart
    agent.restart_gateway()
    
    # Wait a bit to ensure task starts
    await asyncio.sleep(1)
    print("Restart test complete.")

if __name__ == "__main__":
    asyncio.run(test_restart())
