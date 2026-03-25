"""
Microgravity Gateway Runner

This script starts the IntrospectionAgent and its associated 
ChannelGateway (including Telegram) in a persistent mode.
"""

import asyncio
import logging
import signal
from coding_agent.core.agent import IntrospectionAgent

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("GatewayRunner")

async def main():
    logger.info("--- [STARTING MICROGRAVITY GATEWAY] ---")
    
    # Initialize the agent (this starts the gateway dispatcher and telegram polling)
    agent = IntrospectionAgent()
    
    logger.info("Gateway and Telegram Channel initialized.")
    logger.info("Press Ctrl+C to stop.")

    # Event to wait for shutdown
    stop_event = asyncio.Event()

    # Handle termination signals
    def stop_callback():
        logger.info("Shutdown signal received.")
        stop_event.set()

    # Windows alternative for signal handling in asyncio
    try:
        loop = asyncio.get_running_loop()
        # signal.SIGINT and signal.SIGTERM are common
        # On Windows, we might just rely on KeyboardInterrupt if loop.add_signal_handler is not available
        pass 
    except Exception:
        pass

    try:
        # Keep the script running until the event is set or interrupted
        await stop_event.wait()
    except KeyboardInterrupt:
        logger.info("KeyboardInterrupt detected.")
    finally:
        logger.info("--- [GATEWAY SHUTDOWN] ---")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
