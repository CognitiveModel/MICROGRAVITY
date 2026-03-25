"""
Microgravity Channel Manager (Gateway)

Coordinates multiple communication channels and routes 
outbound messages from the internal bus to external targets.
"""

import asyncio
import logging
from typing import Dict, Any, List, Optional
from coding_agent.core.bus import MessageBus, Message

logger = logging.getLogger(__name__)

class ChannelManager:
    """
    Manages communication channels and coordinates message routing.
    """
    def __init__(self, bus: MessageBus):
        self.bus = bus
        self.channels = {}
        self._dispatch_task = None

    def add_channel(self, name: str, channel_instance: Any):
        self.channels[name] = channel_instance
        logger.info(f"Channel registered: {name}")

    async def start_all(self):
        """Starts all registered channels and the outbound dispatcher."""
        self._dispatch_task = asyncio.create_task(self._dispatch_outbound())
        
        tasks = []
        for name, channel in self.channels.items():
            logger.info(f"Starting {name} channel...")
            tasks.append(asyncio.create_task(channel.start()))
        
        await asyncio.gather(*tasks, return_exceptions=True)

    async def _dispatch_outbound(self):
        """Routes messages from the outbound bus to the correct channel."""
        logger.info("Outbound dispatcher started.")
        while True:
            try:
                msg = await self.bus.listen(direction="outbound")
                
                # Check for progress filtering (optional based on user preference)
                if msg.metadata.get("_progress"):
                    # Logic to skip progress if channel config restricts it
                    pass
                
                target_channel = self.channels.get(msg.channel)
                if target_channel:
                    await target_channel.send(msg)
                else:
                    logger.warning(f"Unknown channel: {msg.channel}")
            except Exception as e:
                logger.error(f"Error in dispatcher: {e}")
                await asyncio.sleep(1)

    async def stop_all(self):
        """Stops all channels and the dispatcher."""
        if self._dispatch_task:
            self._dispatch_task.cancel()
        
        for name, channel in self.channels.items():
            try:
                await channel.stop()
            except Exception as e:
                logger.error(f"Error stopping {name}: {e}")
