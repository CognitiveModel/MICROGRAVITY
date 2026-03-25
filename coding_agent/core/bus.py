"""
Microgravity Message Bus

Decouples internal agent cognition from external channel communication.
Supports metadata-based filtering and priority routing.
"""

import asyncio
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, List, Dict, Optional

@dataclass
class Message:
    content: str
    message_type: str = "TEXT" # e.g., "TEXT", "INTERRUPT", "CLARIFICATION", "PROGRESS"
    channel: str = "internal"
    sender_id: str = "system"
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)

class MessageBus:
    def __init__(self):
        self.inbound: asyncio.Queue[Message] = asyncio.Queue()
        self.outbound: asyncio.Queue[Message] = asyncio.Queue()

    async def publish(self, message: Message, direction: str = "outbound"):
        if direction == "inbound":
            await self.inbound.put(message)
        else:
            await self.outbound.put(message)

    async def publish_inbound(self, msg: Message) -> None:
        """Publish a message from a channel to the agent."""
        await self.inbound.put(msg)

    async def consume_inbound(self) -> Message:
        """Consume the next inbound message (blocks until available)."""
        return await self.inbound.get()

    async def publish_outbound(self, msg: Message) -> None:
        """Publish a response from the agent to channels."""
        await self.outbound.put(msg)

    async def consume_outbound(self) -> Message:
        """Consume the next outbound message (blocks until available)."""
        return await self.outbound.get()

    async def listen(self, direction: str = "outbound") -> Message:
        if direction == "inbound":
            return await self.consume_inbound()
        return await self.consume_outbound()

    async def subscribe(self, direction: str = "outbound"):
        """Async generator for subscribing to messages in a specific direction."""
        while True:
            yield await self.listen(direction)
