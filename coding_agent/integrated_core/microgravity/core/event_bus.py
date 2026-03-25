"""
MICROGRAVITY CORE — Durable Event Bus

Redis Streams-backed pub/sub system for inter-agent communication, lifecycle signals,
and scheduled triggers. Supports event types: pause, kill, resume, HITL escalation,
budget depletion, scheduled triggers, TTL expiry, and more.

Falls back to an in-memory asyncio queue if Redis is unavailable.
"""

from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime
from typing import Any, Callable, Coroutine, Optional

from models.schemas import EventType, SwarmEvent

logger = logging.getLogger(__name__)

# Type alias for event handlers
EventHandler = Callable[[SwarmEvent], Coroutine[Any, Any, None]]


class EventBus:
    """
    Durable event bus with Redis Streams backend and in-memory fallback.
    
    Features:
    - Typed event routing (handlers register for specific EventTypes)
    - Broadcast and targeted events (target=None means broadcast)
    - Priority-aware processing
    - TTL-based event expiration
    - Async handler execution
    """

    def __init__(self, redis_url: str | None = None):
        self._redis_url = redis_url
        self._redis = None
        self._handlers: dict[EventType, list[EventHandler]] = {}
        self._global_handlers: list[EventHandler] = []
        self._queue: asyncio.Queue[SwarmEvent] = asyncio.Queue()
        self._running = False
        self._consumer_task: Optional[asyncio.Task] = None

    async def initialize(self) -> None:
        """Connect to Redis if available, otherwise use in-memory queue."""
        if self._redis_url:
            try:
                import redis.asyncio as aioredis
                self._redis = aioredis.from_url(self._redis_url, decode_responses=True)
                await self._redis.ping()
                logger.info(f"Event Bus connected to Redis at {self._redis_url}")
            except Exception as e:
                logger.warning(f"Redis unavailable ({e}), using in-memory queue")
                self._redis = None
        else:
            logger.info("Event Bus running in-memory mode (no Redis URL configured)")

    def subscribe(self, event_type: EventType, handler: EventHandler) -> None:
        """Register a handler for a specific event type."""
        if event_type not in self._handlers:
            self._handlers[event_type] = []
        self._handlers[event_type].append(handler)
        logger.debug(f"Handler registered for {event_type.value}")

    def subscribe_all(self, handler: EventHandler) -> None:
        """Register a handler that receives ALL events."""
        self._global_handlers.append(handler)

    async def emit(self, event: SwarmEvent) -> None:
        """Publish an event to the bus."""
        logger.info(f"Event emitted: {event.event_type.value} from {event.source}"
                     f"{' -> ' + event.target if event.target else ' (broadcast)'}")

        if self._redis:
            # Persist to Redis Stream for durability
            await self._redis.xadd(
                "microgravity:events",
                {"data": event.model_dump_json()},
                maxlen=10000,
            )
        
        # Also route to in-memory handlers immediately
        await self._route_event(event)

    async def emit_simple(self, event_type: EventType, source: str,
                          target: str | None = None, payload: dict | None = None,
                          priority: int = 5) -> SwarmEvent:
        """Convenience method to emit an event without constructing SwarmEvent manually."""
        event = SwarmEvent(
            event_type=event_type,
            source=source,
            target=target,
            payload=payload or {},
            priority=priority,
        )
        await self.emit(event)
        return event

    async def _route_event(self, event: SwarmEvent) -> None:
        """Route an event to all registered handlers."""
        # Check TTL
        if event.ttl_seconds:
            age = (datetime.utcnow() - event.timestamp).total_seconds()
            if age > event.ttl_seconds:
                logger.debug(f"Event {event.event_id} expired (TTL={event.ttl_seconds}s, age={age:.0f}s)")
                return

        tasks = []

        # Type-specific handlers
        handlers = self._handlers.get(event.event_type, [])
        for handler in handlers:
            tasks.append(self._safe_call(handler, event))

        # Global handlers
        for handler in self._global_handlers:
            tasks.append(self._safe_call(handler, event))

        if tasks:
            await asyncio.gather(*tasks)

    @staticmethod
    async def _safe_call(handler: EventHandler, event: SwarmEvent) -> None:
        """Call a handler with error catching."""
        try:
            await handler(event)
        except Exception as e:
            logger.error(f"Event handler error for {event.event_type.value}: {e}", exc_info=True)

    # ── Scheduled Events ──────────────────────────────────────

    async def schedule_event(self, event: SwarmEvent, delay_seconds: float) -> None:
        """Schedule an event to be emitted after a delay."""
        async def _delayed():
            await asyncio.sleep(delay_seconds)
            await self.emit(event)
        asyncio.create_task(_delayed())
        logger.info(f"Scheduled {event.event_type.value} in {delay_seconds}s")

    # ── Redis Stream Consumer (for durability) ────────────────

    async def start_consumer(self) -> None:
        """Start consuming events from Redis Stream (for distributed setups)."""
        if not self._redis:
            return
        self._running = True
        self._consumer_task = asyncio.create_task(self._consume_loop())

    async def _consume_loop(self) -> None:
        """Continuously read from Redis Stream and route events."""
        last_id = "$"  # Only new events
        while self._running:
            try:
                results = await self._redis.xread(
                    {"microgravity:events": last_id},
                    count=10,
                    block=1000,
                )
                for stream_name, messages in results:
                    for msg_id, data in messages:
                        event = SwarmEvent.model_validate_json(data["data"])
                        await self._route_event(event)
                        last_id = msg_id
            except Exception as e:
                logger.error(f"Redis consumer error: {e}")
                await asyncio.sleep(1)

    async def stop(self) -> None:
        """Stop the event bus and close connections."""
        self._running = False
        if self._consumer_task:
            self._consumer_task.cancel()
        if self._redis:
            await self._redis.close()
