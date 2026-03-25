"""
Microgravity Agent Loop

Core processing engine that handles the message lifecycle:
1. Inbound message consumption
2. Context preparation
3. LLM interaction
4. Tool execution
5. Outbound response publication
"""

import asyncio
import logging
import json
from pathlib import Path
from typing import Dict, Any, List, Optional, Callable, Awaitable

from coding_agent.core.bus import MessageBus, Message
from coding_agent.utils.gemini_client import init_gemini, get_gemini_response

logger = logging.getLogger(__name__)

class AgentLoop:
    def __init__(
        self,
        bus: MessageBus,
        workspace: Path,
        model_name: str = "gemini-1.5-pro",
        max_iterations: int = 10
    ):
        self.bus = bus
        self.workspace = workspace
        self.model = init_gemini()
        self.max_iterations = max_iterations
        self._running = False
        self._active_tasks = {}

    async def run(self):
        """Starts the main processing loop."""
        self._running = True
        logger.info("Microgravity Agent Loop started.")
        
        while self._running:
            try:
                # Consume from inbound bus
                msg = await self.bus.listen(direction="inbound")
                
                # Dispatch task
                task = asyncio.create_task(self._process_message(msg))
                self._active_tasks[msg.timestamp.isoformat()] = task
                task.add_done_callback(lambda t: self._active_tasks.pop(msg.timestamp.isoformat(), None))
                
            except Exception as e:
                logger.error(f"Error in AgentLoop: {e}")
                await asyncio.sleep(1)

    async def _process_message(self, msg: Message):
        """Processes a single message through the LLM and tools."""
        logger.info(f"Processing message: {msg.content}")
        
        chat_id = msg.metadata.get("chat_id")
        session_key = f"{msg.channel}:{chat_id}"
        
        from coding_agent.core.session import SessionManager
        sessions = SessionManager(self.workspace / "sessions")
        session = sessions.get_or_create(session_key)
        
        # 1. Progress Update
        await self.bus.publish_outbound(Message(
            content=f"🧠 Microgravity is analyzing: {msg.content[:50]}...",
            channel=msg.channel,
            metadata={"chat_id": chat_id, "_progress": True}
        ))
        
        # 2. LLM Turn
        system_prompt = (
            "You are Microgravity, an advanced swarm operating system. Be concise and technical. Use tool calls for filesystem actions.\n"
            "COGNITIVE FEATURES: Deep-Labeled Task ID, Multi-Path Optimization (PathID), and Autonomous Error/Mistake Distillation.\n"
            "Refer to FEATURE_MANIFEST.md for technical details."
        )
        
        # Build history
        messages = [{"role": "system", "content": system_prompt}]
        messages.extend(session.get_history())
        messages.append({"role": "user", "content": msg.content})
        
        try:
            # For now, we use a simplified call. Real implementation would handle tool calls.
            prompt = json.dumps(messages)
            response = get_gemini_response(self.model, f"Chat History: {prompt}\nRespond to the latest user message.")
            
            # Sync session
            session.add_message("user", msg.content)
            session.add_message("assistant", response)
            sessions.save(session)
            
            # 3. Publish Final Response
            await self.bus.publish_outbound(Message(
                content=response,
                channel=msg.channel,
                metadata={"chat_id": chat_id}
            ))
        except Exception as e:
            logger.error(f"LLM Error: {e}")
            await self.bus.publish_outbound(Message(
                content="❌ Sorry, I encountered an error processing your request.",
                channel=msg.channel,
                metadata={"chat_id": chat_id}
            ))

    def stop(self):
        self._running = False
        logger.info("Microgravity Agent Loop stopping.")
