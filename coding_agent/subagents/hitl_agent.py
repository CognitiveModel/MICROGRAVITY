import asyncio
import logging
from typing import Optional
from coding_agent.core.bus import MessageBus, Message
from coding_agent.utils.gemini_client import init_gemini, get_gemini_response
from coding_agent.storage.world_model import WorldModel
from coding_agent.storage.knowledge_store import KnowledgeStore

logger = logging.getLogger(__name__)

class HumanInTheLoopAgent:
    """
    Sits directly on the MessageBus to provide active interception,
    clarification branching, and interactive task feedback.
    """
    def __init__(self, bus: MessageBus):
        self.bus = bus
        self.gemini = init_gemini()
        self.world_model = WorldModel()
        self.knowledge_store = KnowledgeStore()

    async def start_listening(self):
        """Runs concurrently on the bus to intercept and clarify."""
        print("[HITL] Agent Online. Monitoring bus for clarification requests...", flush=True)
        async for msg in self.bus.subscribe(direction="inbound"):
            # Check for specific user clarification requests
            if msg.message_type == "CLARIFICATION" or self._is_clarification_request(msg.content):
                await self.handle_arbitrary_inquiry(msg)

    def _is_clarification_request(self, txt: str) -> bool:
        txt = txt.lower()
        return any(k in txt for k in ["explain", "clarify", "why ", "what is", "wait "]) and len(txt) < 150

    async def handle_arbitrary_inquiry(self, msg: Message):
        """
        Spins off a lightweight, ephemeral Analysis Task to answer user queries,
        effectively acting as a localized pause/resume via async concurrency.
        """
        chat_id = msg.metadata.get("chat_id")
        await self.bus.publish(Message(
            content="⏸️ [HITL] Analyzing your inquiry...",
            channel=msg.channel,
            metadata={"chat_id": chat_id}
        ))
        
        # Ephemeral analysis task
        prompt = (
            f"User asked for clarification during a Swarm task: '{msg.content}'.\n"
            "Provide a concise, granular explanation analyzing the system behavior, "
            "plan structure, or general technical concept. Keep it under 3 paragraphs."
        )
        
        loop = asyncio.get_event_loop()
        try:
            response = await loop.run_in_executor(None, get_gemini_response, self.gemini, prompt)
        except Exception as e:
            response = f"Failed to run analysis: {e}"
        
        await self.bus.publish(Message(
            content=f"🧠 [HITL Explanatory Branch]\n\n{response}\n\n▶️ Resuming standard processing.",
            channel=msg.channel,
            metadata={"chat_id": chat_id}
        ))
