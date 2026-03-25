"""
Microgravity Telegram Channel

Provides a live bridge to Telegram using python-telegram-bot.
"""

import asyncio
import logging
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
from coding_agent.core.bus import MessageBus, Message
from coding_agent.utils.config import TELEGRAM_BOT_TOKEN

logger = logging.getLogger(__name__)

class TelegramChannel:
    name = "telegram"

    def __init__(self, bus: MessageBus, token: str = TELEGRAM_BOT_TOKEN):
        self.bus = bus
        self.token = token
        self.app = None
        self._running = False

    async def start(self):
        """Starts the Telegram bot polling."""
        if not self.token:
            logger.error("No Telegram token provided.")
            return

        self.app = Application.builder().token(self.token).build()
        
        # Add handlers
        self.app.add_handler(CommandHandler("start", self._start_command))
        self.app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, self._handle_incoming))
        
        logger.info("Starting Telegram Bot...")
        await self.app.initialize()
        await self.app.start()
        await self.app.updater.start_polling()
        self._running = True

    async def stop(self):
        """Stops the Telegram bot."""
        if self.app:
            await self.app.updater.stop()
            await self.app.stop()
            await self.app.shutdown()
        self._running = False

    async def send(self, message: Message):
        """Sends a message to a specific chat_id."""
        if not self.app:
            return
        
        # Use metadata['chat_id'] or a fallback
        chat_id = message.metadata.get("chat_id")
        if not chat_id:
            logger.warning("No chat_id in message metadata for Telegram.")
            return

        await self.app.bot.send_message(chat_id=chat_id, text=message.content)

    async def _start_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        await update.message.reply_text("Microgravity Gateway Online. Send me a task!")

    async def _handle_incoming(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        sender_id = str(update.effective_user.id)
        chat_id = str(update.effective_chat.id)
        content = update.message.text
        
        # Publish to Inbound bus
        msg = Message(
            content=content,
            sender_id=sender_id,
            channel=self.name,
            metadata={"chat_id": chat_id}
        )
        await self.bus.publish(msg, direction="inbound")
