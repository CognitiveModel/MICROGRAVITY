import os
from dotenv import load_dotenv
from pathlib import Path

# Load environment variables from .env file
env_path = Path(__file__).parent.parent.parent / ".env"
load_dotenv(dotenv_path=env_path)

# Configuration for Telegram Bot and Gemini AI
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

# HUD Configuration (1 = Enabled, 0 = Disabled)
HUD_ENABLED = int(os.getenv("HUD_ENABLED", "1"))
