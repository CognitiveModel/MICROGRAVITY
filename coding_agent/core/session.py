"""
Microgravity Session Manager

Handles persistence of chat history across multiple messages.
"""

import json
from pathlib import Path
from typing import Dict, List, Any, Optional

class Session:
    def __init__(self, key: str):
        self.key = key
        self.history: List[Dict[str, str]] = []

    def add_message(self, role: str, content: str):
        self.history.append({"role": role, "content": content})

    def get_history(self) -> List[Dict[str, str]]:
        return self.history

    def clear(self):
        self.history = []

class SessionManager:
    def __init__(self, storage_path: Path):
        self.storage_path = storage_path
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self.sessions: Dict[str, Session] = {}

    def get_or_create(self, key: str) -> Session:
        if key not in self.sessions:
            self.sessions[key] = Session(key)
            # Optional: Load from disk
        return self.sessions[key]

    def save(self, session: Session):
        # Optional: Save to disk
        pass
