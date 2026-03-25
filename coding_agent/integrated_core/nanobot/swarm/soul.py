"""Soul Controller — maintains identity, capabilities, and introspection over time."""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from loguru import logger # type: ignore

from .lmdb_store import LMDBStore # type: ignore


@dataclass
class Capability:
    id: str
    description: str
    confidence: float
    last_used: datetime


class SoulController:
    """
    Core identity system defined in Swarm Architecture.
    Persists continuous sense of self, introspection quality, and learned traits.
    """

    def __init__(self, store: LMDBStore):
        self.store = store
        self._ensure_seeded()

    def _ensure_seeded(self) -> None:
        """Seed initial identity if missing."""
        if not self.store.get("soul:identity:core"):
            logger.info("Seeding initial Soul Identity state.")
            self.store.put("soul:identity:core", {
                "name": "Nanobot Swarm",
                "birth_timestamp": datetime.utcnow().isoformat(), # type: ignore
                "core_directive": "Iterate wisdom, maintain continuity, assist the user.",
            })
            self.store.put("soul:introspection:quality", {
                "depth_score": 1.0,  # 0.0 to 10.0
                "blind_spots": ["physical_world_context", "realtime_audio"],
                "last_evaluated": datetime.utcnow().isoformat(),
            })

    def get_identity(self) -> dict[str, Any]:
        """Get the core identity document."""
        return self.store.get("soul:identity:core", {})

    def get_introspection(self) -> dict[str, Any]:
        """Get current introspection stats."""
        return self.store.get("soul:introspection:quality", {})

    def update_introspection(self, depth_delta: float, discovered_blind_spots: list[str]) -> dict[str, Any]:
        """Adjust introspection quality based on recent interactions."""
        current = self.get_introspection()
        score = current.get("depth_score", 1.0)
        
        # Clamp between 0 and 10
        new_score = max(0.0, min(10.0, score + depth_delta))
        
        blind_spots = set(current.get("blind_spots", []))
        for spot in discovered_blind_spots:
            blind_spots.add(spot)
            
        updated = {
            "depth_score": new_score,
            "blind_spots": list(blind_spots),
            "last_evaluated": datetime.utcnow().isoformat(),
        }
        self.store.put("soul:introspection:quality", updated)
        return updated

    def register_capability(self, cap_id: str, description: str, confidence: float = 0.5) -> None:
        """Add or update a known capability of the system."""
        key = f"soul:capability:{cap_id}"
        self.store.put(key, {
            "id": cap_id,
            "description": description,
            "confidence": confidence,
            "last_used": datetime.utcnow().isoformat()
        })

    def verify_capability(self, cap_id: str) -> bool:
        """Check if a capability is known (and potentially high enough confidence)."""
        data = self.store.get(f"soul:capability:{cap_id}")
        if not data:
            return False
            
        # Update last used
        data["last_used"] = datetime.utcnow().isoformat()
        self.store.put(f"soul:capability:{cap_id}", data)
        return True

    def add_trait(self, trait_name: str, value: Any) -> None:
        """Add a learned stable trait to the identity."""
        self.store.put(f"soul:trait:{trait_name}", {
            "value": value,
            "discovered_at": datetime.utcnow().isoformat()
        })

    def get_traits(self) -> dict[str, Any]:
        """Retrieve all currently known stable traits."""
        traits = {}
        for key, value in self.store.prefix_scan("soul:trait:"):
            trait_name = key.replace("soul:trait:", "") # type: ignore
            traits[trait_name] = value
        return traits
