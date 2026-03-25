"""
Soul Identity Manager

Provides persistent cognitive invariants (personality traits) for the swarm, 
determining its baseline risk, precision, and persistence levels.
Backed by a simple JSON structure to simulate LMDB namespace:soul.
"""

import json
import uuid
import os
import logging
from dataclasses import dataclass, field
from typing import List

logger = logging.getLogger(__name__)

SOUL_FILE = "soul_identity.json"

@dataclass
class SoulTraits:
    persistence_bias: float = 0.85
    creative_risk: float = 0.40
    precision_weight: float = 0.95
    sophistication_tier: str = "STRUCTURAL" # REFLEX | STRUCTURAL | GENERATIVE
    hardened_wisdom_threshold: int = 50
    hardened_wisdom_features: List[str] = field(default_factory=list)

class SoulIdentity:
    """Manages the static personality elements of the CSUSE_CORE architecture."""

    def __init__(self, soul_file: str = SOUL_FILE):
        self.soul_file = soul_file
        self.soul_id = str(uuid.uuid4())
        self.traits = SoulTraits()
        self._load_or_init()

    def _load_or_init(self):
        if os.path.exists(self.soul_file):
            try:
                with open(self.soul_file, "r") as f:
                    data = json.load(f)
                    self.soul_id = data.get("soul_id", self.soul_id)
                    trait_data = data.get("traits", {})
                    # Ensure correct types
                    if "persistence_bias" in trait_data: trait_data["persistence_bias"] = float(trait_data["persistence_bias"])
                    if "creative_risk" in trait_data: trait_data["creative_risk"] = float(trait_data["creative_risk"])
                    if "precision_weight" in trait_data: trait_data["precision_weight"] = float(trait_data["precision_weight"])
                    if "hardened_wisdom_threshold" in trait_data: trait_data["hardened_wisdom_threshold"] = int(trait_data["hardened_wisdom_threshold"])
                    if "hardened_wisdom_features" in trait_data: trait_data["hardened_wisdom_features"] = list(trait_data["hardened_wisdom_features"])
                    
                    self.traits = SoulTraits(**trait_data)
                logger.info(f"Loaded Soul Identity: {self.soul_id}")
            except Exception as e:
                logger.error(f"Failed to load soul file. Initializing default. {e}")
                self._save()
        else:
            logger.info("Initializing new Soul Identity.")
            self._save()

    def _save(self):
        data = {
            "soul_id": self.soul_id,
            "traits": {
                "persistence_bias": self.traits.persistence_bias,
                "creative_risk": self.traits.creative_risk,
                "precision_weight": self.traits.precision_weight,
                "sophistication_tier": self.traits.sophistication_tier,
                "hardened_wisdom_threshold": self.traits.hardened_wisdom_threshold,
                "hardened_wisdom_features": self.traits.hardened_wisdom_features
            }
        }
        with open(self.soul_file, "w") as f:
            json.dump(data, f, indent=4)

    def evaluate_risk(self, proposed_risk: float) -> bool:
        """Determines if a subagent's proposed plan is too risky for this soul."""
        if proposed_risk > self.traits.creative_risk:
            # Too risky
            return False
        return True

    def get_preferred_tier(self) -> str:
        """Returns target depth (REFLEX, STRUCTURAL, GENERATIVE)."""
        return self.traits.sophistication_tier
