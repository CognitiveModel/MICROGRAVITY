"""
Microgravity Tracker Persistence Layer

Logs, updates, and tracks the overall ambition parameters of the system,
including software applications created, financial resources consumed,
mathematical capabilities realized, and emergent behavioral quirks.
"""

import json
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)

MICROGRAVITY_DB = "microgravity_state.json"

@dataclass
class SoftwareEntry:
    name: str
    domain: str
    purpose: str
    complexity_level: str
    compute_resources: str = "TBD"
    cost_estimation: str = "TBD"
    popularity_benchmark: float = 0.0

@dataclass
class MathCapability:
    concept: str
    application: str # Changed from applicability
    confidence_level: int  # 1-10 # Removed default value

@dataclass
class Nuance:
    description: str
    source: str  # 'user' or 'introspection'
    severity: str  # 'info', 'warning', 'critical'
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

@dataclass
class RoadmapItem:
    version: str
    milestone: str
    complexity: int  # 1-10
    status: str  # 'planned', 'in_progress', 'achieved'
    iteration_log: List[str] = field(default_factory=list)

@dataclass
class EmergentQuirk:
    name: str
    determinism: str  # Deterministic, Non-Deterministic, Partial
    impact: str       # High, Medium, Low
    description: str

class MicrogravityTracker:
    """Manages the continuous operational tracking of the Microgravity Ambition."""
    
    def __init__(self, db_path: str = MICROGRAVITY_DB):
        self.db_path = db_path
        
        self.software_directory: Dict[str, SoftwareEntry] = {}
        self.mathematical_library: List[MathCapability] = []
        self.quirks_log: List[str] = [] # Changed type to List[str]
        self.nuances: List[Nuance] = [] # Added
        self.architectural_debt: List[str] = [] # Added
        self.roadmap: List[RoadmapItem] = [] # Added
        
        self._load()

    def _load(self):
        if os.path.exists(self.db_path):
            try:
                with open(self.db_path, "r") as f:
                    data = json.load(f)
                    # Load Software
                    for key, val in data.get("software_directory", {}).items():
                        # Cast to float for popularity_benchmark if present
                        if "popularity_benchmark" in val:
                            val["popularity_benchmark"] = float(val["popularity_benchmark"])
                        self.software_directory[key] = SoftwareEntry(**val)
                    # Load Math
                    for m in data.get("mathematical_library", []):
                        if "confidence_level" in m:
                            m["confidence_level"] = int(m["confidence_level"])
                        # Handle renovation: rename applicability -> application if old data exists
                        if "applicability" in m:
                            m["application"] = m.pop("applicability")
                        self.mathematical_library.append(MathCapability(**m))
                    # Load Quirks
                    self.quirks_log = data.get("quirks_log", []) # Changed loading for List[str]
                    # Load Nuances
                    for n in data.get("nuances", []):
                        self.nuances.append(Nuance(**n))
                    # Load Debt
                    self.architectural_debt = data.get("architectural_debt", [])
                    # Load Roadmap
                    for r in data.get("roadmap", []):
                        if "complexity" in r:
                            r["complexity"] = int(r["complexity"])
                        self.roadmap.append(RoadmapItem(**r))
            except Exception as e:
                logger.error(f"Failed to load Microgravity Tracker: {e}")
                self._save()
        else:
            self._init_defaults()
            self._save()

    def _init_defaults(self):
        """Seed initial capabilities."""
        self.add_math_capability("Maximal Marginal Relevance", "Vector Space Diversification")
        self.add_math_capability("Asymptotic Big-O", "Algorithmic Bound Analysis")
        
        self.add_software("Introspection Governor", "Meta-Control", "Systemic", "High")
        self.add_software("UI Agent", "CV and Native Execution", "OS Hardware", "Medium")

    def _save(self):
        data = {
            "software_directory": {k: v.__dict__ for k, v in self.software_directory.items()},
            "mathematical_library": [m.__dict__ for m in self.mathematical_library],
            "quirks_log": self.quirks_log, # Changed saving for List[str]
            "nuances": [n.__dict__ for n in self.nuances], # Added
            "architectural_debt": self.architectural_debt, # Added
            "roadmap": [r.__dict__ for r in self.roadmap] # Added
        }
        with open(self.db_path, "w") as f:
            json.dump(data, f, indent=4)

    def add_software(self, name: str, domain: str, purpose: str, complexity: str):
        self.software_directory[name] = SoftwareEntry(name=name, domain=domain, purpose=purpose, complexity_level=complexity)
        self._save()

    def add_math_capability(self, concept: str, application: str, confidence: int = 10):
        self.mathematical_library.append(MathCapability(concept=concept, application=application, confidence_level=confidence))
        self._save()

    def log_quirk(self, description: str):
        self.quirks_log.append(description)
        self._save()

    def get_microgravity_context(self) -> str:
        """Returns a stringified context of the Microgravity's current power for system prompts."""
        context = "--- MICROGRAVITY STATE ALIGNMENT ---\n"
        context += f"Apps Deployed: {len(self.software_directory)}\n"
        context += f"Mathematical Constraints Mastered: {', '.join([m.concept for m in self.mathematical_library])}\n"
        context += f"Known Architectural Quirks: {', '.join(self.quirks_log)}\n"
        context += f"Architectural Debt: {', '.join(self.architectural_debt)}\n"
        context += f"User Nuances: {', '.join([n.description for n in self.nuances])}\n"
        return context
