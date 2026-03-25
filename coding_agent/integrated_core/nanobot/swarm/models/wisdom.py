"""Wisdom components, Enums, and Dataclasses spanning the Swarm Architecture."""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, IntEnum
from typing import Optional

from .conviction import ConvictionLevel, SophisticationTier


class WisdomHopType(str, Enum):
    # R1: Extract core phenomena
    CHARACTERIZE = "CHARACTERIZE"
    # R2: Map to ontology/taxonomy
    CLASSIFY = "CLASSIFY"
    # R3: Specify context-dependent meanings
    TERMINOLOGY_SCOPE = "TERMINOLOGY_SCOPE"
    # R4: Handle boundary cases
    INCLUSION_EXCLUSION = "INCLUSION_EXCLUSION"
    # R5: Create/destroy abstractions
    BUILD_DISCARD = "BUILD_DISCARD"
    # R6: Redo R1-R5 on smaller granularity
    ITERATE = "ITERATE"
    # R7: Determine certainty and durability
    CONVICT = "CONVICT"


class PipelinePhase(str, Enum):
    # Initial trigger, intent parsing
    RECEPTION = "RECEPTION"
    # Scope determination, context loading
    BOUNDING = "BOUNDING"
    # R1-R6 hop execution, insight generation
    CONSTRUCTION = "CONSTRUCTION"
    # R7 hop, durability testing, paradox checks
    CONVICTION = "CONVICTION"
    # Tool execution, state emission, UI projection
    ACTIVATION = "ACTIVATION"


class WisdomInterruptType(str, Enum):
    # Triggers rewinds and deepening
    SHALLOWNESS = "SHALLOWNESS"
    TERM_AMBIGUITY = "TERM_AMBIGUITY"
    CATEGORY_ERROR = "CATEGORY_ERROR"
    CONFIRMATION_BIAS = "CONFIRMATION_BIAS"
    PREMATURE_CONVICTION = "PREMATURE_CONVICTION"
    MISSING_BOUNDARY = "MISSING_BOUNDARY"
    CIRCULAR_REASONING = "CIRCULAR_REASONING"
    FALSE_DICHOTOMY = "FALSE_DICHOTOMY"
    SCALE_MISMATCH = "SCALE_MISMATCH"
    TELEOLOGICAL_FALLACY = "TELEOLOGICAL_FALLACY"
    CONTEXT_DROPPING = "CONTEXT_DROPPING"
    EPISTEMIC_TRESPASSING = "EPISTEMIC_TRESPASSING"


class DiscernmentType(str, Enum):
    # Types of subtle insights harvested from operations
    DISTINCTION = "DISTINCTION"
    EXCEPTION_THAT_PROVES = "EXCEPTION_THAT_PROVES"
    HIDDEN_SIMILARITY = "HIDDEN_SIMILARITY"
    BOUNDARY_MARKER = "BOUNDARY_MARKER"
    ABSENCE_SIGNAL = "ABSENCE_SIGNAL"
    CONDITIONAL_FLIP = "CONDITIONAL_FLIP"


class TriggerType(str, Enum):
    # Proactive awakening systems
    RECOGNITION = "RECOGNITION"
    CONTRADICTION = "CONTRADICTION"
    DECAY = "DECAY"
    COMPOSITION = "COMPOSITION"
    ESCALATION = "ESCALATION"


class OutcomeMode(str, Enum):
    # Architectural topology of tool results
    SINGULAR = "SINGULAR"
    DUAL = "DUAL"
    TRIAD = "TRIAD"
    MULTI = "MULTI"
    OPEN = "OPEN"


@dataclass
class WisdomHop:
    """A single discrete step of reasoning or knowledge refinement."""
    hop_id: str
    hop_level: WisdomHopType
    content: str
    topic_id: str
    conviction_level: ConvictionLevel
    sophistication_level: SophisticationTier
    timestamp: datetime = field(default_factory=datetime.utcnow)
    parent_hop_id: Optional[str] = None
    embed_vec: Optional[list[float]] = None
    embed_model: Optional[str] = None


@dataclass
class WisdomInterrupt:
    """A circuit-breaker event halting shallow cognition."""
    interrupt_id: str
    type: WisdomInterruptType
    task_id: str
    fired_at: datetime
    resolution_status: str = "PENDING"
    hops_rewound: int = 0
    triggering_content: str = ""


@dataclass
class DiscernmentPoint:
    """Subtle distinction harvested continuously."""
    dp_id: str
    type: DiscernmentType
    content: str
    source_action_id: str
    seedability: str = "PATTERN"  # IMMEDIATE, CONFIRMED, PATTERN, CROSS_DOMAIN, REQUESTED
    timestamp: datetime = field(default_factory=datetime.utcnow)
