"""Models for Conviction and Sophistication levels from the Swarm Architecture."""

from enum import Enum, IntEnum

class ConvictionLevel(str, Enum):
    """Degrees of certainty attached to knowledge."""
    TENTATIVE = "TENTATIVE"           # Single source, untested
    DEVELOPING = "DEVELOPING"         # Partially tested, some logical coherence
    GROUNDED = "GROUNDED"             # Cross-verified, outcome-proven
    HARDENED = "HARDENED"             # Survived contradiction tests
    PARADOX_AWARE = "PARADOX_AWARE"   # Knows the limits/exceptions of its own truth


class SophisticationTier(IntEnum):
    """Depth of reasoning applied to a problem (Levels 0-6)."""
    REFLEX = 0         # Pattern matching, pre-LLM rules (0ms)
    PROCEDURAL = 1     # Known workflows, tool execution (System 1.5)
    ANALYTICAL = 2     # Basic reasoning, standard Q&A (System 2)
    STRUCTURAL = 3     # Identifying root causes, system diagrams
    SYSTEMIC = 4       # Second-order effects, dynamic loops
    PARADOXICAL = 5    # Contradiction synthesis, tradeoff management
    GENERATIVE = 6     # Creating novel paradigms or definitions


def validate_conviction(level: str | ConvictionLevel, hop_count: int) -> bool:
    """Enforce rules about how conviction corresponds to hop depth."""
    if isinstance(level, str):
        try:
            level = ConvictionLevel(level.upper())
        except ValueError:
            return False

    if level == ConvictionLevel.HARDENED and hop_count < 3:
        return False
    if level == ConvictionLevel.PARADOX_AWARE and hop_count < 5:
        return False
    
    return True


def get_budget_multiplier(tier: SophisticationTier) -> float:
    """Multiplier for LLM token and API budgets based on required sophistication."""
    match tier:
        case SophisticationTier.REFLEX: return 0.0     # Uses LMDB, no LLM tokens
        case SophisticationTier.PROCEDURAL: return 0.5 # Small context, fast models
        case SophisticationTier.ANALYTICAL: return 1.0 # Standard budget
        case SophisticationTier.STRUCTURAL: return 2.0
        case SophisticationTier.SYSTEMIC: return 4.0
        case SophisticationTier.PARADOXICAL: return 8.0
        case SophisticationTier.GENERATIVE: return 10.0
