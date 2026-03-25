"""Modal Swarm Architecture: Advanced memory and reasoning engine."""

from .engine import SwarmEngine # type: ignore
from .context_classifier import ContextClassifier # type: ignore
from .mcp_router import MCPRouter # type: ignore
from .memory_bridge import MemoryBridge # type: ignore
from .parallel_executor import ParallelExecutor # type: ignore
from .pipeline import WisdomPipeline # type: ignore
from .scenario_triggers import ScenarioRegistry # type: ignore
from .soul import SoulController # type: ignore
from .swarm_crons import SwarmCronBridge # type: ignore
from .template_engine import SwarmTemplateEngine # type: ignore

__all__ = [
    "SwarmEngine",
    "WisdomPipeline",
    "SoulController",
    "MCPRouter",
    "MemoryBridge",
    "SwarmCronBridge",
    "SwarmTemplateEngine",
    "ScenarioRegistry",
    "ParallelExecutor",
    "ContextClassifier",
]
