"""Wisdom Pipeline Orchestrator."""

import uuid
from datetime import datetime

from loguru import logger # type: ignore

from .graph_store import GraphStore # type: ignore
from .lmdb_store import LMDBStore # type: ignore
from .models.conviction import ConvictionLevel, SophisticationTier # type: ignore
from .models.wisdom import PipelinePhase, WisdomHop, WisdomHopType # type: ignore


class WisdomPipeline:
    """
    Orchestrates the 5-phase Wisdom Pipeline (RECEPTION -> BOUNDING -> CONSTRUCTION -> CONVICTION -> ACTIVATION)
    as defined in Swarm Architecture §3.
    """

    def __init__(self, lmdb: LMDBStore, graph: GraphStore):
        self.lmdb = lmdb
        self.graph = graph

    def start_pipeline(self, task_id: str, initial_intent: str) -> None:
        """Initialize pipeline state for a given task."""
        self.lmdb.put(f"sprocess:pipeline_phase:{task_id}", {
            "phase": PipelinePhase.RECEPTION.value,
            "intent": initial_intent,
            "started_at": datetime.utcnow().isoformat()
        })
        
        # Log to Kuzu topology
        self.graph.create_node(
            label="Topic",
            node_id=f"topic_{task_id}",
            props={"name": f"Task: {task_id}", "domain": "execution"}
        )

    def advance_phase(self, task_id: str) -> PipelinePhase | None:
        """Move task to the next logical phase in the pipeline."""
        state = self.lmdb.get(f"sprocess:pipeline_phase:{task_id}")
        if not state:
            return None
            
        current = PipelinePhase(state["phase"])
        
        try:
            # Simple linear progression for now
            phases = list(PipelinePhase)
            next_idx = phases.index(current) + 1
            if next_idx >= len(phases):
                return current # already at end
                
            next_phase = phases[next_idx]
            state["phase"] = next_phase.value
            state["phase_updated_at"] = datetime.utcnow().isoformat()
            self.lmdb.put(f"sprocess:pipeline_phase:{task_id}", state)
            
            logger.debug("Task {} advanced from {} to {}", task_id, current.name, next_phase.name)
            return next_phase
            
        except ValueError:
            return current

    def get_current_phase(self, task_id: str) -> PipelinePhase | None:
        """Get the active pipeline phase for a task."""
        state = self.lmdb.get(f"sprocess:pipeline_phase:{task_id}")
        if state:
            return PipelinePhase(state["phase"])
        return None

    def serialize_hop(
        self,
        task_id: str,
        hop_type: WisdomHopType,
        content: str,
        conviction: ConvictionLevel,
        sophistication: SophisticationTier,
        parent_hop_id: str | None = None
    ) -> WisdomHop:
        """Record a discrete wisdom hop into both LMDB (fast access) and KuzuDB (topology)."""
        hop_id = f"hop_{uuid.uuid4().hex[:12]}" # type: ignore
        topic_id = f"topic_{task_id}"
        
        hop = WisdomHop(
            hop_id=hop_id,
            hop_level=hop_type,
            content=content,
            topic_id=topic_id,
            conviction_level=conviction,
            sophistication_level=sophistication,
            parent_hop_id=parent_hop_id
        )
        
        # 1. State Store
        self.lmdb.put(f"sprocess:hop:{hop_id}", {
            "id": hop.hop_id,
            "topic_id": topic_id,
            "type": hop.hop_level.value,
            "content": hop.content, # type: ignore
            "conviction": hop.conviction_level.value,
            "sophistication": int(hop.sophistication_level),
            "timestamp": hop.timestamp.isoformat()
        })
        
        # 2. Topology Store
        self.graph.create_node(
            label="Wisdom_Hop",
            node_id=hop_id,
            props={
                "type": hop.hop_level.value,
                "content": hop.content[:200], # store truncation in graph # type: ignore
                "conviction_level": hop.conviction_level.value,
                "sophistication_level": int(hop.sophistication_level)
            }
        )
        
        # Link to topic
        self.graph.create_edge("BELONGS_TO", hop_id, topic_id)
        
        # Link to parent hop if refinement
        if parent_hop_id:
            self.graph.create_edge("REFINES", hop_id, parent_hop_id) # type: ignore
            
        logger.debug("Serialized Wisdom Hop [{}]: {}", hop_type.name, content[:50]) # type: ignore
        return hop

    def get_context_for_generation(self, task_id: str) -> list[dict]:
        """Fetch all hops for the current task to build the LLM's active reasoning state."""
        hops = []
        topic_id = f"topic_{task_id}"
        
        # Scan LMDB for hops belonging to this topic
        for key, val in self.lmdb.prefix_scan("sprocess:hop:"):
            if val.get("topic_id") == topic_id:
                hops.append(val)
                
        # Sort chronologically
        hops.sort(key=lambda x: x.get("timestamp", ""))
        return hops

    def end_pipeline(self, task_id: str) -> None:
        """Clean up pipeline state after ACTIVATION."""
        self.lmdb.delete(f"sprocess:pipeline_phase:{task_id}")
