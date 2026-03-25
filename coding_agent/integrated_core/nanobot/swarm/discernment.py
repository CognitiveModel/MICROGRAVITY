"""Discernment Point harvester."""

import uuid
from datetime import datetime

from loguru import logger # type: ignore

from .blob_store import BlobStore # type: ignore
from .graph_store import GraphStore # type: ignore
from .lmdb_store import LMDBStore # type: ignore
from .models.wisdom import DiscernmentPoint, DiscernmentType # type: ignore


class DiscernmentHarvester:
    """
    Harvests subtle distinctions, exceptions, and anomalies from execution logs
    to seed future trigger creation and expand the wisdom graph. 
    Implements Swarm Architecture §7.
    """

    def __init__(self, lmdb: LMDBStore, graph: GraphStore, blobs: BlobStore):
        self.lmdb = lmdb
        self.graph = graph
        self.blobs = blobs

    def harvest(self, content: str, source_action_id: str, dp_type: DiscernmentType) -> DiscernmentPoint:
        """Manually trigger a discernment point harvest (usually via LLM evaluation post-task)."""
        
        dp_id = f"dp_{uuid.uuid4().hex[:10]}" # type: ignore
        
        # 1. Classify seedability. Simple placeholder logic for MVP.
        # In a full system, you'd check LMDB for existing patterns to upgrade seedability.
        seedability = "IMMEDIATE" if "fatal" in content.lower() else "PATTERN"
        
        dp = DiscernmentPoint(
            dp_id=dp_id,
            type=dp_type,
            content=content,
            source_action_id=source_action_id,
            seedability=seedability,
            timestamp=datetime.utcnow()
        )
        
        logger.info("🕸️ Harvested Discernment Point [{}]: {}", dp.type.name, dp.content[:60]) # type: ignore
        
        # 1. State Store active cache (for triggers)
        self.lmdb.put(f"discernment:active:{dp_id}", {
            "id": dp.dp_id,
            "type": dp.type.value,
            "content": dp.content,
            "seedability": dp.seedability,
            "timestamp": dp.timestamp.isoformat()
        })
        
        # 2. Topology Store (Graph linking to Outcome or Action)
        self.graph.create_node(
            label="Discernment_Point",
            node_id=dp_id,
            props={
                "type": dp.type.value,
                "content": dp.content[:300] # type: ignore
            }
        )
        
        # Depending on source, link to action or outcome (assume action for now)
        self.graph.create_edge("SEEDED_FROM", dp_id, source_action_id)
        
        # 3. Raw Blob Dump (for background async analysis)
        self.blobs.write_json("discernment_raw", f"{dp_id}.json", {
            "id": dp.dp_id,
            "type": dp.type.value,
            "content": dp.content,
            "source_action_id": dp.source_action_id,
            "seedability": dp.seedability,
            "timestamp": dp.timestamp.isoformat()
        })
        
        return dp

    def promote_pattern(self, pattern_signature: str) -> None:
        """
        Background maintenance task calls this to evaluate 'PATTERN'
        seedability constraints, promoting them if they hit >3 frequency.
        """
        key = f"sprocess:pattern_hits:{pattern_signature}"
        hits = self.lmdb.get(key, 0)
        hits += 1 # type: ignore
        
        if hits >= 3:
             # Promote pattern to confirmed, create trigger
            logger.info("Pattern {0} promoted to CONFIRMED trigger candidate", pattern_signature) # type: ignore
            self.lmdb.put(f"trigger:candidate:{pattern_signature}", {"status": "AWAITING_REVIEW"})
            # Reset counters
            self.lmdb.delete(key)
        else:
            self.lmdb.put(key, hits)
