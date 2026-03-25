import pytest
from nanobot.swarm.lmdb_store import LMDBStore
from nanobot.swarm.graph_store import GraphStore
from nanobot.swarm.pipeline import WisdomPipeline
from nanobot.swarm.models.wisdom import PipelinePhase, WisdomHopType
from nanobot.swarm.models.conviction import ConvictionLevel, SophisticationTier

def test_pipeline_phases(tmp_path):
    lmdb = LMDBStore(tmp_path)
    graph = GraphStore(tmp_path)
    pipeline = WisdomPipeline(lmdb, graph)
    
    task_id = "test_task_123"
    pipeline.start_pipeline(task_id, "Find out why sky is blue")
    
    assert pipeline.get_current_phase(task_id) == PipelinePhase.RECEPTION
    
    next_phase = pipeline.advance_phase(task_id)
    assert next_phase == PipelinePhase.BOUNDING
    assert pipeline.get_current_phase(task_id) == PipelinePhase.BOUNDING

def test_hop_serialization(tmp_path):
    lmdb = LMDBStore(tmp_path)
    graph = GraphStore(tmp_path)
    pipeline = WisdomPipeline(lmdb, graph)
    
    task_id = "test_task_456"
    pipeline.start_pipeline(task_id, "Test Topic")
    
    hop = pipeline.serialize_hop(
        task_id=task_id,
        hop_type=WisdomHopType.CHARACTERIZE,
        content="The sky appears blue due to Rayleigh scattering.",
        conviction=ConvictionLevel.GROUNDED,
        sophistication=SophisticationTier.ANALYTICAL
    )
    
    assert hop is not None
    assert hop.hop_level == WisdomHopType.CHARACTERIZE
    
    # Retrieve from LMDB
    data = lmdb.get(f"sprocess:hop:{hop.hop_id}")
    assert data is not None
    assert data["content"] == hop.content
