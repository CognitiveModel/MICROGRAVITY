import pytest
from nanobot.swarm.lmdb_store import LMDBStore
from nanobot.swarm.graph_store import GraphStore
from nanobot.swarm.blob_store import BlobStore
from nanobot.swarm.interrupts import InterruptMonitor
from nanobot.swarm.models.wisdom import WisdomInterruptType

def test_heuristic_interrupt_detection(tmp_path):
    lmdb = LMDBStore(tmp_path)
    graph = GraphStore(tmp_path)
    blobs = BlobStore(tmp_path)
    monitor = InterruptMonitor(lmdb, graph, blobs)
    
    task_id = "test_task"
    
    # Should flag SHALLOWNESS
    interrupt = monitor.scan_chunk("We can just simply put the file there.", task_id)
    assert interrupt is not None
    assert interrupt.type == WisdomInterruptType.SHALLOWNESS
    assert interrupt.task_id == task_id
    
    # Should flag FALSE_DICHOTOMY
    interrupt2 = monitor.scan_chunk("You always do this wrong.", task_id)
    assert interrupt2 is not None
    assert interrupt2.type == WisdomInterruptType.FALSE_DICHOTOMY
    
    # Should be clean
    interrupt3 = monitor.scan_chunk("Testing semantic differences between nodes.", task_id)
    assert interrupt3 is None

def test_error_chain_interrupt(tmp_path):
    lmdb = LMDBStore(tmp_path)
    graph = GraphStore(tmp_path)
    blobs = BlobStore(tmp_path)
    monitor = InterruptMonitor(lmdb, graph, blobs)
    
    task_id = "error_task"
    
    monitor.record_error(task_id)
    monitor.record_error(task_id)
    
    # Not yet 3 errors
    assert monitor.scan_chunk("Normal chunk", task_id) is None
    
    # Third error throws the interrupt
    monitor.record_error(task_id)
    interrupt = monitor.scan_chunk("Normal chunk", task_id)
    assert interrupt is not None
    assert interrupt.type == WisdomInterruptType.CONFIRMATION_BIAS
