import pytest
from nanobot.swarm.graph_store import GraphStore

def test_graph_node_creation(tmp_path):
    # Using the sqllite fallback for testing if kuzu isn't installed
    graph = GraphStore(tmp_path)
    
    graph.create_node("Topic", "t1", {"name": "Test Topic", "domain": "test"})
    # Verify by looking at the DB directly or through an error check.
    # Since GraphStore MVP focuses on creation wrappers, we'll just test no errors.
    
def test_graph_edge_creation(tmp_path):
    graph = GraphStore(tmp_path)
    graph.create_node("Wisdom_Hop", "h1", {"content": "hop 1"})
    graph.create_node("Wisdom_Hop", "h2", {"content": "hop 2"})
    
    graph.create_edge("REFINES", "h1", "h2", {"reason": "test"})
    
    # Query neighbor
    neighbors = graph.query_neighbors("h2", "REFINES", direction="in")
    assert len(neighbors) == 1
    assert neighbors[0]["id"] == "h1"

def test_graph_neighbor_querying(tmp_path):
    graph = GraphStore(tmp_path)
    graph.create_node("Topic", "t1", {"name": "root"})
    graph.create_node("Wisdom_Hop", "h1", {"content": "child"})
    
    graph.create_edge("BELONGS_TO", "h1", "t1", {})
    
    out_neighbors = graph.query_neighbors("h1", "BELONGS_TO", direction="out")
    assert len(out_neighbors) == 1
    assert out_neighbors[0]["id"] == "t1"
