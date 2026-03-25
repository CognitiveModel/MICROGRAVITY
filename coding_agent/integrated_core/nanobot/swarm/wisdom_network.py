"""Cross-Topic Wisdom Network."""

from loguru import logger # type: ignore

from .graph_store import GraphStore # type: ignore


class WisdomNetwork:
    """
    Interface for querying the topology of multi-domain wisdom.
    Implements Swarm Architecture §9.
    """

    def __init__(self, graph: GraphStore):
        self.graph = graph

    def link_hops(self, source_hop_id: str, target_hop_id: str, relationship: str, reason: str = "") -> None:
        """
        Create a structural link between two independent Wisdom_Hop nodes.
        Relationships: SUPPORTS, CONTRADICTS_WITH, COMPOSES_WITH, HIERARCHY_OF, DEPENDS_ON, SUCCEEDS
        """
        valid = ["SUPPORTS", "CONTRADICTS_WITH", "COMPOSES_WITH", "HIERARCHY_OF", "DEPENDS_ON", "SUCCEEDS"]
        if relationship not in valid:
             raise ValueError(f"Invalid wisdom relationship: {relationship}")
        
        self.graph.create_edge(relationship, source_hop_id, target_hop_id, props={"reason": reason})
        logger.debug("Networked: {} -[{}]-> {}", source_hop_id, relationship, target_hop_id)

    def find_supporting(self, hop_id: str) -> list[dict]:
        """Find hops that SUPPORT this hop, or that this hop SUPPORTS."""
        # Nodes that support this (incoming)
        in_supporters = self.graph.query_neighbors(hop_id, "SUPPORTS", direction="in")
        # Nodes this supports (outgoing)
        out_supporters = self.graph.query_neighbors(hop_id, "SUPPORTS", direction="out")
        return in_supporters + out_supporters

    def find_contradictions(self, hop_id: str) -> list[dict]:
        """Find known contradictions for a hop."""
        return self.graph.query_neighbors(hop_id, "CONTRADICTS_WITH", direction="both")

    def get_wisdom_chain(self, topic_id: str) -> list[dict]:
        """Get the full sequence of hops refining a specific topic."""
        # Assuming we just grab all BELONGS_TO edges pointing to the topic, 
        # then sort by their internal REFINES edges or timestamps.
        # Returning unsorted list for MVP.
        return self.graph.query_neighbors(topic_id, "BELONGS_TO", direction="in")

    def detect_isolated_hops(self) -> list[str]:
        """Maintenance query to find hops with no network connections."""
        # Cypher placeholder: MATCH (w:Wisdom_Hop) WHERE NOT (w)--() RETURN w.id
        # For MVP, we return empty list if not implemented in wrapper
        return []
