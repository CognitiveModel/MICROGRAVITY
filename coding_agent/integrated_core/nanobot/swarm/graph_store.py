"""KuzuDB-backed graph store for the Swarm Architecture."""

import os
import sqlite3
from pathlib import Path
from typing import Any

from loguru import logger # type: ignore

# Try import Kuzu, fallback to SQLite graph representation for systems where Kuzu compilation fails
try:
    import kuzu # type: ignore
    KUZU_AVAILABLE = True
except ImportError:
    KUZU_AVAILABLE = False
    logger.warning("KuzuDB not available. Falling back to SQLite graph emulation.") # type: ignore


class GraphStore:
    """
    Graph topology store for topics, wisdom hops, actions, and outcomes.
    Implements the schema from the Swarm Architecture blueprint.
    """

    def __init__(self, workspace: Path, db_name: str = "swarm_graph.kuzu"):
        self.db_path = workspace / db_name
        os.makedirs(self.db_path, exist_ok=True)
        
        self.kuzu_db = None
        self.kuzu_conn = None
        self.sqlite_conn = None

        if KUZU_AVAILABLE:
            try:
                self.kuzu_db = kuzu.Database(str(self.db_path))
                self.kuzu_conn = kuzu.Connection(self.kuzu_db)
                self._init_kuzu_schema()
                logger.debug("Initialized KuzuDB at {}", self.db_path)
            except Exception as e:
                logger.error("Failed to initialize KuzuDB: {}. Falling back to SQLite.", e) # type: ignore
                self._init_sqlite_fallback(workspace / "swarm_graph.sqlite")
        else:
            self._init_sqlite_fallback(workspace / "swarm_graph.sqlite")

    def _init_kuzu_schema(self):
        """Create the node and edge tables required by the swarm architecture."""
        if not self.kuzu_conn:
            return

        # Check if schema exists (Topic table is a good indicator)
        try:
            self.kuzu_conn.execute("MATCH (t:Topic) RETURN count(t)") # type: ignore
            schema_exists = True
        except RuntimeError:
            schema_exists = False

        if schema_exists:
            return

        logger.info("Creating KuzuDB schema for Swarm Architecture...")

        # NODE TABLES
        nodes = [
            "CREATE NODE TABLE Topic (id STRING, name STRING, domain STRING, PRIMARY KEY (id))",
            "CREATE NODE TABLE Subtopic (id STRING, name STRING, PRIMARY KEY (id))",
            "CREATE NODE TABLE Wisdom_Hop (id STRING, content STRING, type STRING, conviction_level STRING, sophistication_level INT64, PRIMARY KEY (id))",
            "CREATE NODE TABLE Action (id STRING, tool_name STRING, payload STRING, PRIMARY KEY (id))",
            "CREATE NODE TABLE Outcome (id STRING, mode STRING, summary STRING, PRIMARY KEY (id))",
            "CREATE NODE TABLE FailureState (id STRING, error_msg STRING, PRIMARY KEY (id))",
            "CREATE NODE TABLE Discernment_Point (id STRING, type STRING, content STRING, PRIMARY KEY (id))",
            "CREATE NODE TABLE SubAgent (id STRING, purpose STRING, status STRING, PRIMARY KEY (id))",
            "CREATE NODE TABLE Trigger_Definition (id STRING, type STRING, condition STRING, PRIMARY KEY (id))",
            "CREATE NODE TABLE Terminology_Anchor (id STRING, term STRING, definition STRING, PRIMARY KEY (id))",
        ]

        # EDGE TABLES
        edges = [
            "CREATE REL TABLE BIFURCATES_TO (FROM Topic TO Subtopic)",
            "CREATE REL TABLE REFINES (FROM Wisdom_Hop TO Wisdom_Hop)",
            "CREATE REL TABLE BELONGS_TO (FROM Wisdom_Hop TO Topic, FROM Wisdom_Hop TO Subtopic)",
            "CREATE REL TABLE JUSTIFIES (FROM Wisdom_Hop TO Action)",
            "CREATE REL TABLE RESULTED_IN (FROM Action TO Outcome)",
            "CREATE REL TABLE CAUSED_FAILURE (FROM Action TO FailureState)",
            "CREATE REL TABLE CONTRADICTS_WITH (FROM Wisdom_Hop TO Wisdom_Hop)",
            "CREATE REL TABLE SUPPORTS (FROM Wisdom_Hop TO Wisdom_Hop)",
            "CREATE REL TABLE COMPOSES_WITH (FROM Wisdom_Hop TO Wisdom_Hop)",
            "CREATE REL TABLE HIERARCHY_OF (FROM Wisdom_Hop TO Wisdom_Hop)",
            "CREATE REL TABLE DEPENDS_ON (FROM Wisdom_Hop TO Wisdom_Hop)",
            "CREATE REL TABLE SUCCEEDS (FROM Wisdom_Hop TO Wisdom_Hop)",
            "CREATE REL TABLE SEEDED_FROM (FROM Discernment_Point TO Outcome, FROM Discernment_Point TO FailureState)",
            "CREATE REL TABLE ANCHORS_TERM (FROM Wisdom_Hop TO Terminology_Anchor)",
            "CREATE REL TABLE DELEGATES_TO (FROM Action TO SubAgent)",
            "CREATE REL TABLE FIRES_TRIGGER (FROM Action TO Trigger_Definition, FROM Outcome TO Trigger_Definition)",
            "CREATE REL TABLE OUTCOME_SYNERGY (FROM Outcome TO Outcome)",
            "CREATE REL TABLE OUTCOME_TRADEOFF (FROM Outcome TO Outcome, magnitude FLOAT)",
        ]

        for query in nodes + edges:
            try:
                self.kuzu_conn.execute(query) # type: ignore
            except RuntimeError as e:
                logger.warning("Schema creation notice: {}", e) # type: ignore

    def _init_sqlite_fallback(self, db_path: Path):
        """Fallback graph representation using SQLite tables."""
        self.sqlite_conn = sqlite3.connect(str(db_path)) # type: ignore
        self.sqlite_conn.row_factory = sqlite3.Row # type: ignore
        
        cursor = self.sqlite_conn.cursor() # type: ignore
        
        # Generic Node table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS graph_nodes (
                id TEXT PRIMARY KEY,
                label TEXT NOT NULL,
                props_json TEXT
            )
        ''')
        
        # Generic Edge table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS graph_edges (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                source_id TEXT NOT NULL,
                target_id TEXT NOT NULL,
                rel_type TEXT NOT NULL,
                props_json TEXT,
                FOREIGN KEY (source_id) REFERENCES graph_nodes(id),
                FOREIGN KEY (target_id) REFERENCES graph_nodes(id)
            )
        ''')
        
        # Indices for faster graph traversal
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_edge_source ON graph_edges(source_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_edge_target ON graph_edges(target_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_edge_type ON graph_edges(rel_type)")
        
        self.sqlite_conn.commit() # type: ignore
        logger.debug("Initialized SQLite graph fallback at {}", db_path)

    def execute(self, cypher: str, parameters: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        """
        Execute a cypher query. 
        If running KuzuDB, returns results. 
        If running SQLite, raises NotImplementedError (for now - simple wrappers provided instead).
        """
        if self.kuzu_conn:
            result = self.kuzu_conn.execute(cypher, parameters or {}) # type: ignore
            
            # Convert result rows into a list of dicts based on column names
            columns = result.get_column_names()
            rows = []
            while result.has_next():
                values = result.get_next()
                row = {col: val for col, val in zip(columns, values)}
                rows.append(row)
            return rows
            
        raise NotImplementedError("Arbitrary Cypher execution not supported in SQLite fallback mode. Use specific methods instead.")

    def create_node(self, label: str, node_id: str, props: dict[str, Any]) -> None:
        """Create a node with the given label and properties."""
        if self.kuzu_conn:
            # Construct standard KuzuDB cypher (e.g., CREATE (n:Topic {id: $id, name: $name}))
            props["id"] = node_id
            prop_str = ", ".join([f"{k}: ${k}" for k in props.keys()])
            query = f"CREATE (n:{label} {{{prop_str}}})"
            try:
                self.kuzu_conn.execute(query, props) # type: ignore
            except RuntimeError as e:
                # If node exists, Kuzu throws duplicate key error. Handle or log.
                logger.warning("Node creation warning: {}", e) # type: ignore
        elif self.sqlite_conn:
            import json
            cursor = self.sqlite_conn.cursor() # type: ignore
            try:
                cursor.execute(
                    "INSERT INTO graph_nodes (id, label, props_json) VALUES (?, ?, ?)",
                    (node_id, label, json.dumps(props))
                )
                self.sqlite_conn.commit() # type: ignore
            except sqlite3.IntegrityError:
                # Update existing if needed, or ignore
                cursor.execute(
                    "UPDATE graph_nodes SET props_json = ? WHERE id = ? AND label = ?",
                    (json.dumps(props), node_id, label)
                )
                self.sqlite_conn.commit() # type: ignore

    def create_edge(self, rel_type: str, source_id: str, target_id: str, props: dict[str, Any] | None = None) -> None:
        """Create a directed edge between two existing nodes."""
        props = props or {}
        
        if self.kuzu_conn:
            prop_str = ""
            if props:
                prop_str = " {" + ", ".join([f"{k}: ${k}" for k in props.keys()]) + "}"
                
            # Requires knowing the node labels in Kuzu for MATCH, usually Cypher handles dynamic matching poorly if labels aren't specified.
            # Thus, Swarm Architecture blueprint assumes we either provide labels or query them first.
            # Simplified generic match (assuming Kuzu allows label-less match)
            query = f"MATCH (a), (b) WHERE a.id = $source_id AND b.id = $target_id CREATE (a)-[r:{rel_type}{prop_str}]->(b)"
            params = {"source_id": source_id, "target_id": target_id, **props}
            try:
                self.kuzu_conn.execute(query, params) # type: ignore
            except RuntimeError as e:
                 logger.warning("Edge creation warning: {}", e) # type: ignore
                 
        elif self.sqlite_conn:
            import json
            cursor = self.sqlite_conn.cursor() # type: ignore
            cursor.execute(
                "INSERT INTO graph_edges (source_id, target_id, rel_type, props_json) VALUES (?, ?, ?, ?)",
                (source_id, target_id, rel_type, json.dumps(props))
            )
            self.sqlite_conn.commit() # type: ignore

    def query_neighbors(self, node_id: str, rel_type: str, direction: str = "out") -> list[dict[str, Any]]:
        """Get neighboring nodes. direction is 'out', 'in', or 'both'."""
        if self.kuzu_conn:
            if direction == "out":
                query = f"MATCH (a)-[r:{rel_type}]->(b) WHERE a.id = $node_id RETURN b"
            elif direction == "in":
                query = f"MATCH (a)<-[r:{rel_type}]-(b) WHERE a.id = $node_id RETURN b"
            else:
                query = f"MATCH (a)-[r:{rel_type}]-(b) WHERE a.id = $node_id RETURN b"
                
            res = self.kuzu_conn.execute(query, {"node_id": node_id}) # type: ignore
            neighbors = []
            while res.has_next():
                values = res.get_next()
                if values:
                    # Kuzu returns node dictionary representation
                    neighbors.append(values[0])
            return neighbors
            
        elif self.sqlite_conn:
            import json
            cursor = self.sqlite_conn.cursor() # type: ignore
            if direction == "out":
                cursor.execute("""
                    SELECT n.id, n.label, n.props_json FROM graph_nodes n
                    JOIN graph_edges e ON e.target_id = n.id
                    WHERE e.source_id = ? AND e.rel_type = ?
                """, (node_id, rel_type))
            elif direction == "in":
                cursor.execute("""
                    SELECT n.id, n.label, n.props_json FROM graph_nodes n
                    JOIN graph_edges e ON e.source_id = n.id
                    WHERE e.target_id = ? AND e.rel_type = ?
                """, (node_id, rel_type))
            else:
                 cursor.execute("""
                    SELECT n.id, n.label, n.props_json FROM graph_nodes n
                    JOIN graph_edges e ON (e.source_id = n.id OR e.target_id = n.id)
                    WHERE (e.target_id = ? OR e.source_id = ?) AND e.rel_type = ?
                """, (node_id, node_id, rel_type))
                 
            neighbors = []
            for row in cursor.fetchall():
                props = json.loads(row["props_json"]) if row["props_json"] else {}
                props["_label"] = row["label"] # type: ignore
                props["_id"] = row["id"] # type: ignore
                neighbors.append(props)
            return neighbors
        
        return [] # type: ignore

    def close(self):
        """Close connections."""
        if self.kuzu_conn:
            pass # Kuzu connections don't require explicit closing in current API
        if self.sqlite_conn:
            self.sqlite_conn.close() # type: ignore
