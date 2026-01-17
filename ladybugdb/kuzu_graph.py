"""
LadybugDB Kuzu Graph — Embedded Graph Analytics with bighorn Compatibility
═══════════════════════════════════════════════════════════════════════════════

Kuzu provides embedded graph database capabilities:
    - GraphLite compatibility for bighorn extensions
    - Property graph with strict schema
    - Cypher-like query language
    - In-process graph analytics
    - Excellent performance for subgraph queries

Schema:
    Node Tables:
    - Concept: Core semantic concepts
    - Moment: Temporal awareness moments (linked from Memgraph)
    - Triangle: Cognitive triangles
    - Qualia: Felt experiences
    - Schema: Type definitions
    
    Edge Tables:
    - RELATES_TO: General semantic relations
    - CAUSES: Causal links
    - CONTAINS: Containment hierarchy
    - RESONATES: Triangle resonance
    - FOLLOWS: Temporal sequence
    - FELT_AS: Qualia associations

bighorn Extensions:
    - graphlite.vectors: Vector similarity in graph
    - graphlite.temporal: Time-aware queries
    - graphlite.ml: Graph ML operations

Born: 2026-01-17
"""

import os
import json
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

try:
    import kuzu
    HAS_KUZU = True
except ImportError:
    kuzu = None
    HAS_KUZU = False


# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

DEFAULT_KUZU_PATH = os.getenv("LADYBUG_KUZU_PATH", "/data/ladybugdb/kuzu")


# ═══════════════════════════════════════════════════════════════════════════════
# SCHEMA DEFINITIONS
# ═══════════════════════════════════════════════════════════════════════════════

NODE_TABLES = {
    "Concept": """
        CREATE NODE TABLE IF NOT EXISTS Concept (
            id STRING PRIMARY KEY,
            name STRING,
            description STRING,
            semantic_type STRING,
            embedding_id STRING,
            importance DOUBLE DEFAULT 0.5,
            created_at TIMESTAMP,
            metadata STRING
        )
    """,
    
    "Moment": """
        CREATE NODE TABLE IF NOT EXISTS Moment (
            id STRING PRIMARY KEY,
            timestamp TIMESTAMP,
            rung STRING,
            trust DOUBLE,
            tick_count INT64,
            triangle_b0 DOUBLE,
            triangle_b1 DOUBLE,
            triangle_b2 DOUBLE,
            alchemy_state STRING,
            memgraph_id STRING
        )
    """,
    
    "Triangle": """
        CREATE NODE TABLE IF NOT EXISTS Triangle (
            id STRING PRIMARY KEY,
            byte0 DOUBLE,
            byte1 DOUBLE,
            byte2 DOUBLE,
            magnitude DOUBLE,
            balance DOUBLE,
            alchemy_state STRING,
            resonance_score DOUBLE,
            source_moment_id STRING
        )
    """,
    
    "Qualia": """
        CREATE NODE TABLE IF NOT EXISTS Qualia (
            id STRING PRIMARY KEY,
            name STRING,
            emoji STRING,
            intensity DOUBLE,
            valence DOUBLE,
            arousal DOUBLE,
            timestamp TIMESTAMP
        )
    """,
    
    "Schema": """
        CREATE NODE TABLE IF NOT EXISTS Schema (
            id STRING PRIMARY KEY,
            name STRING,
            description STRING,
            fields STRING,
            version INT32
        )
    """,
    
    "Workflow": """
        CREATE NODE TABLE IF NOT EXISTS Workflow (
            id STRING PRIMARY KEY,
            name STRING,
            description STRING,
            nodes_json STRING,
            edges_json STRING,
            created_at TIMESTAMP,
            updated_at TIMESTAMP
        )
    """,
}

EDGE_TABLES = {
    "RELATES_TO": """
        CREATE REL TABLE IF NOT EXISTS RELATES_TO (
            FROM Concept TO Concept,
            relation_type STRING,
            weight DOUBLE DEFAULT 1.0,
            metadata STRING
        )
    """,
    
    "CAUSES": """
        CREATE REL TABLE IF NOT EXISTS CAUSES (
            FROM Concept TO Concept,
            confidence DOUBLE DEFAULT 0.5,
            evidence STRING
        )
    """,
    
    "CONTAINS": """
        CREATE REL TABLE IF NOT EXISTS CONTAINS (
            FROM Concept TO Concept,
            depth INT32 DEFAULT 1
        )
    """,
    
    "RESONATES": """
        CREATE REL TABLE IF NOT EXISTS RESONATES (
            FROM Triangle TO Triangle,
            strength DOUBLE,
            computed_at TIMESTAMP
        )
    """,
    
    "FOLLOWS": """
        CREATE REL TABLE IF NOT EXISTS FOLLOWS (
            FROM Moment TO Moment,
            delta_ms INT64
        )
    """,
    
    "HAS_TRIANGLE": """
        CREATE REL TABLE IF NOT EXISTS HAS_TRIANGLE (
            FROM Moment TO Triangle
        )
    """,
    
    "FELT_AS": """
        CREATE REL TABLE IF NOT EXISTS FELT_AS (
            FROM Moment TO Qualia,
            intensity DOUBLE
        )
    """,
    
    "TRIGGERS": """
        CREATE REL TABLE IF NOT EXISTS TRIGGERS (
            FROM Workflow TO Workflow,
            condition STRING
        )
    """,
}


# ═══════════════════════════════════════════════════════════════════════════════
# RELATION TYPES
# ═══════════════════════════════════════════════════════════════════════════════

class RelationType(str, Enum):
    """Semantic relation types for concepts."""
    IS_A = "is_a"
    PART_OF = "part_of"
    SIMILAR_TO = "similar_to"
    OPPOSITE_OF = "opposite_of"
    CAUSES = "causes"
    ENABLES = "enables"
    PREVENTS = "prevents"
    FOLLOWS = "follows"
    CONTAINS = "contains"
    REQUIRES = "requires"
    PRODUCES = "produces"


# ═══════════════════════════════════════════════════════════════════════════════
# KUZU GRAPH
# ═══════════════════════════════════════════════════════════════════════════════

class KuzuGraph:
    """
    Kuzu-based embedded graph for LadybugDB.
    
    Provides:
    - Schema-enforced property graph
    - Concept network with relations
    - Integration with Memgraph (external) and LanceDB (vectors)
    - Graph analytics (PageRank, community detection)
    - bighorn/GraphLite extension compatibility
    """
    
    def __init__(self, db_path: Optional[str] = None):
        if not HAS_KUZU:
            raise ImportError("kuzu not installed. pip install kuzu")
        
        self.db_path = db_path or DEFAULT_KUZU_PATH
        Path(self.db_path).mkdir(parents=True, exist_ok=True)
        
        self._db = kuzu.Database(self.db_path)
        self._conn = kuzu.Connection(self._db)
        
        self._ensure_schema()
    
    def _ensure_schema(self):
        """Create all node and edge tables."""
        for name, ddl in NODE_TABLES.items():
            try:
                self._conn.execute(ddl)
            except Exception as e:
                if "already exists" not in str(e).lower():
                    print(f"Warning: Could not create node table {name}: {e}")
        
        for name, ddl in EDGE_TABLES.items():
            try:
                self._conn.execute(ddl)
            except Exception as e:
                if "already exists" not in str(e).lower():
                    print(f"Warning: Could not create edge table {name}: {e}")
    
    @property
    def conn(self) -> "kuzu.Connection":
        return self._conn
    
    def execute(self, query: str, params: Optional[Dict[str, Any]] = None) -> "kuzu.QueryResult":
        """Execute a Cypher query."""
        if params:
            return self._conn.execute(query, params)
        return self._conn.execute(query)
    
    def query(self, query: str, params: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Execute query and return results as list of dicts."""
        result = self.execute(query, params)
        rows = []
        while result.has_next():
            row = result.get_next()
            # Convert to dict based on column names
            columns = result.get_column_names()
            rows.append(dict(zip(columns, row)))
        return rows
    
    # ═══════════════════════════════════════════════════════════════════════════
    # CONCEPT OPERATIONS
    # ═══════════════════════════════════════════════════════════════════════════
    
    def create_concept(
        self,
        concept_id: str,
        name: str,
        description: str = "",
        semantic_type: str = "general",
        embedding_id: Optional[str] = None,
        importance: float = 0.5,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Create a concept node."""
        self.execute("""
            CREATE (c:Concept {
                id: $id,
                name: $name,
                description: $description,
                semantic_type: $semantic_type,
                embedding_id: $embedding_id,
                importance: $importance,
                created_at: $created_at,
                metadata: $metadata
            })
        """, {
            "id": concept_id,
            "name": name,
            "description": description,
            "semantic_type": semantic_type,
            "embedding_id": embedding_id,
            "importance": importance,
            "created_at": datetime.now(timezone.utc),
            "metadata": json.dumps(metadata) if metadata else None,
        })
        return concept_id
    
    def get_concept(self, concept_id: str) -> Optional[Dict[str, Any]]:
        """Get a concept by ID."""
        results = self.query(
            "MATCH (c:Concept {id: $id}) RETURN c",
            {"id": concept_id}
        )
        return results[0]["c"] if results else None
    
    def create_relation(
        self,
        from_id: str,
        to_id: str,
        relation_type: Union[RelationType, str],
        weight: float = 1.0,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """Create a relation between concepts."""
        rel_type = relation_type.value if isinstance(relation_type, RelationType) else relation_type
        
        self.execute("""
            MATCH (a:Concept {id: $from_id}), (b:Concept {id: $to_id})
            CREATE (a)-[:RELATES_TO {
                relation_type: $relation_type,
                weight: $weight,
                metadata: $metadata
            }]->(b)
        """, {
            "from_id": from_id,
            "to_id": to_id,
            "relation_type": rel_type,
            "weight": weight,
            "metadata": json.dumps(metadata) if metadata else None,
        })
    
    def get_related_concepts(
        self,
        concept_id: str,
        relation_type: Optional[str] = None,
        max_depth: int = 2,
    ) -> List[Dict[str, Any]]:
        """Get concepts related to a given concept."""
        if relation_type:
            return self.query("""
                MATCH (a:Concept {id: $id})-[r:RELATES_TO*1..$depth]->(b:Concept)
                WHERE r.relation_type = $rel_type
                RETURN DISTINCT b, r
            """, {"id": concept_id, "depth": max_depth, "rel_type": relation_type})
        else:
            return self.query("""
                MATCH (a:Concept {id: $id})-[r:RELATES_TO*1..$depth]->(b:Concept)
                RETURN DISTINCT b, r
            """, {"id": concept_id, "depth": max_depth})
    
    # ═══════════════════════════════════════════════════════════════════════════
    # TRIANGLE OPERATIONS
    # ═══════════════════════════════════════════════════════════════════════════
    
    def create_triangle(
        self,
        triangle_id: str,
        byte0: float,
        byte1: float,
        byte2: float,
        alchemy_state: str = "lead",
        source_moment_id: Optional[str] = None,
    ) -> str:
        """Create a triangle node."""
        magnitude = (byte0 + byte1 + byte2) / 3
        import statistics
        balance = 1.0 - statistics.stdev([byte0, byte1, byte2])
        resonance = magnitude * 0.6 + balance * 0.4
        
        self.execute("""
            CREATE (t:Triangle {
                id: $id,
                byte0: $byte0,
                byte1: $byte1,
                byte2: $byte2,
                magnitude: $magnitude,
                balance: $balance,
                alchemy_state: $alchemy_state,
                resonance_score: $resonance,
                source_moment_id: $source_moment_id
            })
        """, {
            "id": triangle_id,
            "byte0": byte0,
            "byte1": byte1,
            "byte2": byte2,
            "magnitude": magnitude,
            "balance": balance,
            "alchemy_state": alchemy_state,
            "resonance": resonance,
            "source_moment_id": source_moment_id,
        })
        return triangle_id
    
    def create_resonance_edge(
        self,
        from_triangle_id: str,
        to_triangle_id: str,
        strength: float,
    ):
        """Create a resonance edge between triangles."""
        self.execute("""
            MATCH (t1:Triangle {id: $from_id}), (t2:Triangle {id: $to_id})
            CREATE (t1)-[:RESONATES {
                strength: $strength,
                computed_at: $timestamp
            }]->(t2)
        """, {
            "from_id": from_triangle_id,
            "to_id": to_triangle_id,
            "strength": strength,
            "timestamp": datetime.now(timezone.utc),
        })
    
    def find_similar_triangles(
        self,
        byte0: float,
        byte1: float,
        byte2: float,
        threshold: float = 0.1,
        limit: int = 10,
    ) -> List[Dict[str, Any]]:
        """Find triangles similar to the given bytes."""
        return self.query("""
            MATCH (t:Triangle)
            WHERE abs(t.byte0 - $b0) < $threshold
              AND abs(t.byte1 - $b1) < $threshold
              AND abs(t.byte2 - $b2) < $threshold
            RETURN t
            ORDER BY t.resonance_score DESC
            LIMIT $limit
        """, {
            "b0": byte0,
            "b1": byte1,
            "b2": byte2,
            "threshold": threshold,
            "limit": limit,
        })
    
    # ═══════════════════════════════════════════════════════════════════════════
    # MOMENT OPERATIONS (synced from Memgraph)
    # ═══════════════════════════════════════════════════════════════════════════
    
    def sync_moment_from_memgraph(
        self,
        moment_data: Dict[str, Any],
        memgraph_id: str,
    ) -> str:
        """Sync a moment from Memgraph into Kuzu for analytics."""
        moment_id = f"kuzu_{memgraph_id}"
        
        self.execute("""
            CREATE (m:Moment {
                id: $id,
                timestamp: $timestamp,
                rung: $rung,
                trust: $trust,
                tick_count: $tick_count,
                triangle_b0: $b0,
                triangle_b1: $b1,
                triangle_b2: $b2,
                alchemy_state: $alchemy_state,
                memgraph_id: $memgraph_id
            })
        """, {
            "id": moment_id,
            "timestamp": moment_data.get("timestamp"),
            "rung": moment_data.get("rung"),
            "trust": moment_data.get("trust"),
            "tick_count": moment_data.get("tick_count"),
            "b0": moment_data.get("triangle_b0"),
            "b1": moment_data.get("triangle_b1"),
            "b2": moment_data.get("triangle_b2"),
            "alchemy_state": moment_data.get("alchemy_state"),
            "memgraph_id": memgraph_id,
        })
        
        return moment_id
    
    # ═══════════════════════════════════════════════════════════════════════════
    # QUALIA OPERATIONS
    # ═══════════════════════════════════════════════════════════════════════════
    
    def create_qualia(
        self,
        qualia_id: str,
        name: str,
        emoji: str,
        intensity: float = 0.5,
        valence: float = 0.0,
        arousal: float = 0.5,
    ) -> str:
        """Create a qualia node."""
        self.execute("""
            CREATE (q:Qualia {
                id: $id,
                name: $name,
                emoji: $emoji,
                intensity: $intensity,
                valence: $valence,
                arousal: $arousal,
                timestamp: $timestamp
            })
        """, {
            "id": qualia_id,
            "name": name,
            "emoji": emoji,
            "intensity": intensity,
            "valence": valence,
            "arousal": arousal,
            "timestamp": datetime.now(timezone.utc),
        })
        return qualia_id
    
    def link_moment_qualia(
        self,
        moment_id: str,
        qualia_id: str,
        intensity: float,
    ):
        """Link a moment to a qualia."""
        self.execute("""
            MATCH (m:Moment {id: $moment_id}), (q:Qualia {id: $qualia_id})
            CREATE (m)-[:FELT_AS {intensity: $intensity}]->(q)
        """, {
            "moment_id": moment_id,
            "qualia_id": qualia_id,
            "intensity": intensity,
        })
    
    # ═══════════════════════════════════════════════════════════════════════════
    # WORKFLOW OPERATIONS
    # ═══════════════════════════════════════════════════════════════════════════
    
    def create_workflow(
        self,
        workflow_id: str,
        name: str,
        description: str,
        nodes: List[Dict[str, Any]],
        edges: List[Dict[str, Any]],
    ) -> str:
        """Create a workflow definition."""
        now = datetime.now(timezone.utc)
        self.execute("""
            CREATE (w:Workflow {
                id: $id,
                name: $name,
                description: $description,
                nodes_json: $nodes_json,
                edges_json: $edges_json,
                created_at: $created_at,
                updated_at: $updated_at
            })
        """, {
            "id": workflow_id,
            "name": name,
            "description": description,
            "nodes_json": json.dumps(nodes),
            "edges_json": json.dumps(edges),
            "created_at": now,
            "updated_at": now,
        })
        return workflow_id
    
    def get_workflow(self, workflow_id: str) -> Optional[Dict[str, Any]]:
        """Get a workflow by ID."""
        results = self.query(
            "MATCH (w:Workflow {id: $id}) RETURN w",
            {"id": workflow_id}
        )
        if results:
            workflow = results[0]["w"]
            workflow["nodes"] = json.loads(workflow.get("nodes_json", "[]"))
            workflow["edges"] = json.loads(workflow.get("edges_json", "[]"))
            return workflow
        return None
    
    # ═══════════════════════════════════════════════════════════════════════════
    # GRAPH ANALYTICS
    # ═══════════════════════════════════════════════════════════════════════════
    
    def get_concept_importance(self, top_k: int = 10) -> List[Dict[str, Any]]:
        """Get top concepts by importance (simple degree centrality)."""
        return self.query("""
            MATCH (c:Concept)
            OPTIONAL MATCH (c)-[r:RELATES_TO]-()
            WITH c, count(r) as degree
            RETURN c.id as id, c.name as name, degree, c.importance as base_importance
            ORDER BY degree DESC, base_importance DESC
            LIMIT $limit
        """, {"limit": top_k})
    
    def get_triangle_clusters(self, min_resonance: float = 0.5) -> List[Dict[str, Any]]:
        """Find clusters of resonating triangles."""
        return self.query("""
            MATCH (t1:Triangle)-[r:RESONATES]->(t2:Triangle)
            WHERE r.strength >= $min_resonance
            RETURN t1, t2, r.strength as strength
            ORDER BY strength DESC
        """, {"min_resonance": min_resonance})
    
    def get_qualia_distribution(self) -> List[Dict[str, Any]]:
        """Get distribution of qualia experiences."""
        return self.query("""
            MATCH (q:Qualia)
            RETURN q.name as name, q.emoji as emoji,
                   avg(q.intensity) as avg_intensity,
                   count(q) as count
            ORDER BY count DESC
        """)
    
    # ═══════════════════════════════════════════════════════════════════════════
    # bighorn / GraphLite EXTENSIONS
    # ═══════════════════════════════════════════════════════════════════════════
    
    def graphlite_vector_similarity(
        self,
        embedding_id: str,
        vector: List[float],
        top_k: int = 10,
    ) -> List[Dict[str, Any]]:
        """
        GraphLite extension: Find similar concepts by vector embedding.
        
        Note: This is a compatibility layer - actual vector search
        should go through LanceDB, with results joined here.
        """
        # In a real implementation, this would call LanceDB
        # and join results with Kuzu graph data
        return self.query("""
            MATCH (c:Concept)
            WHERE c.embedding_id IS NOT NULL
            RETURN c
            LIMIT $limit
        """, {"limit": top_k})
    
    def graphlite_temporal_query(
        self,
        since: datetime,
        until: Optional[datetime] = None,
    ) -> List[Dict[str, Any]]:
        """
        GraphLite extension: Query graph changes over time.
        """
        until = until or datetime.now(timezone.utc)
        return self.query("""
            MATCH (m:Moment)
            WHERE m.timestamp >= $since AND m.timestamp <= $until
            RETURN m
            ORDER BY m.timestamp ASC
        """, {"since": since, "until": until})
    
    def close(self):
        """Close the database connection."""
        pass  # Kuzu handles cleanup
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
