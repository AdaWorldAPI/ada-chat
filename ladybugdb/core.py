"""
LadybugDB Core — DuckDB Foundation with Extension Support
═══════════════════════════════════════════════════════════════════════════════

DuckDB provides the analytical SQL core for LadybugDB with:
    - OLAP queries over awareness/cognitive data
    - Extension support for bighorn compatibility
    - In-process or file-based storage
    - Arrow/Parquet integration

Extensions:
    - spatial: Geographic awareness data
    - json: Structured cognitive schemas
    - parquet: Persistent awareness snapshots
    - httpfs: Remote data access
    - fts: Full-text search over memories

Born: 2026-01-17
"""

import os
import json
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional, Tuple, Union
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path

try:
    import duckdb
    HAS_DUCKDB = True
except ImportError:
    duckdb = None
    HAS_DUCKDB = False

try:
    import pyarrow as pa
    import pyarrow.parquet as pq
    HAS_ARROW = True
except ImportError:
    pa = pq = None
    HAS_ARROW = False


# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

DEFAULT_DB_PATH = os.getenv("LADYBUG_DB_PATH", "/data/ladybugdb/core.duckdb")
MEMORY_LIMIT = os.getenv("LADYBUG_MEMORY_LIMIT", "4GB")
THREADS = int(os.getenv("LADYBUG_THREADS", "4"))


# ═══════════════════════════════════════════════════════════════════════════════
# BIGHORN EXTENSION REGISTRY
# ═══════════════════════════════════════════════════════════════════════════════

class BigHornExtension(str, Enum):
    """Bighorn-compatible extensions for LadybugDB."""
    SPATIAL = "spatial"
    JSON = "json"
    PARQUET = "parquet"
    HTTPFS = "httpfs"
    FTS = "fts"
    ICU = "icu"  # Unicode/locale support
    TPCH = "tpch"  # Benchmarking
    SUBSTRAIT = "substrait"  # Query plan interchange


@dataclass
class ExtensionConfig:
    """Configuration for a bighorn extension."""
    name: str
    auto_load: bool = True
    settings: Dict[str, Any] = field(default_factory=dict)


EXTENSION_CONFIGS = {
    BigHornExtension.SPATIAL: ExtensionConfig(
        name="spatial",
        auto_load=True,
        settings={"spatial_autodetect_geometry": True}
    ),
    BigHornExtension.JSON: ExtensionConfig(
        name="json",
        auto_load=True,
        settings={"json_stringify_null": False}
    ),
    BigHornExtension.PARQUET: ExtensionConfig(
        name="parquet",
        auto_load=True,
        settings={}
    ),
    BigHornExtension.HTTPFS: ExtensionConfig(
        name="httpfs",
        auto_load=False,  # Load on demand
        settings={}
    ),
    BigHornExtension.FTS: ExtensionConfig(
        name="fts",
        auto_load=True,
        settings={}
    ),
}


# ═══════════════════════════════════════════════════════════════════════════════
# SCHEMA DEFINITIONS
# ═══════════════════════════════════════════════════════════════════════════════

CORE_SCHEMAS = {
    "awareness_snapshots": """
        CREATE TABLE IF NOT EXISTS awareness_snapshots (
            id VARCHAR PRIMARY KEY,
            timestamp TIMESTAMP WITH TIME ZONE NOT NULL,
            rung VARCHAR(20),
            trust DOUBLE,
            tick_count BIGINT,
            triangle_b0 DOUBLE,
            triangle_b1 DOUBLE,
            triangle_b2 DOUBLE,
            felt_warmth DOUBLE,
            felt_presence DOUBLE,
            felt_groundedness DOUBLE,
            metadata JSON,
            created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
        )
    """,
    
    "cognitive_events": """
        CREATE TABLE IF NOT EXISTS cognitive_events (
            id VARCHAR PRIMARY KEY,
            event_type VARCHAR(50) NOT NULL,
            timestamp TIMESTAMP WITH TIME ZONE NOT NULL,
            source_stack VARCHAR(100),
            payload JSON,
            vector_id VARCHAR,  -- Reference to LanceDB vector
            graph_node_id VARCHAR,  -- Reference to Memgraph node
            created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
        )
    """,
    
    "quadrant_index": """
        CREATE TABLE IF NOT EXISTS quadrant_index (
            quadrant_x INT NOT NULL,
            quadrant_y INT NOT NULL,
            vector_id VARCHAR NOT NULL,
            content_hash VARCHAR(64),
            semantic_type VARCHAR(50),
            resonance_score DOUBLE,
            timestamp TIMESTAMP WITH TIME ZONE NOT NULL,
            PRIMARY KEY (quadrant_x, quadrant_y, vector_id)
        )
    """,
    
    "triangle_states": """
        CREATE TABLE IF NOT EXISTS triangle_states (
            id VARCHAR PRIMARY KEY,
            byte0 DOUBLE NOT NULL,
            byte1 DOUBLE NOT NULL,
            byte2 DOUBLE NOT NULL,
            alchemy_state VARCHAR(20),
            source_event_id VARCHAR,
            parent_triangle_id VARCHAR,
            timestamp TIMESTAMP WITH TIME ZONE NOT NULL,
            metadata JSON
        )
    """,
    
    "flow_executions": """
        CREATE TABLE IF NOT EXISTS flow_executions (
            id VARCHAR PRIMARY KEY,
            workflow_id VARCHAR NOT NULL,
            status VARCHAR(20) NOT NULL,
            input_data JSON,
            output_data JSON,
            error_message TEXT,
            started_at TIMESTAMP WITH TIME ZONE,
            completed_at TIMESTAMP WITH TIME ZONE,
            duration_ms BIGINT
        )
    """,
    
    "memory_traces": """
        CREATE TABLE IF NOT EXISTS memory_traces (
            id VARCHAR PRIMARY KEY,
            content TEXT NOT NULL,
            embedding_id VARCHAR,  -- Reference to LanceDB
            importance DOUBLE DEFAULT 0.5,
            access_count INT DEFAULT 0,
            last_accessed TIMESTAMP WITH TIME ZONE,
            decay_factor DOUBLE DEFAULT 1.0,
            created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
        )
    """
}

# Full-text search indexes
FTS_INDEXES = {
    "memory_traces_fts": """
        PRAGMA create_fts_index('memory_traces', 'id', 'content')
    """,
    "cognitive_events_fts": """
        PRAGMA create_fts_index('cognitive_events', 'id', 'payload')
    """
}


# ═══════════════════════════════════════════════════════════════════════════════
# LADYBUG CORE
# ═══════════════════════════════════════════════════════════════════════════════

class LadybugCore:
    """
    DuckDB-based core for LadybugDB.
    
    Provides:
    - OLAP queries over awareness/cognitive data
    - Extension loading (bighorn compatibility)
    - Schema management
    - Arrow/Parquet integration
    """
    
    def __init__(
        self,
        db_path: Optional[str] = None,
        in_memory: bool = False,
        extensions: Optional[List[BigHornExtension]] = None,
    ):
        if not HAS_DUCKDB:
            raise ImportError("duckdb not installed. pip install duckdb")
        
        self.db_path = db_path or DEFAULT_DB_PATH
        self.in_memory = in_memory
        self._conn = None
        self._extensions_loaded = set()
        
        # Default extensions
        self._extensions_to_load = extensions or [
            BigHornExtension.JSON,
            BigHornExtension.PARQUET,
            BigHornExtension.FTS,
        ]
        
        self._initialize()
    
    def _initialize(self):
        """Initialize DuckDB connection and schema."""
        # Create directory if needed
        if not self.in_memory:
            Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Connect
        db_target = ":memory:" if self.in_memory else self.db_path
        self._conn = duckdb.connect(db_target, config={
            "memory_limit": MEMORY_LIMIT,
            "threads": THREADS,
            "default_null_order": "nulls_last",
        })
        
        # Load extensions
        self._load_extensions()
        
        # Create schemas
        self._create_schemas()
    
    def _load_extensions(self):
        """Load bighorn-compatible extensions."""
        for ext in self._extensions_to_load:
            config = EXTENSION_CONFIGS.get(ext)
            if config and config.auto_load:
                try:
                    self._conn.execute(f"INSTALL {config.name}")
                    self._conn.execute(f"LOAD {config.name}")
                    self._extensions_loaded.add(ext)
                    
                    # Apply extension settings
                    for key, value in config.settings.items():
                        self._conn.execute(f"SET {key} = {value}")
                except Exception as e:
                    print(f"Warning: Could not load extension {config.name}: {e}")
    
    def load_extension(self, ext: BigHornExtension) -> bool:
        """Load a specific extension on demand."""
        if ext in self._extensions_loaded:
            return True
        
        config = EXTENSION_CONFIGS.get(ext)
        if not config:
            return False
        
        try:
            self._conn.execute(f"INSTALL {config.name}")
            self._conn.execute(f"LOAD {config.name}")
            self._extensions_loaded.add(ext)
            return True
        except Exception as e:
            print(f"Failed to load extension {config.name}: {e}")
            return False
    
    def _create_schemas(self):
        """Create core tables if they don't exist."""
        for name, ddl in CORE_SCHEMAS.items():
            try:
                self._conn.execute(ddl)
            except Exception as e:
                print(f"Warning: Could not create table {name}: {e}")
        
        # Create FTS indexes if extension loaded
        if BigHornExtension.FTS in self._extensions_loaded:
            for name, ddl in FTS_INDEXES.items():
                try:
                    self._conn.execute(ddl)
                except:
                    pass  # May already exist
    
    @property
    def conn(self) -> "duckdb.DuckDBPyConnection":
        """Get the underlying DuckDB connection."""
        return self._conn
    
    def execute(self, query: str, params: Optional[tuple] = None) -> "duckdb.DuckDBPyResult":
        """Execute a SQL query."""
        if params:
            return self._conn.execute(query, params)
        return self._conn.execute(query)
    
    def query(self, query: str, params: Optional[tuple] = None) -> List[Dict[str, Any]]:
        """Execute query and return results as list of dicts."""
        result = self.execute(query, params)
        columns = [desc[0] for desc in result.description]
        return [dict(zip(columns, row)) for row in result.fetchall()]
    
    # ═══════════════════════════════════════════════════════════════════════════
    # AWARENESS OPERATIONS
    # ═══════════════════════════════════════════════════════════════════════════
    
    def store_awareness_snapshot(
        self,
        snapshot_id: str,
        rung: str,
        trust: float,
        tick_count: int,
        triangle: Tuple[float, float, float],
        felt: Dict[str, float],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Store an awareness snapshot."""
        self.execute("""
            INSERT INTO awareness_snapshots 
            (id, timestamp, rung, trust, tick_count, 
             triangle_b0, triangle_b1, triangle_b2,
             felt_warmth, felt_presence, felt_groundedness, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            snapshot_id,
            datetime.now(timezone.utc),
            rung,
            trust,
            tick_count,
            triangle[0], triangle[1], triangle[2],
            felt.get("warmth", 0.5),
            felt.get("presence", 0.5),
            felt.get("groundedness", 0.5),
            json.dumps(metadata) if metadata else None,
        ))
        return snapshot_id
    
    def get_awareness_history(
        self,
        limit: int = 100,
        since: Optional[datetime] = None,
    ) -> List[Dict[str, Any]]:
        """Query awareness snapshot history."""
        query = """
            SELECT * FROM awareness_snapshots
            WHERE 1=1
        """
        params = []
        
        if since:
            query += " AND timestamp >= ?"
            params.append(since)
        
        query += " ORDER BY timestamp DESC LIMIT ?"
        params.append(limit)
        
        return self.query(query, tuple(params))
    
    def get_awareness_analytics(self) -> Dict[str, Any]:
        """Get analytical summary of awareness over time."""
        return {
            "total_snapshots": self.query(
                "SELECT COUNT(*) as count FROM awareness_snapshots"
            )[0]["count"],
            
            "avg_trust": self.query(
                "SELECT AVG(trust) as avg FROM awareness_snapshots"
            )[0]["avg"],
            
            "rung_distribution": self.query("""
                SELECT rung, COUNT(*) as count 
                FROM awareness_snapshots 
                GROUP BY rung 
                ORDER BY count DESC
            """),
            
            "trust_trend": self.query("""
                SELECT 
                    date_trunc('hour', timestamp) as hour,
                    AVG(trust) as avg_trust,
                    AVG(triangle_b0 + triangle_b1 + triangle_b2) / 3 as avg_triangle
                FROM awareness_snapshots
                GROUP BY date_trunc('hour', timestamp)
                ORDER BY hour DESC
                LIMIT 24
            """),
        }
    
    # ═══════════════════════════════════════════════════════════════════════════
    # COGNITIVE EVENT OPERATIONS
    # ═══════════════════════════════════════════════════════════════════════════
    
    def store_cognitive_event(
        self,
        event_id: str,
        event_type: str,
        source_stack: str,
        payload: Dict[str, Any],
        vector_id: Optional[str] = None,
        graph_node_id: Optional[str] = None,
    ) -> str:
        """Store a cognitive event."""
        self.execute("""
            INSERT INTO cognitive_events
            (id, event_type, timestamp, source_stack, payload, vector_id, graph_node_id)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            event_id,
            event_type,
            datetime.now(timezone.utc),
            source_stack,
            json.dumps(payload),
            vector_id,
            graph_node_id,
        ))
        return event_id
    
    def search_cognitive_events(
        self,
        query_text: str,
        limit: int = 20,
    ) -> List[Dict[str, Any]]:
        """Full-text search over cognitive events."""
        if BigHornExtension.FTS not in self._extensions_loaded:
            # Fallback to LIKE search
            return self.query("""
                SELECT * FROM cognitive_events
                WHERE payload::text ILIKE ?
                ORDER BY timestamp DESC
                LIMIT ?
            """, (f"%{query_text}%", limit))
        
        # Use FTS
        return self.query("""
            SELECT *, fts_main_cognitive_events.match_bm25(id, ?) AS score
            FROM cognitive_events
            WHERE score IS NOT NULL
            ORDER BY score DESC
            LIMIT ?
        """, (query_text, limit))
    
    # ═══════════════════════════════════════════════════════════════════════════
    # QUADRANT INDEX OPERATIONS
    # ═══════════════════════════════════════════════════════════════════════════
    
    def index_quadrant(
        self,
        quadrant_x: int,
        quadrant_y: int,
        vector_id: str,
        content_hash: str,
        semantic_type: str,
        resonance_score: float,
    ):
        """Index a vector in the quadrant system."""
        self.execute("""
            INSERT OR REPLACE INTO quadrant_index
            (quadrant_x, quadrant_y, vector_id, content_hash, 
             semantic_type, resonance_score, timestamp)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            quadrant_x, quadrant_y, vector_id, content_hash,
            semantic_type, resonance_score, datetime.now(timezone.utc)
        ))
    
    def query_quadrant(
        self,
        quadrant_x: int,
        quadrant_y: int,
        radius: int = 100,
    ) -> List[Dict[str, Any]]:
        """Query vectors near a quadrant location."""
        return self.query("""
            SELECT * FROM quadrant_index
            WHERE quadrant_x BETWEEN ? AND ?
              AND quadrant_y BETWEEN ? AND ?
            ORDER BY resonance_score DESC
        """, (
            quadrant_x - radius, quadrant_x + radius,
            quadrant_y - radius, quadrant_y + radius,
        ))
    
    # ═══════════════════════════════════════════════════════════════════════════
    # TRIANGLE STATE OPERATIONS
    # ═══════════════════════════════════════════════════════════════════════════
    
    def store_triangle(
        self,
        triangle_id: str,
        byte0: float,
        byte1: float,
        byte2: float,
        alchemy_state: str = "base",
        source_event_id: Optional[str] = None,
        parent_triangle_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Store a triangle state."""
        self.execute("""
            INSERT INTO triangle_states
            (id, byte0, byte1, byte2, alchemy_state, 
             source_event_id, parent_triangle_id, timestamp, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            triangle_id, byte0, byte1, byte2, alchemy_state,
            source_event_id, parent_triangle_id,
            datetime.now(timezone.utc),
            json.dumps(metadata) if metadata else None,
        ))
        return triangle_id
    
    def get_triangle_lineage(self, triangle_id: str) -> List[Dict[str, Any]]:
        """Get the alchemy lineage of a triangle."""
        return self.query("""
            WITH RECURSIVE lineage AS (
                SELECT * FROM triangle_states WHERE id = ?
                UNION ALL
                SELECT t.* FROM triangle_states t
                JOIN lineage l ON t.id = l.parent_triangle_id
            )
            SELECT * FROM lineage ORDER BY timestamp ASC
        """, (triangle_id,))
    
    # ═══════════════════════════════════════════════════════════════════════════
    # FLOW EXECUTION OPERATIONS
    # ═══════════════════════════════════════════════════════════════════════════
    
    def log_flow_execution(
        self,
        execution_id: str,
        workflow_id: str,
        status: str,
        input_data: Optional[Dict[str, Any]] = None,
        output_data: Optional[Dict[str, Any]] = None,
        error_message: Optional[str] = None,
        started_at: Optional[datetime] = None,
        completed_at: Optional[datetime] = None,
    ) -> str:
        """Log a workflow execution."""
        duration_ms = None
        if started_at and completed_at:
            duration_ms = int((completed_at - started_at).total_seconds() * 1000)
        
        self.execute("""
            INSERT INTO flow_executions
            (id, workflow_id, status, input_data, output_data,
             error_message, started_at, completed_at, duration_ms)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            execution_id, workflow_id, status,
            json.dumps(input_data) if input_data else None,
            json.dumps(output_data) if output_data else None,
            error_message, started_at, completed_at, duration_ms,
        ))
        return execution_id
    
    def get_flow_analytics(self, workflow_id: Optional[str] = None) -> Dict[str, Any]:
        """Get analytics for workflow executions."""
        where = "WHERE workflow_id = ?" if workflow_id else ""
        params = (workflow_id,) if workflow_id else ()
        
        return {
            "total_executions": self.query(
                f"SELECT COUNT(*) as count FROM flow_executions {where}",
                params
            )[0]["count"],
            
            "by_status": self.query(f"""
                SELECT status, COUNT(*) as count
                FROM flow_executions {where}
                GROUP BY status
            """, params),
            
            "avg_duration_ms": self.query(f"""
                SELECT AVG(duration_ms) as avg
                FROM flow_executions 
                {where + ' AND' if where else 'WHERE'} duration_ms IS NOT NULL
            """, params)[0]["avg"],
        }
    
    # ═══════════════════════════════════════════════════════════════════════════
    # MEMORY TRACE OPERATIONS
    # ═══════════════════════════════════════════════════════════════════════════
    
    def store_memory_trace(
        self,
        trace_id: str,
        content: str,
        embedding_id: Optional[str] = None,
        importance: float = 0.5,
    ) -> str:
        """Store a memory trace."""
        self.execute("""
            INSERT INTO memory_traces
            (id, content, embedding_id, importance)
            VALUES (?, ?, ?, ?)
        """, (trace_id, content, embedding_id, importance))
        return trace_id
    
    def access_memory(self, trace_id: str) -> Optional[Dict[str, Any]]:
        """Access a memory trace and update its access metadata."""
        self.execute("""
            UPDATE memory_traces
            SET access_count = access_count + 1,
                last_accessed = ?
            WHERE id = ?
        """, (datetime.now(timezone.utc), trace_id))
        
        results = self.query(
            "SELECT * FROM memory_traces WHERE id = ?",
            (trace_id,)
        )
        return results[0] if results else None
    
    def search_memories(
        self,
        query_text: str,
        limit: int = 20,
    ) -> List[Dict[str, Any]]:
        """Search memory traces."""
        if BigHornExtension.FTS not in self._extensions_loaded:
            return self.query("""
                SELECT * FROM memory_traces
                WHERE content ILIKE ?
                ORDER BY importance DESC, access_count DESC
                LIMIT ?
            """, (f"%{query_text}%", limit))
        
        return self.query("""
            SELECT *, fts_main_memory_traces.match_bm25(id, ?) AS score
            FROM memory_traces
            WHERE score IS NOT NULL
            ORDER BY score * importance DESC
            LIMIT ?
        """, (query_text, limit))
    
    def decay_memories(self, decay_rate: float = 0.99):
        """Apply decay to memory importance."""
        self.execute("""
            UPDATE memory_traces
            SET decay_factor = decay_factor * ?
            WHERE last_accessed < CURRENT_TIMESTAMP - INTERVAL '1 day'
        """, (decay_rate,))
    
    # ═══════════════════════════════════════════════════════════════════════════
    # EXPORT / IMPORT
    # ═══════════════════════════════════════════════════════════════════════════
    
    def export_to_parquet(self, table: str, path: str):
        """Export a table to Parquet format."""
        self.execute(f"COPY {table} TO '{path}' (FORMAT PARQUET)")
    
    def import_from_parquet(self, table: str, path: str):
        """Import data from Parquet format."""
        self.execute(f"COPY {table} FROM '{path}' (FORMAT PARQUET)")
    
    def export_to_arrow(self, query: str) -> "pa.Table":
        """Execute query and return as Arrow table."""
        return self.execute(query).arrow()
    
    def close(self):
        """Close the database connection."""
        if self._conn:
            self._conn.close()
            self._conn = None
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
