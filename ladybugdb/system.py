"""
LadybugDB System — Unified Integration Layer
═══════════════════════════════════════════════════════════════════════════════

The LadybugSystem provides a unified interface to all components:
    - DuckDB Core (analytical SQL)
    - LanceDB VSA (10kd vectors)
    - Memgraph Awareness (real-time graph)
    - Kuzu Graph (embedded analytics)
    - Triangle Alchemy (cognitive transformations)
    - AI Flow (workflow automation)

Integration Points:
    ┌─────────────────────────────────────────────────────────────────────────┐
    │                        LadybugSystem                                    │
    │                                                                         │
    │  ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐          │
    │  │  DuckDB  │◄──►│  LanceDB │◄──►│ Memgraph │◄──►│   Kuzu   │          │
    │  │   Core   │    │   VSA    │    │Awareness │    │  Graph   │          │
    │  └────┬─────┘    └────┬─────┘    └────┬─────┘    └────┬─────┘          │
    │       │               │               │               │                 │
    │       └───────────────┴───────────────┴───────────────┘                 │
    │                               │                                         │
    │                    ┌──────────┴──────────┐                              │
    │                    │  Triangle Alchemy   │                              │
    │                    │   + AI Flow         │                              │
    │                    └─────────────────────┘                              │
    │                                                                         │
    │  Integration Features:                                                  │
    │  - Cross-component queries                                              │
    │  - Unified awareness state                                              │
    │  - VSA ↔ Graph synchronization                                          │
    │  - Workflow-driven processing                                           │
    │                                                                         │
    └─────────────────────────────────────────────────────────────────────────┘

Born: 2026-01-17
"""

import os
import asyncio
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
import uuid

from .core import LadybugCore
from .lance_vsa import LanceVSA, Quadrant, VSAOps
from .memgraph_awareness import MemgraphAwareness, AwarenessMoment, Rung, AlchemyState as MemAlchemyState
from .kuzu_graph import KuzuGraph
from .triangles import Triangle, TriangleAlchemy, AlchemyState
from .ai_flow import AIFlow, Workflow, create_awareness_pipeline, create_resonance_detector


# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

DEFAULT_DATA_PATH = os.getenv("LADYBUG_DATA_PATH", "/data/ladybugdb")


# ═══════════════════════════════════════════════════════════════════════════════
# LADYBUG SYSTEM
# ═══════════════════════════════════════════════════════════════════════════════

class LadybugSystem:
    """
    Unified LadybugDB system integrating all components.
    
    Usage:
        # Initialize
        system = LadybugSystem()
        
        # Store awareness
        await system.store_awareness(
            rung="R5_MODIFY",
            trust=0.75,
            triangle=(0.8, 0.6, 0.9),
            felt={"warmth": 0.8, "presence": 0.9}
        )
        
        # Query across components
        results = await system.query_awareness_with_vectors(
            "recent high-trust moments"
        )
        
        # Run workflow
        result = await system.flow.execute("awareness_pipeline", input_data)
    """
    
    def __init__(
        self,
        data_path: Optional[str] = None,
        enable_memgraph: bool = True,
        in_memory: bool = False,
    ):
        self.data_path = data_path or DEFAULT_DATA_PATH
        self._enable_memgraph = enable_memgraph
        self._in_memory = in_memory
        
        # Components (lazy initialized)
        self._core: Optional[LadybugCore] = None
        self._vsa: Optional[LanceVSA] = None
        self._awareness: Optional[MemgraphAwareness] = None
        self._graph: Optional[KuzuGraph] = None
        self._triangles: Optional[TriangleAlchemy] = None
        self._flow: Optional[AIFlow] = None
        
        self._initialized = False
    
    def _ensure_data_path(self):
        """Ensure data directory exists."""
        import os
        os.makedirs(self.data_path, exist_ok=True)
    
    @property
    def core(self) -> LadybugCore:
        """Get DuckDB core (lazy init)."""
        if self._core is None:
            self._ensure_data_path()
            self._core = LadybugCore(
                db_path=f"{self.data_path}/core.duckdb",
                in_memory=self._in_memory,
            )
        return self._core
    
    @property
    def vsa(self) -> LanceVSA:
        """Get LanceDB VSA (lazy init)."""
        if self._vsa is None:
            self._ensure_data_path()
            self._vsa = LanceVSA(db_path=f"{self.data_path}/lance")
        return self._vsa
    
    @property
    def awareness(self) -> MemgraphAwareness:
        """Get Memgraph awareness connector (lazy init)."""
        if self._awareness is None:
            self._awareness = MemgraphAwareness()
        return self._awareness
    
    @property
    def graph(self) -> KuzuGraph:
        """Get Kuzu graph (lazy init)."""
        if self._graph is None:
            self._ensure_data_path()
            self._graph = KuzuGraph(db_path=f"{self.data_path}/kuzu")
        return self._graph
    
    @property
    def triangles(self) -> TriangleAlchemy:
        """Get triangle alchemy engine (lazy init)."""
        if self._triangles is None:
            self._triangles = TriangleAlchemy()
        return self._triangles
    
    @property
    def flow(self) -> AIFlow:
        """Get AI Flow engine (lazy init)."""
        if self._flow is None:
            self._flow = AIFlow()
            self._register_default_workflows()
        return self._flow
    
    def _register_default_workflows(self):
        """Register default workflows."""
        self._flow.register_workflow(create_awareness_pipeline())
        self._flow.register_workflow(create_resonance_detector())
    
    # ═══════════════════════════════════════════════════════════════════════════
    # UNIFIED AWARENESS OPERATIONS
    # ═══════════════════════════════════════════════════════════════════════════
    
    async def store_awareness(
        self,
        rung: str,
        trust: float,
        triangle: Tuple[float, float, float],
        felt: Dict[str, float],
        tick_count: int = 0,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, str]:
        """
        Store awareness state across all components.
        
        Stores in:
        - DuckDB: Analytical snapshot
        - Memgraph: Real-time graph (if enabled)
        - Kuzu: Embedded graph analytics
        - LanceDB: VSA encoding of triangle
        
        Returns dict of IDs from each store.
        """
        now = datetime.now(timezone.utc)
        moment_id = f"moment_{uuid.uuid4().hex[:12]}"
        
        result = {"moment_id": moment_id}
        
        # 1. Store in DuckDB
        snapshot_id = self.core.store_awareness_snapshot(
            snapshot_id=moment_id,
            rung=rung,
            trust=trust,
            tick_count=tick_count,
            triangle=triangle,
            felt=felt,
            metadata=metadata,
        )
        result["duckdb_id"] = snapshot_id
        
        # 2. Store triangle in LanceDB as VSA encoding
        vsa_triangle = self.vsa.encode_triangle(triangle[0], triangle[1], triangle[2])
        vector_id = self.vsa.store(
            vector_id=f"tri_{moment_id}",
            vector=vsa_triangle,
            quadrant=Quadrant(9999, 9999),  # Resonance quadrant
            semantic_type="triangle",
            metadata={
                "moment_id": moment_id,
                "rung": rung,
                "trust": trust,
            },
        )
        result["lance_id"] = vector_id
        
        # 3. Store in Kuzu graph
        triangle_id = self.graph.create_triangle(
            triangle_id=f"tri_{moment_id}",
            byte0=triangle[0],
            byte1=triangle[1],
            byte2=triangle[2],
            alchemy_state="lead",
            source_moment_id=moment_id,
        )
        result["kuzu_triangle_id"] = triangle_id
        
        # 4. Store in Memgraph if enabled
        if self._enable_memgraph:
            try:
                moment = AwarenessMoment(
                    id=moment_id,
                    timestamp=now,
                    rung=Rung(rung),
                    trust=trust,
                    tick_count=tick_count,
                    triangle=triangle,
                    felt=felt,
                    alchemy_state=MemAlchemyState.LEAD,
                    metadata=metadata or {},
                )
                memgraph_id = await self.awareness.store_moment(moment)
                result["memgraph_id"] = memgraph_id
            except Exception as e:
                result["memgraph_error"] = str(e)
        
        # 5. Log cognitive event
        self.core.store_cognitive_event(
            event_id=f"event_{moment_id}",
            event_type="awareness_stored",
            source_stack="ladybugdb.system",
            payload={
                "rung": rung,
                "trust": trust,
                "triangle": triangle,
            },
            vector_id=vector_id,
            graph_node_id=triangle_id,
        )
        
        return result
    
    async def query_now(self) -> Dict[str, Any]:
        """
        Query current awareness state from all components.
        
        Returns unified view of "now" across:
        - Memgraph: Real-time awareness
        - DuckDB: Recent analytics
        - LanceDB: High-resonance vectors
        - Kuzu: Graph patterns
        """
        result = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        
        # 1. Memgraph "now"
        if self._enable_memgraph:
            try:
                memgraph_now = await self.awareness.query_now()
                result["memgraph"] = memgraph_now
            except Exception as e:
                result["memgraph_error"] = str(e)
        
        # 2. DuckDB recent analytics
        recent = self.core.get_awareness_history(limit=1)
        if recent:
            result["latest_snapshot"] = recent[0]
        
        analytics = self.core.get_awareness_analytics()
        result["analytics"] = analytics
        
        # 3. LanceDB high-resonance vectors
        try:
            resonance_vectors = self.vsa.get_resonance_vectors(limit=5)
            result["resonance_vectors"] = len(resonance_vectors)
        except:
            pass
        
        # 4. Kuzu graph patterns
        try:
            triangle_clusters = self.graph.get_triangle_clusters(min_resonance=0.5)
            result["triangle_clusters"] = len(triangle_clusters)
        except:
            pass
        
        return result
    
    # ═══════════════════════════════════════════════════════════════════════════
    # CROSS-COMPONENT QUERIES
    # ═══════════════════════════════════════════════════════════════════════════
    
    async def search_by_triangle(
        self,
        byte0: float,
        byte1: float,
        byte2: float,
        limit: int = 10,
    ) -> Dict[str, Any]:
        """
        Search across all components by triangle similarity.
        
        Uses VSA encoding to find similar triangles in:
        - LanceDB: Vector similarity
        - Kuzu: Graph pattern matching
        - DuckDB: SQL range queries
        """
        result = {"query_triangle": (byte0, byte1, byte2)}
        
        # 1. VSA similarity search
        query_vector = self.vsa.encode_triangle(byte0, byte1, byte2)
        vsa_results = self.vsa.search(
            query_vector,
            limit=limit,
            semantic_type="triangle",
        )
        result["vsa_matches"] = vsa_results
        
        # 2. Kuzu pattern search
        kuzu_results = self.graph.find_similar_triangles(
            byte0, byte1, byte2,
            threshold=0.1,
            limit=limit,
        )
        result["kuzu_matches"] = kuzu_results
        
        # 3. DuckDB range query
        duckdb_results = self.core.query("""
            SELECT * FROM triangle_states
            WHERE abs(byte0 - ?) < 0.1
              AND abs(byte1 - ?) < 0.1
              AND abs(byte2 - ?) < 0.1
            ORDER BY timestamp DESC
            LIMIT ?
        """, (byte0, byte1, byte2, limit))
        result["duckdb_matches"] = duckdb_results
        
        return result
    
    async def search_by_resonance(
        self,
        min_resonance: float = 0.7,
        limit: int = 20,
    ) -> Dict[str, Any]:
        """
        Find high-resonance triangles across all components.
        """
        result = {"min_resonance": min_resonance}
        
        # 1. Memgraph gold triangles
        if self._enable_memgraph:
            try:
                gold = await self.awareness.find_gold_triangles(limit=limit)
                result["memgraph_gold"] = gold
            except Exception as e:
                result["memgraph_error"] = str(e)
        
        # 2. Kuzu high-resonance clusters
        clusters = self.graph.get_triangle_clusters(min_resonance=min_resonance)
        result["kuzu_clusters"] = clusters
        
        # 3. DuckDB analytics
        analytics = self.core.query("""
            SELECT 
                alchemy_state,
                COUNT(*) as count,
                AVG(byte0) as avg_b0,
                AVG(byte1) as avg_b1,
                AVG(byte2) as avg_b2
            FROM triangle_states
            GROUP BY alchemy_state
        """)
        result["alchemy_distribution"] = analytics
        
        return result
    
    # ═══════════════════════════════════════════════════════════════════════════
    # ALCHEMY OPERATIONS
    # ═══════════════════════════════════════════════════════════════════════════
    
    async def alchemize_triangle(
        self,
        triangle_id: str,
        operation: str = "advance",
    ) -> Dict[str, Any]:
        """
        Apply alchemy transformation and sync across components.
        
        Args:
            triangle_id: ID of triangle to transform
            operation: Alchemy operation (advance, gold, opus, etc.)
        
        Updates:
        - Kuzu: Graph node
        - Memgraph: Real-time state (if enabled)
        - LanceDB: New VSA encoding
        - DuckDB: Audit log
        """
        result = {"triangle_id": triangle_id, "operation": operation}
        
        # Get current state from Kuzu
        kuzu_results = self.graph.query(
            "MATCH (t:Triangle {id: $id}) RETURN t",
            {"id": triangle_id}
        )
        
        if not kuzu_results:
            result["error"] = f"Triangle not found: {triangle_id}"
            return result
        
        node = kuzu_results[0]["t"]
        
        # Create Triangle object
        triangle = Triangle(
            byte0=node["byte0"],
            byte1=node["byte1"],
            byte2=node["byte2"],
            alchemy_state=AlchemyState(node.get("alchemy_state", "lead")),
            id=triangle_id,
        )
        
        # Apply transformation
        transformed = self.triangles.alchemize(triangle, operation)
        result["original"] = triangle.to_dict()
        result["transformed"] = transformed.to_dict()
        
        # Update Kuzu
        self.graph.execute("""
            MATCH (t:Triangle {id: $id})
            SET t.byte0 = $b0, t.byte1 = $b1, t.byte2 = $b2,
                t.alchemy_state = $state, t.resonance_score = $resonance
        """, {
            "id": triangle_id,
            "b0": transformed.byte0,
            "b1": transformed.byte1,
            "b2": transformed.byte2,
            "state": transformed.alchemy_state.value,
            "resonance": transformed.resonance_score,
        })
        
        # Update Memgraph if enabled
        if self._enable_memgraph:
            try:
                memgraph_result = await self.awareness.transform_triangle(
                    triangle_id,
                    operation,
                )
                result["memgraph_update"] = memgraph_result
            except Exception as e:
                result["memgraph_error"] = str(e)
        
        # Store new VSA encoding
        new_vector = self.vsa.encode_triangle(
            transformed.byte0,
            transformed.byte1,
            transformed.byte2,
        )
        new_vector_id = self.vsa.store(
            vector_id=f"{triangle_id}_{operation}",
            vector=new_vector,
            quadrant=Quadrant(9999, 9999),
            semantic_type=f"triangle_{transformed.alchemy_state.value}",
            metadata={
                "source_triangle": triangle_id,
                "operation": operation,
            },
        )
        result["new_vector_id"] = new_vector_id
        
        # Log to DuckDB
        self.core.store_triangle(
            triangle_id=transformed.id,
            byte0=transformed.byte0,
            byte1=transformed.byte1,
            byte2=transformed.byte2,
            alchemy_state=transformed.alchemy_state.value,
            source_event_id=None,
            parent_triangle_id=triangle_id,
            metadata={"operation": operation},
        )
        
        return result
    
    # ═══════════════════════════════════════════════════════════════════════════
    # WORKFLOW OPERATIONS
    # ═══════════════════════════════════════════════════════════════════════════
    
    async def run_workflow(
        self,
        workflow_id: str,
        input_data: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Execute a workflow with LadybugDB components available.
        
        Injects system components into workflow context.
        """
        context_vars = {
            "ladybug_core": self.core,
            "lance_vsa": self.vsa,
            "kuzu_graph": self.graph,
            "triangle_alchemy": self.triangles,
        }
        
        result = await self.flow.execute(
            workflow_id,
            input_data=input_data,
            variables=context_vars,
        )
        
        return result.to_dict()
    
    # ═══════════════════════════════════════════════════════════════════════════
    # CLEANUP
    # ═══════════════════════════════════════════════════════════════════════════
    
    async def close(self):
        """Close all connections."""
        if self._core:
            self._core.close()
        if self._vsa:
            self._vsa.close()
        if self._awareness:
            await self._awareness.close()
        if self._graph:
            self._graph.close()
    
    async def __aenter__(self):
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()


# ═══════════════════════════════════════════════════════════════════════════════
# CONVENIENCE FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def create_ladybug_system(
    data_path: Optional[str] = None,
    enable_memgraph: bool = True,
    in_memory: bool = False,
) -> LadybugSystem:
    """Create a LadybugSystem instance."""
    return LadybugSystem(
        data_path=data_path,
        enable_memgraph=enable_memgraph,
        in_memory=in_memory,
    )


async def quick_store_awareness(
    rung: str,
    trust: float,
    triangle: Tuple[float, float, float],
    felt: Dict[str, float],
) -> Dict[str, str]:
    """Quick helper to store awareness without system setup."""
    async with LadybugSystem(enable_memgraph=False, in_memory=True) as system:
        return await system.store_awareness(rung, trust, triangle, felt)
