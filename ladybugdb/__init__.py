"""
LadybugDB — Unified Database Integration Layer
═══════════════════════════════════════════════════════════════════════════════

Architecture Overview:
    ┌────────────────────────────────────────────────────────────────────────┐
    │                         LadybugDB System                                │
    ├────────────────────────────────────────────────────────────────────────┤
    │  ┌─────────────┐   ┌─────────────┐   ┌─────────────┐   ┌────────────┐ │
    │  │   DuckDB    │   │   LanceDB   │   │  Memgraph   │   │   Kuzu     │ │
    │  │   (Core)    │   │  (VSA 10kd) │   │ (Awareness) │   │ (GraphLite)│ │
    │  └──────┬──────┘   └──────┬──────┘   └──────┬──────┘   └─────┬──────┘ │
    │         │                 │                 │                 │        │
    │         └────────────────┼─────────────────┼─────────────────┘        │
    │                          │                 │                          │
    │                    ┌─────┴─────────────────┴─────┐                    │
    │                    │    Integration Layer        │                    │
    │                    │  - Quadrant Addressing      │                    │
    │                    │  - Triangle Alchemy         │                    │
    │                    │  - VSA Resonance            │                    │
    │                    └─────────────┬───────────────┘                    │
    │                                  │                                    │
    │                    ┌─────────────┴───────────────┐                    │
    │                    │      ai_flow Workflows      │                    │
    │                    │   (N8N-style automation)    │                    │
    │                    └─────────────────────────────┘                    │
    └────────────────────────────────────────────────────────────────────────┘

Components:
    1. DuckDB Core (ladybugdb.core)
       - Analytical SQL engine
       - Extension support for bighorn compatibility
       - OLAP queries over awareness data
       
    2. LanceDB VSA (ladybugdb.lance_vsa)
       - 10,000-dimensional vector storage
       - Quadrant addressing [0:9999, 0:9999]
       - Resonance-based semantic search
       
    3. Memgraph GQL (ladybugdb.memgraph_awareness)
       - Real-time "awareness of now" graph
       - Cypher queries for temporal state
       - Streaming awareness updates
       
    4. Kuzu GraphLite (ladybugdb.kuzu_graph)
       - Embedded graph analytics
       - bighorn extension compatibility
       - Property graph with schema
       
    5. Triangle Alchemy (ladybugdb.triangles)
       - 3-byte cognitive triangles
       - Alchemy transformations
       - Graph pattern matching
       
    6. AI Flow (ladybugdb.ai_flow)
       - N8N-style workflow automation
       - Node-based data pipelines
       - Event-driven execution

Usage:
    from ladybugdb import LadybugSystem
    
    # Initialize system
    system = LadybugSystem()
    
    # Store awareness vector in VSA quadrant
    system.vsa.store_quadrant(
        quadrant=(9999, 9999),
        vector=awareness_vector,
        metadata={"type": "resonance", "timestamp": now}
    )
    
    # Query awareness graph
    now_state = await system.awareness.query_now()
    
    # Execute triangle alchemy
    triangle = system.triangles.create(byte0=0.8, byte1=0.6, byte2=0.9)
    transmuted = system.triangles.alchemize(triangle, "gold")
    
    # Run ai_flow workflow
    result = await system.flow.execute("awareness_pipeline", input_data)

Born: 2026-01-17
"""

__version__ = "0.1.0"
__all__ = [
    "LadybugSystem",
    "LadybugCore",
    "LanceVSA",
    "MemgraphAwareness", 
    "KuzuGraph",
    "TriangleAlchemy",
    "AIFlow",
    "Quadrant",
    "Triangle",
]

from .core import LadybugCore
from .lance_vsa import LanceVSA, Quadrant
from .memgraph_awareness import MemgraphAwareness
from .kuzu_graph import KuzuGraph
from .triangles import TriangleAlchemy, Triangle
from .ai_flow import AIFlow
from .system import LadybugSystem
