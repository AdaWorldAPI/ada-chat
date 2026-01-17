"""
LadybugDB API — FastAPI Router for System Integration
═══════════════════════════════════════════════════════════════════════════════

REST API endpoints for LadybugDB system.
Integrates with the Ada Chat application.

Endpoints:
    GET  /ladybug/health        - System health check
    GET  /ladybug/now           - Current awareness state
    POST /ladybug/awareness     - Store awareness moment
    POST /ladybug/search        - Search by triangle
    POST /ladybug/alchemy       - Apply alchemy transformation
    GET  /ladybug/flow/list     - List workflows
    POST /ladybug/flow/execute  - Execute workflow
    GET  /ladybug/analytics     - Get analytics

Born: 2026-01-17
"""

import os
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional, Tuple

from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field

from .system import LadybugSystem


# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

# Global system instance (lazy initialized)
_system: Optional[LadybugSystem] = None


def get_system() -> LadybugSystem:
    """Get or create the global LadybugSystem instance."""
    global _system
    if _system is None:
        data_path = os.getenv("LADYBUG_DATA_PATH", "/data/ladybugdb")
        enable_memgraph = os.getenv("LADYBUG_ENABLE_MEMGRAPH", "false").lower() == "true"
        _system = LadybugSystem(
            data_path=data_path,
            enable_memgraph=enable_memgraph,
        )
    return _system


# ═══════════════════════════════════════════════════════════════════════════════
# REQUEST/RESPONSE MODELS
# ═══════════════════════════════════════════════════════════════════════════════

class TriangleInput(BaseModel):
    """Triangle bytes input."""
    byte0: float = Field(..., ge=0.0, le=1.0)
    byte1: float = Field(..., ge=0.0, le=1.0)
    byte2: float = Field(..., ge=0.0, le=1.0)


class FeltInput(BaseModel):
    """Felt dimensions input."""
    warmth: float = Field(0.5, ge=0.0, le=1.0)
    presence: float = Field(0.5, ge=0.0, le=1.0)
    groundedness: float = Field(0.5, ge=0.0, le=1.0)
    arousal: Optional[float] = Field(None, ge=0.0, le=1.0)
    valence: Optional[float] = Field(None, ge=-1.0, le=1.0)


class StoreAwarenessRequest(BaseModel):
    """Request to store awareness moment."""
    rung: str = Field(..., description="Cognitive rung (R1-R9)")
    trust: float = Field(..., ge=0.0, le=1.0)
    triangle: TriangleInput
    felt: FeltInput
    tick_count: int = Field(0, ge=0)
    metadata: Optional[Dict[str, Any]] = None


class SearchTriangleRequest(BaseModel):
    """Request to search by triangle."""
    triangle: TriangleInput
    limit: int = Field(10, ge=1, le=100)


class AlchemyRequest(BaseModel):
    """Request to apply alchemy transformation."""
    triangle_id: str
    operation: str = Field("advance", description="Alchemy operation")


class ExecuteWorkflowRequest(BaseModel):
    """Request to execute a workflow."""
    workflow_id: str
    input_data: Optional[Dict[str, Any]] = None


class HealthResponse(BaseModel):
    """Health check response."""
    ok: bool
    service: str
    version: str
    components: Dict[str, bool]
    timestamp: str


class AwarenessNowResponse(BaseModel):
    """Current awareness response."""
    ok: bool
    timestamp: str
    latest_snapshot: Optional[Dict[str, Any]]
    analytics: Optional[Dict[str, Any]]
    memgraph: Optional[Dict[str, Any]]
    resonance_vectors: Optional[int]
    triangle_clusters: Optional[int]


# ═══════════════════════════════════════════════════════════════════════════════
# API ROUTER
# ═══════════════════════════════════════════════════════════════════════════════

router = APIRouter(prefix="/ladybug", tags=["LadybugDB"])


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Health check for LadybugDB system.
    
    Returns status of all components.
    """
    system = get_system()
    
    components = {
        "duckdb": False,
        "lancedb": False,
        "kuzu": False,
        "memgraph": False,
        "ai_flow": False,
    }
    
    try:
        # Check DuckDB
        system.core.execute("SELECT 1")
        components["duckdb"] = True
    except:
        pass
    
    try:
        # Check LanceDB
        system.vsa.db.table_names()
        components["lancedb"] = True
    except:
        pass
    
    try:
        # Check Kuzu
        system.graph.execute("RETURN 1")
        components["kuzu"] = True
    except:
        pass
    
    try:
        # Check AI Flow
        system.flow.list_workflows()
        components["ai_flow"] = True
    except:
        pass
    
    return HealthResponse(
        ok=all([components["duckdb"], components["lancedb"]]),
        service="ladybugdb",
        version="0.1.0",
        components=components,
        timestamp=datetime.now(timezone.utc).isoformat(),
    )


@router.get("/now")
async def get_awareness_now():
    """
    Get current awareness state from all components.
    
    Returns unified view of "now" across:
    - DuckDB: Latest snapshot and analytics
    - Memgraph: Real-time awareness (if enabled)
    - LanceDB: Resonance vector count
    - Kuzu: Triangle cluster count
    """
    system = get_system()
    
    try:
        result = await system.query_now()
        return {"ok": True, **result}
    except Exception as e:
        raise HTTPException(500, f"Failed to query awareness: {str(e)}")


@router.post("/awareness")
async def store_awareness(request: StoreAwarenessRequest):
    """
    Store awareness moment across all components.
    
    Stores in:
    - DuckDB: Analytical snapshot
    - LanceDB: VSA encoding
    - Kuzu: Graph node
    - Memgraph: Real-time graph (if enabled)
    """
    system = get_system()
    
    try:
        result = await system.store_awareness(
            rung=request.rung,
            trust=request.trust,
            triangle=(
                request.triangle.byte0,
                request.triangle.byte1,
                request.triangle.byte2,
            ),
            felt={
                "warmth": request.felt.warmth,
                "presence": request.felt.presence,
                "groundedness": request.felt.groundedness,
                "arousal": request.felt.arousal or 0.5,
                "valence": request.felt.valence or 0.0,
            },
            tick_count=request.tick_count,
            metadata=request.metadata,
        )
        return {"ok": True, **result}
    except Exception as e:
        raise HTTPException(500, f"Failed to store awareness: {str(e)}")


@router.post("/search")
async def search_by_triangle(request: SearchTriangleRequest):
    """
    Search across components by triangle similarity.
    
    Uses VSA encoding to find similar triangles in:
    - LanceDB: Vector similarity
    - Kuzu: Graph pattern matching
    - DuckDB: SQL range queries
    """
    system = get_system()
    
    try:
        result = await system.search_by_triangle(
            byte0=request.triangle.byte0,
            byte1=request.triangle.byte1,
            byte2=request.triangle.byte2,
            limit=request.limit,
        )
        return {"ok": True, **result}
    except Exception as e:
        raise HTTPException(500, f"Search failed: {str(e)}")


@router.post("/alchemy")
async def apply_alchemy(request: AlchemyRequest):
    """
    Apply alchemy transformation to a triangle.
    
    Operations:
    - advance: Move to next alchemical state
    - calcinate, dissolve, separate, conjoin, ferment, distill, coagulate
    - gold: Transform to gold state
    - opus: Full Magnum Opus transformation
    
    Updates all components synchronously.
    """
    system = get_system()
    
    try:
        result = await system.alchemize_triangle(
            triangle_id=request.triangle_id,
            operation=request.operation,
        )
        return {"ok": True, **result}
    except Exception as e:
        raise HTTPException(500, f"Alchemy failed: {str(e)}")


@router.get("/resonance")
async def get_resonance(min_resonance: float = 0.7, limit: int = 20):
    """
    Find high-resonance triangles across all components.
    """
    system = get_system()
    
    try:
        result = await system.search_by_resonance(
            min_resonance=min_resonance,
            limit=limit,
        )
        return {"ok": True, **result}
    except Exception as e:
        raise HTTPException(500, f"Resonance query failed: {str(e)}")


@router.get("/analytics")
async def get_analytics():
    """
    Get analytics from DuckDB.
    
    Returns:
    - Total snapshots
    - Average trust
    - Rung distribution
    - Trust trend over time
    """
    system = get_system()
    
    try:
        analytics = system.core.get_awareness_analytics()
        flow_analytics = system.core.get_flow_analytics()
        
        return {
            "ok": True,
            "awareness": analytics,
            "flows": flow_analytics,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
    except Exception as e:
        raise HTTPException(500, f"Analytics failed: {str(e)}")


# ═══════════════════════════════════════════════════════════════════════════════
# WORKFLOW ENDPOINTS
# ═══════════════════════════════════════════════════════════════════════════════

@router.get("/flow/list")
async def list_workflows():
    """List all registered workflows."""
    system = get_system()
    
    try:
        workflows = system.flow.list_workflows()
        return {"ok": True, "workflows": workflows}
    except Exception as e:
        raise HTTPException(500, f"Failed to list workflows: {str(e)}")


@router.post("/flow/execute")
async def execute_workflow(request: ExecuteWorkflowRequest):
    """
    Execute a workflow.
    
    Available workflows:
    - awareness_pipeline: Process and transform awareness moments
    - resonance_detector: Detect high-resonance triangles
    """
    system = get_system()
    
    try:
        result = await system.run_workflow(
            workflow_id=request.workflow_id,
            input_data=request.input_data,
        )
        return {"ok": True, **result}
    except ValueError as e:
        raise HTTPException(404, str(e))
    except Exception as e:
        raise HTTPException(500, f"Workflow execution failed: {str(e)}")


# ═══════════════════════════════════════════════════════════════════════════════
# QUADRANT ENDPOINTS
# ═══════════════════════════════════════════════════════════════════════════════

@router.get("/quadrant/{x}/{y}")
async def query_quadrant(x: int, y: int, radius: int = 500, limit: int = 20):
    """
    Query vectors near a quadrant location.
    
    Quadrant space: [0:9999, 0:9999]
    
    Named quadrants:
    - [0:4999, 0:4999]: Perception
    - [5000:9999, 0:4999]: Action
    - [0:4999, 5000:9999]: Resonance (thinking substrate)
    - [5000:9999, 5000:9999]: Emergence
    """
    system = get_system()
    
    try:
        from .lance_vsa import Quadrant
        quadrant = Quadrant(x, y)
        
        results = system.vsa.search_quadrant(
            center=quadrant,
            radius=radius,
            limit=limit,
        )
        
        return {
            "ok": True,
            "quadrant": quadrant.address,
            "quadrant_type": quadrant.quadrant_type.value,
            "results": results,
        }
    except Exception as e:
        raise HTTPException(500, f"Quadrant query failed: {str(e)}")


@router.get("/quadrant/resonance")
async def get_resonance_quadrant(limit: int = 10):
    """
    Get vectors from the resonance quadrant [9999:9999].
    
    This is the "thinking substrate" quadrant where
    resonance-based cognitive patterns are stored.
    """
    system = get_system()
    
    try:
        results = system.vsa.get_resonance_vectors(limit=limit)
        return {
            "ok": True,
            "quadrant": "[9999:9999]",
            "quadrant_type": "resonance",
            "results": results,
        }
    except Exception as e:
        raise HTTPException(500, f"Resonance query failed: {str(e)}")


# ═══════════════════════════════════════════════════════════════════════════════
# TRIANGLE ENDPOINTS
# ═══════════════════════════════════════════════════════════════════════════════

@router.post("/triangle/create")
async def create_triangle(triangle: TriangleInput, interpretation: str = "cognitive"):
    """
    Create a new triangle.
    
    Interpretations:
    - cognitive: Attention, Reasoning, Memory
    - emotional: Valence, Arousal, Dominance
    - social: Trust, Warmth, Competence
    - temporal: Past, Present, Future
    - resonance: Query, Key, Value
    """
    system = get_system()
    
    try:
        from .triangles import Triangle, TriangleInterpretation
        
        interp = TriangleInterpretation(interpretation)
        tri = system.triangles.create(
            byte0=triangle.byte0,
            byte1=triangle.byte1,
            byte2=triangle.byte2,
            interpretation=interp,
        )
        
        # Store in Kuzu
        triangle_id = system.graph.create_triangle(
            triangle_id=tri.id,
            byte0=tri.byte0,
            byte1=tri.byte1,
            byte2=tri.byte2,
            alchemy_state=tri.alchemy_state.value,
        )
        
        return {
            "ok": True,
            "triangle": tri.to_dict(),
            "kuzu_id": triangle_id,
        }
    except Exception as e:
        raise HTTPException(500, f"Triangle creation failed: {str(e)}")


@router.get("/triangle/{triangle_id}")
async def get_triangle(triangle_id: str):
    """Get a triangle by ID."""
    system = get_system()
    
    try:
        results = system.graph.query(
            "MATCH (t:Triangle {id: $id}) RETURN t",
            {"id": triangle_id}
        )
        
        if not results:
            raise HTTPException(404, f"Triangle not found: {triangle_id}")
        
        return {"ok": True, "triangle": results[0]["t"]}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"Failed to get triangle: {str(e)}")


@router.get("/triangle/{triangle_id}/lineage")
async def get_triangle_lineage(triangle_id: str):
    """Get the alchemy lineage of a triangle."""
    system = get_system()
    
    try:
        lineage = system.core.get_triangle_lineage(triangle_id)
        return {"ok": True, "lineage": lineage}
    except Exception as e:
        raise HTTPException(500, f"Failed to get lineage: {str(e)}")
