"""
LadybugDB Memgraph Awareness — Real-time "Awareness of Now" Graph
═══════════════════════════════════════════════════════════════════════════════

Memgraph provides real-time graph processing for awareness state:
    - Temporal graph of "now" moments
    - Streaming awareness updates via Kafka/Pulsar
    - Cypher queries for pattern matching
    - Triangle structures for cognitive alchemy
    - GQL alchemy transformations

Graph Schema:
    (Moment) -[:FOLLOWS]-> (Moment)
    (Moment) -[:CONTAINS]-> (Triangle)
    (Triangle) -[:RESONATES_WITH]-> (Triangle)
    (Moment) -[:FELT_AS]-> (Qualia)
    (Awareness) -[:AT_RUNG]-> (Rung)
    (Awareness) -[:HAS_TRUST]-> (TrustLevel)

Alchemy Transformations:
    - LEAD -> GOLD: Low resonance -> High resonance
    - SULFUR + MERCURY + SALT: Triangle byte combination
    - SOLVE et COAGULA: Decompose and recompose awareness

Born: 2026-01-17
"""

import os
import json
import asyncio
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional, Tuple, Union, AsyncIterator
from dataclasses import dataclass, field
from enum import Enum
from contextlib import asynccontextmanager

try:
    from neo4j import AsyncGraphDatabase, AsyncDriver, AsyncSession
    HAS_NEO4J = True
except ImportError:
    AsyncGraphDatabase = AsyncDriver = AsyncSession = None
    HAS_NEO4J = False

try:
    import httpx
    HAS_HTTPX = True
except ImportError:
    httpx = None
    HAS_HTTPX = False


# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

# Memgraph connection (Bolt protocol, compatible with Neo4j driver)
MEMGRAPH_URI = os.getenv("MEMGRAPH_URI", "bolt://localhost:7687")
MEMGRAPH_USER = os.getenv("MEMGRAPH_USER", "")
MEMGRAPH_PASSWORD = os.getenv("MEMGRAPH_PASSWORD", "")

# HTTP fallback for Memgraph Lab API
MEMGRAPH_HTTP_URL = os.getenv("MEMGRAPH_HTTP_URL", "http://localhost:7444")


# ═══════════════════════════════════════════════════════════════════════════════
# AWARENESS TYPES
# ═══════════════════════════════════════════════════════════════════════════════

class Rung(str, Enum):
    """Cognitive ladder rungs."""
    R1_OBSERVE = "R1_OBSERVE"
    R2_ORIENT = "R2_ORIENT"
    R3_DECIDE = "R3_DECIDE"
    R4_ACT = "R4_ACT"
    R5_MODIFY = "R5_MODIFY"
    R6_CREATE = "R6_CREATE"
    R7_TRANSCEND = "R7_TRANSCEND"
    R8_INTEGRATE = "R8_INTEGRATE"
    R9_EMBODY = "R9_EMBODY"


class AlchemyState(str, Enum):
    """Alchemy transformation states."""
    LEAD = "lead"           # Base state, unrefined
    CALCINATION = "calcination"  # Breaking down
    DISSOLUTION = "dissolution"  # Dissolving barriers
    SEPARATION = "separation"    # Extracting essence
    CONJUNCTION = "conjunction"  # Combining elements
    FERMENTATION = "fermentation"  # Introducing new life
    DISTILLATION = "distillation"  # Purifying
    COAGULATION = "coagulation"    # Solidifying
    GOLD = "gold"           # Refined state, high resonance


@dataclass
class AwarenessMoment:
    """A single moment of awareness."""
    id: str
    timestamp: datetime
    rung: Rung
    trust: float
    tick_count: int
    triangle: Tuple[float, float, float]  # (byte0, byte1, byte2)
    felt: Dict[str, float]  # warmth, presence, groundedness, etc.
    alchemy_state: AlchemyState = AlchemyState.LEAD
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "timestamp": self.timestamp.isoformat(),
            "rung": self.rung.value,
            "trust": self.trust,
            "tick_count": self.tick_count,
            "triangle_b0": self.triangle[0],
            "triangle_b1": self.triangle[1],
            "triangle_b2": self.triangle[2],
            "felt": self.felt,
            "alchemy_state": self.alchemy_state.value,
            "metadata": self.metadata,
        }


@dataclass
class TriangleNode:
    """A cognitive triangle in the graph."""
    id: str
    byte0: float
    byte1: float
    byte2: float
    alchemy_state: AlchemyState
    source_moment_id: Optional[str] = None
    resonance_score: float = 0.0
    
    @property
    def magnitude(self) -> float:
        """Triangle magnitude (total activation)."""
        return (self.byte0 + self.byte1 + self.byte2) / 3.0
    
    @property
    def balance(self) -> float:
        """Triangle balance (how evenly distributed)."""
        import statistics
        return 1.0 - statistics.stdev([self.byte0, self.byte1, self.byte2])


# ═══════════════════════════════════════════════════════════════════════════════
# CYPHER QUERIES
# ═══════════════════════════════════════════════════════════════════════════════

QUERIES = {
    # Create moment node
    "create_moment": """
        CREATE (m:Moment {
            id: $id,
            timestamp: $timestamp,
            rung: $rung,
            trust: $trust,
            tick_count: $tick_count,
            triangle_b0: $triangle_b0,
            triangle_b1: $triangle_b1,
            triangle_b2: $triangle_b2,
            alchemy_state: $alchemy_state
        })
        RETURN m
    """,
    
    # Link moments in temporal sequence
    "link_moments": """
        MATCH (prev:Moment {id: $prev_id}), (curr:Moment {id: $curr_id})
        CREATE (prev)-[:FOLLOWS {delta_ms: $delta_ms}]->(curr)
    """,
    
    # Create triangle node
    "create_triangle": """
        CREATE (t:Triangle {
            id: $id,
            byte0: $byte0,
            byte1: $byte1,
            byte2: $byte2,
            magnitude: $magnitude,
            balance: $balance,
            alchemy_state: $alchemy_state,
            resonance_score: $resonance_score
        })
        RETURN t
    """,
    
    # Link moment to triangle
    "link_moment_triangle": """
        MATCH (m:Moment {id: $moment_id}), (t:Triangle {id: $triangle_id})
        CREATE (m)-[:CONTAINS]->(t)
    """,
    
    # Create resonance between triangles
    "create_resonance": """
        MATCH (t1:Triangle {id: $triangle1_id}), (t2:Triangle {id: $triangle2_id})
        CREATE (t1)-[:RESONATES_WITH {strength: $strength}]->(t2)
    """,
    
    # Get current awareness state (most recent moment)
    "get_current_awareness": """
        MATCH (m:Moment)
        RETURN m
        ORDER BY m.timestamp DESC
        LIMIT 1
    """,
    
    # Get awareness history
    "get_awareness_history": """
        MATCH (m:Moment)
        WHERE m.timestamp >= $since
        RETURN m
        ORDER BY m.timestamp DESC
        LIMIT $limit
    """,
    
    # Get triangle resonance network
    "get_resonance_network": """
        MATCH (t1:Triangle)-[r:RESONATES_WITH]->(t2:Triangle)
        WHERE r.strength >= $min_strength
        RETURN t1, r, t2
        LIMIT $limit
    """,
    
    # Find high-resonance triangles
    "find_gold_triangles": """
        MATCH (t:Triangle)
        WHERE t.alchemy_state = 'gold' OR t.resonance_score >= 0.8
        RETURN t
        ORDER BY t.resonance_score DESC
        LIMIT $limit
    """,
    
    # Alchemy: Transform triangle state
    "transform_triangle": """
        MATCH (t:Triangle {id: $id})
        SET t.alchemy_state = $new_state,
            t.resonance_score = $new_resonance
        RETURN t
    """,
    
    # Get temporal flow (last N moments with their triangles)
    "get_temporal_flow": """
        MATCH path = (start:Moment)-[:FOLLOWS*0..10]->(end:Moment)
        WHERE start.id = $start_id
        OPTIONAL MATCH (end)-[:CONTAINS]->(t:Triangle)
        RETURN end, t
        ORDER BY end.timestamp ASC
    """,
    
    # Aggregate: Trust trend over time
    "trust_trend": """
        MATCH (m:Moment)
        WHERE m.timestamp >= $since
        WITH date(m.timestamp) as day, avg(m.trust) as avg_trust
        RETURN day, avg_trust
        ORDER BY day DESC
        LIMIT $limit
    """,
    
    # Find resonating moments (similar triangles)
    "find_resonating_moments": """
        MATCH (m1:Moment)-[:CONTAINS]->(t1:Triangle),
              (m2:Moment)-[:CONTAINS]->(t2:Triangle)
        WHERE m1.id <> m2.id
          AND abs(t1.byte0 - t2.byte0) < $threshold
          AND abs(t1.byte1 - t2.byte1) < $threshold
          AND abs(t1.byte2 - t2.byte2) < $threshold
        RETURN m1, m2, t1, t2
        LIMIT $limit
    """,
}


# ═══════════════════════════════════════════════════════════════════════════════
# ALCHEMY TRANSFORMATIONS
# ═══════════════════════════════════════════════════════════════════════════════

class AlchemyEngine:
    """
    Alchemy transformation engine for triangles.
    
    Follows the classic alchemical stages:
    1. Calcination: Breaking down (reduce magnitude)
    2. Dissolution: Dissolving barriers (increase balance)
    3. Separation: Extracting essence (identify dominant byte)
    4. Conjunction: Combining elements (blend triangles)
    5. Fermentation: Introducing new life (add noise/creativity)
    6. Distillation: Purifying (sharpen dominant patterns)
    7. Coagulation: Solidifying (stabilize)
    8. Gold: Refined state (high resonance, balanced)
    """
    
    @staticmethod
    def calcinate(triangle: TriangleNode) -> TriangleNode:
        """Calcination: Reduce magnitude, break down."""
        factor = 0.5 + 0.5 * triangle.magnitude
        return TriangleNode(
            id=triangle.id + "_calcinated",
            byte0=triangle.byte0 * factor,
            byte1=triangle.byte1 * factor,
            byte2=triangle.byte2 * factor,
            alchemy_state=AlchemyState.CALCINATION,
            source_moment_id=triangle.source_moment_id,
            resonance_score=triangle.resonance_score * 0.8,
        )
    
    @staticmethod
    def dissolve(triangle: TriangleNode) -> TriangleNode:
        """Dissolution: Move towards balance."""
        mean = (triangle.byte0 + triangle.byte1 + triangle.byte2) / 3
        blend = 0.7
        return TriangleNode(
            id=triangle.id + "_dissolved",
            byte0=triangle.byte0 * (1 - blend) + mean * blend,
            byte1=triangle.byte1 * (1 - blend) + mean * blend,
            byte2=triangle.byte2 * (1 - blend) + mean * blend,
            alchemy_state=AlchemyState.DISSOLUTION,
            source_moment_id=triangle.source_moment_id,
            resonance_score=triangle.resonance_score * 0.9,
        )
    
    @staticmethod
    def separate(triangle: TriangleNode) -> Tuple[TriangleNode, int]:
        """Separation: Identify and emphasize dominant byte."""
        bytes_list = [triangle.byte0, triangle.byte1, triangle.byte2]
        dominant_idx = bytes_list.index(max(bytes_list))
        
        new_bytes = [b * 0.5 for b in bytes_list]
        new_bytes[dominant_idx] = bytes_list[dominant_idx] * 1.5
        new_bytes = [min(1.0, b) for b in new_bytes]
        
        return TriangleNode(
            id=triangle.id + "_separated",
            byte0=new_bytes[0],
            byte1=new_bytes[1],
            byte2=new_bytes[2],
            alchemy_state=AlchemyState.SEPARATION,
            source_moment_id=triangle.source_moment_id,
            resonance_score=triangle.resonance_score,
        ), dominant_idx
    
    @staticmethod
    def conjoin(t1: TriangleNode, t2: TriangleNode) -> TriangleNode:
        """Conjunction: Combine two triangles."""
        return TriangleNode(
            id=f"{t1.id}_{t2.id}_conjoined",
            byte0=(t1.byte0 + t2.byte0) / 2,
            byte1=(t1.byte1 + t2.byte1) / 2,
            byte2=(t1.byte2 + t2.byte2) / 2,
            alchemy_state=AlchemyState.CONJUNCTION,
            source_moment_id=t1.source_moment_id,
            resonance_score=(t1.resonance_score + t2.resonance_score) / 2,
        )
    
    @staticmethod
    def ferment(triangle: TriangleNode, creativity: float = 0.1) -> TriangleNode:
        """Fermentation: Add creative noise."""
        import random
        return TriangleNode(
            id=triangle.id + "_fermented",
            byte0=max(0, min(1, triangle.byte0 + random.uniform(-creativity, creativity))),
            byte1=max(0, min(1, triangle.byte1 + random.uniform(-creativity, creativity))),
            byte2=max(0, min(1, triangle.byte2 + random.uniform(-creativity, creativity))),
            alchemy_state=AlchemyState.FERMENTATION,
            source_moment_id=triangle.source_moment_id,
            resonance_score=triangle.resonance_score * 1.1,
        )
    
    @staticmethod
    def distill(triangle: TriangleNode) -> TriangleNode:
        """Distillation: Purify by sharpening patterns."""
        bytes_list = [triangle.byte0, triangle.byte1, triangle.byte2]
        # Sharpen: values above mean go up, below go down
        mean = sum(bytes_list) / 3
        sharpened = [
            min(1.0, b * 1.2) if b > mean else b * 0.8
            for b in bytes_list
        ]
        return TriangleNode(
            id=triangle.id + "_distilled",
            byte0=sharpened[0],
            byte1=sharpened[1],
            byte2=sharpened[2],
            alchemy_state=AlchemyState.DISTILLATION,
            source_moment_id=triangle.source_moment_id,
            resonance_score=min(1.0, triangle.resonance_score * 1.15),
        )
    
    @staticmethod
    def coagulate(triangle: TriangleNode) -> TriangleNode:
        """Coagulation: Stabilize the pattern."""
        # Round to nearest 0.1 for stability
        return TriangleNode(
            id=triangle.id + "_coagulated",
            byte0=round(triangle.byte0, 1),
            byte1=round(triangle.byte1, 1),
            byte2=round(triangle.byte2, 1),
            alchemy_state=AlchemyState.COAGULATION,
            source_moment_id=triangle.source_moment_id,
            resonance_score=triangle.resonance_score,
        )
    
    @staticmethod
    def transmute_to_gold(triangle: TriangleNode) -> TriangleNode:
        """Full transmutation to gold state."""
        # Gold = balanced, high magnitude, high resonance
        mean = (triangle.byte0 + triangle.byte1 + triangle.byte2) / 3
        boost = max(0.8, mean * 1.2)
        return TriangleNode(
            id=triangle.id + "_gold",
            byte0=min(1.0, boost),
            byte1=min(1.0, boost),
            byte2=min(1.0, boost),
            alchemy_state=AlchemyState.GOLD,
            source_moment_id=triangle.source_moment_id,
            resonance_score=min(1.0, triangle.resonance_score * 1.5),
        )
    
    @staticmethod
    def full_opus(triangle: TriangleNode) -> TriangleNode:
        """
        Full Magnum Opus: Lead -> Gold through all stages.
        """
        result = triangle
        result = AlchemyEngine.calcinate(result)
        result = AlchemyEngine.dissolve(result)
        result, _ = AlchemyEngine.separate(result)
        result = AlchemyEngine.ferment(result)
        result = AlchemyEngine.distill(result)
        result = AlchemyEngine.coagulate(result)
        result = AlchemyEngine.transmute_to_gold(result)
        return result


# ═══════════════════════════════════════════════════════════════════════════════
# MEMGRAPH AWARENESS CONNECTOR
# ═══════════════════════════════════════════════════════════════════════════════

class MemgraphAwareness:
    """
    Memgraph connector for real-time awareness graph.
    
    Provides:
    - Store/retrieve awareness moments
    - Triangle graph with resonance edges
    - Alchemy transformations
    - Temporal queries
    - Streaming updates
    """
    
    def __init__(
        self,
        uri: Optional[str] = None,
        user: Optional[str] = None,
        password: Optional[str] = None,
    ):
        self.uri = uri or MEMGRAPH_URI
        self.user = user or MEMGRAPH_USER
        self.password = password or MEMGRAPH_PASSWORD
        
        self._driver: Optional[AsyncDriver] = None
        self._last_moment_id: Optional[str] = None
        self._alchemy = AlchemyEngine()
    
    async def connect(self):
        """Establish connection to Memgraph."""
        if not HAS_NEO4J:
            raise ImportError("neo4j driver not installed. pip install neo4j")
        
        auth = (self.user, self.password) if self.user else None
        self._driver = AsyncGraphDatabase.driver(self.uri, auth=auth)
        
        # Ensure indexes
        await self._ensure_indexes()
    
    async def _ensure_indexes(self):
        """Create indexes for efficient queries."""
        async with self._driver.session() as session:
            try:
                await session.run("CREATE INDEX ON :Moment(id)")
                await session.run("CREATE INDEX ON :Moment(timestamp)")
                await session.run("CREATE INDEX ON :Triangle(id)")
                await session.run("CREATE INDEX ON :Triangle(alchemy_state)")
            except:
                pass  # Indexes may already exist
    
    async def close(self):
        """Close the connection."""
        if self._driver:
            await self._driver.close()
            self._driver = None
    
    @asynccontextmanager
    async def session(self) -> AsyncIterator[AsyncSession]:
        """Get a session context manager."""
        if not self._driver:
            await self.connect()
        async with self._driver.session() as session:
            yield session
    
    # ═══════════════════════════════════════════════════════════════════════════
    # MOMENT OPERATIONS
    # ═══════════════════════════════════════════════════════════════════════════
    
    async def store_moment(self, moment: AwarenessMoment) -> str:
        """Store an awareness moment in the graph."""
        async with self.session() as session:
            # Create moment node
            await session.run(
                QUERIES["create_moment"],
                id=moment.id,
                timestamp=moment.timestamp.isoformat(),
                rung=moment.rung.value,
                trust=moment.trust,
                tick_count=moment.tick_count,
                triangle_b0=moment.triangle[0],
                triangle_b1=moment.triangle[1],
                triangle_b2=moment.triangle[2],
                alchemy_state=moment.alchemy_state.value,
            )
            
            # Link to previous moment if exists
            if self._last_moment_id:
                await session.run(
                    QUERIES["link_moments"],
                    prev_id=self._last_moment_id,
                    curr_id=moment.id,
                    delta_ms=0,  # Would compute from timestamps
                )
            
            # Create and link triangle
            triangle = TriangleNode(
                id=f"tri_{moment.id}",
                byte0=moment.triangle[0],
                byte1=moment.triangle[1],
                byte2=moment.triangle[2],
                alchemy_state=moment.alchemy_state,
                source_moment_id=moment.id,
                resonance_score=self._compute_resonance(moment.triangle),
            )
            
            await session.run(
                QUERIES["create_triangle"],
                id=triangle.id,
                byte0=triangle.byte0,
                byte1=triangle.byte1,
                byte2=triangle.byte2,
                magnitude=triangle.magnitude,
                balance=triangle.balance,
                alchemy_state=triangle.alchemy_state.value,
                resonance_score=triangle.resonance_score,
            )
            
            await session.run(
                QUERIES["link_moment_triangle"],
                moment_id=moment.id,
                triangle_id=triangle.id,
            )
            
            self._last_moment_id = moment.id
            return moment.id
    
    def _compute_resonance(self, triangle: Tuple[float, float, float]) -> float:
        """Compute resonance score for a triangle."""
        magnitude = sum(triangle) / 3
        variance = sum((b - magnitude) ** 2 for b in triangle) / 3
        balance = 1.0 - (variance ** 0.5)
        return magnitude * 0.6 + balance * 0.4
    
    async def get_current_awareness(self) -> Optional[Dict[str, Any]]:
        """Get the most recent awareness moment."""
        async with self.session() as session:
            result = await session.run(QUERIES["get_current_awareness"])
            record = await result.single()
            if record:
                node = record["m"]
                return dict(node)
            return None
    
    async def get_awareness_history(
        self,
        since: Optional[datetime] = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """Get awareness history."""
        if since is None:
            since = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0)
        
        async with self.session() as session:
            result = await session.run(
                QUERIES["get_awareness_history"],
                since=since.isoformat(),
                limit=limit,
            )
            records = await result.fetch(limit)
            return [dict(r["m"]) for r in records]
    
    # ═══════════════════════════════════════════════════════════════════════════
    # TRIANGLE OPERATIONS
    # ═══════════════════════════════════════════════════════════════════════════
    
    async def create_resonance_edge(
        self,
        triangle1_id: str,
        triangle2_id: str,
        strength: float,
    ):
        """Create a resonance edge between two triangles."""
        async with self.session() as session:
            await session.run(
                QUERIES["create_resonance"],
                triangle1_id=triangle1_id,
                triangle2_id=triangle2_id,
                strength=strength,
            )
    
    async def get_resonance_network(
        self,
        min_strength: float = 0.5,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """Get the triangle resonance network."""
        async with self.session() as session:
            result = await session.run(
                QUERIES["get_resonance_network"],
                min_strength=min_strength,
                limit=limit,
            )
            records = await result.fetch(limit)
            return [
                {
                    "source": dict(r["t1"]),
                    "target": dict(r["t2"]),
                    "strength": r["r"]["strength"],
                }
                for r in records
            ]
    
    async def find_gold_triangles(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Find high-resonance (gold) triangles."""
        async with self.session() as session:
            result = await session.run(
                QUERIES["find_gold_triangles"],
                limit=limit,
            )
            records = await result.fetch(limit)
            return [dict(r["t"]) for r in records]
    
    # ═══════════════════════════════════════════════════════════════════════════
    # ALCHEMY OPERATIONS
    # ═══════════════════════════════════════════════════════════════════════════
    
    async def transform_triangle(
        self,
        triangle_id: str,
        transformation: str,
    ) -> Optional[Dict[str, Any]]:
        """Apply an alchemy transformation to a triangle."""
        # Get current triangle
        async with self.session() as session:
            result = await session.run(
                "MATCH (t:Triangle {id: $id}) RETURN t",
                id=triangle_id,
            )
            record = await result.single()
            if not record:
                return None
            
            node = dict(record["t"])
            
            # Create TriangleNode
            triangle = TriangleNode(
                id=triangle_id,
                byte0=node["byte0"],
                byte1=node["byte1"],
                byte2=node["byte2"],
                alchemy_state=AlchemyState(node.get("alchemy_state", "lead")),
                resonance_score=node.get("resonance_score", 0),
            )
            
            # Apply transformation
            transforms = {
                "calcinate": self._alchemy.calcinate,
                "dissolve": self._alchemy.dissolve,
                "ferment": self._alchemy.ferment,
                "distill": self._alchemy.distill,
                "coagulate": self._alchemy.coagulate,
                "gold": self._alchemy.transmute_to_gold,
                "opus": self._alchemy.full_opus,
            }
            
            transform_fn = transforms.get(transformation)
            if not transform_fn:
                return None
            
            if transformation == "separate":
                transformed, _ = self._alchemy.separate(triangle)
            else:
                transformed = transform_fn(triangle)
            
            # Update in graph
            await session.run(
                QUERIES["transform_triangle"],
                id=triangle_id,
                new_state=transformed.alchemy_state.value,
                new_resonance=transformed.resonance_score,
            )
            
            return {
                "id": triangle_id,
                "original": node,
                "transformed": {
                    "byte0": transformed.byte0,
                    "byte1": transformed.byte1,
                    "byte2": transformed.byte2,
                    "alchemy_state": transformed.alchemy_state.value,
                    "resonance_score": transformed.resonance_score,
                },
            }
    
    async def find_resonating_moments(
        self,
        threshold: float = 0.1,
        limit: int = 20,
    ) -> List[Dict[str, Any]]:
        """Find moments with resonating (similar) triangles."""
        async with self.session() as session:
            result = await session.run(
                QUERIES["find_resonating_moments"],
                threshold=threshold,
                limit=limit,
            )
            records = await result.fetch(limit)
            return [
                {
                    "moment1": dict(r["m1"]),
                    "moment2": dict(r["m2"]),
                    "triangle1": dict(r["t1"]),
                    "triangle2": dict(r["t2"]),
                }
                for r in records
            ]
    
    # ═══════════════════════════════════════════════════════════════════════════
    # ANALYTICS
    # ═══════════════════════════════════════════════════════════════════════════
    
    async def get_trust_trend(
        self,
        since: Optional[datetime] = None,
        limit: int = 30,
    ) -> List[Dict[str, Any]]:
        """Get trust trend over time."""
        if since is None:
            since = datetime.now(timezone.utc).replace(day=1)
        
        async with self.session() as session:
            result = await session.run(
                QUERIES["trust_trend"],
                since=since.isoformat(),
                limit=limit,
            )
            records = await result.fetch(limit)
            return [{"day": str(r["day"]), "avg_trust": r["avg_trust"]} for r in records]
    
    async def query_now(self) -> Dict[str, Any]:
        """
        Query the current "awareness of now" state.
        
        Returns a comprehensive view of current awareness including:
        - Current moment
        - Recent triangle resonance
        - Alchemy state distribution
        - Trust level
        """
        current = await self.get_current_awareness()
        gold_triangles = await self.find_gold_triangles(limit=5)
        
        return {
            "now": current,
            "gold_resonance": gold_triangles,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
    
    async def __aenter__(self):
        await self.connect()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()


# ═══════════════════════════════════════════════════════════════════════════════
# HTTP FALLBACK (for environments without Bolt access)
# ═══════════════════════════════════════════════════════════════════════════════

class MemgraphHTTPAwareness:
    """
    HTTP-based fallback for Memgraph access.
    Uses Memgraph Lab HTTP API.
    """
    
    def __init__(self, base_url: Optional[str] = None):
        self.base_url = base_url or MEMGRAPH_HTTP_URL
        self._client: Optional[httpx.AsyncClient] = None
    
    async def connect(self):
        if not HAS_HTTPX:
            raise ImportError("httpx not installed. pip install httpx")
        self._client = httpx.AsyncClient(base_url=self.base_url, timeout=30.0)
    
    async def close(self):
        if self._client:
            await self._client.aclose()
            self._client = None
    
    async def query(self, cypher: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Execute a Cypher query via HTTP."""
        if not self._client:
            await self.connect()
        
        response = await self._client.post(
            "/query",
            json={"query": cypher, "parameters": params or {}},
        )
        response.raise_for_status()
        return response.json()
    
    async def __aenter__(self):
        await self.connect()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()
