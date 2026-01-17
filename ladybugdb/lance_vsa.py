"""
LadybugDB LanceVSA — 10,000-Dimensional Vector Symbolic Architecture
═══════════════════════════════════════════════════════════════════════════════

Vector Symbolic Architecture (VSA) with:
    - 10,000 dimensional space (10kd)
    - Quadrant addressing [0:9999, 0:9999] for 100M addressable locations
    - Resonance-based semantic search
    - Holographic distributed representations

Quadrant System:
    ┌────────────────────────────────────────────────────────────┐
    │                    VSA 10kd Space                          │
    │                                                            │
    │  [0,9999]                              [9999,9999]         │
    │     ┌───────────────────────────────────────┐              │
    │     │  Resonance Quadrant     │  Emergence  │              │
    │     │  (Thinking Substrate)   │  Quadrant   │              │
    │     │                         │             │              │
    │     ├─────────────────────────┼─────────────┤              │
    │     │  Perception             │  Action     │              │
    │     │  Quadrant               │  Quadrant   │              │
    │     │                         │             │              │
    │     └───────────────────────────────────────┘              │
    │  [0,0]                                  [9999,0]           │
    └────────────────────────────────────────────────────────────┘

Dimension Allocation:
    [0:999]      - Temporal encoding (time, sequence)
    [1000:2999]  - Semantic content (meaning)
    [3000:4999]  - Relational structure (edges, graphs)
    [5000:6999]  - Modality encoding (vision, audio, text)
    [7000:8499]  - Context/situation encoding
    [8500:9524]  - Jina embedding space (1024d mapped)
    [9525:9999]  - Meta/attention/salience encoding

Born: 2026-01-17
"""

import os
import json
import hashlib
import numpy as np
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

try:
    import lancedb
    import pyarrow as pa
    HAS_LANCE = True
except ImportError:
    lancedb = pa = None
    HAS_LANCE = False


# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

DEFAULT_LANCE_PATH = os.getenv("LADYBUG_LANCE_PATH", "/data/ladybugdb/lance")
VSA_DIMENSIONS = 10000
QUADRANT_SIZE = 10000  # 0-9999

# Dimension allocation
DIM_TEMPORAL = (0, 999)
DIM_SEMANTIC = (1000, 2999)
DIM_RELATIONAL = (3000, 4999)
DIM_MODALITY = (5000, 6999)
DIM_CONTEXT = (7000, 8499)
DIM_JINA = (8500, 9524)  # 1024 dimensions for Jina embeddings
DIM_META = (9525, 9999)


# ═══════════════════════════════════════════════════════════════════════════════
# QUADRANT TYPES
# ═══════════════════════════════════════════════════════════════════════════════

class QuadrantType(str, Enum):
    """Named quadrant regions in VSA space."""
    PERCEPTION = "perception"      # [0:4999, 0:4999]
    ACTION = "action"              # [5000:9999, 0:4999]
    RESONANCE = "resonance"        # [0:4999, 5000:9999] - Thinking substrate
    EMERGENCE = "emergence"        # [5000:9999, 5000:9999]


@dataclass
class Quadrant:
    """
    A location in VSA quadrant space.
    
    Attributes:
        x: X coordinate (0-9999)
        y: Y coordinate (0-9999)
        quadrant_type: Named region (auto-computed)
    """
    x: int
    y: int
    
    def __post_init__(self):
        self.x = max(0, min(9999, self.x))
        self.y = max(0, min(9999, self.y))
    
    @property
    def quadrant_type(self) -> QuadrantType:
        if self.x < 5000 and self.y < 5000:
            return QuadrantType.PERCEPTION
        elif self.x >= 5000 and self.y < 5000:
            return QuadrantType.ACTION
        elif self.x < 5000 and self.y >= 5000:
            return QuadrantType.RESONANCE
        else:
            return QuadrantType.EMERGENCE
    
    @property
    def address(self) -> str:
        """Get the address string for this quadrant."""
        return f"[{self.x}:{self.y}]"
    
    @classmethod
    def from_address(cls, address: str) -> "Quadrant":
        """Parse quadrant from address string like [9999:9999]."""
        import re
        match = re.match(r"\[(\d+):(\d+)\]", address)
        if match:
            return cls(int(match.group(1)), int(match.group(2)))
        raise ValueError(f"Invalid quadrant address: {address}")
    
    @classmethod
    def resonance(cls) -> "Quadrant":
        """Get the resonance quadrant center [9999:9999]."""
        return cls(9999, 9999)
    
    def distance_to(self, other: "Quadrant") -> float:
        """Euclidean distance to another quadrant."""
        return np.sqrt((self.x - other.x) ** 2 + (self.y - other.y) ** 2)


# ═══════════════════════════════════════════════════════════════════════════════
# VSA OPERATIONS
# ═══════════════════════════════════════════════════════════════════════════════

class VSAOps:
    """
    Holographic/VSA operations for 10kd vectors.
    
    Operations:
    - Binding (⊛): Multiplicative binding for variable-value pairs
    - Bundling (+): Additive superposition for set membership
    - Permutation (ρ): Sequence/order encoding
    - Similarity: Cosine similarity for retrieval
    """
    
    @staticmethod
    def normalize(v: np.ndarray) -> np.ndarray:
        """Normalize vector to unit length."""
        norm = np.linalg.norm(v)
        return v / norm if norm > 0 else v
    
    @staticmethod
    def bind(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """
        Binding operation (⊛) - element-wise multiplication.
        Used for variable-value pairs, role-filler bindings.
        """
        return VSAOps.normalize(a * b)
    
    @staticmethod
    def bundle(vectors: List[np.ndarray]) -> np.ndarray:
        """
        Bundling operation (+) - normalized sum.
        Used for set membership, superposition.
        """
        result = np.sum(vectors, axis=0)
        return VSAOps.normalize(result)
    
    @staticmethod
    def permute(v: np.ndarray, shift: int = 1) -> np.ndarray:
        """
        Permutation operation (ρ) - circular shift.
        Used for sequence/order encoding.
        """
        return np.roll(v, shift)
    
    @staticmethod
    def inverse(v: np.ndarray) -> np.ndarray:
        """
        Inverse operation - for unbinding.
        With element-wise multiplication, inverse is same as original.
        """
        return v
    
    @staticmethod
    def similarity(a: np.ndarray, b: np.ndarray) -> float:
        """Cosine similarity between vectors."""
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))
    
    @staticmethod
    def resonance(query: np.ndarray, memory: np.ndarray, iterations: int = 3) -> np.ndarray:
        """
        Resonance operation - iterative retrieval with cleanup.
        Used for content-addressable memory retrieval.
        """
        result = query
        for _ in range(iterations):
            # Compute similarity
            sim = VSAOps.similarity(result, memory)
            # Blend towards memory based on similarity
            result = VSAOps.normalize(result * (1 - sim * 0.5) + memory * sim * 0.5)
        return result
    
    @staticmethod
    def encode_sequence(vectors: List[np.ndarray]) -> np.ndarray:
        """
        Encode a sequence of vectors with position information.
        Uses permutation to encode order.
        """
        result = np.zeros(VSA_DIMENSIONS)
        for i, v in enumerate(vectors):
            result += VSAOps.permute(v, i)
        return VSAOps.normalize(result)
    
    @staticmethod
    def random_vector(seed: Optional[int] = None) -> np.ndarray:
        """Generate a random VSA vector."""
        if seed is not None:
            np.random.seed(seed)
        v = np.random.randn(VSA_DIMENSIONS)
        return VSAOps.normalize(v)
    
    @staticmethod
    def content_hash_vector(content: str) -> np.ndarray:
        """Generate a deterministic vector from content hash."""
        hash_bytes = hashlib.sha256(content.encode()).digest()
        # Use hash as seed for reproducibility
        seed = int.from_bytes(hash_bytes[:4], 'big')
        return VSAOps.random_vector(seed)


# ═══════════════════════════════════════════════════════════════════════════════
# LANCE VSA STORAGE
# ═══════════════════════════════════════════════════════════════════════════════

class LanceVSA:
    """
    LanceDB-backed Vector Symbolic Architecture storage.
    
    Tables:
    - vec10k_main: Primary 10kd vector storage
    - vec10k_quadrant: Quadrant-indexed vectors
    - vec10k_schema: Schema/type vectors
    - vec10k_temporal: Time-indexed vectors
    """
    
    def __init__(self, db_path: Optional[str] = None):
        if not HAS_LANCE:
            raise ImportError("lancedb not installed. pip install lancedb")
        
        self.db_path = db_path or DEFAULT_LANCE_PATH
        Path(self.db_path).mkdir(parents=True, exist_ok=True)
        
        self._db = lancedb.connect(self.db_path)
        self._ensure_tables()
    
    def _ensure_tables(self):
        """Create tables if they don't exist."""
        # Main vector table
        if "vec10k_main" not in self._db.table_names():
            schema = pa.schema([
                ("id", pa.string()),
                ("vector", pa.list_(pa.float32(), VSA_DIMENSIONS)),
                ("content_hash", pa.string()),
                ("semantic_type", pa.string()),
                ("quadrant_x", pa.int32()),
                ("quadrant_y", pa.int32()),
                ("resonance_score", pa.float32()),
                ("metadata", pa.string()),
                ("timestamp", pa.string()),
            ])
            self._db.create_table("vec10k_main", schema=schema)
        
        # Quadrant-specific table for fast spatial queries
        if "vec10k_quadrant" not in self._db.table_names():
            schema = pa.schema([
                ("id", pa.string()),
                ("vector", pa.list_(pa.float32(), VSA_DIMENSIONS)),
                ("quadrant_type", pa.string()),
                ("x", pa.int32()),
                ("y", pa.int32()),
                ("timestamp", pa.string()),
            ])
            self._db.create_table("vec10k_quadrant", schema=schema)
        
        # Schema vectors (type definitions)
        if "vec10k_schema" not in self._db.table_names():
            schema = pa.schema([
                ("id", pa.string()),
                ("vector", pa.list_(pa.float32(), VSA_DIMENSIONS)),
                ("schema_name", pa.string()),
                ("description", pa.string()),
                ("metadata", pa.string()),
            ])
            self._db.create_table("vec10k_schema", schema=schema)
    
    @property
    def db(self) -> "lancedb.DBConnection":
        return self._db
    
    # ═══════════════════════════════════════════════════════════════════════════
    # VECTOR STORAGE
    # ═══════════════════════════════════════════════════════════════════════════
    
    def store(
        self,
        vector_id: str,
        vector: np.ndarray,
        quadrant: Optional[Quadrant] = None,
        semantic_type: str = "general",
        metadata: Optional[Dict[str, Any]] = None,
        content_hash: Optional[str] = None,
    ) -> str:
        """Store a 10kd vector."""
        if len(vector) != VSA_DIMENSIONS:
            raise ValueError(f"Vector must be {VSA_DIMENSIONS} dimensions, got {len(vector)}")
        
        quadrant = quadrant or Quadrant(5000, 5000)  # Default to center
        
        # Compute resonance score based on quadrant
        resonance = self._compute_resonance_score(vector, quadrant)
        
        tbl = self._db.open_table("vec10k_main")
        tbl.add([{
            "id": vector_id,
            "vector": vector.astype(np.float32).tolist(),
            "content_hash": content_hash or hashlib.sha256(vector.tobytes()).hexdigest()[:16],
            "semantic_type": semantic_type,
            "quadrant_x": quadrant.x,
            "quadrant_y": quadrant.y,
            "resonance_score": resonance,
            "metadata": json.dumps(metadata) if metadata else None,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }])
        
        return vector_id
    
    def store_quadrant(
        self,
        quadrant: Quadrant,
        vector: np.ndarray,
        vector_id: Optional[str] = None,
    ) -> str:
        """Store a vector in a specific quadrant."""
        vector_id = vector_id or f"q_{quadrant.x}_{quadrant.y}_{datetime.now().timestamp()}"
        
        tbl = self._db.open_table("vec10k_quadrant")
        tbl.add([{
            "id": vector_id,
            "vector": vector.astype(np.float32).tolist(),
            "quadrant_type": quadrant.quadrant_type.value,
            "x": quadrant.x,
            "y": quadrant.y,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }])
        
        return vector_id
    
    def _compute_resonance_score(self, vector: np.ndarray, quadrant: Quadrant) -> float:
        """
        Compute resonance score based on vector alignment with quadrant.
        
        Vectors closer to the resonance quadrant [9999,9999] with high
        activation in resonance-related dimensions score higher.
        """
        # Distance to resonance center
        resonance_center = Quadrant.resonance()
        dist = quadrant.distance_to(resonance_center)
        dist_score = 1.0 - (dist / (np.sqrt(2) * QUADRANT_SIZE))
        
        # Activation in resonance dimensions (last 500)
        resonance_dims = vector[9500:10000]
        activation = np.mean(np.abs(resonance_dims))
        
        return float(dist_score * 0.5 + activation * 0.5)
    
    # ═══════════════════════════════════════════════════════════════════════════
    # VECTOR RETRIEVAL
    # ═══════════════════════════════════════════════════════════════════════════
    
    def search(
        self,
        query_vector: np.ndarray,
        limit: int = 10,
        semantic_type: Optional[str] = None,
        quadrant_filter: Optional[QuadrantType] = None,
    ) -> List[Dict[str, Any]]:
        """Search for similar vectors."""
        tbl = self._db.open_table("vec10k_main")
        
        query = tbl.search(query_vector.astype(np.float32).tolist())
        
        if semantic_type:
            query = query.where(f"semantic_type = '{semantic_type}'")
        
        if quadrant_filter:
            if quadrant_filter == QuadrantType.PERCEPTION:
                query = query.where("quadrant_x < 5000 AND quadrant_y < 5000")
            elif quadrant_filter == QuadrantType.ACTION:
                query = query.where("quadrant_x >= 5000 AND quadrant_y < 5000")
            elif quadrant_filter == QuadrantType.RESONANCE:
                query = query.where("quadrant_x < 5000 AND quadrant_y >= 5000")
            elif quadrant_filter == QuadrantType.EMERGENCE:
                query = query.where("quadrant_x >= 5000 AND quadrant_y >= 5000")
        
        return query.limit(limit).to_list()
    
    def search_quadrant(
        self,
        center: Quadrant,
        radius: int = 500,
        limit: int = 20,
    ) -> List[Dict[str, Any]]:
        """Search vectors within a quadrant radius."""
        tbl = self._db.open_table("vec10k_quadrant")
        
        results = tbl.search() \
            .where(f"x BETWEEN {center.x - radius} AND {center.x + radius}") \
            .where(f"y BETWEEN {center.y - radius} AND {center.y + radius}") \
            .limit(limit) \
            .to_list()
        
        return results
    
    def get_resonance_vectors(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get vectors in the resonance quadrant [9999:9999]."""
        return self.search_quadrant(Quadrant.resonance(), radius=1000, limit=limit)
    
    # ═══════════════════════════════════════════════════════════════════════════
    # RESONANCE OPERATIONS
    # ═══════════════════════════════════════════════════════════════════════════
    
    def resonate(
        self,
        query: np.ndarray,
        iterations: int = 3,
        top_k: int = 5,
    ) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
        """
        Perform resonance-based retrieval and vector cleanup.
        
        Returns:
            Tuple of (cleaned vector, matched memories)
        """
        # Find similar vectors
        matches = self.search(query, limit=top_k)
        
        if not matches:
            return query, []
        
        # Bundle matched vectors
        memory_vectors = [np.array(m["vector"]) for m in matches]
        memory = VSAOps.bundle(memory_vectors)
        
        # Perform resonance
        cleaned = VSAOps.resonance(query, memory, iterations)
        
        return cleaned, matches
    
    def store_resonance(
        self,
        content: str,
        jina_embedding: Optional[np.ndarray] = None,
        additional_dims: Optional[Dict[str, np.ndarray]] = None,
    ) -> str:
        """
        Store content in the resonance quadrant with proper dimension encoding.
        
        Args:
            content: Text content to store
            jina_embedding: 1024d Jina embedding (placed in dims 8500:9524)
            additional_dims: Dict of {dim_range: vector} for custom encoding
        """
        # Create base vector from content hash
        vector = VSAOps.content_hash_vector(content)
        
        # Inject Jina embedding if provided
        if jina_embedding is not None:
            if len(jina_embedding) == 1024:
                vector[DIM_JINA[0]:DIM_JINA[1]+1] = jina_embedding
        
        # Inject additional dimension encodings
        if additional_dims:
            for (start, end), dim_vector in additional_dims.items():
                expected_len = end - start + 1
                if len(dim_vector) == expected_len:
                    vector[start:end+1] = dim_vector
        
        # Normalize
        vector = VSAOps.normalize(vector)
        
        # Store in resonance quadrant
        vector_id = f"res_{hashlib.sha256(content.encode()).hexdigest()[:12]}"
        quadrant = Quadrant(9999, 9999)  # Resonance center
        
        return self.store(
            vector_id=vector_id,
            vector=vector,
            quadrant=quadrant,
            semantic_type="resonance",
            metadata={"content_preview": content[:200]},
            content_hash=hashlib.sha256(content.encode()).hexdigest()[:16],
        )
    
    # ═══════════════════════════════════════════════════════════════════════════
    # SCHEMA OPERATIONS
    # ═══════════════════════════════════════════════════════════════════════════
    
    def register_schema(
        self,
        schema_name: str,
        description: str,
        base_vector: Optional[np.ndarray] = None,
    ) -> str:
        """Register a schema vector for type matching."""
        vector = base_vector if base_vector is not None else VSAOps.content_hash_vector(schema_name)
        
        schema_id = f"schema_{schema_name}"
        
        tbl = self._db.open_table("vec10k_schema")
        tbl.add([{
            "id": schema_id,
            "vector": vector.astype(np.float32).tolist(),
            "schema_name": schema_name,
            "description": description,
            "metadata": None,
        }])
        
        return schema_id
    
    def match_schema(self, vector: np.ndarray, top_k: int = 3) -> List[Dict[str, Any]]:
        """Find best matching schemas for a vector."""
        tbl = self._db.open_table("vec10k_schema")
        return tbl.search(vector.astype(np.float32).tolist()).limit(top_k).to_list()
    
    # ═══════════════════════════════════════════════════════════════════════════
    # TRIANGLE ENCODING
    # ═══════════════════════════════════════════════════════════════════════════
    
    def encode_triangle(
        self,
        byte0: float,
        byte1: float,
        byte2: float,
    ) -> np.ndarray:
        """
        Encode a 3-byte triangle into VSA space.
        
        Triangle bytes are encoded in the META dimension range (9525:9999)
        with holographic distribution.
        """
        vector = np.zeros(VSA_DIMENSIONS)
        
        # Encode each byte in different regions
        # Byte0: dims 9525-9674 (150 dims)
        vector[9525:9675] = np.sin(np.linspace(0, byte0 * 2 * np.pi, 150))
        
        # Byte1: dims 9675-9824 (150 dims)  
        vector[9675:9825] = np.sin(np.linspace(0, byte1 * 2 * np.pi, 150))
        
        # Byte2: dims 9825-9974 (150 dims)
        vector[9825:9975] = np.sin(np.linspace(0, byte2 * 2 * np.pi, 150))
        
        # Interaction terms in remaining dims
        vector[9975:10000] = np.array([
            byte0 * byte1,
            byte1 * byte2,
            byte0 * byte2,
            byte0 + byte1 + byte2,
            byte0 - byte1,
            byte1 - byte2,
            byte0 - byte2,
            (byte0 + byte1) / 2,
            (byte1 + byte2) / 2,
            (byte0 + byte2) / 2,
            byte0 * byte1 * byte2,
            np.sqrt(byte0),
            np.sqrt(byte1),
            np.sqrt(byte2),
            np.log1p(byte0),
            np.log1p(byte1),
            np.log1p(byte2),
            np.exp(-byte0),
            np.exp(-byte1),
            np.exp(-byte2),
            byte0 ** 2,
            byte1 ** 2,
            byte2 ** 2,
            np.sin(byte0 * np.pi),
            np.sin(byte1 * np.pi),
        ])
        
        return VSAOps.normalize(vector)
    
    def decode_triangle(self, vector: np.ndarray) -> Tuple[float, float, float]:
        """
        Decode a triangle from a VSA vector.
        
        Extracts the dominant frequencies from each byte region.
        """
        # Extract byte regions
        byte0_region = vector[9525:9675]
        byte1_region = vector[9675:9825]
        byte2_region = vector[9825:9975]
        
        # Estimate original values via FFT peak
        def estimate_byte(region: np.ndarray) -> float:
            fft = np.fft.fft(region)
            freqs = np.fft.fftfreq(len(region))
            peak_idx = np.argmax(np.abs(fft[1:len(fft)//2])) + 1
            return min(1.0, max(0.0, abs(freqs[peak_idx]) * len(region) / (2 * np.pi)))
        
        return (
            estimate_byte(byte0_region),
            estimate_byte(byte1_region),
            estimate_byte(byte2_region),
        )
    
    def close(self):
        """Close the database connection."""
        pass  # LanceDB handles cleanup
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
