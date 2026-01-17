"""
LadybugDB Triangle Alchemy — Cognitive Triangles and Transformations
═══════════════════════════════════════════════════════════════════════════════

The Triangle is the fundamental unit of cognitive state in LadybugDB:
    - 3 bytes representing different cognitive dimensions
    - Alchemy transformations between states
    - Graph patterns for triangle networks
    - Integration with VSA encoding

Triangle Structure:
    ┌─────────────────────────────────────────┐
    │              Triangle                   │
    │                                         │
    │         byte0 (Top Vertex)              │
    │              ▲                          │
    │             /│\                         │
    │            / │ \                        │
    │           /  │  \                       │
    │          /   │   \                      │
    │         /    │    \                     │
    │        ▼     │     ▼                    │
    │    byte1 ────┼──── byte2                │
    │   (Left)     │    (Right)               │
    └─────────────────────────────────────────┘

Byte Interpretations (configurable):
    - Cognitive: Attention, Reasoning, Memory
    - Emotional: Valence, Arousal, Dominance
    - Social: Trust, Warmth, Competence
    - Temporal: Past, Present, Future

Alchemy States:
    LEAD → CALCINATION → DISSOLUTION → SEPARATION →
    CONJUNCTION → FERMENTATION → DISTILLATION →
    COAGULATION → GOLD

Born: 2026-01-17
"""

import math
import hashlib
import numpy as np
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional, Tuple, Union, Callable
from dataclasses import dataclass, field
from enum import Enum


# ═══════════════════════════════════════════════════════════════════════════════
# TRIANGLE TYPES
# ═══════════════════════════════════════════════════════════════════════════════

class AlchemyState(str, Enum):
    """Alchemy transformation states."""
    LEAD = "lead"
    CALCINATION = "calcination"
    DISSOLUTION = "dissolution"
    SEPARATION = "separation"
    CONJUNCTION = "conjunction"
    FERMENTATION = "fermentation"
    DISTILLATION = "distillation"
    COAGULATION = "coagulation"
    GOLD = "gold"


class TriangleInterpretation(str, Enum):
    """How to interpret the three bytes."""
    COGNITIVE = "cognitive"      # Attention, Reasoning, Memory
    EMOTIONAL = "emotional"      # Valence, Arousal, Dominance
    SOCIAL = "social"           # Trust, Warmth, Competence
    TEMPORAL = "temporal"       # Past, Present, Future
    RESONANCE = "resonance"     # Query, Key, Value
    CUSTOM = "custom"


BYTE_LABELS = {
    TriangleInterpretation.COGNITIVE: ("Attention", "Reasoning", "Memory"),
    TriangleInterpretation.EMOTIONAL: ("Valence", "Arousal", "Dominance"),
    TriangleInterpretation.SOCIAL: ("Trust", "Warmth", "Competence"),
    TriangleInterpretation.TEMPORAL: ("Past", "Present", "Future"),
    TriangleInterpretation.RESONANCE: ("Query", "Key", "Value"),
}


# ═══════════════════════════════════════════════════════════════════════════════
# TRIANGLE CLASS
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class Triangle:
    """
    A cognitive triangle with three bytes and alchemy state.
    
    Attributes:
        byte0: First dimension (0.0-1.0)
        byte1: Second dimension (0.0-1.0)
        byte2: Third dimension (0.0-1.0)
        alchemy_state: Current alchemical state
        interpretation: How to interpret the bytes
        id: Optional identifier
        source_id: ID of source (moment, event, etc.)
        metadata: Additional data
    """
    byte0: float
    byte1: float
    byte2: float
    alchemy_state: AlchemyState = AlchemyState.LEAD
    interpretation: TriangleInterpretation = TriangleInterpretation.COGNITIVE
    id: Optional[str] = None
    source_id: Optional[str] = None
    timestamp: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        # Clamp values to [0, 1]
        self.byte0 = max(0.0, min(1.0, self.byte0))
        self.byte1 = max(0.0, min(1.0, self.byte1))
        self.byte2 = max(0.0, min(1.0, self.byte2))
        
        if self.id is None:
            self.id = self._generate_id()
        if self.timestamp is None:
            self.timestamp = datetime.now(timezone.utc)
    
    def _generate_id(self) -> str:
        """Generate a unique ID based on triangle state."""
        content = f"{self.byte0:.4f}:{self.byte1:.4f}:{self.byte2:.4f}:{self.timestamp}"
        return f"tri_{hashlib.sha256(content.encode()).hexdigest()[:12]}"
    
    @property
    def bytes(self) -> Tuple[float, float, float]:
        """Get all bytes as tuple."""
        return (self.byte0, self.byte1, self.byte2)
    
    @property
    def magnitude(self) -> float:
        """Total activation (average of bytes)."""
        return (self.byte0 + self.byte1 + self.byte2) / 3.0
    
    @property
    def balance(self) -> float:
        """How evenly distributed (1.0 = perfectly balanced)."""
        import statistics
        return 1.0 - statistics.stdev([self.byte0, self.byte1, self.byte2])
    
    @property
    def dominant_byte(self) -> int:
        """Index of the dominant byte (0, 1, or 2)."""
        return self.bytes.index(max(self.bytes))
    
    @property
    def dominant_label(self) -> str:
        """Label of the dominant byte."""
        labels = BYTE_LABELS.get(self.interpretation, ("Byte0", "Byte1", "Byte2"))
        return labels[self.dominant_byte]
    
    @property
    def resonance_score(self) -> float:
        """Compute resonance score (higher = more refined)."""
        # Gold-like triangles have high magnitude and balance
        return self.magnitude * 0.6 + self.balance * 0.4
    
    @property
    def entropy(self) -> float:
        """Shannon entropy of the byte distribution."""
        probs = np.array([self.byte0, self.byte1, self.byte2])
        probs = probs / (probs.sum() + 1e-9)
        return float(-np.sum(probs * np.log2(probs + 1e-9)))
    
    @property
    def area(self) -> float:
        """
        Area of the triangle in a normalized coordinate system.
        Uses Heron's formula with bytes as side lengths.
        """
        a, b, c = self.byte0, self.byte1, self.byte2
        s = (a + b + c) / 2
        area_sq = s * (s - a) * (s - b) * (s - c)
        return math.sqrt(max(0, area_sq))
    
    def distance_to(self, other: "Triangle") -> float:
        """Euclidean distance to another triangle."""
        return math.sqrt(
            (self.byte0 - other.byte0) ** 2 +
            (self.byte1 - other.byte1) ** 2 +
            (self.byte2 - other.byte2) ** 2
        )
    
    def cosine_similarity(self, other: "Triangle") -> float:
        """Cosine similarity to another triangle."""
        a = np.array([self.byte0, self.byte1, self.byte2])
        b = np.array([other.byte0, other.byte1, other.byte2])
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "byte0": self.byte0,
            "byte1": self.byte1,
            "byte2": self.byte2,
            "magnitude": self.magnitude,
            "balance": self.balance,
            "resonance_score": self.resonance_score,
            "alchemy_state": self.alchemy_state.value,
            "interpretation": self.interpretation.value,
            "source_id": self.source_id,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "metadata": self.metadata,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Triangle":
        """Create triangle from dictionary."""
        return cls(
            byte0=data["byte0"],
            byte1=data["byte1"],
            byte2=data["byte2"],
            alchemy_state=AlchemyState(data.get("alchemy_state", "lead")),
            interpretation=TriangleInterpretation(data.get("interpretation", "cognitive")),
            id=data.get("id"),
            source_id=data.get("source_id"),
            metadata=data.get("metadata", {}),
        )
    
    @classmethod
    def balanced(cls, value: float = 0.5) -> "Triangle":
        """Create a perfectly balanced triangle."""
        return cls(byte0=value, byte1=value, byte2=value)
    
    @classmethod
    def random(cls) -> "Triangle":
        """Create a random triangle."""
        import random
        return cls(
            byte0=random.random(),
            byte1=random.random(),
            byte2=random.random(),
        )


# ═══════════════════════════════════════════════════════════════════════════════
# TRIANGLE ALCHEMY ENGINE
# ═══════════════════════════════════════════════════════════════════════════════

class TriangleAlchemy:
    """
    Engine for alchemical transformations of triangles.
    
    The Great Work proceeds through stages:
    1. Nigredo (Blackening) - Calcination, Dissolution
    2. Albedo (Whitening) - Separation, Conjunction
    3. Citrinitas (Yellowing) - Fermentation, Distillation
    4. Rubedo (Reddening) - Coagulation, Gold
    """
    
    def __init__(self):
        # Transformation sequence
        self._sequence = [
            AlchemyState.LEAD,
            AlchemyState.CALCINATION,
            AlchemyState.DISSOLUTION,
            AlchemyState.SEPARATION,
            AlchemyState.CONJUNCTION,
            AlchemyState.FERMENTATION,
            AlchemyState.DISTILLATION,
            AlchemyState.COAGULATION,
            AlchemyState.GOLD,
        ]
        
        # Transformation functions
        self._transforms: Dict[AlchemyState, Callable[[Triangle], Triangle]] = {
            AlchemyState.CALCINATION: self._calcinate,
            AlchemyState.DISSOLUTION: self._dissolve,
            AlchemyState.SEPARATION: self._separate,
            AlchemyState.CONJUNCTION: self._conjoin_self,
            AlchemyState.FERMENTATION: self._ferment,
            AlchemyState.DISTILLATION: self._distill,
            AlchemyState.COAGULATION: self._coagulate,
            AlchemyState.GOLD: self._transmute_gold,
        }
    
    def create(
        self,
        byte0: float,
        byte1: float,
        byte2: float,
        interpretation: TriangleInterpretation = TriangleInterpretation.COGNITIVE,
    ) -> Triangle:
        """Create a new triangle."""
        return Triangle(
            byte0=byte0,
            byte1=byte1,
            byte2=byte2,
            interpretation=interpretation,
        )
    
    def next_state(self, state: AlchemyState) -> AlchemyState:
        """Get the next state in the alchemical sequence."""
        idx = self._sequence.index(state)
        if idx < len(self._sequence) - 1:
            return self._sequence[idx + 1]
        return AlchemyState.GOLD
    
    def advance(self, triangle: Triangle) -> Triangle:
        """Advance triangle to next alchemical state."""
        next_state = self.next_state(triangle.alchemy_state)
        transform = self._transforms.get(next_state)
        
        if transform:
            return transform(triangle)
        return triangle
    
    def transform(self, triangle: Triangle, target_state: AlchemyState) -> Triangle:
        """Transform triangle to a specific state."""
        transform = self._transforms.get(target_state)
        if transform:
            result = transform(triangle)
            result.alchemy_state = target_state
            return result
        return triangle
    
    def alchemize(self, triangle: Triangle, target: str) -> Triangle:
        """
        Transform triangle using named alchemy operation.
        
        Args:
            triangle: Source triangle
            target: Target state or operation name
        """
        target_lower = target.lower()
        
        # Map common names to states
        name_map = {
            "gold": AlchemyState.GOLD,
            "lead": AlchemyState.LEAD,
            "calcinate": AlchemyState.CALCINATION,
            "dissolve": AlchemyState.DISSOLUTION,
            "separate": AlchemyState.SEPARATION,
            "conjoin": AlchemyState.CONJUNCTION,
            "ferment": AlchemyState.FERMENTATION,
            "distill": AlchemyState.DISTILLATION,
            "coagulate": AlchemyState.COAGULATION,
        }
        
        state = name_map.get(target_lower)
        if state:
            return self.transform(triangle, state)
        return triangle
    
    def full_opus(self, triangle: Triangle) -> Triangle:
        """
        Perform the full Magnum Opus: Lead → Gold.
        
        Returns a new triangle that has passed through all stages.
        """
        result = triangle
        for state in self._sequence[1:]:  # Skip LEAD
            transform = self._transforms.get(state)
            if transform:
                result = transform(result)
        return result
    
    def solve_et_coagula(self, triangle: Triangle) -> Tuple[Triangle, Triangle]:
        """
        Solve et Coagula: Dissolve and Coagulate.
        
        Returns two triangles:
        - Dissolved (essence)
        - Coagulated (fixed)
        """
        dissolved = self._dissolve(triangle)
        coagulated = self._coagulate(triangle)
        return dissolved, coagulated
    
    # ═══════════════════════════════════════════════════════════════════════════
    # TRANSFORMATION FUNCTIONS
    # ═══════════════════════════════════════════════════════════════════════════
    
    def _calcinate(self, triangle: Triangle) -> Triangle:
        """
        Calcination: Burn away impurities (reduce magnitude).
        
        The fire burns away the gross matter.
        """
        factor = 0.5 + 0.5 * triangle.magnitude
        return Triangle(
            byte0=triangle.byte0 * factor,
            byte1=triangle.byte1 * factor,
            byte2=triangle.byte2 * factor,
            alchemy_state=AlchemyState.CALCINATION,
            interpretation=triangle.interpretation,
            source_id=triangle.id,
            metadata={"parent_state": triangle.alchemy_state.value},
        )
    
    def _dissolve(self, triangle: Triangle) -> Triangle:
        """
        Dissolution: Dissolve barriers, move towards balance.
        
        Water dissolves the ash from calcination.
        """
        mean = triangle.magnitude
        blend = 0.7
        return Triangle(
            byte0=triangle.byte0 * (1 - blend) + mean * blend,
            byte1=triangle.byte1 * (1 - blend) + mean * blend,
            byte2=triangle.byte2 * (1 - blend) + mean * blend,
            alchemy_state=AlchemyState.DISSOLUTION,
            interpretation=triangle.interpretation,
            source_id=triangle.id,
            metadata={"blend_factor": blend},
        )
    
    def _separate(self, triangle: Triangle) -> Triangle:
        """
        Separation: Extract the essence (emphasize dominant).
        
        The valuable is separated from the worthless.
        """
        bytes_list = list(triangle.bytes)
        dominant_idx = bytes_list.index(max(bytes_list))
        
        new_bytes = [b * 0.5 for b in bytes_list]
        new_bytes[dominant_idx] = min(1.0, bytes_list[dominant_idx] * 1.5)
        
        return Triangle(
            byte0=new_bytes[0],
            byte1=new_bytes[1],
            byte2=new_bytes[2],
            alchemy_state=AlchemyState.SEPARATION,
            interpretation=triangle.interpretation,
            source_id=triangle.id,
            metadata={"dominant_byte": dominant_idx},
        )
    
    def _conjoin_self(self, triangle: Triangle) -> Triangle:
        """
        Conjunction: Combine elements (self-integration).
        
        The separated elements are reunited.
        """
        # Self-conjunction: average of original and balanced
        balanced = triangle.magnitude
        return Triangle(
            byte0=(triangle.byte0 + balanced) / 2,
            byte1=(triangle.byte1 + balanced) / 2,
            byte2=(triangle.byte2 + balanced) / 2,
            alchemy_state=AlchemyState.CONJUNCTION,
            interpretation=triangle.interpretation,
            source_id=triangle.id,
        )
    
    def conjoin(self, t1: Triangle, t2: Triangle) -> Triangle:
        """
        Conjunction: Combine two triangles.
        
        The masculine and feminine unite.
        """
        return Triangle(
            byte0=(t1.byte0 + t2.byte0) / 2,
            byte1=(t1.byte1 + t2.byte1) / 2,
            byte2=(t1.byte2 + t2.byte2) / 2,
            alchemy_state=AlchemyState.CONJUNCTION,
            interpretation=t1.interpretation,
            source_id=f"{t1.id}+{t2.id}",
            metadata={"parent1": t1.id, "parent2": t2.id},
        )
    
    def _ferment(self, triangle: Triangle) -> Triangle:
        """
        Fermentation: Add creative life (introduce controlled noise).
        
        The spirit enters the purified matter.
        """
        import random
        creativity = 0.1
        return Triangle(
            byte0=max(0, min(1, triangle.byte0 + random.uniform(-creativity, creativity))),
            byte1=max(0, min(1, triangle.byte1 + random.uniform(-creativity, creativity))),
            byte2=max(0, min(1, triangle.byte2 + random.uniform(-creativity, creativity))),
            alchemy_state=AlchemyState.FERMENTATION,
            interpretation=triangle.interpretation,
            source_id=triangle.id,
            metadata={"creativity": creativity},
        )
    
    def _distill(self, triangle: Triangle) -> Triangle:
        """
        Distillation: Purify by sharpening patterns.
        
        The essence is refined through repeated heating.
        """
        mean = triangle.magnitude
        sharpened = [
            min(1.0, b * 1.2) if b > mean else b * 0.8
            for b in triangle.bytes
        ]
        return Triangle(
            byte0=sharpened[0],
            byte1=sharpened[1],
            byte2=sharpened[2],
            alchemy_state=AlchemyState.DISTILLATION,
            interpretation=triangle.interpretation,
            source_id=triangle.id,
        )
    
    def _coagulate(self, triangle: Triangle) -> Triangle:
        """
        Coagulation: Solidify and stabilize.
        
        The spirit becomes fixed in matter.
        """
        # Round to nearest 0.1 for stability
        return Triangle(
            byte0=round(triangle.byte0, 1),
            byte1=round(triangle.byte1, 1),
            byte2=round(triangle.byte2, 1),
            alchemy_state=AlchemyState.COAGULATION,
            interpretation=triangle.interpretation,
            source_id=triangle.id,
        )
    
    def _transmute_gold(self, triangle: Triangle) -> Triangle:
        """
        Gold: The final transmutation.
        
        The Philosopher's Stone transforms lead to gold.
        """
        # Gold = balanced, high magnitude, high resonance
        mean = triangle.magnitude
        boost = max(0.8, min(1.0, mean * 1.2))
        return Triangle(
            byte0=boost,
            byte1=boost,
            byte2=boost,
            alchemy_state=AlchemyState.GOLD,
            interpretation=triangle.interpretation,
            source_id=triangle.id,
            metadata={"original_magnitude": mean},
        )
    
    # ═══════════════════════════════════════════════════════════════════════════
    # RESONANCE OPERATIONS
    # ═══════════════════════════════════════════════════════════════════════════
    
    def compute_resonance(self, t1: Triangle, t2: Triangle) -> float:
        """
        Compute resonance strength between two triangles.
        
        High resonance indicates harmonic alignment.
        """
        # Cosine similarity of bytes
        similarity = t1.cosine_similarity(t2)
        
        # Alchemy state compatibility (same phase = higher resonance)
        state_map = {
            AlchemyState.LEAD: 0,
            AlchemyState.CALCINATION: 1,
            AlchemyState.DISSOLUTION: 1,
            AlchemyState.SEPARATION: 2,
            AlchemyState.CONJUNCTION: 2,
            AlchemyState.FERMENTATION: 3,
            AlchemyState.DISTILLATION: 3,
            AlchemyState.COAGULATION: 4,
            AlchemyState.GOLD: 4,
        }
        phase1 = state_map.get(t1.alchemy_state, 0)
        phase2 = state_map.get(t2.alchemy_state, 0)
        phase_compatibility = 1.0 - abs(phase1 - phase2) / 4.0
        
        return similarity * 0.7 + phase_compatibility * 0.3
    
    def find_harmonic_triangles(
        self,
        source: Triangle,
        candidates: List[Triangle],
        min_resonance: float = 0.5,
    ) -> List[Tuple[Triangle, float]]:
        """
        Find triangles that resonate with the source.
        
        Returns list of (triangle, resonance_score) tuples.
        """
        results = []
        for candidate in candidates:
            resonance = self.compute_resonance(source, candidate)
            if resonance >= min_resonance:
                results.append((candidate, resonance))
        
        return sorted(results, key=lambda x: x[1], reverse=True)
    
    def blend_triangles(
        self,
        triangles: List[Triangle],
        weights: Optional[List[float]] = None,
    ) -> Triangle:
        """
        Blend multiple triangles with optional weights.
        
        Creates a new triangle from the weighted average.
        """
        if not triangles:
            return Triangle.balanced()
        
        if weights is None:
            weights = [1.0 / len(triangles)] * len(triangles)
        
        total_weight = sum(weights)
        weights = [w / total_weight for w in weights]
        
        byte0 = sum(t.byte0 * w for t, w in zip(triangles, weights))
        byte1 = sum(t.byte1 * w for t, w in zip(triangles, weights))
        byte2 = sum(t.byte2 * w for t, w in zip(triangles, weights))
        
        return Triangle(
            byte0=byte0,
            byte1=byte1,
            byte2=byte2,
            alchemy_state=AlchemyState.CONJUNCTION,
            metadata={
                "blend_sources": [t.id for t in triangles],
                "weights": weights,
            },
        )
    
    # ═══════════════════════════════════════════════════════════════════════════
    # VISUALIZATION HELPERS
    # ═══════════════════════════════════════════════════════════════════════════
    
    def to_rgb(self, triangle: Triangle) -> Tuple[int, int, int]:
        """Convert triangle to RGB color."""
        return (
            int(triangle.byte0 * 255),
            int(triangle.byte1 * 255),
            int(triangle.byte2 * 255),
        )
    
    def to_hex(self, triangle: Triangle) -> str:
        """Convert triangle to hex color."""
        r, g, b = self.to_rgb(triangle)
        return f"#{r:02x}{g:02x}{b:02x}"
    
    def to_barycentric(self, triangle: Triangle) -> Tuple[float, float]:
        """
        Convert triangle to barycentric coordinates for 2D plotting.
        
        Returns (x, y) in equilateral triangle space.
        """
        # Normalize to sum to 1
        total = triangle.byte0 + triangle.byte1 + triangle.byte2 + 1e-9
        b0 = triangle.byte0 / total
        b1 = triangle.byte1 / total
        b2 = triangle.byte2 / total
        
        # Equilateral triangle vertices
        # Top: (0.5, sqrt(3)/2), Left: (0, 0), Right: (1, 0)
        x = b1 + 0.5 * b0
        y = (math.sqrt(3) / 2) * b0
        
        return (x, y)
