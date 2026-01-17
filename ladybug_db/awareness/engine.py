from typing import Dict, Any, List
from ..extensions.memgraph_ext import MemgraphExtension
from .resonance import ResonanceSubstrate

class AwarenessEngine:
    """
    Orchestrates the 'Awareness of Now' (Memgraph) and 'Resonance Substrate' (LanceDB).
    """
    def __init__(self):
        self.graph = MemgraphExtension()
        self.substrate = ResonanceSubstrate(dim=10000) # 10kd VSA
        self.triangle_state = {"byte0": 0.0, "byte1": 0.0, "byte2": 0.0}
        self.quadrant_state = "[9999:9999]"
        self.tick_count = 0
        
    def get_current_state(self) -> Dict[str, Any]:
        """
        Retrieves the current awareness state (Graph of Now).
        """
        # In a real implementation, this would query Memgraph for the 'Now' node or subgraph
        return {
            "rung": "R1", # Placeholder
            "triangle": self.triangle_state,
            "quadrant": self.quadrant_state,
            "tick_count": self.tick_count
        }

    def set_quadrant(self, x: int, y: int):
        """
        Updates the Quadrant state [x:y].
        """
        self.quadrant_state = f"[{x}:{y}]"

    def update_triangle(self, b0, b1, b2):
        """
        Updates the Alchemical Triangle state.
        """
        self.triangle_state = {"byte0": b0, "byte1": b1, "byte2": b2}
        self.log_alchemy(self.triangle_state)

    def perceive(self, input_data: Any):
        """
        Process input, update graph, and check for resonance.
        """
        self.tick_count += 1
        
        # 1. Update Graph (Short term memory / Awareness)
        # self.graph.execute_query(...)
        
        # 2. Encode to VSA
        # vector = encoder.encode(input_data)
        vector = self.substrate.generate_random_vector() # Mock encoding
        
        # 3. Check Resonance (Long term / Associative memory)
        matches = self.substrate.resonate(vector)
        
        # 4. Synthesize
        return {
            "awareness": self.get_current_state(),
            "resonance": matches
        }

    def log_alchemy(self, triangle_data: Dict[str, float]):
        """
        Log GQL alchemy triangle state.
        """
        # Cypher/GQL query to update the triangle state
        query = f"""
        MERGE (a:Awareness {{id: 'current'}})
        SET a.byte0 = {triangle_data.get('byte0', 0)},
            a.byte1 = {triangle_data.get('byte1', 0)},
            a.byte2 = {triangle_data.get('byte2', 0)}
        """
        self.graph.execute_query(query)
