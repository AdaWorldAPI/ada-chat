import numpy as np
import json
from ..extensions.lancedb_ext import LanceDBExtension
import pyarrow as pa

class ResonanceSubstrate:
    """
    Manages the VSA (Vector Symbolic Architecture) 10kd substrate using LanceDB.
    """
    def __init__(self, dim=10000):
        self.dim = dim
        self.db_ext = LanceDBExtension()
        self.ensure_table()

    def ensure_table(self):
        conn = self.db_ext.connect()
        if "resonance_field" not in conn.table_names():
            # Define schema for 10k dimensional vectors
            # Note: 10k is large, we might store as blob or use fixed size list if supported and performant
            # For simplicity/compatibility, we'll use a float32 list
            schema = pa.schema([
                ("id", pa.string()),
                ("vector", pa.list_(pa.float32(), self.dim)),
                ("payload", pa.string()),
                ("timestamp", pa.float64())
            ])
            conn.create_table("resonance_field", schema=schema)

    def resonate(self, query_vector: list, limit=10):
        """
        Finds resonating (nearest neighbor) patterns in the substrate.
        """
        table = self.db_ext.get_table("resonance_field")
        if not table:
            return []
        
        # LanceDB search
        results = table.search(query_vector).limit(limit).to_list()
        return results

    def imprint(self, id: str, vector: list, payload: dict):
        """
        Imprints a new pattern into the substrate.
        """
        table = self.db_ext.get_table("resonance_field")
        if not table:
            return
        
        import time
        table.add([{
            "id": id,
            "vector": vector,
            "payload": json.dumps(payload),
            "timestamp": time.time()
        }])

    def generate_random_vector(self):
        return np.random.randn(self.dim).astype(np.float32).tolist()
