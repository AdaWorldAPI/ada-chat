import lancedb
import os

class LanceDBExtension:
    def __init__(self, uri=None):
        self.uri = uri or os.getenv("LANCEDB_URI", "/data/chat_lancedb")
        self.connection = None

    def connect(self):
        if not self.connection:
            os.makedirs(os.path.dirname(self.uri) if not os.path.isdir(self.uri) else self.uri, exist_ok=True)
            self.connection = lancedb.connect(self.uri)
        return self.connection

    def get_table(self, name):
        conn = self.connect()
        if name in conn.table_names():
            return conn.open_table(name)
        return None
