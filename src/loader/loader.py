from typing import List
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

class QdrantLoader:
    def __init__(self, host: str = "localhost", port: int = 6333, api_key: str = None):
        self.client = QdrantClient(host=host, port=port, api_key=api_key)

    def init_collection(self, collection_name: str, vector_size: int):
        """Crea la colección en Qdrant si aún no existe."""
        collections = self.client.get_collections().collections
        exists = any(c.name == collection_name for c in collections)

        if not exists:
            print(f"--> Creando colección '{collection_name}' (dimensión: {vector_size})...")
            self.client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE)
            )
        else:
            print(f"--> La colección '{collection_name}' ya existe.")

    def load_in_batches(self, collection_name: str, points: List[PointStruct], batch_size: int = 250):
        """Inserción de puntos en lotes a Qdrant."""
        total = len(points)
        print(f"--> Subiendo {total} puntos en lotes de {batch_size}...")

        for i in range(0, total, batch_size):
            batch = points[i:i + batch_size]
            self.client.upsert(
                collection_name=collection_name,
                points=batch
            )
            print(f"   Lote {i // batch_size + 1} enviado ({min(i + batch_size, total)}/{total})")

        print("✅ Inserción completada con éxito.")