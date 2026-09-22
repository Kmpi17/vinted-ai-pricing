from typing import List
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, 
    VectorParams, 
    PointStruct,
    TextIndexParams,
    TokenizerType,
    FieldType
)

class QdrantLoader:
    def __init__(self, host: str = "localhost", port: int = 6333, api_key: str = None):
        self.client = QdrantClient(host=host, port=port, api_key=api_key)

    def init_collection(self, collection_name: str, vector_size: int = 512, force_recreate: bool = False):
        """
        Inicializa o recrea la colección en Qdrant y crea los índices de texto
        para permitir Búsqueda Híbrida (Vectores CLIP + Coincidencia Exacta de Texto).
        """
        collections = self.client.get_collections().collections
        exists = any(c.name == collection_name for c in collections)

        if force_recreate and exists:
            print(f"--> Eliminando colección existente '{collection_name}'...")
            self.client.delete_collection(collection_name=collection_name)
            exists = False

        if force_recreate or not exists:
            print(f"--> Creando colección '{collection_name}' (dimensión: {vector_size}, métrica: COSINE)...")
            self.client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE)
            )

            # -------------------------------------------------------------------
            # Payload Indexes para Búsqueda Híbrida (Full-Text Search)
            # -------------------------------------------------------------------
            print(f"--> Creando índices de texto en '{collection_name}' para búsquedas exactas...")

            
            self.client.create_payload_index(
                collection_name=collection_name,
                field_name="productDisplayName",
                field_schema=TextIndexParams(
                    type=FieldType.TEXT,
                    tokenizer=TokenizerType.WORD,  
                    lowercase=True
                )
            )

            
            self.client.create_payload_index(
                collection_name=collection_name,
                field_name="articleType",
                field_schema=TextIndexParams(
                    type=FieldType.TEXT,
                    tokenizer=TokenizerType.WORD,
                    lowercase=True
                )
            )

            self.client.create_payload_index(
                collection_name=collection_name,
                field_name="masterCategory",
                field_schema=TextIndexParams(
                    type=FieldType.TEXT,
                    tokenizer=TokenizerType.WORD,
                    lowercase=True
                )
            )

            print("✅ Colección e índices de texto inicializados con éxito.")
        else:
            print(f"--> La colección '{collection_name}' ya existe. Omitiendo creación.")

    def load_in_batches(self, collection_name: str, points: List[PointStruct], batch_size: int = 250):
        """Inserción de puntos en lotes utilizando el método upsert (para cargas standalone)."""
        total = len(points)
        print(f"--> Subiendo {total} puntos en lotes de {batch_size}...")

        for i in range(0, total, batch_size):
            batch = points[i:i + batch_size]
            self.client.upsert(
                collection_name=collection_name,
                points=batch
            )
            print(f"   [+] Lote {i // batch_size + 1} enviado ({min(i + batch_size, total)}/{total})")

        print("✅ Inserción en Qdrant completada con éxito.")