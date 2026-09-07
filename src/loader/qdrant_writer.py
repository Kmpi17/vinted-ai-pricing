from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct
from pyspark.sql import DataFrame
from src import config

def ensure_collection_exists():
    """Conecta a Qdrant y crea la colección si no existe."""
    client = QdrantClient(host=config.QDRANT_HOST, port=config.QDRANT_PORT)
    
    if not client.collection_exists(config.COLLECTION_NAME):
        client.create_collection(
            collection_name=config.COLLECTION_NAME,
            vectors_config=VectorParams(
                size=config.VECTOR_SIZE,
                distance=Distance.COSINE
            )
        )
        print(f"✅ Colección '{config.COLLECTION_NAME}' creada con éxito.")
    else:
        print(f"ℹ️ La colección '{config.COLLECTION_NAME}' ya existe.")

def upsert_partition(rows):
    """Instancia el cliente por partición de Spark e inserta los puntos."""
    client = QdrantClient(host=config.QDRANT_HOST, port=config.QDRANT_PORT)
    points = []
    
    for row in rows:
        points.append(
            PointStruct(
                id=int(row["id"]),
                vector=row["vector"],
                payload={
                    "gender": str(row["gender"]),
                    "masterCategory": str(row["masterCategory"]),
                    "subCategory": str(row["subCategory"]),
                    "articleType": str(row["articleType"]),
                    "baseColour": str(row["baseColour"]),
                    "productDisplayName": str(row["productDisplayName"])
                }
            )
        )
    
    if points:
        client.upsert(collection_name=config.COLLECTION_NAME, points=points)

def write_to_qdrant(df):
    # Reagrupar para controlar la concurrencia en local
    df_single = df.coalesce(2)

    def upsert_partition(rows):
        client = QdrantClient(host=config.QDRANT_HOST, port=config.QDRANT_PORT)
        points = []
        
        for row in rows:
            # Reemplaza 'id', 'vector' y 'payload' con los nombres reales de tus columnas
            points.append(
                PointStruct(
                    id=row.id,
                    vector=row.embedding,  # Debe ser una lista/array de floats
                    payload={
                        "title": row.title,
                        "category": row.category if hasattr(row, "category") else None
                    }
                )
            )
            
            # Enviar en batches para evitar saturación de memoria en Python
            if len(points) >= 100:
                client.upsert(collection_name=config.COLLECTION_NAME, points=points)
                points = []
                
        if points:
            client.upsert(collection_name=config.COLLECTION_NAME, points=points)

    df_single.foreachPartition(upsert_partition)