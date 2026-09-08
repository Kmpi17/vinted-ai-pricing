from typing import List
from qdrant_client.models import PointStruct
import uuid

def generate_embedding(text: str) -> List[float]:
    """
    Sustituye esta función con la llamada a tu modelo real 
    (ej. SentenceTransformers, CLIP o API de embeddings).
    """
    # Dimensión de ejemplo de 512 (estándar de modelos multimodal/CLIP)
    return [0.05] * 512  

def prepare_qdrant_points(df_spark) -> List[PointStruct]:
    """
    Extrae información del DataFrame de Spark y genera los objetos PointStruct.
    """
    print("--> Extrayendo registros y generando vectores...")
    rows = df_spark.collect()
    points = []

    for row in rows:
        row_dict = row.asDict()
        
        # 1. Asignar un ID válido para Qdrant (int o UUID)
        raw_id = row_dict.get("id")
        point_id = int(raw_id) if str(raw_id).isdigit() else str(uuid.uuid4())

        # 2. Construir el texto representativo para el embedding
        product_name = str(row_dict.get("productDisplayName", ""))
        article_type = str(row_dict.get("articleType", ""))
        color = str(row_dict.get("baseColour", ""))
        
        text_to_embed = f"{product_name} {article_type} {color}".strip()
        vector = generate_embedding(text_to_embed)

        # 3. Guardar el resto de columnas como payload
        payload = {
            "productDisplayName": product_name,
            "articleType": article_type,
            "baseColour": color,
            "subCategory": row_dict.get("subCategory"),
            "masterCategory": row_dict.get("masterCategory"),
            "gender": row_dict.get("gender"),
        }

        # Eliminar valores nulos del payload
        payload = {k: v for k, v in payload.items() if v is not None}

        points.append(
            PointStruct(
                id=point_id,
                vector=vector,
                payload=payload
            )
        )

    print(f"✅ {len(points)} puntos procesados y listos.")
    return points