import io
import uuid
import torch
import pandas as pd
from typing import List, Iterator
from PIL import Image

from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct
from transformers import CLIPModel, CLIPProcessor

from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from pyspark.sql.types import StructType, StructField, StringType

from src.config import config


# ---------------------------------------------------------------------------
# 1. Función Unificada por Partición (Inferencia CLIP + Carga Qdrant)
# ---------------------------------------------------------------------------
def _process_and_index_partition(iterator: Iterator[pd.DataFrame]) -> Iterator[pd.DataFrame]:
    """
    Se ejecuta una sola vez por partición en cada worker:
    1. Instancia el modelo CLIP en memoria.
    2. Instancia el cliente Qdrant.
    3. Procesa por lotes los textos -> embeddings -> subida a Qdrant.
    """
    # Carga de recursos CLIP dentro del worker
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_name = getattr(config, "CLIP_MODEL_ID", "openai/clip-vit-base-patch32")
    
    model = CLIPModel.from_pretrained(model_name).to(device)
    model.eval()
    processor = CLIPProcessor.from_pretrained(model_name)

    # Conexión con Qdrant
    qdrant_host = getattr(config, "QDRANT_HOST", "localhost")
    qdrant_port = getattr(config, "QDRANT_PORT", 6333)
    collection_name = getattr(config, "COLLECTION_NAME", "fashion_products")
    
    client = QdrantClient(host=qdrant_host, port=qdrant_port)

    batch_size = 64  # Lote interno de PyTorch para no colapsar la VRAM/RAM

    for pdf in iterator:
        if pdf.empty:
            continue

        text_list = pdf["text_to_embed"].tolist()
        embeddings = []

        # 1. Inferencia por lotes en PyTorch
        with torch.no_grad():
            for i in range(0, len(text_list), batch_size):
                batch_texts = text_list[i : i + batch_size]
                inputs = processor(
                    text=batch_texts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True
                ).to(device)

                outputs = model.get_text_features(**inputs)
                features = getattr(outputs, "text_embeds", getattr(outputs, "pooler_output", outputs))
                features = features / features.norm(p=2, dim=-1, keepdim=True)
                embeddings.extend(features.cpu().numpy().tolist())

        # 2. Construcción de PointStructs para Qdrant
        points = []
        for (idx, row), vector in zip(pdf.iterrows(), embeddings):
            raw_id = row.get("id") or (idx + 1)
            point_id = int(raw_id) if str(raw_id).isdigit() else str(uuid.uuid4())

            category = str(row.get("articleType") or row.get("subCategory") or "Apparel")
            gender = str(row.get("gender") or "Unisex")
            display_name = str(row.get("productDisplayName") or f"Fashion Item {point_id} - {category}")

            payload = {
                "productDisplayName": display_name,
                "articleType": category,
                "subCategory": row.get("subCategory", category),
                "masterCategory": row.get("masterCategory", "Apparel"),
                "gender": gender,
                "baseColour": row.get("baseColour", "N/A"),
                "season": row.get("season", "N/A"),
                "year": row.get("year", "N/A"),
                "usage": row.get("usage", "N/A")
            }
            payload = {k: v for k, v in payload.items() if v is not None}

            points.append(
                PointStruct(
                    id=point_id,
                    vector=vector,
                    payload=payload
                )
            )

        # 3. Subida directa a Qdrant
        if points:
            client.upload_points(
                collection_name=collection_name,
                points=points,
                batch_size=500
            )

        yield pd.DataFrame({"status": ["ok"] * len(pdf)})


# ---------------------------------------------------------------------------
# 2. Pipeline ETL Principal
# ---------------------------------------------------------------------------
def prepare_and_index_qdrant_points(df_spark: DataFrame, num_partitions: int = 6) -> int:
    """
    Prepara el dataset, reparte la carga y procesa/ingesta en Qdrant vía mapInPandas.
    """
    # En local limitamos las particiones concurrentes a 4-6 para no ahogar la RAM
    partitions = min(num_partitions, 6)
    print(f"--> Ajustando ejecución a {partitions} particiones para optimizar uso de RAM...")

    df_prepared = df_spark.withColumn(
        "text_to_embed",
        F.concat_ws(
            " ",
            F.coalesce(F.col("productDisplayName"), F.lit("")),
            F.coalesce(F.col("masterCategory"), F.lit("")),
            F.coalesce(F.col("subCategory"), F.lit("")),
            F.coalesce(F.col("articleType"), F.lit("")),
            F.coalesce(F.col("baseColour"), F.lit("")),
            F.coalesce(F.col("gender"), F.lit("Unisex"))
        )
    ).repartition(partitions)

    print("--> Procesando embeddings e ingestando a Qdrant en paralelo vía mapInPandas...")
    output_schema = StructType([StructField("status", StringType(), True)])

    total_indexed = df_prepared.mapInPandas(
        _process_and_index_partition,
        schema=output_schema
    ).count()

    print(f"✅ {total_indexed} puntos procesados e indexados con éxito en Qdrant.")
    return total_indexed


# ---------------------------------------------------------------------------
# 3. Clase Standalone (Inferencia para API FastAPI)
# ---------------------------------------------------------------------------
class CLIPExtractor:
    def __init__(self, model_id: str = None):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        model_name = model_id or getattr(config, "CLIP_MODEL_ID", "openai/clip-vit-base-patch32")

        self.model = CLIPModel.from_pretrained(model_name).to(self.device)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.model.eval()

    def generate_text_embedding(self, text: str) -> List[float]:
        with torch.no_grad():
            inputs = self.processor(text=[text], return_tensors="pt", padding=True, truncation=True).to(self.device)
            outputs = self.model.get_text_features(**inputs)
            features = getattr(outputs, "text_embeds", getattr(outputs, "pooler_output", outputs))
            features = features / features.norm(p=2, dim=-1, keepdim=True)
            return features[0].cpu().numpy().tolist()

    def generate_image_embedding(self, image_bytes: bytes) -> List[float]:
        with torch.no_grad():
            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            inputs = self.processor(images=image, return_tensors="pt").to(self.device)
            outputs = self.model.get_image_features(**inputs)
            features = getattr(outputs, "image_embeds", getattr(outputs, "pooler_output", outputs))
            features = features / features.norm(p=2, dim=-1, keepdim=True)
            return features[0].cpu().numpy().tolist()