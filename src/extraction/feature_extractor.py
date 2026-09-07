import io
import torch
import pandas as pd
from PIL import Image
from typing import Iterator
from transformers import CLIPProcessor, CLIPModel

from pyspark.sql import DataFrame
from pyspark.sql.functions import pandas_udf
from pyspark.sql.types import ArrayType, FloatType
from src import config

def make_clip_udf(model_id: str = config.CLIP_MODEL_ID):
    """
    Crea una Pandas UDF optimizada por lotes (Iterator) para extraer
    embeddings de CLIP sin recargar el modelo en cada llamada.
    """
    @pandas_udf(ArrayType(FloatType()))
    def extract_clip_embeddings(bytes_iter: Iterator[pd.Series]) -> Iterator[pd.Series]:
        # El modelo se carga una sola vez por partición/worker de Spark
        model = CLIPModel.from_pretrained(model_id)
        processor = CLIPProcessor.from_pretrained(model_id)
        model.eval()

        for bytes_series in bytes_iter:
            embeddings = []
            for b in bytes_series:
                try:
                    if b is None:
                        embeddings.append([0.0] * config.VECTOR_SIZE)
                        continue
                    
                    img = Image.open(io.BytesIO(b)).convert("RGB")
                    inputs = processor(images=img, return_tensors="pt")
                    
                    with torch.no_grad():
                        features = model.get_image_features(**inputs)
                        # Normalización L2 para similitud coseno exacta
                        features = features / features.norm(dim=-1, keepdim=True)
                        embeddings.append(features.squeeze().tolist())
                except Exception:
                    embeddings.append([0.0] * config.VECTOR_SIZE)

            yield pd.Series(embeddings)

    return extract_clip_embeddings

def add_image_embeddings(df: DataFrame, image_col: str = "image.bytes", out_col: str = "vector") -> DataFrame:
    """
    Aplica la UDF de CLIP al DataFrame de PySpark para añadir la columna de vectores.
    """
    clip_udf = make_clip_udf()
    return df.withColumn(out_col, clip_udf(df[image_col]))