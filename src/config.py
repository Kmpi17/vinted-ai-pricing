
import os
from dataclasses import dataclass

@dataclass
class Config:
    # Rutas
    DATA_DIR: str = os.getenv("DATA_DIR", "data")
    PARQUET_PATH: str = os.getenv("PARQUET_PATH", "data/fashion_dataset.parquet")
    
    # Qdrant Config
    QDRANT_HOST: str = os.getenv("QDRANT_HOST", "localhost")
    QDRANT_PORT: int = int(os.getenv("QDRANT_PORT", 6333))
    COLLECTION_NAME: str = os.getenv("COLLECTION_NAME", "fashion_products")
    VECTOR_SIZE: int = int(os.getenv("VECTOR_SIZE", 512))
    BATCH_SIZE: int = int(os.getenv("BATCH_SIZE", 250))
    
    # Spark Config
    SPARK_APP_NAME: str = "VintedFashionDatasetPipeline"
    SPARK_DRIVER_MEMORY: str = os.getenv("SPARK_DRIVER_MEMORY", "4g")

config = Config()