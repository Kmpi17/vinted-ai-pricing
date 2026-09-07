
from pathlib import Path
import os

# Rutas de datos
BASE_DIR = Path(__file__).resolve().parent.parent

# Ruta del Parquet
PARQUET_PATH = str(BASE_DIR / "data" / "fashion_dataset.parquet")

# Configuración de Qdrant
QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", 6333))
COLLECTION_NAME = "fashion_items"
VECTOR_SIZE = 512

# Modelo de Embeddings
CLIP_MODEL_ID = "openai/clip-vit-base-patch32"