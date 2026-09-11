from typing import List
from qdrant_client.models import PointStruct
from transformers import CLIPModel, CLIPProcessor
import uuid
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
model.eval()

def generate_embedding(text: str) -> list[float]:
    """Genera un embedding de texto real de 512 dimensiones usando CLIP."""
    with torch.no_grad():
        inputs = processor(text=[text], return_tensors="pt", padding=True, truncation=True).to(device)
        text_outputs = model.get_text_features(**inputs)
        
        # Extraer tensor y normalizar L2
        if hasattr(text_outputs, "text_embeds"):
            text_features = text_outputs.text_embeds
        elif hasattr(text_outputs, "pooler_output"):
            text_features = text_outputs.pooler_output
        else:
            text_features = text_outputs
            
        text_features = text_features / text_features.norm(p=2, dim=-1, keepdim=True)
        return text_features[0].cpu().numpy().tolist()

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