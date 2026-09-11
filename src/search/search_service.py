import io
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor
from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels
from src.config import config

class SearchService:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"--> Inicializando SearchService en dispositivo: {self.device}")
        
        clip_model_id = getattr(config, "CLIP_MODEL_ID", "openai/clip-vit-base-patch32")
        self.model = CLIPModel.from_pretrained(clip_model_id).to(self.device)
        self.processor = CLIPProcessor.from_pretrained(clip_model_id)
        self.model.eval()

        self.client = QdrantClient(host=config.QDRANT_HOST, port=config.QDRANT_PORT)

    def search_by_text(self, text: str, limit: int = 10, category: str = None, gender: str = None):
        with torch.no_grad():
            inputs = self.processor(text=[text], return_tensors="pt", padding=True).to(self.device)
            outputs = self.model.get_text_features(**inputs)
            
            # Si outputs es un objeto BaseModelOutput, extraemos el tensor
            if hasattr(outputs, "text_embeds"):
                text_features = outputs.text_embeds
            elif hasattr(outputs, "pooler_output"):
                text_features = outputs.pooler_output
            elif isinstance(outputs, torch.Tensor):
                text_features = outputs
            else:
                text_features = outputs[0]

            # Normalizar vector L2
            text_features = text_features / text_features.norm(p=2, dim=-1, keepdim=True)
            query_vector = text_features[0].cpu().numpy().tolist()

        return self._query_qdrant(query_vector, limit, category, gender)

    def search_by_image(self, image_bytes: bytes, limit: int = 10, category: str = None, gender: str = None):
        with torch.no_grad():
            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            inputs = self.processor(images=image, return_tensors="pt").to(self.device)
            outputs = self.model.get_image_features(**inputs)
            
            if hasattr(outputs, "image_embeds"):
                image_features = outputs.image_embeds
            elif hasattr(outputs, "pooler_output"):
                image_features = outputs.pooler_output
            elif isinstance(outputs, torch.Tensor):
                image_features = outputs
            else:
                image_features = outputs[0]

            # Normalizar vector L2
            image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)
            query_vector = image_features[0].cpu().numpy().tolist()

        return self._query_qdrant(query_vector, limit, category, gender)

    def _query_qdrant(self, vector: list, limit: int, category: str = None, gender: str = None):
        must_filters = []
        
        if category:
            must_filters.append(
                qmodels.FieldCondition(
                    key="masterCategory",
                    match=qmodels.MatchValue(value=category)
                )
            )
        if gender:
            must_filters.append(
                qmodels.FieldCondition(
                    key="gender",
                    match=qmodels.MatchValue(value=gender)
                )
            )

        query_filter = qmodels.Filter(must=must_filters) if must_filters else None

        # API moderna de Qdrant client (reemplaza a self.client.search)
        response = self.client.query_points(
            collection_name=config.COLLECTION_NAME,
            query=vector,
            query_filter=query_filter,
            limit=limit
        )

        return [
            {
                "id": hit.id,
                "score": round(hit.score, 4),
                "payload": hit.payload
            }
            for hit in response.points
        ]