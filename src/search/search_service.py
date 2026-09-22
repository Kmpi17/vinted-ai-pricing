from typing import List, Optional, Dict, Any
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchText, MatchValue

from src.extraction.extractor import CLIPExtractor
from src.config import config


class SearchService:
    def __init__(self):
        self.extractor = CLIPExtractor()
        self.qdrant_client = QdrantClient(
            host=getattr(config, "QDRANT_HOST", "localhost"),
            port=getattr(config, "QDRANT_PORT", 6333)
        )
        self.collection_name = getattr(config, "COLLECTION_NAME", "fashion_products")

    def _extract_hits(self, response: Any) -> List[Any]:
        """Extrae la lista de puntos independientemente de si Qdrant devuelve un QueryResponse o una tupla."""
        if hasattr(response, "points"):
            return response.points
        elif isinstance(response, tuple):
            return response[0]
        return response

    def _build_metadata_filters(self, category: Optional[str] = None, gender: Optional[str] = None) -> List[FieldCondition]:
        must_conditions = []
        if category:
            must_conditions.append(
                FieldCondition(key="articleType", match=MatchValue(value=category))
            )
        if gender:
            must_conditions.append(
                FieldCondition(key="gender", match=MatchValue(value=gender))
            )
        return must_conditions

    def search_by_text(
        self, 
        text: str, 
        limit: int = 10, 
        category: Optional[str] = None, 
        gender: Optional[str] = None,
        score_threshold: float = 0.62
    ) -> List[Dict[str, Any]]:
        query_vector = self.extractor.generate_text_embedding(text)
        base_must = self._build_metadata_filters(category, gender)

        # 1. Búsqueda híbrida con coincidencia léxica
        hybrid_filter = Filter(
            must=base_must,
            should=[
                FieldCondition(
                    key="productDisplayName",
                    match=MatchText(text=text)
                )
            ]
        )

        exact_response = self.qdrant_client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            query_filter=hybrid_filter,
            limit=limit,
            score_threshold=score_threshold
        )
        exact_hits = self._extract_hits(exact_response)

        found_ids = {hit.id for hit in exact_hits}
        results = [
            {
                "id": hit.id,
                "score": round(hit.score, 4),
                "payload": hit.payload
            }
            for hit in exact_hits
        ]

        # 2. Fallback semántico si se necesitan más resultados
        if len(results) < limit:
            semantic_filter = Filter(must=base_must) if base_must else None

            semantic_response = self.qdrant_client.query_points(
                collection_name=self.collection_name,
                query=query_vector,
                query_filter=semantic_filter,
                limit=limit,
                score_threshold=score_threshold
            )
            semantic_hits = self._extract_hits(semantic_response)

            for hit in semantic_hits:
                if hit.id not in found_ids:
                    results.append({
                        "id": hit.id,
                        "score": round(hit.score, 4),
                        "payload": hit.payload
                    })
                    found_ids.add(hit.id)
                    if len(results) == limit:
                        break

        return results

    def search_by_image(
        self, 
        image_bytes: bytes, 
        limit: int = 10, 
        category: Optional[str] = None, 
        gender: Optional[str] = None,
        score_threshold: float = 0.60
    ) -> List[Dict[str, Any]]:
        query_vector = self.extractor.generate_image_embedding(image_bytes)

        base_must = self._build_metadata_filters(category, gender)
        image_filter = Filter(must=base_must) if base_must else None

        response = self.qdrant_client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            query_filter=image_filter,
            limit=limit,
            score_threshold=score_threshold
        )
        hits = self._extract_hits(response)

        return [
            {
                "id": hit.id,
                "score": round(hit.score, 4),
                "payload": hit.payload
            }
            for hit in hits
        ]