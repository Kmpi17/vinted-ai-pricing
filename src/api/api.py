from fastapi import FastAPI, UploadFile, File, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional
from src.search.search_service import SearchService

app = FastAPI(
    title="Vinted AI - Vector Search API",
    description="API REST para búsqueda semántica y visual con Qdrant y CLIP",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

search_service: Optional[SearchService] = None

@app.on_event("startup")
def startup_event():
    global search_service
    search_service = SearchService()

@app.get("/", tags=["Health Check"])
def health_check():
    return {"status": "ok", "message": "API activa"}

@app.get("/search/text", tags=["Search"])
def search_text(
    q: str = Query(..., description="Texto de búsqueda, ej: 'Nike blue t-shirt'"),
    limit: int = Query(10, ge=1, le=50),
    category: Optional[str] = Query(None),
    gender: Optional[str] = Query(None)
):
    try:
        results = search_service.search_by_text(text=q, limit=limit, category=category, gender=gender)
        return {"query": q, "total": len(results), "results": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/search/image", tags=["Search"])
async def search_image(
    file: UploadFile = File(...),
    limit: int = Query(10, ge=1, le=50),
    category: Optional[str] = Query(None),
    gender: Optional[str] = Query(None)
):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Formato de archivo no válido")

    try:
        contents = await file.read()
        results = search_service.search_by_image(image_bytes=contents, limit=limit, category=category, gender=gender)
        return {"filename": file.filename, "total": len(results), "results": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))