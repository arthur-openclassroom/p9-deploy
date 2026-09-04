"""
API de prediction de segmentation semantique.
Prend une image en entree et renvoie le masque de segmentation predit.
"""

import io
import numpy as np
from PIL import Image
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from .model_loader import load_model, predict, CATEGORIES, CATEGORY_COLORS, WEIGHTS_PATH

app = FastAPI(
    title="API Segmentation Semantique - Future Vision Transport",
    description="API de segmentation d'images pour vehicules autonomes (SegFormer MiT-B0)",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

model, processor, device = load_model()


@app.get("/")
def root():
    return {
        "message": "API Segmentation Semantique - Future Vision Transport",
        "endpoints": {
            "/predict": "POST - Segmentation d'une image (renvoie le masque en PNG)",
            "/predict/json": "POST - Segmentation d'une image (renvoie les classes en JSON)",
            "/categories": "GET - Liste des categories",
            "/health": "GET - Etat de l'API",
        },
    }


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model": "SegFormer MiT-B0",
        "device": str(device),
        "weights": str(WEIGHTS_PATH),
    }


@app.get("/categories")
def get_categories():
    return {
        "categories": CATEGORIES,
        "colors": {k: list(v) for k, v in CATEGORY_COLORS.items()},
    }


@app.post("/predict")
async def predict_image(file: UploadFile = File(...)):
    """Segmente une image et renvoie le masque RGB en PNG."""
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Le fichier doit être une image.")

    contents = await file.read()
    try:
        img = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Impossible de lire l'image.")

    img_array = np.array(img)
    mask_rgb = predict(model, processor, device, img_array)

    mask_img = Image.fromarray(mask_rgb)
    buf = io.BytesIO()
    mask_img.save(buf, format="PNG")
    buf.seek(0)

    return StreamingResponse(buf, media_type="image/png")


@app.post("/predict/json")
async def predict_image_json(file: UploadFile = File(...)):
    """Segmente une image et renvoie les classes predites en JSON."""
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Le fichier doit être une image.")

    contents = await file.read()
    try:
        img = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Impossible de lire l'image.")

    img_array = np.array(img)
    # Un seul forward : le masque de classes suffit pour le comptage.
    mask_classes = predict(model, processor, device, img_array, return_classes=True)

    # Comptage des pixels par categorie
    unique, counts = np.unique(mask_classes, return_counts=True)
    total = counts.sum()
    distribution = {
        CATEGORIES[int(c)]: {
            "pixels": int(cnt),
            "proportion": round(float(cnt / total * 100), 2),
        }
        for c, cnt in zip(unique, counts)
    }

    return JSONResponse(content={
        "width": img_array.shape[1],
        "height": img_array.shape[0],
        "categories_detected": len(unique),
        "distribution": distribution,
    })
