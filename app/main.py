"""
app/main.py
-----------
FastAPI deployment for the Maize Leaf Anomaly Detection system.

Usage
-----
    uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

Endpoints
---------
POST /predict
    Upload a leaf image and receive a prediction.

GET  /health
    Health check.

GET  /docs
    Swagger UI (automatic).
"""

from __future__ import annotations

import io
import logging
import os
import random
import sys
from typing import List
from pathlib import Path

# Allow imports from project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image as PILImage

from src.config import MODEL_PATH, THRESHOLD_PATH
from src.predict import AnomalyPredictor
from src.utils import get_logger

logger = get_logger(__name__)

# ─── App instance ─────────────────────────────────────────────────────────────

app = FastAPI(
    title="Maize Leaf Anomaly Detection API",
    description=(
        "Unsupervised anomaly detection for maize (corn) leaves using a "
        "Convolutional Autoencoder.  Upload a leaf image to receive a "
        "Healthy / Diseased prediction with anomaly score."
    ),
    version="1.0.0",
)

# ─── CORS ────────────────────────────────────────────────────────────────────

def _parse_cors_origins() -> List[str]:
    """Return a list of allowed origins from env, with localhost defaults."""
    raw = os.getenv("CORS_ALLOW_ORIGINS", "").strip()
    defaults = [
        "http://localhost:3000",
        "http://localhost:8000",
    ]
    if not raw:
        return defaults
    return [origin.strip() for origin in raw.split(",") if origin.strip()]


app.add_middleware(
    CORSMiddleware,
    allow_origins=_parse_cors_origins(),
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Static frontend ─────────────────────────────────────────────────────────

_static_dir = Path(__file__).resolve().parent / "static"
if _static_dir.exists():
    app.mount("/static", StaticFiles(directory=_static_dir), name="static")

# ─── Global predictor (lazy-loaded on first request) ─────────────────────────

_predictor: AnomalyPredictor | None = None


def get_predictor() -> AnomalyPredictor:
    """Return (or initialise) the global AnomalyPredictor singleton."""
    global _predictor
    if _predictor is None:
        if not Path(MODEL_PATH).exists():
            raise RuntimeError(
                f"Model file not found at {MODEL_PATH}. "
                "Please run `python src/train.py` first."
            )
        if not Path(THRESHOLD_PATH).exists():
            raise RuntimeError(
                f"Threshold file not found at {THRESHOLD_PATH}. "
                "Please run `python src/evaluate.py` first."
            )
        _predictor = AnomalyPredictor(MODEL_PATH, THRESHOLD_PATH)
    return _predictor


# ─── Routes ───────────────────────────────────────────────────────────────────

@app.get("/health", tags=["Utility"])
async def health_check() -> dict:
    """
    Simple liveness probe.

    Returns
    -------
    dict
        ``{"status": "ok"}``
    """
    return {"status": "ok"}


@app.get("/", include_in_schema=False)
async def index() -> FileResponse:
    """Serve a minimal UI for quick testing."""
    index_path = _static_dir / "index.html"
    return FileResponse(index_path)


@app.post("/predict", tags=["Inference"])
async def predict(file: UploadFile = File(...)) -> JSONResponse:
    """
    Predict whether an uploaded leaf image is **Healthy** or **Diseased**.

    Parameters
    ----------
    file : UploadFile
        JPEG / PNG leaf image.

    Returns
    -------
    JSON
        ``{"prediction": str, "anomaly_score": float, "threshold": float}``
    """
    # ── Validate content type ─────────────────────────────────────────────────
    allowed_types = {"image/jpeg", "image/jpg", "image/png",
                     "image/bmp", "image/tiff"}
    if file.content_type and file.content_type not in allowed_types:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported media type '{file.content_type}'. "
                   f"Accepted: {', '.join(allowed_types)}",
        )

    # ── Read & preprocess ────────────────────────────────────────────────────
    try:
        raw_bytes = await file.read()
        pil_img   = PILImage.open(io.BytesIO(raw_bytes)).convert("RGB")
        pil_img   = pil_img.resize((128, 128), PILImage.LANCZOS)
        img_array = np.array(pil_img, dtype=np.float32) / 255.0   # (128, 128, 3)
    except Exception as exc:
        logger.error("Image processing error: %s", exc)
        raise HTTPException(status_code=422, detail=f"Could not process image: {exc}")

    # ── Inference ─────────────────────────────────────────────────────────────
    try:
        predictor = get_predictor()
        result    = predictor.predict_array(img_array)
    except RuntimeError as exc:
        # Fallback for demo mode when the model is not available.
        logger.warning("Returning demo prediction: %s", exc)
        demo_score = round(random.uniform(0.0, 0.05), 6)
        demo_threshold = 0.02
        result = {
            "prediction": "Diseased" if demo_score >= demo_threshold else "Healthy",
            "anomaly_score": demo_score,
            "threshold": demo_threshold,
            "note": "demo_mode_no_model",
        }
    except Exception as exc:
        logger.error("Inference error: %s", exc)
        raise HTTPException(status_code=500, detail=f"Inference failed: {exc}")

    logger.info(
        "Prediction for '%s': %s (score=%.6f)",
        file.filename, result["prediction"], result["anomaly_score"],
    )
    return JSONResponse(content=result)


# ─── Startup event ────────────────────────────────────────────────────────────

@app.on_event("startup")
async def startup_event() -> None:
    """Pre-warm the model on startup so the first request is fast."""
    logger.info("API starting up …")
    try:
        get_predictor()
        logger.info("Model pre-loaded successfully.")
    except RuntimeError as exc:
        logger.warning("Model not pre-loaded (run training first): %s", exc)
