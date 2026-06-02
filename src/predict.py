"""
predict.py
----------
Inference module for the Maize Leaf Anomaly Detection system.

Can be used:
  1. As a standalone CLI script  →  python src/predict.py <image_path>
  2. As an importable class      →  ``AnomalyPredictor``

Output schema
-------------
{
    "prediction":   "Healthy" | "Diseased",
    "anomaly_score": float,
    "threshold":     float
}
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Optional

# Allow `python src/predict.py` from project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from src.config import MODEL_PATH, THRESHOLD_PATH
from src.model import load_model
from src.preprocess import load_image
from src.utils import get_logger, load_json

logger = get_logger(__name__)


# ─── Predictor class ─────────────────────────────────────────────────────────

class AnomalyPredictor:
    """
    Wraps the trained autoencoder and threshold for single-image inference.

    Parameters
    ----------
    model_path     : Path  — path to saved ``.keras`` model
    threshold_path : Path  — path to saved ``threshold.json``

    Examples
    --------
    >>> predictor = AnomalyPredictor()
    >>> result = predictor.predict("path/to/leaf.jpg")
    >>> print(result)
    {'prediction': 'Diseased', 'anomaly_score': 0.0253, 'threshold': 0.0137}
    """

    def __init__(
        self,
        model_path: Path = MODEL_PATH,
        threshold_path: Path = THRESHOLD_PATH,
    ) -> None:
        logger.info("Loading model from: %s", model_path)
        self.model = load_model(model_path)

        logger.info("Loading threshold from: %s", threshold_path)
        threshold_data   = load_json(threshold_path)
        self.threshold   = float(threshold_data["threshold"])
        self.specificity = float(threshold_data.get("specificity", 0.0))
        logger.info("Operating threshold: %.6f  (specificity=%.4f)",
                    self.threshold, self.specificity)

    # ── Internal ─────────────────────────────────────────────────────────────

    def _reconstruction_error(self, image: np.ndarray) -> float:
        """
        Compute MSE reconstruction error for a single preprocessed image.

        Parameters
        ----------
        image : np.ndarray  (H, W, 3) float32  in [0, 1]

        Returns
        -------
        float  — scalar MSE error
        """
        # Model expects batch dimension
        img_batch  = np.expand_dims(image, axis=0)        # (1, H, W, 3)
        recon_batch = self.model.predict(img_batch, verbose=0)
        error = float(np.mean((img_batch - recon_batch) ** 2))
        return error

    # ── Public API ────────────────────────────────────────────────────────────

    def predict(self, image_path: str | Path) -> dict:
        """
        Predict whether a leaf image is healthy or diseased.

        Parameters
        ----------
        image_path : str | Path
            Path to the leaf image file.

        Returns
        -------
        dict with keys:
            - ``prediction``   : "Healthy" | "Diseased"
            - ``anomaly_score``: float — reconstruction MSE
            - ``threshold``    : float — decision boundary
        """
        image_path = Path(image_path)
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        image = load_image(image_path)
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")

        score      = self._reconstruction_error(image)
        prediction = "Diseased" if score >= self.threshold else "Healthy"

        result = {
            "prediction":    prediction,
            "anomaly_score": round(score, 6),
            "threshold":     round(self.threshold, 6),
        }
        return result

    def predict_array(self, image: np.ndarray) -> dict:
        """
        Predict from a pre-loaded image array (H, W, 3) float32 in [0,1].

        Parameters
        ----------
        image : np.ndarray

        Returns
        -------
        dict — same schema as :meth:`predict`
        """
        score      = self._reconstruction_error(image)
        prediction = "Diseased" if score >= self.threshold else "Healthy"
        return {
            "prediction":    prediction,
            "anomaly_score": round(score, 6),
            "threshold":     round(self.threshold, 6),
        }


# ─── CLI entry point ─────────────────────────────────────────────────────────

def main(argv: list[str] | None = None) -> None:
    """CLI: ``python src/predict.py <image_path>``."""
    argv = argv or sys.argv[1:]

    if not argv:
        print("Usage: python src/predict.py <path_to_image>")
        sys.exit(1)

    image_path = Path(argv[0])

    try:
        predictor = AnomalyPredictor()
        result    = predictor.predict(image_path)
    except FileNotFoundError as exc:
        logger.error("%s", exc)
        sys.exit(1)
    except Exception as exc:
        logger.error("Prediction failed: %s", exc)
        raise

    print(json.dumps(result, indent=4))


if __name__ == "__main__":
    main()
