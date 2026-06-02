"""
tests/test_predict.py
---------------------
Unit tests for the AnomalyPredictor and supporting utilities.

Run with:
    pytest tests/ -v

Note: Tests that require the trained model are skipped if the model
      or threshold file is not found, so the test suite can run in CI
      before training.
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import IMAGE_SHAPE, MODEL_PATH, THRESHOLD_PATH
from src.preprocess import load_image
from src.utils import load_json, save_json


# ─── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture
def dummy_image_path(tmp_path: Path) -> Path:
    """Create a small random JPEG image for testing."""
    import cv2
    img = (np.random.rand(128, 128, 3) * 255).astype(np.uint8)
    path = tmp_path / "test_leaf.jpg"
    cv2.imwrite(str(path), img)
    return path


@pytest.fixture
def dummy_image_array() -> np.ndarray:
    """Return a random normalised (128, 128, 3) float32 array."""
    rng = np.random.default_rng(42)
    return rng.random((128, 128, 3)).astype(np.float32)


@pytest.fixture
def threshold_file(tmp_path: Path) -> Path:
    """Write a minimal threshold.json for testing."""
    data = {"threshold": 0.02, "specificity": 0.95, "sensitivity": 0.80, "auc": 0.93}
    p = tmp_path / "threshold.json"
    save_json(data, p)
    return p


# ─── Utils tests ──────────────────────────────────────────────────────────────

class TestUtils:
    def test_save_load_json(self, tmp_path: Path) -> None:
        data = {"key": "value", "number": 3.14}
        path = tmp_path / "test.json"
        save_json(data, path)
        loaded = load_json(path)
        assert loaded == data

    def test_load_json_missing_file(self) -> None:
        with pytest.raises(FileNotFoundError):
            load_json(Path("/nonexistent/path/file.json"))


# ─── Preprocess tests ─────────────────────────────────────────────────────────

class TestPreprocess:
    def test_load_image_valid(self, dummy_image_path: Path) -> None:
        img = load_image(dummy_image_path)
        assert img is not None
        assert img.shape == IMAGE_SHAPE
        assert img.dtype == np.float32
        assert img.min() >= 0.0
        assert img.max() <= 1.0

    def test_load_image_missing(self, tmp_path: Path) -> None:
        img = load_image(tmp_path / "nonexistent.jpg")
        assert img is None

    def test_load_image_normalisation(self, dummy_image_path: Path) -> None:
        img = load_image(dummy_image_path)
        assert img is not None
        assert 0.0 <= img.min() <= img.max() <= 1.0


# ─── AnomalyPredictor tests ───────────────────────────────────────────────────

class TestAnomalyPredictor:
    """
    Tests for AnomalyPredictor.
    Tests that require model files are skipped if they don't exist.
    """

    def test_predict_file_not_found(
        self, threshold_file: Path, tmp_path: Path
    ) -> None:
        """Predicting a nonexistent image raises FileNotFoundError."""
        if not MODEL_PATH.exists():
            pytest.skip("Trained model not found; skipping inference tests.")

        from src.predict import AnomalyPredictor
        predictor = AnomalyPredictor(threshold_path=threshold_file)
        with pytest.raises(FileNotFoundError):
            predictor.predict(tmp_path / "ghost.jpg")

    def test_predict_array_output_schema(
        self, dummy_image_array: np.ndarray, threshold_file: Path
    ) -> None:
        """predict_array returns the correct schema."""
        if not MODEL_PATH.exists():
            pytest.skip("Trained model not found; skipping inference tests.")

        from src.predict import AnomalyPredictor
        predictor = AnomalyPredictor(threshold_path=threshold_file)
        result = predictor.predict_array(dummy_image_array)

        assert "prediction" in result
        assert "anomaly_score" in result
        assert "threshold" in result
        assert result["prediction"] in ("Healthy", "Diseased")
        assert isinstance(result["anomaly_score"], float)
        assert result["anomaly_score"] >= 0.0

    def test_predict_from_file(
        self, dummy_image_path: Path, threshold_file: Path
    ) -> None:
        """predict() from a real image file returns valid output."""
        if not MODEL_PATH.exists():
            pytest.skip("Trained model not found; skipping inference tests.")

        from src.predict import AnomalyPredictor
        predictor = AnomalyPredictor(threshold_path=threshold_file)
        result = predictor.predict(dummy_image_path)

        assert result["prediction"] in ("Healthy", "Diseased")
        assert 0.0 <= result["anomaly_score"]

    def test_threshold_applied_correctly(self, threshold_file: Path) -> None:
        """A score below threshold → Healthy; score above → Diseased."""
        if not MODEL_PATH.exists():
            pytest.skip("Trained model not found; skipping inference tests.")

        from src.predict import AnomalyPredictor
        predictor = AnomalyPredictor(threshold_path=threshold_file)

        thr = predictor.threshold

        # Mock _reconstruction_error to control score
        predictor._reconstruction_error = MagicMock(return_value=thr - 0.001)
        result = predictor.predict_array(np.zeros((128, 128, 3), dtype=np.float32))
        assert result["prediction"] == "Healthy"

        predictor._reconstruction_error = MagicMock(return_value=thr + 0.001)
        result = predictor.predict_array(np.zeros((128, 128, 3), dtype=np.float32))
        assert result["prediction"] == "Diseased"


# ─── Model architecture tests ─────────────────────────────────────────────────

class TestModelArchitecture:
    def test_build_autoencoder_output_shape(self) -> None:
        """Autoencoder output must match input shape."""
        from src.model import build_autoencoder
        model = build_autoencoder()

        batch = np.random.rand(2, 128, 128, 3).astype(np.float32)
        output = model.predict(batch, verbose=0)
        assert output.shape == batch.shape

    def test_autoencoder_output_range(self) -> None:
        """Sigmoid output must be in [0, 1]."""
        from src.model import build_autoencoder
        model = build_autoencoder()

        batch = np.random.rand(4, 128, 128, 3).astype(np.float32)
        output = model.predict(batch, verbose=0)
        assert output.min() >= 0.0 - 1e-5
        assert output.max() <= 1.0 + 1e-5

    def test_autoencoder_parameter_count(self) -> None:
        """Sanity-check that the model has a reasonable number of parameters."""
        from src.model import build_autoencoder
        model = build_autoencoder()
        n_params = model.count_params()
        assert n_params > 100_000, f"Model has only {n_params} params — likely misconfigured."
        assert n_params < 50_000_000, f"Model has {n_params} params — suspiciously large."
