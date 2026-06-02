"""
config.py
---------
Central configuration for the Maize Leaf Anomaly Detection project.
All paths, hyperparameters, and constants are defined here.
"""

import os
from pathlib import Path

# ─── Root ────────────────────────────────────────────────────────────────────
ROOT_DIR = Path(__file__).resolve().parent.parent

# ─── Directories ─────────────────────────────────────────────────────────────
DATA_RAW_DIR       = ROOT_DIR / "data" / "raw"
DATA_PROCESSED_DIR = ROOT_DIR / "data" / "processed"
MODELS_DIR         = ROOT_DIR / "models"
PLOTS_DIR          = ROOT_DIR / "plots"
ARTIFACTS_DIR      = ROOT_DIR / "artifacts"

# ─── Dataset ─────────────────────────────────────────────────────────────────
# Kaggle dataset: qramkrishna/corn-leaf-infection-dataset
# Expected sub-folders inside DATA_RAW_DIR after download:
#   Healthy/   ← used for training
#   Diseased/  ← used only during evaluation / inference
HEALTHY_CLASS_NAME  = "Healthy"
DISEASED_CLASS_NAME = "Diseased"

# ─── Image Processing ────────────────────────────────────────────────────────
IMAGE_SIZE   = (128, 128)        # (height, width)
IMAGE_SHAPE  = (128, 128, 3)     # H × W × C
NORM_MIN     = 0.0
NORM_MAX     = 1.0

# ─── Dataset Splits ──────────────────────────────────────────────────────────
TRAIN_RATIO      = 0.80
VALIDATION_RATIO = 0.20
RANDOM_SEED      = 42

# ─── Model ───────────────────────────────────────────────────────────────────
LATENT_DIM   = 128
ENCODER_FILTERS = [32, 64, 128]   # successive Conv2D filter counts
DECODER_FILTERS = [128, 64, 32]   # successive Conv2DTranspose counts

# ─── Training ────────────────────────────────────────────────────────────────
BATCH_SIZE          = 32
EPOCHS              = 100
LEARNING_RATE       = 1e-3
EARLY_STOP_PATIENCE = 10
EARLY_STOP_MONITOR  = "val_loss"

# ─── Artifacts ───────────────────────────────────────────────────────────────
MODEL_PATH     = MODELS_DIR    / "autoencoder.keras"
HISTORY_PATH   = ARTIFACTS_DIR / "training_history.pkl"
THRESHOLD_PATH = ARTIFACTS_DIR / "threshold.json"

# ─── Evaluation ──────────────────────────────────────────────────────────────
TARGET_SPECIFICITY = 0.95   # threshold selected at this specificity level

# ─── Plots ───────────────────────────────────────────────────────────────────
TRAINING_LOSS_PLOT    = PLOTS_DIR / "training_loss.png"
ROC_CURVE_PLOT        = PLOTS_DIR / "roc_curve.png"
CONFUSION_MATRIX_PLOT = PLOTS_DIR / "confusion_matrix.png"
RECONSTRUCTION_PLOT   = PLOTS_DIR / "reconstruction_examples.png"
ERROR_DIST_PLOT       = PLOTS_DIR / "error_distribution.png"

# ─── Ensure directories exist ────────────────────────────────────────────────
for _dir in [DATA_RAW_DIR, DATA_PROCESSED_DIR, MODELS_DIR, PLOTS_DIR, ARTIFACTS_DIR]:
    _dir.mkdir(parents=True, exist_ok=True)
