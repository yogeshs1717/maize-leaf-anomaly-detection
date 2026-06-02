"""
train.py
--------
End-to-end training script for the Maize Leaf Anomaly Detection autoencoder.

Usage
-----
    python src/train.py

Steps performed
~~~~~~~~~~~~~~~
1. Set random seeds for reproducibility
2. Load & split the dataset (healthy only for training/validation)
3. Preprocess images (resize + normalise)
4. Build and compile the Convolutional Autoencoder
5. Train with EarlyStopping
6. Save model  → models/autoencoder.keras
7. Save history → artifacts/training_history.pkl
8. Plot training curves → plots/training_loss.png
9. Save reconstruction examples → plots/reconstruction_examples.png
"""

from __future__ import annotations

import sys
from pathlib import Path

# Allow running as `python src/train.py` from the project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

from src.config import (
    BATCH_SIZE,
    DATA_RAW_DIR,
    EARLY_STOP_MONITOR,
    EARLY_STOP_PATIENCE,
    EPOCHS,
    HISTORY_PATH,
    MODEL_PATH,
    PLOTS_DIR,
    RANDOM_SEED,
    RECONSTRUCTION_PLOT,
    TRAINING_LOSS_PLOT,
)
from src.data_loader import build_splits
from src.model import build_autoencoder, save_model
from src.preprocess import load_images_from_paths, prepare_datasets
from src.utils import get_logger, save_pickle, set_seed

logger = get_logger(__name__)


# ─── Plotting helpers ────────────────────────────────────────────────────────

def plot_training_history(history: dict, save_path: Path = TRAINING_LOSS_PLOT) -> None:
    """
    Plot train vs. validation loss and MAE curves.

    Parameters
    ----------
    history : dict
        Keras History.history dict.
    save_path : Path
        Where to save the PNG.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Autoencoder Training", fontsize=14, fontweight="bold")

    # Loss
    axes[0].plot(history["loss"],     label="Train Loss", linewidth=2)
    axes[0].plot(history["val_loss"], label="Val Loss",   linewidth=2, linestyle="--")
    axes[0].set_title("MSE Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("MSE")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # MAE
    if "mae" in history:
        axes[1].plot(history["mae"],     label="Train MAE", linewidth=2)
        axes[1].plot(history["val_mae"], label="Val MAE",   linewidth=2, linestyle="--")
        axes[1].set_title("Mean Absolute Error")
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("MAE")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Training loss plot saved → %s", save_path)


def plot_reconstructions(
    model,
    images: np.ndarray,
    label: str,
    save_path: Path,
    n: int = 8,
) -> None:
    """
    Display *n* original vs. reconstructed image pairs in a grid.

    Parameters
    ----------
    model       : Keras model
    images      : np.ndarray  (N, H, W, 3)
    label       : str         Title prefix
    save_path   : Path
    n           : int         Number of examples to show
    """
    n = min(n, len(images))
    indices = np.random.choice(len(images), size=n, replace=False)
    sample  = images[indices]
    recon   = model.predict(sample, verbose=0)

    fig, axes = plt.subplots(2, n, figsize=(2.5 * n, 5))
    fig.suptitle(f"{label}: Original (top) vs Reconstruction (bottom)",
                 fontsize=12, fontweight="bold")

    for i in range(n):
        axes[0, i].imshow(np.clip(sample[i], 0, 1))
        axes[0, i].axis("off")
        axes[1, i].imshow(np.clip(recon[i], 0, 1))
        axes[1, i].axis("off")

    plt.tight_layout()
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Reconstruction plot saved → %s", save_path)


# ─── Main training function ──────────────────────────────────────────────────

def train() -> None:
    """Run the full training pipeline."""
    logger.info("=" * 60)
    logger.info("Maize Leaf Anomaly Detection — Training")
    logger.info("=" * 60)

    # 1. Reproducibility
    set_seed(RANDOM_SEED)

    # 2. Dataset splits
    logger.info("Step 1/6 — Building dataset splits …")
    train_paths, val_paths, test_paths, test_labels = build_splits(
        data_dir=DATA_RAW_DIR,
        seed=RANDOM_SEED,
    )

    # 3. Preprocessing
    logger.info("Step 2/6 — Preprocessing images …")
    train_ds, val_ds, train_images, val_images = prepare_datasets(
        train_paths, val_paths, batch_size=BATCH_SIZE, seed=RANDOM_SEED
    )

    # 4. Model
    logger.info("Step 3/6 — Building model …")
    model = build_autoencoder()

    # 5. Callbacks
    callbacks = [
        EarlyStopping(
            monitor=EARLY_STOP_MONITOR,
            patience=EARLY_STOP_PATIENCE,
            restore_best_weights=True,
            verbose=1,
        ),
        ReduceLROnPlateau(
            monitor=EARLY_STOP_MONITOR,
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=1,
        ),
        ModelCheckpoint(
            filepath=str(MODEL_PATH),
            monitor=EARLY_STOP_MONITOR,
            save_best_only=True,
            verbose=1,
        ),
    ]

    # 6. Training
    logger.info("Step 4/6 — Training (max %d epochs, patience %d) …",
                EPOCHS, EARLY_STOP_PATIENCE)
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        callbacks=callbacks,
        verbose=1,
    )

    # 7. Save artefacts
    logger.info("Step 5/6 — Saving model and history …")
    save_model(model, MODEL_PATH)
    save_pickle(history.history, HISTORY_PATH)
    logger.info("History saved → %s", HISTORY_PATH)

    # 8. Plots
    logger.info("Step 6/6 — Generating plots …")
    plot_training_history(history.history, TRAINING_LOSS_PLOT)

    # Healthy reconstruction examples
    plot_reconstructions(
        model, val_images, label="Healthy",
        save_path=PLOTS_DIR / "healthy_reconstruction.png",
    )

    # Diseased reconstruction examples (if available)
    from src.data_loader import discover_dataset
    from src.config import DISEASED_CLASS_NAME
    try:
        class_map      = discover_dataset(DATA_RAW_DIR)
        diseased_paths = class_map.get(DISEASED_CLASS_NAME, [])
        if diseased_paths:
            diseased_images = load_images_from_paths(diseased_paths[:50])
            plot_reconstructions(
                model, diseased_images, label="Diseased",
                save_path=PLOTS_DIR / "diseased_reconstruction.png",
            )
    except Exception as exc:
        logger.warning("Could not plot diseased reconstructions: %s", exc)

    logger.info("=" * 60)
    logger.info("Training complete!")
    logger.info("  Model   → %s", MODEL_PATH)
    logger.info("  History → %s", HISTORY_PATH)
    logger.info("  Plots   → %s/", PLOTS_DIR)
    logger.info("=" * 60)
    logger.info("Next step: python src/evaluate.py")


# ─── Entry point ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    train()
