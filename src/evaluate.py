"""
evaluate.py
-----------
Post-training evaluation for the Maize Leaf Anomaly Detection system.

Usage
-----
    python src/evaluate.py

Steps performed
~~~~~~~~~~~~~~~
1. Load trained model and saved threshold (if available)
2. Load test images (healthy + diseased)
3. Compute per-image reconstruction error (MSE)
4. Generate ROC curve and compute AUC
5. Select operating threshold @ 95 % specificity
6. Save threshold → artifacts/threshold.json
7. Compute classification metrics at selected threshold
8. Plot ROC curve, confusion matrix, error distribution
9. Print evaluation report
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    auc,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_curve,
)

from src.config import (
    ARTIFACTS_DIR,
    CONFUSION_MATRIX_PLOT,
    DATA_RAW_DIR,
    ERROR_DIST_PLOT,
    PLOTS_DIR,
    RANDOM_SEED,
    ROC_CURVE_PLOT,
    TARGET_SPECIFICITY,
    THRESHOLD_PATH,
)
from src.data_loader import build_splits
from src.model import load_model
from src.preprocess import load_images_from_paths
from src.utils import get_logger, load_json, save_json, set_seed

logger = get_logger(__name__)


# ─── Anomaly scoring ─────────────────────────────────────────────────────────

def compute_reconstruction_errors(
    model,
    images: np.ndarray,
    batch_size: int = 32,
) -> np.ndarray:
    """
    Compute per-image MSE reconstruction error.

    Parameters
    ----------
    model      : Keras autoencoder
    images     : np.ndarray  (N, H, W, C)
    batch_size : int

    Returns
    -------
    np.ndarray  shape (N,)  — one error value per image
    """
    reconstructions = model.predict(images, batch_size=batch_size, verbose=0)
    # MSE per image: mean over (H, W, C)
    errors = np.mean((images - reconstructions) ** 2, axis=(1, 2, 3))
    return errors.astype(np.float64)


# ─── Threshold selection ─────────────────────────────────────────────────────

def select_threshold_at_specificity(
    fpr: np.ndarray,
    tpr: np.ndarray,
    thresholds: np.ndarray,
    target_specificity: float = TARGET_SPECIFICITY,
) -> tuple[float, float, float]:
    """
    Return the threshold closest to *target_specificity*.

    Specificity = 1 - FPR  ⟹  FPR = 1 - Specificity

    Parameters
    ----------
    fpr, tpr, thresholds : outputs of sklearn.metrics.roc_curve
    target_specificity   : float  (0-1)

    Returns
    -------
    Tuple[float, float, float]
        (threshold, achieved_specificity, achieved_sensitivity)
    """
    target_fpr  = 1.0 - target_specificity
    specificities = 1.0 - fpr

    # Find index of specificity closest to target
    idx = np.argmin(np.abs(specificities - target_specificity))

    chosen_threshold    = float(thresholds[idx])
    achieved_specificity = float(specificities[idx])
    achieved_sensitivity = float(tpr[idx])

    return chosen_threshold, achieved_specificity, achieved_sensitivity


# ─── Plotting helpers ────────────────────────────────────────────────────────

def plot_roc(
    fpr: np.ndarray,
    tpr: np.ndarray,
    roc_auc: float,
    operating_point: tuple,
    save_path: Path = ROC_CURVE_PLOT,
) -> None:
    """Plot ROC curve with the operating point marked."""
    fig, ax = plt.subplots(figsize=(8, 6))

    ax.plot(fpr, tpr, color="steelblue", lw=2,
            label=f"ROC Curve (AUC = {roc_auc:.4f})")
    ax.plot([0, 1], [0, 1], color="gray", lw=1, linestyle="--",
            label="Random Classifier")

    # Mark operating point
    op_fpr, op_tpr, op_thresh = operating_point
    ax.scatter([op_fpr], [op_tpr], color="crimson", s=100, zorder=5,
               label=f"Op. Point @ {TARGET_SPECIFICITY*100:.0f}% Specificity\n"
                     f"(Thr={op_thresh:.5f})")

    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate (1 – Specificity)", fontsize=12)
    ax.set_ylabel("True Positive Rate (Sensitivity)", fontsize=12)
    ax.set_title("ROC Curve — Maize Leaf Anomaly Detection", fontsize=13, fontweight="bold")
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(True, alpha=0.3)

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("ROC curve saved → %s", save_path)


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    save_path: Path = CONFUSION_MATRIX_PLOT,
    classes: list = None,
) -> None:
    """Plot a Seaborn heatmap confusion matrix."""
    if classes is None:
        classes = ["Healthy", "Diseased"]

    cm = confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=classes, yticklabels=classes,
        linewidths=0.5, ax=ax,
    )
    ax.set_ylabel("True Label", fontsize=12)
    ax.set_xlabel("Predicted Label", fontsize=12)
    ax.set_title("Confusion Matrix", fontsize=13, fontweight="bold")

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Confusion matrix saved → %s", save_path)


def plot_error_distribution(
    errors: np.ndarray,
    labels: list,
    threshold: float,
    save_path: Path = ERROR_DIST_PLOT,
) -> None:
    """Plot histogram of reconstruction errors for healthy vs diseased."""
    labels_arr = np.array(labels)
    healthy_errors  = errors[labels_arr == 0]
    diseased_errors = errors[labels_arr == 1]

    fig, ax = plt.subplots(figsize=(10, 5))

    ax.hist(healthy_errors,  bins=50, alpha=0.65, color="green",
            label=f"Healthy (n={len(healthy_errors)})", density=True)
    ax.hist(diseased_errors, bins=50, alpha=0.65, color="red",
            label=f"Diseased (n={len(diseased_errors)})", density=True)
    ax.axvline(threshold, color="black", linestyle="--", linewidth=2,
               label=f"Threshold = {threshold:.5f}")

    ax.set_xlabel("Reconstruction Error (MSE)", fontsize=12)
    ax.set_ylabel("Density", fontsize=12)
    ax.set_title("Reconstruction Error Distribution", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Error distribution plot saved → %s", save_path)


# ─── Main evaluation function ────────────────────────────────────────────────

def evaluate() -> dict:
    """
    Run the full evaluation pipeline and return a metrics dict.

    Returns
    -------
    dict  — all computed metrics
    """
    logger.info("=" * 60)
    logger.info("Maize Leaf Anomaly Detection — Evaluation")
    logger.info("=" * 60)

    set_seed(RANDOM_SEED)

    # ── 1. Load model ─────────────────────────────────────────────────────────
    logger.info("Loading model …")
    model = load_model()

    # ── 2. Load test data ────────────────────────────────────────────────────
    logger.info("Building test split …")
    _, _, test_paths, test_labels = build_splits(data_dir=DATA_RAW_DIR, seed=RANDOM_SEED)

    logger.info("Loading %d test images …", len(test_paths))
    test_images = load_images_from_paths(test_paths)
    # Align labels with successfully loaded images (load_images_from_paths skips bad files)
    # We trust all test images loaded correctly; if lengths differ, truncate labels
    if len(test_images) != len(test_labels):
        logger.warning("Image count (%d) ≠ label count (%d); truncating labels.",
                       len(test_images), len(test_labels))
        test_labels = test_labels[: len(test_images)]
    test_labels = np.array(test_labels)

    # ── 3. Compute reconstruction errors ─────────────────────────────────────
    logger.info("Computing reconstruction errors …")
    errors = compute_reconstruction_errors(model, test_images)

    # ── 4. ROC + AUC ─────────────────────────────────────────────────────────
    logger.info("Computing ROC curve …")
    fpr, tpr, thresholds = roc_curve(test_labels, errors, pos_label=1)
    roc_auc = auc(fpr, tpr)
    logger.info("AUC = %.4f", roc_auc)

    # ── 5. Threshold @ 95 % specificity ──────────────────────────────────────
    threshold, spec, sens = select_threshold_at_specificity(
        fpr, tpr, thresholds, TARGET_SPECIFICITY
    )
    op_fpr = 1.0 - spec
    logger.info("Threshold @ %.0f%% specificity: %.6f  (spec=%.4f, sens=%.4f)",
                TARGET_SPECIFICITY * 100, threshold, spec, sens)

    # ── 6. Save threshold ─────────────────────────────────────────────────────
    threshold_data = {
        "threshold":   round(threshold, 6),
        "specificity": round(spec, 4),
        "sensitivity": round(sens, 4),
        "auc":         round(roc_auc, 4),
    }
    save_json(threshold_data, THRESHOLD_PATH)
    logger.info("Threshold saved → %s", THRESHOLD_PATH)

    # ── 7. Classification metrics ─────────────────────────────────────────────
    y_pred = (errors >= threshold).astype(int)

    cm   = confusion_matrix(test_labels, y_pred)
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, cm[0, 0])

    precision = precision_score(test_labels, y_pred, zero_division=0)
    recall    = recall_score(test_labels, y_pred, zero_division=0)
    f1        = f1_score(test_labels, y_pred, zero_division=0)
    specificity_eval = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    sensitivity_eval = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    # ── 8. Plots ──────────────────────────────────────────────────────────────
    logger.info("Generating evaluation plots …")
    plot_roc(fpr, tpr, roc_auc,
             operating_point=(op_fpr, sens, threshold))
    plot_confusion_matrix(test_labels, y_pred)
    plot_error_distribution(errors, test_labels.tolist(), threshold)

    # ── 9. Report ─────────────────────────────────────────────────────────────
    metrics = {
        "auc":         round(roc_auc, 4),
        "threshold":   round(threshold, 6),
        "specificity": round(specificity_eval, 4),
        "sensitivity": round(sensitivity_eval, 4),
        "precision":   round(precision, 4),
        "recall":      round(recall, 4),
        "f1_score":    round(f1, 4),
        "tn": int(tn), "fp": int(fp),
        "fn": int(fn), "tp": int(tp),
    }

    logger.info("=" * 60)
    logger.info("Evaluation Report")
    logger.info("-" * 40)
    for k, v in metrics.items():
        logger.info("  %-15s: %s", k, v)
    logger.info("=" * 60)

    return metrics


# ─── Entry point ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    evaluate()
