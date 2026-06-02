"""
data_loader.py
--------------
Discovers image paths from the raw dataset directory, separates
healthy from diseased samples, and returns structured splits ready
for the preprocessing pipeline.

Expected raw data layout
------------------------
data/raw/
    Healthy/
        img001.jpg
        img002.jpg
        ...
    Diseased/
        img001.jpg
        ...

The folder names are configurable via src/config.py.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from sklearn.model_selection import train_test_split

from src.config import (
    DATA_RAW_DIR,
    DISEASED_CLASS_NAME,
    HEALTHY_CLASS_NAME,
    RANDOM_SEED,
    TRAIN_RATIO,
    VALIDATION_RATIO,
)
from src.utils import get_logger

logger = get_logger(__name__)

# Supported image extensions
_IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}


# ─── Internal helpers ────────────────────────────────────────────────────────

def _collect_images(folder: Path) -> List[Path]:
    """
    Return a sorted list of image paths inside *folder*.

    Parameters
    ----------
    folder : Path
        Directory to scan (non-recursive).

    Returns
    -------
    List[Path]
        Sorted list of image file paths.
    """
    if not folder.exists():
        logger.warning("Folder not found: %s", folder)
        return []
    paths = sorted(
        p for p in folder.iterdir()
        if p.is_file() and p.suffix.lower() in _IMG_EXTS
    )
    return paths


def _find_class_folder(root: Path, class_name: str) -> Path:
    """
    Case-insensitively locate a sub-folder matching *class_name* inside *root*.
    Falls back to exact match first.

    Parameters
    ----------
    root : Path
        Root dataset directory.
    class_name : str
        Expected folder name (e.g. "Healthy").

    Returns
    -------
    Path
        Resolved folder path.

    Raises
    ------
    FileNotFoundError
        If no matching folder is found.
    """
    # Exact match
    exact = root / class_name
    if exact.exists():
        return exact

    # Case-insensitive search
    for entry in root.iterdir():
        if entry.is_dir() and entry.name.lower() == class_name.lower():
            return entry

    # Partial match (e.g. "Corn_Healthy" contains "healthy")
    for entry in root.iterdir():
        if entry.is_dir() and class_name.lower() in entry.name.lower():
            logger.info("Partial match found: '%s' for class '%s'", entry.name, class_name)
            return entry

    raise FileNotFoundError(
        f"Could not find a folder matching '{class_name}' inside {root}. "
        f"Available: {[e.name for e in root.iterdir() if e.is_dir()]}"
    )


# ─── Public API ──────────────────────────────────────────────────────────────

def discover_dataset(data_dir: Path = DATA_RAW_DIR) -> Dict[str, List[Path]]:
    """
    Walk *data_dir* and return a dict mapping class names to image paths.

    Parameters
    ----------
    data_dir : Path
        Root of the raw dataset (contains sub-folders per class).

    Returns
    -------
    Dict[str, List[Path]]
        ``{"Healthy": [...], "Diseased": [...]}``
    """
    logger.info("Scanning dataset at: %s", data_dir)

    healthy_folder  = _find_class_folder(data_dir, HEALTHY_CLASS_NAME)
    diseased_folder = _find_class_folder(data_dir, DISEASED_CLASS_NAME)

    healthy_paths  = _collect_images(healthy_folder)
    diseased_paths = _collect_images(diseased_folder)

    logger.info("Found %d healthy images in '%s'",  len(healthy_paths),  healthy_folder.name)
    logger.info("Found %d diseased images in '%s'", len(diseased_paths), diseased_folder.name)

    if not healthy_paths:
        raise ValueError(f"No images found in healthy folder: {healthy_folder}")

    return {
        HEALTHY_CLASS_NAME:  healthy_paths,
        DISEASED_CLASS_NAME: diseased_paths,
    }


def build_splits(
    data_dir: Path = DATA_RAW_DIR,
    train_ratio: float = TRAIN_RATIO,
    val_ratio: float = VALIDATION_RATIO,
    seed: int = RANDOM_SEED,
) -> Tuple[List[Path], List[Path], List[Path], List[int], List[Path], List[int]]:
    """
    Split the dataset into train / validation / test sets.

    * **Train** and **Validation** contain only healthy images.
    * **Test** contains the held-out healthy images + all diseased images.

    Parameters
    ----------
    data_dir : Path
        Root dataset directory.
    train_ratio : float
        Fraction of healthy images used for training.
    val_ratio : float
        Fraction of healthy images used for validation.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    Tuple of:
        train_paths   : List[Path]  — healthy images for training
        val_paths     : List[Path]  — healthy images for validation
        test_paths    : List[Path]  — mixed healthy + diseased
        test_labels   : List[int]   — 0=healthy, 1=diseased
    """
    assert abs(train_ratio + val_ratio - 1.0) < 1e-6, (
        "train_ratio + val_ratio must equal 1.0"
    )

    class_map = discover_dataset(data_dir)
    healthy_paths  = class_map[HEALTHY_CLASS_NAME]
    diseased_paths = class_map[DISEASED_CLASS_NAME]

    # Split healthy into train / val
    train_paths, val_paths = train_test_split(
        healthy_paths,
        test_size=val_ratio,
        random_state=seed,
        shuffle=True,
    )

    # Test set: all diseased + a held-out healthy portion for computing
    # true-negative rate (we reuse val for this to keep things simple,
    # ensuring the model never sees these during training).
    test_healthy  = val_paths           # healthy held-out
    test_diseased = diseased_paths      # all diseased

    test_paths  = list(test_healthy) + list(test_diseased)
    test_labels = [0] * len(test_healthy) + [1] * len(test_diseased)

    logger.info("Split summary:")
    logger.info("  Train      : %d healthy images", len(train_paths))
    logger.info("  Validation : %d healthy images", len(val_paths))
    logger.info("  Test       : %d healthy + %d diseased = %d total",
                len(test_healthy), len(test_diseased), len(test_paths))

    return train_paths, val_paths, test_paths, test_labels
