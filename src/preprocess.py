"""
preprocess.py
-------------
Image preprocessing utilities:
  - load a single image (OpenCV → RGB → resize → normalize)
  - batch-load lists of image paths into NumPy arrays
  - build TensorFlow tf.data.Dataset objects for training
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
import tensorflow as tf

from src.config import BATCH_SIZE, IMAGE_SIZE, IMAGE_SHAPE, NORM_MAX, NORM_MIN
from src.utils import get_logger

logger = get_logger(__name__)


# ─── Single-image helpers ────────────────────────────────────────────────────

def load_image(
    path: Path | str,
    size: Tuple[int, int] = IMAGE_SIZE,
) -> Optional[np.ndarray]:
    """
    Load one image from disk, convert BGR→RGB, resize, and normalise to [0,1].

    Parameters
    ----------
    path : Path | str
        File system path to the image.
    size : Tuple[int, int]
        Target (height, width).  Defaults to ``IMAGE_SIZE`` from config.

    Returns
    -------
    np.ndarray | None
        Float32 array of shape (H, W, 3) in [0, 1], or *None* on failure.
    """
    path = str(path)
    img = cv2.imread(path)
    if img is None:
        logger.warning("Could not read image: %s", path)
        return None

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (size[1], size[0]), interpolation=cv2.INTER_AREA)
    img = img.astype(np.float32) / 255.0
    return img


# ─── Batch loading ───────────────────────────────────────────────────────────

def load_images_from_paths(
    paths: List[Path],
    size: Tuple[int, int] = IMAGE_SIZE,
) -> np.ndarray:
    """
    Load and preprocess a list of image paths into a single NumPy array.

    Parameters
    ----------
    paths : List[Path]
        List of image paths to load.
    size : Tuple[int, int]
        Target (height, width).

    Returns
    -------
    np.ndarray
        Array of shape (N, H, W, 3) dtype float32.
        Images that fail to load are silently skipped.
    """
    images: List[np.ndarray] = []
    skipped = 0
    for p in paths:
        img = load_image(p, size)
        if img is not None:
            images.append(img)
        else:
            skipped += 1

    if skipped:
        logger.warning("Skipped %d images that could not be read.", skipped)

    if not images:
        raise ValueError("No images could be loaded from the supplied paths.")

    array = np.stack(images, axis=0)
    logger.info("Loaded %d images → shape %s", len(images), array.shape)
    return array


# ─── TF Dataset builders ─────────────────────────────────────────────────────

def build_tf_dataset(
    images: np.ndarray,
    batch_size: int = BATCH_SIZE,
    shuffle: bool = True,
    seed: int = 42,
) -> tf.data.Dataset:
    """
    Wrap a NumPy image array in a ``tf.data.Dataset`` suitable for autoencoder
    training (input == target).

    Parameters
    ----------
    images : np.ndarray
        Preprocessed images of shape (N, H, W, 3).
    batch_size : int
        Mini-batch size.
    shuffle : bool
        Whether to shuffle the dataset each epoch.
    seed : int
        Shuffle seed.

    Returns
    -------
    tf.data.Dataset
        Yields ``(image_batch, image_batch)`` tuples (input = target).
    """
    dataset = tf.data.Dataset.from_tensor_slices(images)

    if shuffle:
        dataset = dataset.shuffle(buffer_size=len(images), seed=seed)

    dataset = (
        dataset
        .batch(batch_size, drop_remainder=False)
        .map(lambda x: (x, x), num_parallel_calls=tf.data.AUTOTUNE)
        .prefetch(tf.data.AUTOTUNE)
    )
    return dataset


def prepare_datasets(
    train_paths: List[Path],
    val_paths: List[Path],
    batch_size: int = BATCH_SIZE,
    seed: int = 42,
) -> Tuple[tf.data.Dataset, tf.data.Dataset, np.ndarray, np.ndarray]:
    """
    Full pipeline: load → preprocess → return train/val TF datasets plus
    the raw NumPy arrays (needed for plotting / evaluation).

    Parameters
    ----------
    train_paths : List[Path]
    val_paths   : List[Path]
    batch_size  : int
    seed        : int

    Returns
    -------
    Tuple[tf.data.Dataset, tf.data.Dataset, np.ndarray, np.ndarray]
        (train_ds, val_ds, train_images, val_images)
    """
    logger.info("Loading training images …")
    train_images = load_images_from_paths(train_paths)

    logger.info("Loading validation images …")
    val_images = load_images_from_paths(val_paths)

    train_ds = build_tf_dataset(train_images, batch_size=batch_size, shuffle=True,  seed=seed)
    val_ds   = build_tf_dataset(val_images,   batch_size=batch_size, shuffle=False, seed=seed)

    return train_ds, val_ds, train_images, val_images
