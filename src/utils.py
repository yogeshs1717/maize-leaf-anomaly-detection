"""
utils.py
--------
General-purpose utility functions: logging setup, reproducibility,
directory helpers, and JSON I/O.
"""

import json
import logging
import os
import pickle
import random
from pathlib import Path
from typing import Any

import numpy as np
import tensorflow as tf


# ─── Logging ─────────────────────────────────────────────────────────────────

def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """
    Create (or retrieve) a named logger with a consistent format.

    Parameters
    ----------
    name : str
        Logger name (typically ``__name__`` of the calling module).
    level : int
        Logging level (default: INFO).

    Returns
    -------
    logging.Logger
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        fmt = logging.Formatter(
            "[%(asctime)s] %(levelname)-8s %(name)s — %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(fmt)
        logger.addHandler(handler)
    logger.setLevel(level)
    return logger


# ─── Reproducibility ─────────────────────────────────────────────────────────

def set_seed(seed: int = 42) -> None:
    """
    Fix all random seeds for reproducible results.

    Parameters
    ----------
    seed : int
        Seed value (default: 42).
    """
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


# ─── Pickle helpers ──────────────────────────────────────────────────────────

def save_pickle(obj: Any, path: Path) -> None:
    """Serialize *obj* to *path* using pickle."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as fh:
        pickle.dump(obj, fh)


def load_pickle(path: Path) -> Any:
    """Deserialize object from *path* using pickle."""
    with open(path, "rb") as fh:
        return pickle.load(fh)


# ─── JSON helpers ────────────────────────────────────────────────────────────

def save_json(data: dict, path: Path) -> None:
    """Save *data* dict as JSON to *path*."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fh:
        json.dump(data, fh, indent=4)


def load_json(path: Path) -> dict:
    """Load JSON file at *path* and return as dict."""
    with open(path, "r") as fh:
        return json.load(fh)


# ─── Directory helpers ───────────────────────────────────────────────────────

def ensure_dirs(*dirs: Path) -> None:
    """Create all supplied directories (and parents) if they don't exist."""
    for d in dirs:
        Path(d).mkdir(parents=True, exist_ok=True)
