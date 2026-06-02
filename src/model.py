"""
model.py
--------
Convolutional Autoencoder for unsupervised maize-leaf anomaly detection.

Architecture
~~~~~~~~~~~~
Encoder
    3× (Conv2D → BatchNorm → ReLU → MaxPool)
    Dense bottleneck → latent vector of size LATENT_DIM

Decoder
    Dense → reshape
    3× (Conv2DTranspose → BatchNorm → ReLU)
    Final Conv2DTranspose → Sigmoid  (restores 128×128×3)

Loss  : Mean Squared Error
Metric: MAE
Optim : Adam
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import tensorflow as tf
from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import (
    BatchNormalization,
    Conv2D,
    Conv2DTranspose,
    Dense,
    Flatten,
    Input,
    MaxPooling2D,
    Reshape,
    UpSampling2D,
)
from tensorflow.keras.optimizers import Adam

from src.config import (
    DECODER_FILTERS,
    ENCODER_FILTERS,
    IMAGE_SHAPE,
    LATENT_DIM,
    LEARNING_RATE,
    MODEL_PATH,
)
from src.utils import get_logger

logger = get_logger(__name__)


# ─── Architecture ────────────────────────────────────────────────────────────

def build_autoencoder(
    input_shape: Tuple[int, int, int] = IMAGE_SHAPE,
    encoder_filters: list = None,
    decoder_filters: list = None,
    latent_dim: int = LATENT_DIM,
    learning_rate: float = LEARNING_RATE,
) -> Model:
    """
    Build and compile the Convolutional Autoencoder.

    Parameters
    ----------
    input_shape : Tuple[int, int, int]
        (H, W, C) — defaults to ``IMAGE_SHAPE`` from config.
    encoder_filters : list
        List of filter counts for encoder Conv2D layers.
    decoder_filters : list
        List of filter counts for decoder Conv2DTranspose layers.
    latent_dim : int
        Size of the bottleneck dense layer.
    learning_rate : float
        Adam learning rate.

    Returns
    -------
    tf.keras.Model
        Compiled autoencoder model.
    """
    if encoder_filters is None:
        encoder_filters = ENCODER_FILTERS
    if decoder_filters is None:
        decoder_filters = DECODER_FILTERS

    H, W, C = input_shape

    # ── Encoder ──────────────────────────────────────────────────────────────
    inputs = Input(shape=input_shape, name="encoder_input")
    x = inputs

    for i, filters in enumerate(encoder_filters):
        x = Conv2D(filters, kernel_size=3, padding="same",
                   activation="relu", name=f"enc_conv_{i+1}")(x)
        x = BatchNormalization(name=f"enc_bn_{i+1}")(x)
        x = MaxPooling2D(pool_size=2, padding="same", name=f"enc_pool_{i+1}")(x)

    # Calculate spatial dims after pooling
    pool_factor = 2 ** len(encoder_filters)           # e.g. 8 for 3 pools
    spatial_h   = H // pool_factor                    # e.g. 16
    spatial_w   = W // pool_factor
    flat_dim    = spatial_h * spatial_w * encoder_filters[-1]

    x = Flatten(name="enc_flatten")(x)
    latent = Dense(latent_dim, activation="relu", name="latent")(x)

    # ── Decoder ──────────────────────────────────────────────────────────────
    x = Dense(flat_dim, activation="relu", name="dec_expand")(latent)
    x = Reshape((spatial_h, spatial_w, encoder_filters[-1]),
                name="dec_reshape")(x)

    for i, filters in enumerate(decoder_filters):
        x = Conv2DTranspose(filters, kernel_size=3, strides=2,
                            padding="same", activation="relu",
                            name=f"dec_convT_{i+1}")(x)
        x = BatchNormalization(name=f"dec_bn_{i+1}")(x)

    # Final reconstruction layer → Sigmoid to output [0, 1]
    outputs = Conv2DTranspose(C, kernel_size=3, padding="same",
                              activation="sigmoid",
                              name="decoder_output")(x)

    # ── Assemble & compile ────────────────────────────────────────────────────
    autoencoder = Model(inputs, outputs, name="ConvAutoencoder")
    autoencoder.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss="mse",
        metrics=["mae"],
    )
    logger.info("Autoencoder built — parameters: {:,}".format(
        autoencoder.count_params()))
    autoencoder.summary(print_fn=logger.info)
    return autoencoder


# ─── Persistence helpers ─────────────────────────────────────────────────────

def save_model(model: Model, path: Path = MODEL_PATH) -> None:
    """
    Save the Keras model in the native ``.keras`` format.

    Parameters
    ----------
    model : Model
    path  : Path
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(path))
    logger.info("Model saved → %s", path)


def load_model(path: Path = MODEL_PATH) -> Model:
    """
    Load a previously saved ``.keras`` model.

    Parameters
    ----------
    path : Path

    Returns
    -------
    tf.keras.Model
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Model file not found: {path}")
    model = tf.keras.models.load_model(str(path))
    logger.info("Model loaded ← %s", path)
    return model
