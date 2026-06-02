# PROJECT_CONTEXT.md
## Maize Leaf Anomaly Detection — Developer Context

This document explains the *why* behind every major design decision so that
future maintainers and contributors can extend the system without accidentally
breaking assumptions baked into the architecture.

---

## Problem Definition

### Why anomaly detection instead of classification?

A classical supervised classifier (e.g. ResNet with softmax) requires:

1. A large, balanced labelled dataset for **every** disease class.
2. Re-training whenever a new disease variant appears in the field.
3. A "catch-all" background class to avoid overconfidence on unseen diseases.

In practice:
- Healthy leaf data is **abundant** (farms always have healthy reference plants).
- Diseased leaf data is **scarce, unbalanced, and constantly evolving**.
- Farmers and agronomists care about "is this leaf normal?" rather than
  "which of these 12 specific diseases does it have?".

An autoencoder trained exclusively on healthy leaves models the **distribution
of normality**.  Any image that falls outside that distribution — for any
reason, including diseases never seen during development — produces a high
reconstruction error and is correctly flagged.

This is the standard **one-class / anomaly detection** formulation:

```
Train on NORMAL class only  →  model learns P(normal)
At inference: high error = low P(normal) = anomalous
```

---

## Dataset Understanding

### Kaggle: qramkrishna/corn-leaf-infection-dataset

| Folder   | Content | Role |
|----------|---------|------|
| Healthy  | ~1200 images of disease-free maize leaves | Training + test-negative |
| Diseased | ~1000 images with blight, rust, spot, etc. | Test-positive only |

**Critical constraint**: the model must **never** see diseased images during
training.  `data_loader.py::build_splits()` enforces this:
- `train_paths` and `val_paths` contain only healthy images.
- `test_paths` = held-out healthy (val set) + all diseased.

The validation set is reused as the test-healthy pool to avoid a three-way split
that would leave too few images for reliable ROC estimation.  If the dataset
grows, a dedicated `test/healthy` folder should be created.

---

## Data Flow

```
Kaggle download
      │
      ▼
data/raw/
  Healthy/   Diseased/
      │
      ▼  data_loader.py::build_splits()
      │
      ├──► train_paths   (80% healthy)
      ├──► val_paths     (20% healthy)
      └──► test_paths    (val healthy + all diseased)
                │
                ▼  preprocess.py::load_images_from_paths()
                │
      OpenCV read → BGR→RGB → resize(128,128) → /255 → float32
                │
                ▼  preprocess.py::build_tf_dataset()
                │
      tf.data.Dataset  (input, target) = (image, image)
                │
                ▼  model.py::build_autoencoder()
                │
      Convolutional Autoencoder  (trained on train/val healthy only)
                │
                ▼  evaluate.py::compute_reconstruction_errors()
                │
      per-image MSE error  (one scalar per test image)
                │
                ▼  evaluate.py::select_threshold_at_specificity()
                │
      threshold @ 95% specificity  →  artifacts/threshold.json
                │
                ▼  predict.py::AnomalyPredictor.predict()
                │
      { prediction, anomaly_score, threshold }
```

---

## Model Design Decisions

### Autoencoder choice

Alternatives considered:

| Architecture | Pros | Cons | Decision |
|---|---|---|---|
| PCA | Fast, interpretable | Linear, no spatial awareness | Rejected |
| Dense Autoencoder | Simple | Ignores spatial structure | Rejected |
| **Convolutional Autoencoder** | Spatially aware, good for images | More parameters | ✅ **Chosen** |
| Variational Autoencoder (VAE) | Theoretically better anomaly scoring | More complex training | Future work |

A CNN-based autoencoder is the sweet spot: it leverages spatial correlations
in leaf texture and venation patterns while remaining straightforward to train
and debug.

### Architecture specifics

```
Encoder
  Conv2D(32, 3×3) → BN → ReLU → MaxPool(2×2)
  Conv2D(64, 3×3) → BN → ReLU → MaxPool(2×2)
  Conv2D(128,3×3) → BN → ReLU → MaxPool(2×2)
  Flatten → Dense(128) [latent]

Decoder
  Dense(128 × 16 × 16) → Reshape(16,16,128)
  ConvT(128,3×3,stride=2) → BN → ReLU
  ConvT(64, 3×3,stride=2) → BN → ReLU
  ConvT(32, 3×3,stride=2) → BN → ReLU
  ConvT(3,  3×3) → Sigmoid  [output 128×128×3]
```

BatchNorm layers stabilise training and allow higher learning rates.

The `Sigmoid` output maps pixel values to [0,1] to match the normalised input,
making MSE a valid proxy for pixel-level fidelity.

### MSE reconstruction error

```
error(x) = mean( (x − decoder(encoder(x)))² )
```

MSE was chosen over MAE or SSIM because:
- It penalises large deviations heavily (disease lesions cause big pixel shifts).
- It is differentiable everywhere (smooth loss landscape during training).
- It is the natural complement to the MSE training loss (consistent metric).

### ROC thresholding

The ROC curve sweeps all possible thresholds and plots the sensitivity/
specificity trade-off.  This is preferable to picking a threshold manually
because:
- It is **data-driven** and reproducible.
- It decouples model training from deployment policy.
- Changing the operating point (e.g. from 95% to 99% specificity) requires
  **no retraining** — only a JSON update.

### 95% specificity requirement

Specificity = TN / (TN + FP) = fraction of healthy plants correctly classified.

A false positive (healthy classified as diseased) triggers:
- Unnecessary field treatment (pesticide cost + environmental load)
- Farmer distrust in the system

5% false-alarm rate was judged acceptable by domain stakeholders.
Sensitivity (recall on diseased) at this operating point is ~85-90% for a
well-trained model, meaning ~10-15% of diseased plants go undetected —
acceptable given that scouts do secondary verification.

---

## Deployment Flow

```
User / Mobile App
      │
      │  POST /predict  (multipart image upload)
      ▼
FastAPI  (app/main.py)
      │
      │  PIL.Image.open → resize(128,128) → numpy float32 / 255
      ▼
AnomalyPredictor  (src/predict.py)
      │
      │  model.predict(image_batch)
      ▼
Reconstruction  (128×128×3 float32)
      │
      │  MSE(original, reconstruction)
      ▼
Anomaly Score  (scalar float)
      │
      │  score >= threshold  ?
      ▼
"Diseased" / "Healthy"
      │
      ▼
JSON Response  →  User
```

The `AnomalyPredictor` is a singleton (lazy-loaded on first request via
`get_predictor()` in `app/main.py`) to avoid reloading the model on every call.

---

## Key Files Quick Reference

| File | Purpose |
|------|---------|
| `src/config.py` | Single source of truth for paths/hyperparams |
| `src/data_loader.py` | Dataset discovery, case-insensitive class folder matching |
| `src/preprocess.py` | OpenCV image loading, tf.data pipeline |
| `src/model.py` | Autoencoder definition, save/load |
| `src/train.py` | End-to-end training + EarlyStopping + plots |
| `src/evaluate.py` | ROC, AUC, threshold selection, metrics, confusion matrix |
| `src/predict.py` | `AnomalyPredictor` class + CLI |
| `app/main.py` | FastAPI endpoints `/health` and `/predict` |
| `tests/test_predict.py` | pytest suite (skips if model absent) |

---

## Configuration Reference (`src/config.py`)

| Variable | Default | Description |
|---|---|---|
| `IMAGE_SIZE` | (128,128) | Resize target (H,W) |
| `LATENT_DIM` | 128 | Dense bottleneck dimension |
| `ENCODER_FILTERS` | [32,64,128] | Conv2D filter progression |
| `BATCH_SIZE` | 32 | Training mini-batch |
| `EPOCHS` | 100 | Maximum epochs |
| `EARLY_STOP_PATIENCE` | 10 | EarlyStopping patience |
| `LEARNING_RATE` | 1e-3 | Adam LR |
| `TARGET_SPECIFICITY` | 0.95 | Threshold selection target |
| `RANDOM_SEED` | 42 | Global random seed |

---

## Future Extensions

### Variational Autoencoder (VAE)

Replace the encoder's final `Dense(latent_dim)` with two heads:
- `Dense(latent_dim)` for `mu` (mean)
- `Dense(latent_dim)` for `log_var` (log-variance)

Sample latent vector: `z = mu + exp(0.5 * log_var) * epsilon`

Add KL-divergence to the loss.  VAEs provide a smoother latent space and
theoretically better-calibrated anomaly scores.

### Denoising Autoencoder

During training, corrupt inputs with Gaussian noise:
```python
noisy = image + np.random.normal(0, 0.05, image.shape)
noisy = np.clip(noisy, 0, 1)
```
Train to reconstruct the clean image.  Forces more robust feature learning.

### Vision Transformers (ViT)

Replace the CNN layers with a patch-based Transformer encoder.
- Better long-range spatial dependencies
- State-of-the-art on many vision anomaly benchmarks
- Requires more data / pre-training

### Explainability — Grad-CAM

Compute gradient-weighted class activation maps on the decoder output to
highlight which image regions contributed most to the reconstruction error.
This gives agronomists spatial localisation of disease lesions.

```python
# Pseudocode
with tf.GradientTape() as tape:
    tape.watch(img_tensor)
    reconstruction = model(img_tensor)
    error = tf.reduce_mean(tf.square(img_tensor - reconstruction))
grads = tape.gradient(error, img_tensor)
heatmap = tf.reduce_mean(tf.abs(grads), axis=-1)
```

### Real-time Mobile Deployment

1. Export to TFLite: `model.save("autoencoder.tflite")` (use TFLiteConverter)
2. Quantize to INT8 for on-device inference
3. Wrap in an Android/iOS app with camera capture

### Multi-crop Support

The current dataset covers only maize.  Extending to wheat, rice, potato:
1. Train a separate autoencoder per crop (isolated normal distributions)
2. Or use a shared encoder with crop-specific decoder heads
3. Route inference based on GPS location / user-selected crop type
