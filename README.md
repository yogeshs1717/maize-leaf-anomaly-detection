# Maize Leaf Anomaly Detection
### Unsupervised Disease Detection using Convolutional Autoencoders

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/)
[![TensorFlow 2.15](https://img.shields.io/badge/TensorFlow-2.15-orange.svg)](https://www.tensorflow.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110-green.svg)](https://fastapi.tiangolo.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## Project Overview

This project implements an **unsupervised anomaly detection** system for maize (corn) leaves.  A Convolutional Autoencoder is trained **exclusively on healthy leaf images**.  During inference, the model attempts to reconstruct any input image; diseased leaves produce significantly higher reconstruction errors, which serve as the anomaly score.

The system outputs:
- `prediction` — "Healthy" or "Diseased"
- `anomaly_score` — per-image MSE reconstruction error
- `threshold` — decision boundary (selected at 95% specificity)

---

## Problem Statement

Maize diseases (Northern leaf blight, Common rust, Gray leaf spot, etc.) cause 20–40 % annual yield losses.  Early detection is critical.  Classical supervised classifiers require large labelled datasets of every disease variant.  An unsupervised autoencoder approach:

- Requires only **healthy images** for training
- Naturally generalises to **unseen disease types**
- Provides a continuous **anomaly score** (not just a binary label)

---

## Dataset

**Corn Leaf Infection Dataset** — Kaggle  
URL: https://www.kaggle.com/datasets/qramkrishna/corn-leaf-infection-dataset

Classes:
| Class    | Approx. images | Used for |
|----------|---------------|---------|
| Healthy  | ~1200         | Training + evaluation |
| Diseased | ~1000         | Evaluation only |

Download instructions:

```bash
# Option 1 — Kaggle CLI
pip install kaggle
kaggle datasets download -d qramkrishna/corn-leaf-infection-dataset \
    -p data/raw --unzip

# Option 2 — Manual
# Download ZIP from Kaggle, extract into data/raw/
# Expected structure:
#   data/raw/Healthy/   (or any folder name containing "healthy")
#   data/raw/Diseased/  (or any folder name containing "diseased")
```

---

## Autoencoder Approach

```
Input Image (128×128×3)
        │
   ┌────▼────┐
   │ Encoder │  Conv2D × 3 + MaxPool → latent vector (128-D)
   └────┬────┘
        │  latent space
   ┌────▼────┐
   │ Decoder │  Conv2DTranspose × 3 → 128×128×3
   └────┬────┘
        │
   Reconstruction
        │
   MSE(Input, Reconstruction) = Anomaly Score
```

Key design decisions:
- **Train only on healthy images** → the model learns normal leaf texture
- **MSE reconstruction error** → higher for out-of-distribution (diseased) images
- **95 % specificity threshold** → at most 5 % of healthy leaves are falsely flagged

---

## Folder Structure

```
maize-leaf-anomaly-detection/
│
├── app/
│   └── main.py                  ← FastAPI application
│
├── artifacts/
│   ├── threshold.json           ← saved decision threshold
│   └── training_history.pkl     ← Keras training history
│
├── data/
│   ├── raw/                     ← Kaggle dataset (not committed)
│   └── processed/               ← reserved for cached arrays
│
├── models/
│   └── autoencoder.keras        ← trained model weights
│
├── notebooks/
│   └── exploration.ipynb        ← interactive walkthrough
│
├── plots/
│   ├── training_loss.png
│   ├── roc_curve.png
│   ├── confusion_matrix.png
│   ├── error_distribution.png
│   ├── healthy_reconstruction.png
│   └── diseased_reconstruction.png
│
├── src/
│   ├── config.py                ← all paths & hyperparameters
│   ├── data_loader.py           ← dataset discovery & splitting
│   ├── preprocess.py            ← image loading, resizing, normalisation
│   ├── model.py                 ← Convolutional Autoencoder
│   ├── train.py                 ← training pipeline
│   ├── evaluate.py              ← evaluation pipeline
│   ├── predict.py               ← inference module & CLI
│   └── utils.py                 ← logging, seeding, I/O helpers
│
├── tests/
│   └── test_predict.py          ← pytest test suite
│
├── requirements.txt
├── README.md
├── PROJECT_CONTEXT.md
└── .gitignore
```

---

## Installation

```bash
# 1. Clone
git clone <repo-url>
cd maize-leaf-anomaly-detection

# 2. Create virtual environment
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
```

---

## Training Steps

```bash
# 1. Download dataset into data/raw/
#    (see Dataset section above)

# 2. Train the autoencoder
python src/train.py

# Outputs:
#   models/autoencoder.keras
#   artifacts/training_history.pkl
#   plots/training_loss.png
#   plots/healthy_reconstruction.png
#   plots/diseased_reconstruction.png
```

Training configuration (`src/config.py`):
| Parameter | Value |
|-----------|-------|
| Image size | 128 × 128 |
| Encoder filters | [32, 64, 128] |
| Latent dim | 128 |
| Batch size | 32 |
| Max epochs | 100 |
| Early stopping patience | 10 |
| Optimizer | Adam (lr=1e-3) |
| Loss | MSE |

---

## Evaluation Steps

```bash
# Run full evaluation (requires trained model)
python src/evaluate.py

# Outputs:
#   artifacts/threshold.json      ← threshold at 95% specificity
#   plots/roc_curve.png
#   plots/confusion_matrix.png
#   plots/error_distribution.png
```

---

## API Usage

### Start the server

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### Health check

```bash
curl http://localhost:8000/health
# {"status":"ok"}
```

### Predict

```bash
curl -X POST http://localhost:8000/predict \
     -F "file=@path/to/leaf.jpg"
```

Response:
```json
{
    "prediction": "Diseased",
    "anomaly_score": 0.0253,
    "threshold": 0.0137
}
```

Swagger docs: http://localhost:8000/docs

---

## Example Predictions

### CLI

```bash
# Healthy leaf
python src/predict.py data/raw/Healthy/img001.jpg
# {
#     "prediction": "Healthy",
#     "anomaly_score": 0.007,
#     "threshold": 0.0137
# }

# Diseased leaf
python src/predict.py data/raw/Diseased/img001.jpg
# {
#     "prediction": "Diseased",
#     "anomaly_score": 0.029,
#     "threshold": 0.0137
# }
```

### Python API

```python
from src.predict import AnomalyPredictor

predictor = AnomalyPredictor()
result = predictor.predict("leaf.jpg")
print(result)
# {'prediction': 'Diseased', 'anomaly_score': 0.0253, 'threshold': 0.0137}
```

---

## ROC Analysis

The ROC curve plots Sensitivity (True Positive Rate) vs. 1−Specificity (False Positive Rate) across all possible thresholds.

- **AUC** — area under the curve; higher is better (1.0 = perfect)
- A well-trained autoencoder typically achieves AUC ≥ 0.90 on this dataset

The ROC curve is saved at `plots/roc_curve.png` after running `evaluate.py`.

---

## Threshold Selection at 95% Specificity

**Why 95% specificity?**

In agricultural screening, a false positive (flagging a healthy plant as diseased) triggers unnecessary pesticide treatment — costly and harmful.  We therefore require that **at least 95 % of healthy plants are correctly classified**.

**Procedure:**

```
1. Compute ROC curve (FPR, TPR, thresholds)
2. Specificity = 1 − FPR
3. Find index where |specificity − 0.95| is minimised
4. Use that threshold as the operating point
```

Saved to `artifacts/threshold.json`:
```json
{
    "threshold":   0.0137,
    "specificity": 0.9503,
    "sensitivity": 0.8812,
    "auc":         0.9421
}
```

---

## Tests

```bash
pytest tests/ -v
```

Tests cover: image loading, normalisation, model output shape, threshold logic, and predictor output schema.  Tests requiring a trained model are automatically skipped if the model file is absent.

---

## Future Improvements

- **Variational Autoencoder (VAE)** — richer latent space, better anomaly scores
- **Denoising Autoencoder** — improved robustness to image noise
- **Vision Transformer (ViT)** — attention-based reconstruction
- **Grad-CAM / Saliency Maps** — highlight diseased regions in the image
- **Real-time mobile deployment** — TFLite / CoreML export
- **Multi-class anomaly scoring** — per-disease severity estimation
- **Data augmentation** — rotation, flipping, colour jitter during training
