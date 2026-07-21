# NailDCMNet: A Decomposed Color-Morphology Network for Nail Disease Classification

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-blue?style=flat-square&logo=python"/>
  <img src="https://img.shields.io/badge/PyTorch-2.0+-ee4c2c?style=flat-square&logo=pytorch"/>
  <img src="https://img.shields.io/badge/timm-0.9+-green?style=flat-square"/>
  <img src="https://img.shields.io/badge/License-MIT-yellow?style=flat-square"/>
  <img src="https://img.shields.io/badge/Status-Complete-brightgreen?style=flat-square"/>
</p>

<p align="center">
  <b>Accuracy: 95.65% | Macro F1: 0.9589 | Macro Precision: 0.9589</b>
</p>

---

## 📌 Overview

**NailDCMNet** (Decomposed Color-Morphology Network) is a novel deep learning framework for automated nail disease classification. It introduces a **triple-stream parallel encoder** with **cross-stream attention fusion** that explicitly decomposes nail disease diagnosis into three biologically meaningful visual dimensions — color, texture, and morphology — mirroring the diagnostic reasoning of a dermatologist.

Nail diseases such as **Onychomycosis** and **Psoriasis** are clinically challenging to distinguish — both cause thickening, discoloration, and nail separation. NailDCMNet addresses this by processing each diagnostic dimension through a specialized backbone and fusing them via learnable cross-attention and adaptive stream weighting.

---

## 🗂️ Table of Contents

- [Dataset](#-dataset)
- [Architecture](#-architecture)
- [Results](#-results)
- [Baseline Comparison](#-baseline-comparison)
- [Requirements](#-requirements)
- [Usage](#-usage)
- [Project Structure](#-project-structure)
- [Key Design Decisions](#-key-design-decisions)
- [Novelty](#-novelty)

---

## 📦 Dataset

**Nail Disease Image Classification Dataset**
🔗 [Kaggle Link](https://www.kaggle.com/datasets/josephrasanjana/nail-disease-image-classification-dataset)

| Split | Healthy | Onychomycosis | Psoriasis | Total |
|-------|---------|---------------|-----------|-------|
| Train | 248 | 575| 342| 1165 |
| Test  | 62 | 146 | 91 | **299** |

nail_disease_dataset/
├── train/
│ ├── healthy/
│ ├── onychomycosis/
│ └── psoriasis/
└── test/
├── healthy/
├── onychomycosis/
└── psoriasis/
## 🏗️ Architecture

INPUT IMAGE [B, 3, 224, 224]
│
▼
┌─────────────────────────────────────────┐
│ STAGE 1 — Nail Region Focus (NRF) │
│ Conv(3→16)→Conv(16→8)→Conv(8→1) │
│ Soft mask M suppresses background │
│ X_f = X ⊙ (0.7·M + 0.3) │
└──────────────────┬──────────────────────┘
│
┌───────────┼───────────┐
▼ ▼ ▼
┌────────────┐ ┌────────────┐ ┌────────────┐
│ STREAM A │ │ STREAM B │ │ STREAM C │
│ COLOR │ │ TEXTURE │ │ MORPHOLOGY │
│ │ │ │ │ │
│ConvNeXtV2 │ │MobileNetV3 │ │ Swin-Tiny │
│ -Tiny │ │ -Large │ │ │
│ │ │ │ │ │
│ Channel │ │ Spatial │ │ Spatial │
│ Attention │ │ Attention │ │ Attention │
│(SE r=16) │ │(CBAM) │ │(CBAM) │
│ │ │ │ │ │
│[B,256,7,7] │ │[B,256,7,7] │ │[B,256,7,7] │
└─────┬──────┘ └─────┬──────┘ └─────┬──────┘
│Fa │Fb │Fc
└───────────┬──┘ │
▼ │
┌───────────────────────┐ │
│ Cross-Attention A→B │ │
│ Q=Fb, K=Fa, V=Fa │ │
│ Fb* = LN(Fb + Attn) │ │
└───────────┬───────────┘ │
│Fb* │Fc
└────────┬────────┘
▼
┌─────────────────────────┐
│ Cross-Attention B*→C │
│ Q=Fc, K=Fb*, V=Fb* │
│ Fc* = LN(Fc + Attn) │
└────────────┬────────────┘
│
▼
┌────────────────────────────────┐
│ Adaptive Stream Weighting │
│ w = Softmax(Linear(GAP(cat))) │
│ F_w = w₁·Fa + w₂·Fb* + w₃·Fc*│
└───────────────┬────────────────┘
│
▼
┌────────────────────────────────┐
│ STAGE 4 — Dual Attn Gate │
│ Channel Gate (SE-style) │
│ Spatial Gate (CBAM-style) │
│ F_g = gate_proj([F_ch;F_sp]) │
│ + F_w ← residual │
└───────────────┬────────────────┘
│
▼
┌────────────────────────────────┐
│ STAGE 5 — Multi-Pool Agg. │
│ R1 = GlobalAvgPool [global] │
│ R2 = GlobalMaxPool [peak] │
│ R3 = StdPool [contrast]│
│ z = Trunk(Concat[R1;R2;R3]) │
└───────────────┬────────────────┘
│
┌──────────┴──────────┐
▼ ▼
PRIMARY HEAD SEVERITY HEAD
Linear(128→3) Linear(128→1)
Softmax Sigmoid
[inference] [train-time only]


### Loss Function

$$\mathcal{L} = \mathcal{L}_{cls} + 0.10 \cdot \mathcal{L}_{sev} + 0.05 \cdot \mathcal{L}_{div}$$

| Term | Description |
|------|-------------|
| $\mathcal{L}_{cls}$ | Weighted CrossEntropy, $w_i = 1/N_{c_i}$ |
| $\mathcal{L}_{sev}$ | MSE between predicted and pseudo severity |
| $\mathcal{L}_{div}$ | Stream weight entropy — prevents stream collapse |

### Training Configuration

| Hyperparameter | Value |
|----------------|-------|
| Optimizer | AdamW |
| Learning Rate | 1e-4 |
| Weight Decay | 1e-4 |
| LR Scheduler | CosineAnnealingLR (T_max=50, η_min=1e-6) |
| Batch Size | 16 |
| Max Epochs | 50 |
| Early Stopping | patience=7 on val loss |
| Precision | Mixed (AMP FP16) |
| Input Size | 224×224 |
| GPU | Single NVIDIA T4 |

---
## 📊 Results

### NailDCMNet — Classification Report

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| Healthy | 0.9839 | 0.9839 | 0.9839 | 62 |
| Onychomycosis | 0.9589 | 0.9589 | 0.9589 | 146 |
| Psoriasis | 0.9341 | 0.9341 | 0.9341 | 91 |
| **Macro Avg** | **0.9589** | **0.9589** | **0.9589** | 299 |
| Weighted Avg | 0.9565 | 0.9565 | 0.9565 | 299 |

> **Overall Accuracy: 95.65%**

---

## 📈 Baseline Comparison

| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| MobileNetV3-Large | 0.9097 | 0.9134 | 0.9111 | 0.9122 |
| ConvNeXtV2-Tiny | 0.9231 | 0.9365 | 0.9178 | 0.9249 |
| ResNet-50 | 0.9365 | 0.9449 | 0.9383 | 0.9413 |
| Swin-Tiny | 0.9365 | 0.9423 | 0.9335 | 0.9377 |
| DenseNet-121 | 0.9398 | 0.9427 | 0.9372 | 0.9399 |
| **NailDCMNet (Ours)** | **0.9565** | **0.9589** | **0.9589** | **0.9589** |

### Improvements over best baseline (DenseNet-121)

| Metric | DenseNet-121 | NailDCMNet | Gain |
|--------|-------------|------------|------|
| Accuracy | 0.9398 | 0.9565 | **+1.67%** |
| Precision | 0.9427 | 0.9589 | **+1.62%** |
| Recall | 0.9372 | 0.9589 | **+2.17%** |
| F1-Score | 0.9399 | 0.9589 | **+1.90%** |

---

## ⚙️ Requirements

```bash
pip install torch torchvision
pip install timm
pip install scikit-learn
pip install matplotlib seaborn pandas
pip install grad-cam
```

| Package | Tested Version |
|---------|----------------|
| Python | 3.10+ |
| PyTorch | 2.0+ |
| timm | 0.9+ |
| scikit-learn | 1.3+ |
| grad-cam | latest |

---

## 🚀 Usage

### 1. Clone the repository

```bash
git clone https://github.com/<your-username>/NailDCMNet.git
cd NailDCMNet
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run on Kaggle

1. Add dataset from Kaggle
2. Upload notebook code
3. Set accelerator: **GPU T4 x1** (single GPU only)
4. Add this at the top to prevent CUDA errors:

```python
import os
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"]  = "0"
```

### 4. Inference

```python
import torch
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms as transforms

CLASS_NAMES = ["healthy", "onychomycosis", "psoriasis"]

# Load model
model = NailDCMNet(num_classes=3)
model.load_state_dict(
    torch.load("NailDCMNet_best.pt", map_location="cpu"))
model.eval()

# Preprocess
tf = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]),
])

# Predict
img = tf(Image.open("nail_image.jpg")
         .convert("RGB")).unsqueeze(0)

with torch.no_grad():
    logits, severity, stream_weights = model(img)
    pred = logits.argmax(1).item()
    conf = F.softmax(logits, dim=1)[0, pred].item()

print(f"Prediction     : {CLASS_NAMES[pred]}")
print(f"Confidence     : {conf:.4f}")
print(f"Severity score : {severity.item():.4f}")
print(f"Stream weights : color={stream_weights[0,0]:.3f} "
      f"texture={stream_weights[0,1]:.3f} "
      f"morph={stream_weights[0,2]:.3f}")
```


## 🔍 Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| **3 parallel streams** | Onychomycosis vs Psoriasis differ simultaneously in color, texture, and morphology — single backbone cannot capture all three adequately |
| **Cross-stream attention** | Texture stream attends to color to locate lesion positions; morphology stream uses texture-enhanced context for boundary detection |
| **Adaptive stream weighting** | Different diseases emphasize different diagnostic cues — model learns this dynamically per image rather than using fixed fusion |
| **Diversity loss on weights** | Entropy regularization prevents the model from collapsing to one stream and ignoring the other two |
| **StdPool aggregation** | Local contrast variance is a clinically meaningful signal for lesion prominence and disease severity |
| **Severity auxiliary head** | Regularizes shared trunk to encode disease progression information without requiring explicit severity annotations |
| **Single GPU** | EfficientNet and Swin SE blocks cause CUDA misaligned address errors under DataParallel on T4×2 — single GPU is stable |

---

## ✨ Novelty

1. **Nail Region Focus Module** — First learnable soft-mask gate for nail-plate isolation; no existing nail disease paper applies a learnable background suppression gate
2. **Triple-stream decomposed encoder** — Explicit color / texture / morphology decomposition aligned with clinical diagnostic reasoning
3. **Cross-diagnostic attention chain** — Sequential Color→Texture→Morphology cross-attention; information flows in the same order as clinical inspection
4. **Adaptive per-image stream weighting** — Model dynamically learns which diagnostic dimension matters most per input image
5. **Stream weight explainability** — Novel XAI output: bar chart of stream importances per prediction, directly interpretable by dermatologists

---
---

## 📬 Contact

**Abu Bakar Rakib**
AI/ML Researcher | Dhaka, Bangladesh
🔗 GitHub: [@your-username](https://github.com/your-username)

---

<p align="center">
  Made with ❤️ for advancing AI in dermatology
</p>
