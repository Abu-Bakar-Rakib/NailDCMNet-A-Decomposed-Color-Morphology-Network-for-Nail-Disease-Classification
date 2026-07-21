# 💅 NailDCMNet
## A Decomposed Color-Morphology Network for Nail Disease Classification

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10+-3776ab?style=for-the-badge&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c?style=for-the-badge&logo=pytorch&logoColor=white)
![timm](https://img.shields.io/badge/timm-0.9+-6ba539?style=for-the-badge)
![License](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)
![Status](https://img.shields.io/badge/Status-Production-brightgreen?style=for-the-badge)

**Advancing AI-Driven Dermatology Through Multi-Stream Deep Learning**

</div>

---

<div align="center">

### 📊 Model Performance

| Metric | Value |
|--------|-------|
| **Accuracy** | 95.65% |
| **Macro F1-Score** | 0.9589 |
| **Macro Precision** | 0.9589 |

</div>

---

## 🎯 Overview

**NailDCMNet** introduces a cutting-edge **triple-stream parallel architecture** designed to tackle one of dermatology's most challenging diagnostic problems: distinguishing between clinically similar nail diseases.

### The Problem
Diseases like **Onychomycosis** (fungal infection) and **Psoriasis** present nearly identical symptoms:
- Nail thickening and discoloration
- Surface irregularities  
- Nail separation from the nail bed

Traditional single-backbone CNNs struggle because they cannot simultaneously process the **distinct diagnostic cues** that clinicians use:
- 🎨 **Color variations** (yellowish vs. reddish hues)
- 🧵 **Texture patterns** (scaling vs. pitting)
- 🔷 **Morphological changes** (structural deformation)

### Our Solution
NailDCMNet's **three parallel expert streams** work in concert:
1. **Color Stream** (ConvNeXtV2-Tiny) — Captures chromatic diagnostic signals
2. **Texture Stream** (MobileNetV3-Large) — Isolates surface patterns
3. **Morphology Stream** (Swin Transformer-Tiny) — Analyzes structural changes

These streams are linked via **sequential cross-attention**, mirroring how dermatologists examine nails: first observing color, then texture, then overall morphology.

---

## 📑 Table of Contents

- [Dataset](#-dataset)
- [Architecture](#-architecture)
- [Results](#-results)
- [Baseline Comparison](#-baseline-comparison)
- [Requirements](#-requirements)
- [Installation & Usage](#-installation--usage)
- [Project Structure](#-project-structure)
- [Design Decisions](#-key-design-decisions)
- [Novelty](#-novelty)
- [Contact](#-contact)

---

## 📦 Dataset

**Source:** [Kaggle Nail Disease Image Classification Dataset](https://www.kaggle.com/datasets/josephrasanjana/nail-disease-image-classification-dataset)

### Dataset Split

```
Total Images: 1,464
├── Training Set: 1,165 images
│   ├── Healthy:         248
│   ├── Onychomycosis:   575
│   └── Psoriasis:       342
└── Test Set: 299 images
    ├── Healthy:        62
    ├── Onychomycosis:  146
    └── Psoriasis:      91
```

### Directory Structure

```
nail_disease_dataset/
├── train/
│   ├── healthy/           [248 images]
│   ├── onychomycosis/     [575 images]
│   └── psoriasis/         [342 images]
└── test/
    ├── healthy/           [62 images]
    ├── onychomycosis/     [146 images]
    └── psoriasis/         [91 images]
```

---

## 🏗️ Architecture

### System Overview

```
INPUT IMAGE [B, 3, 224, 224]
│
▼
┌─────────────────────────────────────────┐
│ STAGE 1 — Nail Region Focus (NRF)      │
│ Learnable soft mask M                  │
│ X_f = X ⊙ (0.7·M + 0.3)               │
│ [Suppresses background interference]   │
└──────────────────┬──────────────────────┘
│
┌───────────┼───────────┐
▼ ▼ ▼
┌────────────┐ ┌────────────┐ ┌────────────┐
│ STREAM A   │ │ STREAM B   │ │ STREAM C   │
│   COLOR    │ │  TEXTURE   │ │ MORPHOLOGY │
│            │ │            │ │            │
│ConvNeXtV2  │ │MobileNetV3 │ │Swin-Tiny   │
│  -Tiny     │ │  -Large    │ │            │
│            │ │            │ │            │
│ SE Block   │ │ CBAM       │ │ CBAM       │
│ r=16       │ │            │ │            │
│[B,256,7,7] │ │[B,256,7,7] │ │[B,256,7,7] │
└─────┬──────┘ └─────┬──────┘ └─────┬──────┘
      │Fa           │Fb            │Fc
      └───────────┬──┘             │
                  ▼                │
         ┌───────────────────────┐ │
         │ Cross-Attention A→B   │ │
         │ Q=Fb, K=Fa, V=Fa     │ │
         │ Fb* = LN(Fb + Attn)  │ │
         └───────────┬───────────┘ │
                     │Fb*          │Fc
                     └────────┬────────┘
                              ▼
                 ┌─────────────────────────┐
                 │ Cross-Attention B*→C    │
                 │ Q=Fc, K=Fb*, V=Fb*    │
                 │ Fc* = LN(Fc + Attn)   │
                 └────────────┬────────────┘
                              │
                              ▼
                 ┌────────────────────────────────┐
                 │ Adaptive Stream Weighting      │
                 │ w = Softmax(Linear(GAP))      │
                 │ F_w = w₁·Fa + w₂·Fb* + w₃·Fc*│
                 └───────────┬────────────────────┘
                             │
                             ▼
                 ┌────────────────────────────────┐
                 │ STAGE 4 — Dual Attention Gate  │
                 │ Channel Gate (SE-style)        │
                 │ Spatial Gate (CBAM-style)      │
                 │ F_g = gate_proj([F_ch;F_sp])  │
                 └───────────┬────────────────────┘
                             │
                             ▼
                 ┌────────────────────────────────┐
                 │ STAGE 5 — Multi-Pool Agg.      │
                 │ GlobalAvgPool [global]         │
                 │ GlobalMaxPool [peak]           │
                 │ StdPool [contrast]             │
                 └───────────┬────────────────────┘
                             │
                 ┌───────────┴───────────┐
                 ▼                       ▼
            [3-class]              [severity]
           Classification        (train-time only)
          Softmax Output
```

### Loss Function

$$\mathcal{L}_{total} = \mathcal{L}_{cls} + 0.10 \cdot \mathcal{L}_{sev} + 0.05 \cdot \mathcal{L}_{div}$$

| Component | Formula | Purpose |
|-----------|---------|---------|
| **Classification Loss** | Weighted CrossEntropy | Balance class imbalance (w_i = 1/N_c_i) |
| **Severity Loss** | MSE | Regularize trunk with pseudo-labels |
| **Diversity Loss** | Entropy of stream weights | Prevent stream collapse |

### Training Hyperparameters

| Parameter | Value |
|-----------|-------|
| **Optimizer** | AdamW |
| **Learning Rate** | 1e⁻⁴ |
| **Weight Decay** | 1e⁻⁴ |
| **LR Scheduler** | CosineAnnealingLR (T_max=50, η_min=1e⁻⁶) |
| **Batch Size** | 16 |
| **Max Epochs** | 50 |
| **Early Stopping** | patience=7 on validation loss |
| **Precision** | Mixed Precision (AMP FP16) |
| **Input Resolution** | 224×224 |
| **Hardware** | Single NVIDIA T4 GPU |

---

## 📊 Results

### Classification Performance

| Class | Precision | Recall | F1-Score | Support |
|:-----:|:---------:|:------:|:--------:|:-------:|
| 🟢 Healthy | 0.9839 | 0.9839 | **0.9839** | 62 |
| 🟡 Onychomycosis | 0.9589 | 0.9589 | **0.9589** | 146 |
| 🔴 Psoriasis | 0.9341 | 0.9341 | **0.9341** | 91 |
| **Macro Average** | **0.9589** | **0.9589** | **0.9589** | 299 |
| Weighted Average | 0.9565 | 0.9565 | 0.9565 | 299 |

**Overall Accuracy: 95.65%** ✅

---

## 📈 Competitive Comparison

### Against State-of-the-Art Baselines

| Model | Accuracy | Precision | Recall | F1-Score |
|:------|:--------:|:---------:|:------:|:--------:|
| MobileNetV3-Large | 0.9097 | 0.9134 | 0.9111 | 0.9122 |
| ConvNeXtV2-Tiny | 0.9231 | 0.9365 | 0.9178 | 0.9249 |
| ResNet-50 | 0.9365 | 0.9449 | 0.9383 | 0.9413 |
| Swin Transformer-Tiny | 0.9365 | 0.9423 | 0.9335 | 0.9377 |
| DenseNet-121 | 0.9398 | 0.9427 | 0.9372 | 0.9399 |
| **NailDCMNet (Ours)** | **0.9565** ✨ | **0.9589** ✨ | **0.9589** ✨ | **0.9589** ✨ |

### Improvement Over Best Baseline (DenseNet-121)

| Metric | Baseline | NailDCMNet | **Improvement** |
|:-------|:--------:|:----------:|:---------------:|
| Accuracy | 0.9398 | 0.9565 | **+1.67%** ⬆️ |
| Precision | 0.9427 | 0.9589 | **+1.62%** ⬆️ |
| Recall | 0.9372 | 0.9589 | **+2.17%** ⬆️ |
| F1-Score | 0.9399 | 0.9589 | **+1.90%** ⬆️ |

---

## ⚙️ Requirements

### System Requirements

- **Python:** 3.10 or higher
- **GPU:** NVIDIA with CUDA support (tested on T4)
- **CUDA:** 11.8+ recommended

### Python Dependencies

```bash
torch>=2.0.0          # Deep learning framework
torchvision>=0.15.0   # Computer vision utilities
timm>=0.9.0           # PyTorch image models
scikit-learn>=1.3.0   # Metrics & evaluation
matplotlib>=3.7.0     # Visualization
seaborn>=0.12.0       # Advanced plotting
grad-cam>=1.4.8       # Explainability
```

---

## 🚀 Installation & Usage

### Step 1: Clone Repository

```bash
git clone https://github.com/Abu-Bakar-Rakib/NailDCMNet.git
cd NailDCMNet
```

### Step 2: Install Dependencies

```bash
pip install -r requirements.txt
```

Or install manually:

```bash
pip install torch torchvision timm scikit-learn matplotlib seaborn grad-cam
```

### Step 3: Prepare Dataset

Download from [Kaggle](https://www.kaggle.com/datasets/josephrasanjana/nail-disease-image-classification-dataset) and organize as shown in the Dataset section.

### Step 4: Running on Kaggle

1. **Add Dataset:** Import the Kaggle dataset in your notebook
2. **Set Accelerator:** GPU T4 x1 (single GPU only)
3. **Add CUDA configuration** at the top of your notebook:

```python
import os
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
```

4. **Run training notebook** provided in the repository

### Step 5: Inference

```python
import torch
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms as transforms

# Configuration
CLASS_NAMES = ["healthy", "onychomycosis", "psoriasis"]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Load pretrained model
model = NailDCMNet(num_classes=3)
model.load_state_dict(
    torch.load("NailDCMNet_best.pt", map_location=DEVICE)
)
model.to(DEVICE)
model.eval()

# Preprocessing pipeline
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
])

# Load and preprocess image
image = Image.open("path/to/nail_image.jpg").convert("RGB")
image_tensor = transform(image).unsqueeze(0).to(DEVICE)

# Model inference
with torch.no_grad():
    logits, severity, stream_weights = model(image_tensor)
    prediction_idx = logits.argmax(1).item()
    confidence = F.softmax(logits, dim=1)[0, prediction_idx].item()
    severity_score = severity.item()

# Display results
print(f"{'='*50}")
print(f"🩺 DIAGNOSIS REPORT")
print(f"{'='*50}")
print(f"Predicted Class  : {CLASS_NAMES[prediction_idx].upper()}")
print(f"Confidence       : {confidence*100:.2f}%")
print(f"Severity Score   : {severity_score:.4f}")
print(f"\n📊 Stream Contributions:")
print(f"  🎨 Color     : {stream_weights[0,0].item():.1%}")
print(f"  🧵 Texture   : {stream_weights[0,1].item():.1%}")
print(f"  🔷 Morphology: {stream_weights[0,2].item():.1%}")
print(f"{'='*50}")
```

---

## 🗂️ Project Structure

```
NailDCMNet/
├── README.md                      # This file
├── requirements.txt               # Python dependencies
├── NailDCMNet_Training.ipynb      # Full training pipeline
├── NailDCMNet_Inference.ipynb     # Inference demo
│
├── models/
│   └── nail_dcm_net.py            # Architecture definition
│
├── utils/
│   ├── data_loader.py             # Dataset utilities
│   ├── metrics.py                 # Evaluation metrics
│   ├── visualization.py           # Plotting & XAI
│   └── augmentation.py            # Data augmentation
│
├── checkpoints/
│   └── NailDCMNet_best.pt         # Best model weights
│
└── dataset/                       # Download from Kaggle
    ├── train/
    │   ├── healthy/
    │   ├── onychomycosis/
    │   └── psoriasis/
    └── test/
        ├── healthy/
        ├── onychomycosis/
        └── psoriasis/
```

---

## 🔑 Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| **3 Independent Streams** | Onychomycosis vs. Psoriasis differ simultaneously in color, texture, and morphology. Single-backbone architectures lack capacity to represent all diagnostic dimensions simultaneously. |
| **Sequential Cross-Attention** | Models clinical inspection order: observe color → analyze texture patterns → assess morphological changes. Information flows naturally through the diagnostic pipeline. |
| **Adaptive Stream Weighting** | Different diseases emphasize different diagnostic cues (e.g., color is critical for Onychomycosis; morphology for Psoriasis). Model learns per-image importance dynamically. |
| **Entropy Regularization on Weights** | Prevents model from collapsing to a single stream and ignoring others. Encourages genuine multi-stream fusion. |
| **StdPool Aggregation** | Captures local contrast variance—a clinically meaningful signal for lesion prominence and disease severity. |
| **Severity Auxiliary Head** | Regularizes shared trunk to encode disease progression without requiring explicit severity annotations during training. |
| **Single GPU Constraint** | EfficientNet and Swin Transformer SE blocks cause CUDA misaligned address errors under DataParallel on multi-GPU T4 setups. Single GPU is stable and sufficient. |

---

## ✨ Novelty & Innovation

### 1️⃣ **Nail Region Focus Module**
- First learnable **soft-mask gate** for nail-plate isolation
- No existing nail disease classifier applies learnable background suppression
- Improves robustness to lighting and image quality variations

### 2️⃣ **Triple-Stream Decomposed Encoder**
- Explicit **Color / Texture / Morphology** decomposition
- Aligns architecture with **clinical diagnostic reasoning**
- Each stream uses task-optimized backbone (ConvNeXtV2, MobileNetV3, Swin)

### 3️⃣ **Cross-Diagnostic Attention Chain**
- Sequential information flow: **Color → Texture → Morphology**
- First work to model the **inspection order** dermatologists use
- Non-symmetric cross-attention captures diagnostic dependencies

### 4️⃣ **Adaptive Per-Image Stream Weighting**
- Model learns which diagnostic dimension matters **most per input**
- Soft weighting allows graceful degradation if one stream fails
- Prevents spurious reliance on any single modality

### 5️⃣ **Stream Weight Explainability**
- Novel XAI output: **bar chart of stream importances** per prediction
- **Directly interpretable** by dermatologists
- Enables trust and clinical integration

---

## 📚 Citation

If you use NailDCMNet in your research, please cite:

```bibtex
@article{nailDCMNet2024,
  author    = {Abu Bakar Rakib},
  title     = {NailDCMNet: A Decomposed Color-Morphology Network for Nail Disease Classification},
  year      = {2024},
  publisher = {GitHub},
  url       = {https://github.com/Abu-Bakar-Rakib/NailDCMNet}
}
```

---

## 📬 Contact & Support

<div align="center">

**Abu Bakar Rakib**

*AI/ML Researcher | Dermatology AI | Deep Learning*

📍 Dhaka, Bangladesh  
🔗 [GitHub](https://github.com/Abu-Bakar-Rakib)  
📧 [Email](mailto:abubakarrakib.cse@gmail.com)

**Questions or Feedback?**  
Open an issue on [GitHub Issues](https://github.com/Abu-Bakar-Rakib/NailDCMNet/issues)

</div>

---

<div align="center">

### 🙏 Acknowledgments

- Dataset courtesy of [Joseph Rasanjana](https://www.kaggle.com/josephrasanjana) on Kaggle
- Built with [PyTorch](https://pytorch.org/), [timm](https://github.com/huggingface/pytorch-image-models), and ❤️

Made with passion for **advancing AI-driven dermatology** 🩺✨

</div>

---

<div align="center">

![visitors](https://visitor-badge.glitch.me/badge?page_id=Abu-Bakar-Rakib.NailDCMNet)
</div>