# Human Posture Classification Framework

> A CNN-based image classification system that identifies human body postures — **Bending, Lying, Sitting, and Standing** — from silhouette images. Built as part of an Applied Machine Learning course project.

---

## Project Overview

This project trains and compares two deep learning models on a silhouette posture dataset:

- **Custom CNN** — designed from scratch with 4 convolutional blocks
- **MobileNetV2** — pre-trained on ImageNet, fine-tuned for posture classification

The system is framed as a **decision-support tool** for applications like workplace ergonomics monitoring and elderly fall-risk detection — not as a replacement for human experts.

---

## Dataset

| Property | Details |
|---|---|
| Name | Silhouettes of Human Posture |
| Source | [Kaggle — deepshah16](https://www.kaggle.com/datasets/deepshah16/silhouettes-of-human-posture) |
| Total images | 4,800 |
| Classes | Bending, Lying, Sitting, Standing |
| Images per class | 1,200 (balanced) |
| Image type | Silhouette, RGB |
| Resolution used | 128 × 128 × 3 |

---

## Model Architectures

### Custom CNN
```
Input (128×128×3)
 → Block 1: Conv2D(32) × 2 + BN + ReLU + MaxPool  →  64×64
 → Block 2: Conv2D(64) × 2 + BN + ReLU + MaxPool  →  32×32
 → Block 3: Conv2D(128) × 2 + BN + ReLU + MaxPool →  16×16
 → Block 4: Conv2D(256) × 2 + BN + ReLU + MaxPool →   8×8
 → GlobalAveragePooling2D
 → Dense(256) + BN + ReLU + Dropout(0.5)
 → Softmax(4)
```

### MobileNetV2 (Transfer Learning)
- Pre-trained on ImageNet (1.28M images)
- Phase 1: Base frozen, head only — 10 epochs, LR = 1e-3
- Phase 2: Top 30 layers unfrozen, fine-tuned — 10 epochs, LR = 1e-5
- Same classification head as custom CNN

---

## Setup & Usage

### 1. Clone the repo
```bash
git clone https://github.com/YOUR_USERNAME/human-posture-classification.git
cd human-posture-classification
```

### 2. Install dependencies
```bash
pip install tensorflow scikit-learn matplotlib seaborn opencv-python pillow scipy pandas numpy
```

### 3. Download the dataset
Go to [this Kaggle link](https://www.kaggle.com/datasets/deepshah16/silhouettes-of-human-posture), download the dataset and place it so the folder structure looks like:
```
data/
├── Bending/
├── Lying/
├── Sitting/
└── Standing/
```
Then update `DATASET_PATH` in `section1_setup_imports.py` to point to your `data/` folder.

### 4. Run on Kaggle (recommended)
Upload each section file as notebook cells in order — Section 1 through Section 8. All figures and models are saved automatically to `/kaggle/working/outputs/`.

### 5. Run locally
```bash
python section1_setup_imports.py
python section2_dataset_eda.py
python section3_preprocessing.py
python section4_custom_cnn.py
python section5_transfer_learning.py
python section6_evaluation.py
python section7_gradcam.py
python section8_summary.py
```
> ⚠️ Sections must be run in order — each one depends on variables defined in previous sections. In practice, run them as cells inside a single Jupyter notebook.

---

## Results

> Actual values filled in after running the full pipeline.

| Model | Test Accuracy | Macro F1 | Macro AUC | Inference |
|---|---|---|---|---|
| Custom CNN | — | — | — | — ms/img |
| MobileNetV2 | — | — | — | — ms/img |

---

## Sample Outputs

| Figure | Description |
|---|---|
| `fig1_class_distribution.png` | Class balance — bar chart and pie chart |
| `fig2_sample_images.png` | 5 sample images per posture class |
| `fig4_augmentation_examples.png` | Original vs 9 augmented variants |
| `fig5_cnn_training_curves.png` | Custom CNN accuracy and loss curves |
| `fig6_mobilenetv2_curves.png` | MobileNetV2 two-phase training curves |
| `fig9_cm_custom_cnn.png` | Custom CNN confusion matrix |
| `fig11_cm_mobilenetv2.png` | MobileNetV2 confusion matrix |
| `fig19_gradcam_correct.png` | Grad-CAM correct predictions (all 4 classes) |
| `fig20_gradcam_errors.png` | Grad-CAM misclassifications / error analysis |

---

## Grad-CAM Explainability

Grad-CAM (Gradient-weighted Class Activation Mapping) is applied to both models to verify that they focus on meaningful body regions rather than background noise.

- 🔴 **Red regions** = highest model attention
- 🔵 **Blue regions** = low attention

Correct predictions show activation over the torso and limb angles. Misclassifications often show diffuse or background-focused attention — a clear signal that the model was confused.

---

## Evaluation Metrics Used

- Accuracy (overall)
- Precision, Recall, F1-score (macro and per-class)
- Confusion matrix (raw counts + row-normalised)
- ROC-AUC (one-vs-rest, macro average)
- Inference time (ms/image, GPU warm)

---

## Research Questions

| # | Question |
|---|---|
| RQ1 | How accurately can CNN models classify posture from silhouette images? |
| RQ2 | Does MobileNetV2 outperform a custom CNN in accuracy and efficiency? |
| RQ3 | How does augmentation affect per-class robustness? |
| RQ4 | Do Grad-CAM maps confirm the model focuses on real body features? |
| RQ5 | What are the main failure patterns and deployment limitations? |

---

## Limitations

- Trained on clean silhouettes — real-world photos will degrade performance
- 4-class problem on a small balanced dataset — results may not generalise to multi-person or occluded scenes
- Grad-CAM shows attention, not causation — interpretation requires care
- System should always have a human review stage before any deployment

---

## Tech Stack

![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?logo=tensorflow)
![Keras](https://img.shields.io/badge/Keras-API-red?logo=keras)
![Kaggle](https://img.shields.io/badge/Kaggle-Notebook-20BEFF?logo=kaggle)
![scikit-learn](https://img.shields.io/badge/scikit--learn-Evaluation-F7931E?logo=scikit-learn)

---



