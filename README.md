# Human Posture Classification Framework

> An experimental framework for classifying human postures from silhouette images using a Custom CNN and MobileNetV2 with Grad-CAM explainability.

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://www.tensorflow.org/)
[![Kaggle](https://img.shields.io/badge/Kaggle-Dataset-cyan.svg)](https://www.kaggle.com/)

---

## Quick Start

1. Clone the repo and enter the folder:

```bash
git clone <your-repo-url>
cd Human-Posture-Classification
```

2. Install dependencies (create `requirements.txt` first if not present):

```bash
pip install -r requirements.txt
```

3. Download the dataset from Kaggle and place it under `data/silhouettes/` (folder structure: `data/silhouettes/bending`, `.../lying`, `.../sitting`, `.../standing`). Example using Kaggle CLI:

```bash
kaggle datasets download -d deepshah16/silhouettes-of-human-posture --unzip -p data/
```

Alternatively set an environment variable to the dataset root:

```bash
export DATASET_PATH=/full/path/to/silhouettes
```

4. Open the notebook and run cells (the notebook will auto-detect a Kaggle dataset or use the `DATASET_PATH` environment variable):

```bash
jupyter notebook human-posture-notebook.ipynb
```

For a faster recruiter view, open the summary notebook:

```bash
jupyter notebook human-posture-notebook-summary.ipynb
```

If you prefer a quick demo (inference-only), see `src/infer.py`.


## Project Overview

This project investigates the effectiveness of Convolutional Neural Networks (CNNs) for human posture classification (**Bending, Lying, Sitting, Standing**) using silhouette images. 

To identify the most effective approach, two architectures were developed and compared:
1. **Custom CNN:** Designed, optimized, and trained entirely from scratch.
2. **MobileNetV2:** Utilized transfer learning leveraging ImageNet pre-trained weights, followed by fine-tuning.

Beyond standard performance metrics, the project integrates **Grad-CAM (Gradient-weighted Class Activation Mapping)** to visually explain model predictions, ensuring the networks focus on anatomical structures rather than background noise.

### Latent Applications
* **Workplace Ergonomics:** Monitoring posture to prevent musculoskeletal disorders.
* **Smart Healthcare:** Fall risk assessment and aged care monitoring.
* **Human Activity Analysis:** Surveillance and behavioral analytics.

> **Note:** This system is intended strictly as a decision support tool and should not replace professional human judgment or medical expertise.
---

## Results Snapshot

* **Best model:** Custom CNN
* **Test accuracy:** 92.78%
* **Macro F1:** 0.9284
* **Macro AUC:** 0.9938
* **Inference speed:** ~3.5 ms/image

## Dataset Profile

| Property | Details |
| :--- | :--- |
| **Dataset Name** | Silhouettes of Human Posture |
| **Source** | [Kaggle Dataset Link](https://www.kaggle.com/datasets/deepshah16/silhouettes-of-human-posture) |
| **Total Images** | 4,800 images |
| **Target Classes** | 4 (Bending, Lying, Sitting, Standing) |
| **Class Distribution**| Perfectly balanced (1,200 images per class) |
| **Image Resolution** | 128 × 128 × 3 (RGB) |

---

## Model Architectures

### 1. Custom CNN (From Scratch)
A consecutive convolutional network optimized for silhouette pattern recognition:

`Input (128×128×3) ➔ 4 × [Conv2D ➔ BatchNorm ➔ ReLU ➔ MaxPool] ➔ Global Average Pooling ➔ Dense (256) ➔ Dropout ➔ Softmax Output`

* **Feature Extraction:** Four active convolutional blocks scaling filters from 32 up to 256.
* **Classification Head:** Global Average Pooling (GAP) to reduce dimensionality, a dense layer with 256 units, dropout for regularization, and a 4-way softmax layer.

### 2. MobileNetV2 (Transfer Learning)
* **Base Network:** Pre trained on ImageNet (weights frozen initially for feature extraction , then selectively unfrozen for fine tuning).
* **Classification Head:** Identical to the Custom CNN head to ensure a fair evaluation baseline.
---

## Repository Structure

```text
Human-Posture-Classification/
├── human-posture-notebook.ipynb
├── PROJECT.md
├── README.md
├── requirements.txt
├── dataset_link.md
├── output/
│   ├── figures/
│   └── models/  # model files are excluded from Git or stored separately
└── src/
    └── infer.py
```

## Evaluation & Explainability

The models are evaluated and compared using a comprehensive suite of performance criteria:
* **Classification Accuracy:** Overall correctness across posture categories.
* **Precision, Recall, & F1-Score:** Per-class metrics to ensure reliable performance on individual postures.
* **Confusion Matrices:** Identify exact classification overlaps (for example, misclassifying Sitting as Bending).
* **ROC Curves & ROC AUC Scores:** Analyze classifier performance across decision thresholds.
* **Inference Time:** Measured to assess feasibility for real-world or edge deployment.

# Grad-CAM Interpretability
To prevent the models from operating as uninterpretable "black boxes," Gradient-weighted Class Activation Mapping (Grad-CAM) visualizes the optic focus of the last convolutional layer. 

The heatmaps are interpreted as follows:
* Red Regions ➔ Highest model attention (The core features driving the prediction)
* Yellow/Green Regions ➔ Moderate structural focus
* Blue Regions ➔ Ignored background or extraneous spatial areas

Grad-CAM analysis was deliberately applied to both correctly classified and misclassified samples. This allows for a deeper diagnosis of structural edge cases and uncovers exactly why a model failed on a specific posture.

---

# Core Research Questions Addressed

1.Classification Efficacy:How accurately can deep learning CNN architectures distinguish human postures solely from high-contrast silhouette images?
2.Architecture Comparison:Does a highly parameterized, pre-trained model (MobileNetV2) yield a statistically significant performance boost over a lightweight, custom-built CNN on simplified silhouette geometry?
3.Augmentation Utility:How effective are spatial data augmentations (rotations, shearing, scaling) in preventing overfitting when dealing with stark, binary-like boundaries?
4.Attention Verification:Do the Grad-CAM heatmaps verify that the network's attention aligns logically with key anatomical landmarks (e.g., hips, knees, spine alignment)?
5.Failure Mode Analysis:What are the recurring structural failure modes, and what causes the models to confuse highly similar geometries?

---

# Limitations

Silhouette Dependency: The framework relies entirely on clean, pre-segmented silhouettes. Real-world imagery with cluttered backgrounds will severely degrade performance without a preprocessing foreground-extraction pipeline.
Fixed Class Scope:The system is strictly limited to four posture classes (*Bending, Lying, Sitting, Standing*) and cannot classify transitional or hybrid movements.
Single-Subject Constraint:Multi-person scenes are currently unsupported; the presence of multiple silhouettes within a single frame will cause classification errors.
Non-Causal Explanation:Grad-CAM maps indicate statistical feature correlations and spatial attention highlights, but they do not equal strict causal proof of reasoning.

---

# Technologies Used

Core Frameworks:Python, TensorFlow, Keras
Data Engineering & Analytics:NumPy, Pandas, Scikit-learn
Computer Vision & Visualization:OpenCV, Matplotlib, Seaborn
Development Environment:Kaggle Notebooks 
