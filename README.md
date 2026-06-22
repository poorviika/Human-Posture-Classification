# Human Posture Classification Framework

> A artful learning framework for classifying human postures from silhouette images using a Custom CNN and MobileNetV2 with Grad- CAM explainability.

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://www.tensorflow.org/)
[![Kaggle](https://img.shields.io/badge/Kaggle-Dataset-cyan.svg)](https://www.kaggle.com/)

---

## 📌 Project Overview

This project investigates the effectiveness of Convolutional Neural Networks (CNNs) for human posture classification (**Bending, Lying, Sitting, Standing**) using silhouette images. 

To identify the most vigorous approach, two clear architectures were developed and compared:
1. **Custom CNN:** Designed, optimized , and trained entirely from scratch.
2. **MobileNetV2:** Utilized transfer learning leveraging ImageNet pre-trained weights, followed by fine- tuning.

Beyond standard performance metrics, the project integrates **Grad-CAM (Gradient-weighted Class Activation Mapping)** to visually explain model predictions, ensuring the networks focus on anatomical structures rather than background noise.

### 🚀 Latent Applications
* **Workplace Ergonomics:** Monitoring posture to prevent musculoskeletal disorders.
* **Smart Healthcare:** Fall risk assessment and aged care monitoring.
* **Human Activity Analysis:** Surveillance and behavioral analytics.

> ⚠️ **Note:** This system is wilful strictly as a decision support tool and should not replace professional human judgment or medical expertise.
---

## 📊 Dataset Profile

| Property | Details |
| :--- | :--- |
| **Dataset Name** | Silhouettes of Human Posture |
| **Source** | [Kaggle Dataset Link](https://www.kaggle.com/datasets/deepshah16/silhouettes-of-human-posture) |
| **Total Images** | 4,800 images |
| **Target Classes** | 4 (Bending, Lying, Sitting, Standing) |
| **Class Distribution**| Perfectly balanced (1,200 images per class) |
| **Image Resolution** | 128 × 128 × 3 (RGB) |

---

## 🏗️ Model Architectures

### 1. Custom CNN (From Scratch)
A consecutive convolutional network optimized for silhouette pattern recognition:

`Input (128×128×3) ➔ 4 × [Conv2D ➔ BatchNorm ➔ ReLU ➔ MaxPool] ➔ Global Average Pooling ➔ Dense (256) ➔ Dropout ➔ Softmax Output`

* **Feature Extraction:** Four active convolutional blocks scaling filters from 32 up to 256 .
* **Classification Head:** Global ordinary Pooling (GAP) to reduce dimensionality, a dense layer with 256 units , dropout for regularization , and a 4 way Softmax layer.

### 2. MobileNetV2 (Transfer Learning)
* **Base Network:** Pre trained on ImageNet (weights frozen initially for feature extraction , then selectively unfrozen for fine tuning).
* **Classification Head:** Identical to the Custom CNN head to ensure a fair evaluation baseline.
---

## 📂 Repository Structure

```text
Human-Posture-Classification/
├── human-posture-notebook.ipynb   # Complete pipeline (Data prep to Grad-CAM)
├── outputs.zip                    # Exported plots, curves, and weights
├── README.md                      # Project documentation
└── dataset_link.md                # Quick link to source data

## 📈 Evaluation & Explainability

### Metrics Tracked 
The models are strictly evaluated and compared using a comprehensive suite of performance criteria:. 
* **Classification Accuracy:** Overall correctness of the model across all categories. * **Precision , Recall , & F1-Score:** Evaluated per class to ensure reliable performance on individual postures . 
* **Confusion Matrices:** Used to identify exact classification overlaps (e.g ., misclassifying *Sitting* as *Bending*). 
* **ROC Curves & ROC AUC Scores:** To analyze the constant optimistic vs. hollow optimistic trade offs at various thresholds . 
* **Inference Time:** Measured systematically to assess the feasibility of real world , real time edge deployment.

### 🔍 Grad-CAM Interpretability
To prevent the models from operating as uninterpretable "black boxes," **Gradient-weighted Class Activation Mapping (Grad-CAM)** visualizes the optic focus of the last convolutional layer. 

The heatmaps are interpreted as follows:
* **Red Regions** ➔ Highest model attention (The core features driving the prediction)
* **Yellow/Green Regions** ➔ Moderate structural focus
* **Blue Regions** ➔ Ignored background or extraneous spatial areas

Grad-CAM analysis was deliberately applied to both **correctly classified** and **misclassified** samples. This allows for a deeper diagnosis of structural edge cases and uncovers exactly why a model failed on a specific posture.

---

## 🔬 Core Research Questions Addressed

1. **Classification Efficacy:** How accurately can deep learning CNN architectures distinguish human postures solely from high-contrast silhouette images?
2. **Architecture Comparison:** Does a highly parameterized, pre-trained model (MobileNetV2) yield a statistically significant performance boost over a lightweight, custom-built CNN on simplified silhouette geometry?
3. **Augmentation Utility:** How effective are spatial data augmentations (rotations, shearing, scaling) in preventing overfitting when dealing with stark, binary-like boundaries?
4. **Attention Verification:** Do the Grad-CAM heatmaps verify that the network's attention aligns logically with key anatomical landmarks (e.g., hips, knees, spine alignment)?
5. **Failure Mode Analysis:** What are the recurring structural failure modes, and what causes the models to confuse highly similar geometries?

---

## 🛑 Limitations

* **Silhouette Dependency:** The framework relies entirely on clean, pre-segmented silhouettes. Real-world imagery with cluttered backgrounds will severely degrade performance without a preprocessing foreground-extraction pipeline.
* **Fixed Class Scope:** The system is strictly limited to four posture classes (*Bending, Lying, Sitting, Standing*) and cannot classify transitional or hybrid movements.
* **Single-Subject Constraint:** Multi-person scenes are currently unsupported; the presence of multiple silhouettes within a single frame will cause classification errors.
* **Non-Causal Explanation:** Grad-CAM maps indicate statistical feature correlations and spatial attention highlights, but they do not equal strict causal proof of reasoning.

---

## 🛠️ Technologies Used

* **Core Frameworks:** Python, TensorFlow, Keras
* **Data Engineering & Analytics:** NumPy, Pandas, Scikit-learn
* **Computer Vision & Visualization:** OpenCV, Matplotlib, Seaborn
* **Development Environment:** Kaggle Notebooks 