# Project Summary

## What this project does
- Builds and compares two posture classification models for silhouette images:
  - a Custom CNN trained from scratch
  - a MobileNetV2 transfer learning model
- Evaluates models using accuracy, precision, recall, F1-score, ROC AUC, confusion matrices, and inference time
- Uses Grad-CAM to visualize model attention and explain predictions

## My role
- Designed and implemented the data pipeline and tf.data preprocessing
- Built the custom CNN architecture and the transfer learning workflow
- Conducted model evaluation and explainability analysis
- Wrote documentation and created visual results saved under `output/figures`

## Key results
- The dataset is balanced across four posture classes
- The MobileNetV2 model was the strongest performer on validation and test data
- Grad-CAM helped confirm that models focus on posture silhouette regions rather than background noise

## Next improvements
- Add a reproducible `src/` training script for local execution
- Remove large `.keras` model files from Git history and store weights externally
- Add a concise summary notebook for recruiter review
