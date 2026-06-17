# Facial Expression Recognition — Emotion Classification

Deep-learning project classifying facial expressions into seven emotions
(angry, disgust, fear, happy, neutral, sad, surprise) on the FER-2013 dataset.

**Course:** Computer Vision, HSLU · **Date:** June 2026
**Authors:** Bianca Bernasconi, Marion Stutz

## Overview
A fine-tuned ResNet18 CNN (ImageNet transfer learning, PyTorch) is benchmarked
against a CLIP (ViT-B/32) zero-shot baseline. The project covers the full pipeline:
data cleaning, augmentation, training, evaluation, interpretability, and an
out-of-distribution generalisation test.

## Approach
- **Data:** FER-2013 (~36k grayscale 48×48 images, 7 classes). Class-imbalance
  analysis and an automated watermark-detection step (pixel-based detector).
- **Preprocessing:** grayscale → 3-channel RGB, resize to 224×224, augmentation
  (random flip, ±10° rotation, colour jitter).
- **Model:** ResNet18 with frozen backbone; layer4 + a new classification head
  (Dropout 0.5) fine-tuned. Two-phase training (10 + 10 epochs), Adam
  (lr 1e-4, weight decay 1e-4), ReduceLROnPlateau scheduler, batch size 64.
- **Baseline:** CLIP zero-shot classification via text prompts ("a photo of a
  person with a happy expression").

## Results
- Fine-tuned ResNet18: ~61% validation accuracy.
- CLIP zero-shot: ~40% accuracy on the full test set (3,290 images).
- Analysis via confusion matrices, per-class precision/recall/F1, feature-map
  visualisation (layer1 vs layer4), and a domain-shift test on team photos
  (OpenCV Haar-Cascade face detection).

## Limitations & Next Steps
Weighted loss / oversampling for rare classes, face alignment (MTCNN),
a larger backbone (ResNet50/EfficientNet), and full fine-tuning could close
the gap to SOTA (~75%).

## Tech Stack
Python, PyTorch, torchvision, CLIP, OpenCV, NumPy, Matplotlib.

## Disclosure of GenAI
Generative AI (Claude) was used in a supportive role for debugging, code
refinement, and wording. All substantial decisions — model architecture,
transfer-learning strategy, hyperparameters, and interpretation of results —
were made independently by the authors.
