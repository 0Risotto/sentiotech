# sentiotech
# Speech Emotion Recognition

[![Python](https://img.shields.io/badge/Python-3.9+-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red)](https://pytorch.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green)](https://fastapi.tiangolo.com/)
[![License](https://img.shields.io/badge/License-Samsung%20Innovation%20Campus-orange)](LICENSE)

SentioTech is an end-to-end deep learning system for real-time Speech Emotion Recognition (SER), built as part of the Samsung Innovation Campus AI Capstone Project. The system classifies spoken audio into six emotional states: **Anger, Disgust, Fear, Happiness, Neutral, and Sadness**.

---

## Features

- **End-to-End Pipeline**: From raw audio to emotion prediction
- **Multi-Dataset Integration**: Unified CREMA-D, RAVDESS, TESS, and SAVEE datasets
- **Advanced Preprocessing**: Log-Mel spectrograms with delta & delta-delta features
- **Model Comparison**: Tested EfficientNet, ResNet, and PANN architectures
- **Web Interface**: Interactive UI for real-time emotion detection
- **Modular & Reproducible**: Fully configurable training and evaluation pipeline

---

## 📊 Results

Our best-performing model, **EfficientNet-B0**, achieved:

| Metric            | Score   |
|-------------------|---------|
| Test Accuracy     | 72.98%  |
| Weighted F1-Score | 0.7281  |

**Class-wise performance** improved significantly with MixUp augmentation and fine-tuning, especially for challenging emotions like Happiness and Fear.

---

##  Model Architecture

We adopted a **two-stage transfer learning** approach:

1. **Feature Extraction**: 3-channel log-Mel spectrograms (224×224)
2. **Classifier**: Fine-tuned EfficientNet-B0 with:
   - Dropout (p=0.5)
   - Label smoothing (α=0.11)
   - MixUp augmentation (α=0.2)
   - Cosine annealing LR scheduler

---

##  Tech Stack

- **Backend**: Python, PyTorch, TorchAudio, FastAPI
- **Frontend**: HTML, CSS, JavaScript
- **Data Processing**: LibROSA, NumPy, Pandas
- **Deployment**: Local server with interactive web interface

---

