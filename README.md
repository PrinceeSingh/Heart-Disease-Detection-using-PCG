**Heart-Disease-Detection-using-PCG**
# Heart Disease Detection using Phonocardiogram (PCG) and IoT

## Project Overview

A low-cost, real-time cardiac screening system that detects heart disease from
phonocardiogram (PCG) recordings using a CNN-BiLSTM deep learning model with
Grad-CAM explainability, trained on two public datasets.

## Key Results

| Metric | Value |
|--------|-------|
| Test AUC | **0.9567** |
| Sensitivity (at optimal threshold) | **91%** |
| Specificity | **88%** |
| Cross-dataset AUC gap (PhysioNet vs CirCor) | 0.0065 |
| Min. usable SNR (AUC ≥ 0.85) | 15 dB |

## Datasets

- [PhysioNet/CinC Challenge 2016](https://physionet.org/content/challenge-2016/1.0.0/)
- [CirCor DigiScope 2022](https://physionet.org/content/circor-heart-sound/1.0.3/)

## Architecture
PCG (.wav) → Preprocessing → Log-mel Spectrogram → CNN encoder
→ BiLSTM → Attention pooling → Binary classifier

Grad-CAM explainability overlay

## Project Structure
pcg-project/
├── app/dashboard.py          # Streamlit demo dashboard
├── models/                   # Trained model weights + config
├── notebooks/                # Colab notebooks (Phases 1-11)
├── results/                  # Evaluation plots and tables
├── src/                      # Preprocessing and utility modules
└── requirements.txt

## Running the Dashboard

```bash
pip install -r requirements.txt
streamlit run app/dashboard.py
```

Upload any `.wav` PCG recording to get a prediction with Grad-CAM explanation.

## Pipeline Phases

| Phase | Description |
|-------|-------------|
| 1 | Data acquisition (PhysioNet 2016 + CirCor 2022) |
| 2 | Exploration & unified manifest |
| 3 | Preprocessing (bandpass, despike, wavelet denoise) |
| 4 | Segmentation into 4-second windows |
| 5 | Log-mel spectrogram feature extraction |
| 6 | Group-aware train/val/test split |
| 7 | CNN-BiLSTM model training |
| 8 | Test set evaluation |
| 9 | Grad-CAM explainability |
| 10 | Streamlit dashboard |
| 11 | Noise robustness stress test |

## Disclaimer

Research prototype only. Not a certified medical device.
Results must not be used for clinical diagnosis.