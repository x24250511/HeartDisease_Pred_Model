# ❤️ Heart Disease Prediction System (Multimodal ML)

## Project Overview

This project is a multimodal heart disease prediction system built using
machine learning and deep learning.

The system analyzes ECG images, structured diagnostic data (PTB-XL), and
clinical tabular data to provide a risk-aware heart condition
prediction.

Instead of relying on a single model, this project combines:

-   ECG Image Classification (CNN)
-   PTB-XL Diagnostic Model (XGBoost)
-   Clinical Tabular Risk Model (Random Forest)
-   Fusion Logic for final decision

This approach makes the system more reliable and closer to real-world
clinical workflows.

------------------------------------------------------------------------

## System Architecture

### 1️⃣ ECG Image Model (CNN)

-   Model: ResNet18 (PyTorch)
-   Task: Classify ECG as **Normal** or **Abnormal**
-   Dataset Size: \~179,000 ECG images
-   Test Accuracy: \~97%

This model acts as the first screening layer.

If ECG is normal → system returns NORMAL\
If ECG is abnormal → further analysis continues

------------------------------------------------------------------------

### 2️⃣ MI Detector (XGBoost)

Framework: XGBoost
Input: PTB-XL dataset tabular metadata
Task: Binary MI detection
Note: A critical data leakage bug was identified and fixed — an mi_likelihood feature was essentially reading the label directly, producing artificially inflated accuracy. This was audited and removed.

------------------------------------------------------------------------

### 3️⃣ Clinical Tabular Risk Model

-   Model: Random Forest
-   Dataset Size: \~920 patient records
-   Accuracy: \~85%

Uses features such as: - Age - Sex - Cholesterol - Blood Pressure -
Exercise-induced Angina

This model estimates overall patient risk.

------------------------------------------------------------------------

### 4️⃣ Fusion Engine

All predictions are combined using rule-based logic.

Possible outputs: - NORMAL - ABNORMAL_MONITOR - POSSIBLE_MI -
HIGH_RISK_MI

The fusion system follows a clinical-style workflow: 1. Detect abnormal
ECG 2. Check for MI probability 3. Evaluate patient risk 4. Generate
final structured decision

------------------------------------------------------------------------

## Folder Structure

Heart_disease_CML/ │ ├── app/ │ ├── models/ │ ├── fusion/ │ ├── utils/ │
└── main.py │ ├── Jupyter_NB/ ├── data/ ├──
requirements.txt └── README.md

------------------------------------------------------------------------

## How to Run

### 1️⃣ Create Virtual Environment

python -m venv venv source venv/bin/activate

### 2️⃣ Install Dependencies

pip install -r requirements.txt

### 3️⃣ Run API

uvicorn app.main:app --reload

Open browser: http://127.0.0.1:8000/docs

------------------------------------------------------------------------

## Datasets Used

-   ECG Image Dataset (\~179k images)
-   PTB-XL Dataset (PhysioNet)
-   Clinical Heart Disease Dataset

Note: PTB-XL dataset is provided by PhysioNet under its original license.
This repository includes metadata files only.
Users must comply with the original dataset licensing terms.

Datasets
The clinical risk model uses a combined dataset (~5,000+ rows) assembled from multiple public sources to improve generalizability:
DatasetSourceContributionUCI Heart DiseaseCleveland ClinicCore clinical features (303 samples)Fedesoriano Heart Failure PredictionKaggleAdditional clinical recordsCardiovascular DiseaseSulianova, KaggleSubsampled for class balanceStatlog HeartUCI RepositorySupplementary clinical dataPTB-XLPhysioNetECG metadata for MI detection
Combining these datasets improved Random Forest 5-fold cross-validation accuracy from ~77% (original UCI only) to ~87.5%.
------------------------------------------------------------------------

## Key Achievements

-   High-performance ECG classifier (\~97% accuracy)
-   Balanced MI detection using XGBoost
-   Risk-aware tabular model (\~85% accuracy)
-   Multimodal fusion architecture
-   API ready for cloud deployment

------------------------------------------------------------------------

## Future Work

-   Incorporate raw ECG waveform modeling
-   Improve multi-class abnormality detection
-   Deploy system to AWS
-   Build frontend dashboard

------------------------------------------------------------------------

## Academic Context

MSc Cloud Machine Learning Project\
2026
