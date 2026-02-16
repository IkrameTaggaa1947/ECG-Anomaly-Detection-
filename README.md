# ECG Anomaly Detection

## Table of Contents

- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Classification Classes](#classification-classes)
- [Architecture](#architecture)
- [Model Specifications](#model-specifications)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Pipeline Overview](#pipeline-overview)
- [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)
- [Web Application Features](#web-application-features)
- [API Endpoints](#api-endpoints)
- [Tech Stack](#tech-stack)
- [Environment Variables](#environment-variables)
- [License](#license)
- [Acknowledgments](#acknowledgments)

---

## Project Overview

This project implements a **Wide+Deep neural network** for ECG anomaly detection using the PTB-XL dataset (21,481 recordings). It classifies ECG signals into 5 cardiac conditions and includes a multilingual web application for clinical use, featuring real-time predictions and a Groq-powered cardiology chatbot.

**Key Features:**

- ECG signal preprocessing using NeuroKit2
- Multi-lead (12-lead) ECG analysis
- Wide+Deep architecture combining CNN, Transformer, and handcrafted features
- NLP analysis of clinical text reports (XLM-RoBERTa)
- Multilingual chatbot (English, French, Arabic)
- Full-stack application with React.js frontend and FastAPI backend

---

## Dataset

**PTB-XL**: Public 12-lead ECG dataset

| Property | Value |
|----------|-------|
| Records | 21,481 ECG recordings |
| Duration | 10 seconds per recording |
| Sampling Rate | 100 Hz (used), 500 Hz available |
| Annotations | SCP-ECG standard codes |

---

## Classification Classes

| Code | Description |
|------|-------------|
| NORM | Normal ECG |
| MI | Myocardial Infarction |
| STTC | ST/T Changes |
| CD | Conduction Disorders |
| HYP | Hypertrophy |

---

## Architecture

### Wide+Deep Neural Network

```
+-------------------------------------------------------------+
|             INPUT: ECG Signal (12 leads x 1000 samples)     |
+-------------------------------------------------------------+
                              |
        +---------------------+---------------------+
        |                                           |
        v                                           v
+-----------------------+               +-----------------------+
|      DEEP PATH        |               |      WIDE PATH        |
+-----------------------+               +-----------------------+
| CNN (6 blocks):       |               | 32 Handcrafted        |
| Conv1D -> BN -> ReLU  |               | Features:             |
| -> MaxPool            |               | - Heart Rate          |
| Transformer:          |               | - RR Intervals        |
| 8 layers, 8 heads     |               | - Lead Statistics     |
| Output: 64 features   |               |   (mean, std, range)  |
+-----------------------+               +-----------------------+
        |                                           |
        +---------------------+---------------------+
                              |
                              v
+-------------------------------------------------------------+
|                      FUSION LAYER                           |
|         Concat(64 + 32) -> FC(128) -> FC(64) -> FC(5)       |
+-------------------------------------------------------------+
                              |
                              v
+-------------------------------------------------------------+
|          OUTPUT: 5 Probabilities [NORM, MI, STTC, CD, HYP]  |
+-------------------------------------------------------------+
```

---

## Model Specifications

| Component | Details |
|-----------|---------|
| Total Parameters | 11.5M |
| CNN Channels | 12 -> 64 -> 128 -> 256 -> 512 |
| Transformer | d_model=256, nhead=8, num_layers=8 |
| Wide Features | 32 (extracted via NeuroKit2) |
| Loss | Cross-entropy with label smoothing |
| Optimizer | AdamW with cosine annealing |

---

## Installation

### Requirements

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt
```

### GPU Support (Recommended)

```bash
# CUDA 11.8
pip install torch --index-url https://download.pytorch.org/whl/cu118

# CUDA 12.1
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

---

## Quick Start

### Run Web Application

```bash
# Backend (FastAPI)
cd ECG_Analytics_app-main
app run.bat
# Or manually: uvicorn client-ecg.src.main:app --reload --port 8000

# Frontend (React)
app run 2.bat
# Or manually: cd client-ecg && npm start
```

Access the application at `http://localhost:3000`

### Run Training Pipeline

```bash
# Open Jupyter
jupyter notebook

# Execute notebooks in order:
# 1. EDA.ipynb - Data exploration
# 2. ECG_deep_learning_pipeline.ipynb - Model training
```

---

## Pipeline Overview

| Component | Notebook | Description |
|-----------|----------|-------------|
| EDA | `EDA.ipynb` | Demographics, signal visualization, class distribution |
| Deep Learning | `ECG_deep_learning_pipeline.ipynb` | Wide+Deep model (main) |
| Machine Learning | `ECG_machine_learning_based_pipline.ipynb` | Random Forest, clustering baselines |
| NLP | `ECG_NLP_Notebook.ipynb` | XLM-RoBERTa for clinical text |
| Text Analysis | `NLP_EDA.ipynb` | Multilingual analysis, leakage detection |

---

## Exploratory Data Analysis (EDA)

| Analysis | Description |
|----------|-------------|
| Demographics | Age distribution, sex ratio |
| Temporal | Recording date patterns |
| Quality | Signal quality scores |
| Dependencies | Age/Sex vs SCP codes |
| Correlations | Features vs disease superclasses |
| NLP | Multilingual text (German/English), class imbalance |

---

## Web Application Features

### ECG Prediction API

- Upload ECG files (WFDB format: `.dat` + `.hea`)
- Patient information form
- Real-time multi-class prediction
- Probability scores per condition

### Multilingual Chatbot

- **Languages**: English, French, Arabic
- **Categories**: Arrhythmias, Ischemia, Symptoms, Risk Factors
- **Powered by**: Groq LLM API

---

## API Endpoints

```
POST /ecg/predict
  - Upload ECG files + patient info
  - Returns: class probabilities, diagnosis

POST /chatbot/chat
  - Send message with language preference
  - Returns: AI cardiology assistant response
```

---

## Tech Stack

| Layer | Technologies |
|-------|-------------|
| ML/DL | PyTorch, scikit-learn, NeuroKit2 |
| NLP | Transformers (XLM-RoBERTa), Groq |
| Backend | FastAPI, Uvicorn |
| Frontend | React.js, Axios |
| Data | WFDB, Pandas, NumPy |

---

## Environment Variables

Create `.env` file in `client-ecg/src/`:

```
GROQ_API_KEY=your_groq_api_key_here
```

---

## License

This project uses the PTB-XL dataset, publicly available under the Open Data Commons Attribution License.

---

## Acknowledgments

- PTB-XL dataset by PhysioNet
- NeuroKit2 for ECG signal processing
- Groq for LLM API
