# ECG Anomaly Detection

A deep learning system for ECG classification and cardiac anomaly detection using the PTB-XL dataset, with a multilingual web application for clinical use.

## Performance

| Metric | Value |
|--------|-------|
| **AUC Macro** | 91.98% |
| **Dataset** | PTB-XL (21,481 ECG recordings) |
| **Classes** | 5 cardiac conditions |

## Classification Classes

| Code | Description |
|------|-------------|
| NORM | Normal ECG |
| MI | Myocardial Infarction |
| STTC | ST/T Changes |
| CD | Conduction Disorders |
| HYP | Hypertrophy |

## Project Structure

```
ECG-Anomaly-Detection/
├── ECG_Analytics-main/           # Research & Model Development
│   ├── EDA.ipynb                 # Exploratory Data Analysis
│   ├── ECG_deep_learning_pipeline.ipynb    # Wide+Deep model training
│   ├── ECG_machine_learning_based_pipline.ipynb  # ML baselines
│   ├── ECG_NLP_Notebook.ipynb    # NLP text classification
│   └── NLP_EDA.ipynb             # Multilingual text analysis
│
└── ECG_Analytics_app-main/       # Web Application
    ├── client-ecg/
    │   ├── src/
    │   │   ├── App.js            # React frontend
    │   │   ├── main.py           # FastAPI server
    │   │   ├── app.py            # ECG prediction API
    │   │   ├── chatbot.py        # Multilingual chatbot
    │   │   └── model/            # Trained model weights
    │   └── public/
    ├── app run.bat               # Start backend
    └── app run 2.bat             # Start frontend
```

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

### requirements.txt

```
torch>=2.0.0
numpy>=1.24.0
pandas>=2.0.0
matplotlib>=3.7.0
seaborn>=0.12.0
neurokit2>=0.2.0
wfdb>=4.1.0
scipy>=1.10.0
scikit-learn>=1.2.0
transformers>=4.30.0
fastapi>=0.100.0
uvicorn>=0.22.0
python-dotenv>=1.0.0
groq>=0.4.0
axios
```

### GPU Support (Recommended)

The Wide+Deep model (11.5M parameters) benefits from GPU acceleration:

```bash
# CUDA 11.8
pip install torch --index-url https://download.pytorch.org/whl/cu118

# CUDA 12.1
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

## Quick Start

### Run Web Application

```bash
# Terminal 1: Start FastAPI backend
cd ECG_Analytics_app-main
app run.bat
# Or manually: uvicorn client-ecg.src.main:app --reload --port 8000

# Terminal 2: Start React frontend
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

## Model Architecture

### Wide+Deep Neural Network

```
┌─────────────────────────────────────────────────────────────┐
│                INPUT: ECG Signal (12 leads x 1000 samples)  │
└─────────────────────────────────────────────────────────────┘
                              │
        ┌─────────────────────┴─────────────────────┐
        │                                           │
        ▼                                           ▼
┌───────────────────────┐               ┌───────────────────────┐
│      DEEP PATH        │               │      WIDE PATH        │
├───────────────────────┤               ├───────────────────────┤
│ CNN (6 blocks):       │               │ 32 Handcrafted        │
│ Conv1D → BN → ReLU    │               │ Features:             │
│ → MaxPool             │               │ • Heart Rate          │
│                       │               │ • RR Intervals        │
│ Transformer:          │               │ • Lead Statistics     │
│ 8 layers, 8 heads     │               │   (mean, std, range)  │
│                       │               │                       │
│ Output: 64 features   │               │ Output: 32 features   │
└───────────────────────┘               └───────────────────────┘
        │                                           │
        └─────────────────────┬─────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                      FUSION LAYER                           │
│         Concat(64 + 32) → FC(128) → FC(64) → FC(5)          │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│          OUTPUT: 5 Probabilities [NORM, MI, STTC, CD, HYP]  │
└─────────────────────────────────────────────────────────────┘
```

### Model Parameters

- **Total Parameters**: 11.5M
- **CNN Channels**: 12 → 64 → 128 → 256 → 512
- **Transformer**: d_model=256, nhead=8, num_layers=8
- **Wide Features**: 32 (extracted via NeuroKit2)

## Pipeline Overview

| Component | Notebook | Description |
|-----------|----------|-------------|
| EDA | `EDA.ipynb` | Demographics, signal visualization, class distribution |
| Deep Learning | `ECG_deep_learning_pipeline.ipynb` | Wide+Deep model (main) |
| Machine Learning | `ECG_machine_learning_based_pipline.ipynb` | Random Forest, clustering baselines |
| NLP | `ECG_NLP_Notebook.ipynb` | XLM-RoBERTa for clinical text |
| Text Analysis | `NLP_EDA.ipynb` | Multilingual analysis, leakage detection |

## Exploratory Data Analysis

### EDA Components

| Analysis | Description |
|----------|-------------|
| **Demographics** | Age distribution, sex ratio |
| **Temporal** | Recording date patterns |
| **Quality** | Signal quality scores |
| **Dependencies** | Age/Sex vs SCP codes |
| **Correlations** | Features vs disease superclasses |
| **NLP** | Multilingual text (German/English), class imbalance |

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

## API Endpoints

```
POST /ecg/predict
  - Upload ECG files + patient info
  - Returns: class probabilities, diagnosis

POST /chatbot/chat
  - Send message with language preference
  - Returns: AI cardiology assistant response
```

## Usage Example

### Python Prediction

```python
import torch
from model import WideDeepModel

# Load model
model = WideDeepModel(num_wide_features=32, num_classes=5)
model.load_state_dict(torch.load('model_wide_deep_pure_FIXED.pth'))
model.eval()

# Predict
with torch.no_grad():
    logits = model(signal_tensor, wide_features_tensor)
    probs = torch.sigmoid(logits)
    
# Classes: ['NORM', 'MI', 'STTC', 'CD', 'HYP']
```

### API Request

```bash
curl -X POST "http://localhost:8000/ecg/predict" \
  -F "first_name=John" \
  -F "last_name=Doe" \
  -F "age=55" \
  -F "sex=M" \
  -F "dat_file=@ecg_record.dat" \
  -F "hea_file=@ecg_record.hea"
```

## Dataset

**PTB-XL**: Publicly available ECG dataset
- **Records**: 21,481 clinical 12-lead ECGs
- **Duration**: 10 seconds per recording
- **Sampling Rate**: 100 Hz (used) / 500 Hz (available)
- **Annotations**: SCP-ECG standard codes

## Tech Stack

| Layer | Technologies |
|-------|-------------|
| **ML/DL** | PyTorch, scikit-learn, NeuroKit2 |
| **NLP** | Transformers (XLM-RoBERTa), Groq |
| **Backend** | FastAPI, Uvicorn |
| **Frontend** | React.js, Axios |
| **Data** | WFDB, Pandas, NumPy |

## Environment Variables

Create `.env` file in `client-ecg/src/`:

```
GROQ_API_KEY=your_groq_api_key_here
```

## License

This project uses the PTB-XL dataset which is publicly available under the Open Data Commons Attribution License.

## Acknowledgments

- PTB-XL dataset by PhysioNet
- NeuroKit2 for ECG signal processing
- Groq for LLM API
