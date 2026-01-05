# 🏥 Federated Learning for Medical Imaging (Pneumonia Detection)

A privacy-preserving AI system for detecting Pneumonia from Chest X-Rays using Federated Learning.  
This project simulates a Non-IID federated environment where 5 hospitals collaboratively train a global model without sharing sensitive patient data.

A Streamlit-based dashboard is provided to compare predictions from the Centralized model and Federated (hospital-level) models.

---

## 📂 Project Structure (IMPORTANT)

```text
📁 federated-medical-app/
│
├── 📓 Federated_Learning_Notebook.ipynb   # Run first to train models
├── 🐍 app.py                              # Streamlit web interface
├── 🐍 model.py                            # MedicalCNN definition (required)
├── 🐍 utils.py                            # Image preprocessing utilities
├── 📄 requirements.txt
├── 📄 README.md
│
└── 📂 saved_models/                       # Auto-generated after training
    ├── central_model.pth
    ├── client_1_model.pth
    ├── client_2_model.pth
    ├── client_3_model.pth
    ├── client_4_model.pth
    └── client_5_model.pth
```

> ⚠️ **NOTE**  
> `model.py` must be present in the same directory as `app.py`,  
> as saved models depend on its class definition.

---

## 🚀 Technical Features

### 1️⃣ Data Preprocessing & Augmentation

**Normalization (ImageNet statistics)**  
- Mean: `[0.485, 0.456, 0.406]`  
- Std:  `[0.229, 0.224, 0.225]`

**Training Augmentations**
- Random Horizontal Flip  
- Random Rotation (±10°)

**Inference Pipeline**
- Resize to `224 × 224`
- Tensor conversion
- ImageNet normalization  
(Implemented in `utils.py`)

---

### 2️⃣ Federated Learning Architecture

- Algorithm: **Federated Averaging (FedAvg)**
- Clients: **5 hospitals**
- Data Split: **Non-IID class distribution**
- Local Training: **3 epochs per client per round**
- Central Server: **Aggregates client weights per round**
- Evaluation: **Centralized vs Federated performance comparison**

---

### 3️⃣ Model Architecture

- Backbone: **ResNet18 (ImageNet pretrained)**
- Transfer Learning:
  - Backbone frozen
  - Only final FC layer trained
- Task: **Binary classification (Normal vs Pneumonia)**

---

## 📊 Final Evaluation Metrics (Actual Results)

### Overall Accuracy
- **Centralized Model Accuracy:** `0.8958`
- **Federated Model Accuracy:** `0.8526`

---

### Confusion Matrix — Centralized Model (Accuracy ≈ 0.90)

| Actual \ Predicted | Normal (0) | Pneumonia (1) |
|-------------------|------------|---------------|
| Normal (0)        | 184        | 50            |
| Pneumonia (1)     | 15         | 375           |

---

### Confusion Matrix — Federated Model (Accuracy ≈ 0.85)

| Actual \ Predicted | Normal (0) | Pneumonia (1) |
|-------------------|------------|---------------|
| Normal (0)        | 147        | 87            |
| Pneumonia (1)     | 5          | 385           |

---

### Federated Learning Convergence

- Communication Rounds: **5**
- Observations:
  - Accuracy improves steadily until round 3
  - Slight degradation observed in later rounds due to Non-IID client drift
  - Federated accuracy remains close to centralized baseline

---

## 📈 Key Insights

- Federated learning achieves **~95% of centralized performance** without sharing raw medical data
- False negatives are low in both settings, which is critical for medical diagnosis
- Non-IID data causes mild performance drop but does not destabilize convergence
- Demonstrates strong privacy–utility trade-off

---

## 🛠️ Running the Project

1. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Train models**  
   Run all cells in `Federated_Learning_Notebook.ipynb`

3. **Launch dashboard**
   ```bash
   streamlit run app.py
   ```

Dashboard runs at: **https://fed-learn-chestxray.streamlit.app**

---
