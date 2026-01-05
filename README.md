# 🏥 Federated Learning for Medical Imaging (Pneumonia Detection)

A privacy-preserving AI system for detecting Pneumonia from Chest X-Rays using Federated Learning.  
This project simulates a Non-IID federated environment where 5 hospitals collaboratively train a global model without sharing sensitive patient data.

The project also includes a Streamlit Web Dashboard to visualize and compare predictions from the Global Model and individual Local Hospital Models.

---

## 📂 Project Structure (IMPORTANT)

To run the application successfully, your directory structure must look EXACTLY like this to avoid import errors:

federated-medical-app/
│
├── 📓 Federated_Learning_Notebook.ipynb   # Run this first to train models
├── 🐍 app.py                              # Streamlit web interface
├── 🐍 model.py                            # Contains MedicalCNN class (required for loading models)
├── 🐍 utils.py                            # Image preprocessing & utility functions
├── 📄 requirements.txt                    # Python dependencies
├── 📄 README.md                           # Project documentation
│
└── 📂 saved_models/                       # Auto-generated after training
    ├── central_model.pth
    ├── client_1_model.pth
    ├── client_2_model.pth
    ├── client_3_model.pth
    ├── client_4_model.pth
    └── client_5_model.pth

⚠️ NOTE:
model.py MUST be present in the same directory as app.py.
The saved .pth files depend on the class definition inside model.py.

---

## 🚀 Technical Features

### 1️⃣ Data Preprocessing & Augmentation

Normalization  
All images are normalized using standard ImageNet statistics to match the ResNet backbone:

Mean: [0.485, 0.456, 0.406]  
Std:  [0.229, 0.224, 0.225]

Training Augmentations:
- Random Horizontal Flip
- Random Rotation (±10 degrees)

Inference Pipeline:
- Resize images to 224 × 224
- Convert to PyTorch tensors
- Apply ImageNet normalization  
(All preprocessing logic is implemented in utils.py)

---

### 2️⃣ Federated Learning Architecture

- Algorithm: Federated Averaging (FedAvg)
- Clients: 5 hospitals
- Data Distribution: Non-IID (uneven class distribution across hospitals)
- Local Training:
  - Each client trains locally for 3 epochs per communication round
- Communication Efficiency:
  - Tracks theoretical communication cost for FP32 weight transfer
  - Compares against quantized update sizes

---

### 3️⃣ Model Architecture

- Backbone: ResNet18 (pretrained on ImageNet)
- Transfer Learning:
  - Backbone layers are frozen to preserve learned features
  - Reduces training time and computational cost
- Classifier:
  - Final Fully Connected (FC) layer replaced
  - Binary classification: Normal vs Pneumonia

---

### 4️⃣ Evaluation Metrics

The system evaluates models using:
- Confusion Matrix (False Positives vs False Negatives)
- Precision, Recall, F1-Score (critical for medical diagnosis)
- Accuracy (comparison between centralized and federated models)

---

## 🛠️ Installation & Setup

### 1️⃣ Clone the Repository

git clone https://github.com/your-username/federated-medical-app.git  
cd federated-medical-app

---

### 2️⃣ Install Dependencies

pip install -r requirements.txt

---

### 3️⃣ Train the Models

Open Federated_Learning_Notebook.ipynb in Jupyter Notebook or VS Code and run all cells.

This will:
- Download the Chest X-Ray dataset
- Train a Centralized baseline model
- Simulate Federated Learning with 5 clients
- Save trained models into the saved_models/ directory

⚠️ IMPORTANT:
Ensure the final cell executes successfully.
The Streamlit app depends on the saved_models directory.

---

### 4️⃣ Run the Web Application

streamlit run app.py

The dashboard will launch at:
http://localhost:8501

---

## 🖥️ Using the Dashboard

- Upload a Chest X-Ray image (JPEG or PNG)
- The image is preprocessed using ImageNet normalization
- Inference is run across 6 models:
  - Centralized Model
  - Hospital Models 1–5
- Compare predictions to observe:
  - Local model bias due to Non-IID data
  - Robustness of Federated and Centralized models

---

## 📊 Results Comparison

Feature              | Centralized Learning | Federated Learning
-------------------- | -------------------- | ------------------
Data Privacy         | ❌ Low               | ✅ High
Accuracy             | ~96%                 | ~94%
Robustness           | High                 | Good (Non-IID resilient)
Bandwidth Usage      | High (raw data)      | Low (weights only)

---

## 📜 License

This project is open-source and released under the MIT License.
