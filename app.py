import streamlit as st
import torch
import os
import torch.nn.functional as F
from model import MedicalCNN
from utils import process_image

# --- Configuration ---
MODEL_DIR = "saved_models"
CLASS_NAMES = ["NORMAL", "PNEUMONIA"]
DEVICE = torch.device("cpu") # Use CPU for inference on web app usually

# --- Load Models (Cached for speed) ---
@st.cache_resource
def load_models():
    models_dict = {}
    
    # 1. Load Central Model
    try:
        central = MedicalCNN()
        central.load_state_dict(torch.load(os.path.join(MODEL_DIR, "central_model.pth"), map_location=DEVICE))
        central.eval()
        models_dict["Centralized Model"] = central
    except FileNotFoundError:
        st.error("Central model not found in saved_models/")

    # 2. Load 5 Client Models
    for i in range(1, 6):
        try:
            client_model = MedicalCNN()
            path = os.path.join(MODEL_DIR, f"client_{i}_model.pth")
            client_model.load_state_dict(torch.load(path, map_location=DEVICE))
            client_model.eval()
            models_dict[f"Hospital {i} Model"] = client_model
        except FileNotFoundError:
            st.warning(f"Client {i} model not found.")
            
    return models_dict

# --- Main Interface ---
st.title("🏥 Federated Medical Diagnosis")
st.write("Upload a Chest X-Ray to see predictions from the **Centralized Model** vs. **5 Individual Hospital Models**.")

uploaded_file = st.file_uploader("Choose an X-Ray Image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Display Image
    st.image(uploaded_file, caption="Uploaded X-Ray", width=300)
    
    # Process Image
    img_tensor = process_image(uploaded_file).to(DEVICE)
    
    # Get Models
    models = load_models()
    
    if st.button("Analyze Image"):
        st.divider()
        st.subheader("Diagnostic Results")
        
        # Grid Layout for results
        cols = st.columns(3)
        
        for idx, (name, model) in enumerate(models.items()):
            # Prediction
            with torch.no_grad():
                output = model(img_tensor)
                probs = F.softmax(output, dim=1)
                conf, pred_idx = torch.max(probs, 1)
                
                prediction = CLASS_NAMES[pred_idx.item()]
                confidence = conf.item() * 100
                
            # Color logic
            color = "green" if prediction == "NORMAL" else "red"
            
            # Display in Grid
            with cols[idx % 3]:
                st.markdown(f"**{name}**")
                st.markdown(f":{color}[{prediction}]")
                st.progress(int(confidence))
                st.caption(f"Confidence: {confidence:.1f}%")
                st.write("---")