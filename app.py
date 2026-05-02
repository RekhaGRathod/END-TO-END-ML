import streamlit as st
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
import sys
import os

# Add src to path to import utils
sys.path.append(os.path.abspath("src"))
from utils import get_device, build_model

# Setup page
st.set_page_config(page_title="Pneumonia Detection AI", page_icon="🫁")

st.title("🫁 Pediatric Chest X-ray Pneumonia Detection")
st.write("Upload a chest X-ray image to detect whether it is **Normal** or indicates **Pneumonia**.")

@st.cache_resource
def load_model():
    device = get_device()
    model = build_model(num_classes=2)
    model_path = "models/best_model.pt"
    
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        model = model.to(device)
        model.eval()
        return model, device
    return None, device

def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),
        transforms.ToTensor()
    ])
    # Convert to RGB if grayscale
    image = image.convert('RGB')
    tensor = transform(image).unsqueeze(0)
    return tensor

model, device = load_model()

if model is None:
    st.warning("⚠️ Model not found. Please run `src/train.py` to train the model first.")
else:
    uploaded_file = st.file_uploader("Choose an X-ray image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Display the uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded X-ray', use_column_width=True)

        st.write("Classifying...")

        # Process and predict
        input_tensor = preprocess_image(image)
        with torch.no_grad():
            output = model(input_tensor.to(device))
            output = F.softmax(output, dim=1)
            probability, predicted = torch.max(output.data, 1)

        # Mapping
        class_names = {0: 'NORMAL', 1: 'PNEUMONIA'}
        class_name = class_names[predicted.item()]
        conf = probability.item() * 100

        # Display results
        if class_name == "PNEUMONIA":
            st.error(f"Prediction: **{class_name}**")
        else:
            st.success(f"Prediction: **{class_name}**")
            
        st.info(f"Confidence: **{conf:.2f}%**")
