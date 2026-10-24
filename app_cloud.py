#version optimized for cloud deployment

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Page configuration
st.set_page_config(page_title="Waste Classifier", page_icon="♻️")

st.title("♻️ Waste Classifier (V2)")
st.write("Upload an image to classify the type of waste material.")

# List of the 30 classes supported by the model
CLASSES = [
    "aerosol_cans", "aluminum_food_cans", "aluminum_soda_cans", 
    "cardboard_boxes", "cardboard_packaging", "clothing", 
    "coffee_grounds", "disposable_plastic_cutlery", "eggshells", 
    "food_waste", "glass_beverage_bottles", "glass_cosmetic_containers", 
    "glass_food_jars", "magazines", "newspaper", 
    "office_paper", "paper_cups", "plastic_cup_lids", 
    "plastic_detergent_bottles", "plastic_food_containers", "plastic_shopping_bags", 
    "plastic_soda_bottles", "plastic_straws", "plastic_trash_bags", 
    "plastic_water_bottles", "shoes", "steel_food_cans", 
    "styrofoam_cups", "styrofoam_food_containers", "tea_bags"
]

# Cache model loading for fast inference in the cloud
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("hf://Darshan764/waste-classification-v2")

with st.spinner("Loading model from Hugging Face..."):
    model = load_model()

# Image upload widget
uploaded_file = st.file_uploader("Select or drop a waste image...", type=["jpg", "jpeg", "png", "webp"])

if uploaded_file is not None:
    # Process and display uploaded image
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # Preprocessing (224x224 input size for EfficientNet/MobileNet)
    img_resized = image.resize((224, 224))
    arr = np.expand_dims(np.array(img_resized), axis=0)

    # Model inference
    with st.spinner("Classifying..."):
        pred = model.predict(arr, verbose=0)[0]
        top_indices = np.argsort(pred)[::-1][:3]

    # Show top-3 prediction results
    st.subheader("Classification Results:")
    for idx in top_indices:
        label = CLASSES[idx].replace("_", " ").title()
        confidence = pred[idx] * 100
        st.write(f"**{label}**: {confidence:.2f}%")
        st.progress(float(pred[idx]))