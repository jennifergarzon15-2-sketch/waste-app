import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Configuración de la página web
st.title("♻️ Clasificador de Residuos (Waste V2)")
st.write("Sube una imagen para clasificar el tipo de residuo.")

# Lista de las 30 clases del modelo
CLASES = [
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

# Cargar el modelo usando la caché de Streamlit para que sea ultrarrápido
@st.cache_resource
def cargar_modelo():
    return tf.keras.models.load_model("hf://Darshan764/waste-classification-v2")

with st.spinner("Cargando el modelo..."):
    model = cargar_modelo()

# Componente para arrastrar y subir archivos
archivo_subido = st.file_uploader("Selecciona una foto de residuo...", type=["jpg", "jpeg", "png", "webp"])

if archivo_subido is not None:
    # Mostrar la imagen cargada
    image = Image.open(archivo_subido).convert("RGB")
    st.image(image, caption="Imagen cargada", use_container_width=True)

    # Preprocesamiento (224x224)
    img_resized = image.resize((224, 224))
    arr = np.expand_dims(np.array(img_resized), axis=0)

    # Predicción
    with st.spinner("Clasificando..."):
        pred = model.predict(arr, verbose=0)[0]
        top_indices = np.argsort(pred)[::-1][:3]

    st.subheader("Resultados:")
    for idx in top_indices:
        clase = CLASES[idx]
        confianza = pred[idx] * 100
        st.write(f"**{clase}**: {confianza:.2f}%")
        st.progress(float(pred[idx]))