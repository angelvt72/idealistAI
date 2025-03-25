import streamlit as st
from PIL import Image
import os
import tempfile  # Para manejar archivos temporales

# Importar la función de predicción desde models_generator
from models_generator.PredictionProcess_nube import prediction_process

st.title("Clasificación de Imágenes con Transfer Learning")

# Streamlit interface
uploaded_file = st.file_uploader(
    "Elige una imagen", type=["jpg", "jpeg", "png", "bmp", "tif", "tiff", "webp"]
)

if uploaded_file is not None:
    # Guardar archivo cargado en un directorio temporal
    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        temp_file.write(uploaded_file.read())
        temp_file_path = temp_file.name

    # Realizar la predicción
    results = prediction_process(temp_file_path)

    # Mostrar resultados
    if "error" in results:
        st.error(results["error"])
    else:
        st.write("Predicciones:")
        for class_name, prob in results.items():
            st.write(f"{class_name}: {prob:.4f}")
