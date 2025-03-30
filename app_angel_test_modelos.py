import torch
import torchvision.transforms as transforms
from PIL import Image
import os
import torchvision.models as models
import torch.nn as nn
import streamlit as st
from prediction_pipeline_modelos import prediction_process, process_image

# Configuración de la página de Streamlit
st.set_page_config(page_title="Clasificación de Imágenes", layout="centered")

# Encabezado de la aplicación
st.title("🔍 Clasificación de Imágenes con IA")
st.write("Sube una imagen y el modelo te dirá qué es con su probabilidad.")

# Selección del modelo
model_name = st.selectbox(
    "Elige el modelo", ["efficientnet_rank_0", "efficientnet_rank_7"]
)

# Subir imagen
uploaded_file = st.file_uploader(
    "Elige una imagen", type=["jpg", "jpeg", "png", "bmp", "tif", "tiff", "webp"]
)

# Si se subió una imagen, procesarla y hacer la predicción
if uploaded_file is not None:

    # Abrir la imagen y mostrarla
    image = Image.open(uploaded_file)
    st.image(image, caption="Imagen cargada", use_column_width=True)

    # Procesar la imagen
    image_tensor = process_image(image)

    # Verificar si la imagen se procesó correctamente
    if image_tensor is not None:
        
        # Obtener predicciones
        results = prediction_process(image_tensor, model_name)

        # Verificar si hubo un error al cargar el modelo
        if "error" in results:
            st.error(f"❌ Error al cargar el modelo: {results['error']}")
        else:
            # Extraer la mejor predicción
            best_class = max(results, key=results.get)
            best_prob = float(results[best_class]) * 100  # Convertir a porcentaje

            # Mostrar la mejor predicción
            st.subheader(
                f"📌 Predicción principal: **{best_class}** ({best_prob:.2f}%)"
            )

            # Mostrar todas las predicciones con barras de progreso
            st.write("### 📊 Probabilidades:")
            for class_name, prob in results.items():
                prob_float = float(prob)
                st.write(f"**{class_name}**: {prob_float * 100:.2f}%")
                st.progress(prob_float)
    else:
        st.error("❌ No se pudo procesar la imagen. Intenta con otro archivo.")
