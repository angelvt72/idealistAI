import streamlit as st
from PIL import Image
from prediction_pipeline_nube import prediction_process, process_image

# Configurar la página
st.set_page_config(page_title="Clasificación de Imágenes", layout="centered")

# Título principal
st.title("🔍 Clasificación de Imágenes con IA")
st.write("Sube una imagen y el modelo te dirá qué es con su probabilidad.")

# Subir imagen
uploaded_file = st.file_uploader(
    "Elige una imagen", type=["jpg", "jpeg", "png", "bmp", "tif", "tiff", "webp"]
)

if uploaded_file is not None:
    # Mostrar la imagen cargada
    image = Image.open(uploaded_file)
    st.image(image, caption="Imagen cargada", use_column_width=True)

    # Procesar la imagen
    image_tensor = process_image(image)

    if image_tensor is not None:
        # Obtener predicciones
        results = prediction_process(image_tensor)

        # Extraer la mejor predicción
        best_class = max(results, key=results.get)
        best_prob = results[best_class] * 100  # Convertir a porcentaje

        # Mostrar la mejor predicción
        st.subheader(f"📌 Predicción principal: **{best_class}** ({best_prob:.2f}%)")

        # Mostrar todas las predicciones con barras de progreso
        st.write("### 📊 Probabilidades:")
        for class_name, prob in results.items():
            st.write(f"**{class_name}**: {prob * 100:.2f}%")
            st.progress(prob)

    else:
        st.error("❌ No se pudo procesar la imagen. Intenta con otro archivo.")
