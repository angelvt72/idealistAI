import streamlit as st
from PIL import Image
from prediction_pipeline_nube import prediction_process, process_image

st.title("Clasificación de Imágenes con Transfer Learning")

uploaded_file = st.file_uploader(
    "Elige una imagen", type=["jpg", "jpeg", "png", "bmp", "tif", "tiff", "webp"]
)

if uploaded_file is not None:
    # Cargar imagen en memoria y mostrarla
    image = Image.open(uploaded_file)
    st.image(image, caption="Imagen cargada", use_column_width=True)

    # Procesar la imagen
    image_tensor = process_image(image)

    if image_tensor is not None:
        # Hacer predicción
        results = prediction_process(image_tensor)

        # Mostrar resultados
        st.write("### Predicciones:")
        for class_name, prob in results.items():
            st.write(f"**{class_name}**: {prob:.4f}")
    else:
        st.error("No se pudo procesar la imagen. Intenta con otro archivo.")
