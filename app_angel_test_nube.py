import streamlit as st
from PIL import Image
from prediction_pipeline_nube import prediction_process, process_image

# Configuración inicial de la página
st.set_page_config(page_title="Clasificación de Imágenes", layout="centered")

# Estilos personalizados en Streamlit
st.markdown(
    """
    <style>
    .main-title { text-align: center; font-size: 36px; font-weight: bold; color: #4CAF50; }
    .sub-title { text-align: center; font-size: 20px; color: #555; }
    .result-container { background-color: #f9f9f9; padding: 15px; border-radius: 10px; box-shadow: 2px 2px 10px rgba(0,0,0,0.1); }
    .predicted-class { font-size: 24px; font-weight: bold; color: #4CAF50; text-align: center; }
    </style>
""",
    unsafe_allow_html=True,
)

# Título principal
st.markdown(
    '<h1 class="main-title">Clasificación de Imágenes con Transfer Learning</h1>',
    unsafe_allow_html=True,
)
st.markdown(
    '<p class="sub-title">Sube una imagen y el modelo te dirá qué es con su probabilidad.</p>',
    unsafe_allow_html=True,
)

# Cargar imagen
uploaded_file = st.file_uploader(
    "Elige una imagen", type=["jpg", "jpeg", "png", "bmp", "tif", "tiff", "webp"]
)

if uploaded_file is not None:
    # Mostrar imagen cargada
    image = Image.open(uploaded_file)
    st.image(image, caption="Imagen cargada", use_column_width=True)

    # Procesar imagen
    image_tensor = process_image(image)

    if image_tensor is not None:
        # Obtener predicciones
        results = prediction_process(image_tensor)

        # Extraer la mejor predicción
        best_class = max(results, key=results.get)
        best_prob = results[best_class] * 100  # Convertir a porcentaje

        # Mostrar el resultado principal
        st.markdown(
            f'<p class="predicted-class">Predicción principal: {best_class} ({best_prob:.2f}%)</p>',
            unsafe_allow_html=True,
        )

        # Mostrar todas las predicciones con barras de progreso
        st.markdown('<div class="result-container">', unsafe_allow_html=True)
        for class_name, prob in results.items():
            st.write(f"**{class_name}**: {prob * 100:.2f}%")
            st.progress(prob)
        st.markdown("</div>", unsafe_allow_html=True)

        # Botón para probar otra imagen sin recargar la app
        if st.button("Subir otra imagen"):
            st.experimental_rerun()
    else:
        st.error("No se pudo procesar la imagen. Intenta con otro archivo.")
