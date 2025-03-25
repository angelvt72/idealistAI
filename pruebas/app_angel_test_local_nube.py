import streamlit as st
from PIL import Image
import sys
import os
import tempfile  # Se importa para manejar archivos temporales

current_path = os.path.dirname(os.path.abspath(__file__))
parent_path = os.path.abspath(os.path.join(current_path, ".."))
if parent_path not in sys.path:
    sys.path.insert(0, parent_path)

# Importar la función de predicción desde models_generator
from models_generator.PredictionProcess import prediction_process

st.title("Clasificación de Imágenes con Transfer Learning")

uploaded_image = st.file_uploader("Sube una imagen", type=["png", "jpg", "jpeg"])

if uploaded_image is not None:
    image = Image.open(uploaded_image)
    st.image(image, caption="Imagen cargada", use_column_width=True)

    if st.button("Predecir clase"):
        with st.spinner("Realizando predicción..."):
            try:
                # Guardar la imagen cargada en un archivo temporal
                with tempfile.NamedTemporaryFile(
                    delete=False, suffix=os.path.splitext(uploaded_image.name)[1]
                ) as temp_file:
                    temp_file.write(uploaded_image.getvalue())
                    temp_file_path = temp_file.name

                # Llamar a la función de predicción pasando la ruta temporal
                results = prediction_process(temp_file_path)

                # Ordenar los resultados por probabilidad descendente
                sorted_results = dict(
                    sorted(results.items(), key=lambda x: x[1], reverse=True)
                )

                st.subheader("Resultados de la Predicción")
                top_class = list(sorted_results.keys())[0]
                top_prob = sorted_results[top_class]

                st.markdown(f"**Clase predicha:** {top_class}")
                st.markdown(f"**Probabilidad:** {top_prob:.4f}")

                # Eliminar el archivo temporal
                os.unlink(temp_file_path)

            except Exception as e:
                st.error(f"Error en la predicción: {str(e)}")
