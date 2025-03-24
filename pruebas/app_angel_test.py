import streamlit as st
from PIL import Image
import tempfile
import os

from models_generator.PredictionProcess import PredictionProcess


def predict_image(uploaded_image):
    """
    Helper function to process the uploaded image and make predictions.

    Args:
        uploaded_image (UploadedFile): Image uploaded through Streamlit

    Returns:
        tuple: Dictionary of predictions and top prediction
    """
    # Create a temporary file to save the uploaded image
    with tempfile.NamedTemporaryFile(
        delete=False, suffix=os.path.splitext(uploaded_image.name)[1]
    ) as temp_file:
        temp_file.write(uploaded_image.getvalue())
        temp_file_path = temp_file.name

    try:
        # Call PredictionProcess with the temporary file path
        results = PredictionProcess(temp_file_path)

        # Get the top prediction (first item in the dictionary)
        top_prediction = list(results.items())[0]

        return results, top_prediction

    finally:
        # Ensure the temporary file is deleted
        os.unlink(temp_file_path)


def main():
    # Título de la aplicación
    st.title("Clasificación de Imágenes con Transfer Learning")

    # Subir una imagen
    uploaded_image = st.file_uploader(
        "Sube una imagen",
        type=["png", "jpg", "jpeg"],
        help="Sube una imagen para clasificarla",
    )

    if uploaded_image is not None:
        # Mostrar la imagen cargada
        image = Image.open(uploaded_image)
        st.image(image, caption="Imagen cargada", use_column_width=True)

        # Botón para hacer la predicción
        if st.button("Predecir clase"):
            with st.spinner("Realizando predicción..."):
                try:
                    # Llamar a la función de predicción
                    results, (top_class, top_prob) = predict_image(uploaded_image)

                    # Crear una columna para mostrar los resultados
                    st.subheader("Resultados de la Predicción")

                    # Mostrar la predicción principal
                    st.markdown(f"**Clase predicha:** {top_class}")
                    st.markdown(f"**Probabilidad:** {top_prob:.4f}")

                    # Mostrar todas las predicciones en una tabla
                    st.subheader("Top 3 Predicciones")
                    prediction_data = [
                        {"Clase": cls, "Probabilidad": prob}
                        for cls, prob in results.items()
                    ]
                    st.table(prediction_data)

                    # Gráfico de barras de probabilidades
                    st.subheader("Distribución de Probabilidades")
                    st.bar_chart({cls: prob for cls, prob in results.items()})

                except Exception as e:
                    st.error(f"Error en la predicción: {str(e)}")


if __name__ == "__main__":
    main()
