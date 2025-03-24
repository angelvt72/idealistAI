import streamlit as st
from PIL import Image
import tempfile
import os
import sys

# Añadir el directorio raíz del proyecto a sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Ahora puedes importar el módulo desde models_generator
from models_generator.PredictionProcess import prediction_process


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
                    # Crear un archivo temporal para la imagen
                    with tempfile.NamedTemporaryFile(
                        delete=False, suffix=os.path.splitext(uploaded_image.name)[1]
                    ) as temp_file:
                        temp_file.write(uploaded_image.getvalue())
                        temp_file_path = temp_file.name

                    try:
                        # Llamar a la función de predicción
                        results = prediction_process(temp_file_path)

                        # Ordenar resultados por probabilidad descendente
                        sorted_results = dict(
                            sorted(results.items(), key=lambda x: x[1], reverse=True)
                        )

                        # Crear una columna para mostrar los resultados
                        st.subheader("Resultados de la Predicción")

                        # Mostrar la predicción principal
                        top_class = list(sorted_results.keys())[0]
                        top_prob = sorted_results[top_class]

                        st.markdown(f"**Clase predicha:** {top_class}")
                        st.markdown(f"**Probabilidad:** {top_prob:.4f}")

                        # Mostrar todas las predicciones en una tabla
                        st.subheader("Top 3 Predicciones")
                        prediction_data = [
                            {"Clase": cls, "Probabilidad": prob}
                            for cls, prob in sorted_results.items()
                        ]
                        st.table(prediction_data)

                        # Gráfico de barras de probabilidades
                        st.subheader("Distribución de Probabilidades")
                        st.bar_chart(
                            {cls: prob for cls, prob in sorted_results.items()}
                        )

                    finally:
                        # Eliminar el archivo temporal
                        os.unlink(temp_file_path)

                except Exception as e:
                    st.error(f"Error en la predicción: {str(e)}")


if __name__ == "__main__":
    main()
