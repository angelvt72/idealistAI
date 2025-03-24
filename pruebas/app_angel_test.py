# ========================================================================
# Configuración de la aplicación en Streamlit
# ========================================================================
import streamlit as st

# Configurar la página
st.set_page_config(layout="wide", page_title="Carga de Imagen")

# Título de la aplicación
st.title("Sube una imagen desde tu ordenador")

# Componente para cargar una imagen
uploaded_file = st.file_uploader("Selecciona una imagen", type=["png", "jpg", "jpeg"])

# Verificamos si el usuario ha subido una imagen
if uploaded_file is not None:
    st.image(uploaded_file, caption="Imagen cargada", use_column_width=True)
