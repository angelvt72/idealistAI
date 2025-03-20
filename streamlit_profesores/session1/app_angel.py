import streamlit as st
import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import skimage
from PIL import Image

# ========================================================================
# Funciones
# ========================================================================


def load_default_image():
    """
    Carga una imagen predeterminada (la imagen de un gato de skimage) y realiza las siguientes transformaciones:
    1. Convierte la imagen en una imagen PIL.
    2. Convierte la imagen a escala de grises.
    3. Transforma la imagen en un tensor de PyTorch.
    4. Normaliza la imagen para que sus valores estén entre 0 y 1.

    Retorna un diccionario con la imagen original, la imagen en escala de grises y el tensor normalizado.
    """
    img = skimage.data.chelsea()  # Carga una imagen de un gato de la librería skimage
    img = Image.fromarray(img)  # Convierte la imagen NumPy a una imagen de tipo PIL
    img_gray = transforms.Grayscale(num_output_channels=1)(
        img
    )  # Convierte la imagen a escala de grises
    img_tensor = transforms.ToTensor()(
        img_gray
    )  # Convierte la imagen a un tensor de PyTorch
    img_tensor = (img_tensor - torch.min(img_tensor)) / (
        torch.max(img_tensor) - torch.min(img_tensor)
    )  # Normaliza los valores del tensor entre 0 y 1

    return {"img": img, "im_gray": img_gray, "img_tensor": img_tensor}


def get_filter(filter_name):
    """
    Devuelve un kernel de convolución dependiendo del nombre del filtro seleccionado.
    Cada filtro es una matriz de 3x3 que se utilizará para realizar la operación de convolución.
    """
    if filter_name == "No Filter":
        filter = torch.tensor([[0, 0, 0], [0, 1, 0], [0, 0, 0]], dtype=torch.float)
    elif filter_name == "Gaussian":
        filter = torch.tensor([[1, 2, 1], [2, 4, 2], [1, 2, 1]], dtype=torch.float)
    elif filter_name == "Sharpen":
        filter = torch.tensor([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=torch.float)
    elif filter_name == "Edge Detection":
        filter = torch.tensor(
            [[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]], dtype=torch.float
        )
    else:
        raise ValueError("Invalid filter name")

    return filter


def convolve_and_plot(image, filter, stride=1, padding="same"):
    """
    Aplica la operación de convolución a una imagen utilizando un filtro dado.

    Parámetros:
    - image: Tensor de la imagen en escala de grises.
    - filter: Kernel de convolución (tensor 3x3).
    - stride: Paso con el que se mueve el filtro sobre la imagen.
    - padding: Define el tipo de padding a usar para mantener el tamaño de la imagen.

    Retorna la imagen procesada después de aplicar la convolución.
    """
    filter /= torch.sum(torch.abs(filter))  # Normaliza los valores del filtro
    filter = filter.unsqueeze(0).unsqueeze(0)  # Ajusta la forma del tensor para PyTorch

    # Aplica la convolución utilizando la función conv2d de PyTorch
    output_img = torch.nn.functional.conv2d(
        image.unsqueeze(0), filter, bias=None, stride=stride, padding=padding
    )

    # Normaliza la imagen resultante para que sus valores estén entre 0 y 1
    output_img = (output_img - torch.min(output_img)) / (
        torch.max(output_img) - torch.min(output_img)
    )

    return output_img.squeeze()


# ========================================================================
# Configuración de la aplicación en Streamlit
# ========================================================================
st.set_page_config(layout="wide", page_title="DL Lab - Session 1")

# Creamos una interfaz con tres columnas
col01, col02, col03 = st.columns(3)
with col01:
    st.write("# Deep Learning Lab - Session 1")  # Título de la aplicación
with col03:
    # Mostramos una imagen del logo de ICAI
    st.image(
        "/Users/angel/Desktop/MASTER/SEGUNDO_CUATRI/ML2/idealistAI/streamlit_profesores/img/icai.png"
    )

st.write("## Convolution Filters")  # Subtítulo de la aplicación

# Sección para mostrar la imagen original y seleccionar un filtro
col11, col12, col13 = st.columns(3)
with col11:
    st.write("#### Original Image")  # Título para la imagen original
with col12:
    # Dropdown para seleccionar un filtro de convolución
    selected_filter_name = st.selectbox(
        "Select a Filter", ("No Filter", "Gaussian", "Sharpen", "Edge Detection")
    )

# Sección para visualizar la imagen original y la imagen filtrada
col21, col22, col23 = st.columns(3)
with col21:
    # Cargamos la imagen predeterminada
    default_image = load_default_image()

    # Mostramos la imagen en escala de grises
    fig, ax = plt.subplots(figsize=(12, 6))
    plt.imshow(
        default_image["img_tensor"].squeeze(), cmap="gray"
    )  # Muestra la imagen en escala de grises
    plt.colorbar()  # Barra de colores para referencia de valores
    st.pyplot(fig)  # Mostramos la imagen en Streamlit

with col22:
    # Obtiene el filtro seleccionado
    filter = get_filter(selected_filter_name)

    # Aplica la convolución con el filtro seleccionado
    output_image = convolve_and_plot(default_image["img_tensor"], filter)

    # Muestra la imagen filtrada
    fig, ax = plt.subplots(figsize=(12, 6))
    plt.imshow(output_image.squeeze(), cmap="gray")  # Muestra la imagen resultante
    plt.colorbar()  # Barra de colores
    st.pyplot(fig)  # Mostramos la imagen filtrada en Streamlit
