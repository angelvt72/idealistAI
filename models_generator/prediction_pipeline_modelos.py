import torch
import torchvision.transforms as transforms
from PIL import Image
import os
import torchvision.models as models
import torch.nn as nn
import streamlit as st

# Definir transformaciones para la imagen
transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


def load_model(model_name, model_path, num_classes):
    """
    Carga un modelo preentrenado, ajusta su capa final para que tenga 'num_classes' salidas,
    y carga los pesos entrenados desde 'model_path'.

    Args:
        model_name (str): Nombre del modelo a cargar (e.g. "convnext_large", "efficientnet_b0", "efficientnet_b7").
        model_path (str): Ruta al archivo que contiene los pesos entrenados.
        num_classes (int): Número de clases para la capa final del modelo.

    Returns:
        torch.nn.Module: Modelo configurado y con pesos cargados, listo para evaluación.
        None: En caso de error en la carga de pesos.
    """
    import torchvision.models as models
    import torch.nn as nn
    import torch
    import logging

    model_name = model_name.lower().strip()

    # Selección de la arquitectura base y ajuste de la capa final
    if "convnext_large" in model_name:
        # Usamos la versión preentrenada de ConvNeXt Large
        model = models.convnext_large(weights=models.ConvNeXt_Large_Weights.DEFAULT)
        # En este caso, la capa final (posición 2) se reemplaza por una que tenga 'num_classes' salidas
        model.classifier[2] = nn.Linear(model.classifier[2].in_features, num_classes)
        logging.info("ConvNeXt Large model selected")

    elif "efficientnet_b7" in model_name or "efficientnet_rank_7" in model_name:
        model = models.efficientnet_b7(weights=models.EfficientNet_B7_Weights.DEFAULT)
        # Se modifica la última capa (índice 1) para que tenga 'num_classes' salidas
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
        logging.info("EfficientNet-B7 model selected")

    elif "efficientnet_b0" in model_name or "efficient" in model_name:
        model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
        # Se ajusta la capa final (índice 1) para que tenga 'num_classes' salidas
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
        logging.info("EfficientNet-B0 model selected")

    else:
        raise ValueError(
            f"Modelo no soportado: {model_name}. Opciones disponibles: convnext_large, efficientnet_b0, efficientnet_b7"
        )

    # Cargar los pesos entrenados previamente
    try:
        logging.info(f"Cargando pesos del modelo desde: {model_path}")
        state_dict = torch.load(model_path, map_location=torch.device("cpu"))
        # Se usa strict=False por si la arquitectura ajustada difiere en detalles (por ejemplo, la capa final)
        model.load_state_dict(state_dict, strict=False)
        logging.info("Pesos cargados correctamente")
    except Exception as e:
        logging.error(f"Error al cargar el modelo: {e}")
        return None

    model.eval()
    return model


# Función para procesar la imagen
def process_image(image):
    """
    Procesa una imagen para convertirla en un tensor adecuado para un modelo de aprendizaje automático.
    Args:
        image (PIL.Image.Image): La imagen de entrada que se desea procesar.
    Returns:
        torch.Tensor o None: Devuelve un tensor de PyTorch con la imagen procesada y una dimensión de batch añadida.
        Si ocurre un error durante el procesamiento, devuelve None.
    Excepciones:
        Imprime un mensaje de error si ocurre una excepción durante el procesamiento de la imagen.
    """

    try:
        image = image.convert("RGB")
        image_tensor = transform(image).unsqueeze(0)  # Añadir dimensión de batch
        return image_tensor
    except Exception as e:
        print(f"Error al procesar la imagen: {e}")
        return None


# Función para predecir las clases con el modelo cargado
def predict(model, image_tensor, class_names):
    """
    Realiza una predicción utilizando un modelo de aprendizaje profundo y devuelve las
    probabilidades de las tres clases más probables.
    Args:
        model (torch.nn.Module): El modelo de aprendizaje profundo previamente cargado.
        image_tensor (torch.Tensor): El tensor que representa la imagen de entrada,
            preparado para ser procesado por el modelo.
        class_names (list): Lista de nombres de las clases en el mismo orden que las
            salidas del modelo.
    Returns:
        dict: Un diccionario donde las claves son los nombres de las clases y los valores
        son las probabilidades asociadas (en formato string con 4 decimales). Si el modelo
        no está cargado correctamente, devuelve un diccionario vacío.
    """

    if model is None:
        print("Error: El modelo no se ha cargado correctamente.")
        return {}

    with torch.no_grad():
        output = model(image_tensor)
        probabilities = torch.softmax(output, dim=1)
        top_probs, top_indices = torch.topk(probabilities, 3)

        result = {}
        for idx, prob in zip(
            top_indices.squeeze().tolist(), top_probs.squeeze().tolist()
        ):
            result[class_names[idx]] = f"{prob:.4f}"

        return result


# Función principal para procesar la predicción
def prediction_process(image_tensor, model_name):
    """
    Realiza el proceso de predicción utilizando un modelo de aprendizaje automático.
    Args:
        image_tensor (torch.Tensor): Tensor que representa la imagen de entrada para la predicción.
        model_name (str): Nombre del modelo que se utilizará para realizar la predicción.
    Returns:
        dict: Un diccionario con los resultados de la predicción o un mensaje de error si no se pudo cargar el modelo.
            - Si el modelo se carga correctamente, el diccionario contiene las predicciones realizadas por el modelo.
            - Si ocurre un error al cargar el modelo, el diccionario contiene la clave "error" con un mensaje descriptivo.
    """

    model_dir = os.path.join("models_generator", "models")
    for ext in [".pt", ".pth"]:
        model_path = os.path.join(model_dir, f"{model_name}{ext}")
        if os.path.exists(model_path):
            break
    else:
        raise FileNotFoundError(
            f"No model file found for {model_name} with .pt or .pth extension"
        )
    num_classes = 15
    class_names = [
        "Bedroom",
        "Coast",
        "Forest",
        "Highway",
        "Industrial",
        "InsideCity",
        "Kitchen",
        "LivingRoom",
        "Mountain",
        "Office",
        "OpenCountry",
        "Store",
        "Street",
        "Suburb",
        "TallBuilding",
    ]

    model = load_model(model_name, model_path, num_classes)

    if model is None:
        return {"error": "No se pudo cargar el modelo."}

    return predict(model, image_tensor, class_names)
