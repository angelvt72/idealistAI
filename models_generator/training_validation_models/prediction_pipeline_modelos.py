import torch
import torchvision.transforms as transforms
from PIL import Image
import os
import torchvision.models as models
import torch.nn as nn
import streamlit as st

# Definir transformaciones para la imagen
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                         std=[0.229, 0.224, 0.225])
])


def load_model(model_name, model_path, num_classes):
    """
    Carga un modelo preentrenado y ajusta su capa de salida para adaptarse al número de clases especificado.
    Args:
        model_name (str): Nombre del modelo a cargar. Actualmente soporta "efficientnet_rank_0", "efficientnet_rank_7".
        model_path (str): Ruta al archivo que contiene los pesos entrenados previamente del modelo.
        num_classes (int): Número de clases para la tarea de clasificación. Se utiliza para ajustar la capa de salida del modelo.
    Returns:
        torch.nn.Module: Modelo cargado y ajustado, listo para evaluación.
        None: Si ocurre un error al cargar el modelo.
    Raises:
        ValueError: Si el nombre del modelo proporcionado no es soportado.
    Notas:
        - El modelo base utilizado es EfficientNet-B0 con pesos preentrenados en ImageNet.
        - La capa de clasificación del modelo se ajusta para que coincida con el número de clases especificado.
        - Los pesos del modelo se cargan desde la ruta especificada y se asignan al modelo base.
    """

    print(f"Cargando el modelo: {model_name}")  # Agrega este print para depuración

    # Cargar el modelo base según el nombre proporcionado
    if model_name == "efficientnet_rank_0":
        base_model = models.efficientnet_b0(weights="IMAGENET1K_V1")
    elif model_name == "efficientnet_rank_7":
        base_model = models.efficientnet_b7(weights="DEFAULT")
    elif model_name == "convnext_large_1_epoch":
        base_model = models.convnext_large(weights=models.ConvNeXt_Large_Weights.IMAGENET1K_V1)
    elif model_name == "convnext_large_3_epochs":
        base_model = models.convnext_large(weights=models.ConvNeXt_Large_Weights.DEFAULT)

    # Modificar la capa de salida para adaptarse al número de clases
    if model_name in ["efficientnet_rank_0", "efficientnet_rank_7"]:
        base_model.classifier[1] = nn.Linear(
            base_model.classifier[1].in_features, num_classes
        )

    elif model_name == "convnext_large_1_epoch":
        in_features = base_model.classifier[-1].in_features
        classifier = torch.nn.Sequential(
        torch.nn.Linear(in_features, 512),
        torch.nn.ReLU(),
        torch.nn.Dropout(0.2),
        torch.nn.Linear(512, num_classes)
        )
        base_model.classifier[-1] = classifier

    elif model_name == "convnext_large_3_epochs":
        base_model.classifier[2] = nn.Linear(base_model.classifier[2].in_features, num_classes)
    

    else:
        raise ValueError(f"Modelo no soportado: {model_name}")

    # Cargar los pesos entrenados previamente
    try:
        state_dict = torch.load(model_path, map_location=torch.device("cpu"))
        base_model.load_state_dict(state_dict, strict=False)
        base_model.eval()
        return base_model
    except Exception as e:
        print(f"Error al cargar el modelo: {e}")
        return None


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

    model_path = os.path.join("models_generator", "models", f"{model_name}.pt" if model_name != "convnext_large_1_epoch" else f"{model_name}.pth")
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
