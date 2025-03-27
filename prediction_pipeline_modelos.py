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
    ]
)


def load_model(model_name, model_path, num_classes):
    print(f"Cargando el modelo: {model_name}")  # Agrega este print para depuración

    # Cargar el modelo base según el nombre proporcionado
    if model_name == "efficientnet_rank_0":
        base_model = models.efficientnet_b0(weights="IMAGENET1K_V1")
    elif model_name == "efficientnet_rank_7":
        base_model = models.efficientnet_b0(weights="IMAGENET1K_V1")
    else:
        raise ValueError(f"Modelo no soportado: {model_name}")

    # Modificar la capa de salida para adaptarse al número de clases
    if model_name in ["efficientnet_rank_0", "efficientnet_rank_7"]:
        base_model.classifier[1] = nn.Linear(
            base_model.classifier[1].in_features, num_classes
        )
    else:
        raise ValueError(f"Modelo no soportado: {model_name}")

    # Cargar los pesos entrenados previamente
    try:
        state_dict = torch.load(model_path, map_location=torch.device("cpu"))
        base_model.load_state_dict(state_dict)
        base_model.eval()
        return base_model
    except Exception as e:
        print(f"Error al cargar el modelo: {e}")
        return None


# Función para procesar la imagen
def process_image(image):
    try:
        image = image.convert("RGB")
        image_tensor = transform(image).unsqueeze(0)  # Añadir dimensión de batch
        return image_tensor
    except Exception as e:
        print(f"Error al procesar la imagen: {e}")
        return None


# Función para predecir las clases con el modelo cargado
def predict(model, image_tensor, class_names):
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
    model_path = os.path.join("models_generator", "models", f"{model_name}.pt")
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
