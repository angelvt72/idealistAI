import torch
import torchvision.transforms as transforms
from PIL import Image, UnidentifiedImageError
import os
import streamlit as st
import tempfile
import torchvision.models as models
from models_generator.cnn import CNN, load_data


def load_model(model_path, num_classes):
    # Initialize the base model with default weights
    base_model = models.resnet50(weights="DEFAULT")

    # Create the CNN model with the same architecture as during training
    model = CNN(base_model, num_classes)

    # Load the saved state dictionary
    state_dict = torch.load(model_path)

    # Load the weights into the model
    model.load_state_dict(state_dict)

    # Set the model to evaluation mode
    model.eval()

    return model


def process_image(image_path):

    # Valid extensions for the image
    valid_extensions = [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"]

    try:
        # Open and transform the image
        image = Image.open(image_path)
    except UnidentifiedImageError:
        print(f"Error: No se pudo abrir la imagen {image_path}")
        return None

    # Define the transformation
    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),  # Resize to match model's expected input
            transforms.ToTensor(),  # Convert to tensor
        ]
    )

    # Apply transforms and add batch dimension
    image_tensor = transform(image).unsqueeze(0)

    return image_tensor


def predict(model, image_tensor, class_names):

    with torch.no_grad():
        output = model(image_tensor)

    # Apply softmax to get probabilities
    probabilities = torch.softmax(output, dim=1)

    # Get top 3 predictions
    top_probs, top_indices = torch.topk(probabilities, 3)
    top_probs = top_probs.squeeze().tolist()
    top_indices = top_indices.squeeze().tolist()

    # Create results dictionary
    results = {class_names[idx]: prob for idx, prob in zip(top_indices, top_probs)}

    return results


def prediction_process(image_path):

    model_path = os.path.join("models_generator", "models", "resnet50-2epoch.pt")
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

    # Cargar el modelo
    model = load_model(model_path, num_classes)

    # Procesar la imagen
    image_tensor = process_image(image_path)

    # Obtener las predicciones
    results = predict(model, image_tensor, class_names)

    return results
