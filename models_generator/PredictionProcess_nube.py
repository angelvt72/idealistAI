import torch
import torchvision.transforms as transforms
from PIL import Image
import os

from models_generator.cnn import CNN, load_data
import torchvision.models as models

import os

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def load_model(model_path, num_classes):
    """
    Load a pre-trained ResNet50 model with the specified number of classes.

    Args:
        model_path (str): Path to the saved model weights
        num_classes (int): Number of classes in the model

    Returns:
        torch.nn.Module: Loaded model
    """
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
    """
    Prepare the image for model inference.

    Args:
        image_path (str): Path to the input image

    Returns:
        torch.Tensor: Processed image tensor
    """
    # Define transforms for preprocessing
    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),  # Resize to match model's expected input
            transforms.ToTensor(),  # Convert to tensor
            transforms.Normalize(  # Normalization used in pre-trained models
                mean=[0.485, 0.456, 0.406],  # ImageNet mean
                std=[0.229, 0.224, 0.225],  # ImageNet std
            ),
        ]
    )

    # Open and transform the image
    image = Image.open(image_path)

    # Handle grayscale images by converting to RGB
    if image.mode != "RGB":
        image = image.convert("RGB")

    # Apply transforms and add batch dimension
    image_tensor = transform(image).unsqueeze(0)

    return image_tensor


def predict(model, image_tensor, class_names):
    """
    Perform inference and return top predictions.

    Args:
        model (torch.nn.Module): Trained model
        image_tensor (torch.Tensor): Preprocessed image
        class_names (list): List of class names

    Returns:
        dict: Top predictions with their probabilities
    """
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


def prediction_process(
    image_path,
    model_path=None,
    train_dir=None,
    valid_dir=None,
):
    # Establece las rutas absolutas si no se proporcionan
    if train_dir is None:
        train_dir = os.path.join(BASE_DIR, "models_generator", "dataset", "training")
    if valid_dir is None:
        valid_dir = os.path.join(BASE_DIR, "models_generator", "dataset", "validation")
    if model_path is None:
        model_path = os.path.join(
            BASE_DIR, "models_generator", "models", "resnet50-2epoch.pt"
        )

    # Cargar dataset para obtener nombres de clases y número de clases
    train_loader, valid_loader, num_classes = load_data(
        train_dir, valid_dir, batch_size=32, img_size=224
    )
    class_names = train_loader.dataset.classes

    # Cargar el modelo
    model = load_model(model_path, num_classes)

    # Procesar la imagen
    image_tensor = process_image(image_path)

    # Obtener las predicciones
    results = predict(model, image_tensor, class_names)

    return results
