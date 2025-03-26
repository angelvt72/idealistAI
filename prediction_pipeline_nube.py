import torch
import torchvision.transforms as transforms
from PIL import Image
import os
import torchvision.models as models
from models_generator.cnn import CNN

# Definir transformaciones para la imagen
transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ]
)


def load_model(model_path, num_classes):
    base_model = models.resnet50(weights="DEFAULT")
    model = CNN(base_model, num_classes)
    state_dict = torch.load(model_path, map_location=torch.device("cpu"))
    model.load_state_dict(state_dict)
    model.eval()
    return model


def process_image(image):
    """Convierte una imagen en memoria a un tensor compatible con ResNet50."""
    try:
        image = image.convert("RGB")  # Convertir a RGB si no lo está
        image_tensor = transform(image).unsqueeze(0)  # Añadir dimensión de batch
        return image_tensor
    except Exception as e:
        print(f"Error al procesar la imagen: {e}")
        return None


def predict(model, image_tensor, class_names):
    with torch.no_grad():
        output = model(image_tensor)
        probabilities = torch.softmax(output, dim=1)
        top_probs, top_indices = torch.topk(probabilities, 3)
        return {
            class_names[idx]: prob
            for idx, prob in zip(
                top_indices.squeeze().tolist(), top_probs.squeeze().tolist()
            )
        }


def prediction_process(image_tensor):
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
    model = load_model(model_path, num_classes)
    return predict(model, image_tensor, class_names)
