import torch
import torchvision.transforms as transforms
from PIL import Image
import os
import torchvision.models as models
import torch.nn as nn

# Definir transformaciones para la imagen
transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),  # Cambiar el tamaño de la imagen a 224x224 (OJO: podría cambiar según el modelo que queramos [futuro])
        transforms.ToTensor(),  # Convertir la imagen a tensor normalizado
    ]
)

# Función para cargar el modelo y predecir la clase de la imagen
def load_model(model_path, num_classes):
    """
    Carga un modelo preentrenado de EfficientNet(efficientnet_b0) y ajusta su capa de clasificación para que coincida con el número de clases especificado. Luego, carga los pesos personalizados entrenados previamente desde el archivo especificado.

    Params:
    ----------
    model_path : str
        Ruta al archivo que contiene los pesos entrenados previamente del modelo.
    num_classes : int
        Número de clases para la tarea de clasificación. Este valor se utiliza para ajustar la capa de salida del modelo.

    Returns:
    -------
    torch.nn.Module o None
        Devuelve el modelo EfficientNet cargado con los pesos personalizados y configurado para el número de clases especificado. Si ocurre un error durante la carga de los pesos, devuelve `None`.
    """


    # Cargar el modelo EfficientNet sin pesos preentrenados, ya que se usaron pesos personalizados
    base_model = models.efficientnet_b0(weights="DEFAULT")
    base_model.classifier[1] = nn.Linear(
        base_model.classifier[1].in_features, num_classes
    )

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
    """
    Procesa una imagen de entrada convirtiéndola al formato RGB (si no está ya en RGB) y transformándola en un tensor adecuado para la entrada del modelo.
    Args:
        image (PIL.Image.Image): La imagen de entrada a procesar.
    Returns:
        torch.Tensor: Una representación tensorial de la imagen con una dimensión de batch añadida.
        None: Si ocurre un error durante el procesamiento, se devuelve None.
    Raises:
        Exception: Captura e imprime cualquier excepción que ocurra durante el procesamiento de la imagen.
    """

    # Convertir la imagen a RGB y aplicar las transformaciones definidas
    try:
        image = image.convert("RGB")  # Convertir a RGB si no lo está
        image_tensor = transform(image).unsqueeze(0)  # Aplicar transformaciones y añadir dimensión de batch
        return image_tensor
    except Exception as e:
        print(f"Error al procesar la imagen: {e}")
        return None


# Función para predecir las clases con el modelo cargado
def predict(model, image_tensor, class_names):
    """
    Predice las 3 clases principales con sus probabilidades para un tensor de imagen dado utilizando un modelo entrenado.

    Args:
        model (torch.nn.Module): El modelo PyTorch entrenado utilizado para la predicción.
        image_tensor (torch.Tensor): El tensor de imagen de entrada que se clasificará.
        class_names (list of str): Una lista de nombres de clases que corresponden a los índices de salida del modelo.

    Returns:
        dict: Un diccionario donde las claves son los nombres de las clases (str) y los valores son las probabilidades 
              correspondientes (float) para las 3 clases principales predichas.
    """
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
    """
    Realiza el proceso completo de predicción para un tensor de imagen dado.

    Args:
        image_tensor (torch.Tensor): El tensor de imagen de entrada que se clasificará.

    Returns:
        dict: Un diccionario con las clases predichas y sus probabilidades, o un mensaje de error si el modelo no se pudo cargar.
    """
    model_path = os.path.join("models_generator", "models", "efficientnet_rank_7.pt")
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

    # Verificar si el modelo se cargó correctamente
    if model is None:
        return {"error": "No se pudo cargar el modelo."}

    return predict(model, image_tensor, class_names)
