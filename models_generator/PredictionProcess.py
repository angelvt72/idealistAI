import torch
from PIL import Image
import torchvision.transforms as transforms
from cnn import *
import torchvision.models as models


def load_model(model_path):
    """
    Carga el state_dict guardado en el fichero .pt y lo carga en la arquitectura ResNet50.
    Luego, pone el modelo en modo evaluación.
    """
    # Instanciar la arquitectura de ResNet50 (sin pesos pre-entrenados)
    model = models.resnet50(pretrained=False)
    
    # Cargar el state_dict
    state_dict = torch.load(model_path, map_location=torch.device('cpu'))
    model.load_state_dict(state_dict)
    
    # Poner el modelo en modo evaluación
    model.eval()
    return model

def process_image(image_path):
    """
    Abre la imagen, la convierte a escala de grises y la redimensiona a 224x224.
    Luego la transforma en un tensor adecuado para el modelo.
    """
    # Abrir imagen
    image = Image.open(image_path)
    # Convertir a escala de grises (modo 'L' para imágenes en blanco y negro)
    image = image.convert("L")
    # Redimensionar a 224x224
    image = image.resize((224, 224))
    
    # Convertir imagen a tensor; ToTensor() normaliza a [0,1] y cambia la forma a (C, H, W)
    transform = transforms.ToTensor()
    image_tensor = transform(image)
    
    # Añadir dimensión de batch: forma final (1, C, H, W)
    image_tensor = image_tensor.unsqueeze(0)
    return image_tensor

def predict(model, image_tensor, class_names):
    """
    Realiza la inferencia con el modelo, aplica softmax para obtener probabilidades
    y extrae las 3 predicciones más altas.
    Devuelve un diccionario {clase: probabilidad}.
    """
    with torch.no_grad():
        output = model(image_tensor)
    
    # Aplicar softmax para convertir los logits en probabilidades
    probabilities = torch.softmax(output, dim=1)
    
    # Obtener los índices y valores de las 3 predicciones más altas
    top_probs, top_indices = torch.topk(probabilities, 3)
    top_probs = top_probs.squeeze().tolist()
    top_indices = top_indices.squeeze().tolist()
    
    # En caso de que solo se devuelva un valor, asegurarse de trabajar con listas
    if isinstance(top_indices, int):
        top_indices = [top_indices]
        top_probs = [top_probs]
    
    # Crear el diccionario de resultados mapeando índices a nombres de clase
    results = { class_names[i]: top_probs[idx] for idx, i in enumerate(top_indices) }
    return results

def PredictionProcess(image_path):
    
    # Procesar la imagen
    image_tensor = process_image(image_path)
    # Cargar el modelo
    model = load_model(r"C:\Users\pablo\MBD_ICAI_repo\MBD_ICAI\Machine_Learning_2_ML2\idealistAI\models_generator\models\resnet50-2epoch.pt")
    # Obtener las clases del modelo
    train_dir = './dataset/training'
    valid_dir = './dataset/validation'
    train_loader, valid_loader, num_classes = load_data(train_dir, 
                                                    valid_dir, 
                                                    batch_size=32, 
                                                    img_size=224) # ResNet50 requires 224x224 images
    class_names = train_loader.dataset.classes
    # Obtener predicciones
    results = predict(model, image_tensor, class_names)
    
    print("Top 3 predicciones:")
    print(results)
    
    return results



resultados = PredictionProcess(r"C:\Users\pablo\MBD_ICAI_repo\MBD_ICAI\Machine_Learning_2_ML2\idealistAI\models_generator\dataset\training\Bedroom\image_0003.jpg")