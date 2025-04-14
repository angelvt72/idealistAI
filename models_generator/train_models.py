import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision
import os
import logging
import sys
import wandb

# Inicia sesión en wandb
wandb.login()

# Configuración de logging detallada
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("training_log.txt")
    ]
)

def get_device():
    """
    Determina el dispositivo óptimo:
      - GPU NVIDIA si está disponible (CUDA)
      - GPU Apple Silicon si está disponible (MPS)
      - CPU en otro caso
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
        logging.info("Usando GPU NVIDIA (CUDA)")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
        logging.info("Usando GPU Apple Silicon (MPS)")
    else:
        device = torch.device("cpu")
        logging.info("Usando CPU")
    return device

def get_num_workers(default_max=8):
    """
    Devuelve el número óptimo de workers, limitado al menor entre el número de núcleos disponibles y default_max.
    """
    cpu_count = os.cpu_count() or default_max
    num_workers = min(cpu_count, default_max)
    logging.info(f"Número de workers configurados: {num_workers}")
    return num_workers

def load_data(train_dir, valid_dir, batch_size=8, img_size=224):
    # Transformaciones de datos
    transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize((img_size, img_size)),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(
            mean=[0.485, 0.456, 0.406], 
            std=[0.229, 0.224, 0.225]
        )
    ])
    
    if not os.path.exists(train_dir):
        raise ValueError(f"El directorio de entrenamiento no existe: {train_dir}")
    if not os.path.exists(valid_dir):
        raise ValueError(f"El directorio de validación no existe: {valid_dir}")
    
    train_dataset = torchvision.datasets.ImageFolder(train_dir, transform=transform)
    valid_dataset = torchvision.datasets.ImageFolder(valid_dir, transform=transform)
    
    num_workers = get_num_workers()
    pin_mem = True if torch.cuda.is_available() else False
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_mem
    )
    
    valid_loader = DataLoader(
        valid_dataset, 
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_mem
    )
    
    return train_loader, valid_loader, len(train_dataset.classes)

def evaluate(model, valid_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct_predictions = 0
    total_predictions = 0
    with torch.no_grad():
        for inputs, labels in valid_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct_predictions += (predicted == labels).sum().item()
            total_predictions += labels.size(0)
    avg_loss = running_loss / len(valid_loader)
    accuracy = 100 * correct_predictions / total_predictions
    return avg_loss, accuracy

# Función auxiliar para construir y configurar el modelo según la elección del usuario.
def build_model(model_name, num_classes):
    model_name = model_name.lower().strip()
    if model_name == "convnext_large":
        # Cargar ConvNeXt Large preentrenado
        model = torchvision.models.convnext_large(weights=torchvision.models.ConvNeXt_Large_Weights.DEFAULT)
        # Reemplazar la capa final (en posición [2] de la secuencia) para ajustar el número de clases
        model.classifier[2] = nn.Linear(model.classifier[2].in_features, num_classes)
        logging.info("Modelo ConvNeXt Large seleccionado")
    elif model_name == "efficientnet_b0":
        # Cargar EfficientNet-B0 preentrenado
        model = torchvision.models.efficientnet_b0(weights=torchvision.models.EfficientNet_B0_Weights.DEFAULT)
        # Reemplazar la última capa (índice [1]) de la secuencia classifier
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
        logging.info("Modelo EfficientNet-B0 seleccionado")
    else:
        raise ValueError("Modelo no soportado. Opciones disponibles: convnext_large, efficientnet_b0.")
    return model

def train(rank, world_size, model_choice, learning_rate, epochs):
    try:
        logging.info(f"Comenzando entrenamiento en rank {rank}")
        
        train_dir = "./models_generator/dataset/training"
        valid_dir = "./models_generator/dataset/validation"
        
        # Cargar datos
        train_loader, valid_loader, num_classes = load_data(
            train_dir, valid_dir, 
            batch_size=8,  
            img_size=224
        )
        
        # Configurar W&B
        wandb.init(
            project="IdealistAI",
            config={
                "model": f"{model_choice}_{epochs}_epochs_{learning_rate}_lr",
                "epochs": epochs,
                "batch_size": train_loader.batch_size
            },
            name=f"{model_choice}_{epochs}_epochs_{learning_rate}_lr"
        )
        
        logging.info(f"Número de clases: {num_classes}")
        logging.info(f"Longitud del conjunto de entrenamiento: {len(train_loader.dataset)}")
        
        device = get_device()
        
        # Construir el modelo dinámicamente según la elección del usuario
        model = build_model(model_choice, num_classes)
        
        # Si hay más de una GPU NVIDIA, usar DataParallel
        if torch.cuda.device_count() > 1 and device.type == "cuda":
            logging.info(f"Usando {torch.cuda.device_count()} GPUs con DataParallel")
            model = nn.DataParallel(model)
        
        model = model.to(device)
        
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
        criterion = nn.CrossEntropyLoss()
        
        logging.info("Comenzando bucle de entrenamiento")
        for epoch in range(epochs):
            model.train()
            running_loss = 0.0
            correct_predictions = 0
            total_predictions = 0

            for batch_idx, (inputs, labels) in enumerate(train_loader):
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                _, predicted = torch.max(outputs, 1)
                correct_predictions += (predicted == labels).sum().item()
                total_predictions += labels.size(0)
                
                if batch_idx % 10 == 0:
                    accuracy = 100 * correct_predictions / total_predictions
                    logging.info(f"Época {epoch+1}, Batch {batch_idx}, Loss: {loss.item():.4f}, Train Accuracy: {accuracy:.2f}%")

            val_loss, val_accuracy = evaluate(model, valid_loader, criterion, device)
            logging.info(f"Época {epoch+1}, Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.2f}%")
            
            wandb.log({
                "epoch": epoch,
                "train_loss": loss.item(),
                "train_acc": accuracy,
                "val_loss": val_loss,
                "val_acc": val_accuracy,
                "lr": optimizer.param_groups[0]['lr']
            })
            
            scheduler.step()
        
        try:
            os.makedirs('./models_generator/models', exist_ok=True)
        except Exception as e:
            pass
        
        model_path = f"./models_generator/models/{model_choice}_{epochs}epochs_{learning_rate}_lr.pt"
        torch.save(model.state_dict(), model_path)
        logging.info(f"Modelo guardado exitosamente en {model_path}")
        wandb.finish()
        
    except Exception as e:
        logging.error(f"Error durante el entrenamiento: {e}", exc_info=True)

def main():
    # Solicitar al usuario el modelo CNN a entrenar por la terminal
    model_choice = input("Ingrese el nombre del modelo CNN a entrenar (ej: convnext_large o efficientnet_b0): ").strip()

    # Elegir lerning rate y epochs
    epochs = 5
    learning_rate = 5 * 1e-4
    world_size = 1  # En este ejemplo se usa un solo dispositivo
    train(rank=0, world_size=world_size, model_choice=model_choice, learning_rate=learning_rate, epochs=epochs)

if __name__ == '__main__':
    main()