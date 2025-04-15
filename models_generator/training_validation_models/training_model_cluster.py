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
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),  # Salida a consola
        logging.FileHandler("training_log.txt"),
    ],
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


def get_num_workers(default_max=40):
    """
    Devuelve el número óptimo de workers, limitado al menor entre el número de núcleos de CPU disponibles y default_max.
    """
    cpu_count = os.cpu_count() or default_max
    num_workers = min(cpu_count, default_max)
    logging.info(f"Número de workers configurados: {num_workers}")
    return num_workers


def load_data(train_dir, valid_dir, batch_size=64, img_size=224):
    # Transformaciones de datos
    transform = torchvision.transforms.Compose(
        [
            torchvision.transforms.Resize((img_size, img_size)),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            ),
        ]
    )

    # Verificar existencia de directorios
    if not os.path.exists(train_dir):
        raise ValueError(f"El directorio de entrenamiento no existe: {train_dir}")
    if not os.path.exists(valid_dir):
        raise ValueError(f"El directorio de validación no existe: {valid_dir}")

    # Cargar conjuntos de datos
    train_dataset = torchvision.datasets.ImageFolder(train_dir, transform=transform)
    valid_dataset = torchvision.datasets.ImageFolder(valid_dir, transform=transform)

    # Configurar el número de workers
    num_workers = get_num_workers()

    # Determinar si usar pin_memory (útil en GPU)
    pin_mem = True if torch.cuda.is_available() else False

    # Crear dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_mem,
    )

    valid_loader = DataLoader(
        valid_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_mem,
    )

    return train_loader, valid_loader, len(train_dataset.classes)


def evaluate(model, valid_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct_predictions = 0
    total_predictions = 0

    with torch.no_grad():
        for inputs, labels in valid_loader:
            # Mover datos al dispositivo óptimo
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item()

            # Calcular accuracy
            _, predicted = torch.max(outputs, 1)
            correct_predictions += (predicted == labels).sum().item()
            total_predictions += labels.size(0)

    avg_loss = running_loss / len(valid_loader)
    accuracy = 100 * correct_predictions / total_predictions
    return avg_loss, accuracy


def train(rank, world_size):
    try:
        logging.info(f"Comenzando entrenamiento en rank {rank}")

        # Configuración de hiperparámetros
        learning_rate = 1e-3  # Ajuste del learning rate
        epochs = 5  # Aumento de epochs a 5

        # Configuración de los directorios de datos
        train_dir = "../dataset/training"
        valid_dir = "../dataset/validation"

        # Cargar datos
        train_loader, valid_loader, num_classes = load_data(
            train_dir,
            valid_dir,
            batch_size=64,  # Mantener el tamaño del batch
            img_size=224,
        )

        # Configurar W&B con valores numéricos en lugar de strings
        wandb.init(
            project="Understanding-CNNs",
            config={
                "model": f"convnext_large_{epochs}_epochs_{learning_rate}_lr",
                "epochs": epochs,
                "batch_size": train_loader.batch_size,
            },
            name=f"convnext_large_{epochs}_epochs_{learning_rate}_lr",
        )

        logging.info(f"Número de clases: {num_classes}")
        logging.info(
            f"Longitud del conjunto de entrenamiento: {len(train_loader.dataset)}"
        )

        # Determinar el dispositivo óptimo
        device = get_device()

        # Preparar el modelo: carga convnext_large preentrenado
        model = torchvision.models.convnext_large(
            weights=torchvision.models.ConvNeXt_Large_Weights.DEFAULT
        )
        # Reemplazar la capa final para que coincida con el número de clases del dataset
        model.classifier[2] = nn.Linear(model.classifier[2].in_features, num_classes)

        # Si hay más de una GPU NVIDIA, usar DataParallel
        if torch.cuda.device_count() > 1 and device.type == "cuda":
            logging.info(f"Usando {torch.cuda.device_count()} GPUs con DataParallel")
            model = nn.DataParallel(model)

        # Mover el modelo al dispositivo óptimo
        model = model.to(device)

        # Definir optimizador, criterio y scheduler
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
        criterion = nn.CrossEntropyLoss()

        logging.info("Comenzando bucle de entrenamiento")

        # Bucle de entrenamiento
        for epoch in range(epochs):
            model.train()
            running_loss = 0.0
            correct_predictions = 0
            total_predictions = 0

            for batch_idx, (inputs, labels) in enumerate(train_loader):
                # Mover datos al dispositivo óptimo
                inputs, labels = inputs.to(device), labels.to(device)

                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                # Calcular accuracy para el batch
                _, predicted = torch.max(outputs, 1)
                correct_predictions += (predicted == labels).sum().item()
                total_predictions += labels.size(0)

                # Imprimir cada 10 batches
                if batch_idx % 10 == 0:
                    accuracy = 100 * correct_predictions / total_predictions
                    logging.info(
                        f"Época {epoch+1}, Batch {batch_idx}, Loss: {loss.item():.4f}, Train Accuracy: {accuracy:.2f}%"
                    )

            # Evaluar en el conjunto de validación
            val_loss, val_accuracy = evaluate(model, valid_loader, criterion, device)
            logging.info(
                f"Época {epoch+1}, Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.2f}%"
            )

            # Log metrics en wandb utilizando valores numéricos
            wandb.log(
                {
                    "epoch": epoch,
                    "train_loss": loss.item(),
                    "train_acc": accuracy,
                    "val_loss": val_loss,
                    "val_acc": val_accuracy,
                    "lr": optimizer.param_groups[0]["lr"],
                }
            )

            # Actualizar el scheduler
            scheduler.step()

        # Guardar el modelo final
        os.makedirs("./models", exist_ok=True)
        model_path = f"./models/convnext_large_{epochs}_epochs_{learning_rate}_lr"
        torch.save(model.state_dict(), model_path)
        logging.info(f"Modelo guardado exitosamente en {model_path}")
        wandb.finish()

    except Exception as e:
        logging.error(f"Error durante el entrenamiento: {e}", exc_info=True)


def main():
    # Usamos un solo dispositivo para entrenamiento (no se está usando entrenamiento distribuido)
    world_size = 1
    # En este ejemplo se ignora el parámetro rank para cargar el modelo convnext_large
    train(rank=0, world_size=world_size)


# EJECUCIÓN
if __name__ == "__main__":
    main()
