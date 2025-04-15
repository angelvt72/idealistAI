"""
Script para evaluar modelos CNN entrenados con métricas por clase.
Este script carga los modelos entrenados desde una carpeta y evalúa su rendimiento
en un conjunto de validación, mostrando la precisión global y por clase.
"""

import os
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets
import matplotlib.pyplot as plt
import argparse
import logging
import sys

# Configuración de logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("../logs/evaluation_log.txt"),
    ],
)


def get_device():
    """
    Determina el dispositivo óptimo para evaluación:
      - NVIDIA GPU si está disponible (CUDA)
      - Apple Silicon GPU si está disponible (MPS)
      - CPU en caso contrario
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
        logging.info("Usando NVIDIA GPU (CUDA)")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
        logging.info("Usando Apple Silicon GPU (MPS)")
    else:
        device = torch.device("cpu")
        logging.info("Usando CPU")
    return device


def load_data(valid_dir, batch_size=32, img_size=224):
    """
    Carga el conjunto de datos de validación.

    Parámetros:
    - valid_dir: Directorio que contiene los datos de validación
    - batch_size: Tamaño del lote
    - img_size: Tamaño de la imagen para redimensionar

    Retorna:
    - valid_loader: DataLoader con los datos de validación
    - class_names: Nombres de las clases
    """
    # Transformaciones para preprocesado de imágenes
    transform_val = transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    # Verificar si el directorio existe
    if not os.path.exists(valid_dir):
        raise ValueError(f"El directorio de validación no existe: {valid_dir}")

    # Cargar el dataset de validación
    val_dataset = datasets.ImageFolder(valid_dir, transform=transform_val)

    # Configurar el DataLoader
    num_workers = min(os.cpu_count() or 4, 8)
    pin_memory = True if torch.cuda.is_available() else False

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    # Obtener nombres de las clases
    class_names = val_dataset.classes

    logging.info(f"Dataset de validación cargado con {len(val_dataset)} imágenes.")
    logging.info(f"Clases encontradas: {class_names}")

    return val_loader, class_names


def evaluate(model, valid_loader, criterion, class_names, device):
    """
    Evalúa el rendimiento del modelo en un conjunto de datos de validación
    y calcula la precisión por clase.

    Parámetros:
    - model: Modelo a evaluar
    - valid_loader: DataLoader con los datos de validación
    - criterion: Función de pérdida
    - class_names: Lista con los nombres de las clases
    - device: Dispositivo donde se ejecutará la evaluación (cuda, mps, cpu)

    Retorna:
    - avg_loss: Pérdida promedio en el conjunto de validación
    - accuracy: Precisión global del modelo
    - class_accuracies: Diccionario con la precisión por cada clase
    """
    model.eval()
    running_loss = 0.0
    correct_predictions = 0
    total_predictions = 0

    # Inicializar contadores para la precisión por clase
    num_classes = len(class_names)
    correct_per_class = torch.zeros(num_classes)
    total_per_class = torch.zeros(num_classes)

    with torch.no_grad():
        for inputs, labels in valid_loader:
            # Mover datos al dispositivo
            inputs, labels = inputs.to(device), labels.to(device)

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item()

            # Obtener predicciones
            _, predicted = torch.max(outputs, 1)
            correct_predictions += (predicted == labels).sum().item()
            total_predictions += labels.size(0)

            # Calcular precisión por clase
            for i in range(labels.size(0)):
                label = labels[i].item()
                total_per_class[label] += 1
                if predicted[i] == label:
                    correct_per_class[label] += 1

    # Calcular métricas
    avg_loss = running_loss / len(valid_loader)
    accuracy = 100 * correct_predictions / total_predictions

    # Calcular accuracy por clase
    class_accuracies = {
        class_names[i]: (
            (100 * correct_per_class[i] / total_per_class[i]).item()
            if total_per_class[i] > 0
            else 0.0
        )
        for i in range(num_classes)
    }

    return avg_loss, accuracy, class_accuracies


def build_model(model_name, num_classes):
    """
    Construye y configura un modelo basado en la elección del usuario.

    Parámetros:
    - model_name: Nombre del modelo a construir
    - num_classes: Número de clases para la capa final

    Retorna:
    - model: Modelo configurado
    """
    model_name = model_name.lower().strip()

    if model_name == "convnext_large" or "convnext_large" in model_name:
        # Cargar ConvNeXt Large pre-entrenado
        model = torchvision.models.convnext_large(
            weights=torchvision.models.ConvNeXt_Large_Weights.DEFAULT
        )
        # Reemplazar la capa final (en la posición [2] de la secuencia classifier)
        model.classifier[2] = nn.Linear(model.classifier[2].in_features, num_classes)
        logging.info("ConvNeXt Large model selected")

    elif model_name == "efficient" or "efficientnet_b0" in model_name:
        # Cargar EfficientNet B0 pre-entrenado
        model = torchvision.models.efficientnet_b0(
            weights=torchvision.models.EfficientNet_B0_Weights.DEFAULT
        )
        # Reemplazar la última capa (índice [1]) de la secuencia classifier
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
        logging.info("EfficientNet-B0 model selected")

    elif (
        model_name == "efficientnet_b7"
        or "efficientnet_b7" in model_name
        or "efficientnet_rank_7" in model_name
    ):
        # Cargar EfficientNet B7 pre-entrenado
        model = torchvision.models.efficientnet_b7(
            weights=torchvision.models.EfficientNet_B7_Weights.DEFAULT
        )
        # Reemplazar la última capa (índice [1]) de la secuencia classifier
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
        logging.info("EfficientNet-B7 model selected")

    else:
        raise ValueError(
            f"Modelo no soportado: {model_name}. Opciones disponibles: convnext_large, efficientnet_b0, efficientnet_b7"
        )

    return model


def load_model(model_name, model_path, num_classes, device):
    """
    Carga un modelo pre-entrenado.

    Parámetros:
    - model_name: Nombre del modelo a cargar
    - model_path: Ruta al archivo con los pesos del modelo
    - num_classes: Número de clases para la capa de salida
    - device: Dispositivo donde se cargará el modelo

    Retorna:
    - El modelo cargado y configurado
    """
    logging.info(f"Cargando el modelo: {model_name}")

    try:
        # Construir el modelo base usando la función build_model
        model = build_model(model_name, num_classes)

        # Cargar los pesos del modelo entrenado
        state_dict = torch.load(model_path, map_location=device)

        # Intentar cargar con strict=True primero, si falla, probar con strict=False
        try:
            model.load_state_dict(state_dict)
            logging.info(f"Pesos cargados correctamente para {model_name}")
        except Exception as e:
            logging.warning(
                f"No se pudieron cargar los pesos con strict=True. Error: {e}"
            )
            logging.warning("Intentando cargar con strict=False...")
            model.load_state_dict(state_dict, strict=False)
            logging.info(f"Pesos cargados con strict=False para {model_name}")

        # Mover el modelo al dispositivo seleccionado
        model = model.to(device)
        model.eval()

        return model

    except Exception as e:
        logging.error(f"Error al cargar el modelo {model_name}: {e}")
        return None


def plot_class_accuracies(class_names, class_accuracies, model_name, output_dir=None):
    """
    Genera un gráfico de barras mostrando la precisión por clase.

    Parámetros:
    - class_names: Lista de nombres de clases
    - class_accuracies: Diccionario con la precisión de cada clase
    - model_name: Nombre del modelo para el título del gráfico
    - output_dir: Directorio donde guardar el gráfico (opcional)
    """
    accuracies = [class_accuracies[class_name] for class_name in class_names]

    plt.figure(figsize=(12, 7))
    bars = plt.bar(class_names, accuracies, color="skyblue")

    # Añadir valores en las barras
    for bar in bars:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + 1,
            f"{height:.1f}%",
            ha="center",
            va="bottom",
            rotation=0,
        )

    plt.xlabel("Clases", fontsize=12)
    plt.ylabel("Precisión (%)", fontsize=12)
    plt.title(f"Precisión por Clase - {model_name}", fontsize=14)
    plt.xticks(rotation=45, ha="right")
    plt.ylim(0, 105)  # Dejar espacio para los valores
    plt.tight_layout()

    # Guardar el gráfico si se especifica un directorio
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(os.path.join(output_dir, f"{model_name}_class_accuracy.png"))
        logging.info(
            f"Gráfico guardado en {output_dir}/{model_name}_class_accuracy.png"
        )

    plt.show()


def show_predictions(model, dataloader, class_names, device, num_images=5):
    """
    Muestra algunas imágenes de ejemplo con sus predicciones.

    Parámetros:
    - model: Modelo a utilizar para las predicciones
    - dataloader: DataLoader con las imágenes
    - class_names: Lista de nombres de clases
    - device: Dispositivo donde se realizarán las predicciones
    - num_images: Número de imágenes a mostrar
    """
    model.eval()
    # Obtener un lote de imágenes
    images, labels = next(iter(dataloader))

    # Realizar predicciones
    with torch.no_grad():
        images = images.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)

    # Normalización inversa para visualización
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    # Mostrar las imágenes
    fig, axes = plt.subplots(1, min(num_images, len(images)), figsize=(15, 5))
    if num_images == 1:
        axes = [axes]  # Convertir a lista para iterar cuando solo hay un subplot

    for i in range(min(num_images, len(images))):
        # Desnormalizar la imagen
        img = images[i].cpu().detach()
        img = img * std + mean
        img = torch.clamp(img, 0, 1)

        # Mostrar imagen
        axes[i].imshow(img.permute(1, 2, 0))

        # Establecer título con la predicción y etiqueta real
        pred_class = class_names[predicted[i].item()]
        true_class = class_names[labels[i].item()]

        if pred_class == true_class:
            title_color = "green"
        else:
            title_color = "red"

        axes[i].set_title(f"Pred: {pred_class}\nReal: {true_class}", color=title_color)
        axes[i].axis("off")

    plt.tight_layout()
    plt.show()


def main():
    """Función principal que ejecuta la evaluación de los modelos."""

    # Configurar argumentos de línea de comandos
    parser = argparse.ArgumentParser(description="Evaluación de modelos CNN")
    parser.add_argument(
        "--data_dir",
        type=str,
        default="../dataset/validation",
        help="Directorio del dataset de validación",
    )
    parser.add_argument(
        "--model_dir",
        type=str,
        default="./models",
        help="Directorio donde se encuentran los modelos",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="../images/results",
        help="Directorio donde se guardarán los resultados",
    )
    parser.add_argument(
        "--batch_size", type=int, default=32, help="Tamaño del lote para evaluación"
    )
    parser.add_argument(
        "--show_examples", action="store_true", help="Mostrar ejemplos de predicciones"
    )

    args = parser.parse_args()

    # Crear directorio de salida si no existe
    os.makedirs(args.output_dir, exist_ok=True)

    # Obtener el dispositivo
    device = get_device()

    # Cargar datos de validación
    val_loader, class_names = load_data(args.data_dir, batch_size=args.batch_size)
    num_classes = len(class_names)

    # Definir la función de pérdida
    criterion = nn.CrossEntropyLoss()

    # Buscar archivos de modelos
    model_files = [
        f for f in os.listdir(args.model_dir) if f.endswith(".pt") or f.endswith(".pth")
    ]

    if not model_files:
        logging.error(f"No se encontraron archivos de modelo en {args.model_dir}")
        return

    # Evaluar cada modelo
    results = {}
    for model_file in model_files:
        model_name = model_file.split(".")[0]  # Nombre del modelo sin extensión
        model_path = os.path.join(args.model_dir, model_file)

        logging.info(f"\n{'='*50}")
        logging.info(f"Evaluando el modelo: {model_name}")

        # Cargar el modelo
        model = load_model(model_name, model_path, num_classes, device)

        if model is None:
            logging.error(
                f"No se pudo cargar el modelo {model_name}. Continuando con el siguiente."
            )
            continue

        # Evaluar el modelo
        avg_loss, accuracy, class_accuracies = evaluate(
            model, val_loader, criterion, class_names, device
        )

        # Guardar resultados
        results[model_name] = {
            "loss": avg_loss,
            "accuracy": accuracy,
            "class_accuracies": class_accuracies,
        }

        # Mostrar los resultados
        logging.info(f"Loss promedio: {avg_loss:.4f}")
        logging.info(f"Precisión global: {accuracy:.2f}%")
        logging.info("Precisión por clase:")
        for clase, acc in class_accuracies.items():
            logging.info(f"{clase}: {acc:.2f}%")

        # Visualizar precisión por clase
        plot_class_accuracies(
            class_names, class_accuracies, model_name, args.output_dir
        )

        # Mostrar ejemplos de predicciones si se solicita
        if args.show_examples:
            show_predictions(model, val_loader, class_names, device)

    # Comparar todos los modelos
    if len(results) > 1:
        logging.info("\n" + "=" * 80)
        logging.info("COMPARACIÓN DE MODELOS")
        logging.info("=" * 80)

        # Ordenar modelos por precisión global
        sorted_models = sorted(
            results.items(), key=lambda x: x[1]["accuracy"], reverse=True
        )

        for i, (model_name, metrics) in enumerate(sorted_models):
            logging.info(
                f"{i+1}. {model_name}: Precisión = {metrics['accuracy']:.2f}%, Loss = {metrics['loss']:.4f}"
            )

        # Graficar comparación de modelos
        plt.figure(figsize=(12, 6))
        model_names = [name for name, _ in sorted_models]
        accuracies = [metrics["accuracy"] for _, metrics in sorted_models]

        bars = plt.bar(model_names, accuracies, color="lightblue")
        for bar in bars:
            height = bar.get_height()
            plt.text(
                bar.get_x() + bar.get_width() / 2.0,
                height + 1,
                f"{height:.1f}%",
                ha="center",
                va="bottom",
            )

        plt.xlabel("Modelos")
        plt.ylabel("Precisión Global (%)")
        plt.title("Comparación de Precisión Global entre Modelos")
        plt.xticks(rotation=45, ha="right")
        plt.ylim(0, 105)
        plt.tight_layout()

        # Guardar gráfico de comparación
        plt.savefig(os.path.join(args.output_dir, "model_comparison.png"))
        logging.info(
            f"Gráfico de comparación guardado en {args.output_dir}/model_comparison.png"
        )

        plt.show()


if __name__ == "__main__":
    main()
