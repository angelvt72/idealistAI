import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision
import os
import logging
import sys
import wandb

# Log in to wandb
wandb.login()

# Detailed logging configuration
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("training_log.txt")
    ]
)

# Function to set up the device for training
def get_device():
    """
    Determines the optimal device:
      - NVIDIA GPU if available (CUDA)
      - Apple Silicon GPU if available (MPS)
      - CPU otherwise
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
        logging.info("Using NVIDIA GPU (CUDA)")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
        logging.info("Using Apple Silicon GPU (MPS)")
    else:
        device = torch.device("cpu")
        logging.info("Using CPU")
    return device

# Function to determine the number of workers for DataLoader
def get_num_workers(default_max=8):
    """
    Returns the optimal number of workers, limited by the minimum of available CPU cores and default_max.
    """
    cpu_count = os.cpu_count() or default_max
    num_workers = min(cpu_count, default_max)
    logging.info(f"Configured number of workers: {num_workers}")
    return num_workers

# Function to load training and validation data
def load_data(train_dir, valid_dir, batch_size=8, img_size=224):
    # Data transformations
    transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize((img_size, img_size)),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(
            mean=[0.485, 0.456, 0.406], 
            std=[0.229, 0.224, 0.225]
        )
    ])
    
    if not os.path.exists(train_dir):
        raise ValueError(f"Training directory does not exist: {train_dir}")
    if not os.path.exists(valid_dir):
        raise ValueError(f"Validation directory does not exist: {valid_dir}")
    
    train_dataset = torchvision.datasets.ImageFolder(train_dir, transform=transform)
    valid_dataset = torchvision.datasets.ImageFolder(valid_dir, transform=transform)
    
    num_workers = get_num_workers()
    pin_memory = True if torch.cuda.is_available() else False
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    valid_loader = DataLoader(
        valid_dataset, 
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    return train_loader, valid_loader, len(train_dataset.classes)

# Function to evaluate the model on the validation set
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

# Helper function to build and configure the model based on the user's choice.
def build_model(model_name, num_classes):
    model_name = model_name.lower().strip()
    if model_name == "convnext_large":

        # Load pre-trained ConvNeXt Large
        model = torchvision.models.convnext_large(weights=torchvision.models.ConvNeXt_Large_Weights.DEFAULT)

        # Replace the final layer (at position [2] of the classifier sequence) to match the number of classes
        model.classifier[2] = nn.Linear(model.classifier[2].in_features, num_classes)
        logging.info("ConvNeXt Large model selected")
        
    elif model_name == "efficientnet_b0":

        # Load pre-trained EfficientNet-B0
        model = torchvision.models.efficientnet_b0(weights=torchvision.models.EfficientNet_B0_Weights.DEFAULT)

        # Replace the last layer (index [1]) of the classifier sequence
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
        logging.info("EfficientNet-B0 model selected")
    else:
        raise ValueError("Unsupported model. Available options: convnext_large, efficientnet_b0.")
    return model

# Function to train the model
def train(rank, world_size, model_choice, learning_rate, epochs, batch_size):
    try:
        logging.info(f"Starting training on rank {rank}")
        
        train_dir = "./models_generator/dataset/training"
        valid_dir = "./models_generator/dataset/validation"
        
        # Load data with user-specified batch size
        train_loader, valid_loader, num_classes = load_data(
            train_dir, valid_dir, 
            batch_size=batch_size,  
            img_size=224
        )
        
        # Setup W&B
        wandb.init(
            project="IdealistAI",
            config={
                "model": f"{model_choice}_{epochs}_epochs_{learning_rate}_lr",
                "epochs": epochs,
                "batch_size": train_loader.batch_size
            },
            name=f"{model_choice}_{epochs}_epochs_{learning_rate}_lr"
        )
        
        logging.info(f"Number of classes: {num_classes}")
        logging.info(f"Training set size: {len(train_loader.dataset)}")
        
        device = get_device()
        
        # Dynamically build model based on the user's choice
        model = build_model(model_choice, num_classes)
        
        # If there is more than one NVIDIA GPU, use DataParallel
        if torch.cuda.device_count() > 1 and device.type == "cuda":
            logging.info(f"Using {torch.cuda.device_count()} GPUs with DataParallel")
            model = nn.DataParallel(model)
        
        model = model.to(device)
        
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
        criterion = nn.CrossEntropyLoss()
        
        logging.info("Starting training loop")
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
                    logging.info(f"Epoch {epoch+1}, Batch {batch_idx}, Loss: {loss.item():.4f}, Train Accuracy: {accuracy:.2f}%")

            val_loss, val_accuracy = evaluate(model, valid_loader, criterion, device)
            logging.info(f"Epoch {epoch+1}, Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.2f}%")
            
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
        logging.info(f"Model saved successfully at {model_path}")
        wandb.finish()
        
    except Exception as e:
        logging.error(f"Error during training: {e}", exc_info=True)

# Main function to prompt user for input and start training
def main():
    # Prompt the user to enter the CNN model to train via the terminal
    model_choice = input("Enter the CNN model name to train (e.g.: convnext_large or efficientnet_b0): ").strip()
    
    # Ask the user for learning rate, number of epochs and batch size
    learning_rate = float(input("Enter the learning rate (e.g.: 0.0005): ").strip())
    epochs = int(input("Enter the number of epochs (e.g.: 5): ").strip())
    batch_size = int(input("Enter the batch size (e.g.: 8): ").strip())

    world_size = 1  # In this example a single device is used
    train(rank=0, world_size=world_size, model_choice=model_choice, learning_rate=learning_rate, epochs=epochs, batch_size=batch_size)

if __name__ == '__main__':
    main()
