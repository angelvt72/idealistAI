#!/usr/bin/env python
# coding: utf-8

# # **Deep Learning Lab: Transfer Learning**
# 
# <img src ="data:image/png;base64,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" width=300 px>
# 
# ***

# ## Import packages

# In[1]:


from cnn import CNN
import torchvision
from cnn import load_data
from cnn import load_model_weights
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import os
import json
import random
import wandb


# ---
# ## **Model and data loading**

# In[2]:


# Pytorch has many pre-trained models that can be used for transfer learning
classification_models = torchvision.models.list_models(module=torchvision.models)
print(classification_models)


# In[3]:


# Load data and model 
train_dir = './dataset/training'
valid_dir = './dataset/validation'

train_loader, valid_loader, num_classes = load_data(train_dir, 
                                                    valid_dir, 
                                                    batch_size=32, 
                                                    img_size=224) # ResNet50 requires 224x224 images
model = CNN(torchvision.models.resnet50(weights='DEFAULT'), num_classes)


# In[4]:


# Visualize a few images
def imshow(inp, title=None):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated

# Get a few random images
random_indices = np.random.choice(len(valid_loader.dataset), size=4, replace=False)
inputs = []
classes = []
for i in random_indices:
    inputs.append(valid_loader.dataset[i][0])
    classes.append(valid_loader.dataset[i][1])
out = torchvision.utils.make_grid(inputs)
classnames = train_loader.dataset.classes
imshow(out, title=[classnames[x] for x in classes])


# ---
# ## **Train**

# In[5]:


# Parameters
learning_rate = 1e-4
epochs = 2


# #### **Configuración W&B**

# In[ ]:


# Start a new wandb run to track this script.
run = wandb.init(
    # Set the wandb entity where your project will be logged (generally your team name).
    entity="jcriego-prsoria-angelvt",

    # Set the wandb project where this run will be logged.
    project="IdealistAI",

    #name="model_1",  # Aquí defines el nombre del run

    # Track hyperparameters and run metadata.
    config={
        "learning_rate": learning_rate,
        "architecture": "CNN",
        "dataset": "CIFAR-100",
        "epochs": epochs,
    },
)


# In[ ]:


# Train the model

# Watch the model (W&B)
#wandb.watch(model, log="all")

# Define the loss function and optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
criterion = nn.CrossEntropyLoss()

# Train the model
for epoch in range(epochs):

    # Fase de entrenamiento
    model.train()
    running_loss = 0.0
    running_corrects = 0
    total_train = 0

    # Iterate over training data
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # Statistics
        running_loss += loss.item() * inputs.size(0)
        _, preds = torch.max(outputs, 1)
        running_corrects += torch.sum(preds == labels.data)
        total_train += inputs.size(0)

    # Loss and accuracy
    epoch_train_loss = running_loss / total_train
    epoch_train_acc = running_corrects.double() / total_train

    # Registro de métricas en wandb
    run.log({
        "epoch": epoch + 1,
        "train_loss": epoch_train_loss,
        "train_accuracy": epoch_train_acc.item()
    })

    print(f"Epoch {epoch+1}/{epochs} - "
          f"Train Loss: {epoch_train_loss:.4f}, Train Acc: {epoch_train_acc:.4f} - ")

# Guarda el modelo entrenado
model.save_model('resnet50-1epoch')


# In[ ]:


# Train the model
'''
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
criterion = nn.CrossEntropyLoss()
history = model.train_model(train_loader, valid_loader, optimizer, criterion, epochs=1)

model.save_model('resnet50-1epoch')'
'''


# ---
# ## **Predict**

# #### Load model

# In[ ]:


# Load model
model_weights = load_model_weights('resnet50-1epoch')
my_trained_model = CNN(torchvision.models.resnet50(weights='DEFAULT'), num_classes)
my_trained_model.load_state_dict(model_weights)


# ---
# ## **Results**

# In[ ]:


predicted_labels = my_trained_model.predict(valid_loader)


# In[ ]:


# Get a few random images
random_indices = np.random.choice(len(valid_loader.dataset), size=4, replace=False)
inputs = []
classes = []
for i in random_indices:
    inputs.append(valid_loader.dataset[i][0])
    classes.append(predicted_labels[i])

out = torchvision.utils.make_grid(inputs)
classnames = train_loader.dataset.classes
imshow(out, title=[classnames[x] for x in classes])

