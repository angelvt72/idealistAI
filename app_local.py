import torch
import torchvision.transforms as transforms
from PIL import Image
import os
import torchvision.models as models
import torch.nn as nn
import streamlit as st

from models_generator.prediction_pipeline_modelos import (
    prediction_process,
    process_image,
)

# Streamlit page configuration
st.set_page_config(page_title="Image Classification", layout="centered")

# Application header
st.title("🔍 AI Image Classification")
st.write(
    "Upload an image and the model will tell you what it is along with its probability."
)

# Model selection
model_name = st.selectbox(
    "Choose the model",
    [
        "convnext_large_5_epochs_0.001_lr",
        "convnext_large_5_epochs_0.0005_lr",
        "convnext_large_5epochs_0.0001_lr",
        "efficientnet_b0_5epochs_0.0005_lr",
    ],
)


# Upload image
uploaded_file = st.file_uploader(
    "Choose an image", type=["jpg", "jpeg", "png", "bmp", "tif", "tiff", "webp"]
)

# If an image is uploaded, process it and make a prediction
if uploaded_file is not None:

    # Open and display the image
    image = Image.open(uploaded_file)
    st.image(image, caption="Loaded Image", use_column_width=True)

    # Process the image
    image_tensor = process_image(image)

    # Check if the image was processed correctly
    if image_tensor is not None:

        # Get predictions
        results = prediction_process(image_tensor, model_name)

        # Check if there was an error loading the model
        if "error" in results:
            st.error(f"❌ Error loading model: {results['error']}")
        else:
            # Extract the best prediction
            best_class = max(results, key=results.get)
            best_prob = float(results[best_class]) * 100  # Convert to percentage

            # Display the best prediction
            st.subheader(f"📌 Main prediction: **{best_class}** ({best_prob:.2f}%)")

            # Display all predictions with progress bars
            st.write("### 📊 Probabilities:")
            for class_name, prob in results.items():
                prob_float = float(prob)
                st.write(f"**{class_name}**: {prob_float * 100:.2f}%")
                st.progress(prob_float)
    else:
        st.error("❌ Could not process the image. Try another file.")
