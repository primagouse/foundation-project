import streamlit as st
import numpy as np
import torch
import torch.nn as nn
from PIL import Image, ImageOps
from streamlit_drawable_canvas import st_canvas
from datetime import datetime

# Import model architecture
from mnist_neural_net import CNN

# Getting the current date and time
timestamp = datetime.now()

# Load the trained model
@st.cache_resource
def load_model():
    model = CNN()
    model.load_state_dict(torch.load('mnist_cnn.pth', map_location=torch.device('cpu')))
    model.eval()
    return model

# Preprocess drawn image for the model
def preprocess_image(canvas_image):
    # Convert to PIL Image and grayscale
    img = Image.fromarray(canvas_image.astype('uint8'), 'RGBA')
    img = ImageOps.grayscale(img)
    
    # Resize to 28x28 (MNIST size)
    img = img.resize((28, 28))
    
    # Invert colors (MNIST has white digits on black background)
    img_array = np.array(img)
    img_array = 255 - img_array
    
    # Normalize and add batch/channel dimensions
    img_tensor = torch.from_numpy(img_array).float() / 255.0
    img_tensor = img_tensor.unsqueeze(0).unsqueeze(0)  # Shape: [1, 1, 28, 28]
    
    return img_tensor, img_array

# Streamlit UI
st.set_page_config(page_title="MNIST Digit Classifier", layout="wide")
st.title("MNIST Digit Classifier")
st.markdown("""
    Draw a digit (0-9) in the canvas below and click **Predict** to see the classification.
    The model was trained on the MNIST dataset using PyTorch.
    Press on the bin icon to draw a new image.
""")

# Create two columns
col1, col2 = st.columns([2, 1])

with col1:
    # Drawing canvas
    st.subheader("Drawing Canvas")
    canvas_result = st_canvas(
        fill_color="rgba(0, 0, 0, 1)",  # Black background
        stroke_width=18,
        stroke_color="rgba(255, 255, 255, 1)",  # White stroke
        background_color="#000000",
        height=300,
        width=300,
        drawing_mode="freedraw",
        key="canvas",
        update_streamlit=True,
        display_toolbar=True,  # This will show the built-in toolbar with clear button
    )

with col2:
    # Prediction section
    st.subheader("Prediction")
    
    if st.button('Predict', use_container_width=True):
        if canvas_result.json_data is not None and len(canvas_result.json_data["objects"]) > 0:
            # Preprocess and predict
            img_tensor, processed_img = preprocess_image(canvas_result.image_data)
            
            with torch.no_grad():
                outputs = load_model()(img_tensor)
                probs = torch.nn.functional.softmax(outputs, dim=1)[0] * 100
                _, predicted = torch.max(outputs.data, 1)
            
            # Display results
            st.success(f"**Prediction:** {predicted.item()}")
            true_label = st.text_input("True Value?")

            # Show confidence percentages
            st.write("**Confidence:**")
            for i, prob in enumerate(probs):
                st.progress(int(prob), text=f"{i}: {prob:.1f}%")
            
            # Show processed image
            st.image(processed_img, caption='Processed Image (28x28)', width=150)

            #
        else:
            st.warning("Please draw a digit first!")

#python -m streamlit run app.py