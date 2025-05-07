import streamlit as st
import numpy as np
import torch
import torch.nn as nn
import psycopg2
from psycopg2 import sql
import os
from PIL import Image, ImageOps
from streamlit_drawable_canvas import st_canvas
from datetime import datetime
from dotenv import load_dotenv

# Set page config first
st.set_page_config(page_title="MNIST Digit Classifier", layout="wide")

# Load environment variables
load_dotenv()

# Import model architecture
from mnist_neural_net import CNN

# Getting the current date and time
timestamp = datetime.now()

# Database configuration
DB_CONFIG = {
    'dbname': os.getenv('DB_NAME', 'mnist_predictions'),
    'user': os.getenv('DB_USER', 'postgres'),
    'password': os.getenv('DB_PASSWORD', 'postgres'),
    'host': os.getenv('DB_HOST', 'localhost'),
    'port': os.getenv('DB_PORT', '5432')
}

# Load the trained model
def load_model():
    try:
        model = CNN()
        model.load_state_dict(torch.load('mnist_cnn.pth', map_location=torch.device('cpu')))
        model.eval()
        return model
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

# Initialize the model
model = load_model()
if model is None:
    st.error("Failed to load the model. Please check if mnist_cnn.pth exists.")
    st.stop()

# Database connection
def get_db_connection():
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        return conn
    except Exception as e:
        st.error(f"Database connection failed: {str(e)}")
        return None

# Initialize database table
def init_db():
    conn = get_db_connection()
    if conn:
        try:
            with conn.cursor() as cur:
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS predictions (
                        id SERIAL PRIMARY KEY,
                        prediction_date TIMESTAMP NOT NULL,
                        prediction INTEGER NOT NULL,
                        true_label INTEGER NOT NULL
                    )
                """)
                conn.commit()
                st.success("Database initialized successfully")
        except Exception as e:
            st.error(f"Database initialization failed: {e}")
        finally:
            conn.close()

# Store prediction in database
def store_prediction(prediction, true_label):
    try:
        conn = get_db_connection()
        if conn:
            with conn.cursor() as cur:
                # First, let's check if the table exists
                cur.execute("""
                    SELECT EXISTS (
                        SELECT FROM information_schema.tables 
                        WHERE table_name = 'predictions'
                    );
                """)
                table_exists = cur.fetchone()[0]
                
                if not table_exists:
                    cur.execute("""
                        CREATE TABLE predictions (
                            id SERIAL PRIMARY KEY,
                            prediction_date TIMESTAMP NOT NULL,
                            prediction INTEGER NOT NULL,
                            true_label INTEGER NOT NULL
                        )
                    """)
                    conn.commit()
                
                # Insert the data
                cur.execute(
                    "INSERT INTO predictions (prediction_date, prediction, true_label) VALUES (%s, %s, %s) RETURNING id",
                    (datetime.now(), prediction, true_label)
                )
                inserted_id = cur.fetchone()[0]
                conn.commit()
                st.success("Prediction saved successfully!")
                
    except Exception as e:
        st.error(f"Failed to store prediction: {str(e)}")
    finally:
        if 'conn' in locals():
            conn.close()

# Initialize database on app start
init_db()

# Preprocess drawn image for the model
def preprocess_image(canvas_image):
    try:
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
    except Exception as e:
        st.error(f"Error preprocessing image: {e}")
        return None, None

# Streamlit UI
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
        display_toolbar=True,
    )

with col2:
    # Prediction section
    st.subheader("Prediction")
    
    # Initialize session state for prediction if not exists
    if 'current_prediction' not in st.session_state:
        st.session_state['current_prediction'] = None
    
    if st.button('Predict', use_container_width=True):
        if canvas_result.json_data is not None and len(canvas_result.json_data["objects"]) > 0:
            # Preprocess and predict
            img_tensor, processed_img = preprocess_image(canvas_result.image_data)
            
            if img_tensor is not None:
                with torch.no_grad():
                    outputs = model(img_tensor)
                    probs = torch.nn.functional.softmax(outputs, dim=1)[0] * 100
                    _, predicted = torch.max(outputs.data, 1)
                
                # Store prediction in session state
                st.session_state['current_prediction'] = predicted.item()
                
                # Display results
                st.success(f"**Prediction:** {predicted.item()}")
                
                # Show confidence percentages
                st.write("**Confidence:**")
                for i, prob in enumerate(probs):
                    st.progress(int(prob), text=f"{i}: {prob:.1f}%")
                
                # Show processed image
                st.image(processed_img, caption='Processed Image (28x28)', width=150)

    # Always show the input form if we have a prediction
    if st.session_state['current_prediction'] is not None:
        st.write("---")
        st.write("### Save Prediction")
        true_label = st.text_input("Enter the true value (0-9):")
        
        if st.button('Save to Database'):
            if true_label:
                try:
                    true_label = int(true_label)
                    if 0 <= true_label <= 9:
                        store_prediction(st.session_state['current_prediction'], true_label)
                        # Reset everything after successful save
                        st.session_state['current_prediction'] = None
                        st.rerun()  # This will refresh the page and clear the canvas
                    else:
                        st.error("Please enter a number between 0 and 9")
                except ValueError:
                    st.error("Please enter a valid number")
            else:
                st.error("Please enter a true label")
    else:
        st.warning("Please draw a digit and click Predict first!")

# Display recent predictions
st.write("---")
st.subheader("Recent Predictions")

try:
    conn = get_db_connection()
    if conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT prediction_date, prediction, true_label 
                FROM predictions 
                ORDER BY prediction_date DESC 
                LIMIT 10
            """)
            rows = cur.fetchall()
            
            if rows:
                # Create a table with the results
                st.write("Last 10 predictions:")
                for row in rows:
                    date_str = row[0].strftime("%Y-%m-%d %H:%M:%S")
                    st.write(f"Date: {date_str} | Predicted: {row[1]} | True Value: {row[2]}")
            else:
                st.info("No predictions in database yet")
    conn.close()
except Exception as e:
    st.error(f"Error fetching predictions: {str(e)}")

#python -m streamlit run app.py