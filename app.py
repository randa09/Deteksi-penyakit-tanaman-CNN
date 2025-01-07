import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import urllib.request
from io import BytesIO

# Load the trained model
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model('model_penyakit.h5')
    return model

model = load_model()

# Define the classes (replace with your actual class names)
CLASS_NAMES = ['Healthy', 'Powdery', 'Rust']

# Define the input size used by the model
IMG_SIZE = 128

# Function to preprocess image
def preprocess_image(image):
    try:
        # Resize to match model input size
        image = image.resize((IMG_SIZE, IMG_SIZE))  
        image = np.array(image).astype('float32') / 255.0  # Normalize the image
        if image.shape[-1] != 3:  # Ensure RGB format
            raise ValueError("Image must have 3 channels (RGB).")
        image = np.expand_dims(image, axis=0)  # Add batch dimension
        return image
    except Exception as e:
        st.error(f"Error preprocessing image: {e}")
        return None

# Function to predict the class
def predict(image):
    try:
        preprocessed_image = preprocess_image(image)
        if preprocessed_image is None:
            return None, None
        
        # Predict using the loaded model
        predictions = model.predict(preprocessed_image)
        predicted_class_idx = np.argmax(predictions, axis=1)[0]
        predicted_class = CLASS_NAMES[predicted_class_idx]
        confidence = predictions[0][predicted_class_idx]
        return predicted_class, confidence
    except Exception as e:
        st.error(f"Error during prediction: {e}")
        return None, None

# Streamlit app
def main():
    st.title("Plant Disease Detection")
    st.write("Upload an image or provide a URL to detect the plant disease.")

    # File uploader
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

    # URL input
    image_url = st.text_input("Or enter an image URL")

    image = None
    if uploaded_file is not None:
        try:
            image = Image.open(uploaded_file).convert("RGB")  # Ensure RGB format
            st.image(image, caption="Uploaded Image", use_column_width=True)
        except Exception as e:
            st.error(f"Error loading uploaded image: {e}")
    elif image_url:
        try:
            # Use urllib to open the image from the URL
            image_data = urllib.request.urlopen(image_url).read()
            image = Image.open(BytesIO(image_data)).convert("RGB")  # Ensure RGB format
            st.image(image, caption="Image from URL", use_column_width=True)
        except Exception as e:
            st.error(f"Error loading image from URL: {e}")

    # Predict button
    if image is not None:
        if st.button("Predict"):
            with st.spinner("Detecting..."):
                predicted_class, confidence = predict(image)
                if predicted_class and confidence is not None:
                    st.success(f"Prediction: {predicted_class}")
                    st.info(f"Confidence: {confidence * 100:.2f}%")
                else:
                    st.error("Prediction failed. Please check the input image or model.")

if __name__ == "__main__":
    main()
