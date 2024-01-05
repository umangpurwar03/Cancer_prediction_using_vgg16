import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

# Load the saved model
model = tf.keras.models.load_model('best_model_1.h5')

# Function to preprocess and predict
def predict(image):
    # Preprocess the image
    processed_img = preprocess_image(image)
    
    # Make prediction
    prediction = model.predict(processed_img)
    
    return prediction

# Function to preprocess the image
def preprocess_image(image):
    # Resize the image to match model's input shape (150x150) and normalize pixel values
    img = image.resize((150, 150))
    img_array = np.array(img) / 255.0  # Normalize pixel values between 0 and 1
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array

# Custom CSS style for Streamlit app
st.markdown(
    """
    <style>
    .header {
        color: #33B5FF;
        font-size: 36px;
        text-align: center;
        margin-bottom: 30px;
    }
    .upload-box {
        border: 2px dashed #33B5FF;
        border-radius: 10px;
        padding: 20px;
        text-align: center;
        margin-bottom: 30px;
    }
    .classify-button {
        display: block;
        width: 100%;
        padding: 10px;
        margin-top: 20px;
        border: none;
        border-radius: 5px;
        color: white;
        background-color: #33B5FF;
        font-size: 18px;
        cursor: pointer;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Streamlit app code
st.title('Ultrasound Image Classifier')
st.markdown("<p class='header'>Upload an ultrasound image</p>", unsafe_allow_html=True)

uploaded_file = st.file_uploader("", type=['png', 'jpg', 'jpeg'], accept_multiple_files=False)

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True, output_format='JPEG')

    # Adding a border around the uploaded image
    st.markdown("<div class='upload-box'><p>Uploaded Image</p></div>", unsafe_allow_html=True)

    if st.button('Classify', key='classify_button'):
        prediction = predict(image)
        predicted_class = np.argmax(prediction)
        
        # Define class labels based on predicted classes
        class_labels = {
            0: 'Non-cancerous benign tumor',
            1: 'Cancerous malignant tumor',
            2: 'Normal'
        }
        
        # Get the predicted class label and confidence
        predicted_label = class_labels.get(predicted_class)
        confidence = np.max(prediction) * 100
        
        st.write(f"Predicted Class: {predicted_label}")
        st.write(f"Confidence: {confidence:.2f}%")
