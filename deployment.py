import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import cv2
from ultralytics import YOLO

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
        color: #FF69B4;
        font-size: 36px;
        text-align: center;
        margin-bottom: 30px;
    }
    .upload-box {
        border: 2px dashed #FF69B4;
        border-radius: 10px;
        padding: 20px;
        text-align: center;
        margin-bottom: 30px;
    }
    .classify-button {
        display: block;
        width: 100%;
        padding: 2px;
        margin-top: 2px;
        border: none;
        border-radius: 5px;
        color: white;
        background-color: #FF69B4;
        font-size: 18px;
        cursor: pointer;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Streamlit app code
st.sidebar.title('Breast Cancer Prediction')
uploaded_file = st.sidebar.file_uploader("Upload an ultrasound image", type=['png', 'jpg', 'jpeg'], accept_multiple_files=False)

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    # Desired width and height
    desired_width = 650
    desired_height = 400  # Adjust this value to your desired height

    # Calculate aspect ratio
    aspect_ratio = image.width / image.height
    new_width = int(aspect_ratio * desired_height)
    # Resize the image to desired dimensions
    resized_image = image.resize((new_width, desired_height))
    st.image(image, caption='Uploaded Image', width=desired_width , use_column_width=False, output_format='JPEG')

    # Adding a border around the uploaded image
    # st.sidebar.markdown("<div class='upload-box'><p>Uploaded Image</p></div>", unsafe_allow_html=True)

    if st.sidebar.button('Classify', key='classify_button'):
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
        
        st.sidebar.write(f"Predicted Class: {predicted_label}")
        st.sidebar.write(f"Confidence: {confidence:.2f}%")
        
        # If malignant tumor is detected
        if predicted_class == 1:
            # Load the image
            image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            # Initialize YOLO model
            model_main = YOLO(r"best.pt")
            # Run inference on the loaded image
            results = model_main(image_cv, save=False, conf=0.2)
            # Plot the results
            res_plotted = results[0].plot()
            # Convert the plotted image to BGR format (OpenCV uses BGR)
            plotted_bgr = cv2.cvtColor(res_plotted, cv2.COLOR_RGB2BGR)
            # Replace the original image with the detected image
            image = Image.fromarray(plotted_bgr)
            # Display the image
            st.image(image, caption='Detected Malignant Tumor', width=desired_width, use_column_width=False)
