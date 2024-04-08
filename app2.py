from flask import Flask, render_template, request, jsonify
from PIL import Image
import numpy as np
import cv2
import base64
import io
import tensorflow as tf
from ultralytics import YOLO

app = Flask(__name__)

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

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/screening', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Get uploaded file from the request
        uploaded_file = request.files['file']
        
        # Read image using PIL
        image = Image.open(uploaded_file)
        # Define the new size as 300x300 pixels
        new_size = (500, 400)

        # Resize the image
        image = image.resize(new_size)
        # Preprocess the image and make prediction
        processed_image = preprocess_image(image)
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
            # Convert the plotted image to base64 encoding
            img_str = cv2.imencode('.jpg', cv2.cvtColor(res_plotted, cv2.COLOR_RGB2BGR))[1].tostring()
            img_base64 = base64.b64encode(img_str).decode('utf-8')
        else:
            img_str = cv2.imencode('.jpg', cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR))[1].tostring()
            img_base64 = base64.b64encode(img_str).decode('utf-8')

        return jsonify({
            'predicted_label': predicted_label,
            'confidence': confidence,
            'image': img_base64
        })

    return render_template('index2.html')

if __name__ == '__main__':
    app.run(debug=True)



# @app.route('/risk')
# def risk():
#     return render_template('risk.html')

# @app.route('/prevention')
# def prevention():
#     return render_template('prevention.html')

if __name__ == '__main__':
    app.run(debug=True)

# @app.route('/', methods=['GET', 'POST'])
# def index():
#     if request.method == 'POST':
#         # Get uploaded file from the request
#         uploaded_file = request.files['file']
        
#         # Read image using PIL
#         image = Image.open(uploaded_file)
        
#         # Preprocess the image and make prediction
#         processed_image = preprocess_image(image)
#         prediction = predict(image)

#         predicted_class = np.argmax(prediction)
        
#         # Define class labels based on predicted classes
#         class_labels = {
#             0: 'Non-cancerous benign tumor',
#             1: 'Cancerous malignant tumor',
#             2: 'Normal'
#         }
        
#         # Get the predicted class label and confidence
#         predicted_label = class_labels.get(predicted_class)
#         confidence = np.max(prediction) * 100
        
#         # If malignant tumor is detected
#         if predicted_class == 1:
#             # Load the image
#             image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
#             # Initialize YOLO model
#             model_main = YOLO(r"best.pt")
#             # Run inference on the loaded image
#             results = model_main(image_cv, save=False, conf=0.2)
#             # Plot the results
#             res_plotted = results[0].plot()
#             # Convert the plotted image to base64 encoding
#             img_str = cv2.imencode('.jpg', cv2.cvtColor(res_plotted, cv2.COLOR_RGB2BGR))[1].tostring()
#             img_base64 = base64.b64encode(img_str).decode('utf-8')
#         else:
#             img_str = cv2.imencode('.jpg', cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR))[1].tostring()
#             img_base64 = base64.b64encode(img_str).decode('utf-8')

#         return jsonify({
#             'predicted_label': predicted_label,
#             'confidence': confidence,
#             'image': img_base64
#         })

#     return render_template('index2.html')

# if __name__ == '__main__':
#     app.run(debug=True)
