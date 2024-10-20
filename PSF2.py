from flask import Flask, request, jsonify, render_template
import tensorflow as tf
from PIL import Image
import numpy as np
import os

# Initialize Flask app
app = Flask(__name__)

# Load the pre-trained model
model = tf.keras.models.load_model('image_classification_model.h5')

# Define a list of class names based on your training classes
class_names = ['class1', 'class2', 'class3', 'class4', 'class5', 'class6', 'class7', 'class8', 'class9', 'class10']

# Define the home route that renders the HTML form
@app.route('/')
def home():
    return render_template('index.html')

# Define the prediction route that accepts image uploads
@app.route('/predict', methods=['POST'])
def predict():
    # Check if an image file is in the request
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    
    # Open the image using PIL
    img = Image.open(file)
    
    # Preprocess the image to match the input shape the model expects
    img = img.resize((150, 150))  # Assuming the model input is (150, 150, 3)
    img = np.array(img) / 255.0  # Normalize pixel values
    img = np.expand_dims(img, axis=0)  # Add batch dimension

    # Make a prediction
    predictions = model.predict(img)
    predicted_class = np.argmax(predictions, axis=1)[0]
    predicted_label = class_names[predicted_class]

    # Return the prediction result
    return jsonify({'prediction': predicted_label})

if __name__ == '__main__':
    app.run(debug=True)
