from flask import Flask, request, render_template, redirect, url_for
from werkzeug.utils import secure_filename
import os
import cv2
import numpy as np
import tensorflow as tf

app = Flask(__name__)

# Load the trained CNN model
model = tf.keras.models.load_model('rice_leaf_disease_classifier_hsv.h5')

# Folder to store uploaded images (inside static for easier serving)
UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Create uploads folder if it doesn't exist
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Allowed extensions for uploads
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# Class labels (modify according to your model's classes)
CLASS_NAMES = ["Bacterial Blight", "Downy Mildew", "Blast", "Dead Heart", "Healthy"]

# Check if uploaded file is allowed
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Convert image to grayscale
def convert_to_grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Convert image to HSV
def convert_to_hsv(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# Apply segmentation by thresholding based on HSV values
def segment_disease_area(image_hsv):
    lower_bound = np.array([25, 40, 40])  # Lower HSV values
    upper_bound = np.array([85, 255, 255])  # Upper HSV values
    mask = cv2.inRange(image_hsv, lower_bound, upper_bound)
    segmented = cv2.bitwise_and(image_hsv, image_hsv, mask=mask)
    return segmented

# Draw a circle around the detected disease areas
def draw_circle_around_disease(image, mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        (x, y), radius = cv2.minEnclosingCircle(cnt)
        center = (int(x), int(y))
        radius = int(radius)
        if radius > 10:  # Only circle large enough areas
            cv2.circle(image, center, radius, (0, 0, 255), 2)  # Draw red circle around the disease
    return image

# Predict the disease class using the CNN model
def predict_disease(image):
    image_resized = cv2.resize(image, (256, 256))
    image_array = np.expand_dims(image_resized, axis=0)  # Add batch dimension
    image_array = image_array / 255.0  # Normalize the image
    predictions = model.predict(image_array)
    predicted_class = np.argmax(predictions)  # Get the index of the highest probability
    return CLASS_NAMES[predicted_class]  # Return the corresponding class label

# Home route to upload images
@app.route('/')
def home():
    return render_template('index.html')

# Route to process uploaded images
@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return redirect(request.url)
    
    file = request.files['file']
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Load the image using OpenCV
        image = cv2.imread(filepath)
        
        # Process images in different formats
        hsv_image = convert_to_hsv(image)
        grayscale_image = convert_to_grayscale(image)
        segmented_image = segment_disease_area(hsv_image)
        
        # Get the mask for the disease and circle it
        mask = cv2.inRange(hsv_image, np.array([25, 40, 40]), np.array([85, 255, 255]))
        disease_encircled_image = draw_circle_around_disease(image.copy(), mask)
        
        # Predict the disease class using the model
        predicted_disease = predict_disease(image)
        
        # Save the processed images in 'static/uploads'
        hsv_image_path = 'hsv_' + filename
        grayscale_image_path = 'grayscale_' + filename
        segmented_image_path = 'segmented_' + filename
        disease_encircled_image_path = 'disease_encircled_' + filename
        
        cv2.imwrite(os.path.join(app.config['UPLOAD_FOLDER'], hsv_image_path), hsv_image)
        cv2.imwrite(os.path.join(app.config['UPLOAD_FOLDER'], grayscale_image_path), grayscale_image)
        cv2.imwrite(os.path.join(app.config['UPLOAD_FOLDER'], segmented_image_path), segmented_image)
        cv2.imwrite(os.path.join(app.config['UPLOAD_FOLDER'], disease_encircled_image_path), disease_encircled_image)
        
        # Redirect to display results
        return render_template('results.html', 
                               original_image=filename, 
                               hsv_image=hsv_image_path, 
                               grayscale_image=grayscale_image_path, 
                               segmented_image=segmented_image_path, 
                               disease_encircled_image=disease_encircled_image_path,
                               disease=predicted_disease)  # Send the predicted disease class
    else:
        return redirect(request.url)

# Route to display processed images
@app.route('/results')
def display_results():
    return render_template('results.html')

if __name__ == '__main__':
    app.run(debug=True)
