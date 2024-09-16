import logging
import os

import cv2
import numpy as np
import torch
from flask import Flask, render_template, request, send_file
from segment_anything import SamPredictor, sam_model_registry
from werkzeug.utils import secure_filename

# Flask app initialization
app = Flask(__name__)

# Configurations
app.config['UPLOAD_FOLDER'] = r'C:\Users\anany\Desktop\house-ass\uploads'
app.config['PROCESSED_FOLDER'] = r'C:\Users\anany\Desktop\house-ass\processed'
app.config['SAM_CHECKPOINT'] = r'C:\Users\anany\Desktop\house-ass\sam_vit_l_0b3195.pth'

# Ensure folders exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['PROCESSED_FOLDER'], exist_ok=True)

# Enable logging for debugging
logging.basicConfig(level=logging.INFO)

# Initialize SAM model
def initialize_sam_model(checkpoint_path):
    logging.info("Initializing SAM model...")
    sam_model = sam_model_registry["vit_l"](checkpoint=checkpoint_path)
    sam_model.eval()  # Set the model to evaluation mode
    return SamPredictor(sam_model)

# Route for the main page
@app.route('/')
def index():
    return render_template('index03.html')

# Route for processing the images
@app.route('/upload', methods=['POST'])
def upload_images():
    if 'room_image' not in request.files or 'replacement_image' not in request.files:
        return "Please upload both the room image and the replacement image.", 400

    # Get the uploaded files
    room_image_file = request.files['room_image']
    replacement_image_file = request.files['replacement_image']

    # Get the chosen option (wall or floor)
    option = request.form.get('option')

    if not option:
        return "No option selected. Please choose whether to change the wall or flooring.", 400

    # Save the uploaded files
    room_image_path = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(room_image_file.filename))
    replacement_image_path = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(replacement_image_file.filename))
    room_image_file.save(room_image_path)
    replacement_image_file.save(replacement_image_path)

    # Read the images using OpenCV
    room_img = cv2.imread(room_image_path)
    replacement_img = cv2.imread(replacement_image_path)

    if room_img is None or replacement_img is None:
        return "Error loading one or both images.", 400

    # Initialize SAM model
    predictor = initialize_sam_model(app.config['SAM_CHECKPOINT'])
    predictor.set_image(room_img)

    # Coordinates for segmentation (adjusted for finer control)
    if option == 'wall':
        point_coords = np.array([[100, 100], [200, 100], [300, 100]])  # Refined points
        point_labels = np.array([1, 1, 1])
        masks, _, _ = predictor.predict(point_coords=point_coords, point_labels=point_labels, multimask_output=True)

        if masks is not None and len(masks) > 0:
            # Choose the best mask, this is just taking the first one, but you could choose based on other criteria
            mask = masks[0]

            # Optional: Apply morphological erosion to refine the mask
            kernel = np.ones((5, 5), np.uint8)
            mask = cv2.erode(mask.astype(np.uint8), kernel, iterations=1)

            replacement_resized = cv2.resize(replacement_img, (mask.shape[1], mask.shape[0]))
            room_img[mask > 0] = replacement_resized[mask > 0]

    elif option == 'Blinds':
        point_coords = np.array([[100, 100], [200, 100], [300, 100]])  # Refined points
        point_labels = np.array([1, 1, 1])
        masks, _, _ = predictor.predict(point_coords=point_coords, point_labels=point_labels, multimask_output=True)

        if masks is not None and len(masks) > 0:
            # Choose the best mask, this is just taking the first one, but you could choose based on other criteria
            mask = masks[0]

            # Optional: Apply morphological erosion to refine the mask
            kernel = np.ones((5, 5), np.uint8)
            mask = cv2.erode(mask.astype(np.uint8), kernel, iterations=1)

            replacement_resized = cv2.resize(replacement_img, (mask.shape[1], mask.shape[0]))
            room_img[mask > 0] = replacement_resized[mask > 0]

    else:
        return "Invalid option selected. Please choose 'wall' or 'floor'.", 400

    # Save the processed image
    output_filename = f'room_with_{option}.jpg'
    output_filepath = os.path.join(app.config['PROCESSED_FOLDER'], output_filename)
    cv2.imwrite(output_filepath, room_img)

    # Return the processed image
    return send_file(output_filepath, mimetype='image/jpeg')

if __name__ == "__main__":
    app.run(debug=True)
