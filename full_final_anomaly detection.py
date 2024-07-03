import logging
from flask import Flask, render_template, request, redirect, url_for, flash
from werkzeug.utils import secure_filename
import os
import shutil
from datetime import datetime
from anomaly_detection import train_model, detect_anomalies
import uuid
import base64
import requests
import json

# Initialize Flask app
app = Flask(__name__)

# Configuration for file uploads
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv', 'jpg', 'png'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.secret_key = b'_5#y2L"F4Q8z\n\xec]/'

# Initialize logger
logging.basicConfig(level=logging.DEBUG)

# Define application version globally
app_version = '1.0'

# Function to check if file extension is allowed
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Function to encode image as base64
def encode_image(image_path):
    try:
        with open(image_path, "rb") as img_file:
            encoded_string = base64.b64encode(img_file.read()).decode('utf-8')
        return encoded_string
    except FileNotFoundError:
        logging.error(f"File not found: {image_path}")
        return None

# Function to clear a folder
def clear_folder(folder_path):
    if os.path.exists(folder_path):
        for filename in os.listdir(folder_path):
            file_path = os.path.join(folder_path, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                logging.error(f'Failed to delete {file_path}. Reason: {e}')

# Route for index page
@app.route('/')
def index():
    return render_template('index.html')

# Route for file upload
@app.route('/upload', methods=['POST'])
def upload_file():
    # Generate a unique identifier for this run
    run_guid = str(uuid.uuid4())

    # Define folder paths
    train_upload_folder = os.path.join(app.config['UPLOAD_FOLDER'], 'train_upload')
    test_upload_folder = os.path.join(app.config['UPLOAD_FOLDER'], 'test_upload')

    # Clear the previous uploads
    clear_folder(train_upload_folder)
    clear_folder(test_upload_folder)

    # Create a unique anomaly folder
    unique_anomaly_folder = os.path.join(app.config['UPLOAD_FOLDER'], 'anomalies', run_guid)
    os.makedirs(unique_anomaly_folder, exist_ok=True)

    # Retrieve the uploaded files
    training_file = request.files.get('training_file')
    testing_file = request.files.get('testing_file')

    # Check if both training and testing files are uploaded
    if not training_file or not testing_file:
        flash('Please upload both training and testing videos.')
        return redirect(url_for('index'))

    # Handle the uploaded training video
    training_filename = secure_filename(training_file.filename)
    training_file_path = os.path.join(train_upload_folder, training_filename)
    training_file.save(training_file_path)

    # Handle the uploaded testing video
    testing_filename = secure_filename(testing_file.filename)
    testing_file_path = os.path.join(test_upload_folder, testing_filename)
    testing_file.save(testing_file_path)

    # Define input_epoch from request or set default value
    input_epoch_value = request.form.get('input_epoch', None)
    if input_epoch_value is None:
        # Handle the case when input_epoch is not provided
        flash('Number of epochs not provided. Using default value.')
        input_epoch = 10  # Set a default value
    else:
        try:
            input_epoch = int(input_epoch_value)
        except ValueError:
            flash('Invalid value for number of epochs. Using default value.')
            input_epoch = 10  # Set a default value in case of invalid input

    # Call anomaly detection functions for training video
    train_video_frames, training_duration, training_video_duration, training_load_and_preprocess_dur, _ = train_model(
        run_guid, input_epoch, training_file_path, unique_anomaly_folder, app_version, train_upload_folder, training_file_path)

    # Call anomaly detection functions for testing video
    unique_folder_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    anomaly_frames = detect_anomalies(testing_file_path, train_video_frames, training_duration, training_video_duration,
                                      training_load_and_preprocess_dur, input_epoch, unique_anomaly_folder, app_version,
                                      test_upload_folder,
                                      unique_folder_timestamp)

    # Ensure anomaly_frames contains valid paths
    anomaly_frame_urls = []
    for frame in anomaly_frames:
        frame_path = os.path.join(unique_anomaly_folder, f"anomaly_frame_{frame}.png")
        logging.debug(f"Frame path: {frame_path}")
        encoded_image = encode_image(frame_path)
        if encoded_image:
            anomaly_frame_urls.append(encoded_image)
        else:
            logging.error(f"Failed to encode image: {frame_path}")

    flash('Anomaly detection completed successfully.')

    # Render the index.html template and pass the anomaly frame URLs to it
    return render_template('index.html', anomaly_frame_urls=anomaly_frame_urls)

if __name__ == "__main__":
    app.run(debug=True)
