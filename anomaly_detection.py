import os
import sys
from PIL import Image
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Reshape, Flatten, LSTM, Dense
from tensorflow.keras.models import Model
from sklearn.metrics import mean_squared_error
import pandas as pd
import gc
import uuid
# from tkinter import messagebox
# from threading import Thread
from datetime import datetime
import csv
from memory_profiler import profile
# from collections import UserDict

@profile
def load_and_preprocess_video(video_path, h, w, return_original=False):
    print("executing load_and_preprocess_video()")
    start_time = datetime.now()

    print("Video path:", video_path)  # Add this line for debugging

    cap = cv2.VideoCapture(video_path)

    # Check if video opened successfully
    if not cap.isOpened():
        print("Error opening video file:", video_path)  # Add this line for debugging
        return None, 0, 0  # Return None along with other values if video cannot be opened

    frames = []
    original_frames = []

    try:
        # Get total number of frames in the video
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Get frames per second (FPS) of the video
        fps = cap.get(cv2.CAP_PROP_FPS)

        # Calculate video duration in seconds
        video_duration = total_frames / fps

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if return_original:
                original_frames.append(frame)

            # (128,128,1 or 3)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frame = cv2.resize(frame, (h, w))
            # Normalizing (why?)
            frame = frame.astype("float32") / 255.0
            frames.append(frame)

        cap.release()
        video = np.array(frames)
        video = np.expand_dims(video, axis=-1)

        end_time = datetime.now()
        load_and_preprocess_video_duration = (end_time - start_time).seconds

        if return_original:
            return video, video_duration, original_frames, load_and_preprocess_video_duration
        else:
            return video, video_duration, load_and_preprocess_video_duration

    except Exception as e:
        print("Error processing video:", e)
        return None, 0, 0
    if return_original:
        return video, video_duration, original_frames, load_and_preprocess_video_duration
    else:
        return video, video_duration, load_and_preprocess_video_duration


@profile
def build_autoencoder():
        print("executing build_autoencoder()")
        # glogally 
        h, w = 128, 128
        input_img = Input(shape=(h, w, 1)) # channel 1 (gray scale) or 3 (coloured images)

        # Encoding
        print("Encoding...")
        x = Conv2D(16, (3, 3), activation="relu", padding="same")(input_img)
        x = MaxPooling2D((2, 2), padding="same")(x)
        x = Conv2D(8, (3, 3), activation="relu", padding="same")(x)
        x = Conv2D(8, (3, 3), activation="relu", padding="same")(x)
        encoded = MaxPooling2D((2, 2), padding="same")(x)

        # Decoding
        print("Decoding...")
        x = Conv2D(8, (3, 3), activation="relu", padding="same")(encoded)
        x = UpSampling2D((2, 2))(x)
        x = Conv2D(16, (3, 3), activation="relu", padding="same")(x)
        x = Conv2D(16, (3, 3), activation="relu", padding="same")(x)
        x = UpSampling2D((2, 2))(x)
        decoded = Conv2D(1, (3, 3), activation="sigmoid", padding="same")(x)

        # Build the autoencoder model
        print("Building the autoencoder model...")
        autoencoder = Model(input_img, decoded)
        autoencoder.compile(optimizer=tf.keras.optimizers.Adam(), loss=tf.keras.losses.MeanSquaredError())

        optimizer = tf.keras.optimizers.Adam()

        return autoencoder, optimizer

"""
`def save_mse_to_csv(mse_values, file_name, anomaly_indices=None, video_type="training"):` 
function to save mean squared error (MSE) values to a CSV file. 
It takes the MSE values, file name, anomaly indices, and video type as inputs.
"""


@profile
def save_mse_to_csv(mse_values, file_name, run_guid, anomaly_indices=None, video_type="training"):
    print("executing save_mse_to_csv()")
    # Set to an empty list if None is passed
    if anomaly_indices is None:
        anomaly_indices = []
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(file_name), exist_ok=True)

    with open(file_name, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["GUID", "Frame Number", "Anomaly Score", "Anomaly Status"])

        for i, mse in enumerate(mse_values):
            if video_type == "training":
                status = "Normal"  # All frames in training video are normal
            else:  # For testing video
                status = "Anomaly" if i in anomaly_indices else "Normal"
            writer.writerow([run_guid, i, mse, status])

@profile
@tf.function
def train_step(input_video, target_video, autoencoder, optimizer):
        print("executing train_step()")
        with tf.GradientTape() as tape:
            reconstructed_video = autoencoder(input_video, training=True)
            loss = tf.keras.losses.MeanSquaredError()(target_video, reconstructed_video)
        gradients = tape.gradient(loss, autoencoder.trainable_variables)
        optimizer.apply_gradients(zip(gradients, autoencoder.trainable_variables))
        return loss

@profile
def train_model(run_guid, input_epoch, train_video_path, anomaly_dir, app_version, folder_path, file_path):
    print("executing train_model()")
    start_time = datetime.now()

    # Training logic
    h, w = 128, 128
    input_video_path = train_video_path
    print("Input video path:", input_video_path)
    input_video, training_video_duration, training_load_and_preprocess_dur = load_and_preprocess_video(input_video_path,
                                                                                                       h, w)

    if input_video is None:
        # Handle the case where loading and preprocessing failed
        print("Error loading or preprocessing video. Aborting training.")
        return 0, 0, 0, 0, 0

    print("Training video frames: ", len(input_video))

    epochs = input_epoch

    # Calculate total number of batches
    batch_size = 5
    total_batches = len(input_video) // batch_size * epochs

    # Counter for batches
    batch_counter = 0

    autoencoder, optimizer = build_autoencoder()

    # Initialize optimizer's state
    dummy_input = tf.ones((1, h, w, 1))  # Create a dummy input tensor
    autoencoder(dummy_input)
    optimizer.iterations  # This builds the optimizer's state

    # Train your model here using input_video and epochs
    # Convert input_epoch to integer
    epochs = int(input_epoch)
    for epoch in range(epochs):
        print("Epoch: ", epoch + 1)
        for i in range(0, len(input_video), batch_size):
            batch_input_video = input_video[i:i + batch_size]
            batch_target_video = input_video[i:i + batch_size]
            loss = train_step(batch_input_video, batch_target_video, autoencoder, optimizer)
            batch_counter += 1
            #             progress = (batch_counter / total_batches) * 100
            #             progress['value'] = progress
            #             # Update the progress label
            #             progress_label.config(text="{}%".format(int(progress)))
            #             update_idletasks()
            print(f"  Batch {i // batch_size + 1}/{len(input_video) // batch_size}, Loss: {loss:.6f}")

    reconstructed_train_video = autoencoder.predict(input_video)
    train_mse = [mean_squared_error(input_video[i].reshape(h, w), reconstructed_train_video[i].reshape(h, w)) for i in
                 range(len(input_video))]
    threshold = np.mean(train_mse)

    save_mse_to_csv(train_mse, os.path.join(anomaly_dir, "RTAD_v{}_training_anomaly_scores.csv".format(app_version)),
                    run_guid, video_type="training")

    end_time = datetime.now()
    training_duration = (end_time - start_time).seconds

    model_path = os.path.join(anomaly_dir, "RTAD_v{}_saved_model.h5".format(app_version))
    autoencoder.save(model_path)

    train_video_frames = len(input_video)
    gc.collect()
    return train_video_frames, training_duration, training_video_duration, training_load_and_preprocess_dur, input_epoch

@profile
def detect_anomalies(test_video_path, train_video_frames, training_duration, training_video_duration,
                     training_load_and_preprocess_dur, input_epoch, anomaly_dir, app_version, folder_path,
                     unique_folder_timestamp):
    print("Detecting anomalies...")
    autoencoder, optimizer = build_autoencoder()
    start_time = datetime.now()

    h, w = 128, 128  # Ensure that this line is properly indented

    test_video, testing_video_duration, original_frames, testing_load_and_preprocess_dur = load_and_preprocess_video(
        test_video_path, h, w, return_original=True)

    print("Testing video frames: ", len(test_video))

    test_video_frames = len(test_video)

    # Detect anomalies using trained model and test_video
    # Calculate accuracy and update the label text
    reconstructed_test_video = autoencoder.predict(test_video)
    mse = [mean_squared_error(test_video[i].reshape(h, w), reconstructed_test_video[i].reshape(h, w)) for i in
           range(len(test_video))]

    threshold = np.percentile(mse, 98)
    anomaly_frames = [i for i, mse in enumerate(mse) if mse > threshold]

    # print("mse: ", mse)
    print("threshold: ", threshold)
    anomalies = mse > threshold
    anomaly_indices = np.argwhere(anomalies).flatten()

    # print("anomaly_indices: ", anomaly_indices)

    # Display the MSE distribution plot
    # self.create_mse_plot(mse)

    print(f"Number of detected anomalies: {len(anomaly_indices)}")

    # Replace this with trained model calculated accuracy
    # accuracy = 0.95
    # self.accuracy_value.config(text="{:.2f}%".format(accuracy * 100))

    for index in anomaly_indices:
        anomaly_frame = original_frames[index]
        anomaly_frame = Image.fromarray(cv2.cvtColor(anomaly_frame, cv2.COLOR_BGR2RGB))

        # Save the anomaly frame with original frame number in the filename
        anomaly_filename = os.path.join(anomaly_dir, f"anomaly_frame_{index}.png")
        anomaly_frame.save(anomaly_filename)

    # Save the testing video anomaly scores to CSV
    run_guid = str(uuid.uuid4())  # Generate a unique identifier
    save_mse_to_csv(mse, os.path.join(anomaly_dir, "RTAD_v{}_testing_anomaly_scores.csv".format(app_version)), anomaly_indices, video_type="testing")

    end_time = datetime.now()
    testing_duration = (end_time - start_time).seconds

    print("Saving metadata...")

    data = {
        'GUID': run_guid,
        'Model Version': "RTAD_v{}".format(app_version),
        'Created_Folder_Path': folder_path,
        'Training_Video_Src': test_video_path,
        'Training_Video_Dur': training_video_duration,
        'Training_load_and_preprocess_dur': training_load_and_preprocess_dur,
        'Training_Video_Frames': test_video_frames,
        'Model_Training_Dur': str(training_duration) + " seconds",
        'Testing_Video_src': test_video_path,
        'Testing_Video_Dur': testing_video_duration,
        'Testing_load_and_preprocess_dur': testing_load_and_preprocess_dur,
        'Testing_Video_Frames': test_video_frames,
        'Model_Testing_Dur': str(testing_duration) + " seconds",
        'Model_Accuracy_Percent': "To Be Developed",
        'Dectected_Anomalies': len(anomaly_indices),
        'Download_Status': "Success",  # Update this based on actual download status
        'Created On': datetime.now().strftime('%d/%m/%Y %H:%M:%S'),
        'Created By': "Kaushik Jandhayala",
        'Epochs': input_epoch
    }
    df = pd.DataFrame([data])
    df.to_csv(os.path.join(anomaly_dir, 'RTAD_v{}_Metadata_{}.csv'.format(app_version, unique_folder_timestamp)),
              index=False)

    print("Anomalies Detected Successfully and Saved Locally!")

    return anomaly_frames  # Return the anomaly frames list

class Logger(object):
    def __init__(self, filename="output_log.txt"):
        self.terminal = sys.stdout
        self.log = open(filename, "w")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        # This flush method is needed for python 3 compatibility.
        # This handles the flush command by doing nothing.
        # You might want to specify some extra behavior here.
        pass

sys.stdout = Logger("outputs/output_log.txt")



if __name__ == "__main__":
    @profile
    def upload_file():

        run_guid = uuid.uuid4()  # Generate a unique identifier for this run
        train_video_path = "video_path"
        # Pass folder_path as an argument to train_model
        train_video_frames, training_duration, training_video_duration, training_load_and_preprocess_dur, _ = train_model(run_guid, input_epoch, file_path, anomaly_dir, app_version, folder_path, file_path)

        gc.collect()
        cap = None
        frames = None
        original_frames = None
        total_frames = None
        fps = None
        video_duration = None
        video = None

        test_video_path = "video_path"
        unique_folder_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')  # Define unique_folder_timestamp
        # Pass unique_folder_timestamp as an argument to detect_anomalies
        detect_anomalies(test_video_path, train_video_frames, training_duration, training_video_duration,
                         training_load_and_preprocess_dur, input_epoch, anomaly_dir, app_version, folder_path,
                         unique_folder_timestamp)

        return train_video_frames  # Return train_video_frames
    # Call the upload_file function
    train_video_frames = upload_file()
# working and tested on 27/01/24
