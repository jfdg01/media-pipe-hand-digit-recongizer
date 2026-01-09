import os
import tensorflow as tf
from mediapipe_model_maker import gesture_recognizer

# 1. Initialize Dataset Loading (as recommended in report.md lines 133-142)
dataset_path = "hand_digits_data/"

print(f"Loading dataset from: {dataset_path}...")

# Configure preprocessing parameters 
# min_detection_confidence is set to 0.8 to filter out noisy images
data_hparams = gesture_recognizer.HandDataPreprocessingParams(
    shuffle=True,
    min_detection_confidence=0.8
)

# Load the dataset
# This step automatically runs the landmark model on every image
data = gesture_recognizer.Dataset.from_folder(
    dirname=dataset_path,
    hparams=data_hparams
)

print(f"Dataset successfully loaded. Total samples: {len(data)}")

# 2. Data Splitting (Lines 145–146)
# Splitting into 80% Training, 10% Validation, and 10% Testing
train_data, rest_data = data.split(0.8)
validation_data, test_data = rest_data.split(0.5)

print(f"Data split: {len(train_data)} train, {len(validation_data)} validation, {len(test_data)} test samples.")

# 3. Configure Training Implementation (Lines 161–191)
# Using stability recommendations for Laptop/CPU training
print("Initializing training process...")

# Define hyperparameters based on report recommendations (Line 152)
hparams = gesture_recognizer.HParams(
    learning_rate=0.001,
    batch_size=16,
    epochs=50,
    export_dir="digit_model_v1"
)

# Model options: dropout_rate 0.2 to prevent overfitting (Line 157)
model_options = gesture_recognizer.ModelOptions(
    dropout_rate=0.2
)

options = gesture_recognizer.GestureRecognizerOptions(
    model_options=model_options,
    hparams=hparams
)

# Train the model (Starts the transfer learning process)
print("Training model... This Step extracts landmarks and trains the classifier.")
model = gesture_recognizer.GestureRecognizer.create(
    train_data=train_data,
    validation_data=validation_data,
    options=options
)

# 4. Evaluation (Lines 202–204)
print("Performing final evaluation on test set...")
loss, accuracy = model.evaluate(test_data, batch_size=1)
print(f"Final Test Accuracy: {accuracy:.4f}")

# 5. Exporting the Task Bundle (Line 213)
# This creates the 'gesture_recognizer.task' file
print("Exporting model to digit_model_v1/gesture_recognizer.task...")
model.export_model()
print("Model export complete!")
