import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import os
import glob

# Path to your new model and the test folder
model_path = 'digit_model_v1/gesture_recognizer.task'
test_folder = 'test_data'

if not os.path.exists(model_path):
    print(f"Error: Model not found at {model_path}")
    exit()

# 1. Create a GestureRecognizer object
base_options = python.BaseOptions(model_asset_path=model_path)
options = vision.GestureRecognizerOptions(base_options=base_options)
recognizer = vision.GestureRecognizer.create_from_options(options)

# 2. Get all images in the folder
image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']
image_paths = []
for ext in image_extensions:
    image_paths.extend(glob.glob(os.path.join(test_folder, ext)))

image_paths.sort()

# 3. Process each image and compare with filename
print(f"\n--- Batch Inference Results for {test_folder} ---")
correct_count = 0
total_count = 0

# Mapping of digits to word labels (as used in folder names during training)
digit_map = {
    '1': 'one',
    '2': 'two',
    '3': 'three',
    '4': 'four',
    '5': 'five'
}

for image_path in image_paths:
    # Extract filename and expected label
    filename = os.path.basename(image_path).lower()
    expected_label = "none"
    
    # Check for word labels in the filename
    for digit, word in digit_map.items():
        if word in filename:
            expected_label = word
            break
    
    # Load and process image
    image = mp.Image.create_from_file(image_path)
    recognition_result = recognizer.recognize(image)
    
    predicted_label = "none"
    confidence = 0.0
    
    if recognition_result.gestures:
        category = recognition_result.gestures[0][0]
        predicted_label = category.category_name
        confidence = category.score
        
    is_correct = (predicted_label == expected_label)
    total_count += 1
    if is_correct:
        correct_count += 1
        
    status = "✅ PASS" if is_correct else f"❌ FAIL (Expected: {expected_label})"
    print(f"[{filename:20}] Pred: {predicted_label:6} | Conf: {confidence:.4f} | {status}")

# 4. Final tally
if total_count > 0:
    accuracy = (correct_count / total_count) * 100
    print(f"\nSummary: {correct_count}/{total_count} correct ({accuracy:.2f}% accuracy)")
else:
    print("No images found in test_data.")
