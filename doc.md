# MediaPipe Model Maker: Hand Gesture Digit Recognition (1-5)

This documentation details the process of creating a custom hand gesture recognition model to identify digits 1 through 5 using MediaPipe Model Maker.

## 1. Environment Setup

Due to system compatibility requirements (TensorFlow < 2.16, Keras < 3.0), a specific Python 3.9 environment was established.

* **Python Version:** Python 3.9 (Installed via `deadsnakes` PPA on Ubuntu 24.04).
* **Virtual Environment:** `mp_env` created and activated.
* **Key Dependencies:**
  * `mediapipe-model-maker==0.2.1.3`
  * `mediapipe==0.10.9` (Stability pin for `tasks.cc` bindings)
  * `tensorflow==2.15.1`
  * `protobuf==3.20.3`

## 2. Dataset Preparation

MediaPipe Model Maker requires images organized into labeled folders.

### Data Sourcing

* **Target Classes (1-5):** Sourced from the `ardamavi/Sign-Language-Digits-Dataset` (added as a git submodule).
* **None Class (Background/Stability):**
  * **Negative Hand Gestures:** Sourced digits 6, 7, 8, and 9 from the ASL dataset.
  * **Random Noise:** Sourced ~50 images from the `Tiny-COCO` dataset (random objects like chairs, tables, etc.).

### Folder Structure

```text
hand_digits_data/
├── none/  (Neutral hands + COCO background objects)
├── one/   (Gesture for 1)
├── two/   (Gesture for 2)
├── three/ (Gesture for 3)
├── four/  (Gesture for 4)
└── five/  (Gesture for 5)
```

## 3. Model Training (`train.py`)

The training process utilized transfer learning on hand landmark coordinates rather than raw pixels.

### Hyperparameters

* **Learning Rate:** 0.001
* **Batch Size:** 16 (Optimal for CPU/Laptop stability)
* **Epochs:** 50
* **Dropout Rate:** 0.2 (To prevent overfitting to training backgrounds)

### The "Baking" Process

The script performs the following:

1. **Landmark Extraction:** Detects 21 hand points on every image.
2. **Data Splitting:** 80% Train, 10% Validation, 10% Test.
3. **Training:** Executes `GestureRecognizer.create`.
4. **Export:** Produces the production-ready `gesture_recognizer.task` bundle.

## 4. Inference & Usage (`test_inference.py`)

To use the model, use the `mediapipe.tasks` vision library.

```python
base_options = python.BaseOptions(model_asset_path='digit_model_v1/gesture_recognizer.task')
options = vision.GestureRecognizerOptions(base_options=base_options)
recognizer = vision.GestureRecognizer.create_from_options(options)

image = mp.Image.create_from_file("your_image.jpg")
result = recognizer.recognize(image)
```

## 6. Troubleshooting Notes

* **ModuleNotFoundError (tasks.cc):** Resolved by downgrading `mediapipe` to `0.10.9`.
* **DNS Failures:** Base models (`palm_detection_full.tflite`) were manually downloaded to `/tmp/model_maker/` to bypass name resolution errors.
* **Pip Freeze Issues:** Removed system-level packages (`bcc`, `python-apt`) from `requirements.txt` to ensure cross-platform compatibility.
