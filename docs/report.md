# **Stable Workflow for Training a Custom Hand Digit Gesture Classifier via MediaPipe Model Maker on Ubuntu and NVIDIA Ampere Architectures**

The evolution of real-time computer vision has transitioned from complex, hand-engineered feature extraction to streamlined, high-level task-oriented pipelines. Among these, the MediaPipe framework stands as a prominent solution for multi-modal machine learning deployments. For engineering teams operating on the Ubuntu platform equipped with high-performance NVIDIA RTX 3090 hardware, the objective is the establishment of a robust, repeatable, and stable workflow for customizing hand gesture recognizers. This report delineates the end-to-end technical path for developing a digit classifier (1–5) while prioritizing environmental stability and the mitigation of version-based conflicts.

## **Technical Foundation of MediaPipe Gesture Recognition**

The MediaPipe Gesture Recognizer is not a single model but a sophisticated bundle composed of two primary stages: a hand landmark model and a gesture classification model.1 The system begins with a palm detection module that identifies hand regions in the raw input frame, subsequently cropping the image to focus on the hand. A landmark model then processes this crop to predict 21 specific 3D landmarks.2 These landmarks are represented as normalized coordinates $x, y, z \\in \[0.0, 1.0\]$, where $x$ and $y$ are mapped to the image width and height, and $z$ indicates the relative depth from the wrist origin.2

This geometric representation serves as the input for the second stage: the gesture classifier. By operating on 21 points rather than raw pixels, the classifier becomes inherently more robust to lighting variations, skin tones, and background clutter.4 The MediaPipe Model Maker tool leverages transfer learning to retrain these final classification layers using a custom dataset, enabling the recognition of new gestures like the digits 1 through 5 with remarkably small datasets.7

### **Performance Characteristics on RTX 3090**

The RTX 3090, utilizing the Ampere architecture, provides a substantial advantage for the training phase. While the final exported TFLite models are designed for lightweight on-device inference, the initial training process—specifically the landmark extraction from a thousand-image dataset—can be significantly accelerated by GPU acceleration. MediaPipe Tasks support GPU delegation, which reduces latency and enhances the frames-per-second (FPS) during both the validation and inference stages.1

## **Hardware and Software Environment Configuration**

The primary source of instability in Linux-based machine learning environments is the misalignment of the NVIDIA kernel driver, the CUDA Toolkit, and the deep learning framework binaries. For the RTX 3090, a specific stack is required to ensure the Ampere cores are utilized correctly.

### **Driver and Kernel Layer Stability**

The RTX 3090 requires NVIDIA Driver version 450 or higher, though version 525.60.13 or newer is recommended for compatibility with modern CUDA releases.12 On Ubuntu, the most stable installation method utilizes the standard repository rather than manual .run files, which often conflict with kernel updates.

| Command | Purpose |
| :---- | :---- |
| nvidia-smi | Verify driver installation and GPU visibility.13 |
| sudo apt install nvidia-driver-535 | Install a stable, long-term support driver version.15 |
| nvcc \--version | Confirm the CUDA compiler is available in the path.16 |

### **CUDA and cuDNN Integration**

For MediaPipe Model Maker and TensorFlow stability in 2024 and 2025, CUDA 11.8 or 12.3 should be paired with cuDNN 8.9.7.13 A critical step in maintaining stability on Ubuntu is ensuring that the LD\_LIBRARY\_PATH correctly points to the NVIDIA libraries within the virtual environment, preventing the system from falling back to CPU-based execution.13

## **Environment Isolation and Dependency Management**

Given the sensitivity of the mediapipe-model-maker package to its underlying dependencies, environment isolation is mandatory. Conflicts between keras, tensorflow, and tf-models-official frequently lead to installation failures if not pinned correctly.17

### **The Conda Workflow**

Conda is often preferred for its ability to manage both Python packages and binary dependencies like CUDA. However, official TensorFlow guidelines recommend pip inside a venv or Conda environment for the most current stable releases.13 The most stable approach involves creating a clean Python 3.9 environment, which remains the most broadly compatible version for Model Maker.8

Bash

\# Create and activate the environment  
conda create \-n mp\_digit\_classifier python=3.9 \-y  
conda activate mp\_digit\_classifier

\# Upgrade build tools  
pip install \--upgrade pip setuptools

\# Install TensorFlow with GPU support for RTX 3090  
pip install "tensorflow\<2.16" "keras\<3.0.0" "tensorflow\[and-cuda\]"

### **Addressing Binary Conflicts**

A documented issue in the MediaPipe community involves the tensorflow-text dependency, which may not support all platforms or versions.17 If a standard installation fails, practitioners should install the core model maker components with the \--no-deps flag to bypass problematic version checks and then manually resolve requirements.

Bash

\# Stable installation sequence for MediaPipe Model Maker  
pip install "pyyaml\>6.0.0"  
pip install "tf-models-official\<2.16"  
pip install mediapipe-model-maker==0.2.1.3 \--no-deps

## **Dataset Curation and Structural Requirements**

The success of the digit classifier depends on the diversity of the image data. Unlike complex CNNs that require millions of parameters, the landmark-based approach functions optimally with approximately 100 high-quality images per class.7

### **Mandatory Directory Structure**

The Dataset.from\_folder method requires a strict hierarchical structure where sub-folders represent labels. For a digits 1–5 task, a six-class structure is required, including the "none" class.20

dataset\_root/  
├── none/ (Required: Images of neutral hands or random objects) 9  
├── one/ (Hand showing 1 finger)  
├── two/ (Hand showing 2 fingers)  
├── three/ (Hand showing 3 fingers)  
├── four/ (Hand showing 4 fingers)  
└── five/ (Hand showing 5 fingers)

### **The "None" Class Strategy**

The "none" class is a critical stability component. It represents the background state or any hand position that is not a target gesture. Without a robust "none" class, the model may return false positives when the user is simply moving their hand into position.1 The images in this folder should include:

* Relaxed hand positions with no extended fingers.  
* Hands partially out of the frame.  
* The palm-facing-away position (if not used for a digit).  
* Random movements that the user might perform naturally during an interaction.1

## **Quantitative Review of Available Hand Digit Datasets**

Building a dataset from scratch is often unnecessary due to the availability of high-quality open-source collections. For digit classification 1–5, the following datasets provide a reliable baseline.

| Dataset Name | Source | Composition | Format |
| :---- | :---- | :---- | :---- |
| Fingers Image Dataset | [Kaggle](https://www.kaggle.com/datasets/koryakinp/fingers) | 21,600 images (0–5 fingers) | 128x128 PNG.23 |
| Number Gestures 1–5 | [Kaggle](https://www.kaggle.com/datasets/uom190653l/number-gestures-1-5-hand-landmark-dataset) | Pre-processed landmarks for digits 1–5 | CSV/Images.24 |
| ASL Sign Language Detection | (https://universe.roboflow.com/unipi-4sk4y/asl-sign-language-detection) | 475 images, 19 classes (includes digits) | Object Detection/COCO.25 |
| Leap Gesture Recognition | [Kaggle](https://www.kaggle.com/datasets/gti-upm/leapgestrecog) | 10 gestures from 10 subjects | Infra-red PNG.27 |
| HGR Dataset | [Kaggle](https://www.kaggle.com/datasets/nizamuddinmaitlo/hgr-dataset) | Standard camera hand images | JPEG/CSV.28 |

### **Data Acquisition Commands**

To download and prepare a sample dataset such as the "rps\_data\_sample" or a custom zip from Kaggle, use the following commands within the Ubuntu terminal:

Bash

\# Download and unzip a dataset  
wget \[dataset\_link\_url\]  
unzip dataset.zip \-d hand\_digits\_data/

\# Check image counts  
ls hand\_digits\_data/one | wc \-l

## **The Training Implementation**

The training logic in MediaPipe Model Maker is designed to be low-code, but it requires careful handling of the validation split to ensure accuracy metrics are meaningful.

### **Loading and Processing the Dataset**

The Dataset.from\_folder call is where the heavy lifting occurs. During this phase, the system executes the pre-trained MediaPipe landmark model on every image. Any image where a hand cannot be clearly detected is omitted from the dataset.9

Python

import os  
import tensorflow as tf  
from mediapipe\_model\_maker import gesture\_recognizer

\# Define paths  
dataset\_path \= "hand\_digits\_data/"

\# Configure preprocessing parameters  
\# Increase min\_detection\_confidence if the dataset has noisy backgrounds \[20, 29\]  
data\_hparams \= gesture\_recognizer.HandDataPreprocessingParams(  
    shuffle=True,   
    min\_detection\_confidence=0.8  
)

\# Load the dataset  
data \= gesture\_recognizer.Dataset.from\_folder(  
    dirname=dataset\_path,  
    hparams=data\_hparams  
)

\# Split into train, validation, and test sets  
train\_data, rest\_data \= data.split(0.8)  
validation\_data, test\_data \= rest\_data.split(0.5)

### **Hyperparameter Selection for RTX 3090**

While default parameters are often sufficient, the RTX 3090 allows for larger batch sizes and more epochs without significant time penalties. The use of a hidden layer via layer\_widths can assist in separating similar gestures like digit "two" (Victory) and digit "three".9

| Hyperparameter | Default | Stability Recommendation | Role |
| :---- | :---- | :---- | :---- |
| learning\_rate | 0.001 | 0.0005 – 0.001 | Controls the gradient descent step size.4 |
| batch\_size | 2 | 16 – 32 | Number of samples processed per iteration.4 |
| epochs | 10 | 25 – 50 | Total passes over the dataset.4 |
| dropout\_rate | 0.05 | 0.2 | Prevents overfitting to the training samples.4 |
| layer\_widths |  |  | Hidden layers to increase model capacity.9 |
| lr\_decay | 0.99 | 0.99 | Rate at which the learning rate decreases.30 |

### **Executing the Training Task**

The GestureRecognizer.create method initiates the transfer learning process. It is vital to set a distinct export\_dir for each experiment to prevent overwriting checkpoints from previous runs.9

Python

\# Define options  
hparams \= gesture\_recognizer.HParams(  
    learning\_rate=0.001,  
    batch\_size=16,  
    epochs=50,  
    export\_dir="digit\_model\_v1"  
)

model\_options \= gesture\_recognizer.ModelOptions(  
    dropout\_rate=0.2,  
    layer\_widths=  
)

options \= gesture\_recognizer.GestureRecognizerOptions(  
    model\_options=model\_options,  
    hparams=hparams  
)

\# Train the model  
model \= gesture\_recognizer.GestureRecognizer.create(  
    train\_data=train\_data,  
    validation\_data=validation\_data,  
    options=options  
)

## **Evaluation, Benchmarking, and Metadata Export**

Once training completes, the model must be evaluated on the held-out test set. This provides an unbiased estimate of real-world performance.

### **Quantitative Evaluation**

The evaluate method returns the final loss and accuracy. For a simple 1–5 digit task, accuracy should ideally exceed 95% on the test set if the dataset is well-labeled and diverse.9

Python

\# Perform evaluation  
loss, accuracy \= model.evaluate(test\_data, batch\_size=1)  
print(f"Final Test Accuracy: {accuracy:.4f}")

### **Exporting the Task Bundle**

The result of the export\_model function is a .task file. This file is a Flatbuffer that encapsulates the entire MediaPipe pipeline, including the TFLite model, the label map, and the landmark detection metadata.9 This single file is all that is required for deployment.

Python

\# Export the model  
model.export\_model()

\# The file will be saved in digit\_model\_v1/gesture\_recognizer.task

## **Stability-Focused Inference and Deployment**

Deploying the custom digit classifier on the RTX 3090 involves using the MediaPipe Tasks Python API. For maximum stability and throughput, asynchronous processing should be considered for live stream inputs.2

### **GPU Delegation and Configuration**

To ensure the RTX 3090 is utilized during inference, the BaseOptions must specify the GPU delegate. This is currently supported on Linux and macOS but not on Windows native environments.11

Python

import mediapipe as mp  
from mediapipe.tasks import python  
from mediapipe.tasks.python import vision

\# Initialize the Gesture Recognizer  
base\_options \= python.BaseOptions(  
    model\_asset\_path='digit\_model\_v1/gesture\_recognizer.task',  
    delegate=python.BaseOptions.Delegate.GPU  
)

\# Configure running mode  
\# vision.RunningMode.IMAGE for single frames  
\# vision.RunningMode.LIVE\_STREAM for webcam feeds \[1, 2\]  
options \= vision.GestureRecognizerOptions(  
    base\_options=base\_options,  
    running\_mode=vision.RunningMode.IMAGE,  
    num\_hands=1,  
    min\_hand\_detection\_confidence=0.5  
)

\# Create the recognizer  
with vision.GestureRecognizer.create\_from\_options(options) as recognizer:  
    \# Load input image from a file  
    mp\_image \= mp.Image.create\_from\_file('test\_digit\_one.jpg')  
      
    \# Perform recognition  
    recognition\_result \= recognizer.recognize(mp\_image)  
      
    \# Parse results  
    if recognition\_result.gestures:  
        top\_gesture \= recognition\_result.gestures  
        print(f"Detected Digit: {top\_gesture.category\_name} (Score: {top\_gesture.score:.2f})")

### **Handling Real-time Video and Timestamps**

When operating in VIDEO or LIVE\_STREAM modes, each frame must be accompanied by a monotonically increasing timestamp in milliseconds. This is essential for the internal tracking algorithms to maintain continuity between frames.2

Python

\# Live stream example snippet  
import time  
timestamp\_ms \= int(time.time() \* 1000)  
recognizer.recognize\_async(mp\_image, timestamp\_ms)

## **Troubleshooting Common Obstacles on Ubuntu**

Despite following the stable workflow, environmental edge cases can arise.

### **CUDA Library Not Found**

If the GPU is not detected during inference, it is often because the TensorFlow backend cannot locate the cuDNN or CUDA shared objects (.so files). A common fix involves creating symbolic links from the Conda environment's library path to the standard NVIDIA search paths.13

Bash

\# Example symbolic link fix for missing CUDA libraries  
pushd $(dirname $(python \-c 'print(\_\_import\_\_("tensorflow").\_\_file\_\_)'))  
ln \-svf../nvidia/\*/lib/\*.so\*.  
popd

### **Dependency Hell with Keras 3**

The release of Keras 3 introduced significant breaking changes for the MediaPipe Model Maker, which was built against Keras 2 logic. Ensuring that keras\<3.0.0 is installed is the single most effective step in resolving ModuleNotFoundError issues related to keras.src.engine.17

## **Conclusion**

Developing a stable digit gesture classifier on Ubuntu using the RTX 3090 requires a disciplined approach to version management and data curation. By leveraging the MediaPipe Model Maker’s landmark-based transfer learning, developers can bypass the complexities of traditional CNN training. The most stable workflow utilizes Python 3.9, pins TensorFlow to versions below 2.16, and enforces a strict directory structure for labels, including a mandatory "none" class. When deployed with GPU delegation, this setup provides a high-fidelity, real-time solution capable of robust digit classification in demanding professional environments.

#### **Obras citadas**

1. Gesture recognition task guide | Google AI Edge, fecha de acceso: enero 9, 2026, [https://ai.google.dev/edge/mediapipe/solutions/vision/gesture\_recognizer](https://ai.google.dev/edge/mediapipe/solutions/vision/gesture_recognizer)  
2. Gesture recognition guide for Python | Google AI Edge, fecha de acceso: enero 9, 2026, [https://ai.google.dev/edge/mediapipe/solutions/vision/gesture\_recognizer/python](https://ai.google.dev/edge/mediapipe/solutions/vision/gesture_recognizer/python)  
3. How to Install MediaPipe Python for Real-Time Hand Tracking \- Omdena, fecha de acceso: enero 9, 2026, [https://www.omdena.com/blog/mediapipe-python-tutorial](https://www.omdena.com/blog/mediapipe-python-tutorial)  
4. Dynamic Hand Gesture Recognition Using MediaPipe and Transformer \- MDPI, fecha de acceso: enero 9, 2026, [https://www.mdpi.com/2673-4591/108/1/22](https://www.mdpi.com/2673-4591/108/1/22)  
5. mediapipe/docs/solutions/pose.md at master \- GitHub, fecha de acceso: enero 9, 2026, [https://github.com/google-ai-edge/mediapipe/blob/master/docs/solutions/pose.md](https://github.com/google-ai-edge/mediapipe/blob/master/docs/solutions/pose.md)  
6. How to Train Custom Hand Gestures Using Mediapipe \- Instructables, fecha de acceso: enero 9, 2026, [https://www.instructables.com/How-to-Train-Custom-Hand-Gestures-Using-Mediapipe/](https://www.instructables.com/How-to-Train-Custom-Hand-Gestures-Using-Mediapipe/)  
7. MediaPipe Model Maker | Google AI Edge, fecha de acceso: enero 9, 2026, [https://ai.google.dev/edge/mediapipe/solutions/model\_maker](https://ai.google.dev/edge/mediapipe/solutions/model_maker)  
8. How to Create Custom ML Models with Google MediaPipe Model Maker \- DigiKey, fecha de acceso: enero 9, 2026, [https://www.digikey.cz/en/maker/tutorials/2024/how-to-create-custom-ml-models-with-google-mediapipe-model-maker](https://www.digikey.cz/en/maker/tutorials/2024/how-to-create-custom-ml-models-with-google-mediapipe-model-maker)  
9. Hand gesture recognition model customization guide | Google AI Edge, fecha de acceso: enero 9, 2026, [https://ai.google.dev/edge/mediapipe/solutions/customization/gesture\_recognizer](https://ai.google.dev/edge/mediapipe/solutions/customization/gesture_recognizer)  
10. GPU Support | Google AI Edge, fecha de acceso: enero 9, 2026, [https://ai.google.dev/edge/mediapipe/framework/getting\_started/gpu\_support](https://ai.google.dev/edge/mediapipe/framework/getting_started/gpu_support)  
11. How to install and use GPU-supported MediaPipe on a Linux platform? \#5452 \- GitHub, fecha de acceso: enero 9, 2026, [https://github.com/google-ai-edge/mediapipe/issues/5452](https://github.com/google-ai-edge/mediapipe/issues/5452)  
12. The Simple Guide: Deep Learning with RTX 3090 (CUDA, cuDNN, Tensorflow, Keras, PyTorch) | by DeepLCH | Medium, fecha de acceso: enero 9, 2026, [https://medium.com/@deeplch/the-simple-guide-deep-learning-with-rtx-3090-cuda-cudnn-tensorflow-keras-pytorch-e88a2a8249bc](https://medium.com/@deeplch/the-simple-guide-deep-learning-with-rtx-3090-cuda-cudnn-tensorflow-keras-pytorch-e88a2a8249bc)  
13. Install TensorFlow with pip, fecha de acceso: enero 9, 2026, [https://www.tensorflow.org/install/pip](https://www.tensorflow.org/install/pip)  
14. How To Select the Correct TensorFlow Version for Your NVIDIA GPU, fecha de acceso: enero 9, 2026, [https://apxml.com/posts/select-tensorflow-version-nvidia-gpu](https://apxml.com/posts/select-tensorflow-version-nvidia-gpu)  
15. Anybody have a stable Nvidia driver version running with 2x3090 on Ubuntu? \- Reddit, fecha de acceso: enero 9, 2026, [https://www.reddit.com/r/LocalLLaMA/comments/1ir6bsr/anybody\_have\_a\_stable\_nvidia\_driver\_version/](https://www.reddit.com/r/LocalLLaMA/comments/1ir6bsr/anybody_have_a_stable_nvidia_driver_version/)  
16. Tensorflow WSL GPU CUDA recognition issue RTX3090 \#63948 \- GitHub, fecha de acceso: enero 9, 2026, [https://github.com/tensorflow/tensorflow/issues/63948](https://github.com/tensorflow/tensorflow/issues/63948)  
17. Can't pip install mediapipe-model-maker · Issue \#5214 · google-ai ..., fecha de acceso: enero 9, 2026, [https://github.com/google-ai-edge/mediapipe/issues/5214](https://github.com/google-ai-edge/mediapipe/issues/5214)  
18. How to install mediapipe model-maker using an already installed conda tensorflow \#5033, fecha de acceso: enero 9, 2026, [https://github.com/google-ai-edge/mediapipe/issues/5033](https://github.com/google-ai-edge/mediapipe/issues/5033)  
19. Customizing a gesture recognition model with MediaPipe \- Samuel Pröll \- Homepage, fecha de acceso: enero 9, 2026, [https://www.samproell.io/posts/ai/asl-detector-with-mediapipe-wsl/](https://www.samproell.io/posts/ai/asl-detector-with-mediapipe-wsl/)  
20. mediapipe-samples/examples/customization/gesture\_recognizer ..., fecha de acceso: enero 9, 2026, [https://github.com/googlesamples/mediapipe/blob/main/examples/customization/gesture\_recognizer.ipynb](https://github.com/googlesamples/mediapipe/blob/main/examples/customization/gesture_recognizer.ipynb)  
21. mediapipe\_model\_maker.gesture\_recognizer.Dataset | Google AI Edge, fecha de acceso: enero 9, 2026, [https://ai.google.dev/edge/api/mediapipe/python/mediapipe\_model\_maker/gesture\_recognizer/Dataset](https://ai.google.dev/edge/api/mediapipe/python/mediapipe_model_maker/gesture_recognizer/Dataset)  
22. Gesture recognition guide for Python | Google AI Edge | Google AI ..., fecha de acceso: enero 9, 2026, [https://ai.google.dev/edge/mediapipe/solutions/vision/gesture\_recognizer/python\#prepare\_data](https://ai.google.dev/edge/mediapipe/solutions/vision/gesture_recognizer/python#prepare_data)  
23. Fingers Image Dataset \- Kaggle, fecha de acceso: enero 9, 2026, [https://www.kaggle.com/datasets/aviratgupta/finger-identify](https://www.kaggle.com/datasets/aviratgupta/finger-identify)  
24. Number Gestures 1-5: Hand Landmark Dataset \- Kaggle, fecha de acceso: enero 9, 2026, [https://www.kaggle.com/datasets/uom190653l/number-gestures-1-5-hand-landmark-dataset](https://www.kaggle.com/datasets/uom190653l/number-gestures-1-5-hand-landmark-dataset)  
25. ASL Sign Language Detection Object Detection Dataset by Unipi \- Roboflow Universe, fecha de acceso: enero 9, 2026, [https://universe.roboflow.com/unipi-4sk4y/asl-sign-language-detection](https://universe.roboflow.com/unipi-4sk4y/asl-sign-language-detection)  
26. Sign Language Recognition Algorithms Using Hybrid Techniques, fecha de acceso: enero 9, 2026, [https://alife-robotics.co.jp/members2025/icarob/data/html/data/OS/OS26/OS26-4.pdf](https://alife-robotics.co.jp/members2025/icarob/data/html/data/OS/OS26/OS26-4.pdf)  
27. Hand Gesture Recognition Database \- Kaggle, fecha de acceso: enero 9, 2026, [https://www.kaggle.com/datasets/gti-upm/leapgestrecog](https://www.kaggle.com/datasets/gti-upm/leapgestrecog)  
28. HGR-Dataset \- Kaggle, fecha de acceso: enero 9, 2026, [https://www.kaggle.com/datasets/nizamuddinmaitlo/hgr-dataset](https://www.kaggle.com/datasets/nizamuddinmaitlo/hgr-dataset)  
29. Custom Gesture Recognition Model for Video Playback using MediaPipe \- GitBook, fecha de acceso: enero 9, 2026, [https://mycompay.gitbook.io/mediapipe-custom-gesture-model-tutorial](https://mycompay.gitbook.io/mediapipe-custom-gesture-model-tutorial)  
30. mediapipe/mediapipe/model\_maker/python/vision/gesture\_recognizer/gesture\_recognizer.py at master · google-ai-edge/mediapipe \- GitHub, fecha de acceso: enero 9, 2026, [https://github.com/google/mediapipe/blob/master/mediapipe/model\_maker/python/vision/gesture\_recognizer/gesture\_recognizer.py](https://github.com/google/mediapipe/blob/master/mediapipe/model_maker/python/vision/gesture_recognizer/gesture_recognizer.py)  
31. Real-Time Hand Tracking and Gesture Recognition with MediaPipe: Rerun Showcase, fecha de acceso: enero 9, 2026, [https://towardsdatascience.com/real-time-hand-tracking-and-gesture-recognition-with-mediapipe-rerun-showcase-9ec57cb0c831/](https://towardsdatascience.com/real-time-hand-tracking-and-gesture-recognition-with-mediapipe-rerun-showcase-9ec57cb0c831/)  
32. mediapipe-samples/examples/gesture\_recognizer/raspberry\_pi/recognize.py at main \- GitHub, fecha de acceso: enero 9, 2026, [https://github.com/google-ai-edge/mediapipe-samples/blob/main/examples/gesture\_recognizer/raspberry\_pi/recognize.py](https://github.com/google-ai-edge/mediapipe-samples/blob/main/examples/gesture_recognizer/raspberry_pi/recognize.py)  
33. Gesture recognition \- ML on Web with MediaPipe: Episode 3 \- YouTube, fecha de acceso: enero 9, 2026, [https://www.youtube.com/watch?v=cJgDuywJv8Y](https://www.youtube.com/watch?v=cJgDuywJv8Y)