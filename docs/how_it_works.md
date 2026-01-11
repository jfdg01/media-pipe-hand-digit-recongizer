# 🎬 How It Works: Multimodal Film Rating System

This document provides an in-depth technical explanation of the Multimodal Film Rating System, detailing each component, the data flow, and how the various technologies integrate to create an interactive movie rating experience.

---

## 📋 Table of Contents

1. [Overview](#overview)
2. [System Architecture](#system-architecture)
3. [Application Flow](#application-flow)
4. [Core Modules](#core-modules)
   - [Main Entry Point](#1-main-entry-point-mainpy)
   - [Controller](#2-controller-controllerpy)
   - [Speech Recognition](#3-speech-recognition-speechpy)
   - [Film API](#4-film-api-film_apipy)
   - [Gesture Recognition](#5-gesture-recognition-gesturespy)
   - [Display Manager](#6-display-manager-displaypy)
   - [Data Models](#7-data-models-modelspy)
5. [Machine Learning: Gesture Recognition Model](#machine-learning-gesture-recognition-model)
6. [Technology Stack](#technology-stack)
7. [Configuration Options](#configuration-options)

---

## Overview

The **Multimodal Film Rating System** is a Python application that demonstrates the integration of three different input modalities:

1. **🎤 Speech Input** — Users speak the name of a film
2. **🌐 API Integration** — Film data is fetched from the OMDb API
3. **✋ Gesture Input** — Users rate the film (1-5) using hand gestures

This creates a completely hands-free, voice-and-gesture-controlled movie rating experience.

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         USER INTERACTION                                │
│                                                                         │
│    🎤 Microphone          📷 Camera              🖥️ Display            │
│         │                     │                       ▲                 │
└─────────│─────────────────────│───────────────────────│─────────────────┘
          │                     │                       │
          ▼                     ▼                       │
┌─────────────────┐   ┌─────────────────┐   ┌─────────────────────────────┐
│ Speech Module   │   │ Gesture Module  │   │    Display Module           │
│ (Google Speech  │   │ (MediaPipe +    │   │    (OpenCV + PIL)           │
│  Recognition)   │   │  Custom Model)  │   │                             │
└────────┬────────┘   └────────┬────────┘   └──────────────▲──────────────┘
         │                     │                           │
         │                     │                           │
         ▼                     ▼                           │
┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│                     FilmRatingController                                │
│                        (Orchestrator)                                   │
│                                                                         │
│  1. Get film name ──► 2. Fetch film ──► 3. Get rating ──► 4. Display   │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
                                │
                                ▼
                    ┌───────────────────────┐
                    │      OMDb API         │
                    │  (Film Information)   │
                    └───────────────────────┘
```

---

## Application Flow

When you run the application, it follows a **4-step sequential flow**:

### Step 1: Speech Input 🎤

```
User speaks → Microphone captures → Google Speech API → Text
```

- The microphone captures ambient noise for 1 second to calibrate
- User speaks the film title (e.g., "Inception", "Matrix")
- Google's Speech-to-Text API transcribes the audio
- Returns the recognized film title as a string

### Step 2: Film Data Fetch 🌐

```
Film title → OMDb API Request → JSON Response → FilmData object
```

- The recognized title is sent to OMDb API
- API returns movie metadata (title, year, director, genre, poster, etc.)
- Data is parsed into a `FilmData` dataclass for easy access

### Step 3: Gesture Recognition ✋

```
Camera captures → MediaPipe processes → Custom model classifies → Score (1-5)
```

- Camera opens in fullscreen mode
- User shows 1-5 fingers to the camera
- MediaPipe extracts hand landmarks
- Custom-trained model classifies the gesture
- Requires **10 consecutive stable frames** for confirmation
- Returns the numeric score (1-5)

### Step 4: Display Result 🖼️

```
Film data + Score → Generate image → Show fullscreen → Console summary
```

- Downloads the movie poster from the URL
- Creates a composite image with poster, film info, and user rating
- Displays in a fullscreen OpenCV window
- Prints a formatted summary to the console
- Auto-closes after 10 seconds or on keypress

---

## Core Modules

### 1. Main Entry Point (`main.py`)

**Purpose:** Application entry point and CLI argument parsing.

**Key Responsibilities:**

- Load environment variables from `.env` file
- Parse command-line arguments
- Initialize the `FilmRatingController`
- Handle keyboard input mode override

**CLI Arguments:**

| Flag | Description | Default |
|------|-------------|---------|
| `--language`, `-l` | Speech recognition language | `es-ES` |
| `--keyboard`, `-k` | Use keyboard instead of speech | `false` |
| `--camera-url`, `-c` | IP Webcam URL (phone camera) | `None` |

**Example Usage:**

```bash
# Spanish speech + phone camera
python src/main.py --camera-url "http://192.168.1.100:8080/video"

# English speech + local webcam
python src/main.py --language en-US

# Keyboard input mode (testing)
python src/main.py --keyboard
```

---

### 2. Controller (`controller.py`)

**Purpose:** Main orchestrator that coordinates all modules.

**Class: `FilmRatingController`**

```python
class FilmRatingController:
    def __init__(self, language: str, camera_url: str):
        self.speech = SpeechRecognizer(language)
        self.films = FilmFetcher()
        self.gestures = GestureRecognizer(camera_url)
        self.display = DisplayManager()
    
    def run(self):
        # Executes the 4-step flow
```

**The `run()` method implements:**

1. Welcome banner display
2. Speech capture → film title
3. API lookup → film data
4. Gesture capture → user rating
5. Result display

---

### 3. Speech Recognition (`speech.py`)

**Purpose:** Convert spoken words to text.

**Class: `SpeechRecognizer`**

**Dependencies:** `speech_recognition` library

**How it works:**

1. **Initialize microphone** — Opens the default system microphone
2. **Ambient noise calibration** — 1 second of silence to adjust for background noise
3. **Listen for speech** — Captures audio with a 10-second timeout
4. **Transcribe** — Sends audio to Google's Speech-to-Text API
5. **Return text** — Returns the transcribed string or `None` on failure

**Error Handling:**

- `WaitTimeoutError` — No speech detected within timeout
- `UnknownValueError` — Audio was captured but couldn't be understood
- `RequestError` — API connection failure

**Supported Languages:**
Any language code supported by Google Speech API (e.g., `en-US`, `es-ES`, `fr-FR`, `de-DE`).

---

### 4. Film API (`film_api.py`)

**Purpose:** Fetch movie information from the OMDb database.

**Class: `FilmFetcher`**

**API Used:** [OMDb API](https://www.omdbapi.com/)

**How it works:**

1. **Read API key** — From `OMDB_API_KEY` environment variable
2. **Build request** — HTTP GET to `http://www.omdbapi.com/`
3. **Parse response** — Extract relevant fields from JSON
4. **Return FilmData** — Structured dataclass with movie info

**Required Environment Variable:**

```bash
export OMDB_API_KEY="your_api_key_here"
```

**API Response Fields Used:**

- `Title` — Movie title
- `Year` — Release year
- `Director` — Director name
- `Genre` — Genre(s)
- `Plot` — Short plot summary
- `Poster` — Poster image URL
- `imdbRating` — IMDb score
- `Runtime` — Movie duration

---

### 5. Gesture Recognition (`gestures.py`)

**Purpose:** Recognize hand gestures (1-5 fingers) for rating input.

**Class: `GestureRecognizer`**

**Key Components:**

#### MediaPipe Integration

Uses MediaPipe's **Gesture Recognition Task API**:

```python
base_options = python.BaseOptions(model_asset_path="models/gesture_recognizer.task")
options = vision.GestureRecognizerOptions(base_options=base_options)
recognizer = vision.GestureRecognizer.create_from_options(options)
```

#### Gesture Mapping

```python
GESTURE_TO_SCORE = {
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5
}
```

#### Recognition Flow

1. **Open camera** — Local webcam (index 0) or IP Webcam URL
2. **Frame capture loop** — Continuous camera feed
3. **Preprocessing** — Convert BGR→RGB, flip if using phone camera
4. **MediaPipe inference** — Run gesture recognition on each frame
5. **Stability check** — Require 10 consecutive identical gestures
6. **Confirmation** — Return score when stable gesture detected

#### Stability Mechanism

To prevent accidental inputs, the system requires **10 consecutive frames** (about 0.5-1 second) with the same gesture recognized at >70% confidence.

```python
if current_gesture == stable_gesture and current_gesture is not None:
    stable_count += 1
else:
    stable_gesture = current_gesture
    stable_count = 1 if current_gesture else 0

if stable_count >= required_stable_frames:
    return GESTURE_TO_SCORE[stable_gesture]
```

#### Fallback Behavior

If the camera fails or the model isn't available:

- Falls back to keyboard input
- User types a number 1-5 manually

---

### 6. Display Manager (`display.py`)

**Purpose:** Visual output — poster display and rating visualization.

**Class: `DisplayManager`**

**Key Methods:**

#### `download_poster(url)`

- Downloads movie poster from URL
- Converts to numpy array for OpenCV
- Returns `None` if URL is invalid or download fails

#### `create_rating_display(film, user_score)`

- Creates a **1200×900 pixel canvas**
- Centers the movie poster (resized to fit)
- Overlays film information:
  - Title, year, runtime
  - IMDb rating
  - User's star rating (ASCII representation)

#### `show_result(film, user_score)`

- Creates fullscreen OpenCV window
- Displays the composite image
- Prints formatted summary to console
- Auto-closes after 10 seconds or on keypress

**Console Output Example:**

```
==================================================
🎬 Inception (2010)
🎭 Action, Adventure, Sci-Fi
🎥 Director: Christopher Nolan
⭐ IMDb Rating: 8.8
==================================================
📊 YOUR SCORE: ★★★★☆ (4/5)
==================================================
```

---

### 7. Data Models (`models.py`)

**Purpose:** Type-safe data structures.

**Class: `FilmData`**

```python
@dataclass
class FilmData:
    title: str
    year: str
    director: str
    genre: str
    plot: str
    poster_url: str
    imdb_rating: str
    runtime: str
    
    def __str__(self) -> str:
        return f"{self.title} ({self.year}) - Directed by {self.director}"
```

Using Python's `@dataclass` decorator provides:

- Automatic `__init__` generation
- Type hints for IDE support
- Clean `__str__` representation

---

## Machine Learning: Gesture Recognition Model

### Overview

The gesture recognition model is a **custom-trained MediaPipe gesture classifier** that recognizes hand gestures representing digits 1-5.

### Training Pipeline

Located in `scripts/train.py`:

```
┌───────────────────┐     ┌──────────────────┐     ┌───────────────────┐
│  hand_digits_data │ ──► │  MediaPipe       │ ──► │ gesture_recognizer│
│  (labeled images) │     │  Model Maker     │     │      .task        │
└───────────────────┘     └──────────────────┘     └───────────────────┘
```

### Training Configuration

```python
hparams = gesture_recognizer.HParams(
    learning_rate=0.001,
    batch_size=16,
    epochs=50,
    export_dir="digit_model_v1"
)

model_options = gesture_recognizer.ModelOptions(
    dropout_rate=0.2  # Prevent overfitting
)
```

### Dataset Structure

```
hand_digits_data/
├── one/
│   ├── img001.jpg
│   ├── img002.jpg
│   └── ...
├── two/
├── three/
├── four/
├── five/
└── none/  # Background/negative examples
```

### Data Split

- **80%** Training
- **10%** Validation
- **10%** Testing

### Output

The trained model is exported as:

```
models/gesture_recognizer.task
```

This `.task` file is a MediaPipe Task Bundle containing:

- Hand landmark detection model
- Gesture classification model
- All necessary metadata

---

## Technology Stack

| Component | Technology | Purpose |
|-----------|------------|---------|
| **Speech Recognition** | Google Speech-to-Text | Audio transcription |
| **Film Data** | OMDb API | Movie database |
| **Hand Detection** | MediaPipe | Hand landmark extraction |
| **Gesture Classification** | Custom TensorFlow Model | Digit recognition |
| **Image Processing** | OpenCV + PIL | Camera capture, display |
| **Model Training** | MediaPipe Model Maker | Transfer learning |
| **Environment** | python-dotenv | Configuration management |

### Python Dependencies

Key packages from `requirements.txt`:

- `mediapipe` — Hand tracking and gesture recognition
- `opencv-python` — Camera and display
- `SpeechRecognition` — Audio capture
- `requests` — API calls
- `Pillow` — Image processing
- `python-dotenv` — Environment variables
- `mediapipe-model-maker` — Model training

---

## Configuration Options

### Environment Variables (`.env`)

```bash
# OMDb API key for film data
OMDB_API_KEY=your_api_key_here

# Phone camera URL (optional)
CAMERA_URL=http://192.168.1.100:8080/video
```

### CLI Options

```bash
python src/main.py [OPTIONS]

Options:
  -l, --language TEXT    Speech recognition language code (default: es-ES)
  -k, --keyboard         Use keyboard input instead of speech
  -c, --camera-url TEXT  IP Webcam URL for phone camera
```

### Phone Camera Setup (IP Webcam)

1. Install "IP Webcam" app on your phone
2. Start the server in the app
3. Note the URL shown (e.g., `http://192.168.1.100:8080/video`)
4. Pass URL via `--camera-url` or set in `.env`

---

## Summary

The Multimodal Film Rating System demonstrates how multiple input modalities (speech, vision, touch) can be combined to create natural, intuitive user interfaces. The modular architecture allows each component to be developed, tested, and improved independently while the controller orchestrates the complete user experience.

**Key Takeaways:**

- **Speech** enables hands-free film search
- **Gestures** provide an intuitive rating mechanism
- **Visual feedback** confirms the user's actions
- **Modular design** enables easy extension and maintenance
