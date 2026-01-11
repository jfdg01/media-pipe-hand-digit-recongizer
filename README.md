# 🎬 Multimodal Film Rating System

A multimodal application that combines **speech recognition**, **gesture recognition**, and **film API** to create an interactive movie rating experience.

## Features

- **🎤 Speech Recognition**: Say the name of a film
- **🌐 Film API**: Fetches movie data from OMDb API
- **✋ Hand Gesture Recognition**: Rate movies 1-5 using hand gestures
- **🖼️ Visual Display**: Shows movie poster with your rating

## Project Structure

```
├── src/                        # Source code
│   ├── main.py                 # Main application (Film Rating Controller)
│   ├── film_api.py             # Standalone film data fetcher
│   └── speech_test.py          # Speech recognition test script
├── models/                     # Trained ML models
│   └── gesture_recognizer.task # MediaPipe gesture recognition model
├── data/                       # Data files
│   └── test_images/            # Test images for gesture recognition
├── docs/                       # Documentation
│   ├── doc.md
│   └── report.md
├── scripts/                    # Training and utility scripts
│   ├── train.py                # Model training script
│   └── test_inference.py       # Batch inference testing
├── .env.example                # Environment variables template
└── requirements.txt            # Python dependencies
```

## Setup

1. **Create virtual environment:**

   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

2. **Install dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

3. **Configure API key (optional):**

   ```bash
   cp .env.example .env
   # Edit .env and add your OMDb API key
   # Get a free key at: https://www.omdbapi.com/apikey.aspx
   ```

## Usage

### Full Application (with mock data)

```bash
source venv/bin/activate
python src/main.py --mock
```

### Full Application (with real API)

```bash
source .env  # Load your API key
python src/main.py
```

### Testing Modes

```bash
# Use keyboard input instead of speech
python src/main.py --mock --keyboard

# Different speech language
python src/main.py --mock --language en-US
```

## CLI Options

| Flag | Description |
|------|-------------|
| `--mock, -m` | Use mock film data (no API key needed) |
| `--keyboard, -k` | Use keyboard input instead of speech |
| `--language, -l` | Language for speech recognition (default: es-ES) |

## Mock Films Available

When using `--mock` mode, these films are available:

- "matrix" / "Matrix"
- "inception" / "Inception"  
- "interestelar" / "Interstellar"
- "pulp fiction" / "Pulp Fiction"
- "el padrino" / "The Godfather"
