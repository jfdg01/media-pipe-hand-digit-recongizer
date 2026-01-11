# 🎬 Film Rating System - User Manual

A multimodal app that lets you **speak a film name**, then **rate it with hand gestures**.

---

## Quick Start

```bash
# 1. Activate virtual environment
source venv/bin/activate

# 2. Run the app (environment variables are loaded automatically from .env)
python src/main.py
```

---

## How It Works

1. **Say a film name** → The app listens for your voice
2. **Film info appears** → Fetched from OMDb API
3. **Show 1-5 fingers** → Rate the film with hand gestures
4. **See your rating** → Displayed with the movie poster

---

## Environment Setup

Create a `.env` file (copy from `.env.example`):

```bash
cp .env.example .env
```

Then edit `.env` with your settings:

```bash
# Required: OMDb API key (get free at https://www.omdbapi.com/apikey.aspx)
OMDB_API_KEY=your_api_key_here

# Optional: Phone camera URL (if using IP Webcam app)
CAMERA_URL=http://192.168.1.100:8080/video
```

**Note:** The app automatically loads `.env` - no need to run `source .env`!

---

## Running Modes

### 🎤 Full Experience (Speech + Camera)

```bash
python -m src.main
```

Uses your microphone for speech and camera for gestures.

### ⌨️ Keyboard Mode

```bash
python -m src.main --keyboard
```

Type the film name instead of speaking.

### 📱 With Phone Camera

Set `CAMERA_URL` in your `.env` file, then run normally.

### 🌍 Different Language

```bash
python -m src.main --language en-US
```

---

## CLI Flags Reference

| Flag | Short | Description |
|------|-------|-------------|
| `--keyboard` | `-k` | Type film name instead of speaking |
| `--camera-url URL` | `-c` | Phone camera stream URL (overrides .env) |
| `--language CODE` | `-l` | Speech language (default: `es-ES`) |

---

## Hand Gestures

Show fingers to rate:

| Gesture | Score |
|---------|-------|
| ☝️ 1 finger | ★☆☆☆☆ |
| ✌️ 2 fingers | ★★☆☆☆ |
| 🤟 3 fingers | ★★★☆☆ |
| 🖐️ 4 fingers | ★★★★☆ |
| ✋ 5 fingers | ★★★★★ |

Hold the gesture steady for 1-2 seconds until recognized.

---

## Phone Camera Setup

### Android (IP Webcam app)

1. Install "IP Webcam" from Play Store
2. Open app → scroll down → tap "Start server"
3. Note the URL shown (e.g., `http://192.168.1.100:8080`)
4. Add to your `.env`: `CAMERA_URL=http://192.168.1.100:8080/video`

### iOS

Use "EpocCam" or "Camo" app (similar process).

---

## Project Structure

```
src/
├── main.py         # CLI entry point
├── controller.py   # Main orchestration
├── models.py       # Data classes
├── speech.py       # Speech recognition
├── film_api.py     # OMDb API integration
├── gestures.py     # Hand gesture recognition
└── display.py      # Result visualization
```

---

## Troubleshooting

### "OMDB_API_KEY not set"

- Get a free key at <https://www.omdbapi.com/apikey.aspx>
- Add to `.env`: `OMDB_API_KEY=your_key`

### "No speech detected"

- Check your microphone is working
- Speak clearly after the "🎤 Speak now" prompt
- Try `--keyboard` mode instead

### "Could not open camera"

- Your webcam may be in use by another app
- For phone camera, check the URL and WiFi connection
- Make sure IP Webcam app is running on your phone

### "ImportError: attempted relative import"

- Run as a module: `python -m src.main` (not `python src/main.py`)

### "Film not found"

- Check the spelling
- Try the English title
