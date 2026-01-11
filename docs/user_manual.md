# 🎬 Film Rating System - User Manual

A multimodal app that lets you **speak a film name**, then **rate it with hand gestures**.

---

## Quick Start

```bash
# 1. Activate virtual environment
source venv/bin/activate

# 2. Run in test mode (no API key or camera needed)
python src/main.py --mock --keyboard
```

---

## How It Works

1. **Say a film name** → The app listens for your voice
2. **Film info appears** → Fetched from OMDb API
3. **Show 1-5 fingers** → Rate the film with hand gestures
4. **See your rating** → Displayed with the movie poster

---

## Running Modes

### 🧪 Full Test Mode (no hardware needed)

```bash
python src/main.py --mock --keyboard
```

Uses mock data for everything. Type the film name instead of speaking.

### 🎤 With Speech Recognition

```bash
python src/main.py --mock
```

Uses your computer's microphone. Say the film name in Spanish.

### 📱 With Phone Camera

```bash
# Set your phone's IP webcam URL in .env
CAMERA_URL=http://192.168.1.100:8080/video

# Then run
source .env
python src/main.py --mock-filmdb
```

### 🌐 Full Production Mode

```bash
# Set your OMDb API key in .env
OMDB_API_KEY=your_key_here

source .env
python src/main.py
```

---

## CLI Flags Reference

| Flag | Short | Description |
|------|-------|-------------|
| `--mock` | `-m` | Enable all mock modes |
| `--mock-camera` | | Use test images instead of camera |
| `--mock-filmdb` | | Use mock film data instead of API |
| `--keyboard` | `-k` | Type film name instead of speaking |
| `--camera-url URL` | `-c` | Phone camera stream URL |
| `--language CODE` | `-l` | Speech language (default: `es-ES`) |

---

## Available Mock Films

When using `--mock-filmdb`, these films are available:

| Say | Gets |
|-----|------|
| "Matrix" | The Matrix (1999) |
| "Inception" | Inception (2010) |
| "Interestelar" | Interstellar (2014) |
| "Pulp Fiction" | Pulp Fiction (1994) |
| "El Padrino" | The Godfather (1972) |

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

## Troubleshooting

### "No speech detected"

- Check your microphone is working
- Speak clearly after the "🎤 Speak now" prompt
- Try `--keyboard` mode instead

### "Could not open camera"

- Your webcam may be in use by another app
- Try `--mock-camera` to use test images
- For phone camera, check the URL and WiFi connection

### "Film not found"

- Check the spelling
- In mock mode, only 5 films are available (see list above)
- In production mode, ensure your API key is set

---

## File Locations

| What | Where |
|------|-------|
| Main app | `src/main.py` |
| Test images | `data/test_images/` |
| Gesture model | `models/gesture_recognizer.task` |
| Environment vars | `.env` |
