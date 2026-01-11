#!/usr/bin/env python3
"""
Film Rating Controller

Orchestrates the multimodal film rating experience:
1. User speaks the name of a film (speech recognition)
2. API fetches film data (OMDb API)
3. User is prompted to score the film
4. User shows 1-5 with hand gesture (gesture recognition)
5. Movie poster is displayed with the user's score
"""

import os
import sys
import time
from dataclasses import dataclass
from typing import Optional
from io import BytesIO

import cv2
import numpy as np
import requests
import speech_recognition as sr
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from PIL import Image


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class FilmData:
    """Represents film data from the API."""
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


# =============================================================================
# Mock Data for Testing
# =============================================================================

MOCK_FILMS = {
    "matrix": FilmData(
        title="The Matrix",
        year="1999",
        director="Lana Wachowski, Lilly Wachowski",
        genre="Action, Sci-Fi",
        plot="A computer hacker learns about the true nature of reality and his role in the war against its controllers.",
        poster_url="https://m.media-amazon.com/images/M/MV5BNzQzOTk3OTAtNDQ0Zi00ZTVkLWI0MTEtMDllZjNkYzNjNTc4L2ltYWdlXkEyXkFqcGdeQXVyNjU0OTQ0OTY@._V1_SX300.jpg",
        imdb_rating="8.7",
        runtime="136 min"
    ),
    "inception": FilmData(
        title="Inception",
        year="2010",
        director="Christopher Nolan",
        genre="Action, Adventure, Sci-Fi",
        plot="A thief who steals corporate secrets through dream-sharing technology is given the inverse task of planting an idea.",
        poster_url="https://m.media-amazon.com/images/M/MV5BMjAxMzY3NjcxNF5BMl5BanBnXkFtZTcwNTI5OTM0Mw@@._V1_SX300.jpg",
        imdb_rating="8.8",
        runtime="148 min"
    ),
    "interestelar": FilmData(
        title="Interstellar",
        year="2014",
        director="Christopher Nolan",
        genre="Adventure, Drama, Sci-Fi",
        plot="A team of explorers travel through a wormhole in space to ensure humanity's survival.",
        poster_url="https://m.media-amazon.com/images/M/MV5BZjdkOTU3MDktN2IxOS00OGEyLWFmMjktY2FiMmZkNWIyODZiXkEyXkFqcGdeQXVyMTMxODk2OTU@._V1_SX300.jpg",
        imdb_rating="8.6",
        runtime="169 min"
    ),
    "pulp fiction": FilmData(
        title="Pulp Fiction",
        year="1994",
        director="Quentin Tarantino",
        genre="Crime, Drama",
        plot="The lives of two mob hitmen, a boxer, a gangster and his wife, and a pair of diner bandits intertwine.",
        poster_url="https://m.media-amazon.com/images/M/MV5BNGNhMDIzZTUtNTBlZi00MTRlLWFjM2ItYzViMjE3YzI5MjljXkEyXkFqcGdeQXVyNzkwMjQ5NzM@._V1_SX300.jpg",
        imdb_rating="8.9",
        runtime="154 min"
    ),
    "el padrino": FilmData(
        title="The Godfather",
        year="1972",
        director="Francis Ford Coppola",
        genre="Crime, Drama",
        plot="The aging patriarch of an organized crime dynasty transfers control to his reluctant youngest son.",
        poster_url="https://m.media-amazon.com/images/M/MV5BM2MyNjYxNmUtYTAwNi00MTYxLWJmNWYtYzZlODY3ZTk3OTFlXkEyXkFqcGdeQXVyNzkwMjQ5NzM@._V1_SX300.jpg",
        imdb_rating="9.2",
        runtime="175 min"
    ),
}


# =============================================================================
# Speech Recognition Module
# =============================================================================

class SpeechRecognizer:
    """Handles speech-to-text conversion."""
    
    def __init__(self, language: str = "es-ES"):
        self.language = language
        self.recognizer = sr.Recognizer()
    
    def listen(self, prompt: str = "🎤 Listening... Speak now!") -> Optional[str]:
        """
        Listen for speech and convert to text.
        
        Args:
            prompt: Message to display before listening.
        
        Returns:
            Recognized text or None if failed.
        """
        with sr.Microphone() as source:
            print("Adjusting for ambient noise...")
            self.recognizer.adjust_for_ambient_noise(source, duration=1)
            
            print(prompt)
            
            try:
                audio = self.recognizer.listen(source, timeout=10, phrase_time_limit=10)
                print("Processing speech...")
                
                text = self.recognizer.recognize_google(audio, language=self.language)
                print(f"✅ Recognized: \"{text}\"")
                return text
                
            except sr.WaitTimeoutError:
                print("❌ No speech detected within timeout.")
            except sr.UnknownValueError:
                print("❌ Could not understand the audio.")
            except sr.RequestError as e:
                print(f"❌ Speech API error: {e}")
        
        return None


# =============================================================================
# Film Data Fetcher Module
# =============================================================================

class FilmFetcher:
    """Fetches film data from OMDb API or mock data."""
    
    def __init__(self, use_mock: bool = False):
        self.use_mock = use_mock
        self.api_key = os.environ.get("OMDB_API_KEY", "")
        self.base_url = "http://www.omdbapi.com/"
        
        if not self.api_key and not self.use_mock:
            print("⚠️  OMDB_API_KEY not set. Using mock data.")
            self.use_mock = True
    
    def search(self, title: str) -> Optional[FilmData]:
        """
        Search for a film by title.
        
        Args:
            title: The film title to search for.
        
        Returns:
            FilmData object or None if not found.
        """
        if self.use_mock:
            return self._search_mock(title)
        return self._search_api(title)
    
    def _search_mock(self, title: str) -> Optional[FilmData]:
        """Search in mock data."""
        title_lower = title.lower().strip()
        
        # Direct match
        if title_lower in MOCK_FILMS:
            return MOCK_FILMS[title_lower]
        
        # Partial match
        for key, film in MOCK_FILMS.items():
            if title_lower in key or key in title_lower:
                return film
            if title_lower in film.title.lower():
                return film
        
        # Return a default mock film
        print(f"⚠️  '{title}' not in mock data. Returning 'The Matrix' as fallback.")
        return MOCK_FILMS["the matrix"]
    
    def _search_api(self, title: str) -> Optional[FilmData]:
        """Search using the real OMDb API."""
        params = {
            "apikey": self.api_key,
            "t": title,
            "plot": "short"
        }
        
        try:
            response = requests.get(self.base_url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            if data.get("Response") == "False":
                print(f"❌ Film not found: {data.get('Error')}")
                return None
            
            return FilmData(
                title=data.get("Title", "Unknown"),
                year=data.get("Year", "N/A"),
                director=data.get("Director", "Unknown"),
                genre=data.get("Genre", "N/A"),
                plot=data.get("Plot", "No plot available."),
                poster_url=data.get("Poster", ""),
                imdb_rating=data.get("imdbRating", "N/A"),
                runtime=data.get("Runtime", "N/A")
            )
            
        except requests.exceptions.RequestException as e:
            print(f"❌ API request failed: {e}")
            return None


# =============================================================================
# Gesture Recognition Module
# =============================================================================

class GestureRecognizer:
    """Handles hand gesture recognition for scoring 1-5."""
    
    # Mapping from gesture names to numeric scores
    GESTURE_TO_SCORE = {
        "one": 1,
        "two": 2,
        "three": 3,
        "four": 4,
        "five": 5
    }
    
    def __init__(self, model_path: str = "digit_model_v1/gesture_recognizer.task"):
        self.model_path = model_path
        self.recognizer = None
        
        if os.path.exists(model_path):
            base_options = python.BaseOptions(model_asset_path=model_path)
            options = vision.GestureRecognizerOptions(base_options=base_options)
            self.recognizer = vision.GestureRecognizer.create_from_options(options)
        else:
            print(f"⚠️  Gesture model not found at {model_path}")
    
    def recognize_from_camera(self, timeout_seconds: int = 15) -> Optional[int]:
        """
        Open camera and wait for a valid gesture (1-5).
        
        Args:
            timeout_seconds: Maximum time to wait for a gesture.
        
        Returns:
            Score (1-5) or None if failed/timeout.
        """
        if not self.recognizer:
            print("❌ Gesture recognizer not initialized.")
            return self._fallback_input()
        
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("❌ Could not open camera.")
            return self._fallback_input()
        
        print("\n✋ Show your score (1-5 fingers) to the camera...")
        print("   Hold your gesture steady for recognition.")
        print(f"   (Timeout in {timeout_seconds} seconds)\n")
        
        start_time = time.time()
        stable_gesture = None
        stable_count = 0
        required_stable_frames = 10  # Need 10 consistent frames
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Check timeout
                elapsed = time.time() - start_time
                if elapsed > timeout_seconds:
                    print("⏰ Timeout reached.")
                    break
                
                # Convert frame for MediaPipe
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
                
                # Recognize gesture
                result = self.recognizer.recognize(mp_image)
                
                current_gesture = None
                confidence = 0.0
                
                if result.gestures:
                    category = result.gestures[0][0]
                    gesture_name = category.category_name
                    confidence = category.score
                    
                    if gesture_name in self.GESTURE_TO_SCORE and confidence > 0.7:
                        current_gesture = gesture_name
                
                # Check for stable gesture
                if current_gesture == stable_gesture and current_gesture is not None:
                    stable_count += 1
                else:
                    stable_gesture = current_gesture
                    stable_count = 1 if current_gesture else 0
                
                # Display info on frame
                display_text = f"Gesture: {stable_gesture or 'None'} ({stable_count}/{required_stable_frames})"
                cv2.putText(frame, display_text, (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                cv2.putText(frame, f"Time: {int(timeout_seconds - elapsed)}s", (10, 60),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                
                cv2.imshow("Show Your Score (1-5)", frame)
                
                # If we have a stable gesture
                if stable_count >= required_stable_frames:
                    score = self.GESTURE_TO_SCORE[stable_gesture]
                    print(f"✅ Gesture recognized: {stable_gesture} (Score: {score})")
                    cap.release()
                    cv2.destroyAllWindows()
                    return score
                
                # Check for 'q' to quit
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("❌ Cancelled by user.")
                    break
                    
        finally:
            cap.release()
            cv2.destroyAllWindows()
        
        return self._fallback_input()
    
    def _fallback_input(self) -> Optional[int]:
        """Fallback to keyboard input if camera/model fails."""
        print("\n⌨️  Falling back to keyboard input.")
        try:
            score = int(input("Enter your score (1-5): "))
            if 1 <= score <= 5:
                return score
            print("❌ Score must be between 1 and 5.")
        except ValueError:
            print("❌ Invalid input.")
        return None


# =============================================================================
# Display Module
# =============================================================================

class DisplayManager:
    """Handles displaying film posters and results."""
    
    @staticmethod
    def download_poster(url: str) -> Optional[np.ndarray]:
        """Download poster image from URL."""
        if not url or url == "N/A":
            return None
        
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            
            # Convert to numpy array for OpenCV
            image = Image.open(BytesIO(response.content))
            image = image.convert("RGB")
            return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            
        except Exception as e:
            print(f"⚠️  Could not download poster: {e}")
            return None
    
    @staticmethod
    def create_rating_display(film: FilmData, user_score: int) -> np.ndarray:
        """Create a display image with film info and user score."""
        # Create a canvas
        canvas_width = 600
        canvas_height = 500
        canvas = np.zeros((canvas_height, canvas_width, 3), dtype=np.uint8)
        canvas[:] = (30, 30, 30)  # Dark gray background
        
        # Download and add poster
        poster = DisplayManager.download_poster(film.poster_url)
        if poster is not None:
            # Resize poster to fit
            poster_height = 350
            aspect = poster.shape[1] / poster.shape[0]
            poster_width = int(poster_height * aspect)
            poster_resized = cv2.resize(poster, (poster_width, poster_height))
            
            # Center the poster
            x_offset = (canvas_width - poster_width) // 2
            y_offset = 20
            
            # Paste poster onto canvas
            canvas[y_offset:y_offset+poster_height, 
                   x_offset:x_offset+poster_width] = poster_resized
        
        # Add text info
        y_text = 400
        
        # Title
        cv2.putText(canvas, film.title, (20, y_text),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # Year and runtime
        cv2.putText(canvas, f"{film.year} | {film.runtime}", (20, y_text + 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)
        
        # IMDb rating
        cv2.putText(canvas, f"IMDb: {film.imdb_rating}", (20, y_text + 55),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 215, 0), 1)
        
        # User score with stars
        stars = "★" * user_score + "☆" * (5 - user_score)
        cv2.putText(canvas, f"Your Score: {stars}", (20, y_text + 85),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 100), 2)
        
        return canvas
    
    @staticmethod
    def show_result(film: FilmData, user_score: int, window_name: str = "Film Rating Result"):
        """Display the final result in a window."""
        display = DisplayManager.create_rating_display(film, user_score)
        
        print(f"\n{'='*50}")
        print(f"🎬 {film.title} ({film.year})")
        print(f"🎭 {film.genre}")
        print(f"🎥 Director: {film.director}")
        print(f"⭐ IMDb Rating: {film.imdb_rating}")
        print(f"{'='*50}")
        print(f"📊 YOUR SCORE: {'★' * user_score}{'☆' * (5 - user_score)} ({user_score}/5)")
        print(f"{'='*50}")
        
        cv2.imshow(window_name, display)
        print("\nPress any key to close...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()


# =============================================================================
# Main Controller
# =============================================================================

class FilmRatingController:
    """
    Main controller that orchestrates the multimodal film rating experience.
    """
    
    def __init__(self, use_mock: bool = False, language: str = "es-ES"):
        """
        Initialize the controller.
        
        Args:
            use_mock: If True, use mock film data instead of real API.
            language: Language code for speech recognition.
        """
        self.speech = SpeechRecognizer(language=language)
        self.films = FilmFetcher(use_mock=use_mock)
        self.gestures = GestureRecognizer()
        self.display = DisplayManager()
    
    def run(self):
        """Run the complete film rating flow."""
        print("\n" + "="*60)
        print("🎬  MULTIMODAL FILM RATING SYSTEM")
        print("="*60)
        print("This system uses:")
        print("  • 🎤 Speech recognition to get film name")
        print("  • 🌐 OMDb API to fetch film data")
        print("  • ✋ Hand gestures to input your rating (1-5)")
        print("="*60 + "\n")
        
        # Step 1: Get film name via speech
        print("📢 STEP 1: Say the name of a film")
        print("-" * 40)
        
        film_title = self.speech.listen("🎤 Speak the film title now...")
        
        if not film_title:
            print("❌ Could not get film title. Exiting.")
            return
        
        # Step 2: Fetch film data
        print(f"\n🔍 STEP 2: Searching for '{film_title}'...")
        print("-" * 40)
        
        film = self.films.search(film_title)
        
        if not film:
            print(f"❌ Could not find film: '{film_title}'. Exiting.")
            return
        
        print(f"✅ Found: {film}")
        print(f"   Genre: {film.genre}")
        print(f"   Plot: {film.plot[:100]}...")
        
        # Step 3: Get user score via gesture
        print(f"\n📊 STEP 3: Rate '{film.title}'")
        print("-" * 40)
        
        score = self.gestures.recognize_from_camera()
        
        if not score:
            print("❌ Could not get score. Exiting.")
            return
        
        # Step 4: Display result
        print(f"\n🖼️  STEP 4: Displaying result...")
        print("-" * 40)
        
        self.display.show_result(film, score)
        
        print("\n✅ Thank you for using the Film Rating System!")


# =============================================================================
# CLI Entry Point
# =============================================================================

def main():
    """Main entry point with CLI arguments."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Multimodal Film Rating System",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        "--mock", "-m",
        action="store_true",
        help="Use mock film data (for testing without API key)"
    )
    parser.add_argument(
        "--language", "-l",
        default="es-ES",
        help="Language code for speech recognition (default: es-ES)"
    )
    parser.add_argument(
        "--keyboard", "-k",
        action="store_true",
        help="Use keyboard input instead of speech (for testing)"
    )
    
    args = parser.parse_args()
    
    controller = FilmRatingController(use_mock=args.mock, language=args.language)
    
    # Override speech with keyboard input if requested
    if args.keyboard:
        original_listen = controller.speech.listen
        def keyboard_listen(prompt=""):
            print(prompt)
            return input("Enter film title: ").strip()
        controller.speech.listen = keyboard_listen
    
    controller.run()


if __name__ == "__main__":
    main()
