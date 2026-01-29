"""
Display module for the Film Rating System.
Handles poster display and result visualization.
"""

import time
from typing import Optional
from io import BytesIO

import cv2
import numpy as np
import requests
from PIL import Image

from .models import FilmData


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
        # Use a large canvas for fullscreen display
        canvas_width = 1200
        canvas_height = 900
        canvas = np.zeros((canvas_height, canvas_width, 3), dtype=np.uint8)
        canvas[:] = (30, 30, 30)  # Dark gray background
        
        # Download and add poster
        poster = DisplayManager.download_poster(film.poster_url)
        if poster is not None:
            # Resize poster to fit
            poster_height = 650
            aspect = poster.shape[1] / poster.shape[0]
            poster_width = int(poster_height * aspect)
            poster_resized = cv2.resize(poster, (poster_width, poster_height))
            
            # Center the poster
            x_offset = (canvas_width - poster_width) // 2
            y_offset = 30
            
            # Paste poster onto canvas
            canvas[y_offset:y_offset+poster_height, 
                   x_offset:x_offset+poster_width] = poster_resized
        
        # Add text info below poster
        y_text = 720
        
        # Title (larger font)
        cv2.putText(canvas, film.title, (40, y_text),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
        
        # Year and runtime
        cv2.putText(canvas, f"{film.year} | {film.runtime}", (40, y_text + 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (180, 180, 180), 1)
        
        # IMDb rating
        cv2.putText(canvas, f"IMDb: {film.imdb_rating}", (40, y_text + 75),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 215, 0), 2)
        
        # User score with ASCII stars (OpenCV can't render Unicode)
        stars_filled = "[*]" * user_score
        stars_empty = "[ ]" * (5 - user_score)
        score_text = f"Your Score: {stars_filled}{stars_empty} ({user_score}/5)"
        cv2.putText(canvas, score_text, (40, y_text + 120),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 100), 2)
        
        return canvas
    
    @staticmethod
    def show_result(film: FilmData, user_score: int, window_name: str = "Film Rating Result"):
        """Display the final result in a fullscreen window."""
        display = DisplayManager.create_rating_display(film, user_score)
        
        print(f"\n{'='*50}")
        print(f"🎬 {film.title} ({film.year})")
        print(f"🎭 {film.genre}")
        print(f"🎥 Director: {film.director}")
        print(f"⭐ IMDb Rating: {film.imdb_rating}")
        print(f"{'='*50}")
        print(f"📊 YOUR SCORE: {'★' * user_score}{'☆' * (5 - user_score)} ({user_score}/5)")
        print(f"{'='*50}")
        
        # Create fullscreen window
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        cv2.moveWindow(window_name, 0, 0)
        cv2.imshow(window_name, display)
        cv2.resizeWindow(window_name, 1920, 1080)
        
        print("\nPress any key to close (auto-closes in 10 seconds)...")
        
        # Loop with timeout for key events
        start_time = time.time()
        timeout = 10
        while True:
            key = cv2.waitKey(100)
            if key != -1:
                break
            if time.time() - start_time > timeout:
                print("Auto-closing window...")
                break
            if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                break
        
        cv2.destroyAllWindows()
