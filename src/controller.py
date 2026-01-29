"""
Main controller for the Film Rating System.
Orchestrates the multimodal film rating experience.
"""

from .speech import SpeechRecognizer
from .film_api import FilmFetcher
from .gestures import GestureRecognizer
from .display import DisplayManager


class FilmRatingController:
    """
    Main controller that orchestrates the multimodal film rating experience.
    """
    
    def __init__(self, language: str = "es-ES", camera_url: str = None):
        """
        Initialize the controller.
        
        Args:
            language: Language code for speech recognition.
            camera_url: URL for IP Webcam (phone camera).
        """
        self.speech = SpeechRecognizer(language=language)
        self.films = FilmFetcher()
        self.gestures = GestureRecognizer(camera_url=camera_url)
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
        
        score = self.gestures.recognize()
        
        if not score:
            print("❌ Could not get score. Exiting.")
            return
        
        # Step 4: Display result
        print(f"\n🖼️  STEP 4: Displaying result...")
        print("-" * 40)
        
        self.display.show_result(film, score)
        
        print("\n✅ Thank you for using the Film Rating System!")
