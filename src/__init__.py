"""
Film Rating System - Multimodal Application

Modules:
- models: Data classes (FilmData)
- speech: Speech recognition
- film_api: OMDb API integration
- gestures: Hand gesture recognition
- display: Result visualization
- controller: Main orchestration
"""

from .models import FilmData
from .speech import SpeechRecognizer
from .film_api import FilmFetcher
from .gestures import GestureRecognizer
from .display import DisplayManager
from .controller import FilmRatingController

__all__ = [
    'FilmData',
    'SpeechRecognizer', 
    'FilmFetcher',
    'GestureRecognizer',
    'DisplayManager',
    'FilmRatingController',
]
