"""
Data models for the Film Rating System.
"""

from dataclasses import dataclass


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
