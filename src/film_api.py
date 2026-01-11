"""
Film data fetcher module for the Film Rating System.
Uses the OMDb API to fetch movie information.
"""

import os
from typing import Optional

import requests

from models import FilmData


class FilmFetcher:
    """Fetches film data from OMDb API."""
    
    def __init__(self):
        self.api_key = os.environ.get("OMDB_API_KEY", "")
        self.base_url = "http://www.omdbapi.com/"
        
        if not self.api_key:
            print("❌ OMDB_API_KEY not set!")
            print("   Get a free key at: https://www.omdbapi.com/apikey.aspx")
            print("   Then set it: export OMDB_API_KEY='your_key'")
    
    def search(self, title: str) -> Optional[FilmData]:
        """
        Search for a film by title.
        
        Args:
            title: The film title to search for.
        
        Returns:
            FilmData object or None if not found.
        """
        if not self.api_key:
            return None
            
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
