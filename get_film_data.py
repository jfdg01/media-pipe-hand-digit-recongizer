#!/usr/bin/env python3
"""
Film Data Fetcher
Fetches film information from the OMDb API based on a movie title.

Usage:
    python get_film_data.py "The Matrix"
    python get_film_data.py --title "Inception" --year 2010
    python get_film_data.py  # Interactive mode
"""

import argparse
import json
import os
import sys
from typing import Optional

import requests


# Get API key from environment variable
OMDB_API_KEY = os.environ.get("OMDB_API_KEY", "")
OMDB_BASE_URL = "http://www.omdbapi.com/"


def fetch_film_data(title: str, year: Optional[int] = None, plot: str = "full") -> dict:
    """
    Fetch film data from OMDb API.
    
    Args:
        title: The title of the film to search for.
        year: Optional year of release to narrow down the search.
        plot: "short" or "full" - determines the length of the plot summary.
    
    Returns:
        Dictionary containing the film data, or error information.
    """
    params = {
        "apikey": OMDB_API_KEY,
        "t": title,  # Search by title
        "plot": plot,
    }
    
    if year:
        params["y"] = year
    
    try:
        response = requests.get(OMDB_BASE_URL, params=params, timeout=10)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.Timeout:
        return {"Response": "False", "Error": "Request timed out"}
    except requests.exceptions.RequestException as e:
        return {"Response": "False", "Error": f"Request failed: {str(e)}"}


def format_film_output(data: dict) -> str:
    """
    Format the film data into a readable string.
    
    Args:
        data: Dictionary containing the film data from OMDb.
    
    Returns:
        Formatted string representation of the film data.
    """
    if data.get("Response") == "False":
        return f"❌ Error: {data.get('Error', 'Unknown error')}"
    
    lines = [
        "=" * 60,
        f"🎬 {data.get('Title', 'N/A')} ({data.get('Year', 'N/A')})",
        "=" * 60,
        "",
        f"📅 Released:    {data.get('Released', 'N/A')}",
        f"⏱️  Runtime:     {data.get('Runtime', 'N/A')}",
        f"🎭 Genre:       {data.get('Genre', 'N/A')}",
        f"🎥 Director:    {data.get('Director', 'N/A')}",
        f"✍️  Writer:      {data.get('Writer', 'N/A')}",
        f"🌟 Actors:      {data.get('Actors', 'N/A')}",
        "",
        f"📖 Plot:",
        f"   {data.get('Plot', 'N/A')}",
        "",
        f"🌍 Country:     {data.get('Country', 'N/A')}",
        f"🗣️  Language:    {data.get('Language', 'N/A')}",
        f"🏆 Awards:      {data.get('Awards', 'N/A')}",
        "",
    ]
    
    # Add ratings if available
    ratings = data.get("Ratings", [])
    if ratings:
        lines.append("⭐ Ratings:")
        for rating in ratings:
            lines.append(f"   • {rating.get('Source', 'N/A')}: {rating.get('Value', 'N/A')}")
        lines.append("")
    
    # Add additional info
    lines.extend([
        f"📊 Metascore:   {data.get('Metascore', 'N/A')}",
        f"⭐ IMDb Rating: {data.get('imdbRating', 'N/A')} ({data.get('imdbVotes', 'N/A')} votes)",
        f"🔗 IMDb ID:     {data.get('imdbID', 'N/A')}",
        f"📺 Type:        {data.get('Type', 'N/A')}",
    ])
    
    # Add box office if available
    if data.get("BoxOffice"):
        lines.append(f"💰 Box Office:  {data.get('BoxOffice')}")
    
    # Add poster URL if available
    if data.get("Poster") and data.get("Poster") != "N/A":
        lines.extend(["", f"🖼️  Poster URL: {data.get('Poster')}"])
    
    lines.append("=" * 60)
    
    return "\n".join(lines)


def output_as_json(data: dict) -> str:
    """Output the film data as formatted JSON."""
    return json.dumps(data, indent=2, ensure_ascii=False)


def main():
    parser = argparse.ArgumentParser(
        description="Fetch film data from OMDb API",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python get_film_data.py "The Matrix"
    python get_film_data.py --title "Inception" --year 2010
    python get_film_data.py --title "Breaking Bad" --json
        """
    )
    
    parser.add_argument(
        "title",
        nargs="?",
        help="The title of the film to search for"
    )
    parser.add_argument(
        "--title", "-t",
        dest="title_flag",
        help="The title of the film (alternative to positional argument)"
    )
    parser.add_argument(
        "--year", "-y",
        type=int,
        help="Year of release (optional, helps narrow down search)"
    )
    parser.add_argument(
        "--plot", "-p",
        choices=["short", "full"],
        default="full",
        help="Plot length: 'short' or 'full' (default: full)"
    )
    parser.add_argument(
        "--json", "-j",
        action="store_true",
        help="Output raw JSON instead of formatted text"
    )
    
    args = parser.parse_args()
    
    # Determine the title (positional argument takes precedence)
    title = args.title or args.title_flag
    
    # If no title provided, enter interactive mode
    if not title:
        print("🎬 Film Data Fetcher")
        print("-" * 40)
        title = input("Enter film title: ").strip()
        if not title:
            print("❌ Error: No title provided")
            sys.exit(1)
    
    # Check if API key is configured
    if not OMDB_API_KEY:
        print("❌ Error: OMDB_API_KEY environment variable is not set!")
        print("")
        print("   Get a free key at: https://www.omdbapi.com/apikey.aspx")
        print("   Then set it with:")
        print("     export OMDB_API_KEY='your_api_key_here'")
        print("")
        print("   Or add it to your .env file and source it.")
        sys.exit(1)
    
    # Fetch the film data
    print(f"🔍 Searching for: '{title}'..." if not args.json else "", file=sys.stderr)
    data = fetch_film_data(title, year=args.year, plot=args.plot)
    
    # Output the result
    if args.json:
        print(output_as_json(data))
    else:
        print(format_film_output(data))


if __name__ == "__main__":
    main()
