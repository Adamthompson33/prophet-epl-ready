"""
OpenWeatherMap loader for Prophet EPL.
Fetches weather forecast for EPL venues before match kickoff.
PIT-safe: only uses forecasts, never post-match actuals.
"""

import os
import json
import requests
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

# Configuration
CACHE_DIR = Path(__file__).parent / "data_cache"
API_KEY = os.getenv("OPENWEATHERMAP_API_KEY")
BASE_URL = "https://api.openweathermap.org/data/2.5/forecast"

# EPL venue name to OpenWeatherMap city mapping
VENUE_MAP = {
    "Craven Cottage, London": "London",
    "Emirates Stadium, London": "London",
    "Stamford Bridge, London": "London",
    "Tottenham Hotspur Stadium, London": "London",
    "Anfield, Liverpool": "Liverpool",
    "Goodison Park, Liverpool": "Liverpool",
    "Old Trafford, Manchester": "Manchester",
    "Etihad Stadium, Manchester": "Manchester",
    "St James' Park, Newcastle": "Newcastle upon Tyne",
    "Villa Park, Birmingham": "Birmingham",
    "Stamford Bridge, London": "London",
    "Molineux Stadium, Wolverhampton": "Wolverhampton",
    "Elland Road, Leeds": "Leeds",
    "King Power Stadium, Leicester": "Leicester",
    "Vitality Stadium, Bournemouth": "Bournemouth",
    "Brentford Community Stadium, London": "London",
    "St. Mary's Stadium, Southampton": "Southampton",
    "American Express Stadium, Brighton": "Brighton",
    "City Ground, Nottingham": "Nottingham",
    "Selhurst Park, London": "London",
    "Vicarage Road, Watford": "Watford",
    "Kenilworth Road, Luton": "Luton",
    "Gtech Community Stadium, Brentford": "London",
}


def get_cache_path(venue: str, date: str) -> Path:
    """Get cache file path for a venue/date combo."""
    CACHE_DIR.mkdir(exist_ok=True)
    safe_venue = venue.replace(",", "").replace(" ", "_").replace("/", "_")
    return CACHE_DIR / f"{safe_venue}_{date}.json"


def fetch_weather(venue: str, match_date: str) -> Optional[dict]:
    """
    Fetch weather forecast for a venue on a given date.
    
    Args:
        venue: Full venue name (e.g., "Craven Cottage, London")
        match_date: Date in YYYY-MM-DD format
    
    Returns:
        dict with temperature, precipitation, wind_speed or None if unavailable
    """
    if not API_KEY:
        return {"error": "OPENWEATHERMAP_API_KEY not set"}
    
    # Check cache first
    cache_path = get_cache_path(venue, match_date)
    if cache_path.exists():
        with open(cache_path) as f:
            return json.load(f)
    
    # Map venue to city
    city = VENUE_MAP.get(venue, "London")
    
    try:
        # Get forecast for the date
        params = {
            "q": city,
            "appid": API_KEY,
            "units": "metric"
        }
        response = requests.get(BASE_URL, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        # Parse the forecast list to find closest to match time
        # Forecast is in 3-hour blocks, find the one closest to match day 15:00
        target_date = datetime.strptime(match_date, "%Y-%m-%d")
        
        best_forecast = None
        min_diff = float('inf')
        
        for item in data.get("list", []):
            forecast_time = datetime.fromtimestamp(item["dt"])
            diff = abs((forecast_time - target_date).total_seconds())
            if diff < min_diff:
                min_diff = diff
                best_forecast = item
        
        if not best_forecast:
            return {"error": "No forecast available"}
        
        # Extract relevant fields
        result = {
            "venue": venue,
            "date": match_date,
            "city": city,
            "forecast_time": best_forecast["dt_txt"],
            "temperature": best_forecast["main"]["temp"],
            "feels_like": best_forecast["main"]["feels_like"],
            "humidity": best_forecast["main"]["humidity"],
            "precipitation_mm": best_forecast.get("rain", {}).get("3h", 0),
            "wind_speed_ms": best_forecast["wind"]["speed"],
            "wind_direction": best_forecast["wind"].get("deg", None),
            "description": best_forecast["weather"][0]["description"],
            "icon": best_forecast["weather"][0]["icon"],
        }
        
        # Cache the result
        with open(cache_path, "w") as f:
            json.dump(result, f, indent=2)
        
        return result
        
    except requests.RequestException as e:
        return {"error": str(e)}
    except (KeyError, json.JSONDecodeError) as e:
        return {"error": f"Parse error: {str(e)}"}


def test_weather_loader():
    """Test with Craven Cottage."""
    result = fetch_weather("Craven Cottage, London", "2026-03-21")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    test_weather_loader()
