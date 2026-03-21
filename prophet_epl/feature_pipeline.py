"""
Feature Engineering Pipeline for Prophet EPL
Combines weather, API-Football, and Brave Search data to create match features.
"""

import os
import json
import requests
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
import hashlib

# Import our existing weather loader
try:
    from data_ingest.weather_loader import fetch_weather
except ImportError:
    # Fallback if not in package
    import sys
    sys.path.append(str(Path(__file__).parent))
    from data_ingest.weather_loader import fetch_weather

# Configuration
CACHE_DIR = Path(__file__).parent / "feature_cache"
CACHE_DIR.mkdir(exist_ok=True)

API_FOOTBALL_KEY = os.getenv("API_FOOTBALL_KEY")
BRAVE_SEARCH_KEY = os.getenv("BRAVE_SEARCH_API_KEY")

# API endpoints
API_FOOTBALL_BASE = "https://v3.football.api-sports.io"
BRAVE_SEARCH_BASE = "https://api.search.brave.com/res/v1/web/search"

# Rate limiting
API_FOOTBALL_RATE_LIMIT = 10  # requests per minute (be conservative)
BRAVE_SEARCH_RATE_LIMIT = 60  # requests per minute
_last_api_call = {}  # Track last call time per endpoint


def rate_limit(endpoint_name: str, requests_per_minute: int):
    """Apply rate limiting to prevent API throttling."""
    now = time.time()
    
    if endpoint_name not in _last_api_call:
        _last_api_call[endpoint_name] = now
        return
    
    min_interval = 60.0 / requests_per_minute
    elapsed = now - _last_api_call[endpoint_name]
    
    if elapsed < min_interval:
        sleep_time = min_interval - elapsed
        time.sleep(sleep_time)
    
    _last_api_call[endpoint_name] = time.time()

def _make_cache_key(*args) -> str:
    """Create a cache key from arguments."""
    key_string = "_".join(str(arg) for arg in args)
    return hashlib.md5(key_string.encode()).hexdigest()

def get_cached_or_fetch(cache_key: str, fetch_func, *args, expire_hours: int = 6) -> Any:
    """Fetch data or return cached version if fresh."""
    cache_path = CACHE_DIR / f"{cache_key}.json"
    
    if cache_path.exists():
        # Check cache age
        cache_age = datetime.now().timestamp() - cache_path.stat().st_mtime
        if cache_age < expire_hours * 3600:
            with open(cache_path, 'r') as f:
                return json.load(f)
    
    # Fetch fresh data
    data = fetch_func(*args)
    
    # Cache it
    with open(cache_path, 'w') as f:
        json.dump(data, f, indent=2)
    
    return data

def fetch_api_football(endpoint: str, params: Dict) -> Dict:
    """Make a request to API-Football with rate limiting."""
    if not API_FOOTBALL_KEY:
        return {"error": "API_FOOTBALL_KEY not set"}
    
    # Apply rate limiting
    rate_limit("api_football", API_FOOTBALL_RATE_LIMIT)
    
    headers = {
        'x-rapidapi-key': API_FOOTBALL_KEY,
        'x-rapidapi-host': 'v3.football.api-sports.io'
    }
    
    try:
        response = requests.get(f"{API_FOOTBALL_BASE}/{endpoint}", headers=headers, params=params, timeout=15)
        response.raise_for_status()
        return response.json()
    except requests.RequestException as e:
        return {"error": str(e)}
    except json.JSONDecodeError as e:
        return {"error": f"JSON decode error: {str(e)}"}

def fetch_brave_search(query: str, count: int = 5) -> Dict:
    """Make a request to Brave Search with rate limiting."""
    if not BRAVE_SEARCH_KEY:
        return {"error": "BRAVE_SEARCH_API_KEY not set"}
    
    # Apply rate limiting
    rate_limit("brave_search", BRAVE_SEARCH_RATE_LIMIT)
    
    headers = {
        'X-Subscription-Token': BRAVE_SEARCH_KEY,
        'Accept': 'application/json'
    }
    
    params = {
        'q': query,
        'count': count,
        'safesearch': 'moderate'
    }
    
    try:
        response = requests.get(BRAVE_SEARCH_BASE, headers=headers, params=params, timeout=15)
        response.raise_for_status()
        return response.json()
    except requests.RequestException as e:
        return {"error": str(e)}
    except json.JSONDecodeError as e:
        return {"error": f"JSON decode error: {str(e)}"}

# Fallback venue mapping for when weather loader fails
VENUE_FALLBACK_MAP = {
    "Old Trafford": "Manchester",
    "Emirates Stadium": "London",
    "Stamford Bridge": "London",
    "Anfield": "Liverpool",
    "Goodison Park": "Liverpool",
    "Etihad Stadium": "Manchester",
    "Tottenham Hotspur Stadium": "London",
    "Villa Park": "Birmingham",
    "St James' Park": "Newcastle upon Tyne",
    "Molineux Stadium": "Wolverhampton",
    "Elland Road": "Leeds",
    "King Power Stadium": "Leicester",
    "Vitality Stadium": "Bournemouth",
    "Brentford Community Stadium": "London",
    "St. Mary's Stadium": "Southampton",
    "American Express Stadium": "Brighton",
    "City Ground": "Nottingham",
    "Selhurst Park": "London",
    "Craven Cottage": "London",
    "Kenilworth Road": "Luton",
    "Gtech Community Stadium": "London",
    "Vicarage Road": "Watford",
}


def get_weather_features(venue: str, match_date: str) -> Dict:
    """
    Get weather features for a match.
    Returns a dict with weather information or error.
    """
    # Try primary venue first
    venue_name = venue.split(",")[0].strip() if "," in venue else venue
    
    # Use our existing weather loader (PIT-safe)
    weather_data = fetch_weather(venue, match_date)
    
    # If error, try fallback city mapping
    if "error" in weather_data:
        fallback_city = VENUE_FALLBACK_MAP.get(venue_name)
        if fallback_city:
            fallback_venue = f"{fallback_city}, UK"
            weather_data = fetch_weather(fallback_venue, match_date)
        
        # If still error, return default features
        if "error" in weather_data:
            return {
                "weather_error": weather_data.get("error", "Unknown error"),
                "weather_fallback_used": bool(fallback_city),
                "temperature_c": None,
                "feels_like_c": None,
                "humidity_percent": None,
                "precipitation_mm": None,
                "wind_speed_m_s": None,
                "wind_direction_deg": None,
                "weather_description": "unavailable",
                "is_cold": False,
                "is_warm": False,
                "is_wet": False,
                "heavy_rain": False,
                "is_windy": False,
            }
    
    # Extract and format features we want
    features = {
        "temperature_c": weather_data.get("temperature"),
        "feels_like_c": weather_data.get("feels_like"),
        "humidity_percent": weather_data.get("humidity"),
        "precipitation_mm": weather_data.get("precipitation_mm", 0),
        "wind_speed_m_s": weather_data.get("wind_speed_ms"),
        "wind_direction_deg": weather_data.get("wind_direction"),
        "weather_description": weather_data.get("description"),
        "weather_icon": weather_data.get("icon"),
        "forecast_time": weather_data.get("forecast_time"),
        "city": weather_data.get("city"),
        "weather_fallback_used": False,
    }
    
    # Add derived features
    if features["temperature_c"] is not None:
        features["is_cold"] = features["temperature_c"] < 5.0
        features["is_warm"] = features["temperature_c"] > 20.0
    else:
        features["is_cold"] = False
        features["is_warm"] = False
    
    if features["precipitation_mm"] is not None:
        features["is_wet"] = features["precipitation_mm"] > 0.1
        features["heavy_rain"] = features["precipitation_mm"] > 5.0
    else:
        features["is_wet"] = False
        features["heavy_rain"] = False
    
    if features["wind_speed_m_s"] is not None:
        features["is_windy"] = features["wind_speed_m_s"] > 5.0  # ~11 mph
    else:
        features["is_windy"] = False
    
    return features

def get_fixture_features(fixture_id: int) -> Dict:
    """
    Get fixture-specific features from API-Football.
    """
    # Get fixture details
    fixture_data = fetch_api_football("fixtures", {"id": fixture_id})
    
    if "error" in fixture_data:
        return {"fixture_error": fixture_data["error"]}
    
    if not fixture_data.get("response"):
        return {"fixture_error": "No fixture data returned"}
    
    fixture = fixture_data["response"][0]
    
    # Extract basic fixture info
    features = {
        "fixture_id": fixture["fixture"]["id"],
        "referee": fixture["fixture"]["referee"],
        "timezone": fixture["fixture"]["timezone"],
        "status_long": fixture["fixture"]["status"]["long"],
        "status_short": fixture["fixture"]["status"]["short"],
        "elapsed": fixture["fixture"]["status"]["elapsed"],
        "venue_name": fixture["fixture"]["venue"]["name"],
        "venue_city": fixture["fixture"]["venue"]["city"],
        "league_id": fixture["league"]["id"],
        "league_name": fixture["league"]["name"],
        "country": fixture["league"]["country"],
        "season": fixture["league"]["season"],
        "round": fixture["league"]["round"],
    }
    
    # Team info
    features["home_team_id"] = fixture["teams"]["home"]["id"]
    features["home_team_name"] = fixture["teams"]["home"]["name"]
    features["home_team_logo"] = fixture["teams"]["home"]["logo"]
    features["home_team_winner"] = fixture["teams"]["home"]["winner"]
    
    features["away_team_id"] = fixture["teams"]["away"]["id"]
    features["away_team_name"] = fixture["teams"]["away"]["name"]
    features["away_team_logo"] = fixture["teams"]["away"]["logo"]
    features["away_team_winner"] = fixture["teams"]["away"]["winner"]
    
    # Goals (if match has played)
    goals = fixture["goals"]
    features["home_goals"] = goals["home"]
    features["away_goals"] = goals["away"]
    
    # Score at halftime (if available)
    score = fixture["score"]
    features["halftime_home"] = score["halftime"]["home"]
    features["halftime_away"] = score["halftime"]["away"]
    features["fulltime_home"] = score["fulltime"]["home"]
    features["fulltime_away"] = score["fulltime"]["away"]
    features["extratime_home"] = score["extratime"]["home"]
    features["extratime_away"] = score["extratime"]["away"]
    features["penalty_home"] = score["penalty"]["home"]
    features["penalty_away"] = score["penalty"]["away"]
    
    return features

def get_team_search_context(team_name: str, match_date: str, context_days: int = 3) -> Dict:
    """
    Get recent news/context for a team using Brave Search.
    """
    # Create a query for recent team news
    query = f"{team_name} team news injuries form last {context_days} days"
    
    search_data = fetch_brave_search(query, count=5)
    
    if "error" in search_data:
        return {"search_error": search_data["error"]}
    
    # Extract useful snippets
    results = search_data.get("results", [])
    snippets = []
    for result in results[:3]:  # Top 3 results
        snippet = {
            "title": result.get("title", ""),
            "description": result.get("description", ""),
            "url": result.get("url", ""),
            "age": result.get("age", "")  # If available
        }
        snippets.append(snippet)
    
    return {
        "query": query,
        "result_count": len(results),
        "snippets": snippets,
        "search_performed": True
    }

def build_match_features(fixture_id: int) -> Dict:
    """
    Build a complete feature set for a single match.
    Combines weather, fixture, and search data.
    """
    # Get fixture features first (to get venue, date, teams)
    fixture_data = fetch_api_football("fixtures", {"id": fixture_id})
    
    if "error" in fixture_data or not fixture_data.get("response"):
        return {"error": f"Could not fetch fixture data for ID {fixture_id}"}
    
    fixture = fixture_data["response"][0]
    
    # Extract needed info
    venue = f"{fixture['fixture']['venue']['name']}, {fixture['fixture']['venue']['city']}"
    match_date = fixture["fixture"]["date"][:10]  # YYYY-MM-DD part
    
    # Get weather features
    weather_features = get_weather_features(venue, match_date)
    
    # Get search context for both teams
    home_team = fixture["teams"]["home"]["name"]
    away_team = fixture["teams"]["away"]["name"]
    
    home_search = get_team_search_context(home_team, match_date)
    away_search = get_team_search_context(away_team, match_date)
    
    # Combine all features
    features = {
        "fixture_id": fixture_id,
        "match_date": match_date,
        "venue": venue,
        "home_team": home_team,
        "away_team": away_team,
    }
    
    # Add fixture features (flatten the nested structure)
    def flatten_dict(d, parent_key='', sep='_'):
        items = []
        for k, v in d.items():
            new_key = parent_key + sep + k if parent_key else k
            if isinstance(v, dict):
                items.extend(flatten_dict(v, new_key, sep=sep).items())
            else:
                items.append((new_key, v))
        return dict(items)
    
    flattened_fixture = flatten_dict(fixture)
    # Remove the keys we already added explicitly
    for key in ['fixture_id', 'match_date', 'venue', 'home_team', 'away_team']:
        flattened_fixture.pop(key, None)
    
    features.update({f"fixture_{k}": v for k, v in flattened_fixture.items()})
    
    # Add weather features
    features.update({f"weather_{k}": v for k, v in weather_features.items()})
    
    # Add search features (ML-friendly structured format)
    home_snippets = home_search.get("snippets", [])
    away_snippets = away_search.get("snippets", [])
    
    # Extract numeric features from search results
    features["home_search_count"] = home_search.get("result_count", 0)
    features["away_search_count"] = away_search.get("result_count", 0)
    
    # Has news flag (binary feature for ML)
    features["home_has_news"] = len(home_snippets) > 0
    features["away_has_news"] = len(away_snippets) > 0
    
    # Store combined title + description as single text field (for NLP later)
    home_text = " ".join([
        f"{s.get('title', '')} {s.get('description', '')}" 
        for s in home_snippets[:3]
    ]) if home_snippets else ""
    away_text = " ".join([
        f"{s.get('title', '')} {s.get('description', '')}" 
        for s in away_snippets[:3]
    ]) if away_snippets else ""
    
    features["home_search_text"] = home_text[:2000]  # Limit length
    features["away_search_text"] = away_text[:2000]  # Limit length
    
    # Store full snippets as JSON only if explicitly needed
    features["home_search_snippets_json"] = json.dumps(home_snippets)
    features["away_search_snippets_json"] = json.dumps(away_snippets)
    
    return features

def build_features_for_fixtures(fixture_ids: List[int]) -> List[Dict]:
    """
    Build features for a list of fixture IDs.
    """
    all_features = []
    for fid in fixture_ids:
        try:
            features = build_match_features(fid)
            all_features.append(features)
        except Exception as e:
            all_features.append({
                "fixture_id": fid,
                "error": str(e)
            })
    return all_features

def get_upcoming_epl_fixtures(days_ahead: int = 7) -> List[Dict]:
    """
    Get upcoming EPL fixtures for the next N days.
    Returns a list of fixture data from API-Football.
    """
    from datetime import datetime, timedelta
    
    today = datetime.now()
    future = today + timedelta(days=days_ahead)
    
    params = {
        "league": 39,  # Premier League
        "from": today.strftime("%Y-%m-%d"),
        "to": future.strftime("%Y-%m-%d")
    }
    
    data = fetch_api_football("fixtures", params)
    
    if "error" in data:
        return []
    
    return data.get("response", [])

def main():
    """
    Example usage: Build features for upcoming EPL fixtures.
    """
    print("Fetching upcoming EPL fixtures...")
    upcoming = get_upcoming_epl_fixtures(days_ahead=7)
    
    if not upcoming:
        print("No upcoming fixtures found or error in fetching.")
        return
    
    print(f"Found {len(upcoming)} upcoming fixtures.")
    
    # Extract fixture IDs
    fixture_ids = [f["fixture"]["id"] for f in upcoming[:5]]  # Limit to first 5 for demo
    
    print(f"Building features for {len(fixture_ids)} fixtures...")
    features_list = build_features_for_fixtures(fixture_ids)
    
    # Output to JSON file
    output_file = CACHE_DIR / f"match_features_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_file, 'w') as f:
        json.dump(features_list, f, indent=2)
    
    print(f"Features saved to {output_file}")
    
    # Print a summary
    for i, features in enumerate(features_list):
        if "error" not in features:
            print(f"\nFixture {i+1}: {features.get('home_team')} vs {features.get('away_team')}")
            print(f"  Weather: {features.get('weather_temperature_c')}°C, {features.get('weather_precipitation_mm')}mm rain")
            print(f"  Home team search results: {features.get('home_search_count')}")
            print(f"  Away team search results: {features.get('away_search_count')}")
        else:
            print(f"\nFixture {i+1}: Error - {features.get('error')}")

if __name__ == "__main__":
    # Ensure API keys are set
    if not os.getenv("API_FOOTBALL_KEY"):
        print("Warning: API_FOOTBALL_KEY not set in environment")
    if not os.getenv("BRAVE_SEARCH_API_KEY"):
        print("Warning: BRAVE_SEARCH_API_KEY not set in environment")
    if not os.getenv("OPENWEATHERMAP_API_KEY"):
        print("Warning: OPENWEATHERMAP_API_KEY not set in environment")
    
    main()