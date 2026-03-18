#!/usr/bin/env python3
"""
Test script for the feature pipeline using a known fixture.
"""
import os
import sys
import json
from datetime import datetime

# Ensure we can import from the prophet-epl directory
sys.path.insert(0, '/home/adamt/prophet-epl-ready/prophet-epl')

from feature_pipeline import (
    get_weather_features,
    get_fixture_features,
    get_team_search_context,
    build_match_features,
    fetch_api_football
)

def main():
    # Set API keys from environment (should already be set in the shell)
    print("API Keys check:")
    print(f"OPENWEATHERMAP_API_KEY: {'SET' if os.getenv('OPENWEATHERMAP_API_KEY') else 'NOT SET'}")
    print(f"BRAVE_SEARCH_API_KEY: {'SET' if os.getenv('BRAVE_SEARCH_API_KEY') else 'NOT SET'}")
    print(f"API_FOOTBALL_KEY: {'SET' if os.getenv('API_FOOTBALL_KEY') else 'NOT SET'}")
    print()
    
    # Use a known fixture ID from the 2024 season
    fixture_id = 1208021  # Manchester United vs Fulham on 2024-08-16
    print(f"Testing with fixture ID: {fixture_id}")
    print()
    
    # 1. Get fixture features
    print("1. Fetching fixture features...")
    fixture_features = get_fixture_features(fixture_id)
    if "fixture_error" in fixture_features:
        print(f"   Error: {fixture_features['fixture_error']}")
        return
    print(f"   Home: {fixture_features.get('home_team_name')}")
    print(f"   Away: {fixture_features.get('away_team_name')}")
    print(f"   Date: {fixture_features.get('fixture_date')}")  # This will be in fixture_fixture_date after flattening? Let's see.
    # Actually, get_fixture_features returns flattened keys? No, it returns the dict we built.
    # Let's check what keys we have.
    print(f"   Available keys: {list(fixture_features.keys())}")
    print()
    
    # 2. Get weather features (using venue and date from fixture)
    venue = f"{fixture_features.get('venue_name')}, {fixture_features.get('venue_city')}"
    # The date is in fixture_date? Actually, the fixture data has fixture_date under fixture.
    # We need to extract the date from the fixture data we already fetched.
    # Let's refetch the fixture to get the date string, or we can store it.
    # For simplicity, we'll get the fixture data again to get the date.
    fixture_data = fetch_api_football("fixtures", {"id": fixture_id})
    if "error" not in fixture_data and fixture_data.get("response"):
        match_date = fixture_data["response"][0]["fixture"]["date"][:10]  # YYYY-MM-DD
    else:
        # Fallback to today if we can't get it
        match_date = datetime.now().strftime("%Y-%m-%d")
    
    print(f"2. Fetching weather for {venue} on {match_date}...")
    weather_features = get_weather_features(venue, match_date)
    if "weather_error" in weather_features:
        print(f"   Error: {weather_features['weather_error']}")
    else:
        print(f"   Temperature: {weather_features.get('temperature_c')}°C")
        print(f"   Precipitation: {weather_features.get('precipitation_mm')} mm")
        print(f"   Wind speed: {weather_features.get('wind_speed_m_s')} m/s")
        print(f"   Description: {weather_features.get('weather_description')}")
    print()
    
    # 3. Get search context for both teams
    home_team = fixture_features.get("home_team_name")
    away_team = fixture_features.get("away_team_name")
    
    print(f"3. Fetching search context for {home_team}...")
    home_search = get_team_search_context(home_team, match_date)
    if "search_error" in home_search:
        print(f"   Error: {home_search['search_error']}")
    else:
        print(f"   Found {home_search.get('result_count', 0)} results")
        if home_search.get("snippets"):
            print(f"   First snippet: {home_search['snippets'][0].get('title', '')[:50]}...")
    print()
    
    print(f"   Fetching search context for {away_team}...")
    away_search = get_team_search_context(away_team, match_date)
    if "search_error" in away_search:
        print(f"   Error: {away_search['search_error']}")
    else:
        print(f"   Found {away_search.get('result_count', 0)} results")
        if away_search.get("snippets"):
            print(f"   First snippet: {away_search['snippets'][0].get('title', '')[:50]}...")
    print()
    
    # 4. Build combined features
    print("4. Building combined match features...")
    combined = build_match_features(fixture_id)
    if "error" in combined:
        print(f"   Error: {combined['error']}")
    else:
        print("   Successfully built feature set!")
        # Show a few key features
        print(f"   Match: {combined.get('home_team')} vs {combined.get('away_team')}")
        print(f"   Date: {combined.get('match_date')}")
        print(f"   Venue: {combined.get('venue')}")
        print(f"   Temperature: {combined.get('weather_temperature_c')}°C")
        print(f"   Home goals: {combined.get('fixture_goals_home')}")
        print(f"   Away goals: {combined.get('fixture_goals_away')}")
        print(f"   Home search results: {combined.get('home_search_count')}")
        print(f"   Away search results: {combined.get('away_search_count')}")
        
        # Save to file for inspection
        output_path = "/home/adamt/prophet-epl-ready/prophet-epl/feature_cache/test_fixture_features.json"
        with open(output_path, 'w') as f:
            json.dump(combined, f, indent=2)
        print(f"   Full feature set saved to: {output_path}")
    
    print("\nTest complete!")

if __name__ == "__main__":
    main()