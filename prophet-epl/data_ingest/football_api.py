"""
Standalone API-Football data fetcher for Prophet EPL.
Fetches fixtures, teams, standings, injuries, and more.
"""

import os
import json
import time
import requests
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any

# Configuration
API_KEY = os.getenv("API_FOOTBALL_KEY")
BASE_URL = "https://v3.football.api-sports.io"

# Rate limiting
RATE_LIMIT = 8  # requests per minute (conservative)
_last_call_time = 0


def _rate_limit():
    """Apply rate limiting between API calls."""
    global _last_call_time
    now = time.time()
    elapsed = now - _last_call_time
    min_interval = 60.0 / RATE_LIMIT
    if elapsed < min_interval:
        time.sleep(min_interval - elapsed)
    _last_call_time = time.time()


def _api_request(endpoint: str, params: Dict = None) -> Dict:
    """Make a request to API-Football."""
    if not API_KEY:
        return {"error": "API_FOOTBALL_KEY not set"}
    
    _rate_limit()
    
    headers = {
        'x-rapidapi-key': API_KEY,
        'x-rapidapi-host': 'v3.football.api-sports.io'
    }
    
    try:
        resp = requests.get(f"{BASE_URL}/{endpoint}", headers=headers, params=params, timeout=15)
        resp.raise_for_status()
        return resp.json()
    except requests.RequestException as e:
        return {"error": str(e)}


# ============ League & Season ============

def get_league_info(league_id: int = 39) -> Dict:
    """Get league details."""
    return _api_request("leagues", {"id": league_id})


def get_league_seasons(league_id: int = 39) -> List[int]:
    """Get available seasons for a league."""
    data = get_league_info(league_id)
    league = data.get("response", [{}])[0]
    seasons = league.get("seasons", [])
    return [s.get("season") for s in seasons if s.get("season")]


# ============ Teams ============

def get_epl_teams(season: int = 2024) -> List[Dict]:
    """Get all EPL teams for a season."""
    data = _api_request("teams", {"league": 39, "season": season})
    return data.get("response", [])


def get_team_by_id(team_id: int) -> Dict:
    """Get team details by ID."""
    return _api_request("teams", {"id": team_id})


# ============ Fixtures ============

def get_fixtures(league_id: int = 39, season: int = None, 
                 from_date: str = None, to_date: str = None,
                 status: str = None, round_name: str = None) -> List[Dict]:
    """Get fixtures with various filters."""
    params = {"league": league_id}
    if season:
        params["season"] = season
    if from_date:
        params["from"] = from_date
    if to_date:
        params["to"] = to_date
    if status:
        params["status"] = status
    if round_name:
        params["round"] = round_name
    
    data = _api_request("fixtures", params)
    return data.get("response", [])


def get_fixture_by_id(fixture_id: int) -> Dict:
    """Get fixture details by ID."""
    data = _api_request("fixtures", {"id": fixture_id})
    return data.get("response", [{}])[0]


def get_upcoming_fixtures(days: int = 7, season: int = None) -> List[Dict]:
    """Get upcoming EPL fixtures."""
    today = datetime.now()
    future = today + timedelta(days=days)
    
    params = {
        "league": 39,
        "from": today.strftime("%Y-%m-%d"),
        "to": future.strftime("%Y-%m-%d")
    }
    if season:
        params["season"] = season
    
    data = _api_request("fixtures", params)
    return data.get("response", [])


def get_recent_fixtures(days: int = 7, season: int = None) -> List[Dict]:
    """Get recent EPL fixtures."""
    today = datetime.now()
    past = today - timedelta(days=days)
    
    params = {
        "league": 39,
        "from": past.strftime("%Y-%m-%d"),
        "to": today.strftime("%Y-%m-%d"),
        "status": "FT"  # Full time
    }
    if season:
        params["season"] = season
    
    data = _api_request("fixtures", params)
    return data.get("response", [])


# ============ Standings ============

def get_standings(league_id: int = 39, season: int = 2024) -> Dict:
    """Get league standings."""
    return _api_request("standings", {"league": league_id, "season": season})


def parse_standings(standings_data: Dict) -> List[Dict]:
    """Parse standings into a clean format."""
    response = standings_data.get("response", [])
    if not response:
        return []
    
    league_data = response[0]
    league = league_data.get("league", {})
    standings = league_data.get("standings", [])
    
    parsed = []
    for group in standings:
        for row in group:
            parsed.append({
                "position": row.get("rank"),
                "team_id": row.get("team", {}).get("id"),
                "team_name": row.get("team", {}).get("name"),
                "team_logo": row.get("team", {}).get("logo"),
                "played": row.get("all", {}).get("played"),
                "won": row.get("all", {}).get("win"),
                "drawn": row.get("all", {}).get("draw"),
                "lost": row.get("all", {}).get("loss"),
                "goals_for": row.get("all", {}).get("goals", {}).get("for"),
                "goals_against": row.get("all", {}).get("goals", {}).get("against"),
                "goal_difference": row.get("goalsDiff"),
                "points": row.get("points"),
                "form": row.get("form"),
                "group": group[0].get("group") if group else None,
            })
    
    return parsed


# ============ Injuries ============

def get_injuries(league_id: int = 39, season: int = 2024) -> List[Dict]:
    """Get current injuries."""
    data = _api_request("injuries", {"league": league_id, "season": season})
    return data.get("response", [])


# ============ Player Stats ============

def get_team_players(team_id: int, season: int = 2024) -> List[Dict]:
    """Get players for a team."""
    data = _api_request("players", {"team": team_id, "season": season})
    return data.get("response", [])


def get_player_stats(player_id: int, season: int = 2024) -> Dict:
    """Get player statistics."""
    data = _api_request("players", {"id": player_id, "season": season})
    return data.get("response", [{}])[0]


# ============ Head to Head ============

def get_head_to_head(team1_id: int, team2_id: int, last: int = 10) -> List[Dict]:
    """Get head to head between two teams."""
    data = _api_request("fixtures/headtohead", {
        "h2h": f"{team1_id}-{team2_id}",
        "last": last
    })
    return data.get("response", [])


# ============ Odds ============

def get_odds(fixture_id: int, bookmaker: str = "1xbet") -> Dict:
    """Get odds for a fixture."""
    data = _api_request("odds", {"fixture": fixture_id, "bookmaker": bookmaker})
    return data.get("response", [{}])[0]


# ============ Form & Statistics ============

def get_team_form(team_id: int, last: int = 5) -> str:
    """Get team's recent form (e.g., 'WWDLW')."""
    fixtures = get_fixtures(team_id=39, status="FT")
    team_fixtures = []
    
    for f in fixtures:
        teams = f.get("teams", {})
        if teams.get("home", {}).get("id") == team_id or teams.get("away", {}).get("id") == team_id:
            team_fixtures.append(f)
            if len(team_fixtures) >= last:
                break
    
    form = ""
    for f in team_fixtures:
        teams = f.get("teams", {})
        goals = f.get("goals", {})
        
        if teams.get("home", {}).get("id") == team_id:
            # Team was home
            my_goals = goals.get("home")
            opp_goals = goals.get("away")
        else:
            # Team was away
            my_goals = goals.get("away")
            opp_goals = goals.get("home")
        
        if my_goals is None or opp_goals is None:
            continue
        
        if my_goals > opp_goals:
            form = "W" + form
        elif my_goals < opp_goals:
            form = "L" + form
        else:
            form = "D" + form
    
    return form


# ============ Main ============

def main():
    """Demo: Fetch and display current EPL standings."""
    print("Fetching EPL standings...")
    
    standings = get_standings(league_id=39, season=2024)
    parsed = parse_standings(standings)
    
    if not parsed:
        print("No standings data available")
        return
    
    print(f"\nEPL Standings (2024/25):")
    print(f"{'Pos':<4} {'Team':<20} {'P':<3} {'W':<3} {'D':<3} {'L':<3} {'GD':<4} {'Pts':<4}")
    print("-" * 50)
    
    for row in parsed[:10]:
        print(f"{row['position']:<4} {row['team_name'][:19]:<20} {row['played']:<3} {row['won']:<3} {row['drawn']:<3} {row['lost']:<3} {row['goal_difference']:<4} {row['points']:<4}")
    
    # Demo upcoming fixtures
    print("\n\nUpcoming fixtures:")
    upcoming = get_upcoming_fixtures(days=7)
    print(f"Found {len(upcoming)} fixtures in next 7 days")
    
    for f in upcoming[:5]:
        teams = f.get("teams", {})
        fixture = f.get("fixture", {})
        date = fixture.get("date", "")[:16].replace("T", " ")
        home = teams.get("home", {}).get("name")
        away = teams.get("away", {}).get("name")
        print(f"  {date} | {home} vs {away}")


if __name__ == "__main__":
    if not API_KEY:
        print("Warning: API_FOOTBALL_KEY not set")
    main()
