#!/usr/bin/env python3
"""
EPL & MMA Market Simulation Pipeline v2
Compatible with real MiroFish (https://github.com/666ghj/MiroFish)

Combines API-Football + Weather + News -> MiroFish simulation
to predict betting market line movements before they happen.
"""

import os
import sys
import json
import argparse
import requests
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional
import time
import re


def load_env_file(path: Path) -> None:
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, val = line.split("=", 1)
        os.environ.setdefault(key.strip(), val.strip())


# Project layout
SCRIPT_PATH = Path(__file__).resolve()
SKILL_DIR = SCRIPT_PATH.parents[1]  # .../prophet_epl/skills/epl-market-sim
PROJECT_ROOT = SCRIPT_PATH.parents[4]  # .../prophet-epl-ready
CONFIG_DIR = SKILL_DIR / "configs"
DATA_DIR = Path.home() / ".hermes" / "data" / "epl-market-sim"
DATA_DIR.mkdir(parents=True, exist_ok=True)

# Load envs from portable locations (do not overwrite existing env vars)
load_env_file(PROJECT_ROOT / ".env")
load_env_file(Path.cwd() / ".env")

# Add project root to path
sys.path.insert(0, str(PROJECT_ROOT))

# API Keys (from environment)
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
API_FOOTBALL_KEY = os.getenv("API_FOOTBALL_KEY")
WEATHER_API_KEY = os.getenv("OPENWEATHERMAP_API_KEY")
BRAVE_SEARCH_KEY = os.getenv("BRAVE_SEARCH_API_KEY")
ODDS_API_KEY = os.getenv("ODDSSPORTSDATAIO_KEY")

# MiroFish URLs
MIROFISH_URL = os.getenv("MIROFISH_URL", "http://localhost:5001")
MIROFISH_FRONTEND = os.getenv("MIROFISH_FRONTEND", "http://localhost:3000")


def load_config(config_name: str) -> Dict:
    """Load a JSON config file."""
    config_path = CONFIG_DIR / config_name
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    with open(config_path) as f:
        return json.load(f)


def check_mirofish_health() -> Dict:
    """Check MiroFish health."""
    try:
        # Try backend health (main endpoint)
        resp = requests.get(f"{MIROFISH_URL}/health", timeout=5)
        if resp.status_code == 200:
            return {"status": "healthy", "backend": resp.json()}
    except:
        pass
    
    try:
        # Try backend /api/health
        resp = requests.get(f"{MIROFISH_URL}/api/health", timeout=5)
        if resp.status_code == 200:
            return {"status": "healthy", "backend": resp.json()}
    except:
        pass
    
    return {"status": "unreachable"}


def health_check() -> bool:
    """Check all dependencies are available."""
    print("=" * 50)
    print("Health Check - EPL Market Sim")
    print("=" * 50)
    
    issues = []
    
    # Check API keys
    print("\n[API Keys]")
    if OPENROUTER_API_KEY:
        print("  ✓ OpenRouter: set")
    elif OLLAMA_BASE_URL:
        print(f"  ✓ Ollama: {OLLAMA_BASE_URL}")
    else:
        issues.append("No LLM provider: Set OPENROUTER_API_KEY or OLLAMA_BASE_URL")

    if API_FOOTBALL_KEY:
        print("  ✓ API-Football: set")
    else:
        print("  ⚠ API-Football: Not set (optional)")

    if WEATHER_API_KEY:
        print("  ✓ Weather API: set")
    else:
        print("  ⚠ Weather API: Not set (optional)")

    if BRAVE_SEARCH_KEY:
        print("  ✓ Brave Search: set")
    else:
        print("  ⚠ Brave Search: Not set (optional)")

    # Check MiroFish (optional for direct-LLM fallback mode)
    print("\n[MiroFish]")
    mirofish_status = check_mirofish_health()
    if mirofish_status["status"] == "healthy":
        print(f"  ✓ MiroFish: Reachable at {MIROFISH_URL}")
    else:
        print(f"  ⚠ MiroFish: Not reachable at {MIROFISH_URL} (direct LLM mode can still run)")
        print("    Run: docker run -d -p 3000:3000 -p 5001:5001 ghcr.io/666ghj/mirofish")
    
    if issues:
        print("\n❌ Issues found:")
        for issue in issues:
            print(f"  - {issue}")
        return False
    
    print("\n✓ All systems healthy!")
    return True


def fetch_epl_match(match_id: int) -> Dict:
    """Fetch EPL match data from API-Football."""
    if not API_FOOTBALL_KEY:
        return {"error": "API_FOOTBALL_KEY not set"}
    
    url = "https://v3.football.api-sports.io/fixtures"
    headers = {
        'x-rapidapi-key': API_FOOTBALL_KEY,
        'x-rapidapi-host': 'v3.football.api-sports.io'
    }
    
    try:
        resp = requests.get(url, headers=headers, params={"id": match_id}, timeout=15)
        data = resp.json()
        
        if data.get("results", 0) > 0:
            return data["response"][0]
        return {"error": f"No match found for ID {match_id}"}
    except Exception as e:
        return {"error": str(e)}


def fetch_upcoming_epl_fixtures(limit: int = 10) -> List[Dict]:
    """Fetch upcoming EPL fixtures."""
    if not API_FOOTBALL_KEY:
        return []
    
    from datetime import datetime, timedelta
    today = datetime.now()
    week_later = today + timedelta(days=7)
    
    url = "https://v3.football.api-sports.io/fixtures"
    headers = {
        'x-rapidapi-key': API_FOOTBALL_KEY,
        'x-rapidapi-host': 'v3.football.api-sports.io'
    }
    params = {
        "league": 39,  # Premier League
        "from": today.strftime("%Y-%m-%d"),
        "to": week_later.strftime("%Y-%m-%d"),
        "status": "NS"  # Not Started
    }
    
    try:
        resp = requests.get(url, headers=headers, params=params, timeout=15)
        data = resp.json()
        return data.get("response", [])[:limit]
    except Exception:
        return []


def fetch_weather(venue: str, match_date: str = None) -> Dict:
    """Fetch weather for venue."""
    if not WEATHER_API_KEY:
        return {"error": "Weather API not configured"}
    
    url = "https://api.openweathermap.org/data/2.5/forecast"
    params = {
        "q": venue.split(",")[0] if "," in venue else venue,
        "appid": WEATHER_API_KEY,
        "units": "metric"
    }
    
    try:
        resp = requests.get(url, params=params, timeout=10)
        data = resp.json()
        
        if data.get("cod") == "200":
            return {
                "temperature": data["list"][0]["main"]["temp"],
                "description": data["list"][0]["weather"][0]["description"],
                "wind_speed": data["list"][0]["wind"]["speed"],
                "humidity": data["list"][0]["main"]["humidity"]
            }
        return {"error": f"Weather API error: {data.get('message')}"}
    except Exception as e:
        return {"error": str(e)}


def fetch_news(team_name: str, limit: int = 5, sport: str = "epl") -> List[Dict]:
    """Fetch news using Brave Search."""
    if not BRAVE_SEARCH_KEY:
        return []
    
    # Add sport-specific terms to avoid contamination
    sport_terms = ""
    if sport == "mma":
        sport_terms = "UFC MMA"
    
    url = "https://api.search.brave.com/res/v1/web/search"
    headers = {
        'X-Subscription-Token': BRAVE_SEARCH_KEY,
        'Accept': 'application/json'
    }
    
    try:
        query = f"{team_name} {sport_terms} news".strip()
        resp = requests.get(url, headers=headers, params={
            "q": query,
            "count": limit
        }, timeout=10)
        
        data = resp.json()
        return data.get("web", {}).get("results", [])
    except Exception:
        return []


def fetch_mma_odds(fighter1: str, fighter2: str) -> Dict:
    """Fetch MMA odds from OddsSportsDataIO."""
    if not ODDS_API_KEY:
        return {"error": "ODDSSPORTSDATAIO_KEY not set"}
    
    url = "https://api.oddssportsdata.io/v4/mma/upcoming"
    headers = {"Ocp-Apim-Subscription-Key": ODDS_API_KEY}
    
    try:
        resp = requests.get(url, headers=headers, timeout=15)
        data = resp.json()
        
        # Find the fight
        for event in data.get("events", []):
            home_fighter = event.get("homeTeam", {}).get("name", "")
            away_fighter = event.get("awayTeam", {}).get("name", "")
            
            if fighter1.lower() in home_fighter.lower() and fighter2.lower() in away_fighter.lower():
                return {
                    "home_odds": event.get("homeTeamMoneyLine", 0),
                    "away_odds": event.get("awayTeamMoneyLine", 0),
                    "over_under": event.get("overUnder", 0),
                    "event_date": event.get("startDateTime", "")
                }
        
        return {"error": "Fight not found"}
    except Exception as e:
        return {"error": str(e)}


def generate_seed_packet(sport: str, match_data: Dict, weather: Dict, news: Dict) -> Dict:
    """Generate MiroFish-compatible seed packet."""
    
    # Handle both "epl" and "football" as football sports
    if sport in ("football", "epl"):
        teams = match_data.get("teams", {})
        home_team = teams.get("home", {}).get("name", "Unknown")
        away_team = teams.get("away", {}).get("name", "Unknown")
        venue = match_data.get("fixture", {}).get("venue", {}).get("name", "Unknown")
        match_date = match_data.get("fixture", {}).get("date", "")[:10]
        league = match_data.get("league", {}).get("name", "Premier League")
        sport = "epl"  # Normalize
    else:
        # MMA/UFC
        home_team = match_data.get("home_fighter", "Unknown")
        away_team = match_data.get("away_fighter", "Unknown")
        venue = match_data.get("event", {}).get("venue", "Unknown")
        match_date = match_data.get("event", {}).get("date", "")[:10]
        league = "UFC"
    
    return {
        "sport": sport,
        "league": league,
        "match": {
            "home": home_team,
            "away": away_team,
            "venue": venue,
            "date": match_date
        },
        "conditions": {
            "weather": weather,
        },
        "news": news,
        "generated_at": datetime.now().isoformat()
    }


def _parse_percent_value(value) -> Optional[float]:
    """Parse percentage-like values (e.g., 42, '42%', '42.5')."""
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        m = re.search(r"-?\d+(?:\.\d+)?", value)
        if m:
            return float(m.group(0))
    return None


def _extract_three_way_probs(response: Dict) -> Optional[Dict[str, float]]:
    """Extract p(Home)/p(Draw)/p(Away) from structured JSON or free-form text."""
    # 1) Direct structured keys first
    direct_home = _parse_percent_value(response.get("p(Home)"))
    direct_draw = _parse_percent_value(response.get("p(Draw)"))
    direct_away = _parse_percent_value(response.get("p(Away)"))
    if None not in (direct_home, direct_draw, direct_away):
        return {"home": direct_home, "draw": direct_draw, "away": direct_away}

    # 2) Parse from raw text payload
    text_to_parse = " ".join([
        str(response.get("raw_response", "")),
        str(response),
    ])

    home_match = re.search(r"p\(Home\)[^0-9-]*(-?\d+(?:\.\d+)?)%?", text_to_parse, re.IGNORECASE)
    draw_match = re.search(r"p\(Draw\)[^0-9-]*(-?\d+(?:\.\d+)?)%?", text_to_parse, re.IGNORECASE)
    away_match = re.search(r"p\(Away\)[^0-9-]*(-?\d+(?:\.\d+)?)%?", text_to_parse, re.IGNORECASE)

    if not (home_match and draw_match and away_match):
        return None

    return {
        "home": float(home_match.group(1)),
        "draw": float(draw_match.group(1)),
        "away": float(away_match.group(1)),
    }


def run_mirofish_simulation(seed_packet: Dict, config: Dict) -> Dict:
    """
    Run multi-agent simulation directly via OpenRouter.
    
    Falls back to direct LLM calls when MiroFish is unavailable.
    Simulates betting market actors reacting to match data.
    """
    
    if not OPENROUTER_API_KEY:
        return {"error": "No OpenRouter API key available"}
    
    actors = config.get("actors", [])
    if not actors:
        return {"error": "No actors configured"}
    
    # Build simulation prompt from seed packet
    sport = seed_packet.get("sport", "unknown")
    match = seed_packet.get("match", {})
    home_team = match.get("home", "Unknown")
    away_team = match.get("away", "Unknown")
    venue = match.get("venue", "Unknown")
    
    weather = seed_packet.get("conditions", {}).get("weather", {})
    weather_desc = weather.get("description", "Unknown") if isinstance(weather, dict) else "Unknown"
    temp = weather.get("temp", "N/A") if isinstance(weather, dict) else "N/A"
    
    news = seed_packet.get("news", {})
    home_news = news.get("home_news", [])
    away_news = news.get("away_news", [])
    
    # Create simulation context
    context = f"""
MATCH: {home_team} vs {away_team}
VENUE: {venue}
WEATHER: {temp}°C, {weather_desc}

HOME TEAM NEWS:
{chr(10).join(f"- {n}" for n in home_news[:5]) if home_news else "No recent news"}

AWAY TEAM NEWS:
{chr(10).join(f"- {n}" for n in away_news[:5]) if away_news else "No recent news"}
"""
    
    print(f"  Running multi-agent simulation for {home_team} vs {away_team}...")
    
    # Run simulation with multiple actors
    actor_results = []
    
    for i, actor in enumerate(actors[:8]):  # Limit to 8 actors
        actor_name = actor.get("name", f"Actor_{i}")
        actor_role = actor.get("role", actor.get("persona", "Bettor"))
        
        if sport == "mma":
            # MMA prompt (unchanged for now)
            prompt = f"""You are {actor_name}, a {actor_role} in MMA betting.

{context}

Based on your persona and the above fight information:
1. Your prediction: Who will win?
2. Your recommended bet with odds
3. Your confidence level (0-100%)
4. How you think the line will move

Respond in JSON format:
{{"prediction": "home/away", "bet": "description", "confidence": 0-100, "line_movement": "up/down/flat"}}"""
        else:
            # EPL prompt - 3-way probabilities
            prompt = f"""You are {actor_name}, a {actor_role} in sports betting.

{context}

Output EXACTLY three probabilities that sum to 100.
Do not include explanations.
Do not include any other text.

Format exactly:
p(Home): XX%
p(Draw): YY%
p(Away): ZZ%"""

        try:
            result = call_openrouter(prompt, max_tokens=300)
            if result:
                actor_results.append({
                    "actor": actor_name,
                    "role": actor_role,
                    "response": result,
                    "sport": sport
                })
                print(f"    {actor_name}: Done")
        except Exception as e:
            print(f"    {actor_name}: Error - {e}")
    
    # Aggregate results
    if not actor_results:
        return {"error": "No actor responses received"}
    
    # Calculate consensus - EPL/football uses 3-way probability averaging.
    if sport in {"epl", "football"}:
        home_probs: List[float] = []
        draw_probs: List[float] = []
        away_probs: List[float] = []
        valid_actor_count = 0

        for ar in actor_results:
            resp = ar.get("response", {})
            parsed_probs = _extract_three_way_probs(resp)
            if not parsed_probs:
                print(f"    DEBUG: Could not parse 3-way probabilities from actor={ar.get('actor')}")
                continue

            home_probs.append(parsed_probs["home"])
            draw_probs.append(parsed_probs["draw"])
            away_probs.append(parsed_probs["away"])
            valid_actor_count += 1

        # Fail-closed if no valid 3-way agent outputs exist.
        if valid_actor_count == 0:
            return {
                "error": "No valid 3-way probability outputs",
                "sport": sport,
                "agent_consensus": "fail_closed_no_valid_agent_probs",
                "p_home": None,
                "p_draw": None,
                "p_away": None,
                "actors": actor_results,
            }

        avg_home = sum(home_probs) / valid_actor_count
        avg_draw = sum(draw_probs) / valid_actor_count
        avg_away = sum(away_probs) / valid_actor_count

        print(f"    Raw probs: home={home_probs}, draw={draw_probs}, away={away_probs}")

        # Renormalize to sum to 100 to account for imperfect agent totals.
        total = avg_home + avg_draw + avg_away
        if total <= 0:
            return {
                "error": "Invalid 3-way probability total",
                "sport": sport,
                "agent_consensus": "fail_closed_invalid_total",
                "p_home": None,
                "p_draw": None,
                "p_away": None,
                "actors": actor_results,
            }

        avg_home = (avg_home / total) * 100
        avg_draw = (avg_draw / total) * 100
        avg_away = (avg_away / total) * 100

        p_home = round(avg_home, 4)
        p_draw = round(avg_draw, 4)
        # Force exact 100.0000 total after rounding to avoid drift.
        p_away = round(100.0 - p_home - p_draw, 4)

        return {
            "success": True,
            "sport": sport,
            "match": f"{home_team} vs {away_team}",
            "actors": actor_results,
            "agent_consensus": "three_way_mean",
            "valid_agent_count": valid_actor_count,
            "p_home": p_home,
            "p_draw": p_draw,
            "p_away": p_away,
            "probabilities": {
                "p(Home)": p_home,
                "p(Draw)": p_draw,
                "p(Away)": p_away
            },
            "confidence": round(max(p_home, p_draw, p_away) / 100, 4)
        }
    
    # MMA: Calculate consensus (fixed vote mapping)
    home_votes = 0
    away_votes = 0
    total_confidence = 0
    valid_picks = 0
    
    for ar in actor_results:
        resp = ar.get("response", {})
        pred = str(resp.get("prediction", "")).lower()
        
        # Normalize and map correctly - check both team names AND home/away shorthand
        if home_team.lower() in pred or pred == "home":
            home_votes += 1
            valid_picks += 1
        elif away_team.lower() in pred or pred == "away":
            away_votes += 1
            valid_picks += 1
        
        total_confidence += resp.get("confidence", 50)
    
    if valid_picks == 0:
        return {"error": "No valid picks received"}
    
    winner = "home" if home_votes > away_votes else "away"
    confidence = round(max(home_votes, away_votes) / valid_picks * 100, 1)
    
    avg_confidence = total_confidence / len(actor_results) if actor_results else 50
    
    # Determine predicted line movement
    line_movement = "flat"
    if home_votes > away_votes + 2:
        line_movement = "down"  # Home team favored
    elif away_votes > home_votes + 2:
        line_movement = "up"  # Away team favored
    
    return {
        "success": True,
        "sport": sport,
        "match": f"{home_team} vs {away_team}",
        "actors": actor_results,
        "consensus": {
            "home_votes": home_votes,
            "away_votes": away_votes,
            "predicted_winner": "home" if home_votes > away_votes else "away" if away_votes > home_votes else "draw",
            "avg_confidence": avg_confidence
        },
        "predicted_line_movement": line_movement,
        "confidence": avg_confidence / 100
    }


def call_openrouter(prompt: str, max_tokens: int = 500) -> Dict:
    """Call OpenRouter API with prompt."""
    import requests
    
    if not OPENROUTER_API_KEY:
        raise ValueError("No OPENROUTER_API_KEY")
    
    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/nativ3ai/hermes",
        "X-Title": "EPL-Market-Sim"
    }
    
    # Use a free model
    data = {
        "model": "openai/gpt-4o-mini",
        "messages": [
            {"role": "system", "content": "You are a sports betting analyst. Respond only in valid JSON."},
            {"role": "user", "content": prompt}
        ],
        "max_tokens": max_tokens,
        "temperature": 0.7
    }
    
    resp = requests.post(url, json=data, headers=headers, timeout=60)
    
    if resp.status_code != 200:
        raise Exception(f"OpenRouter error: {resp.status_code} - {resp.text}")
    
    result = resp.json()
    content = result.get("choices", [{}])[0].get("message", {}).get("content", "{}")
    
    # Try to parse JSON from response
    import json
    import re
    
    # Find JSON in response
    json_match = re.search(r'\{[^{}]*\}', content, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group())
        except:
            pass
    
    return {"raw_response": content}


def analyze_line_movement(simulation_result: Dict, sport: str) -> Dict:
    """Analyze simulation to predict line movement."""
    
    if "error" in simulation_result:
        return simulation_result
    
    # Check for numeric 3-way probability format (EPL/football)
    if simulation_result.get("p_home") is not None and simulation_result.get("p_draw") is not None and simulation_result.get("p_away") is not None:
        p_home = float(simulation_result.get("p_home", 0))
        p_draw = float(simulation_result.get("p_draw", 0))
        p_away = float(simulation_result.get("p_away", 0))
    elif simulation_result.get("probabilities"):
        probs = simulation_result.get("probabilities", {})
        p_home = float(probs.get("p(Home)", 0))
        p_draw = float(probs.get("p(Draw)", 0))
        p_away = float(probs.get("p(Away)", 0))
    else:
        p_home = p_draw = p_away = None

    if p_home is not None and p_draw is not None and p_away is not None:
        
        # Determine predicted outcome
        if p_home >= p_draw and p_home >= p_away:
            predicted = "home"
        elif p_away >= p_draw and p_away >= p_home:
            predicted = "away"
        else:
            predicted = "draw"
        
        confidence = simulation_result.get("confidence", 0)
        
        return {
            "predicted_direction": predicted,
            "confidence": confidence,
            "probabilities": {
                "p(Home)": p_home,
                "p(Draw)": p_draw,
                "p(Away)": p_away
            },
            "summary": f"Consensus: {predicted} ({p_home:.1f}% home, {p_draw:.1f}% draw, {p_away:.1f}% away)"
        }
    
    # Check for MMA/consensus format
    if simulation_result.get("success"):
        consensus = simulation_result.get("consensus", {})
        predicted_winner = consensus.get("predicted_winner", "unknown")
        avg_confidence = consensus.get("avg_confidence", 0)
        line_movement = simulation_result.get("predicted_line_movement", "unknown")
        
        return {
            "predicted_direction": predicted_winner,
            "confidence": avg_confidence / 100 if avg_confidence else 0,
            "line_movement": line_movement,
            "summary": f"Consensus: {predicted_winner} predicted with {avg_confidence:.1f}% confidence. Line expected to move {line_movement}."
        }
    
    # Look for agent consensus
    agents = simulation_result.get("agents", [])
    if agents:
        directions = [a.get("final_position", a.get("stance")) for a in agents]
        
        home_favor = sum(1 for d in directions if d and "home" in str(d).lower())
        away_favor = sum(1 for d in directions if d and "away" in str(d).lower())
        
        total = home_favor + away_favor
        if total > 0:
            return {
                "predicted_direction": "home_favored" if home_favor > away_favor else "away_favored",
                "confidence": abs(home_favor - away_favor) / total,
                "agents_ analyzed": len(agents),
                "summary": f"{home_favor} favor home, {away_favor} favor away"
            }
    
    return {
        "predicted_direction": "unknown",
        "confidence": 0.0,
        "summary": "Could not analyze simulation results"
    }


def run_simulation(sport: str, match_id: int = None, fighter1: str = None, fighter2: str = None):
    """Main simulation runner."""
    
    print(f"\n{'='*60}")
    print(f"EPL & MMA Market Simulation - Real MiroFish Compatible")
    print(f"{'='*60}\n")
    
    # Load configs
    if sport in ["football", "epl"]:
        actor_config = load_config("actors_epl.json")
    else:
        actor_config = load_config("actors_mma.json")
    
    llm_config = load_config("llm_config.json")
    
    # Fetch match data
    if sport in ["football", "epl"] and match_id:
        print(f"Fetching match data for ID {match_id}...")
        match_data = fetch_epl_match(match_id)
        
        if "error" in match_data:
            print(f"❌ Error: {match_data['error']}")
            
            # Try fetching upcoming fixtures instead
            print("\nFetching upcoming fixtures instead...")
            fixtures = fetch_upcoming_epl_fixtures(5)
            if fixtures:
                print(f"Found {len(fixtures)} upcoming matches:\n")
                for i, f in enumerate(fixtures):
                    home = f.get("teams", {}).get("home", {}).get("name", "?")
                    away = f.get("teams", {}).get("away", {}).get("name", "?")
                    date = f.get("fixture", {}).get("date", "")[:10]
                    print(f"  {i+1}. {home} vs {away} ({date})")
                return
    elif sport == "mma":
        # Use provided fighter names or default
        fighter1 = fighter1 or "Islam Makhachev"
        fighter2 = fighter2 or "Arman Tsarukyan"
        match_data = {
            "home_fighter": fighter1,
            "away_fighter": fighter2,
            "event": {"venue": "T-Mobile Arena, Las Vegas", "date": "2026-03-22"},
            "league": {"name": "UFC"}
        }
        print(f"MMA Fight: {fighter1} vs {fighter2}")
    else:
        print("Fetching upcoming fixtures...")
        fixtures = fetch_upcoming_epl_fixtures(5)
        if fixtures:
            match_data = fixtures[0]
            print(f"Using: {match_data.get('teams', {}).get('home', {}).get('name')} vs {match_data.get('teams', {}).get('away', {}).get('name')}")
        else:
            print("No upcoming matches found")
            return
    
    # Fetch weather - use city, not stadium name
    venue_data = match_data.get("fixture", {}).get("venue", {})
    city = venue_data.get("city", "London")  # Use city, not stadium name
    venue = venue_data.get("name", "London")
    match_date = match_data.get("fixture", {}).get("date", "")[:10]
    print(f"\nFetching weather for {city} ({venue})...")
    weather = fetch_weather(city, match_date)
    if "error" not in weather:
        print(f"  Weather: {weather.get('temperature')}°C, {weather.get('description')}")
    else:
        print(f"  Weather: {weather.get('error')}")
    
    # Fetch news
    teams = match_data.get("teams", {})
    home_team = teams.get("home", {}).get("name", "")
    away_team = teams.get("away", {}).get("name", "")
    
    print(f"\nFetching news...")
    home_news = fetch_news(home_team, sport=sport)
    away_news = fetch_news(away_team, sport=sport)
    print(f"  Found {len(home_news)} home news, {len(away_news)} away news")
    
    news = {
        "home_team": home_team,
        "away_team": away_team,
        "home_news": [n.get("title") for n in home_news[:3]],
        "away_news": [n.get("title") for n in away_news[:3]]
    }
    
    # Generate seed packet
    print("\nGenerating seed packet...")
    seed_packet = generate_seed_packet(sport, match_data, weather, news)
    
    # Run MiroFish simulation
    print("Running MiroFish simulation...")
    result = run_mirofish_simulation(seed_packet, actor_config)
    
    # Analyze results
    print("\nAnalyzing results...")
    analysis = analyze_line_movement(result, sport)
    
    # Save results
    match_id_str = str(match_id) if match_id else datetime.now().strftime("%Y%m%d%H%M%S")
    run_id = f"{sport}_{match_id_str}"
    output_path = DATA_DIR / "runs" / f"{run_id}.json"
    output_path.parent.mkdir(exist_ok=True)
    
    output = {
        "run_id": run_id,
        "sport": sport,
        "match_id": match_id,
        "seed_packet": seed_packet,
        "simulation": result,
        "analysis": analysis,
        "timestamp": datetime.now().isoformat()
    }
    
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    
    # Print summary
    print(f"\n{'='*60}")
    print("RESULTS")
    print(f"{'='*60}")
    print(f"Match: {home_team} vs {away_team}")
    print(f"Weather: {weather.get('temperature', 'N/A')}°C, {weather.get('description', 'N/A')}")
    print(f"\nPredicted Line Movement: {analysis.get('predicted_direction', 'unknown')}")
    print(f"Confidence: {analysis.get('confidence', 0):.2%}")
    print(f"Summary: {analysis.get('summary', 'N/A')}")
    print(f"\nResults saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="EPL & MMA Market Simulation v2")
    subparsers = parser.add_subparsers(dest="command")
    
    # Health check
    subparsers.add_parser("health", help="Check system health")
    
    # Run simulation
    run_parser = subparsers.add_parser("run", help="Run market simulation")
    run_parser.add_argument("--sport", choices=["epl", "mma", "football"], default="epl")
    run_parser.add_argument("--match-id", type=int, help="Match ID (EPL)")
    run_parser.add_argument("--fighter1", type=str, help="MMA Fighter 1 (home)")
    run_parser.add_argument("--fighter2", type=str, help="MMA Fighter 2 (away)")
    run_parser.add_argument("--fixtures", action="store_true", help="Show upcoming fixtures")
    
    args = parser.parse_args()
    
    if args.command == "health":
        health_check()
    elif args.command == "run":
        if args.fixtures:
            fixtures = fetch_upcoming_epl_fixtures(10)
            print("\nUpcoming EPL Fixtures:")
            for i, f in enumerate(fixtures):
                teams = f.get("teams", {})
                home = teams.get("home", {}).get("name")
                away = teams.get("away", {}).get("name")
                date = f.get("fixture", {}).get("date", "")[:16].replace("T", " ")
                print(f"  {i+1}. {home} vs {away} - {date}")
        else:
            run_simulation(args.sport, args.match_id, args.fighter1, args.fighter2)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
