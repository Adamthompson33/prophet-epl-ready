#!/usr/bin/env python3
"""
EPL MiroFish Auto-Scorer
========================
Runs MiroFish simulations on upcoming EPL fixtures,
fetches actual results, and scores predictions.

Usage:
  python3 auto_scorer.py status          # Show calibration stats
  python3 auto_scorer.py next            # Show next fixtures
  python3 auto_scorer.py score [id]      # Score specific fixture
  python3 auto_scorer.py run --limit N   # Run N simulations
  python3 auto_scorer.py collect         # Fetch results for all pending
"""

import os
import sys
import json
import math
import argparse
import requests
import re
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional

# ── Load .env ────────────────────────────────────────────────────────────────
ENV_FILE = Path.home() / ".hermes" / ".env"
if ENV_FILE.exists():
    with open(ENV_FILE) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())

PROJ_ENV = Path.home() / "prophet-epl-ready" / ".env"
if PROJ_ENV.exists():
    with open(PROJ_ENV) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())

# ── Keys ────────────────────────────────────────────────────────────────────
FOOTBALL_DATA_KEY = os.environ.get("FOOTBALL_DATA_KEY", "4d5a35987e7d4c579c8e0cf47458f4bc")
FOOTBALL_DATA_URL = "https://api.football-data.org/v4"
CODING_PLAN_KEY = os.environ.get("CODING_PLAN_KEY", "sk-sp-d69279c539fb4c20a36aa9fd9d1759d1")
CODING_PLAN_URL = "https://coding-intl.dashscope.aliyuncs.com/v1"
SIM_MODEL = os.environ.get("SIM_MODEL", "MiniMax-M2.5")

# ── Paths ────────────────────────────────────────────────────────────────────
SKILL_DIR = Path(__file__).parent.parent.parent / "skills" / "epl-market-sim"
CONFIG_DIR = SKILL_DIR / "configs"
DATA_DIR = Path.home() / ".hermes" / "data" / "epl-market-sim" / "scored"
DATA_DIR.mkdir(parents=True, exist_ok=True)

RUNS_DIR = Path.home() / ".hermes" / "data" / "epl-market-sim" / "runs"

# ── EPL Betting Actors ──────────────────────────────────────────────────────
ACTORS = [
    {"name": "PaddyPower", "role": "Sets aggressive lines to attract action, willing to take positions early"},
    {"name": "Bet365", "role": "Sharp bookmaker with efficient markets, prioritizes accuracy"},
    {"name": "Sharp1", "role": "Syndicate bettor who follows steam and market-moving money"},
    {"name": "Public Bettor", "role": "Amateur recreational bettor influenced by media narratives and team popularity"},
    {"name": "Fan", "role": "Passionate fan who backs their team emotionally regardless of value"},
    {"name": "Market Maker", "role": "Liquidity provider who sets lines based on position-taking ability"},
    {"name": "Quantitative Analyst", "role": "Data-driven analyst using expected goals, xG, and advanced metrics"},
    {"name": "Insider Tipster", "role": "Source with team information advantage, moves on injury news and lineups"},
]

# ── API Helpers ──────────────────────────────────────────────────────────────

def apifootball_get(path: str, params: dict) -> dict:
    """Call football-data.org API."""
    if not FOOTBALL_DATA_KEY:
        return {"error": "FOOTBALL_DATA_KEY not set"}
    headers = {
        "X-Auth-Token": FOOTBALL_DATA_KEY
    }
    url = f"{FOOTBALL_DATA_URL}/{path}"
    try:
        resp = requests.get(url, headers=headers, params=params, timeout=15)
        return resp.json()
    except Exception as e:
        return {"error": str(e)}


def normalize_fd_fixture(f: dict) -> dict:
    """Normalize a football-data.org fixture to the format expected by the rest of the code."""
    return {
        "fixture": {
            "id": f.get("id"),
            "date": f.get("utcDate", ""),
            "venue": {"name": f.get("venue", {}).get("name", "Unknown") if isinstance(f.get("venue"), dict) else "Unknown"},
            "status": {"short": f.get("status", "NS")},
            "matchday": f.get("matchday"),
        },
        "teams": {
            "home": {"name": f.get("homeTeam", {}).get("shortName", f.get("homeTeam", {}).get("name", "Home"))},
            "away": {"name": f.get("awayTeam", {}).get("shortName", f.get("awayTeam", {}).get("name", "Away"))},
        },
        "league": {"name": "Premier League"},
        "goals": {
            "home": f.get("score", {}).get("fullTime", {}).get("home"),
            "away": f.get("score", {}).get("fullTime", {}).get("away"),
        },
        "score": {
            "fulltime": {
                "home": f.get("score", {}).get("fullTime", {}).get("home"),
                "away": f.get("score", {}).get("fullTime", {}).get("away"),
            }
        },
        "_raw": f,
    }


def openrouter_complete(prompt: str, max_tokens: int = 300) -> Optional[str]:
    """Call Alibaba Coding Plan API with a single prompt, return text."""
    if not CODING_PLAN_KEY:
        return None
    headers = {
        "Authorization": f"Bearer {CODING_PLAN_KEY}",
        "Content-Type": "application/json"
    }
    body = {
        "model": SIM_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.7
    }
    try:
        resp = requests.post(
            f"{CODING_PLAN_URL}/chat/completions",
            headers=headers, json=body, timeout=90
        )
        data = resp.json()
        if resp.status_code == 200:
            choices = data.get("choices", [])
            if choices:
                return choices[0].get("message", {}).get("content", "")
        else:
            print(f"    Coding Plan error {resp.status_code}: {str(data)[:200]}")
    except Exception as e:
        print(f"    Coding Plan exception: {e}")
    return None


# ── Data Fetching ────────────────────────────────────────────────────────────

def get_upcoming_fixtures(days_ahead: int = 14, limit: int = 20) -> List[dict]:
    """Fetch upcoming EPL fixtures from football-data.org."""
    today = datetime.now()
    future = today + timedelta(days=days_ahead)
    params = {
        "status": "SCHEDULED",
        "dateFrom": today.strftime("%Y-%m-%d"),
        "dateTo": future.strftime("%Y-%m-%d")
    }
    data = apifootball_get("competitions/PL/matches", params)
    raw_fixtures = data.get("matches", [])
    fixtures = [normalize_fd_fixture(f) for f in raw_fixtures]
    print(f"Found {len(fixtures)} upcoming EPL fixtures")
    return fixtures[:limit]


def get_fixture_result(fixture_id: int) -> Optional[dict]:
    """Get actual result for a fixture from football-data.org."""
    data = apifootball_get(f"matches/{fixture_id}", {})
    if "error" in data or data.get("errorCode"):
        return None
    fix = normalize_fd_fixture(data)
    home_goals = fix.get("goals", {}).get("home")
    away_goals = fix.get("goals", {}).get("away")
    if home_goals is None:
        return None
    teams = fix.get("teams", {})
    score = fix.get("score", {})
    ft = score.get("fulltime", {})
    if home_goals > away_goals:
        actual = "home"
    elif away_goals > home_goals:
        actual = "away"
    else:
        actual = "draw"
    return {
        "fixture_id": fixture_id,
        "home_team": teams.get("home", {}).get("name", "?"),
        "away_team": teams.get("away", {}).get("name", "?"),
        "score": f"{home_goals}-{away_goals}",
        "ft_home": ft.get("home"),
        "ft_away": ft.get("away"),
        "actual": actual
    }


# ── Simulation ──────────────────────────────────────────────────────────────

def build_actors_context(fixture: dict) -> str:
    """Build context string for all actors."""
    teams = fixture.get("teams", {})
    home = teams.get("home", {}).get("name", "Unknown")
    away = teams.get("away", {}).get("name", "Unknown")
    fix = fixture.get("fixture", {})
    venue = fix.get("venue", {}).get("name", "Unknown")
    date = fix.get("date", "")[:10]
    league = fixture.get("league", {}).get("name", "Premier League")

    context = f"""MATCH: {home} vs {away}
VENUE: {venue}
DATE: {date}
LEAGUE: {league}

You are one of several betting market participants. Output EXACTLY the format specified.
"""
    return context


def run_simulation(fixture: dict) -> Optional[dict]:
    """Run multi-actor simulation for a fixture, return aggregated probabilities."""
    teams = fixture.get("teams", {})
    home = teams.get("home", {}).get("name", "Unknown")
    away = teams.get("away", {}).get("name", "Unknown")
    fixture_id = fixture.get("fixture", {}).get("id", "unknown")

    print(f"  Running 8 actors for {home} vs {away}...")

    actor_results = []
    base_context = build_actors_context(fixture)

    for actor in ACTORS:
        actor_name = actor["name"]
        actor_role = actor["role"]

        prompt = f"""{base_context}
ROLE: You are {actor_name}.
BEHAVIOR: {actor_role}

You are setting odds for this match. Output EXACTLY this format (nothing else):
p(Home): XX%
p(Draw): XX%
p(Away): XX%

Where XX are integers summing to 100. Be realistic and consistent with your role."""

        response = openrouter_complete(prompt, max_tokens=100)

        if response:
            actor_results.append({"name": actor_name, "response": response})
            # Parse and print what we got
            hm = re.search(r'p\(Home\):\s*(\d+)%?', response, re.I)
            dm = re.search(r'p\(Draw\):\s*(\d+)%?', response, re.I)
            am = re.search(r'p\(Away\):\s*(\d+)%?', response, re.I)
            if hm and dm and am:
                print(f"    {actor_name}: H={hm.group(1)}% D={dm.group(1)}% A={am.group(1)}%")

    # Aggregate
    home_probs, draw_probs, away_probs = [], [], []
    for ar in actor_results:
        resp = str(ar.get("response", ""))
        hm = re.search(r'p\(Home\):\s*(\d+)%?', resp, re.I)
        dm = re.search(r'p\(Draw\):\s*(\d+)%?', resp, re.I)
        am = re.search(r'p\(Away\):\s*(\d+)%?', resp, re.I)
        if hm and dm and am:
            h, d, a = int(hm.group(1)), int(dm.group(1)), int(am.group(1))
            if h > 0 and d > 0 and a > 0:
                home_probs.append(h)
                draw_probs.append(d)
                away_probs.append(a)

    if not home_probs:
        print(f"    WARNING: Could not parse any actor results")
        return None

    avg_home = sum(home_probs) / len(home_probs)
    avg_draw = sum(draw_probs) / len(draw_probs)
    avg_away = sum(away_probs) / len(away_probs)

    # Normalize to sum to 100
    total = avg_home + avg_draw + avg_away
    if total > 0:
        avg_home = avg_home / total * 100
        avg_draw = avg_draw / total * 100
        avg_away = avg_away / total * 100

    result = {
        "fixture_id": fixture_id,
        "home": home,
        "away": away,
        "date": fixture.get("fixture", {}).get("date", "")[:10],
        "model": SIM_MODEL,
        "n_actors": len(home_probs),
        "home_prob": round(avg_home, 1),
        "draw_prob": round(avg_draw, 1),
        "away_prob": round(avg_away, 1),
        "pred": "HOME" if avg_home >= avg_draw and avg_home >= avg_away else ("DRAW" if avg_draw >= avg_away else "AWAY"),
        "actor_results": actor_results,
        "timestamp": datetime.now().isoformat(),
    }

    print(f"  Consensus: Home={avg_home:.1f}% Draw={avg_draw:.1f}% Away={avg_away:.1f}%")
    return result


# ── Persistence ─────────────────────────────────────────────────────────────

def get_simulation_file(fixture_id: int) -> Path:
    return RUNS_DIR / f"epl_{fixture_id}.json"


def save_simulation(sim: dict) -> None:
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    path = get_simulation_file(sim["fixture_id"])
    with open(path, "w") as f:
        json.dump(sim, f, indent=2)
    print(f"  Saved to {path.name}")


def load_simulation(fixture_id: int) -> Optional[dict]:
    path = get_simulation_file(fixture_id)
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return None


def load_pending_simulations() -> List[dict]:
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    pending = []
    for path in sorted(RUNS_DIR.glob("epl_*.json")):
        with open(path) as f:
            sim = json.load(f)
        # Skip old-format files (e.g. from epl_market_pipeline.py)
        if "fixture_id" not in sim:
            continue
        # Check if already scored
        scored_path = DATA_DIR / f"epl_{sim['fixture_id']}.json"
        if not scored_path.exists():
            pending.append(sim)
    return pending


def save_score(sim: dict, actual_result: dict) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    path = DATA_DIR / f"epl_{sim['fixture_id']}.json"
    scored = dict(sim)
    scored.update({
        "actual": actual_result["actual"],
        "score": actual_result["score"],
        "ft_home": actual_result["ft_home"],
        "ft_away": actual_result["ft_away"],
    })
    with open(path, "w") as f:
        json.dump(scored, f, indent=2)


# ── Scoring ─────────────────────────────────────────────────────────────────

def score_simulation(sim: dict, actual: dict) -> dict:
    """Score a single simulation against actual result."""
    pred = sim["pred"].upper()
    actual_res = actual["actual"].upper()
    correct = pred == actual_res

    pH = sim["home_prob"] / 100
    pD = sim["draw_prob"] / 100
    pA = sim["away_prob"] / 100

    # Log loss
    if pred == "HOME":
        log_loss = -math.log(pH + 1e-15)
    elif pred == "DRAW":
        log_loss = -math.log(pD + 1e-15)
    else:
        log_loss = -math.log(pA + 1e-15)

    # Brier score (probability assigned to correct outcome)
    if actual_res == "HOME":
        brier = (1 - pH) ** 2 + pD ** 2 + pA ** 2
    elif actual_res == "DRAW":
        brier = pH ** 2 + (1 - pD) ** 2 + pA ** 2
    else:
        brier = pH ** 2 + pD ** 2 + (1 - pA) ** 2

    return {
        "correct": correct,
        "log_loss": round(log_loss, 3),
        "brier": round(brier, 3),
        "pH": pH, "pD": pD, "pA": pA,
    }


# ── Calibration Report ──────────────────────────────────────────────────────

def calibration_report() -> None:
    scored_dir = DATA_DIR
    if not scored_dir.exists():
        print("No scored matches yet.")
        return

    scored = sorted(scored_dir.glob("epl_*.json"))
    if not scored:
        print("No scored matches yet.")
        return

    results = []
    for path in scored:
        with open(path) as f:
            results.append(json.load(f))

    print(f"\n{'=' * 80}")
    print(f"Current: {len(results)} scored matches")
    print(f"{'=' * 80}")

    print(f"\n{'=' * 80}")
    print(f"MIROFISH EPL CALIBRATION REPORT — {len(results)} MATCHES")
    print(f"{'=' * 80}")

    header = f"{'Match':<35} {'Score':<8} {'Actual':<8} {'Pred':<8} {'?':<6} {'pH':<7} {'pD':<7} {'pA':<7} {'LogLoss':<9} {'Brier':<6}"
    print(header)
    print("-" * 110)

    correct = 0
    total_ll = 0.0
    total_brier = 0.0

    for r in results:
        s = score_simulation(r, {"actual": r["actual"]})
        correct += s["correct"]
        total_ll += s["log_loss"]
        total_brier += s["brier"]

        short_home = r["home"][:18].ljust(18)
        outcome_mark = "PASS" if s["correct"] else "FAIL"
        print(f"{short_home:<35} {r['score']:<8} {r['actual']:<8} {r['pred']:<8} {outcome_mark:<6} {s['pH']*100:5.1f}% {s['pD']*100:5.1f}% {s['pA']*100:5.1f}% {s['log_loss']:<9.3f} {s['brier']:<6.3f}")

    n = len(results)
    accuracy = correct / n * 100
    avg_ll = total_ll / n
    avg_brier = total_brier / n

    print("-" * 110)
    print(f"\nACCURACY:  {correct}/{n} = {accuracy:.1f}%  (theoretical baseline = 50%)")
    print(f"LOG-LOSS:  avg = {avg_ll:.3f}  (lower=better, random 3-way ~1.099)")
    print(f"BRIER:     avg = {avg_brier:.3f}  (lower=better, uniform ~0.667)")

    # Draw analysis
    actual_draws = sum(1 for r in results if r["actual"] == "draw")
    draw_correct = sum(1 for r in results if r["actual"] == "draw" and r["pred"] == "DRAW")
    max_draw_prob = max((r["draw_prob"] for r in results), default=0)
    print(f"\nDRAWS: {actual_draws}/{n} actual draws | {draw_correct} correctly predicted | max p(Draw) in sample: {max_draw_prob:.1f}%")

    needed = max(0, 30 - n)
    if needed > 0:
        print(f"\n⚠  Need {needed} more matches for meaningful sample (target: 30)")
    else:
        print(f"\n✅ Sample size reached (30+ matches)")


# ── Commands ────────────────────────────────────────────────────────────────

def cmd_next():
    fixtures = get_upcoming_fixtures(limit=20)
    if not fixtures:
        print("No upcoming fixtures found.")
        return
    for f in fixtures:
        fix = f.get("fixture", {})
        teams = f.get("teams", {})
        fid = fix.get("id")
        date = fix.get("date", "")[:16]
        home = teams.get("home", {}).get("name", "?")
        away = teams.get("away", {}).get("name", "?")
        print(f"  [{fid}] {home} vs {away} — {date}")


def cmd_run(limit: int = 3, fixtures: List[dict] = None, dry: bool = False):
    if fixtures is None:
        all_fixtures = get_upcoming_fixtures(limit=50)

    # Filter out already-simulated
    new_fixtures = []
    for f in all_fixtures:
        fid = f.get("fixture", {}).get("id")
        if get_simulation_file(fid).exists():
            print(f"  Skipping {fid} — already simulated")
        else:
            new_fixtures.append(f)

    new_fixtures = new_fixtures[:limit]
    if not new_fixtures:
        print("No new fixtures to simulate.")
        return

    print(f"\nRunning {len(new_fixtures)} simulation(s)...\n")
    for f in new_fixtures:
        sim = run_simulation(f)
        if sim:
            save_simulation(sim)
            print(f"  ✅ Done: {sim['home']} vs {sim['away']}\n")

    print(f"\n  Ran {len(new_fixtures)} new simulations")


def cmd_score(fixture_id: int = None):
    pending = load_pending_simulations()
    if fixture_id:
        pending = [p for p in pending if p["fixture_id"] == fixture_id]

    if not pending:
        print("No pending simulations to score.")
        return

    scored_count = 0
    for sim in pending:
        fid = sim["fixture_id"]
        result = get_fixture_result(fid)
        if result and result.get("actual"):
            save_score(sim, result)
            scored_count += 1
            print(f"  Scored: {sim['home']} vs {sim['away']} → {result['actual'].upper()} ({result['score']})")
        else:
            status = get_fixture_status(fid)
            print(f"  Fixture {fid}: {status}")

    print(f"\n  Total scored: {scored_count}")
    calibration_report()


def get_fixture_status(fixture_id: int) -> str:
    data = apifootball_get(f"matches/{fixture_id}", {})
    if "error" in data:
        return "unknown"
    return data.get("status", "unknown")


def cmd_collect():
    pending = load_pending_simulations()
    if not pending:
        print("No pending simulations.")
        return

    print(f"Unscored simulations: {len(pending)}")
    for p in pending:
        fid = p["fixture_id"]
        status = get_fixture_status(fid)
        print(f"    {fid}: {p['home']} {p.get('score','')} → {p['pred']} [{status}]")

    scored_count = 0
    for sim in pending:
        fid = sim["fixture_id"]
        result = get_fixture_result(fid)
        if result and result.get("actual"):
            save_score(sim, result)
            scored_count += 1

    if scored_count > 0:
        calibration_report()
    else:
        print("\n  No results available yet.")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="EPL MiroFish Auto-Scorer")
    parser.add_argument("cmd", nargs="?", default="status", help="Command: next, run, score, collect, status")
    parser.add_argument("id", nargs="?", type=int, help="Fixture ID")
    parser.add_argument("--limit", "-n", type=int, default=3, help="Number of fixtures to simulate")
    parser.add_argument("--dry", action="store_true", help="Dry run")
    parser.add_argument("--status", action="store_true", help="Show calibration report")

    args = parser.parse_args()
    cmd = args.cmd

    if cmd == "next":
        cmd_next()
    elif cmd == "run":
        cmd_run(limit=args.limit, dry=args.dry)
    elif cmd == "score":
        cmd_score(args.id)
    elif cmd == "collect":
        cmd_collect()
    elif cmd == "status":
        calibration_report()
    else:
        print(f"Unknown command: {cmd}")
        parser.print_help()


if __name__ == "__main__":
    main()
