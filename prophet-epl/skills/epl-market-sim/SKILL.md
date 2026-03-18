---
name: epl-market-sim
description: EPL and MMA market simulation using MiroFish to predict line movements. Combines API-Football/weather data with news to simulate how bookmakers, media, and bettors react before the real crowd.
---

# EPL & MMA Market Simulation Skill

This skill simulates betting market behavior using MiroFish to predict line movements before they happen.

## What It Does

1. **Fetches match data**: API-Football for EPL, UFC/MMA data
2. **Fetches weather**: OpenWeatherMap for venue conditions
3. **Fetches news**: Brave Search for recent team/fighter news
4. **Generates seed packet**: Combines all data for MiroFish
5. **Runs simulation**: MiroFish with betting market actors
6. **Predicts line movement**: Analyzes simulation for market direction

## Sports Supported

| Sport | Use Case |
|-------|----------|
| **EPL** | Predict line movements, detect overreaction to injuries, team news |
| **MMA** | Predict market overreaction to weigh-ins, injuries, fight cancellations |

## Usage

### Health Check
```bash
python3 skills/epl-market-sim/scripts/epl_market_pipeline.py health
```

### Run EPL Simulation
```bash
# Run on specific match
python3 skills/epl-market-sim/scripts/epl_market_pipeline.py run \
  --sport epl --match-id 1208021

# Or just show upcoming fixtures
python3 skills/epl-market-sim/scripts/epl_market_pipeline.py run --fixtures
```

### Run MMA Simulation  
```bash
python3 skills/epl-market-sim/scripts/epl_market_pipeline.py run --sport mma
```

### Counterfactual (What-If)
```bash
# What if key player is injured?
# (Coming soon - requires MiroFish counterfactual API)
```

## Configuration

Edit `configs/` to customize:
- `actors_epl.json` - Bookmakers, media, fans for football
- `actors_mma.json` - Bookmakers, media, analysts for MMA
- `llm_config.json` - OpenRouter/Ollama settings

## Requirements

- **MiroFish**: Run with Docker
  ```bash
  docker run -d -p 3000:3000 -p 5001:5001 ghcr.io/666ghj/mirofish
  ```
- **LLM Provider** (one of):
  - OpenRouter API key (set `OPENROUTER_API_KEY`)
  - Ollama running locally (set `OLLAMA_BASE_URL`)
- **API-Football** (optional): For match data
- **Brave Search** (optional): For news
- **Weather API** (optional): For weather data

## First Checks

```bash
python3 skills/epl-market-sim/scripts/epl_market_pipeline.py health
```

This verifies:
- API keys present
- MiroFish accessible
- Ollama/OpenRouter working
