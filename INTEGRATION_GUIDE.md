# Prophet EPL Pipeline - Integration Guide for Wednesday

## Components Built on Laptop 2

| Component | File | Key Functions |
|-----------|------|---------------|
| Weather loader | `prophet-epl/data_ingest/weather_loader.py` | `fetch_weather(venue, match_date)` |
| API-Football | `prophet-epl/data_ingest/football_api.py` | `get_epl_teams()`, `get_fixtures()`, `get_standings()`, `get_injuries()` |
| Feature pipeline | `prophet-epl/feature_pipeline.py` | `build_match_features(fixture_id)` |
| Dixon-Coles | `prophet-epl/models/dixon_coles.py` | `DixonColesModel.fit(matches)`, `.predict(home, away)` |
| Tests | `tests/test_prophet.py` | Self-contained, no real data needed |

---

## Oracle's Questions Answered

### 1. Dixon-Coles Implementation
**Current:** Single combined implementation (Poisson + home advantage + rho correction)
**Note:** Time decay NOT implemented yet - plain Poisson → home adv → rho only
**To add time decay:** Modify `log_likelihood()` to weight recent matches higher

### 2. Tests Self-Contained?
**Yes.** All tests use sample data:
- `Match("A", "B", 2, 1)` style test data
- No dependency on 33.9k dataset
- Can run: `python -m pytest tests/test_prophet.py -v`

### 3. Feature Pipeline Column Schema
**Current output fields:**
```
fixture_id, match_date, venue, home_team, away_team
fixture_* (50+ fields from API-Football)
weather_temperature_c, weather_precipitation_mm, weather_wind_speed_m_s, weather_is_cold, weather_is_wet, weather_is_windy
home_search_count, away_search_count, home_has_news, away_has_news
home_search_text, away_search_text
```
**Note:** May need schema mapping to match existing evaluator expectations

### 4. API Keys Environment Variables?
**Yes, all use `os.getenv()`:**
- `OPENWEATHERMAP_API_KEY`
- `API_FOOTBALL_KEY`
- `BRAVE_SEARCH_API_KEY`
- `AGENTMAIL_KEY`

---

## Wednesday Integration Checklist

1. [ ] Clone repo to Mac Mini
2. [ ] Copy `.env` or export API keys
3. [ ] Run `pip install -r requirements.txt`
4. [ ] Test: `python -m pytest tests/test_prophet.py -v`
5. [ ] Map feature columns to evaluator schema
6. [ ] Add time decay to Dixon-Coles (optional)
7. [ ] Run on real fixture data

---

## Quick Test Command
```bash
cd prophet-epl-ready
source .env  # or export keys
python -m pytest tests/test_prophet.py -v
```
