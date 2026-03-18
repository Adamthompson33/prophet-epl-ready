# Prophet EPL Tools

## External APIs

### OpenWeatherMap API
- **Purpose**: Weather data for EPL match outcome modeling
- **Env Var**: `OPENWEATHERMAP_API_KEY`
- **Rate Limits**: 60 calls/minute (free tier)
- **Documentation**: https://openweathermap.org/api

### Brave Search API
- **Purpose**: Web search for research, news, data retrieval
- **Env Var**: `BRAVE_SEARCH_API_KEY`
- **Rate Limits**: 2000 requests/month (free tier)
- **Documentation**: https://brave.com/search/api/

### AgentMail
- **Purpose**: Inter-agent communication between Mac Mini and Laptop bots
- **Env Var**: `AGENTMAIL_KEY`

## Data Sources

### API-Football
- **Purpose**: EPL match data, fixtures, odds
- **Env Var**: `API_FOOTBALL_KEY`
- **League ID**: Premier League = 39
- **Documentation**: https://www.api-football.com/

### Betfair
- **Purpose**: Live odds, historical data
- **Authentication**: OAuth / app key

## Environment Setup

```bash
export OPENWEATHERMAP_API_KEY="your-key-here"
export BRAVE_SEARCH_API_KEY="your-key-here"
export AGENTMAIL_KEY="your-key-here"
export API_FOOTBALL_KEY="your-key-here"
export BETFAIR_APP_KEY="your-key-here"
```

## Caching

