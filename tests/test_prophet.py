"""
Test suite for Prophet EPL components.
Run with: python -m pytest tests/ -v
"""

import pytest
import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


class TestWeatherLoader:
    """Tests for weather_loader.py"""
    
    def test_api_key_set(self):
        """Test that API key is set."""
        key = os.getenv("OPENWEATHERMAP_API_KEY")
        assert key is not None, "OPENWEATHERMAP_API_KEY not set"
        assert len(key) > 10, "API key too short"
    
    def test_import_weather_loader(self):
        """Test weather_loader can be imported."""
        from prophet_epl.data_ingest import weather_loader
        assert hasattr(weather_loader, 'fetch_weather')
    
    def test_venue_mapping(self):
        """Test venue to city mapping exists."""
        from prophet_epl.data_ingest.weather_loader import VENUE_MAP
        assert "Craven Cottage, London" in VENUE_MAP
        assert "Old Trafford, Manchester" in VENUE_MAP


class TestFootballAPI:
    """Tests for football_api.py"""
    
    def test_api_key_set(self):
        """Test that API key is set."""
        key = os.getenv("API_FOOTBALL_KEY")
        assert key is not None, "API_FOOTBALL_KEY not set"
    
    def test_import_football_api(self):
        """Test football_api can be imported."""
        from prophet_epl.data_ingest import football_api
        assert hasattr(football_api, 'get_epl_teams')
        assert hasattr(football_api, 'get_fixtures')
        assert hasattr(football_api, 'get_standings')


class TestFeaturePipeline:
    """Tests for feature_pipeline.py"""
    
    def test_import_pipeline(self):
        """Test feature_pipeline can be imported."""
        from prophet_epl import feature_pipeline
        assert hasattr(feature_pipeline, 'build_match_features')
        assert hasattr(feature_pipeline, 'get_weather_features')
    
    def test_rate_limit_function(self):
        """Test rate limiting function exists."""
        from prophet_epl.feature_pipeline import rate_limit
        # Should not raise
        rate_limit("test", 60)
    
    def test_venue_fallback_map(self):
        """Test fallback venue mapping exists."""
        from prophet_epl.feature_pipeline import VENUE_FALLBACK_MAP
        assert len(VENUE_FALLBACK_MAP) > 10
        assert "Old Trafford" in VENUE_FALLBACK_MAP


class TestDixonColes:
    """Tests for Dixon-Coles model"""
    
    def test_import_dixon_coles(self):
        """Test Dixon-Coles model can be imported."""
        from prophet_epl.models.dixon_coles import DixonColesModel, Match
        assert DixonColesModel is not None
    
    def test_create_match(self):
        """Test Match dataclass."""
        from prophet_epl.models.dixon_coles import Match
        m = Match("Team A", "Team B", 2, 1)
        assert m.home_team == "Team A"
        assert m.away_team == "Team B"
        assert m.home_goals == 2
        assert m.away_goals == 1
        assert m.is_home_win == True
        assert m.is_draw == False
    
    def test_model_initialization(self):
        """Test model can be initialized."""
        from prophet_epl.models.dixon_coles import DixonColesModel
        model = DixonColesModel(rho=0.1, home_adv=0.2)
        assert model.rho == 0.1
        assert model.home_adv == 0.2
        assert model.fitted == False
    
    def test_model_fit(self):
        """Test model can be fitted."""
        from prophet_epl.models.dixon_coles import DixonColesModel, Match
        
        # Create simple test data
        matches = [
            Match("A", "B", 2, 1),
            Match("B", "A", 1, 2),
            Match("A", "C", 3, 0),
            Match("C", "A", 0, 1),
            Match("B", "C", 1, 1),
            Match("C", "B", 2, 2),
        ]
        
        model = DixonColesModel()
        model.fit(matches, max_iter=100)
        
        assert model.fitted == True
        assert model.baseline is not None
        assert len(model.teams) == 3
    
    def test_model_predict(self):
        """Test model can make predictions."""
        from prophet_epl.models.dixon_coles import DixonColesModel, Match
        
        matches = [
            Match("A", "B", 2, 1),
            Match("B", "A", 1, 2),
            Match("A", "C", 3, 0),
            Match("C", "A", 0, 1),
        ]
        
        model = DixonColesModel()
        model.fit(matches, max_iter=100)
        
        pred = model.predict("A", "B")
        
        assert "home_win_prob" in pred
        assert "draw_prob" in pred
        assert "away_win_prob" in pred
        assert "expected_home_goals" in pred
        assert "expected_away_goals" in pred
        # Probabilities should sum reasonably close to 1
        total = pred["home_win_prob"] + pred["draw_prob"] + pred["away_win_prob"]
        assert 0.9 < total < 1.1
    
    def test_model_serialization(self):
        """Test model can be saved and loaded."""
        from prophet_epl.models.dixon_coles import DixonColesModel, Match
        
        matches = [
            Match("A", "B", 2, 1),
            Match("B", "A", 1, 2),
        ]
        
        model = DixonColesModel()
        model.fit(matches, max_iter=50)
        
        # Serialize
        data = model.to_dict()
        assert "baseline" in data
        assert "teams" in data
        
        # Deserialize
        model2 = DixonColesModel.from_dict(data)
        assert model2.fitted == True
        assert len(model2.teams) == len(model.teams)
    
    def test_time_decay_weights_recent_higher(self):
        """Test that recent matches contribute more than older matches."""
        from prophet_epl.models.dixon_coles import DixonColesModel, Match
        from datetime import datetime, timedelta
        
        today = datetime.now()
        
        # Recent match (1 week ago)
        recent_date = (today - timedelta(days=7)).strftime("%Y-%m-%d")
        # Old match (2 years ago)
        old_date = (today - timedelta(days=730)).strftime("%Y-%m-%d")
        
        # Matches with same score but different dates
        recent_match = Match("A", "B", 2, 1, date=recent_date)
        old_match = Match("A", "B", 2, 1, date=old_date)
        
        # Create model with time decay
        model = DixonColesModel(decay_half_life=180)
        
        # Get time weights
        recent_weight = model._calculate_time_weight(recent_date)
        old_weight = model._calculate_time_weight(old_date)
        
        # Recent should have higher weight than old
        assert recent_weight > old_weight, f"Recent ({recent_weight}) should be > Old ({old_weight})"
        # Recent should be close to 1.0
        assert recent_weight > 0.9, f"Recent weight should be > 0.9, got {recent_weight}"
        # Old should be significantly less than 1.0
        assert old_weight < 0.5, f"Old weight should be < 0.5, got {old_weight}"
    
    def test_time_decay_disabled(self):
        """Test that time decay can be disabled."""
        from prophet_epl.models.dixon_coles import DixonColesModel, Match
        
        model_no_decay = DixonColesModel(decay_half_life=None)
        model_with_decay = DixonColesModel(decay_half_life=180)
        
        # With decay_half_life=None, all weights should be 1.0
        assert model_no_decay._calculate_time_weight("2020-01-01") == 1.0
        
        # With decay, weights should differ by date
        recent_weight = model_with_decay._calculate_time_weight("2024-01-01")
        old_weight = model_with_decay._calculate_time_weight("2020-01-01")
        assert recent_weight != old_weight
    
    def test_time_decay_parameter_saved(self):
        """Test that time decay parameters are saved in serialization."""
        from prophet_epl.models.dixon_coles import DixonColesModel, Match
        
        model = DixonColesModel(decay_half_life=90)
        
        # Serialize
        data = model.to_dict()
        
        # Check decay params are saved
        assert "decay_half_life" in data
        assert data["decay_half_life"] == 90


class TestIntegration:
    """Integration tests across components."""
    
    def test_tools_md_exists(self):
        """Test TOOLS.md exists."""
        tools_path = project_root / "TOOLS.md"
        assert tools_path.exists(), "TOOLS.md not found"
    
    def test_env_file_exists(self):
        """Test .env file exists."""
        env_path = project_root / ".env"
        assert env_path.exists(), ".env not found"
    
    def test_all_api_keys_in_env(self):
        """Test all required API keys are in .env."""
        env_path = project_root / ".env"
        with open(env_path) as f:
            content = f.read()
        
        assert "OPENWEATHERMAP_API_KEY" in content
        assert "API_FOOTBALL_KEY" in content
        assert "BRAVE_SEARCH_API_KEY" in content


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
