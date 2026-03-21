"""
Dixon-Coles Model for EPL Match Outcome Prediction.

The Dixon-Coles model is a Poisson-based model for predicting football match scores.
It accounts for:
- Team attack strength
- Team defense strength  
- Home advantage
- Draw correlation (rho) for low-scoring draws

Reference: Dixon & Coles (1997) - Modelling Association Football Scores
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from scipy.special import gammaln
from scipy.optimize import minimize
import json


@dataclass
class Team:
    """Represents a team's attacking and defensive strength."""
    name: str
    attack: float = 0.0
    defense: float = 0.0
    home_attack: float = 0.0
    away_attack: float = 0.0
    
    def __repr__(self):
        return f"{self.name}: att={self.attack:.3f}, def={self.defense:.3f}"


@dataclass
class Match:
    """Represents a single match."""
    home_team: str
    away_team: str
    home_goals: int
    away_goals: int
    date: str = None  # Optional: YYYY-MM-DD format for time decay
    
    @property
    def is_draw(self) -> bool:
        return self.home_goals == self.away_goals
    
    @property
    def is_home_win(self) -> bool:
        return self.home_goals > self.away_goals


class DixonColesModel:
    """
    Dixon-Coles Poisson model for football prediction.
    
    The expected goals for each team are:
    - Home: exp(attack_home + defense_away + home_advantage)
    - Away: exp(attack_away + defense_home)
    
    Parameters are estimated via maximum likelihood.
    
    Time Decay:
    Recent matches weight more than old matches. Uses exponential decay:
    weight = exp(-decay_rate * days_since_match)
    where decay_rate = ln(2) / half_life (default ~180 days)
    """
    
    def __init__(self, rho: float = 0.0, home_adv: float = 0.0, 
                 decay_half_life: float = 180.0):
        """
        Initialize model with parameters.
        
        Args:
            rho: Draw correlation parameter
            home_adv: Home advantage (goals)
            decay_half_life: Days for weight to halve (default 180 days)
                             Set to None to disable time decay
        """
        self.rho = rho  # Draw correlation
        self.home_adv = home_adv  # Home advantage
        self.decay_half_life = decay_half_life  # Time decay half-life in days
        self.decay_rate = np.log(2) / decay_half_life if decay_half_life else 0
        self.teams: Dict[str, Team] = {}
        self.attack_strengths: Dict[str, float] = {}
        self.defense_strengths: Dict[str, float] = {}
        self.baseline = 0.0  # Baseline goals (league average)
        self.fitted = False
        
    def _calculate_time_weight(self, match_date: str, reference_date: str = None) -> float:
        """
        Calculate time decay weight for a match.
        
        Weight = exp(-decay_rate * days_since_match)
        Recent matches have weight close to 1.0
        Old matches have weight approaching 0.
        """
        if match_date is None or self.decay_half_life is None:
            return 1.0  # No decay
        
        from datetime import datetime
        
        try:
            match_dt = datetime.strptime(match_date, "%Y-%m-%d")
            if reference_date:
                ref_dt = datetime.strptime(reference_date, "%Y-%m-%d")
            else:
                ref_dt = datetime.now()
            
            days_since = (ref_dt - match_dt).days
            if days_since < 0:
                days_since = 0  # Future matches treated as today
            
            weight = np.exp(-self.decay_rate * days_since)
            return max(weight, 0.01)  # Floor at 1% to prevent near-zero weights
            
        except (ValueError, TypeError):
            return 1.0  # Invalid date format, no decay
        
    def _get_team(self, name: str) -> Team:
        """Get or create team."""
        if name not in self.teams:
            self.teams[name] = Team(name=name)
        return self.teams[name]
    
    def _poisson_log_likelihood(self, observed: int, expected: float) -> float:
        """Log probability of observing 'observed' goals given 'expected'."""
        # log(P(X=k)) = k*log(lambda) - lambda - log(k!)
        # Using gammaln for log(k!) = lgamma(k+1)
        return observed * np.log(expected + 1e-10) - expected - gammaln(observed + 1)
    
    def _dc_correction(self, home_goals: int, away_goals: int, 
                       lambda_home: float, lambda_away: float) -> float:
        """
        Dixon-Coles correction for low-scoring draws.
        Reduces probability of 0-0 and 1-1 draws slightly.
        """
        if home_goals == 0 and away_goals == 0:
            return 1 + self.rho * lambda_home * lambda_away
        elif home_goals == 1 and away_goals == 1:
            return 1 + self.rho
        elif home_goals == 0 and away_goals == 1:
            return 1 - self.rho * lambda_home
        elif home_goals == 1 and away_goals == 0:
            return 1 - self.rho * lambda_away
        return 1.0
    
    def log_likelihood(self, params: np.ndarray, matches: List[Match], 
                       reference_date: str = None) -> float:
        """
        Calculate negative log-likelihood for optimization.
        
        params: [baseline, home_adv, rho, team1_att, team1_def, team2_att, ...]
        
        Each match is weighted by time decay:
        - Recent matches weight more
        - Older matches weight less
        - weight = exp(-decay_rate * days_since_match)
        """
        n_teams = (len(params) - 3) // 2
        
        baseline = params[0]
        home_adv = params[1]
        rho = params[2]
        
        attack = params[3:3+n_teams]
        defense = params[3+n_teams:]
        
        team_names = list(self.teams.keys())
        
        ll = 0.0
        for match in matches:
            i = team_names.index(match.home_team)
            j = team_names.index(match.away_team)
            
            lambda_home = np.exp(baseline + home_adv + attack[i] - defense[j])
            lambda_away = np.exp(baseline + attack[j] - defense[i])
            
            # Home team goals
            ll_home = self._poisson_log_likelihood(match.home_goals, lambda_home)
            # Away team goals  
            ll_away = self._poisson_log_likelihood(match.away_goals, lambda_away)
            # DC correction
            dc = self._dc_correction(match.home_goals, match.away_goals, 
                                      lambda_home, lambda_away)
            
            match_ll = ll_home + ll_away + np.log(dc + 1e-10)
            
            # Apply time decay weight
            time_weight = self._calculate_time_weight(match.date, reference_date)
            weighted_ll = match_ll * time_weight
            
            ll += weighted_ll
        
        return -ll  # Return negative for minimization
    
    def fit(self, matches: List[Match], max_iter: int = 1000, 
            reference_date: str = None) -> 'DixonColesModel':
        """
        Fit the model to historical match data.
        
        Args:
            matches: List of historical matches with scores
            max_iter: Maximum optimization iterations
            reference_date: Date to calculate time decay from (YYYY-MM-DD)
                           Defaults to current date if not provided
        """
        # Build team index
        teams = set()
        for m in matches:
            teams.add(m.home_team)
            teams.add(m.away_team)
        self.teams = {t: Team(name=t) for t in sorted(teams)}
        team_names = list(self.teams.keys())
        
        # Initial parameter guesses
        n_teams = len(team_names)
        
        # Calculate average goals
        total_home = sum(m.home_goals for m in matches) / len(matches)
        total_away = sum(m.away_goals for m in matches) / len(matches)
        baseline = np.log((total_home + total_away) / 2)
        
        # Initial params: [baseline, home_adv, rho, attack[], defense[]]
        initial = np.zeros(3 + 2 * n_teams)
        initial[0] = baseline
        initial[1] = 0.1  # Initial home advantage
        initial[2] = 0.0  # Initial rho
        
        # Optimize
        bounds = [(None, None)] * 3 + [(-2, 2)] * (2 * n_teams)
        
        result = minimize(
            self.log_likelihood,
            initial,
            args=(matches, reference_date),
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': max_iter}
        )
        
        # Store results
        self.baseline = result.x[0]
        self.home_adv = result.x[1]
        self.rho = result.x[2]
        
        for i, name in enumerate(team_names):
            self.teams[name].attack = result.x[3 + i]
            self.teams[name].defense = result.x[3 + n_teams + i]
            
        self.attack_strengths = {n: t.attack for n, t in self.teams.items()}
        self.defense_strengths = {n: t.defense for n, t in self.teams.items()}
        self.fitted = True
        
        return self
    
    def predict(self, home_team: str, away_team: str) -> Dict:
        """
        Predict match outcome probabilities.
        
        Returns:
            Dictionary with expected goals, win/draw/loss probabilities
        """
        if not self.fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        h = self.teams.get(home_team)
        a = self.teams.get(away_team)
        
        if h is None or a is None:
            raise ValueError(f"Unknown team: {home_team} or {away_team}")
        
        # Expected goals
        lambda_home = np.exp(self.baseline + self.home_adv + h.attack - a.defense)
        lambda_away = np.exp(self.baseline + a.attack - h.defense)
        
        # Calculate probability distribution over score combinations
        max_goals = 10  # Cap at reasonable number
        home_probs = []
        away_probs = []
        
        for k in range(max_goals + 1):
            # Poisson probabilities
            p_home = np.exp(k * np.log(lambda_home) - lambda_home - gammaln(k + 1))
            p_away = np.exp(k * np.log(lambda_away) - lambda_away - gammaln(k + 1))
            home_probs.append(p_home)
            away_probs.append(p_away)
        
        home_probs = np.array(home_probs)
        away_probs = np.array(away_probs)
        
        # Joint probability matrix
        joint = np.outer(home_probs, away_probs)
        
        # Apply DC correction
        for i in range(max_goals + 1):
            for j in range(max_goals + 1):
                dc = self._dc_correction(i, j, lambda_home, lambda_away)
                joint[i, j] *= dc
        
        # Normalize
        joint /= joint.sum()
        
        # Outcome probabilities
        home_win = joint[np.triu_indices(max_goals + 1, 1)].sum()
        away_win = joint[np.tril_indices(max_goals + 1, -1)].sum()
        draw = joint.diagonal().sum()
        
        return {
            "home_team": home_team,
            "away_team": away_team,
            "expected_home_goals": lambda_home,
            "expected_away_goals": lambda_away,
            "home_win_prob": home_win,
            "draw_prob": draw,
            "away_win_prob": away_win,
            "most_likely_score": self._most_likely_score(joint, max_goals),
            "joint_distribution": joint.tolist()[:5][:5],  # Top-left 5x5
        }
    
    def _most_likely_score(self, joint: np.ndarray, max_goals: int) -> Tuple[int, int]:
        """Find most likely score."""
        idx = np.unravel_index(joint.argmax(), joint.shape)
        return idx[0], idx[1]
    
    def get_team_ratings(self) -> List[Dict]:
        """Get team attacking and defensive ratings."""
        if not self.fitted:
            return []
        
        ratings = []
        for name, team in self.teams.items():
            ratings.append({
                "team": name,
                "attack": team.attack,
                "defense": team.defense,
                "overall": team.attack - team.defense,  # Positive = better than average
            })
        
        return sorted(ratings, key=lambda x: x["overall"], reverse=True)
    
    def to_dict(self) -> Dict:
        """Serialize model to dictionary."""
        return {
            "baseline": self.baseline,
            "home_advantage": self.home_adv,
            "rho": self.rho,
            "decay_half_life": self.decay_half_life,
            "teams": {
                n: {"attack": t.attack, "defense": t.defense} 
                for n, t in self.teams.items()
            }
        }
    @classmethod
    def from_dict(cls, data: Dict) -> 'DixonColesModel':
        """Load model from dictionary."""
        model = cls(
            rho=data.get("rho", 0), 
            home_adv=data.get("home_advantage", 0),
            decay_half_life=data.get("decay_half_life", 180)
        )
        model.baseline = data.get("baseline", 0)
        model.teams = {
            n: Team(n, a.get("attack", 0), a.get("defense", 0)) 
            for n, a in data.get("teams", {}).items()
        }
        model.attack_strengths = {n: t.attack for n, t in model.teams.items()}
        model.defense_strengths = {n: t.defense for n, t in model.teams.items()}
        model.fitted = True
        return model


# ============ Demo ============

def create_sample_matches() -> List[Match]:
    """Create sample EPL matches for testing with dates."""
    from datetime import datetime, timedelta
    
    today = datetime.now()
    return [
        # Recent results (last 30 days)
        Match("Manchester City", "Arsenal", 3, 1, date=(today - timedelta(days=5)).strftime("%Y-%m-%d")),
        Match("Liverpool", "Chelsea", 2, 1, date=(today - timedelta(days=10)).strftime("%Y-%m-%d")),
        Match("Arsenal", "Tottenham", 2, 0, date=(today - timedelta(days=15)).strftime("%Y-%m-%d")),
        Match("Manchester United", "Newcastle", 1, 0, date=(today - timedelta(days=20)).strftime("%Y-%m-%d")),
        Match("Chelsea", "Arsenal", 1, 2, date=(today - timedelta(days=25)).strftime("%Y-%m-%d")),
        Match("Tottenham", "Liverpool", 1, 3, date=(today - timedelta(days=30)).strftime("%Y-%m-%d")),
        # Older results (180+ days ago - lower weight)
        Match("Newcastle", "Manchester City", 0, 2, date=(today - timedelta(days=200)).strftime("%Y-%m-%d")),
        Match("Arsenal", "Manchester United", 1, 0, date=(today - timedelta(days=210)).strftime("%Y-%m-%d")),
        Match("Chelsea", "Tottenham", 2, 2, date=(today - timedelta(days=220)).strftime("%Y-%m-%d")),
        Match("Liverpool", "Arsenal", 1, 1, date=(today - timedelta(days=365)).strftime("%Y-%m-%d")),
    ]


def main():
    """Demo the Dixon-Coles model."""
    print("Dixon-Coles Model Demo")
    print("=" * 50)
    
    # Create sample matches
    matches = create_sample_matches()
    print(f"Loaded {len(matches)} sample matches")
    
    # Fit model WITH time decay (half-life = 180 days)
    print(f"\nFitting with time decay (half-life = 180 days)...")
    model = DixonColesModel(decay_half_life=180)
    model.fit(matches)
    
    print(f"\nModel Parameters:")
    print(f"  Baseline: {model.baseline:.4f}")
    print(f"  Home Advantage: {model.home_adv:.4f}")
    print(f"  Rho (draw corr): {model.rho:.4f}")
    print(f"  Decay Half-Life: {model.decay_half_life} days")
    
    # Team ratings
    print(f"\nTeam Ratings:")
    for r in model.get_team_ratings():
        print(f"  {r['team']:<20} att: {r['attack']:>6.3f}  def: {r['defense']:>6.3f}  overall: {r['overall']:>+6.3f}")
    
    # Make predictions
    print(f"\nPredictions:")
    test_matches = [
        ("Manchester City", "Arsenal"),
        ("Liverpool", "Chelsea"),
        ("Tottenham", "Newcastle"),
    ]
    
    for home, away in test_matches:
        pred = model.predict(home, away)
        print(f"\n  {home} vs {away}:")
        print(f"    Expected: {pred['expected_home_goals']:.2f} - {pred['expected_away_goals']:.2f}")
        print(f"    Probabilities: {pred['home_win_prob']:.1%} home, {pred['draw_prob']:.1%} draw, {pred['away_win_prob']:.1%} away")
        print(f"    Most likely: {pred['most_likely_score'][0]}-{pred['most_likely_score'][1]}")


if __name__ == "__main__":
    main()
