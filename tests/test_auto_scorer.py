import importlib.util
from pathlib import Path


def _load_auto_scorer_module():
    root = Path(__file__).resolve().parents[1]
    module_path = root / "prophet-epl" / "scripts" / "auto_scorer.py"
    spec = importlib.util.spec_from_file_location("auto_scorer", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_score_simulation_logloss_uses_actual_outcome_probability():
    auto_scorer = _load_auto_scorer_module()

    sim = {
        "pred": "HOME",
        "home_prob": 80,
        "draw_prob": 10,
        "away_prob": 10,
    }
    actual = {"actual": "away"}

    scored = auto_scorer.score_simulation(sim, actual)

    # Must use p(actual=AWAY)=0.10, not p(pred=HOME)=0.80
    assert scored["log_loss"] == 2.303
    assert scored["correct"] is False


def test_score_simulation_clamps_and_renormalizes_distribution():
    auto_scorer = _load_auto_scorer_module()

    sim = {
        "pred": "HOME",
        "home_prob": 120,
        "draw_prob": -10,
        "away_prob": 10,
    }
    actual = {"actual": "home"}

    scored = auto_scorer.score_simulation(sim, actual)

    # After clamp+normalize: [1.0, 0.0, 0.1] -> [0.90909, 0.0, 0.09091]
    assert round(scored["pH"], 6) == round(1.0 / 1.1, 6)
    assert scored["pD"] == 0.0
    assert round(scored["pA"], 6) == round(0.1 / 1.1, 6)


def test_cmd_run_uses_passed_fixtures_and_skips_existing(tmp_path, monkeypatch):
    auto_scorer = _load_auto_scorer_module()

    # If cmd_run incorrectly calls get_upcoming_fixtures when fixtures are passed,
    # this test should fail immediately.
    def _unexpected_get_upcoming_fixtures(*args, **kwargs):
        raise AssertionError("get_upcoming_fixtures should not be called when fixtures arg is provided")

    monkeypatch.setattr(auto_scorer, "get_upcoming_fixtures", _unexpected_get_upcoming_fixtures)

    existing_id = 101
    new_id = 202

    existing_file = tmp_path / f"epl_{existing_id}.json"
    existing_file.write_text("{}")

    monkeypatch.setattr(auto_scorer, "get_simulation_file", lambda fixture_id: tmp_path / f"epl_{fixture_id}.json")

    processed_fixture_ids = []

    def _fake_run_simulation(fixture):
        fid = fixture["fixture"]["id"]
        processed_fixture_ids.append(fid)
        return {
            "fixture_id": fid,
            "home": fixture["teams"]["home"]["name"],
            "away": fixture["teams"]["away"]["name"],
            "home_prob": 50,
            "draw_prob": 25,
            "away_prob": 25,
            "pred": "HOME",
        }

    monkeypatch.setattr(auto_scorer, "run_simulation", _fake_run_simulation)

    saved_ids = []
    monkeypatch.setattr(auto_scorer, "save_simulation", lambda sim: saved_ids.append(sim["fixture_id"]))

    fixtures = [
        {
            "fixture": {"id": existing_id, "date": "2026-01-01"},
            "teams": {"home": {"name": "A"}, "away": {"name": "B"}},
            "league": {"name": "Premier League"},
        },
        {
            "fixture": {"id": new_id, "date": "2026-01-02"},
            "teams": {"home": {"name": "C"}, "away": {"name": "D"}},
            "league": {"name": "Premier League"},
        },
    ]

    auto_scorer.cmd_run(limit=5, fixtures=fixtures)

    assert processed_fixture_ids == [new_id]
    assert saved_ids == [new_id]
