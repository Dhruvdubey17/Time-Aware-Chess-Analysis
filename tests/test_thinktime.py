"""Offline checks for the think-time / complexity logic (no engine, no data)."""

import numpy as np
import pandas as pd

from chess_strength import thinktime as tt


def test_position_complexity_extremes():
    # One legal move is forced, so trivial.
    forced = tt.position_complexity([50], 1)
    assert forced["complexity"] == 0.0
    assert forced["is_only_move"] is True

    # One move clearly best -> low entropy. Several equal -> near max entropy.
    clear = tt.position_complexity([300, -200, -250, -300], 30)["complexity"]
    tied = tt.position_complexity([20, 20, 20, 20], 30)["complexity"]
    assert clear < 0.5
    assert tied > 0.9
    assert tied > clear

    # Distinctiveness (gap) is the other direction: clear best move -> big gap.
    assert tt.position_complexity([300, -200, -250, -300], 30)["eval_gap_1_2"] == 500


def test_ols_recovers_known_line():
    rng = np.random.default_rng(0)
    n = 5000
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    y = 2.0 + 3.0 * x1 - 1.0 * x2 + rng.normal(scale=0.1, size=n)
    X = np.column_stack([np.ones(n), x1, x2])
    fit = tt.ols_fit(X, y)
    assert np.allclose(fit["beta"], [2.0, 3.0, -1.0], atol=0.02)
    assert fit["r2"] > 0.99


def _synthetic_moves(n=4000, seed=0):
    rng = np.random.default_rng(seed)
    complexity = rng.uniform(0, 1, n)
    regime = rng.choice(["blitz", "rapid"], n)
    base = np.where(regime == "blitz", 3.0, 9.0)
    # Real signal: harder positions cost more time, plus regime baseline + noise.
    time_spent = base + 15.0 * complexity + rng.normal(scale=1.0, size=n)
    time_spent = np.clip(time_spent, 0, None)
    return pd.DataFrame(
        {
            "complexity": complexity,
            "move_number": rng.integers(1, 60, n),
            "log_clock_before": rng.uniform(2, 6, n),
            "game_phase": rng.choice(["opening", "middlegame", "endgame"], n),
            "regime": regime,
            "rating_band": rng.choice(["[0, 1300)", "[1300, 1600)", "[1600, 2000)"], n),
            "time_spent_s": time_spent,
            "log_time_spent": np.log1p(time_spent),
        }
    )


def test_standardized_effect_is_positive_when_signal_exists():
    df = _synthetic_moves()
    eff = tt.standardized_complexity_effect(df)
    assert eff["complexity_beta"] > 0
    assert eff["complexity_p"] < 1e-6


def _load_pilot():
    import importlib.util
    import pathlib

    path = pathlib.Path(__file__).resolve().parents[1] / "scripts" / "complexity_pilot.py"
    spec = importlib.util.spec_from_file_location("complexity_pilot", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_sn_curve_turnover_detection():
    pilot = _load_pilot()
    c = np.linspace(0, 1, 1000)
    # Monotonic rise: complexity tracks time, no turnover.
    mono = pd.DataFrame({"x": c, "time_spent_s": 1 + 10 * c})
    _, pear, turn, _, _ = pilot._sn_curve(mono, "x")
    assert pear > 0.9 and turn is False
    # Humped: rises then falls at the top, the STEP 1 saturation shape.
    hump = pd.DataFrame({"x": c, "time_spent_s": 1 + 10 * c - 30 * np.clip(c - 0.8, 0, None)})
    _, _, turn2, top2, peak2 = pilot._sn_curve(hump, "x")
    assert turn2 is True and top2 < peak2


def test_position_complexity_handles_ten_candidates():
    # New pilot cps lists are length up to 10. Spread evals -> low entropy,
    # tied evals -> high entropy, same as the k=4 case.
    spread = tt.position_complexity([400, 100, 50, 0, -40, -80, -120, -160, -200, -240], 40)
    tied = tt.position_complexity([12, 11, 10, 10, 9, 9, 8, 8, 7, 7], 40)
    assert spread["complexity"] < tied["complexity"]
    assert 0.0 <= spread["complexity"] <= 1.0 and 0.0 <= tied["complexity"] <= 1.0


def test_fitted_mapping_predicts_and_residual_centers(tmp_path):
    df = _synthetic_moves()
    model = tt.fit_expected_thinktime(df)

    # A hard position should expect more time than an easy one, same context.
    easy = df.iloc[[0]].copy(); easy["complexity"] = 0.05
    hard = df.iloc[[0]].copy(); hard["complexity"] = 0.95
    assert tt.expected_thinktime(model, hard)[0] > tt.expected_thinktime(model, easy)[0]

    # Residuals over the training data should sit around zero on average.
    resid = tt.thinktime_residual(model, df)
    assert abs(float(resid.mean())) < 0.5

    # Save/load round-trips and predicts the same.
    curve = pd.DataFrame({"regime": ["blitz"], "complexity_bin": [0], "mean_time": [1.0]})
    tt.save_mapping(model, curve, {"note": "test"}, tmp_path)
    reloaded = tt.load_mapping(tmp_path)
    assert np.allclose(
        tt.expected_thinktime(reloaded, df), tt.expected_thinktime(model, df)
    )
