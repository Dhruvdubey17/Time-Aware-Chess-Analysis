"""Phase F: fit a bullet-specific expected think-time model.

The blitz/rapid mapping does not transfer to bullet: bullet think-times are
bimodal, a big near-zero floor (premoves and snap moves) plus a spread of real
thinks. So we fit bullet as its own regime with a two-part (hurdle) shape:

  part 1  the premove / sub-floor split, already decided from clocks by
          backend.premove (deterministic, not learned),
  part 2  a log-normal regression of think-time on the GENUINE moves only, with
          features Maia dispersion, phase, normalized clock remaining, rating
          band, side, and increment.

Premoves and sub-floor moves are excluded from part 2 so they cannot bias the
think-time expectation; they are handled by the guardrail, never upgraded.

    python scripts/build_bullet_thinktime.py --games 150000 --maia-cap 600000

Reads the 2017-04 dump already in data/raw. Writes the mapping to
assets/thinktime/bullet, parallel to the blitz/rapid one. Bullet clocks in
public Lichess data are whole seconds, so this fit is on whole-second think
times; that limit is stated in the report and carried into the app messaging.
"""

from __future__ import annotations

import argparse
import io
import sys
import time
from pathlib import Path

import chess
import chess.pgn
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend import maia_client, premove
from backend.analyze import DEFAULT_ELO, _maia_model_regime, band_label
from backend.cache import FenCache
from chess_strength import maia
from chess_strength.features import game_phase
from chess_strength.stream_filter import _open_lines, classify_time_control, iter_games

BOOK_PLIES = 12
MIN_THINK_S = 0.3
OUT_DIR = ROOT / "assets" / "thinktime" / "bullet"
DUMP = ROOT / "data" / "raw" / "lichess_db_standard_rated_2017-04.pgn.zst"
NUM = ["maia_dispersion", "norm_clock_remaining", "move_number"]
CAT = ["game_phase", "rating_band", "side", "increment"]
TARGET = "log_time_spent"


def _int(headers, key):
    try:
        return int(headers.get(key, ""))
    except ValueError:
        return None


def collect(max_games: int):
    """Walk bullet games from the dump, returning genuine-move records for the
    part-2 fit and the overall premove/sub-floor/genuine counts (part 1)."""
    rates = {premove.PREMOVE: 0, premove.SUB_FLOOR_AMBIGUOUS: 0,
             premove.GENUINE: 0, premove.UNKNOWN: 0}
    rows = []
    seen = kept = 0
    t0 = time.time()
    for text in iter_games(_open_lines(DUMP)):
        if "%clk" not in text:
            continue
        game = chess.pgn.read_game(io.StringIO(text))
        if game is None:
            continue
        h = game.headers
        tc = h.get("TimeControl", "")
        if classify_time_control(tc) != "bullet":
            continue
        seen += 1
        base = int(tc.split("+")[0]) if tc.split("+")[0].isdigit() else 0
        inc = int(tc.split("+")[1]) if "+" in tc and tc.split("+")[1].isdigit() else 0
        welo, belo = _int(h, "WhiteElo"), _int(h, "BlackElo")

        board = game.board()
        nodes = list(game.mainline())
        clocks = [nd.clock() for nd in nodes]
        subsecond = premove.has_subsecond_clocks(clocks)
        prev = {"white": float(base), "black": float(base)}
        for ply, nd in enumerate(nodes, start=1):
            side = "white" if board.turn == chess.WHITE else "black"
            fen = board.fen()
            clk = nd.clock()
            board.push(nd.move)
            if clk is None:
                rates[premove.UNKNOWN] += 1
                continue
            cb = prev[side]
            move_time = cb + inc - clk           # signed, premove signature kept
            time_spent = max(0.0, move_time)     # clamped for the think-time value
            prev[side] = clk
            status = premove.classify(move_time, ply <= 2, subsecond)
            rates[status] += 1
            if status != premove.GENUINE or time_spent < MIN_THINK_S or base <= 0:
                continue
            elo = (welo if side == "white" else belo) or DEFAULT_ELO
            rows.append((fen, band_label(elo), side, "yes" if inc else "no",
                         game_phase(fen, ply, BOOK_PLIES), float(ply),
                         cb / base, np.log1p(time_spent)))
        kept += 1
        if kept >= max_games:
            break
        if kept % 20000 == 0:
            print(f"  parsed {kept} bullet games, {len(rows)} genuine moves, "
                  f"{time.time()-t0:.0f}s", flush=True)
    df = pd.DataFrame(rows, columns=["fen", "rating_band", "side", "increment",
                                     "game_phase", "move_number",
                                     "norm_clock_remaining", TARGET])
    return df, rates, seen


def add_maia(df: pd.DataFrame, maia_cap: int, workers: int) -> pd.DataFrame:
    """Attach Maia dispersion (blitz model, player band) to a capped random
    sample of the genuine moves. This is the heavy step."""
    if len(df) > maia_cap:
        df = df.sample(maia_cap, random_state=0).reset_index(drop=True)
    regime = _maia_model_regime("bullet")  # bullet maps to the blitz Maia model
    band_elo = {b: maia.band_to_elo(b) for b in df["rating_band"].unique()}
    tasks = [(r.fen, regime, band_elo[r.rating_band]) for r in df.itertuples()]
    cache = FenCache(ROOT / "data" / "processed" / "app_cache.sqlite")
    try:
        t = time.time()
        disp = maia_client.dispersion(list(dict.fromkeys(tasks)), cache, workers=workers,
                                      progress=lambda d, n: print(f"  maia {d}/{n}", flush=True)
                                      if d % 50000 == 0 else None)
        print(f"  maia done in {time.time()-t:.0f}s", flush=True)
    finally:
        cache.close()
    df["maia_dispersion"] = [disp.get((r.fen, regime, band_elo[r.rating_band]))
                             for r in df.itertuples()]
    return df.dropna(subset=["maia_dispersion"]).reset_index(drop=True)


def fit_and_report(df: pd.DataFrame, rates: dict, seen: int) -> None:
    from scipy import stats
    from sklearn.compose import ColumnTransformer
    from sklearn.linear_model import LinearRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder

    # Held-out split to report honest fit, then refit on all for the saved model.
    rng = np.random.default_rng(0)
    m = rng.random(len(df)) < 0.8

    def make():
        pre = ColumnTransformer([("cat", OneHotEncoder(handle_unknown="ignore"), CAT),
                                 ("num", "passthrough", NUM)])
        return Pipeline([("pre", pre), ("lin", LinearRegression())])

    model = make()
    model.fit(df[m][CAT + NUM], df[m][TARGET])
    pred = model.predict(df[~m][CAT + NUM])
    resid = df[~m][TARGET].to_numpy() - pred
    ss_res = float((resid ** 2).sum())
    ss_tot = float(((df[~m][TARGET] - df[~m][TARGET].mean()) ** 2).sum())
    r2 = 1 - ss_res / ss_tot if ss_tot else 0.0
    rho, p = stats.spearmanr(df["maia_dispersion"], df[TARGET])

    total = sum(rates.values())
    print("\n" + "=" * 70)
    print(f"bullet games parsed: {seen}   moves: {total}")
    for k in (premove.GENUINE, premove.SUB_FLOOR_AMBIGUOUS, premove.PREMOVE, premove.UNKNOWN):
        print(f"  {k:20} {rates[k]:>9}  ({rates[k]/total*100:4.1f}%)")
    print(f"\ngenuine moves with Maia (part-2 fit set): {len(df)}")
    print(f"held-out R^2 (log think-time): {r2:.3f}")
    print(f"Spearman(Maia dispersion, think-time) on genuine moves: rho={rho:.3f} p={p:.1e}")
    print(f"held-out residual mean={resid.mean():.3f} std={resid.std():.3f} "
          f"(floor-aware: fit on genuine moves only, no zero pile-up)")

    model = make()
    model.fit(df[CAT + NUM], df[TARGET])
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    import joblib
    joblib.dump(model, OUT_DIR / "expected_thinktime_model.joblib")
    spec = {"regime": "bullet", "numeric": NUM, "categorical": CAT, "target": TARGET,
            "min_think_s": MIN_THINK_S, "n_fit": len(df), "heldout_r2": r2,
            "maia_spearman_rho": float(rho), "premove_rates": rates,
            "note": "Whole-second Lichess clocks; sub-second thinks are floored."}
    import json
    (OUT_DIR / "expected_thinktime_spec.json").write_text(json.dumps(spec, indent=2))
    print(f"\nsaved bullet mapping to {OUT_DIR}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=150000)
    ap.add_argument("--maia-cap", type=int, default=600000)
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()
    print(f"collecting up to {args.games} bullet games from {DUMP.name} ...", flush=True)
    df, rates, seen = collect(args.games)
    print(f"collected {len(df)} genuine moves from {seen} bullet games", flush=True)
    df = add_maia(df, args.maia_cap, args.workers)
    fit_and_report(df, rates, seen)


if __name__ == "__main__":
    main()
