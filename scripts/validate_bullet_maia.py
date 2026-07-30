"""Phase G, GO/NO-GO criterion (1): does Maia dispersion predict bullet errors?

Maia-2 was trained excluding bullet, so before we trust its difficulty signal
there we check that positions it calls hard actually produce more mistakes, at
the same rating band. We sample real bullet moves, score each with local
Stockfish (Win% loss) and with Maia dispersion, then ask: within a band, do
higher-dispersion positions lose more Win%, and blunder more often?

Criterion (2), dispersion vs think-time, is already confirmed by the model fit
(rho about 0.14, p about 0 on 700k genuine moves). This script covers (1).

    python scripts/validate_bullet_maia.py --games 4000 --sample 15000
"""

from __future__ import annotations

import argparse
import io
import sys
from pathlib import Path

import chess
import chess.pgn
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend import engine, maia_client
from backend.analyze import DEFAULT_ELO, _maia_model_regime, band_label
from backend.cache import FenCache
from chess_strength import maia
from chess_strength.classify import win_pct_mate
from chess_strength.stream_filter import _open_lines, classify_time_control, iter_games

DUMP = ROOT / "data" / "raw" / "lichess_db_standard_rated_2017-04.pgn.zst"
MATE_CP = 10000


def collect(max_games: int) -> pd.DataFrame:
    rows = []
    kept = 0
    for text in iter_games(_open_lines(DUMP)):
        if "%clk" not in text:
            continue
        game = chess.pgn.read_game(io.StringIO(text))
        if game is None:
            continue
        if classify_time_control(game.headers.get("TimeControl", "")) != "bullet":
            continue
        welo = game.headers.get("WhiteElo", "")
        belo = game.headers.get("BlackElo", "")
        board = game.board()
        for nd in game.mainline():
            side = "white" if board.turn == chess.WHITE else "black"
            fen_before = board.fen()
            board.push(nd.move)
            fen_after = board.fen()
            elo = welo if side == "white" else belo
            rows.append((fen_before, fen_after, side,
                         band_label(int(elo) if elo.isdigit() else DEFAULT_ELO)))
        kept += 1
        if kept >= max_games:
            break
    return pd.DataFrame(rows, columns=["fen_before", "fen_after", "side", "band"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=4000)
    ap.add_argument("--sample", type=int, default=15000)
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()

    df = collect(args.games)
    if len(df) > args.sample:
        df = df.sample(args.sample, random_state=0).reset_index(drop=True)
    print(f"sampled {len(df)} bullet moves from up to {args.games} games", flush=True)

    import os
    sf_path = os.environ["STOCKFISH_PATH"]
    cache = FenCache(ROOT / "data" / "processed" / "app_cache.sqlite")
    try:
        fens = list(dict.fromkeys(df["fen_before"].tolist() + df["fen_after"].tolist()))
        print(f"stockfish over {len(fens)} unique FENs ...", flush=True)
        feats = engine.analyze_fens(fens, cache, sf_path, args.workers)

        def ev(fen):
            return feats[fen]["eval_white"]

        # Win% loss from the mover's point of view.
        wpl = []
        for r in df.itertuples():
            white = r.side == "white"
            bw = ev(r.fen_before) if white else -ev(r.fen_before)
            aw = ev(r.fen_after) if white else -ev(r.fen_after)
            wpl.append(max(0.0, win_pct_mate(bw) - win_pct_mate(aw)))
        df["wpl"] = wpl

        regime = _maia_model_regime("bullet")
        be = {b: maia.band_to_elo(b) for b in df["band"].unique()}
        tasks = [(r.fen_before, regime, be[r.band]) for r in df.itertuples()]
        print(f"maia over {len(set(tasks))} unique positions ...", flush=True)
        disp = maia_client.dispersion(list(dict.fromkeys(tasks)), cache, workers=args.workers)
        df["dispersion"] = [disp.get((r.fen_before, regime, be[r.band])) for r in df.itertuples()]
    finally:
        cache.close()

    df = df.dropna(subset=["dispersion"])
    from scipy import stats
    rho, p = stats.spearmanr(df["dispersion"], df["wpl"])
    print("\n" + "=" * 66)
    print(f"overall: Spearman(dispersion, Win%% loss) rho={rho:.3f} p={p:.1e}  n={len(df)}")
    print("\nwithin band (rho should be positive: harder -> more error):")
    for band, g in df.groupby("band"):
        if len(g) < 200:
            continue
        r, pp = stats.spearmanr(g["dispersion"], g["wpl"])
        print(f"  {band:14} n={len(g):>5}  rho={r:+.3f}  p={pp:.1e}")

    # Blunder rate by dispersion quartile.
    df["q"] = pd.qcut(df["dispersion"], 4, labels=["Q1 easy", "Q2", "Q3", "Q4 hard"])
    print("\nblunder rate (Win%% loss > 20) by dispersion quartile:")
    for q, g in df.groupby("q", observed=True):
        print(f"  {q:9} mean_wpl={g['wpl'].mean():5.2f}  blunder%={100*(g['wpl']>20).mean():4.1f}")


if __name__ == "__main__":
    main()
