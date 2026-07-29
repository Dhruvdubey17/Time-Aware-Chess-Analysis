"""Maia dispersion inference for STEP B. Runs in the arm64 maia venv.

Reads the scoring tasks written by maia_validation.py sample (unique position x
regime x rating-band Elo), scores each with the Maia model matching its regime
at that Elo, turns the move distribution into a dispersion reading, and writes
the result to data/processed/maia_val/dispersion/ in atomic batches.

Process pool, single-threaded torch per worker, one model type per regime pass
so peak memory is one model per worker, not both. Resumable and flushed per
batch, with a hard 90-minute ceiling. Nothing here touches the existing cache or
dataset, it only reads the task list and writes dispersion.

Run with the maia venv:
    .venv_maia/bin/python scripts/maia_infer.py
"""

from __future__ import annotations

import multiprocessing as mp
import os
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd

# The dispersion definition lives in src and is pure math, so it imports here
# even though this venv has numpy<2 and no full pipeline install.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from chess_strength.maia import move_dispersion

# single-threaded torch each, per the standing rule. The service overrides this
# to 1 for a single game, where one model load dominates and a pool just reloads
# the weights N times for no gain.
WORKERS = int(os.environ.get("MAIA_WORKERS", "6"))
BATCH = 250                 # tasks per work unit and per atomic flush
CEILING_MIN = 90            # hard wall-clock safety stop
SAVE_ROOT = "data/processed/maia_val/weights"
OUT_DIR = Path("data/processed/maia_val/dispersion")
TASKS = Path("data/processed/maia_val/val_tasks.parquet")


def log(msg: str) -> None:
    print(msg, flush=True)


def _ensure_weights(regime: str) -> None:
    """Download this regime's weights once, in a child, before the pool starts.
    Six workers all triggering the same download would race and corrupt it."""
    if (Path(SAVE_ROOT) / f"{regime}_model.pt").exists():
        return
    log(f"pre-downloading {regime} weights ...")
    code = (f"from maia2 import model; "
            f"model.from_pretrained(type='{regime}', device='cpu', save_root='{SAVE_ROOT}')")
    subprocess.run([sys.executable, "-c", code], check=True)


# Worker globals, set once per process in the initializer.
_MODEL = None
_PREPARED = None


def _init_worker(regime: str) -> None:
    global _MODEL, _PREPARED
    import warnings

    warnings.filterwarnings("ignore")
    import torch
    from maia2 import inference, model

    torch.set_num_threads(1)
    _MODEL = model.from_pretrained(type=regime, device="cpu", save_root=SAVE_ROOT)
    _PREPARED = inference.prepare()


def _work(batch: list) -> list:
    from maia2 import inference

    out = []
    for fen, elo in batch:
        res = inference.inference_each(_MODEL, _PREPARED, fen, int(elo), int(elo))
        move_probs = res[0] if isinstance(res, tuple) else res
        d = move_dispersion(move_probs)
        d["fen"] = fen
        d["band_elo"] = int(elo)
        out.append(d)
    return out


def _flush(rows: list, regime: str, idx: int) -> None:
    df = pd.DataFrame(rows)
    df["regime"] = regime
    tmp = OUT_DIR / f".batch_{idx:05d}.tmp.parquet"
    final = OUT_DIR / f"batch_{idx:05d}.parquet"
    df.to_parquet(tmp, index=False)
    os.replace(tmp, final)


def _done_keys() -> set:
    done = set()
    for b in sorted(OUT_DIR.glob("batch_*.parquet")):
        d = pd.read_parquet(b, columns=["fen", "regime", "band_elo"])
        done |= set(zip(d["fen"], d["regime"], d["band_elo"]))
    return done


def main() -> None:
    global OUT_DIR, TASKS
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", type=Path, default=TASKS)
    ap.add_argument("--out", type=Path, default=OUT_DIR)
    args = ap.parse_args()
    TASKS, OUT_DIR = args.tasks, args.out

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tasks = pd.read_parquet(TASKS)
    done = _done_keys()
    if done:
        key = list(zip(tasks["fen"], tasks["regime"], tasks["band_elo"]))
        tasks = tasks[[k not in done for k in key]]
    log(f"tasks to score: {len(tasks):,} (already done {len(done):,})")

    start = time.time()
    ceiling_s = CEILING_MIN * 60
    idx = len(list(OUT_DIR.glob("batch_*.parquet")))
    total = 0
    hit_ceiling = False

    for regime in sorted(tasks["regime"].unique()):
        if hit_ceiling:
            break
        sub = tasks[tasks["regime"] == regime]
        pairs = list(zip(sub["fen"], sub["band_elo"]))
        batches = [pairs[i:i + BATCH] for i in range(0, len(pairs), BATCH)]
        log(f"\n== {regime}: {len(pairs):,} tasks in {len(batches)} batches, {WORKERS} workers ==")
        _ensure_weights(regime)

        pool = mp.Pool(WORKERS, initializer=_init_worker, initargs=(regime,))
        try:
            for rows in pool.imap_unordered(_work, batches):
                _flush(rows, regime, idx)
                idx += 1
                total += len(rows)
                elapsed = time.time() - start
                rate = total / elapsed * 60 if elapsed > 0 else 0.0
                if idx % 10 == 0 or total >= len(pairs):
                    log(f"  {regime}: scored {total:,}  {elapsed / 60:.1f} min  {rate:.0f}/min")
                if elapsed > ceiling_s:
                    hit_ceiling = True
                    log(f"  CEILING {CEILING_MIN} min hit, stopping.")
                    break
        finally:
            pool.terminate()
            pool.join()

    elapsed = time.time() - start
    scored = len(_done_keys())
    total_tasks = len(pd.read_parquet(TASKS))
    log(f"\nscored {total:,} this run in {elapsed / 60:.1f} min "
        f"({total / elapsed * 60 if elapsed > 0 else 0:.0f}/min). "
        f"dispersion now holds {scored:,}/{total_tasks:,} tasks.")
    if hit_ceiling or scored < total_tasks:
        log(f"INCOMPLETE ({scored:,}/{total_tasks:,}). Do not analyze a partial sample. "
            f"Re-run to resume.")


if __name__ == "__main__":
    main()
