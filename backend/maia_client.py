"""Maia human-move dispersion, run in its own environment.

Maia needs torch and numpy<2, which fight the main service environment
(numpy 2). So Maia lives in the separate `.venv_maia` interpreter and we call it
as a subprocess, reusing scripts/maia_infer.py unchanged. We hand it a task list
(position, regime, rating Elo), it writes dispersion batches, we read them back.

Dispersion is how much humans at that rating spread their move choice: high means
the position is genuinely hard for a person. That is the complexity signal the
classifier gates on. Everything is local; the weights are already downloaded, so
no network is touched.
"""

from __future__ import annotations

import glob
import os
import platform
import subprocess
import tempfile
from collections.abc import Callable
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
# venvs put the interpreter under Scripts on Windows, bin everywhere else.
MAIA_PYTHON = (REPO_ROOT / ".venv_maia" / "Scripts" / "python.exe"
               if os.name == "nt" else REPO_ROOT / ".venv_maia" / "bin" / "python")
MAIA_SCRIPT = REPO_ROOT / "scripts" / "maia_infer.py"

Task = tuple[str, str, int]  # (fen, regime, band_elo)


def _launch_prefix() -> list[str]:
    """The Maia venv is native arm64, but the main venv here runs x86_64 under
    Rosetta. A child inherits the parent's arch, so without this the arm64 numpy
    in .venv_maia fails to load. `arch -arm64` forces the child to run native.

    ponytail: targets Apple Silicon (the dev and primary target machine). On a
    native-arm64 parent this is a no-op; the Phase 5 installer builds
    arch-matched envs, so Intel-Mac handling can wait until it is needed.
    """
    return ["arch", "-arm64"] if platform.system() == "Darwin" else []


def available() -> bool:
    """True if the Maia environment and script are present to call."""
    return MAIA_PYTHON.exists() and MAIA_SCRIPT.exists()


def dispersion(
    tasks: list[Task],
    cache,
    workers: int = 1,
    progress: Callable[[int, int], None] | None = None,
) -> dict[Task, float]:
    """Maia entropy for each (fen, regime, band_elo), reading and writing cache.

    Only uncached tasks go to the subprocess. One game is a light job, so the
    default is a single worker (one model load, no reload-per-worker waste).
    """
    tasks = list(dict.fromkeys(tasks))
    total = len(tasks)
    out: dict[Task, float] = {}
    missing: list[Task] = []
    for key in tasks:
        hit = cache.get_maia(*key)
        if hit is None:
            missing.append(key)
        else:
            out[key] = hit

    if progress:
        progress(total - len(missing), total)

    if missing:
        if not available():
            raise RuntimeError(
                "The Maia environment (.venv_maia) is missing, so the time-aware "
                "difficulty signal cannot be computed. Run the install script."
            )
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            tasks_path = tmp / "tasks.parquet"
            out_dir = tmp / "dispersion"
            pd.DataFrame(missing, columns=["fen", "regime", "band_elo"]).to_parquet(
                tasks_path, index=False
            )
            env = {**os.environ, "MAIA_WORKERS": str(workers)}
            proc = subprocess.run(
                [*_launch_prefix(), str(MAIA_PYTHON), str(MAIA_SCRIPT),
                 "--tasks", str(tasks_path), "--out", str(out_dir)],
                cwd=REPO_ROOT, env=env, capture_output=True, text=True, check=False,
            )
            if proc.returncode != 0:
                raise RuntimeError(f"Maia inference failed:\n{proc.stdout}\n{proc.stderr}")

            batches = sorted(glob.glob(str(out_dir / "batch_*.parquet")))
            if batches:
                disp = pd.concat((pd.read_parquet(b) for b in batches), ignore_index=True)
                for r in disp.itertuples():
                    key = (r.fen, r.regime, int(r.band_elo))
                    cache.put_maia(*key, float(r.maia_entropy))
                    out[key] = float(r.maia_entropy)

    if progress:
        progress(total, total)
    return out
