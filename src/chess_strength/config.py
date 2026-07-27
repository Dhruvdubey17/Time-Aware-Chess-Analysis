"""Load the project config from config/default.yaml.

Everything downstream calls load_config() instead of hardcoding paths or
thresholds, so there is a single knob to turn.
"""

from __future__ import annotations

from pathlib import Path

import yaml

# Repo root is two levels up from this file (src/chess_strength/config.py).
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = REPO_ROOT / "config" / "default.yaml"


def load_config(path: str | Path | None = None) -> dict:
    """Read the YAML config into a plain dict.

    Relative paths inside the `paths:` block are resolved against the repo
    root so callers get absolute paths regardless of their working directory.
    """
    cfg_path = Path(path) if path else DEFAULT_CONFIG_PATH
    if not cfg_path.exists():
        raise FileNotFoundError(f"config file not found: {cfg_path}")

    with cfg_path.open(encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh) or {}

    for key, rel in (cfg.get("paths") or {}).items():
        cfg["paths"][key] = str((REPO_ROOT / rel).resolve())

    return cfg
