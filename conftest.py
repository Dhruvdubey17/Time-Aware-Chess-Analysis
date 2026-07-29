"""Put the repo root on sys.path so tests can import the top-level `backend`
package alongside the installed `chess_strength` package."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
