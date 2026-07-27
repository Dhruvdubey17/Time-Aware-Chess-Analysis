# Time-Aware Chess Player-Strength Model

Estimates chess player strength from their games. Move quality is judged
conditionally on the time left on the clock, position complexity, and rating.
Skill means playing better than expected for the clock and complexity you
faced, not just low centipawn loss.

Data is the Lichess Open Database. The engine is Stockfish. Everything runs on
free tools and free compute.

## Setup

Requires Python 3.11+ and the Stockfish binary.

```bash
brew install stockfish          # provides /usr/local/bin/stockfish
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

Set `stockfish_path` in `config/default.yaml` if your binary lives elsewhere.

## Check the engine

```bash
python scripts/check_engine.py   # prints a bestmove for the start position
```

## Tests

```bash
pytest
```

## Layout

- `src/chess_strength/` importable pipeline logic, unit tested.
- `config/default.yaml` node budgets, thresholds, and paths in one place.
- `scripts/` thin CLI wrappers over `src/`.
- `tests/` pytest, with a small PGN fixture in `tests/fixtures/`.
- `data/` runtime downloads and outputs (gitignored).
- `notebooks/` for looking at results only, never pipeline logic.
