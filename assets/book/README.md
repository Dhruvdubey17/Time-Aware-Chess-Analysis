# Bundled opening book

`openings.bin` is a small Polyglot opening book used to decide whether a move is
established theory. It holds membership only (which moves are theory from which
position), not master-game frequency stats, so it stays tiny.

## Source and license

Built from the Lichess opening classification, `lichess-org/chess-openings`,
which is released under CC0 1.0 (public domain). CC0 places no restriction on
redistribution, so bundling the derived book here is fine.

- Source data: https://github.com/lichess-org/chess-openings (CC0 1.0)
- License: Creative Commons Zero v1.0 Universal (public domain dedication)

## Rebuilding

Run `python scripts/build_opening_book.py` from the repo root. It downloads the
source TSVs (cached under `data/raw/openings`), replays every named opening line,
and writes this file. The script verifies the result before finishing.
