"""Phase 2: fetch one Lichess monthly standard dump to data/raw/.

We download the compressed `.pgn.zst` and never unpack it to disk; the stream
filter reads it compressed. The download resumes: if a partial file is already
there we ask the server for the rest with a Range request instead of starting
over, which matters because these files are large.
"""

from __future__ import annotations

import urllib.request
from pathlib import Path

LICHESS_BASE = "https://database.lichess.org/standard"


def month_url(month: str) -> str:
    """Build the download URL for a month like '2017-04'."""
    return f"{LICHESS_BASE}/lichess_db_standard_rated_{month}.pgn.zst"


def download_month(
    month: str, dest_dir: str | Path, chunk: int = 1 << 20
) -> Path:
    """Download a monthly dump, resuming a partial file if one exists.

    Returns the path to the (complete) `.pgn.zst`. Safe to re-run: a finished
    file is left alone, a partial one is continued from where it stopped.
    """
    dest = Path(dest_dir) / f"lichess_db_standard_rated_{month}.pgn.zst"
    dest.parent.mkdir(parents=True, exist_ok=True)
    have = dest.stat().st_size if dest.exists() else 0

    req = urllib.request.Request(month_url(month))
    if have:
        req.add_header("Range", f"bytes={have}-")

    try:
        resp = urllib.request.urlopen(req)
    except urllib.error.HTTPError as exc:
        # 416 means our partial file is already the whole thing.
        if exc.code == 416 and have:
            return dest
        raise

    # If we asked to resume but the server ignored Range (200, not 206), start
    # over so we do not append onto a stale prefix.
    append = have and resp.status == 206
    mode = "ab" if append else "wb"
    with resp, dest.open(mode) as out:
        while True:
            block = resp.read(chunk)
            if not block:
                break
            out.write(block)
    return dest
