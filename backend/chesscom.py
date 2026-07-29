"""Fetch a game from the chess.com public API.

This is the app's ONLY network touchpoint, and it is walled off from the offline
core on purpose: the intake and analysis code never import this module, so a user
who only pastes or uploads a PGN makes zero network calls. We reach out only to
fetch a game the user explicitly asks for.

Why it exists: chess.com's manual "download PGN" strips the move clocks, but the
public API carries per-move [%clk] on every move. So to review a chess.com game
with the time-aware signal, we pull it from the API instead of asking the user to
export it by hand.

Endpoints (all public, no auth):
  archives:  /pub/player/{user}/games/archives           -> list of month URLs
  one month: /pub/player/{user}/games/{YYYY}/{MM}         -> that month's games

There is no clean public single-game endpoint, so a game link or id is resolved
by scanning the player's month archives for a matching game URL.
"""

from __future__ import annotations

import json
import re
import urllib.error
import urllib.request
from datetime import UTC, datetime

API = "https://api.chess.com/pub"
# chess.com rejects requests without a descriptive User-Agent, so send one.
USER_AGENT = "TimeAwareChessReview/1.0 (local, offline chess review app)"
TIMEOUT_S = 20


class ChessComError(Exception):
    """A fetch problem we can explain to the user in plain language."""


def _get(url: str) -> dict:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(req, timeout=TIMEOUT_S) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        if e.code == 404:
            raise ChessComError("not_found") from e
        if e.code == 429:
            raise ChessComError(
                "chess.com is rate limiting us. Please wait a minute and try again."
            ) from e
        raise ChessComError(f"chess.com request failed (HTTP {e.code}).") from e
    except urllib.error.URLError as e:
        raise ChessComError(
            "Could not reach chess.com. Check your internet connection."
        ) from e


def normalize_username(username: str) -> str:
    return username.strip().lstrip("@").lower()


def list_months(username: str) -> list[str]:
    """Available archive months for a player, newest first, as 'YYYY/MM'."""
    user = normalize_username(username)
    try:
        data = _get(f"{API}/player/{user}/games/archives")
    except ChessComError as e:
        if str(e) == "not_found":
            raise ChessComError(f"No chess.com player called '{username}'.") from e
        raise
    months = ["/".join(u.rsplit("/", 2)[-2:]) for u in data.get("archives", [])]
    return list(reversed(months))


def _result(game: dict) -> str:
    if game.get("white", {}).get("result") == "win":
        return "1-0"
    if game.get("black", {}).get("result") == "win":
        return "0-1"
    return "1/2-1/2"


def _stub(game: dict) -> dict:
    """The fields the picker shows, plus the PGN that feeds the normal pipeline."""
    end = game.get("end_time")
    date = (
        datetime.fromtimestamp(end, tz=UTC).strftime("%Y-%m-%d")
        if end else ""
    )
    w, b = game.get("white", {}), game.get("black", {})
    return {
        "white": w.get("username", "?"),
        "white_elo": w.get("rating"),
        "black": b.get("username", "?"),
        "black_elo": b.get("rating"),
        "result": _result(game),
        "time_class": game.get("time_class", ""),
        "time_control": game.get("time_control", ""),
        "date": date,
        "url": game.get("url", ""),
        "rules": game.get("rules", "chess"),
        "pgn": game.get("pgn", ""),
    }


def fetch_month(username: str, month: str | None = None) -> dict:
    """A month's standard-chess games, newest first, each with its PGN.

    `month` is 'YYYY/MM'; when omitted we use the most recent month with games.
    """
    user = normalize_username(username)
    months = list_months(user)
    if not months:
        raise ChessComError(f"'{username}' has no games on chess.com yet.")
    month = month or months[0]
    if month not in months:
        raise ChessComError(f"'{username}' has no games for {month}.")

    data = _get(f"{API}/player/{user}/games/{month}")
    games = [_stub(g) for g in data.get("games", []) if g.get("rules", "chess") == "chess"]
    games.reverse()  # API returns oldest first; show newest first
    if not games:
        raise ChessComError(f"No standard games found for '{username}' in {month}.")
    return {"month": month, "months": months, "games": games}


def parse_game_id(link_or_id: str) -> str | None:
    """The numeric id from a chess.com game link or a bare id."""
    m = re.search(r"(\d{6,})", link_or_id)
    return m.group(1) if m else None


def find_game(username: str, link_or_id: str) -> dict:
    """Locate one game by link/id, scanning the player's months newest first.

    There is no public single-game endpoint, so this can make several requests
    for an old game. Requests stay serial and polite.
    """
    gid = parse_game_id(link_or_id)
    if not gid:
        raise ChessComError("Could not read a game id from that link.")
    user = normalize_username(username)
    for month in list_months(user):
        data = _get(f"{API}/player/{user}/games/{month}")
        for g in data.get("games", []):
            if g.get("url", "").rstrip("/").rsplit("/", 1)[-1] == gid:
                return _stub(g)
    raise ChessComError(f"Could not find game {gid} in '{username}'s games.")
