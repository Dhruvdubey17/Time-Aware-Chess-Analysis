"""chess.com fetch layer, all offline (the network is mocked).

Covers month listing, the game list with PGNs, resolving a game by link, the
error cases, that a fetched clock-carrying game flows into time-aware analysis,
and that the paste/upload path makes no network calls at all.
"""

from pathlib import Path

import pytest

from backend import analyze as analyze_mod
from backend import chesscom
from backend import intake as intake_mod
from backend.intake import parse_pgn

FIXTURES = Path(__file__).parent / "fixtures"

CLK_PGN = (
    '[Event "Live Chess"]\n[Site "Chess.com"]\n[White "tester"]\n[Black "rival"]\n'
    '[Result "1-0"]\n[TimeControl "300"]\n'
    '[Link "https://www.chess.com/game/live/9999001"]\n\n'
    "1. e4 {[%clk 0:05:00]} 1... e5 {[%clk 0:05:00]} 2. Nf3 {[%clk 0:04:57]} "
    "2... Nc6 {[%clk 0:04:58]} 1-0\n"
)

ARCHIVES = {"archives": [
    "https://api.chess.com/pub/player/tester/games/2026/06",
    "https://api.chess.com/pub/player/tester/games/2026/07",
]}

# The API returns a month oldest-first; include a bullet game, a blitz game, and
# a Chess960 game (which must be filtered out as non-standard).
MONTH_07 = {"games": [
    {"white": {"username": "tester", "rating": 1499, "result": "checkmated"},
     "black": {"username": "rook", "rating": 1520, "result": "win"},
     "url": "https://www.chess.com/game/live/9999002", "end_time": 1751400000,
     "time_class": "bullet", "time_control": "60", "rules": "chess", "pgn": "1. e4 1-0"},
    {"white": {"username": "tester", "rating": 1500, "result": "win"},
     "black": {"username": "rival", "rating": 1490, "result": "resigned"},
     "url": "https://www.chess.com/game/live/9999001", "end_time": 1751500000,
     "time_class": "blitz", "time_control": "300", "rules": "chess", "pgn": CLK_PGN},
    {"white": {"username": "tester", "result": "win"},
     "black": {"username": "z", "result": "resigned"},
     "url": "https://www.chess.com/game/live/9999003", "end_time": 1751300000,
     "time_class": "blitz", "time_control": "300", "rules": "chess960", "pgn": "..."},
]}


def _fake_get(url: str) -> dict:
    if url.endswith("/games/archives"):
        return ARCHIVES
    if url.endswith("/2026/07"):
        return MONTH_07
    if url.endswith("/2026/06"):
        return {"games": []}
    raise chesscom.ChessComError("not_found")


@pytest.fixture
def mock_net(monkeypatch):
    monkeypatch.setattr(chesscom, "_get", _fake_get)


def test_list_months_newest_first(mock_net):
    assert chesscom.list_months("tester") == ["2026/07", "2026/06"]


def test_fetch_month_lists_standard_games_newest_first(mock_net):
    res = chesscom.fetch_month("tester")  # defaults to newest month
    assert res["month"] == "2026/07"
    # Chess960 filtered out, so two standard games, newest (blitz) first.
    assert len(res["games"]) == 2
    g = res["games"][0]
    assert g["white"] == "tester" and g["result"] == "1-0"
    assert g["time_class"] == "blitz"
    assert len(g["date"]) == 10 and g["date"].count("-") == 2  # YYYY-MM-DD
    assert "%clk" in g["pgn"]


def test_empty_month_is_a_clear_error(mock_net):
    with pytest.raises(chesscom.ChessComError, match="no standard games|no games|No standard"):
        chesscom.fetch_month("tester", "2026/06")


def test_fetch_month_falls_back_to_last_month_with_games(monkeypatch):
    # The newest archive month is empty (a quiet current month). The default view
    # should fall back to the most recent month that has games, not error.
    archives = {"archives": [
        "https://api.chess.com/pub/player/tester/games/2026/07",
        "https://api.chess.com/pub/player/tester/games/2026/08",
    ]}

    def fake(url: str) -> dict:
        if url.endswith("/games/archives"):
            return archives
        if url.endswith("/2026/08"):
            return {"games": []}  # nothing played yet this month
        if url.endswith("/2026/07"):
            return MONTH_07
        raise chesscom.ChessComError("not_found")

    monkeypatch.setattr(chesscom, "_get", fake)
    res = chesscom.fetch_month("tester")  # no month -> fall back to last with games
    assert res["month"] == "2026/07"
    assert len(res["games"]) == 2  # Chess960 still filtered out


def test_unknown_user_is_a_clear_error(monkeypatch):
    monkeypatch.setattr(chesscom, "_get",
                        lambda url: (_ for _ in ()).throw(chesscom.ChessComError("not_found")))
    with pytest.raises(chesscom.ChessComError, match="No chess.com player"):
        chesscom.list_months("ghost")


def test_find_game_by_link(mock_net):
    g = chesscom.find_game("tester", "https://www.chess.com/game/live/9999001")
    assert g["black"] == "rival" and "%clk" in g["pgn"]


def test_find_game_missing_id_errors(mock_net):
    with pytest.raises(chesscom.ChessComError, match="game id"):
        chesscom.find_game("tester", "not-a-link")


def test_fetched_clock_game_flows_to_time_aware(mock_net):
    g = chesscom.fetch_month("tester")["games"][0]
    report = parse_pgn(g["pgn"])[0]
    assert report.accepted and report.has_clocks and not report.has_evals
    assert report.regime == "blitz" and report.time_aware_available


def test_paste_path_makes_no_network_calls(monkeypatch):
    # If anything in the offline path reaches for the network, this blows up.
    def boom(*a, **k):
        raise AssertionError("offline path made a network call")

    monkeypatch.setattr("urllib.request.urlopen", boom)
    for name in ("lichess_clk_eval.pgn", "chesscom_clk.pgn", "bare.pgn"):
        reports = parse_pgn((FIXTURES / name).read_text())
        assert reports and reports[0].accepted
    # The offline modules must not even import the fetch module into their namespace.
    assert not hasattr(intake_mod, "chesscom")
    assert not hasattr(analyze_mod, "chesscom")
