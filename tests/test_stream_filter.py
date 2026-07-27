"""Phase 2 tests: time-control classification, game splitting, and filtering.

Everything runs offline against the plain-PGN fixture; the acceptance gate is
that the fixture yields exactly its three usable games (both tags, wanted regime).
"""

from pathlib import Path

from chess_strength.stream_filter import (
    classify_time_control,
    filter_games,
    game_is_usable,
    iter_games,
    run_filter,
)

FIXTURE = Path(__file__).parent / "fixtures" / "mini.pgn"


def _lines():
    return FIXTURE.read_text(encoding="utf-8").splitlines(keepends=True)


def test_classify_time_control():
    assert classify_time_control("180+2") == "blitz"   # 260s
    assert classify_time_control("300+0") == "blitz"    # 300s
    assert classify_time_control("600+5") == "rapid"    # 800s
    assert classify_time_control("60+0") == "bullet"
    assert classify_time_control("15+0") == "ultrabullet"
    assert classify_time_control("1800+0") == "classical"
    assert classify_time_control("-") is None
    assert classify_time_control("junk") is None


def test_iter_games_splits_all_four():
    games = list(iter_games(_lines()))
    assert len(games) == 4
    assert all(g.startswith("[Event ") for g in games)


def test_game_is_usable_flags_only_tagged_games():
    games = list(iter_games(_lines()))
    # First three have both tags; the fourth (grace/heidi) has no %eval.
    assert game_is_usable(games[0], ["blitz", "rapid"]) == "blitz"
    assert game_is_usable(games[1], ["blitz", "rapid"]) == "blitz"
    assert game_is_usable(games[2], ["blitz", "rapid"]) == "rapid"
    assert game_is_usable(games[3], ["blitz", "rapid"]) is None


def test_regime_filter_excludes_unwanted_speed():
    games = list(iter_games(_lines()))
    # With only rapid wanted, the two blitz games drop out.
    assert game_is_usable(games[0], ["rapid"]) is None
    assert game_is_usable(games[2], ["rapid"]) == "rapid"


def test_filter_games_keeps_exactly_three():
    kept = list(filter_games(_lines(), ["blitz", "rapid"]))
    assert len(kept) == 3
    assert [regime for _, regime in kept] == ["blitz", "blitz", "rapid"]


def test_run_filter_writes_shard_with_kept_games(tmp_path):
    stats = run_filter(FIXTURE, tmp_path, ["blitz", "rapid"])
    assert stats.games_seen == 4
    assert stats.games_kept == 3
    assert stats.shards_written == 1
    assert abs(stats.keep_rate - 0.75) < 1e-9

    shard = tmp_path / "shard_000.pgn"
    text = shard.read_text(encoding="utf-8")
    # The three kept games are present, the eval-less one is not.
    assert "alice" in text and "carol" in text and "erin" in text
    assert "grace" not in text


def test_run_filter_rolls_shards_at_cap(tmp_path):
    # A tiny cap forces one game per shard, proving the roll-over works.
    stats = run_filter(FIXTURE, tmp_path, ["blitz", "rapid"], cap_bytes=100)
    assert stats.games_kept == 3
    assert stats.shards_written == 3
    assert (tmp_path / "shard_002.pgn").exists()
