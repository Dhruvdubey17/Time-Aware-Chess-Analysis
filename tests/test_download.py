"""Phase 2 download tests: URL building only. The actual fetch needs network."""

from chess_strength.download import month_url


def test_month_url():
    assert month_url("2017-04") == (
        "https://database.lichess.org/standard/"
        "lichess_db_standard_rated_2017-04.pgn.zst"
    )
