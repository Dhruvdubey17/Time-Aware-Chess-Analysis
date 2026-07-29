"""Offline checks for the move classifier core."""

from chess_strength import classify as cl
from chess_strength.config import load_config


def C():
    return cl.ClassifyConfig()


def test_win_pct_mate_extremes_and_normal():
    assert cl.win_pct_mate(10000) == 100.0
    assert cl.win_pct_mate(-9999) == 0.0
    assert cl.win_pct_mate(0) == 50.0
    # Mate to mate (different distances) is no loss at all.
    assert cl.win_loss(9998, 9995) == 0.0
    # A mate that becomes a merely huge eval is not a false blunder (small loss).
    assert cl.win_loss(10000, 1500) < cl.ClassifyConfig().excellent_max
    # A real swing from equal to losing is a big Win% loss.
    assert cl.win_loss(0, -300) > 15.0
    # Improving the position gives no negative credit.
    assert cl.win_loss(-100, 200) == 0.0


def test_quality_tiers():
    c = C()
    assert cl.quality_label(0.0, c) == "Best"
    assert cl.quality_label(3.0, c) == "Excellent"
    assert cl.quality_label(7.0, c) == "Good"
    assert cl.quality_label(15.0, c) == "Inaccuracy"
    assert cl.quality_label(25.0, c) == "Mistake"
    assert cl.quality_label(40.0, c) == "Blunder"


def test_baseline_book_and_specials():
    c = C()
    assert cl.baseline_label(0.0, True, False, False, c) == "Book"
    assert cl.baseline_label(1.0, False, True, False, c) == "Brilliant"
    assert cl.baseline_label(1.0, False, False, True, c) == "Great"
    # Book wins even if it would be a brilliant looking move.
    assert cl.baseline_label(1.0, True, True, False, c) == "Book"


def test_brilliant_and_great_rules():
    c = C()
    # All four hold: near-best, not already winning, real sacrifice, hard position.
    assert cl.is_brilliant(1.0, 320, 70.0, 55.0, 2.4, c) is True
    # Already completely winning before -> not brilliant (just winning).
    assert cl.is_brilliant(1.0, 320, 95.0, 90.0, 2.4, c) is False
    # No material given up -> not a sacrifice.
    assert cl.is_brilliant(1.0, 0, 60.0, 55.0, 2.4, c) is False
    # Easy position (low dispersion) -> not brilliant even with a sound sacrifice.
    assert cl.is_brilliant(1.0, 320, 70.0, 55.0, 1.0, c) is False
    # Difficulty unknown (no Maia reading) -> never brilliant.
    assert cl.is_brilliant(1.0, 320, 70.0, 55.0, None, c) is False
    # Not near-best -> not brilliant.
    assert cl.is_brilliant(6.0, 320, 70.0, 55.0, 2.4, c) is False
    # Clear only-move found -> great; gap below the tuned bar -> not great.
    assert cl.is_great(1.0, 600, c) is True
    assert cl.is_great(1.0, 300, c) is False
    # A not-near-best move is never great even with a huge gap.
    assert cl.is_great(6.0, 900, c) is False


def test_gate_and_pressure():
    c = C()
    assert cl.complexity_gate(0.5, c) == 0.0          # below lo, easy
    assert cl.complexity_gate(3.0, c) == 1.0          # above hi, hard
    assert 0.0 < cl.complexity_gate(1.55, c) < 1.0    # midway
    # A full clock earns no pressure no matter how fast (the 887s-castle fix).
    assert cl.pressure_modifier(-30.0, 300.0, c) == 0.0
    assert cl.pressure_modifier(-30.0, 120.0, c) == 0.0
    # On a near-flag clock, a fast move gets strong pressure.
    assert cl.pressure_modifier(-15.0, 2.0, c) >= 0.95
    # On a low clock, a slow move still gets the base floor, not full.
    base = cl.pressure_modifier(+30.0, 0.0, c)
    assert abs(base - c.pressure_base) < 1e-9
    # Speed sharpens pressure within the pressured regime.
    assert cl.pressure_modifier(-15.0, 2.0, c) > cl.pressure_modifier(+5.0, 2.0, c)


def test_skill_preconditions_and_multiplicative_zero():
    c = C()
    # A bad move is never eligible, no matter how hard or fast.
    assert cl.skill_of_move(25.0, 1.0, 1.0, c) == 0.0
    # Zero complexity -> zero skill even when played instantly (the key guard).
    assert cl.skill_of_move(1.0, 0.0, 1.0, c) == 0.0
    # Good move, hard, under pressure -> high skill.
    assert cl.skill_of_move(1.0, 1.0, 1.0, c) == 1.0


def test_option_b_guard_no_upgrade_on_easy_positions():
    c = C()
    # Easy position (gate 0) means skill 0, so labels stay put. Gate never fires easy.
    for base in ["Good", "Excellent", "Best"]:
        assert cl.option_b_label(base, 0.0, False, 1.0, c) == base
    # A near-best move, hard and fast, reaches Great; a Brilliant candidate reaches Brilliant.
    assert cl.option_b_label("Best", 0.40, False, 1.0, c) == "Great"
    assert cl.option_b_label("Excellent", 0.60, True, 1.0, c) == "Brilliant"
    # Never downgrades a strong baseline, never touches Book or bad moves.
    assert cl.option_b_label("Great", 0.0, False, 1.0, c) == "Great"
    assert cl.option_b_label("Book", 0.9, True, 1.0, c) == "Book"
    assert cl.option_b_label("Blunder", 0.9, True, 1.0, c) == "Blunder"


def test_no_two_tier_jump_to_great():
    c = C()
    # A merely good move (not near-best) can never be Great, even with max skill.
    # It caps at Excellent.
    assert cl.option_b_label("Good", 0.99, False, 8.0, c) == "Excellent"
    assert cl.option_b_label("Good", 0.20, False, 8.0, c) == "Excellent"
    # Only a near-best move gets there (Excellent tier, wpl within near-best).
    assert cl.option_b_label("Excellent", 0.40, False, 2.0, c) == "Great"


def test_option_a_threshold():
    c = C()
    # Good move, hard position, real pressure -> one tier up.
    assert cl.option_a_label("Good", 2.0, 2.5, 0.5, c) == "Excellent"
    # Same move in an easy position -> no upgrade.
    assert cl.option_a_label("Good", 2.0, 1.0, 0.5, c) == "Good"


def test_from_config_reads_yaml():
    c = cl.ClassifyConfig.from_config(load_config())
    assert c.good_max == 10.0 and c.disp_hi == 2.20 and c.cut_great == 0.35
