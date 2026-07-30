"""Time-aware move classifier: the headless core.

Two label sets on ONE scale so they are directly comparable:

- Baseline (time-blind): tiers assigned by Win% loss, mirroring chess.com's label
  NAMES and tier structure. This is an honest reconstruction on the Lichess Win%
  scale, NOT chess.com's proprietary expected-points math. Every threshold here
  is our choice and is a config value. The special labels (Brilliant, Great) are
  rules reconstructed from public descriptions, not chess.com's real logic.

- Time-aware: takes a baseline label and may UPGRADE it when a good move was hard
  for a human (high Maia move dispersion) and was played under genuine time
  pressure (faster than the position's difficulty would ordinarily demand, or on
  a low clock). Two upgrade rules: Option B (primary, a continuous score where
  complexity enters multiplicatively) and Option A (a transparent threshold rule
  as a sanity check).

The three components, all read on the Win% scale where it makes sense:
- Quality: Win% loss of the played move (fixed scale, no free weights).
- Complexity: Maia human-move dispersion at the player's band, turned into a
  [0,1] gate. Validated against human think-time, so it stands in for "how hard
  for a human." It gates MULTIPLICATIVELY: zero complexity means zero upgrade.
- Time pressure: the think-time residual from the STEP 1 fitted mapping. Genuine
  pressure means the player moved faster than expected for the context, or the
  absolute clock was low.
"""

from __future__ import annotations

from dataclasses import dataclass, fields

from .winprob import win_pct

# parse_moves stores mate as +-10000 offset by distance, so anything past this
# magnitude is a mate, not a normal eval.
MATE_FLOOR_CP = 9000

# Quality ladder, worst to best. Book sits outside it (an opening-theory tag).
QUALITY_LADDER = ["Blunder", "Mistake", "Inaccuracy", "Good", "Excellent", "Best", "Great", "Brilliant"]
BOOK = "Book"


@dataclass
class ClassifyConfig:
    """Every threshold in one place, all overridable from config/default.yaml.

    Win% loss (wpl) values are in win-percentage points. Dispersion values are
    in nats (raw Maia move-distribution entropy). Times are in seconds.
    """
    # Baseline tier cut points on Win% loss (upper edge of each tier).
    best_max: float = 2.0
    excellent_max: float = 5.0
    good_max: float = 10.0
    inaccuracy_max: float = 20.0
    mistake_max: float = 30.0  # above this is a Blunder

    # A move is "good" (eligible for a time-aware upgrade) at or below this wpl.
    eligibility_max: float = 10.0

    # Near-best precondition. A move below this quality can NEVER be labeled Great,
    # in the baseline or a time-aware upgrade (no two-tier jumps). Used everywhere.
    near_best_max: float = 3.0

    # Brilliant (baseline rule): the rare, special label. All four conditions must
    # hold (see is_brilliant). Near-best reuses near_best_max, so no separate wpl
    # knob here.
    brilliant_sac_cp: float = 200.0         # material given up, mover POV, exchange or more
    brilliant_winning_ceiling: float = 85.0  # not already winning (Win% before the move)
    brilliant_not_losing_floor: float = 45.0  # sound: not losing after (Win% after)
    brilliant_disp_min: float = 2.00        # hard for a human: high Maia dispersion (nats)

    # Great (baseline rule): the only good move in the position. Kept strictly
    # time-blind and complexity-blind (this is the control). Gap tuned so the
    # baseline Great rate lands under ~2%.
    great_gap_cp: float = 550.0             # engine best minus second best

    # Complexity gate: Maia dispersion mapped to [0,1]. Below lo -> 0 (easy, humans
    # agree), above hi -> 1 (hard, humans split). Calibrated on the validation set
    # (roughly the 20th and 90th percentiles of dispersion).
    disp_lo: float = 0.90
    disp_hi: float = 2.20

    # Time pressure. The clock must be a real constraint: pressure fades to zero as
    # the absolute clock rises past the danger zone, so a full clock earns nothing
    # however fast the move. Within a pressured regime, moving faster than expected
    # sharpens the credit; pressure_base is the floor credit for simply being low.
    clock_danger_s: float = 60.0
    pressure_time_ref_s: float = 10.0
    pressure_base: float = 0.5

    # Bullet time pressure. Absolute-second thresholds mean nothing when the whole
    # game is under a minute, so the clock factor is normalized to the base time:
    # zero with most of the clock left, ramping to one near the flag once below
    # this fraction of the base. The fast term uses a bullet-scale reference.
    bullet_clock_danger_frac: float = 0.25
    bullet_pressure_time_ref_s: float = 3.0
    # Mouse-slip guardrail: a big Win% drop played at floor time (a premove or a
    # sub-second move) is likely a slip, not a considered move, so it can never
    # count as a find under pressure.
    misclick_wpl_min: float = 20.0

    # Option B skill-of-move cutoffs (skill is in [0,1]).
    cut_excellent: float = 0.15
    cut_great: float = 0.35
    cut_brilliant: float = 0.55

    # Option A transparent thresholds.
    a_wpl_max: float = 5.0
    a_disp_min: float = 2.00
    a_pressure_min: float = 0.30

    @classmethod
    def from_config(cls, cfg: dict) -> ClassifyConfig:
        block = (cfg or {}).get("classify", {}) or {}
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in block.items() if k in known})


# ---------------------------------------------------------------------------
# Quality on the Win% scale
# ---------------------------------------------------------------------------

def win_pct_mate(cp: float) -> float:
    """Win% for a mover-POV eval, with mate mapped to the extreme of the scale.

    A forced mate reads as 100 (or 0 if the mover is getting mated) instead of a
    clipped 99.4, so a mate-to-huge-eval move is not scored as a false blunder.
    """
    if cp >= MATE_FLOOR_CP:
        return 100.0
    if cp <= -MATE_FLOOR_CP:
        return 0.0
    return win_pct(cp)


def win_loss(before_cp: float, after_cp: float) -> float:
    """Win% thrown away by a move, mover POV, clamped at 0 (no credit for gains)."""
    return max(0.0, win_pct_mate(before_cp) - win_pct_mate(after_cp))


def quality_label(wpl: float, c: ClassifyConfig) -> str:
    """Baseline tier from Win% loss alone."""
    if wpl < c.best_max:
        return "Best"
    if wpl < c.excellent_max:
        return "Excellent"
    if wpl < c.good_max:
        return "Good"
    if wpl < c.inaccuracy_max:
        return "Inaccuracy"
    if wpl < c.mistake_max:
        return "Mistake"
    return "Blunder"


# ---------------------------------------------------------------------------
# Special baseline labels (reconstructed rules, not chess.com's real logic)
# ---------------------------------------------------------------------------

def is_brilliant(wpl: float, sac_cp: float, win_before: float, win_after: float,
                 dispersion: float | None, c: ClassifyConfig) -> bool:
    """Brilliant (!!): the rare, special label. ALL four conditions must hold.

    1. Near-best: the move is best or near-best (Win% loss within near_best_max).
       A losing move is never brilliant, and near_best_max is reused so there is
       one near-best bar everywhere.
    2. Not already winning: Win% before the move is below the ceiling. A sacrifice
       while already crushing is not brilliant.
    3. Sound sacrifice, not regained: `sac_cp` is material given up from the
       mover's point of view that is NOT won back within the next couple of turns
       (see the sac series builder), and the move is not losing after.
    4. Hard to find: Maia human-move dispersion is high, i.e. a move most people
       at this player's rating band would miss. This is the non-obvious criterion.
       Because the dispersion is read at the player's own band, a sacrifice that is
       obvious to a master but not to a beginner is handled naturally, no separate
       rating rule. When dispersion is unknown we do not award Brilliant.
    """
    return (
        wpl <= c.near_best_max
        and win_before <= c.brilliant_winning_ceiling
        and win_after >= c.brilliant_not_losing_floor
        and sac_cp >= c.brilliant_sac_cp
        and dispersion is not None
        and dispersion >= c.brilliant_disp_min
    )


def is_great(wpl: float, eval_gap_cp: float, c: ClassifyConfig) -> bool:
    """Only-good-move: the player found a move that is clearly best while the
    second best is well behind. `eval_gap_cp` is engine best minus second best.
    Time-blind and complexity-blind on purpose, this is the control label."""
    return wpl <= c.near_best_max and eval_gap_cp >= c.great_gap_cp


def baseline_label(wpl: float, in_book: bool, brilliant: bool, great: bool,
                   c: ClassifyConfig) -> str:
    """Full baseline label: Book and the specials sit on top of the quality tier."""
    if in_book:
        return BOOK
    tier = quality_label(wpl, c)
    if brilliant:
        return "Brilliant"
    if great and _rank(tier) < _rank("Great"):
        return "Great"
    return tier


# ---------------------------------------------------------------------------
# The three components and the time-aware upgrade
# ---------------------------------------------------------------------------

def complexity_gate(dispersion: float, c: ClassifyConfig) -> float:
    """Maia dispersion mapped to a [0,1] difficulty gate. 0 for easy positions."""
    if dispersion <= c.disp_lo:
        return 0.0
    if dispersion >= c.disp_hi:
        return 1.0
    return (dispersion - c.disp_lo) / (c.disp_hi - c.disp_lo)


def pressure_modifier(residual_s: float, clock_before_s: float, c: ClassifyConfig) -> float:
    """Genuine time pressure in [0,1]. The clock has to actually be a constraint.

    `clock_factor` is 1 on a near-flag clock and fades to 0 once the clock is above
    the danger zone, so a full clock earns no pressure however fast the move was.
    Within that pressured regime, moving faster than expected (`residual_s` below
    zero) sharpens the credit above the base floor. Fast alone on a high clock
    cannot manufacture pressure.
    """
    clock_factor = min(1.0, max(0.0, c.clock_danger_s - clock_before_s) / c.clock_danger_s)
    fast = min(1.0, max(0.0, -residual_s) / c.pressure_time_ref_s)
    return clock_factor * (c.pressure_base + (1.0 - c.pressure_base) * fast)


def pressure_modifier_bullet(residual_s: float, clock_before_s: float,
                             base_s: float, c: ClassifyConfig) -> float:
    """Genuine time pressure for bullet, in [0,1], normalized to the base clock.

    A bullet game is a minute or two total, so the blitz danger zone in absolute
    seconds is useless here. The clock factor is a fraction of the base instead:
    zero with most of the time left, ramping to one near the flag once the clock
    drops below `bullet_clock_danger_frac` of the base. The fast term (moving
    quicker than the bullet model expects) works like blitz, at a bullet scale.
    """
    frac = clock_before_s / base_s if base_s and base_s > 0 else 1.0
    d = c.bullet_clock_danger_frac
    clock_factor = min(1.0, max(0.0, d - frac) / d) if d > 0 else 0.0
    fast = min(1.0, max(0.0, -residual_s) / c.bullet_pressure_time_ref_s)
    return clock_factor * (c.pressure_base + (1.0 - c.pressure_base) * fast)


def skill_of_move(wpl: float, gate: float, pressure: float, c: ClassifyConfig) -> float:
    """Option B score in [0,1]. Quality is a precondition (bad moves score 0),
    complexity gates multiplicatively (easy positions score 0 no matter how fast),
    and time pressure is the modifier."""
    if wpl > c.eligibility_max:
        return 0.0
    return gate * pressure


def _rank(label: str) -> int:
    return QUALITY_LADDER.index(label) if label in QUALITY_LADDER else -1


def option_b_label(baseline: str, skill: float, brilliant_candidate: bool,
                   wpl: float, c: ClassifyConfig) -> str:
    """Upgrade a baseline label by the Option B score. Never downgrades, never
    touches Book or ineligible (bad) moves. Great needs the move to be near-best,
    so a merely good move caps at Excellent (no two-tier jumps)."""
    if baseline == BOOK or _rank(baseline) < _rank("Good"):
        return baseline
    near_best = wpl <= c.near_best_max
    if brilliant_candidate and skill >= c.cut_brilliant:
        upgraded = "Brilliant"
    elif near_best and skill >= c.cut_great:
        upgraded = "Great"
    elif skill >= c.cut_excellent and baseline == "Good":
        upgraded = "Excellent"
    else:
        upgraded = baseline
    # Keep whichever label is higher, so a strong baseline is never pulled down.
    return upgraded if _rank(upgraded) >= _rank(baseline) else baseline


def option_a_label(baseline: str, wpl: float, dispersion: float, pressure: float,
                   c: ClassifyConfig) -> str:
    """Transparent threshold sanity check: a good move in a hard position played
    under pressure gets bumped one tier."""
    if baseline == BOOK or _rank(baseline) < _rank("Good"):
        return baseline
    if wpl <= c.a_wpl_max and dispersion >= c.a_disp_min and pressure >= c.a_pressure_min:
        idx = min(_rank(baseline) + 1, _rank("Great"))
        label = QUALITY_LADDER[idx]
        # Only near-best moves may reach Great; otherwise cap one below it.
        if label == "Great" and wpl > c.near_best_max:
            label = "Best"
        return label
    return baseline
