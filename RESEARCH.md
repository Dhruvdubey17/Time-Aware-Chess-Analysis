# Time-aware chess review: data and findings

This records the data behind the tool and what we learned from it. The raw and
intermediate data (tens of gigabytes) lived only on the build machine and is
gitignored, so it is not in this repository. This file keeps the findings and the
method after that data is deleted. The trained artifacts the app actually uses are
committed under `assets/`, so nothing here is needed to run the app. It matters
only if someone wants to understand or redo the calibration.

## What the tool does

The app labels each move the way online chess does (Best, Excellent, Good, Book,
Inaccuracy, Mistake, Blunder, and the rare Great and Brilliant), then adds a
second read: a move found in a hard position with little time left is credited as
a find under pressure. "Hard" comes from a human-difficulty signal, not from raw
engine loss, and "under pressure" comes from the clock and from how long the
player actually spent. The studies below are how we decided those two pieces are
real and how to measure them.

## The data

Source: the Lichess standard rated game dumps. The primary month is April 2017
(`lichess_db_standard_rated_2017-04.pgn.zst`, 3.2 GB compressed), with May 2017
added for the player-strength work so each test player had enough games. April
2017 is the first month where Lichess PGNs carry per-move clocks (`[%clk]`), which
the whole time-aware idea depends on. About 6 percent of games also carry a
server eval (`[%eval]`), which we used for a cheap complexity proxy before running
any engine.

Processing pipeline:

1. Stream-filter the compressed dump without fully decompressing it, keeping
   standard chess with usable clocks, and shard the result into plain PGNs.
2. Parse to one row per move: who moved, the move, clocks before and after, time
   spent, game phase, and win-percentage before and after.
3. For sampled positions, run Stockfish at a fixed budget (30,000 nodes, multipv
   4) and cache the top candidate evaluations by FEN, so the same position is
   never analyzed twice.
4. For the difficulty study, score positions with Maia-2 at the player's rating
   band and read the dispersion of its move distribution.

The opening book is separate. It is built from the Lichess `chess-openings` TSV
files (`data/raw/openings/a-e.tsv`, CC0) into the small Polyglot book at
`assets/book/openings.bin`, which is what the app reads to decide whether a move
is still theory.

## Findings

### Position complexity predicts think time, weakly per move but clearly in aggregate

The first question was whether harder positions actually cost players more time,
because the upgrade rule multiplies move quality by position complexity and that
only means something if complexity is real. We defined complexity as the decision
entropy over Stockfish's top-four candidate evaluations, then checked it against
time spent (as log(1 + seconds)) on 1,304,461 moves from 3,001 players.

The raw per-move correlation is weak (Spearman +0.046). Two reasons: per-move
times are noisy, and the top-four entropy metric saturates, with 61 percent of
positions sitting at 0.95 or above. Once the noise is averaged out the signal is
clear. Binned Pearson is 0.78 for blitz and 0.83 for rapid, and in a regression
that controls for the clock and game context the complexity coefficient is +0.104
per unit (t = 148, p below machine precision). The direction is positive and
significant in both regimes and in all six rating bands, and the per-band effect
grows with rating. The verdict was to proceed with the complexity-gated upgrade
(called Option B), with one caveat carried forward: do not feed the saturated raw
entropy into the gate, use a reading with more spread.

### Maia dispersion beats engine entropy as the difficulty signal

The saturation problem above pushed the next question: is a deeper engine read the
fix, or do we need a different signal? We first tested whether more engine
resolution de-saturates entropy, running 8,000 positions at 300,000 nodes and
multipv 10. It did not. Even at that budget 76 percent of positions still sat at
0.95 or above and the relationship with think time still turned over, so the
higher-resolution engine read was a no-go.

We then compared four difficulty signals against think time on 20,000 unique
positions. Maia-2 move dispersion (how much human move choice spreads at that
rating) was the clear winner: raw Spearman +0.28, binned Pearson 0.99, and no
turnover. The decisive result is inside the saturated zone, where engine entropy
has no spread left: there Maia dispersion still holds a binned Pearson of 0.99,
while engine entropy runs backwards at -0.69. Maia dispersion tracks difficulty
across all six rating bands (rho 0.26 to 0.31) and both regimes (blitz 0.27,
rapid 0.31). Maia's top-move probability ran the wrong way (-0.24), so dispersion,
not confidence, is the right reading. This is why the app's difficulty gate is
Maia-2 dispersion rather than an engine number.

### A player-strength baseline: raw move loss beat the conditioned scores

Before the per-move labels, we tested a per-player strength score, a win-percent
version of an intrinsic performance rating where the engine measures how hard each
position was and never grades the move. The first version conditioned expected
move loss on rating band, eval bucket, clock usage, and eval volatility, and
scored only Spearman 0.157 on held-out players. Two causes: too little data (a
median of 29 clean moves per test player) and post-treatment bias, since eval
bucket, clock usage, and volatility are themselves consequences of skill, so
subtracting an expectation built from them removes the signal.

The second version kept only conditioning that is not a consequence of skill (game
phase and engine position complexity) and widened the data to two months, raising
the median to 158 clean moves per player. Plain raw negative mean move loss
reached Spearman 0.549 on 36,345 held-out players with at least 100 moves, past
the 0.30 gate. The phase-plus-complexity residual scored below raw at every cutoff
(0.558 against 0.611 overall). The simpler score won, so raw move loss is the
chosen estimator and the residual stays only as a reported comparison. The lesson
carried into the rest of the project: condition only on things that are not
downstream of skill.

### The time-aware upgrade tracks real skill

We then checked whether the two upgrade rules actually pick out stronger players,
on 321,557 moves (318,281 eligible), reading Maia at a fixed 1500 Elo so rating
could not leak in. Players whose moves earned upgrades did rate higher.
Option B held a Spearman near 0.10 (p below 1e-9) at both the 60-second and
120-second clock thresholds, and Option A ran from about 0.074 to 0.096. Both are
positive and significant, with B a little stronger. We took 120 seconds as the
primary clock threshold and kept 60 seconds as the stricter robustness reading.
The exact same-position test was not feasible because the sampled positions were
effectively unique.

### The special labels stay rare

On a 3,632-move sample, Option B upgraded 18 moves and Option A upgraded 8. The
upgrade is meant to be uncommon, and it is, around half a percent of moves, so the
credit stays meaningful.

### Bullet fits the model, but Lichess clocks limit premove detection

Bullet was fit as its own regime rather than reusing the blitz model, on 700,000
genuine (non-premove) moves. The held-out R-squared is 0.159 and Maia dispersion
still tracks think time (Spearman +0.136), so the difficulty signal survives in
bullet. The honest limit is the clock resolution. Lichess public PGNs report
whole seconds, so any think under a second floors to zero and a true premove
cannot be told apart from a fast genuine move. In the bullet fitting set that left
0 confidently identified premoves, 925,325 sub-floor-ambiguous moves, and
3,102,772 genuine ones. On Lichess bullet the app therefore never credits a
sub-second move as a find under pressure, and says so. Chess.com bullet clocks
carry tenths, so premoves are identifiable there.

## How the research fed the shipped app

The app does not ship or need this data. It ships the fitted results:

- the expected-think-time models under `assets/thinktime/` (blitz and rapid) and
  `assets/thinktime/bullet/`, used to turn time spent into a pressure residual,
- the Polyglot opening book at `assets/book/openings.bin`,
- the classifier thresholds in `config/default.yaml`, set from these studies.

At review time the app recomputes Stockfish (same 30,000 nodes, multipv 4) and
Maia-2 dispersion for the one game in front of it, so the training data was only
ever needed to fit and validate, not to run.

## Reproducing this

The scripts under `scripts/` regenerate the tables and reports from a
re-downloaded Lichess dump: `build_features.py`, `thinktime_validation.py`,
`complexity_pilot.py`, `maia_validation.py`, `validate_skill.py`,
`baseline_residual.py`, `build_bullet_thinktime.py`, and `build_opening_book.py`.
The data directories (`data/raw`, `data/interim`, `data/processed`) are gitignored
and were not committed, so a rebuild starts by fetching the April and May 2017
Lichess standard rated dumps again. The engine budget and rating bands are fixed
in `config/default.yaml`; keep them as they are, since the shipped think-time
models and thresholds were fit at those exact settings.
