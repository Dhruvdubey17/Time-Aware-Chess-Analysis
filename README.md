# Chess Review

Review one of your chess games on your own computer, and get a second opinion
that most tools miss: which good moves you found while the clock was against you.

Every move gets a normal label you already know from online chess: Best,
Excellent, Good, Book, Inaccuracy, Mistake, Blunder, and the rare Great and
Brilliant. On top of that, moves you found in a hard position with little time
left get a special mention, because a strong move played under real pressure
deserves more credit than the same move with all the time in the world.

Everything runs on your machine. Your games are not sent anywhere.

## Get the app

First get the project folder onto your machine. Two ways:

- With git, which also lets you update later with `git pull`:

```bash
git clone https://github.com/Dhruvdubey17/Time-Aware-Chess-Analysis.git
```

- Without git: on the GitHub page use the green **Code** button, **Download
  ZIP**, then unzip it.

Either way you end up with a `Time-Aware-Chess-Analysis` folder. The interface
ships inside it already built, so you do not need Node.js or any web tools. On
Windows, the git option needs Git for Windows; the ZIP option needs nothing
extra.

## Install

You only do this once. Open a terminal in the folder from the step above and run
the line for your system. It downloads what it needs and sets everything up. It
never asks for your password.

**macOS or Linux**

```bash
bash install/install.sh
```

**Windows** (in PowerShell)

```powershell
powershell -ExecutionPolicy Bypass -File install\install.ps1
```

When it finishes it prints the one command to start the app.

## Start the app

**macOS or Linux**

```bash
bash install/launch.sh
```

**Windows**

```powershell
powershell -ExecutionPolicy Bypass -File install\launch.ps1
```

Your browser opens to the app. To stop it, press Ctrl+C in that terminal, or
just close the window.

## How to use it

1. Paste a game, or upload a `.pgn` file, or type a chess.com username to pull a
   game from your account.
2. Wait for the review. Step through the game with the move list, the arrow
   keys, or the graph.
3. Each move shows its label. Moves found under real time pressure light up, and
   selecting one explains why in plain words.

## Remove the app

To clear the app from your machine, run the line for your system. It removes what
the setup created inside this folder (the environments, the chess engine, the Maia
weights, and the cache). It leaves Python and uv alone, since those live outside
this folder and you may use them elsewhere.

**macOS or Linux**

```bash
bash install/delete.sh
```

**Windows**

```powershell
powershell -ExecutionPolicy Bypass -File install\delete.ps1
```

## A note on speed and privacy

The first review of a new game takes a little while, from a few seconds to a
couple of minutes, because the whole analysis runs on your machine, not in the
cloud. A game you have already looked at comes back instantly.

Reviewing a pasted or uploaded game uses no internet at all. The only time the
app goes online is if you ask it to fetch a game from chess.com, and then only
to get that one game.

The time-aware second opinion is available for blitz, rapid, and bullet games
that include move times. Games without move times still get the full normal
review.

## Good to know

- The install and launch are proven on macOS (Apple Silicon). The Windows and
  Linux scripts are written and ready, but are still pending a test run on real
  Windows and Linux machines.
- On Linux, the chess engine download supports Intel and AMD (x86_64) machines.
  Linux on ARM is not covered yet.
- Bullet games get the time-aware review too, with one honest limit. On chess.com
  bullet the clocks carry tenths of a second, so premoves are spotted and the
  under-pressure highlights are reliable. On Lichess bullet the public clocks are
  whole seconds only, so a move under a second cannot be told apart from a premove;
  those moves are still shown, but never credited as a find under pressure, and the
  review says so.
