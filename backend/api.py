"""The local API the browser app talks to.

Everything runs on the user's machine. The flow is: send a PGN and get back a
plain report of each game it holds (so the user can pick one), then submit a game
to analyze. Analysis can take a while on a fresh game because it runs Stockfish
and Maia locally, so it runs in the background and the client polls for progress
and then the result.

The paste and analysis routes touch no network. The only network calls are the
optional chess.com fetch routes, which pull a game the user explicitly asks for;
they live behind their own module (chesscom) so the offline path never uses them.
Start it with:
    uvicorn backend.api:app
"""

from __future__ import annotations

import os
import threading
import uuid
from dataclasses import dataclass
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from chess_strength.config import load_config

from . import chesscom, maia_client
from .analyze import analyze
from .intake import parse_pgn

app = FastAPI(title="Time-Aware Chess Review")

# The app and the API run on the user's machine. In development the frontend is
# served from a different local port, so allow local origins to call in.
app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=r"http://(localhost|127\.0\.0\.1)(:\d+)?",
    allow_methods=["*"],
    allow_headers=["*"],
)

_CFG = load_config()


@dataclass
class Job:
    id: str
    status: str = "running"  # running, done, error
    stage: str = "starting"
    done: int = 0
    total: int = 0
    fraction: float = 0.0
    result: dict | None = None
    error: str | None = None


_jobs: dict[str, Job] = {}
_lock = threading.Lock()


class PGNBody(BaseModel):
    pgn: str


class AnalyzeBody(BaseModel):
    pgn: str
    game: int = 0
    option: str = "B"


class ChessComGamesBody(BaseModel):
    username: str
    month: str | None = None  # "YYYY/MM", defaults to the most recent


class ChessComGameBody(BaseModel):
    username: str
    link: str  # a chess.com game link or bare id


@app.get("/api/health")
def health() -> dict:
    # locked_user is set when the launcher was started with a chess.com username
    # (bash install/launch.sh <username>). The frontend then opens straight to
    # that account's games and stays there across refreshes.
    return {
        "ok": True,
        "stockfish": os.environ.get("STOCKFISH_PATH") or _CFG.get("stockfish_path"),
        "maia_available": maia_client.available(),
        "locked_user": (os.environ.get("CHESS_REVIEW_USER") or "").strip() or None,
    }


@app.post("/api/intake")
def intake(body: PGNBody) -> dict:
    """Parse a PGN and report each game it holds, without analyzing. This is what
    the paste/upload screen and the game picker use."""
    reports = parse_pgn(body.pgn)
    if not reports:
        raise HTTPException(status_code=422, detail="No game found in that PGN.")
    return {"games": [r.summary() for r in reports]}


# The only network routes. They fetch a game the user asked for; the games they
# return carry a PGN that goes back through the same /api/analyze path as a paste.
@app.post("/api/chesscom/games")
def chesscom_games(body: ChessComGamesBody) -> dict:
    try:
        return chesscom.fetch_month(body.username, body.month)
    except chesscom.ChessComError as e:
        raise HTTPException(status_code=422, detail=str(e)) from e


@app.post("/api/chesscom/game")
def chesscom_game(body: ChessComGameBody) -> dict:
    try:
        return {"game": chesscom.find_game(body.username, body.link)}
    except chesscom.ChessComError as e:
        raise HTTPException(status_code=422, detail=str(e)) from e


def _run(job: Job, report, option: str) -> None:
    def progress(stage: str, done: int, total: int) -> None:
        with _lock:
            job.stage, job.done, job.total = stage, done, total
            frac = (done / total) if total else 1.0
            job.fraction = 0.9 * frac if stage == "engine" else 0.9 + 0.1 * frac

    try:
        result = analyze(report, _CFG, option=option, progress=progress)
        with _lock:
            job.result = result.to_dict()
            job.status, job.fraction, job.stage = "done", 1.0, "done"
    except Exception as exc:  # noqa: BLE001 - record any failure so the user hears about it
        with _lock:
            job.status, job.error = "error", str(exc)


@app.post("/api/analyze")
def start_analyze(body: AnalyzeBody) -> dict:
    reports = parse_pgn(body.pgn)
    if not reports:
        raise HTTPException(status_code=422, detail="No game found in that PGN.")
    if not (0 <= body.game < len(reports)):
        raise HTTPException(status_code=422, detail="That game number is out of range.")
    report = reports[body.game]
    if not report.accepted:
        raise HTTPException(status_code=422, detail=report.reject_reason)
    if report.n_plies == 0:
        raise HTTPException(status_code=422, detail="This game has no moves to review.")

    job = Job(id=uuid.uuid4().hex)
    with _lock:
        _jobs[job.id] = job
    threading.Thread(target=_run, args=(job, report, body.option), daemon=True).start()
    return {"job_id": job.id}


@app.get("/api/progress/{job_id}")
def progress(job_id: str) -> dict:
    job = _jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Unknown job.")
    with _lock:
        return {"status": job.status, "stage": job.stage, "done": job.done,
                "total": job.total, "fraction": round(job.fraction, 3),
                "error": job.error}


@app.get("/api/result/{job_id}")
def result(job_id: str) -> dict:
    job = _jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Unknown job.")
    if job.status == "error":
        raise HTTPException(status_code=500, detail=job.error)
    if job.status != "done":
        raise HTTPException(status_code=425, detail="Still analyzing.")
    return job.result


# Serve the built frontend from the same origin as the API, so one launched
# process serves the whole app. Only mounted when a build is present; in
# development the frontend runs on its own port and this is skipped. Must come
# after the /api routes so it does not shadow them.
_FRONTEND = Path(os.environ.get("FRONTEND_DIR")
                 or Path(__file__).resolve().parents[1] / "frontend" / "out")
if _FRONTEND.is_dir():
    app.mount("/", StaticFiles(directory=str(_FRONTEND), html=True), name="site")
