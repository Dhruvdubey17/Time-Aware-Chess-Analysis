# One-shot setup for the Chess Review app on Windows (PowerShell).
#
# You do not need to know anything technical. This sets up everything the app
# needs inside this folder: a small chess engine, the review programs, and the
# human-difficulty model. The interface is already built and ships with the app,
# so you do NOT need Node.js or any web tools. It downloads what it needs while
# it runs, then prints one command to start the app. It never needs administrator
# rights.
#
# If Windows blocks the script, run it like this:
#   powershell -ExecutionPolicy Bypass -File install\install.ps1

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $Root

function Say($m)  { Write-Host "`n==> $m" -ForegroundColor Cyan }
function Info($m) { Write-Host "    $m" }
function Die($m)  { Write-Host "`nSetup stopped: $m" -ForegroundColor Red; exit 1 }

# Temporary things THIS run creates. We remove them ONLY after a fully
# successful self-test (see the end). Nothing the app needs is ever listed here.
$Cleanup = @()

$arch = $env:PROCESSOR_ARCHITECTURE
Say "Setting up Chess Review for Windows on $arch"

# 1. uv: manages Python for us, in your user profile, no admin rights.
Say "Step 1 of 6: preparing the Python toolchain"
if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
  Info "Downloading uv (one small program) ..."
  try { Invoke-RestMethod https://astral.sh/uv/install.ps1 | Invoke-Expression }
  catch { Die "could not download uv. Check your internet connection." }
  # The installer drops uv here; put it on PATH for the rest of this run.
  $env:Path = "$env:USERPROFILE\.local\bin;$env:Path"
}
if (-not (Get-Command uv -ErrorAction SilentlyContinue)) { Die "uv is not on the PATH. Open a new terminal and try again." }
# A uv-managed Python, so we never depend on a system Python being present.
uv python install 3.12
if ($LASTEXITCODE -ne 0) { Die "could not install Python 3.12." }

# 2. Main environment: the review service and web server. uv creates the
#    interpreter at .venv\Scripts\python.exe; we call it by that exact path
#    everywhere, so nothing here relies on a system Python.
Say "Step 2 of 6: installing the review service"
uv venv .venv --python 3.12 --clear
uv pip install --python .venv -e . --no-deps -q
uv pip install --python .venv -q `
  "python-chess>=1.11" numpy pandas pyarrow scikit-learn joblib pyyaml `
  "fastapi>=0.110" "uvicorn>=0.29"
if ($LASTEXITCODE -ne 0) { Die "could not install the review service packages." }

# 3. Maia environment: PyTorch and an older numpy, kept separate on purpose.
Say "Step 3 of 6: installing the human-difficulty model (largest download)"
uv venv .venv_maia --python 3.12 --clear
# PyTorch is the flakiest dependency on a fresh Windows machine, so pin it and
# install a known CPU build explicitly from PyTorch's CPU index. The default
# index can pull a huge GPU build. torch 2.13.0 has a Python 3.12 CPU wheel for
# Windows (x86-64 and ARM64).
uv pip install --python .venv_maia -q "torch==2.13.0" --index-url https://download.pytorch.org/whl/cpu
if ($LASTEXITCODE -ne 0) { Die "could not install PyTorch (the CPU build torch==2.13.0 for Python 3.12). This is the most common failure on a fresh Windows machine. Check your internet connection and run this again. If it keeps failing, see https://pytorch.org/get-started/locally/ for a CPU build for your system." }
# The rest come from the normal index. torch is already installed and satisfies
# maia2, so this step does not touch it. numpy is held below 2 on purpose.
uv pip install --python .venv_maia -q `
  "numpy<2" maia2 "python-chess>=1.11" pandas pyarrow gdown requests tqdm pyzstd einops scikit-learn pyyaml
if ($LASTEXITCODE -ne 0) { Die "could not install the Maia model packages." }

# 4. Stockfish: the official chess engine for your machine.
Say "Step 4 of 6: fetching the Stockfish chess engine"
New-Item -ItemType Directory -Force -Path engines | Out-Null
if ((Test-Path engines\stockfish.exe) -and (Test-Path engines\STOCKFISH_PATH)) {
  Info "Stockfish is already present, skipping the download."
} else {
  if ($arch -eq "ARM64") { $asset = "stockfish-windows-armv8.zip" } else { $asset = "stockfish-windows-x86-64-avx2.zip" }
  $tmp = "engines\_tmp"
  # Download the .zip and unpack it into a scratch folder. Both the .zip and the
  # unpacked files are temporary, so we clean them up on success (see the end).
  Remove-Item -Recurse -Force $tmp -ErrorAction SilentlyContinue
  New-Item -ItemType Directory -Force -Path $tmp | Out-Null
  $Cleanup += $tmp
  Info "Downloading $asset ..."
  try { Invoke-WebRequest "https://github.com/official-stockfish/Stockfish/releases/latest/download/$asset" -OutFile "$tmp\$asset" }
  catch { Die "could not download Stockfish. Check your internet connection." }
  try { Expand-Archive -Path "$tmp\$asset" -DestinationPath $tmp -Force }
  catch { Die "could not unpack the Stockfish download." }
  # The .exe sits inside the .zip under a stockfish*.exe name; find it.
  $exe = Get-ChildItem -Path $tmp -Recurse -Filter "stockfish*.exe" | Select-Object -First 1
  if (-not $exe) { Die "could not find the Stockfish program inside the download." }
  Copy-Item $exe.FullName engines\stockfish.exe -Force
  "$Root\engines\stockfish.exe" | Out-File -Encoding ascii -NoNewline engines\STOCKFISH_PATH
  Info "Stockfish ready at engines\stockfish.exe"
}

# 5. Maia weights, and confirm the prebuilt interface is here.
Say "Step 5 of 6: downloading Maia weights and checking the interface"
Info "Downloading Maia weights (blitz and rapid) ..."
& .venv_maia\Scripts\python.exe -c "from maia2 import model; [model.from_pretrained(type=k, device='cpu', save_root='data/processed/maia_val/weights') for k in ('blitz','rapid')]; print('maia weights ready')"
if ($LASTEXITCODE -ne 0) { Die "could not download the Maia weights." }

# The interface ships prebuilt as static files, so the user never needs Node.js
# and never runs a build. One prebuilt bundle serves macOS, Windows, and Linux.
# If it is missing this is a broken release, not a user problem, so the message
# is aimed at whoever built the release, and we never fall back to needing Node.
if (-not (Test-Path frontend\out\index.html)) {
  Die "this release is missing the prebuilt interface (frontend\out\index.html). Whoever built this release must run 'bash install/prepare_release.sh' and include frontend\out in the package. See install/RELEASE.md."
}
Info "The interface is prebuilt and ready, so no Node.js is needed."

# 6. Self-test: prove every critical piece works, each with its own clear
#    message so a broken piece is named, never a generic failure.
Say "Step 6 of 6: checking that everything works"

& .venv\Scripts\python.exe -c "import backend.intake, backend.analyze, backend.api; from chess_strength.thinktime import load_mapping; load_mapping('assets/thinktime'); print('review service ok')"
if ($LASTEXITCODE -ne 0) { Die "the review service environment is broken (its imports or the think-time model failed to load)." }

& .venv\Scripts\python.exe -c "import chess; from backend import book; r=book.open_book({}); assert r is not None and book.book_moves(r, chess.Board().fen()), 'empty'; r.close(); print('book ok')"
if ($LASTEXITCODE -ne 0) { Die "the bundled opening book (assets\book\openings.bin) is missing or unreadable." }

$env:STOCKFISH_PATH = "$Root\engines\stockfish.exe"
& .venv\Scripts\python.exe -c "import os, chess, chess.engine; e=chess.engine.SimpleEngine.popen_uci(os.environ['STOCKFISH_PATH']); e.analyse(chess.Board(), chess.engine.Limit(nodes=1000)); e.quit(); print('stockfish ok')"
if ($LASTEXITCODE -ne 0) { Die "Stockfish did not answer a UCI handshake. The engine at engines\stockfish.exe may be the wrong build for this machine." }

& .venv_maia\Scripts\python.exe -c "import torch; from maia2 import model; model.from_pretrained(type='blitz', device='cpu', save_root='data/processed/maia_val/weights'); print('maia ok')"
if ($LASTEXITCODE -ne 0) { Die "the Maia-2 model failed to load (weights missing or corrupt, or a torch/numpy mismatch)." }

if (-not (Test-Path frontend\out\index.html)) { Die "the prebuilt interface (frontend\out) is missing from this release." }

# --- Success-only cleanup -------------------------------------------------
# We only reach here if every step above passed (Die exits on any failure), so a
# failed or partial install leaves everything in place for debugging. Remove ONLY
# the temporary things this run created, listed above.
if ($Cleanup.Count -gt 0) {
  Say "Cleaning up temporary download files"
  foreach ($p in $Cleanup) {
    if (Test-Path $p) {
      Info "removing $p (Stockfish download and unpacked files, no longer needed)"
      Remove-Item -Recurse -Force $p
    }
  }
}

Say "All set."
Info "Start the app any time with this one command:"
Write-Host "`n    powershell -ExecutionPolicy Bypass -File install\launch.ps1`n" -ForegroundColor Green
Info "The first review of a new game takes a little while because it all runs on"
Info "your machine. Games you have looked at before come back instantly."
