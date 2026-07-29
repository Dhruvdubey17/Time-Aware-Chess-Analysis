# One-shot setup for the Chess Review app on Windows (PowerShell).
#
# You do not need to know anything technical. This sets up everything the app
# needs inside this folder: a small chess engine, the review programs, and the
# web interface. It downloads what it needs while it runs, then prints one
# command to start the app. It never needs administrator rights.
#
# If Windows blocks the script, run it like this:
#   powershell -ExecutionPolicy Bypass -File install\install.ps1

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $Root

function Say($m)  { Write-Host "`n==> $m" -ForegroundColor Cyan }
function Info($m) { Write-Host "    $m" }
function Die($m)  { Write-Host "`nSetup stopped: $m" -ForegroundColor Red; exit 1 }

$arch = $env:PROCESSOR_ARCHITECTURE
Say "Setting up Chess Review for Windows on $arch"

# 1. uv: manages Python for us, in your user profile, no admin rights.
Say "Step 1 of 6: preparing the Python toolchain"
if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
  Info "Downloading uv (one small program) ..."
  try { Invoke-RestMethod https://astral.sh/uv/install.ps1 | Invoke-Expression }
  catch { Die "could not download uv. Check your internet connection." }
  $env:Path = "$env:USERPROFILE\.local\bin;$env:Path"
}
if (-not (Get-Command uv -ErrorAction SilentlyContinue)) { Die "uv is not on the PATH. Open a new terminal and try again." }
uv python install 3.12

# 2. Main environment: the review service and web server.
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
uv pip install --python .venv_maia -q `
  "numpy<2" torch maia2 "python-chess>=1.11" pandas pyarrow gdown requests tqdm pyzstd einops scikit-learn pyyaml
if ($LASTEXITCODE -ne 0) { Die "could not install the Maia model packages." }

# 4. Stockfish: the official chess engine for your machine.
Say "Step 4 of 6: fetching the Stockfish chess engine"
New-Item -ItemType Directory -Force -Path engines | Out-Null
if ((Test-Path engines\stockfish.exe) -and (Test-Path engines\STOCKFISH_PATH)) {
  Info "Stockfish is already present, skipping the download."
} else {
  if ($arch -eq "ARM64") { $asset = "stockfish-windows-armv8.zip" } else { $asset = "stockfish-windows-x86-64-avx2.zip" }
  $tmp = "engines\_tmp"
  Remove-Item -Recurse -Force $tmp -ErrorAction SilentlyContinue
  New-Item -ItemType Directory -Force -Path $tmp | Out-Null
  Info "Downloading $asset ..."
  try { Invoke-WebRequest "https://github.com/official-stockfish/Stockfish/releases/latest/download/$asset" -OutFile "$tmp\$asset" }
  catch { Die "could not download Stockfish. Check your internet connection." }
  Expand-Archive -Path "$tmp\$asset" -DestinationPath $tmp -Force
  $exe = Get-ChildItem -Path $tmp -Recurse -Filter "stockfish*.exe" | Select-Object -First 1
  if (-not $exe) { Die "could not find the Stockfish program inside the download." }
  Copy-Item $exe.FullName engines\stockfish.exe -Force
  Remove-Item -Recurse -Force $tmp
  "$Root\engines\stockfish.exe" | Out-File -Encoding ascii -NoNewline engines\STOCKFISH_PATH
  Info "Stockfish ready at engines\stockfish.exe"
}

# 5. Maia weights and the web interface.
Say "Step 5 of 6: downloading Maia weights and preparing the interface"
Info "Downloading Maia weights (blitz and rapid) ..."
& .venv_maia\Scripts\python.exe -c "from maia2 import model; [model.from_pretrained(type=k, device='cpu', save_root='data/processed/maia_val/weights') for k in ('blitz','rapid')]; print('maia weights ready')"
if ($LASTEXITCODE -ne 0) { Die "could not download the Maia weights." }

if (Test-Path frontend\out\index.html) {
  Info "Using the interface that ships with this release."
} elseif (Get-Command npm -ErrorAction SilentlyContinue) {
  Info "Building the interface with Node ..."
  Push-Location frontend
  $env:NEXT_PUBLIC_API_BASE = ""
  npm ci --silent; npm run build | Out-Null
  Pop-Location
  if (-not (Test-Path frontend\out\index.html)) { Die "could not build the interface." }
} else {
  Die "the interface is not prebuilt and Node.js was not found. Install Node.js 18+ and try again, or use a release that includes the prebuilt interface."
}

# 6. Self-test.
Say "Step 6 of 6: checking that everything works"
$env:STOCKFISH_PATH = "$Root\engines\stockfish.exe"
& .venv\Scripts\python.exe -c "import os, chess, chess.engine; import backend.intake, backend.analyze, backend.api; from chess_strength.thinktime import load_mapping; load_mapping('assets/thinktime'); e=chess.engine.SimpleEngine.popen_uci(os.environ['STOCKFISH_PATH']); e.analyse(chess.Board(), chess.engine.Limit(nodes=1000)); e.quit(); print('review service ok')"
if ($LASTEXITCODE -ne 0) { Die "the review service self-test failed." }
& .venv_maia\Scripts\python.exe -c "import maia2, torch; from pathlib import Path; w=Path('data/processed/maia_val/weights'); assert (w/'blitz_model.pt').exists() and (w/'rapid_model.pt').exists(); print('maia model ok')"
if ($LASTEXITCODE -ne 0) { Die "the Maia model self-test failed." }
if (-not (Test-Path frontend\out\index.html)) { Die "the interface is missing." }

Say "All set."
Info "Start the app any time with this one command:"
Write-Host "`n    powershell -ExecutionPolicy Bypass -File install\launch.ps1`n" -ForegroundColor Green
Info "The first review of a new game takes a little while because it all runs on"
Info "your machine. Games you have looked at before come back instantly."
