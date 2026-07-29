#!/usr/bin/env bash
# One-shot setup for the Chess Review app on macOS and Linux.
#
# You do not need to know anything technical. This script sets up everything the
# app needs on your own machine: a small chess engine, the review programs, and
# the web interface. It downloads what it needs while it runs. After it finishes,
# it prints one command to start the app.
#
# It never asks for your password and never changes system settings. Everything
# lives inside this folder.

set -euo pipefail

# Always work from the project root (the folder above this script).
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

say()  { printf '\n\033[1m==> %s\033[0m\n' "$*"; }
info() { printf '    %s\n' "$*"; }
die()  { printf '\n\033[1;31mSetup stopped: %s\033[0m\n' "$*" >&2; exit 1; }

OS="$(uname -s)"
ARCH="$(uname -m)"
say "Setting up Chess Review for $OS on $ARCH"

# ---------------------------------------------------------------------------
# 1. uv: a small, self-contained tool that manages Python for us, so we never
#    touch or fight the system Python. Installed into your home folder only.
# ---------------------------------------------------------------------------
say "Step 1 of 6: preparing the Python toolchain"
if ! command -v uv >/dev/null 2>&1; then
  info "Downloading uv (one small program, no admin rights needed) ..."
  curl -LsSf https://astral.sh/uv/install.sh | sh || die "could not download uv. Check your internet connection."
  export PATH="$HOME/.local/bin:$PATH"
fi
command -v uv >/dev/null 2>&1 || die "uv is not on the PATH after install. Open a new terminal and run this again."
uv python install 3.12 || die "could not install Python 3.12."

# ---------------------------------------------------------------------------
# 2. The main environment: the review service and web server. No PyTorch here.
# ---------------------------------------------------------------------------
say "Step 2 of 6: installing the review service"
uv venv .venv --python 3.12 --clear
# Install the project code without its research-only dependencies, then just the
# packages the app actually needs at runtime. Keeps the install small and quick.
uv pip install --python .venv -e . --no-deps -q
uv pip install --python .venv -q \
  "python-chess>=1.11" numpy pandas pyarrow scikit-learn joblib pyyaml \
  "fastapi>=0.110" "uvicorn>=0.29" \
  || die "could not install the review service packages."

# ---------------------------------------------------------------------------
# 3. The Maia environment: the human-difficulty model. It needs PyTorch and an
#    older numpy, which do not mix with the main environment, so it lives on its
#    own. This is the numpy/venv split that has to stay separate.
# ---------------------------------------------------------------------------
say "Step 3 of 6: installing the human-difficulty model (this is the largest download)"
uv venv .venv_maia --python 3.12 --clear
if [ "$OS" = "Linux" ]; then
  # Default PyTorch on Linux pulls a huge GPU build; we only need the CPU one.
  uv pip install --python .venv_maia -q torch --index-url https://download.pytorch.org/whl/cpu \
    || die "could not install PyTorch (CPU)."
fi
uv pip install --python .venv_maia -q \
  "numpy<2" torch maia2 "python-chess>=1.11" pandas pyarrow gdown requests tqdm pyzstd einops scikit-learn pyyaml \
  || die "could not install the Maia model packages."

# ---------------------------------------------------------------------------
# 4. Stockfish: the chess engine, fetched as the official binary for your machine.
# ---------------------------------------------------------------------------
say "Step 4 of 6: fetching the Stockfish chess engine"
mkdir -p engines
if [ -x engines/stockfish ] && [ -f engines/STOCKFISH_PATH ]; then
  info "Stockfish is already present, skipping the download."
else
asset=""
case "$OS-$ARCH" in
  Darwin-arm64)          asset="stockfish-macos-m1-apple-silicon.tar" ;;
  Darwin-x86_64)         asset="stockfish-macos-x86-64-avx2.tar" ;;
  Linux-x86_64)          asset="stockfish-ubuntu-x86-64-avx2.tar" ;;
  *) die "no official Stockfish build for $OS-$ARCH. Please report your platform." ;;
esac
rm -rf engines/_tmp && mkdir -p engines/_tmp
info "Downloading $asset ..."
curl -L --fail -o "engines/_tmp/$asset" \
  "https://github.com/official-stockfish/Stockfish/releases/latest/download/$asset" \
  || die "could not download Stockfish. Check your internet connection."
tar -xf "engines/_tmp/$asset" -C engines/_tmp
sf_bin="$(find engines/_tmp -type f -name 'stockfish*' ! -name '*.tar' | head -n1)"
[ -n "$sf_bin" ] || die "could not find the Stockfish program inside the download."
cp "$sf_bin" engines/stockfish
chmod +x engines/stockfish
rm -rf engines/_tmp
echo "$ROOT/engines/stockfish" > engines/STOCKFISH_PATH
info "Stockfish ready at engines/stockfish"
fi

# ---------------------------------------------------------------------------
# 5. Maia weights and the web interface.
# ---------------------------------------------------------------------------
say "Step 5 of 6: downloading Maia weights and preparing the interface"
info "Downloading Maia weights (blitz and rapid) ..."
.venv_maia/bin/python - <<'PY' || die "could not download the Maia weights."
from maia2 import model
for kind in ("blitz", "rapid"):
    model.from_pretrained(type=kind, device="cpu", save_root="data/processed/maia_val/weights")
print("maia weights ready")
PY

if [ -f frontend/out/index.html ]; then
  info "Using the interface that ships with this release."
elif command -v npm >/dev/null 2>&1; then
  info "Building the interface with Node ..."
  ( cd frontend && npm ci --silent && NEXT_PUBLIC_API_BASE="" npm run build >/dev/null ) \
    || die "could not build the interface."
else
  die "the interface is not prebuilt and Node.js was not found. Install Node.js 18+ and run this again, or use a release that includes the prebuilt interface."
fi

# ---------------------------------------------------------------------------
# 6. Self-test: prove every piece actually works before we say we are done.
# ---------------------------------------------------------------------------
say "Step 6 of 6: checking that everything works"
STOCKFISH_PATH="$ROOT/engines/stockfish" .venv/bin/python - <<'PY' || die "the review service self-test failed."
import os, chess, chess.engine
import backend.intake, backend.analyze, backend.api  # imports must succeed
from chess_strength.thinktime import load_mapping
load_mapping("assets/thinktime")  # the trained model must load
eng = chess.engine.SimpleEngine.popen_uci(os.environ["STOCKFISH_PATH"])
eng.analyse(chess.Board(), chess.engine.Limit(nodes=1000))
eng.quit()
print("review service ok")
PY
prefix=""; [ "$OS" = "Darwin" ] && prefix="arch -arm64"
$prefix .venv_maia/bin/python - <<'PY' || die "the Maia model self-test failed."
import maia2, torch  # noqa: F401
from pathlib import Path
w = Path("data/processed/maia_val/weights")
assert (w / "blitz_model.pt").exists() and (w / "rapid_model.pt").exists(), "weights missing"
print("maia model ok")
PY
[ -f frontend/out/index.html ] || die "the interface is missing."

say "All set."
info "Start the app any time with this one command:"
printf '\n    \033[1mbash install/launch.sh\033[0m\n\n'
info "The first review of a new game takes a little while because it all runs on"
info "your machine. Games you have looked at before come back instantly."
