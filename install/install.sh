#!/usr/bin/env bash
# One-shot setup for the Chess Review app on macOS and Linux.
#
# You do not need to know anything technical. This script sets up everything the
# app needs on your own machine: a small chess engine, the review programs, and
# the human-difficulty model. The interface is already built and ships with the
# app, so you do NOT need Node.js or any web tools. It downloads what it needs
# while it runs. After it finishes, it prints one command to start the app.
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

# Temporary things THIS run creates. We remove them ONLY after a fully
# successful self-test (see the end). Nothing the app needs is ever listed here.
CLEANUP=()

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
  # The installer drops uv here; put it on PATH for the rest of this run.
  export PATH="$HOME/.local/bin:$PATH"
fi
command -v uv >/dev/null 2>&1 || die "uv is not on the PATH after install. Open a new terminal and run this again."
# A uv-managed Python, so we never depend on a system Python being present.
uv python install 3.12 || die "could not install Python 3.12."

# ---------------------------------------------------------------------------
# 2. The main environment: the review service and web server. No PyTorch here.
#    uv creates the interpreter at .venv/bin/python; we call it by that exact
#    path everywhere, so nothing here relies on a system Python.
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
# PyTorch is the flakiest dependency on a fresh machine, so pin it and install a
# known CPU build explicitly. On Linux the default index pulls a huge GPU build,
# so use PyTorch's CPU index there. On macOS the normal wheel is already CPU
# (there is no CUDA on Mac), so no special index is needed. torch 2.13.0 has a
# Python 3.12 CPU wheel for macOS arm64, Linux x86_64, and Windows.
torch_index=""
[ "$OS" = "Linux" ] && torch_index="--index-url https://download.pytorch.org/whl/cpu"
# shellcheck disable=SC2086
uv pip install --python .venv_maia -q "torch==2.13.0" $torch_index \
  || die "could not install PyTorch (the CPU build torch==2.13.0 for Python 3.12). This is the most common failure on a fresh machine. Check your internet connection and run this again. If it keeps failing, see https://pytorch.org/get-started/locally/ for a CPU build for your system."
# The rest come from the normal index. torch is already installed and satisfies
# maia2, so this step does not touch it. numpy is held below 2 on purpose.
uv pip install --python .venv_maia -q \
  "numpy<2" maia2 "python-chess>=1.11" pandas pyarrow gdown requests tqdm pyzstd einops scikit-learn pyyaml \
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
# Download the archive and unpack it into a scratch folder. Both the archive and
# the unpacked files are temporary, so we clean them up on success (see the end).
rm -rf engines/_tmp && mkdir -p engines/_tmp
CLEANUP+=("engines/_tmp")
info "Downloading $asset ..."
curl -L --fail -o "engines/_tmp/$asset" \
  "https://github.com/official-stockfish/Stockfish/releases/latest/download/$asset" \
  || die "could not download Stockfish. Check your internet connection."
tar -xf "engines/_tmp/$asset" -C engines/_tmp || die "could not unpack the Stockfish download."
# The binary sits inside the archive under a stockfish* name; find it, ignoring
# the .tar itself.
sf_bin="$(find engines/_tmp -type f -name 'stockfish*' ! -name '*.tar' | head -n1)"
[ -n "$sf_bin" ] || die "could not find the Stockfish program inside the download."
cp "$sf_bin" engines/stockfish
chmod +x engines/stockfish
echo "$ROOT/engines/stockfish" > engines/STOCKFISH_PATH
info "Stockfish ready at engines/stockfish"
fi

# ---------------------------------------------------------------------------
# 5. Maia weights, and confirm the prebuilt interface is here.
# ---------------------------------------------------------------------------
say "Step 5 of 6: downloading Maia weights and checking the interface"
info "Downloading Maia weights (blitz and rapid) ..."
.venv_maia/bin/python - <<'PY' || die "could not download the Maia weights."
from maia2 import model
for kind in ("blitz", "rapid"):
    model.from_pretrained(type=kind, device="cpu", save_root="data/processed/maia_val/weights")
print("maia weights ready")
PY

# The interface ships prebuilt as static files, so the user never needs Node.js
# and never runs a build. One prebuilt bundle serves macOS, Windows, and Linux.
# If it is missing this is a broken release, not a user problem, so the message
# is aimed at whoever built the release, and we never fall back to needing Node.
if [ ! -f frontend/out/index.html ]; then
  die "this release is missing the prebuilt interface (frontend/out/index.html). Whoever built this release must run 'bash install/prepare_release.sh' and include frontend/out in the package. See install/RELEASE.md."
fi
info "The interface is prebuilt and ready, so no Node.js is needed."

# ---------------------------------------------------------------------------
# 6. Self-test: prove every critical piece actually works before we say we are
#    done. Each check fails with its own clear message so a broken piece is
#    named, never a generic "something went wrong".
# ---------------------------------------------------------------------------
say "Step 6 of 6: checking that everything works"

.venv/bin/python -c "import backend.intake, backend.analyze, backend.api; from chess_strength.thinktime import load_mapping; load_mapping('assets/thinktime'); print('review service ok')" \
  || die "the review service environment is broken (its imports or the think-time model failed to load)."

.venv/bin/python -c "import chess; from backend import book; r=book.open_book({}); assert r is not None and book.book_moves(r, chess.Board().fen()), 'empty'; r.close(); print('book ok')" \
  || die "the bundled opening book (assets/book/openings.bin) is missing or unreadable."

STOCKFISH_PATH="$ROOT/engines/stockfish" .venv/bin/python -c "import os, chess, chess.engine; e=chess.engine.SimpleEngine.popen_uci(os.environ['STOCKFISH_PATH']); e.analyse(chess.Board(), chess.engine.Limit(nodes=1000)); e.quit(); print('stockfish ok')" \
  || die "Stockfish did not answer a UCI handshake. The engine at engines/stockfish may be the wrong build for this machine."

# The Maia venv is native arm64 on Apple Silicon; the main venv can run under
# Rosetta, so force the child native or its arm64 numpy will not load.
prefix=""; [ "$OS" = "Darwin" ] && prefix="arch -arm64"
$prefix .venv_maia/bin/python -c "import torch; from maia2 import model; model.from_pretrained(type='blitz', device='cpu', save_root='data/processed/maia_val/weights'); print('maia ok')" \
  || die "the Maia-2 model failed to load (weights missing or corrupt, or a torch/numpy mismatch)."

[ -f frontend/out/index.html ] || die "the prebuilt interface (frontend/out) is missing from this release."

# --- Success-only cleanup -------------------------------------------------
# We only reach here if every step above passed (set -e plus die exit on any
# failure), so a failed or partial install leaves everything in place for
# debugging. Remove ONLY the temporary things this run created, listed above.
if [ "${#CLEANUP[@]}" -gt 0 ]; then
  say "Cleaning up temporary download files"
  for p in "${CLEANUP[@]}"; do
    if [ -e "$p" ]; then
      info "removing $p (Stockfish download and unpacked files, no longer needed)"
      rm -rf "$p"
    fi
  done
fi

say "All set."
info "Start the app any time with this one command:"
printf '\n    \033[1mbash install/launch.sh\033[0m\n\n'
info "The first review of a new game takes a little while because it all runs on"
info "your machine. Games you have looked at before come back instantly."
