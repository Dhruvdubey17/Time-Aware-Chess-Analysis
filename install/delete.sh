#!/usr/bin/env bash
# Remove everything the setup created for the Chess Review app, so it is cleared
# from your machine.
#
# This only removes things inside this app folder that the setup made: the two
# Python environments, the chess engine, the Maia model weights, the cached
# analysis, and the built interface. It does NOT touch tools that live outside
# this folder or were already on your machine, like Python or uv.
#
# Run it with:  bash install/delete.sh     (add -y to skip the question)

set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

say()  { printf '\n\033[1m%s\033[0m\n' "$*"; }
info() { printf '    %s\n' "$*"; }

say "This removes the Chess Review setup from:"
info "$ROOT"
info "It keeps Python, uv, and anything outside this folder."

if [ "${1:-}" != "-y" ] && [ "${1:-}" != "--yes" ]; then
  printf '\nType y to continue: '
  read -r ans || ans=""
  case "$ans" in
    y | Y | yes | YES) ;;
    *) echo "Cancelled. Nothing was removed."; exit 0 ;;
  esac
fi

remove() { if [ -e "$1" ]; then rm -rf "$1"; info "removed $1"; fi; }

say "Removing setup ..."
remove .venv
remove .venv_maia
remove engines
remove data/processed/maia_val/weights
remove data/processed/app_cache.sqlite
remove frontend/.next
# Only remove the built interface if we built it here (node_modules is present).
# A prebuilt interface that shipped with the app is left in place.
if [ -d frontend/node_modules ]; then
  remove frontend/out
  remove frontend/node_modules
fi
# Tidy up folders that are now empty, but leave anything else you put in data/.
find data -type d -empty -delete 2>/dev/null || true
rmdir data 2>/dev/null || true

say "Done. The app setup has been removed."
info "Python and uv were left in place."
info "To set it up again later, run:  bash install/install.sh"
info "If you also want to remove uv itself (only if nothing else uses it):"
info "  rm -f \"\$HOME/.local/bin/uv\" && rm -rf \"\$HOME/.local/share/uv\" \"\$HOME/.cache/uv\""
