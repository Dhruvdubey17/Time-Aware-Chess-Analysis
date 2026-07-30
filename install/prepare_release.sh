#!/usr/bin/env bash
# Release build step. Run this on YOUR machine before you package the app to
# share it. It builds the web interface into static files (frontend/out) so the
# people you share the app with never need Node.js and never run a build.
#
# This is the ONLY step that needs Node.js, and only you run it, once per release.
# The static files it produces work the same on macOS, Windows, and Linux, so one
# build serves every user. See install/RELEASE.md for how to package afterwards.
#
# Run from the project root:  bash install/prepare_release.sh

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

say()  { printf '\n\033[1m==> %s\033[0m\n' "$*"; }
info() { printf '    %s\n' "$*"; }
die()  { printf '\n\033[1;31mRelease build stopped: %s\033[0m\n' "$*" >&2; exit 1; }

command -v npm >/dev/null 2>&1 || die "Node.js/npm not found. Install Node.js 18+ to build the interface. This is only needed here, on the release machine, not by the people who install the app."

say "Building the interface into static files (frontend/out)"
# Empty API base so the built app calls the backend on its own origin, which is
# how the launcher serves it. Without this the app would call a fixed dev port.
( cd frontend && npm ci && NEXT_PUBLIC_API_BASE="" npm run build ) \
  || die "the interface build failed. Fix the errors above and run this again."

[ -f frontend/out/index.html ] || die "the build finished but frontend/out/index.html is missing. The interface did not export as static files; check frontend/next.config.ts still has output: \"export\"."

say "Interface built. Packaging the release ..."

# We build the shareable archive here so you cannot package before the interface
# is built, and so it does not matter which folder you run this from (paths come
# from this script's own location, not your shell's working directory). The
# archive lands NEXT to the project folder, one level up, so it is never packed
# inside itself.
parent="$(dirname "$ROOT")"
name="$(basename "$ROOT")"
archive="$parent/chess-review.tar.gz"
rm -f "$archive"
# Include the prebuilt interface (frontend/out) and the bundled assets. Leave out
# the big folders the user's install rebuilds (the Python envs, the downloaded
# engine, the model weights, node_modules, build caches) and the private notes.
tar -czf "$archive" -C "$parent" \
  --exclude="$name/.git" \
  --exclude="$name/.venv" \
  --exclude="$name/.venv_maia" \
  --exclude="$name/data" \
  --exclude="$name/engines" \
  --exclude="$name/frontend/node_modules" \
  --exclude="$name/frontend/.next" \
  --exclude='*/__pycache__' \
  --exclude='*.pyc' \
  --exclude='*/.pytest_cache' \
  --exclude='*/.ruff_cache' \
  --exclude='.DS_Store' \
  --exclude="$name/BUILD.md" \
  --exclude="$name/BUILD2.md" \
  --exclude="$name/PLAN.md" \
  --exclude="$name/context.md" \
  --exclude='*chess-review.tar.gz' \
  "$name" \
  || die "could not create the release archive."

# Prove the prebuilt interface actually made it in, so a broken package never
# ships silently.
tar -tzf "$archive" | grep -q "$name/frontend/out/index.html" \
  || die "the archive is missing the prebuilt interface right after a build. Please report this."

say "Release ready."
info "Share this one file:"
info "  $archive"
info "Whoever you send it to unpacks it and runs install/install.sh (macOS or"
info "Linux) or install/install.ps1 (Windows). They never need Node.js."
