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

say "Interface built."
info "Prebuilt files are in frontend/out. Include that folder when you package the app."
info "Next: follow install/RELEASE.md to package and share the release."
