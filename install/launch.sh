#!/usr/bin/env bash
# Start the Chess Review app and open it in your browser.
#
# Everything runs on your machine. Nothing is sent anywhere. Press Ctrl+C in this
# window to stop the app.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

[ -x .venv/bin/python ] || { echo "Please run the setup first:  bash install/install.sh" >&2; exit 1; }
[ -f engines/STOCKFISH_PATH ] || { echo "Setup looks incomplete. Run:  bash install/install.sh" >&2; exit 1; }

export STOCKFISH_PATH="$(cat engines/STOCKFISH_PATH)"

# Find a free local port in 8000..8020. We use the app's own Python (already
# required just below) to test-bind, so this needs no extra tool like lsof and
# behaves the same on macOS and Linux.
port="$(.venv/bin/python - <<'PY'
import socket
for p in range(8000, 8021):
    s = socket.socket()
    try:
        s.bind(("127.0.0.1", p)); s.close(); print(p); break
    except OSError:
        continue
PY
)"
[ -n "$port" ] || { echo "No free port found between 8000 and 8020." >&2; exit 1; }
url="http://127.0.0.1:$port"

echo "Starting Chess Review on $url ..."
.venv/bin/python -m uvicorn backend.api:app --host 127.0.0.1 --port "$port" --log-level warning &
server=$!
trap 'kill "$server" 2>/dev/null || true' INT TERM EXIT

# Wait for the server to answer, then open the browser.
for _ in $(seq 1 40); do
  if curl -s -m 1 "$url/api/health" >/dev/null 2>&1; then break; fi
  sleep 0.25
done
case "$(uname -s)" in
  Darwin) open "$url" 2>/dev/null || true ;;
  Linux)  xdg-open "$url" 2>/dev/null || true ;;
esac

echo "Chess Review is running. Open $url in your browser if it did not open."
echo "Press Ctrl+C here to stop."
wait "$server"
