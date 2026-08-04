# Start the Chess Review app on Windows and open it in your browser.
#
# Everything runs on your machine. Nothing is sent anywhere. Close this window
# or press Ctrl+C to stop the app.
#
# If Windows blocks the script, run it like this:
#   powershell -ExecutionPolicy Bypass -File install\launch.ps1
#
# Optional: pass a chess.com username to open straight to that account's games:
#   powershell -ExecutionPolicy Bypass -File install\launch.ps1 magnuscarlsen

param([string]$User = "")

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $Root

if (-not (Test-Path .venv\Scripts\python.exe)) { Write-Host "Please run the setup first:  powershell -ExecutionPolicy Bypass -File install\install.ps1"; exit 1 }
if (-not (Test-Path engines\STOCKFISH_PATH))   { Write-Host "Setup looks incomplete. Run install\install.ps1"; exit 1 }

$env:STOCKFISH_PATH = (Get-Content engines\STOCKFISH_PATH -Raw).Trim()

# Optional chess.com username. When given, the app opens straight to that
# account's games and stays locked to it, refreshes included. The server passes
# it to the browser app through /api/health.
if ($User.Trim()) {
  $env:CHESS_REVIEW_USER = $User.Trim()
  Write-Host "Locked to chess.com user: $($User.Trim())"
}

# Find a free local port in 8000..8020. We use the app's own Python (required
# just above) to test-bind, so this matches the macOS/Linux launcher and does not
# depend on the NetTCP cmdlets being available.
$findPort = @"
import socket
for p in range(8000, 8021):
    s = socket.socket()
    try:
        s.bind(('127.0.0.1', p)); s.close(); print(p); break
    except OSError:
        continue
"@
$port = (& .venv\Scripts\python.exe -c $findPort).Trim()
if (-not $port) { Write-Host "No free port found between 8000 and 8020."; exit 1 }
$url = "http://127.0.0.1:$port"

Write-Host "Starting Chess Review on $url ..."
$server = Start-Process -PassThru -NoNewWindow .venv\Scripts\python.exe `
  "-m","uvicorn","backend.api:app","--host","127.0.0.1","--port","$port","--log-level","warning"

# Wait for the server to answer, then open the browser.
for ($i = 0; $i -lt 40; $i++) {
  try { Invoke-RestMethod "$url/api/health" -TimeoutSec 1 | Out-Null; break } catch { Start-Sleep -Milliseconds 250 }
}
Start-Process $url

Write-Host "Chess Review is running. Open $url in your browser if it did not open."
Write-Host "Close this window or press Ctrl+C to stop."
try { Wait-Process -Id $server.Id } finally { Stop-Process -Id $server.Id -ErrorAction SilentlyContinue }
