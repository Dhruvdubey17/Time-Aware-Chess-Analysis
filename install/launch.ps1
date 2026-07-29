# Start the Chess Review app on Windows and open it in your browser.
#
# Everything runs on your machine. Nothing is sent anywhere. Close this window
# or press Ctrl+C to stop the app.
#
# If Windows blocks the script, run it like this:
#   powershell -ExecutionPolicy Bypass -File install\launch.ps1

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $Root

if (-not (Test-Path .venv\Scripts\python.exe)) { Write-Host "Please run the setup first:  powershell -ExecutionPolicy Bypass -File install\install.ps1"; exit 1 }
if (-not (Test-Path engines\STOCKFISH_PATH))   { Write-Host "Setup looks incomplete. Run install\install.ps1"; exit 1 }

$env:STOCKFISH_PATH = (Get-Content engines\STOCKFISH_PATH -Raw).Trim()

# Find a free local port starting at 8000.
$port = 8000
while ($port -le 8020) {
  $inUse = Get-NetTCPConnection -State Listen -LocalPort $port -ErrorAction SilentlyContinue
  if (-not $inUse) { break }
  $port++
}
if ($port -gt 8020) { Write-Host "No free port found between 8000 and 8020."; exit 1 }
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
