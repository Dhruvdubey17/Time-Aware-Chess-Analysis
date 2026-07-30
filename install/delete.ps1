# Remove everything the setup created for the Chess Review app on Windows.
#
# This only removes things inside this app folder that the setup made: the two
# Python environments, the chess engine, the Maia model weights, the cached
# analysis, and the built interface. It does NOT touch tools that live outside
# this folder or were already on your machine, like Python or uv.
#
# Run it with:
#   powershell -ExecutionPolicy Bypass -File install\delete.ps1   (add -y to skip the question)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $Root

Write-Host "`nThis removes the Chess Review setup from:" -ForegroundColor Cyan
Write-Host "    $Root"
Write-Host "    It keeps Python, uv, and anything outside this folder."

if ($args[0] -ne "-y" -and $args[0] -ne "--yes") {
  $ans = Read-Host "`nType y to continue"
  if ($ans -notmatch '^(y|yes)$') { Write-Host "Cancelled. Nothing was removed."; exit 0 }
}

function Remove-IfExists($p) {
  if (Test-Path $p) { Remove-Item -Recurse -Force $p; Write-Host "    removed $p" }
}

Write-Host "`nRemoving setup ..." -ForegroundColor Cyan
Remove-IfExists .venv
Remove-IfExists .venv_maia
Remove-IfExists engines
Remove-IfExists data\processed\maia_val\weights
Remove-IfExists data\processed\app_cache.sqlite
Remove-IfExists frontend\.next
if (Test-Path frontend\node_modules) {
  Remove-IfExists frontend\out
  Remove-IfExists frontend\node_modules
}

Write-Host "`nDone. The app setup has been removed." -ForegroundColor Cyan
Write-Host "    Python and uv were left in place."
Write-Host "    To set it up again later, run:  powershell -ExecutionPolicy Bypass -File install\install.ps1"
