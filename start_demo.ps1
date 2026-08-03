# One-command demo launcher (Phase 8). Opens three separate, labeled PowerShell windows — one
# per service — so each one's logs stay visible during a live demo instead of being interleaved
# or hidden in a background job. Run from the repo root: .\start_demo.ps1
#
# Does NOT start MongoDB — it's expected to already be running as a Windows service (see
# docs/demo_script.md's pre-demo checklist).

$repoRoot = $PSScriptRoot

$mongoStatus = (Get-Service -Name MongoDB -ErrorAction SilentlyContinue).Status
if ($mongoStatus -ne "Running") {
    Write-Warning "MongoDB service is not running (status: $mongoStatus). Start it before the gateway will connect."
}

Write-Host "Starting orchestrator (FastAPI, port 8000)..."
Start-Process powershell -ArgumentList "-NoExit", "-Command", `
    "cd '$repoRoot'; ml/.venv/Scripts/python.exe -m uvicorn orchestrator.main:app --port 8000"

Write-Host "Starting gateway (Express, port 4000)..."
Start-Process powershell -ArgumentList "-NoExit", "-Command", `
    "cd '$repoRoot/gateway'; npm start"

Write-Host "Starting dashboard (Vite, port 5173)..."
Start-Process powershell -ArgumentList "-NoExit", "-Command", `
    "cd '$repoRoot/dashboard'; npm run dev"

Write-Host ""
Write-Host "All three services are starting in separate windows. Once ready:"
Write-Host "  Dashboard:    http://localhost:5173"
Write-Host "  Gateway:      http://localhost:4000/health"
Write-Host "  Orchestrator: http://localhost:8000/health"
Write-Host ""
Write-Host "Close each window individually to stop that service."
