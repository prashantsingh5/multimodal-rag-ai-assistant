# setup.ps1 — Windows PowerShell bootstrap for Multimodal RAG AI Assistant
# Run once: .\setup.ps1
# After that, always activate with: .\venv\Scripts\Activate.ps1

$ErrorActionPreference = "Stop"
$ROOT = Split-Path -Parent $MyInvocation.MyCommand.Path

Write-Host "`n=== Multimodal RAG AI Assistant — Environment Setup ===" -ForegroundColor Cyan

# --- 1. Create venv if it doesn't exist ---
$venvPath = Join-Path $ROOT "venv"
if (-Not (Test-Path $venvPath)) {
    Write-Host "`n[1/4] Creating virtual environment..." -ForegroundColor Yellow
    python -m venv $venvPath
    Write-Host "      venv created at: $venvPath" -ForegroundColor Green
} else {
    Write-Host "`n[1/4] Virtual environment already exists — skipping creation." -ForegroundColor Green
}

# --- 2. Activate venv ---
Write-Host "`n[2/4] Activating virtual environment..." -ForegroundColor Yellow
$activateScript = Join-Path $venvPath "Scripts\Activate.ps1"
& $activateScript

# --- 3. Upgrade pip and install dependencies ---
Write-Host "`n[3/4] Installing dependencies from requirements.txt..." -ForegroundColor Yellow
$pip = Join-Path $venvPath "Scripts\pip.exe"
& $pip install --upgrade pip
& $pip install -r (Join-Path $ROOT "requirements.txt")

# --- 4. Create assets directory & .env if missing ---
Write-Host "`n[4/4] Creating runtime directories and .env file..." -ForegroundColor Yellow
$assetsDir = Join-Path $ROOT "assets\vectorstore"
if (-Not (Test-Path $assetsDir)) {
    New-Item -ItemType Directory -Force -Path $assetsDir | Out-Null
}

$envFile = Join-Path $ROOT ".env"
$envExample = Join-Path $ROOT ".env.example"
if (-Not (Test-Path $envFile)) {
    Copy-Item $envExample $envFile
    Write-Host "      .env created from .env.example — please fill in your API keys." -ForegroundColor Magenta
}

Write-Host "`n=== Setup complete! ===" -ForegroundColor Cyan
Write-Host "To start the API server:      python -m uvicorn api.server:app --reload" -ForegroundColor White
Write-Host "To start the Gradio frontend: python frontend/app.py" -ForegroundColor White
Write-Host "Remember to activate venv:    .\venv\Scripts\Activate.ps1`n" -ForegroundColor White
