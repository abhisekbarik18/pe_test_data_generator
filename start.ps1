$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $projectRoot

$venvPath = Join-Path $projectRoot ".venv"

if (-not (Test-Path $venvPath)) {
    Write-Host "Creating virtual environment..."
    python -m venv $venvPath
}

Write-Host "Activating virtual environment..."
& (Join-Path $venvPath "Scripts\Activate.ps1")

Write-Host "Installing dependencies..."
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

Write-Host "Starting Streamlit app..."
$startUrl = "http://localhost:8501"
$browser = "chrome"

Start-Process $browser $startUrl

streamlit run app.py --server.headless false --server.runOnSave false
