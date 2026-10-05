$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $projectRoot

Write-Host "Stopping Streamlit process..."
$streamlitProcesses = Get-CimInstance Win32_Process -Filter "Name = 'streamlit.exe'" -ErrorAction SilentlyContinue
if ($streamlitProcesses) {
    foreach ($process in $streamlitProcesses) {
        $process | Invoke-CimMethod -MethodName Terminate | Out-Null
    }
    Write-Host "Streamlit stopped."
} else {
    Write-Host "No running Streamlit process found."
}

Write-Host "Cleaning generated project artifacts..."
$cleanupPaths = @(
    (Join-Path $projectRoot "logs"),
    (Join-Path $projectRoot "output"),
    (Join-Path $projectRoot "__pycache__"),
    (Join-Path $projectRoot ".streamlit")
)

foreach ($path in $cleanupPaths) {
    if (Test-Path $path) {
        Remove-Item $path -Recurse -Force -ErrorAction SilentlyContinue
        Write-Host "Removed: $path"
    }
}

Write-Host "Cleanup complete."
