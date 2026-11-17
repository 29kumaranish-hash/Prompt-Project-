$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $projectRoot
$env:HF_HOME = Join-Path $projectRoot 'storage/model-cache'
$env:TIKTOKEN_CACHE_DIR = Join-Path $projectRoot 'storage/tiktoken'
& "$projectRoot/.venv/Scripts/python.exe" -m uvicorn app.main:app --app-dir backend --host 127.0.0.1 --port 8000 --workers 1
