$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $projectRoot
$env:OLLAMA_MODELS = Join-Path $projectRoot 'storage/ollama-models'
$env:OLLAMA_HOST = '127.0.0.1:11434'
$env:OLLAMA_NO_CLOUD = 'true'
if (Test-Path -LiteralPath "$projectRoot/storage/ollama/ollama.exe") {
    & "$projectRoot/storage/ollama/ollama.exe" serve
} else {
    ollama serve
}
