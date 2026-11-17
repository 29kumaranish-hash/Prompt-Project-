#!/usr/bin/env sh
set -eu
cd "$(dirname "$0")/.."
export HF_HOME="$PWD/storage/model-cache"
export TIKTOKEN_CACHE_DIR="$PWD/storage/tiktoken"
exec .venv/bin/python -m uvicorn app.main:app --app-dir backend --host 127.0.0.1 --port 8000 --workers 1
