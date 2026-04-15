#!/usr/bin/env bash
set -euo pipefail

export PYTHONPATH="/workspace:${PYTHONPATH:-}"

exec uvicorn app.main:app \
  --host "${HOST:-0.0.0.0}" \
  --port "${PORT:-8888}" \
  --log-level "${LOG_LEVEL:-info}"
