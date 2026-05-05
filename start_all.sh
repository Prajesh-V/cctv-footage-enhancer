#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "$0")" && pwd)

echo "Starting ClarityAI Phase 2 (Video) stack: Backend + Frontend"

BACKEND_CMD="( cd clarityai/backend && uvicorn clarityai.backend.main:app --reload --host 0.0.0.0 --port 8000 )"
FRONTEND_CMD="( cd clarityai/frontend && npm run dev )"

echo "Launching backend..."
( cd clarityai/backend && uvicorn clarityai.backend.main:app --reload --host 0.0.0.0 --port 8000 ) &
BE_PID=$!
echo "Backend PID: ${BE_PID}"

echo "Launching frontend..."
( cd clarityai/frontend && npm run dev ) &
FE_PID=$!
echo "Frontend PID: ${FE_PID}"

cleanup() {
  echo "Shutting down..."
  if [[ -n "${BE_PID:-}" ]]; then
    kill -TERM "$BE_PID" 2>/dev/null || true
  fi
  if [[ -n "${FE_PID:-}" ]]; then
    kill -TERM "$FE_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT

echo "Ready. Backend=$BE_PID Frontend=$FE_PID"
echo "Tip: open http://localhost:8000 (backend) and http://localhost:3000 (frontend)"
wait "$BE_PID" 2>/dev/null || true
wait "$FE_PID" 2>/dev/null || true
