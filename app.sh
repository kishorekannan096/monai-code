#!/usr/bin/env bash
set -e
cd backend
exec uvicorn main:app --host 0.0.0.0 --port "${PORT:-8080}"
