#!/bin/bash
# Run the AskChem web server
#
# Usage:
#   ./run_server.sh              # default: port 8420
#   ./run_server.sh 8080         # custom port
#   ASKCHEM_DB=/path/to/askchem.db ./run_server.sh

set -e

PORT=${1:-8420}
HOST=${2:-0.0.0.0}

cd "$(dirname "$0")"

# The public corpus is distributed separately because it is too large for Git.
DB_PATH=${ASKCHEM_DB:-askchem.db}
if [ ! -f "$DB_PATH" ]; then
    echo "Database not found: $DB_PATH" >&2
    echo "Download askchem.db from https://huggingface.co/datasets/bing-yan/askchem" >&2
    exit 1
fi
export ASKCHEM_DB="$DB_PATH"

echo "Starting AskChem server on http://${HOST}:${PORT}"
echo "  API docs: http://${HOST}:${PORT}/api/docs"
echo "  Frontend: http://${HOST}:${PORT}/"

export PYTHONPATH="${PYTHONPATH:+$PYTHONPATH:}src"
exec uvicorn askchem.server:app --host "$HOST" --port "$PORT" --workers 1
