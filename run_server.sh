#!/bin/bash
# Run the AskChem web server
#
# Usage:
#   ./run_server.sh              # default: port 8420
#   ./run_server.sh --port 8080  # custom port
#   ASKCHEM_DB=/path/to/askchem.db ./run_server.sh

set -e

PORT=${1:-8420}
HOST=${2:-0.0.0.0}

cd "$(dirname "$0")"

# Build database if it doesn't exist (canonical askchem.db, legacy chemtree.db)
if [ ! -f askchem.db ] && [ ! -f chemtree.db ]; then
    echo "Building SQLite database from chemtree_index/..."
    python3 -m src.askchem.db build chemtree_index/
fi

echo "Starting AskChem server on http://${HOST}:${PORT}"
echo "  API docs: http://${HOST}:${PORT}/api/docs"
echo "  Frontend: http://${HOST}:${PORT}/"

export PYTHONPATH="${PYTHONPATH:+$PYTHONPATH:}src"
exec uvicorn askchem.server:app --host "$HOST" --port "$PORT" --workers 1
