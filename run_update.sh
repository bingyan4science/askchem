#!/bin/bash
cd /private/home/bingyan/structure_the_universe
export PYTHONPATH=src
echo "Starting update at $(date)" >> update_run.log
python3 src/update_index.py --days 1 >> update_run.log 2>&1
echo "Finished at $(date), EXIT CODE: $?" >> update_run.log
