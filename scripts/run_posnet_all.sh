#!/usr/bin/env bash
# Run PosNet per cluster, one process at a time, each only when >= MIN_FREE % memory is free.
# Cluster I is skipped: ~58 GB estimated peak (A used 10 GB at 200x200 px; I is 480x480 px).
MIN_FREE="${MIN_FREE:-70}"
cd "$(dirname "$0")/.."
free_pct() { memory_pressure -Q | awk -F': ' '/free percentage/{gsub("%","",$2); print $2}'; }
for L in B C D E H F G; do
  until [ "$(free_pct)" -ge "$MIN_FREE" ]; do sleep 30; done
  echo "$(date +%T) start $L (free $(free_pct)%)"
  /usr/bin/time -l env PYTHONPATH=src ~/.venvs/afm-spinoff/bin/python scripts/run_posnet_baseline.py "$L" 2>&1 \
    | grep -E "pred atoms|Error|maximum resident|real"
done
echo "$(date +%T) DONE"
