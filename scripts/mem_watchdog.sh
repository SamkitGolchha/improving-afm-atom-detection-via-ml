#!/usr/bin/env bash
# Memory watchdog for the overnight pilot. Every 10 s logs free memory % and swap; if free memory
# drops below KILL_PCT, sends SIGTERM to the largest project Python process (by RSS), then SIGKILL
# 20 s later if it is still alive. Only touches processes whose command line contains the project
# venv path, so nothing else on the machine is affected.
# Usage: scripts/mem_watchdog.sh [KILL_PCT=12] [WARN_PCT=25]
KILL_PCT="${1:-12}"; WARN_PCT="${2:-25}"
LOG="$(cd "$(dirname "$0")/.." && pwd)/results/pilot/logs/mem_watchdog.log"
mkdir -p "$(dirname "$LOG")"
while true; do
  free=$(memory_pressure -Q | awk -F': ' '/free percentage/{gsub("%","",$2); print $2}')
  swap=$(sysctl -n vm.swapusage | awk '{print $6}')
  ts=$(date '+%H:%M:%S')
  top=$(ps -axo pid=,rss=,command= | grep "afm-spinoff/bin/python" | grep -v grep | sort -k2 -nr | head -1)
  echo "$ts free=${free}% swap_used=$swap top=[$(echo "$top" | awk '{printf "%s %.1fGB %s %s %s", $1, $2/1048576, $4, $5, $6}')]" >> "$LOG"
  if [ -n "$free" ] && [ "$free" -lt "$KILL_PCT" ] && [ -n "$top" ]; then
    pid=$(echo "$top" | awk '{print $1}')
    echo "$ts KILL: free ${free}% < ${KILL_PCT}%, SIGTERM pid $pid ($top)" >> "$LOG"
    kill -TERM "$pid"; sleep 20; kill -0 "$pid" 2>/dev/null && kill -KILL "$pid" && echo "$ts SIGKILL $pid" >> "$LOG"
  elif [ -n "$free" ] && [ "$free" -lt "$WARN_PCT" ]; then
    echo "$ts WARN: free ${free}%" >> "$LOG"
  fi
  sleep 10
done
