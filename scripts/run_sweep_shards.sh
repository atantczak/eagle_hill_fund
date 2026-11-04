#!/bin/bash
# Run one grand-sweep tag as N side-by-side queues, keep the Mac awake, then rebuild
# the leaderboard. Usage: scripts/run_sweep_shards.sh TAG SHARDS WORKERS [extra args...]
# (pass --shard-base N in the extra args to run a second batch beside a running one)
set -u
cd "$(dirname "$0")/.."
TAG=$1; N=$2; W=$3; shift 3
PY=.venv311/bin/python
mkdir -p var/log
pids=()
for ((i=0; i<N; i++)); do
  $PY -m ehf.research.jobs.grand_sweep --tag "$TAG" --shard $i --shards $N --workers $W "$@" \
    > "var/log/grand_sweep_${TAG}_shard$i${LOGSUFFIX:-}.out" 2>&1 &
  pids+=($!)
done
caffeinate -i -s -w "${pids[0]}" &
for p in "${pids[@]}"; do wait "$p"; done
$PY -m ehf.research.jobs.grand_sweep --tag "$TAG" --leaderboard "$@" > /dev/null
echo "$(date) sweep $TAG done" >> var/log/grand_sweep_${TAG}.done
