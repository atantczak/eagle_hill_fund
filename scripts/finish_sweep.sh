#!/bin/bash
# Wait for every grand-sweep queue to finish, retry failed studies once, rebuild the
# leaderboard, copy one run + scorecard per study into the main results, and push
# var/ to S3 so the website's lab shows them. Usage: scripts/finish_sweep.sh TAG
set -u
cd "$(dirname "$0")/.."
TAG=$1
PY=.venv311/bin/python
LOG=var/log/finish_sweep_${TAG}.out
echo "$(date) waiting for sweep queues" >> "$LOG"
while pgrep -f "jobs.grand_sweep --tag $TAG --shard|build_alt_datasets|run_sweep_shards|run_alt_after_build" > /dev/null; do sleep 60; done
echo "$(date) retrying failures" >> "$LOG"
$PY -m ehf.research.jobs.grand_sweep --tag "$TAG" --retry-failures --shard-base 90 --workers 28 \
  --families basket,dial,alt > "var/log/grand_sweep_${TAG}_retry.out" 2>&1
$PY -m ehf.research.jobs.grand_sweep --tag "$TAG" --leaderboard --families basket,dial,alt >> "$LOG" 2>&1
echo "$(date) exporting" >> "$LOG"
$PY -m ehf.research.jobs.grand_sweep --tag "$TAG" --export >> "$LOG" 2>&1
echo "$(date) syncing var/ to S3" >> "$LOG"
$PY -m ehf.research.jobs.sync_var >> "$LOG" 2>&1
echo "$(date) FINISHED exit=$?" >> "$LOG"
