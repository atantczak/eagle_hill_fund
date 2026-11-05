#!/bin/bash
# Wait for the alt-data dataset build, then run the alt family as extra sweep queues.
cd "$(dirname "$0")/.."
while pgrep -f build_alt_datasets > /dev/null; do sleep 20; done
LOGSUFFIX=_alt scripts/run_sweep_shards.sh "$1" 4 6 --families alt --shard-base 20 --hours 8
