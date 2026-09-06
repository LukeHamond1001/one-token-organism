#!/bin/zsh
# THE TYPIST'S CHAIN: when a served body's typist exits (its --days run out), start the next one at the log's last day + 1.
#   tools/typist_chain.sh PORT LOG.jsonl "TYPIST ARGS (without --port/--day/--log)" "ENV" [ROUNDS]
set -u
PORT=$1; LOG=$2; TARGS=$3; TENV=${4:-}; ROUNDS=${5:-3}
cd /Users/lukehamond/Projects/project
for i in $(seq 1 $ROUNDS); do
  while pgrep -f "body.teacher --port $PORT" >/dev/null; do sleep 60; done
  if ! pgrep -f "body.serve.*--port $PORT" >/dev/null; then echo "$(date +%H:%M:%S) no server on $PORT; the chain stops"; exit 1; fi
  DAY=$(tail -300 "$LOG" | grep -o '"day": [0-9]*' | tail -1 | grep -o '[0-9]*'); NEXT=$((DAY+1))
  env ${=TENV} nohup python3 -m body.teacher --port $PORT --day $NEXT --log "$LOG" ${=TARGS} > "/tmp/typist_${PORT}_day${NEXT}.log" 2>&1 &
  sleep 20; echo "$(date +%H:%M:%S) typist on $PORT relaunched for day $NEXT [$TARGS] env [$TENV]"
done
