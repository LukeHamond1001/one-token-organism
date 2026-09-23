#!/bin/zsh
# A SERVED BODY'S BOUNDARY, race-free: wait for a go-file; then take the day the log is on and wait for THAT day's night
# row; then stop the typist and the server, back the body up, restart the server with new physiology flags, and restart
# the typist for the next day. (The go must come before the night: a go given after a night waits for the next one.)
#   tools/boundary_restart.sh PORT BODY.pt LOG.jsonl GOFILE "SERVE FLAGS" "TYPIST ARGS" "TYPIST ENV"
# The planner for the new day is spawned by hand when the new session starts (a script cannot spawn agents).
set -u
PORT=$1; BODY=$2; LOG=$3; GO=$4; SFLAGS=$5; TARGS=$6; TENV=${7:-}
cd /Users/lukehamond/Projects/project
until [ -f "$GO" ]; do sleep 20; done
DAY=$(tail -300 "$LOG" | grep -o '"day": [0-9]*' | tail -1 | grep -o '[0-9]*')
echo "$(date +%H:%M:%S) go given on day $DAY; waiting for its night"
until grep -q "\"action\": \"night\".*\"day\": $DAY," "$LOG" 2>/dev/null; do sleep 15; done
echo "$(date +%H:%M:%S) night $DAY seen"
pkill -f "body.teacher --port $PORT" ; sleep 3
pkill -f "body.serve.*--port $PORT" ; sleep 8
mkdir -p "data/backups/$(basename "${BODY%.pt}")"; cp "$BODY" "data/backups/$(basename "${BODY%.pt}")/$(basename "${BODY%.pt}")_before_boundary_day$DAY.pt"
nohup python3 -m body.serve --load "$BODY" --tok data/tok_char.json --port $PORT ${=SFLAGS} > "${GO}.serve.log" 2>&1 &
sleep 40
env ${=TENV} nohup python3 -m body.teacher --port $PORT --day $((DAY+1)) --log "$LOG" ${=TARGS} > "${GO}.teach.log" 2>&1 &
sleep 5; echo "$(date +%H:%M:%S) restarted for day $((DAY+1)): serve flags [$SFLAGS] typist [$TARGS] env [$TENV]"; rm -f "$GO"
