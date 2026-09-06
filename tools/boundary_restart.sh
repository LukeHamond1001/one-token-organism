#!/bin/zsh
# A SERVED BODY'S BOUNDARY, race-free: wait for the night row of DAY in the page log AND a go-file, then stop the typist,
# stop the server, back the body up, restart the server with new physiology flags, restart the typist for DAY+1.
#   tools/boundary_restart.sh PORT BODY.pt LOG.jsonl DAY GOFILE "SERVE FLAGS" "TYPIST ARGS" "TYPIST ENV"
# The planner for DAY+1 is spawned by hand when the new session starts (the script cannot spawn agents).
set -u
PORT=$1; BODY=$2; LOG=$3; DAY=$4; GO=$5; SFLAGS=$6; TARGS=$7; TENV=${8:-}
cd /Users/lukehamond/Projects/project
until grep -q "\"action\": \"night\".*\"day\": $DAY," "$LOG" 2>/dev/null && [ -f "$GO" ]; do sleep 20; done
echo "$(date +%H:%M:%S) boundary $DAY: night seen and go given"
pkill -f "body.teacher --port $PORT" ; sleep 3
pkill -f "body.serve.*--port $PORT" ; sleep 8
cp "$BODY" "${BODY%.pt}_before_boundary_day$DAY.pt"
nohup python3 -m body.serve --load "$BODY" --tok data/tok_char.json --port $PORT ${=SFLAGS} > "${GO}.serve.log" 2>&1 &
sleep 40
env ${=TENV} nohup python3 -m body.teacher --port $PORT --day $((DAY+1)) --log "$LOG" ${=TARGS} > "${GO}.teach.log" 2>&1 &
sleep 5; echo "$(date +%H:%M:%S) restarted: serve flags [$SFLAGS] typist [$TARGS] env [$TENV]"; rm -f "$GO"
