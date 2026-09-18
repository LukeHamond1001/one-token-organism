#!/bin/zsh
# A RELOAD NOW (2026-09-17): the served body asked to save, then restarted with the given flags (the code on disk), the typist stopped so
# the chain relaunches it; the day's learning is in the save. For a change that cannot wait for the night's save (the user's word).
#   reload_now.sh SCRATCH "FLAGS"
cd /Users/lukehamond/Projects/project
N=$1; FLAGS=$2; L=data/watch2_caregiver.jsonl
M0=$(stat -f %m data/watch2.pt); curl -s -X POST localhost:8020/save -H 'Content-Type: application/json' -d '{}' > /dev/null
until [ "$(stat -f %m data/watch2.pt)" -gt "$M0" ]; do python3 -c "import time; time.sleep(2)"; done; python3 -c "import time; time.sleep(5)"
P=$(pgrep -f "body.serve --load data/watch[2].pt" | head -1)
mkdir -p data/backups/watch2; cp data/watch2.pt "data/backups/watch2/watch2_before_reload_night$(grep -c '"action": "night"' $L).pt"
ls -t data/backups/watch2/watch2_before_reload_*.pt 2>/dev/null | tail -n +3 | xargs rm -f
echo "reload now at $(date +%H:%M:%S): the body saved by request, server $P stopped; flags [$FLAGS]" >> $N/logs/watch2.log
kill $P; python3 -c "import time; time.sleep(5)"
env nohup python3 -m body.serve --load data/watch2.pt --tok data/tok_char.json --port 8020 ${=FLAGS} > $N/logs/serve_watch2_reload.log 2>&1 &
python3 -c "import time; time.sleep(50)"
T=$(pgrep -f "body.teacher --port 8020" | head -1); [ -n "$T" ] && kill $T
echo "served again (pid $(pgrep -f 'body.serve --load data/watch[2].pt' | head -1)) at $(date +%H:%M:%S) with [$FLAGS]; typist $T stopped for the chain" >> $N/logs/watch2.log
