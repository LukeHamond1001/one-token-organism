#!/bin/zsh
# A RELOAD AFTER THE POST-NIGHT SAVE (2026-09-11): the typist's "save" row comes after the night and its battery of cues; then the
# served body is restarted with the given flags (the code on disk), and the typist is stopped so the chain relaunches it with a fresh
# cursor for the next day. Nothing of the day or the night is lost: the battery is done and the save is written.
#   reload_after_save.sh SCRATCH "FLAGS"
cd /Users/lukehamond/Projects/project
N=$1; FLAGS=$2; L=data/watch2_caregiver.jsonl
N0=$(grep -c '"action": "save"' $L)
until [ "$(grep -c '"action": "save"' $L)" -gt "$N0" ]; do python3 -c "import time; time.sleep(15)"; done
python3 -c "import time; time.sleep(10)"
P=$(pgrep -f "body.serve --load data/watch[2].pt" | head -1)
mkdir -p data/backups/watch2; cp data/watch2.pt "data/backups/watch2/watch2_before_reload_night$(grep -c '"action": "night"' $L).pt"
ls -t data/backups/watch2/watch2_before_reload_*.pt 2>/dev/null | tail -n +3 | xargs rm -f   # the reload copies kept two deep (the disk filled on 2026-09-12)
echo "reload at $(date +%H:%M:%S): server $P stopped after the post-night save; flags [$FLAGS]" >> $N/logs/watch2.log
kill $P; python3 -c "import time; time.sleep(5)"
env nohup python3 -m body.serve --load data/watch2.pt --tok data/tok_char.json --port 8020 ${=FLAGS} > $N/logs/serve_watch2_reload.log 2>&1 &
python3 -c "import time; time.sleep(50)"
T=$(pgrep -f "body.teacher --port 8020" | head -1); [ -n "$T" ] && kill $T      # the chain relaunches it with a fresh cursor
echo "served again (pid $(pgrep -f 'body.serve --load data/watch[2].pt' | head -1)) at $(date +%H:%M:%S) with [$FLAGS]; typist $T stopped for the chain" >> $N/logs/watch2.log
