#!/bin/zsh
# A RELOAD AFTER THE POST-NIGHT SAVE (2026-09-11; rewritten 2026-09-19 after the review): the typist's "save" row comes after the
# night; the typist is stopped AT THAT ROW (its post-save watch has not ended, so no next-day session begins and no line is popped:
# the old order, server first and the typist fifty seconds later, made phantom days and lost a line), the chain is held while the
# served body restarts on the given flags (the code on disk), the page's face is cleared, and the chain is released to relaunch the
# typist for the next day once the server answers. Nothing of the day or the night is lost: the save is written before any of it.
#   reload_after_save.sh SCRATCH "FLAGS"
cd /Users/lukehamond/Projects/project
N=$1; FLAGS=$2; L=data/watch2_caregiver.jsonl
N0=$(grep -c '"action": "save"' $L)
until [ "$(grep -c '"action": "save"' $L)" -gt "$N0" ]; do python3 -c "import time; time.sleep(10)"; done
python3 -c "import time; time.sleep(3)"
T=$(ps -eo pid,args | grep "body.teacher" | grep -v grep | awk '{print $1}' | head -1); [ -n "$T" ] && kill $T           # at the save row, before its next day
C=$(ps -eo pid,args | grep "tools/typist_chain.sh" | grep -v grep | awk '{print $1}' | head -1); [ -n "$C" ] && kill -STOP $C   # the chain waits for the server
P=$(pgrep -f "body.serve --load data/watch[2].pt" | head -1)
mkdir -p data/backups/watch2; cp data/watch2.pt "data/backups/watch2/watch2_before_reload_night$(grep -c '"action": "night"' $L).pt"
ls -t data/backups/watch2/watch2_before_reload_*.pt 2>/dev/null | tail -n +3 | xargs rm -f   # the reload copies kept two deep (the disk filled on 2026-09-12)
echo "reload at $(date +%H:%M:%S): typist $T stopped at the save row, server $P stopped; flags [$FLAGS]" >> $N/logs/watch2.log
kill $P; python3 -c "import time; time.sleep(5)"
env nohup python3 -m body.serve --load data/watch2.pt --tok data/tok_char.json --port 8020 ${=FLAGS} > $N/logs/serve_watch2_reload.log 2>&1 &
n=0; until curl -s -m 3 localhost:8020/state >/dev/null 2>&1 || [ $n -ge 60 ]; do python3 -c "import time; time.sleep(2)"; n=$((n+1)); done
curl -s -m 3 -X POST localhost:8020/face -H "Content-Type: application/json" -d '{"expr": 0}' >/dev/null 2>&1                # a face held by the stopped typist is released
[ -n "$C" ] && kill -CONT $C                                                                                                  # the chain relaunches the typist for the next day
echo "served again (pid $(pgrep -f 'body.serve --load data/watch[2].pt' | head -1)) at $(date +%H:%M:%S) with [$FLAGS]; the chain released" >> $N/logs/watch2.log
