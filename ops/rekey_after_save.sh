#!/bin/zsh
# THE STORE REBUILT AT A SAVE (2026-09-18, item 40's reconsolidation): after the next post-night save row, the typist is held
# (SIGSTOP), the served body is stopped, its save is backed up and its store rebuilt from the utterance memory by tools/rekey_store.py
# under the given flags (the code's key), the body is served again on the rebuilt save with those flags, and the typist is resumed
# and stopped so the chain relaunches it with the server already up (the chain stops for good if it wakes to no server).
#   rekey_after_save.sh SCRATCH FLAGS_FILE
cd /Users/lukehamond/Projects/project
S=$1; FLAGS="$(cat $2)"; L=data/watch2_caregiver.jsonl
N0=$(grep -c '"action": "save"' $L)
until [ "$(grep -c '"action": "save"' $L)" -gt "$N0" ]; do python3 -c "import time; time.sleep(15)"; done
python3 -c "import time; time.sleep(10)"
T=$(pgrep -f "body.teacher --port 8020" | head -1); [ -n "$T" ] && kill -STOP $T
P=$(pgrep -f "body.serve --load data/watch[2].pt" | head -1)
mkdir -p data/backups/watch2; cp data/watch2.pt "data/backups/watch2/watch2_before_rekey_night$(grep -c '"action": "night"' $L).pt"
ls -t data/backups/watch2/watch2_before_*.pt 2>/dev/null | tail -n +3 | xargs rm -f
echo "rekey at $(date +%H:%M:%S): server $P stopped after the post-night save; typist $T held; flags [$FLAGS]" >> $S/logs/watch2.log
kill $P; python3 -c "import time; time.sleep(5)"
nice -n 5 python3 tools/rekey_store.py data/watch2.pt --flags $2 --save-as data/watch2_rekeyed.pt > $S/logs/rekey_served.log 2>&1
if grep -q "the rekeyed copy saved as" $S/logs/rekey_served.log; then
  mv data/watch2_rekeyed.pt data/watch2.pt; echo "rekey done at $(date +%H:%M:%S): $(grep REKEYED $S/logs/rekey_served.log | cut -c1-160)" >> $S/logs/watch2.log
else
  echo "REKEY FAILED at $(date +%H:%M:%S); the save served as it was" >> $S/logs/watch2.log
fi
env nohup python3 -m body.serve --load data/watch2.pt --tok data/tok_char.json --port 8020 ${=FLAGS} > $S/logs/serve_watch2_rekey.log 2>&1 &
python3 -c "import time; time.sleep(50)"
[ -n "$T" ] && kill -CONT $T && python3 -c "import time; time.sleep(2)" && kill $T
echo "served again (pid $(pgrep -f 'body.serve --load data/watch[2].pt' | head -1)) at $(date +%H:%M:%S) on the rebuilt store; typist $T stopped for the chain" >> $S/logs/watch2.log
