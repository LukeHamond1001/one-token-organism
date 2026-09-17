#!/bin/zsh
# A TYPIST RELAUNCH AFTER THE POST-NIGHT SAVE (2026-09-12): the typist's "save" row comes after the night and its battery of cues;
# then the typist is stopped so the chain (already re-chained with the new arguments) relaunches it for the next day. The body is
# not touched.   typist_relaunch_after_save.sh SCRATCH
cd /Users/lukehamond/Projects/project
N=$1; L=data/watch2_caregiver.jsonl
N0=$(grep -c '"action": "save"' $L)
until [ "$(grep -c '"action": "save"' $L)" -gt "$N0" ]; do python3 -c "import time; time.sleep(10)"; done
python3 -c "import time; time.sleep(5)"
T=$(pgrep -f "body.teacher --port 8020" | head -1); [ -n "$T" ] && kill $T
echo "typist $T stopped at $(date +%H:%M:%S) after the post-night save for the chain's relaunch with the new arguments" >> $N/logs/watch2.log
