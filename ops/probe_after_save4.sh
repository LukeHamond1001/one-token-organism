#!/bin/zsh
# THE PROBE AFTER A SAVE: wait until the page log holds at least N save rows, then read the saved body's language-model accuracy
#   probe_after_save.sh SCRATCH N LOGNAME
cd /Users/lukehamond/Projects/project
S=$1; N=$2; OUT=$S/logs/$3
until [ "$(grep -c '"action": "save"' data/watch2_caregiver.jsonl)" -ge "$N" ]; do python3 -c "import time; time.sleep(20)"; done
python3 -c "import time; time.sleep(40)"
cp data/watch2.pt $S/post_night_$N.pt; ls -t $S/post_night_*.pt 2>/dev/null | tail -n +3 | xargs rm -f   # the morning save kept two deep (the dusk probe overwrites the served file by day)
{ date; echo "=== the probe after save $N (nights: $(grep -c '"action": "night"' data/watch2_caregiver.jsonl) rows) ==="; nice -n 19 python3 tools/probe_lm.py data/watch2.pt --lmloss --qa --follow=20 2>&1 | grep -v 'unknown keys'; echo '=== rephrased'; nice -n 19 python3 tools/qa_by_gap.py data/watch2.pt --follow=20 --gaps=2 --set=rephrased 2>&1 | grep -v 'unknown keys' | grep pause; true 2>&1 | grep -v "unknown keys"; } > $OUT 2>&1
