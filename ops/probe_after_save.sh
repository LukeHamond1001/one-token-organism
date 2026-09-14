#!/bin/zsh
# THE PROBE AFTER A SAVE: wait until the page log holds at least N save rows, then read the saved body's language-model accuracy
#   probe_after_save.sh SCRATCH N LOGNAME
cd /Users/lukehamond/Projects/project
S=$1; N=$2; OUT=$S/logs/$3
until [ "$(grep -c '"action": "save"' data/watch2_caregiver.jsonl)" -ge "$N" ]; do python3 -c "import time; time.sleep(20)"; done
python3 -c "import time; time.sleep(40)"
{ date; echo "=== the probe after save $N (nights: $(grep -c '"action": "night"' data/watch2_caregiver.jsonl) rows) ==="; nice -n 19 python3 tools/probe_lm.py data/watch2.pt --lmloss --follow=20 2>&1 | grep -v "unknown keys"; } > $OUT 2>&1
