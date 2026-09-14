#!/bin/zsh
# THE DUSK PROBE (2026-09-13): when the served body's sleep pressure nears its threshold, ask it to save (a five-second pause of the
# tick loop), copy the save aside, and read the language-model accuracy of the body AS THE DAY LEAVES IT, before the night's lesson.
# With the probe after the night's save, the night's effect and the day's are told apart.   probe_at_dusk.sh SCRATCH LOGNAME
cd /Users/lukehamond/Projects/project
S=$1; OUT=$S/logs/$2
until python3 - <<'PY'
import json, urllib.request, sys
try:
    d = json.load(urllib.request.urlopen("http://localhost:8020/insides", timeout=5))
    sys.exit(0 if (not d.get("asleep") and d["sleep_pressure"] >= d["wake_ticks"] - 700) else 1)
except Exception:
    sys.exit(1)
PY
do python3 -c "import time; time.sleep(20)"; done
curl -s -X POST localhost:8020/save -H 'Content-Type: application/json' -d '{}' > /dev/null
python3 -c "import time; time.sleep(5)"
cp data/watch2.pt $S/dusk_copy.pt; cp data/watch2.pt "$S/dusk_copy_$(grep -c '"action": "night"' data/watch2_caregiver.jsonl).pt"; ls -t $S/dusk_copy_*.pt | tail -n +3 | xargs rm -f
{ date; echo "=== the dusk probe (sleep pressure near the threshold; nights so far $(grep -c '"action": "night"' data/watch2_caregiver.jsonl) rows) ==="; nice -n 19 python3 tools/probe_lm.py $S/dusk_copy.pt --lmloss 2>&1 | grep -v "unknown keys"; } > $OUT 2>&1
