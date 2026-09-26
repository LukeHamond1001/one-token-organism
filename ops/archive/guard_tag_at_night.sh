#!/bin/zsh
# THE GUARD ON THE LONG TAG: at the next night, read the day that just ended from the page log (the gate's duty after the parent's
# lines, smiles per line). If the duty fell under 0.2 or smiles per line under 0.6, restart with the tag closed (gate_slow_lr 0);
# otherwise touch nothing and let the five-night reading run.
cd /Users/lukehamond/Projects/project
N=$1; BASE=$2; L=data/watch2_caregiver.jsonl
N0=$(grep -c '"action": "night"' $L)
until [ "$(grep -c '"action": "night"' $L)" -gt "$N0" ]; do python3 -c "import time; time.sleep(15)"; done
cp data/watch2.pt "data/backups/watch2/pre_night_$(grep -c '"action": "night"' $L).pt"; ls -t data/backups/watch2/pre_night_*.pt 2>/dev/null | tail -n +4 | xargs rm -f   # the evening's save kept, three deep
READ=$(python3 - <<'PY'
import json
R = []
for l in open("data/watch2_caregiver.jsonl"):
    try: R.append(json.loads(l))
    except Exception: pass
day = max(r.get("day", 0) for r in R if r.get("action") == "night")
D = [r for r in R if r.get("day") == day]
lines = [r for r in D if r["action"] in ("line", "cue")]; smiles = sum(1 for r in D if r["action"] == "smile")
its = "".join(r.get("its_after", "") for r in lines)
duty = sum(1 for c in its if c != "_") / max(1, len(its)); spl = smiles / max(1, len(lines))
print("%d %.3f %.3f %d" % (day, duty, spl, len(lines)))
PY
)
DAY=$(echo $READ | cut -d' ' -f1); DUTY=$(echo $READ | cut -d' ' -f2); SPL=$(echo $READ | cut -d' ' -f3)
if python3 -c "import sys; sys.exit(0 if (float('$DUTY') < 0.2 or float('$SPL') < 0.6) else 1)"; then
  echo "night after day $DAY at $(date +%H:%M:%S): duty $DUTY, smiles/line $SPL: THE GUARD TRIPS, the long tag closes" >> $N/logs/watch2.log
  python3 -c "import time; time.sleep(20)"
  P=$(pgrep -f "body.serve --load data/watch[2].pt" | head -1); cp data/watch2.pt "data/backups/watch2/watch2_before_guard_day$DAY.pt"; kill $P; python3 -c "import time; time.sleep(5)"
  env nohup python3 -m body.serve --load data/watch2.pt --tok data/tok_char.json --port 8020 ${=BASE} --gate-slow-lr 0.0 > $N/logs/serve_watch2_reload.log 2>&1 &
  python3 -c "import time; time.sleep(50)"; T=$(pgrep -f "body.teacher --port 8020" | head -1); [ -n "$T" ] && kill $T
  echo "served again (pid $(pgrep -f 'body.serve --load data/watch[2].pt' | head -1)) at $(date +%H:%M:%S) with the tag closed; typist $T stopped for the chain" >> $N/logs/watch2.log
else
  echo "night after day $DAY at $(date +%H:%M:%S): duty $DUTY, smiles/line $SPL: the guard holds, the reading runs on" >> $N/logs/watch2.log
fi
