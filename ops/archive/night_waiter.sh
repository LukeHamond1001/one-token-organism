#!/bin/zsh
# THE NIGHT WAITER: wait for the Nth night row (label N+10), re-arm the guard and the dusk probe, tally the last three days, report the dusk, the probe and the branch
#   night_waiter.sh N
S=/private/tmp/claude-501/-Users-lukehamond-Projects-project/a22528f8-bc83-4acb-9044-d5917dc9456c/scratchpad; cd /Users/lukehamond/Projects/project
N=$1; L=$((N+10))
until [ "$(grep -c '"action": "night"' data/watch2_caregiver.jsonl)" -ge "$N" ]; do python3 -c "import time; time.sleep(30)"; done
python3 -c "import time; time.sleep(45)"; date '+%H:%M:%S'; echo "night $L ended (the ${N}th row)"; tail -1 $S/logs/watch2.log | cut -c1-140
# THE GUARD IS RETIRED (2026-09-19 17:15): the long tag has been closed for days (gate_slow_lr 0) and the guard's trip (the gate's duty under 0.2) fired on the ear's trace, restarting the served body with stale flags a minute after the reload. Not re-armed.
# G=$(pgrep -f "^zsh /.*guard_tag_at_night.sh" | wc -l | tr -d ' '); [ "$G" = "0" ] && eval "nohup nice -n 5 $(cat ops/guard_args.txt) > /dev/null 2>&1 &"
D=$(pgrep -f "^zsh /.*probe_at_dusk.sh" | wc -l | tr -d ' '); [ "$D" = "0" ] && { nohup nice -n 5 zsh $S/probe_at_dusk.sh $S probe_dusk_after$N.log > /dev/null 2>&1 & }
python3 -c "import time; time.sleep(2)"; echo "armed: guard $(pgrep -f '^zsh /.*guard_tag_at_night.sh' | wc -l | tr -d ' ') dusk $(pgrep -f '^zsh /.*probe_at_dusk.sh' | wc -l | tr -d ' ')"
DL=$(grep '"action": "night"' data/watch2_caregiver.jsonl | tail -1 | grep -o '"day": [0-9]*' | grep -o '[0-9]*'); echo "=== the day that ended ($DL)"; python3 ops/day.py $DL 2>&1 | head -4; python3 - <<PY
import json
R=[json.loads(l) for l in open("data/watch2_caregiver.jsonl") if l.startswith("{")]
for day in sorted(set(r.get("day") for r in R[-4000:] if r.get("action")=="line"))[-3:]:
    D=[r for r in R if r.get("day")==day]; L=[r for r in D if r["action"]=="line"]
    its="".join(r.get("its_after","") for r in L); own=[c for c in its if c!="_"]
    junk=sum(1 for c in own if not (c.islower() or c in " .?!'")); sm=sum(1 for r in D if r["action"]=="smile"); fr=sum(1 for r in D if r["action"]=="frown")
    ans=sum(1 for r in D if r["action"]=="smile" and str(r.get("why","")).startswith("answer")); A=[r for r in L if r.get("voice")!="b"]
    bb=sum(1 for r in A if any(c.isalpha() for c in r.get("its_after","")))
    print(f"day {day}: {len(L)} lines, own chars {len(own)} junk {junk} '?' {its.count('?')}, smiles {sm} (answer smiles {ans}) frowns {fr}; duty {len(own)/max(1,len(its)):.3f}; A-lines with letters in its turn {bb}/{len(A)}")
PY
python3 ops/queue_depth.py; echo "server: $(pgrep -f 'body.serve --load data/watch[2].pt' | wc -l | tr -d ' ')"; curl -s localhost:8020/insides | python3 -c "import json,sys; d=json.load(sys.stdin); print('store', d['store'], 'nights', d['nights'])"; df -h / | tail -1 | awk '{print $4 " free"}'
echo "=== the dusk before it"; grep "^LM\|^FACTS" $(ls -t $S/logs/probe_dusk_after*.log | head -1) | cut -c1-230
until [ -s $S/logs/probe_after_night$L.log ] && grep -q "rephrased" $S/logs/probe_after_night$L.log && grep -q "pause  2" $S/logs/probe_after_night$L.log; do python3 -c "import time; time.sleep(30)"; done; python3 -c "import time; time.sleep(20)"; date '+%H:%M:%S'; echo "=== the probe after night $L"; grep "^LM\|^FACTS\|^QA\|pause" $S/logs/probe_after_night$L.log | cut -c1-330; echo "=== the branch"; for st in sun day2; do nice -n 19 python3 tools/branch_probe.py data/watch2.pt --reads-along --set=$st 2>&1 | grep -v "unknown keys" | grep "BRANCH"; done
