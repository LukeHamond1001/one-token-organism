#!/bin/zsh
# days back to back for the second body: caregiver day N, then the probes on the saved body, then day N+1
# usage: body/run_days.sh FIRST_DAY LAST_DAY  (the serve must be up on 8018)
cd /Users/lukehamond/Projects/project
S=/private/tmp/claude-501/-Users-lukehamond-Projects-project/b5dd23ff-1e1e-4f1a-a920-a1dd78eb3b6e/scratchpad
for D in $(seq $1 $2); do
  while pgrep -f "body.caregiver" > /dev/null; do sleep 10; done
  echo "=== day $D probe (before)" >> $S/body2_probes.log
  python3 -m body.probe data/body2.pt --n 6 --ticks 24 2>&1 | grep -v Warning >> $S/body2_probes.log
  python3 -u -m body.caregiver --port 8018 --day $D --log data/body2_caregiver.jsonl --period 60 --quiet 3 --cap 45 --seed $D >> $S/body2_care_day$D.log 2>&1
done
echo ALLDAYS >> $S/body2_probes.log
