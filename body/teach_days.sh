#!/bin/zsh
# teaching days for the served body: the probe on the saved body, then the teacher for one day (a queue planner
# fed by whoever teaches), for days FIRST..LAST. usage: body/teach_days.sh FIRST LAST [planner]
cd /Users/lukehamond/Projects/project
S=/private/tmp/claude-501/-Users-lukehamond-Projects-project/b5dd23ff-1e1e-4f1a-a920-a1dd78eb3b6e/scratchpad
P=${3:-queue}
for D in $(seq $1 $2); do
  while pgrep -f "body.caregiver" > /dev/null || pgrep -f "body.teacher" > /dev/null; do sleep 10; done
  echo "=== day $D probe (before)" >> $S/body2_probes.log
  python3 -m body.probe data/body2.pt --n 6 --ticks 24 2>&1 | grep -v Warning >> $S/body2_probes.log
  python3 -u -m body.teacher --port 8018 --day $D --days 1 --log data/body2_caregiver.jsonl --corpus data/body2_corpus.json \
      --planner $P --queue data/teach_queue.jsonl --period 60 --quiet 3 --cap 45 >> $S/body2_teach_day$D.log 2>&1
done
echo ALLDAYS >> $S/body2_probes.log
