#!/bin/zsh
# THE NIGHT CYCLE (2026-09-20): one standing loop for the served body's days and nights from a given save on. While the body is
# awake it keeps the queue fed (the taught pairs, the fox, a story row) whenever fewer than twelve lines lie ahead; at each night's
# start it prints the day's count (ops/day_report.py); at each save it runs the probe (ops/probe_after_save5.sh), waits for the
# reload if one is armed, and queues the morning rows, the day's script, four story rows and three revisits. Each event is one
# line on stdout (a Monitor's event). Nothing inside the body is read; the queue file and the page log are the only hands.
#   night_cycle.sh SCRATCH FIRST_SAVE   (the loop begins once the log holds FIRST_SAVE save rows)
cd /Users/lukehamond/Projects/project
S=$1; FIRST=$2; L=data/watch2_caregiver.jsonl; Q=data/teach_queue_w2.jsonl
until [ "$(grep -c '"action": "save"' $L)" -ge "$FIRST" ]; do sleep 30; done
sleep 240
POOL=(
'{"say": ["the fox is by the wood. what color is the fox?", "b: the fox is red", "and what does the fox say?", "b: the fox says yip"]}'
'{"say": ["the hen pecks by the wall. who gives us eggs?", "b: the hen gives us eggs", "and what does she say?", "b: cluck cluck"]}'
'{"say": ["I cut a lemon. what is sour?", "b: a lemon is sour", "and what is sweet?", "b: a pear is sweet"]}'
'{"say": ["the ducks come to the bread. what do they eat?", "b: bugs and bread", "and who gives us milk?", "b: the cow gives us milk"]}'
'{"say": ["is the egg hard or soft?", "b: the egg is hard", "good. and the bread?", "b: the bread is soft"]}'
'{"say": ["what color is the fox?", "b: the fox is red", "and what does the dog say?", "b: woof woof"]}'
'{"say": ["it is hot. we sit in the shade", "b: the shade is cool", "do you want water?", "b: yes. water is cold"]}'
'{"say": ["what does the fox say?", "b: the fox says yip", "and what does the hen say?", "b: cluck cluck"]}'
'{"say": ["the cow is by the gate. who gives us milk?", "b: the cow gives us milk", "is the milk sweet?", "b: yes. the milk is sweet"]}'
'{"say": ["I hold the pear. what is sweet?", "b: a pear is sweet", "and the lemon?", "b: a lemon is sour"]}'
)
CLOSE=(
'{"say": ["the sun is low. the moon comes up", "b: I see the moon", "we go in. are you tired?", "b: yes. I am sleepy"]}'
'{"say": ["lie down by me. close your eyes", "b: my eyes are closed", "sleep well. I am here", "b: good night"]}'
)
nsave=$(grep -c '"action": "save"' $L); nnight=$(grep -c '"action": "night"' $L); i=0
wake_ts=$(grep '"action": "save"' $L | tail -1 | grep -o '"ts": "[^"]*"' | cut -d'"' -f4 | cut -c1-16)
while true; do
  n2=$(grep -c '"action": "night"' $L)
  if [ "$n2" -gt "$nnight" ]; then
    nnight=$n2; now=$(date +%Y-%m-%dT%H:%M)
    bn=$(curl -s -m 3 localhost:8020/state | python3 -c "import sys,json; print(json.load(sys.stdin).get('nights'))" 2>/dev/null)
    echo "$(date +%H:%M:%S) night $bn has ended (the typist's night row); the day before it: $(python3 ops/day_report.py $wake_ts $now --label "day" 2>&1 | tail -4 | tr '\n' ' ' | cut -c1-400)"
  fi
  s2=$(grep -c '"action": "save"' $L)
  if [ "$s2" -gt "$nsave" ]; then
    nsave=$s2; wake_ts=$(date +%Y-%m-%dT%H:%M)
    nohup zsh ops/probe_after_save5.sh $S $nsave probe_after_night$nnight.log > /dev/null 2>&1 &
    sleep 150                                                              # the reload, if one is armed, and the typist's morning
    cat $S/morning_300.jsonl >> $Q; cat $S/day_358.jsonl >> $Q
    python3 ops/read_stories.py --corpus data/stories_valid.txt --rows 4 --lines 4 > /dev/null 2>&1; python3 ops/revisit_rows.py --rows 3 > /dev/null 2>&1
    echo "$(date +%H:%M:%S) the wake from night $bn: the morning rows, the day's script, four stories and three revisits queued; the probe running (probe_after_night$nnight.log)"
  fi
  if ! curl -s -m 3 localhost:8020/state | grep -q '"asleep": true'; then
    ahead=$(python3 ops/queue_depth.py 2>/dev/null | tail -1 | sed -E 's/lines ahead ([0-9]+).*/\1/')
    if [ -n "$ahead" ] && [ "$ahead" -lt 12 ] 2>/dev/null; then
      for k in 1 2 3 4; do i=$(( (i + 1) % ${#POOL[@]} )); echo "${POOL[$((i + 1))]}" >> $Q; done
      python3 ops/read_stories.py --corpus data/stories_valid.txt --rows 1 --lines 4 > /dev/null 2>&1
      echo "$(date +%H:%M:%S) fed: four rows and a story ($ahead lines were ahead)"
    fi
  fi
  sleep 60
done
