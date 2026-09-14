#!/bin/zsh
# STOP EVERYTHING CLEANLY (for a reboot): the typist chain and typist first (no new day), then the waiters and guards, then the server
# after asking it to save.
cd /Users/lukehamond/Projects/project
pkill -f typist_chain.sh; pkill -f "body.teacher --port 8020"
for p in guard_tag_at_night probe_at_dusk probe_after_save reload_after_save typist_relaunch_after_save lived_after_save pair_chain ctxkey_chain cortexkey_chain horizon_chain janitor_three; do pkill -f $p; done
python3 -c "import time; time.sleep(2)"
curl -s -X POST localhost:8020/save -H 'Content-Type: application/json' -d '{}' > /dev/null; python3 -c "import time; time.sleep(8)"
ls -la data/watch2.pt | awk '{print "saved", $6, $7, $8, $5/1e9, "GB"}'
pkill -f "body.serve --load data/watch2.pt"; python3 -c "import time; time.sleep(3)"
echo "left running:"; pgrep -fl "body.serve|body.teacher|typist_chain|guard_tag|probe_|night_copy|day_on_copy" | cut -c1-80
