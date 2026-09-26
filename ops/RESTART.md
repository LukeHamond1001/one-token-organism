# Restarting the served life (rewritten 2026-09-23; first written 2026-09-14)

How the served body is brought back after a reboot, and how it is changed at a save, as both are done on 2026-09-23.
Run every command from the project root, /Users/lukehamond/Projects/project. 43 other files in body/, tools/ and ops/
hard-code that path. No step here edits body/.

## What runs

| process | started in | what it does |
|---|---|---|
| `python3 -m body.serve` on port 8020 | step 1 | The body. It loads data/watch2.pt with the constants in ops/BASE_FLAGS.txt and runs one symbol per tick (period 0.15 s, about 0.167 s a tick on average). A day is 24000 ticks (about 67 minutes) and a night lasts 25-45 minutes. It serves the page at http://localhost:8020, and /talk. |
| `zsh tools/typist_chain.sh` | step 2 | Relaunches the typist (`python3 -m body.teacher`) whenever it exits. The typist types the queue's rows in the parent's two voices and times the face (the caregiver). After each night it asks the body to save and writes the `"save"` row. The cycle, the probe and the reload all wait on that row. |
| `zsh ops/night_cycle.sh` | step 3 | Keeps the queue fed, counts each day when its night comes, and at each save starts the probe and queues the morning. |
| `python3 ops/talk_proxy.py` on port 8021 | by hand (step 7) | The visitor's page. It pauses the typist while a browser polls it. |

Nothing else is armed. The guard and the night waiter are retired, and the dusk probe is no longer armed (see the last section).

## SCRATCH, and the three counts

SCRATCH is a working directory for the logs, the probe's morning copies and the two staged row files. Any directory works if it has a
`logs/` subfolder and room for two copies of the save (about 2.6 GB). The session scratchpad under /private/tmp did not survive
the reboot of 2026-09-22; a directory outside /private/tmp would. Today's cycle runs on
/private/tmp/claude-501/-Users-lukehamond-Projects-project/81d92d50-4dd9-488b-8268-f1a474117bfc/scratchpad.

    mkdir -p <SCRATCH>/logs

Three counts name the nights, and they differ. Read them; the old rule "night N's save is row N-24" no longer holds.
- **Save rows:** `grep -c '"action": "save"' data/watch2_caregiver.jsonl` (306 at 10:43 on 2026-09-23). Used for the cycle's
  argument, the probe's argument and the morning copy's name (`post_night_<saves>.pt`).
- **Night rows:** `grep -c '"action": "night"' data/watch2_caregiver.jsonl` (320). Used in the probe log's name
  (`probe_after_night<rows>.log`) and the backups' names (`watch2_before_reload_night<rows>.pt`).
- **The body's own nights:** `"nights"` in `GET /state` (328). The diary numbers its nights this way.

## Stopping for a planned reboot

1. `pkill -f "ops/night_cycle.sh"`. ops/stop_all.sh does not stop the cycle.
2. Stop ops/talk_proxy.py if it is running (Ctrl-C in its terminal). It resumes the typist when it exits.
3. `zsh ops/stop_all.sh`. It stops the chain and the typist first (so no new day starts), then any probe or reload waiter. Then it
   asks the body to save and stops it. It also kills some retired process names, which does no harm.

stop_all.sh's save is written mid-day, so the restart continues that day: skip the hand-queued morning in step 3. An
unplanned stop (the Mac's shutdown on the morning of 2026-09-22) loses everything the body lived since its last save. The restart
then begins from the post-night save, which is a morning.

## 1. Serve the body

    nohup python3 -m body.serve --load data/watch2.pt --tok data/tok_char.json --port 8020 $(tr '\n' ' ' < ops/BASE_FLAGS.txt) > <SCRATCH>/logs/serve_watch2.log 2>&1 &
    until curl -s -m 3 localhost:8020/state > /dev/null; do sleep 2; done

ops/serve_command.txt holds the exact command last used, with the same flags. The save is 1.3 GB and takes a moment to load.
When the body answers, run the check under "After any reload or restart" below.

## 2. The typist chain

The quoted command is in ops/typist_chain_command.txt (`PORT LOG "TYPIST ARGS" "ENV" ROUNDS`; the args and the env must stay
quoted). Replace `<SCRATCH>` and run it only after the body answers, because the chain stops at once if no server is on 8020:

    S=<SCRATCH>; eval "$(sed "s|<SCRATCH>|$S|" ops/typist_chain_command.txt)"

The chain launches the typist for the log's last day + 1. Each typist runs `--days 6` and logs to /tmp/typist_8020_day<N>.log;
the chain logs to `<SCRATCH>/logs/typist_chain_8020.log`. The typist's queue cursor (data/teach_queue_w2.jsonl.pos) keeps the
lines it has read but not yet typed, so queued lines survive a restart.

The typist runs its --days in one process, and the chain relaunches it only when that process exits or is killed, taking the day
label from the log. So a change to body/teacher.py or body/caregiver.py reaches the served typist only at a relaunch. To apply one,
kill the typist at a boundary (right after a night row) and the chain starts the next typist on the new code within a minute.
(2026-09-18: the filler removed at 22:07 stayed in use until night 260, because the typist started at 21:56 kept the old planner.)
NEVER look for the typist with its own pattern (`-m body.teacher --port 8020 --day`) in a pgrep of your own. The chain waits
while any process shows that pattern, and a supervisor's repeated check held the relaunch four minutes on 2026-09-18 18:45.
Use `ps -eo args | grep body.teacher | grep -v grep` instead.

## 3. The night cycle

    nohup zsh ops/night_cycle.sh <SCRATCH> <SAVES> > <SCRATCH>/logs/night_cycle.log 2>&1 &

`<SAVES>` is the current number of save rows. The cycle begins four minutes after the log holds that many save rows (four minutes
after launch, if it already does) and acts at every later save. Its first save is therefore the next one. The running cycle
was started at 18:42 on 2026-09-22 as `night_cycle.sh <today's scratchpad> 299`.

What it does:
- **Every minute while the body is awake:** if ops/queue_depth.py counts fewer than 12 lines ahead, it appends four rows from
  its own pool (the fox, the hen, the lemon, the ducks, the shade) and one story row (ops/read_stories.py, from data/stories_valid.txt).
- **At each new night row:** it prints the day's count (ops/day_report.py).
- **At each new save row:** it starts `ops/probe_after_save5.sh <SCRATCH> <saves> probe_after_night<night rows>.log`, then
  waits 150 s for a reload (if one is armed) and the typist's relaunch. Then it appends `<SCRATCH>/morning_300.jsonl` and
  `<SCRATCH>/day_358.jsonl` to the queue, plus four story rows and three revisits (ops/revisit_rows.py).

Each event is one line in its log. Run only one cycle: stop the old one before starting another, or the queue is fed twice.
Do not edit ops/night_cycle.sh while it runs, because zsh reads a running script as it goes. A change takes effect when the
cycle is relaunched.

### The two staged row files (re-stage them after every reboot)

The cycle reads these two files from SCRATCH at every save. They are not in the repo. If they are missing, `cat` writes its
error into the cycle's log, the morning goes without them, and the pool rows fill the queue. After a reboot, write them again
before the next save. The diary describes where they came from:
- **morning_300.jsonl (4 rows):** the morning's greeting ("good morning. did you sleep well?") and seven questions on the facts
  taught in the days' rows. ops/teaching_method.md item 7 describes the morning ("what is sour?" has been asked every morning
  since day 315). DIARY_BODY.md's entries "THE MORNING AFTER NIGHT 299" and "THE MORNING AFTER NIGHT 300" (17:08 and 19:04 on
  the 20th) record the same seven questions as the typist typed them.
- **day_358.jsonl (11 rows):** the day's script, a whole day of two-voice conversation from "now the sun is up" to "good night".
  It was first typed on day 358 (DIARY_BODY.md, "DAY 358 (19:03-20:11 on the 20th", the first day of the nine-tick release).
  From day 359 the cycle queued it every day ("DAY 359 ... the first day fed by the cycle").
- After the reboot of 2026-09-22 both files were rewritten into the new scratchpad (DIARY_BODY.md, "THE REBOOT AND THE RETURN
  (the 22nd)": "the two staged row files rewritten").

Their contents as staged at 13:55 on 2026-09-22, still unchanged at 12:30 on 2026-09-23. If the files in SCRATCH are newer, use
those instead:

```
cat > <SCRATCH>/morning_300.jsonl <<'EOF'
{"say": ["good morning. did you sleep well?", "b: yes. I slept. I am up", "what is sour?", "b: a lemon is sour"]}
{"say": ["who gives us eggs?", "b: the hen gives us eggs", "is the egg hard or soft?", "b: the egg is hard"]}
{"say": ["what is sweet?", "b: a pear is sweet", "what does the hen say?", "b: cluck cluck"]}
{"say": ["what do the ducks eat?", "b: bugs and bread", "who gives us milk?", "b: the cow gives us milk"]}
EOF
cat > <SCRATCH>/day_358.jsonl <<'EOF'
{"say": ["you said the moon. now the sun is up", "b: yes. it is day", "what is up in the sky?", "b: the sun. it is hot"]}
{"say": ["the hen pecks by the wall. see her?", "b: I see her", "what does she give us?", "b: eggs. the hen gives us eggs"]}
{"say": ["is the egg hard or soft?", "b: the egg is hard", "and the bread?", "b: the bread is soft"]}
{"say": ["we walk to the ducks. hold my hand", "b: I hold your hand", "what do they eat?", "b: bugs and bread"]}
{"say": ["what is sweet?", "b: a pear is sweet", "and what is sour?", "b: a lemon is sour"]}
{"say": ["the cow eats grass. and she gives?", "b: milk. the cow gives us milk", "is the milk sweet?", "b: yes. the milk is sweet"]}
{"say": ["the dog runs to us. what does he say?", "b: woof woof", "throw him the ball", "b: I throw it. he runs"]}
{"say": ["the sun is hot. we sit in the shade", "b: the shade is cool", "do you want water?", "b: yes. water is wet and cold"]}
{"say": ["what did we do today?", "b: we saw the hen and the ducks", "and what did we eat?", "b: an egg and bread"]}
{"say": ["the sun is low. the moon comes up", "b: I see the moon", "we go in soon. are you tired?", "b: yes. I am sleepy"]}
{"say": ["lie down by me. close your eyes", "b: my eyes are closed", "sleep well. I am here", "b: good night"]}
EOF
```

A restart from a post-night save puts the body at its morning, but the cycle queues the morning only at the next save. To give
this day its morning, append both files once by hand before the typist's first line:

    cat <SCRATCH>/morning_300.jsonl <SCRATCH>/day_358.jsonl >> data/teach_queue_w2.jsonl

Do not do this after a mid-day save (stop_all.sh). Otherwise, never queue the morning's rows before the night has begun: the
queue is first in, first out (ops/teaching_method.md item 5).

## 4. The probe after each save (started by the cycle)

`ops/probe_after_save5.sh SCRATCH SAVES LOGNAME` waits until the log holds SAVES save rows. It then copies the save to
`SCRATCH/post_night_<SAVES>.pt` (kept two deep) and runs two probes into `SCRATCH/logs/LOGNAME`:
- tools/probe_lm.py: the held-out ruler tools/heldout_stage4.txt, the facts, and every prefix and question.
- tools/qa_by_gap.py --set=rephrased.

Arm it by hand only when the cycle is not running:

    nohup nice -n 5 zsh ops/probe_after_save5.sh <SCRATCH> <next save count> probe_after_night<night rows>.log > /dev/null 2>&1 &

The probe is a ruler kept alongside the main judgment. Progress is judged by sitting with the body and reporting the conversation
(the user's word, 2026-09-18).

## 5. A change of constants: the reload at a night's save

A change is measured on a copy first (step 6). It reaches the served body only at a night's save, and a night is never rolled back.
1. Archive the served flags (`cp ops/BASE_FLAGS.txt ops/archive/flags/BASE_FLAGS_pre_<name>.txt`, the name the ledger item
   uses). Write the new flags into ops/BASE_FLAGS.txt and the new command into ops/serve_command.txt. Pass every changed
   constant explicitly, reverts included (see "After any reload or restart").
2. During the day, arm the reload:

       nohup zsh ops/reload_after_save.sh <SCRATCH> "$(tr '\n' ' ' < ops/BASE_FLAGS.txt)" > /dev/null 2>&1 &

   It waits for the next save row. There it stops the typist (so no next-day session begins and no line is lost) and holds
   the chain with SIGSTOP. It copies data/watch2.pt to `data/backups/watch2/watch2_before_reload_night<rows>.pt` and deletes
   the older reload copies beyond two. Then it restarts the body with the new flags and the code on disk, clears the face, and
   releases the chain, which relaunches the typist for the next day. Its own lines go to `<SCRATCH>/logs/watch2.log`; the new
   server's go to `<SCRATCH>/logs/serve_watch2_reload.log`. The cycle's 150 s wait after each save leaves room for it.
3. Afterwards, run the served_cfg check below and record the change in ITERATIONS.md and DIARY_BODY.md.

A change to body/ reaches the served body only at such a restart (the code on disk is loaded). `ops/reload_now.sh SCRATCH
"FLAGS"` does the same mid-day: it asks the body to save, restarts it, and stops the typist so the chain relaunches it. Use it
only for a change that cannot wait for the night (the user's word).

A change of the KEY'S FORM (key_ctx, ctx_form, key_form) also needs the store rebuilt, because memories written under one form
are not found by a query under another. `ops/rekey_after_save.sh SCRATCH FLAGS_FILE` does this at the next post-night save: it
holds the typist, backs up the save, runs tools/rekey_store.py over the utterance memory, serves the body again and relaunches
the typist (about fifteen minutes of the morning). Night 262 (2026-09-18) took --ctx-form shifted this way. **WARNING (2026-09-23):
rekey_after_save.sh deletes every `data/backups/watch2/watch2_before_*.pt` except the newest two.** That would remove the
guard's, the rekey's and the older reload backups. Do not run it until the user has decided which backups to keep (data/README.md).

## 6. Measuring on copies: tools/inproc_parent.py

The served parent (body/teacher.py and body/caregiver.py, unchanged) teaches a copy of the body inside one process, on a virtual
clock. The copy loads the way a served reload loads, from the save's own constants and then the flags. Each constant under test
is set only on the copy (`--cfg 'k=v'`, `--cfg-default k`, `--reflexes-off`). Copy the probe's morning copy
(`<SCRATCH>/post_night_<N>.pt`). data/watch2.pt also works, read-only, when no save is due.

    nice -n 19 python3 tools/inproc_parent.py <SCRATCH>/post_night_<N>.pt --flags ops/BASE_FLAGS.txt \
        --typist-from ops/typist_chain_command.txt --queue data/teach_queue_w2.jsonl --days 1 \
        --out <SCRATCH>/copy_<name> [--cfg 'k=v ...'] [--report]

- `--out` must be a new or empty directory. The queue, its cursor, the corpus and the log's tail are copied into it. An audit
  hook refuses any write outside `--out` and the system temp dir, anything under body/, ops/, data/ or .git/, any socket other
  than AF_UNIX, and any signal or subprocess. A copy cannot reach port 8020 or the served files.
- `--resume` runs another day of the same copy; `--snapshot-only` with `--inputs-from` gives several runs the same inputs.
  `--queue-pos` starts from a chosen byte of the queue.
- Before each save it checks the disk and keeps room for the served body's nightly save: at least twice the served save plus
  1 GiB stays free (`--disk-reserve-gb` overrides this).
- A copy shares the machine with the served body. Copies lengthen the served typing and nights (the diary, day 300), so run
  one at a time, at nice 19.
- tools/inproc_parent_check.py is its self-check, run on tiny newborns.

Other instruments that run on copies: tools/night_copy.py (the real night), tools/day_on_copy.py, tools/demo_rehearsal.py
(the demo's sequence), tools/silence_probe.py and tools/slow_line_probe.py. See tools/README.md.

## 7. Sitting with it, the talk page, and the parent

THE TALK PAGE (2026-09-17): ops/talk_proxy.py serves the visitor's page on port 8021 and forwards its calls to the body.
While a browser polls it, the typist is stopped (SIGSTOP); it is resumed (SIGCONT) `--pause-after` seconds after the last poll
(6 by default) and on any exit of the server. The body serves the same page at http://localhost:8020/talk without the pause.
Start it after the body:

    nohup python3 ops/talk_proxy.py --port 8021 --body 8020 > <SCRATCH>/logs/talk_proxy.log 2>&1 &

For the demo, the demo script (DEMO_SCRIPT.md, kept locally, not in git) runs it in a terminal with `--pause-after 600`.

THE CHAIR: tools/teach_live.py (`start`, `say`, `listen`, `leave`, `end`) is for sitting with the body from the terminal, which is
how progress is judged (2026-09-18). It holds the typist with SIGSTOP from `start` to `end`.

THE PARENT: the cycle feeds the queue by itself. A parent who writes rows by hand works from ops/parent_brief_human.txt (the human
teacher, the user's word of 2026-09-17) and ops/teaching_method.md. It reads the queue's depth only with ops/queue_depth.py, never
from the .pos file. The method keeps lines to 38 characters or fewer. The typist silently drops any line over 40 characters, or
with any character other than letters, spaces and `. ? !` (so no comma, no apostrophe, no digit).
ops/archive/parent_brief_template.txt is the old brief (see the last section).

## After any reload or restart

AFTER ANY RELOAD OR RESTART (2026-09-16, the reviewer's finding): a save carries its own constants, and the flags only change
some of them. A key left out of the flags keeps the save's value. Every changed constant must be passed explicitly, reverts
included. After the restart, run

    python3 ops/served_cfg.py data/watch2.pt ops/BASE_FLAGS.txt

and read the lines marked "the save alone": none of them may be a constant that the ledger records as reverted. The same check
on the morning copy (`python3 ops/served_cfg.py <SCRATCH>/post_night_<N>.pt ops/BASE_FLAGS.txt`) reads the same bytes without
opening the served file.

## Retired, or no longer armed

THE GUARD ON THE LONG TAG IS RETIRED (2026-09-19 17:15). The tag had been closed for days (gate_slow_lr 0). The guard's trip
(the gate's duty under 0.2 after the parent's lines) fired on the ear's trace at night 286's row. It restarted the served body a
minute after the reload, with the flags captured when it was armed, so the reading share it carried was stale. Do not arm it
(ops/archive/guard_tag_at_night.sh; ops/archive/guard_args.txt holds its old command line, with a scratch path that no longer exists). The reload
scripts (reload_after_save.sh, reload_now.sh, rekey_after_save.sh) are the only restarts.

THE NIGHT WAITER IS RETIRED (2026-09-20). ops/night_waiter.sh (now ops/archive/night_waiter.sh) ran one per night. It re-armed the dusk probe (and the guard, until
2026-09-19), tallied the last days (ops/day.py), waited for the morning probe and ran the branch probe. ops/night_cycle.sh replaced it (step 3).
Its scratch path (a22528f8…) is hard-coded and no longer exists.

THE DUSK PROBE IS NO LONGER ARMED. ops/probe_at_dusk.sh was re-armed each night by the night waiter. When the sleep pressure
comes within 700 ticks of the night, it asks the body to save. That overwrites the served file mid-day, so a copy of
data/watch2.pt taken after it is no longer the morning's. Arm it only by hand, for a dusk reading, and not when the night is
already near: `nohup nice -n 5 zsh ops/probe_at_dusk.sh <SCRATCH> probe_dusk_<label>.log &`.

THE OLD PARENT STEP IS REPLACED. The old step 7 ("the Agent brief in ops/archive/parent_brief_template.txt, with the night count, the day
label, the facts schedule (ten a day, rotating)") gave way on 2026-09-17 to the human teacher (ops/parent_brief_human.txt) and to
the cycle's feeding.

THE POD SCRIPTS (ops/archive/pod_pretrain.sh, ops/archive/pod_pretrain_run.sh) are not run. Everything runs locally.
