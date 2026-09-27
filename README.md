# iga — the body from nothing

iga is a small brain-inspired architecture that lives one symbol per tick, learns while it runs from what it hears and from a person's face, and sleeps and remembers: a transformer cortex predicts the next symbol, a hippocampal store keeps what surprised it, nights replay and imagine what it heard, a learned gate decides when the mouth speaks, and critics, an actor and feelings turn its reward (the caregiver's smiles and frowns, and a small fixed reward for each symbol typed to it) into value and dopamine. Its test bed is a language body, a 179M-parameter character-level organism started from random weights on 2026-09-06, that shares one page with the people who talk to it. What it says it learned from what was typed to it in time (conversations written by Claude and children's stories, typed by a scripted parent, and visitors' lines), and for 37 of its nights (2026-09-19 to 09-22) from story sentences read into its sleep; there is no prompt and no pretraining. The goal is the same architecture in a humanoid robot, so every mechanism is judged by whether it would survive a change of body.



## Run it yourself

The language body runs on a laptop's CPU. Python 3.11 or newer, then:

```bash
git clone https://github.com/LukeHamond1001/one-token-organism.git
cd one-token-organism
pip install -r requirements.txt
python3 -m body.serve --birth data/body.pt --tok assets/tok_char.json --port 8018 --period 0.5
```

Open `http://localhost:8018/talk`. That is a newborn: 6.5M parameters at the defaults (`--d 256`), knowing nothing, living one
letter a tick at two ticks a second, learning from what you type and from your face. Type to it in short lines (`hi. hi baby.`).
Its face is the only reward it has: the up arrow smiles at what it just did, the down arrow frowns, half a step at a time, and a
smile given a second or two after the act is what teaches it. After 24,000 waking ticks it sleeps, dreams and saves itself to
the `--birth` path; start it again with `--load data/body.pt` and it goes on. The served body was born the same way with
`--d 1024 --layers 12 --seed 1` (179M parameters) and raised for weeks by a typist and a person; its constants are in
`ops/BASE_FLAGS.txt`, and `ARCHITECTURE.md` says what the organs are.

To run the raised body rather than a newborn, download its save from this repository's Releases page (`watch2.pt`, 1.3 GB, once
it is published) and serve it with the constants it was raised under:

```bash
python3 -m body.serve --load watch2.pt --tok assets/tok_char.json --port 8020 $(cat ops/BASE_FLAGS.txt)
```

(`ops/BASE_FLAGS.txt` names `data/stories_valid.txt` for the night's story corpus; with `--dream-corpus-n 0`, as served, the
file is not read.) The robot sim needs `pip install -r requirements-sim.txt` (MuJoCo) as well, and its parent's voice is
synthesised by a small Swift server that runs on macOS only (`--voice fake` runs without it); its run command is in the section
below.

## The robot sim: where the project is (2026-09-27)

Since 2026-09-24 the same architecture lives in a simulated body: a Unitree G1 humanoid with hands (MuJoCo, 150 ms ticks, its own eyes,
ears, touch, joint sense and a vocal tract) on a play mat in a room, with a simulated human-proportioned parent who kneels beside it,
talks, shows toys, turns it over and smiles or frowns at what it does. Its reward is her face (±2 on her born reading's rise) and its own
pain (−1 on a gear loaded past its motor); there is no charge, no bottle and no other reward (A88, A91). One seed lives on this Mac,
`data/g1_seed1` (born 2026-09-26; not in git), a day of 24,000 waking ticks and a live dark night of 24,000 more, its pair (`life.pt`
and `world.pt`) saved at every dawn. Every change to the body or the teacher is built on a worktree, gated by the suites, measured on
a copy of a dawn pair when its effect is in doubt, and put in at a dawn, one change a boundary, so each day's ledger has one cause.

Where it stands, day 11 of that life:
- Days 1 to 9 were mostly the parent's faults found and fixed at one dawn each (A95 to A112 in `docs/SIM_DESIGN.md`): her plan frozen
  half-knelt for a day (A108), the child rolled out of the room into the hall (A110: carried back to the mat in its sleep), her turn's
  approach silencing judgment of the child's own acts (A111), her chasing a rolling child round the mat (A112).
- On 2026-09-27 the dawn-10 pair was lived twice on the same tree, the copy with one switch, the striatal actor on (A113, C87). Over
  the day the copy judged 162 acts against the life's 79 (59 motor against 10: reaches, lifts, hits, gets, shakes), hurt itself 65
  ticks against 615, saw her smile 48 times against 6, and ended the day in a positive mood against a negative one. Until then the act
  was the forecaster's own prediction and dopamine could only quieten a gate; nine days of value learning had reached no policy.
- The actor went in at dawn 11. Day 11 (with a supine wake and her side kept): 47 judgments by midday, 26 of them motor, two pain
  ticks in 12,000, no rolling, every lean-in landing, the arms' inverse models risen from 0.11 to 0.31 and 0.40 in three days.
- Open: the turn of a prone child works from a narrow band of kneeling distance (C98); a side-lying G1 looks at the floor and cannot
  see her, which is the cameras' own pitch and not a build (C94); the first transfer test, a toy the child has never seen, is the next
  build (A28's novel toys, B2's colour twins). The C rows of `docs/SIM_DESIGN.md` hold every open question with its evidence.

How it runs:
- The life: `python3 tools/sim_life.py --out data/g1_seed1 --resume --days 6 --d 512 --seed 1 --voice real --threads 4 --page`
  (the `/sim` page on `http://127.0.0.1:8030/`; `--resume` continues from the pair in `--out`). A measurement on a copy: copy the pair
  to another folder and add `--set KEY=VALUE` (the actor's day copy was `--set actor=1 --ticks 24000`, no page).
- The records: `data/g1_seed1/ticks.jsonl` (one row a tick), `report.json` at each night, `run.log` (a line every 500 ticks); the days'
  logs are kept as `run_dayN.log`.
- The suites: `body/tests/test_sim_world.py`, `test_sim_lane.py`, `test_sim_parent.py`, `test_sim_lang.py`, `test_sim_voice.py`,
  `test_sim_ears.py` and `test_motor.py`, each run as a script (`python3 body/tests/test_sim_parent.py`, one at a time beside the
  life). The pins: `tools/determinism_check.py --profile sim` must reproduce `tools/pins/digests.txt`, and the language default
  `7c54199e3c72cf77d45a8e94` at every commit.
- The branches: `main` holds everything (this merge, 2026-09-27); `sim` is the tree the life runs, checked out at
  `../project-worktrees/wt_int` and moved only at a dawn; `a112` is the next dawn's tree, `a114` a parked attempt at C98; `sim-face`
  older face work not yet merged. The earlier `sim-*` branches are merged.

Where to read: `docs/SIM_DESIGN.md` is the design and the ledger in one: the sections, the amendments A1 to A113 (each with what was
found, what was built, what was measured and its boundary), the C rows (open questions and their evidence), the B questions (the
owner's calls on the room's shape) and the plan. `docs/audit/` holds the studies behind the larger decisions. The parent's code is
`body/sim/parent_*.py` and `body/sim/lang/` (her conduct, day plan, templates and percept), the world `body/sim/world.py` and
`g1scene.py`, the room `make_g1room.py`, the lane between them `lane.py`.

## What it does today

The numbers below come from the served body (`data/watch2.pt`, on day 400 by the page log's count after the night that ended at 12:27 on 2026-09-23) and from copies of it. Each is taken from `ITERATIONS.md` items 48 to 52 and the latest entries of `DIARY_BODY.md`, where the full runs are written up.

**It takes turns.** Since 2026-09-23 its turn-taking is sensed (`pace_sense`, item 51). The body measures how long its partner pauses inside a line and how long they take to reply, as running quantiles saved with the body, and waits by those measures. No timing is fitted by hand.
- The first live day (2026-09-23, 00:29-01:38): 0.09 words said over a one-handed line, with 94 percent of lines clean; 0.00 words over fast lines; 0.3 words of babble per silent minute; 4 frowns.
- Three days earlier, under the constants served on 2026-09-20 (item 48, on a copy), it said 0.50 words over each one-handed line (65 percent clean) and 41 words per silent minute. The answer looped through a person's thinking silence: "a lemon is sour a lemon is sour a lemon is s...".

**It answers what it was taught.**
- On the first live day it answered 84 percent of the day's taught questions, a median 2.4 s after the line ended. Under the earlier timing reflexes, days 367 to 377 gave 82 to 95 percent, at a median of 1.2 to 1.3 s on the days whose delay was reported.
- The morning after night 315, 10 of 11 questions were answered whole or nearly whole, and it said nothing over any line.
- Asked the taught questions in new words (a copy after night 310, two seeds), 11 of 16 answers carried the key word, for example "the egg. is it hard?" -> "yes. the egg is hard".

**It learns a fact from a person and keeps it overnight.**
- The owl (the sit-down of 2026-09-22, every line typed one-handed): "what color is the owl?" had never been asked. The other voice said "the owl is brown" once. Asked again, it answered "the owl is brown" at 2.7 s, and again at every later ask (2.3, 6.8 and 2.1 s), after other questions and after a minute alone. During that minute it murmured "he the owl is" to itself.
- The sheep (item 50, a copy): ten facts were told once each during a day. The next morning they were asked among five facts never told. Four came back whole ("the sheep says baa", "the bee says buzz", "the horse says neigh", "the mouse says squeak") and three nearly ("the crow is blach", "the plum is purplum", "the crab is orab"). None of the five never-told facts got a right-sounding answer. This is the best next-day recall of once-told facts so far (item 35 had 1 of 3 live). The copy ran with every timing reflex off and with two fixes that the served flags do not carry (`chunk_gate` and the felt entry of utterances). No matched night was run without them, so their share of the result is not known.
- The fox (item 48): a pair typed into one day's rows ("what does the fox say?" / "the fox says yip") was answered at the first ask the next day, on a copy that had not been asked it before.
- In demo rehearsals on copies, a new fact took three to five tellings.

**It sleeps.** After 24,000 ticks awake (about 67 minutes: the served period is 0.15 s a tick, but a tick takes about 0.17 s in practice) it sleeps for about 25 minutes (24 to 36 minutes for the nights of 2026-09-23). The night replays the utterances it heard (NREM), imagines (REM) and saves. Since night 322 the night no longer reads stories it never heard (item 52), which cut the night from about 41 minutes to about 25.

**The rulers.** On lines no parent typed (`tools/heldout_stage4.txt`), the cortex's next-symbol accuracy is about 0.60 (0.604 after nights 300 and 302). The language probe's thirty fact questions, read greedily from the mouth, get 8 to 10 of 30.

## What it cannot do yet

- Say anything beyond what it was taught, or answer in its own words. A strange question gets a familiar answer ("woof woof").
- Learn a second fact about a subject it already knows when the fact comes in a new form. "where is the fox?" / "the fox is in the wood", taught five times, was never recalled; "the fox is red" won every time (item 49).
- Keep every new fact clean overnight. About a third come back garbled ("the crab is orab").
- Stop an answer or keep to its own questions. It runs on ("a lemon is sour a lemon is sour", "the owl is brown we go in") and asks questions built from the forms it hears ("what color is the frog").
- Hold its mood through a long sitting. The mood fell from 1.2 to -5.3 across the owl sit-down. Its speech gets junkier when the mood is low, just after a restart and just after a night.
- Carry its manners in learned weights alone. With every timing reflex off (a copy, item 50) it talked over each one-handed line all day, about eight words a line, and did not improve within the day. Its turn-taking today comes from `pace_sense`, which measures the partner but is still a built-in mechanism. Getting the learned gate to carry the manners is the open work.
- Answer quickly when the person types slowly. At half the typist's speed its answers come at a median 3.8 s. A faster form (`pace_fore_q`) talked over slow lines and was not adopted (recorded under item 52).
- Live in a robot. The review of 2026-09-22 (`ops/review_2026-09-22.md` §3) lists the gaps:
  - one symbol per tick, on one channel, with one mouth and one readout;
  - the world going quiet is both the event clock and the turn signal;
  - the night replays only the world's lines, never the body's own acts;
  - the actor's unit is a word ended by a space, and one gate serves the whole body;
  - the symbol table is frozen;
  - the loop has no deadline, and sleep comes by tick count;
  - reward is felt only on changes of face.
- Prove that a refactor changes nothing. `tools/determinism_check.py` runs on the flags alone, and 26 served constants live only in the save (review §4).

## How to run it

Everything runs on one Mac, on the CPU, from the repo root. It needs Python 3.11 or later with `torch`, `numpy` and `tokenizers` (`pip install -e .`; see `pyproject.toml`). Many scripts hard-code `/Users/lukehamond/Projects/project`, so the tree must sit there or those paths must be changed. `data/` is not in git: a saved body and `data/tok_char.json` are needed (see the map below).

Several scripts take a scratch directory, written `<SCRATCH>` below. It must hold a `logs/` subfolder: `mkdir -p <SCRATCH>/logs`.

### 1. Serve the body (port 8020)

```bash
nohup python3 -m body.serve --load data/watch2.pt --tok data/tok_char.json --port 8020 $(tr '\n' ' ' < ops/BASE_FLAGS.txt) > logs/serve_watch2.log 2>&1 &
```

The exact command last used, with every flag written out, is `ops/serve_command.txt`. The served constants are `ops/BASE_FLAGS.txt`. A save carries its own constants and the flags are only a delta on it, so every changed constant is passed explicitly, reverts included. After any restart or reload, run:

```bash
python3 ops/served_cfg.py data/watch2.pt ops/BASE_FLAGS.txt
```

Then read the lines marked "the save alone". None of them may be a constant that the ledger records as reverted.

The server's endpoints are `/type`, `/face`, `/state` and `/save`, and the visitor's page is at `/talk`. `/insides` is the supervisor's instrument and never the caregiver's.

### 2. The typist chain (the parent's hands)

The exact quoted command, from `ops/typist_chain_command.txt`:

```bash
nohup zsh tools/typist_chain.sh 8020 data/watch2_caregiver.jsonl "--days 6 --corpus data/watch2_corpus.json --planner queue --queue data/teach_queue_w2.jsonl --tick 0.15 --period 64 --quiet 8 --cap 40 --listen 40 --answer-levels 1 --parent 1 --reply 1 --wait 4" "TALKOVER_FROWN=1 FROWN_GAP=20 HABIT_TICKS=120 ANSWER_SMILE=1 SLOW_SHARE=0.33 SLOW_CPS=2.0 SLOW_PAUSE=0.1 TURN_ONLY_SMILE=1" 1000 > <SCRATCH>/logs/typist_chain_8020.log 2>&1 &
```

The arguments are `PORT LOG "TYPIST ARGS" "ENV" ROUNDS`, and the typist arguments and the environment must stay quoted. The typist (`body/teacher.py`) types the rows queued in `data/teach_queue_w2.jsonl` in two voices, a third of the first voice's lines one-handed (the `b:` voice's lines never). The face comes from the caregiver (`body/caregiver.py`), which reads only the page: a small smile for a known word in the child's turn, a larger one for the answer, and a frown when talked over.

When a typist's `--days` run out, the chain starts the next one with the day label taken from the log. A change to `body/teacher.py` or `body/caregiver.py` therefore reaches the typist only at a relaunch. Never check for the typist with its own pattern in a `pgrep`, because the chain waits while any process shows it. Use this instead:

```bash
ps -eo args | grep body.teacher | grep -v grep
```

### 3. The day and night loop

```bash
zsh ops/night_cycle.sh <SCRATCH> <FIRST_SAVE>
```

The usage line is from the script's header, and `ops/RESTART.md` step 3 describes the loop in full. The loop begins once the page log holds `FIRST_SAVE` save rows, and it prints one line per event.
- It keeps the queue fed while the body is awake.
- At each night it prints the day's three numbers (`ops/day_report.py`).
- After each save it runs the morning probe (`ops/probe_after_save5.sh`: the language probe, the questions and the rephrased set, with the morning save kept two deep).
- It queues the morning rows, the day's script, story rows (`ops/read_stories.py`) and revisits (`ops/revisit_rows.py`).

It reads two staged row files from `<SCRATCH>` (`morning_300.jsonl` and `day_358.jsonl`), and a cleared scratch directory loses them.

What the two voices say is written the way `ops/teaching_method.md` and `ops/parent_brief_human.txt` describe: conversations, not drills, with the child's words recast. `python3 ops/queue_depth.py` gives the true number of lines ahead of the typist; the `.pos` file does not.

A day's numbers for any window of time come from:

```bash
python3 ops/day_report.py 2026-09-19T17:12 2026-09-19T18:17 --label "day 344" [--pace SRC]
```

### 4. Changing the served body

Measure the change on a copy first. Then apply it at a night's save:
- `ops/reload_after_save.sh <SCRATCH> "FLAGS"` applies it at the next save after a night.
- `ops/reload_now.sh <SCRATCH> "FLAGS"` applies it mid-day, only when the change cannot wait.
- `ops/rekey_after_save.sh <SCRATCH> FLAGS_FILE` is for a change to the form of the memory's key (`key_ctx`, `ctx_form`, `key_form`). The store is rebuilt from the utterance memory first. Warning: this script prunes `data/backups/watch2/watch2_before_*.pt` down to two.

Run `ops/served_cfg.py` after each of them. The full restart procedure after a reboot is `ops/RESTART.md`. It was rewritten on 2026-09-23 for the night cycle; the guard and the night waiter are retired, and the dusk probe is armed only by hand.

`zsh ops/stop_all.sh` stops the chain and the typist, asks the body to save, and stops the server. It does not yet stop `ops/night_cycle.sh` or `ops/talk_proxy.py`; stop those by hand.

### 5. Talking to it: the demo page (port 8021)

```bash
python3 ops/talk_proxy.py --port 8021 --body 8020 --pause-after 600
```

Open one tab at http://localhost:8021.
- While a browser polls the page, the typist is stopped, and it resumes `--pause-after` seconds after the last poll or when the proxy exits.
- Each line is a bubble. Your letters flow in as you type them, with no box and no editing.
- The number keys are your face, pressed at least 0.4 s apart:
  - 5 is neutral. The body feels a change of face, not a held face, so go back to 5 after every smile.
  - A sensible word: 7, then 5.
  - An answer: 7, then 9, then 5.
  - A frown when it talks over you: 3, then 5.
- Type each line through without stopping (a pause of more than about 1.3 s splits it into two bubbles), then wait 2 to 4 s.

The body also serves the page at http://localhost:8020/talk, but without the pause: the parent comes back after about 40 s and types over you.

To end, press 5, close the tab and press Ctrl-C in the proxy's terminal. If the typist stays stopped, run `pkill -CONT -f "body.teacher --port 8020"`. Never `kill -9` the proxy.

`tools/teach_live.py` is the supervisor's chair for the same kind of sitting from the terminal.

### 6. The tests and the determinism check

```bash
python3 body/tests/test_organs.py
```

This runs 91 organ tests. Each one fails when its organ stops doing its job.

```bash
python3 tools/determinism_check.py [--flags ops/BASE_FLAGS.txt] [--ticks 400]
python3 tools/determinism_check.py --profile served [--full] [--roundtrip]     # also: --profile switches
```

Run it before and after any edit of `body/`. A tiny body at a fixed seed lives a fixed script, sleeps one night and is hashed. An equal digest means the edit changed nothing the body does on that script. Run it from the main tree. `--profile served` first takes the served save's own constants (read from its pickle, no tensors), so the 26 constants that live only in the save are exercised, and the tiny day ends at the tick's own sleep switch. `--profile switches` adds chunk_gate 1, utt_entry felt and pace_fore_q 0.99. `--full` also hashes every optimizer, the random streams, the saved blob and every working attribute, with a digest per section. `--roundtrip` saves and reloads halfway and names any field a reload does not give back (review §4).

## Map of the repo

- `body/`: the organism, and the only code the served body runs.
  - `model.py` holds the organs.
  - `life.py` is a life: `Life`, its `__init__` and its tick. Its other methods are mixins in `core/`, one module per role: senses, memory,
    cortex, mouth, critics, actor, night, persistence and instruments. The physiology table of constants is `core/physiology.py`,
    re-exported by `life.py`.
  - `serve.py` is the server and the `/talk` page.
  - `teacher.py` and `caregiver.py` are the typist and its face.
  - `fastlife.py` is the fast parent for fresh seeds. One test still uses it.
  - `tests/test_organs.py` holds the tests.
  - Edits here are checked with the determinism check. `ops/review_2026-09-22.md` §4 has the refactor plan.
- `tools/`: the supervisor's instruments, never the caregiver's. Each file's first lines say what it reads and how to run it (index: `tools/README.md`). A parent never reads this folder, because the held-out material lives here.
  - The rulers after every save are `probe_lm.py` and `qa_by_gap.py`. Their material is `heldout_stage4.txt`, `heldout_rephrased.txt`, `facts_stage5.txt` and `fact_prefixes.txt`.
  - Copies that live, sleep or are talked to: `inproc_parent.py` (the served parent teaching a copy in-process), `day_on_copy.py`, `night_copy.py`, `demo_rehearsal.py`, `silence_probe.py`, `slow_line_probe.py`, `once_told.py`, `live_qa.py`, `branch_probe.py`, `floor_cut.py` and `transplant.py`.
  - The store laid open: `read_trace.py`, `store_dist.py`, `store_turnover.py` and `key_separation.py`. `rekey_store.py` is the key rebuild.
  - The page log read back: `word_rate.py`, `served_day.py` and `word_mates.py`.
  - The only tools that act on the live body: `teach_live.py` and `rehearse.py`. Both hold the typist while they sit.
  - Also `determinism_check.py`, `typist_chain.sh`, `gauge_by_position.py` (the cortex alone, by position) and `faststore.py` (a shim that `rekey_store.py` and `key_separation.py` import).
  - The rest are retired and now live in `tools/archive/` (moved 2026-09-23). They include the earlier lineage's recorders and watchers, the body2-era probes, the pod pretraining, and `baseline_train.py`, which was never run, under the no-baseline law. They are kept as the ledger's record and are not run.
- `ops/`: what runs the served body day and night.
  - The constants and exact commands: `BASE_FLAGS.txt`, `serve_command.txt` and `typist_chain_command.txt`.
  - The loop: `night_cycle.sh`, `probe_after_save5.sh`, `day_report.py`, `read_stories.py`, `revisit_rows.py` and `queue_depth.py`.
  - The reloads: `reload_after_save.sh`, `reload_now.sh` and `rekey_after_save.sh`, with `served_cfg.py` as the check after each.
  - The demo page: `talk_proxy.py`. Stopping: `stop_all.sh`.
  - Documents: `RESTART.md`, `teaching_method.md`, `parent_brief_human.txt` and `review_2026-09-22.md`.
  - `archive/flags/` keeps every flag set the ledger names.
  - `archive/` holds the retired scripts: the night waiter, the pod scripts and the guard's old command line (`guard_args.txt`), moved 2026-09-23. The retired guard (`guard_tag_at_night.sh`) and the old brief (`parent_brief_template.txt`) are there too.
- `data/` (not in git): what the served body needs.
  - `watch2.pt`, the save, which every night overwrites.
  - `tok_char.json`, the tokenizer.
  - `watch2_caregiver.jsonl`, the page log.
  - `teach_queue_w2.jsonl` with its `.pos` and `.stories.pos` cursors.
  - `watch2_corpus.json`, the typist's known words.
  - `stories_valid.txt`, the stories the parent reads aloud by day.
  - `backups/watch2/`, the reload copies, kept two deep.
  - Everything else is earlier bodies and the first lineage, kept as history. `data/README.md` lists every file there and which ones the served body needs.
- `body/sim/`: the robot sim (the G1 world, its scene and room, the parent's body, poses, motion, face, feelings and voice, the lane,
  the lang/ conduct); `body/tests/test_sim_*.py` and `test_motor.py` its suites; `tools/sim_life.py` the life's runner,
  `tools/sim_parent_motion.py` the parent's harness, `tools/sim_profile.py` the pinned sim profile; `tools/pins/` the digests.
- `data/g1_seed1/` (not in git): the life, its pair saved at each dawn, its records and its voice cache.
- `docs/`: `SIM_DESIGN.md` and `audit/` (the humanoid: the robot sim project).
- The demo film (its tools, design, takes and renders) is kept on this machine only, never in git: see `.gitignore`.
- `logs/`: the server's log (not in git). `docs-private/`: not in git.
- `legacy/`: the first lineage (the 297M one-token organism, its gestation, its raising and its pod scripts), archived as it was (`legacy/README.md`).

The documents:
- `ARCHITECTURE.md`: the architecture in one read, covering the organs, the tick and the night.
- `BODY_SPEC.md`: the specification. It covers the world, the organs, the one reward, sleep, the disclosed constants, the mathematics with every defect found and its test (§5b), the environment, the instruments and the tests. It was last changed on 2026-09-17, so items 41 to 52 (`pace_sense` among them) are in the ledger but not yet in the spec.
- `ITERATIONS.md`: the ledger, items 1 to 52. Every mechanism is recorded there as derived, measured on copies, and then adopted at a night's save or falsified, with its ruler and its falsifier.
- `DIARY_BODY.md`: the diary of the raising, dated and in order.
- `ops/teaching_method.md` and `ops/parent_brief_human.txt`: how it is taught now. `CURRICULUM.md` is the earlier staged curriculum, which these two replace.
- `ops/review_2026-09-22.md`: the full review, with the confirmed defects, what stands between this body and a robot, and the refactor plan.

## The laws the project runs under

These are the user's standing rulings, dated in the ledger and the diary. A mechanism that breaks one is removed as soon as it is named.

1. **Grounded reward only.** The body's reward is the caregiver's face. Nothing authors or edits its words, and nothing reads its insides on the caregiver's behalf.
2. **No hand-written rules or cheats.** Only mechanisms that biology uses are allowed, and every constant is disclosed. A mechanism must be body-general: it is judged by whether it survives a change of body.
3. **No values fitted to the environment's pace** (2026-09-22). The manners must be learned or sensed by the architecture, not tuned. There are no special typist modes and no environment timed to the store. The shape of the environment is the user's call.
4. **Measure on copies, and change the served body only at a night's save.**
5. **The night stands.** There is no rollback: a night is kept whatever it does.
6. **REM stays on everywhere.** There are no ablations.
7. **No baseline runs.** The body is measured by its own rulers, never against ordinary training.
8. **No side seeds.** There is one seed, and the served body at speed is the experiment.
9. **All local.** It runs on this Mac, with no pods.
10. **Judge progress by sitting with it.** Progress is judged by talking with it live and reporting the conversation; the rulers come second.
11. **The parent is a human teacher.** The teacher listens every minute and answers what the child said, in two voices (the `b:` lines are the other voice), in conversations about the two of them rather than drills.
12. **Every finding is written down.** Every defect found goes into `BODY_SPEC.md` with its date and its test. Every mechanism goes into `ITERATIONS.md` with its ruler and its falsifier.

The robot sim's rulings, 2026-09-25 to 27 (dated in `docs/SIM_DESIGN.md`):

13. **The G1's reward is her face and its own pain.** No charge, no bottle, no charger; the objects in the room are for her teaching.
14. **The parent's method is the lead's to change; the room's shape is the owner's call.** A door that would keep a rolling child in the
    room is his; carrying a sleeping child back to its mat is hers.
15. **One change a boundary, on a copy first when in doubt.** A body change and a teacher change never on the same day; a change goes in
    at a dawn from the dawn's pair; every tree is gated by its suites and the pins before it lives.
16. **The lead builds the G1 directly**, and reports each day's ledger.
17. **The demo film is never in git.** Its tools, design, takes and renders stay on this machine.

## License

Apache License 2.0; see `LICENSE`. From `NOTICE`: iga — Imagination-Gated Agent, Copyright 2026 Luke Hamond.
