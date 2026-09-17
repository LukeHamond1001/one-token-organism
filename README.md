# iga — the body from nothing

A language organism raised from nothing by live conversation: one continuous stream, one grounded reward (the parent's face on the page), learning while it is spoken to, on one Mac. The served body is a 179M-parameter character-level organism with no text corpus at any point in its life: a cortex, a hippocampal store that writes what surprises it and replays it at night, a ladder of eight value bands at clocks 1 to 16384, a fast critic on a striatal input and a slow ventral critic, working memory, a planning actor, a gate that decides when to speak, sleep with REM. No hand-written rules and no reading from inside: the parent sees the page and its face row and decides from that alone. The standing aim: an architecture that, scaled up, given a new body and new inputs, and a grounded reward, learns.

The claim, as the rulers carry it (the numbers by date are in the diary): taught thirty facts as conversational exchanges, it answers about two thirds of their questions on a good morning; a fact told to it once, in a single exchange, in words its caregivers never typed, is answered minutes later and, less reliably, after a night's sleep; with the hippocampal store switched off the same questions answer none. The one-shot memory is hippocampal, not cortical; what sleep contributes is measured on never-replayed held-out lines.

## Map

- `body/` — the organism. `life.py` (a life: the tick, the lessons, sleep, save and load), `model.py` (the organs), `serve.py` (the page and the visitor's page at `/talk`), `caregiver.py` and `teacher.py` (the served typist and its planner), `fastlife.py` (the fast parent, for measurement on fresh seeds), `tests/test_organs.py` (64 organ tests, one per mechanism; each fails when its organ stops doing its job).
- `ops/` — what runs the served body day and night: the flags that are its constants, the restart procedure, the reload at a save, the guard, the dusk and morning probes, the night waiter, the parent's brief, the visitor's page served beside the body (`ops/README.md`).
- `tools/` — the supervisor's instruments, never the caregiver's; each file's first lines say what it reads and how to run it:
  - the rulers on a saved copy: `probe_lm.py` (the language probe: the held-out lines, the fact prefixes as cued recall, the questions at two pauses), `qa_by_gap.py` (the answer by the pause; `--set=rephrased` the questions in words no parent typed), `branch_probe.py` (the branch after a shared start), `once_told.py` (a fact told once, asked later and after a night), `live_qa.py` (the live mouth's form on a copy), `gauge_by_position.py`, `sequence_probe.py`;
  - copies that live or sleep: `day_on_copy.py`, `night_copy.py`, `night_lab.py`, `dream_lived.py`, `floor_cut.py`, `transplant.py`; `baseline_train.py` exists and has not been run on this body's claim, by the house rule against ordinary-training comparisons;
  - the store laid open: `read_trace.py` (which memories carry a read), `store_dist.py`, `store_turnover.py`;
  - the live body, the one exception: `rehearse.py` acts on the served body as a second parent for a rehearsal, typing and smiling; its numbers are rehearsals, not measurements;
  - the page log read back: `served_day.py`, `diary_check.py`, `word_mates.py`, `play_by_play.py`, `watch_trend.py`, `compare_credit.py`;
  - the earlier lineage's watchers and stalkers: `watch_life.py`, `stalk_day.py`, `record_day.py`, `fit_return.py`, `gate_context.py`, `vf_profile.py`, `ceiling_smile.py`, `ceiling_line.py`; the shell helpers (`typist_chain.sh`, which runs the served typist day after day; `boundary_restart.sh`, `run_days.sh`, `teach_days.sh`);
  - the held-out material, never typed by a parent: `heldout_stage4.txt`, `heldout_rephrased.txt`; the facts and their prefixes: `facts_stage5.txt`, `fact_prefixes.txt`. A parent agent never reads this folder.
- `BODY_SPEC.md` — the specification: the world, the organs, the one reward, sleep, the disclosed physiology, the mathematics (§5b, with every defect found and fixed, thirty-one so far), the environment, the instruments, the tests.
- `ITERATIONS.md` — the ledger: every mechanism derived, measured on copies, adopted at a boundary or falsified, with its ruler and its falsifier.
- `DIARY_BODY.md` — the diary of the raising, dated, in order: every night, every parent's report, every incident.
- `CURRICULUM.md` — how the body is taught: connections, not vocabulary. `DEMO_SCRIPT.md` — the one-minute take, scene by scene, and the claim phrased to survive a skeptic. `VIDEO_10MIN.md` — the ten-minute video's script.
- `data/` — bodies, logs, backups; not in git. `data/README.md` says what is where and what must never be touched. `logs/` — the served body's own log; not in git. `docs-private/` — not in git.
- `legacy/` — the first lineage: the 297M one-token organism, its gestation, its raising program, its pod scripts and results, kept as they were (`legacy/README.md`).

## Run

The served body (port 8020, one symbol per tick at 0.2 s; its constants are the flags file, and a save carries its own constants, so every changed constant is passed explicitly — see `ops/RESTART.md`):

```bash
nohup python3 -m body.serve --load data/watch2.pt --tok data/tok_char.json --port 8020 $(tr '\n' ' ' < ops/BASE_FLAGS.txt) > logs/serve_watch2.log 2>&1 &
```

The typist that talks to it, day after day, from the queue a parent agent appends to (`data/teach_queue_w2.jsonl`): the exact quoted command is `ops/typist_chain_command.txt`. The nightly cycle (the guard, the dusk and morning probes, the night waiter, the parent's brief) is `ops/RESTART.md`.

Talking to it yourself: open http://localhost:8020/talk. Each line is a bubble, its parent in brown, the other voice in green, you in blue, the body in black; a line you type goes in one symbol a tick like its parents' lines; smile and frown are the buttons or the arrow keys; both faces run tick by tick at the top; the parent's typist steps back for a minute after you type. When the page in the code is newer than the running server, `python3 ops/talk_proxy.py` serves the same page on http://localhost:8021 beside the body. Nothing on that page reads from inside.

The organ tests:

```bash
python3 body/tests/test_organs.py
```

A fresh body under the fast parent, every tick logged, a digest per day (the earlier recipe's example):

```bash
PARENT=1 REPLY=1 WAIT=4 TALKOVER_FROWN=1 python3 -u tools/watch_life.py data/NAME.pt --birth --seed 1 --d 1024 --layers 12 --heads 16 --window 64 --cfg-from data/watch2.pt --days 60 --threads 6
```

## The rules of the house

- Grounded reward only: the parent's face, and the parent's act of speaking. Nothing authors or edits the body's words; nothing outside the body reads its insides on the caregiver's behalf.
- Biology's mechanisms only, every constant disclosed; a mechanism is judged by whether it survives a change of body.
- The served body takes a change of constant or code only at a night's save, after measurement on copies; a change of forgetting is judged over the store's turnover, never over three mornings. A night, once slept, stands.
- No ordinary-training baseline: the body is measured by its own rulers. No special typist modes and no environment timed to the store.
- CPU, local, one copy run at a time beside the served body; REM stays on everywhere.
- Every defect found goes into `BODY_SPEC.md` with its date and its test; every mechanism into `ITERATIONS.md` with its ruler and its falsifier.
