# iga — the body from nothing

A language organism raised from nothing by live conversation: one continuous stream, one grounded reward (the parent's face on the page), learning while it is spoken to, on one Mac. The diary body is the proving ground for a body-general architecture: a cortex, a ladder of eight bands at clocks 1 to 16384, a fast critic on a striatal input and a slow ventral critic, working memory, a planning actor, sleep with REM. No hand-written rules and no reading from inside: the parent sees the page and its face row and decides from that alone. The standing aim: an architecture that, scaled up, given a new body and new inputs, and a grounded reward, learns.

## Map

- `body/` — the organism. `life.py` (a life: the tick, the lessons, sleep, save and load), `model.py` (the organs), `fastlife.py` (the fast parent, for measurement on fresh seeds), `caregiver.py` and `teacher.py` (the served typist and its planner), `serve.py` (the page), `tests/test_organs.py` (59 organ tests, one per mechanism; each fails when its organ stops doing its job).
- `tools/` — the supervisor's instruments, never the caregiver's (one exception, named: `tools/rehearse.py` acts on the live body as a second parent for a rehearsal, typing and smiling; its numbers are rehearsals, not measurements): `watch_life.py` (a watched life from birth: per-tick logs, a digest per day), `play_by_play.py`, `watch_trend.py`, `diary_check.py`, `compare_credit.py`, `stalk_day.py`, `record_day.py`, `fit_return.py`, `gate_context.py`, `vf_profile.py`, `ceiling_smile.py`, `ceiling_line.py`, the probes, and the shell helpers (`boundary_restart.sh`, `typist_chain.sh`, `run_days.sh`, `teach_days.sh`).
- `BODY_SPEC.md` — the specification: the world, the organs, the one reward, sleep, the disclosed physiology, the mathematics (§5b, with every defect found and fixed), the environment, the instruments, the tests.
- `DIARY_BODY.md` — the diary of the raising, dated, in order.
- `CURRICULUM.md` — how the diary is taught: connections, not vocabulary.
- `data/` — bodies, logs, backups; not in git. `data/README.md` says what is where and what must never be touched.
- `docs-private/` — not in git.
- `legacy/` — the first lineage: the 297M one-token organism, its gestation, its raising program, its pod scripts and results, kept as they were (`legacy/README.md`).

## Run

The served body (the page on port 8018; always with `--period 0.25`):

```bash
python3 -m body.serve --load data/body2.pt --tok data/tok_char.json --port 8018 --period 0.25
```

The typist that talks to it (the reply parent: it waits for the child to fall quiet, and frowns when talked over):

```bash
TALKOVER_FROWN=1 python3 -m body.teacher --port 8018 --day N --days 6 --log data/body2_caregiver.jsonl --corpus data/body2_corpus.json --planner queue --queue data/teach_queue.jsonl --period 40 --quiet 3 --cap 12 --answer-levels 1 --parent 1 --reply 1 --wait 4
```

A watched life from birth, with the fast parent (one seed, every tick logged, a digest per day):

```bash
PARENT=1 REPLY=1 WAIT=4 TALKOVER_FROWN=1 python3 -u tools/watch_life.py data/NAME.pt --birth --seed 1 --d 1024 --layers 12 --heads 16 --window 64 --cfg-from data/body2.pt --set wm=1 --set actor=1 --set actor_form=plan --days 60 --threads 6
```

Talking to the served body yourself: open http://localhost:8020/talk (or, when the page in the code is newer than the running server, `python3 ops/talk_proxy.py` and http://localhost:8021). A line goes in one symbol a tick like the parents' lines; smile and frown are the two buttons or the arrow keys; the parent's typist steps back for a minute after a visitor types. Nothing on that page reads from inside.

The organ tests:

```bash
python3 -m body.tests.test_organs
```

## The rules of the house

- Grounded reward only: the parent's face, and the parent's act of speaking. Nothing authors or edits the body's words; nothing outside the body reads its insides.
- A served body takes a change of recipe or environment only at a day boundary, after measurement on fresh seeds.
- CPU, local, one thread per fast run; REM stays on everywhere.
- Every defect found goes into `BODY_SPEC.md` §5b and the diary with the date.
