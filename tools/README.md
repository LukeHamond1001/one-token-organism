# tools — the supervisor's instruments

Read-only on the served body: every instrument here loads a SAVED COPY of the body (or reads its page log) and never saves back,
with one named exception. Each file's first lines say what it reads, what it prints and how to run it. A parent agent never reads
this folder: the held-out material lives here.

The rulers read on a morning save (what the diary reports every night):
- `probe_lm.py` — the language probe: the held-out lines' accuracy (never typed by a parent), the fact prefixes completed (cued recall) and the thirty questions answered at two pauses (`--qa`, `--qa-all` for every item).
- `qa_by_gap.py` — the answer by the pause, the store read along the question; `--set=rephrased` the questions in words no parent typed.
- `branch_probe.py` — the branch: asked a fact whose first words are shared with others, does the mouth continue with the right one.
- `once_told.py` — a fact told once as an exchange, asked minutes later and after a night.
- `live_qa.py` — the live mouth's form (the gate and the sampled choice) on a copy. `gauge_by_position.py`, `sequence_probe.py` — finer readings of the cortex alone.

Copies that live or sleep (the measurement of a change before it goes on the served body):
- `day_on_copy.py` (a day under the served constants), `night_copy.py` (a night; `--save-as` keeps the copy for the rulers), `night_lab.py`, `dream_lived.py` (what the dreams replay), `floor_cut.py` (a forgetting floor applied at once), `transplant.py` (one organ from a donor save).
- `baseline_train.py` exists and has not been run on this body's claim, by the house rule against ordinary-training comparisons.

The store laid open: `read_trace.py` (which memories carry a read, a prefix or every question, on the real read), `store_dist.py` (the strengths), `store_turnover.py` (two mornings: what was written, dropped and strengthened; the steady state under a floor).
The key's geometry and the store rebuilt (item 40): `key_separation.py` — an empty store written from the last days' typed lines under a chosen form of the key (`--forms ctx:lam:swap`), the facts heard once, each asked in the taught and the rephrased wording: onset hits, chain accuracy, margins in nats. `rekey_store.py` — a copy's store rebuilt from its utterance memory under the code's key (the reconsolidation a change of key needs; the one instrument that writes a copy, `--save-as`; ops/rekey_after_save.sh runs it on the served save at a post-night save). `faststore.py` — the body's Store with room kept ahead and a copy-free eviction, proved identical, for the two above; never the body's.

The behaviour guard: `determinism_check.py` — a tiny body at a fixed seed on a fixed script; its digest must not change under an edit of body/.

The one exception: `rehearse.py` acts ON THE SERVED BODY as a second parent for a rehearsal (the typist frozen, questions typed, smiles given); its numbers are rehearsals, not measurements.

The page log read back: `served_day.py` (a day's digest), `diary_check.py` (every organ's day from the log), `word_mates.py`, `play_by_play.py`, `watch_trend.py`, `compare_credit.py`.

The earlier lineage's watchers and stalkers (a body under the fast parent, every tick logged): `watch_life.py`, `stalk_day.py`, `record_day.py`, `fit_return.py`, `gate_context.py`, `vf_profile.py`, `ceiling_smile.py`, `ceiling_line.py`, `probe.py`; the shell helpers `typist_chain.sh` (runs the served typist day after day; live), `boundary_restart.sh`, `run_days.sh`, `teach_days.sh`.

The material: `heldout_stage4.txt` and `heldout_rephrased.txt` (never typed by a parent), `facts_stage5.txt` (the thirty facts as question | answer) and `fact_prefixes.txt` (their prefixes, cued recall).
