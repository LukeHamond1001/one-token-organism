# tools — the supervisor's instruments

Read-only on the served body: every instrument here loads a SAVED COPY of the body (or reads its page log) and never saves back,
with two named exceptions. Each file's first lines say what it reads, what it prints and how to run it. A parent agent never reads
this folder: the held-out material lives here.

The rulers read on a morning save (what the diary reports every night):
- `probe_lm.py` — the language probe: the held-out lines' accuracy (never typed by a parent), the fact prefixes completed (cued recall) and the thirty questions answered at two pauses (`--qa`, `--qa-all` for every item).
- `qa_by_gap.py` — the answer by the pause, the store read along the question; `--set=rephrased` the questions in words no parent typed.
- `branch_probe.py` — the branch: asked a fact whose first words are shared with others, does the mouth continue with the right one.
- `once_told.py` — a fact told once as an exchange, asked minutes later and after a night.
- `live_qa.py` — the live mouth's form (the gate and the sampled choice) on a copy. `gauge_by_position.py` — a finer reading of the cortex alone.

Copies that live or sleep (the measurement of a change before it goes on the served body):
- `inproc_parent.py` (the served parent teaching a copy in-process, on a virtual clock; `inproc_parent_check.py` is its self-check on tiny newborns), `day_on_copy.py` (a day under the served constants), `night_copy.py` (a night; `--save-as` keeps the copy for the rulers), `floor_cut.py` (a forgetting floor applied at once), `transplant.py` (one organ from a donor save).
- The demo's measures on a copy: `demo_rehearsal.py` (the demo sequence typed one-handed), `silence_probe.py` (the chatter in a thinking silence), `slow_line_probe.py` (the words said over a slow line, item 45).

The store laid open: `read_trace.py` (which memories carry a read, a prefix or every question, on the real read), `store_dist.py` (the strengths), `store_turnover.py` (two mornings: what was written, dropped and strengthened; the steady state under a floor).
The key's geometry and the store rebuilt (item 40): `key_separation.py` — an empty store written from the last days' typed lines under a chosen form of the key (`--forms ctx:lam:swap`), the facts heard once, each asked in the taught and the rephrased wording: onset hits, chain accuracy, margins in nats. `rekey_store.py` — a copy's store rebuilt from its utterance memory under the code's key (the reconsolidation a change of key needs; the one instrument that writes a copy, `--save-as`; ops/rekey_after_save.sh runs it on the served save at a post-night save). `faststore.py` — the body's Store with room kept ahead and a copy-free eviction, proved identical, for the two above; never the body's.

The behaviour guard: `determinism_check.py` — a tiny body at a fixed seed on a fixed script; its digest must not change under an edit of body/. `--profile served` (the served save's constants, read from its pickle, then the flags; the day ends at the sleep switch), `--profile switches` (chunk_gate 1, utt_entry felt, pace_fore_q 0.99), `--full` (optimizers, random streams, the saved blob, every working attribute; a digest per section) and `--roundtrip` (saved and reloaded halfway; the fields a reload does not give back are named); `--cfg` takes the served constants from a frozen pickle of the save's cfg instead of the live save. Run it from the main tree. `pins/` holds the core refactor's guard (docs/SIM_DESIGN.md 8.3): `served_cfg.pkl` (the served save's cfg, frozen) and `digests.txt` (the eight digests every refactor commit must reproduce, and how each is run).

The two exceptions act ON THE SERVED BODY and hold the typist (SIGSTOP) while they sit: `rehearse.py`, a second parent for a rehearsal (questions typed, smiles given; its numbers are rehearsals, not measurements), and `teach_live.py`, the teacher's chair for sitting with the body live (the user's "judge by conversation").

The page log read back: `word_rate.py` (the share of the child's runs of letters that are words its parents typed), `served_day.py` (a day's digest), `word_mates.py`.

The shell helper `typist_chain.sh` runs the served typist day after day (live).

The simulated world's instruments (docs/SIM_DESIGN.md 11: W1, W3; never the body, and nothing of the language body): `sim_babble.py`
(the babbler, smooth random acts in movement units for the world's tests; run, the world's speed on this Mac, `--eyes` with the
eyes), `sim_sink.py` (how fast each posture gives way under the resting servo law), `sim_friction.py` (creep below the sliding
force and the slide at mu x weight, C26), `sim_eye_check.py` (the fovea's identity on the ten toys and the face under three lights,
with and without the sun's shadow, and the face test's rays against a segmentation render: C2, C3). `sim_look.py` renders the
one-arm high chair's stills.

`archive/` (moved 2026-09-23) holds the retired tools, kept as the ledger's record and not run: the earlier lineage's watchers and stalkers (a body under the fast parent, every tick logged: `watch_life.py`, `stalk_day.py`, `record_day.py`, `fit_return.py`, `gate_context.py`, `vf_profile.py`, `ceiling_smile.py`, `ceiling_line.py`, `play_by_play.py`, `watch_trend.py`, `compare_credit.py`), the body2-era readers and probes (`diary_check.py`, `probe.py`, `sequence_probe.py`, `boundary_restart.sh`, `run_days.sh`, `teach_days.sh`), the settled night and gate questions (`night_lab.py`, `dream_lived.py`, `norm_probe.py`), the pod pretraining (`pretrain_cortex.py`, item 44) and `baseline_train.py` (never run, by the house rule against ordinary-training comparisons). Those that find the repo by `dirname(dirname(__file__))` (`watch_life.py`, `stalk_day.py`, `record_day.py`, `pretrain_cortex.py`) would now look inside tools/ and need that path changed before they could run again.

The material: `heldout_stage4.txt` and `heldout_rephrased.txt` (never typed by a parent), `facts_stage5.txt` (the thirty facts as question | answer) and `fact_prefixes.txt` (their prefixes, cued recall).
