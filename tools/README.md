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

The simulated world's instruments (docs/SIM_DESIGN.md 11: W1, W2, W3; never the body, and nothing of the language body): `sim_babble.py`
(the babbler, smooth random acts in movement units for the world's tests; run, the world's speed on this Mac, `--eyes` with the
eyes), `sim_sink.py` (how fast each posture gives way under the resting servo law), `sim_friction.py` (the friction model at
impratio 1 and 10: creep below the sliding force, the slide above it against Coulomb's, the normal force of a pressed contact
starting to slide, and the G1's hip housings under the newborn's flexion with and without friction: C5, C22, C26, the impratio
choice), `sim_grasp.py` (the G1 study's grasp trials on the built room, at a given impratio and toy scale: 3.8, B1, C28),
`sim_eye_check.py` (the fovea's identity on the ten toys and the face under three lights,
with and without the sun's shadow; `--c2 N`, the face test's rays against a segmentation render on babbled frames with the parent
at 0.3-3.5 m with her head kept inside the room, and the born face template's hits, false alarms and best correlation: C2, C3),
`sim_face_template.py` (the born face template on the parent's face at 0.3-2 m in the fovea and the periphery under the room's
lights, and in her attending pose, each match told as a detection of her face or a chance match, with an upside-down control: C3),
`sim_face_photometry.py` (her face's feature contrasts in CIE L* against a young woman's, Russell et al. 2017, photographed under a
studio's frontal light, and the calibration that set her iris's, brows' and lips' colours: 4.1, C3), `sim_face_measure.py` (her face
measured on the drawn geometry against its norms; `--collision`, a hand's rays at her face against her collision shapes; `lids`,
her moving lids against her skin and her eye in nine lid states: 4.1), `sim_pain.py` (pain under babble, the contact pairs that carry it, each withdrawal (the
newborn's flexion) against resting and the babble from the same state, either tick counted and each apart, and phantom pain by the
filter, on still ticks and at rest: C5, C18, C22), `sim_parent_motion.py` (W2: the parent's acts one by one from fresh worlds: each
act's status, reason and ticks, her reach error, every hold's peak force against its cap, her effort against her caps, her body's
contacts with the child and her yields, what her acts did to the G1 (a lift, a slide, a sit-up), her cost a tick, and per physics
step her holds and body together against her caps, her 10 ms force on each link of the child, her runs of pressing it; the W2
verifier's cases: `babble_attend` and `babble_acts` (her acts asked over and over while the G1 babbles: how many she gives up, C8),
`still_*` (a still child, as born or placed elsewhere: getting up and going on), `hands` (her hands against the G1's hulls), `catch`
(C6: falls from hovering under babble, caught or not), `copy_do` (P3's DOES and copy); `guide_pace`, the guide's peak force
against the arm's own push at each candidate pace on the limp arm and under the resting law; since she is a body (2026-09-25),
`c8` (C8 over 12 babbler seeds at p_rest 0.3 and 0.6, attend alone and the mix: her 10 ms force on the child with her holds, each
tick over F_pain told as hers or the child's by who did work on that link in those 10 ms, her hands', forearms' and body's
contacts' depth, her body's work on it in a tick and over each contact, her acts done and refused) and `--replay=save|load|birth:PATH`
(the exact replay across three processes): 4.2, A4-A10, A22, A25, C6, C7, C8, C34). `sim_look.py` renders the one-arm high chair's
stills; `sim_look_g1.py` the G1 room's (main's prototype stills).

`archive/` (moved 2026-09-23) holds the retired tools, kept as the ledger's record and not run: the earlier lineage's watchers and stalkers (a body under the fast parent, every tick logged: `watch_life.py`, `stalk_day.py`, `record_day.py`, `fit_return.py`, `gate_context.py`, `vf_profile.py`, `ceiling_smile.py`, `ceiling_line.py`, `play_by_play.py`, `watch_trend.py`, `compare_credit.py`), the body2-era readers and probes (`diary_check.py`, `probe.py`, `sequence_probe.py`, `boundary_restart.sh`, `run_days.sh`, `teach_days.sh`), the settled night and gate questions (`night_lab.py`, `dream_lived.py`, `norm_probe.py`), the pod pretraining (`pretrain_cortex.py`, item 44) and `baseline_train.py` (never run, by the house rule against ordinary-training comparisons). Those that find the repo by `dirname(dirname(__file__))` (`watch_life.py`, `stalk_day.py`, `record_day.py`, `pretrain_cortex.py`) would now look inside tools/ and need that path changed before they could run again.

The material: `heldout_stage4.txt` and `heldout_rephrased.txt` (never typed by a parent), `facts_stage5.txt` (the thirty facts as question | answer) and `fact_prefixes.txt` (their prefixes, cued recall).
