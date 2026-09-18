# ops — the served body's operations

What runs the served body day and night, and nothing that reads from inside it on the caregiver's behalf.

Live files:
- `BASE_FLAGS.txt` — the served constants as flags, one line; the source of truth beside the save (`served_cfg.py` prints a save's effective constants against it). `guard_args.txt` (the guard's exact command line) and `serve_command.txt` (the exact serve command last used) carry the same flags and must stay identical to it. `typist_chain_command.txt` — the typist chain's quoted command.
- `RESTART.md` — the restart procedure after a reboot, and the rule for every reload (explicit values for every changed constant; then `served_cfg.py`).
- `reload_after_save.sh` — a reload of the served body at the next post-night save with given flags (the boundary at which constants change). `rekey_after_save.sh` — the same boundary with the store rebuilt from the utterance memory first (a change of the key's form needs it; night 262). `guard_tag_at_night.sh` — the guard on the long tag, re-armed after every night. `probe_at_dusk.sh` — the dusk save and its probe. `probe_after_save5.sh` — the morning probe after each night's save (the language probe, every prefix and question listed, the rephrased set; keeps the morning save two deep). `night_waiter.sh` — one per night: re-arms the guard and the dusk probe, tallies the last days, waits for the morning probe, runs the branch. `stop_all.sh` — stop everything cleanly for a reboot.
- `queue_depth.py` — the lines ahead of the typist (the only depth a parent may read). `day.py` — a day's tally from the page log. `served_cfg.py` — a save's constants against the flags. `talk_proxy.py` — the visitor's page served beside the body (the same page the body serves at `/talk`).
- `parent_brief_template.txt` — the brief a parent agent is spawned with.

`archive/` holds what the ledger names and no longer runs: `archive/flags/` the kept flag sets (each the served set before or after a change named in ITERATIONS.md), and the earlier generations of the morning probe and the typist relaunch.
