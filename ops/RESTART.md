# Restarting the served life after a reboot (written 2026-09-14 18:30)

Everything the served body needs is in this directory; the scratchpad under /private/tmp may not survive a reboot.

1. Serve the body (from the project root; the flags are the served constants, key_ctx 0.5 since day 206):
   nohup python3 -m body.serve --load data/watch2.pt --tok data/tok_char.json --port 8020 $(tr '\n' ' ' < ops/BASE_FLAGS.txt) > logs/serve_watch2.log 2>&1 &
   (the exact command last used is in ops/serve_command.txt)
2. The typist chain: the exact quoted command is in ops/typist_chain_command.txt (PORT LOG "TYPIST ARGS" "ENV" ROUNDS; the args and env must stay quoted).
   The typist relaunches at each night with the day label from the log.
3. The guard on the long tag, re-armed after every night row:  nohup nice -n 5 zsh ops/guard_tag_at_night.sh <SCRATCH> <flags> &  (ops/guard_args.txt holds the exact line; replace the scratch path).
4. The dusk probe:  nohup nice -n 5 zsh ops/probe_at_dusk.sh <SCRATCH> probe_dusk_labelNNN.log &   (never arm near the sleep threshold).
5. The probe after each night's save:  nohup nice -n 5 zsh ops/probe_after_save2.sh <SCRATCH> <save-count> probe_after_nightNNN.log &  (the save count is the number of "save" rows in the log; night N's save is row N-24).
6. A parent: the Agent brief in ops/parent_brief_template.txt, with the night count (grep -c '"action": "night"' data/watch2_caregiver.jsonl), the day label, the facts schedule (ten a day, rotating), and the depth script ops/queue_depth.py (never the .pos file).
The scripts expect the scratch directory's logs/ subfolder; create it (mkdir -p <SCRATCH>/logs) and point them at any directory.
