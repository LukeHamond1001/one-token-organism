"""A DAY ON A COPY (a supervisor's instrument, 2026-09-14): load a saved body on a copy with the served constants and any override,
then feed it a day's typed lines from the page log in order, as the typist did (the line's symbols, then the child's turn of
`listen` ticks in which the mouth speaks and the store writes as awake), and save the copy: the day lived again under a changed
constant, for the rulers to read. Never saves back.
usage: python3 tools/day_on_copy.py BODY.pt --flags FLAGS.txt --day 199 --save-as OUT.pt [--listen 24] [--bag-decay 0.9] [--max-lines 0]"""
import sys, os, json, time
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT)
import torch
from tokenizers import Tokenizer
from body.life import Life, PHYSIOLOGY

def arg(name, default):
    names = {f"--{name}", f"--{name.replace('_', '-')}"}
    for i, a in enumerate(sys.argv[1:], 1):
        if "=" in a and a.split("=", 1)[0] in names:
            return type(default)(a.split("=", 1)[1])
        if a in names and i + 1 < len(sys.argv):
            return type(default)(sys.argv[i + 1])
    return default

def parse_flags(s):
    toks = s.split(); cfg = {}; i = 0
    while i < len(toks):
        t = toks[i]
        if t.startswith("--"):
            k = t[2:].replace("-", "_")
            if k in PHYSIOLOGY:
                cfg[k] = type(PHYSIOLOGY[k])(toks[i + 1])
            i += 2
        else:
            i += 1
    return cfg

path = sys.argv[1]; day = arg("day", 0); listen = arg("listen", 24); save_as = arg("save_as", ""); max_lines = arg("max_lines", 0)
cfg = parse_flags(open(arg("flags", "")).read()) if arg("flags", "") else {}
for k in list(PHYSIOLOGY):                                                # any physiology constant may be overridden on the line
    v = arg(k, None) if False else None
for a in sys.argv[1:]:
    if a.startswith("--") and "=" not in a:
        k = a[2:].replace("-", "_")
        if k in PHYSIOLOGY:
            cfg[k] = type(PHYSIOLOGY[k])(sys.argv[sys.argv.index(a) + 1])
TOK = Tokenizer.from_file(os.path.join(ROOT, "data/tok_char.json"))
life = Life.load(path, TOK, device="cpu", cfg=cfg); life.save_path = None; m = life.m
R = [json.loads(l) for l in open(os.path.join(ROOT, "data/watch2_caregiver.jsonl")) if l.strip()]
lines = [(r["text"], "other" if r.get("voice") == "b" else "you") for r in R if r.get("day") == day and r["action"] in ("line", "cue")]
if max_lines: lines = lines[:max_lines]
print(f"body {os.path.basename(path)}: nights {life.nights} store {life.store.n()} | bag_decay {life.cfg['bag_decay']} read_follow {life.cfg.get('read_follow')} | day {day}: {len(lines)} lines, listen {listen}", flush=True)
t0 = time.time(); ticks = 0
with torch.no_grad():
    pass
for i, (text, who) in enumerate(lines):
    life.type_text(text, who=who)
    while life.queue:
        life.tick(); ticks += 1
    for _ in range(listen):
        life.tick(); ticks += 1
    if (i + 1) % 40 == 0:
        print(f"  {i + 1} lines, {ticks} ticks, {time.time() - t0:.0f}s, store {life.store.n()}", flush=True)
print(f"done: {len(lines)} lines in {ticks} ticks, {time.time() - t0:.0f}s", flush=True)
if save_as:
    life.save_path = save_as; life.save(); life.save_path = None; print(f"saved as {save_as}", flush=True)
