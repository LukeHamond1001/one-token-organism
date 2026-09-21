"""A DAY ON A COPY (2026-09-20, late): the taught pairs typed to a copy as the typist types them (the question at the tick, its turn,
the other voice's answer, a pause), N rounds over the eight pairs, the copy learning as it lives (the store's writes, the gate's
lesson, the wake lesson on its own speech); then each question asked and the answers read for their key words, before the day
and after it. The served constants or any given on the line: the own-speech lesson's form is the one under test. Nothing saved.
usage: nice -n 19 python3 tools/day_on_copy.py COPY.pt --flags ops/BASE_FLAGS.txt --rounds 4 [--seed 0] [--own-target-form world]"""
import sys, re
sys.path.insert(0, "/Users/lukehamond/Projects/project")
import torch
from tokenizers import Tokenizer
from body.life import Life, PHYSIOLOGY

def arg(name, default):
    names = {f"--{name}", f"--{name.replace('_', '-')}"}
    for i, a in enumerate(sys.argv[1:], 1):
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

path = sys.argv[1]
cfg = parse_flags(open(arg("flags", "")).read()) if arg("flags", "") else {}
for a in sys.argv[1:]:
    if a.startswith("--") and "=" not in a:
        k = a[2:].replace("-", "_")
        if k in PHYSIOLOGY:
            cfg[k] = type(PHYSIOLOGY[k])(sys.argv[sys.argv.index(a) + 1])
seed = arg("seed", 0); rounds = arg("rounds", 4)
TOK = Tokenizer.from_file("/Users/lukehamond/Projects/project/data/tok_char.json")
life = Life.load(path, TOK, device="cpu", cfg=cfg, seed=seed); life.save_path = None; life.m.eval()
PAIRS = [("what is sour?", "a lemon is sour", "lemon"), ("who gives us eggs?", "the hen gives us eggs", "hen"), ("is the egg hard or soft?", "the egg is hard", "egg is hard"),
         ("what do the ducks eat?", "bugs and bread", "bugs"), ("who gives us milk?", "the cow gives us milk", "cow"), ("what is sweet?", "a pear is sweet", "pear"),
         ("what does the hen say?", "cluck cluck", "cluck"), ("what is round?", "the ball is round", "ball")]
def own_between(p0, p1): return "".join(e[0] for e in life.page[p0:p1] if e[1] == 1 and e[0])
def ask_all(label):
    hits = 0; out = []
    for q, a, key in PAIRS:
        life.type_text(q, "parent")
        while life.queue: life.tick()
        p0 = len(life.page)
        for _ in range(40): life.tick()
        turn = own_between(p0, len(life.page)); ok = key in turn.lower(); hits += ok; out.append(f"{q.split()[-1].strip('?')}:{'Y' if ok else '-'}")
        for _ in range(40): life.tick()
    print(f"  {label}: {hits} of {len(PAIRS)} carry the key word | {' '.join(out)}", flush=True)
    return hits
print(f"body {path.split('/')[-1]} nights {life.nights} | own_target_form {life.cfg.get('own_target_form')} own_gain {life.cfg.get('own_gain')} own_target_conf {life.cfg.get('own_target_conf')} | {rounds} rounds over {len(PAIRS)} pairs, seed {seed}", flush=True)
with torch.no_grad():
    for _ in range(40): life.tick()
before = ask_all("before the day")
for r in range(rounds):                                                   # the day: the pairs as the typist types them, the copy learning
    for q, a, key in PAIRS:
        life.type_text(q, "parent")
        while life.queue: life.tick()
        for _ in range(30): life.tick()                                   # its turn
        life.type_text(a, "other")
        while life.queue: life.tick()
        for _ in range(30): life.tick()                                   # the pause
    print(f"  round {r + 1} typed ({life.ticks} ticks)", flush=True)
after = ask_all("after the day")
print(f"RESULT own_target_form {life.cfg.get('own_target_form')}: {before} of {len(PAIRS)} before the day, {after} of {len(PAIRS)} after {rounds} rounds", flush=True)
