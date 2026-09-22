"""THE BEHAVIOUR GUARD (a supervisor's instrument, 2026-09-17): a tiny body born at a fixed seed under the served constants lives a
fixed script (lines typed as the parents type them, faces at fixed ticks), sleeps one night and lives on; every weight, the store,
the utterance memory, the page and the feelings are hashed. Run before and after an edit of body/: an equal digest means the edit
changed nothing the body does on that script; a different digest means it did. Imports the body from its own tree, so it can guard
a worktree. usage: python3 tools/determinism_check.py [--flags ops/BASE_FLAGS.txt] [--ticks 400]"""
import sys, os, hashlib
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, HERE)
import torch
from tokenizers import Tokenizer
from body.life import Life, PHYSIOLOGY

def arg(name, default):
    for i, a in enumerate(sys.argv[1:], 1):
        if a == "--" + name and i + 1 < len(sys.argv): return type(default)(sys.argv[i + 1])
    return default
def parse_flags(path):
    toks = open(path).read().split(); cfg = {}; i = 0
    while i < len(toks):
        t = toks[i]
        if t.startswith("--"):
            k = t[2:].replace("-", "_")
            if k in PHYSIOLOGY: cfg[k] = type(PHYSIOLOGY[k])(toks[i + 1])
            i += 2
        else: i += 1
    return cfg

flags = arg("flags", os.path.join(HERE, "ops", "BASE_FLAGS.txt")); ticks = arg("ticks", 400)
cfg = parse_flags(flags) if os.path.exists(flags) else {}
cfg.update(dict(night_starts=64, night_starts_max=64, night_rounds=2, night_batch=8, rem_dreams=4, rem_steps=4))
cfg["night_dev"] = str(arg("night_dev", ""))   # the guard stays a CPU guard whatever the flags carry; --night_dev mps makes it a device smoke test   # a night a tiny body can afford
TOK = Tokenizer.from_file("/Users/lukehamond/Projects/project/data/tok_char.json")
torch.manual_seed(0)
life = Life.birth(TOK, device="cpu", d=64, layers=2, heads=2, window=32, cfg=cfg, seed=0)
SCRIPT = [("what do you want?", "parent"), ("I want milk", "other"), ("do you see the ball?", "parent"), ("yes. the ball is red", "other"),
          ("what is cold?", "parent"), ("ice is cold", "other"), ("shall we go out?", "parent"), ("no. it is wet out", "other"),
          ("what do bees make?", "parent"), ("bees make honey", "other"), ("are you warm now?", "parent"), ("I am warm here", "other")]
FACES = {23: 2.0, 24: 2.0, 25: 0.0, 61: -2.0, 62: 0.0, 140: 2.0, 141: 4.0, 142: 0.0, 230: 2.0, 231: 0.0, 333: -2.0, 334: 0.0}
def run(n):
    j = 0
    for t in range(n):
        if t % 30 == 0 and j < len(SCRIPT):
            life.type_text(SCRIPT[j][0], who=SCRIPT[j][1]); j += 1
        if t in FACES: life.set_face(FACES[t])
        life.tick()
run(ticks)
rep = life.night()                                   # as the tick's sleep switch calls it: with grad, the night's own optimizer
assert not rep.get("error"), rep.get("error")
run(100)
h = hashlib.sha256()
def add(name, x):
    if torch.is_tensor(x): h.update(name.encode()); h.update(x.detach().cpu().contiguous().numpy().tobytes())
    else: h.update(name.encode()); h.update(repr(x).encode())
for k, v in sorted(life.m.state_dict().items()): add(k, v)
st = life.store
for k in ("K", "V", "S", "W", "N", "NE", "B", "Bs", "Bq", "A"): add("store." + k, getattr(st, k))
add("utts", [list(u) for u in life.utts]); add("utt_S", list(life.utt_S))
add("page", "".join(e[0] for e in life.page)); add("feel", (round(float(life.mood), 6), round(float(life.fatigue), 6), round(float(life.stress), 6), int(life.ticks), int(life.nights)))
print(f"digest {h.hexdigest()[:24]} | ticks {life.ticks} nights {life.nights} store {st.n()} utts {len(life.utts)} own symbols {sum(1 for e in life.page if e[1] == 1 and e[0])} | dreams {rep.get('dreams')} dropped {rep.get('store_dropped')}")
