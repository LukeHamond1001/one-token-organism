"""THE DREAM'S LIVED SHARE (a supervisor's instrument, 2026-09-13): load a saved body on a copy, draw dreams as the night draws them,
and read how much of their world text is speech as lived (a substring of a line ever typed) against text stitched across lines.
usage: python3 tools/dream_lived.py COPY.pt --flags FLAGS.txt [--n 300]"""
import sys, json
sys.path.insert(0, "/Users/lukehamond/Projects/project")
import torch
from tokenizers import Tokenizer
from body.life import Life, PHYSIOLOGY

def arg(name, default):
    for a in sys.argv[1:]:
        if a.startswith(f"--{name}="):
            return type(default)(a.split("=", 1)[1])
        if a == f"--{name}":
            return type(default)(sys.argv[sys.argv.index(a) + 1])
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

cfg = parse_flags(open(arg("flags", "")).read()) if arg("flags", "") else {}
TOK = Tokenizer.from_file("/Users/lukehamond/Projects/project/data/tok_char.json")
life = Life.load(sys.argv[1], TOK, device="cpu", cfg=cfg); life.save_path = None
R = [json.loads(l) for l in open("/Users/lukehamond/Projects/project/data/watch2_caregiver.jsonl") if l.strip()]
typed = " | ".join(r["text"].strip() for r in R if r["action"] in ("line", "cue"))
st = life.store
print(f"store {st.n()}: links tagged {int((st.NE >= 0).sum())} of {int((st.N >= 0).sum())}, episodes counted {st.episode}, link width {st.NK}, dream_tag {cfg.get('dream_tag', 0)} dream_who {cfg.get('dream_who', 0)}")
n = arg("n", 300)
if int(cfg.get("dream_who", 0)):
    dreams, owns = life.dreams(n, with_who=True)
else:
    dreams = life.dreams(n); owns = [[False] * len(d) for d in dreams]
def world_runs(d, o):
    runs, cur = [], []
    for i, w in zip(d, o):
        if w or i == life.end_id:
            if cur: runs.append(TOK.decode(cur)); cur = []
        else: cur.append(i)
    if cur: runs.append(TOK.decode(cur))
    return runs
n_runs = lived = 0; lens = []; stitched, lived_ex = [], []
for d, o in zip(dreams, owns):
    for r in world_runs(d, o):
        r = r.strip()
        if len(r) < 4: continue
        n_runs += 1; lens.append(len(r))
        if r in typed: lived += 1; lived_ex.append(r)
        else: stitched.append(r)
print(f"{len(dreams)} dreams, mean length {sum(len(d) for d in dreams)/max(1,len(dreams)):.1f}; world runs of 4+ symbols {n_runs}, mean length {sum(lens)/max(1,len(lens)):.1f}; LIVED {lived} ({100*lived/max(1,n_runs):.0f}%)")
print("stitched:", stitched[:10]); print("lived:", lived_ex[:8])
