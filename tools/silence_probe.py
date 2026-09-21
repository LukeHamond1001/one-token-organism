"""THE CHATTER BETWEEN THE LINES (2026-09-20, the user's word: "I need it not to spit random blabber while I'm having a convo"): on
a copy, a question typed at the tick, the child's turn given forty ticks, then a person's thinking silence of N ticks with no
input; every own word in the silence counted and tagged by what let it out (F: the spontaneous floor open at that tick; g: the
floor shut, the learned gate alone), with the tick it came at. The served constants or any given on the line. Nothing saved.
usage: nice -n 19 python3 tools/silence_probe.py COPY.pt --flags ops/BASE_FLAGS.txt --lines 6 --quiet 240 [--gate-quiet-sure 0.9] ..."""
import sys, json, re, random
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
seed = arg("seed", 0); n_lines = arg("lines", 6); quiet = arg("quiet", 240)
TOK = Tokenizer.from_file("/Users/lukehamond/Projects/project/data/tok_char.json")
life = Life.load(path, TOK, device="cpu", cfg=cfg, seed=seed); life.save_path = None; life.m.eval()
questions = ["what is sour?", "who gives us eggs?", "is the egg hard or soft?", "what do the ducks eat?", "who gives us milk?", "what is sweet?", "what does the hen say?", "what is round?"][:n_lines]
if arg("questions_file", ""):                                              # any questions, one a line (the reworded set of 2026-09-21)
    questions = [l.rstrip("\n") for l in open(arg("questions_file", "")) if l.strip()][:n_lines]
def own_since(p0):
    txt = "".join(e[0] for e in life.page[p0:] if e[1] == 1 and e[0])
    return len(re.findall(r"[A-Za-z]{2,}", txt)), txt
print(f"body {path.split('/')[-1]} nights {life.nights} | gate_quiet_tau {life.cfg.get('gate_quiet_tau')} gate_quiet_sure {life.cfg.get('gate_quiet_sure')} gate_tonic {life.cfg.get('gate_tonic')} | {len(questions)} questions, {quiet} quiet ticks each, seed {seed}", flush=True)
tot_words = 0; tot_F = 0; tot_g = 0; firsts = []
with torch.no_grad():
    for _ in range(40): life.tick()
    for q in questions:
        life.type_text(q, "parent")
        while life.queue: life.tick()
        p0 = len(life.page)
        for _ in range(40): life.tick()                                   # its turn
        n_ans, ans = own_since(p0)
        p1 = len(life.page); tags = []; first = None
        for k in range(quiet):                                            # the silence: a person thinking
            p2 = len(life.page); life.tick()
            new = "".join(e[0] for e in life.page[p2:] if e[1] == 1 and e[0])
            if new:
                fl = float(getattr(life, "_floor_now", 0.0))
                tags.append(("F" if fl > 0.0 else "g", k))
                if first is None: first = k
        n_q, txt = own_since(p1)
        F = sum(1 for t, _ in tags if t == "F"); g = sum(1 for t, _ in tags if t == "g")
        tot_words += n_q; tot_F += F; tot_g += g; firsts.append(first)
        print(f"  {q!r:26} answer {ans!r:32} | in {quiet} quiet ticks: {n_q:2d} words {txt!r:60} | own symbols: floor-open {F:3d} gate-alone {g:3d} | first at tick {first}", flush=True)
import statistics
fs = [f for f in firsts if f is not None]
print(f"RESULT tau {life.cfg.get('gate_quiet_tau')} sure {life.cfg.get('gate_quiet_sure')} tonic {life.cfg.get('gate_tonic')}: {tot_words/len(questions):.1f} words per {quiet}-tick silence ({tot_words/(len(questions)*quiet/6/60):.1f} per silent minute at 6 ticks/s) | own symbols floor-open {tot_F} gate-alone {tot_g} | first symbol at a median {statistics.median(fs) if fs else None} ticks into the silence", flush=True)
