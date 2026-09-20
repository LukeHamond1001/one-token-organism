"""THE DEMO REHEARSED ON A COPY (2026-09-20, the user's word: "I teach it something new, then I ask it about it, it responds; then I ask
it other stuff we taught it and it answers because it knows; then I ask the new thing again; no babble in between, no random words
or letters"). The sequence typed one-handed (two symbols a second, a thinking pause of one to four seconds now and then): a new fact
taught as two voices (the question, its turn, the other voice's answer), the new question, three known questions, the new question
again; after every line its turn (forty ticks), then the person's thinking silence (--think ticks). Every own symbol is placed:
over a line, in its turn, or stray in a silence. The served constants or any given on the line. Nothing saved.
usage: nice -n 19 python3 tools/demo_rehearsal.py COPY.pt --flags ops/BASE_FLAGS.txt [--think 60] [--seed 0] [--gate-turn 1 ...]"""
import sys, re, random
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
seed = arg("seed", 0); think = arg("think", 60); cps = arg("cps", 2.0)
TOK = Tokenizer.from_file("/Users/lukehamond/Projects/project/data/tok_char.json")
life = Life.load(path, TOK, device="cpu", cfg=cfg, seed=seed); life.save_path = None; life.m.eval()
rng = random.Random(seed)
NEW_Q, NEW_A, NEW_KEY = arg("new_q", "what does the fox say?"), arg("new_a", "the fox says yip"), arg("new_key", "yip")
repeat = arg("repeat", 1)                                                  # the pair taught this many times (question, its turn, the answer) before the ask
SCRIPT = [("parent", NEW_Q, NEW_KEY, "the new fact asked, untaught"), ("other", NEW_A, None, "the other voice teaches it")]
for _ in range(repeat - 1):
    SCRIPT += [("parent", NEW_Q, NEW_KEY, "the new fact asked, taught once more after"), ("other", NEW_A, None, "the other voice teaches it again")]
SCRIPT += [("parent", NEW_Q, NEW_KEY, "the new fact asked"),
          ("parent", "what is sour?", "lemon", "a known fact"), ("parent", "who gives us milk?", "cow", "a known fact"), ("parent", "what is sweet?", "pear", "a known fact"),
          ("parent", NEW_Q, NEW_KEY, "the new fact asked again")]
gap_ticks = max(1, int(round(6.0 / cps)))                                   # six ticks a second at the served pace
def own_between(p0, p1):
    return "".join(e[0] for e in life.page[p0:p1] if e[1] == 1 and e[0])
def words(t): return re.findall(r"[A-Za-z]{2,}", t)
print(f"body {path.split('/')[-1]} nights {life.nights} | yield {life.cfg.get('gate_yield')} turn {life.cfg.get('gate_turn')} foresee {life.cfg.get('offset_foresee')} sure {life.cfg.get('gate_quiet_sure')} tau {life.cfg.get('gate_quiet_tau')} | one-handed at {cps} a second, thinking {think} ticks between lines", flush=True)
tot_over = 0; tot_stray = 0; tot_stray_sym = 0; answered = []
with torch.no_grad():
    for _ in range(40): life.tick()
    for who, text, key, what in SCRIPT:
        p0 = len(life.page)
        for k, ch in enumerate(text):
            if k > 0:
                gap = gap_ticks + (0 if rng.random() > 0.1 else rng.randint(6, 24))
                for _ in range(gap): life.tick()
            life.type_text(ch, who); life.tick()
        p1 = len(life.page); over = own_between(p0, p1)
        for _ in range(40): life.tick()                                   # its turn
        p2 = len(life.page); turn = own_between(p1, p2)
        for _ in range(think): life.tick()                                # the person thinks
        p3 = len(life.page); stray = own_between(p2, p3)
        tot_over += len(words(over)); tot_stray += len(words(stray)); tot_stray_sym += len(stray.strip())
        ok = (key is not None and key in turn.lower())
        if key is not None and who == "parent" and "taught once more" not in what: answered.append(ok)
        tag = {"parent": "A", "other": "b"}[who]
        print(f"  {tag}: {text!r:26} ({what})\n     over the line: {over!r:20} | its turn: {turn!r:44} {'ANSWERED' if ok else ('-' if key is None else 'no answer')}\n     in the silence after: {stray!r}", flush=True)
print(f"RESULT (taught {repeat}x): answered {sum(answered)} of {len(answered)} questions (the new fact: asked untaught {'yes' if answered[0] else 'no'}, after teaching {'yes' if answered[1] else 'no'}, again at the end {'yes' if answered[-1] else 'no'}) | words over the lines {tot_over} | stray in the silences: {tot_stray} words, {tot_stray_sym} symbols", flush=True)
