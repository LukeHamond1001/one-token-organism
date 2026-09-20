"""THE FORECAST'S NORM BY PHASE (2026-09-20): the sure proposal (gate_quiet_sure) opens the spontaneous floor whole when the forecast's
norm reaches the constant; at 0.45 and at 0.9 it opened on nearly every tick (the silence probe), so the floor's tries were the babble
in a thinking silence and the words in a one-handed line's pauses, and with the proposal off half the lines got no turn. This reads the
norm on a copy by phase: the slot after a question (its answer), the thinking silence after, and the pauses inside a one-handed line,
with the ticks it acted on. Nothing saved.
usage: nice -n 19 python3 tools/norm_probe.py COPY.pt --flags ops/BASE_FLAGS.txt [--gate-quiet-sure 0 ...]"""
import sys, random, statistics
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
seed = arg("seed", 0)
TOK = Tokenizer.from_file("/Users/lukehamond/Projects/project/data/tok_char.json")
life = Life.load(path, TOK, device="cpu", cfg=cfg, seed=seed); life.save_path = None; life.m.eval()
rec = []; phase = ["idle"]
_orig = life._choose
def spy(C1, pred1, u, level, stri):
    out = _orig(C1, pred1, u, level, stri)
    rec.append((phase[0], life.ticks - life._last_world, float(pred1.norm()), bool(out[0]), u != life.sil, bool(life._offset_done)))
    return out
life._choose = spy
questions = ["what is sour?", "who gives us eggs?", "what do the ducks eat?", "what is sweet?"][:arg("lines", 4)]
statements = ["the ball is here on the grass", "the hen sleeps too. and the ducks", "he was helpless and alone."]
rng = random.Random(seed)
with torch.no_grad():
    for _ in range(40): life.tick()
    for q in questions:
        phase[0] = "typed"; life.type_text(q, "parent")
        while life.queue: life.tick()
        phase[0] = "slot"
        for _ in range(40): life.tick()
        phase[0] = "silence"
        for _ in range(int(arg("quiet", 120))): life.tick()
    for s in statements:
        phase[0] = "line"
        for ch in s:
            life.type_text(ch, "parent"); life.tick()
            gap = rng.choice([2, 3]) if rng.random() > 0.1 else rng.randint(6, 24)
            for _ in range(gap - 1): life.tick()
        phase[0] = "after"
        for _ in range(40): life.tick()
def q(xs): return (f"n {len(xs)} p10 {statistics.quantiles(xs, n=10)[0]:.2f} med {statistics.median(xs):.2f} p90 {statistics.quantiles(xs, n=10)[-1]:.2f}" if len(xs) >= 10 else f"n {len(xs)} " + " ".join(f"{x:.2f}" for x in xs))
print(f"body {path.split('/')[-1]} nights {life.nights} | sure {life.cfg.get('gate_quiet_sure')} tau {life.cfg.get('gate_quiet_tau')} yield {life.cfg.get('gate_yield')} | the forecast's norm by phase (quiet ticks only)")
for ph in ("slot", "silence", "line", "after"):
    quiet = [r for r in rec if r[0] == ph and not r[4]]
    print(f"  {ph:8} all quiet ticks: {q([r[2] for r in quiet])}")
    if ph == "slot":
        print(f"  {ph:8} first 8 ticks:   {q([r[2] for r in quiet if r[1] <= 8])}")
    if ph == "line":
        print(f"  {ph:8} perceived pause: {q([r[2] for r in quiet if r[5]])}   (the offset fired inside the line)")
    acted = [r[2] for r in quiet if r[3]]
    print(f"  {ph:8} ticks it acted:  {q(acted)}")
own = "".join(e[0] for e in life.page if e[1] == 1 and e[0])
print(f"  said in all: {own!r}")
