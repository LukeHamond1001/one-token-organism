"""THE ANSWER BY THE PAUSE (a supervisor's instrument, 2026-09-14): each fact's question as the world's line, then k rests, then twelve
symbols read greedily two ways: the cortex alone (its own argmax fed back as its own sound, no recall) and the mouth (the recall in the
forecast). A question is answered when a content word of the fact that the question lacks appears. The exchange dreams put the
answer one rest after the question (dream_gap); the probe's pause is offset_ticks: where the answer lives, if anywhere.
usage: python3 tools/qa_by_gap.py BODY.pt [--follow=20] [--gaps=1,2,4,8]"""
import sys, re, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from tokenizers import Tokenizer
from body.life import Life
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
follow = 0.0; gaps = [1, 2, 4, 8]; bag = None
for a in sys.argv[2:]:
    if a.startswith("--follow="): follow = float(a[9:])
    if a.startswith("--gaps="): gaps = [int(x) for x in a[7:].split(",")]
    if a.startswith("--bag="): bag = float(a[6:])          # the recall query's decay at read time (the keys stay as written)
TOK = Tokenizer.from_file(os.path.join(ROOT, "data/tok_char.json"))
cfg_ = {}
if follow > 1.0: cfg_["read_follow"] = follow
if bag is not None: cfg_["bag_decay"] = bag
life = Life.load(sys.argv[1], TOK, device="cpu", cfg=(cfg_ or None)); m = life.m; m.eval()
zero = torch.zeros(m.d)
STOP = set("is are the a an in on and to do we i you it of my your they can what where who how does".split())
pairs = [tuple(s.strip() for s in l.split("|")[:2]) for l in open(os.path.join(ROOT, "tools/facts_stage5.txt")) if "|" in l]
def run(q, k, use_recall):
    life.win.clear(); life.bag_w.zero_(); life.bag_o.zero_(); life.n_own = 0; life._follow = None
    got = []
    with torch.no_grad():
        for ch in q:
            i = TOK.token_to_id(ch); life.win.append({"x": i, "xo": life.sil, "face": torch.zeros(2), "bundle": life.bands, "read": zero, "r": 0.0})
            life.bag_w = life.cfg["bag_decay"] * m.shift(life.bag_w) + m.E.weight[i]
        for _ in range(k):
            life.win.append({"x": life.sil, "xo": life.sil, "face": torch.zeros(2), "bundle": life.bands, "read": zero, "r": 0.0})
            life.bag_w = life.cfg["bag_decay"] * life.bag_w
        for _ in range(12):
            xs, whos, faces, bundles, reads = life._window_tensors(list(life.win)[-m.window:])
            C = m.stream(m.inputs(xs, whos, faces, bundles, reads))[-1]
            rd = life._recall(life.bag)[0] if use_recall else zero
            lm = m.readout(m.forecast(C, rd)); lm[life.bans] = float("-inf"); lm[life.sil] = float("-inf")
            sym = int(lm.argmax()); got.append(TOK.decode([sym]))
            life.win.append({"x": life.sil, "xo": sym, "face": torch.zeros(2), "bundle": life.bands, "read": rd, "r": 0.0})
            life.bag_o = life.cfg["bag_decay"] * m.shift(life.bag_o) + m.E.weight[sym]; life.n_own += 1
    return "".join(got)
print(f"body {os.path.basename(sys.argv[1])}: nights {life.nights} | bag_decay {life.cfg['bag_decay']} read_follow {life.cfg.get('read_follow')}")
for k in gaps:
    row = []
    for use_recall in (False, True):
        n = 0; ex = []
        for q, fact in pairs:
            keys = [w for w in re.findall(r"[a-z]+", fact.lower()) if w not in STOP and w not in q.lower()]
            t = run(q, k, use_recall); ok = any(w in t.lower() for w in keys); n += int(ok)
            if ok and len(ex) < 3: ex.append(f"{q!r}->{t[:12]!r}")
        row.append((n, ex))
    print(f"  pause {k:2d} rests: cortex alone answers {row[0][0]:2d}/30 {' '.join(row[0][1])} | the mouth {row[1][0]:2d}/30 {' '.join(row[1][1])}")
