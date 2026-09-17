"""THE ANSWER BY THE PAUSE (a supervisor's instrument, 2026-09-14): each fact's question as the world's line, then k rests, then twelve
symbols read greedily two ways: the cortex alone (its own argmax fed back as its own sound, no recall) and the mouth (the recall in the
forecast). A question is answered when a content word of the fact that the question lacks appears. The exchange dreams put the
answer one rest after the question (dream_gap); the probe's pause is offset_ticks: where the answer lives, if anywhere.
usage: python3 tools/qa_by_gap.py BODY.pt [--follow=20] [--gaps=1,2,4,8]"""
import sys, re, re, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from tokenizers import Tokenizer
from body.life import Life
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
follow = 0.0; gaps = [1, 2, 4, 8]; bag = None; key_form = None; show_all = False
for a in sys.argv[2:]:
    if a.startswith("--follow="): follow = float(a[9:])
    if a.startswith("--gaps="): gaps = [int(x) for x in a[7:].split(",")]
    if a.startswith("--bag="): bag = float(a[6:])          # the recall query's decay at read time (the keys stay as written)
    if a.startswith("--key-form="): key_form = a[11:]
    if a == "--all": show_all = True
TOK = Tokenizer.from_file(os.path.join(ROOT, "data/tok_char.json"))
cfg_ = {}
if follow > 1.0: cfg_["read_follow"] = follow
if bag is not None: cfg_["bag_decay"] = bag
if key_form: cfg_["key_form"] = key_form
from body.life import PHYSIOLOGY
for a in sys.argv[2:]:                                                          # any physiology constant may be overridden on the line
    if a.startswith("--") and "=" not in a:
        k = a[2:].replace("-", "_")
        if k in PHYSIOLOGY:
            cfg_[k] = type(PHYSIOLOGY[k])(sys.argv[sys.argv.index(a) + 1])
life = Life.load(sys.argv[1], TOK, device="cpu", cfg=(cfg_ or None)); m = life.m; m.eval()
zero = torch.zeros(m.d)
STOP = set("is are the a an in on and to do we i you it of my your they can what where who how does".split())
qset = "facts"
for a in sys.argv[2:]:
    if a.startswith("--set="): qset = a[6:]
# THE REPHRASED QUESTIONS (2026-09-15; --set=rephrased): each fact asked in words the parents never type (tools/heldout_rephrased.txt),
# the answer the same: recall of the taught exchange against understanding of the question
qfile = "tools/heldout_rephrased.txt" if qset == "rephrased" else "tools/facts_stage5.txt"
pairs = [tuple(s.strip() for s in l.split("|")[:2]) for l in open(os.path.join(ROOT, qfile)) if "|" in l]
def run(q, k, use_recall):
    life.win.clear(); life.bag_w.zero_(); life.bag_o.zero_(); life.ctx_cur.zero_(); life.ctx_prev.zero_(); life._utt_open = False; life.n_own = 0; life._follow = None
    got = []; rd_prev = zero
    with torch.no_grad():
        for ch in q:
            i = TOK.token_to_id(ch); life.win.append({"x": i, "xo": life.sil, "face": torch.zeros(2), "bundle": life.bands, "read": zero, "r": 0.0})
            life.rest_tick(world=True); life.take_world(i)                       # as the tick: the fade, then the world's symbol
            if use_recall: life._recall(life.bag)                     # the store read along the question, as awake (the episode builds)
        for _ in range(k):
            life.win.append({"x": life.sil, "xo": life.sil, "face": torch.zeros(2), "bundle": life.bands, "read": zero, "r": 0.0})
            life.rest_tick()
            if use_recall: life._recall(life.bag)
        for _ in range(20):                                       # twenty symbols: 'bees make honey' is fifteen
            xs, whos, faces, bundles, reads = life._window_tensors(list(life.win)[-m.window:])
            C = m.stream(m.inputs(xs, whos, faces, bundles, reads))[-1]
            cortex_key = str(life.cfg.get("key_form", "bag")) == "cortex"
            rd = (life._recall(life.query_from(C))[0] if cortex_key else life._recall(life.bag)[0]) if use_recall else zero
            lm = m.readout(m.forecast(C, rd)); lm[life.bans] = float("-inf"); lm[life.sil] = float("-inf")
            sym = int(lm.argmax()); got.append(TOK.decode([sym]))
            life.win.append({"x": life.sil, "xo": sym, "face": torch.zeros(2), "bundle": life.bands, "read": (rd_prev if cortex_key else rd), "r": 0.0}); rd_prev = rd
            life.rest_tick(); life.take_own(sym)                       # as the tick: the world half's fade, then its own symbol
    return "".join(got)
print(f"body {os.path.basename(sys.argv[1])}: {qset} questions, twenty symbols read | nights {life.nights} | key_form {life.cfg.get('key_form')} bag_decay {life.cfg['bag_decay']} read_follow {life.cfg.get('read_follow')}")
for k in gaps:
    row = []
    for use_recall in (False, True):
        n = 0; ex = []; first10 = 0
        for qi, (q, fact) in enumerate(pairs):
            keys = [w for w in re.findall(r"[a-z]+", fact.lower()) if w not in STOP and w not in q.lower()]
            t = run(q, k, use_recall); ok = any(re.search(r'\b' + re.escape(w), t.lower()) for w in keys); n += int(ok); first10 += int(ok and qi < 10)
            if ok and len(ex) < 3: ex.append(f"{q!r}->{t[:12]!r}")
            if show_all and use_recall: print(f"      {'*' if ok else ' '} {q:26} -> {t[:16]!r}   (wants one of {keys})")
        row.append((n, ex, first10))
    print(f"  pause {k:2d} rests: cortex alone answers {row[0][0]:2d}/30 {' '.join(row[0][1])} | the mouth {row[1][0]:2d}/30 {' '.join(row[1][1])} | facts 1-10 {row[1][2]}")
