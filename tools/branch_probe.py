"""THE BRANCH AFTER A SHARED START (a supervisor's instrument, 2026-09-14): four facts begin "the sun ", two begin "birds ". After
the question, a rest, and the shared start, which symbol does the cortex alone put first, and which the mouth (the recall in the
forecast)? The asked fact's continuation should win; on the served body after night 177 the cortex alone had no preference by the
question and the mouth followed the store's strongest sibling ("what is hot?" -> "the sun makes...").
usage: python3 tools/branch_probe.py BODY.pt [--follow=20]"""
import sys, os
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT)
import torch
from tokenizers import Tokenizer
from body.life import Life
follow = 20.0; reads_along = False
for a in sys.argv[2:]:
    if a.startswith("--follow="): follow = float(a[9:])
    if a == "--reads-along": reads_along = True
TOK = Tokenizer.from_file(os.path.join(ROOT, "data/tok_char.json"))
from body.life import PHYSIOLOGY
cfg_ = {"read_follow": follow}
for a in sys.argv[2:]:                                                          # any physiology constant may be overridden on the line
    if a.startswith("--") and "=" not in a:
        k = a[2:].replace("-", "_")
        if k in PHYSIOLOGY:
            cfg_[k] = type(PHYSIOLOGY[k])(sys.argv[sys.argv.index(a) + 1])
life = Life.load(sys.argv[1], TOK, device="cpu", cfg=cfg_); m = life.m; m.eval()
zero = torch.zeros(m.d)
SETS = {
    "sun": [("the sun ", [("what is hot?", "i"), ("what makes us warm?", "m"), ("what is up in the day?", "i"), ("what is the sun?", "i")]),
            ("birds ", [("what do birds do?", "f"), ("where do birds live?", "l")]),
            ("the sun is ", [("what is hot?", "h"), ("what is up in the day?", "u"), ("what is the sun?", "a")])],
    # the facts of a second day (11-20): "we " begins eat, drink and sleep; "a " begins wings, legs and big; "an " red and little
    "day2": [("we ", [("what do we eat?", "e"), ("what do we drink?", "d"), ("where do we sleep?", "s")]),
             ("a ", [("what has wings?", "b"), ("what has four legs?", "d"), ("what is big?", "t")]),
             ("an ", [("what is red?", "a"), ("what is little?", "a")]),
             ("we eat ", [("what do we eat?", "b")]), ("we drink ", [("what do we drink?", "w")]), ("a bird ", [("what has wings?", "h")]), ("a dog ", [("what has four legs?", "h")])]}
which = "sun"
for a in sys.argv[2:]:
    if a.startswith("--set="): which = a[6:]
FAMILIES = SETS[which]
def dist(prefix):
    """PREFIX: the question, '|' rests, then the shared start said by the mouth itself (after the rests, the symbols are its own)"""
    life.win.clear(); life.bag_w.zero_(); life.bag_o.zero_(); life.ctx_cur.zero_(); life.ctx_prev.zero_(); life._utt_open = False; life.n_own = 0; life._follow = None
    with torch.no_grad():
        own = False
        for ch in prefix:
            i = TOK.token_to_id(ch) if ch != "|" else life.sil
            if ch == "|": own = True
            if own and ch != "|":
                life.win.append({"x": life.sil, "xo": i, "face": torch.zeros(2), "bundle": life.bands, "read": zero, "r": 0.0})
                life.rest_tick(); life.take_own(i); m.striatum_push(1, i)          # the delay line hears its own symbol
            else:
                life.win.append({"x": i, "xo": life.sil, "face": torch.zeros(2), "bundle": life.bands, "read": zero, "r": 0.0})
                life.rest_tick(world=(ch != "|"))
                if ch != "|": life.take_world(i); m.striatum_push(0, i)          # the delay line hears the world's symbol
                else: m.striatum_push(3, 0)                                      # a tick of quiet
            if reads_along: life._recall(life.bag)                  # the store read at every symbol, as awake: the episode followed builds
        xs, whos, faces, bundles, reads = life._window_tensors(list(life.win)[-m.window:])
        C = m.stream(m.inputs(xs, whos, faces, bundles, reads))[-1]
        rd, conf, _ = life._recall(life.bag)
        lc = m.readout(m.forecast(C, zero)); lm = m.readout(m.forecast(C, rd))
        for l in (lc, lm): l[life.bans] = float("-inf"); l[life.sil] = float("-inf")
        # THE ACTOR'S RAW PREFERENCE (2026-09-15): the striatal head's vote at the branch, before the earned-voice gate scales it
        try:
            if str(life.cfg.get("actor_form", "add")) == "softmax":                 # the chooser: its raw scores on the cortex's state
                za = torch.nn.functional.normalize(C.detach().float() - life._c_mu, dim=0) * float(life.cfg.get("key_scale", 2.5))
                a = m.chooser(za).clone()
            else:
                a = torch.tanh(m.actor(m.stri_in()))
            a[life.bans] = float("-inf"); a[life.sil] = float("-inf")
        except Exception:
            a = torch.zeros_like(lc)
        return torch.softmax(lc, 0), torch.softmax(lm, 0), a
print(f"body {os.path.basename(sys.argv[1])}: nights {life.nights}")
tot_c = tot_m = tot_a = n = 0
for start, qs in FAMILIES:
    for q, want in qs:
        pc, pm, pa = dist(q + "||" + start); w = TOK.token_to_id(want)
        cset = [TOK.token_to_id(x) for _, x in qs]                              # the chooser votes only among the family's candidates
        pa_c = torch.full_like(pa, float("-inf")); pa_c[cset] = pa[cset]; pa = pa_c
        topc = TOK.decode([int(pc.argmax())]); topm = TOK.decode([int(pm.argmax())]); topa = TOK.decode([int(pa.argmax())])
        okc = topc == want; okm = topm == want; oka = topa == want; tot_c += okc; tot_m += okm; tot_a += oka; n += 1
        print(f"  {q!r:24} + {start!r:12} wants {want!r}: cortex alone {topc!r} (p {float(pc[w]):.2f}){'*' if okc else ' '} | the mouth {topm!r} (p {float(pm[w]):.2f}){'*' if okm else ' '} | the actor {topa!r} (vote {float(pa[w]):+.2f}){'*' if oka else ' '}")
print(f"BRANCH: the asked fact's continuation first: cortex alone {tot_c}/{n}, the mouth {tot_m}/{n}, the actor's raw vote {tot_a}/{n}")
