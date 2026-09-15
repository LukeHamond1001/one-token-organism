"""THE FACT TOLD ONCE (a supervisor's instrument, 2026-09-15): the demo's central scene is a fact told once in conversation, a night, and
the question answered the next day. Ten facts the parents never typed (their content words absent from the whole queue) are told once
each inside forty ordinary lines of a logged day, as the typist types (the line's symbols, then the child's turn), the copy saved; then
each question is asked with the pause and twenty symbols read with the recall (qa_by_gap's rule: answered when a content word of the
fact the question lacks appears). Run again with --ask-only 1 on the copy after a night on it (tools/night_copy.py).
usage: python3 tools/once_told.py BODY.pt --flags FLAGS.txt [--day 231] [--listen 48] [--every 4] [--save-as OUT.pt] [--ask-only 0]"""
import sys, os, re, json, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from tokenizers import Tokenizer
from body.life import Life, PHYSIOLOGY
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
def arg(name, default):
    names = {"--" + name, "--" + name.replace("_", "-")}
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
NOVEL = [("what is sour?", "a lemon is sour"), ("what is loud?", "a drum is loud"), ("what is a pear?", "a pear is a fruit"),
         ("what is salty?", "the sea is salty"), ("what is slow?", "a snail is slow"), ("what do owls say?", "owls hoot"),
         ("what gives light?", "a lamp gives light"), ("what has wool?", "a sheep has wool"), ("what has a horn?", "a goat has a horn"),
         ("what is steep?", "a hill is steep")]
path = sys.argv[1]; day = arg("day", 231); listen = arg("listen", 48); every = arg("every", 4); save_as = arg("save_as", ""); ask_only = arg("ask_only", 0)
cfg = parse_flags(open(arg("flags", "")).read()) if arg("flags", "") else {}
TOK = Tokenizer.from_file(os.path.join(ROOT, "data/tok_char.json"))
life = Life.load(path, TOK, device="cpu", cfg=cfg); life.save_path = None; m = life.m
print(f"body {os.path.basename(path)}: nights {life.nights} store {life.store.n()} cap {life.store.cap} | read_follow {life.cfg.get('read_follow')} write_floor {life.cfg.get('write_floor')}", flush=True)
if not ask_only:
    R = [json.loads(l) for l in open(os.path.join(ROOT, "data/watch2_caregiver.jsonl")) if l.strip()]
    plain = [(r["text"], "other" if r.get("voice") == "b" else "you") for r in R if r.get("day") == day and r["action"] in ("line", "cue")][:40]
    lines = []; k = 0
    for i, ln in enumerate(plain):
        lines.append(ln)
        if (i + 1) % every == 0 and k < len(NOVEL):
            q, a = NOVEL[k]; lines += [(q, "you"), (a, "other")]; k += 1
    while k < len(NOVEL):
        q, a = NOVEL[k]; lines += [(q, "you"), (a, "other")]; k += 1
    t0 = time.time(); ticks = 0
    for i, (text, who) in enumerate(lines):
        life.type_text(text, who=who)
        while life.queue:
            life.tick(); ticks += 1
        for _ in range(listen):
            life.tick(); ticks += 1
    print(f"told once: {len(lines)} lines ({len(NOVEL)} novel exchanges among {len(plain)} of day {day}'s), {ticks} ticks, {time.time() - t0:.0f}s, store {life.store.n()}", flush=True)
    if save_as:
        life.save(save_as); print(f"saved {save_as}", flush=True)
m.eval(); zero = torch.zeros(m.d)
STOP = set("is are the a an in on and to do we i you it of my your they can what where who how does".split())
def run(q, k, use_recall):
    life.win.clear(); life.bag_w.zero_(); life.bag_o.zero_(); life.ctx_cur.zero_(); life.ctx_prev.zero_(); life._utt_open = False; life.n_own = 0; life._follow = None
    got = []
    with torch.no_grad():
        for ch in q:
            i = TOK.token_to_id(ch); life.win.append({"x": i, "xo": life.sil, "face": torch.zeros(2), "bundle": life.bands, "read": zero, "r": 0.0})
            life.rest_tick(world=True); life.take_world(i)
            if use_recall: life._recall(life.bag)
        for _ in range(k):
            life.win.append({"x": life.sil, "xo": life.sil, "face": torch.zeros(2), "bundle": life.bands, "read": zero, "r": 0.0})
            life.rest_tick()
            if use_recall: life._recall(life.bag)
        for _ in range(20):
            xs, whos, faces, bundles, reads = life._window_tensors(list(life.win)[-m.window:])
            C = m.stream(m.inputs(xs, whos, faces, bundles, reads))[-1]
            rd = life._recall(life.bag)[0] if use_recall else zero
            lm = m.readout(m.forecast(C, rd)); lm[life.bans] = float("-inf"); lm[life.sil] = float("-inf")
            sym = int(lm.argmax()); got.append(TOK.decode([sym]))
            life.win.append({"x": life.sil, "xo": sym, "face": torch.zeros(2), "bundle": life.bands, "read": rd, "r": 0.0})
            life.rest_tick(); life.take_own(sym)
    return "".join(got)
for k in (2, 8):
    for use_recall in (False, True):
        n = 0; rows = []
        for q, fact in NOVEL:
            keys = [w for w in re.findall(r"[a-z]+", fact.lower()) if w not in STOP and w not in q.lower()]
            t = run(q, k, use_recall); ok = any(w in t.lower() for w in keys); n += int(ok)
            rows.append(f"{'*' if ok else ' '} {q!r:20}->{t[:16]!r}")
        print(f"ONCE pause {k} {'the mouth' if use_recall else 'cortex alone'}: {n}/{len(NOVEL)} | " + " ".join(rows), flush=True)
