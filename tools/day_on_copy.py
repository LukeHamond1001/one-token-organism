"""A DAY ON A COPY (a supervisor's instrument, 2026-09-14): load a saved body on a copy with the served constants and any override,
then feed it a day's typed lines from the page log in order, as the typist did (the line's symbols, then the child's turn of
`listen` ticks in which the mouth speaks and the store writes as awake), and save the copy: the day lived again under a changed
constant, for the rulers to read. Never saves back.
usage: python3 tools/day_on_copy.py BODY.pt --flags FLAGS.txt --day 199 --save-as OUT.pt [--listen 24] [--bag-decay 0.9] [--max-lines 0] [--talkover 1]
(--talkover 1, 2026-09-17: the child's own symbols said while the line was still being typed, the talk-overs, against those said in
its turn after it, per quarter of the day and in all: the ruler for the gate's ear)"""
import sys, os, json, time
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT)
import torch
from tokenizers import Tokenizer
from body.life import Life, PHYSIOLOGY

def arg(name, default):
    names = {f"--{name}", f"--{name.replace('_', '-')}"}
    for i, a in enumerate(sys.argv[1:], 1):
        if "=" in a and a.split("=", 1)[0] in names:
            return type(default)(a.split("=", 1)[1])
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

path = sys.argv[1]; day = arg("day", 0); listen = arg("listen", 24); save_as = arg("save_as", ""); max_lines = arg("max_lines", 0)
cfg = parse_flags(open(arg("flags", "")).read()) if arg("flags", "") else {}
for k in list(PHYSIOLOGY):                                                # any physiology constant may be overridden on the line
    v = arg(k, None) if False else None
for a in sys.argv[1:]:
    if a.startswith("--") and "=" not in a:
        k = a[2:].replace("-", "_")
        if k in PHYSIOLOGY:
            cfg[k] = type(PHYSIOLOGY[k])(sys.argv[sys.argv.index(a) + 1])
TOK = Tokenizer.from_file(os.path.join(ROOT, "data/tok_char.json"))
life = Life.load(path, TOK, device="cpu", cfg=cfg); life.save_path = None; m = life.m
R = [json.loads(l) for l in open(os.path.join(ROOT, "data/watch2_caregiver.jsonl")) if l.strip()]
lines = [(r["text"], "other" if r.get("voice") == "b" else "you") for r in R if r.get("day") == day and r["action"] in ("line", "cue")]
# THE REWARD REPLAYED (--rewards 1; 2026-09-15): the caregiver's smiles and frowns of the day, each at its delay after the line it
# followed (seconds to ticks at five a second), set on the copy's face with the caregiver's own shapes: a smile +2 for twelve ticks,
# the growing smile +2 then +4, a frown -1 for twelve ticks. Without it a copy day carries no reward and nothing that learns from
# dopamine (the actor, the plasticity of the day, the store's strength) can be measured on a copy.
rewards = arg("rewards", 0); events = {}
if rewards:
    import datetime as dt
    D = [r for r in R if r.get("day") == day]
    def ts(r):
        try: return dt.datetime.fromisoformat(r["ts"]).timestamp()
        except Exception: return None
    li = [(k, ts(r)) for k, r in enumerate(D) if r["action"] in ("line", "cue")]
    idx_of_line = {k: n for n, (k, _) in enumerate(li)}
    cur = -1; t_line = None
    for k, r in enumerate(D):
        if r["action"] in ("line", "cue"):
            cur = idx_of_line[k]; t_line = ts(r); continue
        if cur < 0 or t_line is None or r["action"] not in ("smile", "frown"):
            continue
        t = ts(r)
        if t is None: continue
        delay = max(0, int(round((t - t_line) * 5)))
        lev = int(r.get("levels", 1) or 1) if r["action"] == "smile" else -1
        events.setdefault(cur, []).append((delay, lev))
    print(f"rewards replayed: {sum(len(v) for v in events.values())} smiles and frowns over {len(events)} lines", flush=True)
if max_lines: lines = lines[:max_lines]
print(f"body {os.path.basename(path)}: nights {life.nights} store {life.store.n()} | bag_decay {life.cfg['bag_decay']} read_follow {life.cfg.get('read_follow')} | day {day}: {len(lines)} lines, listen {listen}", flush=True)
t0 = time.time(); ticks = 0
with torch.no_grad():
    pass
faces_set = 0; own_syms = []                                          # the child's own symbols, for the junk count (--dump-own 1)
dump_own = arg("dump_own", 0); talkover = arg("talkover", 0); to_rows = []   # per line: own symbols while typing, own symbols in its turn
own_file = arg("own_file", "")                                       # --own-file PATH: the child's own text of the whole day written there (for tools/word_rate.py --text)
for i, (text, who) in enumerate(lines):
    life.type_text(text, who=who)
    span = len(text) + listen                                   # the line's ticks on the copy: its symbols, then the child's turn
    pend = sorted((min(d, max(0, span - 12)), lev) for d, lev in events.get(i, [])) if rewards else []   # a reward later than the turn lands at its end
    t_after = 0; face_until = -1; face_then = None
    def face_tick():
        global t_after, pend, face_until, face_then, faces_set
        while pend and pend[0][0] <= t_after:                       # a smile or frown due now, at its delay after the line
            _, lev = pend.pop(0)
            if lev < 0: life.set_face(-1.0); face_until = t_after + 12; face_then = None
            elif lev >= 2: life.set_face(2.0); face_until = t_after + 12; face_then = (4.0, t_after + 24)
            else: life.set_face(2.0); face_until = t_after + 12; face_then = None
            faces_set += 1
        if face_until >= 0 and t_after >= face_until:
            if face_then is not None and t_after < face_then[1]:
                life.set_face(face_then[0]); face_until = face_then[1]; face_then = None
            else:
                life.set_face(0.0); face_until = -1
        t_after += 1
    n0 = len(life.win)
    own_typing = 0; own_turn = 0
    while life.queue:
        life.tick(); ticks += 1; face_tick(); own_typing += int(list(life.win)[-1]["xo"] != life.sil) if talkover else 0
    for _ in range(listen):
        life.tick(); ticks += 1; face_tick(); own_turn += int(list(life.win)[-1]["xo"] != life.sil) if talkover else 0
    if talkover: to_rows.append((len(text), own_typing, own_turn))
    if rewards: life.set_face(0.0)
    if dump_own:
        own_syms += [int(w["xo"]) for w in list(life.win)[-min(len(life.win), 96):] if int(w["xo"]) != life.sil]
    if own_file:
        own_all = getattr(life, "_own_all", []); own_all += [(int(w["xo"]), (w["x"] != life.sil)) for w in list(life.win)[-(span):] if int(w["xo"]) != life.sil]; life._own_all = own_all
    if (i + 1) % 40 == 0:
        print(f"  {i + 1} lines, {ticks} ticks, {time.time() - t0:.0f}s, store {life.store.n()}", flush=True)
if talkover and to_rows:
    q = max(1, len(to_rows) // 4)
    parts = [to_rows[i:i + q] for i in range(0, len(to_rows), q)][:4]
    print("TALK-OVERS: own symbols while the line was being typed / typed symbols, then own symbols in its turn, per quarter of the day: " +
          " | ".join(f"{sum(r[1] for r in pt)}/{sum(r[0] for r in pt)} typing, {sum(r[2] for r in pt)} in turn" for pt in parts) +
          f" || all: {sum(r[1] for r in to_rows)}/{sum(r[0] for r in to_rows)} typing ({100.0 * sum(r[1] for r in to_rows) / max(1, sum(r[0] for r in to_rows)):.1f}%), {sum(r[2] for r in to_rows)} in turn, lines with a talk-over {sum(1 for r in to_rows if r[1] > 0)} of {len(to_rows)}", flush=True)
if own_file:
    oa = getattr(life, "_own_all", []); turn = "".join(TOK.decode([x]) for x, over in oa if not over); over_ = "".join(TOK.decode([x]) for x, over in oa if over)
    open(own_file, "w").write(json.dumps({"in_turn": turn, "over_the_line": over_})); print(f"own text written to {own_file}: {len(turn)} symbols in its turn, {len(over_)} over the line", flush=True)
if dump_own:
    txt = "".join(TOK.decode([x]) for x in own_syms); junk = sum(1 for ch in txt if not (ch.islower() or ch in " .?!'"))
    print(f"own symbols {len(txt)}: junk (not lowercase, space or .?!) {junk} | sample {txt[:160]!r}", flush=True)
print(f"done: {len(lines)} lines in {ticks} ticks, {time.time() - t0:.0f}s | faces set {faces_set} | the actor's slope {float(getattr(life, '_arel_gain', 0.0)):.3f} corr {float(getattr(life, '_arel_corr', 0.0)):.3f} | mood {life.mood:.2f}", flush=True)
if save_as:
    life.save_path = save_as; life.save(); life.save_path = None; print(f"saved as {save_as}", flush=True)
