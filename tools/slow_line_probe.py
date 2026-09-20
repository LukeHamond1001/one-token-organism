"""A SLOW LINE ON A COPY (2026-09-19, item 45): the parent's own recent lines typed one-handed into a copy of the body (two symbols a
second with a thinking pause of one to four seconds now and then, the served typist's SLOW form), the words the child says over each
line counted, under the served constants or with a constant given on the line. The one measure of item 45 that a copy can give
without the day: whether the utterance's end (offset_ticks, the count that ends a world utterance when the surprise has not settled)
is what lets the child in at a person's pauses. Nothing saved.
usage: nice -n 19 python3 tools/slow_line_probe.py COPY.pt --flags ops/BASE_FLAGS.txt --lines 20 [--seed 0] [--offset-ticks 30]"""
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
for a in sys.argv[1:]:                                                    # any physiology constant may be overridden on the line
    if a.startswith("--") and "=" not in a:
        k = a[2:].replace("-", "_")
        if k in PHYSIOLOGY:
            cfg[k] = type(PHYSIOLOGY[k])(sys.argv[sys.argv.index(a) + 1])
seed = arg("seed", 0); n_lines = arg("lines", 20)
TOK = Tokenizer.from_file("/Users/lukehamond/Projects/project/data/tok_char.json")
life = Life.load(path, TOK, device="cpu", cfg=cfg, seed=seed); life.save_path = None; life.m.eval()
R = [json.loads(l) for l in open("/Users/lukehamond/Projects/project/data/watch2_caregiver.jsonl") if l.strip()][-1500:]
seen = set(); lines = []
for r in reversed(R):
    if r.get("action") == "line" and r.get("voice", "a") == "a" and r.get("text") and r["text"] not in seen and 12 <= len(r["text"]) <= 38:
        seen.add(r["text"]); lines.append(r["text"])
    if len(lines) >= n_lines: break
lines = list(reversed(lines))
rng = random.Random(seed)
def own_since(p0):
    txt = "".join(e[0] for e in life.page[p0:] if e[1] == 1 and e[0])
    return len(re.findall(r"[A-Za-z]{2,}", txt)), txt
print(f"body {path.split('/')[-1]} nights {life.nights} | offset_ticks {life.cfg.get('offset_ticks')} form {life.cfg.get('offset_form')} offset_fast {life.cfg.get('offset_fast')} | {len(lines)} lines at a person's pace, seed {seed}", flush=True)
total = 0; clean = 0; lat = []
diag = arg("diag", 0)
# THE DIAGNOSTIC (--diag 1): where the words over a slow line come from. The offset's firings inside the line are counted by
# wrapping life._offset; each own symbol during the line is tagged with the floor at that tick (life._floor_now: 0 under the
# listening reflex, so a word begun at floor 0 came through the learned gate) and whether an offset had fired inside the line.
n_off = [0]
if diag:
    _orig_offset = life._offset
    def _counted_offset(settled=True):                      # the offset carries whether the settle law ended the utterance (2026-09-19)
        n_off[0] += 1; return _orig_offset(settled=settled)
    life._offset = _counted_offset
with torch.no_grad():
    for _ in range(40): life.tick()                                           # a moment of quiet first
    for line in lines:
        p0 = len(life.page); n_off[0] = 0; tags = []
        for k, ch in enumerate(line):
            if k > 0:
                gap = rng.choice([2, 3]) if rng.random() > 0.1 else rng.randint(6, 24)   # two symbols a second; a thinking pause of 1-4 s
                for _ in range(gap):
                    p1 = len(life.page); life.tick()
                    if diag and any(e[1] == 1 and e[0] for e in life.page[p1:]):
                        tags.append(f"{'F' if float(getattr(life, '_floor_now', 0.0)) > 0.0 else 'g'}{'+' if n_off[0] else '-'}")
            life.type_text(ch, "parent"); p1 = len(life.page); life.tick()
            if diag and any(e[1] == 1 and e[0] for e in life.page[p1:]):
                tags.append(f"{'F' if float(getattr(life, '_floor_now', 0.0)) > 0.0 else 'g'}{'+' if n_off[0] else '-'}")
        n, txt = own_since(p0); total += n; clean += (n == 0)
        extra = f" | offsets inside the line {n_off[0]}; own symbols by (F=floor open, g=floor shut; +=after an offset): {''.join(tags)}" if diag else ""
        p_end = len(life.page); first_at = None
        for k_ in range(40):                                                  # the child's turn (not counted as over): the ticks to its first symbol
            p1 = len(life.page); life.tick()
            if first_at is None and any(e[1] == 1 and e[0] for e in life.page[p1:]):
                first_at = k_ + 1
        lat.append(first_at)
        print(f"  {line!r:40} over {n:2d} {txt!r}{extra} | first own symbol after the line: {first_at}", flush=True)
        q = 0
        while q < 60:                                                         # then its quiet, as the typist waits
            p1 = len(life.page); life.tick(); q += 1
            if any(e[1] == 1 and e[0] for e in life.page[p1:]): q = 0 if q < 30 else q
import statistics
got = [x for x in lat if x is not None]
print(f"RESULT offset_ticks {life.cfg.get('offset_ticks')} ear {life.cfg.get('gate_ear_decay')}: {len(lines)} slow lines, {total/len(lines):.2f} words over per line, {100*clean/len(lines):.0f}% clean | its turn: first symbol at a median {statistics.median(got) if got else None} ticks, none within 40 ticks on {len(lat)-len(got)} of {len(lat)} lines", flush=True)
