"""THE SERVED DAY'S DIGEST: one row per day of a served body from its page log (the caregiver's jsonl), the supervisor's
instrument for the days after the watched phase. Faces, cues asked and answered (from the smile rows whose reason names a
cue), the child's own two-word runs and the pairs no one taught it (from every row's context, the parent's symbols masked),
the known-word count, and the night's memory gauge.
usage: python3 tools/served_day.py data/watch2_caregiver.jsonl data/watch2_corpus.json [FIRST_DAY]"""
import ast
import collections
import json
import re
import sys

log = sys.argv[1]; corpus_path = sys.argv[2]; first = int(sys.argv[3]) if len(sys.argv) > 3 else 0
rows = [json.loads(l) for l in open(log)]
corpus = json.load(open(corpus_path)).get("lines", {})
taught = set(); words = set()
for line in corpus:
    ws = re.findall(r"[a-z]+", line.lower()); words.update(ws); taught.update(zip(ws, ws[1:]))
days = collections.defaultdict(lambda: collections.defaultdict(int))
pairs = collections.defaultdict(collections.Counter); novel = collections.defaultdict(collections.Counter)
gauge = {}; known = {}
for r in rows:
    d = r.get("day"); a = r.get("action")
    if d is None or d < first:
        continue
    days[d][a] += 1
    if a == "smile" and "cue" in str(r.get("why")):
        days[d]["cue_answered"] += 1
    if a == "session_end":
        known[d] = r.get("known")
    if a == "night":
        n = r.get("last_night") or {}
        n = n if isinstance(n, dict) else ast.literal_eval(str(n))
        g = n.get("gauge") or {}; gauge[d] = (g.get("before"), g.get("after"))
    ctx = r.get("context")
    if ctx:
        for seg in re.split(r"_+", ctx):
            ws = re.findall(r"[a-z]+", seg.lower())
            for x, y in zip(ws, ws[1:]):
                if x in words and y in words and len(x) > 1 and len(y) > 1:
                    pairs[d][(x, y)] += 1
                    if (x, y) not in taught:
                        novel[d][(x, y)] += 1
for d in sorted(days):
    c = days[d]; g = gauge.get(d)
    nv = novel[d]
    print(f"day {d:3d} | smiles {c['smile']:3d} frowns {c['frown']:3d} aways {c['away']:2d} | lines {c['line']:3d} cues {c['cue']:2d} answered {c['cue_answered']:2d} | own pairs {sum(pairs[d].values()):3d} distinct {len(pairs[d]):2d} novel {sum(nv.values()):2d} {[' '.join(k) for k, _ in nv.most_common(3)]} | known {known.get(d, '-')} | night {g[0] if g else '-'} -> {g[1] if g else '-'}")
