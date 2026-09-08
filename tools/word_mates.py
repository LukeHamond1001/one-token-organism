"""THE CONNECTIONS A WORD HAS, in the child's own mouth: for each word asked, how many distinct frame-mates the child itself has
paired it with in its own stream (read through the page log's context windows, the parent's symbols masked), how many times,
and how many of those pairings the parent never said before the child did. A floor, not a count: the windows are short.
usage: python3 tools/word_mates.py data/watch2_caregiver.jsonl data/watch2_corpus.json out and me little baby hat eat"""
import collections
import json
import re
import sys

log, corpus_path = sys.argv[1], sys.argv[2]; asked = sys.argv[3:]
rows = [json.loads(l) for l in open(log)]
words = set()
for line in json.load(open(corpus_path))["lines"]:
    words.update(re.findall(r"[a-z]+", line.lower()))
taught = set(); mates = collections.defaultdict(collections.Counter); novel = collections.defaultdict(collections.Counter); first = {}
for r in rows:
    a = r.get("action"); d = r.get("day")
    if a in ("line", "cue"):
        ws = re.findall(r"[a-z]+", (r.get("text") or "").lower())
        for w in ws:
            first.setdefault(w, d)
        taught.update(zip(ws, ws[1:]))
    ctx = r.get("context") or r.get("its_after")
    if not ctx:
        continue
    for seg in re.split(r"_+", ctx):
        ws = re.findall(r"[a-z]+", seg.lower())
        for x, y in zip(ws, ws[1:]):
            if x in words and y in words and len(x) > 1 and len(y) > 1:
                for w, m in ((x, y), (y, x)):
                    mates[w][m] += 1
                    if (x, y) not in taught:
                        novel[w][m] += 1
for w in asked or sorted(mates, key=lambda k: -len(mates[k]))[:12]:
    c, n = mates[w], novel[w]
    print(f"{w:8} (entered day {first.get(w, '?')}) | {len(c):3d} mates, {sum(c.values()):4d} pairings | never taught: {len(n):2d} mates {[m for m, _ in n.most_common(4)]}")
