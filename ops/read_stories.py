"""THE PARENT READS TO IT (2026-09-18, the user's word: all local, all live): story sentences from a corpus queued as the parent's
lines between the conversation's rows, a story's sentences in order, in the body's register (lower case; commas, quotes, colons and
semicolons dropped; split at . ? and !; 8 to 38 symbols of the tokenizer's letters). Each call appends --rows rows of --lines lines
from stories not yet read (a cursor beside the queue: data/teach_queue_w2.stories.pos). Nothing else changes: the typist types
them, the child listens and chimes in, the waking lesson learns them, the night replays them from the utterance memory.
usage: python3 ops/read_stories.py --corpus FILE --rows 3 --lines 4"""
import sys, os, re, json, random
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
def arg(name, default):
    for i, a in enumerate(sys.argv[1:], 1):
        if a == f"--{name}" and i + 1 < len(sys.argv): return type(default)(sys.argv[i + 1])
    return default
corpus = arg("corpus", ""); rows = arg("rows", 3); per = arg("lines", 4); maxlen = 38
Q = os.path.join(ROOT, "data/teach_queue_w2.jsonl"); POS = Q + ".stories.pos"
ok = set("abcdefghijklmnopqrstuvwxyz .?!'")
raw = open(corpus, encoding="utf-8", errors="ignore").read().lower()
stories = raw.split("<|endoftext|>")
try: cur = json.load(open(POS)).get("story", 0)
except Exception: cur = 0
out = []
while len(out) < rows and cur < len(stories):
    st = stories[cur]; cur += 1
    st = st.replace('"', "").replace(",", "").replace(";", "").replace(":", "").replace("\n", " ")
    sents = [re.sub(r"\s+", " ", s).strip() for s in re.split(r"(?<=[.?!])\s+", st)]
    sents = [s for s in sents if 8 <= len(s) <= maxlen and all(ch in ok for ch in s)]
    if len(sents) < per: continue                                   # a story must read whole in short lines
    for i in range(0, len(sents) - per + 1, per):
        out.append(sents[i:i + per])
        if len(out) >= rows: break
with open(Q, "a") as f:
    for r in out: f.write(json.dumps({"say": r}) + "\n")
json.dump({"story": cur}, open(POS, "w"))
print(f"read {len(out)} rows ({sum(len(r) for r in out)} lines, {sum(len(l) for r in out for l in r)} symbols) from stories up to {cur}; e.g. {out[0] if out else None}")
