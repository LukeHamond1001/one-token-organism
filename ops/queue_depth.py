"""THE QUEUE'S DEPTH, TRULY (2026-09-14): the typist reads every new byte of the queue at once and keeps the lines in memory, so the
.pos file's byte position is the file's end and its buffer is not the truth. The depth is the number of queued lines after the
newest occurrence of the last line the typist actually typed (the page log's newest "line" row).
usage: python3 queue_depth.py   (from /Users/lukehamond/Projects/project)"""
import json, os
os.chdir("/Users/lukehamond/Projects/project")
R = [json.loads(l) for l in open("data/watch2_caregiver.jsonl") if l.strip()]
typed = [r["text"].strip() for r in R if r["action"] == "line"]
last = typed[-1] if typed else ""
flat = []
for l in open("data/teach_queue_w2.jsonl"):
    if l.strip():
        for s in json.loads(l).get("say", []):
            s = s.strip()
            flat.append(s[2:].strip() if s[:2].lower() == "b:" else s)
# the newest place where the last THREE typed lines occur in order (a line duplicated later in the queue fooled the single match:
# the sixteenth parent's finding, 12:33)
seq = typed[-3:]
hits = [k for k in range(len(flat)) if flat[max(0, k - len(seq) + 1):k + 1] == seq] or [k for k, s in enumerate(flat) if s == last]
idx = hits[-1] if hits else len(flat) - 1
ahead = len(flat) - 1 - idx
print(f"lines ahead {ahead} (about {ahead * 13 / 60:.0f} min of talk); last typed {last!r}; queue rows {sum(1 for l in open('data/teach_queue_w2.jsonl') if l.strip())}")
