"""REVISITING (2026-09-20): a few of the parent's own earlier rows, drawn at random from the queue's past, appended again, so the
facts taught weeks ago are heard now and then as a parent revisits old lessons. The parent's own material only; nothing from tools/.
usage: python3 ops/revisit_rows.py --rows 3 [--after 2600] [--before 6400] [--seed N]   (the stage-four days; the newest rows excluded so it is a revisit)"""
import sys, json, random, os
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
def arg(name, default):
    for i, a in enumerate(sys.argv[1:], 1):
        if a == "--" + name and i + 1 < len(sys.argv): return type(default)(sys.argv[i + 1])
    return default
n = arg("rows", 3); after = arg("after", 2600); before = arg("before", 6400); seed = arg("seed", int.from_bytes(os.urandom(4), "little"))
path = os.path.join(ROOT, "data/teach_queue_w2.jsonl")
rows = []
with open(path) as f:
    for i, l in enumerate(f):
        if i >= before: break
        if i < after: continue
        try:
            r = json.loads(l)
        except Exception:
            continue
        say = r.get("say", [])
        if len(say) == 4 and any("?" in s for s in say) and all(len(s.replace("b: ", "")) <= 38 for s in say):
            rows.append(say)
rng = random.Random(seed); pick = rng.sample(rows, min(n, len(rows)))
with open(path, "a") as f:
    for say in pick:
        f.write(json.dumps({"say": say}) + "\n")
print(f"revisited {len(pick)} of {len(rows)} early rows: " + " | ".join(s[0][:30] for s in pick))
