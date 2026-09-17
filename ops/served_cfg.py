"""THE SERVED CONSTANTS AS THEY ARE (2026-09-16, the reviewer's finding): a save carries its own constants, and a reload passes only the
flags file's keys, so a key left out of the file keeps the save's value. This prints, for a save, every constant that differs from the
physiology's default, and beside each whether the flags file sets it, sets it differently, or leaves it to the save.
usage: python3 ops/served_cfg.py [SAVE.pt] [FLAGS.txt]"""
import sys, os, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from body.life import PHYSIOLOGY
save = sys.argv[1] if len(sys.argv) > 1 else "data/watch2.pt"; flags = sys.argv[2] if len(sys.argv) > 2 else "ops/BASE_FLAGS.txt"
cfg = torch.load(save, map_location="cpu", weights_only=False).get("cfg") or {}
toks = open(flags).read().split(); fl = {}
i = 0
while i < len(toks):
    if toks[i].startswith("--"):
        k = toks[i][2:].replace("-", "_")
        if k in PHYSIOLOGY: fl[k] = type(PHYSIOLOGY[k])(toks[i + 1])
        i += 2
    else: i += 1
eff = dict(cfg); eff.update(fl)          # what Life.load does: the save's constants, then the flags
rows = []
for k in sorted(set(eff) | set(fl)):
    if k not in PHYSIOLOGY: continue
    v = eff[k]; d = PHYSIOLOGY[k]
    if v == d and k not in fl: continue
    src = "flags" if k in fl and (k not in cfg or cfg[k] == fl[k]) else ("FLAGS OVERRIDE the save's %r" % cfg.get(k) if k in fl else "the save alone (not in the flags)")
    rows.append((k, v, d, src))
print(f"{os.path.basename(save)}: {len(rows)} constants off their defaults or set by the flags")
for k, v, d, src in rows: print(f"  {k:20} = {v!s:12} (default {d!s:10})  <- {src}")
