"""STALKING A DAY: a copy of a saved body lives one fast day and every tick is written down: what the parent typed, what the
body said, the reward it felt, the parent's attention, the ventral value, the fast dopamine error, the fast value itself, the
long error, the gate's probability, the clock, fatigue; and every page row (smiles, aways, frowns, cues, misses) with its tick.

    python3 tools/stalk_day.py BODY.pt SEED [N] [OUT.pt]      (N ticks, default 12000 = the body's own day; the body file is never touched)
"""
import os, sys, shutil, random, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
torch.set_num_threads(2)
from tokenizers import Tokenizer
from body.life import Life
from body.fastlife import FastCaregiver

src = sys.argv[1]; seed = int(sys.argv[2]); N = int(sys.argv[3]) if len(sys.argv) > 3 else 12000
name = os.path.basename(src).replace(".pt", "")
out = sys.argv[4] if len(sys.argv) > 4 else f"data/stalks/nt_stalk_{name}_{seed}.pt"
os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
tok = Tokenizer.from_file("data/tok_char.json")
tmp = f"{os.path.dirname(out) or '.'}/tmp_stalk_{name}_{seed}_{os.getpid()}.pt"; shutil.copy(src, tmp)
L = Life.load(tmp, tok, save_path=None); os.remove(tmp)
L.cfg["wake_ticks"] = L.sleep_pressure + N            # the clock runs 0 -> 1 over this day
fb = int(L.cfg["dopamine_band"])
ticks = []; rows = []; cg_ref = []
orig_tick = L.tick
def tick():
    typing = len(L.queue) > 0                        # the parent's symbol enters this tick (the queue feeds one symbol a tick)
    orig_tick()
    la = L.last
    with torch.no_grad():
        vf = float(L.m.values(L.bands)[fb]) if getattr(L, "bands", None) is not None else None
    ticks.append({"t": L.ticks, "said": la.get("said", ""), "felt": la.get("felt", 0), "gate": la.get("gate"), "dopa": la.get("dopamine"),
                  "vlong": la.get("vlong"), "dlong": la.get("dlong"), "level": la.get("level"), "e": cg_ref[0].e if cg_ref else None,
                  "away": (cg_ref[0].away_until > L.ticks) if cg_ref else False, "clock": float(L.m.vc_clock), "fatigue": la.get("fatigue"),
                  "vf": vf, "typing": typing})
L.tick = tick
cg = FastCaregiver(L, L.day_n + 1, [], random.Random(seed)); cg_ref.append(cg)
orig_row = cg.row
def row(obj):
    obj = dict(obj); obj["t"] = L.ticks; rows.append(obj); return orig_row(obj)
cg.row = row
cg.run_day()
torch.save({"ticks": ticks, "rows": rows, "smiles": cg.smiles, "aways": cg.aways, "frowns": getattr(cg, "frowns", 0), "src": src, "seed": seed}, out)
print(f"stalked {len(ticks)} ticks of {name} (seed {seed}): smiles {cg.smiles} aways {cg.aways} frowns {getattr(cg, 'frowns', 0)} -> {out}")
