"""A LIVED DAY RECORDED: a copy of a saved body lives one fast day; every tick's band states (float16), felt reward, fast value,
fast error and the page rows are saved, so critics of any input and horizon can be fitted and read offline (fit on one day, read
on another of the same body).   python3 tools/record_day.py BODY.pt SEED [N] [OUT.pt]
"""
import os, sys, shutil, random, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
torch.set_num_threads(2)
from tokenizers import Tokenizer
from body.life import Life
from body.fastlife import FastCaregiver

src = sys.argv[1]; seed = int(sys.argv[2]); N = int(sys.argv[3]) if len(sys.argv) > 3 else 12000
name = os.path.basename(src).replace(".pt", "")
out = sys.argv[4] if len(sys.argv) > 4 else f"data/stalks/nt_day_{name}_{seed}.pt"
os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
tok = Tokenizer.from_file("data/tok_char.json")
tmp = f"{os.path.dirname(out) or '.'}/tmp_rec_{name}_{seed}_{os.getpid()}.pt"; shutil.copy(src, tmp)
L = Life.load(tmp, tok, save_path=None); os.remove(tmp)
L.cfg["wake_ticks"] = L.sleep_pressure + N
fb = int(L.cfg["dopamine_band"])
B = []; R = []; VF = []; DF = []; SAID = []; T = []; rows = []; cg_ref = []
orig_tick = L.tick
def tick():
    orig_tick()
    la = L.last
    with torch.no_grad():
        B.append(L.bands.detach().to(torch.float16).cpu().clone()); VF.append(float(L.m.values(L.bands)[fb]))
    R.append(float(la.get("felt", 0) or 0)); DF.append(float(la.get("dopamine") or 0)); SAID.append(la.get("said", "")); T.append(L.ticks)
L.tick = tick
cg = FastCaregiver(L, L.day_n + 1, [], random.Random(seed)); cg_ref.append(cg)
orig_row = cg.row
def row(obj):
    obj = dict(obj); obj["t"] = L.ticks; rows.append(obj); return orig_row(obj)
cg.row = row
cg.run_day()
torch.save({"bands": torch.stack(B), "r": torch.tensor(R), "vf": torch.tensor(VF), "dopa": torch.tensor(DF), "said": SAID, "t": torch.tensor(T),
            "rows": rows, "smiles": cg.smiles, "aways": cg.aways, "frowns": cg.frowns, "src": src, "seed": seed, "clocks": [int(c) for c in L.m.clocks]}, out)
print(f"recorded {len(T)} ticks of {name} (seed {seed}): smiles {cg.smiles} aways {cg.aways} frowns {cg.frowns} -> {out}")
