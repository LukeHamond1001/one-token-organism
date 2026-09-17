"""THE FLOOR'S CUT ON A COPY (a supervisor's instrument, 2026-09-17): load a saved body on a copy, forget the slots below an absolute
strength (what the first night under store_floor_abs would forget beyond the fade), and save the copy under another name for the
rulers (probe_lm, qa_by_gap, branch_probe) to read against the untouched save.
usage: python3 tools/floor_cut.py SAVE.pt OUT.pt 0.07"""
import sys
sys.path.insert(0, "/Users/lukehamond/Projects/project")
import torch
from tokenizers import Tokenizer
from body.life import Life

path, out, F = sys.argv[1], sys.argv[2], float(sys.argv[3])
TOK = Tokenizer.from_file("/Users/lukehamond/Projects/project/data/tok_char.json")
life = Life.load(path, TOK, device="cpu", cfg={}, seed=0)
st = life.store; n0 = st.n(); mean0 = float(st.S.mean())
keep = torch.nonzero(st.S >= F).flatten(); st._keep(keep)
print(f"store {n0} -> {st.n()}: {n0 - st.n()} slots below {F} forgotten (the fade not applied; the store's mean {mean0:.4f} -> {float(st.S.mean()):.4f})")
life.save_path = out; life.save(); print(f"saved as {out}")
