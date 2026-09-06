"""THE GATE'S EARS: from a stalked day (tools/stalk_day.py), how often the body speaks (a symbol of its own on the tick) and
what its gate says, by context: the parent typing vs quiet, the ticks after a talked-over miss, after a smile, after a frown,
while the parent is turned away. A gate with ears speaks less over the parent and after being talked over.
    python3 tools/gate_context.py data/stalks/nt_stalk_X_1.pt [more...]
"""
import sys, statistics as st, torch
W = 12
for f in sys.argv[1:]:
    d = torch.load(f, map_location="cpu", weights_only=False); T = d["ticks"]; rows = d["rows"]
    byt = {x["t"]: x for x in T}; ts = sorted(byt)
    def ctx_after(evt, why=None):
        ev = [r["t"] for r in rows if r.get("action") == evt and (why is None or why in str(r.get("why", "")))]
        s = set()
        for t in ev: s.update(range(t + 1, t + 1 + W))
        return s
    after_over = ctx_after("missed", "talked over"); after_smile = ctx_after("smile"); after_frown = ctx_after("frown")
    def read(name, sel):
        xs = [byt[t] for t in sel if t in byt]
        if len(xs) < 20: print(f"  {name:34s}: n {len(xs):5d} (too few)"); return
        acted = st.mean(1.0 if (x.get("said") or "") != "" else 0.0 for x in xs); g = [x["gate"] for x in xs if x.get("gate") is not None]
        print(f"  {name:34s}: n {len(xs):5d} | spoke {acted:.2f} | gate p(act) {st.mean(g) if g else float('nan'):.2f}")
    has_typing = any("typing" in x for x in T)
    print(f"{f.split('/')[-1]}: {len(T)} ticks | smiles {d.get('smiles')} aways {d.get('aways')} frowns {d.get('frowns')}")
    read("all ticks", ts)
    if has_typing:
        read("parent typing", [t for t in ts if byt[t].get("typing")]); read("parent quiet", [t for t in ts if not byt[t].get("typing")])
    read(f"{W} ticks after a talked-over miss", after_over); read(f"{W} ticks after a smile", after_smile); read(f"{W} ticks after a frown", after_frown)
    read("parent turned away", [t for t in ts if byt[t].get("away")])
