"""THE RISE BEFORE THE SMILE: from a stalked day (tools/stalk_day.py), the fast critic's value and error tick by tick around
each smile, turning-away and frown, and the rise of the value over the eight ticks before a smile as a fraction of the smile.
A slow band cannot cancel an impulse on the reward's tick; expectation is the rise before it.

    python3 tools/vf_profile.py data/stalks/nt_stalk_X_1.pt [more...]
"""
import sys, statistics as st, torch

def profile(byt, rows, mv, evt, label, span=8):
    ev = [r["t"] for r in rows if r.get("action") == evt]
    if not ev:
        print(f"  {label}: none"); return
    prof = {k: [] for k in range(-span, span + 1)}; err = {k: [] for k in range(-span, span + 1)}
    for t in ev:
        for k in range(-span, span + 1):
            x = byt.get(t + k)
            if x is not None and x.get("vf") is not None:
                prof[k].append(x["vf"] - mv); err[k].append(x.get("dopa") or 0.0)
    ks = [k for k in range(-span, span + 1) if prof[k]]
    print(f"  {label} (n {len(ev)}): value minus the day's mean, by tick   " + " ".join(f"{k:+d}:{st.mean(prof[k]):+.2f}" for k in ks))
    print(f"  {' ' * len(label)}          fast error by tick                 " + " ".join(f"{k:+d}:{st.mean(err[k]):+.2f}" for k in ks))
    if prof[-1] and prof[-span]:
        rise = st.mean(prof[-1]) - st.mean(prof[-span])
        print(f"  {' ' * len(label)}   the rise: value at -1 minus value at -{span} = {rise:+.3f} ({100 * rise / 2:.0f}% of a smile of 2)")

for f in sys.argv[1:]:
    d = torch.load(f, map_location="cpu", weights_only=False); T = d["ticks"]; rows = d["rows"]
    byt = {x["t"]: x for x in T}
    vf = [x.get("vf") for x in T if x.get("vf") is not None]
    if not vf:
        print(f"{f}: no fast value recorded"); continue
    mv = st.mean(vf); sv = st.pstdev(vf)
    print(f"{f.split('/')[-1]}: ticks {len(T)} | fast value mean {mv:+.3f} sd {sv:.3f} | felt sum {sum((x.get('felt') or 0) for x in T):+.0f} | smiles {d.get('smiles')} aways {d.get('aways')} frowns {d.get('frowns')}")
    profile(byt, rows, mv, "smile", "smile"); profile(byt, rows, mv, "away", "turning-away"); profile(byt, rows, mv, "frown", "frown")
