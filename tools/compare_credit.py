"""THE BEHAVIORAL YARDSTICK for fast arms: from their page logs (body/fastlife --log data/fast_rN.jsonl), smiles, turnings-away
and frowns per day for two groups of runs, and the paired daily differences on the days both groups lived.
    python3 tools/compare_credit.py 187,188 183,184 [--from DAY]
"""
import sys, json, collections, statistics as st, os
A = [int(x) for x in sys.argv[1].split(",")]; B = [int(x) for x in sys.argv[2].split(",")]
fr = int(sys.argv[sys.argv.index("--from") + 1]) if "--from" in sys.argv else 0
def days_of(runs):
    out = collections.defaultdict(lambda: collections.defaultdict(list))
    for R in runs:
        f = f"data/fast_r{R}.jsonl"
        if not os.path.exists(f): print(f"  (no log for run {R})"); continue
        per = collections.defaultdict(collections.Counter)
        for l in open(f):
            try: r = json.loads(l)
            except Exception: continue
            d = r.get("day")
            if not isinstance(d, int) or d < fr: continue
            per[d][r.get("action")] += 1
        for d, c in per.items():
            out[d]["smiles"].append(c["smile"]); out[d]["aways"].append(c["away"]); out[d]["frowns"].append(c["frown"]); out[d]["talked_over"].append(c["missed"])
    return out
da, db = days_of(A), days_of(B); common = sorted(set(da) & set(db))
def summ(name, dd, runs):
    ds = sorted(dd)
    if not ds: print(f"  {name} runs {runs}: no days"); return
    print(f"  {name:18s} runs {runs}: days {ds[0]}..{ds[-1]} | smiles/day {st.mean([st.mean(dd[d]['smiles']) for d in ds]):5.1f} | aways/day {st.mean([st.mean(dd[d]['aways']) for d in ds]):4.1f} | frowns/day {st.mean([st.mean(dd[d]['frowns']) for d in ds]):5.1f} | misses/day {st.mean([st.mean(dd[d]['talked_over']) for d in ds]):5.1f}")
    print("      smiles by day " + " ".join(f"{d}:{st.mean(dd[d]['smiles']):.0f}" for d in ds) + " | aways by day " + " ".join(f"{d}:{st.mean(dd[d]['aways']):.0f}" for d in ds))
summ("A", da, A); summ("B", db, B)
if common:
    ds_ = [st.mean(da[d]["smiles"]) - st.mean(db[d]["smiles"]) for d in common]; dw = [st.mean(da[d]["aways"]) - st.mean(db[d]["aways"]) for d in common]
    print(f"  A - B on the {len(common)} shared days: smiles {st.mean(ds_):+.1f}/day (sd {st.pstdev(ds_):.1f}; A above on {sum(1 for x in ds_ if x > 0)}/{len(common)}) | aways {st.mean(dw):+.1f}/day (A below on {sum(1 for x in dw if x < 0)}/{len(common)})")
