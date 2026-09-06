"""THE WATCHED LIFE'S TREND: one row per day from the digest, the numbers that say whether each part is learning.
    python3 tools/watch_trend.py data/watch/watch1/digest.txt"""
import sys, re
for line in open(sys.argv[1]):
    m = re.search(r"day\s+(\d+).*?smiles (\d+) \(completions (\d+).*?aways (\d+) frowns (\d+).*?spoke ([\d.]+) while.*?vs ([\d.]+) quiet.*?rise before a smile ([+-]?\d+)%.*?error at the reward ([+-][\d.]+).*?wm latches (\d+).*?planner: (\d+) choices, agreed with the cortex (\d+)", line)
    if not m: continue
    d, sm, comp, aw, fr, ear_t, ear_q, rise, err, wm, pl, ag = m.groups()
    pf = re.search(r"prefrontal: long value vs its realized return ([+-][\d.]+|nan), reliability ([+-][\d.]+), weight in the credit ([\d.]+)", line)
    g = re.search(r"'after': ([\d.]+)", line)
    print(f"day {int(d):2d} | smiles {int(sm):3d} comp {int(comp):2d} aways {int(aw):2d} frowns {int(fr):2d} | ear {float(ear_t):.2f} vs {float(ear_q):.2f} | rise {int(rise):+3d}% err {float(err):+.2f} | wm {int(wm):2d} | actor agreed {int(ag)}/{int(pl)} | prefrontal {('corr ' + pf.group(1) + ' rel ' + pf.group(2) + ' w ' + pf.group(3)) if pf else '-'} | gauge {g.group(1) if g else '-'}")
