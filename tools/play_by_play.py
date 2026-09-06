"""PLAY BY PLAY of a watched day (tools/watch_life.py logs): every event (smile, frown, away, cue, withheld reply) with the ticks
around it: what the parent typed, what the body said, the reward felt, the dopamine, the fast value, the gate, working memory,
and the planner's choice when it decided. Also the day's organ summary.
    python3 tools/play_by_play.py data/watch/watch1/day001 [--events 12] [--span 6] [--kinds smile,frown,away,cue]
"""
import sys, json, argparse, statistics as st
ap = argparse.ArgumentParser(); ap.add_argument("prefix"); ap.add_argument("--events", type=int, default=12); ap.add_argument("--span", type=int, default=6)
ap.add_argument("--kinds", default="smile,frown,away,cue"); a = ap.parse_args()
T = [json.loads(l) for l in open(a.prefix + "_ticks.jsonl")]; R = [json.loads(l) for l in open(a.prefix + "_rows.jsonl")]
byt = {x["t"]: x for x in T}; kinds = set(a.kinds.split(","))
# the parent's typing per tick is known; what it typed is in the rows ("line"/"cue" text) -> mark the span
ev = [r for r in R if r.get("action") in kinds]
step = max(1, len(ev) // a.events); shown = ev[::step][:a.events]
def fmt(x):
    said = x.get("said") or ("·" if not x.get("typing") else "…")
    pl = x.get("plan"); pls = ""
    if pl: pls = " plan[" + " ".join(f"{c!r}:{v:+.2f}" for c, v in zip(pl["cands"], pl["vals"])) + "]"
    return f"t{x['t']:>7} {'P' if x.get('typing') else ' '} said {said!r:5} r {x.get('felt',0):+d} δ {x.get('dopa') or 0:+.2f} V {x.get('vf') or 0:+.2f} gate {x.get('gate') or 0:.2f} wm {int(x.get('wm') or 0)}{'*' if x.get('wm_latch') else ' '}{pls}"
for r in shown:
    t = r["t"]; print(f"== {r.get('action')} at t{t}: {({k: v for k, v in r.items() if k in ('on','why','text','e')})}")
    for k in range(-a.span, a.span + 1):
        if t + k in byt: print(("  >" if k == 0 else "   ") + fmt(byt[t + k]))
sm = [r["t"] for r in R if r.get("action") == "smile"]; vf = [x["vf"] for x in T if x.get("vf") is not None]; mv = st.mean(vf) if vf else 0
print(f"\nday: {len(T)} ticks | smiles {len(sm)} completions {sum(1 for r in R if r.get('action')=='smile' and 'completion' in str(r.get('why')))} frowns {sum(1 for r in R if r.get('action')=='frown')} aways {sum(1 for r in R if r.get('action')=='away')} cues {sum(1 for r in R if r.get('action')=='cue')} | "
      f"spoke while the parent typed {st.mean(1.0 if x.get('said') else 0.0 for x in T if x.get('typing')) if any(x.get('typing') for x in T) else float('nan'):.2f} vs quiet {st.mean(1.0 if x.get('said') else 0.0 for x in T if not x.get('typing')):.2f} | "
      f"wm latches {sum(1 for x in T if x.get('wm_latch'))}, held {st.mean(x.get('wm') or 0 for x in T):.2f} of the day | planner decisions {sum(1 for x in T if x.get('plan'))}")
