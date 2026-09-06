"""EVERY-ORGAN WATCH OF A SERVED BODY'S DAYS, from its page log alone: per day the session length, the sleep pressure at the
session's start (the gap the body lived alone), smiles, cues and their completions, turnings-away, frowns, the parent's
attention at the end, misses by cause with LATE% (the typist arriving after the tick: a load marker), and the night.
    python3 tools/diary_check.py data/body2_caregiver.jsonl [FIRST_DAY] [LAST_DAY]
"""
import sys, json, collections, datetime as dt
f = sys.argv[1]; rows = [json.loads(l) for l in open(f) if l.strip()]
days = sorted({r["day"] for r in rows if isinstance(r.get("day"), int)})
lo = int(sys.argv[2]) if len(sys.argv) > 2 else max(days[0], days[-1] - 6); hi = int(sys.argv[3]) if len(sys.argv) > 3 else days[-1]
def ts(r):
    try: return dt.datetime.fromisoformat(r["ts"]).timestamp()
    except Exception: return None
for d in [x for x in days if lo <= x <= hi]:
    R = [r for r in rows if r.get("day") == d]
    ss = [r for r in R if r.get("action") == "session_start"]; se = [r for r in R if r.get("action") == "session_end"]
    t0 = ts(ss[0]) if ss else None; t1 = ts(se[-1]) if se else (ts(R[-1]) if R else None)
    length = (t1 - t0) / 60 if t0 and t1 else float("nan")
    smiles = [r for r in R if r.get("action") == "smile"]; comp = [r for r in smiles if "complet" in str(r.get("why", ""))]
    cues = [r for r in R if r.get("action") == "cue"]; aways = sum(1 for r in R if r.get("action") == "away"); frowns = sum(1 for r in R if r.get("action") == "frown")
    miss = collections.Counter(str(r.get("why", "")).split(" ")[0] for r in R if r.get("action") == "missed"); nm = sum(miss.values())
    late = 100.0 * miss.get("late", 0) / nm if nm else 0.0
    e_end = se[-1].get("e") if se else None; night = any(r.get("action") == "night" for r in R)
    lines = sum(1 for r in R if r.get("action") == "line")
    print(f"day {d:3d}: session {length:5.0f} min | start pressure {ss[0].get('sleep_pressure') if ss else '?':>5} | lines {lines:3d} cues {len(cues):3d} completions {len(comp):3d} | smiles {len(smiles):3d} aways {aways:2d} frowns {frowns:2d} | e at end {e_end if e_end is not None else '?'} | misses {nm:3d} late {late:3.0f}% {dict(miss.most_common(4))} | night {'yes' if night else 'no'}")
