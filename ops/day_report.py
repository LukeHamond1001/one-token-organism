"""THE DAY'S THREE NUMBERS (2026-09-19, the user's word: judge by the counts, daily): for the rows between two times, the words said
over the parent's one-handed lines (per line, and the share clean; conversation and story lines apart), the answers (the answer's
smile on the questions, its delay after the line's end), and the babble into silence per minute of silence (known words said past
the child's turn, over the minutes with no line in progress and no turn after one).
usage: python3 ops/day_report.py 2026-09-19T17:12 2026-09-19T18:17 [--label "day 344"]"""
import sys, json, statistics, datetime as dt, os
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
a, b = sys.argv[1], sys.argv[2]
label = sys.argv[sys.argv.index("--label") + 1] if "--label" in sys.argv else f"{a[5:16]} to {b[11:16]}"
rows = [json.loads(l) for l in open(os.path.join(ROOT, "data/watch2_caregiver.jsonl")) if l.strip()]
day = [r for r in rows if a <= r.get("ts", "") < b]
def T(r): return dt.datetime.fromisoformat(r["ts"]).timestamp()
lines = [r for r in day if r.get("action") in ("line", "cue") and r.get("voice", "a") == "a" and r.get("over") is not None]
all_lines = [r for r in day if r.get("action") in ("line", "cue")]
slow = [r for r in lines if r.get("slow")]; fast = [r for r in lines if not r.get("slow")]
names = ("she","he","they","one","the","tim","lily","mia","tom","max","sam","ben","anna","lucy","spot","mom","dad","timmy","tommy","it","then","but","when","so","once","a","there","i","we")
story = [r for r in slow if r["text"] and r["text"][0].islower() and r["text"].rstrip()[-1] in ".!?" and "?" not in r["text"] and (" said " in r["text"] or r["text"].split()[0] in names)]
conv = [r for r in slow if r not in story]
def over(L): return (sum(int(r["over"]) for r in L) / len(L), 100 * sum(1 for r in L if int(r["over"]) == 0) / len(L)) if L else (0, 0)
q = [l for l in lines if l["text"].strip().endswith("?")]
ans = [r for r in day if r.get("action") == "smile" and str(r.get("why", "")).startswith("answer")]
lat = []
for s in ans:
    prev = [l for l in lines if T(l) <= T(s)]
    if prev:
        l = max(prev, key=T); lat.append(T(s) - T(l) - len(l["text"]) * (0.75 if l.get("slow") else 0.26))
sil = [r for r in day if r.get("action") == "missed" and r.get("why") == "in silence"]
# the silent minutes: the day's span minus the lines' typing and the turn after each (listen 6 s + 2 s)
span = (T(day[-1]) - T(day[0])) / 60 if len(day) > 1 else 0
busy = sum(len(r["text"]) * (0.75 if r.get("slow") else 0.26) + 8.0 for r in all_lines) / 60
silent = max(0.1, span - busy)
frowns = sum(1 for r in day if r.get("action") == "frown")
print(f"{label}: {len(lines)} parent lines ({len(slow)} slow), {len(all_lines)} lines in all, {span:.0f} min, about {silent:.0f} silent")
o, c = over(slow); print(f"  words over a slow line: {o:.2f} per line, {c:.0f}% clean | conversation {over(conv)[0]:.2f} ({over(conv)[1]:.0f}% clean, {len(conv)} lines) | story {over(story)[0]:.2f} ({len(story)} lines) | fast lines {over(fast)[0]:.2f}")
print(f"  answers: the smile on {len(ans)} of {len(q)} questions ({100*len(ans)/max(1,len(q)):.0f}%), a median {statistics.median(lat) if lat else float('nan'):.1f} s after the line's end")
print(f"  babble into silence: {len(sil)} known words past its turn, {len(sil)/silent:.1f} per silent minute | frowns {frowns}")
