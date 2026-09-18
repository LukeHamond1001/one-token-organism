"""THE WORD RATE (a supervisor's instrument, 2026-09-17, the user's word: "say words and not random letters"): of the child's own runs
of letters on the page, how many are words its parents have typed (the vocabulary of every line ever typed to it). Read from the page
log for given days, or from a text of its own symbols (--text). No mechanism reads this; it is a ruler.
usage: python3 tools/word_rate.py --days 297-300 [--log data/watch2_caregiver.jsonl]     |     python3 tools/word_rate.py --text "..."
prints: own words N, in the parents' vocabulary M (the rate), the commonest of its words in and out of it"""
import sys, json, re, collections
def arg(name, default):
    for i, a in enumerate(sys.argv[1:], 1):
        if a == "--" + name and i + 1 < len(sys.argv): return type(default)(sys.argv[i + 1])
    return default
log = arg("log", "/Users/lukehamond/Projects/project/data/watch2_caregiver.jsonl"); days = arg("days", ""); text = arg("text", "")
R = [json.loads(l) for l in open(log) if l.startswith("{")]
vocab = collections.Counter(w for r in R if r.get("action") in ("line", "cue") for w in re.findall(r"[a-z]+|I", r.get("text", "")))
def rate(own_text, label):
    words = re.findall(r"[a-z]+|I", own_text); words = [w for w in words if len(w) >= 2 or w in ("I", "a")]
    inv = [w for w in words if w in vocab]; out = [w for w in words if w not in vocab]
    print(f"{label}: own words {len(words)}, in the parents' vocabulary {len(inv)} ({100.0 * len(inv) / max(1, len(words)):.1f}%) | "
          f"commonest in: {', '.join(w for w, _ in collections.Counter(inv).most_common(8))} | commonest out: {', '.join(w for w, _ in collections.Counter(out).most_common(8))}")
    return len(words), len(inv)
if text:
    if text.endswith(".json"):
        d = json.load(open(text)); rate(d.get("in_turn", ""), "in its turn"); rate(d.get("over_the_line", ""), "over the line")
    else:
        rate(text, "the text")
else:
    d0, d1 = (int(x) for x in days.split("-")) if "-" in days else (int(days), int(days))
    tot = [0, 0]
    for day in range(d0, d1 + 1):
        own = "".join(r.get("its_after", "").replace("_", "") for r in R if r.get("action") == "line" and r.get("day") == day)
        if own:
            n, m = rate(own, f"day {day}"); tot[0] += n; tot[1] += m
    if d1 > d0: print(f"days {d0}-{d1}: own words {tot[0]}, in the vocabulary {tot[1]} ({100.0 * tot[1] / max(1, tot[0]):.1f}%)")
