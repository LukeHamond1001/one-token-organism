"""THE TEACHER'S CHAIR (2026-09-17, the user's word: "you talk to it live like you have my UI, watch it and teach it"): the supervisor
sits at the served body's page from the terminal, one step per call, the session kept in a small state file. The parent's typist is
held (SIGSTOP) from `start` to `end`, exactly as tools/rehearse.py holds it; the face is timed by this tool as the caregiver times it
(a known word +2 for 2.4 s, an answer 2 then 4) so the words are the teacher's and the smiles land when the word does. `leave`
is a way out of the room: no words, no face, for a while, the typist still held, so the child is alone.
usage: python3 tools/teach_live.py start
       python3 tools/teach_live.py say "what is cold?" [--who parent|other] [--watch 12] [--answer "ice,cold"]
       python3 tools/teach_live.py listen 20          (watch its own talk for 20 s, smiling at its words)
       python3 tools/teach_live.py leave 120          (leave the room for 120 s; prints what it said alone)
       python3 tools/teach_live.py end
The page is its only sense of us: every symbol typed here appears on it as the parent's or the other voice's; nothing here reads
from inside it except the mood shown to the supervisor at start and end (never to the child, never to a rule)."""
import sys, os, re, json, time, signal, subprocess, urllib.request
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STATE = os.path.join(ROOT, "data", "teach_live_state.json"); PORT = 8020; BASE = f"http://localhost:{PORT}"
def get(path): return json.load(urllib.request.urlopen(BASE + path, timeout=10))
def post(path, body):
    req = urllib.request.Request(BASE + path, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"}); return json.load(urllib.request.urlopen(req, timeout=10))
def arg(name, default):
    for i, a in enumerate(sys.argv[1:], 1):
        if a == "--" + name and i + 1 < len(sys.argv): return type(default)(sys.argv[i + 1])
    return default
def own_since(n):
    st = get(f"/state?since={n}"); return "".join(e[0] for e in st["page"] if e[1] == 1 and e[0]), st["n"]
def page_tail(n_entries=400):
    st = get("/state?since=0"); page = st["page"][-n_entries:]; out = []
    for e in page:
        if not e[0]: continue
        tag = "B" if e[1] == 1 else {"parent": "A", "other": "b", "you": "y"}.get(e[4], "?")
        if out and out[-1][0] == tag: out[-1][1] += e[0]
        else: out.append([tag, e[0]])
    return " | ".join(f"{t}:{x}" for t, x in out[-14:])
KNOWN = set(w for w in re.findall(r"[a-z]+", open(os.path.join(ROOT, "data/teach_queue_w2.jsonl")).read().lower()) if len(w) >= 2)
face_off_at = [0.0]; word_tick = {}; smiles = [0]
def smile_at(level, dur=2.4): post("/face", {"expr": level}); face_off_at[0] = time.time() + dur; smiles[0] += 1
def face_tend():
    if face_off_at[0] and time.time() >= face_off_at[0]: post("/face", {"expr": 0}); face_off_at[0] = 0.0
def watch(seconds, keys=None, n0=None):
    """watch the child's turn for `seconds`, smiling at its known words as they complete and at the answer's word; return its text"""
    n0 = get("/state?since=0")["n"] if n0 is None else n0
    got = ""; seen = 0; answered = None; t0 = time.time(); post_then = None
    while time.time() - t0 < seconds:
        time.sleep(0.2); face_tend(); txt, n = own_since(n0)
        if txt != got:
            got = txt; low = got.lower(); words = re.findall(r"[a-z]+", low); done = words[:-1] if low and low[-1].isalpha() else words
            if keys and answered is None and any(re.search(r"\b" + re.escape(k), low) for k in keys):
                answered = round(time.time() - t0, 1); smile_at(2.0); post_then = (4.0, time.time() + 2.4)
            elif len(done) > seen:
                wd = done[seen]; seen = len(done)
                if wd in KNOWN and time.time() - word_tick.get(wd, 0) > 24: smile_at(2.0); word_tick[wd] = time.time()
            else: seen = max(seen, len(done))
        if post_then and time.time() >= post_then[1]: smile_at(post_then[0]); post_then = None
    face_tend(); post("/face", {"expr": 0}); face_off_at[0] = 0.0
    return got, answered
def wait_quiet(quiet_ticks=8, max_wait=150):
    """wait for the child's quiet as the typist does, attending (a known word said meanwhile earns its smile)"""
    n0 = get("/state?since=0")["n"]; silent = 0; waited = 0; got = ""; seen = 0
    while silent < quiet_ticks and waited < max_wait:
        time.sleep(0.2); face_tend(); txt, n = own_since(n0)
        if txt: silent = 0; got += txt; n0 = n
        else: silent += 1
        waited += 1
    return got
cmd = sys.argv[1] if len(sys.argv) > 1 else "help"
if cmd == "start":
    assert not os.path.exists(STATE), "a session is open already (end it first)"
    pids = [int(x) for x in subprocess.run(["pgrep", "-f", f"body.teacher --port {PORT}"], capture_output=True, text=True).stdout.split()]
    for p in pids: os.kill(p, signal.SIGSTOP)
    json.dump({"typist": pids, "t0": time.time()}, open(STATE, "w"))
    md = get("/insides"); print(f"in the chair; the typist held ({pids}); mood {md['mood']:.1f} sharp {md['sharp_now']:.0f} nights {md['nights']} asleep {get('/state?since=0').get('asleep')}")
    print("the page:", page_tail())
elif cmd == "say":
    text = sys.argv[2]; who = arg("who", "parent"); w = arg("watch", 12); keys = [k.strip() for k in arg("answer", "").split(",") if k.strip()]
    before = wait_quiet()
    n0 = get("/state?since=0")["n"]; post("/type", {"text": text, "who": who})
    while get("/state?since=0")["queued"] > 0: time.sleep(0.2)
    got, at = watch(w, keys, n0)
    print(f"[{who}] {text}\n[child] {got!r}" + (f"  (answered at {at}s)" if at else "") + (f"  (before the line it said {before!r})" if before.strip() else ""))
elif cmd == "listen":
    got, _ = watch(int(sys.argv[2]) if len(sys.argv) > 2 else 20); print(f"[child, alone with us listening] {got!r}")
elif cmd == "leave":
    secs = int(sys.argv[2]) if len(sys.argv) > 2 else 120; post("/face", {"expr": 0}); n0 = get("/state?since=0")["n"]; time.sleep(secs)
    got, _ = own_since(n0); print(f"[the room empty for {secs}s; the child alone said] {got!r}")
elif cmd == "end":
    st = json.load(open(STATE)) if os.path.exists(STATE) else {"typist": []}
    post("/face", {"expr": 0})
    for p in st["typist"]:
        try: os.kill(p, signal.SIGCONT)
        except ProcessLookupError: pass
    if os.path.exists(STATE): os.remove(STATE)
    md = get("/insides"); print(f"out of the chair; the typist resumed ({st['typist']}); mood {md['mood']:.1f} sharp {md['sharp_now']:.0f}; the page: {page_tail()}")
else:
    print(__doc__)
