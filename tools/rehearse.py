"""THE DEMO REHEARSED ON THE SERVED BODY (a supervisor's instrument, 2026-09-16): the questions typed to the living body over its page,
as the recording would type them, with the typist frozen meanwhile (SIGSTOP, thawed at the end whatever happens). The parent waits for
the child's quiet, types the question, reads the child's turn for a window, smiles as the caregiver does (a known word +2 for twelve
ticks; the answer's growing smile 2 then 4), and speaks an ordinary exchange of a logged day between questions so the day keeps its
shape. Facts never typed can be told once (--tell "q|a;q|a") and are asked after the set. The copies' mood artifact does not apply:
this is the body itself, its rewards contingent on what it says.
usage: python3 tools/rehearse.py [--port 8020] [--facts 1-15] [--tell "what is sour?|a lemon is sour;..."] [--ask-tell 1] [--day 246] [--window 60] [--smile 1] [--freeze 1]"""
import sys, os, re, json, time, subprocess, signal, urllib.request
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
def arg(name, default):
    for i, a in enumerate(sys.argv[1:], 1):
        if a == "--" + name and i + 1 < len(sys.argv): return type(default)(sys.argv[i + 1])
    return default
port = arg("port", 8020); facts_range = arg("facts", "1-30"); tell = arg("tell", ""); ask_tell = arg("ask_tell", 1); day = arg("day", 246)
window = arg("window", 60); smile = arg("smile", 1); freeze = arg("freeze", 1); quiet_ticks = arg("quiet", 8); max_wait = arg("max_wait", 200)
BASE = f"http://localhost:{port}"
def get(path):
    return json.load(urllib.request.urlopen(BASE + path, timeout=10))
def post(path, body):
    req = urllib.request.Request(BASE + path, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=10))
def own_since(n):
    st = get(f"/state?since={n}"); return "".join(e[0] for e in st["page"] if e[1] == 1 and e[0]), st["n"]
def tick_now(): return get("/state?since=0")["n"] // 2
STOP = set("is are the a an in on and to do we i you it of my your they can what where who how does".split())
pairs = [tuple(s.strip() for s in l.split("|")[:2]) for l in open(os.path.join(ROOT, "tools/facts_stage5.txt")) if "|" in l]
lo, hi = (int(x) for x in facts_range.split("-")); pairs = pairs[lo - 1:hi]
told = [tuple(s.strip() for s in t.split("|")) for t in tell.split(";") if "|" in t]
R = [json.loads(l) for l in open(os.path.join(ROOT, "data/watch2_caregiver.jsonl")) if l.strip()]
L = [(r["text"], "other" if r.get("voice") == "b" else "you") for r in R if r.get("day") == day and r["action"] == "line"]
ordinary = [(L[i], L[i + 1]) for i in range(0, len(L) - 1, 2) if L[i][1] == "you" and L[i + 1][1] == "other" and not any(f in L[i][0] for f, _ in pairs)]
KNOWN = set(w for w in re.findall(r"[a-z]+", open(os.path.join(ROOT, "data/teach_queue_w2.jsonl")).read().lower()) if len(w) >= 2)
typist = subprocess.run(["pgrep", "-f", "body.teacher --port %d" % port], capture_output=True, text=True).stdout.split()
def wait_quiet():
    n0 = get("/state?since=0")["n"]; waited = 0; silent = 0
    while silent < quiet_ticks and waited < max_wait:
        time.sleep(0.2); txt, n = own_since(n0)
        if txt: silent = 0; n0 = n
        else: silent += 1
        waited += 1
    return waited
face_off_at = [0.0]; word_tick = {}
def smile_at(level, dur=2.4):
    post("/face", {"expr": level}); face_off_at[0] = time.time() + dur
def face_tend():
    if face_off_at[0] and time.time() >= face_off_at[0]: post("/face", {"expr": 0}); face_off_at[0] = 0.0
def say(text, who, keys=None, read=True):
    win = window if read else 20                                             # an ordinary line: the typist's four seconds
    wait_quiet(); n0 = get("/state?since=0")["n"]; post("/type", {"text": text, "who": who})
    while get("/state?since=0")["queued"] > 0: time.sleep(0.2)
    got = ""; seen = 0; answered = False; t0 = time.time(); answered_at = None; post_then = None
    while time.time() - t0 < win * 0.2:
        time.sleep(0.2); face_tend(); txt, n = own_since(n0)
        if txt != got:
            got = txt
            if smile:
                low = got.lower(); words = re.findall(r"[a-z]+", low); done = words[:-1] if low and low[-1].isalpha() else words
                if keys and not answered and any(k in low for k in keys):
                    answered = True; answered_at = time.time() - t0; smile_at(2.0); face_off_at[0] = time.time() + 2.4
                    post_then = (4.0, time.time() + 2.4)
                elif not face_off_at[0] and len(done) > seen:
                    wd = done[seen]; seen = len(done)
                    if wd in KNOWN and time.time() - word_tick.get(wd, 0) > 24: smile_at(2.0); word_tick[wd] = time.time()
                else: seen = max(seen, len(done))
        if post_then and time.time() >= post_then[1]:
            smile_at(post_then[0]); post_then = None
    if keys and not answered: answered = any(k in got.lower() for k in keys)
    face_tend(); post("/face", {"expr": 0}); face_off_at[0] = 0.0
    return got, answered, answered_at
def run_set(qs, label):
    n = 0; rows = []
    for qi, (q, fact) in enumerate(qs):
        if ordinary:
            (a_, aw), (b_, bw) = ordinary[qi % len(ordinary)]; say(a_, aw, read=False); time.sleep(4.0); say(b_, bw, read=False)
        keys = [w for w in re.findall(r"[a-z]+", fact.lower()) if w not in STOP and w not in q.lower()]
        got, ok, at = say(q, "you", keys=keys); n += int(ok)
        rows.append(f"{'*' if ok else ' '} {q!r:26} -> {got[:28]!r}{' at %.1fs' % at if at else ''}")
        print("  " + rows[-1], flush=True)
    print(f"REHEARSAL {label}: {n}/{len(qs)} answered in the child's turn on the served body", flush=True)
    return n
try:
    if freeze and typist:
        for p in typist: os.kill(int(p), signal.SIGSTOP)
        print(f"typist frozen ({', '.join(typist)}) at tick {tick_now()}", flush=True)
    st = get("/insides"); print(f"the served body: nights {st['nights']} store {st['store']} mood {st['mood']:.2f} sharp {st['sharp_now']} pressure {st['sleep_pressure']}", flush=True)
    if told:
        for q, a in told:
            say(q, "you", read=False); time.sleep(4.0); say(a, "other", read=False)
        print(f"told once: {len(told)} never-typed facts", flush=True)
    if pairs: run_set(pairs, f"taught {facts_range}")
    if told and ask_tell: run_set(told, "never-typed, the same day")
    st = get("/insides"); print(f"after: mood {st['mood']:.2f} sharp {st['sharp_now']} pressure {st['sleep_pressure']}", flush=True)
finally:
    post("/face", {"expr": 0})
    if freeze and typist:
        for p in typist: os.kill(int(p), signal.SIGCONT)
        print("typist thawed", flush=True)
