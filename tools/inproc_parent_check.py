#!/usr/bin/env python3
"""THE CHECK OF tools/inproc_parent.py, on tiny newborns only, every write inside a fresh folder of SCRATCH_DIR.
The guard is installed first (before this check writes anything), allowing only that folder and the system temp dir.
  - the clock: sleeping moves it and runs one tick per boundary; /type, /face, HTTPError as serve's; the night reported asleep
    with the pre-night count, no tick meanwhile, typing heard in the morning; a night of 0 s held until the parent looks;
    --tick-measured ticks last their compute, the nights the served length unless --night-measured
  - the guard: protected paths made in the scratch folder (a folder, a file) are refused by every spelling (case variant,
    /System/Volumes/Data, symlink, hard link, dir_fd) and every route (open, rename, remove, torch.save through the tool's
    wrapper); writes outside the allowed folders refused; a second thread is judged while the first is inside the hook; the
    network (socket, lookup, UDP), signals, forks, spawns (subprocess, multiprocessing's fork_exec) and ctypes' kill refused
  - the REAL protected folders (body/, ops/, data/, .git/) and their other spellings, with I/O that cannot create or change
    anything even if the guard failed: opening a missing file without O_CREAT, removing a missing file, a save into a
    missing folder (a failed guard gives FileNotFoundError, never a file)
  - --out: never under the protected folders by any spelling, never the repo or above it, new or empty
  - two runs in one process with different typist settings: each runs with its own, os.environ, cwd, torch.save and the
    typist modules' clock and wire put back; the copy's saves record the process's env, not the typist's
  - a tiny newborn takes the served save's constants (gate_ear, gate_opt, ...), then the flags; --reflexes-off on the copy
  - --resume: the copy's own save (ticks and nights go on), the clock after the last row, refused on changed settings
  - the day report reads this run's rows only; a night under 60 s refused, 60 s nights all seen; the disk reserve stops a run
usage: nice -n 19 python3 tools/inproc_parent_check.py SCRATCH_DIR"""
import sys

sys.dont_write_bytecode = True
import json                 # noqa: E402
import os                   # noqa: E402
import socket               # noqa: E402
import threading            # noqa: E402
import time                 # noqa: E402
import urllib.error         # noqa: E402
import urllib.request       # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import inproc_parent as IP  # noqa: E402

if len(sys.argv) != 2:
    raise SystemExit(__doc__)
root = os.path.realpath(os.path.abspath(sys.argv[1]))
repo_dirs = {IP.ident(os.path.join(IP.REPO, d)) for d in IP.PROTECTED_DIRS} - {None}
if IP.is_within(root, repo_dirs) or (IP.ident(root) is not None and IP.ident(root) in {IP.ident(a) for a in IP.chain(IP.REPO)}):
    raise SystemExit("SCRATCH_DIR %s is under body/, ops/, data/ or .git/, or is the repo or above it" % root)
os.makedirs(root, exist_ok=True)
out = os.path.join(root, "check_%d_%d" % (int(time.time()), os.getpid())); os.mkdir(out)
guard = IP.install_guard(base_allow_dirs=[out])           # before any other write of this check
os.nice(max(0, 19 - os.nice(0)))

ok = 0
def check(cond, what):
    global ok
    if not cond:
        raise SystemExit("FAIL: " + what)
    ok += 1; print("ok  " + what, flush=True)


def refused(fn, *a, **k):
    """True if the guard refused; False if it let the call through (whatever the call did then)"""
    try:
        fn(*a, **k)
    except PermissionError as e:
        return "inproc_parent guard" in str(e)
    except Exception:
        return False
    return False


import torch                          # noqa: E402
from tokenizers import Tokenizer      # noqa: E402
from body.life import Life, PHYSIOLOGY  # noqa: E402

os.chdir(IP.REPO)
TOK = Tokenizer.from_file(os.path.join(IP.REPO, "data/tok_char.json"))
CFG = dict(wake_ticks=120, wake_every=8, gate_every=8, night_rounds=2, night_starts=12, rem_dreams=4, rem_steps=4)
torch.save = IP.audited_save(torch.save)             # as the tool does during a run


def tiny(name):
    life = Life.birth(TOK, device="cpu", d=64, layers=2, heads=2, window=32, cfg=dict(CFG), seed=0, save_path=os.path.join(out, name))
    life.save(); return life


def call(body, path, data=None):
    r = urllib.request.Request(IP.BASE + path) if data is None else urllib.request.Request(
        IP.BASE + path, data=json.dumps(data).encode(), headers={"Content-Type": "application/json"})
    with body.urlopen(r, timeout=30) as f:
        return json.loads(f.read().decode() or "{}")


# ---------------- the clock and the endpoints ----------------
life = tiny("check.pt"); T0 = 1.8e9
body = IP.InProcBody(life, T0, tick_s=0.167, night_s=100.0)
check(body.now == T0 and call(body, "/state?since=0")["n"] == 0 and body.ticks == 0, "the clock stands still until slept on")
body.advance(10.0)
check(body.ticks == int(10.0 / 0.167) and call(body, "/state?since=0")["n"] == 2 * body.ticks, "10 s asleep in the typist's loop = %d ticks, 2 page entries each" % body.ticks)
r = call(body, "/type", {"text": "hi", "who": "parent"})
check(r == {"queued": 2} and call(body, "/state?since=0")["queued"] == 2, "/type queues its symbols")
n0 = call(body, "/state?since=0")["n"]; body.advance(0.2)
pg = call(body, "/state?since=%d" % n0)["page"]
check(pg[0][0] == "h" and pg[0][1] == 0 and pg[0][4] == "parent" and pg[1][4] is False, "one symbol heard at the next tick, tagged with who typed it")
check(call(body, "/face", {"expr": 2}) == {"you": 2.0}, "/face answers the clamped face")
body.advance(0.2)
check(life.level == 2, "the face is felt at the next tick")
try:
    call(body, "/nope", {}); check(False, "unknown POST")
except urllib.error.HTTPError as e:
    check(e.code == 404, "an unknown POST raises HTTPError 404, as serve replies")
while body.nights == 0:
    body.advance(0.15)
s = call(body, "/state?since=0")
check(life.nights == 1 and s["asleep"] is True and s["nights"] == 0, "the dusk: asleep reported, nights still the pre-night count")
t_ = body.ticks; n_ = s["n"]
check(call(body, "/type", {"text": "dog", "who": "parent"}) == {"queued": 3}, "typing into the night queues (heard in the morning)")
body.advance(50.0); s = call(body, "/state?since=0")
check(s["asleep"] is True and body.ticks == t_ and s["n"] == n_ and s["queued"] == 3, "50 s into a 100 s night: no tick, the page frozen")
body.advance(51.0); s = call(body, "/state?since=0")
check(s["asleep"] is False and s["nights"] == 1 and body.ticks > t_, "after the night: awake, nights counted, ticks again")
body.advance(1.0)
heard = "".join(e[0] for e in call(body, "/state?since=%d" % n_)["page"][0::2])
check(heard.startswith("dog"), "the night's typing heard first thing in the morning (%r)" % heard[:6])
sv = call(body, "/save", {})
check(sv == {"saved": os.path.join(out, "check.pt")}, "/save writes the copy's own path")

life0 = tiny("check0.pt"); b0 = IP.InProcBody(life0, T0, tick_s=0.167, night_s=0.0)
while b0.nights == 0:
    b0.advance(0.15)
t0_ = b0.ticks; b0.advance(5.0)
check(b0.ticks == t0_, "a night of 0 s: no tick until the parent has looked at the page")
s0 = call(b0, "/state?since=0"); s1 = call(b0, "/state?since=0")
check(s0["asleep"] is True and s0["nights"] == 0 and s1["asleep"] is False and s1["nights"] == 1, "... the parent sees it asleep once, then awake")
b0.advance(1.0)
check(b0.ticks > t0_ and b0.night_log[0].get("held_s", 0) >= 5.0, "... and the ticks go on from its look (the night held %.2f s)" % b0.night_log[0].get("held_s", 0))

life2 = tiny("check2.pt"); b2 = IP.InProcBody(life2, T0, measured=True, floor=0.0)
while b2.nights == 0:
    b2.advance(0.5)
check(abs(b2.night_until - b2.night_log[0]["virtual_s"] - T0 - IP.SERVED_NIGHT_S) < 0.06 and b2.ticks >= 1,   # virtual_s is rounded to 0.1
      "--tick-measured: the ticks last their compute, the night the served %g s" % IP.SERVED_NIGHT_S)
life3 = tiny("check3.pt"); b3 = IP.InProcBody(life3, T0, measured=True, floor=0.0, night_measured=True)
while b3.nights == 0:
    b3.advance(0.5)
check(abs((b3.night_until - T0 - b3.night_log[0]["virtual_s"]) - b3.night_log[0]["real_s"]) < 0.06, "--night-measured: the night lasts its compute (%.2f s)" % b3.night_log[0]["real_s"])

life4 = tiny("check4.pt"); ts0 = torch.save; dk = IP.Disk(out, 0.0, 0.0)
torch.save = IP.audited_save(ts0, before=dk.check, after=dk.saved)
dk.reserve = dk.free() + 1.0                                     # from now on every save would eat the reserve
stop = None
try:
    b4 = IP.InProcBody(life4, T0)
    for _ in range(10000):
        b4.advance(1.0)
except IP.StopRun as e:
    stop = str(e)
finally:
    torch.save = ts0
check(stop is not None and "disk" in stop and life4.asleep is False and not os.path.exists(os.path.join(out, "check4.pt.tmp")),
      "a save that would eat the disk reserve stops the run mid-night (through the body's except Exception), nothing written")

# ---------------- the guard, on protected paths made inside the scratch folder ----------------
pdir = os.path.join(out, "protected_dir"); os.makedirs(pdir)
pfile = os.path.join(out, "protected_file.txt"); open(pfile, "w").write("x")
free = os.path.join(out, "free.txt"); open(free, "w").write("y")
hl = os.path.join(out, "hardlink.txt"); os.link(pfile, hl)
hl2 = os.path.join(out, "free_hardlink.txt"); os.link(free, hl2)
outside_t = os.path.join(out, "outside_target.txt"); open(outside_t, "w").write("t")
lnk_in = os.path.join(pdir, "a_link"); os.symlink(outside_t, lnk_in)
lnk_out = os.path.join(out, "link_into_protected"); os.symlink(os.path.join(pdir, "via_link.txt"), lnk_out)
guard.protect(files=[pfile], dirs=[pdir])
probe = os.path.join(pdir, "probe.txt")
check(refused(open, probe, "w") and not os.path.exists(probe), "the guard refuses a write into a protected folder (nothing created)")
check(refused(open, pfile, "a") and open(pfile).read() == "x", "the guard refuses an append to a protected file")
src = os.path.join(out, "rename_src.txt"); open(src, "w").write("z")
check(refused(os.replace, src, pfile) and os.path.exists(src) and open(pfile).read() == "x", "the guard refuses a rename onto a protected file")
case = os.path.join(out, "PROTECTED_DIR")
if os.path.exists(case):
    check(refused(open, os.path.join(case, "c.txt"), "w") and not os.path.exists(os.path.join(pdir, "c.txt")), "... through a case variant of the folder's name")
fl = "/System/Volumes/Data" + os.path.realpath(pdir)
if os.path.exists(fl):
    check(refused(open, os.path.join(fl, "f.txt"), "w") and not os.path.exists(os.path.join(pdir, "f.txt")), "... through the /System/Volumes/Data name")
check(refused(open, hl, "a") and open(pfile).read() == "x", "... through a hard link to the protected file")
check(refused(open, hl2, "a") and open(free).read() == "y", "a file with a second name (hard link) is never written, protected or not")
check(refused(open, lnk_out, "w") and not os.path.exists(os.path.join(pdir, "via_link.txt")), "... through a symlink that points into the protected folder")
check(refused(os.remove, lnk_in) and os.path.lexists(lnk_in), "the removal of a symlink that lives in the protected folder (pointing out)")
fd = os.open(pdir, os.O_RDONLY)
try:
    check(refused(os.remove, "missing.txt", dir_fd=fd) and refused(os.mkdir, "newdir", dir_fd=fd) and not os.path.exists(os.path.join(pdir, "newdir")),
          "a path relative to a descriptor of the protected folder (dir_fd) is judged in that folder")
finally:
    os.close(fd)
check(guard.verdict("a\0b") is not None and guard.verdict("x", dir_fd=987654) is not None, "a path it cannot resolve is refused (it fails closed)")
check(refused(torch.save, {"x": torch.zeros(2)}, os.path.join(pdir, "t.pt")) and not os.path.exists(os.path.join(pdir, "t.pt")),
      "torch.save through the tool's wrapper into a protected folder is refused (nothing written)")
check(refused(open, os.path.join(IP.REPO, "tools", "never_%d.txt" % os.getpid()), "r+") and refused(open, os.path.expanduser("~/never_%d.txt" % os.getpid()), "r+"),
      "a write outside the allowed folders (tools/, home) is refused")
th_gate = threading.Event(); th_in = threading.Event(); v0 = guard.verdict
def slow_verdict(p, dir_fd=None):
    if threading.current_thread().name == "slow":
        th_in.set(); th_gate.wait(5)
    return v0(p, dir_fd)
guard.verdict = slow_verdict
th = threading.Thread(target=lambda: open(os.path.join(out, "slow.txt"), "w").close(), name="slow"); th.start(); th_in.wait(5)
p2 = os.path.join(pdir, "from_second_thread.txt")
r2 = refused(open, p2, "w")
th_gate.set(); th.join(); guard.verdict = v0
check(r2 and not os.path.exists(p2), "a write from one thread is judged while another thread is inside the hook")

# ---------------- the REAL protected folders, with I/O that cannot create or change anything if the guard failed ----------------
miss = "inproc_check_missing_%d" % os.getpid()
real = []
for d in ("data", "body", "ops", ".git", "DATA", "Body"):
    p = os.path.join(IP.REPO, d, miss)
    if os.path.exists(os.path.dirname(p)):
        real.append(p)
if os.path.exists("/System/Volumes/Data" + IP.REPO):
    real.append("/System/Volumes/Data" + os.path.join(IP.REPO, "data", miss))
for p in real:
    check(refused(open, p, "r+") and refused(os.open, p, os.O_WRONLY) and refused(os.remove, p) and refused(os.truncate, p, 0)
          and refused(torch.save, {"x": 1}, os.path.join(p, "never.pt")) and not os.path.exists(p),
          "the real %s: write, remove, truncate and torch.save refused" % os.path.relpath(p, "/System/Volumes/Data" + IP.REPO if p.startswith("/System") else IP.REPO).replace(miss, "*"))
check(guard.verdict(os.path.join(IP.REPO, "data", "watch2.pt")) is not None and guard.verdict(os.path.join(IP.REPO, "data", "teach_queue_w2.jsonl.pos")) is not None,
      "the served files are protected (a verdict, no I/O)")

# ---------------- the network, signals, spawns ----------------
check(refused(socket.socket, socket.AF_INET, socket.SOCK_STREAM) and refused(socket.socket, socket.AF_INET6, socket.SOCK_DGRAM), "no AF_INET or AF_INET6 socket can be made")
check(refused(socket.getaddrinfo, "localhost", 8020) and refused(socket.gethostbyname, "localhost"), "no name lookup")
a_, b_ = socket.socketpair(); a_.sendall(b"x"); check(b_.recv(1) == b"x", "an AF_UNIX socketpair works (the guard refuses only the network)"); a_.close(); b_.close()
check(refused(os.kill, os.getpid(), 0) and refused(os.killpg, os.getpgid(0), 0), "os.kill and os.killpg refused (tried with signal 0)")
import subprocess                     # noqa: E402
import multiprocessing.util as mu     # noqa: E402
check(refused(subprocess.run, ["/usr/bin/true"]) and refused(mu.spawnv_passfds, b"/usr/bin/true", [b"/usr/bin/true"], []) and refused(os.system, "true"),
      "subprocess, os.system and multiprocessing's fork_exec refused")
def _fork():
    pid = os.fork()
    if pid == 0:
        os._exit(0)
    os.waitpid(pid, 0)
check(refused(_fork), "os.fork refused")
import ctypes                         # noqa: E402
libc = ctypes.CDLL(None)
check(refused(getattr, libc, "kill") and callable(libc.getpid), "ctypes' lookup of kill refused (getpid allowed)")

# ---------------- --out ----------------
def rejected(p, resume=False):
    try:
        IP.check_out(p, resume); return False
    except SystemExit:
        return True
bad = [os.path.join(IP.REPO, "data", "x"), os.path.join(IP.REPO, "DATA", "x"), os.path.join(IP.REPO, ".git", "x"), IP.REPO,
       os.path.dirname(IP.REPO), os.path.join(IP.REPO, "body")]
if os.path.exists("/System/Volumes/Data" + IP.REPO):
    bad.append("/System/Volumes/Data" + os.path.join(IP.REPO, "data", "x"))
check(all(rejected(p) for p in bad), "--out refused under data/, DATA/, .git/, /System/Volumes/Data/.../data/, at the repo or above it")
empty = os.path.join(out, "empty"); os.mkdir(empty)
check(rejected(out) and not rejected(empty) and not rejected(os.path.join(out, "new")) and rejected(empty, resume=True),
      "--out must be new or empty; --resume needs a run of this tool")

# ---------------- the tool's runs: two in one process, a copy, resume, report, night 0, disk ----------------
Q = os.path.join(out, "queue.jsonl")
with open(Q, "w") as f:
    for q, b in (("is the milk cold?", "yes. the milk is cold"), ("what does the dog say?", "the dog says woof"), ("where is the ball?", "the ball is under the bed")):
        f.write(json.dumps({"say": [q, "b: " + b]}) + "\n")
common = ["--birth-tiny", "--flags", "ops/BASE_FLAGS.txt", "--typist-from", "ops/typist_chain_command.txt", "--queue", Q, "--queue-pos", "0",
          "--max-real-s", "600", "--t0", "2026-09-22T09:00:00", "--cfg", "wake_ticks=400 night_rounds=2 night_starts=64"]
env0 = dict(os.environ); cwd0 = os.getcwd(); save0 = torch.save
import body.caregiver as BC           # noqa: E402
ra = os.path.join(out, "runA"); rb = os.path.join(out, "runB")
sA = IP.main(common + ["--out", ra, "--report"])
check(dict(os.environ) == env0 and os.getcwd() == cwd0 and torch.save is save0 and BC.time is time and BC.urllib.request is urllib.request,
      "after a run: os.environ, cwd, torch.save and the typist's clock and wire as they were")
sB = IP.main(common + ["--out", rb, "--env", "TALKOVER_FROWN=0 SLOW_SHARE=1.0 FROWN_GAP=999", "--reflexes-off"])
eA, eB = sA["typist"]["env_in_effect"], sB["typist"]["env_in_effect"]
check(eA["TALKOVER_FROWN"] == 1 and eA["SLOW_SHARE"] == 0.33 and eB["TALKOVER_FROWN"] == 0 and eB["SLOW_SHARE"] == 1.0 and eB["FROWN_GAP"] == 999
      and sB["frowns_by"].get("talked over", 0) == 0 and sA["frowns_by"].get("talked over", 0) > 0 and sB["slow_lines"] == sB["parent_lines"] > 0,
      "a second run in the same process runs with its own typist settings (A: frown on, a third slow; B: no frown, every line slow)")
k2 = len(BC.KNOWN2); IP.fresh_typist({}); import body.caregiver as BC2  # noqa: E402
check(BC2 is BC and k2 > len(BC.KNOWN2) == len({w for w in BC.KNOWN if len(w) >= 2}),
      "the typist's KNOWN2 (%d words after a run) is its own again at the next run's import (%d)" % (k2, len(BC.KNOWN2)))
check(sA["days"] and sA["typist"]["days"] == 1 and sA["nights"] >= 1 and sA["night_rows"] >= 1, "--typist-from does not take the chain's --days 6; a day and its night ran")
bA = torch.load(os.path.join(ra, "tiny_body.pt"), map_location="cpu", weights_only=False)
check(bA["env"]["TALKOVER_FROWN"] == env0.get("TALKOVER_FROWN", "") and bA["cfg"]["gate_ear"] == 1 and bA["cfg"]["gate_opt"] == "adam"
      and bA["cfg"]["actor"] == 1 and bA["cfg"]["gate_yield"] == 20.0,
      "the tiny copy has the served save's constants (gate_ear 1, gate_opt adam, actor 1) and the flags' (gate_yield 20); its save records the process's env")
bB = IP.save_meta(os.path.join(rb, "tiny_body.pt"))["cfg"]
check(all(bB[k] == PHYSIOLOGY[k] for k in IP.REFLEXES) and bB["gate_ear"] == 1, "--reflexes-off: the 8 timing reflexes at their defaults on the copy, the ear kept")
rows_rep = [json.loads(l) for l in open(os.path.join(ra, "report", "data", "watch2_caregiver.jsonl"))]
own_from = json.load(open(os.path.join(ra, "run.json")))["own_rows_from"]
with open(os.path.join(ra, "caregiver.jsonl"), "rb") as f:
    f.seek(own_from); own_rows = [json.loads(l) for l in f.read().decode().splitlines() if l.strip()]
check(rows_rep == own_rows and rows_rep[0]["action"] == "session_start" and own_from > 0,
      "the day report reads this run's own rows only (%d), none of the %d-byte copied served tail, with --t0 before it" % (len(rows_rep), own_from))
check(not os.path.exists(os.path.join(rb, "report")), "no report folder without --report")
check(not sA["guard"]["refused"] and all(IP.is_within(p, {IP.ident(ra)}) or p.startswith(os.path.realpath(IP.tempfile.gettempdir())) or p == os.devnull
                                         for p in sA["guard"]["written_outside_out"] + [ra]),
      "run A: nothing refused, nothing written outside its --out but the system temp dir")
check(sA["disk"]["saves"] >= 3, "run A: every save went through the audited wrapper and the disk check (%d saves)" % sA["disk"]["saves"])

# resume: the copy's own save, the clock after the last row, refused on changed settings
m0 = IP.save_meta(os.path.join(ra, "tiny_body.pt"))
try:
    IP.main(common + ["--out", ra, "--resume", "--cfg", "wake_ticks=500"]); check(False, "resume with changed settings")
except SystemExit as e:
    check("settings differ" in str(e) and "wake_ticks" in str(e), "--resume refuses changed settings, naming them")
sR = IP.main(common + ["--out", ra, "--resume"])
rec = json.load(open(os.path.join(ra, "run.json")))
check(sR["body_nights"][0] == m0["nights"] and sR["days"] == [sA["days"][-1] + 1] and rec["runs"][1]["t0"] > rec["runs"][0]["t_end"],
      "--resume: the copy's own save (nights %d go on), the next day, the clock after the last run's end" % m0["nights"])

# a copy (not a newborn) of a tiny save, a 0 s night, and the disk reserve
rc = os.path.join(out, "runC")
try:
    IP.main(common + ["--out", os.path.join(out, "runN"), "--night-s", "0"]); check(False, "a 0 s night")
except SystemExit as e:
    check("at least 60" in str(e) and not os.path.exists(os.path.join(out, "runN")), "--night-s below 60 s (or four listening turns) is refused before anything is written")
sC = IP.main([os.path.join(ra, "tiny_body.pt")] + [x for x in common if x != "--birth-tiny"] + ["--out", rc, "--night-s", "60", "--days", "3"])
check(sC["days"] == [391, 392, 393] and sC["nights"] == sC["night_rows"] == 3 and sC["body"].startswith("a copy of"),
      "a copy of a save, three days with 60 s nights: the parent's day loop saw every night (3 nights, 3 night rows)")
try:
    IP.main(common + ["--out", os.path.join(out, "runD"), "--disk-reserve-gb", "1000000"]); check(False, "disk reserve")
except SystemExit as e:
    check("disk" in str(e) and not os.path.exists(os.path.join(out, "runD", "tiny_body.pt")), "a run whose save would eat the disk reserve stops before it writes the copy")
print("%d checks passed (in %s)" % (ok, out))
