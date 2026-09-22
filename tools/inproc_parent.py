#!/usr/bin/env python3
"""THE PARENT IN-PROCESS (2026-09-22): the served parent teaches a COPY of a body, in this process, on a virtual clock.

body/teacher.py's typist and body/caregiver.py's caregiver are imported and run UNCHANGED. Three things are swapped inside
this process only:
  - the wire: body.caregiver's name `urllib` is replaced by a shim whose urlopen answers each request the way body/serve.py
    would (GET /state, GET /insides, POST /type, POST /face, POST /save), from a Life object held here. Replies go through
    json.dumps/json.loads as on the wire, and a POST that fails raises HTTPError 500 the way serve's handler replies 500.
  - the clock: body.caregiver's and body.teacher's name `time` is replaced by a virtual clock. time() is the virtual now.
    sleep(dt) moves the now forward and runs one life.tick() at each tick boundary it crosses (serve.py's tick loop).
    strftime and localtime pass through, so the rows' timestamps are virtual times.
  - the night: life.tick() runs the whole night inside the dusk tick, so between ticks `asleep` is never True. The clock
    detects the night (life.nights or life.last_night changed across the tick) and then, for the night's length, reports
    asleep=True and the pre-night `nights` count, and runs no tick. That is what serve shows while its tick thread sleeps.
    A night lasts at least 60 s or four listening turns (the parent looks for the night only between its turns, so a
    shorter night can pass unseen by its day loop), and until the parent has read `asleep` at least once. The typist's
    night loop, its post-night cues and its /save run as served.
No rule of the parent is changed and none is added. The body is not changed either: its constants are the ones a served
reload gives, the save's own constants and then the flags (a tiny newborn takes them from the served save's pickle).

The typist's environment (TALKOVER_FROWN, SLOW_SHARE, ...) is read by body.caregiver and body.teacher when they are imported,
so each run imports them afresh (importlib.reload) with the run's environment, as a new typist process would, and puts
os.environ back at once (the copy's saves record the served process's environment, not the typist's).

Isolation. The queue (and its .pos cursor), the corpus JSON and the tail of the caregiver log are copied into --out, and the
typist and caregiver read and write only those copies. --out must be a new or empty directory (or, with --resume, a run of
this tool). The copy's save path is under --out. A Python audit hook, installed before the run's first write, refuses:
  - any write, creation, rename, removal or metadata change outside --out and the system temp dir (torch's own caches),
    and anything under body/, ops/, data/ or .git/ or to an input file whatever the spelling: paths are compared by device
    and inode, so a case variant, the /System/Volumes/Data name, a symlink or a hard link reach the same verdict; an existing
    file with a second name (a hard link) is never written; a path it cannot resolve is refused (it fails closed);
  - any socket that is not AF_UNIX (creation, connect, bind, sendto, sendmsg) and any name lookup;
  - any kill, killpg, pthread_kill, fork, exec, spawn, posix_spawn, subprocess, os.system, _posixsubprocess.fork_exec
    (multiprocessing's path) and sys.remote_exec, and ctypes symbol lookups of the process, signal, socket and file calls.
torch.save writes an ascii path from C++ with no audit event, so during the run torch.save is wrapped to open the file with
Python's open (the hook sees it; the bytes are torch's own, only the zip's inner folder name differs), and each save first
checks the disk: a copy never takes the space the served body's nightly save needs (--disk-reserve-gb). Bytecode writing
is off, so importing body/ writes nothing to body/__pycache__. The process lowers its own priority to nice 19.

usage (relative paths are taken from the repo root, where every command here runs):
  nice -n 19 python3 tools/inproc_parent.py data/watch2.pt --flags ops/BASE_FLAGS.txt \\
      --typist-from ops/typist_chain_command.txt --queue data/teach_queue_w2.jsonl --queue-pos N --days 1 --out DIR \\
      [--env 'K=V ...'] [--reflexes-off] [--cfg 'k=v ...'] [--report]
  A tiny newborn with the served constants:  ... --birth-tiny [--cfg-from SAVE|none] ...
  The same inputs for several runs:  ... --snapshot-only --out SNAP   then   ... --inputs-from SNAP --out RUN_k
  Another day of the same run:  the same command with --resume (the copy's own save, cursor, corpus and log; the clock goes on)
  The typist's own arguments (--day --days --log --corpus --planner --queue --period --quiet --cap --seed --tick --listen
  --answer-levels --parent --reply --wait --yield) have body/teacher.py's names and defaults. --typist-from reads the
  TARGS, ENV and LOG of a typist_chain.sh command line first (not its --days: the run's length is this tool's --days);
  flags given here then override them.
  The body's own clock: --body-tick S (default 0.167, the served average per tick), or --tick-measured (each tick lasts
  max(--tick-floor, its real compute time); floor default = serve's --period from the flags). A night lasts --night-s
  virtual seconds (default 2450 s, the served nights, in both modes; at least max(60 s, 4 x listen)), or with --night-measured
  the dusk tick's compute time (at least that floor).
"""
import sys

sys.dont_write_bytecode = True           # importing body/ must write nothing under body/__pycache__

import argparse                          # noqa: E402
import collections                       # noqa: E402
import hashlib                           # noqa: E402
import importlib                         # noqa: E402
import io                                # noqa: E402
import json                              # noqa: E402
import os                                # noqa: E402
import pickle                            # noqa: E402
import random                            # noqa: E402
import re                                # noqa: E402
import runpy                             # noqa: E402
import shlex                             # noqa: E402
import shutil                            # noqa: E402
import socket                            # noqa: E402
import stat as _stat                     # noqa: E402
import tempfile                          # noqa: E402
import threading                         # noqa: E402
import time as _time                     # noqa: E402
import types                             # noqa: E402
import urllib.error                      # noqa: E402
import urllib.request                    # noqa: E402
import zipfile                           # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

BASE = "inproc://body"                   # the typist's base URL: never an http URL, so nothing could reach a port by accident
SERVED_TICK = 0.167                      # the served body's measured seconds per tick (0.15 s floor, overrun by its compute)
SERVED_NIGHT_S = 2450.0                  # the served nights' length in seconds (the caregiver log's "slept_s")
NIGHT_MIN_S = 60.0                       # no night shorter: the parent checks for the night only between its turns (its longest
                                         # blind stretch is the listening after a line, s(listen) = 6 s served), so a shorter
                                         # night can pass unseen by its day loop (a 0 s night: 8 nights, 1 night row)
SERVE_COMMAND = "ops/serve_command.txt"  # the served body's command line: its --load is the save whose constants a tiny newborn takes
SERVED_SAVE = "data/watch2.pt"           # ... when that file names none
# the hand-set timing settings the served flags switch on (PHYSIOLOGY defaults are off); --reflexes-off sets each to its default
REFLEXES = ("gate_yield", "gate_turn", "gate_listen", "gate_ear_release", "offset_foresee", "gate_quiet_tau", "gate_ear_decay", "gate_ear_gain")
TYPIST_ENV = ("TALKOVER_FROWN", "FROWN_GAP", "TURN_ONLY_SMILE", "HABIT_TICKS", "ANSWER_SMILE", "SLOW_SHARE", "SLOW_CPS", "SLOW_PAUSE")
PROTECTED_DIRS = ("body", "ops", "data", ".git")
PROTECTED_FILES = ("data/teach_queue_w2.jsonl", "data/teach_queue_w2.jsonl.pos", "data/watch2_corpus.json",
                   "data/watch2_caregiver.jsonl", "data/watch2.pt")
WFLAGS = os.O_WRONLY | os.O_RDWR | os.O_APPEND | os.O_CREAT | os.O_TRUNC
GiB = float(1 << 30)


class StopRun(BaseException):
    """a limit reached. A BaseException, so it passes through the typist's and the body's `except Exception` blocks"""


def hms(s):
    s = int(round(s)); return "%02d:%02d:%02d" % (s // 3600, s % 3600 // 60, s % 60)


def iso(t):
    return _time.strftime("%Y-%m-%dT%H:%M:%S", _time.localtime(t))


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def rpath(p):
    """relative paths are the repo root's, as every command in this project runs from there"""
    if p is None or p == "":
        return None
    p = os.path.expanduser(p)
    return os.path.abspath(p if os.path.isabs(p) else os.path.join(REPO, p))


# ---------------- file identity: every spelling of a path is the same (device, inode) ----------------
def ident(p, follow=True):
    try:
        st = os.stat(p) if follow else os.lstat(p)
        return (st.st_dev, st.st_ino)
    except (OSError, ValueError):
        return None


def chain(p):
    """p and its ancestors, nearest first (p should be resolved already)"""
    out = [p]
    while True:
        q = os.path.dirname(p)
        if q == p:
            return out
        out.append(q); p = q


def is_within(path, dirs_ids):
    """path (resolved) is one of the directories, or lies under one, compared by device and inode"""
    return any(ident(a) in dirs_ids for a in chain(os.path.realpath(path)) if os.path.lexists(a))


def fd_path(fd):
    """the directory a dir_fd names (macOS F_GETPATH; Linux /proc), or None"""
    try:
        import fcntl
        if hasattr(fcntl, "F_GETPATH"):
            return os.fsdecode(fcntl.fcntl(fd, fcntl.F_GETPATH, bytes(1024)).split(b"\0", 1)[0])
    except Exception:
        return None
    try:
        return os.readlink("/proc/self/fd/%d" % fd)
    except Exception:
        return None


# ---------------- the guard: this process writes only where it may, and reaches no one ----------------
# the events that write, create, rename, remove or change a path: event -> [(index of a path argument, index of its dir_fd or None)]
# (an open for writing is judged too; a write through an open descriptor was judged when it was opened)
_WRITE_EVENTS = {
    "os.rename": [(0, 2), (1, 3)], "os.remove": [(0, 1)], "os.rmdir": [(0, 1)], "os.mkdir": [(0, 2)], "os.truncate": [(0, None)],
    "os.chmod": [(0, 2)], "os.chown": [(0, 3)], "os.utime": [(0, 3)], "os.chflags": [(0, None)], "os.lchflags": [(0, None)],
    "os.link": [(0, 2), (1, 3)], "os.symlink": [(1, 2)], "os.mkfifo": [(0, 2)], "os.mknod": [(0, 3)],
    "os.setxattr": [(0, None)], "os.removexattr": [(0, None)], "shutil.rmtree": [(0, 1)], "shutil.move": [(0, None), (1, None)],
    "shutil.copyfile": [(1, None)], "shutil.copytree": [(1, None)], "shutil.chown": [(0, None)], "shutil.make_archive": [(0, None)],
}
_SPAWN_EVENTS = ("os.kill", "os.killpg", "signal.pthread_kill", "os.system", "subprocess.Popen", "os.posix_spawn", "os.exec",
                 "os.spawn", "os.fork", "os.forkpty", "_posixsubprocess.fork_exec", "os.startfile", "sys.remote_exec", "os.pidfd_open")
_LOOKUP_EVENTS = ("socket.getaddrinfo", "socket.gethostbyname", "socket.gethostbyname_ex", "socket.gethostbyaddr",
                  "socket.getnameinfo", "socket.sethostname")
_SOCKET_EVENTS = ("socket.connect", "socket.bind", "socket.sendto", "socket.sendmsg")
_DENY_SYMBOLS = {"kill", "killpg", "pthread_kill", "sigqueue", "fork", "vfork", "forkpty", "execve", "execv", "execvp", "execvpe",
                 "execl", "execlp", "execle", "fexecve", "posix_spawn", "posix_spawnp", "system", "popen", "socket", "connect",
                 "sendto", "sendmsg", "bind", "open", "openat", "creat", "unlink", "unlinkat", "rename", "renameat", "renamex_np",
                 "renameatx_np", "link", "linkat", "symlink", "symlinkat", "truncate", "rmdir", "mkdir", "mkdirat", "chmod",
                 "fchmodat", "chown", "lchown", "fchownat", "utimes", "utimensat", "setxattr", "removexattr", "clonefile",
                 "clonefileat", "fclonefileat", "copyfile", "fcopyfile", "exchangedata", "ptrace", "task_for_pid"}
_AT_FDCWD = (None, -1, -2, -100)          # "relative to the working directory" as the events report it (macOS -2, Linux -100)


class Guard:
    """an allow-list: writes only under the allowed directories (plus the allowed files such as /dev/null), never under the
    protected directories or to the protected files; compared by (device, inode), so no spelling of a path escapes; fails
    closed. Protections only grow within a process; the allowed directories are the base's plus the current run's."""

    def __init__(self, base_allow_dirs=(), allow_files=()):
        self.base_allow = {i for i in (ident(d) for d in base_allow_dirs) if i}
        self.run_allow = set()
        self.allow_files = {i for i in (ident(f) for f in allow_files) if i}
        self.deny_dirs = set(); self.deny_files = set(); self.deny_names = {}
        self.written = set(); self.refused = []
        self._tl = threading.local()

    # -- configuration --
    def protect(self, files=(), dirs=()):
        for f in files:
            i = ident(f) if f else None
            if i:
                self.deny_files.add(i); self.deny_names[i] = f
        for d in dirs:
            i = ident(d)
            if i:
                self.deny_dirs.add(i); self.deny_names[i] = d

    def begin_run(self, allow_dirs):
        self.run_allow = {i for i in (ident(d) for d in allow_dirs) if i}
        self.written = set(); self.refused = []

    def end_run(self):
        self.run_allow = set()

    # -- the verdict --
    def verdict(self, path, dir_fd=None):
        """None if this process may write, create, rename or remove `path`; else the reason"""
        if path is None or isinstance(path, int):
            return None                                               # an open descriptor: its opening was judged
        try:
            s = os.fsdecode(path)
            if not os.path.isabs(s):
                base = os.getcwd() if dir_fd in _AT_FDCWD else fd_path(dir_fd)
                if base is None:
                    return "a path relative to a directory descriptor it cannot name"
                s = os.path.join(base, s)
            target = os.path.realpath(s)
            head, tail = os.path.split(s.rstrip(os.sep) or os.sep)
            entry = os.path.join(os.path.realpath(head), tail) if tail not in ("", ".", "..") else target
            allow = self.base_allow | self.run_allow
            for loc, follow in ((entry, False), (target, True)):
                own = ident(loc, follow=follow)
                if own is not None and own in self.deny_files:
                    return "a protected file (%s)" % self.deny_names.get(own, "")
                try:
                    st = os.lstat(loc)
                    if _stat.S_ISREG(st.st_mode) and st.st_nlink > 1:
                        return "a file with a second name (a hard link)"
                except OSError:
                    pass
                ok = own is not None and own in self.allow_files
                for k, a in enumerate(chain(loc)):
                    i = own if k == 0 else ident(a)
                    if i is None:
                        continue
                    if i in self.deny_dirs:
                        return "under %s/ (protected)" % os.path.relpath(self.deny_names.get(i, a), REPO)
                    if i in allow:
                        ok = True
                if not ok:
                    return "outside --out and the system temp dir"
            return None
        except Exception as e:                                        # cannot resolve: refused
            return "unresolvable (%r)" % (e,)

    def _refuse(self, what, detail):
        self.refused.append((what, str(detail)[:160]))
        raise PermissionError("inproc_parent guard: refused %s: %s" % (what, detail))

    def _judge(self, what, path, dir_fd=None):
        why = self.verdict(path, dir_fd)
        if why is not None:
            self._refuse(what, "%s is %s" % (path, why))
        if path is not None and not isinstance(path, int):
            try:
                self.written.add(os.path.realpath(os.fsdecode(path)))
            except Exception:
                pass

    def hook(self, event, args):
        tl = self._tl
        if getattr(tl, "busy", False):                                # the hook's own calls (per thread)
            return
        tl.busy = True
        try:
            if event == "open":
                a = list(args) + [None, None, None]
                path, mode, flags = a[0], a[1], a[2]
                if (isinstance(mode, str) and any(c in mode for c in "wax+")) or (isinstance(flags, int) and flags & WFLAGS):
                    self._judge("write", path)
            elif event in _WRITE_EVENTS:
                for pi, fi in _WRITE_EVENTS[event]:
                    if pi < len(args):
                        self._judge(event, args[pi], args[fi] if fi is not None and fi < len(args) else None)
            elif event == "socket.__new__":
                fam = args[1] if len(args) > 1 else None
                if fam != getattr(socket, "AF_UNIX", None):
                    self._refuse("a socket", "family %r (no network in this process)" % (fam,))
            elif event in _SOCKET_EVENTS:
                if getattr(args[0], "family", None) != getattr(socket, "AF_UNIX", None):
                    self._refuse(event, "%r (no network in this process)" % (args[1:2],))
            elif event in _LOOKUP_EVENTS:
                self._refuse(event, "%r (no network in this process)" % (args[:1],))
            elif event in _SPAWN_EVENTS:
                self._refuse(event, "%s (this process signals and spawns nothing)" % repr(args)[:80])
            elif event in ("ctypes.dlsym", "ctypes.dlsym/handle"):
                name = args[1] if len(args) > 1 else None
                if isinstance(name, str) and name.lstrip("_") in _DENY_SYMBOLS:
                    self._refuse(event, "the C function %r (process, signal, socket and file calls go through Python here)" % name)
        finally:
            tl.busy = False


_GUARD = None


def install_guard(base_allow_dirs=()):
    """one guard per process (an audit hook cannot be removed): installed once; later calls reuse it"""
    global _GUARD
    if _GUARD is None:
        _GUARD = Guard(list(base_allow_dirs) + [tempfile.gettempdir()], [os.devnull])
        _GUARD.protect(dirs=[os.path.join(REPO, d) for d in PROTECTED_DIRS], files=[os.path.join(REPO, f) for f in PROTECTED_FILES])
        sys.addaudithook(_GUARD.hook)
    return _GUARD


def audited_save(real_save, before=None, after=None):
    """torch.save through Python's open, so the audit hook judges the path (torch's C++ writer raises no event)"""
    def save(obj, f, *a, **k):
        if isinstance(f, (str, bytes, os.PathLike)):
            p = os.fsdecode(f)
            if before:
                before(p)
            with open(p, "wb") as fh:
                r = real_save(obj, fh, *a, **k)
            if after:
                after(p)
            return r
        return real_save(obj, f, *a, **k)
    save.__wrapped__ = real_save
    return save


class Disk:
    """a copy's save never takes the space the served body's own nightly save needs"""

    def __init__(self, out, reserve_bytes, est_bytes):
        self.out, self.reserve, self.est = out, float(reserve_bytes), float(est_bytes); self.saves = 0; self.low = None

    def free(self, p=None):
        return shutil.disk_usage(os.path.dirname(p) if p else self.out).free

    def check(self, p=None, what="save"):
        f = self.free(p)
        if f - self.est < self.reserve:
            self.low = "%s: %.1f GiB free, %.2f GiB needed, %.1f GiB kept for the served body" % (what, f / GiB, self.est / GiB, self.reserve / GiB)
            raise StopRun("disk (%s)" % self.low)

    def saved(self, p):
        self.saves += 1
        try:
            self.est = max(self.est, float(os.path.getsize(p)))
        except OSError:
            pass


# ---------------- a save's constants, read from its pickle without its tensors ----------------
class _Stub:
    def __new__(cls, *a, **k):
        return object.__new__(cls)

    def __init__(self, *a, **k):
        pass

    def __setstate__(self, s):
        pass


class _NoTensors(pickle.Unpickler):
    def find_class(self, mod, name):
        if mod.split(".")[0] in ("torch", "numpy"):
            return _Stub
        return super().find_class(mod, name)

    def persistent_load(self, pid):
        return None


def save_meta(path):
    """the pickled part of a torch save (cfg, arch, life counters) with every tensor left unread: a few hundred KB of a 1.3 GB file"""
    with zipfile.ZipFile(path) as z:
        name = next(n for n in z.namelist() if n == "data.pkl" or n.endswith("/data.pkl"))
        blob = _NoTensors(io.BytesIO(z.read(name))).load()
    life = blob.get("life") or {}
    return {"cfg": dict(blob.get("cfg") or {}), "arch": blob.get("arch"),
            "ticks": life.get("ticks"), "nights": life.get("nights")}


def served_save():
    try:
        toks = open(os.path.join(REPO, SERVE_COMMAND)).read().split()
        return rpath(toks[toks.index("--load") + 1])
    except Exception:
        return rpath(SERVED_SAVE)


# ---------------- body/serve.py, in this process, on a virtual clock ----------------
class _Resp:
    def __init__(self, raw):
        self._raw = raw

    def read(self):
        return self._raw

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


class InProcBody:
    """serve.py's tick loop and endpoints around one Life, the tick loop driven by the typist's own sleeps"""

    def __init__(self, life, t0, tick_s=SERVED_TICK, measured=False, floor=0.15, night_s=None, night_measured=False, night_min=0.0,
                 max_virtual_s=None, max_real_s=None, max_ticks=None, progress=None):
        self.life = life; self.t0 = self.now = float(t0)
        self.tick_s, self.measured, self.floor = float(tick_s), bool(measured), float(floor)
        self.night_s, self.night_measured, self.night_min = night_s, bool(night_measured), float(night_min)
        self.base_t, self.k = self.t0, 1                                # fixed ticks at base_t + k * tick_s (no drift)
        self.next_tick = self.t0 + (self.floor if self.measured else self.tick_s)
        self.night_until = None; self.night_pre = None; self.night_seen = True; self._night_rec = None
        self.ticks = 0; self.tick_errors = 0; self.last_error = None; self.nights = 0; self.night_log = []
        self.own = 0; self.world = 0; self.typed = 0; self.faces = 0
        self.real_ticks_s = 0.0; self.real_nights_s = 0.0; self.real0 = _time.perf_counter()
        self.calls = collections.Counter()
        self.max_virtual_s, self.max_real_s, self.max_ticks = max_virtual_s, max_real_s, max_ticks
        self.progress = progress

    # -- the clock --
    def latched(self):
        return self.night_until is not None and (self.now < self.night_until or not self.night_seen)

    def advance(self, secs):
        self._limits()
        target = self.now + max(0.0, float(secs))
        while self.next_tick <= target:
            if not self.night_seen:                                     # the night goes on until the parent has seen it asleep
                break
            self.now = max(self.now, self.next_tick)
            self._tick()
        self.now = max(self.now, target)

    def _limits(self):
        if self.max_ticks is not None and self.ticks >= self.max_ticks:
            raise StopRun("max ticks %d" % self.max_ticks)
        if self.max_virtual_s is not None and self.now - self.t0 >= self.max_virtual_s:
            raise StopRun("max virtual %.0f s" % self.max_virtual_s)
        if self.max_real_s is not None and _time.perf_counter() - self.real0 >= self.max_real_s:
            raise StopRun("max real %.0f s" % self.max_real_s)

    def _tick(self):
        """one pass of serve.py's loop: `if not life.asleep: life.tick()`, an exception kept in life.last"""
        L = self.life; B = self.next_tick
        self._limits()
        n0, ln0 = L.nights, L.last_night
        p0 = L.page_base + len(L.page)
        r0 = _time.perf_counter()
        try:
            if not L.asleep:
                L.tick()
        except Exception as e:
            L.last = {"error": str(e)[:200], "tick": L.ticks}
            self.tick_errors += 1; self.last_error = str(e)[:200]
        c = _time.perf_counter() - r0
        self.ticks += 1
        for e in L.page[max(0, p0 - L.page_base):]:
            if e[0]:
                if e[1]:
                    self.own += 1
                else:
                    self.world += 1
        if L.nights != n0 or L.last_night is not ln0:                   # the night ran inside this tick
            dur = float(self.night_s) if self.night_s is not None else (max(self.night_min, c) if self.night_measured else SERVED_NIGHT_S)
            self.night_until = B + dur; self.night_pre = n0; self.night_seen = False; self.nights += 1; self.real_nights_s += c
            self.base_t, self.k = self.night_until, 0; self.next_tick = self.night_until   # serve's next tick starts as the night ends
            ln = L.last_night if isinstance(L.last_night, dict) else {}
            rec = {"night": self.nights, "body_nights": L.nights, "body_tick": L.ticks, "virtual_s": round(B - self.t0, 1),
                   "night_s": round(dur, 1), "real_s": round(c, 2), "error": ln.get("error"), "discarded": ln.get("discarded")}
            self.night_log.append(rec); self._night_rec = rec
            if self.progress:
                self.progress("night %d fell at body tick %d (virtual %s, lasting %.0f s); computed in %.1f s real%s"
                              % (self.nights, L.ticks, hms(B - self.t0), dur, c, (" ERROR " + str(ln.get("error"))) if ln.get("error") else ""))
        else:
            self.real_ticks_s += c
            if self.measured:
                self.next_tick = B + max(self.floor, c)
            else:
                self.k += 1; self.next_tick = self.base_t + self.k * self.tick_s

    def _seen(self):
        """the parent has read `asleep`: a night already past its length ends now (it lasted as long as the parent's look)"""
        self.night_seen = True
        if self.now >= self.night_until:
            if self._night_rec is not None:
                self._night_rec["held_s"] = round(self.now - self.night_until, 3)
            self.night_until = self.now; self.base_t, self.k = self.now, 0; self.next_tick = max(self.next_tick, self.now)

    # -- the endpoints (serve.py do_GET / do_POST) --
    def get(self, path):
        if path.startswith("/state"):
            since = 0
            if "since=" in path:
                try:
                    since = int(path.split("since=")[1].split("&")[0])
                except Exception:
                    since = 0
            d = self.life.state(since)
            if self.latched():                                          # the night: asleep, and nights counted at its end
                d = dict(d); d["asleep"] = True; d["nights"] = self.night_pre
                if not self.night_seen:
                    self._seen()
            return 200, d
        if path.startswith("/insides"):
            return 200, self.life.insides()
        return 404, {"error": "the in-process body serves no html pages"}

    def post(self, path, body):
        try:
            if path == "/type":
                r = self.life.type_text(str(body.get("text", "")), who=str(body.get("who", "you")))   # serve's default tag
                self.typed += int(r.get("queued", 0)); return 200, r
            if path == "/face":
                self.faces += 1; return 200, self.life.set_face(body.get("expr", 0))
            if path == "/save":
                return 200, self.life.save()                            # life.save_path: the copy's own path under --out
            return 404, {"error": "unknown path"}
        except Exception as e:
            return 500, {"error": str(e)[:300]}

    def urlopen(self, req, data=None, timeout=None, **kw):
        """the drop-in for urllib.request.urlopen inside body.caregiver"""
        url, payload = (req, data) if isinstance(req, str) else (req.full_url, req.data)
        if not url.startswith(BASE):
            raise urllib.error.URLError("inproc_parent: %s is not the in-process body; this process has no network" % url)
        path = url[len(BASE):] or "/"
        if payload is None:
            code, obj = self.get(path); method = "GET"
        else:
            code, obj = self.post(path, json.loads(payload.decode() or "{}") if payload else {}); method = "POST"
        self.calls[method + " " + path.split("?")[0]] += 1
        raw = json.dumps(obj).encode()
        if code != 200:
            raise urllib.error.HTTPError(url, code, str(obj.get("error", ""))[:300], {}, io.BytesIO(raw))
        return _Resp(raw)


class VirtualTime:
    """stands in for the module `time` inside body.caregiver and body.teacher"""

    def __init__(self, body):
        self._body = body

    def time(self):
        return self._body.now

    def sleep(self, secs):
        self._body.advance(secs)

    def monotonic(self):
        return self._body.now

    perf_counter = monotonic

    def __getattr__(self, name):                                        # strftime, localtime, gmtime, struct_time: the real ones
        return getattr(_time, name)


def fresh_typist(env):
    """body.caregiver and body.teacher imported afresh with the run's environment (each reads it once, at import, as a new typist
    process does); os.environ is put back at once, so nothing else in this process (the copy's saves) sees the typist's env"""
    saved = dict(os.environ)
    try:
        os.environ.update(env)
        mods = []
        for name in ("body.caregiver", "body.teacher"):                 # the caregiver first: the teacher imports its class
            mods.append(importlib.reload(sys.modules[name]) if name in sys.modules else importlib.import_module(name))
    finally:
        for k in [k for k in os.environ if k not in saved]:
            del os.environ[k]
        for k, v in saved.items():
            if os.environ.get(k) != v:
                os.environ[k] = v
    return mods[0], mods[1]


# ---------------- arguments ----------------
def serve_parser():
    """body/serve.py's own parser, to read the flags file exactly as serve reads its command line"""
    from body.life import PHYSIOLOGY
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--birth", default=None); ap.add_argument("--load", default=None)
    ap.add_argument("--tok", default="data/tok_char.json"); ap.add_argument("--dev", default="cpu")
    ap.add_argument("--port", type=int, default=8018); ap.add_argument("--period", type=float, default=0.5)
    ap.add_argument("--d", type=int, default=256); ap.add_argument("--layers", type=int, default=6)
    ap.add_argument("--heads", type=int, default=4); ap.add_argument("--window", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    for k, v in PHYSIOLOGY.items():
        ap.add_argument("--" + k.replace("_", "-"), type=type(v), default=None)
    return ap


def strip_opts(toks, names):
    out = []; skip = False
    for t in toks:
        if skip:
            skip = False; continue
        if t in names:
            skip = True; continue
        if any(t.startswith(n + "=") for n in names):
            continue
        out.append(t)
    return out


def read_chain_command(path):
    """TARGS, ENV and LOG of a `tools/typist_chain.sh PORT LOG "TARGS" "ENV" ROUNDS` command line (TARGS without --days)"""
    toks = shlex.split(open(path).read())
    i = next((j for j, t in enumerate(toks) if t.endswith("typist_chain.sh")), None)
    if i is None or len(toks) < i + 4:
        raise SystemExit("--typist-from %s: no 'typist_chain.sh PORT LOG TARGS [ENV]' in it" % path)
    log, targs = toks[i + 2], toks[i + 3]
    tenv = toks[i + 4] if len(toks) > i + 4 and "=" in toks[i + 4] else ""
    return strip_opts(shlex.split(targs), ("--days", "--day", "--log", "--port")), shlex.split(tenv), log


def build_parser():
    ap = argparse.ArgumentParser(description="the served parent teaching a copy of a body in-process, on a virtual clock",
                                 formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    ap.add_argument("body", nargs="?", help="the save to copy (read only); with --birth-tiny, the save whose constants the newborn takes")
    g = ap.add_argument_group("the body (loaded as body/serve.py loads it)")
    g.add_argument("--flags", default="ops/BASE_FLAGS.txt", help="serve's command-line flags (read with serve's own parser)")
    g.add_argument("--tok", default=None, help="tokenizer (default: the flags' --tok, else data/tok_char.json)")
    g.add_argument("--device", default=None, help="default: the flags' --dev, else cpu")
    g.add_argument("--body-seed", type=int, default=None, help="serve's --seed (default: the flags' --seed, else 0)")
    g.add_argument("--cfg", action="append", default=[], help="'k=v k2=v2': physiology set on the copy after the flags")
    g.add_argument("--cfg-default", action="append", default=[], help="'k1,k2': physiology set to its PHYSIOLOGY default")
    g.add_argument("--reflexes-off", action="store_true", help="--cfg-default " + ",".join(REFLEXES))
    g.add_argument("--birth-tiny", action="store_true", help="a newborn test body d=64 layers=2 heads=2 window=32")
    g.add_argument("--cfg-from", default=None, help="--birth-tiny: the save whose constants the newborn takes before the flags "
                   "(default: the positional save, else the served save of %s); 'none' = the physiology's defaults" % SERVE_COMMAND)
    g.add_argument("--threads", type=int, default=None, help="torch threads (default: torch's own)")
    g.add_argument("--allow-low-memory", action="store_true", help="load a copy even when the free memory looks short")
    g = ap.add_argument_group("isolation")
    g.add_argument("--out", required=True, help="the scratch directory, new or empty (or a run of this tool, with --resume)")
    g.add_argument("--resume", action="store_true", help="another invocation of the run in --out: its own save, cursor, corpus, log and clock")
    g.add_argument("--resume-changes", action="store_true", help="--resume even though the settings differ from the run's first")
    g.add_argument("--save-as", default=None, help="the copy's save path (under --out; default OUT/body_copy.pt or OUT/tiny_body.pt)")
    g.add_argument("--queue-pos", type=int, default=None, help="byte offset in the queue to start from (default: the copied .pos)")
    g.add_argument("--log-tail", type=int, default=300, help="lines of the source caregiver log copied into OUT (0 none, -1 all)")
    g.add_argument("--typist-from", default=None, help="a typist_chain.sh command line (e.g. ops/typist_chain_command.txt)")
    g.add_argument("--env", action="append", default=[], help="'K=V K2=V2' for the typist (read at its import)")
    g.add_argument("--snapshot-only", action="store_true", help="copy the queue, cursor, corpus and log tail into --out and stop")
    g.add_argument("--inputs-from", default=None, help="a --snapshot-only directory: its queue, cursor, corpus and log are the inputs")
    g.add_argument("--disk-reserve-gb", type=float, default=None,
                   help="free space every save leaves (default: twice the served save plus 1 GiB when --out shares its disk)")
    g = ap.add_argument_group("the virtual clock")
    g.add_argument("--body-tick", type=float, default=SERVED_TICK, help="virtual seconds per body tick (served average 0.167)")
    g.add_argument("--tick-measured", action="store_true", help="each tick lasts max(--tick-floor, its real compute time)")
    g.add_argument("--tick-floor", type=float, default=None, help="default: serve's --period from the flags (0.15 served)")
    g.add_argument("--night-s", type=float, default=None, help="a night's virtual seconds (default 2450, the served nights; "
                   "at least max(60, 4 x the typist's listening turn))")
    g.add_argument("--night-measured", action="store_true", help="a night lasts the dusk tick's real compute time (at least that floor)")
    g.add_argument("--t0", default=None, help="the virtual clock's start (epoch seconds or ISO local time; default now; "
                   "with --resume, the run's last virtual time)")
    g.add_argument("--max-virtual-s", type=float, default=None); g.add_argument("--max-real-s", type=float, default=None)
    g.add_argument("--max-ticks", type=int, default=None)
    g.add_argument("--report", action="store_true", help="run ops/day_report.py (a copy in OUT/report) on this run's own rows")
    g = ap.add_argument_group("the typist's own arguments (body/teacher.py main(): same names, same defaults)")
    g.add_argument("--port", type=int, default=None, help="accepted and never used: nothing here opens a socket")
    g.add_argument("--day", type=int, default=None, help="default: the log's last day + 1 (typist_chain.sh's rule), else 1")
    g.add_argument("--days", type=int, default=1)
    g.add_argument("--log", default=None, help="the SOURCE caregiver log (its tail is copied); the typist writes OUT/caregiver.jsonl")
    g.add_argument("--corpus", default="data/body2_corpus.json", help="the SOURCE corpus (copied to OUT/corpus.json)")
    g.add_argument("--planner", choices=["queue", "fixed"], default="queue", help="(the claude planner needs the network: refused)")
    g.add_argument("--queue", default="data/teach_queue.jsonl", help="the SOURCE queue (copied to OUT/queue.jsonl with its .pos)")
    g.add_argument("--model", default="claude-sonnet-5"); g.add_argument("--budget", type=int, default=120)
    g.add_argument("--period", type=float, default=160); g.add_argument("--quiet", type=float, default=12)
    g.add_argument("--cap", type=float, default=48); g.add_argument("--seed", type=int, default=0)
    g.add_argument("--tick", type=float, default=0.25, help="the typist's seconds per tick (its s() conversion; 0.15 served)")
    g.add_argument("--listen", type=float, default=50)
    g.add_argument("--answer-levels", type=int, default=1); g.add_argument("--parent", type=int, default=0)
    g.add_argument("--reply", type=int, default=0); g.add_argument("--wait", type=int, default=0)
    g.add_argument("--yield", dest="yield_ticks", type=int, default=240)
    return ap


def last_day(path):
    """typist_chain.sh's rule: the last '"day": N' in the log's last 300 lines"""
    if not path or not os.path.exists(path):
        return None
    tail = tail_lines(path, 300)
    ds = re.findall(r'"day": ([0-9]+)', "".join(tail))
    return int(ds[-1]) if ds else None


def last_ts(path, start=0):
    """the virtual time of the last row at or after byte `start` (epoch seconds), or None"""
    if not path or not os.path.exists(path) or os.path.getsize(path) <= start:
        return None
    for line in reversed(tail_lines(path, 50)):
        try:
            return _time.mktime(_time.strptime(json.loads(line)["ts"][:19], "%Y-%m-%dT%H:%M:%S"))
        except Exception:
            continue
    return None


def tail_lines(path, n):
    with open(path, "rb") as f:
        f.seek(0, 2); size = f.tell()
        if n < 0:
            f.seek(0); data = f.read()
        else:
            block = 1 << 16; data = b""; pos = size
            while pos > 0 and data.count(b"\n") <= n:
                step = min(block, pos); pos -= step; f.seek(pos); data = f.read(step) + data
    lines = data.decode("utf-8", errors="replace").splitlines(keepends=True)
    if lines and not lines[-1].endswith("\n"):
        lines = lines[:-1]                                             # a row still being written by the served typist
    return lines if n < 0 else lines[-n:] if n else []


def parse_kv(strings, what):
    out = {}
    for s in strings:
        for t in shlex.split(s):
            if "=" not in t:
                raise SystemExit("%s: %r is not K=V" % (what, t))
            k, v = t.split("=", 1); out[k] = v
    return out


def parse_t0(s):
    if s is None:
        return None
    try:
        return float(s)
    except ValueError:
        return _time.mktime(_time.strptime(s[:19], "%Y-%m-%dT%H:%M:%S"))


def check_out(out, resume):
    """--out: never under body/, ops/, data/ or .git/ (any spelling), never the repo or above it; new or empty, or a run to resume"""
    repo_dirs = {ident(os.path.join(REPO, d)) for d in PROTECTED_DIRS} - {None}
    if is_within(out, repo_dirs):
        raise SystemExit("--out %s is under one of %s, which this tool never writes" % (out, ", ".join(d + "/" for d in PROTECTED_DIRS)))
    o = ident(out)
    if o is not None and o in {ident(a) for a in chain(os.path.realpath(REPO))}:
        raise SystemExit("--out %s is the repo or a folder above it" % out)
    if os.path.lexists(out) and not os.path.isdir(out):
        raise SystemExit("--out %s is not a directory" % out)
    if resume:
        if not os.path.exists(os.path.join(out, "run.json")):
            raise SystemExit("--resume: %s holds no run of this tool (no run.json)" % out)
        for name in os.listdir(out):
            p = os.path.join(out, name)
            if os.path.islink(p) or (os.path.isfile(p) and os.stat(p).st_nlink > 1):
                raise SystemExit("--resume: %s is a link; a run's own files are plain files" % p)
    elif os.path.isdir(out) and os.listdir(out):
        raise SystemExit("--out %s is not empty; a run starts in a new or empty directory (or pass --resume for a run of this tool)" % out)
    return os.path.realpath(out)


def norm(x):
    return json.loads(json.dumps(x, default=str, sort_keys=True))


def diff(a, b, path=""):
    if isinstance(a, dict) and isinstance(b, dict):
        out = []
        for k in sorted(set(a) | set(b)):
            out += diff(a.get(k, "<absent>"), b.get(k, "<absent>"), path + "." + str(k) if path else str(k))
        return out
    return [] if a == b else ["%s: %r -> %r" % (path, a, b)]


# ---------------- the run ----------------
def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    pre = argparse.ArgumentParser(add_help=False); pre.add_argument("--typist-from", default=None); pre.add_argument("--inputs-from", default=None)
    p0, _ = pre.parse_known_args(argv)
    preset, preset_env, preset_log = [], [], None
    if p0.typist_from:
        preset, preset_env, preset_log = read_chain_command(rpath(p0.typist_from))
    snap = []
    if p0.inputs_from:
        d = rpath(p0.inputs_from)
        snap = ["--queue", os.path.join(d, "queue.jsonl"), "--corpus", os.path.join(d, "corpus.json"),
                "--log", os.path.join(d, "caregiver.jsonl"), "--log-tail", "-1"]
        for p in snap[1::2][:3]:
            if not os.path.exists(p):
                raise SystemExit("--inputs-from %s: no %s (make it with --snapshot-only)" % (d, os.path.basename(p)))
    a = build_parser().parse_args(preset + (["--log", preset_log] if preset_log else []) + snap + argv)
    progress = lambda msg: print("[inproc] " + msg, file=sys.stderr, flush=True)   # noqa: E731
    night_min = max(NIGHT_MIN_S, 4.0 * a.tick * a.listen)
    if a.night_s is not None and (a.night_s < night_min or a.night_measured):
        raise SystemExit("--night-s must be at least %g s (60 s or four listening turns: the parent looks for the night only between "
                         "its turns), and not with --night-measured" % night_min)
    if a.body_tick <= 0 or (a.tick_floor is not None and a.tick_floor < 0):
        raise SystemExit("--body-tick must be > 0 and --tick-floor >= 0")
    nice0 = os.nice(0)
    if nice0 < 19:
        os.nice(19 - nice0)                                             # the served body first, whatever the caller forgot

    # -- paths, checked before anything is written --
    out = check_out(rpath(a.out), a.resume)
    body_src = rpath(a.body)
    if not a.birth_tiny and not a.snapshot_only and not (body_src and os.path.exists(body_src)) and not a.resume:
        raise SystemExit("a body save to copy is needed (or --birth-tiny)")
    flags_src, queue_src, corpus_src, log_src = rpath(a.flags), rpath(a.queue), rpath(a.corpus), rpath(a.log)
    pos_src = queue_src + ".pos" if queue_src else None
    log_path, corpus_path = os.path.join(out, "caregiver.jsonl"), os.path.join(out, "corpus.json")
    queue_path = os.path.join(out, "queue.jsonl"); pos_path = queue_path + ".pos"
    run_path = os.path.join(out, "run.json")
    if a.planner == "queue" and not a.resume:
        if not (queue_src and os.path.exists(queue_src)):
            raise SystemExit("no queue at %s" % queue_src)
        if a.queue_pos is not None:                                     # checked on the source: the copy is at least as long
            size = os.path.getsize(queue_src)
            if not 0 <= a.queue_pos <= size:
                raise SystemExit("--queue-pos %d is outside the queue (0..%d)" % (a.queue_pos, size))
            if a.queue_pos > 0:
                with open(queue_src, "rb") as f:
                    f.seek(a.queue_pos - 1)
                    if f.read(1) != b"\n":
                        raise SystemExit("--queue-pos %d is not at the start of a row" % a.queue_pos)
    if a.birth_tiny:
        cfg_from = None if (a.cfg_from or "").lower() == "none" else rpath(a.cfg_from) if a.cfg_from else (body_src or served_save())
    else:
        if a.cfg_from:
            raise SystemExit("--cfg-from is for --birth-tiny: a copy always takes its own save's constants (Life.load)")
        cfg_from = body_src
    if cfg_from and not os.path.exists(cfg_from) and not a.snapshot_only:
        raise SystemExit("no save at %s to take the constants from (--cfg-from none for the physiology's defaults)" % cfg_from)
    stored = json.load(open(run_path)) if a.resume else None
    save_path = rpath(a.save_as) if a.save_as else (stored["body"]["save"] if stored else os.path.join(out, "tiny_body.pt" if a.birth_tiny else "body_copy.pt"))
    os.makedirs(out, exist_ok=True)
    inputs = [p for p in (body_src, cfg_from, flags_src, queue_src, pos_src, corpus_src, log_src, rpath(a.tok)) if p and os.path.exists(p)]

    # -- the guard: from here on, before the run's first write --
    guard = install_guard()
    guard.protect(files=inputs)
    guard.begin_run([out])
    for p in (log_path, corpus_path, queue_path, pos_path, save_path, save_path + ".tmp", run_path):
        why = guard.verdict(p)
        if why is not None:
            raise SystemExit("the run's file %s is %s" % (p, why))
    if save_path.endswith(os.sep) or os.path.isdir(save_path):
        raise SystemExit("--save-as %s is a directory" % save_path)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)             # under --out (the guard has judged it)

    cwd0 = os.getcwd(); real_save = None; torch = None; swapped = None
    try:
        os.chdir(REPO)                                  # serve runs from the repo root (cfg's dream_corpus_file is relative)
        # -- the copies (the cursor before the queue, so the cursor never points past the copied queue) --
        copied = {}
        if not a.resume:
            if a.planner == "queue":
                if pos_src and os.path.exists(pos_src) and a.queue_pos is None:
                    shutil.copyfile(pos_src, pos_path); copied["pos"] = pos_src
                shutil.copyfile(queue_src, queue_path); copied["queue"] = queue_src
            if corpus_src and os.path.exists(corpus_src):
                shutil.copyfile(corpus_src, corpus_path); copied["corpus"] = corpus_src
            with open(log_path, "w") as f:
                if log_src and os.path.exists(log_src) and a.log_tail != 0:
                    f.writelines(tail_lines(log_src, a.log_tail)); copied["log_tail"] = "%s (last %s lines)" % (log_src, a.log_tail if a.log_tail > 0 else "all")
        hashes = {os.path.basename(p): {"sha256": sha256(p), "bytes": os.path.getsize(p)}
                  for p in (queue_path, pos_path, corpus_path, log_path) if os.path.exists(p)}
        if a.snapshot_only:
            with open(os.path.join(out, "inputs.json"), "w") as f:
                json.dump({"copied_from": copied, "files": hashes, "made": iso(_time.time())}, f, indent=1)
            print("inproc_parent: a snapshot of the inputs in %s (%s); use --inputs-from %s" % (out, ", ".join(sorted(hashes)), out))
            return {"snapshot": out, "files": hashes, "copied_from": copied}
        log_offset = os.path.getsize(log_path)
        own_rows_from = stored["own_rows_from"] if stored else log_offset
        day0 = a.day if a.day is not None else ((last_day(log_path) if a.resume else None) or last_day(log_src) or 0) + 1

        # -- the typist's environment, read at its import: a fresh import per run --
        env = parse_kv(preset_env, "--typist-from ENV"); env.update(parse_kv(a.env, "--env"))
        import torch                                                         # noqa: F811
        from tokenizers import Tokenizer
        from body.life import Life, PHYSIOLOGY
        C, T = fresh_typist(env)
        swapped = (C, T, C.time, T.time, C.urllib)
        env_seen = {k: getattr(C, k) if hasattr(C, k) else getattr(T, k, None) for k in TYPIST_ENV}
        env_dead = [k for k in env if not hasattr(C, k) and not hasattr(T, k)]
        if env_dead:
            progress("env read by neither body.caregiver nor body.teacher (no effect): " + " ".join(env_dead))
        if a.threads:
            torch.set_num_threads(int(a.threads))

        # -- the constants: the save's own (read without its tensors), then serve's flags, then --cfg-default and --cfg --
        fl = open(flags_src).read().split() if flags_src else []
        sa, unknown = serve_parser().parse_known_args(fl)
        if unknown:
            progress("flags not serve's, ignored: %s" % " ".join(unknown))
        flags_cfg = {k: getattr(sa, k) for k in PHYSIOLOGY if getattr(sa, k) is not None}
        meta = save_meta(cfg_from) if cfg_from else {"cfg": {}, "arch": None}
        cfg = dict(meta["cfg"])
        if cfg_from:
            cfg.setdefault("gate_int_form", "value")                     # Life.load's rule for a save
        cfg.update(flags_cfg)
        save_only = {k: v for k, v in meta["cfg"].items() if k not in flags_cfg and k in PHYSIOLOGY and v != PHYSIOLOGY[k]}
        reset = [k for s in a.cfg_default for k in s.replace(",", " ").split()] + (list(REFLEXES) if a.reflexes_off else [])
        for k in reset:
            if k not in PHYSIOLOGY:
                raise SystemExit("--cfg-default: %s is not physiology" % k)
            cfg[k] = PHYSIOLOGY[k]
        for k, v in parse_kv(a.cfg, "--cfg").items():
            if k not in PHYSIOLOGY:
                raise SystemExit("--cfg: %s is not physiology" % k)
            cfg[k] = type(PHYSIOLOGY[k])(v)
        device = a.device or sa.dev or "cpu"
        bseed = a.body_seed if a.body_seed is not None else int(sa.seed)
        tok_path = rpath(a.tok) or rpath(sa.tok)
        tok = Tokenizer.from_file(tok_path)
        floor = a.tick_floor if a.tick_floor is not None else float(sa.period)
        settings = norm({
            "body": {"kind": "tiny" if a.birth_tiny else "copy", "source": body_src, "cfg_from": cfg_from, "seed": bseed,
                     "device": device, "tok": tok_path, "flags": flags_src},
            "cfg": cfg,
            "typist": {"planner": a.planner, "period": a.period, "quiet": a.quiet, "cap": a.cap, "seed": a.seed, "tick": a.tick,
                       "listen": a.listen, "answer_levels": a.answer_levels, "parent": a.parent, "reply": a.reply, "wait": a.wait,
                       "yield": a.yield_ticks},
            "env": env,
            "clock": {"body_tick": a.body_tick, "tick_measured": a.tick_measured, "tick_floor": floor, "night_s": a.night_s,
                      "night_measured": a.night_measured}})
        changes = diff(stored["settings"], settings) if stored else []
        if changes and not a.resume_changes:
            raise SystemExit("--resume: the settings differ from the run's (pass --resume-changes to accept them):\n  " + "\n  ".join(changes))

        # -- memory and disk, before the copy is loaded --
        src_bytes = float(os.path.getsize(save_path if a.resume else body_src)) if (a.resume or not a.birth_tiny) else 0.0
        if not a.birth_tiny and not a.allow_low_memory:
            try:
                import psutil
                avail = float(psutil.virtual_memory().available); need = 2.5 * src_bytes + GiB
                if avail < need:
                    raise SystemExit("the copy needs about %.1f GiB of memory and %.1f GiB is free: it would take the served body's "
                                     "(pass --allow-low-memory to load it anyway)" % (need / GiB, avail / GiB))
            except ImportError:
                progress("psutil missing: the free memory is not checked before the copy is loaded")
        served = served_save()
        same_disk = ident(os.path.dirname(served)) is not None and os.stat(out).st_dev == os.stat(os.path.dirname(served)).st_dev
        reserve = (a.disk_reserve_gb * GiB if a.disk_reserve_gb is not None else
                   max(2 * GiB, 2.0 * float(os.path.getsize(served)) + GiB) if same_disk and os.path.exists(served) else GiB)
        disk = Disk(out, reserve, src_bytes)
        try:
            disk.check(save_path, "the copy's first save")
        except StopRun as e:
            raise SystemExit(str(e))

        # -- the copy, loaded as serve loads it; its first save is the evening a non-finite night reloads --
        real_save = torch.save
        torch.save = audited_save(real_save, before=disk.check, after=disk.saved)
        real_start = _time.perf_counter()
        torch.manual_seed(bseed)
        if a.resume:
            life = Life.load(save_path, tok, device=device, cfg=cfg, seed=bseed, save_path=save_path)
            what = "%s resumed from its save %s (ticks %d, nights %d)" % ("a tiny newborn" if a.birth_tiny else "a copy", save_path, life.ticks, life.nights)
        elif a.birth_tiny:
            life = Life.birth(tok, device=device, d=64, layers=2, heads=2, window=32, cfg=cfg, seed=bseed, save_path=save_path)
            what = "a tiny newborn (d=64, layers=2, heads=2, window=32, seed %d)" % bseed
            life.save()
        else:
            life = Life.load(body_src, tok, device=device, cfg=cfg, seed=bseed, save_path=save_path)
            what = "a copy of %s (ticks %d, nights %d)" % (body_src, life.ticks, life.nights)
            life.save()
        load_real = _time.perf_counter() - real_start
        nights0 = life.nights
        t0 = parse_t0(a.t0)
        if stored:
            t_prev = stored.get("t_end") or last_ts(log_path, own_rows_from) or 0.0
            if t0 is None or t0 < t_prev:
                if t0 is not None:
                    progress("--t0 is before the run's last virtual time: the clock goes on from %s" % iso(t_prev))
                t0 = float(int(t_prev) + 1)
            if a.queue_pos is not None:
                progress("--queue-pos ignored on --resume: the queue goes on from the run's own cursor")
        elif t0 is None:
            t0 = float(int(_time.time()))
        body = InProcBody(life, t0, tick_s=a.body_tick, measured=a.tick_measured, floor=floor, night_s=a.night_s,
                          night_measured=a.night_measured, night_min=night_min, max_virtual_s=a.max_virtual_s, max_real_s=a.max_real_s,
                          max_ticks=a.max_ticks, progress=progress)
        progress("%s; wake_ticks %d; constants: %s (%d of them set by the save alone), then the flags; typist env in effect %s"
                 % (what, int(life.cfg["wake_ticks"]), cfg_from or "the physiology's defaults", len(save_only),
                    " ".join("%s=%s" % kv for kv in env_seen.items())))
        if reset or a.cfg:
            progress("cfg on the copy: " + ", ".join("%s=%r" % (k, cfg[k]) for k in dict.fromkeys(reset + list(parse_kv(a.cfg, "--cfg")))))
        if changes:
            progress("--resume-changes: " + "; ".join(changes))
        run_rec = {"t0": t0, "t_end": None, "argv": argv, "days": [], "stopped": None, "changes": changes, "started": iso(_time.time())}
        record = stored or {"tool": "inproc_parent", "created": iso(_time.time()), "own_rows_from": own_rows_from, "inputs": hashes,
                            "copied_from": copied, "settings": settings,
                            "body": {"what": what, "save": save_path, "arch": meta.get("arch") if not a.birth_tiny else None}, "runs": []}
        if changes:
            record["settings"] = settings
        record["runs"].append(run_rec)

        def write_record():
            tmp = run_path + ".tmp"
            with open(tmp, "w") as f:
                json.dump(record, f, indent=1, default=str)
            os.replace(tmp, run_path)
        write_record()

        # -- the swap: the wire and the clock, in body.caregiver and body.teacher only --
        vt = VirtualTime(body)
        C.time = vt; T.time = vt
        C.urllib = types.SimpleNamespace(request=types.SimpleNamespace(Request=urllib.request.Request, urlopen=body.urlopen))

        stopped = None; days_done = []; days_begun = []
        real_loop0 = _time.perf_counter()
        try:
            prev_q = None
            for k in range(a.days):                                          # body/teacher.py main(), day by day
                day = day0 + k
                rng = random.Random(a.seed + day)
                if a.planner == "queue":
                    first_pos = a.queue_pos if not a.resume else None
                    planner = T.QueuePlanner(queue_path, rng, pos=(prev_q.pos if prev_q else first_pos), buf=(prev_q.buf if prev_q else None))
                    prev_q = planner
                else:
                    planner = T.FixedPlanner(rng)
                corpus = T.Corpus(corpus_path)                               # reloaded each day, as main() does; persisted in --out
                t = T.Teacher(BASE, day, log_path, corpus, planner, period=a.period, quiet=a.quiet, cap=a.cap, seed=day,
                              answer_levels=a.answer_levels, parent=a.parent, reply=a.reply, wait=a.wait, tick=a.tick,
                              listen=a.listen, yield_ticks=a.yield_ticks)
                v0 = body.now; days_begun.append(day)
                t.run_day()
                days_done.append(day)
                progress("day %d done: %d parent smiles, %d frowns, virtual %s, real %.1f s so far"
                         % (day, t.smiles, t.frowns, hms(body.now - v0), _time.perf_counter() - real_start))
        except StopRun as e:
            stopped = str(e)
        except KeyboardInterrupt:
            stopped = "interrupted"
        finally:
            run_rec.update({"t_end": body.now, "days": days_done, "days_begun": days_begun, "stopped": stopped})
            record["t_end"] = body.now
            write_record()
        real_loop = _time.perf_counter() - real_loop0

        # -- the summary, from the rows this invocation wrote --
        with open(log_path, "rb") as f:
            f.seek(log_offset); rows = [json.loads(l) for l in f.read().decode("utf-8").splitlines() if l.strip()]
        acts = collections.Counter(r.get("action") for r in rows)
        said = [r for r in rows if r.get("action") in ("line", "cue")]
        pl = [r for r in said if r.get("voice", "a") == "a"]; ol = [r for r in said if r.get("voice") == "b"]
        slow = [r for r in pl if r.get("slow")]
        smiles = [r for r in rows if r.get("action") == "smile"]
        sm = collections.Counter("answer" if str(r.get("why", "")).startswith("answer") else "cue" if str(r.get("why", "")).startswith("cue")
                                 else "known word" for r in smiles)
        frowns = [r for r in rows if r.get("action") == "frown"]
        fr = collections.Counter("talked over" if r.get("why") == "talked over" else "symbol run" for r in frowns)
        missed = collections.Counter(re.sub(r"^late .*", "late", str(r.get("why"))) for r in rows if r.get("action") == "missed")
        over_words = sum(int(r.get("over") or 0) for r in pl); over_words_b = sum(int(r.get("over") or 0) for r in ol)
        ends = [r for r in rows if r.get("action") == "session_end"]
        t_first = rows[0]["ts"] if rows else None; t_last = rows[-1]["ts"] if rows else None
        virtual = body.now - body.t0
        tmp = os.path.realpath(tempfile.gettempdir())
        in_out = lambda p: is_within(p, {ident(out)})                   # noqa: E731
        outside = sorted(p for p in guard.written if not in_out(p))
        report_root = os.path.join(out, "report")
        rep_cmd = None
        if rows:
            ds = days_begun or [day0]
            label = "inproc day%s %s" % ("s" if len(ds) > 1 else "", "-".join(str(d) for d in dict.fromkeys([ds[0], ds[-1]])))
            rep_cmd = [os.path.join(report_root, "ops", "day_report.py"), iso(body.t0), iso(body.now + 1), "--label", label]
        summary = {
            "body": what, "save": save_path, "out": out, "days": days_done, "days_begun": days_begun, "stopped": stopped,
            "parent_lines": len(pl), "slow_lines": len(slow), "cues": sum(1 for r in pl if r.get("action") == "cue"), "other_lines": len(ol),
            "symbols_typed": body.typed, "faces_set": body.faces,
            "smiles": len(smiles), "smiles_by": dict(sm), "frowns": len(frowns), "frowns_by": dict(fr),
            "talked_over_rows": missed.get("talked over", 0), "over_words_on_parent_lines": over_words, "over_words_on_other_lines": over_words_b,
            "slow_lines_clean": sum(1 for r in slow if int(r.get("over") or 0) == 0), "missed_by": dict(missed),
            "aways": acts.get("away", 0), "e_end": [r.get("e") for r in ends],
            "nights": body.nights, "night_rows": acts.get("night", 0), "body_nights": [nights0, life.nights], "night_log": body.night_log,
            "ticks": body.ticks, "tick_errors": body.tick_errors, "last_tick_error": body.last_error,
            "own_symbols": body.own, "world_symbols": body.world, "calls": dict(body.calls),
            "virtual_s": round(virtual, 1), "virtual_t0": iso(body.t0), "virtual_first_row": t_first, "virtual_last_row": t_last,
            "real_s": round(_time.perf_counter() - real_start, 1), "real_load_s": round(load_real, 1), "real_loop_s": round(real_loop, 1),
            "real_ticks_s": round(body.real_ticks_s, 1), "real_nights_s": round(body.real_nights_s, 1),
            "clock": ("measured, floor %.3f s" % floor) if a.tick_measured else ("fixed %.3f s per tick" % a.body_tick),
            "night_s": a.night_s if a.night_s is not None else (("measured, at least %g s" % night_min) if a.night_measured else SERVED_NIGHT_S),
            "threads": torch.get_num_threads(), "nice": os.nice(0),
            "typist": {"day0": day0, "days": a.days, "planner": a.planner, "period": a.period, "quiet": a.quiet, "cap": a.cap,
                       "tick": a.tick, "listen": a.listen, "answer_levels": a.answer_levels, "parent": a.parent, "reply": a.reply,
                       "wait": a.wait, "yield": a.yield_ticks, "seed": a.seed, "queue_pos": a.queue_pos if not a.resume else "resumed",
                       "env_requested": env, "env_in_effect": env_seen},
            "constants": {"from_save": cfg_from, "set_by_the_save_alone": save_only, "flags": flags_src,
                          "cfg_set": {k: cfg[k] for k in dict.fromkeys(reset + list(parse_kv(a.cfg, "--cfg")))}},
            "files": {"log": log_path, "log_rows_from_byte": log_offset, "own_rows_from_byte": own_rows_from, "corpus": corpus_path,
                      "queue": queue_path, "pos": pos_path, "run": run_path, "copied_from": record.get("copied_from"),
                      "inputs_at_start": record.get("inputs")},
            "disk": {"saves": disk.saves, "reserve_gib": round(disk.reserve / GiB, 2), "free_gib": round(disk.free() / GiB, 1), "low": disk.low},
            "guard": {"refused": guard.refused, "files_written": len(guard.written), "written_outside_out": outside},
            "day_report": (["python3"] + rep_cmd) if (rep_cmd and a.report) else None,
        }
        with open(os.path.join(out, "summary.json"), "w") as f:
            json.dump(summary, f, indent=1, default=str)

        W = print
        W("inproc_parent: %s" % what)
        W("  days %s (begun %s)%s; typist: planner %s, tick %.3f, period %g, listen %g, parent %d, reply %d, wait %d"
          % (",".join(map(str, days_done)) or "none", ",".join(map(str, days_begun)) or "none", (" (STOPPED: %s)" % stopped) if stopped else "", a.planner, a.tick, a.period, a.listen, a.parent, a.reply, a.wait))
        W("  env in effect: " + " ".join("%s=%s" % kv for kv in env_seen.items()))
        W("  constants: %s then %s%s; %d set by the save alone (%s)"
          % (cfg_from or "the physiology's defaults", os.path.relpath(flags_src, REPO) if flags_src else "no flags",
             (", then " + ", ".join("%s=%r" % kv for kv in summary["constants"]["cfg_set"].items())) if summary["constants"]["cfg_set"] else "",
             len(save_only), ", ".join("%s=%s" % kv for kv in list(save_only.items())[:6]) + (" ..." if len(save_only) > 6 else "")))
        W("  lines typed: %d by the parent (%d slow, %d cues) and %d by the other voice; %d symbols typed"
          % (len(pl), len(slow), summary["cues"], len(ol), body.typed))
        W("  smiles %d (%s); frowns %d (%s)" % (len(smiles), ", ".join("%s %d" % kv for kv in sm.items()) or "none",
                                               len(frowns), ", ".join("%s %d" % kv for kv in fr.items()) or "none"))
        W("  talk-overs: %d words said over a line ('talked over' rows): %d over the parent's lines, %d over the other voice's; slow lines clean %d of %d"
          % (missed.get("talked over", 0), over_words, over_words_b, summary["slow_lines_clean"], len(slow)))
        W("  missed: " + (", ".join("%s %d" % kv for kv in missed.most_common()) or "none") + "; aways %d; e at day end %s" % (acts.get("away", 0), summary["e_end"]))
        W("  nights %d (the typist's night rows %d; the body's nights %d -> %d)" % (body.nights, acts.get("night", 0), nights0, life.nights))
        for n in body.night_log:
            W("    night %d: body tick %d at virtual +%s, %.0f s asleep%s, computed in %.2f s real%s"
              % (n["night"], n["body_tick"], hms(n["virtual_s"]), n["night_s"], (" (+%.2f s until the parent looked)" % n["held_s"]) if n.get("held_s") else "",
                 n["real_s"], (" error " + str(n["error"])) if n["error"] else ""))
        W("  body: %d ticks (%d errors), %d own symbols, %d world symbols heard; calls %s"
          % (body.ticks, body.tick_errors, body.own, body.world, dict(body.calls)))
        W("  virtual time %s (%s; night %s) from %s to %s" % (hms(virtual), summary["clock"], summary["night_s"], iso(body.t0), iso(body.now)))
        W("  real time %.1f s (load and first save %.1f s; loop %.1f s: ticks %.1f s, nights %.1f s); virtual/real x%.0f; nice %d, %d threads"
          % (summary["real_s"], load_real, real_loop, body.real_ticks_s, body.real_nights_s, virtual / max(1e-9, real_loop), os.nice(0), torch.get_num_threads()))
        W("  files: log %s (this run's rows from byte %d), corpus %s, queue %s (+.pos), save %s (%d saves; %.1f GiB free, %.1f kept)"
          % (log_path, log_offset, corpus_path, queue_path, save_path, disk.saves, disk.free() / GiB, disk.reserve / GiB))
        W("  guard: %d refused%s; %d paths written, %s%s" % (len(guard.refused), (" " + repr(guard.refused)) if guard.refused else "", len(guard.written),
                                                           "all under --out" if not outside else "outside --out only in the system temp dir or /dev/null: ",
                                                           ", ".join(outside)))
        if outside and any(not (is_within(p, {ident(tmp)}) or p == os.path.realpath(os.devnull)) for p in outside):
            W("  GUARD FAULT: a path outside --out and the temp dir was written: %s" % outside)
        if a.report and rep_cmd:
            # ops/day_report.py reads ROOT/data/watch2_caregiver.jsonl with ROOT its own grandparent: a copy of it in OUT/report/ops reads
            # OUT/report/data/watch2_caregiver.jsonl, which holds this run's own rows only (no copied served row), in a window from t0
            os.makedirs(os.path.join(report_root, "ops"), exist_ok=True); os.makedirs(os.path.join(report_root, "data"), exist_ok=True)
            shutil.copyfile(os.path.join(REPO, "ops", "day_report.py"), rep_cmd[0])
            with open(log_path, "rb") as src, open(os.path.join(report_root, "data", "watch2_caregiver.jsonl"), "wb") as dst:
                src.seek(own_rows_from); dst.write(src.read())
            W("  day report: python3 " + " ".join(shlex.quote(x) for x in rep_cmd))
            argv0 = sys.argv
            try:
                sys.argv = rep_cmd; runpy.run_path(rep_cmd[0], run_name="__main__")
            finally:
                sys.argv = argv0
        return summary
    finally:
        if torch is not None and real_save is not None:
            torch.save = real_save
        if swapped is not None:                                     # the typist modules' own clock and wire, as imported
            C_, T_, C_.time, T_.time, C_.urllib = swapped
        os.chdir(cwd0)
        guard.end_run()


if __name__ == "__main__":
    main()
