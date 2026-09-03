"""the raw caregiver (CURRICULUM.md, BODY_SPEC.md §6) as a script: it decides from the page and its
face row only. One process: the watcher (smiles, frowns, expansions) and the typist (lines and cues
at the pace, gated by its quiet). Logs one JSON row per action.

  python3 -m body.caregiver --port 8018 --day 1 --log data/body2_caregiver.jsonl \
      --lines "dog will go|give milk|..." --cues "dog will |give |..." --period 120
"""
import argparse
import json
import os
import random
import sys
import time
import urllib.request

KNOWN = set("""hi bye no more up dog ball milk book go big my please the you two in on going
want where not give little under one three went what then can happy will sad why because
first bigger had scared saw balls dogs books all gone""".split())
KNOWN2 = {w for w in KNOWN if len(w) >= 2}
ANSWERS = {"dog will ": ["go"], "scared ": ["dog", "ball"], "give ": ["milk", "ball", "book"], "where ball? ": ["ball"],
           "I had ": ["milk"], "first milk then ": ["ball"], "big dog bigger ": ["dog"], "why dog up? ": ["because"],
           "what? ": ["dog", "ball", "milk", "sad", "scared"], "I can ": ["go"], "sad ": ["dog", "ball"], "happy ": ["dog", "ball"]}
EXPAND = {"b": "big ball", "d": "big dog", "m": "more milk", "g": "go up", "l": "little dog", "u": "up", "w": "what? dog",
          "h": "happy dog", "p": "please", "n": "no more milk", "s": "sad dog", "c": "dog can go", "o": "one ball", "a": "all gone",
          "i": "I go up", "f": "first milk then ball", "t": "the dog", "y": "you go up", "e": "the dog", "r": "more milk", "k": "milk"}


def iso(t=None):
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(t or time.time()))


class Caregiver:
    def __init__(self, base, day, log, period=120.0, quiet=6.0, cap=90.0, seed=0):
        self.base, self.day, self.log = base, day, log
        self.period, self.quiet_needed, self.cap = period, quiet, cap
        self.rng = random.Random(seed)
        self.cursor = 0; self.its = {}; self.tobs = {}; self.maxtick = -1; self.finalized = -1
        self.last_smile = 0.0; self.last_frown = 0.0; self.last_word = None
        self.letter_runs = {}; self.expanded = set(); self.pending_expand = None
        self.cue = None                                  # (text, until, prefixes, full, done)
        self.smiles = 0; self.frowns = 0; self.state = {}
        self.events = []

    def req(self, path, data=None, timeout=30):
        url = self.base + path
        r = urllib.request.Request(url) if data is None else urllib.request.Request(
            url, data=json.dumps(data).encode(), headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(r, timeout=timeout) as f:
            return json.loads(f.read().decode() or "{}")

    def row(self, obj):
        obj = dict(obj); obj.setdefault("day", self.day); obj.setdefault("ts", iso())
        with open(self.log, "a") as f:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")

    # ---- the page ----
    def poll(self):
        try:
            d = self.req("/state?since=%d" % self.cursor)
        except Exception as e:
            self.row({"action": "error", "err": repr(e)[:120]}); time.sleep(2); return None
        n = d.get("n", 0)
        if n < self.cursor:                               # the serve restarted
            self.cursor = 0; d = self.req("/state?since=0"); n = d.get("n", 0)
        t = time.time()
        for i, e in enumerate(d.get("page", [])):
            idx = self.cursor + i; tick = idx // 2
            if idx % 2 == 1:
                self.its[tick] = e[0]; self.tobs[tick] = t; self.maxtick = max(self.maxtick, tick)
        self.cursor = n; self.state = d
        return d

    def quiet_since(self):
        last_write = max([t for t, s in self.its.items() if s != ""] or [-1])
        if last_write < 0 or self.maxtick < 0:
            return 999.0
        return self.tobs.get(self.maxtick, time.time()) - self.tobs.get(last_write, time.time())

    # ---- the face ----
    def face(self, v):
        try:
            self.req("/face", {"expr": v})
        except Exception:
            pass

    def smile(self, on, ctx, why):
        db = (self.state.get("last") or {}).get("doses")
        self.face(2); t0 = time.time(); time.sleep(1.2)
        try:
            la = self.req("/state?since=%d" % max(self.cursor - 2, 0)).get("last", {})
        except Exception:
            la = {}
        self.face(0)
        self.smiles += 1; self.last_smile = time.time(); self.last_word = on
        self.row({"action": "smile", "on": on, "context": ctx, "why": why, "ts": iso(t0),
                  "doses_before": db, "doses_after": la.get("doses"), "mood": la.get("mood"), "gate": la.get("gate")})

    def frown(self, on, ctx):
        self.face(-2); time.sleep(1.2); self.face(0)
        self.frowns += 1; self.last_frown = time.time()
        self.row({"action": "frown", "on": on, "context": ctx})

    def ctx(self, a, b):
        return "".join((self.its.get(t) or "_") for t in range(max(0, a - 10), b + 11) if self.its.get(t) is not None)

    # ---- the watcher ----
    def scan(self):
        start, end = self.finalized + 1, self.maxtick
        t = start
        while t < end:
            sym = self.its.get(t, "")
            if sym in ("", " "):
                self.finalized = t; t += 1; continue
            a = t
            while t < end and self.its.get(t, "") not in ("", " "):
                t += 1
            if t >= end:
                return                                    # still growing
            b = t - 1; self.finalized = b
            self.on_token("".join(self.its.get(k, "") for k in range(a, b + 1)), a, b)
        self.finalized = max(self.finalized, end - 1)

    def on_token(self, tok, a, b):
        wall = self.tobs.get(b, time.time()); low = tok.lower(); ctx = self.ctx(a, b)
        if len(tok) >= 3 and len(set(low)) == 1:
            if low[0].isalpha():
                self.letter_runs[low[0]] = self.letter_runs.get(low[0], 0) + 1
                if self.letter_runs[low[0]] >= 3 and low[0] not in self.expanded and low[0] in EXPAND:
                    self.expanded.add(low[0]); self.pending_expand = EXPAND[low[0]]
            elif low[0] != " " and time.time() - self.last_frown > 60:
                self.frown(tok, ctx); return
        c = self.cue
        if c and wall <= c["until"] and not c["done"] and len(tok) >= 2:
            if low in [x.lower() for x in c["full"]]:
                c["done"] = True; self.smile(tok, ctx, "cue completion: " + c["text"]); return
            if low[:2] in [x[:2].lower() for x in c["full"]] and time.time() - wall < 3.5:
                c["done"] = True; self.smile(tok, ctx, "cue prefix: " + c["text"]); return
        if low in KNOWN2:
            age = time.time() - wall
            if self.last_word == low and time.time() - self.last_smile < 12:
                self.row({"action": "withheld", "on": tok, "why": "same word twice", "context": ctx})
            elif age <= 3.2 and time.time() - self.last_smile >= 2.0:
                self.smile(tok, ctx, "known word")
            else:
                self.row({"action": "missed", "on": tok, "why": "late %.1fs" % age, "context": ctx})

    def watch(self, seconds):
        t0 = time.time()
        while time.time() - t0 < seconds:
            d = self.poll()
            if d is not None:
                self.scan()
            time.sleep(1.5)

    # ---- the typist ----
    def gate(self):
        t0 = time.time()
        while True:
            d = self.poll()
            if d is None:
                continue
            self.scan()
            if d.get("asleep"):
                return -1.0
            if self.quiet_since() >= self.quiet_needed or time.time() - t0 >= self.cap:
                return time.time() - t0
            time.sleep(1.0)

    def event(self, text, kind):
        gw = self.gate()
        if gw < 0:
            return False
        last_before = self.state.get("last") or {}
        if kind == "cue":
            self.cue = {"text": text, "until": time.time() + 90, "full": ANSWERS.get(text, []), "done": False}
        self.req("/type", {"text": text}); t_start = time.time()
        while True:
            d = self.poll(); self.scan()
            if d is not None and d.get("queued", 0) == 0:
                break
            time.sleep(0.4)
        tick_end = self.maxtick                            # the page's own tick count (survives a serve restart)
        self.watch(12.4)
        its = "".join((self.its.get(t) or "_") for t in range(tick_end + 1, tick_end + 26) if self.its.get(t) is not None)
        la = self.state.get("last") or {}
        self.row({"action": kind, "text": text, "gate_wait_s": round(gw, 1), "its_after": its, "ts": iso(t_start),
                  "fatigue": la.get("fatigue"), "stress": la.get("stress"), "mood": la.get("mood"), "gate": la.get("gate"),
                  "own": la.get("own"), "doses": la.get("doses"), "smiles_so_far": self.smiles,
                  "sleep_pressure": self.state.get("sleep_pressure"), "store": self.state.get("store")})
        if kind == "cue":
            time.sleep(3); self.cue = None
        return True

    def run_day(self, lines, cues):
        self.poll(); self.finalized = self.maxtick - 1; self.face(0)
        self.row({"action": "session_start", "sleep_pressure": self.state.get("sleep_pressure"), "nights": self.state.get("nights")})
        plan = []; li = 0; ci = 0
        order = self.rng.sample(lines, len(lines))
        while li < len(order) or ci < len(cues):
            if li < len(order):
                plan.append(("line", order[li])); li += 1
            if li % 2 == 0 and ci < len(cues):
                plan.append(("cue", cues[ci])); ci += 1
        slept = False; prev = None
        for kind, text in plan:
            if self.pending_expand:
                text2 = self.pending_expand; self.pending_expand = None
                self.event(text2, "line")
            if prev is not None:
                while time.time() < prev + self.period:
                    self.watch(min(3.0, prev + self.period - time.time()))
            prev = time.time()
            if not self.event(text, kind):
                slept = True; break
        if not slept:
            self.event("bye", "line")
        # the night
        d = self.poll()
        while d is None or not d.get("asleep"):
            self.watch(5.0); d = self.poll()
            if (d or {}).get("sleep_pressure", 0) >= (d or {}).get("wake_ticks", 1) - 2 and not (d or {}).get("asleep"):
                pass
            if d and d.get("nights", 0) > (self.state.get("nights") or 0):
                break
        t_sleep = time.time()
        while True:
            d = self.poll()
            if d is not None and not d.get("asleep"):
                break
            time.sleep(5)
        self.row({"action": "night", "slept_s": round(time.time() - t_sleep, 1), "last_night": (self.state or {}).get("last_night")})
        # the post-night cues, a known line between each pair
        post = list(cues); lines2 = self.rng.sample(lines, len(lines))
        prev = None; j = 0
        for i, c in enumerate(post):
            for text, kind in ([(c, "cue")] + ([(lines2[j % len(lines2)], "line")] if i % 2 == 1 else [])):
                if prev is not None:
                    while time.time() < prev + self.period:
                        self.watch(min(3.0, prev + self.period - time.time()))
                prev = time.time(); self.event(text, kind)
                if kind == "line":
                    j += 1
        try:
            self.row({"action": "save", "result": self.req("/save", {})})
        except Exception as e:
            self.row({"action": "save", "error": repr(e)[:120]})
        self.watch(120)
        self.row({"action": "session_end", "smiles": self.smiles, "frowns": self.frowns})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8018)
    ap.add_argument("--day", type=int, default=1)
    ap.add_argument("--log", default="data/body2_caregiver.jsonl")
    ap.add_argument("--period", type=float, default=120.0)
    ap.add_argument("--quiet", type=float, default=6.0)
    ap.add_argument("--cap", type=float, default=90.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--lines", default="dog will go|I will go up|you will go in|scared dog|scared ball|what? scared dog|give milk|give ball|give book|ball under|ball on|where ball? ball under|I had milk|you had ball|dog had ball|first milk then ball|first up then in|big dog bigger dog|bigger dog up|I saw dog|you saw dog|why dog up? because big dog")
    ap.add_argument("--cues", default="dog will |scared |give |where ball? |I had |first milk then |big dog bigger |why dog up? ")
    a = ap.parse_args()
    cg = Caregiver("http://localhost:%d" % a.port, a.day, a.log, period=a.period, quiet=a.quiet, cap=a.cap, seed=a.seed)
    cg.run_day([x for x in a.lines.split("|") if x], [x for x in a.cues.split("|") if x])


if __name__ == "__main__":
    main()
