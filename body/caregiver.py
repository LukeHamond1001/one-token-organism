"""the raw caregiver (CURRICULUM.md, BODY_SPEC.md §6) as a script: it decides from the page and its
face row only. One process: the watcher (smiles, frowns, expansions) and the typist (lines and cues
at the pace, gated by its quiet). Logs one JSON row per action.

  python3 -m body.caregiver --port 8018 --day 1 --log data/body2_caregiver.jsonl \
      --lines "dog will go|give milk|..." --cues "dog will |give |..." --period 120
"""
import argparse
import json
import os
TALKOVER_FROWN = int(os.environ.get("TALKOVER_FROWN", "0"))   # the served typist's face when talked over: off until a day boundary after the fast seeds read (the user's word given 2026-09-06)
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
    def __init__(self, base, day, log, period=480, quiet=24, cap=360, seed=0, answer_levels=1, parent=0, reply=0, wait=0.0, tick=0.25, listen=50):
        self.base, self.day, self.log = base, day, log
        # THE PARENT'S CLOCK IS THE BODY'S (the user's word of 2026-09-08, "have it live in speed time"): every timer of this
        # parent is a count of the body's ticks, converted to seconds by the served tick length, so its cadence, its face and
        # its patience are the same in the body's time whatever the wall clock does. It looks at the page once a tick.
        self.tick = float(tick); self.listen = self.s(listen)   # listen: the child's turn after each line, in ticks
        # THE REPLY WITHHELD (the user's word of 2026-09-04, "both are your call"): the parent answers when the child has
        # finished. Its smile for the answer and its reply (the cued line in full, the recast) come after the child's
        # quiet of `wait` ticks (4: a second), and every symbol the child adds before that postpones them: a parent does not praise
        # over a child still talking, and a child who talks over its parent gets no reply until it stops. After the cap
        # it replies anyway; the answer smile is always given. Decided from the page alone. 0 = the smile at once.
        self.wait = int(wait); self.reply_pending = None; self.reply_line = None
        self.answer_levels = int(answer_levels)          # 2: the smile at a cue's completion grows, 2 then 4 (CURRICULUM.md)
        # THE PARENT (the user's word of 2026-09-03): attention that moves, decided from the page alone. e rises at an
        # answer or a known word (more at a word new today), falls at babble, drifts down in silence (150 s); a known
        # word is smiled at with probability e, less for the fiftieth "dog"; the parent talks faster when engaged,
        # answers a smiled word with a line that holds it, and below a floor turns away for 50 s (the still face).
        self.parent = int(parent); self.e = 0.6; self.word_count = {}; self.away_until = 0.0; self.aways = 0
        # THE PARENT WANTS A REPLY (the user's word of 2026-09-04): once its cue is answered, each further word the
        # child adds before the parent's next turn wears its attention and gets no smile, unless the words go on
        # completing the cued line; a word said over the parent's own typing does the same. The answer smile stands.
        self.reply = int(reply); self.reply_cue = None; self.reply_tokens = []; self.answered = False; self.past = False
        self.typing_span = (-1, -1)                       # the parent's own turn, in ticks
        self.expand_next = None; self._e_t = time.time()
        self.period, self.quiet_ticks, self.cap = self.s(period), int(quiet), self.s(cap)
        self.rng = random.Random(seed)
        self.cursor = 0; self.its = {}; self.tobs = {}; self.maxtick = -1; self.finalized = -1
        self.last_smile = 0.0; self.last_frown = 0.0; self.last_word = None
        self.letter_runs = {}; self.expanded = set(); self.pending_expand = None
        self.cue = None                                  # (text, until, prefixes, full, done)
        self.smiles = 0; self.frowns = 0; self.state = {}
        self._face_timer = None; self._cue_clear_at = None      # THE FACE ON TIMERS (the review of 2026-09-10): a held face no longer blinds the typist
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
    def s(self, ticks):
        return float(ticks) * self.tick                  # seconds for a count of the body's ticks

    def pace(self):
        return self.period * (1.6 - self.e) if self.parent else self.period

    def _attend(self):
        now = time.time(); dt = now - self._e_t; self._e_t = now
        self.e += (0.5 - self.e) * min(1.0, dt / self.s(600))   # attention drifts toward its resting level in silence (600 ticks); 0.3 -> 0.5 on 2026-09-11: at 0.3 the parent withheld 150-200 known-word smiles a day as "distracted"
        if self.e < 0.15 and now >= self.away_until:
            self.away_until = now + self.s(200); self.aways += 1
            self.row({"action": "away", "e": round(self.e, 3)}); self.e = 0.35

    def _timers(self):
        now = time.time()
        if self._face_timer is not None and now >= self._face_timer[0]:
            expr, rest = self._face_timer[1], self._face_timer[2]
            self.face(expr); self._face_timer = rest
        if self._cue_clear_at is not None and now >= self._cue_clear_at:
            self.cue = None; self._cue_clear_at = None

    def _hold(self, expr, ticks, then=0.0, then_ticks=None, then_expr=None):
        """set a face now and let it go back after `ticks` (or step to `then_expr` and then back): the loop keeps polling meanwhile"""
        self.face(expr)
        rest = None if then_expr is None else (time.time() + self.s(ticks) + self.s(then_ticks or ticks), then, None)
        self._face_timer = (time.time() + self.s(ticks), then if then_expr is None else then_expr, rest)

    def poll(self):
        self._timers()
        if self.parent:
            self._attend()
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
        """the child's quiet, in the page's own ticks"""
        last_write = max([t for t, s in self.its.items() if s != ""] or [-1])
        if last_write < 0 or self.maxtick < 0:
            return 10 ** 6
        return self.maxtick - last_write

    # ---- the face ----
    def face(self, v):
        try:
            self.req("/face", {"expr": v})
        except Exception:
            pass

    def smile(self, on, ctx, why, extra=None):
        db = (self.state.get("last") or {}).get("doses")
        levels = self.answer_levels if why.startswith("cue") else 1
        t0 = time.time(); la = {}
        if levels >= 2:
            self._hold(2, 2.5, then=0.0, then_ticks=2.5, then_expr=4)
        else:
            self._hold(2, 5)
        self.smiles += 1; self.last_smile = time.time(); self.last_word = on
        self.row(dict({"action": "smile", "on": on, "context": ctx, "why": why, "levels": levels, "e": round(self.e, 3), "ts": iso(t0),
                       "doses_before": db, "doses_after": la.get("doses"), "mood": la.get("mood"), "gate": la.get("gate")}, **(extra or {})))

    # ---- the reply withheld ----
    def reply_line_for(self, cue, answer):
        lines = [l for l in self.lines() if l.startswith(cue) and (l[len(cue):].split() or [""])[0] == answer]
        return self.rng.choice(lines) if lines else (cue + answer).strip()

    def hold_reply(self, tok, ctx, why, cue, answer, b):
        self.reply_pending = {"tok": tok, "ctx": ctx, "why": why, "end": b, "t0": time.time(), "line": self.reply_line_for(cue, answer)}

    def deliver(self):
        rp = self.reply_pending
        if rp is None:
            return
        now = time.time()
        last_write = max([t for t, s_ in self.its.items() if s_ != ""] or [-1])
        yielded = self.maxtick - last_write >= self.wait       # the child's quiet, in the page's own ticks
        if yielded or now - rp["t0"] >= self.cap:
            run_on = sum(1 for t in range(rp["end"] + 2, self.maxtick + 1) if self.its.get(t))
            self.reply_pending = None
            self.smile(rp["tok"], rp["ctx"], rp["why"], extra={"waited_s": round(now - rp["t0"], 1), "run_on": run_on, "yielded": yielded})
            self.reply_line = rp["line"]

    def pace_wait(self, prev, pace):
        """watch until prev + pace; a reply that comes due is spoken at once and the pace restarts from it.
        Returns the new prev, or None if the body fell asleep under the reply."""
        while time.time() < prev + pace:
            if self.reply_line:
                line = self.reply_line; self.reply_line = None
                prev = time.time()
                if not self.event(line, "line"):
                    return None
                continue
            self.watch(min(self.s(12), prev + pace - time.time()))
            if (self.state or {}).get("asleep"):
                break
        return prev

    def frown(self, on, ctx):
        self._hold(-2, 5)
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
                self.deliver(); return                    # still growing
            b = t - 1; self.finalized = b
            self.on_token("".join(self.its.get(k, "") for k in range(a, b + 1)), a, b)
        self.finalized = max(self.finalized, end - 1)
        self.deliver()

    def lines(self):
        return [c + a for c, ans in ANSWERS.items() for a in ans]   # the lines this parent knows (the teacher reads the corpus)

    def on_token(self, tok, a, b):
        wall = self.tobs.get(b, time.time()); ctx = self.ctx(a, b)
        low = tok if tok == "I" else (tok if tok.islower() else "")   # a word is the word as written: "oN" is not "on"
        if len(tok) >= 3 and len(set(tok.lower())) == 1:
            L = tok[0].lower()
            if tok[0].isalpha():
                self.letter_runs[L] = self.letter_runs.get(L, 0) + 1
                if self.letter_runs[L] >= 3 and L not in self.expanded and L in EXPAND:
                    self.expanded.add(L); self.pending_expand = EXPAND[L]
            elif tok[0] != " " and time.time() - self.last_frown > self.s(240):
                self.frown(tok, ctx); return
        if self.parent and time.time() < self.away_until:
            return                                                # the parent is turned away
        if self.parent and (self.reply or TALKOVER_FROWN):
            ts, te = self.typing_span
            if a < te and b >= ts:                                # said over the parent's own turn
                self.e = max(0.0, self.e - 0.04); self.row({"action": "missed", "on": tok, "why": "talked over", "e": round(self.e, 3), "context": ctx})
                if TALKOVER_FROWN and time.time() - self.last_frown > self.s(60):   # THE PARENT'S FACE WHEN INTERRUPTED (the user's word, 2026-09-06):
                    self._hold(-1, 2.5)                                             # a light, brief frown, at most every 60 ticks
                    self.frowns += 1; self.last_frown = time.time(); self.row({"action": "frown", "on": tok, "why": "talked over", "context": ctx})
                return
        c = self.cue
        if c and wall <= c["until"] and not c["done"] and len(tok) >= 2 and low:
            if low in c["full"]:
                c["done"] = True; self.answered = True; self.reply_tokens.append(low)
                self.e = min(1.0, self.e + 0.25)
                if self.wait > 0:
                    self.hold_reply(tok, ctx, "cue completion: " + c["text"], c["text"], low, b); return
                self.smile(tok, ctx, "cue completion: " + c["text"]); return
            if low[:2] in [x[:2] for x in c["full"]] and time.time() - wall < self.s(14):
                c["done"] = True; self.answered = True; self.reply_tokens.append(low)
                self.e = min(1.0, self.e + 0.15)
                if self.wait > 0:
                    full = [x for x in c["full"] if x[:2] == low[:2]][0]
                    self.hold_reply(tok, ctx, "cue prefix: " + c["text"], c["text"], full, b); return
                self.smile(tok, ctx, "cue prefix: " + c["text"]); return
        if self.parent and self.reply and self.answered:
            # past its answer: the parent asked and got a speech (the words may go on completing the cued line)
            said = ((self.reply_cue or "") + " ".join(self.reply_tokens + [tok])).strip()
            if self.past or not any(l.startswith(said) for l in self.lines()):
                self.past = True; self.e = max(0.0, self.e - 0.04)
                self.row({"action": "missed", "on": tok, "why": "past its answer", "e": round(self.e, 3), "context": ctx}); return
            self.reply_tokens.append(tok)
        if low in KNOWN2:
            age = time.time() - wall
            if self.last_word == low and time.time() - self.last_smile < self.s(48):
                self.row({"action": "withheld", "on": tok, "why": "same word twice", "context": ctx})
            elif age <= self.s(13) and time.time() - self.last_smile >= self.s(5):   # a smile may follow as soon as the face has returned (the hold is 5 ticks; 8 withheld 50-90 a day)
                if self.parent:
                    n = self.word_count.get(low, 0); self.word_count[low] = n + 1
                    p = self.e * (0.95 ** max(0, n - 5))         # attention, and the fiftieth "dog"
                    self.e = min(1.0, self.e + (0.1 if n == 0 else 0.05))
                    if self.rng.random() > p:
                        self.row({"action": "missed", "on": tok, "why": "distracted", "e": round(self.e, 3), "context": ctx}); return
                    self.expand_next = low
                self.smile(tok, ctx, "known word")
            else:
                self.row({"action": "missed", "on": tok, "why": "late %.1fs" % age, "context": ctx})
        elif self.parent and low and len(low) >= 3 and low not in KNOWN2 and not any(w.startswith(low) for w in KNOWN2):
            self.e = max(0.0, self.e - 0.04)                      # babble wears the parent's attention

    def watch(self, seconds):
        t0 = time.time()
        while time.time() - t0 < seconds:
            d = self.poll()
            if d is not None:
                self.scan()
            time.sleep(max(0.05, self.s(1)))

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
            if self.quiet_since() >= self.quiet_ticks or time.time() - t0 >= self.cap:
                return time.time() - t0
            time.sleep(max(0.05, self.s(1)))

    def event(self, text, kind):
        gw = self.gate()
        if gw < 0:
            return False
        last_before = self.state.get("last") or {}
        if kind == "cue":
            self.cue = {"text": text, "until": time.time() + self.s(360), "full": ANSWERS.get(text, []), "done": False}
        self.reply_cue = text if kind == "cue" else None; self.reply_tokens = []; self.answered = False; self.past = False
        self.typing_span = (self.maxtick + 1, 10 ** 9)     # the parent's turn: from its first symbol to its last
        self.req("/type", {"text": text}); t_start = time.time()
        while True:
            d = self.poll(); self.scan()
            if d is not None and d.get("queued", 0) == 0:
                break
            time.sleep(max(0.05, self.s(1)))
        tick_end = self.maxtick                            # the page's own tick count (survives a serve restart)
        self.typing_span = (self.typing_span[0], tick_end)
        self.watch(self.listen)
        its = "".join((self.its.get(t) or "_") for t in range(tick_end + 1, tick_end + 26) if self.its.get(t) is not None)
        la = self.state.get("last") or {}
        self.row({"action": kind, "text": text, "gate_wait_s": round(gw, 1), "its_after": its, "ts": iso(t_start),
                  "fatigue": la.get("fatigue"), "stress": la.get("stress"), "mood": la.get("mood"), "gate": la.get("gate"),
                  "own": la.get("own"), "doses": la.get("doses"), "smiles_so_far": self.smiles,
                  "sleep_pressure": self.state.get("sleep_pressure"), "store": self.state.get("store")})
        if kind == "cue":
            self._cue_clear_at = time.time() + self.s(12)
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
            if self.parent and self.expand_next and kind == "line":
                holds = [l for l in lines if self.expand_next in l.split()]
                if holds:
                    text = self.rng.choice(holds)                  # answering its word with a line that holds it
                self.expand_next = None
            if prev is not None:
                prev = self.pace_wait(prev, self.pace())
                if prev is None:
                    slept = True; break
            while self.parent and time.time() < self.away_until:
                self.watch(2.0)
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
        self.reply_pending = None; self.reply_line = None    # a reply due at the night's edge is not given
        # the post-night cues, a known line between each pair
        post = list(cues); lines2 = self.rng.sample(lines, len(lines))
        prev = None; j = 0
        for i, c in enumerate(post):
            for text, kind in ([(c, "cue")] + ([(lines2[j % len(lines2)], "line")] if i % 2 == 1 else [])):
                if prev is not None:
                    prev = self.pace_wait(prev, self.period) or time.time()
                prev = time.time(); self.event(text, kind)
                if kind == "line":
                    j += 1
        try:
            self.row({"action": "save", "result": self.req("/save", {})})
        except Exception as e:
            self.row({"action": "save", "error": repr(e)[:120]})
        self.watch(120)
        self.row({"action": "session_end", "smiles": self.smiles, "frowns": self.frowns, "aways": self.aways, "e": round(self.e, 3)})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8018)
    ap.add_argument("--day", type=int, default=1)
    ap.add_argument("--log", default="data/body2_caregiver.jsonl")
    ap.add_argument("--period", type=float, default=480)    # ticks between the parent's lines
    ap.add_argument("--quiet", type=float, default=24)      # ticks of the child's quiet before the parent speaks
    ap.add_argument("--tick", type=float, default=0.25); ap.add_argument("--listen", type=float, default=50)
    ap.add_argument("--cap", type=float, default=90.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--answer-levels", type=int, default=1)
    ap.add_argument("--parent", type=int, default=0); ap.add_argument("--reply", type=int, default=0)
    ap.add_argument("--wait", type=int, default=0)            # the reply withheld: ticks of the child's quiet before the parent answers (4)
    ap.add_argument("--lines", default="dog will go|I will go up|you will go in|scared dog|scared ball|what? scared dog|give milk|give ball|give book|ball under|ball on|where ball? ball under|I had milk|you had ball|dog had ball|first milk then ball|first up then in|big dog bigger dog|bigger dog up|I saw dog|you saw dog|why dog up? because big dog")
    ap.add_argument("--cues", default="dog will |scared |give |where ball? |I had |first milk then |big dog bigger |why dog up? ")
    a = ap.parse_args()
    cg = Caregiver("http://localhost:%d" % a.port, a.day, a.log, period=a.period, quiet=a.quiet, cap=a.cap, seed=a.seed, answer_levels=a.answer_levels, parent=a.parent, reply=a.reply, wait=a.wait, tick=a.tick, listen=a.listen)
    cg.run_day([x for x in a.lines.split("|") if x], [x for x in a.cues.split("|") if x])


if __name__ == "__main__":
    main()
