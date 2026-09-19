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
FROWN_GAP = int(os.environ.get("FROWN_GAP", "60"))
TURN_ONLY_SMILE = int(os.environ.get("TURN_ONLY_SMILE", "0"))   # THE SMILE IN ITS TURN ONLY (2026-09-19): a known word said into the silence past the child's turn earns nothing
HABIT_TICKS = int(os.environ.get("HABIT_TICKS", "0"))       # THE HABITUATION THE CHILD CAN SEE (2026-09-12): 0 = the day-count rule (0.95^(n-5): the fiftieth "dog" gets
                                                            # one smile in ten, a count the body cannot see, so its critic never learns it and its stress stayed at 20 of 30
                                                            # all day on withheld smiles); N > 0 = a word smiled at within the last N ticks earns nothing, and smiles again after            # the least ticks between two talk-over frowns (2026-09-12: 60 gave seventy frowns a day under the chunk and a body at stress 20 all day, its gate flattened threefold; a parent frowns, then gives it a minute: 240)
import random
import time
import urllib.request

KNOWN = set("""hi bye no more up dog ball milk book go big my please the you two in on going
want where not give little under one three went what then can happy will sad why because
first bigger had scared saw balls dogs books all gone""".split())
KNOWN2 = {w for w in KNOWN if len(w) >= 2}
# THE SMILE FOR THE ANSWER (ANSWER_SMILE, 2026-09-13): a parent rewards relevance, not vocabulary. With a smile at any known word (two a
# line, three hundred a day) the reward did not depend on which word the child said, and the striatal actor, which learns from the
# dopamine error over which symbol, sat still for days (slope 0.04) while the gate, whose reward depends on when, learned turn-taking.
# With it on, the child's word during its turn after the parent's line earns the full smile when it is what the other voice is about to
# answer (a content word of the coming B line, or yes/no when B begins so), and any other known word a faint one.
ANSWER_SMILE = int(os.environ.get("ANSWER_SMILE", "0"))
STOP = {"the", "a", "an", "is", "it", "in", "on", "at", "to", "of", "with", "my", "your", "you", "me", "i", "and", "here", "there", "now",
        "we", "are", "am", "do", "can", "too", "for", "up", "out", "so", "this", "that", "not", "no", "yes", "please"}
def content_words(line):
    return [w for w in line.replace("?", "").replace(".", "").replace("!", "").lower().split() if w not in STOP and len(w) >= 2]
ANSWERS = {"dog will ": ["go"], "scared ": ["dog", "ball"], "give ": ["milk", "ball", "book"], "where ball? ": ["ball"],
           "I had ": ["milk"], "first milk then ": ["ball"], "big dog bigger ": ["dog"], "why dog up? ": ["because"],
           "what? ": ["dog", "ball", "milk", "sad", "scared"], "I can ": ["go"], "sad ": ["dog", "ball"], "happy ": ["dog", "ball"]}
EXPAND = {"b": "big ball", "d": "big dog", "m": "more milk", "g": "go up", "l": "little dog", "u": "up", "w": "what? dog",
          "h": "happy dog", "p": "please", "n": "no more milk", "s": "sad dog", "c": "dog can go", "o": "one ball", "a": "all gone",
          "i": "I go up", "f": "first milk then ball", "t": "the dog", "y": "you go up", "e": "the dog", "r": "more milk", "k": "milk"}


def iso(t=None):
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(t or time.time()))


class Caregiver:
    def __init__(self, base, day, log, period=480, quiet=24, cap=360, seed=0, answer_levels=1, parent=0, reply=0, wait=0.0, tick=0.25, listen=50, yield_ticks=240):
        self.base, self.day, self.log = base, day, log
        # THE PARENT YIELDS TO A VISITOR (2026-09-11, the user's plan: a page to talk to it while its life goes on): the page
        # marks who typed each symbol; for yield_ticks after a symbol typed by anyone but this parent, the parent types nothing,
        # scores nothing and makes no face; the visitor is the parent then. Decided from the page alone.
        self.yield_ticks = int(yield_ticks); self.visitor_tick = -10 ** 9; self.yielding_now = False
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
        self.parent = int(parent); self.e = 0.7; self.word_count = {}; self.away_until = 0.0; self.aways = 0   # the attention's rest 0.7 (2026-09-14: at 0.5 it discarded 45% of the child's words as "distracted" and a sad day spiralled)
        self.word_smiled_tick = {}                        # the page tick of the last smile at each word (HABIT_TICKS)
        # THE PARENT WANTS A REPLY (the user's word of 2026-09-04): once its cue is answered, each further word the
        # child adds before the parent's next turn wears its attention and gets no smile, unless the words go on
        # completing the cued line; a word said over the parent's own typing does the same. The answer smile stands.
        self.reply = int(reply); self.reply_cue = None; self.reply_tokens = []; self.answered = False; self.past = False
        self.typing_span = (-1, -1)                       # the parent's own turn, in ticks
        self.expand_next = None; self._e_t = time.time()
        self.expect = None                                   # the answer the other voice is about to give (ANSWER_SMILE)
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
        """the parent's rhythm: the period stretched by inattention (1.6 - e); and now and then a person's pause (2026-09-18, the
        supervisor's sitting: the child's mood fell three points in six minutes under a partner at a keyboard's pace, its critic
        having learned the typist's even beat). One gap in five is two to four periods long, so the critic learns a slow partner too."""
        base = self.period * (1.6 - self.e) if self.parent else self.period
        if self.parent and self.rng.random() < 0.2:
            return base * self.rng.uniform(2.0, 4.0)
        return base

    def _attend(self):
        now = time.time(); dt = now - self._e_t; self._e_t = now
        self.e += (0.7 - self.e) * min(1.0, dt / self.s(600))   # attention drifts toward its resting level (0.7 since 2026-09-14) in silence (600 ticks); 0.3 -> 0.5 on 2026-09-11: at 0.3 the parent withheld 150-200 known-word smiles a day as "distracted"
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
        if n < self.cursor:                               # the serve restarted: the page begins again, and so must the reading of it
            self.cursor = 0; d = self.req("/state?since=0"); n = d.get("n", 0)
            self.its = {}; self.tobs = {}; self.maxtick = -1; self.finalized = -1   # (2026-09-12: a typist that outlived a restart kept its
            self.typing_span = (-1, -1); self.visitor_tick = -10 ** 9              # old tick count and scanned nothing for half an hour)
        t = time.time()
        for i, e in enumerate(d.get("page", [])):
            idx = self.cursor + i; tick = idx // 2
            if idx % 2 == 1:
                self.its[tick] = e[0]; self.tobs[tick] = t; self.maxtick = max(self.maxtick, tick)
            elif e[0] and len(e) >= 5 and e[4] and e[4] not in ("parent", "other"):   # a visitor's symbol; the other voice is the typist's own
                self.visitor_tick = max(self.visitor_tick, tick)     # a visitor's symbol on the page
        self.cursor = n; self.state = d
        y = self.yielding()
        if y != self.yielding_now:
            self.yielding_now = y
            if y:
                self.cue = None; self.reply_pending = None; self.reply_line = None; self.expand_next = None; self.pending_expand = None
            self.row({"action": "yield" if y else "resume", "visitor_tick": self.visitor_tick, "tick": self.maxtick})
        return d

    def yielding(self):
        """a visitor typed within yield_ticks: the parent steps back"""
        return self.yield_ticks > 0 and self.maxtick - self.visitor_tick < self.yield_ticks

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

    def smile(self, on, ctx, why, extra=None, faint=False):
        db = (self.state.get("last") or {}).get("doses")
        # THE ANSWER'S SMILE IS THE BIGGER ONE (ANSWER_SMILE, 2026-09-13): the ordinary smile at a known word keeps its old strength and
        # the answer earns the growing smile the cues earn (2, then 4). Thinning the ordinary smile instead (a quarter, then a half)
        # halved the day's reward and the gate acted less each day (duty 0.42 -> 0.35 -> 0.29): reward must depend on the word by
        # contrast, not by starvation
        levels = self.answer_levels if why.startswith("cue") else (2 if why.startswith("answer") else 1)
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
        if self.yielding():                                   # the visitor is the parent now: nothing scored, no face
            self.finalized = max(self.finalized, end - 1); return
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
        if self.parent and (self.reply or TALKOVER_FROWN) and len(tok) >= 2:
            # A LETTER IS NOT TALKING (2026-09-12, days 118-131 read): of the tokens said over the parent's typing, 103 of 111 a day were
            # single letters, the mouth shadowing the parent's own letters as they arrived ('n' as "tin" was typed), and the frown fell
            # on them 72 times a day while the child already yielded five to one (the ear ratio 0.18). A parent frowns at a word said
            # over it, not at a murmured letter; the same two-letter bound the smile has.
            ts, te = self.typing_span
            if a < te and b >= ts:                                # said over the parent's own turn
                self.over_count = getattr(self, "over_count", 0) + 1   # the words said over this line (reported in the line's row, 2026-09-19)
                self.e = max(0.0, self.e - 0.04); self.row({"action": "missed", "on": tok, "why": "talked over", "e": round(self.e, 3), "context": ctx})
                if TALKOVER_FROWN and time.time() - self.last_frown > self.s(FROWN_GAP):   # THE PARENT'S FACE WHEN INTERRUPTED (the user's word, 2026-09-06):
                    self._hold(-1, 2.5)                                             # a light, brief frown, at most every FROWN_GAP ticks
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
        ex = self.expect
        if ANSWER_SMILE and self.parent and ex and not ex.get("done") and low and len(low) >= 2 and wall <= ex["until"]:
            w_ = low.strip(".!?")                                # "yes." and "yes!" are yes (days 184-185: 13 such answers, 2 rewarded)
            said_yes = w_.startswith("yes")                      # and "yesee", "yesplease": its yes fuses into the next sound (days 190-192)
            if w_ in ex["words"] or (ex["yesno"] and (said_yes or w_ == "no")):
                ex["done"] = True; self.e = min(1.0, self.e + 0.2); self.word_smiled_tick[low] = b
                self.smile(tok, ctx, "answer: " + ex["line"]); return
        if low in KNOWN2:
            age = time.time() - wall
            ts_, te_ = self.typing_span
            if TURN_ONLY_SMILE and self.parent and te_ < 10 ** 8 and b > te_ + 2 * int(round(self.listen / max(1e-6, self.tick))):
                # THE SMILE IN ITS TURN ONLY (TURN_ONLY_SMILE, 2026-09-19, the user's word on the babble: alone at the wake it recited its
                # evening in a loop until spoken to, and the faint smile at any known word had been paying that chatter): a known word
                # said into the silence past the child's turn after a line (twice the listen window) earns no smile and no expansion,
                # a parent not answering chatter. The turn after a line is rewarded as before. Nothing in the body changes.
                self.row({"action": "missed", "on": tok, "why": "in silence", "context": ctx}); return
            if self.last_word == low and time.time() - self.last_smile < self.s(48):
                self.row({"action": "withheld", "on": tok, "why": "same word twice", "context": ctx})
            elif age <= self.s(13) and time.time() - self.last_smile >= self.s(5):   # a smile may follow as soon as the face has returned (the hold is 5 ticks; 8 withheld 50-90 a day)
                if self.parent:
                    n = self.word_count.get(low, 0); self.word_count[low] = n + 1
                    if HABIT_TICKS > 0:
                        last_t = self.word_smiled_tick.get(low, -10 ** 9)
                        if b - last_t < HABIT_TICKS:               # said and smiled at lately: nothing this time, a smile again later
                            self.row({"action": "missed", "on": tok, "why": "said lately", "e": round(self.e, 3), "context": ctx}); return
                        p = self.e
                    else:
                        p = self.e * (0.95 ** max(0, n - 5))     # attention, and the fiftieth "dog"
                    self.e = min(1.0, self.e + (0.1 if n == 0 else 0.05))
                    if self.rng.random() > p:
                        self.row({"action": "missed", "on": tok, "why": "distracted", "e": round(self.e, 3), "context": ctx}); return
                    self.expand_next = low; self.word_smiled_tick[low] = b
                self.smile(tok, ctx, "known word", faint=bool(ANSWER_SMILE and self.parent))
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
            if self.yielding():
                t0 = time.time()                              # the cap does not run against a visitor
            elif self.quiet_since() >= self.quiet_ticks or time.time() - t0 >= self.cap:
                return time.time() - t0
            time.sleep(max(0.05, self.s(1)))

    def event(self, text, kind):
        gw = self.gate()
        if gw < 0:
            return False
        if kind == "cue":
            self.cue = {"text": text, "until": time.time() + self.s(360), "full": ANSWERS.get(text, []), "done": False}
        self.reply_cue = text if kind == "cue" else None; self.reply_tokens = []; self.answered = False; self.past = False
        self.typing_span = (self.maxtick + 1, 10 ** 9)     # the parent's turn: from its first symbol to its last
        self.req("/type", {"text": text, "who": "parent"}); t_start = time.time()
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
