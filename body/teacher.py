"""the teaching caregiver (CURRICULUM.md): the same raw rules as body/caregiver.py, decided from the
page and its face row only, but the SPEECH comes from a planner: new lines and phrasings, a
vocabulary that grows by the curriculum. Two honest generalizations of the scripted rules:
a word is known (smiled at) once the teacher has typed it three times; a cue's answers are
the continuations of the lines the body has actually heard. Nothing here reads the body's
insides or writes its words.

Planners:
  --planner claude   the Anthropic SDK plans the next utterances (ANTHROPIC_API_KEY in the
                     environment of the shell that launches this; --budget calls per day)
  --planner queue    utterances are read from a JSONL queue file (a Claude Code subagent or a
                     person appends {"say": [...]} rows); the known lines fill any gap
  --planner fixed    the 22 lines and 8 cues of body/caregiver.py (a test of the mechanics)

  python3 -m body.teacher --port 8018 --day 7 --log data/body2_caregiver.jsonl --planner queue \
      --queue data/teach_queue.jsonl --days 3
"""
import argparse
import json
import os
import random
import re
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from body.caregiver import Caregiver, KNOWN, EXPAND, iso  # noqa: E402

LINES0 = ["dog will go", "I will go up", "you will go in", "scared dog", "scared ball", "what? scared dog", "give milk", "give ball",
          "give book", "ball under", "ball on", "where ball? ball under", "I had milk", "you had ball", "dog had ball",
          "first milk then ball", "first up then in", "big dog bigger dog", "bigger dog up", "I saw dog", "you saw dog",
          "why dog up? because big dog"]
CUES0 = ["dog will ", "scared ", "give ", "where ball? ", "I had ", "first milk then ", "big dog bigger ", "why dog up? "]
ALLOWED = re.compile(r"^[a-zA-Z ?.!]+$")
KNOWN_AFTER = 3          # a word is known once typed this many times (over all the teacher's days)
HEARD_FOR_CUE = 2        # a line's prefix can be a cue once the line was typed this many times


def words_of(text):
    return [w for w in re.split(r"[^a-zA-Z]+", text) if w]


class Corpus:
    """what the body has heard from its teachers: the lines, their counts, the words' counts.
    Persisted in a JSON file so the vocabulary grows across days and restarts."""

    def __init__(self, path):
        self.path = path; self.lines = {}; self.words = {}
        if path and os.path.exists(path):
            d = json.load(open(path)); self.lines = d.get("lines", {}); self.words = d.get("words", {})
        for line in LINES0:                                   # the raised lines count as heard
            if line not in self.lines:
                self.lines[line] = HEARD_FOR_CUE
                for w in words_of(line):
                    self.words[w] = self.words.get(w, 0) + KNOWN_AFTER

    def typed(self, text):
        t = text.strip()
        if not t:
            return
        self.lines[t] = self.lines.get(t, 0) + 1
        for w in words_of(t):
            self.words[w] = self.words.get(w, 0) + 1
        if self.path:
            tmp = self.path + ".tmp"
            json.dump({"lines": self.lines, "words": self.words}, open(tmp, "w")); os.replace(tmp, self.path)

    def known(self):
        return {w for w, n in self.words.items() if n >= KNOWN_AFTER and len(w) >= 2 and (w == "I" or w.islower())} | {w for w in KNOWN if len(w) >= 2}

    def answers(self, prefix):
        """the next words of every heard line that begins with the prefix (the prefix ends in a space)"""
        out = []
        for line, n in self.lines.items():
            if n >= HEARD_FOR_CUE and line.startswith(prefix) and len(line) > len(prefix):
                rest = words_of(line[len(prefix):])
                if rest and rest[0] not in out:
                    out.append(rest[0])
        return out

    def heard_lines(self, n_min=1):
        return [l for l, n in self.lines.items() if n >= n_min]


class Teacher(Caregiver):
    def __init__(self, base, day, log, corpus, planner, period=60.0, quiet=3.0, cap=45.0, seed=0):
        super().__init__(base, day, log, period=period, quiet=quiet, cap=cap, seed=seed)
        self.corpus, self.planner = corpus, planner
        self.said_today = []                                  # (text, kind, its_after)

    # the rules read the corpus, not the fixed tables
    def on_token(self, tok, a, b):
        import body.caregiver as C
        C.KNOWN2 = self.corpus.known()                        # the scripted rule, over the grown vocabulary
        return super().on_token(tok, a, b)

    def event(self, text, kind):
        if kind == "cue":
            ans = self.corpus.answers(text)
            if not ans:
                kind = "line"                                 # a prefix the body never heard continued is just a line
        gw = self.gate()
        if gw < 0:
            return False
        if kind == "cue":
            self.cue = {"text": text, "until": time.time() + 90, "full": ans, "done": False}
        self.req("/type", {"text": text}); t_start = time.time()
        self.corpus.typed(text if kind == "line" else text.strip())
        while True:
            d = self.poll(); self.scan()
            if d is not None and d.get("queued", 0) == 0:
                break
            time.sleep(0.4)
        tick_end = self.maxtick
        self.watch(12.4)
        its = "".join((self.its.get(t) or "_") for t in range(tick_end + 1, tick_end + 26) if self.its.get(t) is not None)
        la = self.state.get("last") or {}
        self.row({"action": kind, "text": text, "answers": (self.cue or {}).get("full") if kind == "cue" else None,
                  "gate_wait_s": round(gw, 1), "its_after": its, "ts": iso(t_start), "teacher": self.planner.name,
                  "fatigue": la.get("fatigue"), "stress": la.get("stress"), "mood": la.get("mood"), "gate": la.get("gate"),
                  "own": la.get("own"), "doses": la.get("doses"), "smiles_so_far": self.smiles,
                  "sleep_pressure": self.state.get("sleep_pressure"), "store": self.state.get("store")})
        self.said_today.append((text, kind, its.replace("_", "")))
        if kind == "cue":
            time.sleep(3); self.cue = None
        return True

    def recent(self, n=8):
        return self.said_today[-n:]

    def run_day(self):
        self.poll(); self.finalized = self.maxtick - 1; self.face(0)
        self.row({"action": "session_start", "teacher": self.planner.name, "sleep_pressure": self.state.get("sleep_pressure"),
                  "nights": self.state.get("nights"), "known": len(self.corpus.known()), "lines_heard": len(self.corpus.heard_lines())})
        slept = False; prev = None; n_events = 0
        while True:
            d = self.poll()
            if d is not None and d.get("asleep"):
                slept = True; break
            if self.pending_expand:
                text2 = self.pending_expand; self.pending_expand = None
                if not self.event(text2, "line"):
                    slept = True; break
            item = self.planner.next(self)
            if item is None:
                self.watch(3.0); continue
            text, kind = item
            if prev is not None:
                while time.time() < prev + self.period:
                    self.watch(min(3.0, prev + self.period - time.time()))
                    if (self.state or {}).get("asleep"):
                        break
            prev = time.time()
            if not self.event(text, kind):
                slept = True; break
            n_events += 1
            if (self.state or {}).get("sleep_pressure", 0) >= (self.state or {}).get("wake_ticks", 10 ** 9) - 300 and n_events > 4:
                self.event("bye", "line")
                while not (self.poll() or {}).get("asleep"):
                    self.watch(5.0)
                slept = True; break
        # the night
        t_sleep = time.time()
        while True:
            d = self.poll()
            if d is not None and not d.get("asleep"):
                break
            time.sleep(5)
        self.row({"action": "night", "slept_s": round(time.time() - t_sleep, 1), "last_night": (self.state or {}).get("last_night")})
        # the post-night cues: the eight raised cues (the fixed yardstick), a heard line between each pair
        lines2 = self.rng.sample(self.corpus.heard_lines(HEARD_FOR_CUE), min(4, len(self.corpus.heard_lines(HEARD_FOR_CUE))))
        prev = None; j = 0
        for i, c in enumerate(CUES0):
            for text, kind in ([(c, "cue")] + ([(lines2[j % len(lines2)], "line")] if i % 2 == 1 and lines2 else [])):
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
        self.watch(60)
        self.row({"action": "session_end", "teacher": self.planner.name, "smiles": self.smiles, "frowns": self.frowns,
                  "calls": getattr(self.planner, "calls", 0), "known": len(self.corpus.known()), "lines_heard": len(self.corpus.heard_lines())})


# ---------------- planners ----------------
def clean(s):
    s = re.sub(r"\s+", " ", str(s)).strip("\n")
    return s if ALLOWED.match(s) and 1 <= len(s.strip()) <= 40 else None


class FixedPlanner:
    name = "fixed"

    def __init__(self, rng):
        self.rng = rng; self.plan = []

    def next(self, teacher):
        if not self.plan:
            order = self.rng.sample(LINES0, len(LINES0)); ci = 0
            for i, line in enumerate(order):
                self.plan.append((line, "line"))
                if i % 2 == 1 and ci < len(CUES0):
                    self.plan.append((CUES0[ci], "cue")); ci += 1
        return self.plan.pop(0)


class QueuePlanner:
    """rows {"say": ["dog will go", "dog will "]} appended to a JSONL file by whoever teaches; an
    utterance ending in a space is a cue. Empty queue: a heard line (the day goes on)."""
    name = "queue"

    def __init__(self, path, rng):
        self.path, self.rng = path, rng; self.pos = 0; self.buf = []; self.calls = 0

    def next(self, teacher):
        if os.path.exists(self.path):
            with open(self.path) as f:
                f.seek(self.pos); new = f.read(); self.pos = f.tell()
            for line in new.splitlines():
                try:
                    for s in json.loads(line).get("say", []):
                        c = clean(s)
                        if c:
                            self.buf.append(c); self.calls += 1
                except Exception:
                    pass
        if self.buf:
            s = self.buf.pop(0)
            return (s, "cue" if s.endswith(" ") else "line")
        heard = teacher.corpus.heard_lines(HEARD_FOR_CUE) or LINES0
        return (self.rng.choice(heard), "line")


SYSTEM = """You are the teacher of a small language organism that lives on a shared page: it sees one
character per tick of what you type, and answers one character per tick. It has no eyes, no world,
only the page and your face (a smile after a word it says well, decided by fixed rules, not by you).
You raise it as a parent raises a child: short lines, few words, much repetition, one new thing at a
time. Everything you say is lowercase letters, spaces, '?' and '.' only (the word 'I' may be capital).
Rules of the curriculum:
- lines of 2 to 5 words, from the KNOWN words; at most ONE new word per day, and a new word must be
  said in at least three different short lines before the day ends;
- vary the phrasings around the same words (that is how it generalizes): 'dog will go', 'big dog will
  go', 'dog will go up', 'you will go', 'where dog? dog under';
- a CUE is a prefix of a line it has heard at least twice, ending with a space: 'dog will ' — then
  wait; NEVER type the answer right after a cue; the next utterance after a cue must be a full line;
- about one cue in four utterances; repeat a new line at least three times across the day;
- answer what it says: if it said a known word, use that word in your next line;
- never mention the page, rules, yourself, or anything but the small world of dog, ball, milk, book."""


class ClaudePlanner:
    """the Anthropic SDK plans batches of utterances; one call per batch; a daily call budget"""
    name = "claude"

    def __init__(self, model, budget, rng, batch=6):
        import anthropic
        self.client = anthropic.Anthropic()
        self.model, self.budget, self.rng, self.batch = model, budget, rng, batch
        self.calls = 0; self.buf = []; self.new_word_today = None

    def next(self, teacher):
        if not self.buf and self.calls < self.budget:
            self.buf = self.plan(teacher)
        if self.buf:
            s = self.buf.pop(0)
            return (s, "cue" if s.endswith(" ") else "line")
        heard = teacher.corpus.heard_lines(HEARD_FOR_CUE) or LINES0
        return (self.rng.choice(heard), "line")

    def plan(self, teacher):
        known = sorted(teacher.corpus.known())
        recent = "\n".join(f"you: {t!r} ({k}) -> it: {a!r}" for t, k, a in teacher.recent(10)) or "(the day begins)"
        heard = sorted(teacher.corpus.heard_lines(HEARD_FOR_CUE))[-40:]
        user = (f"KNOWN words: {' '.join(known)}\nLines it has heard at least twice (the cues may be their prefixes):\n"
                + "\n".join(heard) + f"\n\nThe last exchanges:\n{recent}\n\nSmiles so far today: {teacher.smiles}. "
                f"Day {teacher.day}. Give the next {self.batch} utterances as a JSON list of strings, nothing else.")
        self.calls += 1
        try:
            msg = self.client.messages.create(model=self.model, max_tokens=300, system=SYSTEM,
                                              messages=[{"role": "user", "content": user}])
            text = "".join(b.text for b in msg.content if getattr(b, "type", "") == "text")
            m = re.search(r"\[.*\]", text, re.S)
            items = json.loads(m.group(0)) if m else []
            out = [c for c in (clean(s) for s in items) if c]
            teacher.row({"action": "plan", "calls": self.calls, "say": out, "usage": {"in": msg.usage.input_tokens, "out": msg.usage.output_tokens}})
            return out
        except Exception as e:
            teacher.row({"action": "plan_error", "calls": self.calls, "err": repr(e)[:160]})
            return []


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8018)
    ap.add_argument("--day", type=int, default=1); ap.add_argument("--days", type=int, default=1)
    ap.add_argument("--log", default="data/body2_caregiver.jsonl")
    ap.add_argument("--corpus", default="data/body2_corpus.json")
    ap.add_argument("--planner", choices=["claude", "queue", "fixed"], default="queue")
    ap.add_argument("--queue", default="data/teach_queue.jsonl")
    ap.add_argument("--model", default="claude-sonnet-5"); ap.add_argument("--budget", type=int, default=120)
    ap.add_argument("--period", type=float, default=60.0); ap.add_argument("--quiet", type=float, default=3.0)
    ap.add_argument("--cap", type=float, default=45.0); ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    for k in range(a.days):
        day = a.day + k
        rng = random.Random(a.seed + day)
        planner = {"claude": lambda: ClaudePlanner(a.model, a.budget, rng), "queue": lambda: QueuePlanner(a.queue, rng),
                   "fixed": lambda: FixedPlanner(rng)}[a.planner]()
        corpus = Corpus(a.corpus)
        t = Teacher("http://localhost:%d" % a.port, day, a.log, corpus, planner, period=a.period, quiet=a.quiet, cap=a.cap, seed=day)
        t.run_day()


if __name__ == "__main__":
    main()
