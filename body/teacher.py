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
        # only what a parent taught (the review of 2026-09-10: the first lineage's hard-coded list had 10 untaught words smiled at 62 times)
        return {w for w, n in self.words.items() if n >= KNOWN_AFTER and len(w) >= 2 and (w == "I" or w.islower())}

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
    def __init__(self, base, day, log, corpus, planner, period=160, quiet=12, cap=48, seed=0, answer_levels=1, parent=0, reply=0, wait=0, tick=0.25, listen=50, yield_ticks=240):
        super().__init__(base, day, log, period=period, quiet=quiet, cap=cap, seed=seed, answer_levels=answer_levels, parent=parent, reply=reply, wait=wait, tick=tick, listen=listen, yield_ticks=yield_ticks)
        self.corpus, self.planner = corpus, planner
        self.said_today = []                                  # (text, kind, its_after)

    # the rules read the corpus, not the fixed tables
    def on_token(self, tok, a, b):
        import body.caregiver as C
        C.KNOWN2 = self.corpus.known()                        # the scripted rule, over the grown vocabulary
        return super().on_token(tok, a, b)

    def lines(self):
        return self.corpus.heard_lines()                      # what continues a cue: the lines it has heard

    def reply_line_for(self, cue, answer):
        # the recast: the heard line (twice at least) that begins with the cue and goes on with its answer. THE MINIMAL RECAST
        # (2026-09-05): the shortest such line, the most heard among equals; the most heard alone was 'dog will go down' for
        # every 'dog will' -> 'go', the typist feeding the stutter's frame that its teachers avoided. A parent recasts the
        # child's words with the least added.
        lines = [(len(l), -n, l) for l, n in self.corpus.lines.items() if n >= HEARD_FOR_CUE and l.startswith(cue) and (l[len(cue):].split() or [""])[0] == answer]
        return min(lines)[2] if lines else (cue + answer).strip()

    def event(self, text, kind):
        if kind == "cue":
            ans = self.corpus.answers(text)
            if not ans:
                kind = "line"                                 # a prefix the body never heard continued is just a line
        gw = self.gate()
        if gw < 0:
            return False
        if kind == "cue":
            self.cue = {"text": text, "until": time.time() + self.s(360), "full": ans, "done": False}
        self.reply_cue = text if kind == "cue" else None; self.reply_tokens = []; self.answered = False; self.past = False
        self.typing_span = (self.maxtick + 1, 10 ** 9)         # the parent's turn: from its first symbol to its last
        self.req("/type", {"text": text, "who": "parent"}); t_start = time.time()
        self.corpus.typed(text if kind == "line" else text.strip())
        while True:
            d = self.poll(); self.scan()
            if d is not None and d.get("queued", 0) == 0:
                break
            time.sleep(max(0.05, self.s(1)))
        tick_end = self.maxtick
        self.typing_span = (self.typing_span[0], tick_end)
        self.watch(self.listen)
        its = "".join((self.its.get(t) or "_") for t in range(tick_end + 1, tick_end + 26) if self.its.get(t) is not None)
        la = self.state.get("last") or {}
        self.row({"action": kind, "text": text, "answers": (self.cue or {}).get("full") if kind == "cue" else None,
                  "gate_wait_s": round(gw, 1), "its_after": its, "ts": iso(t_start), "teacher": self.planner.name,
                  "fatigue": la.get("fatigue"), "stress": la.get("stress"), "mood": la.get("mood"), "gate": la.get("gate"),
                  "own": la.get("own"), "doses": la.get("doses"), "smiles_so_far": self.smiles,
                  "sleep_pressure": self.state.get("sleep_pressure"), "store": self.state.get("store")})
        self.said_today.append((text, kind, its.replace("_", "")))
        if kind == "cue":
            self._cue_clear_at = time.time() + self.s(12)   # the cue closes 12 ticks after the listening, without a blind sleep
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
                self.watch(self.s(12)); continue
            text, kind = item
            if self.parent and self.expand_next and kind == "line" and not getattr(self.planner, "planned", False):
                # answering its word with a line that holds it: only in place of the typist's own filler. Replacing
                # the planner's lines too (2026-09-04 and before), with a hundred smiled words a day, it ate nearly
                # every planned phrasing and word: the corpus grew by one word in thirty days
                holds = [l for l in self.corpus.heard_lines(HEARD_FOR_CUE) if self.expand_next in l.split()]
                if holds:
                    text = self.rng.choice(holds)
            self.expand_next = None
            if prev is not None:
                prev = self.pace_wait(prev, self.pace())
                if prev is None:
                    slept = True; break
            while self.parent and time.time() < self.away_until and not (self.state or {}).get("asleep"):
                self.watch(self.s(8))
            prev = time.time()
            if not self.event(text, kind):
                slept = True; break
            n_events += 1
            if False:   # (review 2026-09-06) the parent used the body's sleep pressure to say bye before the night; a parent sees sleep, not adenosine
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
        self.reply_pending = None; self.reply_line = None    # a reply due at the night's edge is not given
        # the post-night cues: the day's own cues, the eight most recent distinct (2026-09-11: the eight raised cues of the first lineage
        # were answered one time in five after a hundred days; the window after a night is where cues land best), a heard line between
        # each pair; the raised cues only when the day asked fewer than four
        lines2 = self.rng.sample(self.corpus.heard_lines(HEARD_FOR_CUE), min(4, len(self.corpus.heard_lines(HEARD_FOR_CUE))))
        day_cues = list(dict.fromkeys(t for t, k, _ in reversed(self.said_today) if k == "cue"))[:8][::-1]
        battery = day_cues if len(day_cues) >= 4 else CUES0
        prev = None; j = 0
        for i, c in enumerate(battery):
            for text, kind in ([(c, "cue")] + ([(lines2[j % len(lines2)], "line")] if i % 2 == 1 and lines2 else [])):
                if prev is not None:
                    prev = self.pace_wait(prev, self.period) or time.time()
                prev = time.time(); self.event(text, kind)
                if kind == "line":
                    j += 1
        try:
            self.row({"action": "save", "result": self.req("/save", {})})
        except Exception as e:
            self.row({"action": "save", "error": repr(e)[:120]})
        self.watch(60)
        self.row({"action": "session_end", "teacher": self.planner.name, "smiles": self.smiles, "frowns": self.frowns, "aways": self.aways, "e": round(self.e, 3),
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

    def __init__(self, path, rng, pos=None, buf=None):
        self.path, self.rng = path, rng; self.buf = list(buf or []); self.calls = 0
        # a new day's planner starts where the last one stopped (2026-09-11): the parent's unread lines used to be dropped at every
        # boundary; with no last one, at the file's end (yesterday's rows were yesterday's speech)
        # THE POSITION AND THE UNSAID LINES SURVIVE A RELAUNCH (2026-09-11, the parent of days 119-121: a fresh typist began at the
        # file's end and the unsaid lines of the boundary were lost): after each utterance the position read to and the lines read
        # but not yet said are written beside the queue, and a new typist takes them from there; the file's end only when nothing
        # was ever written.
        self.pos_path = path + ".pos"
        if pos is not None:
            self.pos = int(pos)
        else:
            try:
                st = json.load(open(self.pos_path))
                self.pos = int(st["pos"]); self.buf = [str(x) for x in st.get("buf", [])]
                if os.path.exists(path) and self.pos > os.path.getsize(path):
                    self.pos = os.path.getsize(path); self.buf = []   # the queue was replaced by a shorter file
            except Exception:
                self.pos = os.path.getsize(path) if os.path.exists(path) else 0
        # THE FILLER IS RECENT SPEECH (2026-09-05): when the queue is empty the typist used to say a random heard line,
        # and the most-heard lines are the oldest frames ('ball go down', 'dog go down' opened day 41 while the teachers
        # avoided them for the stutter). A parent with nothing new to say repeats what was said lately: the filler is
        # one of the last two dozen planned lines (yesterday's tail before today's rows arrive), a heard line only when
        # nothing was ever planned.
        self.recent = []
        if os.path.exists(path):
            try:
                with open(path) as f:
                    tail = f.readlines()[-12:]
                for line in tail:
                    for s in json.loads(line).get("say", []):
                        c = clean(s).strip()
                        if c:
                            self.recent.append(c)
            except Exception:
                pass

    def _remember(self):
        try:
            tmp = self.pos_path + ".tmp"
            with open(tmp, "w") as f:
                json.dump({"pos": self.pos, "buf": self.buf}, f)
            os.replace(tmp, self.pos_path)
        except Exception:
            pass

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
            s = self.buf.pop(0); self.planned = True
            self._remember()
            if s.strip():
                self.recent.append(s.strip()); self.recent = self.recent[-24:]
            return (s, "cue" if s.endswith(" ") else "line")
        self._remember()
        self.planned = False                                  # the typist's own filler
        if self.recent:
            return (self.rng.choice(self.recent), "line")     # a recent planned line, said as a line (a cue's prefix too)
        heard = teacher.corpus.heard_lines(HEARD_FOR_CUE) or LINES0
        return (self.rng.choice(heard), "line")               # a heard line only when nothing was ever planned


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
        self.recent = []                                     # the filler's memory (the review: it was never set here)
        import anthropic
        self.client = anthropic.Anthropic()
        self.model, self.budget, self.rng, self.batch = model, budget, rng, batch
        self.calls = 0; self.buf = []; self.new_word_today = None

    def next(self, teacher):
        if not self.buf and self.calls < self.budget:
            self.buf = self.plan(teacher)
        if self.buf:
            s = self.buf.pop(0); self.planned = True
            if s.strip():
                self.recent.append(s.strip()); self.recent = self.recent[-24:]
            return (s, "cue" if s.endswith(" ") else "line")
        self.planned = False                                  # the typist's own filler
        if self.recent:
            return (self.rng.choice(self.recent), "line")     # a recent planned line, said as a line (a cue's prefix too)
        heard = teacher.corpus.heard_lines(HEARD_FOR_CUE) or LINES0
        return (self.rng.choice(heard), "line")               # a heard line only when nothing was ever planned

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
    ap.add_argument("--period", type=float, default=160); ap.add_argument("--quiet", type=float, default=12)   # in the body's ticks (2026-09-08)
    ap.add_argument("--cap", type=float, default=48); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tick", type=float, default=0.25)      # the served body's seconds per tick: the parent's clock is the body's
    ap.add_argument("--listen", type=float, default=50)      # the child's turn after each line, in ticks
    ap.add_argument("--answer-levels", type=int, default=1); ap.add_argument("--parent", type=int, default=0)
    ap.add_argument("--reply", type=int, default=0)           # the parent wants a reply (the user's word of 2026-09-04)
    ap.add_argument("--wait", type=int, default=0)            # the reply withheld: ticks of the child's quiet before the parent answers (4)
    ap.add_argument("--yield", dest="yield_ticks", type=int, default=240)   # the parent steps back for this many ticks after a visitor types (2026-09-11)
    a = ap.parse_args()
    prev_q = None
    for k in range(a.days):
        day = a.day + k
        rng = random.Random(a.seed + day)
        planner = {"claude": lambda: ClaudePlanner(a.model, a.budget, rng),
                   "queue": lambda: QueuePlanner(a.queue, rng, pos=(prev_q.pos if prev_q else None), buf=(prev_q.buf if prev_q else None)),
                   "fixed": lambda: FixedPlanner(rng)}[a.planner]()
        if a.planner == "queue":
            prev_q = planner
        corpus = Corpus(a.corpus)
        t = Teacher("http://localhost:%d" % a.port, day, a.log, corpus, planner, period=a.period, quiet=a.quiet, cap=a.cap, seed=day, answer_levels=a.answer_levels, parent=a.parent, reply=a.reply, wait=a.wait, tick=a.tick, listen=a.listen, yield_ticks=a.yield_ticks)
        t.run_day()


if __name__ == "__main__":
    main()
