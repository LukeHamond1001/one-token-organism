"""THE PARENT'S LEDGER (docs/SIM_DESIGN.md 4.8, A18, A26, A28; package P3): every line she says and every word of the child's she
hears, and each word's standing, from outward events only. Deterministic, saved with the world, and checked on a replay.

THE RECORD. One JSON line per event, keys sorted, in the life's folder (ledger.jsonl), each also folded into a running sha256
(`digest`), so two lives that said and heard the same things have the same digest:
  said        tick, the line, its intent, register, focus and emphasis, the objects it named, its source (the templates or
              Claude's), its frame, its clip's key and digest (the voice's ledger, body/sim/voice/synth.py), its length in
              ticks, when each word ended, and which of its words the child heard (below)
  cut         the talk-over stopped her line (4.6)
  child       a word (or a wordless vocal turn) of the child's as the transcriber read it (body/sim/lang/transcriber.py): its
              channel (the tract, or the silent token output: kept apart from birth, A26), the word, exact or approximate, the
              expected set, the ear's distances, an echo or not, and whether it counted toward "says"
  ask, base   an ask she made (a gaze, an act, a name, a call) and its window; a base-rate trial (the same test at a random
              moment with no ask: scheduled by the day plan, P4)
  outcome     an ask's or a trial's result (met, missed, or void: a base trial is void if she said its word in its window)

EACH WORD'S STANDING (4.8), from those events alone:
  heard       said by her with its referent in the child's view (an object word: an object of that name the child sees; a body
              word: the child in her view; any other word: said)
  understood  after "where is the X?" or "look at the X" (X in the child's view, not in its fovea), X lands in its fovea within
              20 ticks and stays 2 (an act word: its act within 40): met on at least 5 of the last 10 asks, and above the
              child's own base rate (the same test at random moments) by a one-sided binomial p < 0.05, once the base rate has
              at least BASE_MIN = 10 trials (ours: the design names no minimum)
  says        (per channel) the child says X with X in its fovea or hand ("mama": her face in its fovea, or she is away), or
              right after its act (an act word: the act within the last 40 ticks), or, for a word with no referent or act
              ("hi", "more", "the"), accepted in context; never within ECHO_WINDOW ticks of her saying it (an echo); 3 times
              over at least 2 life days
  exact       (per channel) how often the word was said exactly: an approximation earns a recast and a smile until this
              reaches 3 (4.6, A27)

SAVED WITH THE WORLD. state() is plain data (the standing, the open trials, the digest, the file's length); load_state() puts it
back and cuts the file to that length. The events a crash lost are then replayed (A18: the lost part of the day replays exactly),
and each replayed line is checked against the line the file held there: a difference raises LedgerDiverged, as a changed voice
raises VoiceChanged, and the life pauses rather than go on differently.
"""
import hashlib
import json
import math
import os
from pathlib import Path

from . import consts as K
from . import templates as TP
from .lexicon import PARENT_NAME

ACT_WORDS = {"roll": ("rolled",), "sit": ("sat",), "give": ("gave",), "up": ("sat",), "down": ("fell",), "look": ()}


class LedgerDiverged(RuntimeError):
    """a replayed event differs from the one the ledger's file held: the life did not replay exactly."""


def binom_tail(k, n, p0):
    """P(X >= k) for X ~ Binomial(n, p0)."""
    if k <= 0:
        return 1.0
    if p0 <= 0.0:
        return 0.0
    if p0 >= 1.0:
        return 1.0
    return float(sum(math.comb(n, i) * p0 ** i * (1 - p0) ** (n - i) for i in range(k, n + 1)))


def _blank():
    return dict(heard=0, said=0, asks=[], base=[], says={"tract": [], "token": []}, echoes={"tract": 0, "token": 0},
                exact={"tract": 0, "token": 0}, approx={"tract": 0, "token": 0}, understood_at=None, says_at={})


class Ledger:
    def __init__(self, path=None):
        self.path = None if path is None else Path(path)
        self.words = {}
        self.trials = []                       # open: dict(id, kind, word, obj, tick, until, base, run, void)
        self.n = 0                             # events recorded
        self.chain = hashlib.sha256(b"the parent's ledger").hexdigest()
        self.said_ticks = {}                   # word -> the ticks she said it (for void base trials), trimmed
        self.recent_events = []                # (tick, kind, obj) the child's acts she saw, for act words
        self.size = 0                          # the file's length
        self._tail = []                        # a replay's lines to check against
        if self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.size = self.path.stat().st_size if self.path.exists() else 0

    # ------------------------------------------------------------------ the record
    def _rec(self, d):
        line = json.dumps(d, sort_keys=True, separators=(",", ":"))
        self.chain = hashlib.sha256((self.chain + line).encode()).hexdigest()
        self.n += 1
        if self._tail:
            want = self._tail.pop(0)
            if want != line:
                raise LedgerDiverged(f"event {self.n}: replayed {line[:160]} where the ledger held {want[:160]}")
        if self.path is not None:
            b = (line + "\n").encode()
            with open(self.path, "ab") as f:
                f.write(b)
            self.size += len(b)

    def _w(self, w):
        if w not in self.words:
            self.words[w] = _blank()
        return self.words[w]

    @property
    def digest(self):
        return self.chain

    # ------------------------------------------------------------------ her lines
    def said(self, t, line, p, clip=None, n_ticks=0, word_ends=()):
        ws = TP.words(line.text)
        heard = []
        named = {s.id: s for s in p.seen}
        for w in ws:
            if w in TP.OBJECT_NOUNS:
                objs = [named[r] for r in line.refs if r in named and named[r].name == w] or \
                    [s for s in p.seen if s.name == w]              # the object the line named, else any of that name
                ok = any(s.child_sees for s in objs)
            elif w in TP.CHILD_BODY:
                ok = p.child_in_view
            else:
                ok = True
            st = self._w(w)
            st["said"] += 1
            if ok:
                st["heard"] += 1
                heard.append(w)
        for w, te in word_ends:
            self.said_ticks.setdefault(w, []).append(int(te))
            self.said_ticks[w] = self.said_ticks[w][-8:]
        self._rec(dict(ev="said", t=int(t), text=line.text, intent=line.intent, register=line.register, focus=line.focus,
                       emphasis=line.emphasis, refs=list(line.refs), source=line.source, frame=line.frame,
                       clip=None if clip is None else clip.key, clip_digest=None if clip is None else clip.digest,
                       ticks=int(n_ticks), ends=[[w, int(te)] for w, te in word_ends], heard=heard))

    def cut(self, t, line):
        self._rec(dict(ev="cut", t=int(t), text=None if line is None else line.text))

    # ------------------------------------------------------------------ the child's words
    def accepted(self, cw, p):
        """a ChildWord read this tick (the transcriber), with the moment's percept (the referent's place)."""
        counted, why = False, ""
        if cw.word is not None:
            st = self._w(cw.word)
            (st["exact"] if cw.exact else st["approx"])[cw.channel] += 1
            if cw.echo:
                st["echoes"][cw.channel] += 1
                why = "echo"
            else:
                counted, why = self._referent(cw.word, cw.tick, p)
                if counted:
                    st["says"][cw.channel].append(int(cw.tick))
                    days = {tk // K.TICKS_PER_DAY for tk in st["says"][cw.channel]}
                    if len(st["says"][cw.channel]) >= K.SAYS_TIMES and len(days) >= K.SAYS_DAYS and \
                            cw.channel not in st["says_at"]:
                        st["says_at"][cw.channel] = int(cw.tick)
        d = cw.as_dict()
        self._rec(dict(ev="child", counted=counted, why=why, **d))

    def _referent(self, w, t, p):
        if w in TP.OBJECT_NOUNS:
            tg = p.target_obj()
            held = [p.obj(h) for h in p.child_holds]
            if (tg is not None and tg.name == w) or any(o is not None and o.name == w for o in held):
                return True, "in its fovea or hand"
            return False, "its referent not in its fovea or hand"
        if w == PARENT_NAME:
            if p.child_target == "mama" or not p.present or not p.seen_by_child:
                return True, "her face in its fovea, or she is away"
            return False, "her face not in its fovea"
        if w in ACT_WORDS and ACT_WORDS[w]:
            if any(k in ACT_WORDS[w] for tk, k, _o in self.recent_events if tk > t - K.JUDGE_ACT):
                return True, "right after its act"
            return False, "no act of it just before"
        return True, "accepted in context"

    # ------------------------------------------------------------------ asks and their outcomes
    def ask(self, t, word, kind, obj, window, base=False):
        tid = self.n
        self.trials.append(dict(id=tid, kind=kind, word=word, obj=obj, tick=int(t), until=int(t + window), base=bool(base),
                                run=0, void=False))
        self._rec(dict(ev="base" if base else "ask", t=int(t), word=word, kind=kind, obj=obj, window=int(window)))
        return tid

    def base_trial(self, t, word, kind="gaze", obj=None, window=K.JUDGE_GAZE):
        """the child's own base rate: the same test at a random moment with no word said (4.8; the day plan draws the moments)."""
        return self.ask(t, word, kind, obj, window, base=True)

    def observe(self, t, p):
        """one tick of the open trials against what she sees; -> [(word, kind)] of her asks met this tick."""
        self.recent_events = [e for e in self.recent_events if e[0] > t - K.JUDGE_ACT] + [(t, k, o) for k, o in p.events]
        met, keep = [], []
        tg = p.target_obj()
        for tr in self.trials:
            ok = None
            if tr["base"] and any(tr["tick"] <= tk <= t for tk in self.said_ticks.get(tr["word"], ())):
                tr["void"] = True
            if tr["kind"] == "gaze":
                tr["run"] = tr["run"] + 1 if (tg is not None and tg.name == tr["word"]) else 0
                ok = tr["run"] >= K.HOLD or None
            elif tr["kind"] == "call":
                tr["run"] = tr["run"] + 1 if p.child_target == "mama" else 0
                ok = tr["run"] >= K.HOLD or None
            elif tr["kind"] == "act":
                ok = any(k == "gave" and o is not None and p.obj(o) is not None and p.obj(o).name == tr["word"]
                         for k, o in p.events) or any(k in ACT_WORDS.get(tr["word"], ()) for k, _o in p.events) or None
            if ok is None and t >= tr["until"]:
                ok = False
            if ok is None:
                keep.append(tr)
                continue
            self._outcome(t, tr, ok)
            if ok and not tr["base"]:
                met.append((tr["word"], tr["kind"]))
        self.trials = keep
        return met

    def named(self, t, word):
        """a name ask ("what is this?", "more?") met by the child's word (the conduct calls it when the word is heard)."""
        for tr in list(self.trials):
            if tr["kind"] == "name" and tr["word"] == word and not tr["base"]:
                self.trials.remove(tr)
                self._outcome(t, tr, True)
                return True
        return False

    def _outcome(self, t, tr, ok):
        st = self._w(tr["word"])
        if tr["base"]:
            if not tr["void"]:
                st["base"].append(int(bool(ok)))
        elif tr["kind"] in ("gaze", "act", "call"):
            st["asks"].append(int(bool(ok)))
            st["asks"] = st["asks"][-K.UNDERSTOOD_LAST:]
            if st["understood_at"] is None and self._understood(st):
                st["understood_at"] = int(t)
        self._rec(dict(ev="outcome", t=int(t), trial=tr["id"], word=tr["word"], kind=tr["kind"], base=tr["base"],
                       result="void" if tr["void"] else ("met" if ok else "missed")))

    # ------------------------------------------------------------------ standing
    @staticmethod
    def _understood(st):
        asks = st["asks"][-K.UNDERSTOOD_LAST:]
        k, n = sum(asks), len(asks)
        if k < K.UNDERSTOOD_MIN or len(st["base"]) < K.BASE_MIN:
            return False
        p0 = sum(st["base"]) / len(st["base"])
        return binom_tail(k, n, p0) < K.UNDERSTOOD_P

    def understood(self, w):
        st = self.words.get(w)
        return st is not None and st["understood_at"] is not None

    def says(self, w, channel):
        st = self.words.get(w)
        return st is not None and channel in st["says_at"]

    def exact_count(self, w, channel=None):
        st = self.words.get(w)
        if st is None:
            return 0
        return sum(st["exact"].values()) if channel is None else st["exact"][channel]

    def standing(self, w):
        st = self.words.get(w) or _blank()
        return dict(heard=st["heard"], understood=st["understood_at"] is not None, says_tract="tract" in st["says_at"],
                    says_token="token" in st["says_at"], exact=dict(st["exact"]), asks=list(st["asks"]),
                    base=list(st["base"]))

    # ------------------------------------------------------------------ save
    def state(self):
        return json.loads(json.dumps(dict(words=self.words, trials=self.trials, n=self.n, chain=self.chain,
                                          said_ticks=self.said_ticks, recent_events=self.recent_events, size=self.size)))

    def load_state(self, s):
        s = json.loads(json.dumps(s))
        self.words, self.trials, self.n, self.chain = s["words"], s["trials"], s["n"], s["chain"]
        self.said_ticks = s["said_ticks"]
        self.recent_events = [tuple(e) for e in s["recent_events"]]
        self.size = s["size"]
        self._tail = []
        if self.path is not None and self.path.exists():
            raw = self.path.read_bytes()
            if len(raw) < self.size:
                raise ValueError(f"the ledger's file is shorter ({len(raw)} bytes) than its save ({self.size})")
            self._tail = [ln for ln in raw[self.size:].decode(errors="replace").split("\n") if ln]
            if raw[self.size:] and not raw.endswith(b"\n"):
                self._tail = self._tail[:-1]              # a last line cut short (a kill) is not held against the replay
            os.truncate(self.path, self.size)

    def replay_left(self):
        """lines of the file not yet replayed since load_state()."""
        return len(self._tail)
