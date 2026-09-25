"""THE PARENT'S LEDGER (docs/SIM_DESIGN.md 4.8, 12, A18, A26, A28; package P3): every line she says, every word of the child's she
hears, her asks and her formal trials, and each word's standing, from outward events only. Deterministic, saved with the world,
and checked on a replay.

THE RECORD. One JSON line per event, keys sorted, in the life's folder (ledger.jsonl), each also folded into a running sha256
(`digest`), so two lives that said and heard the same things have the same digest:
  said        tick, the line, its intent, register, focus and emphasis, the objects it named, its source (the templates or
              Claude's), its frame, its clip's key and digest (the voice's ledger, body/sim/voice/synth.py), its length in
              ticks and when each word's sound is due to end
  voiced      the line's words once their sound has ended: which she said and which of them the child heard (below). A word
              counts as said only when its sound has ended (voiced()), so a word the talk-over stopped is never said
  cut         the talk-over stopped her line (4.6): the tick her voice stops, the words said and the words withdrawn (never
              said: not heard, not in her echo window)
  child       a word (or a wordless vocal turn) of the child's as the transcriber read it (body/sim/lang/transcriber.py): its
              channel (the tract, or the silent token output: kept apart from birth, A26), the word, exact or approximate, the
              expected set, the ear's distances, an echo or not, and whether it counted toward "says"
  ask         an everyday ask she made (a gaze, an act, a name, a call): her TEACHING, recorded with its outcome and counted
              toward nothing (the lead's decision: understanding is scored only in formal trials); the tick it is judged from
              (its word heard) and its window
  outcome     an ask's result: met, missed, or void (a gaze ask whose X she already reads the child attending, its head's line
              on it, in its hand or reached toward; a call while she reads it looking at her face; a gaze ask whose X is out of
              its view, or an X in her own hands, or a call while the child cannot see her, when it is judged from: A40; an act
              ask with no X within its reach then, as she sees it, or whose act was done before its word was heard; an ask cut
              before its word; a name ask answered only by an echo of her own word, or by a word begun before its question was
              heard)
  trial       a formal trial (4.8, 12; the conduct's probe): its form (consts.TRIAL_FORMS), the target and the distractor (their
              ids and the words they stand for), the sides, the word said (its name or a foil, for the name test), the tick
              its window opens (the target word's onset on the trial's one timeline: the latest of every sentence its draw
              could have given) and its length, the never-taught items presented, and what it scores
              (each of its two things that is a fresh never-taught item scores for its word, named or the other named; the
              name test for its name, its name or a foil said)
  trial_outcome  met (the child's first look or reach on the target; for the name test its turn to her face), missed (on the
              distractor; no turn), none (no look to either: counted neither way), or void (logged with why: it already attended
              one at the onset, both at once, her attention log showed anything but her mouth moving, the world stopped her
              sentence; never the child's voice: P3's eleventh round)
  displayed   a probe dropped after its two things were brought into the child's view, never said: the never-taught items it
              displayed, each one presentation more (fail-closed: her follow-in naming may label a thing in its new place)

EACH WORD'S STANDING (4.8), from those events alone:
  heard       said by her with its referent in the child's view (an object word: an object of that name the child sees; a body
              word: the child in her view; any other word: said)
  understood  from formal trials only, over its last UNDERSTOOD_LAST = 20 scored trials (its thing a fresh never-taught item,
              A28, whichever of the two was named: about 10 of each, since which is named is a fair coin), both at a
              one-sided binomial p < UNDERSTOOD_P = 0.05 against 1/2:
                (i)  of the trials naming it, its first look on its thing, at least UNDERSTOOD_MIN = 5 times;
                (ii) over all of them, its first look where the word said sent it (on its thing when it was named, on the
                     other when the other was).
              (ii) is chance by counterbalancing made exact. Which is named is a fair coin her trial stream draws fresh each
              trial, so for a child whose first look does not depend on the word said (a favourite of either thing at any
              strength, one that grows or shifts, a child that repeats whatever her smiles rewarded before) each trial goes
              where the word sent it with probability exactly 1/2 whatever came before: the count is Binomial(n, 1/2), no rate
              estimated (the design's randomization test; a martingale, so it holds whatever the child learns between
              trials). (i) keeps the credit on this word: a child that knows only the other word looks at the other thing when
              that is named, which (ii) alone would credit here. Together they are the design's "50%, or its yoked rate where
              higher" (the yoked rate: its first looks on its thing when the other was named), tested on one stretch of trials
              instead of against a rate carried from other times and treated as known (P3's ninth verifier: a child whose
              favourite, or whose turning to any voice, grew over its life reached "understood" far more often than a fair
              coin). The name: (ii) over its last 20 name-test trials (a turn to her face after its name, no turn after a
              foil in the same voice: Mandel, Jusczyk and Pisoni 1995), with at least 5 turns after its name; a child that turns
              to any voice, at any rate and however it changes, meets (ii) on exactly half. Only looks, reaches and turns meet
              a trial: a word never does. Re-tested after every trial and kept once reached (4.8, C64: a child whose looks do
              not depend on the word reaches it at most as often as a fair coin's trials reach (ii) alone)
  says        (per channel) the child says X with X where she reads it looking, in its hand or reached toward ("mama": she reads
              it looking at her face, or she is away), or right after its act (an act word: the act within the last 40 ticks), or,
              for a word with no referent or act ("hi", "more"), in context: among the words she expected then (A27's set: the
              tract's words always are, a token's letters read as a word she did not expect are not); never within ECHO_WINDOW
              ticks of her saying it (an echo), and never begun or ended while an ask of hers is open (its line to its
              window's end) or a trial of hers is (its sentence to its window's end), whatever the word (teaching only, the
              lead's decision: the conduct's accepted(teaching=)); 3 times over at least 2 life days
  exact       (per channel) how often the word was said exactly: an approximation earns a recast and a smile until this
              reaches 3 (4.6, A27)
  asks        her everyday asks' last ASKS_KEEP = 10 outcomes (1 met, 0 missed): her teaching's record, read by nothing that
              paces or claims
SECTION 12'S TEST OF A FORM (pooled(): A19, A28): (i) and (ii) over the form's trials, each trial counted once however many of its
things were fresh, one-sided at CLAIM_P = 0.01; a form whose trials are too few for a perfect score to reach it is "not testable
yet", never loosened.

SAVED WITH THE WORLD. state() is plain data (the standing, the open asks and trials, the items presented, each form's trials,
the digest, the file's length); load_state() puts it back and cuts the file to that length. The events a crash lost are then
replayed (A18: the lost part of the day replays exactly), and each replayed line is checked against the line the file held
there: a difference raises LedgerDiverged, as a changed voice raises VoiceChanged, and the life pauses rather than go on
differently.
"""
import hashlib
import json
import math
import os
from fractions import Fraction
from pathlib import Path

from . import consts as K
from . import templates as TP
from .lexicon import NAME, PARENT_NAME

ACT_WORDS = {"roll": ("rolled",), "sit": ("sat",), "give": ("gave",), "up": ("sat",), "down": ("fell",), "look": ()}


class LedgerDiverged(RuntimeError):
    """a replayed event differs from the one the ledger's file held: the life did not replay exactly."""


def binom_tail(k, n, p0=0.5):
    """P(X >= k) for X ~ Binomial(n, p0), exact (in rationals, then one rounding: a pooled form's thousands of trials neither
    overflow nor lose the tail)."""
    k, n = int(k), int(n)
    if k <= 0:
        return 1.0
    if k > n:
        return 0.0
    q = Fraction(p0)
    if q <= 0:
        return 0.0
    if q >= 1:
        return 1.0
    if q == Fraction(1, 2):
        return float(Fraction(sum(math.comb(n, i) for i in range(k, n + 1)), 2 ** n))
    return float(sum(math.comb(n, i) * q ** i * (1 - q) ** (n - i) for i in range(k, n + 1)))


def two_part(seq, name=False, last=None):
    """the ledger's test (4.8) over a record of scored trials [[named, on], ...] in their order (named: 1 when its word was said
    (the name test: its name), 0 when the other thing's (a foil); on: 1 when its first look was on its thing (its turn to her
    face)), over the last `last` of them (all when None) -> dict(n, named, on_named, where, p_named, p_where): (i) its first
    looks on its thing when it was named, against 1/2; (ii) its first looks where the word said sent it (named and on, or not
    named and not on), against 1/2. Both are one-sided binomial tails; (ii) is exact for any child whose look does not depend on
    the word said (which is named is a fair coin drawn fresh each trial), (i) is not needed for the name (its foils are no
    words)."""
    w = [(int(a), int(b)) for a, b in (seq if last is None else seq[-int(last):])]
    named = [on for nm, on in w if nm]
    where = sum(int(on == nm) for nm, on in w)
    return dict(n=len(w), named=len(named), on_named=sum(named), where=where,
                p_named=None if name else binom_tail(sum(named), len(named), K.CHANCE_2AFC),
                p_where=binom_tail(where, len(w), K.CHANCE_2AFC))


def _blank():
    return dict(heard=0, said=0, asks=[], trials=[], yoked=[], seq=[], says={"tract": [], "token": []},
                echoes={"tract": 0, "token": 0}, exact={"tract": 0, "token": 0}, approx={"tract": 0, "token": 0},
                understood_at=None, says_at={})


class Ledger:
    def __init__(self, path=None):
        self.path = None if path is None else Path(path)
        self.words = {}
        self.trials = []                       # her open asks: dict(id, kind, word, obj, tick, open, until, run, opened, void)
        self.formal = {}                       # her open formal trials: str(id) -> dict(id, form, score)
        self.items = {}                        # a never-taught item -> its presentations in trials (A28: its first 3 count)
        self.forms = {}                        # section 12's record of each form's scored trials, each once, in order:
                                               # form -> [[named fresh (the name test: its name said), where, [its words]]]
        self.n = 0                             # events recorded
        self.chain = hashlib.sha256(b"the parent's ledger").hexdigest()
        self.recent_events = []                # (tick, kind, obj) the child's acts she saw, for act words
        self.voicing = []                      # her line sounding: its words not yet ended (dict(t, text, refs, words, done, ...))
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
        if not isinstance(w, str):
            raise ValueError(f"the ledger keeps words, not {w!r}")
        if w not in self.words:
            self.words[w] = _blank()
        return self.words[w]

    @property
    def digest(self):
        return self.chain

    # ------------------------------------------------------------------ her lines
    def said(self, t, line, p, clip=None, n_ticks=0, word_ends=()):
        """she starts a line at t; its words count as said as their sound ends (voiced)."""
        self.voicing.append(dict(t=int(t), text=line.text, refs=list(line.refs), words=[[w, int(te)] for w, te in word_ends],
                                 done=0, said=[], heard=[]))
        self._rec(dict(ev="said", t=int(t), text=line.text, intent=line.intent, register=line.register, focus=line.focus,
                       emphasis=line.emphasis, refs=list(line.refs), source=line.source, frame=line.frame,
                       clip=None if clip is None else clip.key, clip_digest=None if clip is None else clip.digest,
                       ticks=int(n_ticks), ends=[[w, int(te)] for w, te in word_ends]))

    def voiced(self, t, p):
        """her words whose sound has ended by t -> [(word, the tick it ended, whether it was its line's last word)]: each
        counted as said, and as heard when its referent is in the child's view now (an object word: the object the line
        named, else one of that name the child sees; a body word: the child in her view; any other word: said)."""
        out = []
        named = {s.id: s for s in p.seen}
        for v in self.voicing:
            while v["done"] < len(v["words"]) and v["words"][v["done"]][1] <= t:
                w, te = v["words"][v["done"]]
                v["done"] += 1
                if w in TP.OBJECT_NOUNS:
                    objs = [named[r] for r in v["refs"] if r in named and named[r].name == w] or \
                        [s for s in p.seen if s.name == w]
                    ok = any(s.child_sees for s in objs)
                elif w in TP.CHILD_BODY:
                    ok = p.child_in_view
                else:
                    ok = True
                st = self._w(w)
                st["said"] += 1
                v["said"].append(w)
                if ok:
                    st["heard"] += 1
                    v["heard"].append(w)
                out.append((w, int(te), v["done"] == v.get("n", len(v["words"]))))
        for v in [v for v in self.voicing if v["done"] >= len(v["words"])]:
            self.voicing.remove(v)
            self._rec(dict(ev="voiced", t=int(t), line=v["t"], text=v["text"], said=v["said"], heard=v["heard"]))
        return out

    def cut(self, t, line, kept, stop):
        """the talk-over at t stopped her line at tick `stop`, after its first `kept` words: the rest are withdrawn, never
        said (4.6)."""
        withdrawn = []
        for v in self.voicing:
            if line is not None and v["text"] == line.text:
                v.setdefault("n", len(v["words"]))
                withdrawn = [w for w, _te in v["words"][kept:]]
                v["words"] = v["words"][:kept]
        self._rec(dict(ev="cut", t=int(t), text=None if line is None else line.text, stop=int(stop), kept=int(kept),
                       withdrawn=withdrawn))

    # ------------------------------------------------------------------ the child's words
    def accepted(self, cw, p, teaching=False):
        """a ChildWord read this tick (the transcriber), with the moment's percept (the referent's place). teaching: the word was
        begun or ended while an ask or a trial of hers was open (the conduct's _teaching): never toward "says"."""
        counted, why = False, ""
        if cw.word is not None:
            st = self._w(cw.word)
            (st["exact"] if cw.exact else st["approx"])[cw.channel] += 1
            if cw.echo:
                st["echoes"][cw.channel] += 1
                why = "echo"
            elif teaching:
                why = "while her ask or trial was open: teaching only, never 'says' (the lead's decision)"
            else:
                counted, why = self._referent(cw.word, cw.tick, p, cw.expected)
                if counted:
                    st["says"][cw.channel].append(int(cw.tick))
                    days = {tk // K.TICKS_PER_DAY for tk in st["says"][cw.channel]}
                    if len(st["says"][cw.channel]) >= K.SAYS_TIMES and len(days) >= K.SAYS_DAYS and \
                            cw.channel not in st["says_at"]:
                        st["says_at"][cw.channel] = int(cw.tick)
        d = cw.as_dict()
        self._rec(dict(ev="child", counted=counted, why=why, **d))

    def _referent(self, w, t, p, expected=()):
        if w in TP.OBJECT_NOUNS:
            if any(o.name == w for o in p.attended()):
                return True, "where she reads it looking, in its hand or reached toward"
            return False, "its referent not where she reads it looking, nor in or toward its hand"
        if w == PARENT_NAME:
            if p.child_target == "mama" or not p.present or not p.seen_by_child:
                return True, "she reads it looking at her face, or she is away"
            return False, "she does not read it looking at her face"
        if w in ACT_WORDS and ACT_WORDS[w]:
            if any(k in ACT_WORDS[w] for tk, k, _o in self.recent_events if tk > t - K.JUDGE_ACT):
                return True, "right after its act"
            return False, "no act of it just before"
        if w in expected:
            return True, "accepted in context"
        return False, "not in context: not a word she expected then"

    # ------------------------------------------------------------------ asks and their outcomes
    def ask(self, t, word, kind, obj, window, open_at=None):
        """an everyday ask she made at t (her teaching: its outcome is recorded and counted toward nothing), judged from open_at
        (the tick its word has been heard) over `window` ticks -> the ask's id."""
        if not isinstance(word, str) or not word:
            raise ValueError(f"an ask needs its word: {word!r}")
        tid = self.n
        op = int(t if open_at is None else open_at)
        self.trials.append(dict(id=tid, kind=kind, word=word, obj=obj, tick=int(t), open=op, until=int(op + window), run=0,
                                opened=False, void=None))
        self._rec(dict(ev="ask", t=int(t), word=word, kind=kind, obj=obj, open=op, window=int(window), teaching=True))
        return tid

    @staticmethod
    def _act_done(tr, p):
        """the act an act trial asks for, done this tick: its toy given her (an X: either twin), or its act word's act."""
        return any(k == "gave" and o is not None and p.obj(o) is not None and p.obj(o).name == tr["word"]
                   for k, o in p.events) or any(k in ACT_WORDS.get(tr["word"], ()) for k, _o in p.events)

    @staticmethod
    def _gave_other(tr, p):
        """the child gave her a toy other than the one an act ask asks for ("give me the ball", the cup given): its answer,
        missed (4.8; P3's eighth round)."""
        return tr["word"] in TP.OBJECT_NOUNS and any(k == "gave" and o is not None and p.obj(o) is not None and
                                                     p.obj(o).name != tr["word"] for k, o in p.events)

    def observe(self, t, p):
        """one tick of her open asks against what she sees -> [(word, kind, ask id, "met" | "missed" | "void")] for those
        resolved this tick."""
        self.recent_events = [e for e in self.recent_events if e[0] > t - K.JUDGE_ACT] + [(t, k, o) for k, o in p.events]
        out, keep = [], []
        att = {o.name for o in p.attended()}                         # its head's line, its hands: as she reads them (A40)
        for tr in self.trials:
            ok = None
            if t < tr["open"]:
                if tr["kind"] == "act" and self._act_done(tr, p):   # done before its word was heard: no answer to it (4.8)
                    tr["void"], ok = f"its act done before {tr['word']!r} was heard", False
                else:
                    keep.append(tr)
                    continue
            if not tr["opened"]:
                tr["opened"] = True
                if tr["kind"] == "gaze":
                    if tr["word"] in att:
                        tr["void"] = f"a {tr['word']} already where she reads it looking, in its hand or reached toward " \
                                     f"when the word was heard"
                    elif not any(s.name == tr["word"] and s.child_sees for s in p.seen):
                        tr["void"] = f"no {tr['word']} in its view when the word was heard"
                    elif any(s.name == tr["word"] and s.on == PARENT_NAME for s in p.seen):
                        tr["void"] = f"a {tr['word']} in her own hands when the word was heard: a look at her would meet it"
                elif tr["kind"] == "act" and tr["word"] in TP.OBJECT_NOUNS and \
                        not any(s.name == tr["word"] and s.child_can_reach for s in p.seen):
                    tr["void"] = f"no {tr['word']} within its reach when the word was heard: no act could answer it"
                elif tr["kind"] == "call" and p.child_target == "mama":
                    tr["void"] = "she read it already looking at her face when its name was heard"
                elif tr["kind"] == "call" and not (p.present and p.seen_by_child):
                    tr["void"] = "the child cannot see her when its name was heard: no look can answer it"
                if tr["void"] is not None:
                    ok = False                                       # resolved now, as void
            if ok is None and tr["void"] is None:
                if tr["kind"] == "gaze":
                    tr["run"] = tr["run"] + 1 if tr["word"] in att else 0
                    ok = tr["run"] >= K.HOLD or None
                elif tr["kind"] == "call":
                    tr["run"] = tr["run"] + 1 if p.child_target == "mama" else 0
                    ok = tr["run"] >= K.HOLD or None
                elif tr["kind"] == "act":
                    ok = self._act_done(tr, p) or (False if self._gave_other(tr, p) else None)
            if ok is None and t >= tr["until"]:
                ok = False
            if ok is None:
                keep.append(tr)
                continue
            res = self._outcome(t, tr, ok)
            out.append((tr["word"], tr["kind"], tr["id"], res))
        self.trials = keep
        return out

    def withdraw(self, t, tid, why):
        """an ask she did not finish saying (the talk-over cut it before its word): void."""
        for tr in list(self.trials):
            if tr["id"] == tid:
                self.trials.remove(tr)
                tr["void"] = why
                self._outcome(t, tr, False)
                return True
        return False

    def named(self, t, word, start):
        """a name ask ("what is this?") met by the child's word, read at t and begun at start (the tick the child made it): met
        only when begun once the question had been heard (4.8; a word made before it and read at its end is none). The conduct
        never calls it for an echo of her own word (an echo counts toward nothing here: it voids the ask instead)."""
        for tr in list(self.trials):
            if tr["kind"] == "name" and tr["word"] == word and start >= tr["open"]:
                self.trials.remove(tr)
                self._outcome(t, tr, True)
                return True
        return False

    def _outcome(self, t, tr, ok):
        """an ask's result -> "met" | "missed" | "void": her teaching's record (asks), counted toward nothing."""
        st = self._w(tr["word"])
        res = "void" if tr["void"] is not None else ("met" if ok else "missed")
        if res != "void":
            st["asks"].append(int(bool(ok)))
            st["asks"] = st["asks"][-K.ASKS_KEEP:]
        self._rec(dict(ev="outcome", t=int(t), trial=tr["id"], word=tr["word"], kind=tr["kind"], result=res, why=tr["void"],
                       teaching=True))
        return res

    # ------------------------------------------------------------------ formal trials (4.8, 12; the lead's decision)
    def presented(self, item):
        """a never-taught item's presentations in trials so far (A28: a probe only on its first 3)."""
        return int(self.items.get(item, 0))

    def trial(self, t, form, target, distractor, open_at, window, said, sides=None, score=(), items=()):
        """a formal trial's test sentence said at t -> its id. target, distractor: dict(id, word) (distractor None for the name
        test); open_at: the target word's onset (its window's first tick); said: the word said there (its name or a foil, for
        the name test); score: [[key, "trials" | "yoked", value if met, value if missed]], one for each of its things that is a
        fresh never-taught item ("trials": its word the one said; "yoked": the other's; the value 1 when the first look is on
        that thing: for the name test, its turn to her face); items: the never-taught items it presents (each one presentation
        more)."""
        if form not in K.TRIAL_FORMS:
            raise ValueError(f"not a trial form: {form!r}")
        tid = self.n
        for it in items:
            self.items[it] = self.items.get(it, 0) + 1
        sc = [list(x) for x in score]
        self.formal[str(tid)] = dict(id=tid, form=form, score=sc)
        self._rec(dict(ev="trial", t=int(t), form=form, target=target, distractor=distractor, said=said,
                       sides=None if sides is None else list(sides), open=int(open_at), window=int(window), score=sc,
                       items=list(items)))
        return tid

    def displayed(self, t, form, items, why):
        """a probe dropped after its things were brought into the child's view, never said: each never-taught item it displayed
        is one presentation more (fail-closed: shown there, it may have been named there, as her follow-in naming does)."""
        items = [it for it in items if it]
        for it in items:
            self.items[it] = self.items.get(it, 0) + 1
        self._rec(dict(ev="displayed", t=int(t), form=form, items=list(items), why=why))

    def trial_outcome(self, t, tid, result, why=None):
        """a formal trial's result: "met", "missed", "none" or "void" (with why). Met and missed score it: each of its words'
        record (named or not, its first look on its thing or not), its form's record (section 12's, the trial once); none and
        void count for nothing."""
        tr = self.formal.pop(str(tid), None)
        if tr is None:
            raise ValueError(f"no open trial {tid!r}")
        if result not in ("met", "missed", "none", "void"):
            raise ValueError(f"not a trial's result: {result!r}")
        if result in ("met", "missed") and tr["score"]:
            where = None
            for key, lst, on_met, on_missed in tr["score"]:
                st = self._w(key)
                named, v = int(lst == "trials"), int(on_met if result == "met" else on_missed)
                st[lst].append(v)
                st["seq"].append([named, v])
                where = int(v == named)          # its look where the word said sent it: the same for each of its things
                if st["understood_at"] is None and self._understood(key, st):
                    st["understood_at"] = int(t)
            fresh_named = int(any(x[1] == "trials" for x in tr["score"]))
            self.forms.setdefault(tr["form"], []).append([fresh_named, where, sorted({x[0] for x in tr["score"]})])
        self._rec(dict(ev="trial_outcome", t=int(t), trial=int(tid), form=tr["form"], result=result, why=why))
        return result

    # ------------------------------------------------------------------ standing
    def test(self, w):
        """its word's test now (4.8): two_part() over its last UNDERSTOOD_LAST scored trials, with whether it holds."""
        st = self.words.get(w) or _blank()
        r = two_part(st["seq"], name=w == NAME, last=K.UNDERSTOOD_LAST)
        r["holds"] = self._understood(w, st)
        return r

    @staticmethod
    def _understood(key, st):
        """(i) and (ii) at UNDERSTOOD_P over its last UNDERSTOOD_LAST scored trials, at least UNDERSTOOD_MIN first looks on its
        thing when named (the name: (ii) with at least UNDERSTOOD_MIN turns after its name)."""
        r = two_part(st["seq"], name=key == NAME, last=K.UNDERSTOOD_LAST)
        if r["on_named"] < K.UNDERSTOOD_MIN or r["p_where"] >= K.UNDERSTOOD_P:
            return False
        return key == NAME or r["p_named"] < K.UNDERSTOOD_P

    def pooled(self, form, keys=None):
        """section 12's test of a trial form (A19, A28): -> dict(k, n, where, n_where, p_named, p_where, p, testable, name): over
        the form's scored trials (those of `keys` alone when given), each counted once, (i) its first looks on the named thing
        in the trials whose named thing was fresh (k of n; none for the name test) and (ii) its first looks where the word said
        sent it (where of n_where), each a one-sided binomial against 1/2; p the larger, to hold at CLAIM_P. testable: a perfect
        score could reach CLAIM_P (a form with fewer trials is "not testable yet", never loosened: A28)."""
        name = form == "name"
        rows = [r for r in self.forms.get(form, ()) if keys is None or set(r[2]) & set(keys)]
        a = [r[1] for r in rows if r[0]]
        k, n = sum(a), len(a)
        where, nw = sum(r[1] for r in rows), len(rows)
        p_named = None if name else binom_tail(k, n, K.CHANCE_2AFC)
        p_where = binom_tail(where, nw, K.CHANCE_2AFC)
        p = p_where if name else max(p_named, p_where)
        testable = 0.5 ** nw < K.CLAIM_P and (name or 0.5 ** n < K.CLAIM_P)
        return dict(k=k, n=n, where=where, n_where=nw, p_named=p_named, p_where=p_where, p=p if testable else 1.0,
                    testable=testable, name=name)

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
                    trials=list(st["trials"]), yoked=list(st["yoked"]), seq=[list(x) for x in st["seq"]])

    # ------------------------------------------------------------------ save
    def state(self):
        return json.loads(json.dumps(dict(words=self.words, trials=self.trials, formal=self.formal, items=self.items,
                                          forms=self.forms, n=self.n, chain=self.chain, recent_events=self.recent_events,
                                          voicing=self.voicing, size=self.size)))

    def load_state(self, s):
        s = json.loads(json.dumps(s))
        self.words, self.trials, self.n, self.chain = s["words"], s["trials"], s["n"], s["chain"]
        for st in self.words.values():                    # a save before formal trials: its asks kept as teaching, never
            if "trials" not in st:                        # "understood" by them
                st["understood_at"] = None
            if "seq" not in st:                           # a save before P3's tenth round: its trials' order unknown, so its
                st["seq"] = []                            # record starts again, and an "understood" the ninth round's rule
                st["understood_at"] = None                # gave is not kept (fail-closed: its trials are tested again)
            for k_, v in _blank().items():
                st.setdefault(k_, v)
            st.pop("base", None)
            st.pop("forms", None)
        self.trials = [tr for tr in self.trials if not tr.get("base")]
        self.formal, self.items = dict(s.get("formal", {})), dict(s.get("items", {}))
        self.forms = {f: [list(r) for r in rows] for f, rows in dict(s.get("forms", {})).items()}
        self.recent_events = [tuple(e) for e in s["recent_events"]]
        self.voicing = s["voicing"]
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
