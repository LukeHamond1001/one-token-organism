"""THE PARENT'S LEDGER (docs/SIM_DESIGN.md 4.8, 12, A18, A26, A28; package P3): every line she says, every word of the child's she
hears, her asks and her formal trials, and each word's standing, from outward events only. Deterministic, saved with the world,
and checked on a replay.

THE RECORD. One JSON line per event, keys sorted, in the life's folder (ledger.jsonl), each also folded into a running sha256
(`digest`), so two lives that said and heard the same things have the same digest:
  said        tick, the line, its intent, register, focus and emphasis, the objects it named (kept: a thing among them is never
              a new exemplar after, A60b), its source (the templates or Claude's), its frame, its clip's key and digest (the
              voice's ledger, body/sim/voice/synth.py), its length in ticks and when each word's sound is due to end
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
              ids and the words they stand for; an exemplar trial's target its new exemplar, named or not, its second
              distractor in others, and her hearings so far of its noun and its foil, exposure), the sides, the word said (its
              name or a foil, for the name test; its word or its never-told foil, for an exemplar trial), the test
              word's onset tick on the trial's one timeline (every sentence its draw could have given made on it: P3's twelfth
              round), its window's first and last tick (consts.TRIAL_LOOK: the onset + 2 to the onset + 23, the same whichever
              is named), its stimuli (their ticks, the onset, the sentences, their test words' shapes, the words channel held
              to it or not), the never-taught items presented, and what it scores (each of its two things that is a fresh
              never-taught item scores for its word, named or the other named; the name test for its name, its name or a foil
              said)
  trial_outcome  SCORED BY THE PROPORTION OF LOOKING over its window (the lead's decision A60b, as infant labs score it:
              Golinkoff, Hirsh-Pasek, Cauley and Gordon 1987; Fernald et al. 2008; Bergelson and Swingley 2012): "scored" with
              on, the window's ticks her reading of its head line was on the target (T; the name test: on her face), and off,
              those on the distractor (D; the name test: the window's other ticks), whatever their order and wherever they
              fell in it (an exemplar trial: T its new exemplar's, D the two things' beside it); or "void", logged with why and
              counted, never scored: a trial of things with T + D < TRIAL_LOOK_MIN, her
              attention log showing anything but her mouth moving from the sentence to the window's end (a tick not read
              among it), the world stopping her sentence, its pain, distress or hit (she answers it: 4.10), a sentence not its
              stimulus's timeline, a save from before this rule mid-window. Never the child's voice (P3's eleventh round).
              For a child whose looking does not depend on the word said, what it does in a trial, a void among it, is the
              same whichever was named (her trial stream's fair coin, drawn fresh), so dropping a void leaves its labels
              exchangeable (intention to treat, P3's thirteenth round, is retired with the first look it was built for)
  test        a word's test taken (below): its block (1, its record; 2, its new-exemplar block, with the foils said in it and
              her hearings of the noun and its foil so far), its
              registered trials, how many named it, the difference, the p, the test's level, whether it was testable, whether
              its p was exact, whether it passed, and whether the word is understood
  displayed   a probe dropped after its two things were brought into the child's view, never said: the never-taught items it
              displayed, each one presentation more (fail-closed: her follow-in naming may label a thing in its new place)

EACH WORD'S STANDING (4.8), from those events alone:
  heard       said by her with its referent in the child's view (an object word: an object of that name the child sees; a body
              word: the child in her view; any other word: said)
  understood  from formal trials only (the lead's decision A60b), over its registered trials (each scored trial in which its
              thing was a fresh never-taught item, A28, whichever of the two was named): q, the share of the window's looking
              at either thing that went to its thing (T / (T + D) when it was named, D / (T + D) when the other was), and the
              difference between q's mean over the trials naming it and over those naming the other (the same thing, the foil
              condition: the design's yoked comparison), one-sided, by a PERMUTATION TEST OVER THE DRAW LABELS (perm_test: every
              way of choosing which k of its n trials named it, each as likely, the p the share whose difference is at least
              the one seen; Golinkoff et al. 1987, Bergelson and Swingley 2012). The permutation test is the invariance
              property: which of a trial's two things is named is a fair coin her trial stream draws fresh, so for any child
              whose looking does not depend on the word said (a favourite of either thing at any strength, one that grows or
              shifts, a side's, a follower of her gaze or hands, a child timed from her voice's start or her sound's stop as
              a whole, which since P3's fourteenth round are one moment whichever is named), its looking is the same
              whichever was named and its labels are exchangeable: the difference is null by construction, its p uniform or
              larger, no rate estimated. A child that looks where the test word's own sound sends it, whatever it hears in
              it, tells the words apart by their sound and goes to the named thing: that is what the trial credits (C69, C69b).
              The name: its face's share of the window, q = F / N (F the window's ticks on her face), after its name against
              after a foil (Mandel, Jusczyk and Pisoni 1995), the same test. Taken at its tests (consts.UNDERSTOOD_FIRST: its 12th
              registered trial, then 24, 48, ...), the j-th at UNDERSTOOD_P / 2^j, only where a perfect separation could reach
              it; the levels sum to UNDERSTOOD_P = 0.01, so such a child reaches "understood" in at most 1 life in 100 (a union
              over the tests, each exact), however long it lives (C64's "alpha spent over the looks"). Kept once reached.
              TWO LEVELS (the lead's decision A60b, after 535e32b): the test over its record, trials of its trained thing
              ("place"), is level 1, "maps" (maps_at); an object noun is "understood" (level 2) once the same test is passed
              also over its NEW-EXEMPLAR BLOCK (seq2: "exemplar" trials of new exemplars of its kind, each never among her
              lines' referents before its probe and fresh, set down beside new exemplars of two other kinds as new as it (the
              conduct's draw, since the lead's decision after 67741fd), registered apart, a block of 24 tested at its own 24th,
              48th, ... trial at 0.005, 0.0025, ... (consts.KIND_FIRST; kind_at), whose control since the lead's decision after
              200e57a is its NEVER-TOLD FOIL (consts.NOUN_FOILS, the skeptic's design; heard as often as its noun since the
              decision after 67741fd, exposure(), each trial's counts logged): q its new exemplar's share of the looking at the
              three, after its word against after its foil, so a child that knows only the other words, turning from what it
              can name at any word it cannot, looks alike after both and passes no word by exclusion. A child keyed to one
              particular thing passes level 1 and fails level 2, a knower of the kind passes both (Waxman and Booth 2001). The
              name, the colours and a combination's key have one level, "understood" when it passes (the name's own limit: to
              recognize one's name is to recognize its sound pattern, Mandel, Jusczyk and Pisoni 1995, so a child keyed to a
              band of its sound is a recognizer by any looking test, and need be no more: A60b, C69b). Only looks meet a
              trial: a word never does, nor a reach, a hit or its voice. Disclosed, the yoked comparison's own limit at level
              1: a child that knows only the other word looks at the other thing when that is named and at either when this
              one is, so this word's record passes too ("maps" by exclusion: A60b); level 2's foil closes it for
              "understood"
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
SECTION 12'S TEST OF A FORM (pooled(): A19, A28): over the form's scored trials, each counted once however many of its things were
fresh, x = (T - D) / (T + D), the named thing's share of the looking at either less the other's; their sum one-sided at CLAIM_P
= 0.01 by the FLIP TEST over the draw labels (flip_test: every trial's draw turned over or not, all 2^n as likely, turning x to
-x, as her trial stream's fair coins would have given them); the name form by its permutation test, its name against its foils.
A form whose trials are too few for a perfect score to reach it is "not testable yet", never loosened; its voids counted beside
it.

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

import numpy as np

from . import consts as K
from . import templates as TP
from .lexicon import NAME, PARENT_NAME

ACT_WORDS = {"roll": ("rolled",), "sit": ("sat",), "give": ("gave",), "up": ("sat",), "down": ("fell",), "look": ()}
SCORING = "look-share-4"                  # its trials scored by the proportion of looking over a fixed window and tested by
                                          # permutation over the draw labels (the lead's decision A60b), at two levels (maps:
                                          # the trained thing; understood: a new exemplar too, beside new exemplars as new as
                                          # it, against its never-told foil heard as often as it, from its block's 24th trial):
                                          # a save of "look-share-2" (level 2 against another thing's known word) or of
                                          # "look-share-3" (a foil beside familiar things, its exposure not matched) has its
                                          # new-exemplar blocks started again, one before them (first looks, or one level) its
                                          # whole record (load_state)
LEVEL2_OLD = ("look-share-2", "look-share-3")


class LedgerDiverged(RuntimeError):
    """a replayed event differs from the one the ledger's file held: the life did not replay exactly."""


# ------------------------------------------------------------------ the tests over the draw labels (A60b)
def _scaled(fracs):
    """exact shares -> integers on one scale (their denominators' least common multiple), so sums compare exactly."""
    den = 1
    for f in fracs:
        den = den * f.denominator // math.gcd(den, f.denominator)
    return np.array([f.numerator * (den // f.denominator) for f in fracs], np.int64), den


def _halves(vals):
    """every subset of vals: its sum and its size (2^len values each), built a value at a time."""
    sums, sizes = np.zeros(1, np.int64), np.zeros(1, np.int16)
    for v in vals:
        sums, sizes = np.concatenate([sums, sums + v]), np.concatenate([sizes, sizes + 1])
    return sums, sizes


def _count_at_least(vals, s, size=None):
    """how many subsets of the integers vals (of `size` members, or of any size) sum to at least s, exactly: the two halves'
    subsets each enumerated, one sorted, the other searched in it (meet in the middle)."""
    h = len(vals) // 2
    sa, na = _halves(vals[:h])
    sb, nb = _halves(vals[h:])
    if size is None:
        sb = np.sort(sb)
        return int((len(sb) - np.searchsorted(sb, s - sa, "left")).sum())
    total = 0
    for j in range(max(0, size - (len(vals) - h)), min(size, h) + 1):
        a, b = sa[na == j], np.sort(sb[nb == size - j])
        if len(a) and len(b):
            total += int((len(b) - np.searchsorted(b, s - a, "left")).sum())
    return total


def _draws(level):
    return max(K.PERM_DRAWS, int(math.ceil(20.0 / level))) if level else K.PERM_DRAWS


def perm_test(seq, level=None):
    """the permutation test of a word's record (4.8, the lead's decision A60b) over the draw labels: seq [[named, a, b], ...],
    named 1 when its word was said (the name test: its name), 0 when the other thing's (a foil); a, b the window's ticks on its
    thing (its face) and on the other (its other ticks) -> dict(n, named, diff, p, exact, testable): q = a / (a + b) each trial,
    diff the mean q over the trials naming it less the mean over the rest (None when either side is empty), p the share of the
    C(n, k) ways of choosing which k trials named it whose difference is at least diff (exact up to PERM_EXACT_N trials: every
    relabeling counted; past them drawn, PERM_DRAWS or 20 / level relabelings from PERM_SEED's stream, p = (1 + those at least as
    far) / (draws + 1): a valid p at any count), testable when a perfect separation could reach `level` (1 / C(n, k) under it).
    For any child whose looking does not depend on the word said, its labels are exchangeable (her trial stream's fair coin,
    drawn fresh): P(p <= a) <= a exactly."""
    rows = [(int(nm), int(a), int(b)) for nm, a, b in seq if int(a) + int(b) > 0]
    n, k = len(rows), sum(nm for nm, _a, _b in rows)
    if n == 0 or k in (0, n):
        return dict(n=n, named=k, diff=None, p=1.0, exact=True, testable=False)
    q = [Fraction(a, a + b) for _nm, a, b in rows]
    s_named = sum((x for x, (nm, _a, _b) in zip(q, rows) if nm), Fraction(0))
    diff = s_named / k - (sum(q, Fraction(0)) - s_named) / (n - k)   # for a fixed k, larger with the named trials' sum alone
    vals, den = _scaled(q)
    s_obs = int(sum(v for v, (nm, _a, _b) in zip(vals, rows) if nm))
    ways = math.comb(n, k)
    if n <= K.PERM_EXACT_N:
        p, exact = _count_at_least(vals, s_obs, size=k) / ways, True
    else:
        rng, m, hits = np.random.Generator(np.random.PCG64(K.PERM_SEED)), _draws(level), 0
        for c in range(0, m, 2000):
            idx = np.argpartition(rng.random((min(2000, m - c), n)), k - 1, axis=1)[:, :k]   # k of n, each set as likely
            hits += int((vals[idx].sum(1) >= s_obs).sum())
        p, exact = (1 + hits) / (m + 1), False
    return dict(n=n, named=k, diff=float(diff), p=float(p), exact=exact,
                testable=level is None or 1.0 / ways < level)


def flip_test(rows, level=K.CLAIM_P):
    """section 12's test of a form (A19, A28, the lead's decision A60b): rows [[on, off], ...] a trial each, the window's ticks
    on the named thing and on the other -> dict(n, x, p, exact, testable): x = (on - off) / (on + off) each trial, their sum
    against its every flip (each trial's draw turned over or not, all 2^n as likely: x to -x), p the share at least as large
    (exact up to PERM_EXACT_N trials, meet in the middle; past them drawn as perm_test draws); testable when a perfect score
    could reach `level` (2^-n under it). For any child whose looking does not depend on the word said, each trial's x is as
    likely turned over (its named thing a fresh fair coin): exact whatever the child favours."""
    rows = [(int(a), int(b)) for a, b in rows if int(a) + int(b) > 0]
    n = len(rows)
    if n == 0:
        return dict(n=0, x=None, p=1.0, exact=True, testable=False)
    x = [Fraction(a - b, a + b) for a, b in rows]
    vals, den = _scaled([abs(v) for v in x])
    s_obs = sum(int(v) if f >= 0 else -int(v) for v, f in zip(vals, x))
    tot = int(vals.sum())
    if n <= K.PERM_EXACT_N:                     # sum(e v) >= s  <=>  the + set's sum >= (s + sum v) / 2 (an integer: even)
        p, exact = _count_at_least(vals, (s_obs + tot) // 2) / 2 ** n, True
    else:
        rng, m, hits = np.random.Generator(np.random.PCG64(K.PERM_SEED)), _draws(level), 0
        for c in range(0, m, 2000):
            sg = np.where(rng.random((min(2000, m - c), n)) < 0.5, 1, -1)
            hits += int(((sg * vals).sum(1) >= s_obs).sum())
        p, exact = (1 + hits) / (m + 1), False
    return dict(n=n, x=float(sum(x, Fraction(0)) / n), p=float(p), exact=exact, testable=level is None or 0.5 ** n < level)


def test_level(n, first=None):
    """a word's test at its n-th registered trial: the level of its test there (UNDERSTOOD_P / 2^j at its j-th: its 12th
    trial, 24th, 48th, ...; a new-exemplar block's from its first = KIND_FIRST = 24th), or None where it takes none
    (UNDERSTOOD_FIRST, C64's alpha spent over the looks)."""
    j, m = 1, K.UNDERSTOOD_FIRST if first is None else int(first)
    while m < n:
        j, m = j + 1, m * 2
    return K.UNDERSTOOD_P / 2 ** j if m == n else None


def _blank():
    return dict(heard=0, said=0, asks=[], trials=[], yoked=[], seq=[], tests=[], voids=0, seq2=[], tests2=[], maps_at=None,
                kind_at=None, says={"tract": [], "token": []}, echoes={"tract": 0, "token": 0}, exact={"tract": 0, "token": 0},
                approx={"tract": 0, "token": 0}, understood_at=None, says_at={})


def kinded(key):
    """a word whose "understood" is its second level too (A60b): an object noun, which has exemplars of its kind; the name, the
    colours and a combination's key have one level, their own test."""
    return key in TP.OBJECT_NOUNS


class Ledger:
    def __init__(self, path=None):
        self.path = None if path is None else Path(path)
        self.words = {}
        self.trials = []                       # her open asks: dict(id, kind, word, obj, tick, open, until, run, opened, void)
        self.formal = {}                       # her open formal trials: str(id) -> dict(id, form, score)
        self.items = {}                        # a never-taught item -> its presentations in trials (A28: its first 3 count)
        self.forms = {}                        # section 12's record of each form's scored trials, each once, in order:
                                               # form -> [[named fresh (the name test: its name said), on, off, [its words]]]
        self.voided = {}                       # each form's void trials that scored something, counted (A60b: disclosed)
        self.named_ids = set()                 # the things her lines have named (their refs): a new exemplar never named before
                                               # its probe is one only these do not hold (A60b's second level)
        self.named_unknown = False
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
        self.named_ids.update(str(r) for r in line.refs)
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

    def trial(self, t, form, target, distractor, open_at, window, said, sides=None, score=(), items=(), stimulus=None,
              others=None, exposure=None):
        """a formal trial's test sentence said at t -> its id. target, distractor: dict(id, word) (distractor None for the name
        test; an exemplar trial's target its new exemplar, named or not, and others its second distractor, [dict(id, word)]);
        open_at: the test word's onset tick; window: its window's first and last tick (consts.TRIAL_LOOK from the onset);
        said: the word said there (its name or a foil, for the name test; its word or a foil, for an exemplar trial); score:
        [[key, "trials" | "yoked", 1 | 0, 0 | 1, level]], one for each of its things that is a fresh never-taught item
        ("trials": its word the one said, its thing the target (1, 0); "yoked": the other's, its thing the distractor (0, 1);
        the name test's key its name, its region her face, "trials" after its name and "yoked" after a foil, (1, 0) both; level
        1, its word's record, or 2, its new-exemplar block, A60b: a new exemplar of its kind never named before the probe, set
        down beside two things of other words, "trials" after its word and "yoked" after a never-told foil, its region the
        target (1, 0) both; left out, 1); items: the never-taught items it presents (each one presentation more);
        stimulus: its sentences' one timeline (their ticks, the test word's onset tick, the sentences, their test words' shapes,
        whether the words channel was held to it: P3's twelfth round); exposure: an exemplar trial's {its noun: her hearings of
        it, its foil: of the foil} before the sentence (A60b 13: matched in exposure, logged beside each other)."""
        if form not in K.TRIAL_FORMS:
            raise ValueError(f"not a trial form: {form!r}")
        w0, w1 = (int(x) for x in window)
        if w1 < w0:
            raise ValueError(f"a trial's window runs forward: {window!r}")
        tid = self.n
        for it in items:
            self.items[it] = self.items.get(it, 0) + 1
        sc = [list(x) for x in score]
        self.formal[str(tid)] = dict(id=tid, form=form, score=sc, open=int(open_at), window=[w0, w1],
                                     words=[(target or {}).get("word"), (distractor or {}).get("word")], said=said)
        row = dict(ev="trial", t=int(t), form=form, target=target, distractor=distractor, said=said,
                   sides=None if sides is None else list(sides), open=int(open_at), window=[w0, w1], score=sc,
                   items=list(items), stimulus=stimulus)
        if others is not None:
            row["others"] = [dict(x) for x in others]
        if exposure is not None:
            row["exposure"] = {str(k): int(v) for k, v in sorted(dict(exposure).items())}
        self._rec(row)
        return tid

    def displayed(self, t, form, items, why):
        """a probe dropped after its things were brought into the child's view, never said: each never-taught item it displayed
        is one presentation more (fail-closed: shown there, it may have been named there, as her follow-in naming does)."""
        items = [it for it in items if it]
        for it in items:
            self.items[it] = self.items.get(it, 0) + 1
        self._rec(dict(ev="displayed", t=int(t), form=form, items=list(items), why=why))

    def trial_outcome(self, t, tid, result, why=None, on=None, off=None):
        """a formal trial's result (the lead's decision A60b): "scored", with on and off, its window's ticks on the target and on
        the distractor (the name test: on her face and off it; an exemplar trial: on its new exemplar and on either of the two
        things beside it), whatever their order; or "void" (with why), never scored and counted (a trial with things whose on +
        off is under TRIAL_LOOK_MIN is void: given as scored, it is refused). Scored, each of its fresh things' words' record
        takes the trial (named or not; its thing's ticks and the rest: [named, a, b], its thing the target or the distractor as
        its score entry says; an entry of level 2, its word's new-exemplar block, with the word said, its own or a foil) and its
        word is tested where a test is due (_test_at); its form's record (section 12's) takes it once. For a child whose
        looking does not depend on the word said, whether a trial is void, and its on and off turned over or not (a pair's; an
        exemplar trial's and the name's the same), are all its draw changes."""
        tr = self.formal.pop(str(tid), None)
        if tr is None:
            raise ValueError(f"no open trial {tid!r}")
        if result not in ("scored", "void"):
            raise ValueError(f"not a trial's result: {result!r}")
        name = tr["form"] == "name"
        if result == "scored":
            on, off = int(on), int(off)
            if on < 0 or off < 0 or (not name and on + off < K.TRIAL_LOOK_MIN):
                raise ValueError(f"a scored trial needs its window's looks, at least {K.TRIAL_LOOK_MIN} ticks on either "
                                 f"thing: {on!r}, {off!r} (void, not scored)")
        else:
            on = off = None
        scored = result == "scored"
        self._rec(dict(ev="trial_outcome", t=int(t), trial=int(tid), form=tr["form"], result=result, why=why, scored=scored,
                       on=on, off=off))
        if not tr["score"]:
            return result
        if not scored:
            for key, _lst, *_x in tr["score"]:
                self._w(key)["voids"] += 1
            self.voided[tr["form"]] = self.voided.get(tr["form"], 0) + 1
            return result
        due = []
        for key, lst, *x_ in tr["score"]:
            st = self._w(key)
            named = int(lst == "trials")
            level = int(x_[2]) if len(x_) > 2 else 1
            mine = int(x_[0]) if x_ else int(name or named)          # its thing the target (1, 0) or the distractor (0, 1)
            a, b = (on, off) if mine else (off, on)                  # its thing's ticks: the target's, or the distractor's
            if level == 2:                                           # its new-exemplar block: and the word said (its own, or
                st["seq2"].append([named, a, b, tr.get("said")])     # a never-told foil)
            else:
                st["trials" if named else "yoked"].append([a, b])
                st["seq"].append([named, a, b])
            due.append((key, level))
        self.forms.setdefault(tr["form"], []).append([int(any(x[1] == "trials" for x in tr["score"])), on, off,
                                                      sorted({x[0] for x in tr["score"]})])
        for key, level in sorted(set(due)):
            self._test_at(t, key, level)
        return result

    def _test_at(self, t, key, block=1):
        """its word's test where one is due in a block (UNDERSTOOD_FIRST: its 12th registered trial, 24th, 48th, ...;
        test_level), over all that block's registered trials, at that test's level, the permutation test (A60b): block 1, its
        word's record ("maps": the trained thing, against the other thing's word); block 2, its new-exemplar block (its new
        exemplar's share after its word against after a never-told foil, beside two things of other words: the lead's decision
        after 200e57a). Passed when testable and p under its level; recorded, and kept once reached. "understood" once both are
        (kinded words), or the first (the rest)."""
        st = self.words[key]
        seq = st["seq"] if block == 1 else st["seq2"]
        at = "maps_at" if block == 1 else "kind_at"
        level = test_level(len(seq), None if block == 1 else K.KIND_FIRST)
        if level is None or st[at] is not None:
            return
        r = perm_test([x[:3] for x in seq], level)
        testable = bool(r["testable"])
        hit = bool(testable and r["p"] < level)
        if hit:
            st[at] = int(t)
        if st["understood_at"] is None and st["maps_at"] is not None and (st["kind_at"] is not None or not kinded(key)):
            st["understood_at"] = int(t)
        st["tests" if block == 1 else "tests2"].append([r["n"], r["named"], r["p"], level, testable])
        row = dict(ev="test", t=int(t), word=key, block=block, n=r["n"], named=r["named"], diff=r["diff"], p=r["p"],
                   level=level, testable=testable, exact=bool(r["exact"]), passed=hit,
                   understood=st["understood_at"] is not None)
        if block == 2:
            row["foils"] = sorted({x[3] for x in seq if not x[0] and x[3] is not None})
            row["exposure"] = {w: self.exposure(w) for w in sorted({key} | set(row["foils"]))}
        self._rec(row)

    # ------------------------------------------------------------------ standing
    def test(self, w, block=1):
        """its word's test now (4.8), over all its registered trials of a block (1: its record, "maps"; 2: its new-exemplar
        block): perm_test at its next test's level (UNDERSTOOD_P / 2^j for its j-th), with its tests so far and whether the
        block's test is passed (holds)."""
        st = self.words.get(w) or _blank()
        tests = st["tests"] if block == 1 else st["tests2"]
        nxt = K.UNDERSTOOD_P / 2 ** (len(tests) + 1)          # (block 2's first at its KIND_FIRST-th trial)
        r = perm_test([x[:3] for x in (st["seq"] if block == 1 else st["seq2"])], nxt)
        r.update(level=nxt, tests=[list(x) for x in tests],
                 holds=st["maps_at" if block == 1 else "kind_at"] is not None)
        return r

    def pooled(self, form, keys=None):
        """section 12's test of a trial form (A19, A28, A60b): -> dict(n, p, testable, name, voids, ...): over the form's scored
        trials (those of `keys` alone when given), each counted once, the flip test of the named thing's share of the looking at
        either (flip_test; the name form: the permutation test of its face's share, its name against its foils; the exemplar
        form likewise, its new exemplar's share after its word against after a never-told foil), one-sided, to hold at
        CLAIM_P; testable: a perfect score could reach CLAIM_P (a form with fewer trials is "not testable yet", never loosened:
        A28); voids: the form's void trials, counted."""
        name = form == "name"
        rows = [r for r in self.forms.get(form, ()) if keys is None or set(r[3]) & set(keys)]
        r = perm_test([[x[0], x[1], x[2]] for x in rows], K.CLAIM_P) if form in ("name", "exemplar") else \
            flip_test([x[1:3] for x in rows], K.CLAIM_P)
        return dict(r, p=r["p"] if r["testable"] else 1.0, name=name, voids=int(self.voided.get(form, 0)))

    def understood(self, w):
        """its word understood (A60b): level 2 for a kinded word (maps, and its new-exemplar block passed), level 1 for the
        rest (the name, the colours, a combination's key)."""
        st = self.words.get(w)
        return st is not None and st["understood_at"] is not None

    def maps(self, w):
        """level 1 (A60b): its word maps to the trained thing, its record's test passed."""
        st = self.words.get(w)
        return st is not None and st["maps_at"] is not None

    def exposure(self, w):
        """how often she has said the word so far, its sound ended (voiced(): each word of her lines, whatever it named): a
        noun's and its foil's, matched in exposure (A60b 13)."""
        return int((self.words.get(w) or {}).get("said", 0))

    def was_named(self, oid):
        """a thing her lines have named (their refs) before now: never a new exemplar for her word's second level (an older save
        with no file to read them from: every thing, fail-closed)."""
        return getattr(self, "named_unknown", False) or str(oid) in self.named_ids

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
        return dict(heard=st["heard"], understood=st["understood_at"] is not None, maps=st["maps_at"] is not None,
                    kind=st["kind_at"] is not None, says_tract="tract" in st["says_at"],
                    says_token="token" in st["says_at"], exact=dict(st["exact"]), asks=list(st["asks"]),
                    trials=[list(x) for x in st["trials"]], yoked=[list(x) for x in st["yoked"]],
                    seq=[list(x) for x in st["seq"]], tests=[list(x) for x in st["tests"]], seq2=[list(x) for x in st["seq2"]],
                    tests2=[list(x) for x in st["tests2"]], voids=int(st["voids"]))

    # ------------------------------------------------------------------ save
    def state(self):
        return json.loads(json.dumps(dict(words=self.words, trials=self.trials, formal=self.formal, items=self.items,
                                          forms=self.forms, voided=self.voided, named_ids=sorted(self.named_ids), n=self.n,
                                          chain=self.chain, recent_events=self.recent_events, voicing=self.voicing,
                                          size=self.size, scoring=SCORING)))

    def load_state(self, s):
        s = json.loads(json.dumps(s))
        self.words, self.trials, self.n, self.chain = s["words"], s["trials"], s["n"], s["chain"]
        for w_, st in self.words.items():                 # a save before formal trials: its asks kept as teaching, never
            if "trials" not in st:                        # "understood" by them
                st["understood_at"] = None
            if "seq" not in st or any(not isinstance(x, list) or len(x) != 3 for x in st["seq"]) or \
                    any(not isinstance(x, list) for x in st.get("trials", []) + st.get("yoked", [])):
                st["seq"], st["trials"], st["yoked"] = [], [], []   # a save before P3's tenth round (its trials' order
                st["understood_at"] = None                # unknown), or a record of first looks: its record starts again,
                                                          # and an "understood" an older rule gave is not kept (fail-closed)
            for k_, v in _blank().items():
                st.setdefault(k_, v)
            st.pop("base", None)
            st.pop("forms", None)
            if s.get("scoring") in LEVEL2_OLD:            # its new-exemplar block stood against another thing's known word,
                st["seq2"], st["tests2"], st["kind_at"] = [], [], None   # or a foil beside familiar things, not heard as often:
                if kinded(w_):                            # that block starts again (A60b 9, 13, 14), no "understood" of it kept
                    st["understood_at"] = None            # (fail-closed); its record (level 1) stands
            elif s.get("scoring") != SCORING:             # a save before A60b scored first looks (by intention to treat, or
                st["seq"], st["trials"], st["yoked"], st["tests"] = [], [], [], []   # before it), which a child timed from
                st["seq2"], st["tests2"], st["maps_at"], st["kind_at"] = [], [], None, None   # a moment of her sound could
                st["understood_at"], st["voids"] = None, 0   # pass, or one level: its record starts again, no "understood" of
                                                          # it kept (fail-closed)
        self.trials = [tr for tr in self.trials if not tr.get("base")]
        self.formal, self.items = dict(s.get("formal", {})), dict(s.get("items", {}))
        self.forms = {f: [list(r) for r in rows] for f, rows in dict(s.get("forms", {})).items()}
        self.voided = dict(s.get("voided", {}))
        self.named_ids = set(s.get("named_ids", ()))
        self.named_unknown = "named_ids" not in s          # an older save: the things named read again from its file below,
                                                          # or, with no file, every thing counted as named (fail-closed)
        if s.get("scoring") in LEVEL2_OLD:                # (section 12's rows the same: the exemplar form's against a known
            self.forms.pop("exemplar", None)              # word dropped; an open trial of an older save is the conduct's to
            self.voided.pop("exemplar", None)             # void: its window was not measured, or not against a foil)
        elif s.get("scoring") != SCORING:
            self.forms, self.voided = {}, {}
        self.recent_events = [tuple(e) for e in s["recent_events"]]
        self.voicing = s["voicing"]
        self.size = s["size"]
        self._tail = []
        if self.path is not None and self.path.exists():
            raw = self.path.read_bytes()
            if len(raw) < self.size:
                raise ValueError(f"the ledger's file is shorter ({len(raw)} bytes) than its save ({self.size})")
            if self.named_unknown:
                for ln in raw[:self.size].decode(errors="replace").split("\n"):
                    if '"ev":"said"' in ln:
                        self.named_ids.update(str(r) for r in json.loads(ln).get("refs", ()))
                self.named_unknown = False
            self._tail = [ln for ln in raw[self.size:].decode(errors="replace").split("\n") if ln]
            if raw[self.size:] and not raw.endswith(b"\n"):
                self._tail = self._tail[:-1]              # a last line cut short (a kill) is not held against the replay
            os.truncate(self.path, self.size)

    def replay_left(self):
        """lines of the file not yet replayed since load_state()."""
        return len(self._tail)
