"""THE TRANSCRIBER: HOW THE PARENT HEARS WHAT THE CHILD SAYS (docs/SIM_DESIGN.md 4.6, 4.8, 4.9, A26, A27; package P3). The child
has two outputs, and she reads both here, on the world's side:

  THE TRACT (effector 0, its voice). Each tick the child's tract samples (engine units at 1 m in front of its speaker,
  body/sim/tract.py) reach her head as sound pressure falling as 1 / distance (ours: the room has no echo yet, B9; she attends to
  the child's voice among the other sounds, as a person does by where it comes from, so no other source is mixed in). A tick is
  sounding to her when any band of her cochlea (body/sim/ears.cochlea, the child's own) is over her hearing threshold (10 dB SPL);
  a tick of exact silence (the tract at rest) is silent without computing. An utterance runs from its first sounding tick and
  ends when the voice has rested TURN_END_REST = 2 ticks (4.6: the child's turn ends there); its sound, those 2 quiet ticks
  included, goes to her ear (body/sim/parent_ear.ParentEar) with the words she expects at that moment (expected_words, A27).
  Every utterance is a vocal turn (a ChildWord whose word may be None); in_pause marks one begun while her voice was silent.
  THE TOKEN OUTPUT (effector 1, silent). One symbol a tick from the born table (body/sim/lang/lexicon.py). She reads it as an
  exact transcript (4.9): a word token is its word, exactly; letters run until a space, the end row, a token, or 2 rest ticks,
  and the run is read as the word it spells exactly, else a word within edit distance 1 (2 for words of 6 letters or more),
  else a word of which it is a prefix of at least 2 letters among the names of what the child sees or holds; nearest first,
  ties to what it sees or holds, then her words' order. Only her words are read (A15: a queue word said before it is introduced
  is not recognised). A token made while she is speaking is read when her line ends (A26: it never talks over her). Each
  reading carries the words she expected at that moment (A27's set, as for the tract), so a letter run read as a word she did
  not expect ("z" read as "a", "qp" as "up") is not taken as said in context (the ledger, the conduct's judgments).

A ChildWord's echo mark: said within ECHO_WINDOW = 10 ticks of her saying that word (4.8: it counts only as an echo, never toward
"says"). The ledger keeps the two channels apart from birth (A26).

expected_words() is A27's list, fixed before birth: the names of what she reads the child attending, where its head's line is,
in its hand or reached toward ("mama" when she reads it looking at her face; A40: never its fovea's window), the word of a pending
ask, the focus word of her last line, the words of the routine under way (consts.EXPECT_ROUTINES), and "mama" while she is away
or out of its view; kept to her words, in their order.

Nothing here draws a random number. state() and load_state() carry an utterance under way, the letters read so far and the tokens
held during her line, so a replay continues exactly.
"""
from dataclasses import dataclass, field

import numpy as np

from . import consts as K
from . import lexicon as LX
from .. import ears as E
from ..voice.synth import PA_PER_UNIT


@dataclass(frozen=True)
class ChildWord:
    tick: int                     # the tick it was read: the tract's turn end, or the token's reading
    channel: str                  # "tract" or "token"
    word: str = None              # the word she heard, or None (a vocal turn with no word)
    exact: bool = False
    start: int = 0                # the tick the utterance (or the letters) began
    end: int = 0
    expected: tuple = ()
    echo: bool = False
    in_pause: bool = False        # begun while her voice was silent
    detail: dict = field(default_factory=dict)

    def as_dict(self):
        return dict(tick=self.tick, channel=self.channel, word=self.word, exact=self.exact, start=self.start, end=self.end,
                    expected=list(self.expected), echo=self.echo, in_pause=self.in_pause, detail=dict(self.detail))


def expected_words(p, pending, last_focus, routine, vocab):
    """A27's expected set for this moment, kept to her words, in their order."""
    exp = set(p.attended_names())                    # its head's line, its hands, as she reads them (A40); "mama" at her face
    if pending is not None and pending.get("word"):
        exp.add(pending["word"])
    if last_focus:
        exp.add(last_focus)
    exp.update(K.EXPECT_ROUTINES.get(routine, ()))
    if not p.present or not p.seen_by_child:
        exp.update(K.EXPECT_AWAY)
    return tuple(w for w in vocab if w in exp)


def edit_distance(a, b):
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def read_letters(s, vocab, near=()):
    """a run of letters -> (word or None, exact, how: the edit distance or 'prefix'), by 4.9's rule: the word it spells exactly;
    else a word within edit distance 1 (2 for words of 6 letters or more), or a word it begins (at least 2 letters) among the
    names of what the child sees or holds (near). Among several, what it sees or holds first (the order of 4.9's rule does not
    say; a run the child writes while holding the ball, "ba", is the ball's before any other's), then the fewest letters
    wrong or missing, then her words' order."""
    if not s:
        return None, False, None
    if s in vocab:
        return s, True, 0
    best = None
    for i, w in enumerate(vocab):
        d = edit_distance(s, w)
        how = d if d <= (K.EDIT_MAX_LONG if len(w) >= K.EDIT_LONG else K.EDIT_MAX) else None
        if how is None and w in near and len(s) >= K.PREFIX_MIN and w.startswith(s):
            how, d = "prefix", len(w) - len(s)
        if how is None:
            continue
        key = (0 if w in near else 1, d, i)
        if best is None or key < best[0]:
            best = (key, w, how)
    return (best[1], False, best[2]) if best is not None else (None, False, None)


class Transcriber:
    """ear: body/sim/parent_ear.ParentEar (None: the tract's turns are heard as turns with no word). pin: the ear's (digest,
    method) taken here, when her ear is loaded, and saved with the world: the conduct verifies her ear against it at every night
    boundary, and a saved world restored with another ear is refused."""

    def __init__(self, ear=None):
        self.ear = ear
        self.pin = None if ear is None else tuple(ear.pin)
        self.utt = None                   # the utterance under way: dict(start, in_pause, quiet, x=[pa per tick])
        self.sounding = False
        self.turn_start = None            # the tick the current turn started (this tick's value, for the conduct)
        self.turn_end = None              # the tick the last turn ended
        self.letters = ""                 # the token output's letters so far
        self.letters_start = None
        self.rests = 0
        self.held = []                    # (tick, symbol) made while she spoke, read when her line ends
        self.n_utts = 0
        self.wall = []                    # (ticks sounding, ms to hear it): the cost instrument, not saved
        self.last_x = None                # the last utterance's sound (Pa at her head): an instrument, not saved

    expected = staticmethod(expected_words)

    # ------------------------------------------------------------------ one tick
    def tick(self, t, tract, distance_m, token, parent_speaking, expected, p, said_word=None, vocab=None):
        """-> [ChildWord] read this tick. tract: the child's samples this tick (engine units at 1 m) or None; token: its
        symbol this tick or None (rest); expected: expected_words() now; vocab: the words she has (default: the lexicon's)."""
        vocab = tuple(vocab) if vocab is not None else LX.BIRTH_WORDS
        said_word = said_word or {}
        out = []
        pa = None
        audible = False
        if tract is not None and np.any(tract):
            pa = np.asarray(tract, np.float64) * (PA_PER_UNIT / float(distance_m))
            audible = bool(np.any(E.cochlea(pa) > E.E_THR))
        self.sounding = audible
        if audible and self.utt is None:
            self.utt = dict(start=t, in_pause=not parent_speaking, quiet=0, x=[])
        if self.utt is not None:
            self.utt["x"].append(pa if pa is not None else np.zeros(E.TICK))
            self.utt["quiet"] = 0 if audible else self.utt["quiet"] + 1
            if self.utt["quiet"] >= K.TURN_END_REST:
                out.append(self._hear(t, expected, vocab, said_word))
        self.turn_start = self.utt["start"] if self.utt is not None else None
        # the token output
        if token is not None and token != LX.ID[LX.REST] or self.held or self.letters:
            if parent_speaking:
                if token is not None and token != LX.ID[LX.REST]:
                    self.held.append((t, int(token)))
            else:
                for tk, s in self.held:
                    out += self._symbol(tk, s, t, vocab, p, said_word, expected)
                self.held = []
                out += self._symbol(t, LX.ID[LX.REST] if token is None else int(token), t, vocab, p, said_word, expected)
        return out

    def _hear(self, t, expected, vocab, said_word):
        import time                                                          # noqa: PLC0415
        u, self.utt = self.utt, None
        self.turn_end = t
        self.n_utts += 1
        x = np.concatenate(u["x"])
        self.last_x = x
        detail = {}
        word, exact = None, False
        if self.ear is not None:
            t0 = time.perf_counter()
            h = self.ear.hear(x, expected, vocab)
            self.wall = self.wall[-999:] + [(len(u["x"]), (time.perf_counter() - t0) * 1e3)]
            word, exact = h.word, bool(h.exact)
            detail = {k: v for k, v in h.as_dict().items() if k not in ("expected",)}
            detail = {k: (None if isinstance(v, float) and not np.isfinite(v) else v) for k, v in detail.items()}
        return ChildWord(t, "tract", word, exact, u["start"], t, tuple(expected), self._echo(word, u["start"], said_word),
                         u["in_pause"], detail)

    def _echo(self, word, start, said_word):
        return word is not None and word in said_word and abs(start - said_word[word]) <= K.ECHO_WINDOW

    def _symbol(self, tk, s, t, vocab, p, said_word, expected=()):
        out = []
        if s < LX.N_BIRTH:                                                   # a word token
            out += self._close(t, vocab, p, said_word, expected)
            w = LX.TABLE[s]
            if w in vocab:
                out.append(ChildWord(t, "token", w, True, tk, tk, tuple(expected), self._echo(w, tk, said_word), True,
                                     dict(symbols=[w])))
            self.rests = 0
        elif s in LX.LETTER_ID.values():
            if not self.letters:
                self.letters_start = tk
            self.letters += LX.TABLE[s]
            self.rests = 0
        elif s == LX.ID[LX.REST]:
            self.rests += 1
            if self.rests >= K.TURN_END_REST:
                out += self._close(t, vocab, p, said_word, expected)
        else:                                                                # a space or the end row
            out += self._close(t, vocab, p, said_word, expected)
            self.rests = 0
        return out

    def _close(self, t, vocab, p, said_word, expected=()):
        if not self.letters:
            return []
        s, start = self.letters, self.letters_start
        self.letters, self.letters_start = "", None
        near = set()
        if p is not None:
            for o in p.seen:
                if o.child_sees:
                    near.add(o.name)
            near |= {o.name for o in p.attended()}
        w, exact, how = read_letters(s, vocab, near)
        return [ChildWord(t, "token", w, exact, start, t, tuple(expected), self._echo(w, start, said_word), True,
                          dict(letters=s, read=how))]

    # ------------------------------------------------------------------ save
    def state(self):
        return dict(utt=None if self.utt is None else dict(self.utt, x=[np.array(a) for a in self.utt["x"]]),
                    sounding=self.sounding, turn_start=self.turn_start, turn_end=self.turn_end, letters=self.letters,
                    letters_start=self.letters_start, rests=self.rests, held=[list(h) for h in self.held], n_utts=self.n_utts,
                    pin=None if self.pin is None else list(self.pin))

    def load_state(self, s):
        if s["pin"] is not None and (self.pin is None or tuple(s["pin"]) != self.pin):
            raise ValueError(f"the saved world heard with her ear {s['pin'][0][:16]}, not this one "
                             f"{None if self.pin is None else self.pin[0][:16]}: the ear is fixed for the life (A27)")
        self.utt = None if s["utt"] is None else dict(s["utt"], x=[np.array(a) for a in s["utt"]["x"]])
        self.sounding, self.turn_start, self.turn_end = s["sounding"], s["turn_start"], s["turn_end"]
        self.letters, self.letters_start, self.rests = s["letters"], s["letters_start"], s["rests"]
        self.held, self.n_utts = [tuple(h) for h in s["held"]], s["n_utts"]
