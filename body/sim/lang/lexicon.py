"""THE BORN TABLE AND THE WORDS CHANNEL'S SCHEDULE (docs/SIM_DESIGN.md 3.4 channel 0, 4.8, 4.9; the scaffold).

The table is fixed at birth and shared by the ear's scaffold channel and the voice's silent token output: the 50 birth words
(section 4.8, in its order), the 26 letters, a space, rest and end: 79 rows. No row is added in life: a word after the first 50
arrives as its letters, one a tick from its onset, then a space, so no code changes and no word switches from letters to a token.

The words channel (the scaffold, channel 0) carries one symbol a tick: the parent's own label for the word it said, beside the
sound the ears hear. Words.word() queues a birth word's token for the tick its sound ends, a later word's letters from the tick
its sound starts, then a space; Words.end() queues END on the tick the line's sound ends (the voice's playback, body/sim/voice/
playback.py, calls both as each word starts to sound, so a line cut at a word's end labels exactly what was said, and withdraws
a word broken off by the talk-over stop's cap);
Words.tick() hands out one symbol a tick, in the words' order, each once it is due (so a token can be a tick late behind a busy
channel, and a word after a spelled one waits for its letters; measured in tools/sim_voice_check.py), and REST when nothing is
due. Symbols arrive only while the parent is audible (the caller says so each tick, by audible() below: a symbol due while the
parent is not audible is dropped, never delivered late).

Ticks: tick k of a line holds its samples [k x 2400, (k + 1) x 2400) at 16 kHz (150 ms); a word whose sound ends at sample e
(its last sample e - 1) ends on tick (e - 1) // 2400, and the token comes in that tick's frame, with the sound's last part.

END is our reading of the table's "end" row for the ear: the line's sound is over (the language body's end of a line). REST is
nothing heard on the channel this tick.
"""
from dataclasses import dataclass

NAME, PARENT_NAME = "pip", "mama"          # the owner's defaults (section 14, item 7)
GROUPS = (
    ("names", (NAME, PARENT_NAME)),
    ("social", ("hi", "bye", "yes", "no", "good", "uh", "oh", "night", "peekaboo")),
    ("toys", ("ball", "duck", "block", "cup", "bear", "car", "drum", "bottle")),   # "bottle" stays a first word (A88 took the bottle
                                                                                  # from the room, not the word: she never shows it)
    ("body", ("hand", "foot", "head", "tummy")),
    ("room", ("mat", "sofa", "window")),
    ("actions", ("look", "give", "roll", "sit", "up", "down", "crawl", "more")),
    ("function", ("the", "a", "is", "you", "it", "here", "there", "on", "in", "your", "where", "what", "this", "at", "me",
                  "see")),
)
BIRTH_WORDS = tuple(w for _, ws in GROUPS for w in ws)
LETTERS = tuple("abcdefghijklmnopqrstuvwxyz")
SPACE, REST, END = "<space>", "<rest>", "<end>"
TABLE = BIRTH_WORDS + LETTERS + (SPACE, REST, END)
N_BIRTH, N_TABLE = len(BIRTH_WORDS), len(TABLE)
WORD_ID = {w: i for i, w in enumerate(BIRTH_WORDS)}                  # "a" is a word (row 39) and a letter (row 50): kept apart
LETTER_ID = {ch: N_BIRTH + k for k, ch in enumerate(LETTERS)}
ID = {SPACE: N_TABLE - 3, REST: N_TABLE - 2, END: N_TABLE - 1}
TICK_SAMPLES = 2400                         # 150 ms at 16 kHz
AUDIBLE_DB = 20.0                           # the parent is audible when its plain speech reaches the nearer ear above this


def audible(speech_db_at_1m, gains_db):
    """the words channel's gate: would the parent's speech (its level at 1 m, dB SPL) reach the child's nearer ear above
    AUDIBLE_DB (dB SPL; ours: speech is detected at about 10-20 dB SPL in quiet, from memory)? gains_db: the ears' path gains for
    the parent's mouth (body/sim/ears.Heard.gains). With no walls in the room it holds everywhere in it (62 dB at 1 m reaches
    20 dB at 125 m); it matters once the house has doors."""
    return speech_db_at_1m + max(gains_db) >= AUDIBLE_DB

assert N_BIRTH == 50 and len(set(BIRTH_WORDS)) == 50 and N_TABLE == 79 and TABLE[ID[END]] == END


def symbols_for(word):
    """a word -> its symbols on the table: its token if it is a birth word, else its letters and a space."""
    w = word.lower()
    if w in WORD_ID:
        return [WORD_ID[w]]
    letters = [LETTER_ID[ch] for ch in w if ch in LETTER_ID]
    return letters + [ID[SPACE]] if letters else []


def spell(ids):
    return " ".join(TABLE[i] for i in ids)


@dataclass
class _Due:
    tick: int
    order: int
    sym: int
    word: str


class Words:
    """the words channel: one symbol a tick from a queue of the parent's labels."""

    def __init__(self):
        self.queue = []
        self.order = 0
        self.late = []                       # (symbol, ticks after its due tick) per delivered symbol: an instrument
        self.dropped = 0
        self.withdrawn = 0

    def word(self, word, on_tick, end_tick):
        """queue one word's symbols: a birth word's token for end_tick (the tick its sound ends), a later word's letters one a
        tick from on_tick (the tick its sound starts), then a space. Returns their orders (for withdraw())."""
        syms = symbols_for(word)
        if len(syms) == 1:
            due = [end_tick]
        else:
            due = [on_tick + k for k in range(len(syms))]
        orders = []
        for s, t in zip(syms, due):
            self.queue.append(_Due(t, self.order, s, word))
            orders.append(self.order)
            self.order += 1
        return orders

    def withdraw(self, orders, close_tick=None):
        """drop a word's symbols still queued (a word broken off by the talk-over stop is not labelled as said); returns how
        many were withdrawn. A spelled word some of whose letters were already delivered keeps its closing space, due by
        close_tick (the tick its broken sound stops), so the letters that came are closed off as a run, as every word's are."""
        drop = set(orders)
        queued = {d.order for d in self.queue}
        if len(orders) > 1 and any(o not in queued for o in orders[:-1]):      # letters out: the space (its last symbol) stays
            drop.discard(orders[-1])
            if close_tick is not None:
                for d in self.queue:
                    if d.order == orders[-1]:
                        d.tick = min(d.tick, close_tick)
        keep = [d for d in self.queue if d.order not in drop]
        n = len(self.queue) - len(keep)
        self.queue = keep
        self.withdrawn += n
        return n

    def end(self, tick):
        """queue END for the tick a line's sound ends (whole, or cut at a word's end)."""
        self.queue.append(_Due(tick, self.order, ID[END], ""))
        self.order += 1

    def push(self, words, start_tick, n_samples=None):
        """a whole line at once: words [(word, first sample, end sample)] of a line starting to sound on start_tick;
        n_samples, if given, queues END on the line's last tick."""
        for w, on, end in words:
            self.word(w, start_tick + on // TICK_SAMPLES, start_tick + (max(end, 1) - 1) // TICK_SAMPLES)
        if n_samples:
            self.end(start_tick + (max(n_samples, 1) - 1) // TICK_SAMPLES)

    def tick(self, t, audible=True):
        """the symbol delivered on tick t: the queue's first symbol once it is due, else REST. The queue keeps the words' order,
        so a spelled word is never split and its line's END comes last."""
        if not audible:
            keep = [d for d in self.queue if d.tick > t]
            self.dropped += len(self.queue) - len(keep)
            self.queue = keep
            return ID[REST]
        if not self.queue or self.queue[0].tick > t:
            return ID[REST]
        d = self.queue.pop(0)
        self.late.append((d.sym, t - d.tick))
        return d.sym

    def state(self):
        return dict(queue=[(d.tick, d.order, d.sym, d.word) for d in self.queue], order=self.order, dropped=self.dropped,
                    withdrawn=self.withdrawn)

    def load_state(self, s):
        self.queue = [_Due(*q) for q in s["queue"]]
        self.order, self.dropped, self.withdrawn = s["order"], s["dropped"], s["withdrawn"]
