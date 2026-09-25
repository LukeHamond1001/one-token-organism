"""A LINE BEING SPOKEN (docs/SIM_DESIGN.md 4.4, 4.6; package P1): the parent's clip played tick by tick into the world.

Utterance(clip, start_tick).tick(t, words) gives the tick's 2,400 samples as sound pressure at 1 m in front of the parent's mouth
(pascals; the ears' spatializer places them, body/sim/ears.py) and hands each word to the words channel (the scaffold,
body/sim/lang/lexicon.Words) on the tick its sound starts, with the tick its sound ends, so the channel labels what was said and
nothing else. cut(words) is the talk-over stop (4.6: "the parent finishes the current word (at most 3 ticks), stops"): the word
sounding now is finished and the line stops at its end, unless that end is more than CUT_MAX ticks away, when the parent breaks
off at CUT_MAX ticks and the broken word's symbols still queued in the words channel are withdrawn (its sound was not finished, so
it is not labelled as said; letters of a later word already delivered stay, as they came, and its closing space stays too, due
by the tick its sound stops, so the run is closed); between words it stops at once. A line cut before any of its words sounded
ends unheard: no END is queued for it (the P1-P2 verifier's nit: a lone END with no sound).
Measured over the 331 birth lines (plain) cut at every tick (tools/sim_voice_check.py): without the cap 82 of the 2,913 cuts
(2.8%) needed 4-5 ticks to finish the word (a long word said slowly, "peekaboo!", or a line's last word); over the same lines as
new-word lines (the new word lengthened, about 630 ms) 405 of 3,636 (11.1%) needed 4-6; with the cap none passes 3. The line's
last 10 ms before a cut fade out on a raised cosine (ours: at a word's end the clip is already 40 dB down, so the fade only removes the step
a hard stop would add). mouth() is the tick's loudness for the face.

Nothing here draws a random number; state() and Utterance.restore(clip, state) carry the play position, a cut and the words
handed to the channel, so a replay continues exactly (the clip itself comes back from the voice cache by its key).
"""
import numpy as np

from .synth import HOP

TICK = 2400
FADE = HOP                                  # 10 ms
CUT_MAX = 3                                 # ticks: the talk-over stop's longest finish (docs/SIM_DESIGN.md 4.6; teacher method)


class Utterance:
    def __init__(self, clip, start_tick):
        self.clip = clip
        self.start = int(start_tick)
        self.x = clip.pa()
        self.stop_at = len(self.x)
        self.pos = 0
        self.next_word = 0
        self.done = len(self.x) == 0
        self.cut_at_tick = None
        self.fade_from = None
        self.broken = None                  # the word index broken off by a capped cut
        self.handed = {}                    # word index -> the words channel's orders of its symbols
        self._last = np.zeros(TICK, np.float32)

    def tick(self, t, words=None):
        k = t - self.start
        assert k == self.pos // TICK, f"the line was played out of order (tick {t}, expected {self.start + self.pos // TICK})"
        seg = np.zeros(TICK, np.float32)
        if self.done:
            self.pos += TICK
            self._last = seg
            return seg
        n = max(0, min(TICK, self.stop_at - self.pos))
        seg[:n] = self.x[self.pos:self.pos + n]
        ws = self.clip.words
        while self.next_word < len(ws) and ws[self.next_word][1] < min(self.pos + TICK, self.stop_at):
            w, on, end = ws[self.next_word]
            if words is not None:
                self.handed[self.next_word] = words.word(w, self.start + on // TICK,
                                                         self.start + (max(min(end, self.stop_at), on + 1) - 1) // TICK)
                for k in [k for k in self.handed if k < self.next_word - 1]:
                    del self.handed[k]                               # only the word sounding at a cut can be withdrawn
            self.next_word += 1
        self.pos += TICK
        if self.pos >= self.stop_at:
            self.done = True
            if words is not None and self.next_word > 0:                # a line cut before any word sounded ends unheard
                words.end(self.start + (max(self.stop_at, 1) - 1) // TICK)
        self._last = seg
        return seg

    def _fade(self, a, stop):
        if stop > a:
            self.x = self.x.copy()
            ramp = 0.5 + 0.5 * np.cos(np.pi * (np.arange(a, stop) - (stop - FADE)) / FADE)
            self.x[a:stop] *= ramp.astype(np.float32)

    def cut(self, words=None):
        """finish the word sounding now and stop at its end, or break off after CUT_MAX ticks if its end is further (its
        queued symbols withdrawn from `words`, the words channel); between words stop now. Returns the ticks still to sound."""
        if self.done or self.cut_at_tick is not None:
            return int(np.ceil(max(0, self.stop_at - self.pos) / TICK))
        stop = self.pos
        for i, (w, on, end) in enumerate(self.clip.words):
            if on < self.pos < end:
                stop = end
                if end - self.pos > CUT_MAX * TICK:                  # too long to finish: broken off
                    stop = self.pos + CUT_MAX * TICK
                    self.broken = i
                break
        self.stop_at = min(self.stop_at, stop)
        if self.broken is not None and words is not None and self.broken in self.handed:
            words.withdraw(self.handed[self.broken], close_tick=self.start + (self.stop_at - 1) // TICK)
        self.fade_from = max(self.pos, self.stop_at - FADE)
        self._fade(self.fade_from, self.stop_at)
        self.cut_at_tick = self.start + self.pos // TICK
        return int(np.ceil((self.stop_at - self.pos) / TICK))

    def mouth(self):
        """the last tick's RMS sound pressure at 1 m (pascals): how far the parent's mouth opens this tick."""
        return float(np.sqrt(np.mean(self._last.astype(np.float64) ** 2)))

    def state(self):
        return dict(key=self.clip.key, start=self.start, pos=self.pos, stop_at=self.stop_at, next_word=self.next_word,
                    done=self.done, cut_at_tick=self.cut_at_tick, fade_from=self.fade_from, broken=self.broken,
                    handed={k: list(v) for k, v in self.handed.items()}, last=self._last.copy())

    @classmethod
    def restore(cls, clip, s):
        assert clip.key == s["key"], "not the clip this line was playing"
        u = cls(clip, s["start"])
        u.pos, u.next_word, u.done, u.cut_at_tick = s["pos"], s["next_word"], s["done"], s["cut_at_tick"]
        u.stop_at, u.fade_from, u._last = s["stop_at"], s["fade_from"], np.array(s["last"], np.float32)
        u.broken, u.handed = s["broken"], {int(k): list(v) for k, v in s["handed"].items()}
        if u.cut_at_tick is not None:
            u._fade(u.fade_from, u.stop_at)
        return u
