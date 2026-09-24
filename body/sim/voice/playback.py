"""A LINE BEING SPOKEN (docs/SIM_DESIGN.md 4.4, 4.6; package P1): the parent's clip played tick by tick into the world.

Utterance(clip, start_tick).tick(t, words) gives the tick's 2,400 samples as sound pressure at 1 m in front of the parent's mouth
(pascals; the ears' spatializer places them, body/sim/ears.py) and hands each word to the words channel (the scaffold,
body/sim/lang/lexicon.Words) on the tick its sound starts, with the tick its sound ends, so the channel labels what was said and
nothing else. cut() is the talk-over stop (4.6): the word sounding now is finished and the line stops at its end; between words it
stops at once. The line's last 10 ms before a cut fade out on a raised cosine (ours: a voice stops at a word's end, where the clip
is already 40 dB down, so the fade only removes the step a hard stop would add). mouth() is the tick's loudness for the face.

Nothing here draws a random number; state() and Utterance.restore(clip, state) carry the play position and a cut, so a replay
continues exactly (the clip itself comes back from the voice cache by its key).
"""
import numpy as np

from .synth import HOP

TICK = 2400
FADE = HOP                                  # 10 ms


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
                words.word(w, self.start + on // TICK, self.start + (max(min(end, self.stop_at), on + 1) - 1) // TICK)
            self.next_word += 1
        self.pos += TICK
        if self.pos >= self.stop_at:
            self.done = True
            if words is not None:
                words.end(self.start + (max(self.stop_at, 1) - 1) // TICK)
        self._last = seg
        return seg

    def _fade(self, a, stop):
        if stop > a:
            self.x = self.x.copy()
            ramp = 0.5 + 0.5 * np.cos(np.pi * (np.arange(a, stop) - (stop - FADE)) / FADE)
            self.x[a:stop] *= ramp.astype(np.float32)

    def cut(self):
        """finish the word sounding now and stop at its end (between words: stop now). Returns the ticks still to sound."""
        if self.done or self.cut_at_tick is not None:
            return int(np.ceil(max(0, self.stop_at - self.pos) / TICK))
        stop = self.pos
        for w, on, end in self.clip.words:
            if on < self.pos < end:
                stop = end
                break
        self.stop_at = min(self.stop_at, stop)
        self.fade_from = max(self.pos, self.stop_at - FADE)
        self._fade(self.fade_from, self.stop_at)
        self.cut_at_tick = self.start + self.pos // TICK
        return int(np.ceil((self.stop_at - self.pos) / TICK))

    def mouth(self):
        """the last tick's RMS sound pressure at 1 m (pascals): how far the parent's mouth opens this tick."""
        return float(np.sqrt(np.mean(self._last.astype(np.float64) ** 2)))

    def state(self):
        return dict(key=self.clip.key, start=self.start, pos=self.pos, stop_at=self.stop_at, next_word=self.next_word,
                    done=self.done, cut_at_tick=self.cut_at_tick, fade_from=self.fade_from, last=self._last.copy())

    @classmethod
    def restore(cls, clip, s):
        assert clip.key == s["key"], "not the clip this line was playing"
        u = cls(clip, s["start"])
        u.pos, u.next_word, u.done, u.cut_at_tick = s["pos"], s["next_word"], s["done"], s["cut_at_tick"]
        u.stop_at, u.fade_from, u._last = s["stop_at"], s["fade_from"], np.array(s["last"], np.float32)
        if u.cut_at_tick is not None:
            u._fade(u.fade_from, u.stop_at)
        return u
