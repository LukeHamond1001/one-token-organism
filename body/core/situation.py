"""THE GROUNDING OF WORDS IN THE SITUATION (A214, 2026-10-08; docs/SIM_DESIGN.md A214).

WHY: A202's organ binds a word to the look of a thing (the fovea's colour), so a noun can be learned; a word for what is going on
('up' as it is stood, 'walk' as it steps, 'good' as she smiles, 'fall', 'gone') has no thing to look at. The owner's bar is a child that
describes its reality in language; a child's first words are as often events and states as things (Nelson 1973; Tomasello 2003).

WHAT, body-general: the ANATOMY declares a `Situation` (body/core/anatomy.py): the band of the cortex's ladder whose state is the
situation's code (the body's own representation of what is going on over that band's clock, every channel in it: posture, touch, her
face, the sounds) and the symbols never bound. The ORGAN holds, per symbol of the words channel, the running mean of that band's state
(centred on the band's own running mean, band_mu) at its hearings (`_situ_A` [vocab, d]; the hearings counted, `_situ_n`), by the same
cross-situational law as A202's rows (A202b's consistency, A202e's final-position weight, A202k's shared mean taken off, A202q's gate
that a name ends her lines), and:
  (1) THE BINDING, at the tick's end (`_situ_learn`): the symbol heard this tick moves its row toward the situation's code now.
  (2) THE WORD, at the voice's choice (`_situ_say`): the word whose situation matches the situation now (the best cosine past
      GROUND_MATCH, GROUND_MARGIN ahead of the second, among words with GROUND_SAY_MIN_N hearings, consistent, ending her lines) is a
      prior on the token output's logits, as A202's name is: standing primes 'up'; the actor and the lexicon decide as before.
No cue: a situation is not a place to look. Every constant is A202's (body/core/grounding.py). The organ's state lives in the body's
working day (the day blob keeps every attribute); a body whose anatomy declares no Situation (the diary) has none of it.

THE RULERS: situ_bind a day, situ_say a day and the words it primed against the child's posture and her acts (the record's `ssay`)."""
import numpy as np
import torch

from . import grounding as GR


class SituationMixin:
    def _situ_on(self):
        return getattr(self.anatomy, "situation", None) is not None

    def _situ_state(self):
        """the organ's rows and counts, born at first use (zeros); kept as CPU tensors, worked on as arrays that share their memory"""
        d = int(self.bands.shape[1])
        A = getattr(self, "_situ_A", None)
        if A is None or int(A.shape[0]) != int(self.m.vocab) or int(A.shape[1]) != d or getattr(self, "_situ_v", None) != 1:
            self._situ_A = torch.zeros((int(self.m.vocab), d), dtype=torch.float64)
            self._situ_n = torch.zeros(int(self.m.vocab), dtype=torch.float64)
            self._situ_s = torch.zeros(int(self.m.vocab), dtype=torch.float64)
            self._situ_h = torch.zeros(int(self.m.vocab), dtype=torch.float64)
            self._situ_f = torch.zeros(int(self.m.vocab), dtype=torch.float64)
            self._situ_mu = torch.zeros(d, dtype=torch.float64); self._situ_mu_n = 0.0
            self._situ_stats = {"bind": 0, "say": 0}
            self._situ_last = None; self._situ_line_n = 0; self._situ_v = 1
        return self._situ_A.numpy(), self._situ_n.numpy()

    def _situ_code(self):
        """the situation now: the declared band's state less its running mean (the body's own representation of what is going on)"""
        b = int(self.anatomy.situation.band)
        with torch.no_grad():
            z = (self.bands[b] - self.m.band_mu[b]).detach().cpu().to(torch.float64).numpy()
        return z

    def _situ_skip(self, u):
        g = self.anatomy.situation
        return u == int(self.sil) or u in tuple(g.skip) or u in tuple(getattr(self, "bans", ()) or ())

    def _situ_bind(self, u, T, w):
        A, n = self._situ_state(); nT = GR._norm(T)
        if nT > 0.0 and w > 0.0:
            n[u] += w; rate = max(w / float(n[u]), GR.GROUND_RATE * w)
            A[u] += min(1.0, rate) * (T - A[u]); self._situ_s.numpy()[u] += w * nT
            self._situ_stats["bind"] += 1
            mu = self._situ_mu.numpy(); tot = float(self._situ_mu_n) + w
            mu += max(w / tot, GR.GROUND_RATE * w / 10.0) * (T - mu); self._situ_mu_n = tot

    def _situ_learn(self, u):
        """(1) THE BINDING, at the tick's end: the symbol heard this tick bound to the situation now; the line's last word weighs
        GROUND_FINAL (A202e) when the line ends"""
        if not self._situ_on():
            return
        self._situ_state(); u = int(u)
        if u == int(getattr(self, "end_id", -1)) or u == int(getattr(self, "eot", -1)):
            last_ = getattr(self, "_situ_last", None)
            if last_ is not None and int(getattr(self, "_situ_line_n", 0)) >= 2:
                self._situ_bind(int(last_[0]), np.asarray(last_[1], dtype=np.float64), GR.GROUND_FINAL - 1.0)
                self._situ_f.numpy()[int(last_[0])] += 1.0
            self._situ_last = None; self._situ_line_n = 0
            return
        if self._situ_skip(u):
            return
        T = self._situ_code()
        self._situ_bind(u, T, 1.0); self._situ_h.numpy()[u] += 1.0
        self._situ_line_n = int(getattr(self, "_situ_line_n", 0)) + 1
        self._situ_last = (u, T.astype(np.float32).tolist())

    def _situ_consist(self):
        A, n = self._situ_state(); s_ = self._situ_s.numpy()
        return np.where(n > 0, GR._norm(A, axis=1) * n / np.maximum(s_, 1e-12), 0.0)

    def _situ_say(self):
        """(2) THE WORD, at the voice's choice: (the word whose situation is the situation now, its margin), or None"""
        if not self._situ_on():
            return None
        A, n = self._situ_state()
        T = self._situ_code() - self._situ_mu.numpy(); nt = GR._norm(T)
        if nt <= 1e-9:
            return None
        cons = self._situ_consist(); h_ = self._situ_h.numpy(); f_ = self._situ_f.numpy()
        ok = (n >= GR.GROUND_SAY_MIN_N) & (cons >= GR.GROUND_CONSIST) & (h_ >= GR.GROUND_MIN_N) & (f_ >= GR.GROUND_FINAL_SHARE * np.maximum(h_, 1.0))
        for i_ in tuple(getattr(self.anatomy.situation, "name_skip", ()) or ()):
            if 0 <= int(i_) < ok.shape[0]:
                ok[int(i_)] = False
        if getattr(self.anatomy, "grounding", None) is not None and hasattr(self, "_ground_name_ok"):
            # A216 (2026-10-08): A THING'S NAME IS NOT A WORD FOR THE SITUATION. Day 122 live, the organ's first hours: 'ball' primed 628
            # times, 'drum' 393, 'cup' 151, all on its back in floor play (the situation every noun is heard in), 'block' said 651 times.
            # A word bound to a thing's look is the thing's name (the whole-object assumption: Markman 1990) and its road is the look;
            # the situation primes the words the look organ does not name
            ok = ok & ~self._ground_name_ok()
        if int(ok.sum()) < 1:
            return None
        Ac = A - self._situ_mu.numpy(); na = GR._norm(Ac, axis=1)
        cos = (Ac @ (T / nt)) / np.maximum(na, 1e-9)
        cos = np.where(ok & (na > 1e-9), cos * cons, -1.0)
        order = np.argsort(-cos, kind="stable")
        b = int(order[0]); second = float(cos[order[1]]) if cos.size >= 2 else -1.0
        if cos[b] >= GR.GROUND_MATCH and cos[b] - second >= GR.GROUND_MARGIN and not self._situ_skip(b):
            self._situ_stats["say"] += 1
            return b, float(cos[b] - second)
        return None

    def situ_report(self):
        if not self._situ_on():
            return None
        A, n = self._situ_state()
        return {**{k_: int(v_) for k_, v_ in self._situ_stats.items()}, "words": int(np.sum((n >= GR.GROUND_SAY_MIN_N) & (self._situ_consist() >= GR.GROUND_CONSIST)))}
