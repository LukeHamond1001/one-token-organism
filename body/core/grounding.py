"""THE GROUNDING OF WORDS IN JOINT ATTENTION (A202, 2026-10-06; docs/SIM_DESIGN.md A202).

WHY: the G1's word rulers stood flat for thirty days (the heard words' top-1 at 0.33 from day 70 to day 103; one met ask a day) while
its body learned what her face paid for (standing, grasping, the door). A word reached the brain only as one more channel of the
frame; nothing bound the word to the thing the eyes were on when it was heard, and nothing turned a heard word into a place in the
present view to look at. Infants learn their first nouns in joint attention: the word heard while the eyes are on the thing (Tomasello
and Farrar 1986; Baldwin 1991), bound after a few hearings (fast mapping: Carey and Bartlett 1978), and a heard word then draws the eyes
to the thing's look in the periphery (feature-based attention biasing the priority map: Treue and Martinez-Trujillo 1999; Desimone and
Duncan 1995; Bisley and Goldberg 2010), and the thing's look primes its name (the ventral stream's road to the lexicon).

WHAT, body-general: the ANATOMY declares a `Grounding` (body/core/anatomy.py): an appearance function (the frame -> the look of what
the fovea holds, contrast-coded, k numbers), a periphery function (the frame -> every periphery cell's look in the same k numbers and
its direction from the fovea's centre) and the frame's observation the organ writes its cue into. The ORGAN holds, per symbol of the
words channel, the running mean of the appearance at its hearings (a Hebbian trace, `_ground_A` [vocab, k]; the hearings counted,
`_ground_n`), and:
  (1) THE BINDING, at the tick's end (`_ground_learn`): the symbol heard this tick (its sound ended), not the rest and not the end,
      moves its row toward the fovea's look by GROUND_RATE, when the fovea holds a look at all (its contrast above GROUND_FLOOR: a
      word heard while the eyes are on the empty floor binds nothing). A function word heard over everything averages to no look.
  (2) THE CUE, at the senses (`_ground_sense`): for GROUND_TRACE ticks after a word is heard (the orienting response's span), when its
      row has GROUND_MIN_N hearings and a look, every periphery cell's look is matched to it (cosine); the cell that matches best, past
      GROUND_MATCH and GROUND_MARGIN ahead of the second, is the cue: [1, yaw, pitch] from the fovea's centre, written into the frame as
      the born cues are (the anatomy's OrientCue reads it; the born saccade and the orienting bias take it as they take her face).
  (3) THE NAME, at the voice's choice (`_ground_say`): the word whose look fills the fovea now (the best cosine past GROUND_MATCH, the
      margin past GROUND_MARGIN, among words with GROUND_MIN_N hearings) is a prior on the token output's logits, GROUND_SAY times the
      margin: seeing the ball primes "ball"; the actor and the lexicon decide as before.
Every number below is ours and disclosed. The organ's state lives in the body's working day (persistence: the day blob keeps every
attribute), so a save continues it; a body whose anatomy declares no Grounding (the diary) has none of it and is unchanged (the
language default's digest 7c54199e3c72cf77d45a8e94 kept).

THE RULERS: ground_bind a day (bindings), ground_cue a day (the cue firing) and the asks met with the gaze on the named thing (the
ledger's met_ask, the 'nearer' result), ground_say a day and the words it said with a thing in view; the first look-at-the-named-thing
and the first naming on the film."""
import numpy as np
import torch

GROUND_RATE = 0.05      # A202b: the least step of a row toward this hearing's look; a row moves by max(1/n, this), the running mean of
                        # its hearings (cross-situational: a word heard over many things averages to no look, Yu and Smith 2007),
                        # drifting at this rate once it has many (ours). 0.15 at first: 'is' then tracked whatever was shown last
GROUND_CONSIST = 0.75   # A202b: the least consistency of a word's look (its mean's size over its hearings' mean size: 1 for one look
                        # always, 0.7 for two colours by turns, near 0 for a word heard over everything) for the cue and the name (ours)
GROUND_MIN_N = 3        # hearings before a word's look is trusted for the cue and the name (ours: fast mapping's few)
GROUND_TRACE = 20       # ticks (3 s) a heard word keeps drawing the eyes toward its look (ours: the orienting response's span)
GROUND_FLOOR = 0.02     # the least contrast (opponent units) that counts as a look at a thing, in the fovea or a cell (ours)
GROUND_MATCH = 0.6      # the least cosine between a word's look and a cell's for the cue or the name (ours)
GROUND_MARGIN = 0.15    # the best match's lead over the second (ours)
GROUND_SAY = 2.0        # A202i: the prior on the logit of the word whose look fills the fovea, in the readout's own units: this many standard
                        # deviations of the readout's logits for a margin at GROUND_MARGIN, in proportion past it (ours). It was 1.0 times
                        # the margin, 0.2 of a logit against a top gap of 2.6 (day 109: 'ball' primed 47 times and never said)
GROUND_FINAL = 3.0      # A202e: the weight of a line's last word's hearing (its look counted this many times over a word within the line):
                        # the utterance-final word is the one infants bind (Fernald and Mazzie 1991's final-position prominence; her lines
                        # end in the focus word, templates.py's rule), so 'look', 'is' and 'the' bind a third as hard as the name (ours)


def _norm(x, axis=None):
    return np.sqrt(np.sum(np.asarray(x, dtype=np.float64) ** 2, axis=axis))


def ground_prior(logits, margin):
    """A202i: the size of the name's prior on its logit: GROUND_SAY readout standard deviations per GROUND_MARGIN of margin. The
    readout's logits set the scale (the seen thing's name competes in the lexical choice's own currency: a day-109 readout's
    logits spread 1.6 with a top gap of 2.6, and a prior of 0.2 could never be said)"""
    sd = float(logits.std()) if hasattr(logits, "std") else float(np.std(np.asarray(logits, dtype=np.float64)))
    return GROUND_SAY * (float(margin) / GROUND_MARGIN) * sd


class GroundingMixin:
    def _ground_on(self):
        return getattr(self.anatomy, "grounding", None) is not None

    def _ground_state(self):
        """the organ's rows and counts, born at first use (zeros: no word has a look); kept as CPU tensors (the save's and the
        determinism check's forms), worked on as the arrays that share their memory"""
        A = getattr(self, "_ground_A", None)
        if A is None or int(A.shape[0]) != int(self.m.vocab):
            k = int(self.anatomy.grounding.size)
            self._ground_A = torch.zeros((int(self.m.vocab), k), dtype=torch.float64)
            self._ground_n = torch.zeros(int(self.m.vocab), dtype=torch.float64)   # (A202e: the hearings' weights, a line's last word's GROUND_FINAL)
            self._ground_s = torch.zeros(int(self.m.vocab), dtype=torch.float64)   # A202b: the sum of its hearings' look sizes
            self._ground_trace = [-1, 0]
            self._ground_stats = {"bind": 0, "cue": 0, "say": 0}
        if getattr(self, "_ground_v", None) != 3:   # (3: A202e's weighted counts)                      # A202d: the rows bound before the consistency law (A202b) carried
            k = int(self.anatomy.grounding.size)                          # looks averaged with a recency weight and no record of their
            self._ground_A = torch.zeros((int(self.m.vocab), k), dtype=torch.float64)   # hearings' sizes, so 'good' read as consistent for
            self._ground_n = torch.zeros(int(self.m.vocab), dtype=torch.float64)        # hundreds of hearings to come: begun again, once
            self._ground_s = torch.zeros(int(self.m.vocab), dtype=torch.float64)        # (fast mapping rebuilds a day's rows in an hour)
            self._ground_v = 3
        if getattr(self, "_ground_mu", None) is None or int(self._ground_mu.shape[0]) != int(self._ground_A.shape[1]):
            # A202k: the grand mean of the looks over every hearing (what the eyes hold while she speaks, whatever the word), begun
            # from the rows there are (their weighted mean), so a life mid-way carries it from its first tick under the law
            n_ = self._ground_n; tot = float(n_.sum())
            self._ground_mu = (self._ground_A * n_[:, None]).sum(0) / tot if tot > 0 else torch.zeros(int(self._ground_A.shape[1]), dtype=torch.float64)
            self._ground_mu_n = tot
        return self._ground_A.numpy(), self._ground_n.numpy(), self._ground_trace

    def _ground_centred(self, X):
        """A202k: THE LOOK EVERY WORD SHARES IS NO WORD'S LOOK. Day 109's rows: 'the' (160 hearings), 'see' (145), 'you', 'is', 'look',
        'it' all carried one look, the blue-green of her sweater (the fovea rests on her body while she speaks), 'see' primed 145 times
        in day 110's first hour and said 36; a word heard over the speaker's own appearance at every hearing learns that appearance as
        its meaning. A cue present on every trial earns no association (Rescorla and Wagner 1972's blocking; the cross-situational
        learner's competition among words for a referent, Yu and Smith 2007): the grand mean of the looks is taken off each word's row
        and off the look in view at the cue and the name, so what the speaker's presence adds to every hearing falls out, and a word
        whose look is only that baseline has no look left (below GROUND_FLOOR). The rows themselves stay the raw running means"""
        return np.asarray(X, dtype=np.float64) - self._ground_mu.numpy()

    def _ground_consist(self, w=None):
        """A202b: a word's look's consistency, |mean| / (the mean of its hearings' |look|), 0 with no hearing (for every word when w is None)"""
        A, n, _ = self._ground_state(); s_ = self._ground_s.numpy()
        if w is None:
            return np.where(n > 0, _norm(A, axis=1) * n / np.maximum(s_, 1e-12), 0.0)
        return float(_norm(A[w]) * n[w] / max(float(s_[w]), 1e-12)) if n[w] > 0 else 0.0

    def _ground_skip(self, u):
        g = self.anatomy.grounding
        return u == int(self.sil) or u in tuple(g.skip) or u in tuple(getattr(self, "bans", ()) or ())

    def _ground_sense(self, frame):
        """(2) THE CUE, at the senses: the heard word's look found in the periphery -> frame.obs[g.obs] = [fired, yaw, pitch]"""
        g = getattr(self.anatomy, "grounding", None)
        if g is None:
            return
        A, n, tr = self._ground_state()
        out = np.zeros(3)
        w, left = int(tr[0]), int(tr[1])
        if left > 0 and w >= 0 and int(n[w]) >= GROUND_MIN_N and self._ground_consist(w) >= GROUND_CONSIST:
            T = self._ground_centred(A[w]); nt = _norm(T)                   # A202k: the word's look past what every word shares
            if nt > GROUND_FLOOR:
                feats, dirs = g.periphery(frame)
                feats = self._ground_centred(feats); dirs = np.asarray(dirs, dtype=np.float64)
                nf = _norm(feats, axis=1)
                cos = (feats @ (T / nt)) / np.maximum(nf, 1e-9)
                cos = np.where(nf > GROUND_FLOOR, cos, -1.0)
                if cos.size >= 2:
                    order = np.argsort(-cos, kind="stable")
                    b, s = int(order[0]), int(order[1])
                    if cos[b] >= GROUND_MATCH and cos[b] - cos[s] >= GROUND_MARGIN:
                        out = np.array([1.0, float(dirs[b][0]), float(dirs[b][1])])
                        self._ground_stats["cue"] += 1
            tr[1] = left - 1
        frame.obs[g.obs] = out
        self._ground_now = [float(x_) for x_ in out]                     # (plain numbers: the working day's hasher reads them)

    def _ground_learn(self, u, frame):
        """(1) THE BINDING, at the tick's end: the symbol heard this tick bound to the fovea's look; the word's trace armed"""
        g = getattr(self.anatomy, "grounding", None)
        if g is None or frame is None:
            return
        A, n, tr = self._ground_state()
        u = int(u)
        if u == int(getattr(self, "end_id", -1)) or u == int(getattr(self, "eot", -1)):
            last_ = getattr(self, "_ground_last", None)                    # A202e: the line over: its last word's hearing weighs GROUND_FINAL
            if last_ is not None and int(getattr(self, "_ground_line_n", 0)) >= 2:   # A202f: within a line of two words or more: a line of
                self._ground_bind(int(last_[0]), np.asarray(last_[1]), GROUND_FINAL - 1.0)   # one word ('oh.') has no final position (day 106:
            self._ground_last = None; self._ground_line_n = 0                           # 'oh' primed 37 times, the names 4)
            return
        if self._ground_skip(u):
            return
        T = np.asarray(g.appearance(frame), dtype=np.float64)
        self._ground_bind(u, T, 1.0)
        self._ground_line_n = int(getattr(self, "_ground_line_n", 0)) + 1   # (A202f: the line's words so far)
        self._ground_last = (u, [float(x_) for x_ in T]) if _norm(T) > GROUND_FLOOR else None   # (plain numbers: the working day's hasher)
        tr[0] = u; tr[1] = GROUND_TRACE

    def _ground_bind(self, u, T, w):
        """one hearing of word u with look T at weight w: the weighted running mean (A202b, A202e), when the fovea holds a look"""
        A, n, _ = self._ground_state(); nT = _norm(T)
        if nT > GROUND_FLOOR and w > 0.0:
            n[u] += w; rate = max(w / float(n[u]), GROUND_RATE * w)
            A[u] += min(1.0, rate) * (T - A[u]); self._ground_s.numpy()[u] += w * nT
            self._ground_stats["bind"] += 1
            mu = self._ground_mu.numpy(); tot = float(self._ground_mu_n) + w   # A202k: the grand mean over every hearing (its drift
            mu += max(w / tot, GROUND_RATE * w / 10.0) * (np.asarray(T, dtype=np.float64) - mu)   # a tenth of a row's: the baseline
            self._ground_mu_n = tot                                                                 # moves slower than any word)

    def _ground_say(self, frame):
        """(3) THE NAME, at the voice's choice: (the word whose look fills the fovea, its margin), or None"""
        g = getattr(self.anatomy, "grounding", None)
        if g is None or frame is None:
            return None
        A, n, _ = self._ground_state()
        T = self._ground_centred(g.appearance(frame)); nt = _norm(T)      # A202k: the look in view past what every word shares
        if nt <= GROUND_FLOOR:
            return None
        cons = self._ground_consist()
        ok = (n >= GROUND_MIN_N) & (cons >= GROUND_CONSIST)
        if int(ok.sum()) < 1:
            return None
        A = self._ground_centred(A)
        na = _norm(A, axis=1)
        cos = (A @ (T / nt)) / np.maximum(na, 1e-9)
        cos = np.where(ok & (na > GROUND_FLOOR), cos * cons, -1.0)         # A202b: weighed by the word's consistency: the word that
        order = np.argsort(-cos, kind="stable")                             # always came with this look over one that sometimes did
        b = int(order[0]); second = float(cos[order[1]]) if cos.size >= 2 else -1.0
        if cos[b] >= GROUND_MATCH and cos[b] - second >= GROUND_MARGIN and not self._ground_skip(b):
            self._ground_stats["say"] += 1
            return b, float(cos[b] - second)
        return None

    def ground_report(self):
        """the organ's counts and the words with a look (for the record and the day's report)"""
        if not self._ground_on():
            return None
        A, n, tr = self._ground_state()
        return {**{k_: int(v_) for k_, v_ in self._ground_stats.items()}, "words": int(np.sum((n >= GROUND_MIN_N) & (self._ground_consist() >= GROUND_CONSIST))),
                "trace": [int(tr[0]), int(tr[1])], "mu": round(float(_norm(self._ground_mu.numpy())), 3)}   # (A202k: the shared look's size)
