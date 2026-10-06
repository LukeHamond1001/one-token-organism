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

GROUND_RATE = 0.15      # the Hebbian step of a row toward this hearing's look (ours: a word's look settles in about seven hearings)
GROUND_MIN_N = 3        # hearings before a word's look is trusted for the cue and the name (ours: fast mapping's few)
GROUND_TRACE = 20       # ticks (3 s) a heard word keeps drawing the eyes toward its look (ours: the orienting response's span)
GROUND_FLOOR = 0.02     # the least contrast (opponent units) that counts as a look at a thing, in the fovea or a cell (ours)
GROUND_MATCH = 0.6      # the least cosine between a word's look and a cell's for the cue or the name (ours)
GROUND_MARGIN = 0.15    # the best match's lead over the second (ours)
GROUND_SAY = 1.0        # the logit prior on the word whose look fills the fovea, times the margin (ours)


def _norm(x, axis=None):
    return np.sqrt(np.sum(np.asarray(x, dtype=np.float64) ** 2, axis=axis))


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
            self._ground_n = torch.zeros(int(self.m.vocab), dtype=torch.int64)
            self._ground_trace = [-1, 0]
            self._ground_stats = {"bind": 0, "cue": 0, "say": 0}
        return self._ground_A.numpy(), self._ground_n.numpy(), self._ground_trace

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
        if left > 0 and w >= 0 and int(n[w]) >= GROUND_MIN_N:
            T = A[w]; nt = _norm(T)
            if nt > GROUND_FLOOR:
                feats, dirs = g.periphery(frame)
                feats = np.asarray(feats, dtype=np.float64); dirs = np.asarray(dirs, dtype=np.float64)
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
        if self._ground_skip(u):
            return
        T = np.asarray(g.appearance(frame), dtype=np.float64)
        if _norm(T) > GROUND_FLOOR:
            A[u] += GROUND_RATE * (T - A[u]); n[u] += 1
            self._ground_stats["bind"] += 1
        tr[0] = u; tr[1] = GROUND_TRACE

    def _ground_say(self, frame):
        """(3) THE NAME, at the voice's choice: (the word whose look fills the fovea, its margin), or None"""
        g = getattr(self.anatomy, "grounding", None)
        if g is None or frame is None:
            return None
        A, n, _ = self._ground_state()
        T = np.asarray(g.appearance(frame), dtype=np.float64); nt = _norm(T)
        if nt <= GROUND_FLOOR:
            return None
        ok = n >= GROUND_MIN_N
        if int(ok.sum()) < 1:
            return None
        na = _norm(A, axis=1)
        cos = (A @ (T / nt)) / np.maximum(na, 1e-9)
        cos = np.where(ok & (na > GROUND_FLOOR), cos, -1.0)
        order = np.argsort(-cos, kind="stable")
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
        return {**{k_: int(v_) for k_, v_ in self._ground_stats.items()}, "words": int(np.sum(n >= GROUND_MIN_N)),
                "trace": [int(tr[0]), int(tr[1])]}
