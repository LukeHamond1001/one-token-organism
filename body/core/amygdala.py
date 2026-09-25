"""the amygdala: the valence tagger (a named organ of the core, and a mixin of `Life`, body/life.py; the owner's decision 2; the core
refactor's step R7d; docs/SIM_DESIGN.md 7.4, 10, A16). From what the body senses now it learns fast which cues predict good and bad, from
each of the body's reward sources, over the next few seconds, and marks each moment with its emotional weight, the tag: what the moment
is expected to bring plus what it brought. The tag sets how strongly the moment is written to the fast memory (R7b's writes, their later
boosts), how likely the night is to replay it (R8) and whether orienting turns toward or away (R7e). It never makes reward, changes
reward, produces dopamine, enters any gate's credit or trains the cortex: it cannot pay the child for anything. Its constants are
physiology.py's AMYG, absent from a body's cfg unless given; its switch `amyg` is off by its absence, so the language body builds none
of it.

BIOLOGY: the amygdala as the Pavlovian cue-to-outcome learner, with LeDoux's fast thalamic "low road" and slower cortical "high road";
separate positive- and negative-valence neurons in the basolateral amygdala (Paton et al. 2006; Namburi et al. 2015; Kim et al. 2016);
the amygdala strengthening the storage of emotional events (McGaugh 2004); behavioural tagging (Frey and Morris 1997; Redondo and Morris
2011); rewarded and emotional experiences replayed more in sleep (Ambrose, Pfeiffer and Foster 2016; Girardeau et al. 2017);
conditioned orienting through the central amygdala (Gallagher and Holland 1999); the learning law as the least-squares (Kalman) form of
the Rescorla-Wagner rule (Dayan and Kakade 2001).

THE ORGAN (`Amygdala`, m.amyg, built last by the organs; every number float64, every tensor a buffer saved with the body; it stays on the
host, as the critics' evidence does, and draws no random number):
- WHAT IT READS, each tick: x = [C / sqrt(d), the event lines, 1]: the high road, the cortex's stream as the tick's choice reads it,
  detached (everything the cortex has made of the eyes, the ears, the joints, touch, balance and the charge: a voice's tone, the
  parent's face turning, the charge falling); the low road, the anatomy's born event lines (R7a, the same declaration the critics read:
  13 for the G1); the level. 526 inputs for the G1 at d 512.
- ITS HEADS: one per reward source and sign (each RewardSource declares its signs and whether it reaches the amygdala): the G1's 5, face
  +, face -, pain -, charge +, charge -. Head h, of source s and sign sg, receives u = max(0, sg x the source's term) and forecasts
  y_t = sum over k >= 1 of g^(k-1) u_(t+k), g dopamine's own discount (0.9375), the sum not cut off. Good and bad kept apart: a single
  signed forecast would net to zero a cue that brings a smile and then pain, although that cue matters most.
- ITS LAW, the critics' least squares with a shorter memory (it does not bootstrap: the value learner's deadly triad does not apply): the
  eligibility trace e_t = g e_(t-1) + x_t; the evidence b <- beta b + e_(t-1) u_t^T, A <- beta A + x_t x_t^T, beta = 1 - 1/tau_a
  (tau_a the ladder's clock at band amyg_clock 6: 4,096 ticks), the exact backward form of regressing y on x, so an outcome enters the
  evidence on the tick it is felt; every amyg_every 8 ticks the solve W = (A + R)^-1 b, R_ii = amyg_prior 0.3 x tau_a x each input's
  running variance (its rate 1 / min(n, tau_a)), the level free, a Cholesky solve falling back to least squares; the forecast
  y^ = max(0, W^T x) per head. An input that has never varied is left out of the solve, its weight 0 (the minimum-norm answer: its row
  of the evidence is zero or a copy of the level's), so the Cholesky solve holds where a line has never fired (THE BUILDER'S READING,
  for the lead). It learns only while awake; its evidence is kept across the night, its trace begins afresh each morning.
- ITS RELIABILITY, the voice it has earned: each head's rho is the running correlation of its forecast with the realized target, the
  target finalized tag_reach 64 ticks later (the sum of the outcomes at ages 1-64) or at nightfall (what arrived before it), the moments
  decaying over amyg_rel_tau 36,000 ticks (the face organ's own measure), clipped to 0-1 and 0 until amyg_pairs 64 finalized pairs.
  Anticipated good A+ = the sum over positive heads of rho y^; anticipated bad A- over the negative heads; the net valence N = A+ - A-.
- THE TAG: tag_t = min(the judgment's clip 2, A+ + A- + R_t), R_t the sum of |term| over the sources that reach it (what was felt this
  tick; body/core/frames.py `_received_tag`), in reward units. At a write the tag is this tick's received part plus the forecast made the
  tick before (R7b's gate and strength). No running-mean scaling (defect 7).

THE LIFE'S PART (`AmygdalaMixin`): a life whose switch is on checks its organs' amygdala against its anatomy (its inputs and heads) and
runs `_amygdala` each awake tick; `_orient_gain` (R7e, body/core/cord.py) and the store's later boosts (R7d, body/core/frames.py) read
what it made; nothing else takes the tag or the forecast: not reward, dopamine, mood, stress, the gates' credit, the actor, working
memory's latch, the cortex's loss, REM's scorer or anything the parent sees.

THE NIGHT'S SIDE (R8's to wire; its tests are the amygdala's, 7.4 items 7 and 8): `tag_star` (the tag reaching back, tag*_t = the largest
g^k tag_(t+k) for k < 64), `episode_entries` (an episode's entry [its mean surprise x (1 + |dopamine|)] x (1 + T_e) over a saved,
bias-corrected running mean of 64 episodes), `night_draw` (the tagged dreamt first: every episode with T_e >= 1 once, highest first, at most
half the night; the rest drawn by entry) and `act_pred_night_weight` (act_pred's lesson at a replayed position weighted clip(1 + G, 0, 1),
G the replayed dopamine's credit: acts followed by net harm are not taught as acts to make)."""
import math

import torch
from torch import nn

from .physiology import AMYG, FRAMES


def amygdala_spec(anatomy, cfg):
    """WHAT THE ORGANS BUILD (Organs(..., amygdala=)): None while the switch is off (the language body's: its cfg has no `amyg`); else the
    number of its event lines, its heads (source name, sign) in the anatomy's order of sources and signs, and its horizon (tag_reach)"""
    c = cfg or {}
    if not int(c.get("amyg", AMYG["amyg"])):
        return None
    heads = [(s.name, float(sg)) for s in anatomy.rewards if s.amyg for sg in s.signs]
    if not heads:
        raise ValueError("the amygdala is switched on (amyg 1) and no reward source of the anatomy reaches it (RewardSource.amyg)")
    return dict(events=len(anatomy.events or ()), heads=heads, reach=int(c.get("tag_reach", FRAMES["tag_reach"])))


class Amygdala(nn.Module):
    """THE ORGAN (the module's doc): its evidence, weights, trace, input statistics, reliability's moments and pending forecasts, every one
    a float64 (or count) buffer. `n_in` its inputs (d + the event lines + the level), `heads` its (source, sign) pairs, `reach` the
    horizon its forecasts are finalized at"""

    def __init__(self, n_in, heads, reach):
        super().__init__()
        n, H, R = int(n_in), len(heads), int(reach)
        if n < 2 or H < 1 or R < 1:
            raise ValueError(f"Amygdala: {n} inputs, {H} heads, a horizon of {R}")
        self.n_in, self.heads, self.reach = n, tuple((str(a), float(b)) for a, b in heads), R
        f8 = torch.float64
        self.register_buffer("A", torch.zeros(n, n, dtype=f8))            # the inputs' second moment, forgetting
        self.register_buffer("b", torch.zeros(n, H, dtype=f8))            # the inputs' trace against the outcomes, forgetting
        self.register_buffer("W", torch.zeros(n, H, dtype=f8))            # the heads' weights, solved every amyg_every ticks
        self.register_buffer("e", torch.zeros(n, dtype=f8))               # the eligibility trace of the inputs
        self.register_buffer("mu", torch.zeros(n - 1, dtype=f8))          # each input's running mean and variance (the prior's metric)
        self.register_buffer("var", torch.zeros(n - 1, dtype=f8))
        self.register_buffer("cnt", torch.zeros((), dtype=f8))
        self.register_buffer("rel", torch.zeros(H, 6, dtype=f8))          # the reliability's moments: count, f, y, f^2, y^2, f y
        self.register_buffer("pairs", torch.zeros(H, dtype=torch.long))   # the finalized pairs, counted exactly
        self.register_buffer("ring_f", torch.zeros(R, H, dtype=f8))       # the pending forecasts, the newest first (age 1 at index 0)
        self.register_buffer("ring_y", torch.zeros(R, H, dtype=f8))       # and their targets so far
        self.register_buffer("ring_n", torch.zeros((), dtype=torch.long))
        self.register_buffer("n_tick", torch.zeros((), dtype=torch.long))  # the awake ticks it has lived (the solve's clock)
        self.register_buffer("n_solve", torch.zeros((), dtype=torch.long))

    def _apply(self, fn, recurse=True):
        """the amygdala stays on the host whatever the organs are sent to (the night's device): its numbers are float64, as the critics'"""
        return self

    def solve(self, prior, tau):
        """W = (A + R)^-1 b over the inputs that have varied and the level (the module's doc): R_ii = prior x tau x var_i, the level free;
        Cholesky, else least squares; the rest of W 0. An input that has not varied has its row and column of the system made the
        identity's and its evidence 0, so the system decouples: the solve over the others is theirs alone and its weight is 0 (the
        reduced system's answer, without copying it out)"""
        A_ = self.A.clone()
        A_.diagonal()[:-1].add_(self.var.clamp_min(0.0), alpha=float(prior) * float(tau))   # R_ii = prior x tau x var_i; the level free
        b_ = self.b.clone()
        off = (self.var <= 0.0).nonzero().flatten()
        if off.numel():
            A_[off, :] = 0.0; A_[:, off] = 0.0; A_[off, off] = 1.0; b_[off] = 0.0
        try:
            L_ = torch.linalg.cholesky(A_)
            w = torch.cholesky_solve(b_, L_)
        except Exception:
            w = torch.linalg.lstsq(A_, b_).solution
        if off.numel():
            w[off] = 0.0
        self.W.copy_(w); self.n_solve.add_(1)

    def _finalize(self, f, y, rel_tau):
        """one finalized pair per head (the forecast `f` [H], the target `y` [H]) into the reliability's moments"""
        d = 1.0 - 1.0 / float(rel_tau); m = self.rel
        m.mul_(d)
        m[:, 0] += 1.0; m[:, 1] += f; m[:, 2] += y; m[:, 3] += f * f; m[:, 4] += y * y; m[:, 5] += f * y
        self.pairs.add_(1)

    def reliability(self, min_pairs):
        """each head's rho: the running correlation of its forecast with its target, clipped to 0-1, 0 until `min_pairs` pairs"""
        m = self.rel; n = m[:, 0].clamp_min(1e-12)
        mf, my = m[:, 1] / n, m[:, 2] / n
        vf = (m[:, 3] / n - mf * mf).clamp_min(1e-9); vy = (m[:, 4] / n - my * my).clamp_min(1e-9); cov = m[:, 5] / n - mf * my
        rho = (cov / torch.sqrt(vf * vy)).clamp(0.0, 1.0)
        return torch.where(self.pairs >= int(min_pairs), rho, torch.zeros_like(rho))

    @torch.no_grad()
    def step(self, x, u, gamma, beta, prior, tau, every, rel_tau):
        """ONE AWAKE TICK (the module's doc): `x` [n_in] this tick's input, `u` [H] the outcomes felt this tick (each head's max(0, sign x
        term)); `gamma` the discount, `beta` the evidence's forgetting, `prior` and `tau` the ridge's, `every` the solve's period,
        `rel_tau` the reliability's memory. In order: the outcome enters the evidence with the trace of the inputs before it, the input
        its own second moment, the trace and the input statistics move; the pending forecasts take the outcome (age k at g^(k-1)) and the
        one `reach` ticks old is finalized; every `every` ticks the solve; the forecast made now joins the pending. Returns y^ [H]"""
        x = x.to(torch.float64); u = u.to(torch.float64)
        felt = bool(u.any())
        if felt:
            self.b.addr_(self.e, u, beta=float(beta))
        else:
            self.b.mul_(float(beta))                                   # no outcome: the evidence only forgets
        self.A.addr_(x, x, beta=float(beta))
        self.e.mul_(gamma).add_(x)
        self.cnt.add_(1.0); eta = 1.0 / min(float(self.cnt), float(tau))
        xv = x[:-1]; dv = xv - self.mu; self.mu.add_(dv, alpha=eta); self.var.add_(dv * (xv - self.mu) - self.var, alpha=eta)
        n_ = int(self.ring_n); R = self.reach
        if n_:
            if felt:
                w = float(gamma) ** torch.arange(n_, dtype=torch.float64)
                self.ring_y[:n_] += w[:, None] * u[None, :]
            if n_ == R:
                self._finalize(self.ring_f[R - 1].clone(), self.ring_y[R - 1].clone(), rel_tau)
        self.n_tick.add_(1)
        if int(self.n_tick) % int(every) == 0:
            self.solve(prior, tau)
        yhat = (x @ self.W).clamp_min(0.0)
        self.ring_f.copy_(torch.roll(self.ring_f, 1, 0)); self.ring_y.copy_(torch.roll(self.ring_y, 1, 0))
        self.ring_f[0] = yhat; self.ring_y[0] = 0.0
        self.ring_n.fill_(min(n_ + 1, R))
        return yhat

    @torch.no_grad()
    def nightfall(self, rel_tau):
        """the night: every pending forecast finalized with what arrived before it, the pending let go, the trace begun afresh (the
        evidence and the weights kept)"""
        for i in range(int(self.ring_n) - 1, -1, -1):                 # the oldest first
            self._finalize(self.ring_f[i].clone(), self.ring_y[i].clone(), rel_tau)
        self.ring_f.zero_(); self.ring_y.zero_(); self.ring_n.zero_(); self.e.zero_()


class AmygdalaMixin:
    def _amyg_const(self, k):
        """an amygdala constant: the body's cfg when it was given, else AMYG's"""
        return self.cfg.get(k, AMYG[k])

    def _amyg_on(self):
        """the switch `amyg` (step R7d): 0 for the diary, whose cfg holds no such key"""
        return bool(int(self.cfg.get("amyg", AMYG["amyg"])))

    def _amyg_attach(self):
        """BORN OR LOADED WITH THE SWITCH ON: the organs' amygdala must be the one the anatomy declares (its inputs: d, the event lines, the
        level; its heads: the sources that reach it, each sign); with the switch off, organs that hold one are refused (a body is born
        with its switches: SIM_DESIGN.md A20)"""
        org = self.m._modules.get("amyg")
        if not self._amyg_on():
            if org is not None:
                raise ValueError("Life: the organs hold an amygdala (m.amyg) and the life's constants switch it off (amyg 0)")
            return
        spec = amygdala_spec(self.anatomy, self.cfg)
        want = (int(self.m.d) + int(spec["events"]) + 1, tuple((a, float(b)) for a, b in spec["heads"]), int(spec["reach"]))
        if org is None or not isinstance(org, Amygdala) or (org.n_in, org.heads, org.reach) != want:
            raise ValueError(f"Life: the organs' amygdala ({None if org is None else (org.n_in, org.heads, org.reach)}) is not the one the anatomy "
                             f"declares ({want}; built by Organs(..., amygdala=amygdala_spec(anatomy, cfg)))")

    def _amyg_input(self, C1, frame):
        """x = [C / sqrt(d), the event lines, 1] (float64, detached): the high road, the low road, the level"""
        d = float(self.m.d)
        ev = self._event_lines(frame)
        return torch.cat([C1.detach().to("cpu", torch.float64) / math.sqrt(d), torch.tensor(ev, dtype=torch.float64),
                          torch.ones(1, dtype=torch.float64)])

    def _amygdala(self, C1, frame):
        """THE AMYGDALA'S TICK (the module's doc; after the critics' lesson and the face organ, before the choice): its input and the
        outcomes felt this tick into its law; its reliable forecasts' anticipated good and bad, the net valence and the tag, kept for the
        tick (`_amyg_now`), and the forecast part of the tick before (`_amyg_prev`, the tag at a write)"""
        org = self.m.amyg
        x = self._amyg_input(C1, frame)
        terms = getattr(self, "_terms_now", None) or {}
        u = torch.tensor([max(0.0, float(sg) * float(terms.get(name, 0.0))) for name, sg in org.heads], dtype=torch.float64)
        tau = float(self.m.clocks[int(self._amyg_const("amyg_clock"))])
        yhat = org.step(x, u, self._tag_gamma(), 1.0 - 1.0 / tau, float(self._amyg_const("amyg_prior")), tau,
                        int(self._amyg_const("amyg_every")), float(self._amyg_const("amyg_rel_tau")))
        rho = org.reliability(int(self._amyg_const("amyg_pairs")))
        rf = rho * yhat
        Ap = float(sum(float(v) for v, (_, sg) in zip(rf, org.heads) if sg > 0.0))
        Am = float(sum(float(v) for v, (_, sg) in zip(rf, org.heads) if sg < 0.0))
        R = float(getattr(self, "_rtag_now", 0.0))
        cap = self.anatomy.rewards[0].clip
        tag = Ap + Am + R
        tag = min(float(cap), tag) if cap is not None else tag
        prev = getattr(self, "_amyg_now", None)
        self._amyg_prev = (float(prev["Ap"]) + float(prev["Am"])) if prev is not None else 0.0
        self._amyg_now = {"Ap": Ap, "Am": Am, "N": Ap - Am, "tag": tag, "R": R}

    def _amyg_nightfall(self):
        """the night (called as it begins): the pending forecasts finalized and let go, the trace afresh; the tick's own let go"""
        self.m.amyg.nightfall(float(self._amyg_const("amyg_rel_tau")))
        self._amyg_now = None; self._amyg_prev = 0.0


# ---------------- the night's side (R8 wires them; their tests are the amygdala's, SIM_DESIGN.md 7.4 items 7 and 8) ----------------

def tag_star(tags, gamma, reach):
    """THE TAG REACHES BACK (7.4; behavioural tagging): tag*_t = the largest gamma^k x tag_(t+k) for k = 0 .. reach - 1 within the sequence
    `tags` (a day's record's tag column); a list of floats"""
    T = len(tags); out = [0.0] * T
    for t in range(T):
        best = 0.0
        for k in range(min(int(reach), T - t)):
            v = (float(gamma) ** k) * float(tags[t + k])
            if v > best:
                best = v
        out[t] = best
    return out


def episode_entries(episodes, surprise, delta, tstar, mean_state, tau=64):
    """THE NIGHT'S ENTRIES (7.4 item 2; R8): for each episode (start, end) of the day's record, [the mean over its ticks of surprise x
    (1 + |dopamine|)] x (1 + T_e), T_e the largest tag* over its ticks and the `reach` ticks past its end (tag* already reaches back, so
    T_e is the largest tag* from its start to its end), over a running mean of `tau` episodes, bias-corrected (defect 7's fix: m_n = b
    m_n-1 + (1 - b) x_n over (1 - b^n), b = 1 - 1/tau), the mean before each episode (the first 1). `mean_state` = [m, n], moved in place
    (saved with the body by R8). Returns [(entry, T_e)]"""
    beta = 1.0 - 1.0 / float(tau); out = []
    for (a, b) in episodes:
        s_ = sum(float(surprise[t]) * (1.0 + abs(float(delta[t]))) for t in range(a, b)) / max(1, b - a)
        T_e = max(float(tstar[t]) for t in range(a, b)) if b > a else 0.0
        x = s_ * (1.0 + T_e)
        m_, n_ = float(mean_state[0]), int(mean_state[1])
        mhat = m_ / (1.0 - beta ** n_) if n_ > 0 else 0.0
        out.append((x / mhat if mhat > 1e-12 else 1.0, T_e))
        mean_state[0] = beta * m_ + (1.0 - beta) * x; mean_state[1] = n_ + 1
    return out


def night_draw(entries, tags, n, gen):
    """THE TAGGED COME FIRST (7.4 item 2; R8): the night's `n` dreams over episodes with `entries` and T_e `tags`: every episode with T_e >= 1
    dreamt once before the weighted draw, highest first, taking at most half the night; the rest drawn by entry, with replacement, on the
    body's generator `gen`. Returns the episodes' indices in the order dreamt"""
    first = sorted([i for i, T_ in enumerate(tags) if float(T_) >= 1.0], key=lambda i: (-float(tags[i]), i))[: int(n) // 2]
    rest = int(n) - len(first)
    w = torch.tensor([max(0.0, float(e)) for e in entries], dtype=torch.float64)
    drawn = torch.multinomial(w / w.sum(), rest, replacement=True, generator=gen).tolist() if rest > 0 and float(w.sum()) > 0 else []
    return first + [int(i) for i in drawn]


def act_pred_night_weight(G):
    """act_pred's lesson at a replayed position (7.4 item 2; R8): clip(1 + G, 0, 1), G the replayed dopamine's credit over the next 64
    ticks: an act followed by net harm (G <= -1) is not taught as an act to make (weighting by the tag, as first written, would have taught
    the acts that led to a fall)"""
    return max(0.0, min(1.0, 1.0 + float(G)))
