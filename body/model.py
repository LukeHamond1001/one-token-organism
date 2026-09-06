"""the organs of the second body (BODY_SPEC.md §2).

Everything learned lives in `Organs` (an nn.Module, saved with the body). The hippocampus
is `Store`, a table of slots whose tensors are saved beside the weights. Nothing here decides
behaviour by hand: constants are physiology (disclosed in the spec) or plumbing.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

CLOCKS = (1, 4, 16, 64, 256, 1024, 4096, 16384)      # the PFC ladder's clocks, in ticks


def sigreg(z, n_dirs=64, grid=17, span=5.0):
    """SIGReg (LeJEPA), sketched: random unit directions, each projection tested against N(0,1)
    by the Epps-Pulley statistic on its characteristic function; the collapse guard for a
    latent prediction with no tokens in it. z [N, d] (normalize before calling)."""
    N, d = z.shape
    if N < 4:
        return z.new_zeros(())
    g = torch.Generator(device="cpu").manual_seed(int(N) * 7919 + int(d))
    P = F.normalize(torch.randn(d, n_dirs, generator=g), dim=0).to(z.device, z.dtype)
    x = z @ P
    ts = torch.linspace(-span, span, grid, device=z.device, dtype=z.dtype)
    tx = x.unsqueeze(-1) * ts
    re = torch.cos(tx).mean(0); im = torch.sin(tx).mean(0)
    target = torch.exp(-0.5 * ts * ts)
    w = target / math.sqrt(2.0 * math.pi)
    stat = (((re - target) ** 2 + im ** 2) * w).sum(-1) * (2.0 * span / (grid - 1))
    return float(N) * stat.mean()


class Store:
    """THE HIPPOCAMPUS: slots of (key, value, strength, who). Keys and values are unit vectors.
    Read = strength-weighted attention over keys at the organ's own temperature (pattern
    completion); write = a new slot or a merge into a near-identical one; fade = strengths
    shrink each night and slots far below the store's own mean are forgotten."""

    def __init__(self, d, cap=8192, temp=0.05, device="cpu", read_strength=0.0):
        self.d, self.cap, self.temp, self.dev = d, int(cap), float(temp), device
        self.read_strength = float(read_strength)      # weight of log-strength in the read: 0 = recall by content alone
        self.K = torch.zeros(0, d, device=device)
        self.V = torch.zeros(0, d, device=device)
        self.S = torch.zeros(0, device=device)
        self.W = torch.zeros(0, dtype=torch.long, device=device)
        self.B = torch.zeros(0, dtype=torch.bool, device=device)   # THE BOUNDARY: this slot's symbol ended the world's utterance
        self.Bs = torch.zeros(0, dtype=torch.bool, device=device)  # and this slot's symbol began one (the first after a pause)
        self.Bq = torch.zeros(0, dtype=torch.bool, device=device)  # THE SEAM: this slot's symbol is an utterance's first, under the last line's faded context

    def n(self):
        return int(self.K.shape[0])

    @torch.no_grad()
    def read(self, q, adapt=None, end_vec=None):
        """q [d] -> (the recalled next embedding [d], its norm the confidence in 0..1, winner index);
        adapt [n] (optional) multiplies strengths: the recall adaptation a dream runs under;
        end_vec [d] (optional): THE MARKS SPEAK IN THE RECALL. A seam slot (an utterance's first symbol written under
        the last line's faded context, whose unit key is the line's own) recalls not that symbol but the turn's end
        (the unit direction of <eot_human>): the memory that a new utterance began here is the memory that the turn
        ended here. Without it the recall after a whole line is the next line's first letter, the seam the mouth
        said ('downg', served body day 10; runs 61/62 neutral for this reason)"""
        if self.n() == 0:
            return torch.zeros(self.d, device=self.dev), 0.0, -1
        # A DOT PRODUCT, not a cosine: the keys are unit directions, the query is the context as it is,
        # so its norm is the inverse temperature of the recall. Normalised, a context faded to nothing
        # (norm 0.01 after 24 quiet ticks) still recalled at confidence 0.87, its faint tail amplified
        # into a full direction, and the mouth chained across lines through the pauses (run 17).
        sims = self.K @ q.to(self.dev).float()                          # [n]
        S = self.S if adapt is None else self.S * adapt
        # RECALL BY CONTENT: the match decides; strength decides how long a memory lasts and which are
        # replayed (with it in the read, the most-reinforced memory won every cue: measured 2026-09-03)
        logits = sims / self.temp
        if self.read_strength > 0 or adapt is not None:
            logits = logits + (self.read_strength if adapt is None else 1.0) * torch.log(S + 1e-6) * (1.0 if adapt is not None else 1.0)
        w = torch.softmax(logits, 0)
        # the recall is the attended mean of unit values: its norm is the agreement among the memories
        # attended, the calibrated confidence (the largest weight understated it once duplicate slots
        # that no longer merge split the mass eight ways at conf 0.11, all of them saying 'g': run 15)
        V = self.V
        if end_vec is not None and bool(self.Bq.any()):
            V = torch.where(self.Bq.unsqueeze(1), end_vec.to(self.dev).float().unsqueeze(0).expand_as(self.V), self.V)
        pred = w @ V
        self._last_w = w
        return pred, float(pred.norm()), int(w.argmax())

    @torch.no_grad()
    def write(self, k, v, strength, who, merge_cos=0.97):
        if strength <= 1e-4:
            return False
        k = F.normalize(k.to(self.dev).float(), dim=0); v = F.normalize(v.to(self.dev).float(), dim=0)
        if self.n() > 0:
            sims = self.K @ k
            # the same memory, stronger: the best-matching slot AMONG THOSE THAT SAY THE SAME (judged by
            # the single best key, a slot with the same key and another value blocked the merge, and each
            # hearing of "ball on" added a voter: six identical slots outvoted an exact match, run 17)
            same = (sims > merge_cos) & ((self.V @ v) > merge_cos)
            if bool(same.any()):
                j = int(torch.where(same, sims, torch.full_like(sims, -2.0)).argmax())
                self.S[j] += float(strength)
                return True
        self.K = torch.cat([self.K, k.unsqueeze(0)]); self.V = torch.cat([self.V, v.unsqueeze(0)])
        self.S = torch.cat([self.S, torch.tensor([float(strength)], device=self.dev)])
        self.W = torch.cat([self.W, torch.tensor([int(who)], device=self.dev)])
        self.B = torch.cat([self.B, torch.zeros(1, dtype=torch.bool, device=self.dev)])
        self.Bs = torch.cat([self.Bs, torch.zeros(1, dtype=torch.bool, device=self.dev)])
        self.Bq = torch.cat([self.Bq, torch.zeros(1, dtype=torch.bool, device=self.dev)])
        if self.n() > self.cap:                                         # the weakest gives way
            keep = torch.argsort(self.S, descending=True)[: self.cap]
            self._keep(keep)
        return True

    @torch.no_grad()
    def _find(self, k, v, merge_cos=0.97):
        if self.n() == 0:
            return -1
        k = F.normalize(k.to(self.dev).float(), dim=0); v = F.normalize(v.to(self.dev).float(), dim=0)
        sims = self.K @ k
        same = (sims > merge_cos) & ((self.V @ v) > merge_cos)
        if not bool(same.any()):
            return -1
        return int(torch.where(same, sims, torch.full_like(sims, -2.0)).argmax())

    @torch.no_grad()
    def mark_boundary(self, k, v, merge_cos=0.97):
        """THE BOUNDARY: the slot that holds this context -> this symbol (the same match as a merge) ended the
        world's utterance; a dream that recalls it ends there (the memory's own event boundary)"""
        j = self._find(k, v, merge_cos)
        if j < 0:
            return False
        self.B[j] = True
        return True

    @torch.no_grad()
    def mark_start(self, k, v, merge_cos=0.97):
        """the slot that holds the first symbol after a pause began an utterance: dreams start at starts (replay
        runs from an episode's onset)"""
        j = self._find(k, v, merge_cos)
        if j < 0:
            return False
        self.Bs[j] = True
        return True

    def _keep(self, idx):
        self.K, self.V, self.S, self.W = self.K[idx], self.V[idx], self.S[idx], self.W[idx]
        self.B, self.Bs, self.Bq = self.B[idx], self.Bs[idx], self.Bq[idx]

    @torch.no_grad()
    def mark_seam(self, k, v, merge_cos=0.97):
        """THE SEAM: the slot that holds an utterance's first symbol under the last line's faded context"""
        j = self._find(k, v, merge_cos)
        if j < 0:
            return False
        self.Bq[j] = True
        return True

    @torch.no_grad()
    def fade(self, f=0.9, floor_rel=0.1):
        """each night: strengths x f; slots below floor_rel x the store's own mean are forgotten"""
        if self.n() == 0:
            return 0
        self.S *= float(f)
        thr = float(floor_rel) * float(self.S.mean())
        keep = torch.nonzero(self.S >= thr).flatten()
        dropped = self.n() - int(keep.numel())
        self._keep(keep)
        return dropped

    @torch.no_grad()
    def self_confidence(self, qnorm=1.0):
        """the store's own mean read confidence, querying each slot with its own key (the reference
        a dream is judged against: it stops when recall is half as sure as a memory of its own)"""
        if self.n() == 0:
            return 0.0
        logits = (float(qnorm) * self.K @ self.K.t()) / self.temp     # each slot read by its own key at a full context's norm
        w = torch.softmax(logits, 1)
        return float((w @ self.V).norm(dim=1).mean())

    @torch.no_grad()
    def sample_starts(self, n, gen=None):
        """dream starts: the utterance onsets the store knows, drawn by strength (an episode replayed from its
        beginning; with fewer onsets than dreams the draw is with replacement: ten onsets gave ten short dreams a
        night and the cortex's trace fell from 60 to 36 of 82, run 54); any memory by strength, without
        replacement, when the store knows no onset. (Half the night drawn from any memory instead, runs 57/58,
        moved the cortex's trace within seed noise and cost the mouth two to four of 48 against runs 59/60.)"""
        if self.n() == 0:
            return []
        n = int(n)
        starts = torch.nonzero(self.Bs).flatten()
        pool = starts if starts.numel() >= 1 else torch.arange(self.n(), device=self.dev)
        p = self.S[pool] / self.S[pool].sum()
        idx = torch.multinomial(p.cpu(), n, replacement=bool(pool.numel() < n), generator=gen)
        return [int(pool[i]) for i in idx]

    def state_dict(self):
        return {"K": self.K.cpu(), "V": self.V.cpu(), "S": self.S.cpu(), "W": self.W.cpu(), "B": self.B.cpu(), "Bs": self.Bs.cpu(), "Bq": self.Bq.cpu(), "temp": self.temp}

    def load_state_dict(self, sd):
        self.K = sd["K"].to(self.dev); self.V = sd["V"].to(self.dev)
        self.S = sd["S"].to(self.dev); self.W = sd["W"].to(self.dev)
        self.B = sd["B"].to(self.dev) if "B" in sd else torch.zeros(self.n(), dtype=torch.bool, device=self.dev)
        self.Bs = sd["Bs"].to(self.dev) if "Bs" in sd else torch.zeros(self.n(), dtype=torch.bool, device=self.dev)
        self.Bq = sd["Bq"].to(self.dev) if "Bq" in sd else torch.zeros(self.n(), dtype=torch.bool, device=self.dev)
        self.temp = float(sd.get("temp", self.temp))


class Block(nn.Module):
    def __init__(self, d, heads):
        super().__init__()
        self.ln1 = nn.LayerNorm(d); self.ln2 = nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, heads, batch_first=True)
        self.mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))

    def forward(self, x, mask):
        h = self.ln1(x)
        a, _ = self.attn(h, h, h, attn_mask=mask, need_weights=False)
        x = x + a
        return x + self.mlp(self.ln2(x))


class Organs(nn.Module):
    """all the learned organs, one module, saved with the body"""

    def __init__(self, vocab, d=256, layers=6, heads=4, window=64, clocks=CLOCKS, birth_act=0.25):
        super().__init__()
        self.vocab, self.d, self.window = int(vocab), int(d), int(window)
        self.clocks = tuple(int(c) for c in clocks)
        nb = len(self.clocks)
        # THE LEXICON: one table, input and readout (cosine); no vocabulary softmax is trained
        self.E = nn.Embedding(self.vocab, d)
        nn.init.normal_(self.E.weight, std=1.0 / math.sqrt(d))
        with torch.no_grad():
            self.E.weight.copy_(F.normalize(self.E.weight, dim=-1))
        self.E.weight.requires_grad_(False)               # a fixed lexicon: no trivial solution to predicting it
        self.who_emb = nn.Embedding(2, d)                 # (unused since the two-bag key; kept for old bodies' files)
        nn.init.normal_(self.who_emb.weight, std=0.3 / math.sqrt(d))
        self.sil_id = None                                # set by Life: the rest symbol
        # THE LAG CODE of the hippocampal context: a fixed permutation of the dimensions. Each symbol
        # shifts the whole context through it before entering at lag 0, so "l" at lag 0 and "l" at lag 1
        # are different directions (theta sequence coding; the mathematics of holographic reduced
        # representations). A bag was blind to order and count: after its own "ball" it recalled the
        # "bal" key at 0.976 against the "ball" key at 0.962 and stuttered the l (run 18, day 1).
        self.register_buffer("perm", torch.randperm(d))
        self.own_gain = 0.5                               # corollary discharge on its own sound in the stream
        self.face_in = nn.Linear(2, d)                    # the caregiver's face and its change, as a sense
        nn.init.zeros_(self.face_in.weight); nn.init.zeros_(self.face_in.bias)
        # THE HIPPOCAMPAL PATHWAY: the store's recall reaches the cortex through one learned map,
        # identity at birth (the pathway exists; the cortex learns to modulate it)
        self.store_in = nn.Linear(d, d, bias=False)
        nn.init.eye_(self.store_in.weight)
        # THE PFC LADDER handed over: the bundle of band states enters the cortex
        self.bundle_in = nn.Linear(nb * d, d)
        nn.init.normal_(self.bundle_in.weight, std=0.02); nn.init.zeros_(self.bundle_in.bias)
        self.in_ln = nn.LayerNorm(d)
        # THE CORTEX: a small transformer over the last `window` steps
        self.blocks = nn.ModuleList([Block(d, heads) for _ in range(int(layers))])
        self.lnf = nn.LayerNorm(d)
        # its forecasts: the next embedding it will receive, and the next bundle
        self.latent_pred = nn.Linear(d, d)
        # born unsure: a forecast of norm about 0.1 (the default init gave norm 4 to 9 of pure noise,
        # a deterministic junk symbol at the mouth until the first lessons shrank it)
        nn.init.normal_(self.latent_pred.weight, std=4e-4); nn.init.zeros_(self.latent_pred.bias)
        self.pfc_pred = nn.ModuleList([nn.Linear(d, d) for _ in range(nb)])
        # THE PFC LADDER: leaky integrators of the stream at each clock, each with a fixed input map
        # (born, like the lexicon, and never trained by the critic's error: features that learn from a
        # bootstrapped error are the deadly triad, and they saturated by day 15 in run 28), a Go/NoGo
        # gate on its own update, and a value head (the critic at that timescale)
        self.band_in = nn.ModuleList([nn.Linear(d, d) for _ in range(nb)])
        self.band_gate = nn.ModuleList([nn.Linear(d, 1) for _ in range(nb)])
        for g in self.band_gate:
            nn.init.zeros_(g.weight); nn.init.constant_(g.bias, 2.0)      # open at birth (sigmoid 0.88)
        self.value = nn.ModuleList([nn.Linear(d, 1) for _ in range(nb)])
        for v in self.value:
            nn.init.zeros_(v.weight); nn.init.zeros_(v.bias)            # V = 0 at birth: the error IS the reward
        # THE RELATIVE VALUE PINNED: a differential band (its clock at or beyond the differential
        # horizon; the mask is set by the life) is a relative value, defined up to a constant, and a
        # linear head over raw states has two directions that constant can walk in: its bias, and the
        # states' mean. Both walked (run 28, day 15: two bands past a thousand with their states at the
        # tanh ceiling). So a differential head has no bias and reads its state centered on a running
        # mean of the states (adaptation): the gradient's persistent direction is gone.
        self.register_buffer("diff", torch.zeros(nb, dtype=torch.bool))
        self.register_buffer("band_mu", torch.zeros(nb, d))
        self.register_buffer("v_scale", torch.ones(nb))           # each band's value's running mean square (the level input's normalizer)
        # THE VENTRAL CRITIC: one relative (average-reward) value over the WHOLE ladder, fast bands and slow, so
        # its error moves with the act itself (the fast bands change within a tick) while it predicts the long run
        # (the reward rate's horizon). The ladder's own slow heads read states that move a thousandth per tick and
        # cannot register an act (runs 37/38); the ventral striatum reads the cue it sees now and predicts far.
        # Discounted, so it has a level (the reward rate over its horizon) that the bias holds; an older body's
        # bias is born at zero. Its lesson may carry an eligibility trace at its own horizon (vcrit_lambda).
        self.vcrit = nn.Linear(nb * d + nb + 1, 1, bias=True)  # over the bands' states, then the eight tonic traces, then the clock
        nn.init.zeros_(self.vcrit.weight); nn.init.zeros_(self.vcrit.bias)
        # which bands feed it (vcrit_bands): all by default; the ceiling instrument of 2026-09-04 read +0.51 for the
        # slow bands alone against +0.38 for all eight, the fast bands adding overfit
        self.register_buffer("vcrit_mask", torch.ones(nb))
        # THE TONIC TRACES (2026-09-06): the felt reward averaged at each of the ladder's clocks, tonic dopamine at eight timescales;
        # the ventral critic may read them (vcrit_traces 1). On two recorded fresh days a head on these eight numbers alone read the
        # 1024-return at +0.83/+0.59 where every head on the bands' states, fit the same way, read the next day at random sign.
        self.register_buffer("r_tr", torch.zeros(nb)); self.register_buffer("r_tr_prev", torch.zeros(nb))
        self.vcrit_traces = False
        # THE CLOCK (2026-09-06): the body's own time of day, its sleep pressure over the threshold that brings the night (adenosine;
        # a robot's uptime); the ventral critic may read it (vcrit_clock 1). The one regularity of the return that carries across days
        # is the day's profile: on the diary's days the 1024-return falls across the day at -0.8..-0.95 with time; time alone, fit on
        # one lived fast day, read the next at +0.5..+0.7 where the traces flipped sign between days.
        self.register_buffer("vc_clock", torch.zeros(1)); self.register_buffer("vc_clock_prev", torch.zeros(1))
        self.vcrit_clock = False
        # THE DECORRELATED CRITIC (vcrit_rls): the head's lesson is recursive least-squares TD(lambda) with forgetting, the
        # eligibility trace carried through a precision matrix (the Kalman form of TD; the online LSTD of Xu et al. 2002).
        # A gradient head fit for one pass to correlated inputs reads their dominant common component, which on the slow
        # bands runs against the return (the cross-page instrument, 2026-09-05: -0.49, -0.74, -0.60 on days the body never
        # lived, where least squares read +0.66, +0.92, +0.71); the inverse covariance divides that component out, as a
        # decorrelating inhibitory input layer does. A and b are the accumulated statistics (sized by the life to the
        # head's active inputs plus the level), float64 for the solve.
        self.register_buffer("vc_A", torch.zeros(0, 0, dtype=torch.float64))
        self.register_buffer("vc_b", torch.zeros(0, dtype=torch.float64))
        # THE CRITIC'S HOMEOSTATIC INPUT (vcrit_norm_tau): each input dimension standardized by its own running mean and scale
        # (adaptation and synaptic scaling at a slow time constant; the rate 1/min(n, tau), so the first ticks are exact
        # averages), because a prior that is scale-free in standardized units serves every dimension while on raw inputs,
        # whose scales span two orders, no single prior does (the pages instrument, 2026-09-05).
        self.register_buffer("vc_mu", torch.zeros(0, dtype=torch.float64))
        self.register_buffer("vc_var", torch.zeros(0, dtype=torch.float64))
        self.register_buffer("vc_n", torch.zeros((), dtype=torch.float64))
        self.register_buffer("vc_form", torch.full((), 2.0, dtype=torch.float64))   # 2: evidence in raw coordinates, the prior as the metric
        # THE MOUTH'S GATE (basal ganglia): whether to act, from the stream, the feelings, and the
        # salience of the mouth's proposal (the forecast's certainty, as the striatum reads the
        # strength of a cortical request for action)
        self.mouth_gate = nn.Linear(d + 5, 1)                      # + fatigue, mood, stress, salience, the level
        nn.init.zeros_(self.mouth_gate.weight)
        nn.init.constant_(self.mouth_gate.bias, math.log(birth_act / (1.0 - birth_act)))

        # ITS FACE: a forecast of the caregiver's face (a readout)
        self.face_head = nn.Linear(d, 1)
        nn.init.zeros_(self.face_head.weight); nn.init.zeros_(self.face_head.bias)
        self.read_sharp = 10.0                            # PHYSIOLOGY (to become an organ): logit = sharpness x cosine
        self.register_buffer("_mask", torch.triu(torch.ones(self.window, self.window, dtype=torch.bool), 1))

    # ---- the cortex over a window ----

    @torch.no_grad()
    def widen_gate(self, n_extra):
        """THE EAR: more inputs to the gate (the world's symbol this tick, its own act last tick), their weights born at zero"""
        old = self.mouth_gate
        new = nn.Linear(old.in_features + int(n_extra), 1)
        new.weight.zero_(); new.weight[:, :old.in_features] = old.weight; new.bias.copy_(old.bias)
        self.mouth_gate = new.to(old.weight.device)

    def inputs(self, xs, xos, faces, bundles, reads):
        """one position per tick: xs [T] the world's symbol (or its quiet), xos [T] its own symbol in
        the same tick (or its quiet), faces [T, 2], bundles [T, nb, d], reads [T, d] -> u [T, d].
        All the sounds of a tick superpose in one time step, its own attenuated by corollary discharge
        (own_gain; measured in cortex at a third to a half). With two positions per tick (the world's,
        then its own, mostly a rest) the stream read "d . o . g ." awake and "d o g" in the dreams
        the night trains on, and the cortex forecast "d" after everything awake (run 17, day 6)."""
        u = self.E(xs) + self.face_in(faces) + self.bundle_in(bundles.reshape(bundles.shape[0], -1))
        if xos is not None and self.sil_id is not None:
            # its own sound attenuated (corollary discharge, own_gain 0.5) and superposed on the tick's
            # position. Heard at full weight with the lessons hearing the world only (run 22), the cortex
            # alone was worse at day 6 ("big big big"); the loops that motivated that change were an
            # instrument's fault (a store holding only the cue), not the cortex's.
            own = (xos != self.sil_id).to(u.dtype).unsqueeze(-1)
            u = u + float(self.own_gain) * own * self.E(xos)
        return self.in_ln(u)

    def shift(self, v):
        """the context one lag older: v permuted (the last dimension)"""
        return v.index_select(-1, self.perm)

    def forecast(self, C, reads, conf=1.0):
        """what the mouth reads: the cortex's own forecast (the conditional mean, its norm its certainty)
        plus the hippocampus's recall through its pathway (the attended mean of memories, its norm
        their agreement). Two calibrated votes; agreement adds, and the readout's dot product makes
        agreement sharp. Recall is NOT an input to the stream (entered there it looked like the
        current symbol and the trunk advanced it a step); the night trains the cortex with recall off."""
        return self.latent_pred(C) + self.store_in(reads)     # both calibrated: each norm its certainty

    def stream(self, u):
        """u [T, d] -> C [T, d], the cortex stream (causal over the window)"""
        T = u.shape[0]
        x = u.unsqueeze(0)
        mask = self._mask[:T, :T]
        for blk in self.blocks:
            x = blk(x, mask)
        return self.lnf(x[0])

    def readout(self, pred, prior=None):
        """the lexicon read: logits [.., vocab] = sharpness x (pred . E). The forecast is the conditional
        mean of the next unit embedding (trained by squared error), so pred . E_k is its probability of
        symbol k, and its norm its certainty: a sure forecast is read decisively, an unsure one babbles
        the symbols it has heard in their proportions (the unconditional mean IS the heard distribution,
        so a separate prior counted it twice: with it the commonest symbol, the space, won every flat
        context on run 15). The pairwise cosines of the lexicon (about 0.06) put a floor on the
        sharpness: at 25, a symbol at probability 0.5 outweighs fifty strangers at their noise."""
        En = F.normalize(self.E.weight, dim=-1)
        return float(self.read_sharp) * pred @ En.t()

    def nearest(self, pred):
        return int(self.readout(pred).argmax(-1))

    # ---- the PFC ladder ----
    def band_update(self, states, c):
        """one tick: states [nb, d] -> new states; c [d] the stream now (detached by the caller if
        the bands are to be judges, live if they are to learn)"""
        out = []
        for b, tau in enumerate(self.clocks):
            s = states[b]
            g = torch.sigmoid(self.band_gate[b](s.detach()))            # Go/NoGo on its own update
            target = torch.tanh(self.band_in[b](c))
            # the leaky average at the clock, the gate scaling its rate. A gated write (the gate loading
            # the band in full, the clock forgetting) let the states jump as the gates learned, and the
            # bootstrapped values on jumping features diverged by day 15 (run 25: three bands in the
            # hundreds with TD errors in the thousands); reverted on measurement.
            out.append(s + (g / float(tau)) * (target - s))
        return torch.stack(out)

    def value_of(self, b, s):
        """V_b(s): a linear head; a differential band's head has no bias and reads the state centered
        on the running mean (band_mu), so the relative value has no constant to learn"""
        if bool(self.diff[b]):
            return (s - self.band_mu[b]) @ self.value[b].weight[0]
        return self.value[b](s).squeeze(-1)

    def vcrit_input(self, states, traces=None, clock=None):
        """what the ventral critic reads: the bands' states (vcrit_center 0; the bias holds the level) or the states
        centered on their running means (vcrit_center 1, the form of 2026-09-03). THE CENTERING WAS THE DEFECT
        (the night-transfer instrument, 2026-09-05): a running mean at 1024 ticks tracks a band whose clock is 4096 or
        16384 and leaves it a thousand-tick recency residual; on the served body's day 40 the same TD(0) rule read the
        return at horizon 1024 at +0.52 from the raw slow bands and -0.56 from the centered ones, and a ridge head from
        the raw slow bands read +0.50 on that day and +0.51 one, two and ten nights away: the slow bands do not drift"""
        x = states - self.band_mu if getattr(self, "vcrit_center", True) else states
        x = (x * self.vcrit_mask[:, None]).reshape(-1)
        tr = (self.r_tr if traces is None else traces).to(x.dtype)
        ck = (self.vc_clock if clock is None else clock).to(x.dtype)
        return torch.cat([x, tr if getattr(self, "vcrit_traces", False) else tr * 0.0,          # the full width always; the traces and
                          ck if getattr(self, "vcrit_clock", False) else ck * 0.0])              # the clock zeroed unless read

    def value_long(self, states, traces=None, clock=None):
        """the ventral critic's value over the bands' states and, when it reads them, the tonic traces and the clock (see vcrit_input)"""
        return (self.vcrit_input(states, traces, clock) @ self.vcrit.weight[0]) + self.vcrit.bias[0]

    def vcrit_rls_solve(self, idx, prior=None):
        """the decorrelated head from its evidence: w = (A + R)^-1 b over the active inputs (raw coordinates) and the level. With the
        homeostatic statistics (prior given) R = prior x sd_i^2 on each weight and 0 on the level: the standardized prior in raw
        coordinates, so the statistics shape only the prior and the evidence never changes coordinates. Without them the prior
        already sits inside A."""
        with torch.no_grad():
            A = self.vc_A
            if prior is not None and self.vc_var.numel() == A.shape[0] - 1:
                R = torch.zeros(A.shape[0], dtype=A.dtype, device=A.device); R[:-1] = float(prior) * (self.vc_var.clamp_min(0) + 1e-9)
                A = A + torch.diag(R)
            try:
                w = torch.linalg.solve(A, self.vc_b)
            except Exception:
                w = torch.linalg.lstsq(A, self.vc_b.unsqueeze(1)).solution.squeeze(1)
            self.vcrit.weight.zero_(); self.vcrit.weight[0, idx] = w[:-1].to(self.vcrit.weight.dtype); self.vcrit.bias[0] = w[-1].to(self.vcrit.bias.dtype)

    def vcrit_norm_update(self, x, tau):
        """the homeostatic statistics take this tick's (active) input; returns the standardized input"""
        with torch.no_grad():
            self.vc_n += 1.0; eta = 1.0 / min(float(self.vc_n), float(tau))
            self.vc_mu += eta * (x - self.vc_mu); self.vc_var += eta * ((x - self.vc_mu) ** 2 - self.vc_var)
            return (x - self.vc_mu) / (self.vc_var.clamp_min(0).sqrt() + 1e-3)

    def values(self, states):
        """V_b(s_b) for every band: [nb]"""
        return torch.stack([self.value_of(b, states[b]) for b in range(len(self.clocks))])

    def gammas(self):
        return [1.0 - 1.0 / float(t) if t > 1 else 0.0 for t in self.clocks]

    # ---- the lessons ----
    def latent_loss(self, pred, target_ids, w=None, sig=0.0, C=None):
        """the squared error between the forecast of the next embedding and the unit embedding
        received (a fixed lexicon: the target set is discrete and near-orthogonal, so the loss has
        no trivial solution), weighted by w [T] (the world's symbols); cmean reports the cosine. SIGReg, if asked, goes on the
        stream C, never on a forecast that must hit discrete targets."""
        En = self.E.weight.detach()
        tgt = En[target_ids]
        cos = F.cosine_similarity(pred.float(), tgt.float(), dim=-1)
        if w is None:
            w = torch.ones_like(cos)
        w = w.to(cos.dtype)
        # squared error to the unit target: the minimiser is the conditional mean of the next embedding,
        # whose norm is the certainty (1 when one symbol follows, small when many can). The cosine loss
        # left the norm meaningless, so a flat forecast and a sure one read the same at the mouth and
        # the prior's commonest letter won every tie (run 14, day 6: 'l' after every cue).
        se = 0.5 * ((pred.float() - tgt.float()) ** 2).sum(-1)
        loss = (se * w).sum() / w.sum().clamp(min=1e-6)
        cmean = float((cos.detach() * w).sum() / w.sum().clamp(min=1e-6))
        if sig > 0 and C is not None:
            loss = loss + float(sig) * sigreg(F.normalize(C, dim=-1).float())
        return loss, cmean

    def forecast_loss(self, C, bundles_next, sig=0.0):
        """the cortex stream at t forecasts the bundle handed over at t+1 (stop-grad): C [T, d],
        bundles_next [T, nb, d]; one minus the cosine, averaged over bands and steps"""
        terms, cos_all = [], []
        for b in range(len(self.clocks)):
            pred = self.pfc_pred[b](C)
            cos = F.cosine_similarity(pred.float(), bundles_next[:, b].detach().float(), dim=-1)
            terms.append((1.0 - cos).mean()); cos_all.append(float(cos.detach().mean()))
        loss = torch.stack(terms).mean()
        if sig > 0:
            loss = loss + float(sig) * sigreg(F.normalize(C, dim=-1).float())
        return loss, (sum(cos_all) / len(cos_all))
