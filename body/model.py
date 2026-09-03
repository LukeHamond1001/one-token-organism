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

    def n(self):
        return int(self.K.shape[0])

    @torch.no_grad()
    def read(self, q, adapt=None):
        """q [d] -> (predicted next embedding [d] (unit) or zeros, confidence in 0..1, winner index);
        adapt [n] (optional) multiplies strengths: the recall adaptation a dream runs under"""
        if self.n() == 0:
            return torch.zeros(self.d, device=self.dev), 0.0, -1
        qn = F.normalize(q.to(self.dev).float(), dim=0)
        sims = self.K @ qn                                              # [n]
        S = self.S if adapt is None else self.S * adapt
        # RECALL BY CONTENT: the match decides; strength decides how long a memory lasts and which are
        # replayed (with it in the read, the most-reinforced memory won every cue: measured 2026-09-03)
        logits = sims / self.temp
        if self.read_strength > 0 or adapt is not None:
            logits = logits + (self.read_strength if adapt is None else 1.0) * torch.log(S + 1e-6) * (1.0 if adapt is not None else 1.0)
        w = torch.softmax(logits, 0)
        pred = F.normalize(w @ self.V, dim=0)
        self._last_w = w
        return pred, float(w.max()), int(w.argmax())

    @torch.no_grad()
    def write(self, k, v, strength, who, merge_cos=0.97):
        if strength <= 1e-4:
            return False
        k = F.normalize(k.to(self.dev).float(), dim=0); v = F.normalize(v.to(self.dev).float(), dim=0)
        if self.n() > 0:
            sims = self.K @ k
            j = int(sims.argmax())
            if float(sims[j]) > merge_cos and float(self.V[j] @ v) > merge_cos:
                self.S[j] += float(strength)                             # the same memory, stronger
                return True
        self.K = torch.cat([self.K, k.unsqueeze(0)]); self.V = torch.cat([self.V, v.unsqueeze(0)])
        self.S = torch.cat([self.S, torch.tensor([float(strength)], device=self.dev)])
        self.W = torch.cat([self.W, torch.tensor([int(who)], device=self.dev)])
        if self.n() > self.cap:                                         # the weakest gives way
            keep = torch.argsort(self.S, descending=True)[: self.cap]
            self._keep(keep)
        return True

    def _keep(self, idx):
        self.K, self.V, self.S, self.W = self.K[idx], self.V[idx], self.S[idx], self.W[idx]

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
    def self_confidence(self):
        """the store's own mean read confidence, querying each slot with its own key (the reference
        a dream is judged against: it stops when recall is half as sure as a memory of its own)"""
        if self.n() == 0:
            return 0.0
        logits = (self.K @ self.K.t()) / self.temp + torch.log(self.S + 1e-6).unsqueeze(0)
        w = torch.softmax(logits, 1)
        return float(w.max(1).values.mean())

    @torch.no_grad()
    def sample_starts(self, n, gen=None):
        """dream starts: slots drawn by strength, without replacement"""
        if self.n() == 0:
            return []
        n = min(int(n), self.n())
        p = self.S / self.S.sum()
        idx = torch.multinomial(p.cpu(), n, replacement=False, generator=gen)
        return [int(i) for i in idx]

    def state_dict(self):
        return {"K": self.K.cpu(), "V": self.V.cpu(), "S": self.S.cpu(), "W": self.W.cpu(), "temp": self.temp}

    def load_state_dict(self, sd):
        self.K = sd["K"].to(self.dev); self.V = sd["V"].to(self.dev)
        self.S = sd["S"].to(self.dev); self.W = sd["W"].to(self.dev)
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
        self.who_emb = nn.Embedding(2, d)                 # the speaker sense: 0 the world, 1 the body
        nn.init.normal_(self.who_emb.weight, std=0.3 / math.sqrt(d))
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
        self.pfc_pred = nn.ModuleList([nn.Linear(d, d) for _ in range(nb)])
        # THE PFC LADDER: leaky integrators of the stream at each clock, each with a learned input
        # map, a Go/NoGo gate on its own update, and a value head (the critic at that timescale)
        self.band_in = nn.ModuleList([nn.Linear(d, d) for _ in range(nb)])
        self.band_gate = nn.ModuleList([nn.Linear(d, 1) for _ in range(nb)])
        for g in self.band_gate:
            nn.init.zeros_(g.weight); nn.init.constant_(g.bias, 2.0)      # open at birth (sigmoid 0.88)
        self.value = nn.ModuleList([nn.Linear(d, 1) for _ in range(nb)])
        for v in self.value:
            nn.init.zeros_(v.weight); nn.init.zeros_(v.bias)            # V = 0 at birth: the error IS the reward
        # THE MOUTH'S GATE (basal ganglia): whether to act, from the stream and the feelings
        self.mouth_gate = nn.Linear(d + 3, 1)
        nn.init.zeros_(self.mouth_gate.weight)
        nn.init.constant_(self.mouth_gate.bias, math.log(birth_act / (1.0 - birth_act)))
        # ITS FACE: a forecast of the caregiver's face (a readout)
        self.face_head = nn.Linear(d, 1)
        nn.init.zeros_(self.face_head.weight); nn.init.zeros_(self.face_head.bias)
        self.read_sharp = 10.0                            # PHYSIOLOGY (to become an organ): logit = sharpness x cosine
        self.register_buffer("_mask", torch.triu(torch.ones(self.window, self.window, dtype=torch.bool), 1))

    # ---- the cortex over a window ----
    def inputs(self, xs, whos, faces, bundles, reads):
        """xs [T] long, whos [T] long, faces [T, 2], bundles [T, nb, d], reads [T, d] -> u [T, d].
        The cortex hears its own symbols as it hears the world's (a sound is a sound): the speaker
        sense lives in the hippocampal key and the corollary discharge, not in the stream, so what
        it learned after the world's "d" applies after its own (run 7: runs of one letter otherwise)."""
        u = self.E(xs) + self.face_in(faces) + self.bundle_in(bundles.reshape(bundles.shape[0], -1))
        if getattr(self, "cortex_who", False):
            u = u + self.who_emb(whos)
        return self.in_ln(u)

    def forecast(self, C, reads, conf=1.0):
        """what the mouth reads: the cortex's own forecast (its norm its certainty) plus the hippocampus's
        recall as a unit direction through its pathway, weighted by the recall's confidence. Two
        calibrated votes; agreement adds, and the readout's dot product makes agreement sharp. Recall is NOT an input to the stream (entered there it looked like the
        current symbol and the trunk advanced it a step); the night trains the cortex with recall off."""
        own = self.latent_pred(C)                             # calibrated: its norm is its certainty
        rec = self.store_in(reads)
        rn = rec.norm(dim=-1, keepdim=True)
        rec = torch.where(rn > 1e-6, rec / rn.clamp(min=1e-6), rec)
        if torch.is_tensor(conf):
            conf = conf.to(rec.dtype).unsqueeze(-1) if conf.dim() == 1 else conf
        return own + float(conf) * rec if not torch.is_tensor(conf) else own + conf * rec

    def stream(self, u):
        """u [T, d] -> C [T, d], the cortex stream (causal over the window)"""
        T = u.shape[0]
        x = u.unsqueeze(0)
        mask = self._mask[:T, :T]
        for blk in self.blocks:
            x = blk(x, mask)
        return self.lnf(x[0])

    def readout(self, pred, prior=None):
        """the lexicon read: logits [.., vocab] = sharpness x (pred . E) + log prior. The forecast is the
        conditional mean of the next unit embedding (trained by squared error), so its norm is its
        certainty: a sure forecast overrides the prior, an unsure one lets the prior babble. The prior
        is the symbols it has heard (perceptual narrowing; Bayes: prior x likelihood)."""
        En = F.normalize(self.E.weight, dim=-1)
        lg = float(self.read_sharp) * pred @ En.t()          # a dot product: the forecast's norm is its certainty
        if prior is not None:
            lg = lg + torch.log(prior + 1e-4).to(lg.dtype)
        return lg

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
            out.append(s + (g / float(tau)) * (target - s))
        return torch.stack(out)

    def values(self, states):
        """V_b(s_b) for every band: [nb]"""
        return torch.stack([self.value[b](states[b]).squeeze(-1) for b in range(len(self.clocks))])

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
