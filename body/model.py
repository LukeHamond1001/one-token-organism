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

    def __init__(self, d, cap=8192, temp=0.05, device="cpu", read_strength=0.0, links=4):
        self.d, self.cap, self.temp, self.dev = d, int(cap), float(temp), device
        self.read_strength = float(read_strength)      # weight of log-strength in the read: 0 = recall by content alone
        self.saturate = False                          # REPETITION SUPPRESSION (2026-09-11): a repeat strengthens its slot less the stronger it already is
        self.sat_done = False                          # the strengths were converted to the saturating law once (a save from before it)
        self.K = torch.zeros(0, d, device=device)
        self.V = torch.zeros(0, d, device=device)
        self.S = torch.zeros(0, device=device)
        self.W = torch.zeros(0, dtype=torch.long, device=device)
        self.B = torch.zeros(0, dtype=torch.bool, device=device)   # THE BOUNDARY: this slot's symbol ended the world's utterance
        self.Bs = torch.zeros(0, dtype=torch.bool, device=device)  # and this slot's symbol began one (the first after a pause)
        self.Bq = torch.zeros(0, dtype=torch.bool, device=device)  # THE SEAM: this slot's symbol is an utterance's first, under the last line's faded context
        self.NK = int(links)                                       # THE SEQUENCE (2026-09-12): each slot keeps the last NK slots written next in the same utterance
        self.N = torch.zeros(0, self.NK, dtype=torch.long, device=device)   # (-1: none); a dream draws one by strength: the recent and the rewarded replayed more
        # THE EPISODE TAG (2026-09-13, the twenty-second defect): each link remembers which utterance wrote it (a count of utterances,
        # the hippocampal time context), so a dream can follow ONE utterance as lived through slots that many utterances share
        self.NE = torch.zeros(0, self.NK, dtype=torch.long, device=device)  # (-1: untagged, a link from before the tags)
        self.episode = 0
        self.last_idx = -1                                         # the slot the last write went to (new or merged)
        # THE EPISODE KEPT PER UTTERANCE (episode_chain; 2026-09-14, the twenty-ninth defect): a slot's link table holds only its
        # sixteen newest continuations, and a slot shared by every "the " sees hundreds of utterances, so the thread from a question
        # to its own answer was cut within two symbols. Here each utterance keeps the ordered list of the slots it wrote (CA3's
        # sequence, held per episode, not per element), the newest ep_cap utterances; a slot knows the episodes it belongs to.
        self.EP = {}; self.EPI = {}; self.ep_cap = 1000
        self.A = torch.zeros(0, device=device)                     # THE WAKING RECALL TIRES (2026-09-12): a slot's short-term availability, 1 rested

    def n(self):
        return int(self.K.shape[0])

    @torch.no_grad()
    def read(self, q, adapt=None, end_vec=None, tire=None, follow=None, follow_gain=1.0, boost=None):
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
        if tire is not None and tire.numel() == logits.numel():
            logits = logits + torch.log(tire + 1e-6)             # a memory recalled lately is harder to recall again (synaptic depression)
        if follow is not None and follow_gain > 1.0:
            # THE RECALL CARRIES THE EPISODE (read_follow; 2026-09-13, the twenty-third defect): the slots that the utterance being
            # recalled wrote next (its links under its tag) are easier to recall now, as a retrieved sequence continues along its chain
            # (CA3's recurrent chain, the mechanism the dreams follow). Without it the waking read returned, at every tick, the
            # continuation of whichever memory matched the last five symbols, and its speech was a five-gram walk across lines.
            slot, tag = follow
            if 0 <= slot < self.n():
                row = self.N[slot]; hit = (row >= 0) & (self.NE[slot] == int(tag))
                if bool(hit.any()):
                    logits = logits.clone(); logits[row[hit]] += math.log(float(follow_gain))
        if boost is not None and 0 <= int(boost) < self.n() and follow_gain > 1.0:
            logits = logits.clone(); logits[int(boost)] += math.log(float(follow_gain))   # the episode's next element (episode_chain)
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
                self.last_idx = j
                if self.saturate:
                    # THE SIXTEENTH DEFECT (2026-09-11, night 117): strength summed linearly over repeats, so the typist's most repeated
                    # prefixes ("that is my ", said thirty times a day) reached four hundred against a store mean of two, the dreams
                    # (drawn by strength) collapsed onto nine distinct sequences of ninety-six, and the night diverged on them. A
                    # hippocampus encodes the familiar weakly (repetition suppression, novelty-gated encoding): a repeat's increment is
                    # scaled by m / (m + S), m the store's own mean strength, so a slot grows like the log of its repetitions, relative
                    # to the store and not to a constant. A new memory is written in full.
                    m_ = float(self.S.mean()) if self.n() > 0 else 1.0
                    self.S[j] += float(strength) * m_ / (m_ + float(self.S[j]))
                else:
                    self.S[j] += float(strength)
                return True
        self.K = torch.cat([self.K, k.unsqueeze(0)]); self.V = torch.cat([self.V, v.unsqueeze(0)])
        self.S = torch.cat([self.S, torch.tensor([float(strength)], device=self.dev)])
        self.W = torch.cat([self.W, torch.tensor([int(who)], device=self.dev)])
        self.B = torch.cat([self.B, torch.zeros(1, dtype=torch.bool, device=self.dev)])
        self.Bs = torch.cat([self.Bs, torch.zeros(1, dtype=torch.bool, device=self.dev)])
        self.Bq = torch.cat([self.Bq, torch.zeros(1, dtype=torch.bool, device=self.dev)])
        self.N = torch.cat([self.N, torch.full((1, self.NK), -1, dtype=torch.long, device=self.dev)])
        self.NE = torch.cat([self.NE, torch.full((1, self.NK), -1, dtype=torch.long, device=self.dev)])
        self.A = torch.cat([self.A, torch.ones(1, device=self.dev)])
        self.last_idx = self.n() - 1
        if self.n() > self.cap:                                         # the weakest gives way
            keep = torch.argsort(self.S, descending=True)[: self.cap]
            self._keep(keep)
        return True

    @torch.no_grad()
    def link(self, a, b, tag=-1):
        """THE SEVENTEENTH DEFECT (2026-09-12, night 128 read on a copy): the night's dreams were pattern completions from the
        utterance onsets, and every line that began with the same word shared one onset, so ninety-six dreams were seventeen
        fragments ("the ", "put ", "yes. ") of five symbols: the night replayed first words and the day's line endings, the nouns,
        never. A hippocampus keeps the order of an episode (CA3's recurrent chain; replay runs it as lived): each slot remembers
        the slot written next in the same utterance, and a dream follows that chain, pattern completion only where it breaks."""
        if 0 <= a < self.n() and 0 <= b < self.n() and a != b:
            row = self.N[a]; tags = self.NE[a]
            keep = row != int(b)                                     # the same continuation again: moved to the front, under this utterance's tag
            row = row[keep]; tags = tags[keep]
            self.N[a] = torch.cat([torch.tensor([int(b)], device=self.dev), row])[:self.NK]      # the newest first, the oldest forgotten
            self.NE[a] = torch.cat([torch.tensor([int(tag)], device=self.dev), tags])[:self.NK]

    def note(self, tag, slot):
        """the slot just written (new or merged) is the next element of this utterance's episode"""
        tag = int(tag); slot = int(slot)
        if slot < 0:
            return
        if tag not in self.EP:
            self.EP[tag] = []
            while len(self.EP) > self.ep_cap:                       # the oldest utterance's episode forgotten
                old = next(iter(self.EP)); ids = self.EP.pop(old)
                for sl in ids:
                    if sl >= 0 and sl in self.EPI:
                        self.EPI[sl] = [(t, q) for (t, q) in self.EPI[sl] if t != old]
        self.EP[tag].append(slot); self.EPI.setdefault(slot, []).append((tag, len(self.EP[tag]) - 1))

    def next_in(self, tag, pos):
        """the slot that followed position pos in utterance tag's episode; -1 at its end or where the slot was dropped"""
        ids = self.EP.get(int(tag))
        if ids is None or pos + 1 >= len(ids):
            return -1
        return int(ids[pos + 1])

    def episodes_of(self, slot):
        """the (tag, position) pairs of the episodes a slot belongs to, oldest first"""
        return list(self.EPI.get(int(slot), []))

    def _remap_episodes(self, remap):
        rm = remap.tolist(); EP = {}
        for tag, ids in self.EP.items():
            EP[tag] = [(rm[i] if 0 <= i < len(rm) else -1) if i >= 0 else -1 for i in ids]
        self.EP = EP; self.EPI = {}
        for tag, ids in self.EP.items():
            for q, sl in enumerate(ids):
                if sl >= 0:
                    self.EPI.setdefault(sl, []).append((tag, q))

    @torch.no_grad()
    def successor(self, a, gen=None, tag=None):
        """the next slot of an episode from slot a: a draw among the last NK continuations by their strength (the surprise and the
        reward at the moment of writing), the way replay favors the recent and the rewarded; -1 when there is none. With a tag,
        the newest continuation written by that utterance, or -1 where the utterance's trace is gone (the episode ends there)"""
        if not (0 <= a < self.n()):
            return -1
        row = self.N[a]; ok = row >= 0
        if tag is not None:
            hit = ok & (self.NE[a] == int(tag))
            return int(row[hit][0]) if bool(hit.any()) else -1
        if not bool(ok.any()):
            return -1
        cand = row[ok]; w = self.S[cand].clamp_min(1e-6)
        i = int(torch.multinomial((w / w.sum()).cpu(), 1, generator=gen))
        return int(cand[i])

    @torch.no_grad()
    def draw_link(self, a, gen=None):
        """a dream's first step from an onset: a continuation drawn by strength as successor() draws, and the utterance that wrote it
        (its tag; -1 for a link from before the tags): the dream then follows that utterance"""
        if not (0 <= a < self.n()):
            return -1, -1
        row = self.N[a]; ok = row >= 0
        if not bool(ok.any()):
            return -1, -1
        idx = torch.nonzero(ok).flatten(); w = self.S[row[idx]].clamp_min(1e-6)
        i = int(idx[int(torch.multinomial((w / w.sum()).cpu(), 1, generator=gen))])
        return int(row[i]), int(self.NE[a][i])

    @torch.no_grad()
    def compress(self):
        """the strengths of a store kept under the linear law converted once to what the saturating law would have summed:
        S -> m ln(1 + S/m), m the store's mean; the order is kept, the skew is not"""
        if self.n() > 0 and not self.sat_done:
            m_ = float(self.S.mean())
            if m_ > 0:
                self.S = m_ * torch.log1p(self.S / m_)
        self.sat_done = True

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
        idx = idx.to(self.dev).long()
        remap = torch.full((self.n(),), -1, dtype=torch.long, device=self.dev); remap[idx] = torch.arange(int(idx.numel()), device=self.dev)
        self.last_idx = int(remap[self.last_idx]) if 0 <= self.last_idx < self.n() else -1
        self.K, self.V, self.S, self.W = self.K[idx], self.V[idx], self.S[idx], self.W[idx]
        self.B, self.Bs, self.Bq = self.B[idx], self.Bs[idx], self.Bq[idx]
        N = self.N[idx]; self.N = torch.where(N >= 0, remap[N.clamp_min(0)], N)   # the links follow the slots that stay; a dropped successor is none
        NE = self.NE[idx] if self.NE.shape[0] == remap.numel() else torch.full_like(self.N, -1)
        self.NE = torch.where(self.N >= 0, NE, torch.full_like(NE, -1))
        self.A = self.A[idx] if self.A.numel() == remap.numel() else torch.ones(int(idx.numel()), device=self.dev)
        if self.EP:
            self._remap_episodes(remap)

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
    def sample_starts(self, n, gen=None, mask=None):
        """dream starts: the utterance onsets the store knows, drawn by strength (an episode replayed from its
        beginning; with fewer onsets than dreams the draw is with replacement: ten onsets gave ten short dreams a
        night and the cortex's trace fell from 60 to 36 of 82, run 54); any memory by strength, without
        replacement, when the store knows no onset. (Half the night drawn from any memory instead, runs 57/58,
        moved the cortex's trace within seed noise and cost the mouth two to four of 48 against runs 59/60.)"""
        if self.n() == 0:
            return []
        n = int(n)
        starts = torch.nonzero(self.Bs if mask is None else (self.Bs & mask.to(self.Bs.device))).flatten()   # mask: which onsets may start a dream
        pool = starts if starts.numel() >= 1 else torch.arange(self.n(), device=self.dev)
        p = self.S[pool] / self.S[pool].sum()
        idx = torch.multinomial(p.cpu(), n, replacement=bool(pool.numel() < n), generator=gen)
        return [int(pool[i]) for i in idx]

    def state_dict(self):
        return {"K": self.K.cpu(), "V": self.V.cpu(), "S": self.S.cpu(), "W": self.W.cpu(), "B": self.B.cpu(), "Bs": self.Bs.cpu(), "Bq": self.Bq.cpu(), "temp": self.temp, "sat": bool(self.sat_done), "N": self.N.cpu(), "NE": self.NE.cpu(), "episode": int(self.episode), "EP": {int(t): list(v) for t, v in self.EP.items()}}

    def load_state_dict(self, sd):
        self.K = sd["K"].to(self.dev); self.V = sd["V"].to(self.dev)
        self.S = sd["S"].to(self.dev); self.W = sd["W"].to(self.dev)
        self.B = sd["B"].to(self.dev) if "B" in sd else torch.zeros(self.n(), dtype=torch.bool, device=self.dev)
        self.Bs = sd["Bs"].to(self.dev) if "Bs" in sd else torch.zeros(self.n(), dtype=torch.bool, device=self.dev)
        self.Bq = sd["Bq"].to(self.dev) if "Bq" in sd else torch.zeros(self.n(), dtype=torch.bool, device=self.dev)
        self.temp = float(sd.get("temp", self.temp)); self.sat_done = bool(sd.get("sat", False))
        def fit(M):                                                     # a saved link table brought to this store's width (padded or trimmed, newest first)
            M = M.to(self.dev)
            if M.shape[1] < self.NK:
                return torch.cat([M, torch.full((M.shape[0], self.NK - M.shape[1]), -1, dtype=torch.long, device=self.dev)], dim=1)
            return M[:, :self.NK]
        N = sd.get("N")
        if N is None or N.dim() != 2:
            self.N = torch.full((self.n(), self.NK), -1, dtype=torch.long, device=self.dev)
            if N is not None and N.dim() == 1 and N.numel() == self.n():
                self.N[:, 0] = N.to(self.dev)                           # a save with one link a slot
        else:
            self.N = fit(N)
        NE = sd.get("NE")
        self.NE = fit(NE) if (NE is not None and NE.dim() == 2 and NE.shape[0] == self.n()) else torch.full_like(self.N, -1)
        self.episode = int(sd.get("episode", 0))
        self.last_idx = -1
        self.A = torch.ones(self.n(), device=self.dev)            # rested after a load
        EP = sd.get("EP")
        self.EP = {}; self.EPI = {}
        if isinstance(EP, dict):
            n = self.n()
            for t, ids in EP.items():
                self.EP[int(t)] = [int(i) if (0 <= int(i) < n) else -1 for i in ids]
            for t, ids in self.EP.items():
                for q, sl in enumerate(ids):
                    if sl >= 0:
                        self.EPI.setdefault(sl, []).append((t, q))


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
        # THE FAST CRITIC DECORRELATED (fast_rls 1; 2026-09-06): the dopamine band's own value head solved from evidence like the
        # ventral critic (raw coordinates, the prior as the metric), at its own horizon. On a stalked day the fast error at a smile was
        # +1.97 on a reward of +2: the band predicted nothing of it, so nothing could be disappointed, so talking over the parent cost
        # nothing the gate could feel. The infant's turn-taking needs the expected answer.
        self.register_buffer("vf_A", torch.zeros(0, 0, dtype=torch.float64)); self.register_buffer("vf_b", torch.zeros(0, dtype=torch.float64))
        self.register_buffer("vf_mu", torch.zeros(0, dtype=torch.float64)); self.register_buffer("vf_var", torch.zeros(0, dtype=torch.float64))
        self.register_buffer("vf_n", torch.zeros((), dtype=torch.float64))
        # THE STRIATAL INPUT (fast_input "striatum"; 2026-09-06, the tenth defect): a short delay line of the stream's events
        # (heard symbols, own symbols, felt faces) through a born random sparse expansion, read linearly by the fast critic.
        # Neither the ladder's bands nor the cortex's vector linearly carry which word is ending (0.13 on an unseen day); a
        # fixed thresholded expansion of the last eight events does (0.47, the value rising half a smile before the smile),
        # as the striatum's and the cerebellum's afferent layers do. Sized by the life (stri_k events, stri_m units); 0 = absent.
        self.register_buffer("stri_W", torch.zeros(0, 0)); self.register_buffer("stri_b", torch.zeros(0))
        self.register_buffer("stri_line", torch.full((0,), -1, dtype=torch.long))
        self.vfast = nn.Linear(1, 1)
        # THE ACTOR (actor 1; 2026-09-06): the striatum's second head, from the same delay-line expansion to a bias over the next
        # symbol, added to the cortex's forecast (the cortex proposes, the striatum disposes); learned by the three-factor rule
        # over which-symbol, as the gate learns act-or-rest. Sized with the striatum; a placeholder until then.
        self.actor = nn.Linear(1, 1)
        self.register_buffer("wm_slot", torch.zeros(0)); self.register_buffer("wm_on", torch.zeros(())); self.register_buffer("wm_age", torch.zeros(()))
        self.stri_wm = 0
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
        u = self.E(xs) + self.face_in(faces) + self.bundle_in(bundles.reshape(*bundles.shape[:-2], -1))   # [T, nb, d] or [B, T, nb, d]
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
        """u [T, d] -> C [T, d], the cortex stream (causal over the window); a batch u [B, T, d] -> C [B, T, d] (the
        night's dreams in lockstep, 2026-09-13: the same arithmetic, many at once)"""
        batched = u.dim() == 3
        T = u.shape[-2]
        x = u if batched else u.unsqueeze(0)
        mask = self._mask[:T, :T]
        for blk in self.blocks:
            x = blk(x, mask)
        x = self.lnf(x)
        return x if batched else x[0]

    def stream_step(self, u_t, cache):
        """THE STREAM ONE POSITION AT A TIME (2026-09-13): u_t [B, d] the newest position's input; cache a list, one entry per
        block, of the keys and values of the positions before it ([B, heads, t, hd] each, or None at the first position) ->
        C_t [B, d], the cache grown by this position. The same arithmetic as stream() at the last position (the attention is
        causal, so the earlier positions' keys and values never change), without recomputing the positions before: the night's
        lockstep loop over a dream's prefixes falls from quadratic to linear in its length."""
        x = u_t.unsqueeze(1)                                                       # [B, 1, d]
        for i, blk in enumerate(self.blocks):
            h = blk.ln1(x)
            q, k, v = F.linear(h, blk.attn.in_proj_weight, blk.attn.in_proj_bias).chunk(3, dim=-1)
            B_, _, d = q.shape; H = blk.attn.num_heads; hd = d // H
            q = q.view(B_, 1, H, hd).transpose(1, 2); k = k.view(B_, 1, H, hd).transpose(1, 2); v = v.view(B_, 1, H, hd).transpose(1, 2)
            if cache[i] is not None:
                k = torch.cat([cache[i][0], k], dim=2); v = torch.cat([cache[i][1], v], dim=2)
            assert k.shape[2] <= self.window, "the stream's cache outgrew the window"
            cache[i] = (k, v)
            att = torch.softmax((q @ k.transpose(-1, -2)) / math.sqrt(hd), dim=-1)  # [B, H, 1, t]
            x = x + blk.attn.out_proj((att @ v).transpose(1, 2).reshape(B_, 1, d))
            x = x + blk.mlp(blk.ln2(x))
        return self.lnf(x[:, 0])

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

    def band_update_b(self, states, c):
        """band_update for a batch (the night's dreams in lockstep, 2026-09-13): states [B, nb, d], c [B, d] -> [B, nb, d]"""
        out = []
        for b, tau in enumerate(self.clocks):
            s = states[:, b]
            g = torch.sigmoid(self.band_gate[b](s.detach()))
            target = torch.tanh(self.band_in[b](c))
            out.append(s + (g / float(tau)) * (target - s))
        return torch.stack(out, dim=1)

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

    def fast_rls_solve(self, b, prior=None):
        """the dopamine band's head from its evidence: w = (A + R)^-1 b over the band's state (raw) and the level (see vcrit_rls_solve)"""
        with torch.no_grad():
            A = self.vf_A
            if prior is not None and self.vf_var.numel() == A.shape[0] - 1:
                R = torch.zeros(A.shape[0], dtype=A.dtype, device=A.device); R[:-1] = float(prior) * (self.vf_var.clamp_min(0) + 1e-9)
                A = A + torch.diag(R)
            try: w = torch.linalg.solve(A, self.vf_b)
            except Exception: w = torch.linalg.lstsq(A, self.vf_b.unsqueeze(1)).solution.squeeze(1)
            if self.stri_W.numel() > 0:                                       # the striatal head
                self.vfast.weight[0].copy_(w[:-1].to(self.vfast.weight)); self.vfast.bias[0] = w[-1].to(self.vfast.bias)
            else:
                self.value[b].weight[0].copy_(w[:-1].to(self.value[b].weight)); self.value[b].bias[0] = w[-1].to(self.value[b].bias)

    def to(self, *args, **kwargs):
        """the critics' double-precision evidence (A, b, the running statistics) stays on the CPU: Metal has no float64 and
        no solve; the cortex, the bands, the store and the heads go where they are sent"""
        dbl = {n: self._buffers.pop(n) for n in list(self._buffers) if self._buffers[n] is not None and self._buffers[n].dtype == torch.float64}
        out = super().to(*args, **kwargs)
        for n, b in dbl.items():
            out.register_buffer(n, b.cpu())
        return out

    def striatum_init(self, k, m, seed=0, wm=0):
        """born: the expansion of a delay line of k events (a heard symbol, an own symbol, or a felt face each) into m
        thresholded units; the rows of the born map are summed over the line's occupied positions (the input is one-hot)"""
        n_in = int(k) * (2 * self.vocab + 3)                                   # heard | own | warm face, cold face, a tick of quiet
        g = torch.Generator().manual_seed(int(seed) + 7919); dev = self.E.weight.device
        self.stri_W = (torch.randn(n_in, int(m), generator=g) / math.sqrt(float(k))).to(dev)
        self.stri_b = (-torch.rand(int(m), generator=g) * 1.5).to(dev)          # random thresholds: about a third of the units fire
        self.stri_line = torch.full((int(k),), -1, dtype=torch.long, device=dev)
        # WORKING MEMORY (wm 1; 2026-09-06): a slot beside the line, latching the striatal input at a dopamine burst and
        # holding it until the reward comes or it ages out, so what began a sequence is still readable at its end
        # (prefrontal gating by dopamine: update at a burst, maintain otherwise). The heads read [line, slot].
        self.stri_wm = int(wm)
        self.wm_slot = torch.zeros(int(m), device=dev); self.wm_on = torch.zeros((), device=dev); self.wm_age = torch.zeros((), device=dev)   # buffers (assigned by name)
        width = int(m) * (1 + self.stri_wm)
        self.vfast = nn.Linear(width, 1).to(dev)
        self.actor = nn.Linear(width, self.vocab).to(dev)
        with torch.no_grad():
            self.vfast.weight.zero_(); self.vfast.bias.zero_(); self.actor.weight.zero_(); self.actor.bias.zero_()

    def striatum_push(self, kind, idx):
        """an event enters the delay line: kind 0 a heard symbol, 1 an own symbol, 2 a felt face (idx 0 warm, 1 cold), 3 a tick
        of quiet (nothing heard, nothing said: the line carries time, as time cells do, so a smile that comes after the
        child's quiet can be seen approaching through the quiet)"""
        with torch.no_grad():
            self.stri_line = torch.roll(self.stri_line, 1)
            self.stri_line[0] = int(kind) * self.vocab + int(idx) if int(kind) < 2 else 2 * self.vocab + (int(idx) if int(kind) == 2 else 2)

    def striatum_read(self):
        """the expansion now: relu(the born rows of the line's events + thresholds), [m]"""
        with torch.no_grad():
            width = 2 * self.vocab + 3; z = self.stri_b.clone()
            for p_ in range(self.stri_line.numel()):
                e = int(self.stri_line[p_])
                if e >= 0:
                    z += self.stri_W[p_ * width + e]
            return torch.relu(z)

    def stri_in(self):
        """what the striatal heads read: the line's expansion, and the working-memory slot beside it when the slot exists"""
        z = self.striatum_read()
        if getattr(self, "stri_wm", 0):
            return torch.cat([z, self.wm_slot * self.wm_on])
        return z

    def wm_latch(self, z):
        with torch.no_grad():
            self.wm_slot.copy_(z[: self.wm_slot.numel()]); self.wm_on.fill_(1.0); self.wm_age.fill_(0.0)

    def wm_clear(self):
        with torch.no_grad():
            self.wm_on.fill_(0.0); self.wm_age.fill_(0.0)

    def wm_tick(self):
        with torch.no_grad():
            self.wm_age += 1.0

    def striatum_reset(self):
        with torch.no_grad():
            self.stri_line.fill_(-1)

    def fast_value(self, z):
        """the fast critic's value of a striatal input"""
        return self.vfast(z).squeeze(-1)

    def fast_norm_update(self, x, tau):
        """running mean and variance of the fast head's inputs (Welford, the count from 1), for the prior's metric only"""
        with torch.no_grad():
            self.vf_n += 1.0; eta = 1.0 / min(float(self.vf_n), float(tau))
            x = x.detach().cpu().double(); d = x - self.vf_mu; self.vf_mu += eta * d; self.vf_var += eta * (d * (x - self.vf_mu) - self.vf_var)

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
            self.vcrit.weight.zero_(); self.vcrit.weight[0, idx.to(self.vcrit.weight.device)] = w[:-1].to(self.vcrit.weight); self.vcrit.bias[0] = w[-1].to(self.vcrit.bias)

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
