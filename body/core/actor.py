"""the actor (a mixin of `Life`, body/life.py): the chooser's eligibility (`_chooser_credit`) and its lesson from dopamine
(`_chooser_learn`), and the actor's reliability (`_arel_update`). The actor's vote is read in `_choose` and its credit set in `_act`
(mouth.py).

Moved verbatim from body/life.py (review 2026-09-22 section 4, step 2)."""
import math

import torch


class ActorMixin:
    def _chooser_credit(self, nxt, gamma):
        """the chooser's eligibility: the log-softmax's gradient over the candidates for the one said, on the cortex's state; decays by
        dopamine's discount; nothing new when the moment was not torn"""
        m = self.m
        e = getattr(self, "_e_chooser", None)
        e = (gamma * e) if e is not None else torch.zeros(m.vocab, m.d, device=self.dev)
        cands = getattr(self, "_cands_now", None)
        if cands is not None and int(nxt) in cands:
            pa = self._pa_now; za = self._za_now
            for j, c in enumerate(cands):
                e[c] += ((1.0 if c == int(nxt) else 0.0) - float(pa[j])) * za
        self._e_chooser = e

    def _chooser_learn(self, delta):
        """dopamine times the eligibility on the chooser's head; every row bounded (actor_wmax) so no candidate can saturate the vote"""
        e = getattr(self, "_e_chooser", None)
        if e is None or abs(float(delta)) < 1e-9:
            return
        with torch.no_grad():
            W = self.m.chooser.weight
            W.add_(float(self.cfg.get("actor_lr", 0.02)) * float(delta) * e)
            n = W.norm(dim=1, keepdim=True); wmax = float(self.cfg.get("actor_wmax", 3.0))
            W.mul_(torch.clamp(wmax / (n + 1e-9), max=1.0))

    def _arel_update(self, v, g):
        """the actor's reliability: the running moments of (its vote for the act taken, the reward of the ticks after) over actor_tau acts"""
        d = 1.0 - 1.0 / float(self.cfg.get("actor_tau", 36000)); m = self._arel
        m[0] = d * m[0] + 1.0; m[1] = d * m[1] + v; m[2] = d * m[2] + g; m[3] = d * m[3] + v * v; m[4] = d * m[4] + g * g; m[5] = d * m[5] + v * g
        n = m[0]; mv, mg = m[1] / n, m[2] / n
        var_v, var_g = m[3] / n - mv * mv, m[4] / n - mg * mg; cov = m[5] / n - mv * mg
        self._arel_corr = float(cov / math.sqrt(max(var_v, 1e-9) * max(var_g, 1e-9))) if n > 64 else 0.0
        self._arel_gain = float(max(0.0, min(1.0, cov / max(var_v, 1e-9)))) if n > 64 else 0.0
