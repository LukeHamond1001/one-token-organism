"""the critics (a mixin of `Life`, body/life.py): the fast critic's value now (`fast_value`); the tick's `_learn_values` (every
critic learns from the felt reward, the fast band's error is dopamine, the least-squares evidence and its solves, the chooser's
lesson); the reliability gain of the prefrontal voice (`_vrel_update`); and the face organ (`_own_face`, `_face_solve`,
`_face_input_vec`, `_foresee`, `_frel_update`, `_face_weight`).

Moved verbatim from body/life.py (review 2026-09-22 section 4, step 2); the floating-point order of every update is as it was."""
import math

import torch
import torch.nn.functional as F


class CriticsMixin:
    def fast_value(self):
        """the fast critic's value now (the striatal head on the delay line, or the dopamine band's head on its state)"""
        with torch.no_grad():
            if str(self.cfg.get("fast_input", "band")) == "striatum" and self.m.stri_W.numel() > 0:
                return float(self.m.fast_value(self.m.stri_in()))
            return float(self.m.values(self.bands)[int(self.cfg["dopamine_band"])])

    def _actor_scaling(self, m):
        """A147 (2026-10-01): SYNAPTIC SCALING OF THE STRIATAL ACTORS (Turrigiano 2008; Turrigiano and Nelson 2004): a neuron whose
        drive runs past its set point scales all its synapses down together, keeping its output in the range where its plasticity
        works. Each actor's set point is the unit scale of its squashing (the tanh's own: 1, as A97's "1 is the readout's own scale"):
        while this tick's mean square of its pre-activations (w . z over its settings) stands above 1 the actor's weights are
        multiplied by (mean square) ** (-1 / (2 x reach)), the reach the synaptic tag's (tag_reach, 64 ticks: the time over which an
        act's credit is assigned, so the scaling is no faster than the lesson's own memory; ours): the log of the drive decays with
        that time constant, a drive of 2,000 (its mean square 4 million) loses 11% at the first tick and is back at 1 in some 300
        ticks, under a minute of life, and stops there (the tick's own drive is the gate, so no lagging mean carries the scaling past
        the set point: a running mean of 1,024 ticks took the test body from 2,000 to 0.32, and the critics' horizon of 1,024 as the
        time constant left it at 8 after 1,500 ticks). A151 (2026-10-01): AND UP AS WELL AS DOWN. Scaling ran down only through day 49
        ("an actor that has learned little should not be made loud"), and the actors' forgetting (1 - 1/36,000 a tick, set when a lesson
        moved a bias by hundreds) shrank every drive by half a day once A148 had bounded the lessons: the ruler read 0.98 at A148's
        landing and 0.60 at dusk, e^(-t/36,000) exactly, no lesson refilling it; by dusk 51 the actors would have been whispering.
        Biology's scaling is bidirectional (Turrigiano 2008: synapses scaled up under activity deprivation as down under excess), so a
        drive under 1 is scaled up at the same reach, and the forgetting, a uniform shrink the scaling would undo each tick, is removed
        from the actors' lessons (the critics keep theirs). The newborn: an actor born at zero has a drive of exactly 0 and is left
        at zero until its first lesson; its first lessons are then brought to the unit scale within a minute, a confident early bias
        the proposal and the gate weigh against (as the saturated actors of days 1 to 48 were at their rails from the first day). The
        gradient through the squashing (mouth._actor_squash_grad) keeps a scaled actor from running up again. A running mean square
        per actor is kept beside it for the rulers (`_actor_ms`, by actor name, at the same reach; born at 1, not saved)"""
        if getattr(self, "_z_now", None) is None:
            return
        from .physiology import FRAMES
        H = float(self.cfg.get("tag_reach", FRAMES["tag_reach"])); a_ = 1.0 / H
        ms = getattr(self, "_actor_ms", None)
        if ms is None:
            ms = self._actor_ms = {}
        with torch.no_grad():
            for name, mod in [("voice", m.actor)] + [(e_.name, m.get_submodule(e_.actor)) for e_ in self.anatomy.motors]:
                raw = mod(self._z_now); now_ = float((raw * raw).mean())
                ms[name] = ms.get(name, 1.0) + a_ * (now_ - ms.get(name, 1.0))
                if now_ > 0.0:                                                 # A151: both ways; an actor born at zero stays zero until it learns
                    mod.weight.mul_(float(now_ ** (-0.5 / H)))

    def _learn_values(self, r, felt, stri):
        """dopamine: every critic learns from the felt reward; the fast band's error is dopamine; the chain closes; the synaptic tag is captured"""
        m = self.m
        # --- dopamine: the fast band's error of the world's reward; the critic learns at every band ---
        with torch.no_grad():
            v_now = m.values(self.bands)
            # THE LEVEL: the slow critic's value of this moment, read by the gate below, scaled by its own
            # running root mean square (divisive normalization; born at one so a newborn's noise reads small)
            lb = int(self.cfg["gate_level_band"]); vb = float(v_now[lb])
            m.v_scale[lb] += (1.0 / float(self.cfg["diff_horizon"])) * (vb * vb - float(m.v_scale[lb]))
            level = float(self.cfg["gate_level_w"]) * max(-5.0, min(5.0, vb / (1.0 + math.sqrt(max(0.0, float(m.v_scale[lb]))))))
        gam = m.gammas()
        with torch.no_grad():                                    # THE CLOCK advances: the day's fraction, the night at 1
            m.vc_clock_prev.copy_(m.vc_clock); m.vc_clock.fill_(min(1.0, float(self.sleep_pressure) / max(1.0, float(self.cfg["wake_ticks"]))))
        with torch.enable_grad():
            # the organs hold no dropout or batch statistics, so train()/eval() changed nothing but cost 12% of the tick in
            # Python (a recursive mode flip over every module twice a tick); the body stays in eval mode from construction
            v_prev_live = m.values(self._bands_prev.detach()) if getattr(self, "_bands_prev", None) is not None else None
            if v_prev_live is not None:
                # SEMI-GRADIENT TD, the convergent form: the target r + gamma V(s_now) is detached and
                # only the heads learn; the bands' states are fixed features of the stream (their input
                # maps are born, not trained by this error). The differential heads read their states
                # centered on a running mean at the reward rate's horizon (adaptation), so the relative
                # value's free constant has no direction to walk in.
                with torch.no_grad():
                    eta = 1.0 / float(self.cfg["diff_horizon"])
                    for b in range(len(gam)):                          # every band's mean (the ventral critic centers on all)
                        m.band_mu[b] += eta * (self._bands_prev[b].detach() - m.band_mu[b])
                # DISCOUNTED TD below the differential horizon, AVERAGE-REWARD (differential) TD at and
                # above it: with the discount near 1 the bootstrapped value ran away (the slowest band
                # read 7000 against a true return near 85, correlation -0.995: run 21, day 20). The
                # baseline is the reward rate estimated at the band's own clock, tonic dopamine.
                N_ = int(self.cfg.get("night_ticks", 0)); an_ = bool(N_) and bool(getattr(self, "_after_night", False)); self._after_night = False
                nf = [(gam[b] ** N_) if an_ else 1.0 for b in range(len(gam))]      # THE NIGHT TAKES TIME
                td = torch.stack([(r + gam[b] * nf[b] * v_now[b].detach() - v_prev_live[b]) if not self._differential[b]
                                  else (r - float(self.rbar) + v_now[b].detach() - v_prev_live[b]) for b in range(len(gam))])
                self._td_last = [float(x) for x in td.detach().cpu().tolist()]       # C165: the ladder's errors this tick, a ruler's (the record's bands)
                self.rbar += (1.0 / float(self.cfg["diff_horizon"])) * (r - self.rbar)   # the reward rate, tonic dopamine
                # THE VENTRAL CRITIC: discounted TD at a definite long horizon over the whole ladder's states (semi-gradient,
                # the target detached; linear on fixed features, convergent)
                with torch.no_grad():
                    vl_now = m.value_long(self.bands, m.r_tr, m.vc_clock)
                gl = float(self.cfg.get("vcrit_gamma", 1.0 - 1.0 / 1024)); nfl = (gl ** N_) if an_ else 1.0   # THE NIGHT TAKES TIME
                if int(self.cfg.get("vcrit_diff", 0)):
                    td_long = (r - float(self.rbar)) + vl_now.detach() - m.value_long(self._bands_prev.detach(), m.r_tr_prev, m.vc_clock_prev)
                    gl_tr = 1.0                                       # the trace decays at lambda alone
                else:
                    td_long = r + gl * nfl * vl_now.detach() - m.value_long(self._bands_prev.detach(), m.r_tr_prev, m.vc_clock_prev)
                    gl_tr = gl
                lam = float(self.cfg.get("vcrit_lambda", 0.0)); tau = float(self.cfg.get("vcrit_tau", 0.0))
                rls = int(self.cfg.get("vcrit_rls", 0))
                if rls:
                    # THE DECORRELATED CRITIC: the statistics of the trace against the state's discounted change and the reward
                    with torch.no_grad():
                        one = torch.ones(1, dtype=torch.float64, device="cpu")
                        x_p = m.vcrit_input(self._bands_prev.detach(), m.r_tr_prev, m.vc_clock_prev).detach().cpu()[self._vc_idx].double(); x_n = m.vcrit_input(self.bands.detach(), m.r_tr, m.vc_clock).detach().cpu()[self._vc_idx].double()
                        ntau = int(self.cfg.get("vcrit_norm_tau", 0))
                        if ntau:
                            # THE PRIOR AS THE METRIC (2026-09-05 21:40): the homeostatic statistics take the state but do NOT transform
                            # it. The evidence A, b is accumulated in the raw coordinates, which never drift; the statistics act only on
                            # the prior at the solve (delta x sd_i^2 on each weight, the level free), which is the standardized prior
                            # expressed in raw coordinates. Standardizing the inputs themselves while the statistics formed put every
                            # tick's evidence in different coordinates: on a recorded lived day the body's own head read -0.59 where the
                            # same evidence in fixed coordinates read +0.71 (nt_worlds3/4).
                            m.vcrit_norm_update(x_n, ntau)
                        xa_prev = torch.cat([x_p, one]); xa_now = torch.cat([x_n, one])
                        e = getattr(self, "_vc_e", None)
                        self._vc_e = (gl_tr * lam * e if e is not None else torch.zeros_like(xa_prev)) + xa_prev
                        vf_ = float(self.cfg.get("vcrit_forget", 0) or 0); beta = (1.0 - 1.0 / vf_) if vf_ > 0 else 1.0
                        dlt = xa_prev - gl * (nfl if not int(self.cfg.get("vcrit_diff", 0)) else 1.0) * xa_now
                        m.vc_A.mul_(beta).addr_(self._vc_e, dlt)
                        if beta < 1.0 and not ntau:
                            m.vc_A.diagonal().add_((1.0 - beta) * self._vc_delta)   # the constant prior inside A (the form without statistics)
                        m.vc_b.mul_(beta).add_(self._vc_e * float(r))
                        if self.ticks % int(self.cfg.get("vcrit_rls_every", 64)) == 0:
                            m.vcrit_rls_solve(self._vc_idx, prior=(self._vc_delta if ntau else None))
                if int(self.cfg.get("fast_rls", 0)) and (not stri or getattr(self, "_z_prev", None) is not None):
                    # THE FAST CRITIC DECORRELATED: the dopamine band's evidence on its own state (or the striatal input), at its own horizon
                    with torch.no_grad():
                        fb = int(self.cfg["dopamine_band"]); gf = float(gam[fb]); one = torch.ones(1, dtype=torch.float64, device="cpu")
                        if stri:
                            xf_p = self._z_prev.detach().cpu().double(); xf_n = self._z_now.detach().cpu().double()
                        else:
                            xf_p = self._bands_prev[fb].detach().cpu().double(); xf_n = self.bands[fb].detach().cpu().double()
                        m.fast_norm_update(xf_n, float(self.cfg.get("fast_rls_forget", 36000)))
                        xa_p = torch.cat([xf_p, one]); xa_n = torch.cat([xf_n, one])
                        ef = getattr(self, "_vf_e", None)
                        self._vf_e = (gf * ef if ef is not None else torch.zeros_like(xa_p)) + xa_p           # the trace at gamma (lambda 1)
                        bf = 1.0 - 1.0 / float(self.cfg.get("fast_rls_forget", 36000))
                        m.vf_A.mul_(bf).addr_(self._vf_e, xa_p - gf * xa_n); m.vf_b.mul_(bf).add_(self._vf_e * float(r))
                        if self.ticks % int(self.cfg.get("fast_rls_every", 64)) == 0:
                            m.fast_rls_solve(fb, prior=self._vf_delta)
                with torch.no_grad():
                    x_prev = torch.cat([m.vcrit_input(self._bands_prev.detach(), m.r_tr_prev, m.vc_clock_prev), torch.ones(1, device=self.dev)])
                    if lam > 0.0:
                        # THE CRITIC'S ELIGIBILITY TRACE (TD(lambda), backward view): the trace of the critic's inputs
                        # decays at gamma * lambda; the error captures it
                        tr = getattr(self, "_vtrace", None)
                        self._vtrace = (gl_tr * lam * tr if tr is not None else torch.zeros_like(x_prev)) + x_prev
                    e_in = self._vtrace if lam > 0.0 else x_prev
                if rls:
                    loss_vl = td_long.detach() * 0.0                   # the decorrelated head learns above, not by the gradient
                elif tau > 0.0:
                    # THE NORMALIZED STEP at the critic's time constant: the input's energy tracked over a horizon
                    with torch.no_grad():
                        en = float((e_in * e_in).sum())
                        self._vcrit_energy = en if getattr(self, "_vcrit_energy", None) is None else (1.0 - (1.0 - gl)) * self._vcrit_energy + (1.0 - gl) * en
                        step = ((1.0 - gl) / tau) / (self._vcrit_energy + 1e-6) * float(td_long.detach())
                        m.vcrit.weight += step * e_in[:-1].unsqueeze(0); m.vcrit.bias += step * e_in[-1:]
                    loss_vl = td_long.detach() * 0.0
                elif lam > 0.0:
                    loss_vl = -(td_long.detach() * (m.vcrit.weight[0] @ self._vtrace[:-1] + m.vcrit.bias[0] * self._vtrace[-1]))   # gradient -error x trace
                else:
                    loss_vl = td_long ** 2
                if int(self.cfg.get("fast_rls", 0)):
                    mask_ = torch.ones_like(td); mask_[int(self.cfg["dopamine_band"])] = 0.0; loss_v = ((td * mask_) ** 2).mean() + loss_vl
                else:
                    loss_v = (td ** 2).mean() + loss_vl
                # Go/NoGo on the bands' own updates: a positive error pulls the gate open, a negative one shut
                gates = torch.stack([torch.sigmoid(m.band_gate[b](self._bands_prev[b].detach())).squeeze()
                                     for b in range(len(gam))])
                tgt = (td.detach() > 0).float()
                loss_g = (td.detach().abs() * F.binary_cross_entropy(gates.clamp(1e-6, 1 - 1e-6), tgt, reduction="none")).mean()
                self.opt_value.zero_grad(set_to_none=True)
                (loss_v + 0.01 * loss_g).backward()
                self.opt_value.step()
                vf = float(self.cfg.get("vcrit_forget", 0) or 0)
                if vf > 0 and not rls:
                    with torch.no_grad():
                        m.vcrit.weight.mul_(1.0 - 1.0 / vf)       # the forgetting head
                bvf = float(self.cfg.get("value_forget", 0) or 0)
                if bvf > 0:
                    # A146 (2026-09-30): THE LADDER'S VALUE HEADS FORGET. Each band's linear value head decays by 1 - 1/value_forget a tick
                    # (36,000, a day: actor_forget's and fast_rls_forget's constant). Why: the differential heads (the bands at or past the
                    # differential horizon) read a relative value on states centered on a running mean, so their weights have no scale to
                    # rest at, and Adam at a fixed step on a near-zero-mean noisy error random-walks them: on life day 44 the clock-1,024
                    # head's weights stood at 651, its value swung from +102 to -14 with a spread of 50 within half a day, and its gate,
                    # trained by the sign of that noise, shut to 0.20 (the slow memory updated at a fifth of its clock). A weight that is
                    # not refreshed by the error decays (the leaky LMS critic; synapses without reinforcement fade); the discounted heads
                    # (weights 2 to 26) are refreshed every tick and lose nothing they use
                    with torch.no_grad():
                        for v_ in m.value:
                            v_.weight.mul_(1.0 - 1.0 / bvf)
                # DOPAMINE: the TD error of the band whose discount matches dopamine's (clock 16,
                # gamma 0.9375): an expected reward fires before it lands, a missed one dips
                if stri and getattr(self, "_z_prev", None) is not None:
                    with torch.no_grad():                        # the dopamine from the striatal head
                        fb_ = int(self.cfg["dopamine_band"])
                        delta = float(r + float(gam[fb_]) * m.fast_value(self._z_now) - m.fast_value(self._z_prev))
                else:
                    delta = float(td[int(self.cfg["dopamine_band"])].detach())
                self._dop_last = float(delta)                   # C186 (2026-10-01): the tick's dopamine, for the record (a ruler)
                if int(self.cfg.get("actor", 0)) and str(self.cfg.get("actor_form", "add")) == "softmax":
                    self._chooser_learn(delta)
                elif int(self.cfg.get("actor", 0)) and getattr(self, "_e_actor", None) is not None:
                    with torch.no_grad():                        # THE ACTOR'S LESSON: dopamine times the eligibility, the weights forgetting
                        m.actor.weight.add_(float(self.cfg.get("actor_lr", 0.02)) * delta * self._e_actor)   # A151: no forgetting (moot under the scaling)
                if int(self.cfg.get("actor", 0)):
                    slr_ = float(self.cfg.get("actor_slow_lr", 0.0))
                    for e_, st_ in zip(self.anatomy.motors, getattr(self, "motor", ())):   # each later effector's actor, the same lesson (step R5)
                        if st_["e_actor"] is not None:
                            with torch.no_grad():
                                fast_ = float(self.cfg.get("actor_lr", 0.02)) * delta * st_["e_actor"]
                                m.get_submodule(e_.actor).weight.add_(fast_)   # A151: no forgetting (a uniform shrink the scaling undid each tick)
                                if slr_ > 0.0:
                                    st_["a_upd"][0] += float(fast_.norm())
                        if slr_ > 0.0 and st_.get("a_tag") is not None and abs(float(delta)) > 1e-9:
                            with torch.no_grad():                        # A142/C153: the tag captured by the same phasic dopamine
                                upd = slr_ * delta * st_["a_tag"]        # (mouth._actor_tag_step: the 64-tick trace)
                                m.get_submodule(e_.actor).weight.add_(upd)
                                st_["a_upd"][1] += float(upd.norm())
                    self._actor_scaling(m)                           # A147: each actor's synapses scaled toward its set point
                if stri and int(self.cfg.get("wm", 0)) and getattr(m, "stri_wm", 0):
                    with torch.no_grad():                        # WORKING MEMORY: latch at a burst, clear at the reward or with age
                        m.wm_tick()
                        if felt > 0 or float(m.wm_age) > float(self.cfg.get("wm_max", 512)):
                            m.wm_clear()
                        elif delta > float(self.cfg.get("wm_burst", 0.5)):
                            m.wm_latch(m.striatum_read()); self._z_now = m.stri_in()
                delta_slow = float(td[int(self.cfg["gate_slow_band"])].detach())
                delta_long = float(td_long.detach()); vlong = float(vl_now)
                if int(self.cfg.get("vcrit_auto", 0)):
                    self._vrel_update(vlong, r)
                for b in range(len(gam)):
                    self.v_buf[b].append((self._bands_prev[b].detach().cpu(), r, self.bands[b].detach().cpu()))
            else:
                delta = r; delta_slow = r; delta_long = r; vlong = 0.0
            # THE CHAIN CLOSES (the eleventh defect, found by review 2026-09-06): the state that was this update's target is the
            # next update's source. Before, the source was the end of the previous tick (after its own symbol) and the target the
            # middle of this one (after the world's), so the body's own step fell in a gap no transition covered and every
            # critic's evidence matrix was asymmetric (16-30 percent) where a closed chain's is symmetric.
            self._bands_prev = self.bands.clone()
            if stri:
                self._z_prev = self._z_now
        with torch.no_grad():                                    # THE TONIC TRACES advance with this tick's felt reward
            m.r_tr_prev.copy_(m.r_tr); m.r_tr += (float(r) - m.r_tr) / torch.tensor([float(c) for c in m.clocks], device=m.r_tr.device)
        self._dopa = delta
        self._dopa_since_utt = float(getattr(self, "_dopa_since_utt", 0.0)) + max(0.0, float(delta))   # the reward since the last utterance kept
        # THE SYNAPTIC TAG: every act (or rest) leaves a tag on the gate's weights, (act - p) x the gate's input, that
        # decays at the ventral critic's own horizon; the ventral error, as it arrives over the following minutes,
        # captures the tags (Frey and Morris 1997: a tag set by activity, captured by later dopamine). The expected
        # update is the sum over acts of (act - p) x (the long return that followed minus the critic's estimate): the
        # policy gradient at the critic's horizon, where the twelve-tick sum of the lesson could not reach the parent's
        # attention. gate_slow_lr 0 = off.
        slr = float(self.cfg.get("gate_slow_lr", 0.0))
        if slr > 0.0 and getattr(self, "_gate_tag", None) is not None:
            with torch.no_grad():
                m.mouth_gate.weight += slr * delta_long * self._gate_tag[:-1].unsqueeze(0)
                m.mouth_gate.bias += slr * delta_long * self._gate_tag[-1:]
        return delta, delta_slow, delta_long, vlong, level, gam

    def _own_face(self, C1, r):
        """its own face, learned from yours: a foresight of the felt reward (face_form "foresee") or a readout of it"""
        m = self.m
        # --- its face learns from yours (a readout) ---
        if str(self.cfg.get("face_form", "read")) == "foresee":
            # THE FACE ORGAN FORESEES (2026-09-08, §5c): from the stream a tick ago it predicts the felt reward of this tick; its
            # reliability, the slope of the felt reward on that foresight, is measured, so imagination can be weighed by it
            # a least-squares readout (as the fast critic's): decorrelated, in reward units, its evidence forgetting over face_tau
            # ticks and solved every face_every; tick-by-tick gradient steps on a target that is zero on most ticks swung it wildly
            with torch.no_grad():
                xin = self._face_input_vec(C1)
                if getattr(self, "_fin_prev", None) is not None and xin is not None:
                    x = torch.cat([self._fin_prev, torch.ones(1, dtype=torch.float64)])
                    f_prev = float(x @ self._fh_w)                                   # the foresight made a tick ago, before this evidence
                    bf = 1.0 - 1.0 / float(self.cfg.get("face_tau", 36000))
                    self._fh_A.mul_(bf).addr_(x, x); self._fh_b.mul_(bf).add_(x * float(r))
                    self._frel_update(f_prev, float(r))
                    if (self.ticks + 32) % int(self.cfg.get("face_every", 64)) == 0:   # offset from the fast critic's solve
                        self._face_solve()
                f_pred = self._foresee(xin) if xin is not None else 0.0            # the felt reward it foresees for the next tick
            self._fin_prev = xin; self._C1_prev = C1.detach()
        else:
            with torch.enable_grad():
                f_pred = m.face_head(C1.detach()).squeeze() * 6.0
                lf = (f_pred - torch.tensor(float(self.face_now), device=self.dev)) ** 2
                self.opt_face.zero_grad(set_to_none=True); lf.backward(); self.opt_face.step()
        its_face = float(f_pred.detach()) if torch.is_tensor(f_pred) else float(f_pred); self._fpred_now = its_face
        return its_face

    def _vrel_update(self, vlong, r):
        """THE RELIABILITY GAIN: every 256 ticks, once the buffer holds 4096, the return at the head's horizon for the
        oldest 256 ticks (3072 rewards each, 95 percent of the discounted mass) against the head's value then; the
        moments decay over 8192 samples; the gain is max(0, corr)"""
        self._vbuf_v.append(float(vlong)); self._vbuf_r.append(float(r))
        if len(self._vbuf_r) < 4096 or self.ticks % 256 != 0:
            return
        gl = float(self.cfg.get("vcrit_gamma", 1.0 - 1.0 / 1024))
        rs = torch.tensor(list(self._vbuf_r)); vs = torch.tensor(list(self._vbuf_v))
        disc = gl ** torch.arange(3072, dtype=torch.float32)
        G = torch.stack([(rs[i:i + 3072] * disc).sum() for i in range(256)]); V = vs[:256]
        d = 1.0 - 1.0 / 65536; m = self._vrel                     # ~5 days of samples (review: 8192 held ~8 independent returns)
        for v_, g_ in zip(V.tolist(), G.tolist()):
            m[0] = d * m[0] + 1.0; m[1] = d * m[1] + v_; m[2] = d * m[2] + g_
            m[3] = d * m[3] + v_ * v_; m[4] = d * m[4] + g_ * g_; m[5] = d * m[5] + v_ * g_
        n = m[0]; mv, mg = m[1] / n, m[2] / n
        var_v, var_g = m[3] / n - mv * mv, m[4] / n - mg * mg; cov = m[5] / n - mv * mg
        corr = cov / math.sqrt(max(var_v, 1e-9) * max(var_g, 1e-9))
        self._vrel_corr = float(corr); self._vrel_gain = float(max(0.0, min(1.0, cov / max(var_v, 1e-9))))   # the slope (review: corr flickered at random)

    def _face_solve(self):
        """the foreseeing face organ from its evidence: w = (A + ridge)^-1 b, the ridge scaled to the evidence's own size"""
        m = self.m
        with torch.no_grad():
            A = self._fh_A; lam = float(self.cfg.get("face_ridge", 1.0)) * float(A.diagonal()[:-1].mean().clamp_min(1e-9))
            R = torch.full((A.shape[0],), lam, dtype=A.dtype); R[-1] = 0.0
            try:
                w = torch.linalg.solve(A + torch.diag(R), self._fh_b)
            except Exception:
                w = torch.linalg.lstsq(A + torch.diag(R), self._fh_b.unsqueeze(1)).solution.squeeze(1)
            self._fh_w = w
            if w.numel() == m.face_head.weight.numel() + 1:
                m.face_head.weight[0].copy_(w[:-1].to(m.face_head.weight)); m.face_head.bias[0] = w[-1].to(m.face_head.bias)

    def _face_input_vec(self, C1):
        """the face organ's input this tick, as a double vector: the stream, or the striatal input the fast critic reads"""
        if str(self.cfg.get("face_input", "cortex")) == "striatum" and self.m.stri_W.numel() > 0:
            z = getattr(self, "_z_now", None)
            return None if z is None else z.detach().cpu().double()
        return C1.detach().cpu().double()

    def _foresee(self, xin):
        """the felt reward the face organ foresees for the next tick, from its input"""
        if xin is None or xin.numel() != self._fh_n:
            return 0.0
        return float(torch.cat([xin, torch.ones(1, dtype=torch.float64)]) @ self._fh_w)

    def _frel_update(self, f, r):
        """the face organ's reliability: the running moments of (foresight, felt reward) over face_tau ticks; the slope, clipped to [0, 1]"""
        d = 1.0 - 1.0 / float(self.cfg.get("face_tau", 36000)); m = self._frel
        m[0] = d * m[0] + 1.0; m[1] = d * m[1] + f; m[2] = d * m[2] + r; m[3] = d * m[3] + f * f; m[4] = d * m[4] + r * r; m[5] = d * m[5] + f * r
        n = m[0]; mf, mr = m[1] / n, m[2] / n
        var_f, var_r = m[3] / n - mf * mf, m[4] / n - mr * mr; cov = m[5] / n - mf * mr
        self._frel_corr = float(cov / math.sqrt(max(var_f, 1e-9) * max(var_r, 1e-9))) if n > 64 else 0.0
        self._frel_gain = float(max(0.0, min(1.0, cov / max(var_f, 1e-9)))) if n > 64 else 0.0

    def _face_weight(self):
        """how much an imagined tick counts: the face organ's correlation with the felt reward, clipped at zero. Not its slope: a
        predictor of small variance clips its slope at one while discriminating nothing (night 50: slope 1.0, correlation 0.15, a
        mean foreseen reward of 0.03 over 336 imagined transitions at full weight). The correlation is the share it has proved."""
        return max(0.0, float(self._frel_corr))
