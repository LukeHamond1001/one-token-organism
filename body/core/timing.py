"""the motor timing part (a mixin of `Life`, body/life.py; the core refactor's step R6, docs/SIM_DESIGN.md 5.4 and 5.8): for each later
effector (the voice has none, so the language body runs none of this), act_pred (the forecast of its own next act, its proposal),
the forward half (the forecast of its body sense at the next position, and the correction its error makes to the proposal) and
act_inv (from its body sense at t and t+1 to the act that explains the motion, learning online from its own acts, its running
reliability Cohen's kappa over its running confusion, joint by joint). The organs are m.timing[name] (body/model.py `MotorTiming`); the
working state is the effector's in life.motor; the constants are physiology.py's MOTOR, absent from a body's cfg unless given.

WHEN EACH PART RUNS, for a later effector at tick t (after the voice's choice, in `_choose_effector`, body/core/mouth.py):
- `_timing_sense`: its body sense now, s_t (its channel's observation this tick, its own numbers); act_inv's lesson on the tick before
  when it acted then (the pair (s_t-1, s_t), the efference copy a_t-1 the label; its reliability updated on the label it gave before
  the lesson); the forward half's error now, e_t = s_t - the sense it foresaw at t-1's own step.
- `_timing_propose` (the base Effector's `propose`): act_pred(C) + cor(e_t), C the stream the tick's choice reads.
- after the act (`_act_effectors`): the forward half foresees s_t+1 from the stream after the tick's own step (which holds the act).
- in the waking lesson (`_timing_loss`, body/core/cortex.py), over the window, whose positions are the ticks: at every position t >= 1
  act_pred(C[t-1]) + cor(s[t] - fwd(C[t-1])) is taught the act at t, its efference copy where it acted; where it rested (whatever
  moved it: the parent's hand, a collision, a reflex, or nothing, which reads as the hold) act_inv's label for (s[t], s[t+1]),
  weighted by act_inv's reliability (a demonstration counts only as far as the inverse model has earned: the weights are absolute, the
  weighted errors averaged over the window's positions), or, for an effector with no inverse model, its rest; and fwd(C[t]) is taught
  s[t+1]. Through the cortex, at the waking lesson's rate, as a channel's forecast, except act_inv's labels, which reach act_pred and
  the correction and never the stream. ACT_PRED'S PLASTICITY IS GATED BY ITS LABELS' RELIABILITY (`GatedAdam`, `_timing_step`):
  act_pred and its correction step with an optimizer of their own, a group per effector, each lesson a sample of weight the lesson's
  mean label weight in the group's moments and in its step size, its gradient bounded by its own norm; every other parameter steps
  with the waking lesson's Adam as before, bounded by theirs.
THE LEARNED STOPS (chunk_gate, every later effector; `_choose_effector`): a chunk of acts continues, the act act_pred's best guess
(each joint's most likely setting, no draw), until that guess is the effector's rest (the learned end), its gate's own draw closes,
its reflex fires, its declared end act closed the chunk before, or chunk_max acts have run (a ceiling: the next act is a fresh
decision). With chunk_gate 0 every tick is a fresh decision, as before R6.

The one-tick timing the forward half serves: on a tick the world is quiet the choice reads the stream of the last full position, the
tick before, so the sense felt now enters the proposal only through the forward half's error (SIM_DESIGN.md 5.2's one forward a tick
for the sim, its own acts entering at the next frame, is a later step). On a tick a world symbol opens a position the choice reads it
there, while the lesson teaches act_pred from the position before, as the words' lesson does."""
import torch
import torch.nn.functional as F

from .physiology import MOTOR


class GatedAdam(torch.optim.Optimizer):
    """ACT_PRED'S PLASTICITY, GATED BY THE RELIABILITY OF ITS LABELS (the R6 verifier's third finding, 2026-09-24; SIM_DESIGN.md 5.4):
    Adam in which each step is one sample of weight `gain` (each group's, set before the step: the lesson's mean label weight over
    the window's positions, 1 at an own act, its efference copy, and act_inv's reliability at a rest act_inv labelled). The gradient is
    taken per unit of that weight, u = the gradient / gain (the lesson's weighted error averages the weights over the positions; u is
    the gradient of the error averaged over the labels' weights), and the gain enters twice, as a neuromodulator gates plasticity:
    - THE MOMENTS ARE MOVED IN PROPORTION: m <- m + (1 - beta1) gain (u - m), v <- v + (1 - beta2) gain (u^2 - v), the share of each
      still its birth's zero q <- q (1 - (1 - beta) gain), their estimates m / (1 - q1) and v / (1 - q2). A lesson whose labels are
      barely earned moves the running averages barely, so its direction is not carried into later lessons by the momentum.
    - THE STEP IS SCALED: lr gain m^ / (sqrt(v^) + eps). Adam divides each parameter's step by its recent gradient size, so a
      gradient made small (the lesson's loss scaled by the reliability, as R6 fix 1 made it) is re-inflated to a whole step: in the
      day's Adam, over 300 lessons on a window of rests alone or of the parent's guidance alone at the served rate, act_pred learned
      at reliability 0.01 0.95 and 0.87 of what it learned at 1 from a fresh optimizer (the moments built by the small gradients
      themselves: after one lesson m^ is its gradient and v^ its square, whatever its size), 0.48 and 0.44 after 100 lessons of its
      own acts had built them (their size forgotten over the lessons). The gain on the step size is what scales the learning: a
      lesson at reliability r moves act_pred r of a whole lesson's step, and over N lessons it learns about what N r whole lessons
      teach (anatomy 31: the same label mass given whole teaches as much).
    At gain 1 on every step this is Adam exactly (the same betas and eps, its moments, bias corrections and step: an effector with no
    inverse model, and a window of own acts, learn as with Adam; the lesson's gradient bounded per effector, `_timing_step`); at gain 0
    nothing moves, the moments included (nothing learned, nothing forgotten). A lesson's plasticity is the weight of the labels it
    carries, so own acts in a window whose rests act_inv has not earned (reliability 0) teach at the share of the window they fill
    (20 of 31 positions: 0.65 of a whole step), where the day's Adam taught them whole (the R6 verifier: 86% of the old learning over
    1000 lessons at the served rate, early in life, before act_inv earns its labels). Its gain gates act_pred and the correction only:
    act_inv's labels never reach the stream (`_timing_loss`). No constant is added: the rate is the waking lesson's (live_lr), the
    betas and eps Adam's defaults, as the waking lesson's optimizer has them."""

    def __init__(self, groups, lr, betas=(0.9, 0.999), eps=1e-8):
        super().__init__(groups, dict(lr=float(lr), betas=tuple(betas), eps=float(eps), gain=0.0))

    @torch.no_grad()
    def step(self):
        for g_ in self.param_groups:
            w = float(g_["gain"])
            if not w > 0.0:
                continue                                           # no earned label: no plasticity, the moments kept
            b1, b2 = g_["betas"]; a1, a2 = (1.0 - b1) * w, (1.0 - b2) * w
            for p in g_["params"]:
                if p.grad is None:
                    continue
                st = self.state[p]
                if not st:
                    st["m"] = torch.zeros_like(p, memory_format=torch.preserve_format); st["v"] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    st["q1"] = 1.0; st["q2"] = 1.0
                gh = p.grad / w                                    # u: the gradient per unit of the labels' weight
                st["m"].lerp_(gh, a1)
                st["v"].mul_(1.0 - a2).addcmul_(gh, gh, value=a2)
                st["q1"] *= 1.0 - a1; st["q2"] *= 1.0 - a2
                den = (st["v"].sqrt() / (1.0 - st["q2"]) ** 0.5).add_(g_["eps"])
                p.addcdiv_(st["m"], den, value=-float(g_["lr"]) * w / (1.0 - st["q1"]))


class TimingMixin:
    def _motor_const(self, k):
        """a motor timing constant: the body's cfg when it was given, else MOTOR's"""
        return self.cfg.get(k, MOTOR[k])

    def _gated_params(self, e):
        """later effector e's parameters whose waking plasticity its labels' reliability gates (`GatedAdam`): act_pred and the
        correction, the proposal's two terms (the forward half's lesson is its body sense's, a label felt, never read by act_inv)"""
        tm = self.m.timing[e.name]
        return list(tm.pred.parameters()) + (list(tm.cor.parameters()) if tm.sense_n else [])

    def _timing_step(self, rep, clip):
        """ACT_PRED'S AND THE CORRECTION'S STEP IN THE WAKING LESSON (after the waking lesson's own step): each later effector's group
        at the gain of its lesson, the mean weight of the window's labels (`_timing_loss`'s report "w"; 0 where it had no lesson), its
        gradient bounded by its own norm: the gradient per unit of the labels' weight (u, what GatedAdam's moments take) at most
        `clip`, the waking lesson's bound, so a lesson is bounded alike at every reliability. EACH OPTIMIZER'S OWN BOUND (the R6
        verifier's fourth look, 2026-09-24): with one bound over every parameter, as before, act_pred's gradient, which grows with
        act_inv's reliability, set the size of every step of the stream (on the tiny arm the bound held at every lesson, the joint
        norm 9 to 13, act_pred's and the corrections' the larger part), so the labels' reliability still moved the stream's learning
        with their route to it closed: with act_pred held still, the proposal's error on a trained body moved by up to 0.5 between
        reliabilities; with the bounds apart, not at all (the stream the same to the bit at every reliability, anatomy 31)"""
        for g_ in self.opt_pred.param_groups:
            r_ = rep.get(g_["name"])
            g_["gain"] = float(r_["w"]) if r_ is not None else 0.0
            if g_["gain"] > 0.0:
                torch.nn.utils.clip_grad_norm_(g_["params"], float(clip) * g_["gain"])   # |u| = |the gradient| / gain at most clip
        self.opt_pred.step()

    def _body_sense(self, e):
        """effector e's body sense this tick [sense_n] (float32, detached): its sense channel's observation at the tick's position (the
        channel of the world's frames reads the frame the tick is lived on), its own numbers there"""
        c = self.anatomy.channel(e.sense)
        with torch.no_grad():
            o = torch.as_tensor(c.observe(self, self.sil, 0), dtype=torch.float32, device=self.dev).reshape(int(c.size))
            if e.sense_idx is not None:
                o = o[torch.tensor([int(j) for j in e.sense_idx], device=o.device)]
        return o.detach().clone()

    def _window_sense(self, e, obs):
        """effector e's body sense at every position of a window's observations: [T, sense_n]"""
        x = obs[e.sense].float()
        if e.sense_idx is not None:
            x = x[..., torch.tensor([int(j) for j in e.sense_idx], device=x.device)]
        return x

    def _timing_sense(self, i):
        """THE TICK'S BODY SENSE for later effector i (step R6), before its choice: act_inv's lesson on the tick before (when the
        effector acted then: an own act, its efference copy the label), the forward half's error now, the sense kept for the next tick"""
        e = self.anatomy.effectors[i]; st = self.motor[i - 1]
        if e.sense is None:
            return
        s_now = self._body_sense(e)
        prev = st["now"]
        if e.inverse and st["sense"] is not None and prev is not None and prev["acted"]:
            self._inverse_lesson(i, st["sense"], s_now, int(prev["act"]))
        st["err"] = (s_now - st["fwd"]) if st["fwd"] is not None else None
        st["sense"] = s_now

    def _inverse_lesson(self, i, s0, s1, act):
        """ACT_INV LEARNS ONLINE (step R6; SIM_DESIGN.md 5.4): the pair of senses (s0 before the act, s1 after it) and the act the body
        made, its efference copy, the label. Its reliability is updated first, on the label it gives now against the efference copy
        (a prediction, not a fit), then one step of its own optimizer on the cross-entropy of each joint's setting"""
        e = self.anatomy.effectors[i]; st = self.motor[i - 1]
        tm = self.m.timing[e.name]; tab = self.m.get_submodule(e.organ)
        true = [int(x) for x in tab.digits(torch.tensor(int(act))).tolist()]
        with torch.no_grad():
            label = [int(lg.argmax()) for lg in tm.inverse_logits(s0, s1)]
        self._inv_rel_update(e, st, label, true)
        with torch.enable_grad():
            self.opt_inv.zero_grad(set_to_none=True)
            lg = tm.inverse_logits(s0, s1)
            loss = None
            for l_, t_ in zip(lg, true):
                ce = F.cross_entropy(l_.unsqueeze(0), torch.tensor([t_], device=l_.device))
                loss = ce if loss is None else loss + ce
            loss.backward()
            self.opt_inv.step()
            self.opt_inv.zero_grad(set_to_none=True)
        st["inv_n"] = int(st["inv_n"]) + 1
        st["inv_last"] = {"tick": self.ticks, "loss": round(float(loss.detach()), 4), "hit": [int(a == b) for a, b in zip(label, true)]}

    @staticmethod
    def _inv_conf_new(e):
        """act_inv's running confusion at birth: per joint a K x K table of zeros (the row the setting act_inv's label chose, the
        column the setting the efference copy made)"""
        return [[[0.0] * int(K) for _ in range(int(K))] for K in e.factors]

    def _inv_rel_update(self, e, st, label, true):
        """ACT_INV'S RELIABILITY: COHEN'S KAPPA OVER ITS RUNNING CONFUSION, JOINT BY JOINT (the R6 verifier's second finding,
        2026-09-24). Each own act adds one count to each joint's confusion (st["inv_conf"][j]: the row the setting act_inv's label
        chose, the column the setting the efference copy made), every count decaying over act_inv_tau acts. A joint's kappa is its
        labels' agreement with the acts with the agreement chance would reach AT THE ACTS' AND THE LABELS' OWN RATES taken out:
        (p_o - p_e) / (1 - p_e), p_o the diagonal's share, p_e the sum over settings of the label's rate times the act's rate; a joint
        whose labels and acts have never varied (p_e 1) has shown nothing, its kappa 0. The reliability, the weight a demonstration's
        label earns in act_pred's lesson, is the joints' mean kappa, each clipped to [0, 1] (the act's row is the sum of its joints'
        rows, so each joint's label is its own part of the target); zero until 64 acts, as the critics' estimators are zero until 64
        samples. st["inv_kappa"] holds each joint's kappa.
        Until 2026-09-24 this was the critics' estimator (the running moments of prediction and outcome, the slope clipped to [0, 1]),
        its samples every setting of every joint pooled: chance was 1/K for every setting, so the body's own base rates read as skill
        (the verifier's probe: a label that always said 'hold' earned 0.51 when 60% of the acts held the joint, and 0.75 at 80%; a
        label drawn at the acts' own rates, blind to the act, 0.26 at 60%). Kappa reads all of them 0.
        ITS LIMIT (the R6 verifier's third look, measured 2026-09-24; not corrected yet): the chance is taken from the label's and the
        acts' rates pooled over the confusion's whole horizon (act_inv_tau acts). Where the acts' rates shift between regimes within it,
        a label that follows the regime's most common setting without reading the motion agrees beyond the pooled chance and reads as
        skill: a label that is the mode read 0.26 to 0.65 when a joint's mode flipped (hold <-> +big at 80%, hold <-> +small at 50%,
        +small <-> -small at 60%) every 250 to 6000 acts; act_inv itself, blind (its sense pair constant) and learning online at the
        served rate, follows the mode and read 0.02 to 0.16 at 250 acts a regime, 0.20 to 0.55 at 1000, 0.24 to 0.62 at 6000 (the
        means); 0 whenever the mode stays, however far the other rates shift (hold 80% <-> 50%). For the designed child such flips are
        plausible once act_pred's acts depend on the episode (floor play, motor time, the feeds: some 250 to 2500 acts of a limb each at
        a quarter to a half of the ticks acting), never at birth (the born proposal is flat, so no setting is the mode); the parent's
        guidance never enters (a guided move is not an own act). The body sense's posture can name the regime at once, the worst case.
        THE PROPOSED CORRECTION, tested on a probe (acts drawn at known rates): each label's chance the probability the act's own choice
        gave the label's setting (the per-joint probabilities of the draw, known before the act, st["now"]["probs"], conditioned on the
        draw not being the rest; 1 or 0 for an act chosen without a draw), accumulated over the same horizon, kappa_j = (p_o - mean
        chance) / (1 - mean chance): for any label made before the act E[agree - chance] = 0 whatever the rates do, so blind labels read
        at most 0.018 in every regime above, and a label right 90% reads as Cohen's on stationary acts (0.87 at hold 60%) and 0.80 to
        0.89 when the mode flips. Left for R6h, which defines the choice's probability for a held act (the movement units) and weighs
        chunk_gate's continuations (an act chosen without a draw shows no skill when the label is right, only its absence when wrong)"""
        d = 1.0 - 1.0 / float(self._motor_const("act_inv_tau"))
        if st.get("inv_conf") is None:
            st["inv_conf"] = self._inv_conf_new(e)
        kap = []; n = 0.0
        for j, Cj in enumerate(st["inv_conf"]):
            for row in Cj:
                for k in range(len(row)):
                    row[k] *= d
            Cj[int(label[j])][int(true[j])] += 1.0
            K = len(Cj); n = sum(sum(r) for r in Cj)
            po = sum(Cj[k][k] for k in range(K)) / n
            pe = sum(sum(Cj[k]) * sum(Cj[r][k] for r in range(K)) for k in range(K)) / (n * n)
            kap.append((po - pe) / (1.0 - pe) if 1.0 - pe > 1e-9 else 0.0)
        on = n > 64
        st["inv_kappa"] = [float(k) if on else 0.0 for k in kap]
        st["inv_gain"] = float(sum(max(0.0, min(1.0, k)) for k in kap) / len(kap)) if on else 0.0

    def _timing_propose(self, e, C):
        """ACT_PRED'S PROPOSAL (the base Effector's `propose`; step R6): the forecast of its own next act from the stream C, plus the
        forward half's correction of the error its body sense shows now (none before the forward half has foreseen a tick)"""
        st = self.motor[self.anatomy.effectors.index(e) - 1]
        tm = self.m.timing[e.name]
        p = tm.pred(C)
        if e.sense is not None and st["err"] is not None:
            p = p + tm.cor(st["err"])
        return p

    def _timing_foresee(self, i):
        """THE FORWARD HALF AFTER THE ACT (step R6): the body sense at the next tick foreseen from the stream after this tick's own step
        (the act just taken is in it), kept for the next tick's error"""
        e = self.anatomy.effectors[i]; st = self.motor[i - 1]
        if e.sense is None:
            return
        C = getattr(self, "_C_last", None)
        with torch.no_grad():
            st["fwd"] = self.m.timing[e.name].fwd(C).detach().clone() if C is not None else None

    def _best_guess(self, e, pred):
        """act_pred's best guess: each joint's most likely setting under its proposal (a one-joint alphabet's reserved acts never), as
        the flat act; the rest when it proposes nothing"""
        if pred is None:
            return int(e.rest_id)
        tab = self.m.get_submodule(e.organ)
        lg = tab.logits(pred, 1.0)
        if e.reserved:
            lg[0] = lg[0].clone(); lg[0][list(e.reserved)] = float("-inf")
        return tab.flat([int(x.argmax()) for x in lg])

    def _timing_loss(self, i, C, obs):
        """THE WAKING LESSON OF LATER EFFECTOR i'S TIMING PART (step R6; body/core/cortex.py `_wake_lesson`), over the window's stream
        C [T, d] (with its gradient) and observations: act_pred's squared error to the target act's row at every position t >= 1, from
        the stream at t-1 and the forward half's error at t, weighted (1 at its own acts; at its rests act_inv's reliability on act_inv's
        label, none at the last position, whose next sense is not yet felt; 1 on its rest for an effector with no inverse model); and
        the forward half's squared error to the sense at t+1 from the stream at t. Returns (loss, report).
        THE WEIGHTS ARE ABSOLUTE (the R6 verifier's first finding, 2026-09-24): the weighted errors are averaged over the window's
        positions (T - 1), never over the weights' sum. Divided by the weights' sum, the reliability only reweighted the rests against
        the own acts and never scaled them: a window of rests alone, or of the parent's hand alone, taught at full strength at any
        reliability above 0 (the verifier's probe: the same loss and gradient at 0.01 as at 1). Over the positions, a label act_inv
        has not earned teaches only in proportion to what it has earned, and an own act teaches as it did in a window of own acts.
        The report's "w" is the lesson's mean label weight over the same positions, the gain at which act_pred and the correction step
        (`GatedAdam`: the loss's weights reach the gradient, and Adam alone would undo their scale).
        ACT_INV'S LABELS TEACH act_pred AND THE CORRECTION, NEVER THE STREAM (the R6 verifier's fourth look, 2026-09-24): at a position
        whose target is act_inv's label the proposal reads the stream detached. The stream steps with the waking lesson's Adam, which
        divides each parameter's step by its recent gradient size, so the share of these labels that reached it through the proposal
        was taught at about its whole strength whatever the reliability: once act_pred had learned (the tiny arm at the served rate),
        1000 lessons at reliability 0.01 taught the proposal 9 to 21 times what the same labels taught given whole (every hundredth
        lesson at 1), nearly all of it through the stream. No scale on the loss can gate a parameter Adam steps; only an optimizer whose
        step carries the gain can, and the stream is shared by every lesson (the words, the channels' forecasts, the forward halves,
        the own acts). So the labels an organ infers teach only the organs that read the stream, as the prefrontal heads learn from the
        stream and not through it: the stream learns what it senses and what it does. The motion act_inv reads still reaches it whole,
        as the body sense the forward half foresees (a label felt, weight 1), and so does every own act, through its efference copy
        (weight 1), and the rest of an effector with no inverse model. THE COST, measured on the tiny arm at reliability 1 against
        letting the stream learn these labels through a gated optimizer of its own (a second Adam over the stream, its step at the
        labels' weight; from the same trained state): the parent's guidance taught the proposal 0.35 / 0.79 / 1.19 in 100 / 300 / 1000
        lessons against 0.97 / 1.40 / 1.50, and on a held-out window of the same guidance 0.14 / 0.32 / 0.60 against 0.33 / 0.48 /
        0.42: slower, with no lower ceiling (what the stream's route added was mostly the window taught); newborn, about the same as
        the old route (0.20 / 0.33 / 0.62 against 0.20 / 0.34 / 0.87). The gated second optimizer was not taken: it gives the stream
        a second whole step beside the day's wherever demonstrations are, needs a gain for the shared stream where several effectors'
        labels meet (their weights' sum passes one), costs a second backward pass through the stream and its moments twice over, and
        its own ratio to the same labels given whole wandered (0.17 to 1.46), the stream's path then depending on the reliability."""
        e = self.anatomy.effectors[i]; st = self.motor[i - 1]
        tm = self.m.timing[e.name]; tab = self.m.get_submodule(e.organ)
        acts = obs[e.name]
        T = int(acts.shape[0])
        if T < 2:
            return None, None
        rest = int(e.rest_id)
        own = acts != rest
        tgt = acts.clone()
        wt = torch.ones(T, device=C.device)
        s = self._window_sense(e, obs) if e.sense is not None else None
        gain = 0.0
        if e.inverse:
            gain = float(st["inv_gain"])
            with torch.no_grad():
                lab = tab.flat_many([lg.argmax(-1) for lg in tm.inverse_logits(s[:-1], s[1:])])   # [T-1]: what moved it from t to t+1
            rested = ~own
            tgt[:-1] = torch.where(rested[:-1], lab, acts[:-1])
            wt = torch.where(rested, torch.full_like(wt, gain), wt)
            if bool(rested[-1]):
                wt[-1] = 0.0                                               # its next sense is not yet felt: no label
        Cp = C[:-1]
        if e.inverse and bool(rested[1:].any()):
            # ACT_INV'S LABELS TEACH THE PROPOSAL, NOT THE STREAM: at a position whose target is act_inv's label the stream is read
            # detached, so that lesson reaches act_pred and the correction alone (their plasticity gated by the label's reliability)
            Cp = torch.where(rested[1:].unsqueeze(-1), Cp.detach(), Cp)
        P = tm.pred(Cp)
        lf = None
        if s is not None:
            fw = tm.fwd(C[:-1])                                            # the sense at t+1 foreseen from the stream at t
            P = P + tm.cor(s[1:] - fw.detach())                          # the proposal at t+1 reads the error there
            lf = 0.5 * ((fw.float() - s[1:].float()) ** 2).sum(-1).mean()
        with torch.no_grad():
            rows = tab(tgt[1:])
        w1 = wt[1:]
        lp = (0.5 * ((P.float() - rows.float()) ** 2).sum(-1) * w1).sum() / float(T - 1)   # over the positions: the weights absolute
        loss = lp if lf is None else lp + lf
        rep = {"pred": round(float(lp.detach()), 4), "own": int(own[1:].sum()), "demo_w": round(gain, 3),
               "w": float(w1.sum()) / float(T - 1)}                       # the lesson's mean label weight: act_pred's plasticity (GatedAdam)
        if lf is not None:
            rep["fwd"] = round(float(lf.detach()), 4)
        return loss, rep

    def _timing_report(self, i):
        """the timing part's instruments for later effector i: act_inv's reliability and last lesson, the chunks and their stops"""
        e = self.anatomy.effectors[i]; st = self.motor[i - 1]
        out = {"chunks": int(st["chunks"]), "stops": dict(st["stops"]), "chunk": int(st["chunk"])}
        if e.inverse:
            out.update({"inv_gain": round(float(st["inv_gain"]), 3), "inv_kappa": [round(float(k_), 3) for k_ in st["inv_kappa"]],
                        "inv_n": int(st["inv_n"]), "inv_last": st["inv_last"]})
        return out
