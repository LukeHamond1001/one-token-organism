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
  s[t+1]. Through the cortex, at the waking lesson's rate, as a channel's forecast. ACT_PRED'S PLASTICITY IS GATED BY ITS LABELS'
  RELIABILITY (`GatedAdam`, `_timing_step`): act_pred and its correction step with an optimizer of their own, a group per effector,
  each lesson a sample of weight the lesson's mean label weight in the group's moments and in its step size; every other parameter
  steps with the waking lesson's Adam as before.
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
      teach (anatomy 31: 0.011 to 0.024 of it at 0.01).
    At gain 1 on every step this is the waking lesson's Adam exactly (the same betas and eps, its moments, bias corrections and step:
    an effector with no inverse model, and a window of own acts, learn as they did); at gain 0 nothing moves, the moments included
    (nothing learned, nothing forgotten). No constant is added: the rate is the waking lesson's (live_lr), the betas and eps Adam's
    defaults, as the waking lesson's optimizer has them."""

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

    def _timing_step(self, rep):
        """ACT_PRED'S AND THE CORRECTION'S STEP IN THE WAKING LESSON (after the waking lesson's own step; the gradients clipped with
        every other parameter's): each later effector's group at the gain of its lesson, the mean weight of the window's labels
        (`_timing_loss`'s report "w"; 0 where it had no lesson)"""
        for g_ in self.opt_pred.param_groups:
            r_ = rep.get(g_["name"])
            g_["gain"] = float(r_["w"]) if r_ is not None else 0.0
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
        label drawn at the acts' own rates, blind to the act, 0.26 at 60%). Kappa reads all of them 0"""
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
        (`GatedAdam`: the loss's weights reach the gradient, and Adam alone would undo their scale)."""
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
        P = tm.pred(C[:-1])
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
