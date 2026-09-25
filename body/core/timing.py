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
  act_pred and its correction step with an optimizer of their own (opt_pred, a group per effector), one step a lesson: what it
  writes is the lesson at its labels' weights (1 at an own act, and at an effector's rest where it has no inverse model; act_inv's
  reliability at a label), in its first moment and its step size at the lesson's mean label weight; the scale it writes against,
  its second moment, is the same lesson with every label earned, whatever the reliability; its gradient bounded by its own norm.
  Every other parameter steps with the waking lesson's Adam as before, bounded by theirs.
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
    Adam in which each step is one lesson of weight `gain` (each group's, set before the step: the lesson's mean label weight over the
    window's positions, 1 at an own act, its efference copy, and act_inv's reliability at a rest act_inv labelled). The gradient is
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
    nothing is written. Its gain gates act_pred and the correction only: act_inv's labels never reach the stream (`_timing_loss`). No
    constant is added: the rate is the waking lesson's (live_lr), the betas and eps Adam's defaults, as the waking lesson's optimizer
    has them.
    THE SCALE IS THE LESSON WITH EVERY LABEL EARNED (R6 fix 6, the R6 verifier's fifth and sixth looks, 2026-09-24). A lesson of an
    effector with an inverse model carries two kinds of target, its own acts (weight 1) and act_inv's labels (weight the reliability
    g), on the same act_pred and correction. What it writes, its first moment and its step, is the lesson at those weights: its
    gradient the own acts' plus g times the labels' at full reliability, per unit of its mean label weight w = w_own + g s_lab (s_lab
    the labels' share of the positions), moved and stepped in proportion to w as above. The scale it is written against, its second
    moment, is the same lesson with every label earned: the own acts' gradient plus the labels' whole, per unit of its weight W = w_own
    + s_lab, moved in proportion to W, whatever g (`step`'s `full`). So the step is Adam's step on the whole lesson with each label's
    part of it taken at its reliability: the reliability gates what is written, never the scale it is written against. Where the
    lesson has no label the two are one (W = w), as before.
    - THE FIFTH LOOK: with the scale taken from the lesson as written (R6 fix 4, one sample of gain w), a lesson whose own acts were
      fitted had a small gradient but for its labels', which the reliability had made small, and Adam's division by the recent
      gradient size re-inflated them to w of a step at any g, the own acts' share (the verifier, the tiny arm with its own acts fitted
      in the mixed window, at the served rate: act_pred's step at reliability 0.01 / 0.05 / 0.25 was 0.070 / 0.27 / 0.66 of its step at
      1, and 1000 lessons at 0.01 taught 0.18 of what they taught at 1, 3.1 times what the same labels taught given whole). With the
      scale the lesson with every label earned, it does not shrink with g, and the labels' part of the step is g times its part at 1:
      from the same state (the verifier's fixture, the stream learning) 0.011 / 0.047 / 0.23 of a whole step, the share learned at 0.01
      0.010 and 0.007 after 300 and 1000 lessons, the same labels given whole teaching as much (1.26 and 1.04); a reliability of 0.0001
      and 0.001 taught 0.00012 and 0.0012 of what 1 did after 300 lessons from birth, the stream still (anatomy 31).
    - THE SIXTH LOOK: R6 fix 5 made the own acts and the labels two samples, each with moments of its own (a second GatedAdam,
      opt_lab), so each sample's step was divided by the recent size of its own gradient alone: each step the size of its weight
      whatever the size of its error, and the two together did not descend the lesson's loss. Where the own acts and the labels pulled
      act_pred apart, the sample of the larger weight won and the other was un-learned below its birth (the verifier, from birth at
      reliability 1, the stream still, at a hundred times the served rate: the labels' error 1.26 -> 2.95 beside own acts fitted to
      0.0003; in a window of labels mostly the own acts' error rose; the same at the served rate over 30000 lessons). One step through
      one scale descends the whole lesson, as Adam does: from birth both errors fall together (anatomy 34), and at reliability 1 the
      lesson is Adam's own on the whole lesson, as R6 fix 4's was (the same errors to four places on anatomy 34's windows).
    - WHAT IT COSTS: the labels' size enters the scale whatever their reliability, at 0 too (a reliability of 0.0001 then writes 0.0001
      of the labels' part and changes nothing else; a scale that took them only above 0 would change the own acts' steps at the
      first earned label, as much at 0.0001 as at 1). So the own acts beside rests step as Adam would step them in the whole lesson,
      their part of it, where R6 fix 3 to 5 stepped them at their share of the window against their own gradient's size alone: at
      reliability 0 beside the mixed window's rests, the served rate, the own acts' error after 1000 lessons 0.25 from birth (0.23
      under R6 fix 3 to 5), 0.17 after 300 lessons of own acts first (0.13), 0.005 from states trained on own acts (0.002 to 0.003);
      at reliability 1, 0.14 from birth, as under R6 fix 4 (0.28 under R6 fix 5)."""

    def __init__(self, groups, lr, betas=(0.9, 0.999), eps=1e-8):
        super().__init__(groups, dict(lr=float(lr), betas=tuple(betas), eps=float(eps), gain=0.0))

    @torch.no_grad()
    def step(self, full=None):
        """one lesson's step. `full` (the waking lesson's, `_timing_step`): {group name: (W, [tensors])} for each group whose lesson
        carried act_inv's labels, the lesson's gradient with every label at full reliability (bounded as p.grad) and W its weight,
        which the second moment takes whatever the reliability; a group without it is its own acts' lesson alone (W = the gain, the
        second moment on p.grad), as before"""
        full = full or {}
        for g_ in self.param_groups:
            w = float(g_["gain"])
            W, fl_ = full.get(g_.get("name"), (w, None))
            W = float(W)
            if not W > 0.0:
                continue                                           # no label weighed, earned or not: nothing moves
            b1, b2 = g_["betas"]; a1, a2 = (1.0 - b1) * w, (1.0 - b2) * W
            for k_, p in enumerate(g_["params"]):
                if p.grad is None:
                    continue
                st = self.state[p]
                if not st:
                    st["m"] = torch.zeros_like(p, memory_format=torch.preserve_format); st["v"] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    st["q1"] = 1.0; st["q2"] = 1.0
                gh = p.grad / w if w > 0.0 else None               # u: the gradient per unit of the labels' weight
                fh = gh if fl_ is None else fl_[k_] / W            # the scale's: the same lesson with every label earned, per unit of W
                if gh is not None:
                    st["m"].lerp_(gh, a1)
                st["v"].mul_(1.0 - a2).addcmul_(fh, fh, value=a2)
                if gh is not None:
                    st["q1"] *= 1.0 - a1
                st["q2"] *= 1.0 - a2
                if gh is None:
                    continue                                       # nothing earned: nothing written (the scale taken)
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

    def _gated_names(self, e):
        """the names of `_gated_params(e)` in its order, within e's timing organ (pred.weight, pred.bias, cor.weight)"""
        tm = self.m.timing[e.name]
        return [f"pred.{n}" for n, _ in tm.pred.named_parameters()] + ([f"cor.{n}" for n, _ in tm.cor.named_parameters()] if tm.sense_n else [])

    def _timing_label_grads(self, labs):
        """act_inv's labels' gradient on act_pred and the correction, AT FULL RELIABILITY: for each later effector whose lesson carried
        labels (`_timing_loss`'s third value, every label at weight 1, at the waking lesson's scale), taken on act_pred's and the
        correction's parameters alone (the labels never reach the stream) before the lesson's backward frees the graph: {name: [grads]}"""
        out = {}
        for g_ in (self.opt_pred.param_groups if labs else ()):
            lb_ = labs.get(g_["name"])
            if lb_ is None:
                continue
            gs_ = torch.autograd.grad(lb_, g_["params"], retain_graph=True, allow_unused=True)
            out[g_["name"]] = [torch.zeros_like(p_) if q_ is None else q_ for p_, q_ in zip(g_["params"], gs_)]
        return out

    def _timing_step(self, rep, clip, glab=None):
        """ACT_PRED'S AND THE CORRECTION'S STEP IN THE WAKING LESSON (after the waking lesson's own step): each later effector's group
        of opt_pred (GatedAdam), one step a lesson. A lesson without act_inv's labels (an effector with no inverse model, a window of
        own acts) steps at its own acts' share of the positions ("w_own") on the gradient the lesson's backward left, bounded per unit
        of that weight at `clip`, as before. A lesson with them: the gradient written is the lesson's at the labels' reliability g, the
        own acts' (the backward's) plus g times the labels' at full reliability (`glab`), at the gain w = w_own + g s_lab; the scale is
        the same lesson with every label earned, the own acts' plus the labels' whole, at its weight W = w_own + s_lab ("s_lab": the
        labels' share of the positions), whatever g; both bounded by the one factor that bounds the whole lesson per unit of W at
        `clip`, so the bound too is the same at every reliability. EACH OPTIMIZER'S OWN BOUND (the R6 verifier's fourth look,
        2026-09-24): with one bound over every parameter, as before, act_pred's gradient, which grows with act_inv's reliability, set
        the size of every step of the stream (on the tiny arm the bound held at every lesson, the joint norm 9 to 13, act_pred's and the
        corrections' the larger part), so the labels' reliability still moved the stream's learning with their route to it closed: with
        act_pred held still, the proposal's error on a trained body moved by up to 0.5 between reliabilities; with the bounds apart,
        not at all (the stream the same to the bit at every reliability, anatomy 31)"""
        full = {}
        for g_ in self.opt_pred.param_groups:
            r_ = rep.get(g_["name"]); q_ = (glab or {}).get(g_["name"])
            if r_ is None:
                g_["gain"] = 0.0
                continue
            if q_ is None:                                         # no label: the own acts' lesson alone, as before
                g_["gain"] = float(r_["w_own"])
                if g_["gain"] > 0.0:
                    torch.nn.utils.clip_grad_norm_(g_["params"], float(clip) * g_["gain"])   # |u| = |the gradient| / gain at most clip
                continue
            W = float(r_["w_own"]) + float(r_["s_lab"]); rel = float(r_["rel"])
            own_ = [p_.grad if p_.grad is not None else torch.zeros_like(p_) for p_ in g_["params"]]
            whole = [a_ + b_ for a_, b_ in zip(own_, q_)]          # the lesson with every label earned
            tot = torch.linalg.vector_norm(torch.stack([torch.linalg.vector_norm(x_) for x_ in whole]))
            k_ = torch.clamp(float(clip) * W / (tot + 1e-6), max=1.0)   # its bound per unit of W, the one factor for both
            for p_, a_, b_ in zip(g_["params"], own_, q_):
                p_.grad = (a_ + rel * b_) * k_                     # what is written: the labels at their reliability
            g_["gain"] = float(r_["w_own"]) + float(r_["w_lab"])
            full[g_["name"]] = (W, [x_ * k_ for x_ in whole])
        if full:
            self.opt_pred.step(full=full)
        else:
            self.opt_pred.step()                                   # no label in any lesson: as before

    def _gated_moments(self, e):
        """later effector e's act_pred and correction moments in opt_pred, by parameter name, copied to the CPU: {"lesson": {name: {m,
        v, q1, q2}}} (empty before its first lesson). Saved with the body (body/core/persistence.py)"""
        opt = getattr(self, "opt_pred", None)
        g_ = next((x_ for x_ in (opt.param_groups if opt is not None else ()) if x_["name"] == e.name), None)
        if g_ is None:
            return {}
        return {"lesson": {n_: {"m": opt.state[p_]["m"].detach().cpu().clone(), "v": opt.state[p_]["v"].detach().cpu().clone(),
                                "q1": float(opt.state[p_]["q1"]), "q2": float(opt.state[p_]["q2"])}
                           for n_, p_ in zip(self._gated_names(e), g_["params"]) if opt.state.get(p_)}}

    def _gated_moments_load(self, e, saved):
        """a save's moments for later effector e (`_gated_moments`' form) into opt_pred, where each parameter's name and shape match;
        returns what could not be loaded (as born): a parameter this organ has not, or a shape it has not, and every moment of a save
        whose form is not this one (R6 fix 5's two samples, "own" and "labels", 2026-09-24: moments of another optimizer)"""
        opt = getattr(self, "opt_pred", None)
        saved = saved or {}
        bad = [f"{k_}.{n_}" for k_, v_ in saved.items() if k_ != "lesson" for n_ in (v_ or {})]
        sv_ = saved.get("lesson") or {}
        g_ = next((x_ for x_ in (opt.param_groups if opt is not None else ()) if x_["name"] == e.name), None)
        if g_ is None:
            return bad + [f"lesson.{n_}" for n_ in sv_]
        names = self._gated_names(e)
        bad += [f"lesson.{n_}" for n_ in sv_ if n_ not in names]
        for n_, p_ in zip(names, g_["params"]):
            s_ = sv_.get(n_)
            if s_ is None:
                continue
            if tuple(s_["m"].shape) != tuple(p_.shape) or tuple(s_["v"].shape) != tuple(p_.shape):
                bad.append(f"lesson.{n_}"); continue
            opt.state[p_] = {"m": s_["m"].to(p_.device, p_.dtype).clone(), "v": s_["v"].to(p_.device, p_.dtype).clone(),
                             "q1": float(s_["q1"]), "q2": float(s_["q2"])}
        return bad

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
        the forward half's squared error to the sense at t+1 from the stream at t. Returns (loss, report, labels): `loss` the own acts'
        errors (and the rests of an effector with no inverse model) and the forward half's, which the waking lesson's backward carries;
        `labels` act_inv's labels' errors AT FULL RELIABILITY (weight 1 at each labelled rest whose next sense is felt, over the same
        positions), None where the window has none, whatever the reliability (the scale `_timing_step` steps against takes them whole,
        R6 fix 6). The whole lesson is loss + reliability x labels. The report's "w" is the lesson's mean label weight, "w_own" and
        "w_lab" its own acts' and its labels' parts, "s_lab" the labels' share of the positions (their weight at full reliability) and
        "rel" the reliability they were weighed at.
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
            return None, None, None
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
        err = 0.5 * ((P.float() - rows.float()) ** 2).sum(-1)
        # THE LABELS APART (the R6 verifier's fifth and sixth looks): the own acts' errors (and an uninverted effector's rests) and
        # act_inv's labels' at full reliability (weight 1 at each labelled rest whose next sense is felt), each over the window's
        # positions (the weights absolute); the lesson is the own acts' plus the reliability times the labels'. A window with no
        # labelled rest is the own acts' alone, as before
        w_own, lab_f = w1, None
        if e.inverse and bool(rested[1:].any()):
            lab1 = rested[1:]
            lab_f = lab1.clone()
            if bool(rested[-1]):
                lab_f[-1] = False                                          # the last rest: its next sense not felt, no label
            w_own = torch.where(lab1, torch.zeros_like(w1), w1)
        lp = (err * w_own).sum() / float(T - 1)                           # over the positions: the weights absolute
        n_lab = int(lab_f.sum()) if lab_f is not None else 0
        lb = (err * lab_f.float()).sum() / float(T - 1) if n_lab else None   # act_inv's labels, every one earned
        loss = lp if lf is None else lp + lf
        s_lab = n_lab / float(T - 1)
        rep = {"pred": round(float(lp.detach()) + (gain * float(lb.detach()) if lb is not None else 0.0), 4), "own": int(own[1:].sum()),
               "demo_w": round(gain, 3),
               "w": float(w1.sum()) / float(T - 1),                       # the lesson's mean label weight
               "w_own": float(w_own.sum()) / float(T - 1),                # the own acts' share of it
               "w_lab": gain * s_lab,                                     # act_inv's labels' share of it: the reliability times their share
               "s_lab": s_lab, "rel": gain}                               # their share of the positions, the reliability they were weighed at
        if lf is not None:
            rep["fwd"] = round(float(lf.detach()), 4)
        return loss, rep, lb

    def _timing_report(self, i):
        """the timing part's instruments for later effector i: act_inv's reliability and last lesson, the chunks and their stops"""
        e = self.anatomy.effectors[i]; st = self.motor[i - 1]
        out = {"chunks": int(st["chunks"]), "stops": dict(st["stops"]), "chunk": int(st["chunk"])}
        if e.inverse:
            out.update({"inv_gain": round(float(st["inv_gain"]), 3), "inv_kappa": [round(float(k_), 3) for k_ in st["inv_kappa"]],
                        "inv_n": int(st["inv_n"]), "inv_last": st["inv_last"]})
        return out
