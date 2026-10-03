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
  the correction and never the stream. ACT_PRED LEARNS BY PLAIN STEPS, ITS LABELS AT THEIR RELIABILITY (`GatedDescent`,
  `_timing_step`; R6 fix 7): act_pred and its correction step with an optimizer of their own (opt_pred, a group per effector), one
  plain gradient step a lesson on the lesson's gradient, its labels weighted as above (1 at an own act, and at an effector's rest
  where it has no inverse model; act_inv's reliability at a label), each element's step bounded, with no state, so a label teaches its
  reliability times what it teaches whole. Every other parameter steps with the waking lesson's Adam as before, bounded by theirs.
THE LEARNED STOPS (chunk_gate, every later effector; `_choose_effector`): a chunk of acts continues, the act act_pred's best guess
(each joint's most likely setting, no draw), until that guess is the effector's rest (the learned end), its gate's own draw closes,
its reflex fires, its declared end act closed the chunk before, or chunk_max acts have run (a ceiling: the next act is a fresh
decision). With chunk_gate 0 every tick is a fresh decision, as before R6.

The one-tick timing the forward half serves: on a tick the world is quiet the choice reads the stream of the last full position, the
tick before, so the sense felt now enters the proposal only through the forward half's error (SIM_DESIGN.md 5.2's one forward a tick
for the sim, its own acts entering at the next frame, is a later step). On a tick a world symbol opens a position the choice reads it
there, while the lesson teaches act_pred from the position before, as the words' lesson does.

STEP R6h (SIM_DESIGN.md 3.5, 3.6; each under a MOTOR constant absent unless given, R6's law without it): THE MOVEMENT UNIT (`_unit_hold`,
unit_margin): a unit under way holds its act joint by joint unless the choice prefers another setting by more than the persistence
margin, in place of act_pred's best guess; ACT_INV BATCHED (`_inverse_batch`, act_inv_every): its pairs gathered and learned in one step
every so many ticks, a batch of one R6's lesson; THE KAPPA CORRECTION (`_act_chance`, act_inv_chance): each label's chance the act's own
choice's probability of it; THE FORWARD ERROR AT THE GATE (`_fwd_err_in`, the effector's `fwd_gate`): the size of the error its body
sense shows now against its forward half's foresight, one of its gate's own inputs."""
import torch
import torch.nn.functional as F

from .physiology import MOTOR



def _against(P, rows):
    """A153 (2026-10-01): TEACHING AGAINST AN ACT, BOUNDED. act_pred's lesson toward an act is the squared distance of its forecast P to
    the act's row (toward); a negative weight (A150, A152: dopamine's credit against the act) cannot be the same distance with its sign
    turned, since pushing P away from a row has no end: on the night of life day 50 the reel's 2,048 dreams, half their positions
    weighted against, drove the forecasts' norm from 2.3 to 3,600 and the readout (sharpness x cos x |P|^g, C148) to a probability of 1
    on one setting of every joint, the arms frozen at dawn 51. Against is the hinge of the cosine between P and the act's row,
    relu(cos): the forecast is turned away from the act until it is indifferent to it (cos 0), the gradient orthogonal to P (the norm
    untouched) and zero past indifference; the association weakened toward nothing, as LTD weakens a trace, never driven negative.
    [T] per position"""
    return torch.relu(torch.nn.functional.cosine_similarity(P.float(), rows.float(), dim=-1))

class GatedDescent(torch.optim.Optimizer):
    """ACT_PRED'S PLASTICITY: PLAIN GRADIENT DESCENT, GATED BY THE RELIABILITY OF ITS LABELS (R6 fix 7, the lead's decision of
    2026-09-24 after the R6 verifiers' third finding and their fourth to seventh looks; SIM_DESIGN.md 5.4). Each lesson moves each
    element of act_pred and its correction by

        step = clamp(lr_motor x G, -B, +B),    G = (the sum over the window's positions of w_i x grad_i) / (the number of positions)

    grad_i the gradient of position i's squared error (the proposal against its target's row), w_i its weight: 1 at an own act (its
    efference copy) and at an uninverted effector's rest, act_inv's reliability at act_inv's label, 0 at the last rest (`_timing_loss`).
    The waking lesson's plasticity scale ((1 + stress / 10) x the wake gate, which the lesson's backward leaves on every parameter's
    gradient, `_wake_lesson`) is divided out before this step (`_timing_step`; R6 fix 8), so G is the lesson's own gradient at any
    stress and any gate. NO STATE: nothing of one lesson reaches the next but the weights themselves. So WHAT A LABEL TEACHES IS ITS
    RELIABILITY TIMES WHAT IT TEACHES WHOLE, BY CONSTRUCTION: below the bound the step is linear in G, and G is linear in each weight;
    no running normalisation divides a small gradient back up to a whole step. A synapse changes with its error and with a
    neuromodulatory signal of how far the teaching may be trusted (acetylcholine as expected uncertainty; Yu and Dayan 2005), with no
    running renormalisation.
    WHY NOT ADAM (R6 fix 1 to 6): Adam divides each parameter's step by the recent size of its gradient, which undoes any scale put on a
    lesson (fix 1: at reliability 0.01 act_pred learned 0.44 to 0.95 of what it learned at 1). Every adaptation left a state that could
    re-inflate a small gradient or outgrow the rate: a gain on Adam's step and moments (fix 3) re-inflated the labels beside fitted own
    acts (the fifth look: 0.07 of a whole step at 0.01); moments per source (fix 5) fought, un-learning one another (the sixth look); a
    scale taken from another gradient (fix 6) had no bound per element, up to thousands of times the rate (the seventh look). The class
    is closed: no adaptive second moment reaches act_pred or its correction.
    THE CONSTANTS (physiology.py MOTOR; `lr` the group's, the waking lesson's rate live_lr, so a raised waking rate raises both):
        lr_motor = act_pred_rate x sqrt(d) x live_lr = 2 sqrt(d) x live_lr,    B = act_pred_bound x sqrt(d) x lr_motor = 2 d x live_lr
    d act_pred's fan-in, the stream's width. HOW THEY ARE DERIVED: at birth Adam moves every element of act_pred by exactly its rate.
    The plain step moves element ij by lr_motor |G_ij|, and at birth |G_ij| is about c sqrt(J / d): the stream is LayerNorm'd (its
    features, act_pred's inputs, of unit size), an act's row is the sum of its J joints' unit rows and the born proposal nearly 0, so the
    error, of norm sqrt(J), is spread over act_pred's d outputs; c is the coherence of the window's positions (their targets and inputs
    partly cancel). So lr_motor = sqrt(d) x live_lr / (c sqrt(J)): sqrt(d) from the fan-in, and 1 / (c sqrt(J)) = 2, act_pred_rate,
    MEASURED on the tiny arm (d 64, J 2; 2026-09-24, under R6 fix 6's GatedAdam, which is Adam at weight 1): over its first 50 lessons
    from birth on its own acts Adam moved act_pred 16.0 times the rate per unit of the lesson's gradient norm, the plain step 2 sqrt(64)
    = 16. From birth on the other windows Adam's was 6.3 (rests alone: one label, the hold, makes the gradient coherent) to 12.2, and
    from trained and fitted states 9 to 240 (Adam
    steps the rate whatever the gradient's size; the plain step goes by the size of the error, so a fitted act steps little); over
    every state and window at weight 1 the geometric mean was 23. THE BOUND is the step of a per-element gradient of sqrt(d): a whole
    unit row's error on an input carrying the LayerNorm's whole norm (|x_j| <= sqrt(d)). No lesson at birth asks that much of an element,
    so it holds only a gradient no lesson at birth can make (a proposal run away, a body sense far outside its range through the
    correction), never an ordinary lesson (anatomy 31 and 35: the largest element's step in every state measured, at the served rate
    and a hundred times it, 0.17 of the bound at most, some 20 times the rate; over 3000-lesson runs from every state 0.23; the same
    at any stress and gate, the scale divided out, R6 fix 8), and the linearity above holds wherever it is not reached. For the sim body (SIM_DESIGN.md 6.3: d 512, the served rate 1e-5): lr_motor = 2 sqrt(512) x 1e-5 = 4.5e-4 per unit of
    gradient, B = 1.0e-2 per element per lesson (1024 times the rate).
    WHAT IT CHANGES (the tiny arm, the served rate unless said; the tables in R6 fix 7's commit message):
    - A LABEL TEACHES ITS RELIABILITY'S SHARE IN EVERY STATE: one lesson's gradient at 0.01, 0.05 and 0.25 is that share of its change
      from 0 to 1 to float32's rounding (anatomy 31), and the same label mass given whole teaches the same: on rests, guidance and the
      mixed window from birth, own acts first, trained and fitted, 300 to 3000 lessons at the served rate, 0.93 to 1.03 at 0.05 and
      0.99 to 1.00 at 0.25 (0.67 to 1.01 at 0.01, the side of teaching less, where 300 lessons hold three whole ones); nothing
      re-inflates, as nothing is kept.
    - OWN ACTS learn at weight 1 whatever the labels beside them, by the size of their error: the first lessons about as under Adam
      (from birth on its own acts, the error 1.01 -> 0.76 after 300 lessons; Adam 0.66), the later ones slower, as descent slows
      where the error is small and in the directions the lesson curves least, where Adam's division does not (0.53 and 0.087 after
      1000 and 3000 lessons; Adam 0.14 and 0.006); at a hundred times the rate both fit (0.0004 against 0.0000 after 3000). With the
      stream still (anatomy 34), 3000 lessons at a hundred times the rate leave 0.03 to 0.29 of the birth errors, at ten times 0.17
      to 0.37 (Adam 0.0001 to 0.08, and 0.009 to 0.10). A state whose own acts are fitted steps them little, where Adam stepped the
      rate.
    - THE CORRECTION steps by its input's size, the body sense's error in the world's units, where Adam stepped each weight the rate
      whatever that size (from birth on the arm's own acts Adam moved it 30 to 33 times the rate per unit of its gradient, the plain
      step 16; where the sense does not move, as on rests alone, it does not learn).
    - STRESS AND THE WAKE GATE DO NOT REACH THIS STEP (R6 fix 8, the R6 verifier's eighth look): the lesson's scale ((1 + stress / 10)
      x the wake gate, "stress raises plasticity") multiplies every parameter's gradient; Adam cancels a steady scale (it divides each
      step by the gradient's recent size), so on the day's parameters it moves the step only while it changes. The plain step has no
      such division, and R6 fix 7 kept the scale on it, so act_pred was the one waking parameter stress reached, and to the full: four
      times at the ceiling (30), one lesson's largest element 0.63 of the bound from birth on the tiny arm (0.16 at stress 0); at a
      hundred times the served rate the bound held every element and the arm diverged (its errors 1.1 -> 257 in 10 lessons at
      reliability 1, 59 at 0.25; the verifier's own states read 181 to 231, reliability's order inverted), and at the sim body's width
      (the arm born at d 512) four times the served rate under stress 30 diverged (1.0 -> 2530 in 25 lessons). The lead's step has no
      scale in it (G above), so `_timing_step` divides the scale out: the step is the one at stress 0 and gate 1 at any stress and any
      gate, to float32's rounding (anatomy 35); at a hundred times the served rate under stress 30 the arm learns as at stress 0, and
      at d 512 eight times the served rate under stress 30 learns (1.0 -> 0.08 in 200 lessons, its largest element 0.016 of the
      bound). A lesson the gate shuts wholly (a scale of 0) leaves no gradient to step, act_pred's included: it writes nothing.
    - STABILITY: plain descent is stable while lr_motor x the lesson's largest curvature stays below 2. That curvature is about d on
      these windows (the LayerNorm'd stream's positions share a direction: 0.11 d to 0.98 d measured), so the product is about 2 d^1.5 x
      live_lr: on the arm 0.01 at the served rate and at most 1.0 at a hundred times it (where anatomy 31 and 35 run); for the sim body
      (d 512, 1e-5) about 0.23, so stable up to some eight times the served rate, at any stress (R6 fix 8). Past that the stiffest
      direction oscillates, each element held at the bound (anatomy 25 raises act_pred's waking rate to 1e-3, not the day's 3e-3).
      act_pred_rate could rise to about 4 before a hundred times the rate turns unstable on the arm.
    - THE SAVE holds no optimizer state for act_pred (R6 fix 5 and 6's saves load with a note, their moments not read); the lesson
      takes one backward pass where R6 fix 5 and 6 took a second for the labels, and keeps no moments."""

    def __init__(self, groups, lr, rate, bound):
        super().__init__(groups, dict(lr=float(lr), rate=float(rate), bound=float(bound)))

    @torch.no_grad()
    def step(self):
        """one lesson's step: each element moved by the plain step on its gradient, at most the bound"""
        for g_ in self.param_groups:
            s_ = float(g_["lr"]) * float(g_["rate"]); b_ = float(g_["lr"]) * float(g_["bound"])
            if not s_ > 0.0:
                continue
            for p in g_["params"]:
                if p.grad is None:
                    continue
                p.sub_((p.grad * s_).clamp_(-b_, b_))


class TimingMixin:
    def _motor_const(self, k):
        """a motor timing constant: the body's cfg when it was given, else MOTOR's"""
        return self.cfg.get(k, MOTOR[k])

    def _gated_params(self, e):
        """later effector e's parameters whose waking plasticity its labels' reliability gates (`GatedDescent`): act_pred and the
        correction, the proposal's two terms (the forward half's lesson is its body sense's, a label felt, never read by act_inv), and
        (step R7f) its recall map, the proposal's third, which learns through the same lesson and the same plain step"""
        tm = self.m.timing[e.name]
        rc = list(self.m.recall[e.name].parameters()) if "recall" in self.m._modules else []   # step R7f: its recall map, the proposal's third term
        return list(tm.pred.parameters()) + (list(tm.cor.parameters()) if tm.sense_n else []) + rc

    def _gated_names(self, e):
        """the names of `_gated_params(e)` in its order, within e's timing organ (pred.weight, pred.bias, cor.weight), then (step R7f)
        its recall map's beside it (recall.weight: the organs' recall[e.name])"""
        tm = self.m.timing[e.name]
        rc = [f"recall.{n}" for n, _ in self.m.recall[e.name].named_parameters()] if "recall" in self.m._modules else []
        return [f"pred.{n}" for n, _ in tm.pred.named_parameters()] + ([f"cor.{n}" for n, _ in tm.cor.named_parameters()] if tm.sense_n else []) + rc

    def _timing_step(self, scale=1.0):
        """ACT_PRED'S AND THE CORRECTION'S STEP IN THE WAKING LESSON (after the waking lesson's own step): each later effector's group
        of opt_pred (`GatedDescent`), one plain step a lesson on the gradient the lesson's backward left on them, the own acts' and
        act_inv's labels' at their reliability (`_timing_loss`, body/core/cortex.py), each element's step bounded (R6 fix 7).
        THE LESSON'S OWN GRADIENT (R6 fix 8, the R6 verifier's eighth look): `scale` is the waking lesson's plasticity scale ((1 +
        stress / 10) x the wake gate), which its backward left on every parameter's gradient; it is divided out of act_pred's and the
        corrections' before their step, so the step is the plain step on the lesson's own gradient at any stress and gate (kept,
        stress stepped act_pred up to four times over and a hundred times the served rate diverged: `GatedDescent`). The day's
        parameters keep it, their Adam as before. A scale of 1 divides nothing (the step to the bit as before); a scale of 0 left
        nothing to divide (the lesson shut, no gradient anywhere). EACH
        OPTIMIZER'S OWN BOUND (the R6 verifier's fourth look, 2026-09-24): with one bound over every parameter, as before, act_pred's
        gradient, which grows with act_inv's reliability, set the size of every step of the stream (on the tiny arm the bound held at
        every lesson, the joint norm 9 to 13, act_pred's and the corrections' the larger part), so the labels' reliability still moved
        the stream's learning with their route to it closed: with act_pred held still, the proposal's error on a trained body moved by
        up to 0.5 between reliabilities; with the bounds apart, not at all (the stream the same to the bit at every reliability, anatomy
        31). The day's norm bound stays the day's; act_pred's bound is GatedDescent's, element by element and out of ordinary lessons'
        reach, never a norm over the group (R6 fix 3 to 6 bounded each group's norm per unit of the lesson's weight; a norm bound that
        binds sets the step's size whatever the gradient's, as Adam does)"""
        s_ = float(scale)
        if s_ != 1.0 and s_ != 0.0:
            with torch.no_grad():
                for g_ in self.opt_pred.param_groups:
                    for p_ in g_["params"]:
                        if p_.grad is not None:
                            p_.grad.div_(s_)
        self.opt_pred.step()
        self._forecast_scaling()                                           # A154: the forecast heads held at the act rows' scale

    def _forecast_scaling(self):
        """A154 (2026-10-01): THE FORECAST'S SCALE IS THE ACT ROWS'. act_pred's forecast P is read as sharpness x cos(P, row) x |P|^g (C148),
        and its lesson toward an act pulls P to the act's row, whose norm is the sum of its joints' unit rows (about 2.6 for an arm of
        seven): the scale the forecasts held for weeks. Nothing else should move that scale, yet two lessons did: the lesson against an
        act as a distance with its sign turned (the night of day 50: 3,600; A153 bounded it), and A153's own hinge of the cosine, whose
        gradient is orthogonal to P but whose finite steps lengthen it (|P + e v|^2 = |P|^2 + e^2 |v|^2), with no toward-lesson left under
        A152 to anchor it except at the acts that paid: the night of day 51 took the heads from 2 to 4 (hands), 6 (left arm), 7 (gaze)
        and 22 (right arm). After each step of act_pred's optimizer, by day (_timing_step) and in the night's reel (sleep.py), each later
        effector's head (pred, and its correction cor) is multiplied by (rows' mean norm / P's mean norm over the lesson's positions) **
        (1 / tag_reach): the scale drifts to the rows' at the tag's reach (64 lessons), both ways, the direction (the forecast's content)
        untouched. Homeostatic synaptic scaling (Turrigiano 2008), as A147/A151 hold the actors' drive; the set point is the rows' own,
        no new constant. The measured scales come from _timing_loss (`_fc_scale`, by effector), consumed here"""
        fs = getattr(self, "_fc_scale", None)
        if not fs:
            return
        from .physiology import FRAMES
        reach = float(self.cfg.get("tag_reach", FRAMES["tag_reach"]))
        with torch.no_grad():
            for name, (pn, rn) in fs.items():
                if pn > 1e-6 and rn > 1e-6 and name in self.m.timing:
                    f_ = float((rn / pn) ** (1.0 / reach))
                    tm = self.m.timing[name]
                    for mod in (tm.pred,) + ((tm.cor,) if tm.sense_n else ()):   # cor exists for an effector with a body sense alone
                        mod.weight.mul_(f_)
                        if mod.bias is not None:
                            mod.bias.mul_(f_)
        fs.clear()

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
        e = self.anatomy.motors[i - 1]; st = self.motor[i - 1]
        if e.sense is None:
            return
        s_now = self._body_sense(e)
        prev = st["now"]
        every_ = int(self._motor_const("act_inv_every")); chance_ = int(self._motor_const("act_inv_chance"))
        if e.inverse and st["sense"] is not None and prev is not None and prev["acted"]:
            if every_ <= 1 and not chance_:
                self._inverse_lesson(i, st["sense"], s_now, int(prev["act"]))       # R6's: a step a tick, its reliability Cohen's kappa
            else:
                # STEP R6h: the pair gathered for the batch, with the chance the act's own choice gave each setting (the kappa correction)
                st["inv_batch"].append((st["sense"], s_now, int(prev["act"]), self._act_chance(e, prev) if chance_ else None))
        if e.inverse and st.get("inv_batch") and self.ticks % max(1, every_) == 0:
            self._inverse_batch(i)                                             # step R6h: one step on the pairs gathered since the last
        st["err"] = (s_now - st["fwd"]) if st["fwd"] is not None else None
        st["sense"] = s_now

    def _inverse_lesson(self, i, s0, s1, act):
        """ACT_INV LEARNS ONLINE (step R6; SIM_DESIGN.md 5.4): the pair of senses (s0 before the act, s1 after it) and the act the body
        made, its efference copy, the label. Its reliability is updated first, on the label it gives now against the efference copy
        (a prediction, not a fit), then one step of its own optimizer on the cross-entropy of each joint's setting"""
        e = self.anatomy.motors[i - 1]; st = self.motor[i - 1]
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

    def _act_chance(self, e, now):
        """THE CHANCE THE ACT'S OWN CHOICE GAVE EACH SETTING (step R6h, the kappa correction; R6's verifiers, 6d6d246): for an act its gate
        drew (a fresh choice, each joint drawn from its probabilities), per joint the probability of each setting given that the draw
        was not the effector's rest (the act was an own act): q_j(s) = (p_j(s) - [s = r_j] P(rest)) / (1 - P(rest)), P(rest) the
        product of the joints' probabilities of the rest's settings r_j; for an act chosen without a draw (a movement unit's held act, a
        chunk's continuation: `cont`), None, its chance 1 where a label names its setting and 0 elsewhere. Known before the act, so a
        label made before it agrees beyond this chance only as far as it reads the motion"""
        if now.get("cont"):
            return None
        tab = self.m.get_submodule(e.organ)
        r_ = [int(x_) for x_ in tab.digits(torch.tensor(int(e.rest_id))).tolist()]
        ps = [p_.detach().float().cpu() for p_ in now["probs"]]
        pr = 1.0
        for p_, k_ in zip(ps, r_):
            pr *= float(p_[k_])
        if not 1.0 - pr > 1e-12:
            return [p_.clone() for p_ in ps]
        out = []
        for p_, k_ in zip(ps, r_):
            q_ = p_.clone(); q_[k_] = q_[k_] - pr
            out.append((q_ / (1.0 - pr)).clamp_(min=0.0))
        return out

    def _inverse_batch(self, i):
        """ACT_INV'S LESSONS BATCHED (step R6h; SIM_DESIGN.md 3.6: unbatched they cost 4-6 ms a tick at the humanoid's size): the pairs
        gathered since the last batch (act_inv_every ticks), each (the sense before the act, after it, the act made, the chance its
        choice gave each setting), labelled by act_inv as it stands, its reliability updated on each label in the order they came (a
        prediction, not a fit), then one step of its optimizer on the mean over the pairs of the joints' summed cross-entropy (a batch
        of one is R6's lesson to the float)"""
        e = self.anatomy.motors[i - 1]; st = self.motor[i - 1]
        pairs = st["inv_batch"]; st["inv_batch"] = []
        if not pairs:
            return
        tm = self.m.timing[e.name]; tab = self.m.get_submodule(e.organ)
        S0 = torch.stack([p_[0] for p_ in pairs]); S1 = torch.stack([p_[1] for p_ in pairs])
        trues = [[int(x_) for x_ in tab.digits(torch.tensor(int(p_[2]))).tolist()] for p_ in pairs]
        with torch.no_grad():
            labels = torch.stack([lg.argmax(-1) for lg in tm.inverse_logits(S0, S1)], dim=-1).tolist()
        for lab_, true_, p_ in zip(labels, trues, pairs):
            self._inv_rel_update(e, st, lab_, true_, chance=p_[3], held=p_[3] is None)
        tt = torch.tensor(trues, dtype=torch.long)
        with torch.enable_grad():
            self.opt_inv.zero_grad(set_to_none=True)
            lg = tm.inverse_logits(S0, S1)
            loss = None
            for j_, l_ in enumerate(lg):
                ce = F.cross_entropy(l_, tt[:, j_].to(l_.device))
                loss = ce if loss is None else loss + ce
            loss.backward()
            self.opt_inv.step()
            self.opt_inv.zero_grad(set_to_none=True)
        st["inv_n"] = int(st["inv_n"]) + len(pairs)
        st["inv_last"] = {"tick": self.ticks, "loss": round(float(loss.detach()), 4), "n": len(pairs),
                          "hit": [round(sum(int(a == b) for a, b in zip(lab_, true_)) / len(true_), 3) for lab_, true_ in zip(labels, trues)]}

    @staticmethod
    def _inv_conf_new(e):
        """act_inv's running confusion at birth: per joint a K x K table of zeros (the row the setting act_inv's label chose, the
        column the setting the efference copy made)"""
        return [[[0.0] * int(K) for _ in range(int(K))] for K in e.factors]

    def _inv_rel_update(self, e, st, label, true, chance=None, held=False):
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
        if int(self._motor_const("act_inv_chance")):
            # STEP R6h, THE KAPPA CORRECTION (6d6d246's proposal): per joint the running agreement and the running chance the act's own
            # choice gave the label's setting (`_act_chance`; an act chosen without a draw: 1 where the label names its setting, else 0),
            # over the same horizon; kappa_j = (p_o - mean chance) / (1 - mean chance); zero until 64 acts. It replaces the pooled kappa
            # above as the reliability (the confusion is kept, its pooled kappa reported as inv_kappa_pooled)
            if st.get("inv_ch") is None:
                st["inv_ch"] = [[0.0, 0.0, 0.0] for _ in e.factors]
            kc = []
            for j, row in enumerate(st["inv_ch"]):
                c_ = (1.0 if int(label[j]) == int(true[j]) else 0.0) if (held or chance is None) else float(chance[j][int(label[j])])
                row[0] = d * row[0] + 1.0; row[1] = d * row[1] + (1.0 if int(label[j]) == int(true[j]) else 0.0); row[2] = d * row[2] + c_
                pc = row[2] / row[0]
                kc.append((row[1] / row[0] - pc) / (1.0 - pc) if 1.0 - pc > 1e-9 else 0.0)
            on = st["inv_ch"][0][0] > 64
            st["inv_kappa_pooled"] = st["inv_kappa"]
            st["inv_kappa"] = [float(k) if on else 0.0 for k in kc]
            st["inv_gain"] = float(sum(max(0.0, min(1.0, k)) for k in kc) / len(kc)) if on else 0.0

    def _timing_propose(self, e, C):
        """ACT_PRED'S PROPOSAL (the base Effector's `propose`; step R6): the forecast of its own next act from the stream C, plus the
        forward half's correction of the error its body sense shows now (none before the forward half has foreseen a tick)"""
        st = self.motor[self.anatomy.motors.index(e)]
        tm = self.m.timing[e.name]
        p = tm.pred(C)
        if e.sense is not None and st["err"] is not None:
            p = p + tm.cor(st["err"])
        if self._recall_on():
            p = p + self._recall_term(e, self._frec_now)               # step R7f: the recalled act through its map (body/core/frames.py)
        if int(self.cfg.get("imagine_vte", 0)):
            p = p + self._vte_term(e)                                  # A182: the lean toward the better imagined future's first act
        return p

    def _timing_foresee(self, i):
        """THE FORWARD HALF AFTER THE ACT (step R6): the body sense at the next tick foreseen from the stream after this tick's own step
        (the act just taken is in it), kept for the next tick's error"""
        e = self.anatomy.motors[i - 1]; st = self.motor[i - 1]
        if e.sense is None:
            return
        C = getattr(self, "_C_last", None)
        with torch.no_grad():
            st["fwd"] = self.m.timing[e.name].fwd(C).detach().clone() if C is not None else None

    def _fwd_err_in(self, st):
        """THE FORWARD ERROR AS A GATE INPUT (step R6h; SIM_DESIGN.md 3.5, 3.6: each limb's forward error feeds its gate): the size of the
        error its body sense shows now against what its forward half foresaw at the tick before, the root mean square over its sense's
        numbers (in the sense's own units); 0 before the forward half has foreseen a tick"""
        err = st.get("err")
        if err is None:
            return 0.0
        return float(err.float().pow(2).mean().sqrt())

    def _unit_hold(self, e, st, logits, margin):
        """THE MOVEMENT UNIT (step R6h; SIM_DESIGN.md 3.6, 10: the persistence margin, unit_margin): a unit under way holds its act, joint
        by joint: a joint takes another setting only where the choice's logits prefer it to the held setting by more than the margin
        (then the most preferred); the flat act. Infant movement units (von Hofsten) and newborns' smooth general movements (Prechtl)
        are never white noise; at birth act_pred knows nothing and its best guess is a fresh random act every tick (3.6's evidence)"""
        held = st["unit"]; out = []
        for lg, h_ in zip(logits, held):
            b_ = int(lg.argmax())
            out.append(b_ if float(lg[b_]) - float(lg[int(h_)]) > float(margin) else int(h_))
        return self.m.get_submodule(e.organ).flat(out)

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

    def _timing_loss(self, i, C, obs, wpos=None, night=False, wlab=None):
        """THE WAKING LESSON OF LATER EFFECTOR i'S TIMING PART (step R6; body/core/cortex.py `_wake_lesson`), over the window's stream
        C [T, d] (with its gradient) and observations: act_pred's squared error to the target act's row at every position t >= 1, from
        the stream at t-1 and the forward half's error at t, weighted (1 at its own acts; at its rests act_inv's reliability on act_inv's
        label, none at the last position, whose next sense is not yet felt; 1 on its rest for an effector with no inverse model); and
        the forward half's squared error to the sense at t+1 from the stream at t. Returns (loss, report, labels): `loss` the own acts'
        errors (and the rests of an effector with no inverse model) and the forward half's, which the waking lesson's backward carries;
        `labels` act_inv's labels' errors AT FULL RELIABILITY (weight 1 at each labelled rest whose next sense is felt, over the same
        positions), None where the window has none, whatever the reliability. The whole lesson is loss + reliability x labels (the
        waking lesson adds them so, body/core/cortex.py, and act_pred steps plainly on its gradient, `GatedDescent`; kept apart since R6
        fix 5, whose instruments and tests read the labels' part at full reliability). The report's "w" is the lesson's mean label weight, "w_own" and
        "w_lab" its own acts' and its labels' parts, "s_lab" the labels' share of the positions (their weight at full reliability) and
        "rel" the reliability they were weighed at.
        THE WEIGHTS ARE ABSOLUTE (the R6 verifier's first finding, 2026-09-24): the weighted errors are averaged over the window's
        positions (T - 1), never over the weights' sum. Divided by the weights' sum, the reliability only reweighted the rests against
        the own acts and never scaled them: a window of rests alone, or of the parent's hand alone, taught at full strength at any
        reliability above 0 (the verifier's probe: the same loss and gradient at 0.01 as at 1). Over the positions, a label act_inv
        has not earned teaches only in proportion to what it has earned, and an own act teaches as it did in a window of own acts.
        The report's "w" is the lesson's mean label weight over the same positions (the loss's weights reach the gradient, and act_pred's
        plain step keeps their scale, `GatedDescent`; Adam undid it, R6 fix 1 to 6).
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
        its own ratio to the same labels given whole wandered (0.17 to 1.46), the stream's path then depending on the reliability.
        THE NIGHT'S WEIGHT (step R7d, the amygdala's side of R8; SIM_DESIGN.md 7.4 item 2): `wpos` [T], when given, weighs act_pred's error
        at each position of its own acts, and `wlab` [T] (A152; `wpos` when not given) at act_inv's labels: the night replays a window with act_pred's lesson at position t
        weighted clip(1 + G_t, 0, 1), G_t the replayed dopamine's credit (body/core/amygdala.py `act_pred_night_weight`), so acts followed
        by net harm are not taught as acts to make; the forward half's lesson is not weighted. None (the day's): as before, to the bit.
        `night` (step R8b, the night over frames: body/core/sleep.py): act_pred reads the stream detached at every position, so its
        weighted targets (the own acts' too) reach act_pred, the correction and the recall map alone, stepped plainly (GatedDescent), and
        never the stream, whose night Adam would teach whatever share of a weighted target reached it whole (the night's note since R6
        fix 7); the forward half still learns through the stream at weight 1, as by day. False (the day's): as before, to the bit"""
        e = self.anatomy.motors[i - 1]; st = self.motor[i - 1]
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
        if night:
            Cp = Cp.detach()                                               # step R8b: the night's weighted targets never reach the stream
        elif e.inverse and bool(rested[1:].any()):
            # ACT_INV'S LABELS TEACH THE PROPOSAL, NOT THE STREAM: at a position whose target is act_inv's label the stream is read
            # detached, so that lesson reaches act_pred and the correction alone (their plasticity gated by the label's reliability)
            Cp = torch.where(rested[1:].unsqueeze(-1), Cp.detach(), Cp)
        P = tm.pred(Cp)
        lf = None
        if s is not None:
            fw = tm.fwd(C[:-1])                                            # the sense at t+1 foreseen from the stream at t
            P = P + tm.cor(s[1:] - fw.detach())                          # the proposal at t+1 reads the error there
            lf = 0.5 * ((fw.float() - s[1:].float()) ** 2).sum(-1).mean()
        if "@frec" in obs:
            P = P + self._recall_term(e, obs["@frec"][1:])               # step R7f: the proposal at t+1 read the recall made at its choice
        with torch.no_grad():
            rows = tab(tgt[1:])
        w1 = wt[1:]
        err = 0.5 * ((P.float() - rows.float()) ** 2).sum(-1)
        # THE LABELS APART (since R6 fix 5): the own acts' errors (and an uninverted effector's rests) and act_inv's labels' at full
        # reliability (weight 1 at each labelled rest whose next sense is felt), each over the window's positions (the weights
        # absolute); the lesson is the own acts' plus the reliability times the labels'. A window with no labelled rest is the own
        # acts' alone, as before
        w_own, lab_f = w1, None
        if e.inverse and bool(rested[1:].any()):
            lab1 = rested[1:]
            lab_f = lab1.clone()
            if bool(rested[-1]):
                lab_f[-1] = False                                          # the last rest: its next sense not felt, no label
            w_own = torch.where(lab1, torch.zeros_like(w1), w1)
        if wpos is not None:                                              # step R7d: the weight on act_pred's error at its own acts (A152: dopamine's credit)
            w_ = wpos[1:].to(err.dtype)
            err_o = torch.where(w_ >= 0.0, w_ * err, (-w_) * _against(P, rows))   # A153: a negative weight teaches against by the cosine's hinge, not the distance
        else:
            err_o = err
        err_l = err * (wlab if wlab is not None else wpos.clamp_min(0.0))[1:].to(err.dtype) if (wlab is not None or wpos is not None) else err   # and at act_inv's labels (R8's)
        lp = (err_o * w_own).sum() / float(T - 1)                         # over the positions: the weights absolute
        n_lab = int(lab_f.sum()) if lab_f is not None else 0
        lb = (err_l * lab_f.float()).sum() / float(T - 1) if n_lab else None   # act_inv's labels, every one earned
        loss = lp if lf is None else lp + lf
        s_lab = n_lab / float(T - 1)
        with torch.no_grad():                                              # A154: the forecast's scale against the act rows' (the lesson's targets), for _forecast_scaling
            fs = getattr(self, "_fc_scale", None)
            if fs is None:
                fs = self._fc_scale = {}
            fs[e.name] = (float(P.detach().norm(dim=-1).mean()), float(rows.norm(dim=-1).mean()))
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
        e = self.anatomy.motors[i - 1]; st = self.motor[i - 1]
        out = {"chunks": int(st["chunks"]), "stops": dict(st["stops"]), "chunk": int(st["chunk"])}
        if e.inverse:
            out.update({"inv_gain": round(float(st["inv_gain"]), 3), "inv_kappa": [round(float(k_), 3) for k_ in st["inv_kappa"]],
                        "inv_n": int(st["inv_n"]), "inv_last": st["inv_last"]})
            if st.get("inv_kappa_pooled") is not None:                    # step R6h: the pooled kappa beside the corrected one
                out["inv_kappa_pooled"] = [round(float(k_), 3) for k_ in st["inv_kappa_pooled"]]
        if int(self._motor_const("own_fatigue")):
            out["fatigue"] = round(float(st.get("fatigue", 0.0)), 3)       # step R6h: its own fatigue
        if st.get("cord_n"):
            out["cord"] = dict(st["cord_n"])                               # step R6h: the cord's patterns' ticks (the reflex's log)
        return out
