"""the born patterns summed at the cord (a mixin of `Life`, body/life.py; the core refactor's step R6h, docs/SIM_DESIGN.md 3.6, 3.7, 10,
A35, A47, A48, C53, C54): the spinal pattern generator of each limb that declares one and the born cry of the tract, each added to its
effector's own act below the gate, as the palmar grasp is (A35: in the cord a reflex and the descending command meet at the same motor
neurons). The gate draws every tick and its acts keep their eligibility: the pattern's part is no act of the gate's (no efference copy,
no striatal event, no credit), and the cortex feels it only through the body's senses and its forward model's error. Its constants are
physiology.py's REFLEX, absent from a body's cfg unless given; the effectors declare where each acts (`Effector.spg`, `Effector.cry`),
so the language body, which declares none, runs none of it.

WHAT THE WORLD RECEIVES (body/core/world.py `Acts`): the tick's acts as before, and `acts.cord`, per effector with a pattern this tick,
one additive step per joint in the joint's own units (rad for the G1's joints, a fraction of the range for the tract's articulators),
which the world adds to the target its own act re-anchors (target = measured + own step + cord step), under the same clip. A joint
the pattern does not move gets 0.

THE SPINAL PATTERN GENERATOR (`_spg_step`; A48): a half-centre oscillator per limb (Brown 1911), its phase advancing each tick,
phi = frac(phase0 + ticks / spg_period): in its flexion half (phi < 1/2) a step of +A along each of the limb's declared flexion joints in
its flexion sense (the withdrawal's joints: a leg's hip pitch, knee and ankle pitch; an arm's shoulder pitch and elbow), in its
extension half the opposite, A = the limb's gate's p_act this tick x spg_amp (the gate's tonic readiness drives it). An own act
against the step on a joint cancels it there (its setting's step of the opposite sign); an own act with it, or holding, sums with it.
The legs' born phases are half a cycle apart (newborns' kicks alternate in 74% of kicks: Sylos-Labini et al. 2020; Thelen 1979); a
limb that declares no phase takes one drawn at birth from the body's seed (the organs' `spg_phase`: the arms, uncoupled from the legs
and from each other, as the neonate model's oscillators were coupled only through the body: Kuniyoshi and Sangawa 2006; C54 open). No
posture, balance or gravity term: it reads the tick, its phase, the gate's p_act and the own act, never a sense.

THE BORN CRY (`_cry_step`; A47; Jurgens 2002): while the body felt pain this tick (the named reward source's term below 0) or its charge
is below cry_charge, the tract's declared cry posture is added to its targets in breath groups: cry_expire ticks of the posture (the
lungs pushing, the glottis pressed, the pitch raised, the jaw open), then cry_inspire ticks of the lungs drawn back (breathing in: the
reservoir refills at rest), the expiration ending early when the reservoir is empty (breath left at 0, the tract's own physics). The
tract's own act overrides it articulator by articulator: where the own act steps an articulator (any setting but the hold) the cry's
step there is dropped, so the cortex can hush it. Its ticks are logged as reflex (st["cord_n"]["cry"], the tick's record); a cry is
never a vocal turn (the parent's side reads acts.cord).

THE BORN BIASES (step R6h; SIM_DESIGN.md 3.7, A23, A43): ORIENTING (`_orient_bias`): each tick the anatomy's born cues
(`Anatomy.orienting`: the face template in the periphery, a sound's side at an onset, a sudden local change) are read from the frame
once (`_orient_cues`); on each joint an effector declares for it (the gaze's yaw and pitch, the waist's yaw), every cue that fires with
a direction on that joint's axis (outside the fovea's zone, or a side) adds orient_bias x the orienting gain to each setting stepping
toward it and takes as much from each stepping away, the hold untouched: a bias on the choice's logits, never a forced move, so a
learned proposal can outweigh it and it fades by learning; the gain is the amygdala's, exactly 1 at birth (`_orient_gain`; R7e). The
born gate input (`_orient_in`): 1 on a tick a cue appeared (a sound's or a sudden change's onset; the face's fire after a tick without
it). THE VOR (`_vor_acts`): the body's born reflex on the gaze's window, handed to the world with the tick's acts (Acts.vor: the axes,
the born gain, the quick phase's jump) for it to apply at its samples of the gyro."""
import torch

from .physiology import REFLEX


class CordMixin:
    def _reflex_const(self, k):
        """a born pattern's or bias's constant: the body's cfg when it was given, else REFLEX's"""
        return self.cfg.get(k, REFLEX[k])

    @staticmethod
    def _setting_sign(k, K):
        """the sense of setting k of a joint of K settings: -1 below the middle, 0 the middle (the hold), +1 above"""
        c = (int(K) - 1) / 2.0
        return 0 if int(k) == c else (1 if int(k) > c else -1)

    def _cord(self, i, frame, p_act, dig, reflex):
        """THE CORD'S PATTERNS FOR MOTOR EFFECTOR i THIS TICK: the per-joint additive steps its spinal pattern generator and its born cry
        add below the gate, after the own act's cancellation, or None when neither adds anything. A tick its reflex (the withdrawal)
        took has none: the reflex takes the limb"""
        e = self.anatomy.motors[i - 1]; st = self.motor[i - 1]
        if reflex:
            return None
        out = None
        if e.spg and int(self._reflex_const("spg")):
            s_ = self._spg_step(i, e, st, p_act, dig)
            if s_ is not None:
                out = s_; st["cord_n"]["spg"] = int(st["cord_n"].get("spg", 0)) + 1
        if e.cry and int(self._reflex_const("cry")):
            c_ = self._cry_step(e, st, frame, dig)
            if c_ is not None:
                out = c_ if out is None else [a_ + b_ for a_, b_ in zip(out, c_)]
                st["cord_n"]["cry"] = int(st["cord_n"].get("cry", 0)) + 1
        return out

    def _spg_phase0(self, i, e):
        """the limb's born phase: its declaration's, else the one drawn at birth from the body's seed (the organs' spg_phase)"""
        if e.spg_phase is not None:
            return float(e.spg_phase)
        return float(self.m.spg_phase[i - 1])

    def _spg_step(self, i, e, st, p_act, dig):
        """the half-centre oscillator's step this tick (the module's doc): +-A along each declared flexion joint, cancelled where the own
        act steps that joint against it; None when every step is 0"""
        per = float(self._reflex_const("spg_period"))
        phi = (self._spg_phase0(i, e) + float(self.ticks) / per) % 1.0
        half = 1.0 if phi < 0.5 else -1.0                              # the flexion half, then the extension half
        A = float(p_act) * float(self._reflex_const("spg_amp"))
        out = [0.0] * len(e.factors); moved = False
        for j, sg in e.spg.items():
            step = half * float(sg) * A
            own = self._setting_sign(dig[int(j)], e.factors[int(j)])
            if own != 0 and (own > 0) != (step > 0):
                continue                                               # an own act against the step cancels it
            out[int(j)] = step; moved = moved or step != 0.0
        return out if moved else None

    def _cry_step(self, e, st, frame, dig):
        """the born cry's step this tick (the module's doc), or None when it does not cry: its breath groups kept in st["cry_t"] (the
        ticks since the cry began, 0 when it is quiet)"""
        cy = e.cry
        terms = getattr(self, "_terms_now", None) or {}
        pain = float(terms.get(cy["pain"], 0.0)) < 0.0 if cy.get("pain") else False
        ch_, ci_ = cy["charge"]
        o_ = frame.obs.get(ch_)
        low = o_ is not None and float(o_[int(ci_)]) < float(self._reflex_const("cry_charge"))
        if not (pain or low):
            st["cry_t"] = 0
            return None
        E, I = int(self._reflex_const("cry_expire")), int(self._reflex_const("cry_inspire"))
        k = int(st.get("cry_t", 0)) % (E + I)
        bc_, bi_ = cy["breath"]
        b_ = frame.obs.get(bc_)
        if k < E and b_ is not None and float(b_[int(bi_)]) <= 0.0:
            k = E; st["cry_t"] = (int(st.get("cry_t", 0)) // (E + I)) * (E + I) + E    # the reservoir empty: breathe in now
        lungs = int(cy["lungs"])
        out = [0.0] * len(e.factors)
        if k < E:
            for j, v in cy["posture"].items():
                out[int(j)] = float(v)                                 # the cry's posture: lungs pushing, glottis pressed, pitch up, jaw open
        else:
            out[lungs] = -abs(float(cy["posture"][lungs]))             # breathing in: the lungs drawn back
        st["cry_t"] = int(st.get("cry_t", 0)) + 1
        for j in range(len(out)):                                      # its own act overrides it, articulator by articulator
            if self._setting_sign(dig[j], e.factors[j]) != 0:
                out[j] = 0.0
        return out if any(v != 0.0 for v in out) else None             # wholly overridden: hushed this tick

    # ---------------- the born biases: orienting and the VOR ----------------
    def _orient_gain(self):
        """the orienting gain: the amygdala's clip(1 + N, -0.5, 2) once it is built (SIM_DESIGN.md 7.4, R7e); exactly 1 at birth and until
        then"""
        return 1.0

    def _orient_cues(self, frame):
        """the tick's born orienting cues, read once a tick from its frame (cached for the tick): [(cue, direction on yaw, direction on
        pitch, appeared)], each direction +1 (right / up), -1 or 0 (no such axis, not firing, or inside the fovea's zone)"""
        now_ = getattr(self, "_orient_now", None)
        if now_ is not None and now_[0] == self.ticks:
            return now_[1]
        last = getattr(self, "_orient_last", None)
        if last is None:
            last = {}; self._orient_last = last
        out = []
        for c in (self.anatomy.orienting or ()):
            o_ = frame.obs.get(c.obs)
            fired = o_ is not None and float(o_[int(c.fired)]) > 0.0
            dirs = []
            for ax in (c.yaw, c.pitch):
                if not fired or ax is None:
                    dirs.append(0); continue
                v = float(c.sense) * float(o_[int(ax)])
                dirs.append(0 if (v == 0.0 or (not c.side_only and abs(v) <= float(c.zone))) else (1 if v > 0 else -1))
            appeared = fired and (bool(c.onset) or not last.get(c.name, False))
            last[c.name] = fired
            out.append((c, dirs[0], dirs[1], appeared))
        self._orient_now = (self.ticks, out)
        return out

    def _orient_in(self, frame):
        """THE BORN GATE INPUT (3.7): 1 on a tick a face, a sound's onset or a sudden change appeared, else 0 (0 while orienting is off)"""
        if not int(self._reflex_const("orient")):
            return 0.0
        return 1.0 if any(a_ for *_, a_ in self._orient_cues(frame)) else 0.0

    def _orient_bias(self, e, frame, tab):
        """the born orienting bias on effector e's joints this tick: one piece of logits per joint (zeros where it declares none), or None
        when nothing pulls"""
        cues = self._orient_cues(frame)
        if not any(dy or dp for _, dy, dp, _ in cues):
            return None
        g = float(self._orient_gain()) * float(self._reflex_const("orient_bias"))
        out = [torch.zeros(int(K), device=self.dev) for K in e.factors]; pulled = False
        for j, (ax, sg) in e.orient.items():
            d = sum((dy if ax == "yaw" else dp) for _, dy, dp, _ in cues)
            if d == 0:
                continue
            K = int(e.factors[int(j)])
            out[int(j)] = torch.tensor([g * float(d) * float(sg) * self._setting_sign(k, K) for k in range(K)], device=self.dev)
            pulled = True
        return out if pulled else None

    def _vor_acts(self):
        """the VOR's born constants for the world this tick (Acts.vor), per effector that declares it: its axes, the born gain, the quick
        phase's jump (a fraction of the reach)"""
        if not int(self._reflex_const("vor")):
            return {}
        return {e.name: {"axes": [int(j) for j in e.vor], "gain": float(self._reflex_const("vor_gain")),
                         "quick": float(self._reflex_const("vor_quick"))} for e in self.anatomy.motors if e.vor}
