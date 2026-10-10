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

THE SPINAL PATTERN GENERATOR (`_spg_step`; A48; C54 closed on the measured newborn rhythm): a half-centre generator per limb (Brown
1911) whose cycle is A SHORT MOVEMENT FOLLOWED BY A PAUSE: flexion for spg_flex ticks (2: 0.30 s; Thelen and Fisher 1982, 1983),
then extension for spg_ext ticks (3: 0.45 s; Thelen and Fisher 1982), then the pause until the cycle's end. In the flexion a step of
+A along each of the limb's declared flexion joints in its flexion sense (the withdrawal's joints: a leg's hip pitch, knee and ankle
pitch; an arm's shoulder pitch and elbow), in the extension a step of -A x spg_flex / spg_ext (2A/3), in the pause none; A = the limb's
gate's p_act this tick x spg_amp (the gate's tonic readiness drives it). So a movement moves the targets 2A toward flexion and 2A back,
and a cycle returns where it began: THE LEAD'S DECISION (2026-09-25). A kick is a flexion and a return; the sources give the phases'
durations, not their excursions. Until then the extension stepped A as the flexion did (R6h, C54), and each cycle drifted the targets
A toward extension. The durations are the sources' and unchanged; only the extension's step is scaled. EACH CYCLE'S LENGTH IS
DRAWN FROM THE BODY'S SEED (`_spg_cycle`): a log-normal of mean spg_cycle and SD spg_cycle_sd (3.56 and 1.93 s: newborns' kicking at
birth, Hinnekens et al. 2023; physiology.py REFLEX gives every source), clipped to spg_cycle_min..spg_cycle_max (1.0-8.5 s); the
movement's 5 ticks are fixed and only the pause stretches (the pause is what varies: Thelen 1981). A cycle's length is real (in
ticks), its movement beginning at the first tick at or after the cycle's start, so the long run's cycles average the law's mean
exactly, and a limb's movements begin at least 6 ticks apart (the shortest cycle is 6.67), each ending before the next. THE RHYTHM (`_spg_rhythm`, `_spg_where`): a limb keeps its own rhythm unless it names one
(`Effector.spg_rhythm`) that limbs share: the first of them in the declared order leads it, the rhythm's cycles beginning at its
movements, and each other moves its own phase's lag behind the leader's within every cycle, so the G1's legs (born at phases 0 and
0.5, both naming "legs") share their drawn cycles and the right leg moves half of each cycle after the left (newborns' kicks alternate
in 74% of kicks: Sylos-Labini et al. 2020; Thelen 1979); since the right leg's movement sits half-way through each cycle, its own
movement-to-movement intervals are half of one cycle and half of the next (the same mean, a smaller spread). A limb that declares no
phase takes one drawn at birth from the body's seed (the organs' `spg_phase`: the fraction of its first cycle lived at birth) and keeps
a rhythm of its own (the arms: uncoupled from the legs and from each other, as the neonate model's oscillators were coupled only
through the body: Kuniyoshi and Sangawa 2006). Cycle n of the rhythm led by motor effector r is drawn from a generator of its own
seeded spg_seed + (the motor effectors' count) x n + r (the organs' buffer, drawn at birth from the body's seed), so the rhythm is a
function of the tick alone: it draws nothing from the life's streams, it goes on through the night, and after a load it is found again
from birth, where the save left it. The step SUMS with the own act on every joint, with it or against it (A97, 2026-09-26; A48's law: the pattern
and the descending command meet at the same motoneurons and add); until A97 an own act against the step cancelled it there, which let a
constant act cancel every return and ratchet the joint to its limit (life 1's second day). No posture, balance or gravity term: it reads the tick, its rhythm, the gate's p_act and
the own act, never a sense.

THE BORN CRY (`_cry_step`; A47; Jurgens 2002): while the body felt pain this tick (the named reward source's term below 0, paid or not:
a cortisol source's term is kept in the tick's terms, A88) or, on a body with a charge, its charge is below cry_charge, the tract's
declared cry posture is added to its targets in breath groups: cry_expire ticks of the posture (the lungs pushing, the glottis pressed,
the pitch raised, the jaw open), then cry_inspire ticks of the lungs drawn back (breathing in: the
reservoir refills at rest), the expiration ending early when the reservoir is empty (breath left at 0, the tract's own physics). The
tract's own act overrides it articulator by articulator: where the own act steps an articulator (any setting but the hold) the cry's
step there is dropped, so the cortex can hush it. Its ticks are logged as reflex (st["cord_n"]["cry"], the tick's record); a cry is
never a vocal turn (the parent's side reads acts.cord). Its breath clock (st["cry_t"]) is saved with the body's day (A70; body/core/
persistence.py), so a life saved mid-cry goes on in the same breath group.

THE BORN BIASES (step R6h; SIM_DESIGN.md 3.7, A23, A43): ORIENTING (`_orient_bias`): each tick the anatomy's born cues
(`Anatomy.orienting`: the face template in the periphery, a sound's side at an onset, a sudden local change) are read from the frame
once (`_orient_cues`); on each joint an effector declares for it (the gaze's yaw and pitch, the waist's yaw), every cue that fires with
a direction on that joint's axis (outside the fovea's zone, or a side) adds orient_bias x the orienting gain to each setting stepping
toward it and takes as much from each stepping away, the hold untouched: a bias on the choice's logits, never a forced move, so a
learned proposal can outweigh it and it fades by learning; the gain is the amygdala's, clip(1 + N, -0.5, 2) on all three cues alike,
exactly 1 at birth (`_orient_gain`; step R7e, body/core/amygdala.py). The
born gate input (`_orient_in`): 1 on a tick a cue appeared (a sound's or a sudden change's onset; the face's fire after a tick without
it). The memory of each cue's last fire (`_orient_last`) runs on across the night and is saved with the body's day (A70), so a face held
across a save does not appear again at the load. THE VOR (`_vor_acts`): the body's born reflex on the gaze's window, handed to the world
with the tick's acts (Acts.vor: the axes, the born gain, the quick phase's jump) for it to apply at its samples of the gyro; the body
keeps none of its state (the quick phase's is the world's)."""
import math

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
        took has none: the reflex takes the limb, and the generator's rhythm runs on under it (its place this tick is kept for the
        instruments, st["now"]["spg"]: +1 flexion, -1 extension, 0 the pause)"""
        e = self.anatomy.motors[i - 1]; st = self.motor[i - 1]
        spg_on = bool(e.spg) and bool(int(self._reflex_const("spg")))
        where = self._spg_where(i, e, st) if spg_on else None
        if spg_on and st.get("now") is not None:
            st["now"]["spg"] = where
        if reflex:
            return None
        out = None
        if spg_on:
            s_ = self._spg_step(e, where, p_act, dig)
            if s_ is not None:
                out = s_; st["cord_n"]["spg"] = int(st["cord_n"].get("spg", 0)) + 1
        crying = False
        if e.cry and int(self._reflex_const("cry")):
            c_ = self._cry_step(e, st, frame, dig)
            if c_ is not None:
                out = c_ if out is None else [a_ + b_ for a_, b_ in zip(out, c_)]
                st["cord_n"]["cry"] = int(st["cord_n"].get("cry", 0)) + 1
                crying = True
        if e.cry and not crying and int(self._reflex_const("breath")):   # A173: the born breath runs while the tract does not cry
            b_ = self._breath_step(e, st, frame)
            if b_ is not None:
                out = b_ if out is None else [a_ + c2 for a_, c2 in zip(out, b_)]
                st["cord_n"]["breath"] = int(st["cord_n"].get("breath", 0)) + 1
        if e.orient and e.vor and int(self._reflex_const("orient")) and int(self._reflex_const("orient_saccade")):
            sc_ = self._orient_saccade(e, frame)                           # A186: the born saccade toward a cue, on the eyes
            if sc_ is not None:
                out = sc_ if out is None else [a_ + b_ for a_, b_ in zip(out, sc_)]
                st["cord_n"]["saccade"] = int(st["cord_n"].get("saccade", 0)) + 1
        if e.approach and int(self._reflex_const("orient")) and int(self._reflex_const("approach")):
            ap_ = self._approach_step(e, frame)                            # A216: the born approach toward a face, on the locomotor command
            if ap_ is not None:
                out = ap_ if out is None else [a_ + b_ for a_, b_ in zip(out, ap_)]
                st["cord_n"]["approach"] = int(st["cord_n"].get("approach", 0)) + 1
        if st.get("now") is not None:
            st["now"]["cry"] = bool(crying)                                 # (the world's crying flag: the cry's step, never the breath's)
        return out

    def _breath_step(self, e, st, frame):
        """A173 (2026-10-02): THE BORN BREATH. The brainstem's respiratory rhythm (the pre-Botzinger complex: Smith et al. 1991) drives the
        tract's lungs in a tidal cycle below the gate while the tract does not cry: breath_expire ticks pushing at breath_amp of the lungs'
        range, then breath_inspire ticks drawn back (the reservoir empty on an expiration: breathe in now, as the cry does). Its clock
        st["breath_t"] is saved with the body's day. Why: on life day 64's copy the tract had sounded on none of 300 ticks (its pressure
        0 Pa), the voice's inverse model read chance agreement on every articulator (kappa 0.00 to 0.06) for want of any sound to label,
        and the voice actor sat at one setting on 78 to 91% of its draws: the lungs rested at zero drive, so no act of the glottis could
        phonate. An infant breathes always, and coos on an expiration when the glottis closes (the vocal play of 6 to 8 weeks); the
        breath is the brainstem's, the glottis the child's. Nothing of the glottis or the other articulators is touched here"""
        cy = e.cry
        E, I = int(self._reflex_const("breath_expire")), int(self._reflex_const("breath_inspire"))
        amp = float(self._reflex_const("breath_amp"))
        k = int(st.get("breath_t", 0)) % (E + I)
        bc_, bi_ = cy["breath"]
        b_ = frame.obs.get(bc_)
        if k < E and b_ is not None and float(b_[int(bi_)]) <= 0.0:
            k = E; st["breath_t"] = (int(st.get("breath_t", 0)) // (E + I)) * (E + I) + E   # the reservoir empty: breathe in now
        lungs = int(cy["lungs"])
        out = [0.0] * len(e.factors)
        out[lungs] = amp if k < E else -amp
        brake = float(self._reflex_const("breath_brake"))
        if brake > 0.0 and cy.get("glottis") is not None:
            # A179 (2026-10-02): THE NEWBORN'S EXPIRATORY BRAKING. The glottis narrowed through the expiration by brake of its range a tick and
            # opened again on the inspiration (the laryngeal adductors' expiratory activity that holds the newborn's lung volume, Kosch and Stark
            # 1984; Harding 1984): the breath sounds as a newborn's does (its grunt and sigh), so the ears hear the articulators at work on most
            # expirations. Why: on the day-66 copy the voice's inverse model had had 272,759 pairs and sat at chance on nine articulators of ten
            # (kappa -0.01 to 0.04), 95% of its acts sounding nothing (74 phonated ticks of 1,500 under A173's silent breath). The own act's
            # glottis step replaces this one as it replaces the cry's (the glottis the child's where it acts)
            out[int(cy["glottis"])] = brake if k < E else -brake
        st["breath_t"] = int(st.get("breath_t", 0)) + 1
        return out

    def _spg_phase0(self, i, e):
        """the limb's born phase (the fraction of its cycle lived at birth, a cycle beginning with its movement): its declaration's, else
        the one drawn at birth from the body's seed (the organs' spg_phase)"""
        if e.spg_phase is not None:
            return float(e.spg_phase)
        return float(self.m.spg_phase[i - 1])

    def _spg_rhythm(self, i, e):
        """THE RHYTHM LIMB i KEEPS (the module's doc): (r, the leader's born phase, this limb's lag within each cycle), r the place among
        the motor effectors of the limb that leads it (the first in the declared order that names the rhythm; the limb itself when it
        names none), the lag (the leader's phase less its own, a fraction of the cycle: 0 for the leader, 0.5 for the G1's right leg)"""
        motors = self.anatomy.motors
        r = i - 1
        if e.spg_rhythm is not None:
            r = next(k for k, x in enumerate(motors) if x.spg and x.spg_rhythm == e.spg_rhythm)
        lead = self._spg_phase0(r + 1, motors[r])
        return r, lead, (lead - self._spg_phase0(i, e)) % 1.0

    def _spg_cycle(self, r, n):
        """CYCLE n OF THE RHYTHM LED BY MOTOR EFFECTOR r (0 is the one under way at birth): its length in ticks, drawn from a generator
        of its own seeded spg_seed + (the motor effectors' count) x n + r (a seed per cycle and rhythm; the body's seed through the organs'
        spg_seed), one standard normal z: exp(mu + sigma z), the log-normal whose mean and SD are spg_cycle and spg_cycle_sd (sigma^2 =
        ln(1 + (sd / mean)^2), mu = ln(mean) - sigma^2 / 2), held to spg_cycle_min..spg_cycle_max"""
        mean, sd = float(self._reflex_const("spg_cycle")), float(self._reflex_const("spg_cycle_sd"))
        key = int(self.m.spg_seed) + len(self.anatomy.motors) * int(n) + int(r)
        z = float(torch.randn((), generator=torch.Generator().manual_seed(key), dtype=torch.float64))
        s2 = math.log1p((sd / mean) ** 2)
        x = math.exp(math.log(mean) - 0.5 * s2 + math.sqrt(s2) * z)
        return min(max(x, float(self._reflex_const("spg_cycle_min"))), float(self._reflex_const("spg_cycle_max")))

    def _spg_where(self, i, e, st):
        """WHERE LIMB i'S GENERATOR STANDS THIS TICK: +1 in its movement's flexion, -1 in its extension, 0 in the pause. Its rhythm's
        cycle n runs from s_n (s_0 = -the leader's born phase x its length, s_n+1 = s_n + its length) and the limb's movement in it
        begins at the first tick at or after s_n + its lag x the cycle's length; the limb stands in the movement that began last. Kept
        in st["spg_cyc"] (the cycle under way and the next), a function of the tick: found from birth when none is kept (a new body, a
        load), then advanced one cycle at a time"""
        t = int(self.ticks)
        cy = st.get("spg_cyc")
        if cy is None or int(cy["t"]) > t:
            r, lead, lag = self._spg_rhythm(i, e)
            L0 = self._spg_cycle(r, 0); s0 = -lead * L0
            cy = {"r": r, "lag": lag, "n": 0, "s": s0, "L": L0, "m": math.ceil(s0 + lag * L0), "next": None, "t": t}
            st["spg_cyc"] = cy
        while True:
            if cy["next"] is None:
                n1 = int(cy["n"]) + 1; s1 = float(cy["s"]) + float(cy["L"]); L1 = self._spg_cycle(int(cy["r"]), n1)
                cy["next"] = (n1, s1, L1, math.ceil(s1 + float(cy["lag"]) * L1))
            n1, s1, L1, m1 = cy["next"]
            if m1 > t:
                break
            cy.update(n=n1, s=s1, L=L1, m=m1, next=None)              # the next movement has begun: its cycle is the one under way
        cy["t"] = t
        d = t - int(cy["m"])
        F, E = int(self._reflex_const("spg_flex")), int(self._reflex_const("spg_ext"))
        return 1 if 0 <= d < F else (-1 if F <= d < F + E else 0)

    def _spg_step(self, e, where, p_act, dig):
        """the generator's step this tick (the module's doc): along each declared flexion joint +A in the movement's flexion (+1),
        -A x spg_flex / spg_ext in its extension (-1: the extension returns what the flexion moved, the lead's decision), summed with
        the own act whatever its sense (A97); None in the pause or when every step is 0"""
        if not where:
            return None
        A = float(p_act) * float(self._reflex_const("spg_amp"))
        if where < 0:
            A = A * float(self._reflex_const("spg_flex")) / float(self._reflex_const("spg_ext"))   # 2A/3: the cycle's net excursion 0
        out = [0.0] * len(e.factors); moved = False
        for j, sg in e.spg.items():
            step = float(where) * float(sg) * A                        # A97: summed with the own act whatever its sense (A48's law: the
            out[int(j)] = step; moved = moved or step != 0.0           # pattern and the descending command meet at the motoneurons and
        return out if moved else None                                  # add; until A97 an own act against the step cancelled it, so a
                                                                       # constant act cancelled every return and ratcheted the joint to
                                                                       # its limit: life 1's second day)

    def _cry_step(self, e, st, frame, dig):
        """the born cry's step this tick (the module's doc), or None when it does not cry: its breath groups kept in st["cry_t"] (the
        ticks since the cry began, 0 when it is quiet)"""
        cy = e.cry
        terms = getattr(self, "_terms_now", None) or {}
        pain = float(terms.get(cy["pain"], 0.0)) < 0.0 if cy.get("pain") else False
        low = False
        if cy.get("charge") is not None:                                 # a body with a charge (none since A88: the G1 has no charge)
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
        """THE ORIENTING GAIN (step R7e; SIM_DESIGN.md 7.4 item 3, A16): the amygdala's clip(1 + N, amyg_orient_lo, amyg_orient_hi), N the
        net valence of the tick's reliable forecasts (body/core/amygdala.py), on all three born cues alike (the face, the sound's side, the
        sudden change: `_orient_bias` scales their summed pull); a context that predicts one reward unit of good doubles the born pull, one
        that predicts 1.5 units of bad turns it into a weak turn away (at most half the born pull: the owner's "toward or away"). Exactly
        1 at birth (every reliability 0, so N is 0) and while the amygdala is off"""
        now = getattr(self, "_amyg_now", None) if self._amyg_on() else None
        if now is None:
            return 1.0
        return max(float(self._amyg_const("amyg_orient_lo")), min(float(self._amyg_const("amyg_orient_hi")), 1.0 + float(now["N"])))

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

    def _orient_saccade(self, e, frame):
        """A186 (2026-10-03): THE BORN SACCADE TO A FACE. The superior colliculus turns the eyes to a target in the periphery through the
        brainstem's saccade generator, where its command and the cortex's meet and add (the colliculus's map of the retina gives the
        turn its direction and size: Wurtz and Albano 1980), as a limb's reflex and the descending command meet at its motor neurons
        (A35, A48); and a newborn's eyes follow a face-like pattern by a subcortical road before the cortex guides them (Johnson,
        Dziurawiec, Ellis and Morton 1991; Morton and Johnson 1991's CONSPEC). For an effector that orients AND carries the VOR (the
        eyes), on each joint it declares for orienting: the offset from the fovea's centre of the first STANDING cue that fires
        outside the fovea's zone (a cue that is not itself an onset: the G1's born face cue), times orient_saccade_gain (0.5: a
        newborn's saccades fall short, a target reached in steps of about half the distance: Aslin and Salapatek 1975), at most
        orient_saccade_max a tick, times the orienting gain (the amygdala's toward or away, 1 at birth). A cord step: summed with the
        eyes' own act below the gate (an own step against it cancels it: the cortex can hold its gaze), no efference copy, no
        credit; the born logit bias (`_orient_bias`) stands beside it as it was, for every cue. -> one step per joint, or None.
        THE ONSET CUES (a sudden change, a sound's side) KEEP THE BIAS ALONE: tried with them on the dawn-76 copy (a new onset
        leading), the saccade fired on 87% of ticks, the eye chasing each flicker of its own moving hands (now in view, A185), and
        her face fell from 19% of ticks in the fovea to 2%; a transient everywhere is no target.
        WHY: life days 75 and 76, read from the record: with her face's cue off-centre the gaze's next step went toward it on 55
        to 63% of ticks (yaw; 46 to 52% pitch) and away on 36 to 42%, the cue's distance from the fovea 17.7 degrees before and
        17.6 after; her face in its fovea on 11% of ticks with the cue on 33%. A bias of log 4 on the choice's logits is drowned by
        76 days of habit and the movement unit's hold; her face is where her smile, her mouth's words and her eyes' direction are"""
        cues = [c_ for c_ in self._orient_cues(frame) if (c_[1] or c_[2]) and not c_[0].onset]
        if not cues:
            return None
        c, dy, dp, _ = cues[0]
        o_ = frame.obs.get(c.obs)
        g = float(self._orient_gain()); k = float(self._reflex_const("orient_saccade_gain")); mx = float(self._reflex_const("orient_saccade_max"))
        out = [0.0] * len(e.factors); moved = False
        for j, (ax, sg) in e.orient.items():
            d = dy if ax == "yaw" else dp
            idx = c.yaw if ax == "yaw" else c.pitch
            if not d or idx is None:
                continue
            v = float(d) * k * mx if c.side_only else max(-mx, min(mx, k * float(c.sense) * float(o_[int(idx)])))
            out[int(j)] = g * float(sg) * v; moved = True
        return out if moved else None

    def _approach_step(self, e, frame):
        """A216 (2026-10-10): THE BORN APPROACH. The superior colliculus's crossed descending road turns the head and body toward a target
        and drives approach (its uncrossed road, defence: Dean, Redgrave and Westby 1989), reading the target in the eyes' coordinates
        together with where the eyes point in the head (Freedman and Sparks 1997: gaze shifts share the eyes' and the head's part); a
        newborn turns toward its mother's face and voice (Goren, Sarty and Wu 1975; Muir and Field 1979), and the infant's approach to
        the caregiver is the attachment system's set goal (Bowlby 1969). For an effector that declares `approach` (the G1's locomotor
        command, W8), the first of its named cues that fires, in their declared order (her face; then her voice's side at an onset,
        the newborn's head turn to its mother's voice: a turn alone, no step, the ears' angle being a side and not a place), gives its
        bearing from the body: for the face the eyes' yaw in the head (approach["eye"]: the frame's observation and index) plus the
        cue's offset from the fovea (rad, + right), for the voice the ears' born lateral read; the turn joint's
        step is approach_turn_gain x the bearing, signed by the joint's declared sense (`orient`), less approach_damp x the body's own
        yaw rate (approach["gyro"]: the trunk's gyro, + left; the vestibular damping of a turn), at most approach_turn_max a tick; the
        speed joint's step is approach_go a tick while the bearing lies within approach_zone of straight ahead (it walks when it faces
        her); both times the orienting gain (the amygdala's toward or away: a face that forecasts bad turns it away). A cord step: summed
        with the brain's own command below the gate, no efference copy, no credit; the brain's own acts on the command earn their credit
        from her face as every act does, and can hold against it. -> one step per joint, or None"""
        ap = e.approach
        names = tuple(ap.get("cues", ("face",)))
        cues_ = {c_[0].name: c_[0] for c_ in self._orient_cues(frame)}
        for name in names:                                                  # in the declared order: the face before the voice
            c = cues_.get(name)
            if c is None or c.yaw is None:
                continue
            o_ = frame.obs.get(c.obs)
            if o_ is None or float(o_[int(c.fired)]) <= 0.0:
                continue
            heard = bool(c.onset or c.side_only)                            # a sound's side at its onset (the ears' angle in the head: no eye
            bearing = float(c.sense) * float(o_[int(c.yaw)])                # term; a turn toward the voice, never a step: its angle is a
            eye = ap.get("eye")                                             # side, not a place, and a voice behind reads as straight ahead)
            if eye is not None and not heard:
                ob_ = frame.obs.get(eye[0])
                if ob_ is not None:
                    bearing += float(ob_[int(eye[1])])
            g = float(self._orient_gain())
            out = [0.0] * len(e.factors); moved = False
            jt = ap.get("turn")
            if jt is not None:
                sg = float((e.orient or {}).get(int(jt), ("yaw", 1))[1])
                k = float(self._reflex_const("approach_turn_gain")); mx = float(self._reflex_const("approach_turn_max"))
                rate = 0.0                                                  # the body's own yaw rate (approach["gyro"]: the trunk's gyro,
                gy = ap.get("gyro")                                         # + left), the vestibular damping of the turn
                if gy is not None:
                    gb_ = frame.obs.get(gy[0])
                    if gb_ is not None:
                        rate = float(gb_[int(gy[1])])
                c = float(self._reflex_const("approach_damp"))
                if heard:                                                   # a voice: one onset, one pulse that turns it by about the angle
                    k = float(self._reflex_const("approach_sound_gain")); mx = float(self._reflex_const("approach_sound_max"))
                out[int(jt)] = g * max(-mx, min(mx, sg * k * bearing - c * rate)); moved = True
            jg = ap.get("go")
            if jg is not None and not heard and abs(bearing) <= float(self._reflex_const("approach_zone")):
                out[int(jg)] = g * float(self._reflex_const("approach_go")); moved = True
            return out if moved else None
        return None

    def _vor_acts(self):
        """the VOR's born constants for the world this tick (Acts.vor), per effector that declares it: its axes, the born gain, the quick
        phase's jump (a fraction of the reach)"""
        if not int(self._reflex_const("vor")):
            return {}
        return {e.name: {"axes": [int(j) for j in e.vor], "gain": float(self._reflex_const("vor_gain")),
                         "quick": float(self._reflex_const("vor_quick"))} for e in self.anatomy.motors if e.vor}
