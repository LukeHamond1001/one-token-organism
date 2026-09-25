"""the cerebellum: a loop below the tick (a named organ of the core, and a mixin of `Life`, body/life.py; the core refactor's step R6c,
docs/SIM_DESIGN.md 7.5, A44 and C50). The Marr-Albus cerebellum (Marr 1969; Albus 1971), read as an adaptive filter (Fujita 1982),
scoped to what its two teachers can teach: load compensation at the limbs and the VOR's gain. It never proposes an act, never enters a
gate's credit and never reads reward; its constants are physiology.py's CEREB, absent from a body's cfg unless given, and its switch
`cereb` is off by its absence, so the language body (whose anatomy declares no cerebellar interface) builds none of it.

THE ORGAN (`Cerebellum`, m.cereb, built last by the organs from a generator of its own seeded by the body's seed; every number float64,
every tensor a buffer saved with the body; it stays on the host, as the robot runs it on its own computer beside the servo loop):
- MOSSY FIBRES: the numbers the anatomy declares (body/core/anatomy.py `Cerebellar`; 7.5 lists for the humanoid the joints' angles,
  velocities, estimated torques and the servo's current targets, the efference copy of the tick's act, the inertial units and the
  hands' touch), each read as a fibre's rate r = 1 + (x - offset) / scale, held to [0, 2]: a tonic rate modulated both ways by the
  declared middle and half-range of its number, saturating at silence and at twice the tonic rate. MEASURED IN R6c: with each joint's
  estimated torque among them (the motor's torque, which carries the servo's correction and the cerebellum's own torque back into its
  input) the readout's gain, raised by a lesson blind to that loop, carries the limb into oscillation at its torque limit within a life
  day on the test limb; with the angles, velocities and targets alone it held all day (tools/cereb_day.py; cereb 10). The organ reads
  whatever is declared; which numbers the humanoid declares is the lead's call.
- THE GRANULE LAYER, born and fixed: `cereb_granule` units, each summing `cereb_fan_in` distinct fibres (drawn at birth) through born
  weights (normal, signed: fibres that rise and fibres that fall with their number, as the vestibular nuclei's two types), and the
  Golgi cells' inhibition holding the active fraction constant (Marr 1969; Albus 1971): the threshold each sub-step is the value that
  leaves `cereb_coding` of the units above it (k = its share of the units, rounded: 410 of 4,096), and each of those passes its excess
  over the threshold, g = z - theta; the rest are silent. The fibres' tonic rate gives each unit its own resting drive, so the code is
  not the same along every ray from the middle of the input's range. The active units are kept in ascending order (a canonical order
  for the sums below).
- PURKINJE READOUTS, one per declared joint (the humanoid's 29 of the arms, the legs and the waist): tau_j = sum over the active units
  of g_i w_ij, a torque (N m) the world adds to the joint's servo, inside the joint's limit and the weakness clip. Born at zero.
- THE FLOCCULUS, per declared VOR axis: a gain correction G_a = sum g_i f_ia and an offset O_a = sum g_i f_i(A+a), read once a tick
  at sub-step 0 and held through the tick by the world (the window counter-shifts by -(born gain + G) turn - O). Born at zero.
- ITS TEACHERS, the climbing fibres:
  * at the limbs, every sub-step, THE SERVO LAW'S OWN CORRECTIVE TORQUE e_j (feedback-error learning: Kawato and Gomi 1992): the
    readout is read first, then taught on the same sub-step's granule activity, w_ij += cereb_rate e_j g_i / |g|^2 (least mean
    squares normalized by the granule layer's activity), so each readout learns to supply what the servo would have spent, before the
    error appears. With no teacher (e all zero, or none) nothing is written;
  * at the flocculus, once a tick, RETINAL SLIP s_a (Ito 1982) with the gyro's turn t_a over the tick it was seen in: the granule
    activity the flocculus read that tick is its eligibility (kept in the organ, so a save between ticks loses nothing), and at the
    next tick's sub-step 0 it is taught, least mean squares on its two regressors (the turn for the gain, 1 for the offset) normalized
    by their power and the eligibility's activity: f_ia -= cereb_vor_rate s_a t_a g_i / (|g|^2 (t_a^2 + 1)) and f_i(A+a) -=
    cereb_vor_rate s_a g_i / (|g|^2 (t_a^2 + 1)). The slip is the gradient: s = (born gain + G - magnification) x turn + O plus the
    gyro's bias carried through the gain, so the gain goes where the image's motion asks and the offset cancels the gyro's drifting
    bias (A39). No slip (None) writes nothing.
- WHAT IT CANNOT DO (A44): feedback-error learning needs an innate feedback controller for what it learns, and its teacher knows joint
  angles only (righting and equilibrium reactions are refused, 3.7), so it learns load compensation and the VOR's gain, never a sit or
  a balance. WHAT IT MAY DO, disclosed: a limb left to rest sinks against the servo's lagging target, and that lag is a corrective
  torque, so the readout may learn part of a limb's own weight and slow a rested posture's sink (body/tests/test_cerebellum.py writes
  down the sink with it learning; W4 again on the G1).
- REPLAY: it draws no random number after birth, every weight and its eligibility are buffers in the save, and the world calls it at
  fixed sub-steps, so a replay is exact.

THE LIFE'S PART (`CerebellumMixin`): a life whose switch is on checks its organs' cerebellum against its anatomy's declaration and sets
its world's sub-tick hook (`Below`, holding the life weakly: body/core/world.py `World.below`); each sub-step the world calls it, and
`_cereb_sub` runs the organ's law at the life's constants. Nothing of the tick reads it: it adds no working attribute to the life, and
the tick's senses, choices, lessons and reward are what they were (the cortex feels its work only through the body's senses)."""
import time
import weakref

import numpy as np
import torch
from torch import nn

from .physiology import CEREB
from .world import SubActs


def cerebellum_spec(anatomy, cfg):
    """WHAT THE ORGANS BUILD (Organs(..., cerebellum=)): None while the switch is off (the language body's case: its cfg has no `cereb`);
    else the anatomy's declaration with the organ's born sizes, from the cfg or CEREB. The switch on with no declaration is refused."""
    c = cfg or {}
    if not int(c.get("cereb", CEREB["cereb"])):
        return None
    decl = getattr(anatomy, "cerebellar", None)
    if decl is None:
        raise ValueError("the cerebellum is switched on (cereb 1) and the anatomy declares no cerebellar interface (body/core/anatomy.py "
                         "Cerebellar: what the world feeds it below the tick)")
    return dict(decl=decl, granule=int(c.get("cereb_granule", CEREB["cereb_granule"])), fan_in=int(c.get("cereb_fan_in", CEREB["cereb_fan_in"])),
                coding=float(c.get("cereb_coding", CEREB["cereb_coding"])))


class Cerebellum(nn.Module):
    """THE ORGAN (see the module's doc): the born expansion, the Purkinje readouts, the flocculus, their eligibility and counters, every
    one a float64 (or index) buffer. `decl` is the anatomy's `Cerebellar`; `gen` the organ's own generator (the body's seed); `granule`,
    `fan_in` and `coding` its born sizes (CEREB)."""

    def __init__(self, decl, gen, granule, fan_in, coding):
        super().__init__()
        M, J, A = int(decl.n_mossy), len(decl.joints), len(decl.vor)
        G, K = int(granule), int(fan_in)
        k = int(round(float(coding) * G))
        if not 1 <= K <= M:
            raise ValueError(f"Cerebellum: each granule unit reads {K} distinct mossy fibres of the {M} declared")
        if not 1 <= k < G:
            raise ValueError(f"Cerebellum: {k} of {G} granule units active (coding {coding}): at least one, and fewer than all")
        self.n_mossy, self.joints, self.vor = M, tuple(str(j) for j in decl.joints), tuple(str(a) for a in decl.vor)
        self.granule, self.fan_in, self.k = G, K, k
        # the born expansion: each unit's fibres (distinct, drawn at birth) and its synapses' weights; the fibres' declared middles and
        # half-ranges, as born (a declaration changed later is refused by the life, not read)
        self.register_buffer("mossy_idx", torch.multinomial(torch.ones(G, M, dtype=torch.float64), K, replacement=False, generator=gen))
        self.register_buffer("mossy_w", torch.randn(G, K, generator=gen, dtype=torch.float64))
        self.register_buffer("mossy_off", torch.tensor([float(x) for x in decl.mossy_offset], dtype=torch.float64))
        self.register_buffer("mossy_scale", torch.tensor([float(x) for x in decl.mossy_scale], dtype=torch.float64))
        # the readouts, born at zero: the limbs' Purkinje cells [G, J], the flocculus's gain and offset per axis [G, 2A]
        self.register_buffer("pc_w", torch.zeros(G, J, dtype=torch.float64))
        self.register_buffer("fl_w", torch.zeros(G, 2 * A, dtype=torch.float64))
        # the flocculus's eligibility: the active units and their activity when it last read (taught by the slip of that tick), the
        # first fl_n of each (0: none held, as at birth)
        self.register_buffer("fl_top", torch.zeros(k, dtype=torch.long))
        self.register_buffer("fl_g", torch.zeros(k, dtype=torch.float64))
        self.register_buffer("fl_n", torch.zeros((), dtype=torch.long))
        # counters: sub-steps lived, limb lessons (sub-steps a teacher wrote), flocculus lessons
        self.register_buffer("n_sub", torch.zeros((), dtype=torch.long))
        self.register_buffer("n_limb", torch.zeros((), dtype=torch.long))
        self.register_buffer("n_vor", torch.zeros((), dtype=torch.long))

    def _apply(self, fn, recurse=True):
        """the cerebellum stays on the host whatever the organs are sent to (the night's device, Organs.to): its numbers are float64,
        which Metal lacks, and it runs below the tick on the CPU, as the robot runs it beside its servo loop"""
        return self

    def declares(self, decl):
        """whether the organ was born for this declaration (the same fibres with the same middles and half-ranges, joints and axes)"""
        return (int(decl.n_mossy) == self.n_mossy and tuple(str(j) for j in decl.joints) == self.joints and tuple(str(a) for a in decl.vor) == self.vor
                and np.array_equal(self.mossy_off.numpy(), np.array([float(x) for x in decl.mossy_offset], dtype=np.float64))
                and np.array_equal(self.mossy_scale.numpy(), np.array([float(x) for x in decl.mossy_scale], dtype=np.float64)))

    def granule_code(self, mossy):
        """the granule layer's activity for one mossy input (raw, the declared order): (the active units in ascending order, their
        activity, each its excess over the Golgi threshold): the k units whose drive is above the (k+1)-th largest drive, fewer where
        drives tie at it (a unit at the threshold passes nothing)"""
        x = np.asarray(mossy, dtype=np.float64).reshape(self.n_mossy)
        r = np.clip((x - self.mossy_off.numpy()) / self.mossy_scale.numpy(), -1.0, 1.0) + 1.0
        z = np.einsum("ij,ij->i", self.mossy_w.numpy(), r[self.mossy_idx.numpy()])
        theta = np.partition(z, self.granule - self.k - 1)[self.granule - self.k - 1]   # the (k+1)-th largest drive: the Golgi threshold
        top = np.flatnonzero(z > theta)                                # the units above it, in ascending order (k of them, fewer at a tie)
        return top, z[top] - theta

    @torch.no_grad()
    def sub_step(self, sf, rate, vor_rate):
        """ONE SUB-STEP OF THE LAW on the world's SubFrame `sf` at the life's rates: the limbs' torques read, then taught by the servo's
        corrective torque; at sub-step 0, the flocculus taught by the last tick's slip on its eligibility, then read. Returns
        (torque [J] or None, vor_gain [A] or None, vor_offset [A] or None), numpy float64."""
        J, A = len(self.joints), len(self.vor)
        top, g = self.granule_code(sf.mossy)
        n2 = float(np.einsum("i,i->", g, g))
        tau = None
        if J:
            pc = self.pc_w.numpy()
            P = pc[top]
            tau = np.einsum("i,ij->j", g, P)
            if sf.teach is not None and n2 > 0.0:
                e = np.asarray(sf.teach, dtype=np.float64).reshape(J)
                if np.any(e != 0.0):
                    pc[top] = P + (float(rate) / n2) * (g[:, None] * e[None, :])
                    self.n_limb.add_(1)
        gain = off = None
        if A and int(sf.sub) == 0:
            fw = self.fl_w.numpy()
            nh = int(self.fl_n)
            if sf.slip is not None and sf.turn is not None and nh:
                s = np.asarray(sf.slip, dtype=np.float64).reshape(A)
                t = np.asarray(sf.turn, dtype=np.float64).reshape(A)
                ft, fg = self.fl_top.numpy()[:nh], self.fl_g.numpy()[:nh]
                m2 = float(np.einsum("i,i->", fg, fg))
                if m2 > 0.0 and np.any(s != 0.0):
                    den = m2 * (t * t + 1.0)
                    c = np.concatenate([-float(vor_rate) * s * t / den, -float(vor_rate) * s / den])
                    fw[ft] = fw[ft] + fg[:, None] * c[None, :]
                    self.n_vor.add_(1)
            out = np.einsum("i,ij->j", g, fw[top])
            gain, off = out[:A].copy(), out[A:].copy()
            ft, fg = self.fl_top.numpy(), self.fl_g.numpy()             # the eligibility: this read's active units and activity
            ft[:] = 0; fg[:] = 0.0; ft[:top.size] = top; fg[:top.size] = g
            self.fl_n.fill_(int(top.size))
        self.n_sub.add_(1)
        return tau, gain, off


class Below:
    """THE HOOK A WORLD CALLS BELOW THE TICK (step R6c; body/core/world.py `World.below`): a life's cerebellum, held weakly (a world
    never keeps its body alive), with the period and the interface a world needs to call it (`period_s`, `joints`, `vor`, `n_mossy`,
    the anatomy's `Cerebellar`). Calling it with a SubFrame runs one sub-step of the law (`Life._cereb_sub`) and gives SubActs; once its
    life is gone it gives None. `calls` and `seconds` are an instrument (the wall time inside the law), never saved or sensed."""

    def __init__(self, life):
        self._life = weakref.ref(life)
        decl = life.anatomy.cerebellar
        self.period_s = float(life._cereb_const("cereb_period_s"))
        self.joints, self.vor, self.n_mossy = tuple(decl.joints), tuple(decl.vor), int(decl.n_mossy)
        self.calls = 0; self.seconds = 0.0

    def __call__(self, sf):
        life = self._life()
        if life is None:
            return None
        t0 = time.perf_counter()
        out = life._cereb_sub(sf)
        self.calls += 1; self.seconds += time.perf_counter() - t0
        return out


class CerebellumMixin:
    def _cereb_const(self, k):
        """a cerebellum constant: the body's cfg when it was given, else CEREB's"""
        return self.cfg.get(k, CEREB[k])

    def _cereb_attach(self):
        """BORN OR LOADED WITH THE SWITCH ON: the organs' cerebellum must be the one the anatomy declares (built by Organs(...,
        cerebellum=cerebellum_spec(anatomy, cfg))), and the world's sub-tick hook is set to this life's. With the switch off, organs that
        hold a cerebellum are refused (a body is born with its switches: SIM_DESIGN.md A20)."""
        on = bool(int(self._cereb_const("cereb")))
        org = self.m._modules.get("cereb")
        if not on:
            if org is not None:
                raise ValueError("Life: the organs hold a cerebellum (m.cereb) and the life's constants switch it off (cereb 0)")
            return
        decl = self.anatomy.cerebellar
        if decl is None:
            raise ValueError("Life: the cerebellum is switched on (cereb 1) and the anatomy declares no cerebellar interface (Cerebellar)")
        if org is None or not isinstance(org, Cerebellum) or not org.declares(decl):
            raise ValueError(f"Life: the organs' cerebellum ({None if org is None else (org.n_mossy, org.joints, org.vor)}) is not the one the anatomy "
                             f"declares ({decl.n_mossy} mossy numbers, joints {list(decl.joints)}, VOR axes {list(decl.vor)}; built by Organs(..., "
                             f"cerebellum=cerebellum_spec(anatomy, cfg)))")
        per_ = float(self._cereb_const("cereb_period_s"))
        if not per_ > 0.0:
            raise ValueError(f"Life: the cerebellum's period is {per_} s")
        self.world.below = Below(self)

    def _cereb_sub(self, sf):
        """one sub-step below the tick (the world's call through Below): the organ's law at this life's rates, as SubActs"""
        tau, gain, off = self.m.cereb.sub_step(sf, float(self._cereb_const("cereb_rate")), float(self._cereb_const("cereb_vor_rate")))
        return SubActs(tau, gain, off)
