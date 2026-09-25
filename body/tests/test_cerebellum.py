"""the cerebellum below the tick (docs/SIM_DESIGN.md 7.5, A44 and C50; the core refactor's step R6c). Run: python3 -m body.tests.test_cerebellum
(the organ tests run these too). The organ is body/core/cerebellum.py; the world's sub-tick hook is body/core/world.py's `World.below`.

What must hold: the language body has none of it (no organ, no attribute, no key, no hook; the eight pinned digests are the guard's,
tools/pins/digests.txt); a body whose switch is on has it built last from its own generator, every other organ born as without it; the
law is least mean squares on the granule code, exact against numpy; with no teacher it writes nothing and adds nothing; it learns a
limb's load (a weight added to a hand: the servo's corrective torque falls over time with it on, not off); the flocculus learns the VOR's
gain from retinal slip under a magnifying lens and cancels a biased gyro; through a life, the world calls it fifteen times a tick and
nothing of the tick reads it; a save round trip is exact and a replay across processes equal; its cost at the humanoid's size.

The test limb is a MuJoCo arm of two hinges in a vertical plane (a shoulder and an elbow, 1 kg and 0.8 kg links of 0.2 m, a hand), each
joint a position servo at the servo law's form (SIM_DESIGN.md 3.3: stiffness at the joint's limit for 0.25 rad of error, damping 0.04 s x
the stiffness, the torque held to the limit; an act re-anchors the target at the measured angle plus the step; at rest the target
relaxes to the measured angle with a time constant of 3 ticks), the cerebellum's torque the actuators' constant bias term, as the
SimWorld interface says. It is an instrument of these tests, not the G1's world. Its mossy numbers are the joints' state and the
efference copy (each joint's angle, velocity and servo target: MOSSY_KINDS "state"); SIM_DESIGN.md 7.5's list adds each joint's
estimated torque ("design"), with which the loop goes unstable on this limb within a life day (cereb 10; tools/cereb_day.py)."""
import math
import os
import pickle
import shutil
import subprocess
import sys
import tempfile
import time

import numpy as np
import torch
from tokenizers import Tokenizer

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)   # this tree's body, not a fixed one
from body.life import Life, PHYSIOLOGY  # noqa: E402
from body.core.anatomy import Cerebellar, Channel, Effector, LanguageAnatomy  # noqa: E402
from body.core.cerebellum import Below, Cerebellum, cerebellum_spec  # noqa: E402
from body.core.physiology import CEREB, MOTOR, SWITCHES  # noqa: E402
from body.core.world import Frame, SimWorld, SubActs, SubFrame, World, WorldLoop  # noqa: E402
from body.model import Organs  # noqa: E402

TOK = Tokenizer.from_file("/Users/lukehamond/Projects/project/data/tok_char.json")
STEPS = (-0.27, -0.09, 0.0, 0.09, 0.27)            # the arm's settings per joint, rad a tick (SIM_DESIGN.md 3.5)
REST = 12                                           # the arm's rest: both joints at setting 2
TICK_S, DT, N_STEPS = 0.15, 0.002, 75
_LR0 = dict(live_lr=0.0, value_lr=0.0, band_lr=0.0, night_lr=0.0, gate_lr=0.0, gate_adam_lr=0.0, vcrit_lr=0.0, actor_lr=0.0, face_lr=0.0,
            act_inv_lr=0.0, wake_ticks=100000, wake_every=8, gate_every=8, write_floor=1e-30, fast_rls=0)

ARM_XML = """
<mujoco model="cerebellum test limb">
  <compiler angle="radian"/>
  <option timestep="0.002" gravity="0 0 -9.81" integrator="implicitfast"/>
  <worldbody>
    <body name="upper" pos="0 0 1">
      <joint name="shoulder" type="hinge" axis="0 -1 0" range="-2.5 2.5" limited="true" damping="0.05" armature="0.02"
             actuatorfrclimited="true" actuatorfrcrange="-25 25"/>
      <geom type="capsule" fromto="0 0 0 0.2 0 0" size="0.03" mass="1.0" contype="0" conaffinity="0"/>
      <body name="fore" pos="0.2 0 0">
        <joint name="elbow" type="hinge" axis="0 -1 0" range="-0.2 2.6" limited="true" damping="0.05" armature="0.02"
               actuatorfrclimited="true" actuatorfrcrange="-25 25"/>
        <geom type="capsule" fromto="0 0 0 0.2 0 0" size="0.025" mass="0.8" contype="0" conaffinity="0"/>
        <body name="hand" pos="0.2 0 0">
          <geom type="sphere" size="0.03" mass="0.05" contype="0" conaffinity="0"/>
        </body>
      </body>
    </body>
  </worldbody>
  <actuator>
    <position name="shoulder" joint="shoulder" kp="1" kv="1"/>
    <position name="elbow" joint="elbow" kp="1" kv="1"/>
  </actuator>
</mujoco>"""
ARM_VEL = 10.0                                      # the test limb's declared velocity scale, rad/s (an instrument's)
# THE LIMB'S MOSSY NUMBERS, by kind: each quantity per joint (quantity by quantity, the joints in order), about its declared middle and
# half-range. "state", which these tests declare, is the joints' state and the efference copy: each joint's angle, velocity and the
# servo's target. "design" is SIM_DESIGN.md 7.5's list for the humanoid, which adds each joint's estimated torque: the torque the motor
# applies, which carries the servo's correction and the cerebellum's own torque back into its input, and with it the loop goes unstable
# on this limb within a life day (cereb 10; tools/cereb_day.py measures each kind over a whole day). The others are that instrument's
MOSSY_KINDS = {"state": ("angle", "velocity", "target"), "design": ("angle", "velocity", "torque", "target"), "efference": ("target", "step"),
               "efference+angle": ("target", "step", "angle"), "efference+velocity": ("target", "step", "velocity"),
               "efference+torque": ("target", "step", "torque")}


def arm_cerebellar(kind="state"):
    """the test limb's cerebellar interface: its mossy numbers of `kind` (MOSSY_KINDS), each about its declared middle and half-range
    (an angle or a target: its joint's range; a velocity: the declared scale; a torque: the joint's limit; a step: the big step), then
    the two joints' readouts"""
    import mujoco
    m = mujoco.MjModel.from_xml_string(ARM_XML)
    lo, hi = m.jnt_range[:, 0], m.jnt_range[:, 1]
    mid, half, lim = (lo + hi) / 2, (hi - lo) / 2, m.jnt_actfrcrange[:, 1]
    decl = {"angle": (mid, half), "target": (mid, half), "velocity": ((0.0, 0.0), (ARM_VEL, ARM_VEL)), "torque": ((0.0, 0.0), lim),
            "step": ((0.0, 0.0), (STEPS[-1], STEPS[-1]))}
    off, sc = [], []
    for q in MOSSY_KINDS[kind]:
        off += [float(x) for x in decl[q][0]]; sc += [float(x) for x in decl[q][1]]
    return Cerebellar(off, sc, joints=["shoulder", "elbow"])


class ArmWorld(SimWorld):
    """THE TEST LIMB (see the module's doc), lockstep, implementing the SimWorld interface with its loop below the tick: each tick the
    arm's act (its flat id of two joints of five settings; the rest holds by tone), 75 physics steps of 2 ms, and every
    below.period_s the SubFrame (the mossy numbers in arm_cerebellar's order; the teacher the servo law's own torque) and the answer's
    torque set as the actuators' bias. `load(kg)` adds a weight to the hand. It records each tick's mean |teacher| per joint (over its
    sub-steps) and the cerebellum's torque. Its state (the physics, the torque applied, the tick's step, the load, its clock) saves and
    restores exactly. `mossy` names the kind of mossy numbers it hands below the tick (MOSSY_KINDS; its declaration arm_cerebellar's)."""

    def __init__(self, q0=(-0.3, 1.0), mossy="state"):
        import mujoco
        self.mj = mujoco
        self.m = mujoco.MjModel.from_xml_string(ARM_XML); self.d = mujoco.MjData(self.m)
        m, d = self.m, self.d
        self.hand = m.body("hand").id; self.hand_mass0 = float(m.body_mass[self.hand])
        self.lo, self.hi = m.jnt_range[:, 0].copy(), m.jnt_range[:, 1].copy()
        self.lim = m.jnt_actfrcrange[:, 1].copy()
        self.kp = self.lim / 0.25; self.kv = 0.04 * self.kp            # the servo law's form (SIM_DESIGN.md 3.3's first law)
        for a in range(2):
            m.actuator_gainprm[a, :] = 0.0; m.actuator_gainprm[a, 0] = self.kp[a]
            m.actuator_biasprm[a, :] = 0.0; m.actuator_biasprm[a, 1] = -self.kp[a]; m.actuator_biasprm[a, 2] = -self.kv[a]
        self.alpha = 1.0 - math.exp(-DT / (3.0 * TICK_S))              # the tone at rest: 3 ticks
        d.qpos[:] = q0; d.ctrl[:] = q0; mujoco.mj_forward(m, d)
        self.t = 0; self.paused = False; self.step = np.zeros(2); self.kind = MOSSY_KINDS[mossy]
        self.teach_log = []; self.tau_log = []; self.calls = 0

    def load(self, kg):
        self.m.body_mass[self.hand] = self.hand_mass0 + float(kg)

    def q(self):
        return self.d.qpos.copy()

    def frame(self):
        assert not self.paused, "a frame taken while the world is paused"
        d = self.d
        return Frame(self.t, {"body": [float(d.qvel[0]) / ARM_VEL, float(d.qvel[1]) / ARM_VEL]}, 0.0, {"q": d.qpos.copy()})

    def teacher(self):
        """the servo law's own corrective torque now, held to the limit (the cerebellum's torque not in it)"""
        d = self.d
        return np.clip(self.kp * (d.ctrl - d.qpos) - self.kv * d.qvel, -self.lim, self.lim)

    def apply(self, acts, force=None):
        assert not self.paused, "the world moved while paused"
        mj, m, d = self.mj, self.m, self.d
        a = int((acts or {}).get("arm", REST))
        rest = a == REST
        self.step = np.zeros(2) if rest else np.array([STEPS[a // 5], STEPS[a % 5]])
        if not rest:
            d.ctrl[:] = np.clip(d.qpos + self.step, self.lo, self.hi)
        below = self.below
        n = int(round(below.period_s / DT)) if below is not None else 0
        assert not n or N_STEPS % n == 0, f"the loop's period ({n} steps) does not divide the tick"
        te = []
        for s in range(N_STEPS):
            if n and s % n == 0:
                teach = self.teacher(); te.append(np.abs(teach))
                got = {"angle": d.qpos, "velocity": d.qvel, "torque": d.qfrc_actuator, "target": d.ctrl, "step": self.step}
                mossy = np.concatenate([got[q] for q in self.kind])
                ans = self.sub_tick(SubFrame(self.t, s // n, mossy, teach)); self.calls += 1
                if ans is not None and ans.torque is not None:
                    m.actuator_biasprm[:, 0] = np.asarray(ans.torque, dtype=np.float64)
            elif not n and s % 5 == 0:
                te.append(np.abs(self.teacher()))
            if rest:
                d.ctrl[:] += self.alpha * (d.qpos - d.ctrl)
            if force is not None:
                d.xfrc_applied[self.hand, :3] = force
            mj.mj_step(m, d)
        d.xfrc_applied[self.hand, :] = 0.0
        mj.mj_forward(m, d)                                            # the tick's end: its forces from its end state (as a restore's)
        self.teach_log.append(np.mean(te, axis=0)); self.tau_log.append(m.actuator_biasprm[:, 0].copy())
        self.t += 1

    def pause(self):
        self.paused = True

    def resume(self):
        self.paused = False

    def save_state(self):
        mj = self.mj
        spec = mj.mjtState.mjSTATE_INTEGRATION
        st = np.zeros(mj.mj_stateSize(self.m, spec)); mj.mj_getState(self.m, self.d, st, spec)
        return pickle.dumps(dict(state=st, bias=self.m.actuator_biasprm[:, 0].copy(), mass=float(self.m.body_mass[self.hand]), t=self.t,
                                 paused=self.paused, step=self.step.copy()), protocol=4)

    def load_state(self, blob):
        mj = self.mj
        b = pickle.loads(blob)
        self.m.actuator_biasprm[:, 0] = b["bias"]; self.m.body_mass[self.hand] = b["mass"]
        mj.mj_setState(self.m, self.d, b["state"], mj.mjtState.mjSTATE_INTEGRATION)
        mj.mj_forward(self.m, self.d)
        self.t = int(b["t"]); self.paused = bool(b["paused"]); self.step = np.array(b["step"], dtype=np.float64)


class OrganHook:
    """an organ's law as a world's hook, for the organ's own tests (the life's hook is body/core/cerebellum.py `Below`, whose call is the
    same law at the life's rates: test cereb 7)"""

    def __init__(self, organ, rate=CEREB["cereb_rate"], vor_rate=CEREB["cereb_vor_rate"], period_s=CEREB["cereb_period_s"]):
        self.organ, self.rate, self.vor_rate, self.period_s = organ, float(rate), float(vor_rate), float(period_s)
        self.joints, self.vor, self.n_mossy = organ.joints, organ.vor, organ.n_mossy

    def __call__(self, sf):
        return SubActs(*self.organ.sub_step(sf, self.rate, self.vor_rate))


def organ(decl, seed=1, **kw):
    """an organ born as the organs build it (its own generator, the body's seed and the prime the organs add), at CEREB's sizes unless given"""
    spec = dict(granule=CEREB["cereb_granule"], fan_in=CEREB["cereb_fan_in"], coding=CEREB["cereb_coding"]); spec.update(kw)
    return Cerebellum(decl, torch.Generator().manual_seed(int(seed) + 49979687), spec["granule"], spec["fan_in"], spec["coding"])


# the instrument's driver for the limb: four postures in turn, 25 ticks each, every joint stepped toward its posture by the setting whose
# step is nearest the error (0 when within half the small step); both joints at 0 is the rest. It reads the world's truth, never the body
POSTURES = ((-0.2, 1.0), (0.3, 0.6), (-0.6, 1.5), (0.1, 1.2))


def drive(q, t):
    goal = POSTURES[(t // 25) % len(POSTURES)]
    s = [int(np.argmin([abs((g - x) - st) for st in STEPS])) for g, x in zip(goal, q)]
    return s[0] * 5 + s[1]


def run_limb(world, ticks, load_at=None, kg=0.5, t0=0):
    for t in range(t0, t0 + ticks):
        if load_at is not None and t == load_at:
            world.load(kg)
        world.apply({"arm": drive(world.q(), t)})


def blocks(log, j, size=100):
    x = np.array(log)[:, j]
    return [float(x[i:i + size].mean()) for i in range(0, len(x) - size + 1, size)]


class Head:
    """THE TEST HEAD FOR THE VOR (an instrument): the trunk turns the head about two axes (yaw and pitch of the fovea window) by a
    scripted rhythm; a gyro reads each tick's turn with a bias (rad/s) and white noise from a seeded stream; a thing held still in the
    world is fixated; the optics magnify the world's image motion by `mag` (a magnifying lens: 2). Each tick: sub-step 0 hands the
    cerebellum the gyro's rates and the head's attitude (its mossy numbers) with the last tick's slip and turn; the window counter-shifts
    by -(1 + gain) x turn - offset (the VOR, born gain 1); the slip is the image's motion relative to the window, -mag x the true turn
    minus the counter-shift. Sub-steps 1 to 14 hand it the same numbers (no teacher: the head has no limb)."""
    AMP = (0.3, 0.2)                                  # rad a tick at the rhythm's peak (2 and 1.3 rad/s: brisk turns of the trunk)
    PER = (37, 53)                                    # ticks per cycle, per axis

    def __init__(self, hook=None, mag=1.0, bias=(0.0, 0.0), noise=0.002, seed=0):
        self.hook, self.mag, self.bias, self.noise = hook, float(mag), np.array(bias, dtype=np.float64), float(noise)
        self.rng = np.random.Generator(np.random.PCG64(seed))
        self.t = 0; self.att = np.zeros(2); self.slip = None; self.turn = None
        self.slip_log = []; self.gain_log = []; self.off_log = []

    @staticmethod
    def cerebellar():
        """the head's interface: the gyro's rates on three axes and the attitude on two (rad/s over 5, rad over 1.5), no joints, two VOR axes"""
        return Cerebellar([0.0] * 5, [5.0, 5.0, 5.0, 1.5, 1.5], joints=[], vor=["yaw", "pitch"])

    def tick(self):
        t = self.t
        true = np.array([self.AMP[a] * math.sin(2 * math.pi * t / self.PER[a]) for a in range(2)])
        turn = true + self.bias * TICK_S + self.noise * self.rng.standard_normal(2)
        mossy = np.concatenate([turn / TICK_S, [0.0], self.att])
        gain = off = np.zeros(2)
        if self.hook is not None:
            for s in range(15):
                ans = self.hook(SubFrame(t, s, mossy, None, self.slip if s == 0 else None, self.turn if s == 0 else None))
                if s == 0:
                    gain, off = ans.vor_gain, ans.vor_offset
        shift = -(1.0 + gain) * turn - off
        self.slip = -self.mag * true - shift; self.turn = turn
        self.att = self.att + true
        self.slip_log.append(self.slip.copy()); self.gain_log.append(np.array(gain, dtype=np.float64)); self.off_log.append(np.array(off, dtype=np.float64))
        self.t += 1


class ArmAnatomy(LanguageAnatomy):
    """the diary's words and face, the arm's body sense (its joints' velocities over the declared scale, a channel of the world's frames
    encoded by the face's map), the arm (two joints of five settings, its rest 12) and the test limb's cerebellar interface"""

    def __init__(self, tok, cfg=None):
        super().__init__(tok, cfg)
        self.channels.append(Channel("body", "vector", 2, organ="face_in"))
        self.effectors.append(Effector("arm", [5, 5], rest_id=REST, effort=0.05, sense="body"))
        self.cerebellar = arm_cerebellar()


def born(cfg, world=None, seed=0, anatomy=None):
    return Life.birth(anatomy or ArmAnatomy(TOK, cfg), device="cpu", d=64, layers=2, heads=2, window=32, cfg=cfg, seed=seed, world=world)


def _hash(items):
    import hashlib
    from body.tests.test_anatomy import _canon
    g = hashlib.sha256()
    for k, v in items:
        g.update(k.encode() + b"="); _canon(g, v, k)
    return g.hexdigest()[:16]


def _life_but_cerebellum(L):
    """a life's every part but its cerebellum, a hash per part: the organs (every state entry, gradient and plain attribute but the
    cerebellum's), the store, the optimizers, the random streams, every working attribute but the interface objects, the constants less
    the switch"""
    m = L.m; OPTS = sorted(k for k, v in vars(L).items() if isinstance(v, torch.optim.Optimizer))
    org = [("sd", {k: v for k, v in m.state_dict().items() if not k.startswith("cereb.")}), ("grad", {n: p.grad for n, p in m.named_parameters()})]
    for mn, mod in m.named_modules():
        if mn.split(".")[0] != "cereb":
            org.append(("attrs:" + (mn or "."), {k: v for k, v in vars(mod).items() if not k.startswith("_") and not isinstance(v, torch.nn.Module)}))
    SKIP = {"m", "store", "tok", "gen", "cfg", "save_path", "_t_feel", "anatomy", "effectors", "world"} | set(OPTS)
    return {"organs": _hash(org), "store": _hash(sorted(vars(L.store).items())), "optim": _hash([(k, getattr(L, k).state_dict()) for k in OPTS]),
            "rng": _hash([("gen", L.gen.get_state()), ("torch", torch.get_rng_state())]),
            "work": _hash([(k, v) for k, v in sorted(vars(L).items()) if k not in SKIP]),
            "cfg": _hash([("cfg", {k: v for k, v in L.cfg.items() if k != "cereb"})])}


# ---------------------------------------------------------------- the tests

def test_absent_for_language():
    """cereb 1 (SIM_DESIGN.md 8.3; A44): the language body has none of it. Its cfg holds no cerebellum constant and the physiology's
    table none (CEREB is a table of its own, absent unless given); its anatomy declares no cerebellar interface (the class's None, no
    attribute); its organs build no cerebellum and its save holds no key of one; its world has no hook (the class's None) and a call
    below the tick answers nothing; born, lived and reloaded, the life never runs the cerebellum's attach (it is replaced here by one that
    fails); the mixin holds no class attribute. (The eight pinned digests are the guard's: tools/pins/digests.txt.)"""
    from body.core.cerebellum import CerebellumMixin
    assert not set(CEREB) & (set(PHYSIOLOGY) | set(MOTOR) | set(SWITCHES)) and CEREB["cereb"] == 0
    assert all(callable(v) or k.startswith("__") for k, v in vars(CerebellumMixin).items()), "the mixin holds a class attribute"

    def refuse(self):
        raise AssertionError("the language body ran the cerebellum's attach")
    keep = CerebellumMixin._cereb_attach; CerebellumMixin._cereb_attach = refuse
    tmp = tempfile.mkdtemp()
    try:
        L = Life.birth(TOK, device="cpu", d=64, layers=2, heads=2, window=32, cfg=dict(wake_ticks=400, wake_every=8, gate_every=8), seed=0)
        assert "cereb" not in L.m._modules and not any(k.startswith("cereb") for k in L.m.state_dict())
        assert L.anatomy.cerebellar is None and "cerebellar" not in vars(L.anatomy) and cerebellum_spec(L.anatomy, L.cfg) is None
        assert type(L.world).below is None and "below" not in vars(L.world) and L.world.sub_tick(SubFrame(0, 0, [0.0])) is None
        assert not set(CEREB) & set(L.cfg) and not any("cereb" in k for k in vars(L))
        L.type_text("hello there")
        for _ in range(40):
            L.tick()
        p = os.path.join(tmp, "lang.pt"); L.save(p)
        blob = torch.load(p, map_location="cpu", weights_only=False)
        assert not any("cereb" in k for k in blob["organs"]) and not any("cereb" in str(k) for k in blob["life"]) and not set(CEREB) & set(blob["cfg"])
        L2 = Life.load(p, TOK, device="cpu", seed=0)
        assert "cereb" not in L2.m._modules and "below" not in vars(L2.world)
    finally:
        CerebellumMixin._cereb_attach = keep
        shutil.rmtree(tmp, ignore_errors=True)
    print("cereb 1: the language body has none of it: no constant in its cfg, no declaration, no organ, no save key, no hook; its attach never runs")


def test_built_last_from_its_own_generator():
    """cereb 2 (SIM_DESIGN.md 7.5, 8.3): a body whose switch is on has the cerebellum built after every other organ, from a generator of
    its own seeded by the body's seed: every other organ is born bit for bit as with the switch off, the global random stream and the
    life's own are left where they are, and the life gains no working attribute; the same seed gives the same expansion, another seed
    another; the readouts are born at zero; each granule unit reads four distinct fibres; exactly 410 of 4,096 units are active on any
    input; the world's hook is the life's. Refused: the switch on with no declaration, organs holding one under a switch that is off, an
    organ born for another declaration, and a declaration that does not hold together"""
    a = ArmAnatomy(TOK, {}); V = a.vocab
    spec = cerebellum_spec(a, dict(cereb=1))
    assert spec["decl"] is a.cerebellar and (spec["granule"], spec["fan_in"], spec["coding"]) == (4096, 4, 0.1)
    torch.manual_seed(9); o0 = Organs(V, d=64, layers=2, heads=2, window=32, channels=a.channels, effectors=a.effectors, born_seed=3); r0 = torch.get_rng_state()
    torch.manual_seed(9); o1 = Organs(V, d=64, layers=2, heads=2, window=32, channels=a.channels, effectors=a.effectors, born_seed=3, cerebellum=spec); r1 = torch.get_rng_state()
    torch.manual_seed(1); o2 = Organs(V, d=64, layers=2, heads=2, window=32, channels=a.channels, effectors=a.effectors, born_seed=3, cerebellum=spec)
    torch.manual_seed(9); o3 = Organs(V, d=64, layers=2, heads=2, window=32, channels=a.channels, effectors=a.effectors, born_seed=4, cerebellum=spec)
    s0, s1, s2, s3 = o0.state_dict(), o1.state_dict(), o2.state_dict(), o3.state_dict()
    assert torch.equal(r0, r1) and list(s1)[:len(s0)] == list(s0) and all(torch.equal(s0[k], s1[k]) for k in s0), "the other organs are born otherwise"
    new = list(s1)[len(s0):]
    assert new == ["cereb.mossy_idx", "cereb.mossy_w", "cereb.mossy_off", "cereb.mossy_scale", "cereb.pc_w", "cereb.fl_w", "cereb.fl_top",
                   "cereb.fl_g", "cereb.fl_n", "cereb.n_sub", "cereb.n_limb", "cereb.n_vor"], new
    assert all(torch.equal(s1[k], s2[k]) for k in new), "the expansion is not the body's seed's alone"
    assert not torch.equal(s1["cereb.mossy_idx"], s3["cereb.mossy_idx"]) and not torch.equal(s1["cereb.mossy_w"], s3["cereb.mossy_w"])
    c = o1.cereb
    assert c.granule == 4096 and c.k == 410 and c.fan_in == 4 and c.n_mossy == 6 and c.joints == ("shoulder", "elbow") and c.vor == ()
    assert tuple(c.pc_w.shape) == (4096, 2) and not c.pc_w.any() and c.fl_w.numel() == 0 and all(b.dtype in (torch.float64, torch.long) for b in c.buffers())
    idx = c.mossy_idx
    assert int(idx.min()) >= 0 and int(idx.max()) < 6 and all(len(set(r)) == 4 for r in idx.tolist())
    rng = np.random.default_rng(0); decl = a.cerebellar
    for _ in range(50):
        x = np.array(decl.mossy_offset) + np.array(decl.mossy_scale) * rng.uniform(-1.3, 1.3, 6)
        top, g = c.granule_code(x)
        assert top.size == 410 and np.all(np.diff(top) > 0) and np.all(g > 0)
    assert o1.to("cpu").cereb is c and c.pc_w.device.type == "cpu"
    # born as lives, switch on and off
    cfg = dict(_LR0)
    w0, w1 = ArmWorld(), ArmWorld()
    torch.manual_seed(0); L0 = born(dict(cfg), w0); g0 = torch.get_rng_state()
    torch.manual_seed(0); L1 = born(dict(cfg, cereb=1), w1); g1 = torch.get_rng_state()
    assert torch.equal(g0, g1) and torch.equal(L0.gen.get_state(), L1.gen.get_state()) and sorted(vars(L0)) == sorted(vars(L1))
    assert _life_but_cerebellum(L0) == _life_but_cerebellum(L1)
    assert "cereb" not in L0.m._modules and w0.below is None and isinstance(w1.below, Below) and w1.below._life() is L1
    assert (w1.below.period_s, w1.below.joints, w1.below.vor, w1.below.n_mossy) == (0.01, ("shoulder", "elbow"), (), 6)
    # refused
    bad = []
    for label, fn in (("the switch on with no declaration", lambda: Life.birth(TOK, device="cpu", d=64, layers=2, heads=2, window=32, cfg=dict(cereb=1), seed=0)),
                      ("organs holding one under a switch that is off", lambda: Life(Organs(V, d=64, layers=2, heads=2, window=32, channels=a.channels,
                                                                                               effectors=a.effectors, cerebellum=spec), a, cfg={})),
                      ("an organ born for another declaration", lambda: Life(Organs(V, d=64, layers=2, heads=2, window=32, channels=a.channels, effectors=a.effectors,
                                                                                    cerebellum=dict(spec, decl=Cerebellar([0.0] * 6, [2.0] * 6, joints=["shoulder", "elbow"]))),
                                                                             a, cfg=dict(cereb=1), world=ArmWorld()))):
        try:
            fn(); bad.append(label)
        except ValueError:
            pass
    assert not bad, f"not refused: {bad}"
    for label, cb in (("offsets and scales of two lengths", Cerebellar([0.0] * 3, [1.0] * 4, joints=["a"])),
                      ("a scale at zero", Cerebellar([0.0] * 4, [1.0, 1.0, 0.0, 1.0], joints=["a"])),
                      ("a joint named twice", Cerebellar([0.0] * 4, [1.0] * 4, joints=["a", "a"])),
                      ("nothing to learn", Cerebellar([0.0] * 4, [1.0] * 4)),
                      ("no mossy number", Cerebellar([], [], joints=["a"]))):
        b = ArmAnatomy(TOK, {}); b.cerebellar = cb
        try:
            b.check(); bad.append(label)
        except ValueError:
            pass
    assert not bad, f"the declaration's check let through: {bad}"
    print("cereb 2: built last from its own generator (the other organs, both random streams and the life's attributes as with it off);",
          "410 of 4,096 units active, 4 distinct fibres each, the readouts born at zero; the hook the life's; eight refusals")


class RefLaw:
    """7.5's law written again plainly, for cereb 3: the rates, the granule drive as a sum over each unit's fibres, the threshold as the
    (k+1)-th largest drive found by a full sort, the readouts and the lessons unit by unit"""

    def __init__(self, o):
        self.idx, self.w = o.mossy_idx.numpy().copy(), o.mossy_w.numpy().copy()
        self.off, self.sc = o.mossy_off.numpy().copy(), o.mossy_scale.numpy().copy()
        self.k = o.k; self.W = np.zeros_like(o.pc_w.numpy()); self.F = np.zeros_like(o.fl_w.numpy()); self.elig = None
        self.A = len(o.vor)

    def code(self, x):
        r = np.minimum(np.maximum((np.asarray(x, float) - self.off) / self.sc, -1.0), 1.0) + 1.0
        z = np.array([sum(float(self.w[i, j]) * float(r[self.idx[i, j]]) for j in range(self.idx.shape[1])) for i in range(self.idx.shape[0])])
        theta = np.sort(z)[::-1][self.k]
        top = [i for i in range(z.size) if z[i] > theta]
        return top, [float(z[i] - theta) for i in top]

    def step(self, sf, rate, vor_rate):
        top, g = self.code(sf.mossy)
        n2 = sum(v * v for v in g)
        tau = [sum(g[n] * self.W[i, j] for n, i in enumerate(top)) for j in range(self.W.shape[1])]
        if sf.teach is not None and n2 > 0 and any(e != 0 for e in sf.teach):
            for n, i in enumerate(top):
                for j, e in enumerate(sf.teach):
                    self.W[i, j] += rate * float(e) * g[n] / n2
        gain = off = None
        if self.A and sf.sub == 0:
            if sf.slip is not None and self.elig is not None:
                et, eg = self.elig; m2 = sum(v * v for v in eg)
                for a in range(self.A):
                    s, t = float(sf.slip[a]), float(sf.turn[a])
                    for n, i in enumerate(et):
                        self.F[i, a] -= vor_rate * s * t * eg[n] / (m2 * (t * t + 1.0))
                        self.F[i, self.A + a] -= vor_rate * s * eg[n] / (m2 * (t * t + 1.0))
            gain = [sum(g[n] * self.F[i, a] for n, i in enumerate(top)) for a in range(self.A)]
            off = [sum(g[n] * self.F[i, self.A + a] for n, i in enumerate(top)) for a in range(self.A)]
            self.elig = (top, g)
        return top, tau, gain, off


def test_the_law_against_numpy():
    """cereb 3 (SIM_DESIGN.md 7.5's law, least mean squares normalized by the granule layer's activity, exact against numpy): an organ of
    1,024 granule units over 12 mossy numbers, 3 joints and 2 VOR axes, and the law written again plainly (RefLaw), fed the same 240
    sub-steps (a wandering input, teachers on most sub-steps and silent or absent on others, slips on most ticks' first sub-step, none on
    others): the same active units at every sub-step, the torques, gains and offsets and every weight equal to 1e-12 of their size"""
    decl = Cerebellar([0.1 * i for i in range(12)], [0.5 + 0.1 * i for i in range(12)], joints=["a", "b", "c"], vor=["yaw", "pitch"])
    o = organ(decl, seed=5, granule=1024); ref = RefLaw(o)
    rng = np.random.default_rng(3); x = np.array(decl.mossy_offset, float); worst = 0.0
    for t in range(16):
        for s in range(15):
            x = x + 0.2 * np.array(decl.mossy_scale) * rng.standard_normal(12)
            teach = None if (t, s) in ((2, 3), (7, 0)) else (np.zeros(3) if s == 5 else rng.standard_normal(3) * 2.0)
            slip = turn = None
            if s == 0 and t % 5 != 4:
                slip, turn = rng.standard_normal(2) * 0.02, rng.standard_normal(2) * 0.2
            sf = SubFrame(t, s, x.copy(), teach, slip, turn)
            tau, gain, off = o.sub_step(sf, CEREB["cereb_rate"], CEREB["cereb_vor_rate"])
            rtop, rtau, rgain, roff = ref.step(sf, CEREB["cereb_rate"], CEREB["cereb_vor_rate"])
            assert list(o.granule_code(x)[0]) == rtop, f"the active units differ at tick {t}, sub-step {s}"
            for mine, theirs in ((tau, rtau), (gain, rgain), (off, roff)):
                if theirs is None:
                    assert mine is None
                else:
                    worst = max(worst, float(np.max(np.abs(np.asarray(mine) - np.asarray(theirs)) / (1.0 + np.abs(np.asarray(theirs))))))
    for mine, theirs in ((o.pc_w.numpy(), ref.W), (o.fl_w.numpy(), ref.F)):
        worst = max(worst, float(np.max(np.abs(mine - theirs)) / (1e-300 + np.max(np.abs(theirs)))))
    assert worst < 1e-12, worst
    assert int(o.n_sub) == 240 and int(o.n_limb) == 240 - 16 - 2 and int(o.n_vor) == 16 - 1 - 3, (int(o.n_sub), int(o.n_limb), int(o.n_vor))
    print(f"cereb 3: the law against numpy over 240 sub-steps: the same active units, every output and weight within {worst:.1e} of its size")


def test_silent_without_a_teacher():
    """cereb 4 (A44: nothing but its teachers writes it): a newborn cerebellum at the humanoid's size, fed 600 sub-steps of states wandering
    over their whole range with every teacher silent (zeros, or none) and no slip, gives exactly zero torque, gain and offset at every
    one and writes nothing; one that has learned, given a state that does not change and a silent teacher, gives the same torque to the
    bit at every sub-step and writes nothing; and the test limb resting in no gravity, its servo's teacher exactly zero, is given exactly
    no torque"""
    M, J = 150, 29
    decl = Cerebellar([0.0] * M, [1.0] * M, joints=[f"j{i}" for i in range(J)], vor=["yaw", "pitch"])
    o = organ(decl, seed=2); rng = np.random.default_rng(1)
    for t in range(40):
        for s in range(15):
            sf = SubFrame(t, s, rng.uniform(-1.5, 1.5, M), None if (t + s) % 3 == 0 else np.zeros(J), None if t % 2 else np.zeros(2), None if t % 2 else rng.standard_normal(2))
            tau, gain, off = o.sub_step(sf, CEREB["cereb_rate"], CEREB["cereb_vor_rate"])
            assert np.all(tau == 0.0) and (s or (np.all(gain == 0.0) and np.all(off == 0.0))), (t, s)
    assert not o.pc_w.any() and not o.fl_w.any() and int(o.n_limb) == 0 and int(o.n_vor) == 0 and int(o.n_sub) == 600
    for t in range(20):                                                  # it learns something
        for s in range(15):
            o.sub_step(SubFrame(t, s, rng.uniform(-1, 1, M), rng.standard_normal(J), rng.standard_normal(2) * 0.01, rng.standard_normal(2) * 0.1), 0.01, 0.05)
    pc0, fl0 = o.pc_w.clone(), o.fl_w.clone(); x = rng.uniform(-1, 1, M); outs = set()
    for t in range(15):
        for s in range(15):
            tau, gain, off = o.sub_step(SubFrame(t, s, x, np.zeros(J)), 0.01, 0.05)
            outs.add(tau.tobytes()); outs.add(("vor", gain.tobytes(), off.tobytes()) if s == 0 else None)
    assert torch.equal(o.pc_w, pc0) and torch.equal(o.fl_w, fl0) and len(outs - {None}) == 2 and np.any(np.frombuffer(sorted(b for b in outs if isinstance(b, bytes))[0]) != 0)
    w = ArmWorld(); w.m.opt.gravity[:] = 0.0; o2 = organ(arm_cerebellar(), seed=3); w.below = OrganHook(o2)
    for _ in range(30):
        w.apply({"arm": REST})
    assert not np.array(w.teach_log).any() and not np.array(w.tau_log).any() and not o2.pc_w.any() and int(o2.n_sub) == 450
    print("cereb 4: silent without a teacher: 600 sub-steps of any state give exactly no torque, gain or offset and write nothing; a learned",
          "one at a still state and a silent teacher gives one torque to the bit; the limb at rest in no gravity gets exactly none")


def test_it_learns_a_limbs_load():
    """cereb 5 (SIM_DESIGN.md 7.5: feedback-error learning, Kawato and Gomi 1992): the test limb moves through four postures in turn
    (the instrument's driver, 25 ticks each, a cycle of 100 ticks), a 0.5 kg weight added to its hand at tick 400, as the fifth cycle
    begins. Compared posture by posture: with the cerebellum off, the servo's corrective torque at the shoulder over the first posture
    rises with the load and stays there, cycle after cycle; with it on, it rises at the load and falls over the next cycles, to a small
    part of the off level, and every cycle after the load stays well below the off level. Written down, not asserted (C50): the whole
    curve per cycle, the sink of the rested limb after learning, and how far a 10 N push at the hand moves the resting limb, on and off"""
    res, probe = {}, {}
    for on in (False, True):
        w = ArmWorld()
        if on:
            o = organ(arm_cerebellar(), seed=1); w.below = OrganHook(o)
        run_limb(w, 1600, load_at=400)
        x = np.array(w.teach_log)[:, 0]
        res[on] = ([float(x[c * 100:c * 100 + 25].mean()) for c in range(16)], blocks(w.teach_log, 0), blocks(w.teach_log, 1))
        run_limb(w, 40, t0=1600)                                         # to the first posture again, then rest 20 ticks
        q0 = w.q()
        for _ in range(20):
            w.apply({"arm": REST})
        sink = (w.q() - q0) / (20 * TICK_S)
        run_limb(w, 40, t0=1600)
        q1 = w.q()
        for _ in range(7):
            w.apply({"arm": REST}, force=np.array([0.0, 0.0, 10.0]))
        probe[on] = (sink, w.q() - q1)
    (off0, off_s, off_e), (on0, on_s, on_e) = res[False], res[True]
    assert off0[4] > 1.3 * off0[3] and max(off0[4:7]) - min(off0[4:7]) < 0.02 * off0[4], ("off, the first posture per cycle", off0)
    assert on0[4] > 1.5 * on0[3] and on0[6] < 0.5 * on0[4] and on0[5] < on0[4], ("on, the first posture per cycle", on0)
    assert max(on_s[4:]) < 0.5 * min(off_s[4:]) and max(on_e[4:]) < 0.75 * min(off_e[4:]), ("on against off, per cycle", on_s, off_s, on_e, off_e)
    print(f"cereb 5: a 0.5 kg load at tick 400: the shoulder's corrective torque over the first posture of cycles 4-7 (N m), off",
          f"{' '.join(f'{b:.2f}' for b in off0[3:7])}, on {' '.join(f'{b:.2f}' for b in on0[3:7])}; per cycle, off {' '.join(f'{b:.2f}' for b in off_s)};",
          f"on {' '.join(f'{b:.2f}' for b in on_s)}; the elbow's per cycle after the load at most {max(on_e[4:]):.2f} on, at least {min(off_e[4:]):.2f} off.",
          f"Rested 3 s after learning, the limb sinks {probe[True][0].round(4)} rad/s on, {probe[False][0].round(4)} off; a 10 N push for",
          f"1.05 s at rest moves it {probe[True][1].round(3)} rad on, {probe[False][1].round(3)} off")


def test_the_vor_learns_from_slip():
    """cereb 6 (SIM_DESIGN.md 7.5 and 3.7: the flocculus, Ito 1982): the test head turns briskly about both axes. A MAGNIFYING LENS (the
    image moves twice as far for a turn) from tick 200: with the cerebellum off the slip stays where the lens put it; with it on, the gain
    correction climbs toward 1 (the VOR's gain toward 2) and the slip falls to a small part of its first level. A BIASED GYRO (0.2 and
    -0.1 rad/s) from tick 200: off, the slip carries the bias every tick; on, the offset cancels it (-bias x the tick) and the slip falls to
    the gyro's own noise"""
    out = {}
    for label, change in (("lens", dict(mag=2.0)), ("bias", dict(bias=np.array([0.2, -0.1])))):
        for on in (False, True):
            o = organ(Head.cerebellar(), seed=4); h = Head(OrganHook(o) if on else None)
            for t in range(3200):
                if t == 200:
                    for k, v in change.items():
                        setattr(h, k, v)
                h.tick()
            S = np.abs(np.array(h.slip_log))
            out[(label, on)] = ([float(S[i:i + 200].mean()) for i in range(0, 3200, 200)], np.array(h.gain_log), np.array(h.off_log), int(o.n_vor))
    lo, ln = out[("lens", False)][0], out[("lens", True)][0]
    assert max(lo[1:]) - min(lo[1:]) < 0.05 * max(lo[1:]) and lo[1] > 20 * lo[0], lo
    gain = out[("lens", True)][1]
    assert ln[-1] < 0.15 * ln[1] and np.all(gain[-200:].mean(0) > 0.6) and np.all(gain[-200:].mean(0) > gain[1000:1200].mean(0)), (ln, gain[-200:].mean(0))
    bo, bn = out[("bias", False)][0], out[("bias", True)][0]
    offs = out[("bias", True)][2][-200:].mean(0)
    assert max(bo[1:]) - min(bo[1:]) < 0.05 * max(bo[1:]) and bn[-1] < 0.15 * bo[-1] and bn[-1] < 1.5 * bn[0], (bo, bn)
    assert abs(offs[0] + 0.2 * TICK_S) < 0.1 * 0.2 * TICK_S and abs(offs[1] - 0.1 * TICK_S) < 0.1 * 0.1 * TICK_S, offs
    assert out[("lens", True)][3] == 3199, out[("lens", True)][3]
    print(f"cereb 6: under a 2x lens the mean |slip| per 200 ticks (rad a tick) off {lo[1]:.3f} throughout, on {' '.join(f'{b:.3f}' for b in ln[1:])},",
          f"the gain correction {gain[-200:].mean(0).round(2)} at the end (toward 1); a biased gyro: off {bo[1]:.4f}, on {bn[1]:.4f} -> {bn[-1]:.4f}",
          f"(the noise alone {bn[0]:.4f}), the offset {offs.round(4)} (-bias x the tick: {np.array([-0.03, 0.015])})")


class _Recorded(ArmWorld):
    """the test limb, keeping every SubFrame it hands below the tick and every answer"""

    def __init__(self):
        super().__init__(); self.frames = []; self.answers = []

    def sub_tick(self, sf):
        ans = super().sub_tick(sf); self.frames.append(sf); self.answers.append(ans)
        return ans


class _Unhooked(ArmWorld):
    """the test limb calling nothing below the tick, whatever hook it holds"""

    def sub_tick(self, sf):
        return None


def test_through_a_life():
    """cereb 7 (SIM_DESIGN.md 7.5, 8.2): a life whose switch is on, in the test limb's world through the world loop, its arm chosen by its
    born gates (every other learning rate at 0), a 0.5 kg load from tick 100: the world calls the life's hook 15 times a tick, and each
    answer is the organ's law (a twin of the born organ fed the same SubFrames gives the same torques to the bit); the cerebellum learns
    from the servo's torque and its torque reaches the limb. Written down, not asserted: the servo's corrective torque at the shoulder
    after the load, on and off (under the born body's babble, big random steps whose transients the servo meets at its limit, the load
    is a small part of it). NOTHING OF THE TICK READS IT: a life whose switch is on, in a world that calls nothing below the tick, is after
    200 ticks the life with the switch off in every part but its cerebellum (its organs, store, optimizers, random streams, working state
    and constants), so it proposes no act, enters no credit and reads no reward"""
    import copy
    cfg = dict(_LR0, gate_floor=0.5)
    res = {}
    for on in (False, True):
        w = _Recorded(); torch.manual_seed(0); L = born(dict(cfg, cereb=1) if on else dict(cfg), w)
        twin = copy.deepcopy(L.m.cereb) if on else None
        run = WorldLoop(L); acted = 0
        for t in range(400):
            if t == 100:
                w.load(0.5)
            acts = run.step(); acted += int(acts["arm"] != REST)
        res[on] = (np.array(w.teach_log)[100:, 0].mean(), acted)
        if on:
            assert w.below._life() is L and w.below.calls == len(w.frames) == 400 * 15 == int(L.m.cereb.n_sub) and int(L.m.cereb.n_limb) > 5000
            assert [sf.sub for sf in w.frames[:30]] == list(range(15)) * 2 and all(isinstance(a_, SubActs) for a_ in w.answers)
            for sf, ans in zip(w.frames, w.answers):
                tau, _, _ = twin.sub_step(sf, CEREB["cereb_rate"], CEREB["cereb_vor_rate"])
                assert np.array_equal(tau, ans.torque)
            assert all(torch.equal(a_, b_) for a_, b_ in zip(twin.state_dict().values(), L.m.cereb.state_dict().values()))
            assert np.abs(np.array(w.tau_log)).max() > 0.5
        else:
            assert not w.frames and w.below is None
    assert res[True][1] > 100 and res[False][1] > 100, res
    # nothing of the tick reads it
    lives = {}
    for on in (False, True):
        w = _Unhooked(); torch.manual_seed(0); L = born(dict(cfg, live_lr=1e-4, value_lr=1e-3, gate_lr=0.05, cereb=1) if on else dict(cfg, live_lr=1e-4, value_lr=1e-3, gate_lr=0.05), w)
        run = WorldLoop(L)
        for t in range(200):
            if t == 50:
                w.load(0.5)
            run.step()
        lives[on] = (_life_but_cerebellum(L), L)
    assert lives[True][0] == lives[False][0], (lives[True][0], lives[False][0])
    assert int(lives[True][1].m.cereb.n_sub) == 0
    print(f"cereb 7: through a life, the hook called 15 times a tick for 400 ticks, each answer the organ's law to the bit; the shoulder's",
          f"corrective torque after the load under the born body's babble {res[True][0]:.3f} N m on, {res[False][0]:.3f} off (the arm acting on",
          f"{res[True][1]} and {res[False][1]} of 400 ticks);",
          "unhooked, a life with it on is the life with it off in every part but the cerebellum after 200 learning ticks")


def _organ_digest(o, w, last):
    import hashlib
    g = hashlib.sha256()
    for k, v in o.state_dict().items():
        g.update(k.encode()); g.update(v.contiguous().numpy().tobytes())
    g.update(w.save_state()); g.update(np.array(w.teach_log[-last:]).tobytes())
    return g.hexdigest()[:24]


def _life_digest(L, w, last):
    import hashlib
    g = hashlib.sha256()
    for k, v in L.m.state_dict().items():
        g.update(k.encode()); g.update(v.detach().contiguous().numpy().tobytes())
    g.update(w.save_state()); g.update(np.array(w.teach_log[-last:]).tobytes()); g.update(repr(L.page[-60:]).encode())
    return g.hexdigest()[:24]


LIFE_CFG = dict(_LR0, gate_floor=0.5, cereb=1)


def child(mode, tmp):
    """a separate process's part of cereb 8: the organ and the limb uninterrupted (organ-full), or saved halfway (organ-save) and taken up
    by another process (organ-load); a life with its cerebellum born, lived, saved, reloaded and lived on (life); each prints its digest"""
    torch.set_num_threads(1)
    if mode.startswith("organ"):
        w = ArmWorld(); o = organ(arm_cerebellar(), seed=1 if mode != "organ-load" else 7); w.below = OrganHook(o)
        if mode == "organ-full":
            run_limb(w, 600, load_at=200)
        elif mode == "organ-save":
            run_limb(w, 300, load_at=200)
            torch.save(o.state_dict(), os.path.join(tmp, "organ.pt"))
            with open(os.path.join(tmp, "limb.bin"), "wb") as f:
                f.write(w.save_state())
            return
        else:
            o.load_state_dict(torch.load(os.path.join(tmp, "organ.pt")))
            with open(os.path.join(tmp, "limb.bin"), "rb") as f:
                w.load_state(f.read())
            run_limb(w, 300, load_at=200, t0=300)
        print("DIGEST", _organ_digest(o, w, 300), flush=True)
    else:
        w = ArmWorld(); torch.manual_seed(0); L = born(dict(LIFE_CFG), w); run = WorldLoop(L)
        for t in range(120):
            if t == 40:
                w.load(0.5)
            run.step()
        p = os.path.join(tmp, f"life_{os.getpid()}.pt"); L.save(p); blob = w.save_state()
        w2 = ArmWorld(); w2.load_state(blob)
        L2 = Life.load(p, ArmAnatomy(TOK, {}), device="cpu", seed=0, world=w2)
        os.remove(p)
        run2 = WorldLoop(L2)
        for _ in range(60):
            run2.step()
        print("DIGEST", _life_digest(L2, w2, 60), flush=True)


def test_save_round_trip_and_replay():
    """cereb 8 (SIM_DESIGN.md 7.5: it draws no random number after birth, its weights are in the save, the world calls it at fixed
    sub-steps, so a replay is exact): the organ's whole state (the expansion, the readouts, the flocculus's eligibility, the counters) is
    its state dict; the test limb and the organ saved halfway through 600 ticks (a load at 200) and taken up by ANOTHER PROCESS (the
    organ born there from another seed, then loaded) end exactly where one process running all 600 ends (the organ, the limb and the
    teacher's last 300 ticks to the bit), and a second uninterrupted process agrees. A life with its cerebellum saves it among its organs
    (cereb.*, nothing in the save's life dict); reloaded, its cerebellum is the saved one to the bit, saved again at once it writes the
    same, its new world's hook is the new life's; two processes that each bear, live, save, reload and live on such a life end with the
    same digest"""
    tmp = tempfile.mkdtemp(prefix="cereb_")
    try:
        env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
        def run(mode):
            r = subprocess.run([sys.executable, os.path.abspath(__file__), "--child", mode, tmp], capture_output=True, text=True, env=env, timeout=600)
            assert r.returncode == 0, (mode, r.stderr[-2000:])
            out = [l_ for l_ in r.stdout.splitlines() if l_.startswith("DIGEST ")]
            return out[-1].split()[1] if out else None
        full1 = run("organ-full"); run("organ-save"); cont = run("organ-load"); full2 = run("organ-full")
        assert full1 == full2 == cont, (full1, full2, cont)
        life1, life2 = run("life"), run("life")
        assert life1 == life2 and life1 is not None, (life1, life2)
        # in this process: the save holds the organ, the reload gives it back to the bit, and a save at once writes it again
        w = ArmWorld(); torch.manual_seed(0); L = born(dict(LIFE_CFG), w); run_ = WorldLoop(L)
        for t in range(60):
            if t == 20:
                w.load(0.5)
            run_.step()
        p = os.path.join(tmp, "life.pt"); p2 = os.path.join(tmp, "again.pt"); L.save(p)
        blob = torch.load(p, map_location="cpu", weights_only=False)
        keys = [k for k in blob["organs"] if k.startswith("cereb.")]
        assert len(keys) == 12 and not any("cereb" in str(k) for k in blob["life"]) and blob["cfg"]["cereb"] == 1
        w2 = ArmWorld(); w2.load_state(w.save_state())
        L2 = Life.load(p, ArmAnatomy(TOK, {}), device="cpu", seed=0, world=w2)
        assert all(torch.equal(L2.m.cereb.state_dict()[k[6:]], blob["organs"][k]) for k in keys) and int(L2.m.cereb.n_sub) == 60 * 15
        assert w2.below._life() is L2 and w.below._life() is L
        L2.save(p2); again = torch.load(p2, map_location="cpu", weights_only=False)
        assert all(torch.equal(again["organs"][k], blob["organs"][k]) for k in keys)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print(f"cereb 8: saved halfway and taken up by another process, the organ and the limb end where one process ends ({cont});",
          f"a life with its cerebellum reloads it to the bit and writes it again at once; two processes living, saving, reloading and living on agree ({life1})")


def test_its_cost():
    """cereb 9 (SIM_DESIGN.md 7.5's estimate, about 1 ms a tick; C50): the organ at the humanoid's size (4,096 granule units over 150
    mossy numbers, 29 joints, 2 VOR axes), 15 sub-steps a tick, every one taught, the flocculus taught at each tick's first: the wall
    time of the law per tick, one thread, beside whatever else this machine runs (written down; the bound here is only against a runaway)"""
    torch.set_num_threads(1)
    M, J = 150, 29
    decl = Cerebellar([0.0] * M, [1.0] * M, joints=[f"j{i}" for i in range(J)], vor=["yaw", "pitch"])
    o = organ(decl, seed=6); rng = np.random.default_rng(2); x = rng.uniform(-1, 1, M); frames = []
    for t in range(330):
        for s in range(15):
            x = np.clip(x + 0.02 * rng.standard_normal(M), -1.2, 1.2)
            frames.append(SubFrame(t, s, x.copy(), rng.standard_normal(J) * 0.5, rng.standard_normal(2) * 0.01 if s == 0 else None,
                                   rng.standard_normal(2) * 0.1 if s == 0 else None))
    hook = OrganHook(o)
    for sf in frames[:450]:
        hook(sf)
    best = []
    for rep in range(3):
        t0 = time.perf_counter()
        for sf in frames[450 + rep * 1500:450 + (rep + 1) * 1500]:
            hook(sf)
        best.append((time.perf_counter() - t0) / 100 * 1e3)
    ms = min(best)
    assert ms < 20.0, best
    print(f"cereb 9: its cost at the humanoid's size: {ms:.2f} ms a tick at best of three runs of 100 ticks ({', '.join(f'{b:.2f}' for b in best)};",
          f"15 sub-steps, one thread, under this machine's load: {os.getloadavg()[0]:.1f})")


def test_the_estimated_torque_unsettles_it():
    """cereb 10 (C50; the reason the test limb declares no torque): the test limb as in cereb 5 for 6,000 ticks, the 0.5 kg load at
    tick 400. With the joints' state and the efference copy as its mossy numbers (each joint's angle, velocity and target) the servo's
    corrective torque at the shoulder stays a small part of the off level to the end, the weights bounded. With SIM_DESIGN.md 7.5's list,
    which adds each joint's estimated torque (the motor's torque, carrying the servo's correction and the cerebellum's own torque back into
    its input), the readout's gain, raised by a lesson blind to that loop, carries the limb into oscillation at its torque limit: its
    corrective torque ends above the off level. Written down for the lead (the humanoid's declaration); over a whole day,
    tools/cereb_day.py"""
    out = {}
    for kind in (None, "state", "design"):
        w = ArmWorld(mossy=kind or "state"); o = None
        if kind:
            o = organ(arm_cerebellar(kind), seed=1); w.below = OrganHook(o)
        run_limb(w, 6000, load_at=400)
        x = np.array(w.teach_log)[:, 0]
        out[kind] = ([float(x[i:i + 1000].mean()) for i in range(0, 6000, 1000)], 0.0 if o is None else float(o.pc_w.abs().max()))
    off, (st, st_w), (de, de_w) = out[None][0], out["state"], out["design"]
    assert st[-1] < 0.4 * off[-1] and st_w < 5.0, (st, st_w, off)
    assert de[-1] > off[-1] and de_w > 10 * st_w, (de, de_w, off)
    print(f"cereb 10: the shoulder's corrective torque per 1000 ticks (N m): off {' '.join(f'{b:.2f}' for b in off)}; the state and the",
          f"efference copy {' '.join(f'{b:.2f}' for b in st)} (largest weight {st_w:.2f}); 7.5's list with the estimated torque",
          f"{' '.join(f'{b:.2f}' for b in de)} (largest weight {de_w:.1f}): carried into oscillation at the limit")


CEREB_TESTS = [test_absent_for_language, test_built_last_from_its_own_generator, test_the_law_against_numpy, test_silent_without_a_teacher,
               test_it_learns_a_limbs_load, test_the_vor_learns_from_slip, test_through_a_life, test_save_round_trip_and_replay, test_its_cost,
               test_the_estimated_torque_unsettles_it]


if __name__ == "__main__":
    if len(sys.argv) >= 4 and sys.argv[1] == "--child":
        child(sys.argv[2], sys.argv[3])
        sys.exit(0)
    t0 = time.time(); failed = 0
    for t in CEREB_TESTS:
        try:
            t()
        except AssertionError as e:
            failed += 1; print("FAIL", t.__name__, ":", e)
        except Exception as e:
            failed += 1; print("ERROR", t.__name__, ":", type(e).__name__, str(e)[:300])
    print(f"{len(CEREB_TESTS) - failed}/{len(CEREB_TESTS)} passed in {time.time() - t0:.0f}s")
    sys.exit(1 if failed else 0)
