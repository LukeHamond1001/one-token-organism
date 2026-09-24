"""THE SIMULATED WORLD: the stock Unitree G1 in the living room, in lockstep with the body (docs/SIM_DESIGN.md sections 3.3-3.7,
5, 6 and 9; the build plan's W1; the G1 amendment of 2026-09-24). `G1World` implements the core's `SimWorld` (body/core/world.py,
step R9) over MuJoCo: `frame()` shows the world at this tick as the body meets it (no sim time passes, nothing drawn from the
body's random streams), `apply(acts)` lives one tick of 150 ms (75 physics steps of 2 ms) with the body's acts, `pause()` and
`resume()` freeze and free it at night, `save_state()` / `load_state(blob)` give the whole world back exactly, every random stream
included. The world waits for the child: sim time passes only in `apply`.

WHAT THE WORLD DOES WITH AN ACT (the servo law with tone, 3.3). Each joint effector (g1scene.EFFECTORS: the waist, each arm, each
Dex3 hand, each leg; 43 joints) acts by one flat id, its joints' settings as base-5 digits, joint 0 the most significant (the
core's Effector). Setting k of a joint is the step SETTINGS[k] (rad): -big, -small, 0, +small, +big. The act RE-ANCHORS the servo
target: target = the measured angle at the tick's start + the step, clipped to the joint's range, held through the tick. At REST
(the effector's rest id, every joint's setting 2; an effector the acts do not name rests) each target relaxes toward the measured
angle with a time constant of TONE_TAU_TICKS, so a rest holds a posture for a moment and then gives way. The servo is the G1's
own position actuator on each joint with the body's gains: stiffness at the joint's torque limit for SERVO_ERR_AT_LIMIT of error
(kp = limit / 0.25 rad; the Dex3's joints at 0.1 rad) and damping SERVO_DAMP_S x kp; the torque is clamped at the joint's limit (Unitree's own, TORQUE_LIMITS)
x (WEAK_FLOOR + (1 - WEAK_FLOOR) h), weakness when the charge h is empty. No gravity compensation, no balance law: nothing but
the servo acts on the G1. The gains replace the stock file's kp 500 at load (Menagerie's placeholder, its README says "needs
tuning"); the file stays byte for byte as committed.

WHAT THE BODY SENSES (3.4; the channels of the frame, each the raw observation the anatomy's born code will read):
  body         43 joints x [sin, cos of the angle scaled over its range to -pi/2..pi/2, velocity (rad/s), servo effort (the
               joint's actuator torque over its present limit)], in the effectors' order: joint sense (the real motors report
               angle, velocity and torque)
  touch        per touch zone (ZONES: 45, one per G1 link's collision shape, the head and the palm apart from the torso and the
               wrist they are fixed to) [log(1 + F / 1 N), onset], F the zone's summed normal contact force (self-contact
               included) averaged over the tick's 75 steps, onset the rise of the log force since the last tick (the slowly and
               rapidly adapting afferents); a hold of the parent's (a weld on a G1 body) adds its force to that body's zone
  vestibular   per IMU (imu_in_torso, which moves with the head: the vestibule; imu_in_pelvis: the trunk's graviceptors)
               [accelerometer mean (3), peak (3), gyro mean (3), peak (3)] over the tick's 75 samples (the peak per axis the
               signed sample of largest size), with the model's own declared noise and ranges
  charge       [h, the change of h this tick]: the body's own need (the robot's battery)
and, for the reward's pain source and the spinal reflexes (a disclosed exception, 3.4: pain is the contact force on a zone),
  pain         per zone 1 when the tick's largest 10 ms mean (PAIN_WINDOW_STEPS physics steps) of its force exceeds F_PAIN, the
               threshold from the body's declared mass: PAIN_WEIGHTS x its weight from the model file (3 x 337.4 N = 1012 N)
The frame's `face` is the parent's face level (0 until the parent lane's feelings drive it) and its `truth` (object poses, the
forces in newtons, the charge's parts) is for the parent and the instruments only, never the body. The eyes are body/sim/eyes.py
(W3) and the ears the parent lane's (P2); their channels join the frame there.

THE CHARGE (6.3, 5.3): h drains DRAIN_BASE a tick plus DRAIN_EFFORT x the tick's mean of sum(tau^2) / sum(tau_max^2) over the 43
joints (the body's own actuator torque, tau_max the declared limits); a charger (a geom named bottle*, none in the room until the
bottle is built) touching a palm feeds FEED_RATE a tick. h is born full.

THE REFLEXES are the body's (spinal, declared on its effectors: body/sim/reflexes.py computes each from the frame); the world
applies their acts as any act.

FAULTS (A18). MuJoCo resets its state by itself on a bad acceleration and counts it in its warning counters, and a few contacts
can stop it outright (mujoco.FatalError). Every tick checks the counters and that the state is finite; on any fault the world
puts back the state it had when the tick began and raises WorldFault, so the fault never reaches the body.

DETERMINISM. One seed; the world's own stream (the IMU noise) is a numpy PCG64 from SeedSequence(seed, spawn_key=(1,)), saved
with the world. A saved world restored anywhere continues bit for bit (body/tests/test_sim_world.py, the exact replay test)."""
import math
import pickle
import sys
import time
from pathlib import Path

import mujoco
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))
import g1scene as G  # noqa: E402
from body.core.world import Frame, SimWorld  # noqa: E402

# ---------------------------------------------------------------- the disclosed constants (SIM_DESIGN.md section 10)
TICK_S = 0.150                  # a tick: 150 ms of sim time (clock; ours)
STEPS_PER_TICK = 75             # 2 ms physics steps, the model's option timestep (world; ours)
SETTINGS_PER_JOINT = 5          # an act's setting per joint: -big, -small, 0, +small, +big (3.5; ours)
STEP_SMALL, STEP_BIG = 0.09, 0.27                 # rad a tick, the trunk, arms, hands and legs (3.5; ours)
SETTINGS = (-STEP_BIG, -STEP_SMALL, 0.0, STEP_SMALL, STEP_BIG)
SERVO_ERR_AT_LIMIT = 0.25       # rad of error at which a servo reaches its torque limit (3.3; anatomy, ours)
SERVO_ERR_AT_LIMIT_HAND = 0.1   # the Dex3's joints reach theirs at 0.1 rad (3.3; anatomy, ours)
SERVO_DAMP_S = 0.04             # damping = 0.04 s x stiffness (3.3; anatomy, ours)
TONE_TAU_TICKS = 3.0            # at rest the target relaxes to the measured angle with this time constant (3.3; innate, ours)
WEAK_FLOOR = 0.3                # weakness when empty: the limits x (0.3 + 0.7 h) (3.3; anatomy, ours)
PAIN_WEIGHTS = 3.0              # F_pain = 3 x the body's weight from the model file (6.2, A12; innate, ours)
PAIN_WINDOW_STEPS = 5           # a zone's force for pain: the tick's largest 10 ms mean (A12; innate, ours)
TOUCH_UNIT_N = 1.0              # touch's log force is log(1 + F / 1 N) (3.4; anatomy, ours)
DRAIN_BASE, DRAIN_EFFORT = 4e-5, 2e-3             # the charge's drain a tick (6.3; world, the owner's default)
FEED_RATE = 0.01                # a charger touching a palm, a tick (5.3; world, the owner's default)
H_BIRTH = 1.0                   # born full (ours)
WORLD_STREAM = 1                # the world's random stream: SeedSequence(seed, spawn_key=(1,)) (ours)
CHARGER_PREFIX = "bottle"       # a geom named bottle* feeds through a palm (5.3; the bottle itself is W3's)

# THE G1'S TORQUE LIMITS (N m), per joint: Unitree's own robot description of this model's revision, the <limit effort> of
# unitree_ros robots/g1_description/g1_29dof_with_hand_rev_1_0.urdf (read 2026-09-24). The Menagerie file (derived from Unitree's
# MJCF) declares the same for 37 of the 43 joints; for the ankles' pitch and roll and the waist's roll and pitch it declares 50,
# where Unitree's URDF says 35: Unitree's value is taken (set at load; the file is unchanged). Unitree's product page gives the
# knee's maximum as 120 N m for the G1 EDU (90 for the standard G1), where the URDF says 139: flagged, the URDF kept (one source
# for every joint).
TORQUE_LIMITS = dict(
    hip_pitch=88.0, hip_roll=139.0, hip_yaw=88.0, knee=139.0, ankle_pitch=35.0, ankle_roll=35.0,
    waist_yaw=88.0, waist_roll=35.0, waist_pitch=35.0,
    shoulder_pitch=25.0, shoulder_roll=25.0, shoulder_yaw=25.0, elbow=25.0, wrist_roll=25.0, wrist_pitch=5.0, wrist_yaw=5.0,
    hand_thumb_0=2.45, hand_thumb_1=1.4, hand_thumb_2=1.4, hand_index_0=1.4, hand_index_1=1.4, hand_middle_0=1.4, hand_middle_1=1.4)


def torque_limit(joint):
    """the declared limit (N m) of a G1 joint by its full name"""
    base = joint[:-len("_joint")]
    for side in ("left_", "right_"):
        if base.startswith(side):
            base = base[len(side):]
    return TORQUE_LIMITS[base]


# ---------------------------------------------------------------- the effectors
EFFECTOR_NAMES = tuple(n for n, _ in G.EFFECTORS)
JOINTS = tuple(j for _, js in G.EFFECTORS for j in js)          # the 43 joints in the effectors' order (the body channel's)


def rest_id(n_joints):
    """an effector's rest: every joint at setting 2 (no step)"""
    return (SETTINGS_PER_JOINT ** n_joints - 1) // 2


def act_digits(act, n_joints):
    """a flat act's settings, joint 0 first (the most significant base-5 digit)"""
    a = int(act)
    if not 0 <= a < SETTINGS_PER_JOINT ** n_joints:
        raise ValueError(f"an act of {n_joints} joints is in [0, {SETTINGS_PER_JOINT ** n_joints}); given {act}")
    out = [0] * n_joints
    for j in range(n_joints - 1, -1, -1):
        out[j] = a % SETTINGS_PER_JOINT; a //= SETTINGS_PER_JOINT
    return out


def act_flat(digits):
    """the flat act of a joint-by-joint list of settings"""
    a = 0
    for k in digits:
        if not 0 <= int(k) < SETTINGS_PER_JOINT:
            raise ValueError(f"a setting is in [0, {SETTINGS_PER_JOINT}); given {k}")
        a = a * SETTINGS_PER_JOINT + int(k)
    return a


EFFECTOR_FACTORS = {n: [SETTINGS_PER_JOINT] * len(js) for n, js in G.EFFECTORS}
EFFECTOR_REST = {n: rest_id(len(js)) for n, js in G.EFFECTORS}


# ---------------------------------------------------------------- the touch zones
def touch_zones(m, g1_set):
    """The G1's touch zones and each geom's zone: one zone per G1 link's collision shape (the four sole spheres of a foot one
    zone, its ankle_roll link), the head's and the palm's shapes their own zones apart from the torso and the wrist they are fixed
    to (the URDF's head_link and hand_palm_link). Returns (zone names in model order, the zone index of every geom or -1, each G1
    body's own zone or -1)."""
    names, zone_of_geom = [], np.full(m.ngeom, -1, dtype=np.int64)
    for g in range(m.ngeom):
        b = int(m.geom_bodyid[g])
        if b not in g1_set or not (m.geom_contype[g] or m.geom_conaffinity[g]):
            continue
        nm = m.body(b).name
        if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH:
            mesh = m.mesh(m.geom_dataid[g]).name
            if mesh == "head_link" or mesh.endswith("hand_palm_link"):
                nm = mesh
        nm = nm[:-len("_link")] if nm.endswith("_link") else nm
        if nm not in names:
            names.append(nm)
        zone_of_geom[g] = names.index(nm)
    body_zone = np.full(m.nbody, -1, dtype=np.int64)
    for b in g1_set:
        nm = m.body(b).name
        nm = nm[:-len("_link")] if nm.endswith("_link") else nm
        if nm in names:
            body_zone[b] = names.index(nm)
    return names, zone_of_geom, body_zone


def zone_groups(zones):
    """The zones in the 8 groups the core's event lines read (the amygdala spec, R7a: head, trunk, each arm, each hand, each
    leg), {group: [zone index]}."""
    hand_parts = ("hand_palm", "hand_thumb", "hand_index", "hand_middle")
    arm_parts = ("shoulder", "elbow", "wrist")
    leg_parts = ("hip", "knee", "ankle")
    out = {"head": [], "trunk": [], "arm_l": [], "arm_r": [], "hand_l": [], "hand_r": [], "leg_l": [], "leg_r": []}
    for i, z in enumerate(zones):
        side = "l" if z.startswith("left_") else "r" if z.startswith("right_") else None
        part = z.split("_", 1)[1] if side else z
        if z == "head":
            out["head"].append(i)
        elif side is None:
            out["trunk"].append(i)
        elif part.startswith(hand_parts):
            out[f"hand_{side}"].append(i)
        elif part.startswith(arm_parts):
            out[f"arm_{side}"].append(i)
        elif part.startswith(leg_parts):
            out[f"leg_{side}"].append(i)
        else:
            raise ValueError(f"touch zone {z!r} belongs to no group")
    return out


MUJOCO_MESSAGES = []            # MuJoCo's warning texts this process, the latest last (kept here, never printed or logged to a file)


def _mujoco_warning(msg):
    MUJOCO_MESSAGES.append(str(msg).strip())
    del MUJOCO_MESSAGES[:-64]


def _catch_mujoco_warnings():
    """MuJoCo's warnings reach the world, not the terminal or a MUJOCO_LOG.TXT in the working folder (the world's counters
    decide a fault; the text goes into its reason). Set once per process, unless a handler is already set."""
    if mujoco.get_mju_user_warning() is None:
        mujoco.set_mju_user_warning(_mujoco_warning)


class WorldFault(RuntimeError):
    """a tick the physics could not live (a MuJoCo reset, a stop, a state not finite): the world stands where the tick began"""

    def __init__(self, tick, reason):
        super().__init__(f"world fault at tick {tick}: {reason}")
        self.tick, self.reason = int(tick), str(reason)


# the model's fields the world (and the parent's drivers) change at run time: saved with the world so a restore is exact
MUTABLE_MODEL_FIELDS = ("jnt_actfrcrange", "eq_data", "geom_pos", "geom_quat", "geom_size", "geom_rgba", "geom_contype",
                        "geom_conaffinity", "light_active", "light_castshadow", "light_diffuse", "light_ambient", "light_specular",
                        "mat_rgba", "mat_emission")
STATE_SPEC = mujoco.mjtState.mjSTATE_INTEGRATION


class G1World(SimWorld):
    """THE G1 IN THE LIVING ROOM, lockstep (see the module's doc). `seed` is the body's one seed (the world's own stream is derived
    from it); `extra(spec)` adds an instrument's rig to the scene before it compiles (tests only). Born at construction: the G1 on
    its back on the mat, settled, the charge full, tick 0."""

    def __init__(self, seed=1, extra=None, xml=G.XML):
        _catch_mujoco_warnings()
        self.scene = G.Scene(xml, extra)
        m, d = self.m, self.d = self.scene.m, self.scene.d
        self.seed = int(seed)
        self.rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence(self.seed, spawn_key=(WORLD_STREAM,))))
        if abs(m.opt.timestep * STEPS_PER_TICK - TICK_S) > 1e-12:
            raise ValueError(f"the model's step {m.opt.timestep} s x {STEPS_PER_TICK} is not a {TICK_S} s tick")
        # the joints, their actuators, dofs, ranges and limits, in the effectors' order
        self.jid = np.array([m.joint(j).id for j in JOINTS])
        self.qadr = m.jnt_qposadr[self.jid].copy()
        self.dof = m.jnt_dofadr[self.jid].copy()
        self.aid = np.array([m.actuator(j).id for j in JOINTS])           # the G1's actuators are named for their joints
        if not all(int(m.actuator_trnid[a, 0]) == int(j) for a, j in zip(self.aid, self.jid)):
            raise ValueError("an actuator does not drive the joint it is named for")
        self.lo, self.hi = m.jnt_range[self.jid, 0].copy(), m.jnt_range[self.jid, 1].copy()
        self.tau_max = np.array([torque_limit(j) for j in JOINTS])
        self.tau_max_sq = float(np.sum(self.tau_max ** 2))
        self.eff_slices = {}
        k = 0
        for n, js in G.EFFECTORS:
            self.eff_slices[n] = slice(k, k + len(js)); k += len(js)
        self._set_servo_law()
        # the touch zones, the pain threshold, the chargers, the parent's holds on the G1
        self.zones, self.zone_of_geom, self.body_zone = touch_zones(m, self.scene.g1_set)
        self.nz = len(self.zones)
        self.groups = zone_groups(self.zones)
        self.body_mass = float(m.body_subtreemass[m.body("pelvis").id])
        self.f_pain = PAIN_WEIGHTS * self.body_mass * float(np.linalg.norm(m.opt.gravity))
        self.palm_zones = [self.zones.index(f"{s}_hand_palm") for s in ("left", "right")]
        self.chargers = np.array([g for g in range(m.ngeom) if (m.geom(g).name or "").startswith(CHARGER_PREFIX)], dtype=np.int64)
        self.g1_welds = [e for e in range(m.neq) if m.eq_type[e] == mujoco.mjtEq.mjEQ_WELD and int(m.eq_obj2id[e]) in self.scene.g1_set]
        self.weld_zone = np.array([self.body_zone[int(m.eq_obj2id[e])] for e in self.g1_welds], dtype=np.int64)
        # the IMUs: the model's four sensors (gyro and accelerometer on each site), their declared noise and ranges
        names = [m.sensor(i).name for i in range(m.nsensor)]
        self.imu_order = [names.index(n) for n in ("imu-torso-linear-acceleration", "imu-torso-angular-velocity",
                                                   "imu-pelvis-linear-acceleration", "imu-pelvis-angular-velocity")]
        self.imu_adr = np.concatenate([np.arange(m.sensor_adr[i], m.sensor_adr[i] + m.sensor_dim[i]) for i in self.imu_order])
        self.imu_noise = np.repeat([float(m.sensor_noise[i]) for i in self.imu_order], 3)
        self.imu_cut = np.repeat([float(m.sensor_cutoff[i]) if m.sensor_cutoff[i] > 0 else np.inf for i in self.imu_order], 3)
        self.alpha = 1.0 - math.exp(-m.opt.timestep / (TONE_TAU_TICKS * TICK_S))
        # birth
        self.tick = 0
        self.h = H_BIRTH; self.dh = 0.0
        self.paused = False
        self._apply_weakness()
        self.scene.birth()
        self._last_acts = {}
        self._sense_birth()
        self.timing = {"ticks": 0, "apply_s": 0.0, "physics_s": 0.0}      # wall clock, an instrument (never saved, never sensed)

    # ---------------------------------------------------------------- the servo law
    def _set_servo_law(self):
        """each G1 actuator a position servo at the body's gains: kp = limit / SERVO_ERR_AT_LIMIT (the Dex3's joints: / 0.1 rad),
        kv = SERVO_DAMP_S x kp"""
        m = self.m
        err = np.array([SERVO_ERR_AT_LIMIT_HAND if "_hand_" in j else SERVO_ERR_AT_LIMIT for j in JOINTS])
        kp = self.tau_max / err
        kv = SERVO_DAMP_S * kp
        for a, p, v in zip(self.aid, kp, kv):
            m.actuator_gainprm[a, :] = 0.0; m.actuator_gainprm[a, 0] = p
            m.actuator_biasprm[a, :] = 0.0; m.actuator_biasprm[a, 1] = -p; m.actuator_biasprm[a, 2] = -v
        self.kp, self.kv = kp, kv

    def limits_now(self):
        """the torque limits this tick: the declared limits x (WEAK_FLOOR + (1 - WEAK_FLOOR) h)"""
        return self.tau_max * (WEAK_FLOOR + (1.0 - WEAK_FLOOR) * self.h)

    def _apply_weakness(self):
        lim = self.limits_now()
        self.m.jnt_actfrcrange[self.jid, 0] = -lim
        self.m.jnt_actfrcrange[self.jid, 1] = lim
        self.m.jnt_actfrclimited[self.jid] = 1

    # ---------------------------------------------------------------- the world interface
    def frame(self):
        """the world at this tick as the body meets it (no sim time passes; nothing drawn)"""
        if self.paused:
            raise RuntimeError("G1World: a frame taken while the world is paused (the night)")
        d, s = self.d, self._sensed
        q = d.qpos[self.qadr]
        mid, half = (self.hi + self.lo) / 2, (self.hi - self.lo) / 2
        ang = (q - mid) / half * (math.pi / 2)
        effort = d.qfrc_actuator[self.dof] / self.m.jnt_actfrcrange[self.jid, 1]     # over the limit the torque was clamped at
        body = np.stack([np.sin(ang), np.cos(ang), d.qvel[self.dof], effort], axis=1).reshape(-1)
        touch = np.stack([s["touch_log"], s["touch_onset"]], axis=1).reshape(-1)
        obs = {"body": body, "touch": touch, "vestibular": s["vestibular"].copy(), "charge": np.array([self.h, self.dh]),
               "pain": (s["pain_force"] > self.f_pain).astype(np.float64)}
        return Frame(self.tick, obs, 0.0, self._truth())

    def apply(self, acts):
        """one tick (150 ms, 75 steps of 2 ms) with the body's acts {effector name: flat act}; names the world does not move (the
        voice, the gaze until the eyes are built) are the other lanes'"""
        if self.paused:
            raise RuntimeError("G1World: the world moved while paused (the night)")
        m, d = self.m, self.d
        t_apply = time.perf_counter(); t_phys = 0.0
        steps = {}                                                      # every act read before anything moves (a bad act moves nothing)
        for name, js in G.EFFECTORS:
            a = acts.get(name) if acts else None
            if a is not None and int(a) != EFFECTOR_REST[name]:
                steps[name] = np.array([SETTINGS[k] for k in act_digits(a, len(js))])
        start = self._capture()
        warn0 = [int(d.warning[i].number) for i in range(int(mujoco.mjtWarning.mjNWARNING))]
        self._apply_weakness()
        rest_idx = []
        q = d.qpos[self.qadr]
        for name, js in G.EFFECTORS:
            sl = self.eff_slices[name]
            if name not in steps:
                rest_idx.extend(range(sl.start, sl.stop))
                continue
            d.ctrl[self.aid[sl]] = np.clip(q[sl] + steps[name], self.lo[sl], self.hi[sl])
        self._last_acts = {k: int(v) for k, v in (acts or {}).items() if k in EFFECTOR_REST}
        rest_a = self.aid[rest_idx] if rest_idx else None
        rest_q = self.qadr[rest_idx] if rest_idx else None
        n, nz = STEPS_PER_TICK, self.nz
        F = np.zeros((n, nz)); imu = np.zeros((n, 12)); eff = np.zeros(n); fed = False
        zg, alpha = self.zone_of_geom, self.alpha
        try:
            for s in range(n):
                if rest_a is not None:
                    d.ctrl[rest_a] += alpha * (d.qpos[rest_q] - d.ctrl[rest_a])
                t0 = time.perf_counter()
                mujoco.mj_step(m, d)
                t_phys += time.perf_counter() - t0
                F[s] = self._zone_forces()
                imu[s] = d.sensordata[self.imu_adr]
                tq = d.qfrc_actuator[self.dof]
                eff[s] = float(tq @ tq)
                if self.chargers.size and not fed:
                    fed = self._palm_on_charger()
        except mujoco.FatalError as e:
            self._restore(start)
            raise WorldFault(self.tick, f"MuJoCo stopped: {e}")
        warned = {mujoco.mjtWarning(i).name: int(d.warning[i].number) - warn0[i]
                  for i in range(int(mujoco.mjtWarning.mjNWARNING)) if int(d.warning[i].number) != warn0[i]}
        if warned or not (np.isfinite(d.qpos).all() and np.isfinite(d.qvel).all()):
            self._restore(start)
            said = f" ({MUJOCO_MESSAGES[-1]})" if MUJOCO_MESSAGES else ""
            raise WorldFault(self.tick, f"MuJoCo's warnings {warned}{said}" if warned else "a state not finite")
        # the tick's senses
        self._sense_tick(F, imu)
        drain = DRAIN_BASE + DRAIN_EFFORT * float(eff.mean()) / self.tau_max_sq
        h0 = self.h
        self.h = float(min(1.0, max(0.0, self.h - drain + (FEED_RATE if fed else 0.0))))
        self.dh = self.h - h0
        self._drain = drain; self._fed = fed
        self.tick += 1
        mujoco.mj_forward(m, d)
        self.timing["ticks"] += 1; self.timing["physics_s"] += t_phys; self.timing["apply_s"] += time.perf_counter() - t_apply

    def pause(self):
        """the night: the world freezes exactly where it is"""
        self.paused = True

    def resume(self):
        """the morning: on from the same state"""
        self.paused = False

    def save_state(self):
        """the whole world as bytes: the physics, the model's run-time fields, the charge, the senses' carry and the world's random
        stream"""
        return pickle.dumps(self._capture(), protocol=4)

    def load_state(self, blob):
        """the whole world back exactly from save_state's bytes"""
        st = pickle.loads(bytes(blob))
        if st.get("version") != 1 or st.get("nstate") != mujoco.mj_stateSize(self.m, STATE_SPEC):
            raise ValueError("G1World.load_state: not a save of this world")
        self._restore(st)

    # ---------------------------------------------------------------- the senses
    def _zone_forces(self):
        """this step's summed normal force on each touch zone (N): every contact touching a G1 zone (self-contact counts on both
        zones), plus each of the parent's active holds on a G1 body on that body's zone"""
        d, zg, nz = self.d, self.zone_of_geom, self.nz
        out = np.zeros(nz)
        nc = d.ncon
        if nc:
            con = d.contact
            adr = con.efc_address
            ok = adr >= 0
            fn = np.where(ok, d.efc_force[np.where(ok, adr, 0)], 0.0)     # the elliptic cone's first row: the normal force
            geom = con.geom
            for col in (0, 1):
                z = zg[geom[:, col]]
                sel = z >= 0
                if sel.any():
                    out += np.bincount(z[sel], weights=fn[sel], minlength=nz)
        if self.g1_welds:
            act = d.eq_active[self.g1_welds]
            if act.any():
                eq = mujoco.mjtConstraint.mjCNSTR_EQUALITY
                for e, z, on in zip(self.g1_welds, self.weld_zone, act):
                    if on and z >= 0:
                        rows = np.nonzero((d.efc_type[:d.nefc] == eq) & (d.efc_id[:d.nefc] == e))[0][:3]
                        out[z] += float(np.linalg.norm(d.efc_force[rows]))
        return out

    def _palm_on_charger(self):
        d = self.d
        if not d.ncon:
            return False
        geom = d.contact.geom
        zg = self.zone_of_geom
        for a, b in geom:
            if (a in self.chargers and zg[b] in self.palm_zones) or (b in self.chargers and zg[a] in self.palm_zones):
                return True
        return False

    def _sense_tick(self, F, imu):
        """the tick's aggregates: touch (the mean force's log, its onset), pain's filtered force, the IMUs with their noise"""
        s = self._sensed
        lf = np.log1p(F.mean(axis=0) / TOUCH_UNIT_N)
        onset = np.maximum(0.0, lf - s["touch_log"])
        P = np.vstack([s["carry"], F])
        c = np.vstack([np.zeros((1, self.nz)), np.cumsum(P, axis=0)])
        w = PAIN_WINDOW_STEPS
        win = (c[w:] - c[:-w]) / w
        pain_force = win[-len(F):].max(axis=0)
        vest = self._vestibular(imu)
        self._sensed = {"touch_log": lf, "touch_onset": onset, "touch_force": F.mean(axis=0), "pain_force": pain_force,
                        "carry": F[-(w - 1):].copy(), "vestibular": vest, "peak_force": F.max(axis=0)}

    def _vestibular(self, imu):
        x = imu + self.rng.standard_normal(imu.shape) * self.imu_noise
        x = np.clip(x, -self.imu_cut, self.imu_cut)
        mean = x.mean(axis=0)
        peak = x[np.abs(x).argmax(axis=0), np.arange(x.shape[1])]
        out = []
        for k in range(4):
            sl = slice(3 * k, 3 * k + 3)
            out.append((mean[sl], peak[sl]))
        # per IMU: accelerometer mean and peak, gyro mean and peak (torso, then pelvis)
        return np.concatenate([out[0][0], out[0][1], out[1][0], out[1][1], out[2][0], out[2][1], out[3][0], out[3][1]])

    def _sense_birth(self):
        """the senses at birth, before any tick: one sample of the born state stands for the tick (its mean and its peak)"""
        mujoco.mj_forward(self.m, self.d)
        F = self._zone_forces()[None, :]
        imu = self.d.sensordata[self.imu_adr][None, :]
        lf = np.log1p(F[0] / TOUCH_UNIT_N)
        self._sensed = {"touch_log": lf, "touch_onset": np.zeros(self.nz), "touch_force": F[0].copy(), "pain_force": F[0].copy(),
                        "carry": np.repeat(F, PAIN_WINDOW_STEPS - 1, axis=0), "vestibular": self._vestibular(imu),
                        "peak_force": F[0].copy()}
        self._drain = 0.0; self._fed = False

    def _truth(self):
        """world truth for the parent and the instruments (never the body)"""
        m, d = self.m, self.d
        toys = {m.body(b).name[4:]: d.xpos[b].copy() for b in range(m.nbody) if m.body(b).name.startswith("toy_")}
        s = self._sensed
        return {"time": float(d.time), "pelvis": d.qpos[0:7].copy(), "torso": d.xpos[m.body("torso_link").id].copy(),
                "toys": toys, "touch_N": s["touch_force"].copy(), "pain_N": s["pain_force"].copy(), "peak_N": s["peak_force"].copy(),
                "f_pain": self.f_pain, "drain": self._drain, "fed": self._fed, "ncon": int(d.ncon), "acts": dict(self._last_acts)}

    # ---------------------------------------------------------------- the state
    def _capture(self):
        m, d = self.m, self.d
        phys = np.zeros(mujoco.mj_stateSize(m, STATE_SPEC))
        mujoco.mj_getState(m, d, phys, STATE_SPEC)
        return {"version": 1, "nstate": int(phys.size), "physics": phys,
                "model": {f: getattr(m, f).copy() for f in MUTABLE_MODEL_FIELDS},
                "tick": self.tick, "h": self.h, "dh": self.dh, "paused": self.paused, "seed": self.seed,
                "sensed": {k: v.copy() for k, v in self._sensed.items()}, "drain": self._drain, "fed": self._fed,
                "last_acts": dict(self._last_acts), "rng": self.rng.bit_generator.state,
                "warnings": np.array([int(d.warning[i].number) for i in range(int(mujoco.mjtWarning.mjNWARNING))])}

    def _restore(self, st):
        m, d = self.m, self.d
        for f, v in st["model"].items():
            getattr(m, f)[...] = v
        mujoco.mj_setState(m, d, st["physics"], STATE_SPEC)
        for i, n in enumerate(st["warnings"]):
            d.warning[i].number = int(n)
        self.tick, self.h, self.dh, self.paused = int(st["tick"]), float(st["h"]), float(st["dh"]), bool(st["paused"])
        self.seed = int(st["seed"])
        self._sensed = {k: v.copy() for k, v in st["sensed"].items()}
        self._drain, self._fed = st["drain"], st["fed"]
        self._last_acts = dict(st["last_acts"])
        self.rng.bit_generator.state = st["rng"]
        mujoco.mj_forward(m, d)
