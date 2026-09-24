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
(kp = limit / 0.25 rad; the Dex3's joints at 0.1 rad) and damping SERVO_DAMP_S x kp; the torque is clamped at the joint's limit (the
model's own, 3.2) x (WEAK_FLOOR + (1 - WEAK_FLOOR) h), weakness when the charge h is empty. No gravity compensation, no balance law: nothing but
the servo acts on the G1. The gains replace the stock file's kp 500 at load (Menagerie's placeholder, its README says "needs
tuning"); the file stays byte for byte as committed.

THE GAZE (3.4, 3.5, 3.7; W3). The G1's eyes do not turn: a software fovea, a 32 px window in each camera's image (body/sim/eyes.py),
sits where the gaze state puts it (yaw and pitch, tangent angles in the image; vergence, the two windows' yaw apart). The gaze
effector (first of the effectors) steps it by +-0.07 / +-0.2 rad (vergence +-0.03 / +-0.1) at the tick's start, held in reach
(each window inside its image, vergence up to the 25 cm near point, A23); a rest holds it (nothing pulls it back to the centre).
THE VOR counter-turns it through the tick by the torso gyro's samples (the same noisy afferent the vestibular channel reads), gain
1, so a thing fixated stays fixated while the trunk turns: its slow phase. Its QUICK PHASE (A23): when the counter-turn would carry
the window past its reach, the window jumps back, in the direction of the trunk's turn, by half that axis's reach, within the
tick (the truth's vor_quick counts them, as reflex; C33 reports them). The gaze's own acts add on top.

WHAT THE BODY SENSES (3.4; the channels of the frame, each the raw observation the anatomy's born code will read):
  body         43 joints x [sin, cos of the angle scaled over its range to -pi/2..pi/2, velocity (rad/s), servo effort (the
               joint's actuator torque over the limit it was clamped at)], in the effectors' order (joint sense: the real motors
               report angle, velocity and torque); then the gaze's state and its velocity (6): BODY_SIZE 178 (the tract's 21
               numbers are the voice lane's to append)
  touch        per touch zone (ZONES: 45, one per G1 link's collision shape, the head and the palm apart from the torso and the
               wrist they are fixed to) [log(1 + F / 1 N), onset], F the zone's summed normal contact force averaged over the
               tick's 75 steps (self-contact included: A12's blind spots are the pairs pressing at rest, `rest_blind`, none for
               the G1 as born), onset the rise of the log force since the last tick (the slowly and rapidly adapting
               afferents); a hold of the parent's (a weld on a G1 body) adds its force to that body's zone
  vestibular   per IMU (imu_in_torso, which moves with the head: the vestibule; imu_in_pelvis: the trunk's graviceptors)
               [accelerometer mean (3), peak (3), gyro mean (3), peak (3)] over the tick's 75 samples (the peak per axis the
               signed sample of largest size), with the model's own declared noise and ranges
  charge       [h, the change of h this tick]: the body's own need (the robot's battery)
and, for the reward's pain source and the spinal reflexes (a disclosed exception, 3.4: pain is the contact force on a zone),
  pain         per zone 1 when the tick's largest 10 ms mean (PAIN_WINDOW_STEPS physics steps) of its force exceeds F_PAIN, the
               threshold from the body's declared mass: PAIN_WEIGHTS x its weight from the model file (3 x 337.4 N = 1012 N). Pain
               alone, no place on the link: the newborn's withdrawal is generalized over the limb (body/sim/reflexes.py), so no
               approved part reads where on a link it hurts (the W1 fix 2's skin-site afferent, `pain_site`, is gone with the
               local sign it served)
  eye_p, eye_f the eyes' retina codes (2 x 168, 2 x 384), when eyes are attached (body/sim/eyes.py, W3), with the born face
               template's two readings from the same pixels: face_fovea (1: the event line "a face in the fovea", either eye) and
               face_periph (orienting's cue: 1 and where the best match lies from the window, or 0s). A1's face test is world truth
               (the reward carrier's gate, 3.4's disclosed exception) and stays in the truth, never in the body's channels
The frame's `face` is the parent's face level (0 until the parent lane's feelings drive it) and its `truth` (object poses, the
forces in newtons, the charge's parts, the eyes' images) is for the parent and the instruments only, never the body. The ears are
the parent lane's (P2); their channel joins the frame there.

THE CHARGE (6.3, 5.3): h drains DRAIN_BASE a tick plus DRAIN_EFFORT x the tick's mean of sum(tau^2) / sum(tau_max^2) over the 43
joints (the body's own actuator torque, tau_max the declared limits); a charger (a geom named bottle*, none in the room until the
bottle is built) touching a palm feeds FEED_RATE a tick. h is born full.

THE REFLEXES are the body's (body/sim/reflexes.py). The withdrawal is declared on each limb's effector (the core's hook: it takes
the limb's tick) and its act reaches the world as any act. The palmar grasp is the spinal cord's: `apply` sums it into each hand's
own act before anything moves (the truth's `spinal` logs it), so the hand's own act that tick can open it (A11). `spinal=False`
switches it off (an instrument's switch). The grasp fires on anything pressing the palm, its own fingers included (a fist closed
on nothing keeps itself closed until the hand's own act opens it, as newborns' hands are fisted); the truth's `palm_own_N` (the
palm's tick-mean force from its own hand's links) lets W4 count those fists (the W1 verifier's eighth finding).

FAULTS (A18). The scene disables MuJoCo's auto-reset (`<flag autoreset="disable"/>`, checked at load), so a bad state is never
silently replaced by the start pose; MuJoCo still counts it in its warning counters, and a few contacts can stop it outright
(mujoco.FatalError, caught in every step and in the tick's closing forward pass). Every tick checks the counters and the end state
under MuJoCo's own bound (positions, velocities and accelerations finite and within mjMAXVAL: the last step's result, which MuJoCo
itself would check only at the next tick's first step); on any fault the world puts back the state it had when the tick began and
raises WorldFault, so the fault never reaches the body.

DETERMINISM. One seed; the world's own stream (the IMU noise) is a numpy PCG64 from SeedSequence(seed, spawn_key=(1,)), saved
with the world, as are the gaze and its velocity. A saved world restored anywhere continues bit for bit (body/tests/test_sim_world.py, the exact replay test)."""
import json
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

R = None                        # body/sim/reflexes.py, the body's spinal cord: bound at the first world's birth (it imports this module)

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

# THE GAZE: the software fovea's effector (3.4, 3.5; W3). The G1 has no eyes that turn: a 32 x 32 px window inside each camera's
# native image is its fovea, placed by a gaze state (yaw, pitch: tangent angles in the image, right and up positive; vergence: the
# two windows' yaw apart, positive converging), moved by the gaze's acts (Hering's law: 3 commands move both windows) and counter-
# turned by the VOR. The window stays inside its image: yaw within +-38.1 deg and pitch +-20.3 deg of the axis.
GAZE_NAME, GAZE_JOINTS = "gaze", ("yaw", "pitch", "vergence")
GAZE_SMALL, GAZE_BIG = 0.07, 0.20                  # rad a tick, yaw and pitch: +-4 / +-11.5 deg (3.5; anatomy, ours)
VERG_SMALL, VERG_BIG = 0.03, 0.10                  # rad a tick, vergence: +-1.7 / +-5.7 deg (3.5; anatomy, ours)
GAZE_SETTINGS = ((-GAZE_BIG, -GAZE_SMALL, 0.0, GAZE_SMALL, GAZE_BIG),) * 2 + ((-VERG_BIG, -VERG_SMALL, 0.0, VERG_SMALL, VERG_BIG),)
NEAR_POINT_M = 0.25             # the nearest thing both windows can fixate: vergence 0 to 2 atan(0.025 / 0.25) = 11.4 deg, the
                                # nearest her face comes (A3); anything nearer is seen double, as inside an infant's near point
                                # (A23; anatomy, ours)
VOR_GAIN = 1.0                  # the window counter-turns by the torso gyro's rotation, gain 1 (3.7; innate, ours)
FOVEA_PX = 32                   # the fovea window, px of the native image (about 21 deg; 3.4; anatomy, ours)

# THE G1'S TORQUE LIMITS (N m) are the model's own (SIM_DESIGN.md 3.2, 10 and A21): each joint's actuatorfrcrange in the stock
# Menagerie file, read at load and never changed there (weakness scales them each tick, the one change A21 allows). Unitree's own
# sources agree neither with it nor with each other (the W1 verifier's sixth finding; each read again 2026-09-24):
#   - Unitree's URDF of this revision (unitree_ros robots/g1_description/g1_29dof_with_hand_rev_1_0.urdf, <limit effort>) gives
#     35 for the ankles' pitch and roll and the waist's roll and pitch, where Menagerie gives 50; its hip roll is Menagerie's 139;
#   - Unitree's MJCF (unitree_mujoco unitree_robots/g1/g1_29dof.xml, its motors' ctrlrange) gives those four Menagerie's 50, but
#     the hip roll 88, where Menagerie and the URDF give 139;
#   - Unitree's G1 page gives the knee 90 N m (the G1) or 120 (the EDU), where all three files give 139.
# So these are Menagerie's limits, not "the real motors' limits as Unitree publishes them" (3.2's wording: flagged for the
# design); the design takes the model's (A21), and the robot's own are a question for before the robot (C37's kind).
# body/tests/test_sim_world.py checks the limits read against 3.2's table and against the stock file.


# ---------------------------------------------------------------- the effectors
EFFECTOR_NAMES = (GAZE_NAME,) + tuple(n for n, _ in G.EFFECTORS)   # the gaze first, then the joint effectors (3.5's order)
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


EFFECTOR_FACTORS = {GAZE_NAME: [SETTINGS_PER_JOINT] * len(GAZE_JOINTS), **{n: [SETTINGS_PER_JOINT] * len(js) for n, js in G.EFFECTORS}}
EFFECTOR_REST = {n: rest_id(len(f)) for n, f in EFFECTOR_FACTORS.items()}
BODY_SIZE = 4 * len(JOINTS) + 6  # the body channel: 43 joints x 4, then the gaze's state and its velocity


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


def rest_blind(m, d, g1_set):
    """THE SKIN'S BLIND SPOTS (A12, exactly as the design writes it): MuJoCo already ignores the contacts between a link and the
    link it hinges on; any OTHER pair of the G1's links that presses into itself AT REST is listed in the world (never by editing
    the G1's file), as a sensor's blind spot of touch and pain, and stays in collision (the G1's self-collision as shipped).
    Derived from the state it is given, the settled born state (the G1 at rest: g1scene.birth), never listed by hand: every pair
    of G1 bodies in a contact carrying a positive normal force there. For the G1 as born the list is EMPTY (no link of it touches
    another at rest: body/tests/test_sim_world.py, worlds 4 and 10), so every self-contact is felt, a kick to its own leg and two
    motor housings struck together alike. (The W1 fix listed the pairs inside a compound joint as blind, reasoning that they meet
    only at a joint's range; the W1 verifier measured under babble that 35% of their contacts over F_pain came with every hinge of
    the joint more than 0.2 rad from its range's ends: the collision shapes' hulls meeting, the robot's housings struck. That rule
    went beyond A12 and decided how much of the body has skin, the owner's call (B18), so it is gone.) Returns (a boolean nbody x
    nbody matrix, the pairs as (body, body) names)."""
    blind = np.zeros((m.nbody, m.nbody), dtype=bool)
    pairs = []
    for i in range(d.ncon):
        c = d.contact[i]
        a, b = int(m.geom_bodyid[c.geom[0]]), int(m.geom_bodyid[c.geom[1]])
        if a in g1_set and b in g1_set and c.efc_address >= 0 and d.efc_force[c.efc_address] > 0 and not blind[a, b]:
            blind[a, b] = blind[b, a] = True
            pairs.append((m.body(a).name, m.body(b).name))
    return blind, pairs


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


def gaze_reach(w=G.EYE_W, h=G.EYE_H, fovy_deg=G.EYE_FOVY, px=FOVEA_PX):
    """the window centre's reach, (yaw, pitch) in rad, and the eye's focal length in px"""
    f = (h / 2) / math.tan(math.radians(fovy_deg) / 2)
    return math.atan((w / 2 - px / 2) / f), math.atan((h / 2 - px / 2) / f), f


GAZE_REACH_YAW, GAZE_REACH_PITCH, EYE_F_PX = gaze_reach()
VERG_MAX = 2 * math.atan(G.STEREO_BASELINE / 2 / NEAR_POINT_M)


def clamp_gaze(g):
    """the gaze held in reach: vergence in [0, VERG_MAX], pitch within the reach, yaw such that both windows stay in their images"""
    v = min(VERG_MAX, max(0.0, float(g[2])))
    ry = GAZE_REACH_YAW - v / 2
    return np.array([min(ry, max(-ry, float(g[0]))), min(GAZE_REACH_PITCH, max(-GAZE_REACH_PITCH, float(g[1]))), v])


def vor(yaw, pitch, omega_cam, dt, gain=VOR_GAIN, reach=None):
    """THE VOR on the software fovea: the conjugate gaze (tangent angles in the image) counter-turned by the head's rotation. The
    ray it names in the camera's frame (x right, y up, z back) is rotated by -gain x omega dt for each gyro sample (omega already
    in the camera's frame), so a thing fixated stays fixated while the head turns (roll about the axis cannot be undone by a
    window and is left): THE SLOW PHASE. THE QUICK PHASE (A23; 3.7): `reach` (yaw, pitch), each axis's reach in rad; when a sample's
    counter-turn carries the window past its reach on an axis, the window jumps back on that axis, in the direction of the head's
    turn (against the slow phase's drift), by half that axis's reach, within the sample, and the slow phase goes on from there: the
    brainstem's nystagmus, slow and quick phases together. `reach` None: the slow phase alone. Returns (yaw, pitch, the quick
    phases this call)."""
    r = np.array([math.tan(yaw), math.tan(pitch), -1.0]); r /= np.linalg.norm(r)
    quick = 0
    for w in np.asarray(omega_cam, float):
        th = -gain * w * dt
        a = float(np.linalg.norm(th))
        if a > 0:
            k = th / a
            r = r * math.cos(a) + np.cross(k, r) * math.sin(a) + k * float(k @ r) * (1 - math.cos(a))
        if reach is not None:
            zf = max(-r[2], 1e-6)
            y, p = math.atan2(r[0], zf), math.atan2(r[1], zf)
            jump = False
            if abs(y) > reach[0]:
                y -= math.copysign(reach[0] / 2, y); jump = True
            if abs(p) > reach[1]:
                p -= math.copysign(reach[1] / 2, p); jump = True
            if jump:
                quick += 1
                r = np.array([math.tan(y), math.tan(p), -1.0]); r /= np.linalg.norm(r)
    zf = max(-r[2], 1e-6)
    return math.atan2(r[0], zf), math.atan2(r[1], zf), quick


MJ_MAXVAL = 1e10                # MuJoCo's mjMAXVAL (mjmodel.h): the bound its own checks (mj_checkPos, _checkVel, _checkAcc) call bad
MUJOCO_MESSAGES = []            # MuJoCo's warning texts this process, the latest last (kept here, never printed or logged to a file)


def _mujoco_warning(msg):
    MUJOCO_MESSAGES.append(str(msg).strip())
    del MUJOCO_MESSAGES[:-64]


def _catch_mujoco_warnings():
    """MuJoCo's warnings reach the world, not the terminal or a MUJOCO_LOG.TXT in the working folder (the world's counters
    decide a fault; the text goes into its reason). Set once per process, unless a handler is already set."""
    if mujoco.get_mju_user_warning() is None:
        mujoco.set_mju_user_warning(_mujoco_warning)


def _pose_state(pose):
    """the parent's kinematic pose (parent_kin.Pose) as the scene holds it, in a canonical plain form for the save (None: none
    drawn): arrays rebuilt from their numbers, floats, every name interned and every mapping sorted, so equal poses always
    pickle to equal bytes (a pickled array keeps its dtype object, and an unpickled one's is not numpy's own: rebuilding them
    keeps a save made after a restore the same bytes as one made without it)"""
    if pose is None:
        return None
    k = sys.intern
    arr = lambda v: np.array(np.asarray(v, float).tolist(), dtype=np.float64)
    expr = tuple(sorted((k(a), float(v)) for a, v in pose.expr.items())) if isinstance(pose.expr, dict) else float(pose.expr)
    return (arr(pose.pos), arr(pose.R),
            tuple(sorted((k(a), arr(v)) for a, v in pose.local.items())),
            tuple(sorted((k(a), arr(v)) for a, v in pose.world_override.items())),
            tuple(sorted((k(sd), tuple(sorted((k(a), None if v is None else float(v)) for a, v in h.items()))) for sd, h in pose.hand.items())),
            expr, None if pose.gaze is None else arr(pose.gaze),
            k(json.dumps(pose.report, sort_keys=True, default=float)))


def _pose_from_state(st):
    """the pose back from _pose_state's form"""
    if st is None:
        return None
    pos, R, local, override, hand, expr, gaze, report = st
    p = G.kin.Pose(pos, R)
    p.local = {a: v.copy() for a, v in local}
    p.world_override = {a: v.copy() for a, v in override}
    p.hand = {sd: dict(h) for sd, h in hand}
    p.expr = dict(expr) if isinstance(expr, tuple) else float(expr)
    p.gaze = None if gaze is None else gaze.copy()
    p.report = json.loads(report)
    return p


class WorldFault(RuntimeError):
    """a tick the physics could not live (a MuJoCo reset, a stop, a state not finite): the world stands where the tick began"""

    def __init__(self, tick, reason):
        super().__init__(f"world fault at tick {tick}: {reason}")
        self.tick, self.reason = int(tick), str(reason)


# the model's fields the world (and the parent's drivers) change at run time: saved with the world so a restore is exact
MUTABLE_MODEL_FIELDS = ("jnt_actfrcrange", "eq_data", "geom_pos", "geom_quat", "geom_size", "geom_rgba", "geom_contype",
                        "geom_conaffinity", "light_active", "light_castshadow", "light_diffuse", "light_ambient", "light_specular",
                        "light_pos", "light_dir", "mat_rgba", "mat_emission")     # (the day's light moves the sun: W5)
STATE_SPEC = mujoco.mjtState.mjSTATE_INTEGRATION


class G1World(SimWorld):
    """THE G1 IN THE LIVING ROOM, lockstep (see the module's doc). `seed` is the body's one seed (the world's own stream is derived
    from it); `extra(spec)` adds an instrument's rig to the scene before it compiles (tests only). Born at construction: the G1 on
    its back on the mat, settled, the charge full, tick 0."""

    def __init__(self, seed=1, extra=None, xml=G.XML, spinal=True):
        global R
        from body.sim import reflexes as R                              # the body's spinal cord (it reads this module's constants)
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
        self.tau_max = m.jnt_actfrcrange[self.jid, 1].copy()             # the model's own limits (3.2), before weakness scales them
        if not (np.all(m.jnt_actfrclimited[self.jid]) and np.all(self.tau_max > 0) and np.array_equal(m.jnt_actfrcrange[self.jid, 0], -self.tau_max)):
            raise ValueError("a G1 joint declares no symmetric torque limit")
        self.tau_max_sq = float(np.sum(self.tau_max ** 2))
        self.eff_slices = {}
        k = 0
        for n, js in G.EFFECTORS:
            self.eff_slices[n] = slice(k, k + len(js)); k += len(js)
        self._set_servo_law()
        # the touch zones, the pain threshold, the chargers, the parent's holds on the G1
        self.zones, self.zone_of_geom, self.body_zone = touch_zones(m, self.scene.g1_set)
        self.nz = len(self.zones)
        self.zone_body = np.full(self.nz, -1, dtype=np.int64)          # each zone's link (one link a zone)
        for g in range(m.ngeom):
            z = int(self.zone_of_geom[g])
            if z >= 0:
                if self.zone_body[z] not in (-1, int(m.geom_bodyid[g])):
                    raise ValueError(f"touch zone {self.zones[z]!r} spans two links")
                self.zone_body[z] = int(m.geom_bodyid[g])
        self.groups = zone_groups(self.zones)
        self.body_mass = float(m.body_subtreemass[m.body("pelvis").id])
        self.f_pain = PAIN_WEIGHTS * self.body_mass * float(np.linalg.norm(m.opt.gravity))
        self.palm_zones = [self.zones.index(f"{s}_hand_palm") for s in ("left", "right")]
        self.palm_of_hand = {"hand_l": self.palm_zones[0], "hand_r": self.palm_zones[1]}
        self.own_hand = np.zeros((2, self.nz + 1), dtype=bool)          # each hand's other links (index nz: no zone), for palm_own_N
        for h, grp in enumerate(("hand_l", "hand_r")):
            self.own_hand[h, [z for z in self.groups[grp] if z != self.palm_zones[h]]] = True
        self.spinal = bool(spinal)                                      # the palmar grasp at the spinal cord (off: an instrument's switch)
        if not m.opt.disableflags & mujoco.mjtDisableBit.mjDSBL_AUTORESET:
            raise ValueError("the scene leaves MuJoCo's auto-reset on (A18: <flag autoreset=\"disable\"/>)")
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
        # the gaze: the torso gyro's samples turned into the eyes' camera frame (both eyes share it; the site rides the torso)
        R_site = np.zeros(9); mujoco.mju_quat2Mat(R_site, m.site_quat[m.site("imu_in_torso").id])
        R_cam = G.eye_frames()["L"][1]
        self.gyro_to_cam = R_cam.T @ R_site.reshape(3, 3)
        self.gaze = np.zeros(3); self.gaze_v = np.zeros(3)
        self.eyes = None
        # birth
        self.tick = 0
        self.h = H_BIRTH; self.dh = 0.0
        self.paused = False
        self._apply_weakness()
        self.scene.birth()
        self.blind, self.blind_pairs = rest_blind(m, d, self.scene.g1_set)     # A12: the pairs pressing at rest (none as born)
        self._last_acts = {}
        self._spinal = {}; self._vor_quick = 0
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
        body = np.concatenate([np.stack([np.sin(ang), np.cos(ang), d.qvel[self.dof], effort], axis=1).reshape(-1), self.gaze, self.gaze_v])
        touch = np.stack([s["touch_log"], s["touch_onset"]], axis=1).reshape(-1)
        obs = {"body": body, "touch": touch, "vestibular": s["vestibular"].copy(), "charge": np.array([self.h, self.dh]),
               "pain": (s["pain_force"] > self.f_pain).astype(np.float64)}
        truth = self._truth()
        if self.eyes is not None:                                       # the eyes' retina codes (W3; body/sim/eyes.py)
            seen = self.eyes.see()
            for k in ("eye_p", "eye_f", "face_fovea", "face_periph"):   # the pixels' own; A1's face test stays in the truth
                obs[k] = seen[k]
            truth["eyes"] = seen["truth"]
        return Frame(self.tick, obs, 0.0, truth)

    def apply(self, acts):
        """one tick (150 ms, 75 steps of 2 ms) with the body's acts {effector name: flat act}: the gaze's and the joint effectors';
        names the world does not move (the voice, the word output) are the other lanes'. The hands' acts pass the spinal cord
        first (the palmar grasp summed with them, body/sim/reflexes.py). A tick MuJoCo cannot live raises WorldFault and leaves the
        world where the tick began (A18)."""
        if self.paused:
            raise RuntimeError("G1World: the world moved while paused (the night)")
        m, d = self.m, self.d
        t_apply = time.perf_counter(); t_phys = 0.0
        acts = dict(acts or {})
        spinal = {}
        if self.spinal:                                                 # THE SPINAL CORD: the palmar grasp on each hand's own act
            for hand, z in self.palm_of_hand.items():
                a, ev = R.grasp(hand, acts.get(hand), float(self._sensed["touch_log"][z]))
                if ev is not None:
                    spinal[hand] = ev
                    acts[hand] = a
        steps = {}                                                      # every act read before anything moves (a bad act moves nothing)
        for name, js in G.EFFECTORS:
            a = acts.get(name)
            if a is not None and int(a) != EFFECTOR_REST[name]:
                steps[name] = np.array([SETTINGS[k] for k in act_digits(a, len(js))])
        a = acts.get(GAZE_NAME)
        gaze_step = np.zeros(3) if a is None else np.array([GAZE_SETTINGS[j][k] for j, k in enumerate(act_digits(a, len(GAZE_JOINTS)))])
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
        self._last_acts = {k: int(v) for k, v in acts.items() if k in EFFECTOR_REST}
        self._spinal = spinal
        gaze0 = self.gaze.copy()
        self.gaze = clamp_gaze(self.gaze + gaze_step)                  # the gaze's act: the windows jump at the tick's start
        rest_a = self.aid[rest_idx] if rest_idx else None
        rest_q = self.qadr[rest_idx] if rest_idx else None
        n, nz = STEPS_PER_TICK, self.nz
        F = np.zeros((n, nz)); imu = np.zeros((n, 12)); eff = np.zeros(n); fed = False
        own = np.zeros(2)                                              # the palms' own-hand force
        alpha = self.alpha
        try:
            for s in range(n):
                if rest_a is not None:
                    d.ctrl[rest_a] += alpha * (d.qpos[rest_q] - d.ctrl[rest_a])
                t0 = time.perf_counter()
                mujoco.mj_step(m, d)
                t_phys += time.perf_counter() - t0
                F[s] = self._zone_forces()
                own += self._palm_own
                imu[s] = d.sensordata[self.imu_adr]
                tq = d.qfrc_actuator[self.dof]
                eff[s] = float(tq @ tq)
                if self.chargers.size and not fed:
                    fed = self._palm_on_charger()
            mujoco.mj_forward(m, d)                                     # the tick's end: its accelerations, contacts and sensors
        except mujoco.FatalError as e:
            self._restore(start)
            raise WorldFault(self.tick, f"MuJoCo stopped: {e}")
        # A18: MuJoCo's counters (its own checks, on every step but the last one's result) and the end state under the same bound
        # (the last step's position, velocity and acceleration, which MuJoCo would check only at the next tick's first step)
        warned = {mujoco.mjtWarning(i).name: int(d.warning[i].number) - warn0[i]
                  for i in range(int(mujoco.mjtWarning.mjNWARNING)) if int(d.warning[i].number) != warn0[i]}
        bad = self._unsound()
        if warned or bad:
            self._restore(start)
            said = f" ({MUJOCO_MESSAGES[-1]})" if MUJOCO_MESSAGES and warned else ""
            raise WorldFault(self.tick, f"MuJoCo's warnings {warned}{said}" if warned else f"the tick's end state: {bad}")
        # the tick's senses
        imu = self._imu_noisy(imu)
        self._sense_tick(F, imu, own / n)
        reach = (GAZE_REACH_YAW - self.gaze[2] / 2, GAZE_REACH_PITCH)
        yaw, pitch, quick = vor(self.gaze[0], self.gaze[1], imu[:, 3:6] @ self.gyro_to_cam.T, m.opt.timestep, reach=reach)   # the VOR
        self.gaze = clamp_gaze([yaw, pitch, self.gaze[2]])
        self.gaze_v = (self.gaze - gaze0) / TICK_S
        self._vor_quick = int(quick)
        drain = DRAIN_BASE + DRAIN_EFFORT * float(eff.mean()) / self.tau_max_sq
        h0 = self.h
        self.h = float(min(1.0, max(0.0, self.h - drain + (FEED_RATE if fed else 0.0))))
        self.dh = self.h - h0
        self._drain = drain; self._fed = fed
        self.tick += 1
        self.timing["ticks"] += 1; self.timing["physics_s"] += t_phys; self.timing["apply_s"] += time.perf_counter() - t_apply

    def _unsound(self):
        """what MuJoCo's own checks would call bad in the state as it stands (a position, velocity or acceleration not finite or
        beyond MJ_MAXVAL), or "" when it is sound"""
        d = self.d
        for name, x in (("qpos", d.qpos), ("qvel", d.qvel), ("qacc", d.qacc)):
            if not np.all(np.abs(x) <= MJ_MAXVAL):                     # NaN fails the comparison too
                i = int(np.argmin(np.abs(x) <= MJ_MAXVAL))
                return f"{name}[{i}] = {x[i]:g} (not finite, or beyond MuJoCo's bound {MJ_MAXVAL:g})"
        return ""

    def pause(self):
        """the night: the world freezes exactly where it is"""
        self.paused = True

    def resume(self):
        """the morning: on from the same state"""
        self.paused = False

    def save_state(self):
        """the whole world as bytes: the physics, the model's run-time fields, the charge, the senses' carry, the world's random
        stream and the parent's pose as the scene last drew it (Scene.pose: the W1 verifier's third round)"""
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
        zones; A12's rest blind spots, none as born, felt by neither), plus each of the parent's active holds on a G1 body on
        that body's zone. Also this step's force on each palm from its own hand's other links (`_palm_own`)"""
        d, zg, nz = self.d, self.zone_of_geom, self.nz
        out = np.zeros(nz)
        self._palm_own = np.zeros(2)
        nc = d.ncon
        if nc:
            con = d.contact
            adr = con.efc_address
            ok = adr >= 0
            geom = con.geom
            bod = self.m.geom_bodyid[geom]
            ok &= ~self.blind[bod[:, 0], bod[:, 1]]                     # a pair pressing at rest: the skin's blind spot (A12)
            fn = np.where(ok, d.efc_force[np.where(ok, adr, 0)], 0.0)     # the elliptic cone's first row: the normal force
            z = zg[geom]
            for col in (0, 1):
                zc = z[:, col]
                sel = zc >= 0
                if sel.any():
                    out += np.bincount(zc[sel], weights=fn[sel], minlength=nz)
            for h, pz in enumerate(self.palm_zones):                    # the palm pressed by its own hand's links (the grasp's log)
                mine = ((z[:, 0] == pz) & self.own_hand[h][z[:, 1]]) | ((z[:, 1] == pz) & self.own_hand[h][z[:, 0]])
                if mine.any():
                    self._palm_own[h] = float(fn[mine].sum())
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

    def _sense_tick(self, F, imu, palm_own=None):
        """the tick's aggregates: touch (the mean force's log, its onset), pain's filtered force, the IMUs (their noisy samples),
        the palms' own-hand force"""
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
                        "carry": F[-(w - 1):].copy(), "vestibular": vest, "peak_force": F.max(axis=0),
                        "palm_own": np.zeros(2) if palm_own is None else np.asarray(palm_own, float).copy()}

    def _imu_noisy(self, imu):
        """the IMUs' samples as the sensors give them: the model's declared noise from the world's stream, clipped at their ranges"""
        return np.clip(imu + self.rng.standard_normal(imu.shape) * self.imu_noise, -self.imu_cut, self.imu_cut)

    def _vestibular(self, x):
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
                        "carry": np.repeat(F, PAIN_WINDOW_STEPS - 1, axis=0), "vestibular": self._vestibular(self._imu_noisy(imu)),
                        "peak_force": F[0].copy(), "palm_own": self._palm_own.copy()}
        self._drain = 0.0; self._fed = False

    def _truth(self):
        """world truth for the parent and the instruments (never the body)"""
        m, d = self.m, self.d
        toys = {m.body(b).name[4:]: d.xpos[b].copy() for b in range(m.nbody) if m.body(b).name.startswith("toy_")}
        s = self._sensed
        return {"time": float(d.time), "pelvis": d.qpos[0:7].copy(), "torso": d.xpos[m.body("torso_link").id].copy(),
                "toys": toys, "touch_N": s["touch_force"].copy(), "pain_N": s["pain_force"].copy(), "peak_N": s["peak_force"].copy(),
                "f_pain": self.f_pain, "drain": self._drain, "fed": self._fed, "ncon": int(d.ncon), "acts": dict(self._last_acts),
                "gaze": self.gaze.copy(), "spinal": dict(self._spinal), "vor_quick": self._vor_quick,
                "palm_own_N": {"hand_l": float(s["palm_own"][0]), "hand_r": float(s["palm_own"][1])}}

    # ---------------------------------------------------------------- the state
    def _capture(self):
        m, d = self.m, self.d
        phys = np.zeros(mujoco.mj_stateSize(m, STATE_SPEC))
        mujoco.mj_getState(m, d, phys, STATE_SPEC)
        return {"version": 1, "nstate": int(phys.size), "physics": phys,
                "model": {f: getattr(m, f).copy() for f in MUTABLE_MODEL_FIELDS},
                "tick": self.tick, "h": self.h, "dh": self.dh, "paused": self.paused, "seed": self.seed,
                "sensed": {k: v.copy() for k, v in self._sensed.items()}, "drain": self._drain, "fed": self._fed,
                "last_acts": dict(self._last_acts), "rng": self.rng.bit_generator.state, "gaze": self.gaze.copy(), "gaze_v": self.gaze_v.copy(),
                "spinal": dict(self._spinal), "vor_quick": self._vor_quick, "scene_pose": _pose_state(self.scene.pose),
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
        self._spinal = dict(st.get("spinal", {})); self._vor_quick = int(st.get("vor_quick", 0))
        self.rng.bit_generator.state = st["rng"]
        self.gaze = np.asarray(st.get("gaze", np.zeros(3)), float).copy()        # (a save from before the gaze: born at 0)
        self.gaze_v = np.asarray(st.get("gaze_v", np.zeros(3)), float).copy()
        if "scene_pose" in st:                                          # the parent's pose as the scene last drew it (W2 goes on
            self.scene.pose = _pose_from_state(st["scene_pose"])        # from it); her mocap and face geoms are in the physics and
        mujoco.mj_forward(m, d)                                         # the model fields above
