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
model's own, 3.2). No gravity compensation, no balance law: nothing but
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
               afferents); a hold of the parent's (a capped spring on a G1 link: body/sim/parent_motion.py) adds its force to
               that link's zone, each physics step (being held is felt, 4.2)
  vestibular   per IMU (imu_in_torso, which moves with the head: the vestibule; imu_in_pelvis: the trunk's graviceptors)
               [accelerometer mean (3), peak (3), gyro mean (3), peak (3)] over the tick's 75 samples (the peak per axis the
               signed sample of largest size), with the model's own declared noise and ranges
and, for the pain source (cortisol since A88) and the spinal reflexes (a disclosed exception, 3.4: pain is the contact force on a zone),
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
forces in newtons, the eyes' images) is for the parent and the instruments only, never the body. The ears are
the parent lane's (P2); their channel joins the frame there.

NO CHARGE (A88, the owner's decision 2026-09-26): the body has no need the room meets. The charge, its drain, the charger (the bottle on
its dock) and the weakness law that followed the charge are gone; the joints' limits are the declared ones at every tick. Her face is
the only reward; pain is cortisol (body/sim/anatomy.py).

THE REFLEXES are the body's (body/sim/reflexes.py). The withdrawal is declared on each limb's effector (the core's hook: it takes
the limb's tick) and its act reaches the world as any act. The palmar grasp is the spinal cord's: `apply` sums it into each hand's
own act before anything moves (the truth's `spinal` logs it), so the hand's own act that tick can open it (A11). `spinal=False`
switches it off (an instrument's switch). The grasp fires on anything pressing the palm, its own fingers included (a fist closed
on nothing keeps itself closed until the hand's own act opens it, as newborns' hands are fisted); the truth's `palm_own_N` (the
palm's tick-mean force from its own hand's links) lets W4 count those fists (the W1 verifier's eighth finding).

THE PARENT'S MOTION (W2; body/sim/parent_motion.py) runs inside the tick: `parent.tick_begin()` before the physics (her acts
advance and her pose at the tick's end is found), `before_step(s)` and `after_step(s)` around each of the 75 steps (her body's joints
driven toward her plan with a woman's strength, her holds' capped springs applied as outside forces and their reaction on her
hands, her contacts with the child read for her stop and her pain), `tick_end()` after. She is a body in the same physics
(the lead's decision of 2026-09-25; body/sim/parent_body.py). No equality constraint may tie the G1 to anything outside it, a weld or a connect by
body or by site, or a joint of it held (a world with one refuses to be born: every hold on it is a capped spring, 4.1). Her motion's state is saved and restored with the world's, and rolled back with it when a tick faults.
`parent=False` builds the world without her motion (an instrument's switch); the truth's `parent` is her motion as her conduct and
the instruments see it.

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
from body.core.world import Frame, SimWorld, SubFrame  # noqa: E402
from body.sim import anatomy as AN  # noqa: E402  (the G1's anatomy: the frame's contract, S5a)
from body.sim.tract import Tract  # noqa: E402  (the voice effector's physics, 4.9)
from body.sim.voice.synth import PA_PER_UNIT  # noqa: E402  (the tract's engine units -> pascals at 1 m)
from body.sim import ears as EA  # noqa: E402  (the ears, P2)
from body.sim import observer as OBS  # noqa: E402  (the born momentum observer: contact from the robot's own sensors, A37)
from body.sim import sounds as SND  # noqa: E402  (the room's sounds from its physics, W5)

R = None                        # body/sim/reflexes.py, the body's spinal cord: bound at the first world's birth (it imports this module)

# ---------------------------------------------------------------- the disclosed constants (SIM_DESIGN.md section 10)
TICK_S = 0.150                  # a tick: 150 ms of sim time (clock; ours)
STEPS_PER_TICK = 75             # 2 ms physics steps, the model's option timestep (world; ours)
SETTINGS_PER_JOINT = 5          # an act's setting per joint: -big, -small, 0, +small, +big (3.5; ours)
STEP_SMALL, STEP_BIG = 0.09, 0.27                 # rad a tick, the trunk, arms, hands and legs (3.5; ours)
SETTINGS = (-STEP_BIG, -STEP_SMALL, 0.0, STEP_SMALL, STEP_BIG)
STRAIN_LINE = 0.7               # A191: a gear's load over this share of its pain line is strain (the high-threshold mechanoreceptor's and
                                # the tendon organ's warning under the pain line; ours, measured on the day-82 copy: pain follows within
                                # 4 ticks in 13.6% of such ticks against 1.4% of the others)
SERVO_ERR_AT_LIMIT = 0.25       # rad of error at which a servo reaches its torque limit (3.3; anatomy, ours)
SERVO_ERR_AT_LIMIT_HAND = 0.1   # the Dex3's joints reach theirs at 0.1 rad (3.3; anatomy, ours)
SERVO_DAMP_S = 0.04             # damping = 0.04 s x stiffness (3.3; anatomy, ours)
TONE_TAU_TICKS = 3.0            # at rest the target relaxes to the measured angle with this time constant (3.3; innate, ours)
PASSIVE_MARGIN = 0.15           # A143 (2026-09-30): THE PASSIVE END-RANGE STIFFNESS. In the last PASSIVE_MARGIN of a joint's range (either end) the
PASSIVE_FRAC = 1.0              # body's passive tissues push back toward mid-range, rising linearly to PASSIVE_FRAC x the joint's torque limit at
                                # the stop (both ours; the form: passive elastic joint moments are near zero mid-range and rise steeply toward the
                                # range ends, Riener and Edrich 1999; a real G1 drives its joints inside software limits with a margin). Why: on
                                # life day 43 the body lay folded into its range stops (6.5 of the 14 arm joints at a stop on average, the knees, the
                                # waist and three hip joints too), since the resting target follows the measured angle and nothing pulls a joint
                                # off a stop once gravity or the floor has pressed it there; a joint at a stop shows its inverse model nothing of
                                # its acts (kappa near chance), so its readout stays flat and it draws big steps at random (C148). The push is the
                                # tissue's, not the motor's: it is applied to the joint (qfrc_applied), never read as gear load, pain or heat.
                                # C163 (the same evening): at 0.2 the push could not hold a DRIVEN joint off its stop (day 43 at tick 15,000: the
                                # left wrist roll's flat readout commanding big steps into the stop, the target clipped at the range end, the motor
                                # pushing in at 3 Nm against the tissue's 4.9: the joint 0.01 rad from the hard stop, every act there still blind).
                                # At 1.0 the tissue's push at the stop equals the motor's limit, so no drive reaches the hard stop: under full drive
                                # the joint settles where kp x (stop - q) meets the push, about 0.17 rad inside (kp = limit / 0.25 rad), where every
                                # step still moves it and the gear load stays under the pain line. An infant's muscles cannot push a joint past
                                # its anatomical limit against the tissue: the ratio is the infant's, the number ours. The hand's joints (Dex3) are
                                # left out: their tiny inertia under a stiff spring would need a finer physics step, and the fingers were never
                                # the problem
PAIN_WEIGHTS = 3.0              # F_pain = 3 x the body's weight from the model file (6.2, A12; innate, ours)
PAIN_WINDOW_STEPS = 5           # a zone's force for pain: the tick's largest 10 ms mean (A12; innate, ours)
TOUCH_UNIT_N = 1.0              # touch's log force is log(1 + F / 1 N) (3.4; anatomy, ours)
HAND_DEPTH_M = 0.03             # A171: a held toy's centre this far before the palm's face (half a small toy; the hand's grasp centre, ours)
SIDE_COS = 0.64                 # A171: a push counts on a side of the hand when within 50 deg of the palm's normal (ours; across it, the mat
                                # under a hand lying on its edge, it is no side's)
WORLD_STREAM = 1                # the world's random stream: SeedSequence(seed, spawn_key=(1,)) (ours)

# ---------------------------------------------------------------- S5a: the world as the G1's anatomy meets it (body/sim/anatomy.py)
# THE OBSERVER (A37; 3.4): a joint's OUTSIDE TORQUE is the generalized force on its dof of every contact on the G1 (J^T f over the
# contact rows, MuJoCo's own Jacobian; self-contact included, as a real observer sees a limb struck against the body) and of the
# parent's holds (her capped springs, applied as outside forces on the held links); the base's OUTSIDE WRENCH is the same on the
# floating base's six dofs. It is the world's truth, which the robot's born momentum observer estimates from its own encoders,
# currents and inertial unit (Haddadin et al. 2017). The observer's own error is C43's, open: until it is measured the truth is given,
# no noise added (ours, disclosed). A joint's mechanical stop is not a contact and is not in it.
OBS_WINDOW_STEPS = 5            # 10 ms means: the pain law's window (A37) and the sub-step's period below the tick (7.5; innate, ours)
# PAIN FROM THE JOINTS (A37: "a load from outside larger than a joint's motor can hold back-drives its gear: the robot's own damage
# line"; the lead's reading of 2026-09-25, S5a): what loads a joint's gear is the torque the gear carries, the motor's torque less what
# the rotor's inertia takes of the joint's acceleration, tau_gear = tau_motor - I_rotor x q'' (I_rotor the model's armature through the
# gear; a rotor resisting a sudden acceleration passes the rest of the blow through the gear). A joint is in pain on a tick when its
# gear's 10 ms mean passes the joint's declared limit. A support the motor does not fight (the mat carrying the lying torso: a large
# outside torque at the waist with the waist's motor near zero) loads no gear and is no pain; a blow that accelerates a joint does.
# Computed from what the robot's own sensors give (the torque from the motor's current, the acceleration from its encoder). The first
# reading, outside torque past the limit, put the lying, babbling G1 in pain on 70-80% of its ticks, almost all of it the mat's
# support mapped onto the waist and the shoulders (measured, S5a); the base is in pain past F_pain as before.
# THE MOTORS' HEAT (A39: a sense only, never a reward or a drive): a first-order thermal model per motor, the winding's temperature
# above the room's relaxing toward HEAT_RISE_C x (torque / the declared limit)^2 with the time constant HEAT_TAU_S (copper losses go
# as the current squared, the current as the torque). Unitree's thermal constants are C46's, open: these two are ours until read
# (disclosed). The body channel carries the rise over HEAT_RISE_C.
HEAT_TAU_S = 600.0
HEAT_RISE_C = 60.0
# THE NIGHT'S LIGHT (5.4, A46: dimmed, never switched off): each room light's diffuse, ambient and specular x NIGHT_LIGHT at dusk, back
# over the wake's first DAWN_TICKS ticks (ours)
NIGHT_LIGHT = 0.05
TWIST_MARGIN_RAD = 0.05     # A110 (C103): a joint of the sleeping child's within this of a stop of its range is at its stop (the
TWIST_JOINTS = ("waist", "hip", "shoulder")   # ... the joints whose stop twists the body: the trunk and the limbs' roots (C103's waist yaw and
                            # hip pitch); not a knee, an ankle, a wrist or a finger (the dawn-17 pair's ankle roll and wrist yaw)
                            # servo's target clipped at the range, the measured angle a few hundredths inside it); ours
DAWN_TICKS = 30
VOICE_NAME, WORDS_NAME = "voice", "words"                          # the anatomy's tract (effector 0) and its words' output (1)
VOICE_REST = (SETTINGS_PER_JOINT ** len(AN.TRACT) - 1) // 2      # the tract at rest: every articulator at setting 2

# THE GAZE: the software fovea's effector (3.4, 3.5; W3). The G1 has no eyes that turn: a 64 x 64 px window inside each camera's
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
FOVEA_PX = 64                   # the fovea window, px of the native image (about 21 deg at 3 px a degree: A42; 3.4; anatomy, ours)

# THE SERVO GAINS (A39; 3.3): Unitree's published position gains for the G1 and the Dex3, read 2026-09-25:
#   - the body (legs, waist, arms): unitree_sdk2's whole-body low-level example (example/g1/low_level/g1_ankle_swing_example.cpp: Kp
#     60 60 60 100 40 40 a leg (hip pitch, roll, yaw, knee, ankle pitch, roll), 60 40 40 the waist (yaw, roll, pitch), 40 each arm joint;
#     Kd 1 each, the knee 2), the one set Unitree publishes for all 29 joints (unitree_rl_gym's walking policies use stiffer legs, 100 100
#     100 150 40 40 with Kd 2 2 2 4 2 2: a locomotion controller's, not a resting law's);
#   - the Dex3's 14 joints: unitree_sdk2's example/g1/dex3/g1_dex3_example.cpp, its grip (kp 1.5, kd 0.1; its sweep 0.5).
# The torque is still clamped at the joint's limit x the weakness. At rest the targets relax to the measured angles (TONE_TAU_TICKS).
UNITREE_KP = {"hip_pitch": 60.0, "hip_roll": 60.0, "hip_yaw": 60.0, "knee": 100.0, "ankle_pitch": 40.0, "ankle_roll": 40.0,
              "waist_yaw": 60.0, "waist_roll": 40.0, "waist_pitch": 40.0, "shoulder_pitch": 40.0, "shoulder_roll": 40.0,
              "shoulder_yaw": 40.0, "elbow": 40.0, "wrist_roll": 40.0, "wrist_pitch": 40.0, "wrist_yaw": 40.0}
UNITREE_KD = {"knee": 2.0}                                           # every other body joint 1
DEX3_KP, DEX3_KD = 1.5, 0.1


SERVO_LAW = "unitree"


def first_law_gains(tau_max):
    """the first law's gains (3.3): kp = limit / SERVO_ERR_AT_LIMIT (the Dex3's / 0.1 rad), kv = SERVO_DAMP_S x kp"""
    err = np.array([SERVO_ERR_AT_LIMIT_HAND if "_hand_" in j else SERVO_ERR_AT_LIMIT for j in JOINTS])
    kp = tau_max / err
    return kp, SERVO_DAMP_S * kp


def servo_gains():
    """(kp, kd) per joint in JOINTS' order: Unitree's (A39, the constants above)"""
    kp, kd = [], []
    for j in JOINTS:
        base = j.replace("_joint", "")
        if "_hand_" in base:
            kp.append(None); kd.append(DEX3_KD); continue
        key = base.replace("left_", "").replace("right_", "")
        kp.append(UNITREE_KP[key]); kd.append(UNITREE_KD.get(key, 1.0))
    return kp, np.array(kd)


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


def vor(yaw, pitch, omega_cam, dt, gain=VOR_GAIN, reach=None, offset=(0.0, 0.0), quick_frac=0.5):
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
    g = np.asarray(gain, float)
    if g.ndim == 0:
        gv = np.array([float(g)] * 3)
    else:                                                             # per axis (S5a: the body's born gain + the flocculus's
        gv = np.array([float(g[1]), float(g[0]), 0.5 * float(g[0] + g[1])])   # correction): pitch about the camera's x, yaw
    for w in np.asarray(omega_cam, float):                            # about its y; roll (never undone) at their mean
        th = -gv * w * dt
        a = float(np.linalg.norm(th))
        if a > 0:
            k = th / a
            r = r * math.cos(a) + np.cross(k, r) * math.sin(a) + k * float(k @ r) * (1 - math.cos(a))
        if reach is not None:
            zf = max(-r[2], 1e-6)
            y, p = math.atan2(r[0], zf), math.atan2(r[1], zf)
            jump = False
            if abs(y) > reach[0]:
                y -= math.copysign(reach[0] * quick_frac, y); jump = True
            if abs(p) > reach[1]:
                p -= math.copysign(reach[1] * quick_frac, p); jump = True
            if jump:
                quick += 1
                r = np.array([math.tan(y), math.tan(p), -1.0]); r /= np.linalg.norm(r)
    zf = max(-r[2], 1e-6)
    return math.atan2(r[0], zf) - float(offset[0]), math.atan2(r[1], zf) - float(offset[1]), quick


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
    """the parent's planned pose (parent_kin.Pose) as the scene holds it, in a canonical plain form for the save (None: none
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
                        "light_pos", "light_dir", "mat_rgba", "mat_emission",     # (the day's light moves the sun: W5)
                        "actuator_biasprm")      # the cerebellum's torque below the tick, the servos' constant bias term (7.5)
STATE_SPEC = mujoco.mjtState.mjSTATE_INTEGRATION


def _canon_acts(acts):
    """the tick's acts as the save keeps them: {effector: int}, sorted, every name interned, so equal acts always pickle to equal
    bytes (pickle memoizes a string by its identity: the same names from two callers, as two objects, had made the save's bytes
    differ with the same content; the W2 verifier's finding)"""
    return {sys.intern(str(k)): int(v) for k, v in sorted(dict(acts).items()) if k in EFFECTOR_REST or k in (VOICE_NAME, WORDS_NAME)}


def _canon(x):
    """a state in a canonical plain form for the save, so equal states always pickle to equal bytes (a pickled array keeps its dtype
    object, and an unpickled one's is not numpy's own; a dict keeps its insertion order): arrays as ("nd", dtype, shape, bytes),
    numpy scalars as Python numbers, mappings as sorted (key, value) pairs, names interned. _uncanon gives it back"""
    if isinstance(x, np.ndarray):
        return ("__nd__", x.dtype.str, tuple(int(n) for n in x.shape), np.ascontiguousarray(x).tobytes())
    if isinstance(x, np.generic):
        return x.item()
    if isinstance(x, dict):
        return ("__map__", tuple(sorted(((sys.intern(k) if isinstance(k, str) else k), _canon(v)) for k, v in x.items())))
    if isinstance(x, (list, tuple)):
        return ("__seq__", type(x).__name__, tuple(_canon(v) for v in x))
    if isinstance(x, str):
        return sys.intern(x)
    return x


def _fresh(a):
    """an array rebuilt on numpy's own dtype object (an unpickled array's dtype is a copy, which pickles to other bytes)"""
    return np.frombuffer(np.ascontiguousarray(a).tobytes(), dtype=np.dtype(a.dtype.str)).reshape(a.shape).copy()


def _uncanon(x):
    if isinstance(x, tuple) and x and isinstance(x[0], str):
        if x[0] == "__nd__":
            return np.frombuffer(x[3], dtype=np.dtype(x[1])).reshape(x[2]).copy()
        if x[0] == "__map__":
            return {k: _uncanon(v) for k, v in x[1]}
        if x[0] == "__seq__":
            v = [_uncanon(y) for y in x[2]]
            return tuple(v) if x[1] == "tuple" else v
    return x


BUCKET_NEAR_M = 0.5                    # C130: the bucket farther than this from the child's chest at dawn is set beside it (ours)
BUCKET_BESIDE_M = 0.35                 # C130: where it is set: a ring this far round its chest, within a G1 arm's reach (ours)


def PM_Child(m, d, g1_set):
    from body.sim import parent_motion as PM
    return PM.Child(m, d, g1_set)


def _tendon_bodies(m, t):
    """the bodies a tendon passes through: its sites' and wrapping geoms' bodies (a spatial tendon), its joints' (a fixed one)"""
    out = set()
    for w in range(int(m.tendon_adr[t]), int(m.tendon_adr[t]) + int(m.tendon_num[t])):
        typ, oid = int(m.wrap_type[w]), int(m.wrap_objid[w])
        if oid < 0:
            continue
        if typ == int(mujoco.mjtWrap.mjWRAP_SITE):
            out.add(int(m.site_bodyid[oid]))
        elif typ in (int(mujoco.mjtWrap.mjWRAP_SPHERE), int(mujoco.mjtWrap.mjWRAP_CYLINDER)):
            out.add(int(m.geom_bodyid[oid]))
        elif typ == int(mujoco.mjtWrap.mjWRAP_JOINT):
            out.add(int(m.jnt_bodyid[oid]))
    return out


def _eq_owner(m, e, oid):
    """the body an equality constraint's end belongs to: a body's own id, a site's body, a joint's body; the world (0) for an end
    left open (a weld or connect to the world, a joint held at a value); None for a tendon's (its bodies: _tendon_bodies) or a
    flex's (none in the scene)"""
    if oid < 0:
        return 0
    t = int(m.eq_objtype[e])
    if t == int(mujoco.mjtObj.mjOBJ_SITE):
        return int(m.site_bodyid[oid])
    if t == int(mujoco.mjtObj.mjOBJ_BODY):
        return int(oid)
    if int(m.eq_type[e]) == int(mujoco.mjtEq.mjEQ_JOINT):
        return int(m.jnt_bodyid[oid])
    return None


class G1World(SimWorld):
    """THE G1 IN THE LIVING ROOM, lockstep (see the module's doc). `seed` is the body's one seed (the world's own stream is derived
    from it); `extra(spec)` adds an instrument's rig to the scene before it compiles (tests only). Born at construction: the G1 on
    its back on the mat, settled, tick 0."""

    def __init__(self, seed=1, extra=None, xml=G.XML, spinal=True, parent=True, righting=True, tone=True, standing=True, posture=True):
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
        self.passive_mask = np.array([0.0 if j in G.DEX3 else 1.0 for j in JOINTS])   # A143/C163: the passive end-range push, not at the hand's joints
        self.tau_hold = np.array([G.DEX3[j]["hold"] if j in G.DEX3 else float(t) for j, t in zip(JOINTS, self.tau_max)])   # each joint's
        # pain line (A37, A73): the gear's load it takes before it back-drives: the Dex3's holding figure (A80), else the declared limit
        if not (np.all(m.jnt_actfrclimited[self.jid]) and np.all(self.tau_max > 0) and np.array_equal(m.jnt_actfrcrange[self.jid, 0], -self.tau_max)):
            raise ValueError("a G1 joint declares no symmetric torque limit")
        self.eff_slices = {}
        k = 0
        for n, js in G.EFFECTORS:
            self.eff_slices[n] = slice(k, k + len(js)); k += len(js)
        self.eff_joint_idx = {n: list(range(sl.start, sl.stop)) for n, sl in self.eff_slices.items()}   # (A139: the tendon reflex's map)
        self._set_servo_law()
        # the touch zones, the pain threshold, the parent's holds on the G1
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
        self.sole_zones = [[i for i, z in enumerate(self.zones) if z.startswith(f"{s}_ankle")] for s in ("left", "right")]   # A193: each foot's zones
        self._trunk_zones = [self.zones.index(z) for z in ("torso", "pelvis")]   # A195: where a hold on its trunk is felt
        self.palm_of_hand = {"hand_l": self.palm_zones[0], "hand_r": self.palm_zones[1]}
        self.own_hand = np.zeros((2, self.nz + 1), dtype=bool)          # each hand's other links (index nz: no zone), for palm_own_N
        for h, grp in enumerate(("hand_l", "hand_r")):
            self.own_hand[h, [z for z in self.groups[grp] if z != self.palm_zones[h]]] = True
        self.palm_body = [int(self.zone_body[pz]) for pz in self.palm_zones]   # A171: each palm link, for its normal (out of the palm)
        self.palm_geom = [int(np.where(self.zone_of_geom == pz)[0][0]) for pz in self.palm_zones]   # A171: each palm's shape, for its face
        self.hand_of_zone = np.full(self.nz + 1, -1, dtype=np.int64)          # A171: each zone's hand (0 left, 1 right, -1 none; nz: no zone)
        for h, grp in enumerate(("hand_l", "hand_r")):
            self.hand_of_zone[list(self.groups[grp])] = h
        self._sides = np.zeros((2, 2))                                          # A171: this step's push on each hand from things not
                                                                                # its own: [palmar (toward its back), dorsal (toward its palm)]
        self.spinal = bool(spinal)                                      # the palmar grasp at the spinal cord (off: an instrument's switch)
        self.standing = bool(standing)                                  # A193: the standing and stepping reflexes (off: an instrument's switch)
        self.posture = bool(posture)                                    # A195: the postural tone and the vestibulospinal reflex below the tick (off: an instrument's switch)
        self._vest_pitch, self._vest_on = 0.0, False
        self._stepping = {"leg_l": ["stance", 0], "leg_r": ["stance", 0]}
        self.righting = bool(righting)                                  # the prone pattern at the cord (A92; off: an instrument's switch)
        self.tone = bool(tone)                                          # the arms' resting tone at the cord (A185; off: an instrument's switch)
        if not m.opt.disableflags & mujoco.mjtDisableBit.mjDSBL_AUTORESET:
            raise ValueError("the scene leaves MuJoCo's auto-reset on (A18: <flag autoreset=\"disable\"/>)")
        g1 = self.scene.g1_set
        for e in range(m.neq):                                          # no equality ties the G1 to anything outside it: a weld, a
            ends = [_eq_owner(m, e, int(m.eq_obj1id[e])), _eq_owner(m, e, int(m.eq_obj2id[e]))]   # connect (by body or by
            if None not in ends and (ends[0] in g1) != (ends[1] in g1):                            # site), a joint held
                raise ValueError("an equality constraint on the G1 (a weld, a connect or a joint held): every hold on it is a "
                                 "capped spring (SIM_DESIGN.md 4.1, 4.2, A25)")
            if int(m.eq_type[e]) == int(mujoco.mjtEq.mjEQ_TENDON):     # a tendon equality: the bodies its tendons pass through
                bodies = set()
                for t in (int(m.eq_obj1id[e]), int(m.eq_obj2id[e])):
                    bodies |= _tendon_bodies(m, t) if t >= 0 else {0}
                if bodies & g1 and bodies - g1:
                    raise ValueError("a tendon equality ties the G1 to something outside it: every hold on it is a capped spring "
                                     "(SIM_DESIGN.md 4.1, 4.2, A25)")
        for t in range(m.ntendon):                                      # nor a tendon (a rope) from the world or the room to it
            bodies = _tendon_bodies(m, t)
            if bodies & g1 and bodies - g1:
                raise ValueError("a tendon runs from the G1 to something outside it: every hold on it is a capped spring "
                                 "(SIM_DESIGN.md 4.1, 4.2, A25)")
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
        # S5a: THE ANATOMY'S CONTRACT (body/sim/anatomy.py): the joints in its order, the Dex3's 16 zones, the floating base, the
        # cerebellum's readout joints, the motors' heat, the tract, the ears, the night's light
        if tuple(JOINTS) != tuple(AN.BODY_JOINTS):
            raise ValueError("the world's joints are not the anatomy's BODY_JOINTS in order")
        rng_ok = np.allclose(np.stack([self.lo, self.hi], 1), np.array(AN.RANGES), atol=1e-4)
        if not rng_ok:
            raise ValueError("the model's joint ranges are not the anatomy's RANGES (the cerebellum's angle fibres)")
        self.dex = np.array([self.zones.index(z) for z in AN.ZONES])        # the Dex3's 16 zones among the world's per-link zones
        self.base_dof = int(m.jnt_dofadr[m.joint("floating_base_joint").id])
        self.pelvis_id = m.body("pelvis").id
        self.g1_bodies = np.array([b for b in range(m.nbody) if int(m.body_rootid[b]) == int(m.body_rootid[m.body('pelvis').id])], dtype=np.int64)
        self.cereb_idx = np.array([JOINTS.index(j) for j in AN.CEREB_JOINTS])
        self.weight = self.body_mass * float(np.linalg.norm(m.opt.gravity))
        self.armature = m.dof_armature[self.dof].copy()                     # each joint's rotor inertia through its gear (the model's)
        self.observer = OBS.Observer(m, self.d, self.dof, self.base_dof, self.pelvis_id, "imu_in_pelvis", self.armature)
        self.sounds = SND.Sounds(self)                                      # the room's sounds from its physics (W5)
        self.heat = np.zeros(len(JOINTS))                                   # each motor's temperature above the room's (degC)
        self.tract = Tract(seed=self.seed)                                   # the voice effector's physics (4.9)
        self.ears = EA.Ears()                                                # two cochleas at the head's ear sites (P2)
        self.lane = None                                                     # the parent's lane (her voice, the words, her face)
        self.words_out = None                                                # the child's words' output this tick (to her ear)
        self.tract_pa = np.zeros(EA.TICK)                                    # the tract's sound this tick (Pa at 1 m)
        self.tract_raw = np.zeros(EA.TICK)                                   # the same in the engine's units (her transcriber's)
        self.crying = False                                                  # the cord's born cry pushed the tract this tick
        self.vor_corr = np.zeros(4)      # the flocculus's gain correction (yaw, pitch) and offset (yaw, pitch), held through a tick
        self._vor_slip = None; self._vor_turn = None   # A174: the last tick's retinal slip and head turn (yaw, pitch; rad), the flocculus's teacher
        cl_, cr_, cc_ = m.camera("eye_L").id, m.camera("eye_R").id, m.camera("eye_C").id   # A174: the eyes' separation and the cameras' lever
        self.eye_ipd = float(np.linalg.norm(m.cam_pos[cl_] - m.cam_pos[cr_]))                # about the trunk's yaw axis (the parallax of a near
        jy_ = m.joint("waist_yaw_joint").id                                                   # fixation: the slip a gain-1 VOR leaves), as
        mujoco.mj_forward(m, self.d)                                                          # the born pose stands: the camera's distance from
        ax_ = np.asarray(self.d.xaxis[jy_], float); ax_ /= max(float(np.linalg.norm(ax_)), 1e-9)   # the yaw axis's line (its anchor and axis
        dv_ = np.asarray(self.d.cam_xpos[cc_], float) - np.asarray(self.d.xanchor[jy_], float)      # in the world)
        self.cam_lever = float(np.linalg.norm(dv_ - ax_ * float(dv_ @ ax_)))
        self.night = False
        self.dawn_left = 0
        self.carried = []                                               # A110: the child carried to the mat at a dawn (tick, from, to)
        self.tidied = []                                                # A117 (B8): the lost toys put back at a dawn (tick, toy, from, to)
        self.light_day = {f: getattr(m, f).copy() for f in ("light_diffuse", "light_ambient", "light_specular")}
        self._below_n = 0                                                    # sub-steps called this life (an instrument)
        # birth
        self.tick = 0
        self.paused = False
        self._apply_limits()
        self.scene.birth()
        self.blind, self.blind_pairs = rest_blind(m, d, self.scene.g1_set)     # A12: the pairs pressing at rest (none as born)
        self._last_acts = {}
        self._spinal = {}; self._vor_quick = 0
        self._tendon = np.zeros(len(JOINTS), int)                          # A139: the tendon organ's inhibition, a countdown per joint
        self._grasp_hab = {h: [0, 0] for h in self.palm_of_hand}            # A162: the grasp's habituation per hand [ticks pressed, ticks free]
        self._dorsal_hab = {h: [0, 0] for h in self.palm_of_hand}           # A171: the dorsal response's, the same bookkeeping
        self._traction_hab = {a: [0, 0] for a in ("arm_l", "arm_r")}       # A177: the traction response's, the same bookkeeping
        self._forearm_body = {"arm_l": int(self.m.body("left_elbow_link").id), "arm_r": int(self.m.body("right_elbow_link").id)}   # A177: the forearm links her pull holds
        self.parent = None
        self._sense_birth()
        if parent:                                                      # THE PARENT'S MOTION (W2; body/sim/parent_motion.py): her
            from body.sim import parent_motion as PM                    # acts, holds and yield, run inside the tick's physics
            self.parent = PM.ParentMotion(self)
        self.timing = {"ticks": 0, "apply_s": 0.0, "physics_s": 0.0, "parent_s": 0.0}   # wall clock, an instrument (never saved or sensed)

    # ---------------------------------------------------------------- the servo law
    def _set_servo_law(self):
        """each G1 actuator a position servo at UNITREE'S OWN GAINS (A39, the lead's decision of 2026-09-25 read from the sources,
        replacing the first law's kp = limit / 0.25 rad, which was ours and 3-15 times stiffer): servo_gains()"""
        m = self.m
        if SERVO_LAW == "unitree":
            kp, kv = servo_gains()
            hand = np.array([k is None for k in kp])
            kp = np.array([self.tau_max[i] / STEP_BIG if h else k for i, (k, h) in enumerate(zip(kp, hand))], float)
        else:
            kp, kv = first_law_gains(self.tau_max)
        for a, p, v in zip(self.aid, kp, kv):
            m.actuator_gainprm[a, :] = 0.0; m.actuator_gainprm[a, 0] = p
            m.actuator_biasprm[a, :] = 0.0; m.actuator_biasprm[a, 1] = -p; m.actuator_biasprm[a, 2] = -v
        self.kp, self.kv = kp, kv

    def passive_torque(self, q):
        """A143: the passive tissues' torque at every joint for the angles `q` (JOINTS order): zero over the middle of the range, and in the last
        PASSIVE_MARGIN of the range at either end a push back toward the middle rising linearly to PASSIVE_FRAC x tau_max at the stop"""
        s = (q - self.lo) / np.maximum(self.hi - self.lo, 1e-9)
        hi_ = np.clip((s - (1.0 - PASSIVE_MARGIN)) / PASSIVE_MARGIN, 0.0, 1.0)
        lo_ = np.clip((PASSIVE_MARGIN - s) / PASSIVE_MARGIN, 0.0, 1.0)
        return PASSIVE_FRAC * self.tau_max * self.passive_mask * (lo_ - hi_)

    def limits_now(self):
        """the torque limits this tick: the declared limits (the weakness that followed the charge is gone with it, A88)"""
        return self.tau_max.copy()

    def _apply_limits(self):
        lim = self.limits_now()
        self.m.jnt_actfrcrange[self.jid, 0] = -lim
        self.m.jnt_actfrcrange[self.jid, 1] = lim
        self.m.jnt_actfrclimited[self.jid] = 1

    # ---------------------------------------------------------------- the world interface
    def frame(self):
        """the world at this tick AS THE G1'S ANATOMY MEETS IT (S5a; body/sim/anatomy.py's contract, every channel at its size, raw and
        unit-scaled here; no sim time passes, nothing drawn):
          body 242        per joint in BODY_JOINTS' order [sin, cos of the angle scaled over its range to -pi/2..pi/2, velocity, servo
                          effort (the actuator torque over the limit it was clamped at), the motor's heat over HEAT_RISE_C], the
                          gaze's state and velocity (6), the tract's 10 positions, 10 velocities and breath left (21)
          touch 130       the Dex3's 16 zones [log(1 + F / 1 N), its onset]; per joint [the born observer's outside torque (the
                          tick's mean of its 10 ms samples, from the robot's own encoders, motor torques and pelvis unit: observer.py)
                          over the joint's declared limit, its onset: the rise of its size]; the base's outside wrench as the observer
                          estimates it, force then torque in the pelvis's frame, each [its tick's mean over the body's weight, its onset]
          pain 44         per joint 1 where the gear's load (the sensed torque less the rotor's inertia times the encoder's velocity
                          change) as a 10 ms mean passed the joint's declared limit (A73), then the base's (the observer's outside force
                          in a 10 ms sample past F_pain) (A37)
          vestibular 24, imu_torso 6 (the torso unit's accelerometer and gyro, the tick's means of the noisy samples)
        and by day only (at night the eyes and ears are off and the parent asleep: those channels absent, quiet):
          words           the words channel's symbol this tick (the parent's word token as its sound ends, or letters), 0 the rest
          face 2          [2 x (smile - frown) of the face she shows while with the child, its change] (A1, A49, A158: the lane's)
          ears 1,725      both cochleas and the delay lines (body/sim/ears.py) of the tick's sounds; sound_side [an onset heard,
                          the born lateral read's angle, + left]
          eye_p 172, eye_f 1,536, onset_periph 3   the eyes (body/sim/eyes.py when attached: the D435's three views, A78, and
                          the born visual onset cue, A43; zeros without them)
          face_periph 3   the born face cue's stand-in (A157; eyes.face_cue): [fired, yaw, pitch], 1 and her mouth's direction from the
                          fovea's centre while her face lies in an eye's image under A1's conditions but the window's; zeros without the eyes
          face_fovea 1    zeros: no born face detector on the pixels at birth (C39, the lead's decision of 2026-09-25, option a; its
                          route to faces, her voice and her face brought into view, measured over days 49 to 55: her face seen on 1
                          to 3% of ticks, her smiles paying nothing; A157 reversed it)"""
        if self.paused:
            raise RuntimeError("G1World: a frame taken while the world is paused (the night)")
        d, s = self.d, self._sensed
        q = d.qpos[self.qadr]
        mid, half = (self.hi + self.lo) / 2, (self.hi - self.lo) / 2
        ang = (q - mid) / half * (math.pi / 2)
        effort = d.qfrc_actuator[self.dof] / self.m.jnt_actfrcrange[self.jid, 1]     # over the limit the torque was clamped at
        body = np.concatenate([np.stack([np.sin(ang), np.cos(ang), d.qvel[self.dof], effort, self.heat / HEAT_RISE_C], axis=1).reshape(-1),
                               self.gaze, self.gaze_v, self.tract.proprio()])
        touch = np.concatenate([np.stack([s["touch_log"][self.dex], s["touch_onset"][self.dex]], 1).reshape(-1),
                                np.stack([s["obs_j"], s["obs_j_on"]], 1).reshape(-1),
                                np.stack([s["obs_b"], s["obs_b_on"]], 1).reshape(-1)])
        pain = np.concatenate([(s["bd_peak"] > self.tau_hold).astype(np.float64), [float(s["base_peak"] > self.f_pain)]])
        strain = np.array([float((s["bd_peak"] / self.tau_hold).max() > STRAIN_LINE)])   # A191: a gear loaded near its pain line
        obs = {"body": body, "touch": touch, "pain": pain, "strain": strain, "vestibular": s["vestibular"].copy(), "imu_torso": s["imu_torso"].copy()}
        face = 0.0
        if not self.night:
            ln = self.lane
            word = 0 if ln is None else int(ln.word_now)
            fl = np.zeros(2) if ln is None else np.asarray(ln.face_seen, float)
            face = float(fl[0])
            obs.update(words=word, face=fl.copy(), ears=s["ears"].copy(), sound_side=s["sound_side"].copy(),
                       eye_p=np.zeros(AN.SIZES["eye_p"]), eye_f=np.zeros(AN.SIZES["eye_f"]), onset_periph=np.zeros(3),
                       face_periph=np.zeros(3), face_fovea=np.zeros(1))
        truth = self._truth()
        if self.eyes is not None and not self.night:                    # the eyes (W3r: the D435's three views at the anatomy's sizes,
            seen = self.eyes.see()                                      # body/sim/eyes.py; A38, A42)
            for k in ("eye_p", "eye_f", "face_fovea", "face_periph", "onset_periph"):
                obs[k] = seen[k]
            truth["eyes"] = seen["truth"]
        return Frame(self.tick, obs, face, truth)

    def _efference(self, acts):
        """each motor effector's own act this tick as joint settings, for the cerebellum's efference copy (7.5, A67), in the
        anatomy's order: the tract's 10, the gaze's 3, then the waist's, the arms', the hands' and the legs' (rest: every setting 2)"""
        out = []
        for name, n in ((VOICE_NAME, len(AN.TRACT)), (GAZE_NAME, len(GAZE_JOINTS))) + tuple((nm, len(js)) for nm, js in G.EFFECTORS):
            a = acts.get(name)
            out.extend(act_digits(rest_id(n) if a is None else int(a), n))
        return np.array(out, float)

    def _outside(self):
        """this step's outside forces on the G1 (the observer's truth, A37): the generalized force of every contact row (J^T f) and
        of the parent's holds (outside forces on its links), on the 43 joints' dofs (N m) and the floating base's 6 (N, N m)"""
        m, d = self.m, self.d
        qf = np.zeros(m.nv)
        if d.nefc:
            ty = d.efc_type
            cm = (ty == int(mujoco.mjtConstraint.mjCNSTR_CONTACT_FRICTIONLESS)) | (ty == int(mujoco.mjtConstraint.mjCNSTR_CONTACT_PYRAMIDAL)) \
                | (ty == int(mujoco.mjtConstraint.mjCNSTR_CONTACT_ELLIPTIC))
            if cm.any():
                mujoco.mj_mulJacTVec(m, d, qf, np.where(cm, d.efc_force, 0.0))
        if self.parent is not None and self.parent.holds:               # her holds: the outside forces on the G1's links, as MuJoCo
            qh = np.zeros(m.nv)                                         # applies xfrc_applied (at each body's centre of mass)
            for bb in self.g1_bodies:
                fr = d.xfrc_applied[bb]
                if fr.any():
                    mujoco.mj_applyFT(m, d, fr[:3].copy(), fr[3:].copy(), d.xipos[bb].copy(), int(bb), qh)
            qf += qh
        b = self.base_dof
        return np.concatenate([qf[self.dof], qf[b:b + 6]])

    def _below_step(self, sub, own, imu_last, F_last, obs_last, lim):
        """one sub-step of the loop below the tick (7.5; SimWorld's contract): the SubFrame (the mossy fibres in the anatomy's order,
        the teacher, the limits, and at sub-step 0 the slip and turn: none until the eyes' W3 reopening measures the fovea's slip)
        handed to the body's hook, its torque added to each cerebellar joint's servo as the actuator's constant bias term"""
        d = self.d
        J = len(JOINTS)
        mossy = np.concatenate([own, d.qpos[self.qadr], d.qvel[self.dof], imu_last,
                                np.log1p(np.asarray(F_last, float)[self.dex] / TOUCH_UNIT_N),
                                obs_last[:J] / self.tau_max, obs_last[J:] / self.weight])
        ci = self.cereb_idx
        a = self.aid[ci]
        teach = np.clip(self.kp[ci] * (d.ctrl[a] - d.qpos[self.qadr[ci]]) - self.kv[ci] * d.qvel[self.dof[ci]], -lim[ci], lim[ci])
        slip_ = turn_ = None
        if int(sub) == 0 and self._vor_slip is not None:               # A174: at sub-step 0 the last tick's slip and turn (the flocculus's teacher)
            slip_, turn_ = list(self._vor_slip), list(self._vor_turn)
        sa = self.sub_tick(SubFrame(self.tick, int(sub), mossy.tolist(), teach.tolist(), slip_, turn_, lim[ci].tolist()))
        self._below_n += 1
        if sa is None:
            return
        if sa.torque is not None:
            self.m.actuator_biasprm[a, 0] = np.asarray(sa.torque, float)
        if sub == 0 and sa.vor_gain is not None:
            self.vor_corr = np.concatenate([np.asarray(sa.vor_gain, float), np.asarray(sa.vor_offset if sa.vor_offset is not None else (0.0, 0.0), float)])

    def apply(self, acts):
        """one tick (150 ms, 75 steps of 2 ms) with the body's acts (S5a: body/core/world.py `Acts`): {effector name: flat act} for the
        tract ("voice"), the words' output ("words"), the gaze and the joint effectors, with `acts.cord` (the born patterns' steps below
        the gates, per joint, added to the targets the acts re-anchor) and `acts.vor` (the body's VOR: its gain and quick phase). The
        hands' acts pass the spinal cord first (the palmar grasp). Every 10 ms (5 steps) the loop below the tick is called when a body
        hooks it (`below`). The tract sounds and the ears hear the tick (by day). A tick MuJoCo cannot live raises WorldFault and
        leaves the world where the tick began (A18)."""
        if self.paused:
            raise RuntimeError("G1World: the world moved while paused (the night)")
        m, d = self.m, self.d
        t_apply = time.perf_counter(); t_phys = 0.0
        cord = dict(getattr(acts, "cord", None) or {})
        cry_flag = getattr(acts, "crying", None)                       # A173: the body's word on whether the cord's cry stepped (read before
                                                                        # the acts become a plain dict below)
        if hasattr(acts, "vor"):                                        # the body's own VOR (the core's Acts.vor: empty when it has none)
            vora = dict(acts.vor or {})
        else:                                                           # a plain dict of acts (an instrument's, a test's): the born VOR
            vora = {GAZE_NAME: {"axes": [0, 1], "gain": VOR_GAIN, "quick": 0.5}}
        acts = dict(acts or {})
        spinal = {}
        if self.spinal:                                                 # THE SPINAL CORD: the palmar grasp on each hand's own act
            for hand, z in self.palm_of_hand.items():
                i_h = 0 if hand == "hand_l" else 1
                a, ev = R.grasp(hand, acts.get(hand), float(self._sensed["palm_log"][i_h]),   # A171: the palm's side (its face and the
                                self._grasp_hab.setdefault(hand, [0, 0]),   # A162: the reflex habituates under a constant pressure   # fingers'
                                float(self._sensed["palm_onset"][i_h]))     # A164: and a new press wakes it                           # insides)
                if ev is not None:
                    spinal[hand] = ev
                    if a is not None:                                       # (habituated at rest: no act of the hand's this tick)
                        acts[hand] = a
                if ev in (None, "habituated"):                              # A171: the grasp not firing: a push on the back of the hand
                                                                            # (a toy set against a fist's knuckles) opens it, unless the
                    a2, ev2 = R.dorsal_open(hand, acts.get(hand), float(self._sensed["sides"][i_h, 1]),   # palm's side is pressed by
                                            float(self._sensed["sides"][i_h, 0]),                           # anything but its own fingers
                                            self._dorsal_hab.setdefault(hand, [0, 0]))
                    if ev2 is not None:
                        spinal[hand] = ev2 if ev is None else ev + "+" + ev2
                        if a2 is not None:
                            acts[hand] = a2
            for arm, n_ in self._traction_N().items():                      # A177: THE TRACTION RESPONSE: a pull on the forearm along the arm
                a3, ev3 = R.traction(arm, acts.get(arm), float(n_), self._traction_hab.setdefault(arm, [0, 0]))   # flexes the elbow and
                if ev3 is not None:                                         # shoulder (Prechtl): the newborn's part in its own pull to sit
                    spinal[arm] = ev3 if arm not in spinal else spinal[arm] + "+" + ev3
                    if a3 is not None:
                        acts[arm] = a3
            if self.righting:                                           # and the prone pattern: face down, the arms flex under, the
                spinal.update(R.prone(acts, self._sensed["imu_torso"]))   # trunk yaws toward the side that is up (A92)
        if self.tone:                                                   # A185: THE RESTING TONE: each arm's proximal joints drawn toward
            q_ = d.qpos[self.qadr]                                      # their born resting angles by a saturating step, summed with the
            for limb in R.TONE_REST:                                    # cord's other patterns and the own act (reflexes.tone)
                t_ = R.tone(limb, q_[self.eff_slices[limb]])
                c_ = cord.get(limb)
                cord[limb] = t_ if c_ is None else tuple(float(a_) + float(b_) for a_, b_ in zip(c_, t_))
        posture_ = {}
        if self.standing:                                               # A193: THE STANDING AND STEPPING REFLEXES (reflexes.stand): upright
            q_ = d.qpos[self.qadr]                                      # on a loaded sole the legs carry the body and, the stance hip
            tf_ = self._sensed["touch_force"]                           # extended, step; lying or sitting, nothing
            st_, ev_ = R.stand({n_: q_[self.eff_slices[n_]] for n_ in R.STAND_LIMBS}, self._sensed["imu_torso"],
                               (float(tf_[self.sole_zones[0]].sum()), float(tf_[self.sole_zones[1]].sum())), self._stepping)
            posture_ = R.posture(ev_, {n_: q_[self.eff_slices[n_]] for n_ in R.STAND_LIMBS}, self._sensed["imu_torso"], getattr(self, "_tone_on", False)) if self.posture else {}
            sup_n_ = float(np.asarray(self._sensed["touch_force"], float)[self._trunk_zones].sum())
            self._sup_lp = getattr(self, "_sup_lp", 0.0) + R.SUPPORT_LP * (sup_n_ - getattr(self, "_sup_lp", 0.0))   # (the hold felt, smoothed over
            sup_n_ = self._sup_lp                                           # about half a second: her steadying wavers 40 to 150 N in a walk)
            self._tone_w = float(np.clip(1.0 - (sup_n_ - R.SUPPORT_N) / (R.SUPPORT_FULL_N - R.SUPPORT_N), 0.0, 1.0))   # C287: the tone's share: all of it
            if posture_ and self._tone_w <= 0.0:                            # unheld, none of it at a full hold felt on its trunk
                posture_ = {}                                               # (SUPPORT_FULL_N), in between as the hold eases: the legs'
                                                                            # postural responses fall away as the body is held by a
                                                                            # support (Cordo and Nashner 1982; recalled), by degrees: let
                                                                            # go from a full hold it went down before its tone began
            self._tone_on = bool(posture_)              # A195: the limbs under the postural tone this tick

            for limb, t_ in st_.items():
                if limb in posture_:                                        # (the tone below the tick holds them; the standing
                    cord.setdefault(limb, tuple(0.0 for _ in t_))           # reflex's one step a tick is not added on top)
                else:
                    c_ = cord.get(limb)
                    cord[limb] = t_ if c_ is None else tuple(float(a_) + float(b_) for a_, b_ in zip(c_, t_))
                spinal[limb] = ev_[limb] if limb not in spinal else spinal[limb] + "+" + ev_[limb]
        own = self._efference(acts)                                     # the efference copy (the own acts, after the grasp's sum)
        steps = {}                                                      # every act read before anything moves (a bad act moves nothing)
        for name, js in G.EFFECTORS:
            a = acts.get(name); c = cord.get(name)
            if (a is not None and int(a) != EFFECTOR_REST[name]) or c is not None:
                st_ = np.array([SETTINGS[k] for k in act_digits(EFFECTOR_REST[name] if a is None else a, len(js))])
                if c is not None:
                    st_ = st_ + np.asarray(c, float)
                steps[name] = st_
        if self.spinal:                                                 # A139: THE TENDON ORGAN: a joint at its load line last tick has
            loaded = self._sensed["bd_peak"] >= self.tau_hold          # its drive relaxed this tick (the withdrawal's own blocked push
            ev = R.tendon(steps, loaded, self._tendon, self.eff_joint_idx)   # made day 26's 440-tick elbow storm, C113)
            for eff, js in ev.items():
                spinal[eff] = "tendon" if eff not in spinal else spinal[eff] + "+tendon"
                self.stats_tendon = getattr(self, "stats_tendon", 0) + len(js)
        a = acts.get(GAZE_NAME)
        gaze_step = np.zeros(3) if a is None else np.array([GAZE_SETTINGS[j][k] for j, k in enumerate(act_digits(a, len(GAZE_JOINTS)))])
        cg_ = cord.get(GAZE_NAME)                                         # A186: the born saccade, summed with the gaze's own step (the
        if cg_ is not None:                                               # brainstem's saccade generator: the colliculus's command and the
            gaze_step = gaze_step + np.asarray(cg_, float)                # cortex's add)
        va = acts.get(VOICE_NAME)
        vdig = None if va is None or int(va) == VOICE_REST else act_digits(va, len(AN.TRACT))
        wo = acts.get(WORDS_NAME)
        start = self._capture(fast=True)                                # A187: the tick's start kept raw (canonical only if a fault restores it)
        warn0 = [int(d.warning[i].number) for i in range(int(mujoco.mjtWarning.mjNWARNING))]
        par = self.parent
        t_par = time.perf_counter()
        if par is not None:
            par.tick_begin()                                            # her acts advance; her pose at the tick's end (L1)
        t_par = time.perf_counter() - t_par
        lim = self.limits_now()
        rest_idx = []
        q = d.qpos[self.qadr]
        for name, js in G.EFFECTORS:
            sl = self.eff_slices[name]
            if name not in steps:
                rest_idx.extend(range(sl.start, sl.stop))
                continue
            d.ctrl[self.aid[sl]] = np.clip(q[sl] + steps[name], self.lo[sl], self.hi[sl])
        self._last_acts = _canon_acts(acts)
        self._spinal = spinal
        gaze0 = self.gaze.copy()
        self.gaze = clamp_gaze(self.gaze + gaze_step)                  # the gaze's act: the windows jump at the tick's start
        rest_a = self.aid[rest_idx] if rest_idx else None
        rest_q = self.qadr[rest_idx] if rest_idx else None
        n, nz, J, W_ = STEPS_PER_TICK, self.nz, len(JOINTS), OBS_WINDOW_STEPS
        F = np.zeros((n, nz)); imu = np.zeros((n, 12)); BD = np.zeros((n, J))
        OW = np.zeros((n // W_, J + 6)); TW = np.zeros((n // W_, J + 6))   # each 10 ms sample: the observer's estimate, and the truth
        obsv = self.observer                                                # (an instrument's: the world's own contact torques)
        heat_in = np.zeros(J)
        own_p = np.zeros(2)                                            # the palms' own-hand force
        sides = np.zeros((2, 2))                                       # A171: the push on each hand's two sides
        alpha = self.alpha
        below = self.below
        s0 = self._sensed
        imu_last, F_last, obs_last = s0["imu_last"].copy(), s0["F_last"].copy(), s0["obs_last"].copy()   # (obs_last: the observer's last
                                                                                                        # sample, the mossy fibres')
        try:
            for s in range(n):
                if below is not None and s % W_ == 0:
                    self._below_step(s // W_, own, imu_last, F_last, obs_last, lim)
                if posture_ and s % W_ == 0:                             # A195: THE POSTURAL TONE AND THE VESTIBULOSPINAL REFLEX, every 10 ms:
                    pit_ = self._vest_pitch                             # the pelvis's lean as the vestibular sense holds it (below)
                    sup_ = False; tw_ = float(getattr(self, "_tone_w", 1.0))
                    for limb, (ref_, ank_, hip_, roll_) in posture_.items():
                        sl = self.eff_slices[limb]
                        q_ = d.qpos[self.qadr[sl]]
                        off_ = R.posture_step(q_, d.qvel[self.dof[sl]], ref_, steps.get(limb, 0.0), ank_, pit_, float(imu_last[10]), hip_, sup_, roll_, tw_)
                        d.ctrl[self.aid[sl]] = np.clip(q_ + off_, self.lo[sl], self.hi[sl])
                if rest_a is not None:
                    d.ctrl[rest_a] += alpha * (d.qpos[rest_q] - d.ctrl[rest_a])
                d.qfrc_applied[self.dof] = self.passive_torque(d.qpos[self.qadr])   # A143: the passive end-range stiffness, the tissue's push
                if par is not None:
                    t0 = time.perf_counter()
                    par.before_step(s)                                  # her segments drawn, her holds' capped springs (L0)
                    t_par += time.perf_counter() - t0
                t0 = time.perf_counter()
                mujoco.mj_step(m, d)
                t_phys += time.perf_counter() - t0
                self.sounds.step(self, s)                               # the step's impacts (W5: body/sim/sounds.py)
                BD[s] = obsv.gear_load(d.qfrc_actuator[self.dof], d.qvel[self.dof])   # THE GEAR'S LOAD from the sensed torque and the
                                                                                       # encoder's velocity change (A37, A73)
                if par is not None:
                    t0 = time.perf_counter()
                    par.after_step(s)                                   # her contacts with the room: her stop (L0)
                    t_par += time.perf_counter() - t0
                F[s] = self._zone_forces()
                own_p += self._palm_own
                sides += self._sides
                imu[s] = self._imu_noisy(d.sensordata[self.imu_adr])
                if self.posture:                                        # A195: the lean the reflex reads (kept at every step, standing or not): the pelvis gyro's turn about
                    self._vest_pitch += float(imu[s, 10]) * m.opt.timestep   # the side-to-side axis summed (the canals), drawn slowly
                    an_ = float(np.linalg.norm(imu[s, 6:9]))            # (R.VEST_TAU) toward the accelerometer's tilt when it reads
                    if 0.8 * 9.81 < an_ < 1.2 * 9.81:                   # about one g (the otoliths): one sample of the accelerometer
                        self._vest_pitch += (math.atan2(-float(imu[s, 6]), float(imu[s, 8])) - self._vest_pitch) * m.opt.timestep / R.VEST_TAU   # alone read 10 to 40 deg off a body swaying 2 deg
                tq = d.qfrc_actuator[self.dof]
                heat_in += (tq / self.tau_max) ** 2
                imu_last, F_last = imu[s], F[s]
                if (s + 1) % W_ == 0:                                   # THE OBSERVER'S 10 ms SAMPLE, from the sensed signals alone
                    k_ = s // W_
                    r_, wb_ = obsv.sample(d.qpos[self.qadr], d.qvel[self.dof], d.qpos[self.base_dof + 3:self.base_dof + 7], imu[s, 6:9],
                                          imu[s, 9:12])
                    OW[k_] = np.concatenate([r_, wb_])
                    obs_last = OW[k_].copy()
                    tr_ = self._outside()                               # the truth at the sample (the instruments' and the parent's
                    Rp_ = d.xmat[self.pelvis_id].reshape(3, 3)          # eyes', never the body's)
                    TW[k_] = np.concatenate([tr_[:J], Rp_.T @ tr_[J:J + 3], tr_[J + 3:]])
            mujoco.mj_forward(m, d)                                     # the tick's end: its accelerations, contacts and sensors
            if par is not None:
                par.tick_end()
        except mujoco.FatalError as e:
            self._restore(self._from_fast(start))
            raise WorldFault(self.tick, f"MuJoCo stopped: {e}")
        warned = {mujoco.mjtWarning(i).name: int(d.warning[i].number) - warn0[i]
                  for i in range(int(mujoco.mjtWarning.mjNWARNING)) if int(d.warning[i].number) != warn0[i]}
        bad = self._unsound()
        if warned or bad:
            self._restore(self._from_fast(start))
            said = f" ({MUJOCO_MESSAGES[-1]})" if MUJOCO_MESSAGES and warned else ""
            raise WorldFault(self.tick, f"MuJoCo's warnings {warned}{said}" if warned else f"the tick's end state: {bad}")
        # the voice: the tract sounds this tick (its act, the cord's cry below it), heard with the parent's voice by day
        cv = cord.get(VOICE_NAME)
        self.crying = bool(cry_flag) if cry_flag is not None else \
            (cv is not None and bool(np.any(np.asarray(cv, float) != 0.0)))    # the born cry sounding (world truth of a sound; A173: the
                                                                                # breath's steps are not a cry, so the body says which)
        self.tract_raw = np.asarray(self.tract.tick(vdig, cord=cv), float)
        self.tract_pa = self.tract_raw * PA_PER_UNIT
        self.words_out = None if wo is None or int(wo) == 0 else int(wo)
        heard = None
        room = self.sounds.tick_end(self)                               # the room's sounds this tick (W5), made day and night
        if not self.night:
            tp = d.xpos[m.body("torso_link").id]; tR = d.xmat[m.body("torso_link").id].reshape(3, 3)
            ear_l, ear_r, mouth = EA.head_from_torso(tp, tR)
            sources = {"tract": (self.tract_pa, mouth)}
            sources.update(room)
            if self.lane is not None:
                sources.update(self.lane.tick(self))                    # her tick: her conduct on this tick, her voice from her
                                                                            # mouth (body/sim/lane.py)
            heard = self.ears.tick(ear_l, ear_r, sources)
        # the tick's senses
        self._sense_tick(F, imu, own_p / n, OW, heard, BD, TW, sides=sides / n)
        vg = vora.get(GAZE_NAME)
        if vg is not None:                                              # THE BODY'S VOR (the core's Acts.vor) with the flocculus's
            gain = (float(vg["gain"]) + self.vor_corr[0], float(vg["gain"]) + self.vor_corr[1])   # correction and offset
            reach = (GAZE_REACH_YAW - self.gaze[2] / 2, GAZE_REACH_PITCH)
            yaw, pitch, quick = vor(self.gaze[0], self.gaze[1], imu[:, 3:6] @ self.gyro_to_cam.T, m.opt.timestep, gain=gain,
                                    reach=reach, offset=(self.vor_corr[2], self.vor_corr[3]), quick_frac=float(vg["quick"]))
            self.gaze = clamp_gaze([yaw, pitch, self.gaze[2]])
            self._vor_quick = int(quick)
        else:
            self._vor_quick = 0
        self.gaze_v = (self.gaze - gaze0) / TICK_S
        # A174 (2026-10-02): THE RETINAL SLIP TEACHES THE VOR. The flocculus's climbing fibres carry the image's motion left on the retina
        # after the VOR's counter-turn (Ito 1982; the cerebellum's VOR lesson, body/core/cerebellum.py, learns down its gradient); the
        # SubFrame has carried None since R6c ('none until the eyes' W3 reopening measures the fovea's slip'), so in 64 days the flocculus
        # learned nothing (its lesson count 0 against 46.8 million at the limbs: the audit of 13:45). The slip is read kinematically, as
        # the test head reads it (body/tests/test_cerebellum.py Head: slip = -magnification x the true turn - the counter-shift; the
        # magnification here the parallax of the fixated distance, 1 + the cameras' lever about the trunk's yaw axis over the vergence's
        # distance, 1 for a far thing): the head's rotation over the tick in the camera's frame (the gyro's samples, yaw about the camera's y, pitch
        # about its x) and the gaze windows' shift the VOR made this tick, both counted from after the gaze's own jump (the saccade
        # is not in the slip: a gaze that acts every tick, as this child's does, would otherwise never teach it); a tick on which the VOR
        # jumped (its quick phase) teaches nothing and hands None, as the SubFrame's contract says. A thing held still in the fovea is assumed (the room
        # stands still; her hands and the toys in motion are the noise the lesson averages out)
        if vg is not None:
            g_s = clamp_gaze(gaze0 + gaze_step)                          # the gaze after its act, before the VOR
            yp_, pp_, _q = vor(g_s[0], g_s[1], imu[:, 3:6] @ self.gyro_to_cam.T, m.opt.timestep, gain=(1.0, 1.0), reach=None)
            perfect_ = np.array([yp_, pp_])                              # where a VOR of gain 1 with no offset would have put the windows:
            turn_ = -(perfect_ - g_s[:2])                                # the head's turn in the windows' own axes (yaw, pitch) is its negative,
            shift_ = self.gaze[:2] - g_s[:2]                             # and the slip is what the real counter-shift left of it, the fixated
            verg_ = float(g_s[2])                                        # thing's parallax included: the cameras ride the trunk's yaw axis at
            d_fix = self.eye_ipd / (2.0 * math.tan(0.5 * verg_)) if verg_ > 1e-3 else float("inf")   # a lever, so a thing fixated at the
            mag_ = 1.0 + (self.cam_lever / d_fix if np.isfinite(d_fix) else 0.0)   # vergence's distance moves on the retina by more than
            if int(self._vor_quick) == 0:                                # the turn (a gain over 1 is asked of the VOR for near things,
                self._vor_slip = -mag_ * turn_ - shift_; self._vor_turn = turn_   # as of the eye: Viirre et al. 1986). The gaze's own act is
                                                                         # not in this slip (both counts start after its jump); a quick
            else:                                                        # phase corrupts the counter-shift and hands None)
                self._vor_slip = None; self._vor_turn = None
        else:
            self._vor_slip = None; self._vor_turn = None
        k = 1.0 - math.exp(-TICK_S / HEAT_TAU_S)                       # the motors' heat: first order toward the tick's load
        self.heat += (HEAT_RISE_C * heat_in / n - self.heat) * k
        if self.dawn_left > 0:                                          # the morning's light returning
            self.dawn_left -= 1
            f_ = NIGHT_LIGHT + (1.0 - NIGHT_LIGHT) * (DAWN_TICKS - self.dawn_left) / DAWN_TICKS
            for fld, v in self.light_day.items():
                getattr(m, fld)[...] = v * f_
        if self.lane is not None:
            self.lane.after_apply(self)                                 # her conduct's tick, on what the world just did
        self.tick += 1
        self.timing["ticks"] += 1; self.timing["physics_s"] += t_phys; self.timing["apply_s"] += time.perf_counter() - t_apply
        self.timing["parent_s"] += t_par

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
        """a frozen night (the G1's world runs through the night instead, live_night: dusk() and dawn()); kept for the tests"""
        self.paused = True

    def resume(self):
        """the morning of a frozen night: on from the same state"""
        self.paused = False

    live_night = True                                                   # the G1's world runs through the night (R8c; 5.4, A46)

    def dusk(self):
        """THE NIGHT FALLS (5.4, A46; SimWorld's contract): the room's lights dim to NIGHT_LIGHT of the day's (never off), the eyes
        and the ears are off (the frame carries the body's own senses alone), the parent goes to sleep on the sofa touching nothing
        (every hold of hers released; the lane's conduct sleeps with her). The physics runs on,
        stepped by the body tick by tick"""
        m = self.m
        self.night = True
        self.dawn_left = 0
        for fld, v in self.light_day.items():
            getattr(m, fld)[...] = v * NIGHT_LIGHT
        if self.parent is not None:
            self.parent.sleep()
        if self.lane is not None:
            self.lane.dusk(self)

    def carry_to_mat(self, to=None):
        """A110 (2026-09-27, C92): a child that has rolled off the mat is carried back onto it in its sleep, as a person carries a
        sleeping baby to its bed: at dawn, before the light, its body is set down at the mat's centre on its back in the birth pose
        (amended 2026-09-27: laid as it lay, on its side, it woke looking at the floor), settled under its servos, its velocities
        zero, and the toys stay where they are. Why: life day 6 the child rolled off the mat (dawn 8: its pelvis at (1.57, 0.22), the mat's edge at 1.4), day 7
        through the door into the hall (dawn 9: (3.41, -1.56); the room's edge 2.65), where no toy lay within two metres, her
        planner found no spot to kneel ("no path on the floor") and she counted as away (lane.ROOM_EDGE_X): no reach, no lesson,
        no judgment of its acts for two days. Nothing here reaches the body but the morning's new view. Logged in `carried`
        (the tick, from, to). -> whether it was carried"""
        m, d = self.m, self.d
        g = m.geom("mat").id
        c, h = m.geom_pos[g][:2].copy(), m.geom_size[g][:2]
        xy = d.qpos[:2].copy()                                          # its pelvis (the floating base) on the floor plan
        on_mat = abs(xy[0] - c[0]) <= h[0] and abs(xy[1] - c[1]) <= h[1]
        supine = float(d.xmat[m.body("torso_link").id].reshape(3, 3)[2, 0]) > 0.6   # its chest's normal up: on its back (the posture
                                                                        # law of parent_motion.Child: fz > 0.6)
        twisted = self._twisted()                                       # A110 amended a fourth time (2026-09-27 23:40, C103): a joint
        if to is None and on_mat and supine and not twisted:            # at its stop while it sleeps (the dawn-16 pair: the waist yaw
            return False                                                # 2.62, the left hip pitch 2.88, the right wrist roll -1.98; it
                                                                        # woke to 176 pain ticks in 500) is laid straight like the rest:
                                                                        # a carer straightens a baby sleeping twisted. On the mat, on
                                                                        # its back, its joints off their stops: left where it lies
        to = np.asarray(to, float) if to is not None else (c if not on_mat else xy)   # (C281: or laid where the caller says) (A110 amended 2026-09-27 16:10: a child asleep on its side or
        joints = dict(G.BIRTH, waist_yaw=0.0, waist_roll=0.0, waist_pitch=0.0,   # laid STRAIGHT: the birth's limbs and the waist at
                      ankle_roll=0.0, wrist_pitch=0.0, wrist_yaw=0.0)           # rest, and the joints the birth pose does not name at
                                                                        # the model's rest too (2026-09-28 01:25: the dawn-17 pair's
                                                                        # left ankle roll and wrist yaw stayed at their stops through
                                                                        # the laying)
        self.scene.set_g1(joints, root=np.r_[to[0], to[1], 1.0, G.kin.mjquat(G.kin.ry(-math.pi / 2))])   # rest (2026-09-27 17:50: the
                                                                        # dawn-13 pair's waist yaw was 2.59 rad, the torso twisted almost
                                                                        # backwards on a supine pelvis, so the laid child read "front";
                                                                        # a person lays a baby straight). Front ON the mat is laid on its
        for b_ in self.scene.g1_set:                                    # (C281: no hold's force of the tick before goes with it)
            d.xfrc_applied[b_] = 0.0
        mujoco.mj_forward(m, d)                                         # pose (A110 amended 2026-09-27: a sleeping baby is laid on its
        d.qpos[2] += (.012 + .004) - self.scene.lowest_g1_point()      # back, never on its side or front; life dawn 11 laid it as it
        d.qvel[:] = 0.0                                                 # lay, on its side, and it woke looking at the floor, C94, with
        mujoco.mj_forward(m, d)                                         # no reward possible all morning), set down on the mat as at
        b = self.scene.bmap                                             # birth (g1scene.place_on_mat) and settled under its servos
        her_q = d.qpos[b.qadr].copy()                                   # holding the pose, the parent held where she sleeps, the toys
        m.actuator_biasprm[:, 0] = 0.0                                  # (2026-09-27 17:45: the servos' constant bias carries the
        d.qacc_warmstart[:] = 0.0                                       # cerebellum's last torque of the dusk, MUTABLE_MODEL_FIELDS; under
                                                                        # it the settle writhed and rolled the laid child back onto its
                                                                        # front on the life's dawn-13 pair; the cerebellum writes it anew
                                                                        # each waking tick)
        for _ in range(int(round(G.BIRTH_SETTLE_S / m.opt.timestep))):  # where they are (no reset of the world)
            mujoco.mj_step(m, d)
            d.qpos[b.qadr] = her_q
            d.qvel[b.vadr] = 0.0
        d.qvel[:] = 0.0
        mujoco.mj_forward(m, d)
        self.carried.append((int(self.tick), [float(xy[0]), float(xy[1])], [float(to[0]), float(to[1])]))   # back where it lies, too)
        return True

    def _twisted(self):
        """whether a joint of the child's trunk or of a limb's root (TWIST_JOINTS: the waist, the hips, the shoulders) rests within
        TWIST_MARGIN_RAD of a stop of its range (C103): the actor pins a joint at its stop (C100's waist yaw at 2.62 for days); a body
        asleep twisted against those stops is straightened by the carry"""
        m, d = self.m, self.d
        for j in range(m.njnt):
            if m.jnt_type[j] != mujoco.mjtJoint.mjJNT_HINGE or not m.jnt_limited[j]:
                continue
            if int(m.jnt_bodyid[j]) not in self.scene.g1_set:               # the child's own joints (its bodies: g1scene.g1_set)
                continue
            nm = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j) or ""
            if not any(k in nm for k in TWIST_JOINTS):
                continue                                                # the trunk and the limbs' roots only: a knee, an ankle, a wrist
                                                                        # or a finger at its stop does not twist the body (the dawn-17
                                                                        # pair: the left ankle roll and wrist yaw at theirs, laid for it)
            q = float(d.qpos[m.jnt_qposadr[j]]); lo, hi = m.jnt_range[j]
            if min(q - lo, hi - q) < TWIST_MARGIN_RAD:
                return True
        return False

    def dawn(self):
        """THE MORNING (5.4, A46): the light returns over the wake's first DAWN_TICKS ticks, the eyes and ears on, the parent awake;
        the world goes on from wherever the night left it, the child carried back onto the mat if it rolled off (A110)"""
        self.carry_to_mat()
        self.tidy_toys()
        self.bucket_beside()
        self.night = False
        self.dawn_left = DAWN_TICKS
        if self.parent is not None:
            self.parent.wake()
        if self.lane is not None:
            self.lane.dawn(self)

    def tidy_toys(self):
        """THE MORNING TIDY (B8, A117, C102): at a dawn, a toy the parent cannot get to (under solid furniture: a ray cast up from the
        floor beside it meets the room; or with no spot she can kneel at to pick it up, parent_motion._toy_spot) and out of the child's
        reach (0.5 m clear of it) is put back where it stood at birth (the model's qpos0), or on the first free point of a ring 0.3 m
        round that place, as someone tidies a room overnight; a toy the child can reach is never moved. The cup lay under the low table
        from life day 12 and the ball and the stacker in the room's corner behind the plant: her reach lessons asked for the cup 17 times
        in two days and were refused each time. Returns the toys moved; each is logged in `tidied` and saved"""
        from body.sim import parent_motion as PM
        m, d = self.m, self.d
        par = self.parent
        if par is None:
            return []
        par.child = PM.Child(m, d, self.scene.g1_set)
        gid = np.zeros(1, dtype=np.int32)

        def covered(xy):
            z = 0.02
            for _ in range(8):
                h = mujoco.mj_ray(m, d, np.array([xy[0], xy[1], z]), np.array([0.0, 0.0, 1.0]), None, 1, -1, gid)
                if h < 0 or z + h > 1.5:
                    return False
                g = int(gid[0]); b = int(m.geom_bodyid[g])
                if m.body_dofnum[b] == 0 and m.body_mocapid[b] < 0 and (m.geom_contype[g] or m.geom_conaffinity[g]):
                    return True
                z = z + h + 0.005
            return False
        moved = []
        for k, b in sorted(par.toys.items()):
            xy = d.xpos[b][:2].copy()
            if par.child.clearance_xy(xy) < 0.5 or k in par.holding.values():
                continue
            left = set(getattr(getattr(getattr(self, "lane", None), "conduct", None), "left", {}) or {})   # C139: a toy she left where it lay
            if not covered(xy) and par._toy_spot(xy) is not None and k not in left:                    # (her hand could not get to it) is
                continue                                                                               # put back too
            j = m.body_jntadr[b]; adr = m.jnt_qposadr[j]; dof = m.jnt_dofadr[j]
            home = m.qpos0[adr:adr + 7].copy()
            others = [d.xpos[b2][:2] for k2, b2 in par.toys.items() if k2 != k]
            place = None
            for ring, n in ((0.0, 1), (0.3, 8)):
                for i in range(n):
                    q = home[:2] + ring * np.array([math.cos(2 * math.pi * i / n), math.sin(2 * math.pi * i / n)])
                    if par.child.clearance_xy(q) >= 0.45 and all(np.linalg.norm(q - o) >= 0.15 for o in others) and par._in_plan(q) \
                            and par.plan.dist[par.plan.cell(q)] >= 0.10:
                        place = q; break
                if place is not None:
                    break
            if place is None:
                continue
            d.qpos[adr:adr + 2] = place; d.qpos[adr + 2] = home[2]; d.qpos[adr + 3:adr + 7] = home[3:7]
            d.qvel[dof:dof + 6] = 0.0
            self.tidied.append((int(self.tick), k, [float(xy[0]), float(xy[1])], [float(place[0]), float(place[1])]))
            moved.append(k)
        if moved:
            mujoco.mj_forward(m, d)
        return moved

    def bucket_beside(self):
        """THE BUCKET SET BESIDE THE CHILD AT DAWN (C130, an environment change disclosed; 2026-09-29): the hide game's container stands
        where it was left (she never carries it, C119), and life day 31's four hides were played 2.0 to 2.3 m from the child, which was
        never within reach of the bucket all day (its nearest 0.95 m): no hide could be found. Whoever tidies the room overnight sets
        the bucket within the child's reach at its side: at dawn, a bucket farther than BUCKET_NEAR_M from both its shoulders is set upright on
        the free point of a ring BUCKET_BESIDE_M round the point between its shoulders nearest one of them (off its body, on the floor she can plan on, clear of the other
        toys and of nothing solid above), or left where it stands when the ring has none; a toy lying in it stays in it. Logged in
        `tidied` as ("bucket", from, to). -> whether it was moved"""
        m, d = self.m, self.d
        par = self.parent
        if par is None or "bucket" not in par.toys or "bucket" in par.holding.values():
            return False
        b = par.toys["bucket"]
        sh = [d.xpos[m.body(f"{s}_shoulder_pitch_link").id][:2].copy() for s in ("left", "right")]
        chest = (sh[0] + sh[1]) / 2.0                                    # between its shoulders: where its arms reach from
        xy = d.xpos[b][:2].copy()
        if min(float(np.linalg.norm(xy - p_)) for p_ in sh) <= BUCKET_NEAR_M:
            return False
        par.child = PM_Child(m, d, self.scene.g1_set)
        inside = [k2 for k2, b2 in par.toys.items() if k2 != "bucket" and np.linalg.norm(d.xpos[b2][:2] - xy) < 0.12 and d.xpos[b2][2] < d.xpos[b][2] + 0.25]
        others = [d.xpos[b2][:2] for k2, b2 in par.toys.items() if k2 != "bucket" and k2 not in inside]
        cands = []
        for i in range(12):
            q = chest + BUCKET_BESIDE_M * np.array([math.cos(2 * math.pi * i / 12), math.sin(2 * math.pi * i / 12)])
            if par.child.clearance_xy(q) >= 0.12 and all(np.linalg.norm(q - o) >= 0.15 for o in others) and par._in_plan(q) \
                    and par.plan.dist[par.plan.cell(q)] >= 0.10:
                cands.append((min(float(np.linalg.norm(q - p_)) for p_ in sh), q))
        if not cands:
            return False
        place = min(cands, key=lambda x: x[0])[1]                        # the free point nearest one of its shoulders
        shift = place - xy
        for k2 in ["bucket"] + inside:                                   # the bucket and what lies in it, moved together, upright
            b2 = par.toys[k2]; j = m.body_jntadr[b2]; adr = m.jnt_qposadr[j]; dof = m.jnt_dofadr[j]
            d.qpos[adr:adr + 2] += shift; d.qvel[dof:dof + 6] = 0.0
            if k2 == "bucket":
                home = m.qpos0[adr:adr + 7]
                d.qpos[adr + 2] = home[2]; d.qpos[adr + 3:adr + 7] = home[3:7]
        mujoco.mj_forward(m, d)
        self.tidied.append((int(self.tick), "bucket", [float(xy[0]), float(xy[1])], [float(place[0]), float(place[1])]))
        return True

    def toys_beside(self, names, rings=(0.45, 0.57)):
        """C286: THE CARRY BRINGS ITS TOYS ALONG. After a carry back to its mat (C277, C281) the toys lay where the walk had left
        them: on the day-92 copy the child saw a toy on 40 of 500 ticks, her 'you see the X.' lines were refused 'not true', and
        her asks found nothing to ask about. Each named toy (the day's focus toys) that neither holds and that lies farther than
        the outer ring from both its shoulders is set upright on a free point of a ring round the point between its shoulders
        (off its body, 0.15 m from the other toys, on the floor she can plan on, 0.5 m from where she is), nearest a shoulder;
        whoever carries a child to its mat brings its toys. An environment's act, disclosed; logged in `tidied`. -> the toys moved"""
        m, d = self.m, self.d
        par = self.parent
        if par is None:
            return []
        par.child = PM_Child(m, d, self.scene.g1_set)
        sh = [d.xpos[m.body(f"{s_}_shoulder_pitch_link").id][:2].copy() for s_ in ("left", "right")]
        chest = (sh[0] + sh[1]) / 2.0
        her = np.asarray(par.base["at"], float)[:2]
        held = set(v for v in par.holding.values() if v is not None) | set(getattr(getattr(getattr(self, "lane", None), "_p", None), "child_holds", None) or ())
        moved = []
        for k in names:
            if k not in par.toys or k in held or k == "bucket":
                continue
            b = par.toys[k]; xy = d.xpos[b][:2].copy()
            if min(float(np.linalg.norm(xy - p_)) for p_ in sh) <= rings[-1] + 0.05:
                continue
            others = [d.xpos[b2][:2] for k2, b2 in par.toys.items() if k2 != k]
            cands = []
            for r_ in rings:
                for i in range(12):
                    q = chest + r_ * np.array([math.cos(2 * math.pi * i / 12), math.sin(2 * math.pi * i / 12)])
                    if par.child.clearance_xy(q) >= 0.12 and all(np.linalg.norm(q - o) >= 0.15 for o in others) and par._in_plan(q) \
                            and par.plan.dist[par.plan.cell(q)] >= 0.10 and float(np.linalg.norm(q - her)) >= 0.5:
                        cands.append((min(float(np.linalg.norm(q - p_)) for p_ in sh), q))
                if cands:
                    break
            if not cands:
                continue
            place = min(cands, key=lambda x: x[0])[1]
            j = m.body_jntadr[b]; adr = m.jnt_qposadr[j]; dof = m.jnt_dofadr[j]; home = m.qpos0[adr:adr + 7]
            d.qpos[adr:adr + 2] = place; d.qpos[adr + 2] = home[2]; d.qpos[adr + 3:adr + 7] = home[3:7]; d.qvel[dof:dof + 6] = 0.0
            mujoco.mj_forward(m, d)
            self.tidied.append((int(self.tick), k, [float(xy[0]), float(xy[1])], [float(place[0]), float(place[1])]))
            moved.append(k)
        return moved

    def save_state(self):
        """the whole world as bytes: the physics, the model's run-time fields, the senses' carry, the world's random
        stream and the parent's pose as the scene last drew it (Scene.pose: the W1 verifier's third round)"""
        return pickle.dumps(self._capture(), protocol=4)

    def load_state(self, blob):
        """the whole world back exactly from save_state's bytes"""
        st = pickle.loads(bytes(blob))
        if st.get("version") != 1 or st.get("nstate") != mujoco.mj_stateSize(self.m, STATE_SPEC):
            raise ValueError("G1World.load_state: not a save of this world")
        self._restore(st)

    # ---------------------------------------------------------------- the senses
    def _traction_N(self):
        """A177: the parent's pull on each forearm along the arm, away from the shoulder (N): over her holds on the forearm link, the hold's
        force projected on the unit from the elbow link to the hand; nothing below zero (a push toward the shoulder is not traction);
        {} with no parent or no hold"""
        out = {}
        if self.parent is None or not getattr(self.parent, "holds", None):
            return out
        for arm, b in self._forearm_body.items():
            n_ = 0.0
            for h in self.parent.holds:
                if int(h.body) == b and h.force is not None:
                    hand = self.d.xpos[self.palm_body[0 if arm == "arm_l" else 1]]
                    u = hand - self.d.xpos[b]; nu = float(np.linalg.norm(u))
                    if nu > 1e-6:
                        n_ += max(0.0, float(np.asarray(h.force, float) @ (u / nu)))
            if n_ > 0.0:
                out[arm] = n_
        return out

    def palm_normal(self, h):
        """the unit normal out of a hand's palm (0 left, 1 right), from its palm link's frame (the Dex3's palm faces -y on the left
        hand and +y on the right in the link's frame, as the parent reads it: parent_motion.Child)"""
        return self.d.xmat[self.palm_body[h]].reshape(3, 3) @ np.array([0.0, -1.0 if h == 0 else 1.0, 0.0])

    def grasp_centre(self, h):
        """where a toy held in a hand's grasp sits (A171): HAND_DEPTH_M out from the middle of the palm's face along its normal"""
        g = self.palm_geom[h]
        R_ = self.d.geom_xmat[g].reshape(3, 3); n = self.palm_normal(h)
        aabb = self.m.geom_aabb[g]
        return self.d.geom_xpos[g] + R_ @ aabb[:3] + n * (float(np.abs(R_.T @ n) @ aabb[3:]) + HAND_DEPTH_M)

    def _zone_forces(self):
        """this step's summed normal force on each touch zone (N): every contact touching a G1 zone (self-contact counts on both
        zones; A12's rest blind spots, none as born, felt by neither), plus each of the parent's active holds on a G1 body on
        that body's zone. Also this step's force on each palm from its own hand's other links (`_palm_own`)"""
        d, zg, nz = self.d, self.zone_of_geom, self.nz
        out = np.zeros(nz)
        self._palm_own = np.zeros(2)
        self._sides = np.zeros((2, 2))
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
            hz = np.where(z >= 0, self.hand_of_zone[np.where(z >= 0, z, 0)], -1)   # A171: THE TWO SIDES OF THE HAND: each contact of one
            for h in (0, 1):                                                        # hand's link with anything not that hand (a toy, the
                in0, in1 = hz[:, 0] == h, hz[:, 1] == h                             # floor, her hand), read ALONG THE PALM'S NORMAL, both its
                sel = in0 ^ in1                                                     # push and its place from the hand's grasp centre (a held
                if sel.any():                                                       # toy's: HAND_DEPTH_M before the palm's face): a push along
                    into = np.where(in1[sel, None], con.frame[sel, :3], -con.frame[sel, :3])   # the normal AWAY from the centre is a thing on
                    n_ = self.palm_normal(h)                                        # the palm's side (the face, the fingers' insides: palmar);
                    a_ = into @ n_                                                  # TOWARD it a thing against the back of the hand or a fist's
                    s_ = (con.pos[sel] - self.grasp_centre(h)) @ n_                 # knuckles (dorsal); a push ACROSS the normal (the mat under
                    f_ = fn[sel]                                                    # a hand lying on its edge, the supine child's every hour) is
                    face = np.abs(a_) >= SIDE_COS                                   # neither and counts for nothing
                    away = face & (a_ * s_ > 0.0)
                    toward = face & (a_ * s_ < 0.0)
                    self._sides[h, 0] = float(f_[away].sum()); self._sides[h, 1] = float(f_[toward].sum())
        if self.parent is not None and self.parent.holds:            # being held is felt (4.2): each capped spring's force on the
            out += self.parent.hold_zone                                # zone of the link it holds, this step's
        return out

    def _sense_tick(self, F, imu, palm_own=None, OW=None, heard=None, BD=None, TW=None, sides=None):
        """the tick's aggregates: touch (the mean force's log per zone, its onset), the IMUs (their noisy samples), the palms' own-hand
        force; THE OBSERVER'S (OW: its 15 samples of 10 ms, each the 43 joints' outside torques and the base's outside force, in the
        pelvis's frame, and torque, from the robot's own sensors: body/sim/observer.py): the tick's mean over each joint's declared
        limit and over the body's weight, their onsets (the rise of their sizes), each joint's largest sample (an instrument) and
        the base's largest force for its pain (A37); THE GEAR'S LOAD (BD, each step's, sensed): each joint's largest 10 ms mean for its
        pain (A73); the ears' code and the sound's side (by day). TW: the world's own contact torques at the same samples, for the
        instruments and the parent's eyes alone (C43's observer error)"""
        s = self._sensed
        J, W_ = len(JOINTS), OBS_WINDOW_STEPS
        lf = np.log1p(F.mean(axis=0) / TOUCH_UNIT_N)
        onset = np.maximum(0.0, lf - s["touch_log"])
        OW = np.zeros((len(F) // W_, J + 6)) if OW is None else OW
        TW = np.zeros_like(OW) if TW is None else TW
        BD = np.zeros((len(F), J)) if BD is None else BD               # PAIN (A37, A73): the torque each joint's gear carries, 10 ms
        Pb = np.vstack([s["bd_carry"], BD])                             # means (the module's PAIN note)
        cb = np.vstack([np.zeros((1, J)), np.cumsum(Pb, axis=0)])
        winb = ((cb[W_:] - cb[:-W_]) / W_)[-len(BD):]
        obs_j = OW[:, :J].mean(axis=0) / self.tau_max
        obs_b = OW[:, J:].mean(axis=0) / self.weight
        vest = self._vestibular(imu)
        po_ = np.zeros(2) if palm_own is None else np.asarray(palm_own, float)   # A171: THE GRASP'S STIMULUS IS THE PALM'S SIDE: the push on the
        sd_ = np.zeros((2, 2)) if sides is None else np.asarray(sides, float)    # hand from its palm's side by things not its own (the palm's face,
        pl = np.log1p((sd_[:, 0] + po_) / TOUCH_UNIT_N)                           # the fingers' insides) plus its own fingers on the palm (A162's
                                                                                   # fist), never the mat under the back of a palm-up hand
        self._sensed = {"touch_log": lf, "touch_onset": onset, "touch_force": F.mean(axis=0), "vestibular": vest,
                        "peak_force": F.max(axis=0), "palm_own": np.zeros(2) if palm_own is None else np.asarray(palm_own, float).copy(),
                        "sides": np.zeros((2, 2)) if sides is None else np.asarray(sides, float).copy(),   # A171: the tick's mean push on each hand
                                                                                                           # from things not its own [palmar, dorsal] (N)
                        "palm_log": pl, "palm_onset": np.maximum(0.0, pl - s["palm_log"]),               # A171: the grasp's stimulus, the palm's SIDE
                        "obs_j": obs_j, "obs_j_on": np.maximum(0.0, np.abs(obs_j) - np.abs(s["obs_j"])),
                        "obs_b": obs_b, "obs_b_on": np.maximum(0.0, np.abs(obs_b) - np.abs(s["obs_b"])),
                        "obs_peak": np.abs(OW[:, :J]).max(axis=0), "base_peak": float(np.linalg.norm(OW[:, J:J + 3], axis=1).max()),
                        "true_peak": np.abs(TW[:, :J]).max(axis=0), "true_base_peak": float(np.linalg.norm(TW[:, J:J + 3], axis=1).max()),
                        "true_j": TW[:, :J].mean(axis=0), "true_b": TW[:, J:].mean(axis=0),
                        "bd_peak": np.abs(winb).max(axis=0), "bd_carry": BD[-(W_ - 1):].copy(),
                        "obs_last": OW[-1].copy(), "imu_last": imu[-1].copy(),
                        "F_last": F[-1].copy(), "imu_torso": np.concatenate([imu[:, 0:3].mean(axis=0), imu[:, 3:6].mean(axis=0)]),
                        "ears": s["ears"] if heard is None else heard.code().astype(np.float64),
                        "sound_side": s["sound_side"] if heard is None else np.array([float(heard.onset_heard), float(heard.lateral)])}

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
        """the senses at birth, before any tick: one sample of the born state stands for the tick (its mean and its peak); the
        observer takes its first sample, the static estimate of the born state at rest"""
        mujoco.mj_forward(self.m, self.d)
        F = self._zone_forces()[None, :]
        imu = self._imu_noisy(self.d.sensordata[self.imu_adr])[None, :]
        J = len(JOINTS)
        d = self.d
        r0, w0 = self.observer.sample(d.qpos[self.qadr], d.qvel[self.dof], d.qpos[self.base_dof + 3:self.base_dof + 7], imu[0, 6:9],
                                      imu[0, 9:12], tau=d.qfrc_actuator[self.dof].copy())
        ob = np.concatenate([r0, w0])
        tr = self._outside()
        Rp = d.xmat[self.pelvis_id].reshape(3, 3)
        tw = np.concatenate([tr[:J], Rp.T @ tr[J:J + 3], tr[J + 3:]])
        lf = np.log1p(F[0] / TOUCH_UNIT_N)
        self._sensed = {"touch_log": lf, "touch_onset": np.zeros(self.nz), "touch_force": F[0].copy(), "vestibular": self._vestibular(imu),
                        "peak_force": F[0].copy(), "palm_own": self._palm_own.copy(), "sides": self._sides.copy(), "palm_log": np.log1p((self._sides[:, 0] + self._palm_own) / TOUCH_UNIT_N), "palm_onset": np.zeros(2), "obs_j": ob[:J] / self.tau_max, "obs_j_on": np.zeros(J),
                        "obs_b": ob[J:] / self.weight, "obs_b_on": np.zeros(6), "obs_peak": np.abs(ob[:J]),
                        "base_peak": float(np.linalg.norm(ob[J:J + 3])), "true_peak": np.abs(tw[:J]),
                        "true_base_peak": float(np.linalg.norm(tw[J:J + 3])), "true_j": tw[:J].copy(), "true_b": tw[J:].copy(),
                        "bd_peak": np.zeros(J), "bd_carry": np.zeros((OBS_WINDOW_STEPS - 1, J)),
                        "obs_last": ob.copy(), "imu_last": imu[0].copy(),
                        "F_last": F[0].copy(), "imu_torso": np.concatenate([imu[0, 0:3], imu[0, 3:6]]),
                        "ears": np.zeros(AN.SIZES["ears"]), "sound_side": np.zeros(2)}

    def _truth(self):
        """world truth for the parent and the instruments (never the body)"""
        m, d = self.m, self.d
        toys = {m.body(b).name[4:]: d.xpos[b].copy() for b in range(m.nbody) if m.body(b).name.startswith("toy_")}
        s = self._sensed
        return {"time": float(d.time), "pelvis": d.qpos[0:7].copy(), "torso": d.xpos[m.body("torso_link").id].copy(),
                "toys": toys, "touch_N": s["touch_force"].copy(), "peak_N": s["peak_force"].copy(),
                "outside_peak_Nm": s["obs_peak"].copy(), "base_peak_N": float(s["base_peak"]), "heat_C": self.heat.copy(),
                "outside_true_Nm": s["true_j"].copy(), "outside_true_peak_Nm": s["true_peak"].copy(),
                "base_true_peak_N": float(s["true_base_peak"]), "base_true_wrench": s["true_b"].copy(),
                "night": self.night, "below_n": self._below_n, "crying": bool(self.crying),
                "sound_events": list(getattr(self.sounds, "last_events", [])),
                "f_pain": self.f_pain, "ncon": int(d.ncon), "acts": dict(self._last_acts),
                "gaze": self.gaze.copy(), "spinal": dict(self._spinal), "vor_quick": self._vor_quick, "tendon": self._tendon.copy(),
                "palm_own_N": {"hand_l": float(s["palm_own"][0]), "hand_r": float(s["palm_own"][1])},
                "parent": None if self.parent is None else self.parent.truth()}

    # ---------------------------------------------------------------- the state
    def _from_fast(self, blob):
        """a tick-start snapshot (`_capture(fast=True)`) as the state `_restore` takes"""
        st = pickle.loads(blob)
        st["s5"] = _canon(st["s5"])
        if st.get("parent") is not None:
            st["parent"] = sys.modules[type(self.parent).__module__]._canon(st["parent"])
        return st

    def _capture(self, fast=False):
        """the world's whole state. fast (A187, 2026-10-04, speed only): the tick's own start, kept in case the physics faults: the
        same content pickled raw, made canonical only when a fault restores it (`_from_fast`). The canonical form is the save's (equal
        states, equal bytes); a snapshot that is thrown away a tick later needs none, and making it cost 39 of a 436 ms tick"""
        m, d = self.m, self.d
        phys = np.zeros(mujoco.mj_stateSize(m, STATE_SPEC))
        mujoco.mj_getState(m, d, phys, STATE_SPEC)
        st = {"version": 1, "nstate": int(phys.size), "physics": phys,
                "model": {f: getattr(m, f).copy() for f in MUTABLE_MODEL_FIELDS},
                "tick": self.tick, "paused": self.paused, "seed": self.seed,
                "sensed": {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in self._sensed.items()},
                "last_acts": dict(self._last_acts), "rng": self.rng.bit_generator.state, "gaze": self.gaze.copy(), "gaze_v": self.gaze_v.copy(),
                "spinal": dict(self._spinal), "vor_quick": self._vor_quick, "tendon": self._tendon.copy(), "scene_pose": _pose_state(self.scene.pose),
                "grasp_hab": {h: list(v) for h, v in self._grasp_hab.items()},   # A162
                "dorsal_hab": {h: list(v) for h, v in self._dorsal_hab.items()},  # A171
                "traction_hab": {a: list(v) for a, v in self._traction_hab.items()},  # A177
                "stepping": {k_: [str(v_[0]), int(v_[1])] for k_, v_ in self._stepping.items()},   # A193
                "parent": None if self.parent is None else (self.parent._state() if fast else self.parent.state()),
                "s5": ((lambda x: x) if fast else _canon)({"heat": self.heat, "tract": self.tract.state(), "ears": self.ears.state(), "vor_corr": self.vor_corr,
                              "vor_slip": ([] if self._vor_slip is None else list(map(float, self._vor_slip))),        # A174 (empty: none)
                              "vor_turn": ([] if self._vor_turn is None else list(map(float, self._vor_turn))),
                              "observer": self.observer.state(), "sounds": self.sounds.state(),
                              "eyes": None if self.eyes is None else self.eyes.state(),
                              "words_out": self.words_out, "tract_pa": self.tract_pa, "tract_raw": self.tract_raw, "crying": self.crying, "night": self.night, "dawn_left": int(self.dawn_left),
                              "carried": [[int(t), list(a), list(b)] for t, a, b in self.carried],
                              "tidied": [[int(t), k, list(a), list(b)] for t, k, a, b in self.tidied],
                              "lane": None if self.lane is None else self.lane.state()}),
                "warnings": np.array([int(d.warning[i].number) for i in range(int(mujoco.mjtWarning.mjNWARNING))])}
        return pickle.dumps(st, protocol=4) if fast else st

    def _restore(self, st):
        m, d = self.m, self.d
        for f, v in st["model"].items():
            getattr(m, f)[...] = v
        mujoco.mj_setState(m, d, st["physics"], STATE_SPEC)
        for i, n in enumerate(st["warnings"]):
            d.warning[i].number = int(n)
        self.tick, self.paused = int(st["tick"]), bool(st["paused"])
        self.seed = int(st["seed"])
        self._sensed = {k: (_fresh(v) if isinstance(v, np.ndarray) else v) for k, v in st["sensed"].items()}   # numpy's own dtypes
        for k, v in (("sides", np.zeros((2, 2))), ("palm_log", np.zeros(2)), ("palm_onset", np.zeros(2))):   # (A171; a save from before it:
            self._sensed.setdefault(k, v)                                                                     # the hands unpressed this tick)
                                                                        # (a carried array, the night's ears, pickles as a fresh one)
        self._last_acts = _canon_acts(st["last_acts"])
        self._spinal = dict(st.get("spinal", {})); self._vor_quick = int(st.get("vor_quick", 0))
        self._tendon = np.asarray(st.get("tendon", np.zeros(len(JOINTS), int)), int).copy()   # (A139; older saves: none)
        self._grasp_hab = {h: [int(x) for x in st.get("grasp_hab", {}).get(h, [0, 0])] for h in self.palm_of_hand}   # (A162; older saves: fresh)
        self._dorsal_hab = {h: [int(x) for x in st.get("dorsal_hab", {}).get(h, [0, 0])] for h in self.palm_of_hand}   # (A171; older saves: fresh)
        self._traction_hab = {a: [int(x) for x in st.get("traction_hab", {}).get(a, [0, 0])] for a in ("arm_l", "arm_r")}   # (A177; older saves: fresh)
        self._stepping = {k_: [str(st.get("stepping", {}).get(k_, ["stance", 0])[0]), int(st.get("stepping", {}).get(k_, ["stance", 0])[1])] for k_ in ("leg_l", "leg_r")}   # (A193; older saves: stance)
        self.rng.bit_generator.state = st["rng"]
        self.gaze = np.asarray(st.get("gaze", np.zeros(3)), float).copy()        # (a save from before the gaze: born at 0)
        self.gaze_v = np.asarray(st.get("gaze_v", np.zeros(3)), float).copy()
        if "scene_pose" in st:                                          # the parent's pose as the scene last drew it (W2 goes on
            self.scene.pose = _pose_from_state(st["scene_pose"])        # from it); her body and face geoms are in the physics and
        if self.parent is not None and st.get("parent") is not None:    # the model fields above; her motion's own state
            self.parent.load_state(st["parent"])
        s5 = _uncanon(st["s5"])
        self.heat = np.asarray(s5["heat"], float).copy()
        self.tract.load_state(s5["tract"])
        self.observer.load_state(s5["observer"])
        self.sounds.load_state(s5["sounds"])
        if self.eyes is not None and s5.get("eyes") is not None:
            self.eyes.load_state(s5["eyes"])
        self.ears.load_state(s5["ears"])
        self.vor_corr = np.asarray(s5["vor_corr"], float).copy()
        vs_, vt_ = s5.get("vor_slip"), s5.get("vor_turn")               # (A174; older saves or an empty pair: none)
        self._vor_slip = None if not vs_ else np.asarray(vs_, float).copy()
        self._vor_turn = None if not vt_ else np.asarray(vt_, float).copy()
        self.words_out = s5["words_out"]; self.tract_pa = np.asarray(s5["tract_pa"], float).copy()
        self.tract_raw = np.asarray(s5["tract_raw"], float).copy(); self.crying = bool(s5["crying"])
        self.night, self.dawn_left = bool(s5["night"]), int(s5["dawn_left"])
        self.carried = [(int(t), [float(x) for x in a], [float(x) for x in b]) for t, a, b in s5.get("carried", [])]   # (A110; older saves: none)
        self.tidied = [(int(t), str(k), [float(x) for x in a], [float(x) for x in b]) for t, k, a, b in s5.get("tidied", [])]   # (A117; older saves: none)
        if self.lane is not None and s5.get("lane") is not None:
            self.lane.load_state(s5["lane"])
        mujoco.mj_forward(m, d)
