"""THE G1'S ANATOMY AS THE CORE MEETS IT (docs/SIM_DESIGN.md 3.4, 3.5, 3.6, 3.7, 6 and 10; the core refactor's step R6h, C61): the stock
Unitree G1's nine channels at 3.4's sizes, its ten effectors numbered as 3.5 numbers them (the vocal tract effector 0, the words'
silent output effector 1, the gaze 2, the waist 3, the arms 4-5, the Dex3 hands 6-7, the legs 8-9), and its three reward sources in
section 6's order (face, pain, charge); its cerebellar interface (7.5, A44: the mossy input the world hands the cerebellum below the
tick, its readouts, the flocculus's axes); and SIM_CFG, the core's constants the sim is born with that R6h decides (the gates' drives,
the switches of the motor effectors, the cord's patterns and the born biases) and the cerebellum's switch. A declaration: it builds no
module, draws no random number and keeps no state (the core's law for an anatomy, body/core/anatomy.py). S5a builds the world against
it (the frames' keys below) and completes what is marked for it; the words are the parent's birth words (body/sim/lang's BIRTH_WORDS on
the parent's branch), passed in (`words`), placeholders in the core's tests.

THE NUMBERING (C61): 3.5 numbers the tract 0 and the words' output 1; the code's voice is the lexicon's effector (the words' symbols,
mouth_gate), which since R6h may stand at any place (`Anatomy.voice`), so the numbering is the design's: effectors[0] is the tract,
effectors[1] the words. The performance error (A41) is declared on the tract (`intrinsic`, per articulator) and not on the words, so
under gate_int 0.5 it reaches the tract's gate and no other (body/tests/test_motor.py, motor 11).

THE FRAME THE WORLD HANDS THE BODY (S5a's contract; every observation the body's own sensors give, raw, unit-scaled by the world):
  words        the parent's word token on the tick its sound ends, or letters (79 symbols: `born_table`), or nothing (the rest)
  face         [the born expression reading 2 x (smile - frown), its change] (A1, A49): the level held 30 ticks out of view, then neutral
  ears         1,725: both cochleas' 15 frames x 40 bands and the brainstem's delay lines (body/sim/ears.py)
  eye_p        172: each grey eye's periphery cells (7 x 4 x luminance ON, OFF) and the colour camera's 5 x 3 x (red-green, blue-yellow) x
               (ON, OFF) (A38)
  eye_f        1,536: each grey fovea's 8 x 8 cells x the born bank's 10 and the colour window's 8 x 8 x 4 (A42)
  body         242: per joint in BODY_JOINTS' order [sin, cos of the scaled angle, velocity, servo effort (the torque estimated from current
               over the limit), temperature] (215), the gaze's [yaw, pitch, vergence] and their velocities (6: numbers 215-220), the
               tract's 10 positions, 10 velocities and breath left (21: 221-241, breath left 241)
  touch        130: the Dex3 hands' 16 zones in ZONES' order, each [log(1 + F / 1 N), its onset] (32), then per joint in BODY_JOINTS'
               order [the observer's outside torque over the joint's limit, its onset] (86), then the base's outside wrench, each of its
               6 numbers [its value over the body's weight, its onset] (12)
  vestibular   24: both inertial units' accelerometer and gyro, each the tick's mean and peak
  charge       [h, the change of h this tick]
and beside the channels, read by the reward, the born reflexes and the gates' own inputs (never a channel):
  pain         44: per joint in BODY_JOINTS' order 1 where the observer's outside torque (its tick's largest 10 ms mean) passed the
               joint's own limit, then the base's (its outside force past 3 x the body's weight) (A37)
  face_periph  [fired, direction from the fovea's centre in yaw (rad, + right), in pitch (rad, + up)]: the born face template's
               strongest fire in the periphery (orienting's cue; C39)
  face_fovea   [fired]: the born face template's fire on either fovea's own pixels (the event line "a face in the fovea"; never the
               world's face test: 3.4, A42, A49; C39)
  sound_side   [an onset heard, the born lateral read's angle (rad, + left)] (the cochlea's onset cells and the lateral read)
  onset_periph [fired, yaw, pitch]: the born sudden local change in the grey periphery, habituating (A43; C49)
  imu_torso    [accelerometer x, y, z (m/s^2), gyro x, y, z (rad/s)]: the torso's inertial unit, the tick's means, raw (through its sensor's
               models: noise, the gyro's walking bias), in the unit's frame: the heading's source (R7f, 7.6: the trunk's yaw integrated
               from the torso gyro since birth; the vestibular channel carries the same, unit-scaled)
THE EVENT LINES (R7a; 7.2, 7.4's low road, A37, A43; body/core/anatomy.py `EventLine`), 13, read from the frame above by the born rule
(a line fires when any of its numbers is above 0; on its side where it has one; not where a line further along its limb fires):
  touch_trunk, touch_arm_l, touch_arm_r, touch_hand_l, touch_hand_r, touch_leg_l, touch_leg_r: touch onset in 7 groups from the
      observer and the hands' arrays: the touch channel's onsets on the group's joints (the trunk: the waist's joints and the base's
      wrench, since the observer cannot tell the head from the torso and a touch on the pelvis shows only in the base's wrench; a hand:
      its 7 joints and its 8 zones of the Dex3's arrays). ISOLATION (Haddadin et al. 2017): a contact shows on every joint between the
      pelvis and the touched link, so the furthest group that feels it names it: an arm's line fires only when its hand's does not, the
      trunk's only when no limb's does. THE WORLD'S ONSET is a rise beyond the observer's noise (C43, W1 reopened; the line fires on any
      onset above 0 the world reports, as the limbs' gates read it)
  pain: any of the pain flags (a joint's, or the base's)
  face_fovea: the born face template in either fovea
  sound_l, sound_r: a sound's onset, on the left (the lateral read's angle above 0) or the right (below 0); both where it has no side
  visual_l, visual_r: a sudden local change in the periphery, on the left (its yaw below 0) or the right (above 0); both at the centre
THE ACTS THE WORLD RECEIVES (body/core/world.py `Acts`): each effector's flat act by its name (the tract's 10 articulators and every
limb's joints as base-5 digits, joint 0 the most significant, in the joint orders below; the words' symbol), `acts.cord` (the spinal
pattern generators' and the born cry's steps below the gates, per joint in the joint's units) and `acts.vor` (the gaze's VOR).

THE CEREBELLUM'S MOSSY INPUT (SIM_DESIGN.md 7.5, A44; the lead's decision for the G1: THE BODY'S OWN SIGNALS ONLY, NO VISION AND NO
HEARING AT BIRTH): the numbers the world hands the cerebellum at each sub-step below the tick (every 10 ms, body/core/world.py
`SubFrame.mossy`), 219 in this order, each named in `SimAnatomy.mossy` and declared to the organ by `SimAnatomy.cerebellar`:
  (1) THE EFFERENCE COPY OF EVERY MOTOR EFFECTOR'S ACTS (the pontine route of the motor command: Ito 1984, The Cerebellum and Neural
      Control; Apps and Garwicz 2005, Nat Rev Neurosci 6:297-311), the tick's own act held through its sub-steps, in the effectors'
      declared order: each joint's setting of the tract (10: its articulators), the gaze (3: yaw, pitch, vergence), the waist (3), the
      arms (7 + 7), the Dex3 hands (7 + 7) and the legs (6 + 6): 56. Each setting is 0-4, the hold 2. The gate's own act: never the
      cord's patterns or a reflex's forced act. THE WORDS' SILENT OUTPUT (effector 1, the token output) IS NOT AMONG THEM (the lead's
      decision, 2026-09-25): the cerebellum smooths joints, and the token output moves no joint and has no body sense to predict; its
      79 symbol lines, which e351813 read into "every effector", left the efference copy (298 numbers then, 219 now).
  (2) THE JOINTS' POSITIONS AND VELOCITIES (the spinocerebellar proprioception: Bosco and Poppele 2001, Physiol Rev 81:539-568): each
      of the 43 joints' angle (rad, through the encoder) in BODY_JOINTS' order, then each one's velocity (rad/s): 86. The gaze's window
      and the tract, whose states the body channel also carries, are not spinal joints and are not among them (their acts are in 1).
  (3) THE INERTIAL UNITS' ORIENTATION AND ANGULAR VELOCITY (the vestibular mossy fibres: Apps and Garwicz 2005): the torso's unit (it
      moves with the head) then the pelvis's, each its accelerometer (m/s^2) then its gyro (rad/s), the sub-step's samples through the
      sensors' models, both units as 3.4's vestibular channel declares them: 12. THE ANATOMY DECLARES NO ORIENTATION ESTIMATE (no
      attitude, no quaternion: the vestibular channel carries the accelerometer and the gyro alone), so the accelerometer's specific
      force, which way is down, stands for the orientation, as the otoliths' own signal does; nothing is estimated for it.
  (4) TOUCH AND CONTACT (the cutaneous spinocerebellar input: Apps and Garwicz 2005): the Dex3's 16 zones' log(1 + F / 1 N) of the
      sub-step's normal force on the arrays' faces (ZONES' order), then the observer's outside torque on each of the 43 joints over its
      limit, then the base's outside wrench over the body's weight (force, then torque, each x y z in the pelvis's frame): 65, each as
      the touch channel carries it, at the sub-step (the observer runs every 10 ms, the sub-step's period), onsets not among them.
Each number's declared middle and half-range (the organ reads it as the fibre's rate 1 + (x - middle) / half-range, held to [0, 2]): a
setting 2 and 2; an angle its joint's range's middle and half (RANGES, the model's own); a velocity and a gyro's turn 0 and MOSSY_SPEED;
an accelerometer's axis 0 and MOSSY_G; touch and contact 0 and 1 (the touch channel's own units: the log of newtons, each joint's limit,
where its pain begins, the body's weight). THE READOUTS (7.5): the 29 joints of the waist, the arms and the legs (CEREB_JOINTS; the
Dex3's joints none), each a torque added to its servo inside its limit this tick; THE FLOCCULUS on the gaze window's yaw and pitch (the
VOR's axes)."""
import math
from dataclasses import dataclass

from tokenizers import Tokenizer, models

from body.core.anatomy import Cerebellar, Channel, EarChannel, Effector, EventLine, Heading, LanguageAnatomy, OrientCue, RewardSource, VoiceEffector

# ---------------------------------------------------------------- the G1's joints, in the order the world writes them (3.2, 3.5)
WAIST = ("waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint")
ARM = ("shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow", "wrist_roll", "wrist_pitch", "wrist_yaw")
HAND = ("hand_thumb_0", "hand_thumb_1", "hand_thumb_2", "hand_index_0", "hand_index_1", "hand_middle_0", "hand_middle_1")
LEG = ("hip_pitch", "hip_roll", "hip_yaw", "knee", "ankle_pitch", "ankle_roll")
LIMBS = (("waist", WAIST),
         ("arm_l", tuple(f"left_{j}_joint" for j in ARM)), ("arm_r", tuple(f"right_{j}_joint" for j in ARM)),
         ("hand_l", tuple(f"left_{j}_joint" for j in HAND)), ("hand_r", tuple(f"right_{j}_joint" for j in HAND)),
         ("leg_l", tuple(f"left_{j}_joint" for j in LEG)), ("leg_r", tuple(f"right_{j}_joint" for j in LEG)))
BODY_JOINTS = tuple(j for _, js in LIMBS for j in js)                     # 43
ZONES = tuple(f"{s}_hand_{z}" for s in ("left", "right") for z in ("palm", "thumb_0", "thumb_1", "thumb_2", "index_0", "index_1",
                                                                   "middle_0", "middle_1"))   # the Dex3's 16 zones (C44)
HAND_OF_ARM = {"arm_l": "hand_l", "arm_r": "hand_r"}
TRACT = ("lungs", "glottis", "pitch", "jaw", "tongue_front", "tongue_height", "tongue_tip", "lips", "rounding", "velum")   # body/sim/tract.py
PER_JOINT = 5                                                             # sin, cos, velocity, effort, temperature (3.4)
GAZE_AT = PER_JOINT * len(BODY_JOINTS)                                    # 215: the gaze's state and velocity
TRACT_AT = GAZE_AT + 6                                                    # 221: the tract's positions, velocities, breath left
BREATH_AT = TRACT_AT + 20                                                 # 241
BODY_SIZE = TRACT_AT + 21                                                 # 242
TOUCH_SIZE = 2 * len(ZONES) + 2 * len(BODY_JOINTS) + 2 * 6                # 130
SIZES = dict(face=2, ears=1725, eye_p=172, eye_f=1536, body=BODY_SIZE, touch=TOUCH_SIZE, vestibular=24, charge=2)
# each limb's flexion joints and their flexion senses, measured on the G1 (the withdrawal's: sim-world's body/tests/test_sim_world.py,
# world 8; 3.7): a leg's hip pitch -1, knee +1, ankle pitch -1; an arm's shoulder pitch -1 and elbow -1. The spinal pattern generator
# moves these (A48)
FLEXION = {"leg": {0: -1, 3: 1, 4: -1}, "arm": {0: -1, 3: -1}}
# THE FOVEA'S ZONE: half the software fovea's width, 64 px at 3 px a degree (3.4, A42): 10.7 degrees; a cue inside it is foveated (ours,
# the fovea's own anatomy)
FOVEA_HALF = math.radians(64.0 / 3.0 / 2.0)
# THE CRY'S POSTURE (A47, C53; Jurgens 2002): the lungs pushing, the glottis pressed, the pitch raised, the jaw open: each the big step
# up (+0.6 of the range a tick, the tract's largest; so the posture is reached within a tick or two and held at the ranges' ends while the
# cry lasts: the pitch at 546 Hz, inside newborns' phonated cries of 250-700 Hz). Ours, from its sources' description; C53 reads the
# pattern's figures (its breath groups are REFLEX's cry_expire and cry_inspire, from Robb, Sinton-White and Kaipa 2011)
CRY_POSTURE = {0: 0.6, 1: 0.6, 2: 0.6, 3: 0.6}
# ---------------------------------------------------------------- the cerebellum's interface (7.5, A44; the module's doc)
# THE G1'S JOINT RANGES (rad, in BODY_JOINTS' order): the model's own, each joint's `range` in Menagerie's g1_with_hands.xml (the file
# committed in dd8640e and loaded unchanged; 3.2 gives them in degrees), the angle fibres' middles and half-ranges. The cerebellum is
# born with them (its organ keeps its declared middles and half-ranges and a load checks them), so they are the file's, written here;
# S5a's world reads the same file and checks them against it
RANGES = ((-2.618, 2.618), (-0.52, 0.52), (-0.52, 0.52),                                                            # the waist
          (-3.0892, 2.6704), (-1.5882, 2.2515), (-2.618, 2.618), (-1.0472, 2.0944),                                   # the left arm
          (-1.97222, 1.97222), (-1.61443, 1.61443), (-1.61443, 1.61443),
          (-3.0892, 2.6704), (-2.2515, 1.5882), (-2.618, 2.618), (-1.0472, 2.0944),                                   # the right arm
          (-1.97222, 1.97222), (-1.61443, 1.61443), (-1.61443, 1.61443),
          (-1.0472, 1.0472), (-0.724312, 1.0472), (0.0, 1.74533), (-1.5708, 0.0), (-1.74533, 0.0), (-1.5708, 0.0),     # the left hand
          (-1.74533, 0.0),
          (-1.0472, 1.0472), (-1.0472, 0.724312), (-1.74533, 0.0), (0.0, 1.5708), (0.0, 1.74533), (0.0, 1.5708),       # the right hand
          (0.0, 1.74533),
          (-2.5307, 2.8798), (-0.5236, 2.9671), (-2.7576, 2.7576), (-0.087267, 2.8798), (-0.87267, 0.5236),           # the left leg
          (-0.2618, 0.2618),
          (-2.5307, 2.8798), (-2.9671, 0.5236), (-2.7576, 2.7576), (-0.087267, 2.8798), (-0.87267, 0.5236),           # the right leg
          (-0.2618, 0.2618))
# THE CEREBELLUM'S READOUTS (7.5): one per joint of the waist, the arms and the legs (29), in BODY_JOINTS' order; the Dex3's joints none
CEREB_JOINTS = tuple(j for name, js in LIMBS if not name.startswith("hand") for j in js)
# THE HALF-RANGES THE MODEL FILE DOES NOT GIVE (ours, disclosed): the file declares no joint velocity, and the inertial units' ranges only
# as their sensors' full scales (cutoff 34.9 rad/s and 157 m/s^2, under which a turn or a tilt would barely move a fibre). A joint's
# velocity and a gyro's turn are read over MOSSY_SPEED, the big step's pace (0.27 rad a tick of 0.15 s, 1.8 rad/s: the fastest the body's
# own act moves a joint's target, the waist's the trunk's; 3.5), faster saturating; an accelerometer's axis over MOSSY_G, one g (9.81
# m/s^2), so which way is down spans the fibre's range and a jolt past one g saturates
MOSSY_SPEED = 0.27 / 0.15
MOSSY_G = 9.81


def born_table(words=None):
    """THE WORD SCAFFOLD'S BORN TABLE (4.9: 50 word tokens, 26 letters, a space, rest and end: 79 rows), as a tokenizer for the language
    anatomy: the rest `<rest>` 0, the end `<end>` 1, the space 2, the letters a-z 3-28, the 50 words 29-78. `words`: the parent's 50
    birth words (placeholders qaa-qbx when none are given: the core's tests); a word spelled as one letter is refused (it is that letter's
    row)"""
    words = list(words) if words is not None else ["q" + chr(97 + k // 26) + chr(97 + k % 26) for k in range(50)]
    if len(words) != 50 or len(set(words)) != 50 or any(len(w) < 2 or not w.isalpha() or w != w.lower() for w in words):
        raise ValueError(f"born_table: 50 distinct lower-case words of at least two letters, given {len(words)}")
    vocab = {"<rest>": 0, "<end>": 1, " ": 2}
    for k, ch in enumerate("abcdefghijklmnopqrstuvwxyz"):
        vocab[ch] = 3 + k
    for k, w in enumerate(words):
        vocab[w] = 29 + k
    return Tokenizer(models.WordLevel(vocab=vocab, unk_token="<rest>"))


class FaceIncrement(RewardSource):
    """THE FACE AS REWARD FOR THE SIM (6.1, the increment rule; A1, A49): the born expression reading's level (the frame's face channel,
    number 0: 2 x (smile - frown), held while out of view as the world holds it); a rise of its positive part is felt positive, a rise of
    its negative part negative, no fall is ever felt; clipped to +-2 (the source's clip). The last reading is the life's `level` (as the
    diary's FaceReward keeps it), a float here. Source 0: the world's judgment, always answering (0 when nothing is felt)"""

    def felt(self, frame, life):
        o_ = frame.obs.get("face")
        now = float(o_[0]) if o_ is not None else 0.0
        prev = float(life.level)
        felt = max(0.0, max(0.0, now) - max(0.0, prev)) - max(0.0, max(0.0, -now) - max(0.0, -prev))
        life.level = now
        return felt


class JointPain(RewardSource):
    """PAIN (6.2, A37): -1 on a tick any of the frame's `pain` flags is set (a joint's outside torque past its limit, or the base's force
    past 3 x the body's weight, as the observer estimates them from the robot's own sensors); silent otherwise"""

    def felt(self, frame, life):
        p_ = frame.obs.get("pain")
        return -1.0 if p_ is not None and any(float(x) > 0.0 for x in p_) else None


class ChargeRelief(RewardSource):
    """THE CHARGE (6.3): 4 x [D(h_t-1) - D(h_t)], D(h) = (1 - h)^2 (drive reduction: Keramati and Gutkin 2014), from the charge channel's
    [h, its change this tick]; a full charge earns nothing"""

    def felt(self, frame, life):
        o_ = frame.obs.get("charge")
        if o_ is None:
            return None
        h, dh = float(o_[0]), float(o_[1])
        return 4.0 * ((1.0 - (h - dh)) ** 2 - (1.0 - h) ** 2)


def _joint_index(name):
    return BODY_JOINTS.index(name)


def event_lines():
    """THE G1'S 13 BORN EVENT LINES (R7a; the module's doc; SIM_DESIGN.md 7.4's low road, A37, A43), in the design's order: touch onset in 7
    groups (the trunk, each arm, each hand, each leg: the touch channel's onsets, its joints' in BODY_JOINTS' order and a hand's zones'
    in ZONES' order), pain, a face in the fovea, a sound onset on the left and the right, a visual onset on the left and the right"""
    Z, J, B = len(ZONES), len(BODY_JOINTS), 6
    joint_on = lambda j: 2 * Z + 2 * j + 1                                  # noqa: E731  (a joint's onset in the touch channel)
    zone_on = lambda z: 2 * z + 1                                           # noqa: E731  (a zone's onset)
    base_on = tuple(2 * Z + 2 * J + 2 * w + 1 for w in range(B))             # the base's wrench, its 6 onsets
    touch = []
    for name, js in LIMBS:
        on = tuple(joint_on(_joint_index(j)) for j in js)
        if name.startswith("hand"):
            side = "left" if name.endswith("_l") else "right"
            on += tuple(zone_on(k) for k, z in enumerate(ZONES) if z.startswith(side))
        grp = "trunk" if name == "waist" else name
        if grp == "trunk":
            on += base_on
            distal = tuple(f"touch_{n}" for n, _ in LIMBS if n != "waist")    # any limb's contact also loads the waist or the base
        elif grp.startswith("arm"):
            distal = ("touch_" + HAND_OF_ARM[grp],)                           # a hand's contact also loads its arm
        else:
            distal = ()
        touch.append(EventLine(f"touch_{grp}", "touch", fired=on, distal=distal))
    assert [x.name for x in touch] == ["touch_trunk", "touch_arm_l", "touch_arm_r", "touch_hand_l", "touch_hand_r", "touch_leg_l", "touch_leg_r"]
    return touch + [EventLine("pain", "pain", fired=tuple(range(J + 1))),
                    EventLine("face_fovea", "face_fovea", fired=(0,)),
                    EventLine("sound_l", "sound_side", fired=(0,), side=(1, 1.0)),       # the lateral read's angle: + left
                    EventLine("sound_r", "sound_side", fired=(0,), side=(1, -1.0)),
                    EventLine("visual_l", "onset_periph", fired=(0,), side=(1, -1.0)),   # the change's yaw: + right
                    EventLine("visual_r", "onset_periph", fired=(0,), side=(1, 1.0))]


@dataclass(eq=False)
class Limb(Effector):
    """A LIMB OF THE G1 (the waist, an arm, a hand, a leg; 3.5): its gate's own inputs are its own act last tick and its forward
    model's error (the core's, `fwd_gate`), the born orienting input where it orients (the waist), then the touch onset on the limb
    (the largest onset among its joints' outside torques and, for a hand, its zones) and the pain on the limb (any of its joints' pain
    flags; an arm's hand's joints count for it: 3.7). `joints` are its joints' places in BODY_JOINTS, `zones` its touch zones' places
    in ZONES, `pain_joints` the joints whose pain is its. Its cost: 0.12 x the mean over its joints of the servo effort squared,
    sensed on the tick (section 10's 0.12 x sum tau^2 / sum tau_max^2 per tick of motion, weighted by the limits once S5a passes them:
    `limits`), charged when it acts"""
    joints: tuple = ()
    zones: tuple = ()
    pain_joints: tuple = ()
    limits: tuple = ()

    def gate_inputs(self, frame, life, state):
        own = super().gate_inputs(frame, life, state)
        t_ = frame.obs.get("touch")
        onset = 0.0
        if t_ is not None:
            ons = [float(t_[2 * len(ZONES) + 2 * j + 1]) for j in self.joints] + [float(t_[2 * z + 1]) for z in self.zones]
            onset = max([0.0] + ons)
        p_ = frame.obs.get("pain")
        pain = 1.0 if p_ is not None and any(float(p_[j]) > 0.0 for j in self.pain_joints) else 0.0
        return own + [onset, pain]

    def cost(self, act, frame, life):
        b_ = frame.obs.get("body")
        if b_ is None:
            return 0.0
        w = self.limits if self.limits else (1.0,) * len(self.joints)
        num = sum((float(b_[PER_JOINT * j + 3]) * float(l_)) ** 2 for j, l_ in zip(self.joints, w))
        return 0.12 * num / sum(float(l_) ** 2 for l_ in w)


class Tract(Effector):
    """THE VOCAL TRACT (4.9; effector 0): its gate's own inputs are its own act last tick, the forward error on its hearing (its
    consequence sense is the ears) and its breath left (the body channel's number 241); its cost 0.12 an act (section 10's "0.12 a
    sounding tick": whether a tick sounded is heard only the tick after; S5a may charge it from the world's report)"""

    def gate_inputs(self, frame, life, state):
        own = super().gate_inputs(frame, life, state)
        b_ = frame.obs.get("body")
        return own + [float(b_[BREATH_AT]) if b_ is not None else 1.0]

    def cost(self, act, frame, life):
        return 0.12


class Gaze(Effector):
    """THE GAZE (3.4, 3.5; effector 2): the software fovea's window, no joints of the body: yaw and pitch (both windows together) and
    vergence; its consequence sense is its state in the body channel; its gate reads its own act last tick, its forward error and the
    born orienting input; its cost 0.03 a step (section 10)"""

    def cost(self, act, frame, life):
        return 0.03


class SimAnatomy(LanguageAnatomy):
    """THE G1 (the module's doc): `tok` is `born_table(words)`, `cfg` the life's constants (SIM_CFG's symbols: the rest `<rest>`, the end
    `<end>`, end_symbol "eot"); `limits` the G1's torque limits in BODY_JOINTS' order (the model's own: S5a reads them from the file;
    None weighs every joint alike in the limbs' cost)."""

    def __init__(self, tok, cfg=None, limits=None):
        super().__init__(tok, cfg)
        words = EarChannel("words", "symbol", self.vocab, organ="E", field="x", forecast=True, rest_id=self.sil, end_id=self.end_id,
                           reserved=self.reserved, partner=True)
        chans = [words] + [Channel(n, "vector", SIZES[n], organ=f"encs.{n}", forecast=True)
                           for n in ("face", "ears", "eye_p", "eye_f", "body", "touch", "vestibular", "charge")]
        lim = list(limits) if limits is not None else None

        def idx(js):
            return [PER_JOINT * _joint_index(j) + k for j in js for k in range(PER_JOINT)]
        tract = Tract("voice", [5] * len(TRACT), rest_id=(5 ** len(TRACT) - 1) // 2, sense="ears", inverse=True, fwd_gate=True, n_in=3,
                      intrinsic=True, cry={"posture": dict(CRY_POSTURE), "lungs": 0, "breath": ("body", BREATH_AT), "charge": ("charge", 0),
                                           "pain": "pain"})
        voice = VoiceEffector("words", [self.vocab], rest_id=self.sil, end_id=self.space_id, reserved=self.bans, intrinsic=False)
        gaze = Gaze("gaze", [5, 5, 5], rest_id=62, sense="body", sense_idx=list(range(GAZE_AT, GAZE_AT + 6)), fwd_gate=True,
                    orient={0: ("yaw", 1), 1: ("pitch", 1)}, orient_gate=True, vor=[0, 1], n_in=3)
        limbs = []
        for name, js in LIMBS:
            J = len(js); ji = tuple(_joint_index(j) for j in js)
            kind = name.split("_")[0]
            pain_j = ji + (tuple(_joint_index(j) for j in dict(LIMBS)[HAND_OF_ARM[name]]) if name in HAND_OF_ARM else ())
            zones = tuple(k for k, z in enumerate(ZONES) if name.startswith("hand") and z.startswith("left" if name.endswith("_l") else "right"))
            kw = dict(joints=ji, zones=zones, pain_joints=pain_j, limits=tuple(lim[j] for j in ji) if lim else ())
            if kind in FLEXION:
                # the spinal pattern generators (A48, C54): the legs keep one rhythm, the left leading it and the right half of each cycle
                # behind (born at phases 0 and 0.5); each arm a rhythm of its own, its phase drawn at birth from the body's seed
                kw.update(spg=dict(FLEXION[kind]), spg_phase=({"leg_l": 0.0, "leg_r": 0.5}.get(name)),
                          spg_rhythm=("legs" if kind == "leg" else None))
            if name == "waist":
                # the waist yaw's positive step turns the trunk, and the head's cameras with it, to the left (a rotation about the torso's
                # up axis), so a cue on the right (+ yaw in the image) is turned toward by its negative step: sense -1 (S5a checks the sign
                # on the G1, as W1 measured the withdrawal's)
                kw.update(orient={0: ("yaw", -1)}, orient_gate=True)
            n_in = 2 + (1 if name == "waist" else 0) + 2
            limbs.append(Limb(name, [5] * J, rest_id=(5 ** J - 1) // 2, sense="body", sense_idx=idx(js), inverse=True, fwd_gate=True,
                              n_in=n_in, **kw))
        rewards = [FaceIncrement("face", clip=2), JointPain("pain", signs=(-1.0,)), ChargeRelief("charge")]   # R7d: the heads face +/-, pain -, charge +/-
        self.channels, self.effectors, self.rewards, self.inner_at = chans, [tract, voice, gaze] + limbs, rewards, 2
        self.orienting = [OrientCue("face", "face_periph", fired=0, yaw=1, pitch=2, sense=1.0, zone=FOVEA_HALF),
                          OrientCue("sound", "sound_side", fired=0, yaw=1, sense=-1.0, side_only=True, onset=True),
                          OrientCue("onset", "onset_periph", fired=0, yaw=1, pitch=2, sense=1.0, zone=FOVEA_HALF, onset=True)]
        # THE CEREBELLUM'S INTERFACE (7.5, A44; the module's doc): the mossy numbers in their order, each named and declared by its
        # middle and half-range; the readouts on the waist's, the arms' and the legs' joints; the flocculus on the gaze's yaw and pitch
        mossy, off, half = [], [], []

        def fibre(name, mid, hr):
            mossy.append(name); off.append(float(mid)); half.append(float(hr))
        for e in self.effectors:                                       # (1) the efference copy of every motor effector's acts
            if e is voice:
                continue                                               # the words' token output: no joint, no body sense (the lead's)
            jn = TRACT if e is tract else (("yaw", "pitch", "vergence") if e is gaze else tuple(BODY_JOINTS[j] for j in e.joints))
            for j in jn:
                fibre(f"act {e.name}.{j}", 2.0, 2.0)                   # a joint's setting, 0-4 about the hold
        for (lo, hi), j in zip(RANGES, BODY_JOINTS):                   # (2) the joints' positions, then their velocities
            fibre(f"angle {j}", (lo + hi) / 2.0, (hi - lo) / 2.0)
        for j in BODY_JOINTS:
            fibre(f"velocity {j}", 0.0, MOSSY_SPEED)
        for unit in ("torso", "pelvis"):                               # (3) the inertial units: which way is down, and the turn
            for ax in "xyz":
                fibre(f"acc {unit}.{ax}", 0.0, MOSSY_G)
            for ax in "xyz":
                fibre(f"gyro {unit}.{ax}", 0.0, MOSSY_SPEED)
        for z in ZONES:                                                # (4) touch, then contact from the joints' efforts
            fibre(f"touch {z}", 0.0, 1.0)
        for j in BODY_JOINTS:
            fibre(f"contact {j}", 0.0, 1.0)
        for w in ("force.x", "force.y", "force.z", "torque.x", "torque.y", "torque.z"):
            fibre(f"contact base.{w}", 0.0, 1.0)
        self.mossy = tuple(mossy)
        self.cerebellar = Cerebellar(off, half, joints=list(CEREB_JOINTS), vor=["yaw", "pitch"])
        self.events = event_lines()                                    # R7a: the born event lines (the module's doc)
        self.heading = Heading("imu_torso", acc=(0, 1, 2), gyro=(3, 4, 5), dt=0.15)   # R7f: the heading from the torso's unit (the module's doc)


# THE CORE'S CONSTANTS THE SIM IS BORN WITH THAT R6h DECIDES (SIM_DESIGN.md 3.5, 3.6, 3.7, 10; A41, A47, A48; the language body holds none
# of them), and the cerebellum's switch (7.5, A44: on at birth, the anatomy declaring its mossy list, the lead's). S5a adds the rest of
# section 10's sim constants (the critics' solve every 256 ticks, the ventral critic's bands, the store's capacity).
SIM_CFG = dict(
    rest_token="<rest>", end_token="<end>", end_symbol="eot",          # the born table's rest and end (4.9)
    # the switches at birth (10): fixes #4, #5 and #8; chunk_gate 1 for every effector; chunk_max 8, a ceiling only (3.6)
    gate_own_draw=1, actor_trace_tick=1, elig_from=1, chunk_gate=1, chunk_max=8,
    # THE GATES' DRIVES, DISCLOSED (3.5, A41): the tonic drive following the reward rate in every gate, 0.25 + (sum over k < 12 of 0.8^k,
    # the gate's own eligibility window: 4.656) x the felt reward's running mean at the ladder's 256-tick clock (band 4); the performance
    # error at 0.5 on the gates that declare it (the tract's alone); gate_vigor 0, so the reward rate is not counted twice
    gate_tonic=0.25, gate_tonic_rate=sum(0.8 ** k for k in range(12)), gate_tonic_clock=4,
    gate_int=0.5, gate_int_form="error", gate_vigor=0.0,
    # R6h's motor effectors (MOTOR): movement units, act_inv batched every 8 ticks with the kappa correction, fatigue per effector
    unit_margin=math.log(4.0), act_inv_every=8, act_inv_chance=1, own_fatigue=1,
    # the born patterns and biases (REFLEX): the spinal pattern generators (their shape and cycles REFLEX's, C54: a movement of 2 ticks'
    # flexion and 3 ticks' extension, the extension returning the flexion's excursion, then a pause, each cycle drawn from the seed,
    # 3.56 +- 1.93 s held to 1.0-8.5 s), the born cry, orienting, the VOR
    spg=1, cry=1, orient=1, vor=1,
    # THE CEREBELLUM ON AT BIRTH (7.5, A44; SimAnatomy.cerebellar, the lead's mossy list): its constants CEREB's, as R6c and its fix
    # settled them and none given here (the rate 0.01 a sub-step; the leak rate / 3, after Smith, Ghazizadeh and Shadmehr 2006; the bound
    # that steps a lesson back onto the limit; each readout held inside its joint's limit this tick; the flocculus's 0.05 a tick; 4,096
    # granule units of 4 fibres, 10% active), its readouts per joint of the waist, the arms and the legs
    cereb=1,
    # STEP R7 (FRAMES; body/core/frames.py): the body lives in frames (R7b: the frame's surprise, the event's end for frames, the
    # surprise-gated writes at its running 0.9 quantile, the tick's record), its constants FRAMES' (none given here)
    frames=1,
    # STEP R7c: the tired memories recover (defect 1, SWITCHES; 10's switches at birth: fix #1 on) and each channel's forecast error
    # scaled by its own running mean in the waking lesson (FRAMES' err_scale; 10's "forecast heads"); tag_trace (defect 6) stays off,
    # the language body's utterance entry (the sim's tags reach its store and its night through the amygdala, R7d and R8)
    tire_recover=1, err_scale=1,
    # STEP R7d: THE AMYGDALA ON AT BIRTH (7.4, 10, A16; body/core/amygdala.py): its 5 heads (face +/-, pain -, charge +/-) over the
    # stream, the 13 event lines and the level; its constants AMYG's (none given here: tau_a band 6's clock, 4,096 ticks; the critics'
    # prior 0.3; the solve every 8 ticks; the reliability over 36,000 ticks, earned after 64 pairs)
    amyg=1,
    # STEP R7f: RECALL INTO ACTION (7.6, A45; FRAMES' recall): a frame's key the stream plus the heading from the torso gyro, its value
    # its codes and every effector's efference copy, each motor effector's map from the recalled act born at zero; and working memory
    # latching at the frames' event ends in place of the utterances' (wm_frames)
    recall=1, wm_frames=1,
)
