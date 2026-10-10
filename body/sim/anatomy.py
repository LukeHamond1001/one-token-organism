"""THE G1'S ANATOMY AS THE CORE MEETS IT (docs/SIM_DESIGN.md 3.4, 3.5, 3.6, 3.7, 6 and 10; the core refactor's step R6h, C61): the stock
Unitree G1's eight channels at 3.4's sizes, its ten effectors numbered as 3.5 numbers them (the vocal tract effector 0, the words'
silent output effector 1, the gaze 2, the waist 3, the arms 4-5, the Dex3 hands 6-7, the legs 8-9), and its two reward sources in
section 6's order (the face, pain; no charge: A88; pain pays again: A91); its cerebellar interface (7.5, A44: the mossy input the world hands the cerebellum below the
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
(the charge channel is gone with the charge: A88)
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

import numpy as np

from tokenizers import Tokenizer, models

from body.core.anatomy import Cerebellar, Channel, EarChannel, Effector, EventLine, Grounding, Situation, Heading, LanguageAnatomy, OrientCue, RewardSource, VoiceEffector


def _ground_appearance(frame):
    """A202: the look of what the fovea holds this tick (body/sim/eyes.py: ground_appearance over the frame's eye_f) -> [4]"""
    from body.sim import eyes as _eyes                                         # (eyes imports the world; the anatomy is imported first)
    ef = frame.obs.get("eye_f"); ep = frame.obs.get("eye_p")
    if ef is None or len(ef) < 4:                                              # the eyes off (the night): no look (A202m: None, no thing)
        return None
    fp = frame.obs.get("face_periph")
    face_ = None
    if fp is not None and len(fp) >= 3 and float(fp[0]) > 0.0 and abs(float(fp[1])) <= FOVEA_HALF and abs(float(fp[2])) <= FOVEA_HALF:
        # A202o (2026-10-07): THE SPEAKER'S FACE IS NOT A REFERENT. The rows rebuilt under the figure law (A202n, 40 minutes of day 110):
        # 'good' (17 hearings), 'the', 'see', 'oh', 'you', 'pip' all carried one look, [0.04, 0, 0, 0.03] on the opponent axes, her skin
        # (rgba .86 .66 .54): she speaks face to face and the child's eyes rest on her face (the born face cue draws them there, A157),
        # so her face is the figure at most hearings and every word becomes its name ('see' primed 69 times and said 16 in half an
        # hour). An infant maps a word onto an object, not onto the speaker (the whole-object and novel-name assumptions, Markman
        # 1990; the speaker is the source of the sound); her face is read by its own born cue, and while it lies within the fovea's
        # window the window is hers: not a thing. (Her own name, 'mama', is the word-level mouth's road.) A202p: the object stream
        # looks beside her face (eyes.ground_appearance's `face`): the toy she holds up at her cheek as she names it is the look
        face_ = (float(fp[1]), float(fp[2]))
    body_ = frame.obs.get("body")
    gaze_ = body_[GAZE_AT:GAZE_AT + 3] if body_ is not None and len(body_) >= GAZE_AT + 3 else (0.0, 0.0, 0.0)
    return _eyes.ground_appearance(ef, ep if ep is not None and len(ep) >= 4 else None, face=face_, gaze=gaze_, own=frame.obs.get("self_cells"),
                                   her=frame.obs.get("her_cells"))         # (A202s; A217: her body's cells left out)   # (A202g: against the scene; A202m: None with no figure)


def _ground_periphery(frame):
    """A202: every colour periphery cell's look and its direction from the fovea's centre (eyes.ground_periphery over the frame's eye_p,
    the gaze from the body channel) -> (feats [15, 4], dirs [15, 2])"""
    from body.sim import eyes as _eyes
    ep = frame.obs.get("eye_p")
    if ep is None or len(ep) < 4:                                              # the eyes off (the night): no cells
        return np.zeros((0, 4)), np.zeros((0, 2))
    g = frame.obs["body"][GAZE_AT:GAZE_AT + 3]
    return _eyes.ground_periphery(ep, g)

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
SIZES = dict(face=2, ears=1725, eye_p=172, eye_f=1536, body=BODY_SIZE, touch=TOUCH_SIZE, vestibular=24)
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
# THE WITHDRAWAL'S LENGTH (3.7; body/sim/reflexes.py's WITHDRAW_TICKS: a big flexion step a tick for 2 ticks; innate, ours)
WITHDRAW_TICKS = 2
WITHDRAW_REST_TICKS = 10        # A178 (2026-10-02): after a withdrawal has run, the reflex rests this long (1.5 s) and the pain its own
                                # whip makes does not re-arm it; the rest counts down on pain-free ticks (the grasp's recovery scale, A162; ours)
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
    birth words (placeholders qaa-qbx when none are given: the core's tests); a word spelled as one letter keeps its own row, keyed by
    word_key"""
    words = list(words) if words is not None else ["q" + chr(97 + k // 26) + chr(97 + k % 26) for k in range(50)]
    if len(words) != 50 or len(set(words)) != 50 or any(not w or not w.isalpha() or w != w.lower() for w in words):
        raise ValueError(f"born_table: 50 distinct lower-case words, given {len(words)}")
    vocab = {"<rest>": 0, "<end>": 1, " ": 2}
    for k, ch in enumerate("abcdefghijklmnopqrstuvwxyz"):
        vocab[ch] = 3 + k
    for k, w in enumerate(words):
        vocab[word_key(w)] = 29 + k
    return Tokenizer(models.WordLevel(vocab=vocab, unk_token="<rest>"))


def word_key(w):
    """a birth word's key in the born table's vocabulary: the word itself, and a one-letter word (the parent's article "a") as
    "a_", so its own row never meets the letter's (the lexicon keeps the two apart: its word row and its letter row; S5a, the
    lead's reading of the two branches' tables, which disagreed only here)"""
    return w if len(w) > 1 else w + "_"


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
    """PAIN (6.2, A37; A91): -1 on a tick any of the frame's `pain` flags is set (a joint's outside torque past its limit, or the base's
    force past 3 x the body's weight, as the observer estimates them from the robot's own sensors); silent otherwise. It PAYS: a
    negative reward, the dopamine dip (the lateral habenula's road, Matsumoto and Hikosaka 2007), which also raises the body's stress
    as every dip does (body/core/mouth.py: cortisol with the dip), reaches the amygdala (its pain- head, the tag) and sets off the born
    cry. A88 had made it cortisol alone (dopamine off); the owner reversed that on 2026-09-26 (A91) after the first plumbing day
    showed the cortisol loop (C79): with nothing paying for pain, no value could learn to avoid it. Biology pays for pain."""

    def felt(self, frame, life):
        p_ = frame.obs.get("pain")
        return -1.0 if p_ is not None and any(float(x) > 0.0 for x in p_) else None


class Novelty(RewardSource):
    """NOVELTY AS REWARD (A127, the brain sprint; 6.3 to be): dopamine to the new, the hippocampus's mismatch signal to the VTA (Lisman and
    Grace 2005; Schultz's novelty responses): on a tick after the store kept a FRAME as new (body/core/frames.py _frame_write: its
    surprise past the write gate and no slot it merged into), NOVELTY_GAIN is felt, once per new frame; a frame that merges into a memory
    it has (the same thing seen again) pays nothing, so the drive habituates by the store's own law. Silent otherwise. Signs: positive
    only. On since dawn 24, a switch of the body. A127b (2026-09-28 16:25): PAID BY THE MISMATCH, NOVELTY_GAIN x (1 - the new frame's nearest
    stored key's cosine): the write gate is a quantile and passed a tenth of every day's frames forever (day 24's record: 296 payments in
    3,000 ticks, 1,200 a day against her smiles' 50), so a drive that paid the gain for each could never habituate; the comparator's own
    mismatch (Lisman and Grace 2005's CA1 signal) is small for a frame like the day's others and full for a new thing. A155 (2026-10-01):
    AND BY LEARNING PROGRESS, NOVELTY_GAIN x the mismatch x the share of the frame's surprise the body is learning away (frames._ferr_progress:
    per channel the relative fall of its error of late against its long run, the channels weighed by their share of this tick's error).
    Why: life day 52's sweep read the drive paying on a tenth of the day's ticks, 82% of them at a frame whose surprise was her words (the
    words channel's error 0.87 and never falling: her next word is unpredictable, not unlearned), and with the habit dopamine's to make
    (A152) the arms' habits were being shaped by the timing of her speech. The unpredictable pays nothing; what the body is getting
    better at foreseeing pays in proportion (Oudeyer and Kaplan 2007; Gottlieb et al. 2013)"""
    signs = (1.0,)

    def felt(self, frame, life):
        new = float(getattr(life, "_frame_novel", 0.0))
        if new <= 0.0:
            return None
        life._frame_novel = 0.0
        prog = float(life._ferr_progress()) if hasattr(life, "_ferr_progress") else 1.0   # A155: by learning progress (body/core/frames.py)
        self.last_progress = prog                                           # (the runner's record: `nov_p`)
        self.n_paid = int(getattr(self, "n_paid", 0)) + 1                  # the payments counted (the runner's record: `nov`; attribution)
        self.paid = float(getattr(self, "paid", 0.0)) + NOVELTY_GAIN * new * prog
        return NOVELTY_GAIN * new * prog                                   # A127b: graded by the mismatch (0 to 1); A155: and by the progress (0 to 1)


class Competence(RewardSource):
    """COMPETENCE AS REWARD (A181, 2026-10-03, the owner's word for the complete architecture): dopamine for the body's OWN ACTS' consequences
    becoming foreseeable. On the tick after an event's end (the frames', A180) the drive pays COMPETENCE_GAIN x the event's effort (the
    share of its ticks the body acted on: an event it lay still through pays nothing) x the competence progress (frames._ferr_own_progress:
    A184: per channel the rise, of late against the long run, of the forward model's SKILL on the ticks the body acted on, its forecast's
    error against the naive forecast's 'nothing changes', the channels counted alike; A181 paid the fall of the error itself, which a
    quieter world earns as well as a better model). The novelty drive (A127, A155) pays for the learnable surprise of the WORLD at a new frame;
    this pays for mastery of the body's own doing, whatever the world does: an infant repeats a movement while its effect is becoming
    predictable and drops it once it is (Piaget's circular reactions; competence motivation, White 1959; intrinsically motivated learning
    of own-action consequences, Oudeyer and Kaplan 2007; the sense of agency as forward-model reliability, Blakemore, Wolpert and Frith
    1998). Progress, never accuracy and never quiet: a body that lies still foresees itself perfectly and earns nothing; an end the night closed is not
    paid at dawn. Silent otherwise; signs positive only. A switch of the body (competence), measured on a day copy"""
    signs = (1.0,)

    def felt(self, frame, life):
        due = getattr(life, "_comp_due", None)
        if due is None:
            return None
        life._comp_due = None
        share, prog, t_ = due
        if int(getattr(life, "ticks", 0)) - int(t_) > 1 or share <= 0.0 or prog <= 0.0:
            return None
        self.last = (float(share), float(prog))                             # (the runner's record: comp_eff, comp_p)
        self.n_paid = int(getattr(self, "n_paid", 0)) + 1                  # the payments counted (the runner's record: `comp`)
        v = COMPETENCE_GAIN * float(share) * float(prog)
        self.paid = float(getattr(self, "paid", 0.0)) + v
        return v


COMPETENCE_GAIN = 0.25                  # an event's competence dopamine at full effort and full progress (A181; A200c: 0.5 -> 0.25 with the fall read against the error's spread): the novelty drive's magnitude,
                                       # a quarter of her smile's rise (ours, disclosed; the magnitude the day copy reads)
NOVELTY_GAIN = 0.5                     # a new frame's dopamine: a quarter of her smile's rise (+2), half of pain's -1 (ours, disclosed; the
                                       # magnitude the day copy reads)


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

    def reflex(self, frame, life, state):
        """THE FLEXOR WITHDRAWAL (3.7, A11, A37; S5a moves it from the world's zones to the joints' pain): a leg or an arm (the limbs
        with flexion joints, FLEXION) whose pain joints (an arm's include its hand's) carry a pain flag this tick is taken for
        WITHDRAW_TICKS ticks: a big flexion step on each of its flexion joints in its flexion sense a tick, its other joints held
        (setting 2), whatever its last move was and wherever on the limb it hurts: generalized and crude, as a newborn's is (Andrews
        and Fitzgerald 1994; Cornelissen et al. 2013; Holmberg and Schouenborg 1996). The count is kept in the effector's working
        state (saved with the body's day, A70). None when it does not fire"""
        if not self.spg:
            return None
        p_ = frame.obs.get("pain")
        hurt = p_ is not None and any(float(p_[j]) > 0.0 for j in self.pain_joints)
        k = int(state.get("withdraw", 0)); rest = int(state.get("withdraw_rest", 0))
        if k <= 0 and rest > 0 and not hurt:
            state["withdraw_rest"] = rest - 1                    # A178: the rest counts down on pain-free ticks alone
        if hurt and k <= 0 and rest <= 0:
            # A178 (2026-10-02): THE WITHDRAWAL'S REFRACTORY PERIOD. A pain re-arms the reflex only when none is running and its rest has
            # passed. Until A178 every pain tick reset the count, and day 66's copy (wrist_pain66.py, storm66.py) showed the loop: the
            # withdrawal's whip drove the resting wrist into its range end, that pain re-armed the withdrawal, and so on; 22 of 22 pain
            # joint-ticks fell on reflex ticks, the pain 15 to 20 a thousand all day where the frozen arm of day 65 had 1. A reflex that
            # fires once per episode and recovers in the quiet is the response decrement of habituation (Rankin et al. 2009), the
            # newborn's flexion withdrawal habituating to a repeated stimulus (Andrews and Fitzgerald 1994); the pain itself is felt as
            # before (its reward, the critics', the gate's input), only the reflex rests
            state["withdraw"] = WITHDRAW_TICKS; k = WITHDRAW_TICKS
        if k <= 0:
            return None
        state["withdraw"] = k - 1
        if k - 1 <= 0:
            state["withdraw_rest"] = WITHDRAW_REST_TICKS          # A178: the reflex has run: it rests
        a = 0
        for j in range(len(self.factors)):
            sg = self.spg.get(j)
            a = a * 5 + (2 if sg is None else (4 if sg > 0 else 0))
        return a

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


class Loco(Effector):
    """THE LOCOMOTION COMMAND (W8, 2026-10-09; effector 10, the last): the brain's drive to the body's gait circuit, go and turn, two
    channels of five settings stepping the commanded speed and turn (body/sim/world.py LOCO_*; the circuit: the expert gait the ghost
    holds in its command blocks, body/sim/ghost.py). The cortex does not make the steps; it says where (the locomotor command of the
    brainstem: Grillner 2006). Its consequence sense is the vestibular channel (its own motion felt), no joint of the body; its cost the
    gaze's. Born fresh at a load (its organs where the save has none, as the gaze's inverse was)"""

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
                           for n in ("face", "ears", "eye_p", "eye_f", "body", "touch", "vestibular")]
        lim = list(limits) if limits is not None else None

        def idx(js):
            return [PER_JOINT * _joint_index(j) + k for j in js for k in range(PER_JOINT)]
        tract = Tract("voice", [5] * len(TRACT), rest_id=(5 ** len(TRACT) - 1) // 2, sense="ears", inverse=True, fwd_gate=True, n_in=3,
                      intrinsic=True, cry={"posture": dict(CRY_POSTURE), "lungs": 0, "glottis": 1, "breath": ("body", BREATH_AT), "pain": "pain"})   # (A179: the glottis, for the breath's braking)
        voice = VoiceEffector("words", [self.vocab], rest_id=self.sil, end_id=self.space_id, reserved=self.bans, intrinsic=False)
        gaze = Gaze("gaze", [5, 5, 5], rest_id=62, sense="body", sense_idx=list(range(GAZE_AT, GAZE_AT + 6)), fwd_gate=True,
                    orient={0: ("yaw", 1), 1: ("pitch", 1)}, orient_gate=True, vor=[0, 1], n_in=3,
                    inverse=True)                                       # A98 (C82): an inverse model of its own windows' motion, so its
                                                                        # decisiveness is earned as the limbs' (A97) and the born
                                                                        # orienting weighs while it is soft (born fresh at a load)
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
                              n_in=n_in, twitch=True, **kw))                        # R8c: its joints twitch in active sleep (3.7, A46)
        rewards = [FaceIncrement("face", clip=2), JointPain("pain", signs=(-1.0,))]   # her face pays (+/-), pain pays (-): A91 (R7d: the
                                                                                     # heads face +/-, pain -); no charge (A88)
        if int((cfg or {}).get("novelty", SIM_CFG.get("novelty", 0))):
            rewards.append(Novelty("novelty", clip=NOVELTY_GAIN, signs=(1.0,)))        # A127: the new pays (+), a switch of the body
        if int((cfg or {}).get("competence", SIM_CFG.get("competence", 0))):
            rewards.append(Competence("competence", clip=COMPETENCE_GAIN, signs=(1.0,)))   # A181: mastery of its own doing pays (+), a switch of
        loco = Loco("loco", [5, 5], rest_id=12, sense=None, fwd_gate=False, n_in=2, inverse=False,   # W8 (the last: a loaded life's window grows at the end).
                    orient={1: ("yaw", -1)}, orient_gate=True,                # A216: the turn's positive step (+ rad/s) turns the body left, so a cue on
                    approach={"go": 0, "turn": 1, "cues": ("face",), "eye": ("body", GAZE_AT), "gyro": ("imu_torso", 5)})   # the right is turned toward by its negative step: sense -1;
                                                        # the born approach on the command (body/core/cord.py _approach_step): her face's bearing
                                                        # from the body is the eyes' yaw in the head (the body channel at GAZE_AT) plus the cue's
                                                        # offset from the fovea; the gate's second input "a cue appeared" (the waist's).
                                                        # W8 amended (2026-10-09, 23:58): NO CONSEQUENCE SENSE. Declared on the vestibular channel at first, its
                                                        # newborn forward half forecast raw accelerations (gravity at 9.8) and its squared error swamped the first
                                                        # night's lesson (night 138: the NREM loss 1091 against 61, REM's coherence 0.74 against 0.99). The command's
                                                        # consequence is the whole body's motion, read by the channels' own heads; the effector learns from its
                                                        # actor and gate under dopamine, as the words' output does
        self.channels, self.effectors, self.rewards, self.inner_at = chans, [tract, voice, gaze] + limbs + [loco], rewards, 2
        self.orienting = [# A202: the heard word's look found in the periphery (body/core/grounding.py): a standing cue, as her face is, so
                          # the born saccade turns the eyes to it and the orienting bias pulls while it stands; first among the standing
                          # cues (A208): a child hearing 'ball' looks at the ball before its hand or her face
                          OrientCue("named", "named_periph", fired=0, yaw=1, pitch=2, sense=1.0, zone=FOVEA_HALF),
                          # A215 (2026-10-08): where she looks draws the eyes next (gaze following, joint attention: the thing she names is
                          # the thing her eyes are on; body/sim/eyes.py gaze_cue, a stand-in read from the world as the face cue is)
                          OrientCue("gaze", "gaze_periph", fired=0, yaw=1, pitch=2, sense=1.0, zone=FOVEA_HALF),
                          # A211 (2026-10-07): the thing that stands out in view draws the eyes next (bottom-up salience with inhibition of
                          # return, body/sim/eyes.py salience_cue): the eyes go from thing to thing and back to her; before its own hand
                          # (the day-116 copy: the hand cue led the saccade on 125 ticks of 300 and the eyes held a toy on 7%; in a reach
                          # guided by the eye the eye is on the target, not the hand: Johansson et al. 2001)
                          OrientCue("salient", "salient_periph", fired=0, yaw=1, pitch=2, sense=1.0, zone=FOVEA_HALF),
                          # A208 (2026-10-07): its own moving hand draws the eyes (hand regard, visually guided reaching: the thing reached
                          # for enters the fovea with the hand; body/sim/eyes.py hand_cue), a standing cue the born saccade turns to (A186)
                          OrientCue("hand", "hand_periph", fired=0, yaw=1, pitch=2, sense=1.0, zone=FOVEA_HALF),
                          OrientCue("face", "face_periph", fired=0, yaw=1, pitch=2, sense=1.0, zone=FOVEA_HALF),
                          OrientCue("sound", "sound_side", fired=0, yaw=1, sense=-1.0, side_only=True, onset=True),
                          OrientCue("onset", "onset_periph", fired=0, yaw=1, pitch=2, sense=1.0, zone=FOVEA_HALF, onset=True)]
        # A202: THE GROUNDING OF WORDS IN JOINT ATTENTION: the G1's look is its colour (the colour window's and the colour periphery's
        # opponent code, body/sim/eyes.py: ground_appearance, ground_periphery; the gaze from the body channel, GAZE_AT); the end the
        # offset teaches and the space are never bound
        # A214: the words for what is going on bind to the ladder's band 3 (its clock 64 ticks, about 10 s: an event's span)
        self.situation = Situation(band=3, skip=tuple(int(x_) for x_ in (self.end_id, self.space_id) if x_ is not None),
                                   name_skip=tuple(int(i_) for i_ in (tok.token_to_id(ch_) for ch_ in "abcdefghijklmnopqrstuvwxyz") if i_ is not None))
        self.grounding = Grounding("named_periph", 4, _ground_appearance, _ground_periphery,
                                   skip=tuple(int(x_) for x_ in (self.end_id, self.space_id) if x_ is not None),
                                   name_skip=tuple(int(i_) for i_ in (tok.token_to_id(ch_) for ch_ in "abcdefghijklmnopqrstuvwxyz") if i_ is not None))   # A202q: the letters
        # THE CEREBELLUM'S INTERFACE (7.5, A44; the module's doc): the mossy numbers in their order, each named and declared by its
        # middle and half-range; the readouts on the waist's, the arms' and the legs' joints; the flocculus on the gaze's yaw and pitch
        mossy, off, half = [], [], []

        def fibre(name, mid, hr):
            mossy.append(name); off.append(float(mid)); half.append(float(hr))
        for e in self.effectors:                                       # (1) the efference copy of every motor effector's acts
            if e is voice or e is loco:
                continue                                               # the words' token output: no joint, no body sense (the lead's); W8: the
                                                                       # gait command is no joint either, and a living body's cerebellum is not
                                                                       # widened at a load (A20): its efference copy stays out of the mossy fibres
            jn = TRACT if e is tract else (("yaw", "pitch", "vergence") if e is gaze else (("go", "turn") if e is loco else tuple(BODY_JOINTS[j] for j in e.joints)))   # (W8: the command's two channels)
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
        # A191 (2026-10-04): THE AMYGDALA'S OWN LOW ROAD FOR STRAIN. A line only the amygdala reads (the nociceptive road to the
        # central amygdala through the parabrachial nucleus: Bernard and Besson 1990), firing while any gear is loaded over
        # STRAIN_LINE of its pain line (body/sim/world.py; the frame's "strain"): the warning under the pain line. The amygdala's pain
        # head forecast the next 8 ticks' pain at r 0.10 (reliability 0.16) from the stream and the 13 lines, of which "pain" fires
        # only once it hurts; strain stands at 0.82 of the pain line a tick before a pain onset and 0.66 four ticks before
        # (p1/strain_probe.py). The striatal expansion does not read it (its born rows are laid by the 13)
        self.amyg_events = [EventLine("strain", "strain", fired=(0,))]
        self.proprio = ("body",)                                       # A149: the body channel is the proprioceptive one (the limbs' and the gaze's own sense)
        self.heading = Heading("imu_torso", acc=(0, 1, 2), gyro=(3, 4, 5), dt=0.15)   # R7f: the heading from the torso's unit (the module's doc)


# THE CORE'S CONSTANTS THE SIM IS BORN WITH THAT R6h DECIDES (SIM_DESIGN.md 3.5, 3.6, 3.7, 10; A41, A47, A48; the language body holds none
# of them), and the cerebellum's switch (7.5, A44: on at birth, the anatomy declaring its mossy list, the lead's). S5a adds the rest of
# section 10's sim constants (the critics' solve every 256 ticks, the ventral critic's bands, the store's capacity).
SIM_CFG = dict(
    rest_token="<rest>", end_token="<end>", end_symbol="eot",          # the born table's rest and end (4.9)
    # the switches at birth (10): fixes #4, #5 and #8; chunk_gate 1 for every effector; chunk_max 8, a ceiling only (3.6)
    gate_own_draw=1, actor_trace_tick=1, elig_from=1, chunk_gate=1, chunk_max=8,
    # THE STRIATAL ACTOR ON (A113, 2026-09-27; C87): dopamine's lesson reaches a policy, so value becomes behaviour (the day copy of dawn 10:
    # pain 0.25% against 3.65%, 11 motor acts judged against 0 in the first 2,000 ticks). Its form, rate and forgetting are the core's
    # (physiology: actor_form 'add', actor_lr 0.02, actor_forget 36000)
    actor=1,
    stress_slow=1,      # A215 (2026-10-09): the stress follows the slow band's dip (the tonic loss), not dopamine's phasic dip (body/core/mouth.py)
    dopamine_adapt=1,   # A172 (2026-10-02): the actors learn from dopamine in units of its own spread (Tobler, Fiorillo and Schultz 2005); the
                        # actors had frozen from day 50 under A148's unit-power step with dopamine at 0.13 RMS and lr 0.02 (0.001 a logit a tick)
    # THE GATES' DRIVES, DISCLOSED (3.5, A41): the tonic drive following the reward rate in every gate, 0.25 + (sum over k < 12 of 0.8^k,
    # the gate's own eligibility window: 4.656) x the felt reward's running mean at the ladder's 256-tick clock (band 4); the performance
    # error at 0.5 on the gates that declare it (the tract's alone); gate_vigor 0, so the reward rate is not counted twice
    gate_tonic=0.25, gate_tonic_rate=sum(0.8 ** k for k in range(12)), gate_tonic_clock=4,
    gate_int=0.5, gate_int_form="error", gate_vigor=0.0,
    # R6h's motor effectors (MOTOR): movement units, act_inv batched every 8 ticks with the kappa correction, fatigue per effector
    unit_margin=math.log(4.0), act_inv_every=8, act_inv_chance=1, own_fatigue=1,
    gate_ceiling=1, gate_scaling=1,   # A183: the motor gates' ceiling (rest sampled as activity is: p <= 1 - gate_floor) and their synaptic
                                      # scaling (a logit past logit(1 - gate_floor) scaled back at the tag's reach); on from its landing
    # the born patterns and biases (REFLEX): the spinal pattern generators (their shape and cycles REFLEX's, C54: a movement of 2 ticks'
    # flexion and 3 ticks' extension, the extension returning the flexion's excursion, then a pause, each cycle drawn from the seed,
    # 3.56 +- 1.93 s held to 1.0-8.5 s), the born cry, orienting, the VOR
    spg=1, cry=1, breath=1, orient=1, vor=1, orient_saccade=1, approach=1,   # A216: the born approach (the locomotor command's step toward her face)   # A173: the born breath (the tract's tidal cycle below the gate)
    breath_brake=0.30,   # A179: the newborn's expiratory braking, the glottis narrowed this much of its range a tick through the expiration
                         # (the tract's position follows its target with its own lag: the glottis reaches 0.55 of its range by the fifth tick,
                         # just under the cry's 0.6, where its aerodynamics give a soft voicing on the expiration's last ticks, 5 to 7 mPa at
                         # 1 m against the open breath's 1: the newborn's end-expiratory grunt) and opened again on the inspiration (ours;
                         # Kosch and Stark 1984; at 0.24 the glottis reaches 0.48 and sounds 1.4 mPa, at 0.4 0.73 and 10 mPa)
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
    # STEP R7d: THE AMYGDALA ON AT BIRTH (7.4, 10, A16; body/core/amygdala.py): its 3 heads (face +/-, pain -; A88) over the
    # stream, the 13 event lines and the level; its constants AMYG's (none given here: tau_a band 6's clock, 4,096 ticks; the critics'
    # prior 0.3; the solve every 8 ticks; the reliability over 36,000 ticks, earned after 64 pairs)
    amyg=1,
    # A93: THE REVERSE VALUE SWEEP in the frames' night (physiology.SLEEP night_reverse): the day's transitions swept backwards from its
    # rewards, the tagged windows first, so a smile's value reaches the acts before it in one night (risk 1)
    night_reverse=1,
    # STEP R7f: RECALL INTO ACTION (7.6, A45; FRAMES' recall): a frame's key the stream plus the heading from the torso gyro, its value
    # its codes and every effector's efference copy, each motor effector's map from the recalled act born at zero; and working memory
    # latching at the frames' event ends in place of the utterances' (wm_frames)
    recall=1, wm_frames=1,
    novelty=1,                     # A127: the novelty drive (Novelty, NOVELTY_GAIN); on since dawn 24 (2026-09-28 13:56, the owner's all-in word)
    competence=1,                  # A181: the competence drive (Competence, COMPETENCE_GAIN): the body's own acts' consequences becoming foreseeable
                                   # pays at each event's end; on from its landing (2026-10-03, the owner's word for the complete architecture)
    goal_key=1,                    # A130: the held word as a recall key (frames._goal_trace, memory.GOAL_TAU/GOAL_SCALE); on since dawn 24
                                   # (the private-speech ruler, speech_act_reading.py)
    ctx_key=1,                     # A128: the held context as a recall key (memory.query_from, CTX_TAU/CTX_SCALE); on since dawn 24 (the
                                   # re-find and re-grasp rulers)
    value_forget=36000,            # A146 (2026-09-30): the ladder's value heads forget at a day's constant (critics.py): the differential heads'
                                   # weights random-walked to 651 and their values swung by 50 within half a day (day 44), shutting their gates
    inner_speech=1,                # A137: the inner word (frames._goal_trace: the voice's sure, unsounded top choice held; memory.INNER_P); on from
                                   # dawn 29 (the private-speech ruler with --covert)
    sharp_per_joint=1,             # A140 (C117): each joint's decisiveness its own, earned by its kappa and its settings' variety (mouth._motor_sharp)
    habit_by_credit=1,             # A141 (C149, 2026-09-30): the day's habit lesson weighted by dopamine's credit as the night's is (cortex._habit_weights)
    actor_slow_lr=1e-3,            # A142/C153 (2026-09-30): the actor's 64-tick tag captured by phasic dopamine (mouth._actor_tag_step, critics): at
                                   # 60 ticks after an act its credit through the tag (1e-3 x 0.39) equals the fast trace's (0.02 x 0.02); ours
    sharp_earned_norm=1,           # C148 (2026-09-30): the proposal's certainty earned by the same exponent (model.ActTable.logits): a joint that has
                                   # shown nothing draws from the cosines alone (day 40: the right shoulder yaw at +big 98% of ticks, kappa 0.06)
    imagine_key=1, imagine_pav=1,  # A138: waking imagination at each event's end (sleep._imagine: the REM rollout awake; IMAG_SCALE; the
                                   # amygdala's forecast on the imagined future into the gates); on from its dawn (the brain sprint)
    vte_act=1,                     # A190: an act weighed against not doing it, one effector a deliberation, the lean past the usual difference
    imagine_vte=1,                 # A182: vicarious trial and error: a second imagined future weighed against the first, the lean toward the
                                   # better one's first acts; on from its landing (2026-10-03, the owner's word for the complete architecture)
    # STEP R8: THE NIGHT OVER FRAMES (SIM_DESIGN.md 7.4 item 2, 8's R8 row, 9's tape; body/core/sleep.py; physiology.py SLEEP): each awake
    # tick taped beside its record, the day cut into episodes at nightfall at the frames' event ends, their entries and windows from the
    # tag reaching back over the day's record, kept across nights up to the episodes' cap
    night_frames=1,
    # THE NIGHT'S PASSES OVER FRAMES (R8b; PHYSIOLOGY's night_* and rem_*: "the core's night passes run as before", 5.4): the served
    # language body's own (ops/BASE_FLAGS.txt and its save's constants, tools/pins/served_cfg.pkl): the dreams in lockstep batches of 16,
    # 6 rounds, a dream for each memory the day added, never fewer than 1,024 or more than 2,048; the night's Adam at 1e-5 (the waking
    # rate), warmed over 8 steps, its second moment at 0.99; REM's words drawn at temperature 1 (its 6 rounds of 8 dreams of 8 free steps
    # are PHYSIOLOGY's, the served body's too; the served REM form, "imagine", reads the striatal critic, which the sim does not hold at
    # birth: REM on frames is the forecast form)
    night_batch=16, night_rounds=6, night_starts=1024, night_starts_max=2048, night_load=1.0, night_lr=1e-5, night_warm=8,
    night_beta2=0.99, rem_temp=1.0,
    rem_limbs=1,                   # A132: the limbs dream in REM (sleep._rem_limb_act: act_pred's proposal drawn at the REM temperature, an
                                   # efference copy in the dream's inputs; the motor cortex under atonia); on since dawn 25 (the all-in sprint)
    # THE LIFE DAY AND THE NIGHT (10, 5.4, A46, B19): 24,000 waking ticks, then a night as long (the night takes time: the critics
    # discount across it, night_ticks, and the live night steps the world that many ticks, R8c)
    wake_ticks=24000, night_ticks=24000,
    # STEP R8c: THE TWITCHES OF ACTIVE SLEEP IN THE LIVE, DARK NIGHT (3.7, 5.4, A46; body/core/sleep.py; physiology.py SLEEP): the world
    # stepped dark through the night (its `live_night`), the waist's, the arms', the hands' and the legs' joints twitching one at a time in
    # active sleep, each twitch's pair teaching act_inv and the forward half, the cerebellum learning wherever the world runs
    twitch=1,
    # A71, THE BORN CONFIG AT THE SERVED VALUES (the lead's decision of 2026-09-25 on the PFC-maturation study, docs/audit/pfc_maturation.md:
    # the prefrontal parts mature by use, not by calendar; born whole at full strength, their say grown through readouts born small and
    # reliability-weighted voices). Each value is the served language body's (tools/pins/served_cfg.pkl, its save's constants), where the
    # tests had added them by hand, unless said:
    # - THE SLOW BANDS KEPT ACROSS THE NIGHT (night_keep_bands 1): zeroed nightly, band 7 reached only about 72% of its settled level by the
    #   day's end and bands 6-7 became a clock of time since waking (the study, section 3);
    # - THE STRIATAL FAST CRITIC (fast_input "striatum", fast_rls 1, stri_k 8, stri_m 2,048, stri_quiet 1, fast_rls_prior 0.3): dopamine from
    #   the striatal expansion's least squares (R7a's event lines' rows then live), its solve every 256 ticks (SIM_DESIGN.md 7.2, 10: the
    #   sim's constant, the served body's 64);
    # - WORKING MEMORY'S SLOT (wm 1, wm_burst 0.5, wm_max 512: the served values, wm_max ours and unsourced): latched at the frames' event
    #   ends (wm_frames, R7f) and at dopamine's bursts;
    # - THE LONG CRITIC WITH ITS EARNED VOICE (vcrit_rls 1, vcrit_auto 1, vcrit_ceiling "earned", vcrit_forget 36,000, vcrit_traces 1,
    #   vcrit_clock 1, and its law's own served prior and horizons: vcrit_norm_tau 36,000, vcrit_rls_prior 3.0, vcrit_lambda 1.0,
    #   vcrit_center 0, without which its least squares would run in a form the served body never ran), reading the fast ladder bands 0-2
    #   (SIM_DESIGN.md 7.2's sim constant; the served "-") and solved every 256 ticks (10);
    # - AMYG_PAV BORN ON, ITS WEIGHT EARNED (amyg_pav 1, amyg_pav_form "earned": its weight the aversive heads' largest reliability, 0 at birth
    #   and grown by use), in place of a switch after birth (A20: a change of body)
    night_keep_bands=1,
    fast_input="striatum", fast_rls=1, stri_k=8, stri_m=2048, stri_quiet=1, fast_rls_prior=0.3, fast_rls_every=256,
    wm=1, wm_burst=0.5, wm_max=512,
    vcrit_rls=1, vcrit_auto=1, vcrit_ceiling="earned", vcrit_forget=36000, vcrit_traces=1, vcrit_clock=1, vcrit_norm_tau=36000,
    vcrit_rls_prior=3.0, vcrit_lambda=1.0, vcrit_center=0, vcrit_bands="0,1,2", vcrit_rls_every=256,
    amyg_pav=1, amyg_pav_form="earned",
)
