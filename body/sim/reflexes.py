"""THE G1'S SPINAL REFLEXES (docs/SIM_DESIGN.md 3.7 and the decision log A11, A12, C22; the build plan's W1): the two the design
keeps for the limbs, computed from the frame as the body's own afferents give it (the world's touch and pain, body/sim/world.py),
the withdrawal as the act it forces on its limb this tick (or None), the grasp as its sum with the hand's own act. They are the
body's, below the gate. The sim's anatomy declares the withdrawal on each limb's effector (the core's `Effector.reflex(frame, life,
state)`, step R6), whose tick then gives the gate no eligibility and the actor no credit, and whose act reaches the world as any
act; the grasp is summed at the spinal cord with the hand's own act (`grasp`, run by the world's apply). Of the refused
reflexes (stepping, the tonic neck and labyrinthine reflexes, Moro, rooting, Galant, Babinski, placing) none is here but one, kept
in part since A92: the newborn's prone pattern (`prone`, below). The VOR
is the world's (body/sim/world.py, on the software fovea); orienting is a bias on the gaze's and the waist's proposals, the
core's (R6h).

THE FLEXOR WITHDRAWAL, AS A NEWBORN HAS IT (3.7 and section 10 as the design approved them; the owner's decision of 2026-09-24,
made for him in the W1 verifier's third round). When any zone of a limb feels pain (the frame's `pain`: the zone's largest 10 ms
mean force over F_pain), that limb takes one big flexion step (0.27 rad) of each of its flexion joints a tick, for WITHDRAW_TICKS
ticks, whatever its last move was and wherever on the limb it hurts; its other joints hold (setting 2, re-anchored where they
are). The head and the trunk have none. A hand's zones (the palm, the fingers) are the arm's: a hand in pain withdraws the arm.
It is generalized and crude, whole-limb, as a newborn's is:
  - in human newborns the withdrawal is evoked from the whole limb (from the sole to the top of the thigh and the buttock) at low
    thresholds, and the spatial tuning of the reflex grows with age: Andrews and Fitzgerald 1994 (Pain 56:95-101); young infants'
    nociceptive flexion reflexes are bilateral and non-specific, their spatial organisation increasing with age: Cornelissen et
    al. 2013 (PLoS ONE 8:e76470);
  - in newborn rats the withdrawal reflexes are functionally unadapted and often move the limb TOWARD the stimulus; the adult,
    site-specific organisation (each muscle's module withdrawing the skin it moves away: the "local sign") emerges over the first
    three postnatal weeks: Holmberg and Schouenborg 1996 (J Physiol 493:239-252);
  - that tuning is learned from the body's own movements: the tactile feedback of spontaneous twitches in sleep guides the
    spinal self-organisation by a correlation-based learning (Petersson, Waldenstrom, Fahraeus and Schouenborg 2003, Nature
    424:72-75; Waldenstrom et al. 2003, J Neurosci 23:7719-7725).
  So the precise local withdrawal of the W1 fix 2 (a local sign computed on an exact kinematic copy of the body, born tuned) is
  gone, with its skin-site afferent (`pain_site`: no approved part reads it). A newborn's withdrawal can press a limb into a worse
  contact (a flexing hip drives the thigh's housing into the pelvis, a knee toward the chest): C22 writes down how often, at birth,
  with the limb at rest and the babble as the chance comparisons (tools/sim_pain.py); it falls only if a learned mechanism tunes
  the reflex, which is an open design item, not built.
Each limb's flexion joints and their flexion signs, measured on the G1's own geometry (body/tests/test_sim_world.py, world 8):
  a leg:  the triple flexion of the lower limb: the hip's flexion (hip pitch, negative: the knee forward), the knee's (positive)
          and the ankle's dorsiflexion (ankle pitch, negative: the toe up); the hip's roll and yaw and the ankle's roll hold;
  an arm: the shoulder's flexion (shoulder pitch, negative: the arm forward, off the mat for the body on its back) and the elbow's
          (negative: the hand toward the shoulder); the shoulder's roll and yaw hold (abduction and rotation, no flexion), and so do
          the wrist's three: the Dex3's fingers close across the wrist's pitch axis, so no wrist joint has a flexion sense on this
          hand (measured: a wrist pitch of 0.1 rad moves the index finger 0.2-0.5 mm along the fingers' closing direction), as the
          thumb's rotation has no closing sense in the grasp.

THE PALMAR GRASP (spinal, present at birth). A palm touch above GRASP_N (the palm zone's tick-mean force) closes the hand's
flexing joints one big step a tick (A82: under the Dex3's real motors, A80, a small step's servo torque, 0.25 N m, is under the
finger gear's friction, 0.27 N m, and could never close the hand; the newborn's grasp is strong, bearing its weight for moments in
the traction response: recalled, Twitchell 1965) (the thumb's two flexing joints and both fingers' two joints each; the thumb's rotation,
hand_thumb_0, turns the thumb about the palm and has no closing sense, so it takes the hand's own setting), unless the hand's own
act THAT TICK opens it (any closing joint stepped toward open: the cortex overrides). It never opens by itself: letting go is the
child's own act, learned (A11). It is not a core hook that takes the hand's tick: a hook is decided before the tick's own act is
drawn, takes the tick from the gate and sends the effector's rest as its act, so on a touched palm the hand's own act could never
be drawn, and a grasp would hold for ever (the W1 verifier's finding: a ball kept in the palm, 40 ticks of 40 taken by the reflex,
the hand's gate drawn 0 times). It is the spinal summation instead, as in the cord, where a reflex and the descending command
meet at the same motor neurons: the hand's gate draws every tick, its own act (or its rest) comes down, and `grasp` sums the
closing step into it where the own act does not close a joint already (the world's apply runs it on each hand's act before
anything moves: body/sim/world.py). The own act keeps its eligibility, since it was the gate's; the reflex's part is no act of the
gate's (the cortex sees it through touch and joint sense, and through its forward model's error: a closing it did not send).
The design's "their ticks are logged as reflex and carry no gate eligibility" (3.7) holds for the withdrawal; for the grasp it
would forbid learning to let go, which A11 asks for (a conflict written in the W1 fix's report). It fires on its own fingers
too (a fist closed on nothing stays closed until the hand's own act opens it); the world's truth `palm_own_N` counts those (W4).
THE PRONE PATTERN (A92, the owner's word 2026-09-26: "add 1"; a born reflex of the brainstem and cord, kept in part). A newborn laid
on its front turns its head to one side and holds its arms flexed under its chest, the weight on the forearms and the chest, not
on extended wrists: the protective head turn is present from birth (Prechtl and Beintema 1964, The Neurological Examination of
the Full-term Newborn Infant; Prechtl 1977), and the physiological flexor tone of the newborn's limbs holds the arms flexed in
prone (Bly 1994, Motor Skills Acquisition in the First Year). Rolling over is not a newborn's reflex (it is learned, at 3-6
months), so no roll is here. The first plumbing day after A88 found why the pattern matters on this body: it rolled onto its
front under babble and lay with its wrists under its 34 kg, the 5 N m gears back-driven, in pain on 12% of ticks (C79). The
G1 has no neck (A22), so its head turn is the waist's yaw. As the cord has it, from the body's own afferent: while the torso's
accelerometer reads its chest normal pointing down (its x component below -PRONE_G, the parent's own reading of "face down":
parent_motion.Child), each arm's flexion joints (the withdrawal's, FLEXION) take one small flexion step a tick, and the waist
yaws one small step toward the side that is up (the accelerometer's y component's sign), unless the own act that tick steps a
joint the other way (the own act wins there, as it opens the grasp); an own step the same way, or a hold, sums with it. Logged as
reflex (the truth's spinal: "prone" per effector), no gate eligibility. The legs keep their own acts (the newborn's flexed hips in
prone would drive the thigh housings into the pelvis on this body: C22's artifact), disclosed."""
import math
import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from body.sim import world as W  # noqa: E402

G = W.G

LIMBS = ("leg_l", "leg_r", "arm_l", "arm_r")                     # the limbs with a withdrawal (the waist and the head have none)
WITHDRAW_TICKS = 2              # the withdrawal's big flexion step, a tick for 2 ticks (3.7; innate, ours)
TENDON_TICKS = 2                # A139: the tendon organ's inhibition holds a joint's drive relaxed this long after a tick at its load line (ours)
GRASP_N = 0.3                   # palm touch that closes the hand (3.7; the prototype's value; innate, ours)
PRONE_G = 0.6 * 9.81            # face down: the torso accelerometer's chest-normal component below -0.6 g (Child's fz < -0.6; A92, ours)
PRONE_LIMBS = ("arm_l", "arm_r")  # the arms flex under the chest; the head turn is the waist's yaw (no neck: A22)
WAIST_YAW = "waist_yaw_joint"   # its positive step turns the trunk, and the cameras, to the left (body/sim/anatomy.py)

# each limb's flexion: {joint: the sign of its flexion} (measured on the G1: body/tests/test_sim_world.py, world 8)
FLEXION = {"leg_l": {"left_hip_pitch_joint": -1, "left_knee_joint": +1, "left_ankle_pitch_joint": -1},
           "leg_r": {"right_hip_pitch_joint": -1, "right_knee_joint": +1, "right_ankle_pitch_joint": -1},
           "arm_l": {"left_shoulder_pitch_joint": -1, "left_elbow_joint": -1},
           "arm_r": {"right_shoulder_pitch_joint": -1, "right_elbow_joint": -1}}
# each Dex3 hand's closing: {joint: the sign that closes it} (the URDF's ranges: a finger is open at 0; measured on the G1)
CLOSING = {"hand_l": {"left_hand_thumb_1_joint": +1, "left_hand_thumb_2_joint": +1, "left_hand_index_0_joint": -1,
                      "left_hand_index_1_joint": -1, "left_hand_middle_0_joint": -1, "left_hand_middle_1_joint": -1},
           "hand_r": {"right_hand_thumb_1_joint": -1, "right_hand_thumb_2_joint": -1, "right_hand_index_0_joint": +1,
                      "right_hand_index_1_joint": +1, "right_hand_middle_0_joint": +1, "right_hand_middle_1_joint": +1}}
HAND_OF_ARM = {"arm_l": "hand_l", "arm_r": "hand_r"}
PALM_ZONE = {"hand_l": "left_hand_palm", "hand_r": "right_hand_palm"}

_JOINTS = dict(G.EFFECTORS)
_MID = W.SETTINGS_PER_JOINT // 2                        # setting 2: no step
_BIG = {+1: W.SETTINGS.index(W.STEP_BIG), -1: W.SETTINGS.index(-W.STEP_BIG)}
_SMALL = {+1: W.SETTINGS.index(W.STEP_SMALL), -1: W.SETTINGS.index(-W.STEP_SMALL)}


def flexion_act(limb):
    """the limb's withdrawal act: a big flexion step on each of its flexion joints, the rest held"""
    return W.act_flat([_BIG[FLEXION[limb][j]] if j in FLEXION[limb] else _MID for j in _JOINTS[limb]])


def closing_act(hand):
    """the hand's grasp act: a big closing step on its closing joints, the thumb's rotation held (A82)"""
    return W.act_flat([_BIG[CLOSING[hand][j]] if j in CLOSING[hand] else _MID for j in _JOINTS[hand]])


def limb_zones(zones, limb):
    """the touch zones whose pain withdraws the limb (an arm's includes its hand's)"""
    groups = W.zone_groups(zones)
    out = list(groups[limb])
    if limb in HAND_OF_ARM:
        out += groups[HAND_OF_ARM[limb]]
    return sorted(out)


class Reflexes:
    """the withdrawal over one world's zones (their order is the frame's; given the world or its zone names):
    `withdrawal(frame, limb, state)`, the act it forces or None, the core's hook on each limb's effector (step R6: it takes the
    limb's tick); `state` is the effector's working state (the core's life.motor entry), where the withdrawal keeps its count. It
    reads the frame's pain alone. The grasp is the spinal cord's (`grasp`), run by the world."""

    def __init__(self, world_or_zones):
        self.zones = list(getattr(world_or_zones, "zones", world_or_zones))
        self.limb_zones = {limb: limb_zones(self.zones, limb) for limb in LIMBS}
        self.palm = {h: self.zones.index(z) for h, z in PALM_ZONE.items()}

    def withdrawal(self, frame, limb, state):
        if limb not in FLEXION:
            return None
        pain = frame.obs.get("pain")
        if pain is not None and any(pain[z] > 0 for z in self.limb_zones[limb]):
            state["withdraw"] = WITHDRAW_TICKS
        if state.get("withdraw", 0) > 0:
            state["withdraw"] -= 1
            return flexion_act(limb)
        return None


GRASP_LOG = math.log1p(GRASP_N / W.TOUCH_UNIT_N)                 # the palm's touch (log force) at the grasp's threshold


def grasp(hand, own, palm_log):
    """THE PALMAR GRASP at the spinal cord (see the module's doc): the hand's own act this tick (`own`, its flat act; None its rest)
    and its palm's touch (the frame's log force) give (the act its servos take, the event): (own, None) when the palm is not
    touched at GRASP_N; (own, "overridden") when the own act opens the hand; else (own with each closing joint stepped at least one
    big step closed, "grasp"; A82)."""
    if hand not in CLOSING:
        raise ValueError(f"no palmar grasp on {hand!r}")
    if palm_log < GRASP_LOG:
        return own, None
    n = len(_JOINTS[hand])
    rest = W.rest_id(n)
    dig = W.act_digits(rest if own is None else own, n)
    if own is not None and opens(hand, own):
        return own, "overridden"
    for i, j in enumerate(_JOINTS[hand]):
        if j in CLOSING[hand] and W.SETTINGS[dig[i]] * CLOSING[hand][j] < W.STEP_BIG:
            dig[i] = _BIG[CLOSING[hand][j]]
    return W.act_flat(dig), "grasp"


def tendon(steps, loaded, state, joints_of):
    """THE TENDON ORGAN'S AUTOGENIC INHIBITION at the spinal cord (A139, C113; the Golgi tendon organ's Ib afferent inhibits the motor
    neurons of the muscle whose tension is excessive: Houk and Henneman 1967; the clasp-knife's road). `loaded`: per joint (JOINTS' order)
    whether the gear's sensed 10 ms peak load reached its line last tick (the pain flag's own afferent); a joint that did is inhibited
    for TENDON_TICKS ticks: its step this tick is zeroed (its servo target re-anchored to its angle: the drive relaxed), over the own
    act, the cord's step and the withdrawal alike. `steps` {effector: per-joint steps} changed in place; `state` the per-joint
    countdown (the world's, saved with it) changed in place; `joints_of` {effector: [joint index in JOINTS]} -> {effector: [joint
    names relaxed]} for the truth's spinal (no gate eligibility moves: the own act stays the gate's, as the grasp leaves it; the
    cortex sees the relaxed joint through its joint sense and its forward error). Day 26's morning (C113): the elbow's withdrawal
    fired on the pain its own blocked flexion step made, 440 ticks at the motor's limit; this reflex breaks that loop in one tick"""
    state[loaded] = TENDON_TICKS
    ev = {}
    for eff, st_ in steps.items():
        idx = joints_of[eff]
        hit = [k for k, j in enumerate(idx) if state[j] > 0 and st_[k] != 0.0]
        if hit:
            st_[hit] = 0.0
            ev[eff] = [_JOINTS[eff][k] for k in hit]
    state[state > 0] -= 1
    return ev


def prone(acts, imu_torso):
    """THE PRONE PATTERN at the spinal cord (the module's doc, A92): `acts` the tick's acts (a dict, changed in place), `imu_torso` the
    torso unit's tick means [acc 3, gyro 3] in the torso's frame -> {effector: "prone"} for the effectors it stepped (none when the
    body is not face down). Each arm's flexion joints one small flexion step where the own act does not step against; the waist's
    yaw one small step toward the side that is up."""
    ax, ay = float(imu_torso[0]), float(imu_torso[1])
    if ax > -PRONE_G:
        return {}
    ev = {}
    plan = {limb: {j: FLEXION[limb][j] for j in FLEXION[limb]} for limb in PRONE_LIMBS}
    plan["waist"] = {WAIST_YAW: (1 if ay > 0 else -1)}
    for eff, steps in plan.items():
        n = len(_JOINTS[eff])
        own = acts.get(eff)
        dig = W.act_digits(W.rest_id(n) if own is None else own, n)
        changed = False
        for i, j in enumerate(_JOINTS[eff]):
            s = steps.get(j)
            if s is None:
                continue
            cur = W.SETTINGS[dig[i]] * s
            if cur < 0:
                continue                                          # its own act against it: the own act wins on this joint
            if cur < W.STEP_SMALL:
                dig[i] = _SMALL[s]; changed = True
        if changed:
            acts[eff] = W.act_flat(dig); ev[eff] = "prone"
    return ev


def opens(hand, act):
    """whether a hand's act steps any closing joint toward open"""
    for j, k in zip(_JOINTS[hand], W.act_digits(act, len(_JOINTS[hand]))):
        if j in CLOSING[hand] and W.SETTINGS[k] * CLOSING[hand][j] < 0:
            return True
    return False
