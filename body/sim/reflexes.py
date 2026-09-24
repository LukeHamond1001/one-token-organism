"""THE G1'S SPINAL REFLEXES (docs/SIM_DESIGN.md 3.7 and the decision log A11, A12; the build plan's W1): the two the design keeps
for the limbs, computed from the frame as the body's own afferents give it (the world's touch and pain, body/sim/world.py), the
withdrawal as the act it forces on its limb this tick (or None), the grasp as its sum with the hand's own act. They are the body's,
below the gate. The sim's anatomy declares the withdrawal on each limb's effector (the core's `Effector.reflex(frame, life,
state)`, step R6), whose tick then gives the gate no eligibility and the actor no credit, and whose act reaches the world as any
act; the grasp is summed at the spinal cord with the hand's own act (`grasp`, run by the world's apply). None of the refused
reflexes (stepping, righting, the tonic neck and labyrinthine reflexes, Moro, rooting, Galant, Babinski, placing) is here. The VOR is the world's (body/sim/world.py,
on the software fovea); orienting is a bias on the gaze's and the waist's proposals, the core's (R6h).

THE FLEXOR WITHDRAWAL (Sherrington; spinal and lifelong). When a zone of a limb feels pain (the frame's `pain`: the zone's
largest 10 ms mean force over F_pain), that limb takes one big flexion step of its flexion joints a tick for WITHDRAW_TICKS
ticks, whatever its last move was; its other joints hold (setting 2, re-anchored where they are). The head and the trunk have
none. Each limb's flexion step, fixed here (the build plan's W1), with the sign of each joint's flexion measured on the G1's
geometry (body/tests/test_sim_world.py checks each sign):
  a leg:  the triple flexion of the lower limb's flexion reflex: the hip's flexion (hip pitch, negative), the knee's (positive)
          and the ankle's dorsiflexion (ankle pitch, negative);
  an arm: the elbow's flexion (negative), which draws the hand toward the shoulder; the shoulder's and the wrist's joints hold
          (an arm's withdrawal on a supine body moves the hand off what struck it without swinging the whole arm into the mat).
  A hand's zones (the palm, the fingers) are the arm's: a hand in pain withdraws the arm.
THE PALMAR GRASP (spinal, present at birth). A palm touch above GRASP_N (the palm zone's tick-mean force) closes the hand's
flexing joints one small step a tick (the thumb's two flexing joints and both fingers' two joints each; the thumb's rotation,
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
would forbid learning to let go, which A11 asks for (a conflict written in the W1 fix's report)."""
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from body.sim import world as W  # noqa: E402

G = W.G

WITHDRAW_TICKS = 2              # the withdrawal's big flexion step, a tick for 2 ticks (3.7; innate, ours)
GRASP_N = 0.3                   # palm touch that closes the hand (3.7; the prototype's value; innate, ours)

# each limb's flexion: {joint: the sign of its flexion} (measured on the G1: body/tests/test_sim_world.py)
FLEXION = {"leg_l": {"left_hip_pitch_joint": -1, "left_knee_joint": +1, "left_ankle_pitch_joint": -1},
           "leg_r": {"right_hip_pitch_joint": -1, "right_knee_joint": +1, "right_ankle_pitch_joint": -1},
           "arm_l": {"left_elbow_joint": -1}, "arm_r": {"right_elbow_joint": -1}}
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
    """the limb's withdrawal act: a big flexion step on its flexion joints, the rest held"""
    return W.act_flat([_BIG[FLEXION[limb][j]] if j in FLEXION[limb] else _MID for j in _JOINTS[limb]])


def closing_act(hand):
    """the hand's grasp act: a small closing step on its closing joints, the thumb's rotation held"""
    return W.act_flat([_SMALL[CLOSING[hand][j]] if j in CLOSING[hand] else _MID for j in _JOINTS[hand]])


def limb_zones(zones, limb):
    """the touch zones whose pain withdraws the limb (an arm's includes its hand's)"""
    groups = W.zone_groups(zones)
    out = list(groups[limb])
    if limb in HAND_OF_ARM:
        out += groups[HAND_OF_ARM[limb]]
    return sorted(out)


class Reflexes:
    """the withdrawal over one world's zones (their order is the frame's): `withdrawal(frame, limb, state)`, the act it forces or
    None, the core's hook on each limb's effector (step R6: it takes the limb's tick); `state` is the effector's working state (the
    core's life.motor entry), where the withdrawal keeps its count. The grasp is the spinal cord's (`grasp`), run by the world."""

    def __init__(self, zones):
        self.zones = list(zones)
        self.limb_zones = {limb: limb_zones(self.zones, limb) for limb in FLEXION}
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
    small step closed, "grasp")."""
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
        if j in CLOSING[hand] and W.SETTINGS[dig[i]] * CLOSING[hand][j] < W.STEP_SMALL:
            dig[i] = _SMALL[CLOSING[hand][j]]
    return W.act_flat(dig), "grasp"


def opens(hand, act):
    """whether a hand's act steps any closing joint toward open"""
    for j, k in zip(_JOINTS[hand], W.act_digits(act, len(_JOINTS[hand]))):
        if j in CLOSING[hand] and W.SETTINGS[k] * CLOSING[hand][j] < 0:
            return True
    return False
