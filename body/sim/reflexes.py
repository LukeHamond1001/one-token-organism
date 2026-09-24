"""THE G1'S SPINAL REFLEXES (docs/SIM_DESIGN.md 3.7 and the decision log A11, A12, C22; the build plan's W1): the two the design
keeps for the limbs, computed from the frame as the body's own afferents give it (the world's touch, pain and pain's skin site,
body/sim/world.py), the withdrawal as the act it forces on its limb this tick (or None), the grasp as its sum with the hand's own
act. They are the body's, below the gate. The sim's anatomy declares the withdrawal on each limb's effector (the core's
`Effector.reflex(frame, life, state)`, step R6), whose tick then gives the gate no eligibility and the actor no credit, and whose act
reaches the world as any act; the grasp is summed at the spinal cord with the hand's own act (`grasp`, run by the world's apply). None
of the refused reflexes (stepping, righting, the tonic neck and labyrinthine reflexes, Moro, rooting, Galant, Babinski, placing) is
here. The VOR is the world's (body/sim/world.py, on the software fovea); orienting is a bias on the gaze's and the waist's proposals,
the core's (R6h).

THE WITHDRAWAL, WITH ITS LOCAL SIGN (Sherrington 1910 named the flexion reflex's "local sign": its form changes with the skin that is
stimulated; Schouenborg and colleagues found it built of modules, one per muscle, each module's receptive field the skin its own
contraction withdraws from a stimulus, strongest where it withdraws it best: Schouenborg and Kalliomaki 1990, Schouenborg and Weng
1994; in people, Andersen, Sonnenborg and Arendt-Nielsen 1999 on the foot sole; all recalled, to check). When a zone of a limb feels
pain (the frame's `pain`), its nociceptors' place is the frame's `pain_site` for that zone: where on its link the force pressed and
the link's outward normal there, in the link's own frame. The spinal cord knows its own limb: from the body's joint sense (the
frame's `body` channel) it places the site and its normal, and for each joint of the limb between the trunk and that link it reads how
far a turn of the joint withdraws the site (the site's motion along the inward normal per radian: -(axis x (site - joint)) . normal,
the joint's module's drive), summed over the limb's zones in pain. Each joint whose drive is at least RF_HALF of the strongest takes
one big step (0.27 rad) the way that withdraws; the limb's other joints hold where they are (setting 2, re-anchored at the measured
angle), so the push that pressed stops. It lasts WITHDRAW_TICKS ticks: the act computed at the pain's tick, again on the next
(computed afresh while the pain lasts). A zone pressed from opposite sides (a shin between a weight and the mat) has no direction to
withdraw in: its sites' mean normal is short, and below RF_HALF of a unit the modules cancel; if no zone of the limb gives a
direction, the limb rests for the reflex's ticks (its targets relax: the push stops; the limb is not driven into either side). The
head and the trunk have none. A hand's zones (the palm, the fingers) are the arm's: a hand in pain withdraws the arm (the hand's own
joints are the hand's effector).
WHY NOT THE GENERALIZED FLEXION (3.7 as first written: "one big flexion step of its joints"): the W1 fix built it, and the W1
verifier measured it raising the pain it answered (C22: 2 of 9 onsets; pressing harder than resting in 6 of 9): a flexion drives a
knee into its own chest and a thigh's housing into the pelvis. The design's own bar (C22: no withdrawal that raises the pain it
answers) is what the local sign is for in biology. Ours is born tuned (a newborn's is coarser, its receptive fields larger and
tuned after birth by the body's own movements: Andrews and Fitzgerald 1994; Petersson et al. 2003, in rats; recalled): flagged for
the design (3.7, 10). The spinal cord reads its own limb's geometry from a kinematic copy of the body (the model's own links and
axes, set to the frame's joint angles), the anatomy it was born with; nothing of the world.

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
would forbid learning to let go, which A11 asks for (a conflict written in the W1 fix's report). It fires on its own fingers
too (a fist closed on nothing stays closed until the hand's own act opens it); the world's truth `palm_own_N` counts those (W4)."""
import math
import sys
from pathlib import Path

import mujoco
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from body.sim import world as W  # noqa: E402

G = W.G

LIMBS = ("leg_l", "leg_r", "arm_l", "arm_r")                     # the limbs with a withdrawal (the waist and the head have none)
WITHDRAW_TICKS = 2              # the withdrawal's big step, a tick for 2 ticks (3.7; innate, ours)
RF_HALF = 0.5                   # a withdrawal module's receptive field at its half-maximum: a joint takes part where its drive is at
                                # least half the strongest joint's, and the pressing sites of a zone give a direction only when their
                                # mean outward normal is at least half a unit long (3.7, C22; innate, ours)
GRASP_N = 0.3                   # palm touch that closes the hand (3.7; the prototype's value; innate, ours)

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


def joint_angles(frame):
    """the 43 joints' angles (rad, W.JOINTS order) as the body's joint sense gives them: the frame's body channel carries each
    angle scaled over its range to -pi/2..pi/2 as its sine and cosine, so atan2 gives it back"""
    b = np.asarray(frame.obs["body"])[:4 * len(W.JOINTS)].reshape(len(W.JOINTS), 4)
    return np.arctan2(b[:, 0], b[:, 1])


class Reflexes:
    """the withdrawal over one world's zones (their order is the frame's) and its own kinematic copy of the body:
    `withdrawal(frame, limb, state)`, the act it forces or None, the core's hook on each limb's effector (step R6: it takes the
    limb's tick); `state` is the effector's working state (the core's life.motor entry), where the withdrawal keeps its count and
    its act. `drive(frame, limb, zones)` is each of the limb's joints' module drive (m of withdrawal per rad), for the
    instruments. The grasp is the spinal cord's (`grasp`), run by the world."""

    def __init__(self, world):
        m = world.m
        self.zones = list(world.zones)
        self.limb_zones = {limb: limb_zones(self.zones, limb) for limb in LIMBS}
        self.palm = {h: self.zones.index(z) for h, z in PALM_ZONE.items()}
        self.m = m                                                      # read only: the body's links and axes
        self.d = mujoco.MjData(m)                                       # the spinal cord's own copy of the limb's pose
        self.zone_body = np.asarray(world.zone_body).copy()
        jid = np.array([m.joint(j).id for j in W.JOINTS])
        self.qadr = m.jnt_qposadr[jid].copy()
        lo, hi = m.jnt_range[jid, 0], m.jnt_range[jid, 1]
        self.mid, self.half = (hi + lo) / 2, (hi - lo) / 2
        self.limb_jid = {limb: np.array([m.joint(j).id for j in _JOINTS[limb]]) for limb in LIMBS}
        self.moves = {}                                                 # (limb, zone): which of the limb's joints move that zone's link
        for limb in LIMBS:
            for z in self.limb_zones[limb]:
                up, b = set(), int(self.zone_body[z])
                while b > 0:
                    up.add(b); b = int(m.body_parentid[b])
                self.moves[(limb, z)] = np.array([int(m.jnt_bodyid[j]) in up for j in self.limb_jid[limb]])

    def drive(self, frame, limb, zones):
        """each of the limb's joints' withdrawal drive from the frame's pain sites on `zones` (m of the site's withdrawal per rad of
        the joint; + means a positive turn withdraws), and how many zones gave a direction"""
        m, d = self.m, self.d
        d.qpos[self.qadr] = self.mid + self.half * joint_angles(frame) / (math.pi / 2)
        mujoco.mj_kinematics(m, d)
        site = np.asarray(frame.obs["pain_site"]).reshape(len(self.zones), 6)
        e = np.zeros(len(self.limb_jid[limb])); used = 0
        for z in zones:
            n_l = site[z, 3:]
            c = float(np.linalg.norm(n_l))
            if c < RF_HALF:                                             # pressed from opposite sides: no direction
                continue
            b = int(self.zone_body[z])
            Rb = d.xmat[b].reshape(3, 3)
            p = d.xpos[b] + Rb @ site[z, :3]
            n = Rb @ (n_l / c)
            used += 1
            for k, (j, mv) in enumerate(zip(self.limb_jid[limb], self.moves[(limb, z)])):
                if mv:
                    e[k] -= float(np.cross(d.xaxis[j], p - d.xanchor[j]) @ n)
        return e, used

    def withdrawal_act(self, frame, limb, zones):
        """the withdrawal's act for the limb's zones in pain: a big step, the withdrawing way, on each joint whose drive is at
        least RF_HALF of the strongest, the others held; the limb's rest when no zone gives a direction"""
        e, used = self.drive(frame, limb, zones)
        top = float(np.abs(e).max()) if used else 0.0
        if top <= 0.0:
            return W.EFFECTOR_REST[limb]
        return W.act_flat([_BIG[1 if x > 0 else -1] if abs(x) >= RF_HALF * top else _MID for x in e])

    def withdrawal(self, frame, limb, state):
        if limb not in LIMBS:
            return None
        pain = frame.obs.get("pain")
        hurt = [z for z in self.limb_zones[limb] if pain is not None and pain[z] > 0]
        if hurt:
            state["withdraw"] = WITHDRAW_TICKS
            state["withdraw_act"] = int(self.withdrawal_act(frame, limb, hurt))
        if state.get("withdraw", 0) > 0:
            state["withdraw"] -= 1
            return state["withdraw_act"]
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
