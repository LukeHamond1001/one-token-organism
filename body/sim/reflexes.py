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


def opening_act(hand):
    """the hand's opening act (A171): a big opening step on its closing joints, the thumb's rotation held"""
    return W.act_flat([_BIG[-CLOSING[hand][j]] if j in CLOSING[hand] else _MID for j in _JOINTS[hand]])


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
GRASP_HOLD_TICKS = 40           # A162: the grasp fires at full strength this long (6 s) under a constant pressure on the palm, then habituates
                                # (the reflex's response to a sustained, unchanging stimulus wanes: Thompson and Spencer 1966; the palmar grasp
                                # holds an object placed in a newborn's hand for seconds, not minutes: Twitchell 1965); the span is A4's hand-over
                                # window (her release once the hand has closed), ours
GRASP_RECOVER_TICKS = 10        # A162: a palm free of pressure this long (1.5 s) re-arms the grasp in full (the habituated response recovers
                                # once the stimulus is withdrawn: Thompson and Spencer 1966; ours). Not a touch onset: the grasp's own squeeze
                                # makes onsets at the palm every tick, so an onset would never let it habituate (measured on the test's ball)


def grasp(hand, own, palm_log, state=None, onset=0.0):
    """THE PALMAR GRASP at the spinal cord (see the module's doc): the hand's own act this tick (`own`, its flat act; None its rest)
    and its palm's touch (the frame's log force) give (the act its servos take, the event): (own, None) when the palm is not
    touched at GRASP_N; else the sum JOINT BY JOINT (A160, 2026-10-01; A35's own principle: the reflex and the descending command meet
    at the same motor neurons): each closing joint whose own step opens it keeps that step (the cortex overrides that joint), every
    other closing joint is stepped at least one big step closed (A82); the event "overridden" when every closing joint was opened by
    the own act (the hand opened as a whole), "grasp" otherwise. Until A160 one closing joint's opening step cancelled the reflex on
    the whole hand, which a hand acting at random does on 95 of 100 ticks (1 - (3/5)^6): life day 55's hand-overs saw the grasp
    overridden on 481 ticks and firing on 5, the toy set in its palm never held (22 of 27 released unclosed; 303 toys lost in the day).
    HABITUATION (A162, 2026-10-01; `state`: the hand's [ticks pressed running, ticks free running], the world's, saved with it): under a
    constant pressure the reflex fires in full for GRASP_HOLD_TICKS, then falls silent ((own, "habituated"): the hand's own acts rule the
    fingers, which stay where they are until an act moves them); a palm free GRASP_RECOVER_TICKS running re-arms it in full, and so does
    a touch ONSET at the palm while it is silent (A164, 2026-10-01: dishabituation by a changed stimulus, Thompson and Spencer 1966; the
    frame's touch_onset, a rise of the palm's force: a toy pressed into a resting fist. While the reflex fires its own squeeze makes
    onsets every tick, so an onset counts only once it is silent: day 56's hand-overs into a fist habituated on its own fingers ended
    'its hand never closed on it' 5 of 7 times). Day 56's
    first 3,000 ticks under A160 alone: the right hand a
    fist on nothing (its own fingers pressing its palm) on 42, 88 and 74% of the ticks, holding a toy on 14, 2 and 4% (16 to 25%
    before): a reflex that never wanes locks the hand; the reflex scaffolds the learned grasp and gives way to it. Without `state`
    the pure sum of the same tick (the tests' instrument)."""
    if hand not in CLOSING:
        raise ValueError(f"no palmar grasp on {hand!r}")
    if palm_log < GRASP_LOG:
        if state is not None:
            state[0] = 0; state[1] += 1
        return own, None
    if state is not None:
        if state[1] >= GRASP_RECOVER_TICKS:                          # the palm was free: the reflex in full again
            state[0] = 0
        state[1] = 0
        if state[0] > GRASP_HOLD_TICKS and float(onset) > 0.0:         # silent, and the palm's force rose: a new press, the reflex in full
            state[0] = 0
        state[0] += 1
        if state[0] > GRASP_HOLD_TICKS:
            return own, "habituated"
    n = len(_JOINTS[hand])
    rest = W.rest_id(n)
    dig = W.act_digits(rest if own is None else own, n)
    closed = 0
    for i, j in enumerate(_JOINTS[hand]):
        if j in CLOSING[hand]:
            step = W.SETTINGS[dig[i]] * CLOSING[hand][j]
            if step < 0:                                                 # its own step opens this joint: the cortex's, kept
                continue
            if step < W.STEP_BIG:
                dig[i] = _BIG[CLOSING[hand][j]]
            closed += 1
    if closed == 0:
        return own, "overridden"
    return W.act_flat(dig), "grasp"


DORSAL_N = GRASP_N              # A171: a push on the back of the hand that opens it (the palm's grasp threshold, the one force the skin
                                # is given a line at; ours)


def dorsal_open(hand, own, dorsal_N, palm_other_N, state=None):
    """THE DORSAL HAND RESPONSE at the spinal cord (A171, 2026-10-02): a touch on the BACK of the hand opens it. Stimulation of the
    dorsum of the hand or fingers extends the fingers in the newborn (the avoiding response: Twitchell 1965, 'The automatic grasping
    responses of infants'; a nurse opens a fisted hand by stroking its back), the palmar grasp's opposite number at the same motor
    neurons. `dorsal_N`: the tick's mean push on the hand's links by things not its own from their back (toward the palm's face: the
    world's `sides[h, 1]`); `palm_other_N`: the same things' push from the palm's side (toward the back: `sides[h, 0]`; a palm or
    fingers pressed from that side at GRASP_N is a grasp's, which wins; the hand's own fingers on its palm count for nothing here).
    -> (the act its servos take, the event): (own, None) when the back is not pushed at DORSAL_N or the palm is pressed; else the sum
    joint by joint as the grasp's (A160): each closing joint whose own step closes it keeps that step (the cortex's), every other is
    stepped at least one big step OPEN; the event "dorsal", or (own, None) when the own act closed every joint. HABITUATION as the
    grasp's (A162; `state` [ticks pushed running, ticks free running], the world's, saved with it): under a constant push it fires in
    full GRASP_HOLD_TICKS then falls silent ("dorsal_habituated": a hand resting on its back on the mat is not held open for ever),
    a back free GRASP_RECOVER_TICKS re-arms it. Why: on life days 55 to 61 both of the child's hands lay fisted on nothing (66 to 85
    deg closed: the grasp, habituated on its own fingers, leaves them where they closed) and her hand-overs set the toy against the
    knuckles: 16 of 28 'never closed on it' and the 12 'closed' mostly a toy pressed to the back of a fist (the probe on day 61's
    copy: 0 N on the palm, 1.7 to 7.9 N on the middle finger's back). The hand must open for a toy to reach the palm; this is how"""
    if hand not in CLOSING:
        raise ValueError(f"no dorsal response on {hand!r}")
    if dorsal_N < DORSAL_N or palm_other_N >= GRASP_N:
        if state is not None:
            state[0] = 0; state[1] += 1
        return own, None
    if state is not None:
        if state[1] >= GRASP_RECOVER_TICKS:
            state[0] = 0
        state[1] = 0
        state[0] += 1
        if state[0] > GRASP_HOLD_TICKS:
            return own, "dorsal_habituated"
    n = len(_JOINTS[hand])
    dig = W.act_digits(W.rest_id(n) if own is None else own, n)
    opened = 0
    for i, j in enumerate(_JOINTS[hand]):
        if j in CLOSING[hand]:
            step = W.SETTINGS[dig[i]] * CLOSING[hand][j]                 # positive: its own step closes this joint (the cortex's, kept)
            if step > 0:
                continue
            if -step < W.STEP_BIG:
                dig[i] = _BIG[-CLOSING[hand][j]]
            opened += 1
    if opened == 0:
        return own, None
    return W.act_flat(dig), "dorsal"


TRACTION_N = 2.0                # A177: a steady pull on the forearm along the arm, away from the shoulder, that elicits the traction response (ours:
                                # the examiner's pull at the wrists is a few newtons; Prechtl and Beintema 1964)
TRACTION_HOLD_TICKS = 40        # A177: the response holds its flexion this long (6 s) under a steady pull, then habituates (the grasp's A162 bookkeeping)
TRACTION_RECOVER_TICKS = 10     # A177: an arm free of traction this long (1.5 s) re-arms the response in full


def traction(limb, own, pull_N, state=None):
    """THE TRACTION RESPONSE (A177, 2026-10-02): a newborn pulled by the forearms from supine flexes its elbows and shoulders and takes part
    in its own pull to sit (Prechtl and Beintema 1964's neurological examination of the newborn; the response is present from birth and
    wanes over the first months). `pull_N` the pull along the limb away from the shoulder this tick (the world's reading of the parent's
    holds on the forearm); under TRACTION_N nothing. Composed with the own act as the grasp is (A160): a big flexion step on each of the
    limb's flexion joints (FLEXION: the shoulder pitch and the elbow) unless its own step extends that joint (the cortex's, kept); the
    other joints as its own act has them. Habituation as the grasp's (A162): full for TRACTION_HOLD_TICKS under a steady pull, then
    silent ((own, "habituated"): the arm's own acts rule) until the arm has been free TRACTION_RECOVER_TICKS. Returns (act, event):
    (own, None) when it does not fire. Life day 65: the pull-to-sit stopped at the guide's cap 4 of 4 times ('the child resisted, or its
    joint is at its range'), the arm driving into its stops against her; A9 lets the pull rise only with the child's own flexion, which
    a newborn supplies by this response"""
    if limb not in FLEXION:
        raise ValueError(f"no traction response on {limb!r}")
    if pull_N < TRACTION_N:
        if state is not None:
            state[0] = 0; state[1] += 1
        return own, None
    if state is not None:
        if state[1] >= TRACTION_RECOVER_TICKS:
            state[0] = 0
        state[1] = 0
        state[0] += 1
        if state[0] > TRACTION_HOLD_TICKS:
            return own, "habituated"
    n = len(_JOINTS[limb])
    rest = W.rest_id(n)
    dig = W.act_digits(rest if own is None else own, n)
    flexed = 0
    for i, j in enumerate(_JOINTS[limb]):
        if j in FLEXION[limb]:
            step = W.SETTINGS[dig[i]] * FLEXION[limb][j]
            if step < 0:                                                 # its own step extends this joint: the cortex's, kept
                continue
            if step < W.STEP_BIG:
                dig[i] = _BIG[FLEXION[limb][j]]
            flexed += 1
    if flexed == 0:
        return own, "overridden"
    return W.act_flat(dig), "traction"


# THE ARMS' RESTING TONE (A185): the born resting posture of each arm's proximal joints, measured on the G1's own geometry (the lead,
# 2026-10-03; p1/rest_pose_probe.py): the upper arm along the trunk (shoulder pitch 0), a little off it (roll 0.2 rad outward), no
# turn about its own axis (yaw 0), the elbow flexed past its right angle (-0.7 rad of its flexion sense: the forearm some 130 degrees to
# the upper arm), which holds the hand before the chest, about 0.3 m before the eye and in its image's middle. Ours, disclosed: the
# sources give the posture in words (the arms flexed and adducted, the hands before the chest), never in this body's joint angles
TONE_REST = {"arm_l": {"left_shoulder_pitch_joint": 0.0, "left_shoulder_roll_joint": 0.2, "left_shoulder_yaw_joint": 0.0, "left_elbow_joint": -0.7},
             "arm_r": {"right_shoulder_pitch_joint": 0.0, "right_shoulder_roll_joint": -0.2, "right_shoulder_yaw_joint": 0.0, "right_elbow_joint": -0.7}}
TONE_GAIN = 0.3                 # a tick: the tone's step is this share of the joint's distance from its rest (ours): a soft spring, at the
                                # shoulder's and the elbow's gain (40 N m a rad) 12 N m a rad, three and a half times the gradient of the
                                # forearm's weight about the elbow (3.4 N m a rad), so a forearm standing near its rest stays standing;
                                # an own small step held against it moves the joint 0.3 rad, the babble's range about the rest
TONE_STEP = W.STEP_SMALL        # rad a tick: the tone's pull at its strongest, one small step (ours: the body's smallest own act; 3.6 N m at
                                # those joints, just over the 3.4 N m the forearm and hand weigh at the elbow lying flat, so the tone can
                                # lift the forearm off the mat, as the newborn's arm recoil does, and no more)


def tone(limb, q):
    """THE RESTING TONE (A185, 2026-10-03): the tonic stretch reflex holds a limb's resting posture (postural tone: Sherrington 1909;
    Liddell and Sherrington 1924, the stretch reflex), and the spinal cord's own circuits define equilibrium postures the limb converges
    to from wherever it is (the convergent force fields of the spinalized frog: Bizzi, Mussa-Ivaldi and Giszter 1991, Science 253:287-291;
    the equilibrium point as what the cord holds and the descending command shifts: Feldman 1986). The term newborn's resting posture is
    flexion, the arms flexed and adducted, and an arm drawn out of it springs back (the posture and the arm recoil of the newborn's
    neurological examination: Amiel-Tison 1968, Arch Dis Child 43:89-93; Dubowitz, Dubowitz and Goldberg 1970, J Pediatr 77:1-10;
    Ballard et al. 1991, J Pediatr 119:417-423). `q`: the limb's joint angles this tick in its joints' order (the cord's own afferent, the
    spindles'). Returns one additive step per joint toward the joint's resting angle, TONE_GAIN x its distance from the rest and at
    most TONE_STEP (a soft spring that saturates: near the rest a gentle pull an own small step outweighs, far from it one small step a
    tick), 0 on a joint with no declared rest; None for a limb with none. Summed with the limb's own act and the cord's other patterns on every joint, with it or
    against it (A48, A97's law), below the gate: no efference copy, no credit. WHY: this body's servo re-anchors every target at the
    measured angle each tick (3.5), so a limb the cortex does not drive has no posture at all and lies where gravity and its last
    pushes left it; day 76's arms lay wedged beside and behind the trunk (the left shoulder 1.7 rad back, the right arm overhead), the
    hands 59 to 76 degrees off the eye's axis, in its image on 0.0% and 4.1% of ticks: a body that never sees its own hands has no
    road to hand regard (White, Castle and Held 1964), to seeing what its own acts do, or to imitation"""
    rest = TONE_REST.get(limb)
    if rest is None:
        return None
    out = []
    for i, j in enumerate(_JOINTS[limb]):
        if j in rest:
            out.append(min(max(TONE_GAIN * (float(rest[j]) - float(q[i])), -TONE_STEP), TONE_STEP))
        else:
            out.append(0.0)
    return tuple(out)


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


def opens_all(hand, act):
    """whether a hand's act steps every closing joint toward open (the hand opened as a whole: the grasp's override, A160)"""
    return all(W.SETTINGS[k] * CLOSING[hand][j] < 0 for j, k in zip(_JOINTS[hand], W.act_digits(act, len(_JOINTS[hand]))) if j in CLOSING[hand])


def opens(hand, act):
    """whether a hand's act steps any closing joint toward open"""
    for j, k in zip(_JOINTS[hand], W.act_digits(act, len(_JOINTS[hand]))):
        if j in CLOSING[hand] and W.SETTINGS[k] * CLOSING[hand][j] < 0:
            return True
    return False


# ---------------------------------------------------------------- A193 (2026-10-04): the standing and the stepping reflexes
# THE POSITIVE SUPPORTING REACTION AND THE STEPPING REFLEX, born (the owner's word: walking, as fast as it can be had). A newborn held
# upright with its soles on a surface stiffens its legs and bears weight (the positive supporting reaction: Magnus 1926; Peiper 1963),
# and, moved forward, steps: a leg in stance whose hip has extended while the other leg stands swings forward and is set down ahead
# (the stepping reflex; in supported infants a leg's swing begins on its hip's extension with its load taken by the other leg: Pang
# and Yang 2000, J Physiol 528:389-404; newborn stepping and supine kicking are one pattern: Thelen and Fisher 1982). Both at the cord,
# below the gate, summed with the own act and the cord's other patterns (A48, A97's law), read from the body's own senses alone: the
# torso unit's specific force (upright: its long axis carries STAND_UP_G of gravity) and the soles' touch (a sole loaded over SOLE_N).
# Lying, sitting, or held with its feet off the floor, neither fires: the life on the mat is untouched until it is stood up.
# Measured on the day-84 copy (p1/upright*.py; the child stood up by an instrument, her hands stood in for by a capped spring at the
# trunk: 80 N sideways, 40 N m, at most 0.6 of its weight carried): alone it falls in a second with or without them (no balance:
# that is hers to give and its cerebellum's to learn); held, without them it sinks in 12 s; with the supporting reaction it stands
# 39 s on 0.9 of its own weight; with the stepping reflex and her hold pacing 0.20 m ahead of its feet at 0.10 m/s it walked 0.64 m
# in 45 s in 10 steps without sinking. The constants below are ours, from that grid; the joint's own rate (one big step, 0.27 rad a
# tick) sets the swing's length: 4 ticks of lift, 3 of placing, about a second, as supported infants' steps are.
STAND_LIMBS = ("waist", "leg_l", "leg_r")
STAND_UP_G = 5.6                # m/s2 along the torso's long axis: upright within about 55 deg (9.81 cos 55 deg; ours). 8.5 (30 deg) as
                                # first built: raised by her hands (C268) it hung at 35 to 50 deg with its soles loaded and its legs
                                # limp, and never came under itself; at 55 deg the thrust takes its weight as soon as it is half up
SOLE_N = 20.0                   # N on a sole: loaded (ours: a sixteenth of the body's weight)
STAND_GAIN, STAND_STEP = 1.0, W.STEP_BIG        # the supporting reaction is an extensor THRUST: the whole distance to the straight leg,
                                                # at most one big step a tick. With the tone's soft spring (0.3, one small step) the
                                                # held child stood but could not rise: led up by the chest within her sustained cap
                                                # (156 N) it stayed sitting; with the thrust it came from lying to its feet in 9 s
                                                # on her 100 to 156 N and stood on its own legs with 20 to 45 N of her steadying
                                                # (p1/raise.py, the day-84 copy)
STEP_EXT = 0.08                 # rad: the stance hip's extension that starts the swing (ours; 0.12 on the stand-in's grid, 0.08 for
                                # her hands' slower lead: C268's copy walked 0.40 m at it)
STEP_UNLOAD = 0.30 # A194: the swing begins only in a leg carrying no more than this share of the soles' load: the
                                # stance-to-swing transition needs the hip extended AND the leg unloaded (the cat's and the infant's
                                # stepping: Grillner and Rossignol 1978 the hip's extension, Duysens and Pearson 1980 the extensors'
                                # unloading; recalled). A loaded leg that swung dragged its foot and pulled the pelvis back (the day-85
                                # copy's held walk: 0.04 m in 125 ticks). The share: ours
STEP_LIFT, STEP_PLACE = 4, 3    # ticks: hip and knee flexing (the foot lifted and brought forward), then the knee extending (set down)
STEP_HIP, STEP_KNEE = -0.9, 1.3 # rad: the swing's hip flexion and knee flexion targets (ours)
STEP_HIP_PLACE = 0.8            # the hip's target while the foot is set down, as a share of STEP_HIP


def stand(q, imu_torso, soles, state):
    """THE STANDING AND STEPPING REFLEXES' TICK. `q` {limb: its joints' angles, in the effector's order} for STAND_LIMBS; `imu_torso`
    the torso unit's tick means [acc 3, gyro 3]; `soles` (left, right) the soles' forces (N); `state` {leg: [phase, ticks]} kept by
    the world and changed in place -> ({limb: its additive steps}, {limb: "stand" | "step"}); ({}, {}) when the body is not upright
    on a loaded sole (the legs' phases return to stance)"""
    az = float(imu_torso[2])
    if az < STAND_UP_G or max(float(soles[0]), float(soles[1])) < SOLE_N:
        for k in state:
            state[k][0], state[k][1] = "stance", 0
        return {}, {}
    out, ev = {}, {}
    out["waist"] = tuple(float(min(max(-float(x), -W.STEP_BIG), W.STEP_BIG)) for x in q["waist"])   # the trunk held over the pelvis
    ev["waist"] = "stand"
    load = float(soles[0]) + float(soles[1])
    for leg, other in (("leg_l", "leg_r"), ("leg_r", "leg_l")):
        mine = float(soles[0 if leg == "leg_l" else 1])
        js = _JOINTS[leg]; iq = {j.split("_", 1)[1].replace("_joint", ""): i for i, j in enumerate(js)}
        st = state[leg]; ql = q[leg]
        if st[0] == "stance" and float(ql[iq["hip_pitch"]]) > STEP_EXT and state[other][0] == "stance" and mine <= STEP_UNLOAD * load:
            st[0], st[1] = "swing", 0
        tgt = {k: 0.0 for k in iq}; cap = {k: STAND_STEP for k in iq}; gain = {k: STAND_GAIN for k in iq}
        if float(ql[iq["hip_pitch"]]) > 0.0:                            # the thrust EXTENDS: a hip already extended (the leg trailing as
            cap["hip_pitch"] = W.STEP_SMALL; gain["hip_pitch"] = 0.3    # the body passes over its foot) is held softly (the tone's
                                                                        # spring), or the stance hip could never extend and no step begin
        if st[0] == "stance" and state[other][0] == "swing":            # the stance leg stands firm while the other swings
            for k in ("knee", "hip_pitch", "hip_roll", "ankle_pitch", "ankle_roll"):
                cap[k] = W.STEP_BIG; gain[k] = 1.0
        if st[0] == "swing":
            st[1] += 1
            if st[1] <= STEP_LIFT:
                tgt.update(hip_pitch=STEP_HIP, knee=STEP_KNEE)
            else:
                tgt.update(hip_pitch=STEP_HIP * STEP_HIP_PLACE, knee=0.05)
            for k in ("hip_pitch", "knee", "ankle_pitch"):
                cap[k] = W.STEP_BIG; gain[k] = 1.0
            ev[leg] = "step"
            if st[1] >= STEP_LIFT + STEP_PLACE:
                st[0], st[1] = "stance", 0
        else:
            ev[leg] = "stand"
        steps = [0.0] * len(js)
        for k, i in iq.items():
            steps[i] = float(min(max(gain[k] * (tgt[k] - float(ql[i])), -cap[k]), cap[k]))
        out[leg] = tuple(steps)
    return out, ev
