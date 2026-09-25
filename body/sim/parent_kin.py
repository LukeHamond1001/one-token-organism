"""Shared numbers and kinematics for the living room (docs/SIM_DESIGN.md; from the 2026-09-24 prototype; the world's side, never the body).

The parent's skeleton: 16 segments posed here in Python (forward kinematics, two-bone inverse kinematics for the arms and
legs, look-at for the head and eyes). Her planner makes the pose she means with it; her body is 16 dynamic MuJoCo bodies with
this skeleton's joints, driven toward that pose with a woman's strength (parent_body.py, make_g1room.py: the lead's decision of
2026-09-25). Its fingers, eyes, brows, lids and mouth are geoms whose local poses this file sets (hand shapes, gaze,
expression). Nothing here is the child's body or its control: the child is a physical humanoid in the XML.

Frames: every segment's frame is aligned with the parent's root in the rest pose (standing, arms down, palms to the
thighs): x forward, y to its left, z up. Limbs hang along -z from their proximal joint. A right-side segment is the
left one mirrored in y. Units m, radians (degrees only where a name says _DEG)."""
import math

import numpy as np
from scipy.spatial.transform import Rotation as Rot


# ---------------------------------------------------------------- rotations
def rx(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def ry(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def rz(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def axang(axis, a):
    axis = np.asarray(axis, float)
    return Rot.from_rotvec(axis / np.linalg.norm(axis) * a).as_matrix()


def mjquat(R):
    x, y, z, w = Rot.from_matrix(R).as_quat()
    return np.array([w, x, y, z])


def unit(v):
    v = np.asarray(v, float)
    n = np.linalg.norm(v)
    return v / n if n > 1e-12 else v


def perp(v, axis):
    """v's component perpendicular to the unit axis, normalised (falls back to any perpendicular)."""
    p = np.asarray(v, float) - np.dot(v, axis) * axis
    if np.linalg.norm(p) < 1e-9:
        p = np.cross(axis, [0, 0, 1]) if abs(axis[2]) < .9 else np.cross(axis, [1, 0, 0])
    return unit(p)


def frame(x, z):
    """A rotation matrix with columns x, y = z cross x, z (x made perpendicular to z)."""
    z = unit(z)
    x = perp(x, z)
    return np.column_stack([x, np.cross(z, x), z])


# ---------------------------------------------------------------- the parent's skeleton (H = 1.68 m)
H = 1.68
HIP_Z = .89                     # standing hip-joint height (Winter's segment ratios x H, rounded)
L_UA, L_FA, L_TH, L_SH = .30, .25, .41, .41
SEG_PARENT = {"pelvis": None, "abdomen": "pelvis", "chest": "abdomen", "head": "chest",
              "upper_arm_L": "chest", "forearm_L": "upper_arm_L", "hand_L": "forearm_L",
              "upper_arm_R": "chest", "forearm_R": "upper_arm_R", "hand_R": "forearm_R",
              "thigh_L": "pelvis", "shin_L": "thigh_L", "foot_L": "shin_L",
              "thigh_R": "pelvis", "shin_R": "thigh_R", "foot_R": "shin_R"}
SEGS = list(SEG_PARENT)
_OFF = {"abdomen": (0, 0, .10), "chest": (0, 0, .16), "head": (0, 0, .235),
        "upper_arm_L": (0, .178, .215), "forearm_L": (0, 0, -L_UA), "hand_L": (0, 0, -L_FA),
        "thigh_L": (0, .09, 0), "shin_L": (0, 0, -L_TH), "foot_L": (0, 0, -L_SH)}
OFFSET = {}
for k, v in _OFF.items():
    OFFSET[k] = np.array(v, float)
    if k.endswith("_L"):
        OFFSET[k[:-1] + "R"] = np.array([v[0], -v[1], v[2]], float)

# human joint ranges (degrees; typical adult active ranges, AAOS-style, rounded; the knee's 160 is its passive range,
# which kneeling on the heels uses). Signs: flexion positive; abduction/external rotation positive on both sides.
LIM_DEG = dict(shoulder_flex=(-60, 180), shoulder_abd=(-40, 180), shoulder_rot=(-90, 90), elbow=(0, 150),
               pronation=(-90, 90), wrist_flex=(-70, 80), wrist_dev=(-25, 35),
               hip_flex=(-25, 135), hip_abd=(-30, 50), hip_rot=(-45, 45), knee=(0, 160),
               ankle_dorsi=(-55, 25), ankle_inv=(-30, 25), ankle_twist=(-15, 15),
               lumbar_flex=(-30, 60), lumbar_lat=(-30, 30), lumbar_rot=(-20, 20),
               chest_flex=(-20, 40), chest_lat=(-25, 25), chest_rot=(-40, 40),
               neck_flex=(-50, 60), neck_lat=(-40, 40), neck_rot=(-75, 75))


def side_sign(side):
    return 1 if side == "L" else -1


class Pose:
    """A full parent pose: root placement plus each segment's rotation relative to its parent segment."""

    def __init__(self, pos=(0, 0, HIP_Z), R=None):
        self.pos = np.array(pos, float)
        self.R = np.eye(3) if R is None else np.array(R, float)
        self.local = {s: np.eye(3) for s in SEGS}
        self.world_override = {}          # segment -> world rotation (set by IK; converted to local by fk)
        self.hand = {"L": dict(curl=.25, thumb=.2, index=None), "R": dict(curl=.25, thumb=.2, index=None)}
        self.expr = 0.0                   # -1 frown .. 0 neutral .. +1 smile
        self.gaze = None                  # world point the eyes look at (None: straight ahead)
        self.report = {}                  # joint angles (deg) and limit checks from the IK calls

    def copy(self):
        p = Pose(self.pos, self.R)
        p.local = {k: v.copy() for k, v in self.local.items()}
        p.world_override = {k: v.copy() for k, v in self.world_override.items()}
        p.hand = {k: dict(v) for k, v in self.hand.items()}
        p.expr, p.gaze, p.report = self.expr, None if self.gaze is None else np.array(self.gaze), dict(self.report)
        return p


def fk(pose):
    """World (pos, R) of every segment's frame (its proximal joint)."""
    out = {}
    for s in SEGS:
        par = SEG_PARENT[s]
        if par is None:
            out[s] = (pose.pos.copy(), pose.R.copy())
            continue
        pp, pR = out[par]
        p = pp + pR @ OFFSET[s]
        R = pose.world_override[s] if s in pose.world_override else pR @ pose.local[s]
        out[s] = (p, R)
    return out


def _check(report, name, val_deg):
    lo, hi = LIM_DEG[name]
    report[name] = round(float(val_deg), 1)
    if not (lo - .5 <= val_deg <= hi + .5):
        report.setdefault("violations", []).append(f"{name} {val_deg:.0f} not in [{lo}, {hi}]")


def _ball_angles(Rrel, side, kind, report, prefix):
    """Decompose a ball joint's relative rotation into flex (about -y), abd (about +x on the left) and rot (about +z on
    the left), intrinsic in that order, and check the limits; tries both Euler solutions and keeps the one inside."""
    s = side_sign(side)
    best = None
    b, a, g = Rot.from_matrix(Rrel).as_euler("YXZ")
    for (bb, aa, gg) in ((b, a, g), (b + math.pi, math.pi - a, g + math.pi)):
        wrap = lambda t: (t + math.pi) % (2 * math.pi) - math.pi
        q = (math.degrees(wrap(-bb)), math.degrees(wrap(aa)) * s, math.degrees(wrap(gg)) * s)
        names = (f"{kind}_flex", f"{kind}_abd", f"{kind}_rot")
        excess = sum(max(0, LIM_DEG[n][0] - v, v - LIM_DEG[n][1]) for n, v in zip(names, q))
        if best is None or excess < best[0]:
            best = (excess, q, names)
    for n, v in zip(best[2], best[1]):
        _check(report, n if not prefix else n, v)
    return best[1]


def arm_ik(pose, side, wrist, pole, hand_R=None, segs=None):
    """Two-bone IK for one arm. wrist: world target of the wrist joint; pole: world direction the elbow points to;
    hand_R: the hand's desired world rotation (None: a straight wrist). Writes world rotations into the pose and
    returns the reach error in metres (0 when the target is reachable)."""
    segs = segs or fk(pose)
    cp, cR = segs["chest"]
    s = cp + cR @ OFFSET[f"upper_arm_{side}"]
    d = np.asarray(wrist, float) - s
    dist = np.linalg.norm(d)
    dcl = float(np.clip(dist, abs(L_UA - L_FA) + 1e-3, L_UA + L_FA - 1e-4))
    err = max(0.0, dist - dcl)
    dh = unit(d)
    ph = perp(pole, dh)
    a = (L_UA ** 2 + dcl ** 2 - L_FA ** 2) / (2 * L_UA * dcl)
    e = s + L_UA * (a * dh + math.sqrt(max(0, 1 - a * a)) * ph)
    w = s + dh * dcl
    b1, b2 = unit(e - s), unit(w - e)
    Ru = frame(-ph, -b1)                                  # the forearm bends toward the segment's +x
    th = math.acos(float(np.clip(np.dot(b1, b2), -1, 1)))
    Rf0 = Ru @ axang((0, -1, 0), th)
    rep = pose.report.setdefault(f"arm_{side}", {})
    rep.clear()
    _check(rep, "elbow", math.degrees(th))
    _ball_angles(cR.T @ Ru, side, "shoulder", rep, "")
    if hand_R is None:
        Rf, Rh = Rf0, Rf0
        _check(rep, "pronation", 0.0)
    else:
        rel = Rf0.T @ np.asarray(hand_R)
        psi, phi, chi = Rot.from_matrix(rel).as_euler("ZXY")
        sg = side_sign(side)
        Rf = Rf0 @ rz(psi)
        Rh = np.asarray(hand_R)
        _check(rep, "pronation", math.degrees(psi) * sg)
        _check(rep, "wrist_flex", -math.degrees(phi) * sg)
        _check(rep, "wrist_dev", math.degrees(chi))
    pose.world_override[f"upper_arm_{side}"] = Ru
    pose.world_override[f"forearm_{side}"] = Rf
    pose.world_override[f"hand_{side}"] = Rh
    rep["reach_err_m"] = round(err, 4)
    return err


def leg_ik(pose, side, ankle, knee_dir, foot_R=None, segs=None):
    """Two-bone IK for one leg: ankle target (world), the direction the knee points (world), the foot's world rotation
    (None: the shin's). Returns the reach error."""
    segs = segs or fk(pose)
    pp, pR = segs["pelvis"]
    h = pp + pR @ OFFSET[f"thigh_{side}"]
    d = np.asarray(ankle, float) - h
    dist = np.linalg.norm(d)
    dcl = float(np.clip(dist, 1e-3 + abs(L_TH - L_SH), L_TH + L_SH - 1e-4))
    err = max(0.0, dist - dcl)
    dh = unit(d)
    ph = perp(knee_dir, dh)
    a = (L_TH ** 2 + dcl ** 2 - L_SH ** 2) / (2 * L_TH * dcl)
    k = h + L_TH * (a * dh + math.sqrt(max(0, 1 - a * a)) * ph)
    b1, b2 = unit(k - h), unit(h + dh * dcl - k)
    Rt = frame(ph, -b1)                                   # the knee points along the thigh's +x
    th = math.acos(float(np.clip(np.dot(b1, b2), -1, 1)))
    Rs = Rt @ axang((0, 1, 0), th)
    rep = pose.report.setdefault(f"leg_{side}", {})
    rep.clear()
    _check(rep, "knee", math.degrees(th))
    _ball_angles(pR.T @ Rt, side, "hip", rep, "")
    if foot_R is None:
        Rfoot = Rs
    elif np.isscalar(foot_R):                              # an ankle dorsiflexion angle (rad) relative to the shin
        Rfoot = Rs @ axang((0, -1, 0), float(foot_R))
    else:
        Rfoot = np.asarray(foot_R)
    b, a_, g = Rot.from_matrix(Rs.T @ Rfoot).as_euler("YXZ")
    _check(rep, "ankle_dorsi", -math.degrees(b))
    _check(rep, "ankle_inv", math.degrees(a_) * side_sign(side))
    _check(rep, "ankle_twist", math.degrees(g))
    pose.world_override[f"thigh_{side}"] = Rt
    pose.world_override[f"shin_{side}"] = Rs
    pose.world_override[f"foot_{side}"] = Rfoot
    rep["reach_err_m"] = round(err, 4)
    return err


def spine(pose, lumbar=(0, 0, 0), chest=(0, 0, 0)):
    """Spine bends in degrees: (flex forward, lateral to the right, twist to the left) for the lumbar and chest joints."""
    rep = pose.report.setdefault("spine", {})
    for seg, (f_, l_, t_), nm in (("abdomen", lumbar, "lumbar"), ("chest", chest, "chest")):
        pose.local[seg] = ry(math.radians(f_)) @ rx(math.radians(l_)) @ rz(math.radians(t_))
        _check(rep, f"{nm}_flex", f_); _check(rep, f"{nm}_lat", l_); _check(rep, f"{nm}_rot", t_)


def look(pose, target, segs=None):
    """Point the head at a world target (neck yaw and pitch within their limits); the eyes take the remainder."""
    segs = segs or fk(pose)
    cp, cR = segs["chest"]
    neck = cp + cR @ OFFSET["head"]
    v = cR.T @ (np.asarray(target, float) - (neck + cR @ np.array([0, 0, PUPIL_Z])))
    yaw = math.degrees(math.atan2(v[1], v[0]))
    pitch = math.degrees(-math.atan2(v[2], math.hypot(v[0], v[1])))       # flexion (look down) positive
    lo, hi = LIM_DEG["neck_rot"]; yaw_c = float(np.clip(yaw, lo + 5, hi - 5))
    lo, hi = LIM_DEG["neck_flex"]; pit_c = float(np.clip(pitch * .7, lo + 5, hi - 5))    # the eyes take the rest
    pose.local["head"] = rz(math.radians(yaw_c)) @ ry(math.radians(pit_c))
    rep = pose.report.setdefault("neck", {})
    _check(rep, "neck_rot", yaw_c); _check(rep, "neck_flex", pit_c)
    pose.gaze = np.asarray(target, float)


# ---------------------------------------------------------------- the parent's face (body/sim/parent_face.py)
# HER FACE is built in parent_face.py from cited adult female norms (the W1 verifier's third and fourth rounds; the owner's decision
# of 2026-09-24, made for him: fix the world, not the detector): one smooth head sheet with the eye fissures cut between the canthi
# the norms place, the lids that turn about the canthal line, the brows and lips lying on her skin, her collision the hulls of her
# own sheet's pieces. The names the rest of the world reads are re-exported here.
import parent_face as face  # noqa: E402
from parent_face import (IPD, EYE_C, EYE_MID, EYE_R, PUPIL_Z, X_CORNEA, MOUTH_Z, MOUTH_HALF_W, MOUTH_CURVE, IRIS_D,  # noqa: E402,F401
                         PUPIL_D, EN, EX, MRD1, MRD2, head_surface_x, FACE_GEOM_NAMES)

FACE_GEOMS = list(FACE_GEOM_NAMES)


def face_geoms(expr, gaze_head=None, blink=0.0):
    """Local (pos, quat, size or None) of her face's moving parts for an expression: a dict of graded face parameters
    (FACE_NEUTRAL's keys), or the old one-number expression (-1 frown .. +1 smile, through scalar_to_params); the gaze in the head
    frame (None: straight ahead). Size None marks a mesh (the scene composes its compiled offset)"""
    if not isinstance(expr, dict):
        expr = scalar_to_params(expr)
    return face_geoms_graded(expr, gaze_head, blink)


# ---------------------------------------------------------------- the graded face (the parent's feelings, shown)
# Graded face parameters, each an action unit of the Facial Action Coding System (Ekman and Friesen) at an intensity
# in [0, 1] (lid_up in [-1, 1]; tilt in degrees, applied to the head's pose, not here). The parent's feelings set them
# (parent_feel.py); the born expression reading reads only the mouth corners: face_reading().
FACE_NEUTRAL = dict(smile=0.0,      # AU12 lip-corner puller (+ AU25 lips part above .25: the open smile)
                    frown=0.0,      # AU15 lip-corner depressor
                    cheek=0.0,      # AU6 cheek raiser: the lower lids rise (the eyes of a felt smile)
                    brow_in=0.0,    # AU1 inner-brow raiser (surprise, interest, concern's oblique brows)
                    brow_out=0.0,   # AU2 outer-brow raiser
                    brow_low=0.0,   # AU4 brow lowerer (displeasure; with AU1: concern)
                    lid_up=0.0,     # AU5 upper-lid raiser (+: eyes wide) / relaxed lids (-: drowsy)
                    lid_tight=0.0,  # AU7 lid tightener (displeasure)
                    jaw=0.0,        # AU26 jaw drop (surprise, speech)
                    round=0.0,      # AU18/AU22 lips funnelled: the "oh!" of surprise
                    press=0.0,      # AU24 lip presser (concern, a flat held mouth)
                    blink=0.0,
                    tilt=0.0)       # the head's tilt toward its shoulder, degrees (the question face)
FACE_READING_GAIN = 2.0     # the born reading: 2 x (corner pull), so a full smile reads +2 and a full frown -2


def face_params(**kw):
    fp = dict(FACE_NEUTRAL)
    for k, v in kw.items():
        if k not in fp:
            raise KeyError(k)
        fp[k] = float(v)
    return fp


def scalar_to_params(e):
    """The old one-number expression (-1 frown .. +1 smile) as graded parameters (for a model with the extra geoms)."""
    e = float(np.clip(e, -1, 1))
    return face_params(smile=max(e, 0), cheek=max(e, 0), brow_in=.4 * max(e, 0), brow_out=.2 * max(e, 0),
                       frown=max(-e, 0), brow_low=.8 * max(-e, 0), lid_tight=max(-e, 0))


def face_reading(fp):
    """The born expression reading of a graded face (disclosed, SIM_DESIGN A1): the mouth corners' pull only, in
    [-2, 2]. Brows, lids, jaw, rounding and pressing read 0, so surprise, interest, concern, the question face, speech
    and the greeting flash never reach the reward."""
    return float(np.clip(FACE_READING_GAIN * (fp.get("smile", 0.0) - fp.get("frown", 0.0)), -2, 2))


def face_geoms_graded(fp, gaze_head=None, blink=None):
    """Local (pos, quat, size or None) of her face's moving parts for graded face parameters (FACE_NEUTRAL's keys): the lips (upper
    and lower vermilion), the mouth's opening and teeth, the brows, the irises and pupils (turned to the gaze), the lids (upper and
    lower, turned about the lids' hinge), the upper lash lines (parent_face.face_geoms)"""
    f = dict(FACE_NEUTRAL); f.update(fp)
    return face.face_geoms(f, gaze_head, blink)


# the hand's fingers (left hand frame; palm faces -y, thumb on +x): knuckle x positions, lengths, radii
FINGER_X = (.028, .0095, -.0095, -.027)
FINGER_L = ((.046, .040), (.050, .044), (.047, .041), (.037, .032))
FINGER_R = .0085
KNUCKLE_Z = -.094


def hand_geoms(side, curl=.25, thumb=.2, index=None, spread=.06):
    """Local (pos, quat, half-length) of one hand's finger capsules. curl: 0 open .. 1.5 fist (radians per joint, split
    over two joints); thumb: 0 open .. 1 opposed; index: the index finger's own curl (None: same as the others),
    so pointing is index=0 with curl=1.4."""
    sg = side_sign(side)
    out = {}
    for i, x0 in enumerate(FINGER_X):
        c = curl if (i > 0 or index is None) else index
        base = np.array([x0, 0, KNUCKLE_Z])
        splay = (i - 1.5) * spread * (1 - min(c, 1))
        R = ry(-splay) @ rx(-sg * c * .95)                      # flex toward the palm (-y on the left)
        l1, l2 = FINGER_L[i]
        mid = base + R @ np.array([0, 0, -l1])
        R2 = R @ rx(-sg * c * 1.05)
        tip = mid + R2 @ np.array([0, 0, -l2])
        for j, (a, b, RR) in enumerate(((base, mid, R), (mid, tip, R2))):
            out[f"f{i}{j}_{side}"] = ((a + b) / 2, mjquat(RR), np.linalg.norm(b - a) / 2)
    # thumb: from the palm's radial edge; opens along the palm, opposes across its face
    tb = np.array([.036, -sg * .010, -.036])
    t = float(np.clip(thumb, 0, 1))
    d0 = unit([.55, -sg * .15, -.82])
    d1 = unit([.05, -sg * .75, -.66])
    d = unit((1 - t) * d0 + t * d1)
    m = tb + d * .036
    d2 = unit(d + np.array([-.5 * t, -sg * .1, -.2]))
    tip = m + d2 * .031
    for j, (a, b) in enumerate(((tb, m), (m, tip))):
        out[f"t{j}_{side}"] = ((a + b) / 2, mjquat(frame(perp([1, 0, 0], unit(b - a)), -(b - a))), np.linalg.norm(b - a) / 2)
    return out


def mirror(v):
    return np.array([v[0], -v[1], v[2]])


# ---------------------------------------------------------------- the child (numbers shared with the XML builder)
CHILD_MASS_KG = 9.4   # a 12-month-old: about 0.75 m and 9-10 kg (WHO growth standards, 50th percentile, rounded)


CHILD_EYE_GLOW = {"L": np.array([.0925, .022, .082]), "R": np.array([.0925, -.022, .082])}
CHILD_MOUTH = dict(half_w=.017, z=.052, x=.0925, curve=.0075)


def child_face_geoms(expr, eyes=None):
    """The child's screen face (head frame): the glow eyes slide toward where the eyes look (eyes: {'L': (yaw, pitch),
    'R': ...} in radians) and squint a little in a smile; the mouth's corners rise with expr > 0 and fall below 0."""
    e = float(np.clip(expr, -1, 1))
    out = {}
    for sd in ("L", "R"):
        yaw, pit = (0.0, 0.0) if eyes is None else eyes[sd]
        c = CHILD_EYE_GLOW[sd]
        dy = float(np.clip(math.tan(yaw) * .009, -.006, .006))
        dz = float(np.clip(math.tan(pit) * .009, -.006, .006))
        sz = .0118 * (1 - .38 * max(e, 0))
        out[f"child_glow_eye_{sd}"] = (np.array([c[0], c[1] + dy, c[2] + dz + .002 * max(e, 0)]), mjquat(np.eye(3)), (.0015, .0088, sz))
    n = 5
    ys = np.linspace(-CHILD_MOUTH["half_w"], CHILD_MOUTH["half_w"], n)
    pts = [np.array([CHILD_MOUTH["x"], y, CHILD_MOUTH["z"] + e * CHILD_MOUTH["curve"] * (y / CHILD_MOUTH["half_w"]) ** 2
                     - .002 * e * (1 - (y / CHILD_MOUTH["half_w"]) ** 2)]) for y in ys]
    for i in range(n - 1):
        a, b = pts[i], pts[i + 1]
        out[f"child_mouth{i}"] = ((a + b) / 2, mjquat(frame([1, 0, 0], b - a)), (.0032, np.linalg.norm(b - a) / 2, 0))
    return out
