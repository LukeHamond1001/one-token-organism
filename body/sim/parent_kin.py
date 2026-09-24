"""Shared numbers and kinematics for the living room (docs/SIM_DESIGN.md; from the 2026-09-24 prototype; the world's side, never the body).

The parent is kinematic: a skeleton of 16 segments posed here in Python (forward kinematics, two-bone inverse
kinematics for the arms and legs, look-at for the head and eyes) and written into 16 MuJoCo mocap bodies each
physics step. Its fingers, eyes, brows, lids and mouth are geoms whose local poses this file sets (hand shapes,
gaze, expression). Nothing here is the child's body or its control: the child is a physical humanoid in the XML.

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


# ---------------------------------------------------------------- the parent's face: human proportions (geom poses)
# THE FACE OF A REAL WOMAN'S PROPORTIONS (the owner's decision of 2026-09-24, made for him in the W1 verifier's third round: fix the
# world, not the detector). The prototype's face was a cartoon: irises 19.6 mm across, eye whites 27 mm tall, eyes 72 mm apart, no
# eye socket, so at the child's fovea her eye regions read as light as her skin. Every measure below is a published adult female
# norm, cited; the two anchors (where the face sits on her head) are ours. Head frame: x forward, y to her left, z up, metres.
# SOURCES
#   [F]  Farkas LG et al., North American white women 18-25 y, n = 200 (Farkas, Hreczko, Kolar and Munro 1985, Plast Reconstr Surg
#        75:328-338; Farkas, Katic and Forrest 2005, J Craniofac Surg 16:615-646; 2007, Ann Plast Surg 59:692-698), as tabulated
#        in Husein et al. 2010 (J Plast Reconstr Aesthet Surg 63:1825-1831, table 1) and Virdi, Wertheim and Naini 2019 (Maxillofac
#        Plast Reconstr Surg 41:9, table 3)
#   [D]  Dodgson NA 2004, Variation and extrema of human interpupillary distance (Proc SPIE 5291): women's mean 61.7 mm
#   [Ru] Ruefer F, Schroeder A and Erb C 2005, White-to-white corneal diameter: normal values in healthy humans (Cornea 24:259-261):
#        11.71 +- 0.42 mm (the visible iris's diameter)
#   [G]  Gao T, Guo Y et al. 2025 (Quant Imaging Med Surg): young Caucasian women, the eyebrow's lower margin 18.5 mm above the
#        lateral canthus and 14.2 mm above the upper lid margin laterally; its lower length 61.4 mm
#   [M]  McKinney P, Mossie RD and Zukowski ML 1991 (Aesthetic Plast Surg 15:141-147): 50 young women, 25 mm from the mid-pupil to
#        the top of the brow
#   [Y]  Yaremchuk MJ, Atlas of Facial Implants, "Infraorbital rim": in young adults the soft tissue over the supraorbital rim lies
#        10 mm in front of the cornea, the cheek prominence 2 mm in front of it, the soft tissue over the infraorbital rim 3 mm
#        behind it
#   [R]  Russell R, Kramer SS and Jones AL 2017 (Adapt Hum Behav Physiol, table 2): young women (19-31 y, no cosmetics), the
#        luminance contrast (skin - feature) / (skin + feature) in CIE L* of each feature against the skin around it: the eyes
#        (the eye with the band of skin around its lashes) 0.152, the brows 0.126, the lips 0.092 (the photometry, set on the
#        materials in body/sim/make_g1room.py and checked by body/tests/test_sim_eyes.py)
#   recalled, flagged: the eyeball's radius 12 mm (the adult globe about 24 mm across);
#   a room-light pupil of 4 mm; the upper lid over the top 1.5 mm of the iris (so the lid margin 4.3 mm above the pupil and the
#   lower lid at the lower limbus: 10.9 mm apart, [F]'s fissure height); the nasion between the upper lash line and the lid crease
#   (the rhinoplasty literature's normal nasion level): 9 mm above the pupils; the nose's dorsum about 33 deg from the vertical.
IPD = .0617                     # interpupillary distance [D]
EN_EN, EX_EX, EX_EN = .0318, .0878, .0307    # intercanthal, biocular width, eye fissure length [F]
PS_PI = .0109                   # eye fissure height [F]
CANTHAL_TILT_DEG = 4.1          # the lateral canthus above the medial [F]
IRIS_D = .0117                  # the visible iris [Ru]
PUPIL_D = .004                  # room light (recalled; inside the dark iris at the fovea's resolution)
EYE_R = .012                    # the globe (recalled)
IRIS_CAP_R = .010               # the iris, seen through the clear cornea, drawn as the shallow cap a 10 mm sphere makes on the
                                # globe through the limbus (0.37 mm proud of it: ours)
MRD1, MRD2 = .0043, PS_PI - .0043            # the upper lid margin above the pupil, the lower below it (recalled; sum [F])
BROW_LOW, BROW_TOP = .0185, .025             # the brow's lower margin [G] and its top [M] above the pupil
BROW_LEN = .0614                # its lower length [G] (the arc drawn: the medial head above the medial canthus to the tail beyond
                                # the lateral one)
CH_CH, LS_ST, ST_LI, SN_ST = .0502, .0087, .0094, .0201      # mouth width, upper and lower vermilion, upper lip height [F]
N_ST, N_SN, N_GN, TR_GN, N_PN, SN_PN = .0694, .0506, .1118, .1725, .0447, .0197   # midface 2, nasal height, face height,
                                # total face height, nasal length, tip protrusion [F]
AL_AL, ZY_ZY, GO_GO = .0314, .130, .0911     # nasal width, face width, mandible width [F]
BROW_AHEAD, CHEEK_AHEAD, INFRAORB_BEHIND = .010, .002, .003     # the relief about the cornea [Y]
NASION_AHEAD = .010             # the nasion at the brow's soft tissue depth [Y] (the nose's root continuous with the brow)
PUPIL_BELOW_NASION = .009       # (recalled, see above)
DORSUM_DEG = 33.0               # (recalled)
# THE ANCHORS (ours): the nasion's height on her head and the corneal plane's depth, chosen so the chin comes to the head's lower
# front and the eyes sit just in front of its base; nothing else is chosen
FACE_Z_N = .170
X_CORNEA = .100

PUPIL_Z = FACE_Z_N - PUPIL_BELOW_NASION
EYE_C = {"L": np.array([X_CORNEA - EYE_R, IPD / 2, PUPIL_Z]), "R": np.array([X_CORNEA - EYE_R, -IPD / 2, PUPIL_Z])}   # globe centres
EYE_MID = np.array([X_CORNEA - EYE_R, 0.0, PUPIL_Z])
Z_ST = FACE_Z_N - N_ST                                      # the stomion (where the lips meet)
Z_SN = FACE_Z_N - N_SN                                      # the subnasale
Z_GN = FACE_Z_N - N_GN                                      # the chin's lowest point
Z_BROW = PUPIL_Z + (BROW_LOW + BROW_TOP) / 2                # the brow's midline over the pupil
FACE_MOUTH_N = 7                # the upper lip: 6 capsules through 7 points
MOUTH_HALF_W, MOUTH_CURVE = CH_CH / 2, .0075                # half the mouth's width [F]; the corners' lift at a full smile (ours)
MOUTH_Z = Z_ST + LS_ST / 2      # the upper vermilion's midline (the born reading's corners ride it; the face test's mouth point)
LIP_UP_R, LIP_LO_R = LS_ST / 2, ST_LI / 2

# THE SKIN'S MASSES: her head (the base, whose front is set back so the eyes can sit in it) and the relief laid on it, each an
# ellipsoid in the head frame (centre, semi-axes), placed from the landmarks above: the forehead, the brow ridge (its soft tissue
# BROW_AHEAD in front of the cornea over the pupil), the cheeks (their prominence CHEEK_AHEAD in front of it, below and outside
# the eye), the lower face and jaw (its width at the jaw's corners GO_GO), the muzzle under the nose (the lips' base) and the
# chin (its lowest point at Z_GN). head_surface_x reads the frontmost of them.
HEAD_BASE = ((.015, 0.0, .16), (.080, .084, .106))          # the collision shape parent_head (its back, top and sides as before)
_BR_D, _BR_W, _BR_H = .016, .054, .015
FACE_SKIN = {
    "head": HEAD_BASE,
    "forehead": ((.050, 0.0, .187), (.062, .066, .038)),
    "brow_ridge": ((X_CORNEA + BROW_AHEAD - _BR_D * math.sqrt(1 - (IPD / 2 / _BR_W) ** 2), 0.0, Z_BROW), (_BR_D, _BR_W, _BR_H)),
    "midface": ((.058, 0.0, .127), (.045, .056, .034)),
    "cheek_mass_L": ((X_CORNEA + CHEEK_AHEAD - .016, .040, PUPIL_Z - .027), (.016, .026, .026)),
    "cheek_mass_R": ((X_CORNEA + CHEEK_AHEAD - .016, -.040, PUPIL_Z - .027), (.016, .026, .026)),
    "jaw": ((.042, 0.0, .100), (.060, GO_GO / 2 + .012, .050)),
    "muzzle": ((.093, 0.0, Z_ST + .003), (.024, .028, .022)),
    "chin": ((.094, 0.0, Z_GN + .015), (.016, .030, .016)),
}


SKIN_SOFT = .005                # the skin's masses are blended into one smooth surface: along each ray from the head's centre a
                                # soft maximum of the masses' far surfaces, its fillet about 5 mm (ours: her head drawn as one
                                # sheet, no seams between the masses)
HEAD_MESH_STEP_DEG = 2.5        # the head sheet's grid of directions (ours)
HEAD_MESH_AZ, HEAD_MESH_EL = (-105.0, 105.0), (-78.0, 84.0)   # its extent about the head's centre: the face and the sides the hair
                                # does not cover (deg; ours)


def skin_radius(U):
    """the head sheet's distance from the head's centre (HEAD_BASE's) along unit directions U (n x 3): the soft maximum (SKIN_SOFT)
    over the skin's masses (FACE_SKIN) of where each ray leaves the mass (a ray that misses a mass takes a value falling away with
    its miss distance, so the soft maximum leaves it)"""
    U = np.atleast_2d(np.asarray(U, float))
    O = np.asarray(HEAD_BASE[0], float)
    rs = []
    for c, s in FACE_SKIN.values():
        c, s = np.asarray(c, float), np.asarray(s, float)
        p0, d = (O - c) / s, U / s
        a_ = (d * d).sum(1); b_ = d @ p0; c0 = p0 @ p0 - 1
        disc = b_ * b_ - a_ * c0
        t_far = (-b_ + np.sqrt(np.maximum(disc, 0))) / a_
        t_mid = -b_ / a_
        miss = np.sqrt(np.maximum(p0 @ p0 - b_ * b_ / a_, 0)) - 1
        rs.append(np.where(disc >= 0, t_far, t_mid - 3 * miss * float(s.mean())))
    r = np.stack(rs)
    top = r.max(0)
    return top + SKIN_SOFT * np.log(np.exp((r - top) / SKIN_SOFT).sum(0))


def head_mesh():
    """the head sheet: vertices O + r u over the grid of directions (azimuth about her up axis from her front, elevation), as
    (vertices n x 3, triangles m x 3) in the head frame, facing out"""
    O = np.asarray(HEAD_BASE[0], float)
    az = np.radians(np.arange(HEAD_MESH_AZ[0], HEAD_MESH_AZ[1] + 1e-9, HEAD_MESH_STEP_DEG))
    el = np.radians(np.arange(HEAD_MESH_EL[0], HEAD_MESH_EL[1] + 1e-9, HEAD_MESH_STEP_DEG))
    A, E = np.meshgrid(az, el)
    U = np.stack([np.cos(E) * np.cos(A), np.cos(E) * np.sin(A), np.sin(E)], -1).reshape(-1, 3)
    V = O + U * skin_radius(U)[:, None]
    na, ne = len(az), len(el)
    tri = []
    for k in range(ne - 1):
        for j in range(na - 1):
            a, b, c, d = k * na + j, k * na + j + 1, (k + 1) * na + j, (k + 1) * na + j + 1
            tri += [(a, b, d), (a, d, c)]
    return V, np.array(tri, dtype=np.int64)


def _surface_table():
    """head_surface_x's table: for each (y, z) on a 1 mm grid over the face, the x where the head sheet is (bisection along x
    for the point whose distance from the head's centre equals the sheet's radius in its direction)"""
    O = np.asarray(HEAD_BASE[0], float)
    ys = np.arange(-.075, .0751, .001); zs = np.arange(.040, .2451, .001)
    Y, Z = np.meshgrid(ys, zs)
    lo = np.full(Y.shape, O[0]); hi = np.full(Y.shape, O[0] + .2)
    for _ in range(30):
        mid = (lo + hi) / 2
        P = np.stack([mid, Y, Z], -1).reshape(-1, 3) - O
        n = np.linalg.norm(P, axis=1)
        inside = (n < skin_radius(P / n[:, None])).reshape(Y.shape)
        lo = np.where(inside, mid, lo); hi = np.where(inside, hi, mid)
    return ys, zs, (lo + hi) / 2


_SURF = None


def head_surface_x(y, z):
    """the front of her face at (y, z) in the head frame (m): where the head sheet is, from a 1 mm table (bilinear)"""
    global _SURF
    if _SURF is None:
        _SURF = _surface_table()
    ys, zs, X = _SURF
    fy = float(np.clip((y - ys[0]) / .001, 0, len(ys) - 1.001)); fz = float(np.clip((z - zs[0]) / .001, 0, len(zs) - 1.001))
    j, k = int(fy), int(fz); ty, tz = fy - j, fz - k
    return float((X[k, j] * (1 - ty) + X[k, j + 1] * ty) * (1 - tz) + (X[k + 1, j] * (1 - ty) + X[k + 1, j + 1] * ty) * tz)


# THE NOSE (drawn, not in head_surface_x): its root at the nasion NASION_AHEAD in front of the cornea, its dorsum N_PN long at
# DORSUM_DEG from the vertical to the tip, the tip SN_PN in front of the subnasale, the alae AL_AL wide
_N_ROOT = np.array([X_CORNEA + NASION_AHEAD, 0.0, FACE_Z_N])
_N_TIP = _N_ROOT + N_PN * np.array([math.sin(math.radians(DORSUM_DEG)), 0.0, -math.cos(math.radians(DORSUM_DEG))])
NOSE = dict(root=_N_ROOT, tip=_N_TIP, tip_r=.0105, dorsum_r=.0072, subnasale=np.array([_N_TIP[0] - SN_PN, 0.0, Z_SN]),
            ala_y=AL_AL / 2 - .0068, ala_semi=(.0100, .0068, .0078))

# THE LIDS: each a sphere wrapping the globe, set above (the upper) or below (the lower) and behind its centre, so the line where
# it leaves the globe is the lid margin: an arch over the iris for the upper, a shallow curve for the lower, closing toward the
# canthi as real lids do (ours: the spheres' sizes and offsets; the margins' heights are MRD1 and MRD2)
LID_UP_R, LID_UP_BACK = .0150, .0025
LID_LO_R, LID_LO_BACK = .0150, .0025


def lid_centre_z(margin_z, r_lid, back, upper=True):
    """the lid sphere's centre height (above the pupil, m) that puts its margin at margin_z over the globe's front"""
    mz = float(np.clip(margin_z, -EYE_R + 1e-4, EYE_R - 1e-4))
    xm = math.sqrt(EYE_R ** 2 - mz ** 2)
    s = math.sqrt(max(r_lid ** 2 - (xm + back) ** 2, 0.0))
    return mz + s if upper else mz - s


def face_geoms(expr, gaze_head=None, blink=0.0):
    """Local (pos, quat, size or None) of her face's moving geoms (names as the XML's parent_<name>) for an expression: a dict of
    graded face parameters (FACE_NEUTRAL's keys), or the old one-number expression (-1 frown .. +1 smile, through
    scalar_to_params); the gaze in the head frame (None: straight ahead)"""
    if not isinstance(expr, dict):
        expr = scalar_to_params(expr)
    return face_geoms_graded(expr, gaze_head, blink)


def _seg(a, b, r):
    return (a + b) / 2, mjquat(frame([1, 0, 0], b - a)), (r, float(np.linalg.norm(b - a)) / 2, 0)


LIP_DEPTH = .0028               # a lip's half-depth: the vermilion is drawn as a flattened ellipsoid, its height the source's (ours)


def _lip(a, b, h):
    """a lip segment from a to b, an ellipsoid: half-depth LIP_DEPTH forward, its half-length along the segment (plus the half
    height, so neighbours overlap), half-height h"""
    u = unit(b - a)
    x = perp(np.array([1.0, 0, 0]), u)
    R = np.column_stack([x, u, np.cross(x, u)])
    return (a + b) / 2, mjquat(R), (LIP_DEPTH, float(np.linalg.norm(b - a)) / 2 + h * 1.1, h)


def _axis_x(v):
    """a rotation whose local x is the unit vector v"""
    x = unit(v)
    z = perp(np.array([0.0, 0.0, 1.0]), x)
    return np.column_stack([x, np.cross(z, x), z])


def lid_margin_at(dy, cz, r_lid, back, upper=True):
    """the lid margin's height (about the pupil, m) at dy across the globe, in closed form: where the lid's sphere (centre (-back,
    dy-independent, cz) about the globe's centre) leaves the globe's front; None where it does not meet it"""
    R2 = EYE_R ** 2 - dy * dy
    if R2 <= 0:
        return None
    A = r_lid ** 2 - EYE_R ** 2 - back ** 2 - cz ** 2
    a, b, c = 4 * (cz * cz + back * back), 4 * A * cz, A * A - 4 * back * back * R2
    disc = b * b - 4 * a * c
    if disc < 0:
        return None
    roots = [(-b + s * math.sqrt(disc)) / (2 * a) for s in (1, -1)]
    roots = [z for z in roots if A + 2 * cz * z >= -1e-12 and z * z <= R2]
    if not roots:
        return None
    return min(roots) if upper else max(roots)


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


def _mouth_upper(y, W, net, smile, frown, lift):
    u = min((y / W) ** 2, 1.0)
    return MOUTH_Z + net * MOUTH_CURVE * u - (.004 * smile - .002 * frown) * (1 - u) + lift * (1 - u)


def face_geoms_graded(fp, gaze_head=None, blink=None):
    """Local (pos, quat, size or None) of her face's moving geoms for graded face parameters (FACE_NEUTRAL's keys): the lips, the
    mouth's opening and teeth, the brows (two arcs each), the eyes (the globe, the iris as the cornea's cap, the pupil, the
    corneal highlight), the lids (the upper with its lash line, the lower), the cheeks' blush. The static relief (the forehead, the
    brow ridge, the cheeks' mass, the nose, the jaw, the muzzle, the chin) is the maker's (face_static_xml)."""
    f = dict(FACE_NEUTRAL); f.update(fp)
    c01 = lambda k: float(np.clip(f[k], 0, 1))
    sm, fr, ck = c01("smile"), c01("frown"), c01("cheek")
    bi, bo, bl = c01("brow_in"), c01("brow_out"), c01("brow_low")
    lu, lt = float(np.clip(f["lid_up"], -1, 1)), c01("lid_tight")
    jw, rd, pr = c01("jaw"), c01("round"), c01("press")
    bk = c01("blink") if blink is None else float(np.clip(max(blink, f["blink"]), 0, 1))
    net = sm - fr
    out = {}
    # THE MOUTH. The upper lip: 7 points, corners up for a smile and down for a frown (the only part the born reading reads); its
    # centre lifts as the mouth opens; wider in a smile, narrower in an "oh". Each lip tapers toward the corners.
    W = MOUTH_HALF_W * (1 + .12 * sm + .05 * pr - .30 * rd)
    sm_open = max(0.0, sm - .25) / .75
    lift = .0025 * jw + .004 * rd + .0012 * sm_open
    up = []
    for y in np.linspace(-W, W, FACE_MOUTH_N):
        z = _mouth_upper(y, W, net, sm, fr, lift)
        up.append(np.array([head_surface_x(y, z) + .0015, y, z]))
    for i in range(FACE_MOUTH_N - 1):
        u = abs((up[i][1] + up[i + 1][1]) / 2) / W
        out[f"mouth{i}"] = _lip(up[i], up[i + 1], LIP_UP_R * (1 - .4 * pr) * (1 - .5 * u * u))
    h = .0105 * sm_open + .017 * jw + .016 * rd                       # the opening between the lips
    lo = []
    for y in np.linspace(-W * .96, W * .96, 5):
        u = min((y / W) ** 2, 1.0)
        z = _mouth_upper(y, W, net, sm, fr, lift) - (LIP_UP_R + LIP_LO_R) * (1 - .7 * u) - h * (1 - u) ** (.75 if rd > .2 else 1.0)
        lo.append(np.array([head_surface_x(y, z) + .0010, y, z]))
    for i in range(4):
        u = abs((lo[i][1] + lo[i + 1][1]) / 2) / W
        out[f"lip_lo{i}"] = _lip(lo[i], lo[i + 1], LIP_LO_R * (1 - .4 * pr) * (1 - .5 * u * u))
    zu0 = _mouth_upper(0, W, net, sm, fr, lift)
    zl0 = zu0 - (LIP_UP_R + LIP_LO_R) - h
    oc = np.array([head_surface_x(0, (zu0 + zl0) / 2) - .0012, 0, (zu0 + zl0) / 2])
    shown = h > .0008
    out["mouth_open"] = (oc if shown else oc - [.02, 0, 0], mjquat(np.eye(3)),
                         (.004, max(.001, W * (.80 + .15 * rd)), max(.001, h / 2 + .0012)))
    teeth_k = sm_open * (1 - rd) + .5 * max(0.0, jw - .35) * (1 - rd)
    out["teeth"] = (np.array([oc[0] + .001, 0, zu0 - LIP_UP_R - .0012 * teeth_k]) if teeth_k > .02 else oc - [.02, 0, 0],
                    mjquat(np.eye(3)), (.003, .001 + .016 * min(teeth_k, 1) * W / MOUTH_HALF_W, .001 + .0024 * min(teeth_k, 1)))
    for sd, sg in (("L", 1), ("R", -1)):
        # THE BROWS: two arcs over each eye on the brow ridge, from the head (over the medial canthus) through the peak (over the
        # lateral limbus) to the tail (past the lateral canthus); the head rises with AU1 and falls and draws in with AU4, the
        # tail rises with AU2
        pts = []
        for y0, z0, dz in ((EN_EN / 2 - .001 + .004 * bl, Z_BROW - .0028, .011 * bi - .0065 * bl + .002 * sm),
                           (IPD / 2 + .006, Z_BROW + .0010, .5 * (.011 * bi + .009 * bo) - .004 * bl + .0015 * sm),
                           (EX_EX / 2 + .018, Z_BROW - .0035, .009 * bo - .002 * bl + .001 * sm)):
            y, z = sg * y0, z0 + dz
            pts.append(np.array([head_surface_x(y, z) + .0012, y, z]))
        out[f"brow_{sd}"] = _seg(pts[0], pts[1], (BROW_TOP - BROW_LOW) / 2)          # over the pupil: BROW_LOW to BROW_TOP
        out[f"brow2_{sd}"] = _seg(pts[1], pts[2], (BROW_TOP - BROW_LOW) / 2 - .0008)
        # THE EYE: the globe; the iris drawn as the cap the cornea's sphere makes on it (its rim the limbus, IRIS_D across), turned
        # to the gaze (within 35 deg across and 25 up or down); the pupil on the cap's apex; the corneal highlight above it
        ec = EYE_C[sd]
        g = np.array([1.0, 0, 0]) if gaze_head is None else unit(gaze_head[sd] if isinstance(gaze_head, dict) else gaze_head)
        yaw = float(np.clip(math.atan2(g[1], max(g[0], 1e-6)), -math.radians(35), math.radians(35)))
        pit = float(np.clip(math.atan2(g[2], math.hypot(g[0], g[1])), -math.radians(25), math.radians(25)))
        g = np.array([math.cos(pit) * math.cos(yaw), math.cos(pit) * math.sin(yaw), math.sin(pit)])
        rl = IRIS_D / 2
        xi = math.sqrt(EYE_R ** 2 - rl ** 2)
        di = xi - math.sqrt(IRIS_CAP_R ** 2 - rl ** 2)
        out[f"sclera_{sd}"] = (ec.copy(), mjquat(np.eye(3)), (EYE_R, EYE_R, EYE_R))
        out[f"iris_{sd}"] = (ec + di * g, mjquat(np.eye(3)), (IRIS_CAP_R, IRIS_CAP_R, IRIS_CAP_R))
        apex = ec + (di + IRIS_CAP_R) * g
        Rg = _axis_x(g)
        out[f"pupil_{sd}"] = (apex + .00012 * g, mjquat(Rg), (.00025, PUPIL_D / 2, PUPIL_D / 2))
        out[f"glint_{sd}"] = (apex - .0004 * g + Rg @ np.array([0, sg * .0011, .0017]), mjquat(np.eye(3)), (.0006, .0006, .0006))
        # THE LIDS: the upper margin MRD1 over the pupil, raised with AU5, lowered with AU7, drowsiness and the blink (to the lower
        # margin); the lower margin MRD2 under it, raised with AU6 (the cheeks) and AU7
        lo_m = -MRD2 + .0045 * ck + .002 * lt
        up_m = MRD1 + .002 * max(lu, 0) - .005 * lt - .007 * max(-lu, 0)
        up_m = max(lo_m + .0003, up_m - (up_m - lo_m - .0003) * bk)
        shut = float(np.clip((MRD1 - up_m) / (MRD1 + MRD2), 0, 1))     # how far the upper lid has come down: it slides forward
        back_up = LID_UP_BACK * (1 - shut)                               # over the cornea as it closes
        cz_up = lid_centre_z(up_m, LID_UP_R, back_up, True)
        cz_lo = lid_centre_z(lo_m, LID_LO_R, LID_LO_BACK, False)
        out[f"lid_up_{sd}"] = (ec + [-back_up, 0, cz_up], mjquat(np.eye(3)), (LID_UP_R, LID_UP_R, LID_UP_R))
        out[f"lid_lo_{sd}"] = (ec + [-LID_LO_BACK, 0, cz_lo], mjquat(np.eye(3)), (LID_LO_R, LID_LO_R, LID_LO_R))
        # the upper lash line along the upper margin, from the medial third to the lateral canthus
        lash = []
        for dy in (-sg * .0095, sg * .0005, sg * .0112):
            z = lid_margin_at(dy, cz_up, LID_UP_R, back_up, True)
            z = up_m if z is None else z
            x = math.sqrt(max(EYE_R ** 2 - dy * dy - z * z, 0.0))
            lash.append(ec + np.array([x + .0004, dy, z + .0002]))
        out[f"lash_{sd}"] = _seg(lash[0], lash[1], LASH_R)
        out[f"lash2_{sd}"] = _seg(lash[1], lash[2], LASH_R)
        # the cheeks' blush rises and rounds with AU6
        cy, cz = sg * (.047 + .002 * ck), PUPIL_Z - .026 + .006 * ck
        out[f"cheek_{sd}"] = (np.array([head_surface_x(cy, cz) - .0002, cy, cz]), mjquat(rz(math.radians(sg * 32))),
                              (.0015, .011 * (1 + .1 * ck), .007 * (1 + .15 * ck)))
    return out


LASH_R = .0008                  # the upper lash line's half-thickness (ours: the lashes' dark band at the margin)


def face_mesh_asset_xml():
    """the head sheet as an inline mesh asset (for the maker's <asset>); a shell's inertia (an open sheet has no volume)"""
    V, T = head_mesh()
    return (f'<mesh name="parent_face" inertia="shell" vertex="{" ".join(f"{x:.5g}" for x in V.reshape(-1))}" '
            f'face="{" ".join(str(int(i)) for i in T.reshape(-1))}"/>')


def face_static_xml(skin, decor_attrs):
    """her face's static relief for the maker's head segment: the face sheet (the skin's masses, FACE_SKIN, blended into one
    surface: face_mesh_asset_xml) and the nose (its dorsum, tip, alae and columella)"""
    f = lambda *v: " ".join(f"{x:.5g}" for x in v)
    g = [f'<geom name="parent_face" type="mesh" mesh="parent_face" material="{skin}" {decor_attrs}/>']
    N = NOSE
    tip_back = N["tip"] - unit(N["tip"] - N["root"]) * N["tip_r"] * .6 - [N["dorsum_r"], 0, 0]
    root_axis = N["root"] - [N["dorsum_r"], 0, 0]                        # the dorsum's front at the nasion
    g.append(f'<geom name="parent_nose_dorsum" type="capsule" fromto="{f(*root_axis, *tip_back)}" size="{N["dorsum_r"]:.5g}" '
             f'material="{skin}" {decor_attrs}/>')
    g.append(f'<geom name="parent_nose" type="sphere" pos="{f(*(N["tip"] - [N["tip_r"], 0, 0]))}" size="{N["tip_r"]:.5g}" '
             f'material="{skin}" {decor_attrs}/>')
    for sd, sg in (("L", 1), ("R", -1)):
        al = np.array([N["subnasale"][0] + .0025, sg * N["ala_y"], Z_SN + .0035])
        g.append(f'<geom name="parent_nose_ala_{sd}" type="ellipsoid" pos="{f(*al)}" size="{f(*N["ala_semi"])}" material="{skin}" '
                 f'{decor_attrs}/>')
    g.append(f'<geom name="parent_nose_columella" type="capsule" fromto="{f(*(N["tip"] - [N["tip_r"], 0, .004]), *(N["subnasale"] + [.002, 0, .001]))}" '
             f'size=".0035" material="{skin}" {decor_attrs}/>')
    return g


# THE SOCKETS' SHADE OF THE ROOM'S INDIRECT LIGHT (the owner's decision: her eye sockets in shade, as real faces are at low
# spatial frequency). MuJoCo's renderer draws a light's ambient term (the room's indirect light, make_g1room.ROOM_INDIRECT) with no
# occlusion, so a recessed eye under its brow ridge and beside its nose would be lit as fully as the forehead. The occlusion is
# computed here from her own geometry, never set: for sample points on the eye (the globe's front in the fissure) and on each lid,
# the share of a cosine-weighted hemisphere of directions that leaves her head without meeting it (the head sheet, the nose, the
# lids, the globe): the ambient occlusion a renderer with indirect light would give under a uniform sky, baked into the albedo of
# the eye's and the lids' materials (the maker's), as renderers without indirect light bake it. Only the socket's own parts carry
# it; the sheet around them is one material and is left at 1 (its socket skin is small beside the lids).
AO_RAYS, AO_STEP, AO_REACH = 256, .0006, .08          # rays per sample point, the march's step and reach (m; ours: the instrument)


def _nose_parts():
    N = NOSE
    tip_back = N["tip"] - unit(N["tip"] - N["root"]) * N["tip_r"] * .6 - [N["dorsum_r"], 0, 0]
    root_axis = N["root"] - [N["dorsum_r"], 0, 0]
    caps = [(root_axis, tip_back, N["dorsum_r"]),
            (N["tip"] - [N["tip_r"], 0, .004], N["subnasale"] + [.002, 0, .001], .0035)]
    ells = [(N["tip"] - [N["tip_r"], 0, 0], (N["tip_r"],) * 3)]
    for sg in (1, -1):
        ells.append((np.array([N["subnasale"][0] + .0025, sg * N["ala_y"], Z_SN + .0035]), N["ala_semi"]))
    return caps, ells


def _occluded(Q, lids):
    """whether points Q (k x 3, the head frame) are inside her head: the sheet, the nose, the lid spheres, the globes"""
    O = np.asarray(HEAD_BASE[0], float)
    D = Q - O
    n = np.linalg.norm(D, axis=1)
    inside = n < skin_radius(D / np.maximum(n, 1e-9)[:, None])
    caps, ells = _nose_parts()
    for a, b, r in caps:
        ab = b - a
        t = np.clip(((Q - a) @ ab) / (ab @ ab), 0, 1)
        inside |= np.linalg.norm(Q - (a + t[:, None] * ab), axis=1) < r
    for c, sz in ells + [(c, (r,) * 3) for c, r in lids]:
        inside |= (((Q - c) / np.asarray(sz)) ** 2).sum(1) < 1
    return inside


SOCKET_MATERIALS = {"eye": ("sclera", "iris", "pupil", "glint", "lash"), "lid_up": ("lid_up",), "lid_lo": ("lid_lo",)}


def socket_occlusion(seed=0):
    """the ambient light reaching the left eye's parts, as a share of an unoccluded surface's: {"eye", "lid_up", "lid_lo"}, each
    the mean over sample points on the part's visible surface (neutral face) of the cosine-weighted share of directions leaving
    her head (deterministic: its own seeded directions)"""
    g = face_geoms_graded(FACE_NEUTRAL)
    ec = EYE_C["L"]
    lids = [(g["lid_up_L"][0], LID_UP_R), (g["lid_lo_L"][0], LID_LO_R)]
    rng = np.random.default_rng(seed)
    u1, u2 = rng.random(AO_RAYS), rng.random(AO_RAYS)
    local = np.stack([np.sqrt(u1), np.sqrt(1 - u1) * np.cos(2 * np.pi * u2), np.sqrt(1 - u1) * np.sin(2 * np.pi * u2)], 1)  # cos-weighted, +x
    steps = np.arange(AO_STEP, AO_REACH, AO_STEP)
    out = {}
    cz_up, cz_lo = g["lid_up_L"][0][2] - ec[2], g["lid_lo_L"][0][2] - ec[2]
    parts = {"eye": [(ec, EYE_R, dy, z) for dy in (-.007, -.0035, 0, .0035, .007) for z in (-.004, 0, .003)],
             "lid_up": [(g["lid_up_L"][0], LID_UP_R, dy, z) for dy in (-.006, 0, .006) for z in (MRD1 + .002, MRD1 + .004)],
             "lid_lo": [(g["lid_lo_L"][0], LID_LO_R, dy, z) for dy in (-.006, 0, .006) for z in (-MRD2 - .0015, -MRD2 - .003)]}
    for part, pts in parts.items():
        vis = []
        for c, r, dy, z in pts:
            zz = ec[2] + z - c[2]
            x = math.sqrt(max(r * r - dy * dy - zz * zz, 0))
            P = c + np.array([x, dy, zz])
            nrm = unit(P - c)
            R = _axis_x(nrm)
            W = local @ R.T
            Q = (P + .0004 * nrm)[None, None, :] + W[:, None, :] * steps[None, :, None]
            occ = lids if part == "eye" else [lids[1 - ["lid_up", "lid_lo"].index(part)], (ec, EYE_R)]
            hit = _occluded(Q.reshape(-1, 3), occ).reshape(len(W), len(steps)).any(1)
            vis.append(1 - hit.mean())
        out[part] = float(np.mean(vis))
    return out


FACE_MATERIAL = dict(mouth="lips", lip_lo="lips", mouth_open="mouth_in", teeth="teeth", brow="brow", brow2="brow", sclera="sclera",
                     iris="iris", pupil="pupil", glint="glint", lid_up="lid_up", lid_lo="lid_lo", lash="lash", lash2="lash", cheek="blush")


def _material_of(name):
    base = name.rstrip("0123456789")
    base = base[:-2] if base.endswith(("_L", "_R")) else base
    return FACE_MATERIAL[base]


def face_moving_xml(decor_attrs):
    """her face's moving geoms for the maker's head segment, each at its neutral pose (the scene's set_parent moves them)"""
    f = lambda *v: " ".join(f"{x:.6g}" for x in v)
    g = []
    for n, (p, q, sz) in face_geoms_graded(FACE_NEUTRAL).items():
        kind = "capsule" if n.startswith(("brow", "lash")) else "ellipsoid"
        size = f(sz[0], sz[1]) if kind == "capsule" else f(*sz)
        g.append(f'<geom name="parent_{n}" type="{kind}" pos="{f(*p)}" quat="{f(*q)}" size="{size}" material="{_material_of(n)}" '
                 f'{decor_attrs}/>')
    return g


FACE_GEOMS = list(face_geoms_graded(FACE_NEUTRAL).keys())


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
