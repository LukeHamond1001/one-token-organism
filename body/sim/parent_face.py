"""THE PARENT'S FACE: a woman's head of human proportions, built from cited norms, with the photometry real faces have at low
spatial frequency (docs/SIM_DESIGN.md 4.1 and 4.3; the owner's decision of 2026-09-24, made for him in the W1 verifier's third
and fourth rounds: FIX THE WORLD, NOT THE DETECTOR). The world's side only: nothing here is the child's body.

WHY THIS FILE (the W1 verifier's fourth round). The third round's face declared the norms but did not build them (its eye opening
22.0 mm wide where the norm is 30.7, its inner eye corners 41.6 mm apart where the norm is 31.8, its head 168 mm wide), looked
lumpy (cheek bulges, rod brows off the forehead, a nose of a cylinder and three balls, segmented lips, faceted shading, bright
specks) and stood 2-5 cm in front of her collision shape. Here the landmarks are the geometry: the eye opening IS the cut between
the canthi the norms place, and eyes 12 (body/tests/test_sim_eyes.py) measures every norm on the drawn geometry by rays, never on
the declared constants.

HOW SHE IS BUILT
  THE HEAD SHEET   one smooth surface: a thin-plate spline r(theta, z) about a vertical axis through her head (the smoothest surface
                   through its points: no lumps), fitted through the landmarks below (her face) and through her cranium, jaw and
                   neck (her sides and back); the nose a smooth union of a tapered dorsum, a tip, the alae and the columella, the
                   nostrils cut; meshed by surface nets on a 1.6 mm grid and projected onto the surface; its normals the surface's
                   own gradient (smooth shading, no facets). One mesh (assets/parent/face.msh), its texture her skin's shade of
                   the room's indirect light, computed from her own geometry (the ambient occlusion MuJoCo does not draw:
                   textures/parent_face_ao.png).
  THE EYE FISSURE  cut into the sheet: a ring patch from the sheet's hole around each eye in to the fissure's outline, which runs
                   from the medial canthus (en) to the lateral (ex) through the lid margins, and curls back at the rim into the
                   socket. The outline is the norms': en-en 31.8, ex-ex 87.8, ex-en 30.7, the fissure 10.9 tall, the canthal
                   tilt 4.1 deg [F]; the upper lid's margin 4.3 mm over the pupil [EW]. Behind it the globe (24 mm), the iris
                   (11.7 mm [Ru]) as a finely drawn cap, the pupil, the caruncle at the medial corner and the conjunctiva.
  THE LIDS         two thin shells per eye that turn about a hinge through the globe's centre, behind the rim: the upper comes
                   down for the blink, AU7 and drowsiness and goes up 2 mm for AU5 (the rim is cut at AU5's full rise, so at rest
                   the upper lid shows a 2 mm band below it, as a lid below its fold does); the lower rises for AU6 and AU7. Each
                   lies just behind her skin wherever it ever passes under it (so it never shows through), and the blink carries
                   the upper lid until its margin is past the lower's along the whole fissure (a rigid lid cannot bend to the
                   lower's shape: it goes on behind the lower lid). The upper lash line rides the upper lid's margin and stops on
                   the lower lid in a blink. Measured in nine lid states by tools/sim_face_measure.py lids (eyes 17).
  THE BROWS        thin shapes lying on the skin (never standing off it), their lower margin at Gao's heights over the lid
                   margin and the canthi, their top 25 mm over the pupil [M]; AU1, AU2 and AU4 move them on the skin.
  THE LIPS         the vermilion as overlapping thin shapes on the skin of her mouth, the mouth 50.2 wide, the upper vermilion
                   8.7 and the lower 9.4 tall [F]; the smile, the frown, the jaw, the "oh" and the press move them (the born
                   reading still reads the corners only: parent_kin.face_reading).
  THE COLLISION    her head's collision is the convex hulls of her own sheet's pieces (assets/parent/collide_*.msh) and her
                   cranium: a hand stops at her drawn face, within the few millimetres a piece's hull bridges (body/tests/
                   test_sim_eyes.py eyes 16 measures it).

SOURCES (looked up 2026-09-24 unless marked)
  [F]  Farkas LG et al., North American white women 18-25 y, n = 200, as tabulated in Husein OF, Sepehr A, Garg R et al. 2010,
       J Plast Reconstr Aesthet Surg 63:1825-1831, table 1, and Virdi SS, Wertheim D, Naini FB 2019, Maxillofac Plast Reconstr
       Surg 41:9, table 3: tr-n 63.0, tr-g 52.7, n-gn 111.8, n-st 69.4, sn-gn 64.3, zy-zy 130.0, go-go 91.1, en-en 31.8, ex-en
       30.7, ps-pi 10.9, canthal tilt 4.1 deg, n-sn 50.6, n-pn 44.7, sn-pn 19.7, al-al 31.4, ch-ch 50.2, sn-st 20.1, ls-st 8.7,
       li-st 9.4, nasofrontal angle 134.3 deg, nasofacial angle 29.9 deg, ear length 59.6, ear incline 17.5 deg (mm)
  [D]  Dodgson NA 2004, Proc SPIE 5291: women's interpupillary distance 61.7 mm
  [Ru] Ruefer F, Schroeder A, Erb C 2005, Cornea 24:259-261: the visible iris (white to white) 11.71 mm
  [G]  Gao T, Guo Y et al. 2025, Quant Imaging Med Surg ("Racial and sexual differences of eyebrow and eyelid morphology"),
       Caucasian women: the brow's lower margin 17.52 mm above en, 11.64 over the lid margin at the medial limbus, 10.93 at the
       pupil, 12.04 at the lateral limbus, 18.53 above ex
  [M]  McKinney P, Mossie RD, Zukowski ML 1991, Aesthetic Plast Surg 15:141-147: 25 mm from the mid-pupil to the brow's top
  [Y]  Yaremchuk MJ, Atlas of Facial Implants, "Infraorbital rim": the soft tissue over the supraorbital rim 10 mm in front of
       the cornea, the cheek's prominence 2 mm in front, the soft tissue over the infraorbital rim 3 mm behind
  [P]  Park B-KD, Corner BD, Hudson JA, Whitestone J, Mullenger CR, Reed MP, A three-dimensional parametric adult head model
       (humanshape.org/head), table 1, women n = 100: head breadth 146.6 mm, head length 186.5 mm
  [EW] EyeWiki, "Margin to Reflex Distance 1, 2, 3": the upper lid rests 1-2 mm below the superior limbus (so 4.3 mm over the
       pupil with the 11.7 mm iris; the lower margin at 10.9 - 4.3 = 6.6 below, [F]'s fissure)
  recalled, flagged: the globe 24 mm across; a room-light pupil 4 mm
  ours (the anchors and the relief between the landmarks): the corneal plane X_CORNEA and the pupils' height PUPIL_Z in her head's
  frame; the medial canthus 3 mm behind the corneal apex (EN_BEHIND; the lateral's depth then follows from [F]'s three widths);
  the nasion 9 mm over the pupils and 10 mm in front of the cornea; the spline's points between the landmarks; the brow's head
  over the ala and its tail on the line from the ala through ex (the clinical convention), its taper; the lips' fullness; the
  lids' 2 mm band under the rim and their thickness; the lids' hinge (HINGE_BACK_DEG, HINGE_TILT_DEG: the one, of a grid, that
  closes a blink with the least turn), their clearance behind the skin (LID_CLEAR), the share of the fissure a blink closes
  (BLINK_INNER), the lash line's reach (LASH_SPAN) and thickness; the pupil's disc seated on the iris (PUPIL_SEAT); the vertex
  280 mm up (under her hair)."""
import hashlib
import math
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ASSETS = HERE / "assets" / "parent"
MM = 1e-3

# ---------------------------------------------------------------- the norms (m) [sources above]
IPD = .0617                                         # [D]
IRIS_D = .01171                                     # [Ru]
EN_EN, EX_EX, EX_EN, PS_PI = .0318, .0878, .0307, .0109   # [F]
CANTHAL_TILT_DEG = 4.1                              # [F]
MRD1 = .0043                                        # [EW]: the lid 1.5 mm below the superior limbus (11.7 / 2 - 1.5)
MRD2 = PS_PI - MRD1
TR_N, TR_G, N_GN, N_ST, SN_GN = .0630, .0527, .1118, .0694, .0643     # [F]
ZY_ZY, GO_GO = .1300, .0911                         # [F]
N_SN, N_PN, SN_PN, AL_AL = .0506, .0447, .0197, .0314                 # [F]
NASOFACIAL_DEG, NASOFRONTAL_DEG = 29.9, 134.3        # [F]
CH_CH, SN_ST, LS_ST, ST_LI = .0502, .0201, .0087, .0094               # [F]
EAR_LEN, EAR_INCLINE_DEG = .0596, 17.5              # [F]
BROW_OVER_EN, BROW_OVER_EX = .01752, .01853         # [G]: the brow's lower margin above the canthi
BROW_OVER_LID = ((-.00585, .01164), (0.0, .01093), (.00585, .01204))  # [G]: over the lid margin at the medial limbus, pupil, lateral limbus
BROW_TOP_OVER_PUPIL = .025                          # [M]
BROW_AHEAD, CHEEK_AHEAD, INFRAORB_BEHIND = .010, .002, .003           # [Y]
HEAD_BREADTH, HEAD_LENGTH = .1466, .1865            # [P]
EYE_R = .012                                        # recalled
PUPIL_D = .004                                      # recalled

# ---------------------------------------------------------------- the anchors and the landmarks (ours where not cited)
X_CORNEA, PUPIL_Z = .095, .161
EN_BEHIND = .003
NASION_AHEAD, PUPIL_BELOW_NASION = .010, .009
EYE_C = {"L": np.array([X_CORNEA - EYE_R, IPD / 2, PUPIL_Z]), "R": np.array([X_CORNEA - EYE_R, -IPD / 2, PUPIL_Z])}
EYE_MID = np.array([X_CORNEA - EYE_R, 0.0, PUPIL_Z])
_dy = (EX_EX - EN_EN) / 2
_dz = _dy * math.tan(math.radians(CANTHAL_TILT_DEG))
_dx = math.sqrt(EX_EN ** 2 - _dy ** 2 - _dz ** 2)      # the lateral canthus lies this far behind the medial ([F]'s three widths)
EN = np.array([X_CORNEA - EN_BEHIND, EN_EN / 2, PUPIL_Z - _dz * ((IPD / 2 - EN_EN / 2) / _dy)])   # the left eye's (y > 0)
EX = EN + np.array([-_dx, _dy, _dz])
N_PT = np.array([X_CORNEA + NASION_AHEAD, 0.0, PUPIL_Z + PUPIL_BELOW_NASION])            # the nasion
PN = N_PT + N_PN * np.array([math.sin(math.radians(NASOFACIAL_DEG)), 0, -math.cos(math.radians(NASOFACIAL_DEG))])   # the tip


def _sn():
    """the subnasale: [F]'s triangle n-sn 50.6, sn-pn 19.7, n-pn 44.7 (the root below the nasion)"""
    a, b = N_SN, SN_PN
    p = PN - N_PT
    # solve |s| = a, |p - s| = b in the sagittal plane: s = (sx, -sz)
    c = (a * a + p @ p - b * b) / 2
    # s . p = c with s = a (sin t, -cos t)
    px, pz = p[0], p[2]
    R_, ph = math.hypot(px, pz), math.atan2(-pz, px)
    t = math.asin(c / (a * R_)) - ph
    return N_PT + a * np.array([math.sin(t), 0, -math.cos(t)])


SN = _sn()
ST = np.array([SN[0] + .001, 0.0, N_PT[2] - .0696])      # the stomion: n-st 70.0 and sn-st 19.4 (the least squares of [F]'s 69.4 and
                                                         # 20.1, which a protruding subnasale cannot both meet; ours: 1 mm ahead of sn)
MOUTH_HALF_W = CH_CH / 2
MOUTH_Z = ST[2] + LS_ST / 2                              # the upper vermilion's midline (the born reading's corners ride it)
Z_GN = N_PT[2] - N_GN
TR_Z = N_PT[2] + TR_N

# ---------------------------------------------------------------- the head sheet: a thin-plate spline r(theta, z) about an axis
AX_X = .015                     # the axis's x (m; ours)
TPS_L = 80.0                    # mm per radian in the spline's metric (ours)
TPS_SMOOTH = 30.0               # the spline's smoothing (ours: residuals at the points median 0.1 mm)
# her face's points (x, y, z) mm, y >= 0 (mirrored); the nose is added apart. The cited ones: the brow over the pupil (105: [Y]),
# the closed lid 2 mm in front of the cornea (97), the infraorbital rim (92-93: [Y]), the cheek's prominence (97: [Y]), the
# canthi (en, ex: [F] and EN_BEHIND), the stomion and the chin's gnathion ([F]), the mouth's corners (y 25.1: [F]); the rest ours
FACE_POINTS = [
    (96, 0, 233), (102.5, 0, 215), (106.5, 0, 197), (107.8, 0, 185), (107.4, 0, 180), (107.2, 0, 174), (104.2, 0, 170),
    (109.0, 0, 163), (106.5, 0, 158), (106.5, 0, 146), (106.5, 0, 136), (104, 0, 132), (110.5, 0, 120), (112.2, 0, 109), (110.6, 0, 100.4), (107.6, 0, 91),
    (102.8, 0, 82.6), (97.6, 0, 72), (94.4, 0, 63), (89.5, 0, 57.8), (80.5, 0, 54), (67, 0, 50), (56, 0, 45),
    (94, 25, 233), (85, 45, 233), (66, 58, 233), (101, 20, 215), (95, 40, 215), (82, 55, 215), (62, 65, 215),
    (105.5, 18, 197), (100, 38, 197), (88, 54, 197), (66, 67, 197),
    (106.8, 15, 182), (105, 31, 181), (100, 45, 181), (88, 56, 180), (66, 67, 180),
    (103.5, 10, 172), (99.5, 18, 172), (99, 31, 172.5), (95.5, 42, 172), (85, 53, 171), (64, 67, 170),
    (100, 8, 161), (92, 15.9, 160), (95.2, 22, 161), (97.0, 30.85, 161), (94.6, 38, 161.5), (87.0, 42.5, 161.8), (79.6, 43.9, 162),
    (78, 48, 162), (73, 56, 162), (52, 67, 161),
    (99.5, 12, 150), (95, 20, 150.5), (94, 31, 150), (90, 42, 151), (79, 52, 152), (58, 64, 153),
    (99, 14, 145), (94.5, 24, 145.5), (93, 31, 146), (90.5, 42, 146.5), (81, 53, 147), (58, 63.5, 148),
    (103, 14, 136), (99, 26, 136), (97, 38, 137), (91, 48, 138), (78, 56, 140), (55, 63, 142),
    (104.5, 14, 126), (101, 24, 126), (96, 36, 127), (90, 46, 128), (79, 55, 130), (54, 63, 133),
    (110.5, 8, 117), (107, 16, 116), (101.5, 26, 116), (95, 38, 117), (84, 48, 118), (56, 60, 121),
    (111, 8, 109), (107.5, 16, 108.5), (102, 24, 107), (95, 34, 107), (84, 45, 108), (54, 57, 110),
    (109.5, 8, 100.4), (105, 16, 100.4), (99.5, 25.1, 100.4), (93, 32, 100), (83, 42, 100), (52, 53, 100),
    (106.5, 8, 91), (102, 16, 91), (96.5, 24, 91.5), (90, 33, 92), (79, 41, 93), (48, 50, 93),
    (102, 8, 82.6), (99, 16, 83), (93.5, 25, 84), (86, 33, 85), (73, 40, 86), (44, 47, 86),
    (96.5, 9, 72), (93, 17, 72.5), (86, 26, 74), (75, 34, 76), (58, 41.5, 78), (35, 44, 79),
    (92, 9, 63), (87, 17, 64), (78, 26, 66), (64, 34, 69), (45, 39.5, 72), (25, 43, 75),
    (78, 12, 56), (70, 22, 58), (58, 30, 61), (40, 37, 65),
]
CRANIUM = ((12.0, 0.0, 185.0), (HEAD_LENGTH / 2 / MM - 1.75, HEAD_BREADTH / 2 / MM, 95.0))   # centre, semi-axes (mm): [P]'s
                                # breadth; its length from the glabella (107.8) back 186.5 [P]; the vertex 280 (ours)
NECK = ((8.0, 0.0, -10.0), (20.0, 0.0, 80.0), 41.0)       # the neck (the maker's capsule), mm
LOWER = ((45, 60, 75, 90, 105, 120, 135, 150), (41, 42, 44, 46.5, 50.5, 56, 61.5, 66), (-27, -27.5, -29, -33, -40, -50, -60, -68))
                                # the lower head's half-width and back per height (mm; ours: jaw, nape, ears' level)


def _cranium_r(th, z):
    (cx, _, cz), (a, b, c) = CRANIUM
    q = 1 - ((z - cz) / c) ** 2
    if q <= 0:
        return 0.0
    ux, uy = math.cos(th), math.sin(th)
    ox = AX_X / MM - cx
    A = (ux / a) ** 2 + (uy / b) ** 2; B = 2 * ox * ux / a ** 2; C = (ox / a) ** 2 - q
    return (-B + math.sqrt(B * B - 4 * A * C)) / (2 * A)


def _neck_r(th, z):
    (x0, _, z0), (x1, _, z1), r = NECK
    t = min(max((z - z0) / (z1 - z0), 0.0), 1.0)
    cx = x0 + (x1 - x0) * t
    ux = math.cos(th)
    ox = AX_X / MM - cx
    B = 2 * ox * ux; C = ox * ox - r * r
    return (-B + math.sqrt(B * B - 4 * C)) / 2


def _lower_r(th, z):
    zs, ws, bs = LOWER
    w = float(np.interp(z, zs, ws)); bk = float(np.interp(z, zs, bs))
    cx, a = bk + 60.0, 60.0
    ux, uy = math.cos(th), math.sin(th)
    ox = AX_X / MM - cx
    A = (ux / a) ** 2 + (uy / w) ** 2; B = 2 * ox * ux / a ** 2; C = (ox / a) ** 2 - 1
    return (-B + math.sqrt(B * B - 4 * A * C)) / (2 * A)


TEMPLE_PINCH = 6.0              # mm (ours)


def spline_points():
    """(theta, z, r) mm: her face's points and her head's sides and back, mirrored"""
    pts = [(math.atan2(y, x - AX_X / MM), z, math.hypot(x - AX_X / MM, y)) for x, y, z in FACE_POINTS]
    for z in range(30, 271, 8):
        for thd in (95, 110, 125, 140, 160, 180):
            if thd == 95 and 50 < z < 150:
                continue
            th = math.radians(thd)
            rc = _cranium_r(th, z)
            r = (rc if rc > 0 else None) if z >= 150 else (_neck_r(th, z) - 2 if z <= 45 else max(_lower_r(th, z), rc))
            if r:
                # the temples: the head in front of the ears narrower than the cranium's smooth ellipsoid (the temporal fossa), so the
                # face is [F]'s zy-zy wide at the zygia (ours: the pinch's size and place)
                r -= TEMPLE_PINCH * math.exp(-((z - 150) / 14) ** 2) * max(0.0, math.cos(th - math.radians(95)) ** 2) * (thd <= 125)
                pts.append((th, z, r))
    for z in (244, 252, 260, 268, 276):
        for thd in (0, 15, 30, 45, 60, 75, 90):
            r = _cranium_r(math.radians(thd), z)
            if r > 5:
                pts.append((math.radians(thd), z, r))
    for z in (30, 38):
        for thd in (0, 25, 50):
            pts.append((math.radians(thd), z, _neck_r(math.radians(thd), z) - 2))
    return pts + [(-t, z, r) for t, z, r in pts if t > 1e-9]


TH_STEP, Z_STEP = math.radians(0.5), 0.5
_TH = np.arange(-math.pi, math.pi + 1e-9, TH_STEP)
_ZT = np.arange(20.0, 280.0 + 1e-9, Z_STEP)
_RTAB = None


def radius_table():
    """the spline's radius (mm) on a (theta, z) grid, computed once (the build's)"""
    global _RTAB
    if _RTAB is None:
        from scipy.interpolate import RBFInterpolator
        pts = spline_points()
        P = np.array([(t * TPS_L, z) for t, z, r in pts]); R = np.array([r for t, z, r in pts])
        fit = RBFInterpolator(P, R, kernel="thin_plate_spline", smoothing=TPS_SMOOTH)
        TT, ZZ = np.meshgrid(_TH, _ZT, indexing="ij")
        q = np.stack([TT.reshape(-1) * TPS_L, ZZ.reshape(-1)], 1)
        _RTAB = np.concatenate([fit(q[i:i + 20000]) for i in range(0, len(q), 20000)]).reshape(TT.shape)
    return _RTAB


def _radius(th, z):
    from scipy.ndimage import map_coordinates
    tab = radius_table()
    th = np.asarray(th, float); z = np.asarray(z, float)
    ci = (th - _TH[0]) / TH_STEP; cj = (np.clip(z, _ZT[0], _ZT[-1]) - _ZT[0]) / Z_STEP
    return map_coordinates(tab, [ci.reshape(-1), cj.reshape(-1)], order=3, mode="nearest").reshape(th.shape)


# ---------------------------------------------------------------- signed distances (m)
def sd_ell(P, c, r):
    q = (P - np.asarray(c)) / np.asarray(r)
    k0 = np.linalg.norm(q, axis=-1)
    k1 = np.linalg.norm(q / np.asarray(r), axis=-1)
    return k0 * (k0 - 1.0) / np.maximum(k1, 1e-9)


def sd_cap(P, a, b, ra, rb=None):
    rb = ra if rb is None else rb
    a, b = np.asarray(a, float), np.asarray(b, float)
    ab = b - a
    t = np.clip(((P - a) @ ab) / (ab @ ab), 0, 1)
    return np.linalg.norm(P - (a + t[..., None] * ab), axis=-1) - (ra + (rb - ra) * t)


def smin(a, b, k):
    h = np.maximum(k - np.abs(a - b), 0.0) / k
    return np.minimum(a, b) - h * h * k / 4


def smax(a, b, k):
    return -smin(-a, -b, k)


def nose_sdf(P):
    """her nose: the dorsum from the nasion to the supratip, the tip (pronasale PN), the alae (al-al 31.4 [F]), the columella to the
    subnasale SN (the landmarks [F]; the parts' radii ours)"""
    Pm = P.copy(); Pm[..., 1] = np.abs(P[..., 1])
    root = N_PT - [.0058, 0, .007]
    tipc = PN - [.0095, 0, 0]
    d = sd_cap(P, root, tipc + [-.001, 0, .007], .0056, .0072)
    d = smin(d, np.linalg.norm(P - tipc, axis=-1) - .0095, .007)
    d = smin(d, sd_ell(Pm, (SN[0] - .005, AL_AL / 2 - .0055, SN[2] + .005), (.0075, .0055, .0065)), .006)
    d = smin(d, sd_cap(P, PN - [.008, 0, .006], SN + [-.0042, 0, .0025], .0030), .004)
    return d


NOSTRIL = ((.001, .0060, .0015), (.0045, .0025, .0020))  # (offset from sn, semi-axes), ours


def head_sdf(P):
    """the head sheet's signed distance (m; negative inside): the spline about the axis, the cranium's crown (under her hair), the
    neck, the nose, the nostrils cut. Her eyes' fissures are cut into the mesh apart (cut_eyes)"""
    P = np.asarray(P, float)
    x, y, z = P[..., 0] / MM, P[..., 1] / MM, P[..., 2] / MM
    th = np.arctan2(y, x - AX_X / MM); r = np.hypot(x - AX_X / MM, y)
    zc = np.clip(z, 25, 275)
    e = 0.3
    R = _radius(th, zc)
    Rz = (_radius(th, zc + e) - _radius(th, zc - e)) / (2 * e)
    Rt = (_radius(th + e / TPS_L, zc) - _radius(th - e / TPS_L, zc)) / (2 * e / TPS_L)
    d = (r - R) / np.sqrt(1 + Rz ** 2 + (Rt / np.maximum(r, 5)) ** 2) * MM
    d = smax(d, (z - 276) * MM, .006)
    (cc, cr) = CRANIUM
    top = smax(sd_ell(P, np.array(cc) * MM, np.array(cr) * MM), (270 - z) * MM, .006)
    d = smin(d, top, .004)
    (a, b, rn) = NECK
    d = smin(d, sd_cap(P, np.array(a) * MM, np.array(b) * MM, rn * MM), .010)
    d = smin(d, nose_sdf(P), .005)
    Pm = P.copy(); Pm[..., 1] = np.abs(P[..., 1])
    d = smax(d, -sd_ell(Pm, SN + NOSTRIL[0], NOSTRIL[1]), .0012)
    return d


def sdf_grad(f, P, eps=1e-4):
    g = np.zeros_like(P)
    for i in range(3):
        e = np.zeros(3); e[i] = eps
        g[:, i] = (f(P + e) - f(P - e)) / (2 * eps)
    return g


# ---------------------------------------------------------------- meshing: surface nets, projected onto the surface
SHEET_H = .0016                 # the grid (m; ours)
SHEET_BOX = ((-.085, -.085, .035), (.140, .085, .292))


def surface_nets(f, lo, hi, h, chunk=400000):
    xs = np.arange(lo[0], hi[0] + 1e-9, h); ys = np.arange(lo[1], hi[1] + 1e-9, h); zs = np.arange(lo[2], hi[2] + 1e-9, h)
    X, Y, Z = np.meshgrid(xs, ys, zs, indexing="ij")
    Pg = np.stack([X, Y, Z], -1).reshape(-1, 3)
    F = np.concatenate([f(Pg[i:i + chunk]) for i in range(0, len(Pg), chunk)]).reshape(X.shape)
    nx, ny, nz = F.shape
    inside = F < 0
    cs = np.stack([inside[a:nx - 1 + a, b:ny - 1 + b, c:nz - 1 + c] for a in (0, 1) for b in (0, 1) for c in (0, 1)], -1)
    active = cs.any(-1) & ~cs.all(-1)
    idx = -np.ones(active.shape, np.int64)
    idx[active] = np.arange(active.sum())
    acc = np.zeros(active.shape + (3,)); cnt = np.zeros(active.shape)
    edges = [((0, 0, 0), (1, 0, 0)), ((0, 1, 0), (1, 1, 0)), ((0, 0, 1), (1, 0, 1)), ((0, 1, 1), (1, 1, 1)),
             ((0, 0, 0), (0, 1, 0)), ((1, 0, 0), (1, 1, 0)), ((0, 0, 1), (0, 1, 1)), ((1, 0, 1), (1, 1, 1)),
             ((0, 0, 0), (0, 0, 1)), ((1, 0, 0), (1, 0, 1)), ((0, 1, 0), (0, 1, 1)), ((1, 1, 0), (1, 1, 1))]
    sl = lambda o: (slice(o[0], nx - 1 + o[0]), slice(o[1], ny - 1 + o[1]), slice(o[2], nz - 1 + o[2]))
    I, J, K = np.meshgrid(np.arange(nx - 1), np.arange(ny - 1), np.arange(nz - 1), indexing="ij")
    base = np.stack([xs[I], ys[J], zs[K]], -1)
    for a, b in edges:
        fa, fb = F[sl(a)], F[sl(b)]
        cross = (fa < 0) != (fb < 0)
        t = np.where(cross, fa / np.where(cross, fa - fb, 1), 0)
        p = base + h * (np.asarray(a) + t[..., None] * (np.asarray(b) - np.asarray(a)))
        acc += np.where(cross[..., None], p, 0); cnt += cross
    V = acc[active] / cnt[active][:, None]
    quads = []
    for ax in range(3):
        o = [0, 0, 0]; o[ax] = 1
        fa = F[:nx - o[0], :ny - o[1], :nz - o[2]]
        fb = F[o[0]:, o[1]:, o[2]:]
        ii = np.argwhere((fa < 0) != (fb < 0))
        u, v = [d_ for d_ in range(3) if d_ != ax]
        ok = (ii[:, u] >= 1) & (ii[:, v] >= 1) & (ii[:, u] < F.shape[u] - 1) & (ii[:, v] < F.shape[v] - 1)
        ii = ii[ok]

        def cell(du, dv):
            c = ii.copy(); c[:, u] -= du; c[:, v] -= dv
            return idx[c[:, 0], c[:, 1], c[:, 2]]
        q = np.stack([cell(1, 1), cell(0, 1), cell(0, 0), cell(1, 0)], 1)
        flip = fa[ii[:, 0], ii[:, 1], ii[:, 2]] < 0
        if ax == 1:
            flip = ~flip
        q[~flip] = q[~flip][:, ::-1]
        quads.append(q)
    Q = np.concatenate(quads)
    return V, Q[(Q >= 0).all(1)]


def project(f, V, iters=6):
    for _ in range(iters):
        g = sdf_grad(f, V)
        V = V - (f(V) / np.maximum((g * g).sum(1), 1e-12))[:, None] * g
    return V


def triangulate(V, Q):
    d02 = np.linalg.norm(V[Q[:, 0]] - V[Q[:, 2]], axis=1); d13 = np.linalg.norm(V[Q[:, 1]] - V[Q[:, 3]], axis=1)
    a = d02 <= d13
    return np.concatenate([Q[a][:, [0, 1, 2]], Q[a][:, [0, 2, 3]], Q[~a][:, [0, 1, 3]], Q[~a][:, [1, 2, 3]]])


def vertex_normals(V, T):
    fn = np.cross(V[T[:, 1]] - V[T[:, 0]], V[T[:, 2]] - V[T[:, 0]])
    N = np.zeros_like(V)
    for c in range(3):
        np.add.at(N, T[:, c], fn)
    return N / np.maximum(np.linalg.norm(N, axis=1)[:, None], 1e-15)


# ---------------------------------------------------------------- the fissure (the left eye's; the right is its mirror)
AU5_UP = .002                   # the upper lid's rise at AU5 = 1 over the pupil (the old face's)
R_M = .0136                     # the rim's front edge from the globe's centre over the globe (ours: the lids ~1.6 mm thick there)
R_COV = .0133                   # the moving upper lid's front surface (ours)
R_COV_LO = .01345               # the lower lid's (over the upper where they meet in a blink; ours)
BLINK_OVERLAP = .0005           # a blink brings the upper lid's margin this far past the lower's (behind it; ours)
LID_CLEAR = .0002               # the lids' clearance behind her drawn skin over their sweep (ours)
R_COV_IN = .01245               # their rolled edge's inner end (clears the iris's cap, 12.37 mm)
_U_P = (EYE_C["L"][1] - EN[1]) / (EX[1] - EN[1])
_ALPHA = math.log(0.5) / math.log(_U_P)
FISSURE_BETA = (0.70, 1.00)     # the upper margin's arch fuller than the lower's (ours)


def _zline(u):
    return EN[2] + (EX[2] - EN[2]) * u


def margin_yz(u, which, up=0.0):
    """the lid margin's frontal (y, z) at u in [0, 1] from en to ex ('up' or 'lo'); up: the upper margin's extra rise at the
    pupil. The margins peak at the pupil: MRD1 over it, MRD2 under it [EW, F]"""
    u = np.asarray(u, float)
    y = EN[1] + (EX[1] - EN[1]) * u
    zl = _zline(u); zp = _zline(_U_P)
    if which == "up":
        f = np.sin(np.pi * np.clip(u, 0, 1) ** _ALPHA) ** FISSURE_BETA[0]
        return y, zl + ((PUPIL_Z + MRD1 + up) - zp) * f
    f = np.sin(np.pi * np.clip(u, 0, 1) ** _ALPHA) ** FISSURE_BETA[1]
    return y, zl - (zp - (PUPIL_Z - MRD2)) * f


def margin_x(y, z, R=R_M):
    """the rim's depth at (y, z): on the sphere R about the globe's centre over the globe, the canthal line toward the canthi"""
    C = EYE_C["L"]
    u = np.clip((y - EN[1]) / (EX[1] - EN[1]), 0, 1)
    xl = EN[0] + (EX[0] - EN[0]) * u
    q = R * R - (y - C[1]) ** 2 - (z - C[2]) ** 2
    xs = C[0] + np.sqrt(np.maximum(q, 0)) - np.where(q < 0, 0.02 * np.sqrt(-np.minimum(q, 0)), 0)
    k = .0015
    h = np.maximum(k - np.abs(xl - xs), 0) / k
    x = np.maximum(xl, xs) + h * h * k / 4
    # toward the lateral canthus the margins leave the globe's sphere and run back along the canthal line to ex (the lateral canthus
    # lies behind the globe's equator: [F]'s widths), so the drawn corner is ex itself
    w = np.clip((1 - u) / LATERAL_TUCK, 0, 1); w = w * w * (3 - 2 * w)
    return w * x + (1 - w) * xl


LATERAL_TUCK = 0.20            # the share of the fissure's length over which the margins run back to ex (ours)
PATCH_C = np.array([(EN[1] + EX[1]) / 2, PUPIL_Z])
PATCH_AX = (.021, .0135)
PATCH_RINGS = 14


def fissure_outline(n=400, up=AU5_UP):
    """the rim's outline, frontal (y, z): the upper margin at AU5's full rise (ex to en), then the lower at rest (en to ex)"""
    u = np.linspace(0, 1, n)
    yu, zu = margin_yz(u[::-1], "up", up)
    yl, zl = margin_yz(u[1:-1], "lo")
    return np.concatenate([np.stack([yu, zu], 1), np.stack([yl, zl], 1)])


def _ray_outline(poly, ang):
    d = np.stack([np.cos(ang), np.sin(ang)], 1)
    A = poly - PATCH_C; E = np.roll(poly, -1, 0) - poly
    out = np.zeros((len(ang), 2))
    for i, di in enumerate(d):
        den = -di[0] * E[:, 1] + di[1] * E[:, 0]
        with np.errstate(divide="ignore", invalid="ignore"):
            t = (-A[:, 0] * E[:, 1] + A[:, 1] * E[:, 0]) / den
            s = (di[0] * A[:, 1] - di[1] * A[:, 0]) / den
        ok = (t > 0) & (s >= -1e-9) & (s <= 1 + 1e-9) & np.isfinite(t)
        out[i] = PATCH_C + t[ok].min() * di
    return out


def front_x(f, y, z, lo=.05, hi=.14, step=2e-4):
    """the frontmost surface x of f at each (y, z) (a march from the front, then bisection)"""
    y = np.atleast_1d(np.asarray(y, float)); z = np.atleast_1d(np.asarray(z, float))
    xs = np.arange(hi, lo, -step)
    P = np.stack([np.repeat(xs[None, :], len(y), 0), np.repeat(y[:, None], len(xs), 1), np.repeat(z[:, None], len(xs), 1)], -1)
    d = f(P.reshape(-1, 3)).reshape(len(y), len(xs))
    i = np.argmax(d < 0, axis=1)
    a = xs[np.maximum(i - 1, 0)]; b = xs[i]
    for _ in range(20):
        m = (a + b) / 2
        dm = f(np.stack([m, y, z], 1))
        a = np.where(dm >= 0, m, a); b = np.where(dm >= 0, b, m)
    return (a + b) / 2


def cut_eyes(f, V, T, N):
    """cut each eye's fissure into the sheet: the sheet inside the patch ellipse removed, a ring patch from the hole's edge in to
    the rim, the rim rolled back into the socket; returns V, T, N and the patches' vertex ranges"""
    V, T, N = V.copy(), T.copy(), N.copy()
    C = EYE_C["L"]
    ranges = []
    for sg in (1, -1):
        cy, cz = PATCH_C
        q = ((sg * V[:, 1] - cy) / PATCH_AX[0]) ** 2 + ((V[:, 2] - cz) / PATCH_AX[1]) ** 2
        inside = (q < 1) & (V[:, 0] > C[0] - .015) & (sg * V[:, 1] > 0)
        T = T[~inside[T].any(1)]
        E = np.sort(np.concatenate([T[:, [0, 1]], T[:, [1, 2]], T[:, [2, 0]]]), 1)
        uk, cnt = np.unique(E, axis=0, return_counts=True)
        bd = uk[cnt == 1]
        mid = V[bd].mean(1)
        qm = ((sg * mid[:, 1] - cy) / PATCH_AX[0]) ** 2 + ((mid[:, 2] - cz) / PATCH_AX[1]) ** 2
        bd = bd[(qm < 1.6) & (mid[:, 0] > C[0] - .02) & (sg * mid[:, 1] > 0)]
        nb = {}
        for a_, b_ in bd:
            nb.setdefault(int(a_), []).append(int(b_)); nb.setdefault(int(b_), []).append(int(a_))
        start = int(bd[0][0]); loop = [start]; prev = None; cur = start
        while True:
            nxt = [v for v in nb[cur] if v != prev]
            prev, cur = cur, nxt[0]
            if cur == start:
                break
            loop.append(cur)
        loop = np.array(loop)
        if len(loop) != len(bd):
            raise RuntimeError(f"the eye patch's hole is not one loop ({len(loop)} of {len(bd)} edges)")
        ang = np.arctan2(V[loop, 2] - cz, sg * V[loop, 1] - cy)
        if np.sum(np.diff(np.unwrap(ang))) < 0:
            loop = loop[::-1]; ang = ang[::-1]
        mg = _ray_outline(fissure_outline(), ang)
        for corner in (EN, EX):                             # the canthi drawn exactly: the ring's point nearest each corner put on it
            ca = math.atan2(corner[2] - cz, corner[1] - cy)
            i_c = int(np.argmin(np.abs(np.angle(np.exp(1j * (ang - ca))))))
            mg[i_c] = corner[1:3]
        byz = np.stack([sg * V[loop, 1], V[loop, 2]], 1)
        # the rings' outer edge smoothed (the first strip takes the hole's grid steps)
        sm = byz.copy()
        for _ in range(3):
            sm = (np.roll(sm, 1, 0) + 2 * sm + np.roll(sm, -1, 0)) / 4
        ks = np.linspace(0, 1, PATCH_RINGS + 1) ** 0.75
        rings = [loop]
        newV = []
        for k in ks[1:]:
            src = sm if k > ks[1] * 1.01 else (byz + sm) / 2
            yz = (1 - k) * src + k * mg
            xf = front_x(f, sg * yz[:, 0], yz[:, 1])
            xm = margin_x(yz[:, 0], yz[:, 1])
            g = np.clip((k - 0.55) / 0.45, 0, 1); g = g * g * (3 - 2 * g)
            x = xm if k >= 1 else xf + g * (xm - xf)
            newV.append(np.stack([x, sg * yz[:, 0], yz[:, 1]], 1))
        m3 = newV[-1] * np.array([1, sg, 1])
        din = np.concatenate([np.zeros((len(mg), 1)), (PATCH_C - mg) / np.linalg.norm(PATCH_C - mg, axis=1)[:, None]], 1)
        rc = m3 - C
        rn = np.linalg.norm(rc, axis=1)[:, None]
        for dr in (.00035, .00075):                     # rolled straight back: the rim's edge is the margin itself
            p = m3 - np.array([dr * 1.4, 0, 0])
            newV.append(p * np.array([1, sg, 1]))
        base = len(V)
        for R_ in newV:
            rings.append(np.arange(len(V), len(V) + len(R_)))
            V = np.concatenate([V, R_]); N = np.concatenate([N, np.zeros_like(R_)])
        M = len(loop)
        quads = []
        for k in range(len(rings) - 1):
            a0, a1 = rings[k], rings[k + 1]
            for i in range(M):
                j = (i + 1) % M
                quads.append((a0[i], a0[j], a1[j], a1[i]))
        Qp = np.array(quads)
        Tp = np.concatenate([Qp[:, [0, 1, 2]], Qp[:, [0, 2, 3]]])
        fn = np.cross(V[Tp[:, 1]] - V[Tp[:, 0]], V[Tp[:, 2]] - V[Tp[:, 0]])
        if np.median(fn[:, 0]) < 0:
            Tp = Tp[:, ::-1]
        T = np.concatenate([T, Tp])
        ranges.append((base, len(V)))
    NN = vertex_normals(V, T)
    for a_, b_ in ranges:
        N[a_:b_] = NN[a_:b_]
    return V, T, N, ranges


# ---------------------------------------------------------------- the moving lids (the left eye's, relative to its globe's centre)
# the lids' hinge: they turn about an axis through the globe's centre (so their shells slide over themselves), its direction set back
# HINGE_BACK_DEG from the frontal plane and tilted HINGE_TILT_DEG (ours: of back-angles 0-33 deg and tilts -12..15 deg, the one at
# which a blink closes the whole fissure with the least turn, hinge_search(); the frontal line alone, ex 3.4 mm behind it, swung
# the lateral margins apart in a blink)
HINGE_BACK_DEG, HINGE_TILT_DEG = 12.0, -9.0
LID_AXIS = np.array([-math.sin(math.radians(HINGE_BACK_DEG)) * math.cos(math.radians(HINGE_TILT_DEG)),
                     math.cos(math.radians(HINGE_BACK_DEG)) * math.cos(math.radians(HINGE_TILT_DEG)), math.sin(math.radians(HINGE_TILT_DEG))])
_E_FWD = np.array([1.0, 0, 0]) - LID_AXIS[0] * LID_AXIS
_E_FWD = _E_FWD / np.linalg.norm(_E_FWD)
_E_UP = np.cross(_E_FWD, LID_AXIS)
LID_SPAN_DEG = (110.0, 60.0)    # the upper lid's reach up and back from its margin, the lower's down (ours: past a full blink,
                                # to the rim at AU5 even at the fissure's ends, where the margin lies near the hinge)


def _to_axis(p):
    return p @ LID_AXIS, np.arctan2(p @ _E_UP, p @ _E_FWD), np.hypot(p @ _E_FWD, p @ _E_UP)


def _from_axis(a, psi, rho):
    return a[..., None] * LID_AXIS + (rho * np.cos(psi))[..., None] * _E_FWD + (rho * np.sin(psi))[..., None] * _E_UP


def _rodrigues(axis, theta):
    K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return np.eye(3) + math.sin(theta) * K + (1 - math.cos(theta)) * (K @ K)


def lid_turn(theta, side="L"):
    """a lid's rotation (matrix) turned down by theta (rad) about the hinge through the globe's centre (the right eye's mirrored)"""
    R = _rodrigues(LID_AXIS, float(theta))
    if side == "R":
        M = np.diag([1.0, -1.0, 1.0])
        R = M @ R @ M
    return R


SHEET_TRIS = None                # the drawn sheet's triangles around the left eye (the build's; the lids keep behind them)
SKIN_TRIS = None                 # those more than RIM_BAND from the lid margins (her skin, not the opening's rolled rim)
RIM_BAND = .001                  # the rolled rim's reach from the drawn margins (it rolls back 1.05 mm: cut_eyes)


def rim_curves(n=400):
    """the drawn lid margins (3D, the left eye's): the upper at AU5's rise (the rim) and at rest, the lower at rest"""
    u = np.linspace(0, 1, n)
    return np.concatenate([np.stack([margin_x(*margin_yz(u, w_, a_)), *margin_yz(u, w_, a_)], 1)
                           for w_, a_ in (("up", AU5_UP), ("up", 0.0), ("lo", 0.0))])


def set_sheet(V, T):
    """the lids' reference: the drawn sheet's triangles near the left eye, and those of them that are skin"""
    global SHEET_TRIS, SKIN_TRIS
    from scipy.spatial import cKDTree
    near = np.linalg.norm(V[T].mean(1) - EYE_C["L"], axis=1) < .030
    SHEET_TRIS = V[T[near]]
    SKIN_TRIS = SHEET_TRIS[cKDTree(rim_curves()).query(SHEET_TRIS.mean(1))[0] > RIM_BAND]


def _ray_tris(o, dirs, tris):
    """the nearest positive hit distance of rays from o along dirs with triangles (n x 3 x 3) (Moller-Trumbore); inf where none"""
    v0, e1, e2 = tris[:, 0], tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0]
    out = np.full(len(dirs), np.inf)
    for s0 in range(0, len(dirs), 256):
        D = dirs[s0:s0 + 256]
        pv = np.cross(D[:, None, :], e2[None, :, :])
        det = (e1[None] * pv).sum(-1)
        ok = np.abs(det) > 1e-14
        inv = np.where(ok, 1.0 / np.where(ok, det, 1.0), 0.0)
        tv = o - v0
        u = (tv[None] * pv).sum(-1) * inv
        qv = np.cross(tv, e1)
        v = (D[:, None, :] * qv[None]).sum(-1) * inv
        t = (e2 * qv).sum(-1)[None] * inv
        hit = ok & (u >= 0) & (v >= 0) & (u + v <= 1) & (t > 1e-6)
        out[s0:s0 + 256] = np.where(hit, t, np.inf).min(1)
    return out


def _static_radius(f, dirs, rmax=.022, skin_only=False):
    """the static skin's distance from the left globe's centre along unit directions: the drawn sheet's (the patch with its tucked rim;
    skin_only: not the rim within RIM_BAND of the margins) when the build has it, else the head's surface"""
    C = EYE_C["L"]
    if SHEET_TRIS is not None:
        r = _ray_tris(C, dirs, SKIN_TRIS if skin_only else SHEET_TRIS)
        return np.minimum(r, rmax)
    rs = np.arange(.0115, rmax, 1e-4)
    P = C[None, None, :] + dirs[:, None, :] * rs[None, :, None]
    d = f(P.reshape(-1, 3)).reshape(len(dirs), len(rs))
    out = d >= 0
    return np.where(out.any(1), rs[np.argmax(out, axis=1)], rmax)


def _lid_frame(which, n_t=64):
    """a lid's margin row in the hinge's coordinates: t along the fissure (en 0, ex 1; past them under the sheet), the axial
    coordinate a, the angle psi and the distance rho from the hinge"""
    t = np.linspace(-0.12, 1.12, n_t)
    tc = np.clip(t, 0, 1)
    y, z = margin_yz(tc, which)
    a, psi, rho = _to_axis(np.stack([margin_x(y, z), y, z], -1) - EYE_C["L"])
    a = a + (t - tc) * float((EX - EN) @ LID_AXIS)
    if which == "lo":
        psi = psi - math.radians(1.0)                           # at rest just under the rim
    return t, a, psi, rho


def lid_mesh(f, which, n_t=64, n_psi=26):
    """a lid's shell (vertices relative to the left globe's centre, faces, the margin row): from its rest margin up and back
    ('up') or down ('lo'), at R_COV (R_COV_LO) or just behind her skin wherever the lid ever passes under it, whichever is nearer
    (so it never shows through the skin), the margin rolled in to R_COV_IN; beyond the canthi it carries on under the sheet"""
    t, a, psi, rho = _lid_frame(which, n_t)
    sgn = 1 if which == "up" else -1
    span = LID_SPAN_DEG[0 if which == "up" else 1]
    rows = [(2.6, -.00085), (1.2, -.0004), (0.0, 0.0)] + [(k, 0.0) for k in span * (np.linspace(0, 1, n_psi)[1:] ** 1.3)]
    # the rolled edge turns back behind the margin (the margin itself the lid's lowest drawn point)
    verts = []
    # every place the lid takes as it turns (lid_range: the upper from AU5's rise to a full blink, the lower from rest to AU6 and
    # AU7 at once), wherever her skin lies in front of it (the drawn sheet, not the opening's rolled rim within RIM_BAND of its
    # margins, nor sheet the globe itself hides), the lid stays LID_CLEAR behind it: each column (along the fissure) at the nearest
    # the skin comes over the column's whole sweep (one radius a column, eroded and smoothed along the fissure, so the lid stays a
    # smooth shell, drawing back only toward the lateral canthus, where the rim tucks in behind the globe)
    sweep = np.linspace(*lid_range(which), 14)
    dir_rows = []
    col = np.full(n_t, np.inf)
    for dpsi, dr in rows:
        dirs = _from_axis(a, psi + sgn * math.radians(dpsi), rho)
        dirs = dirs / np.linalg.norm(dirs, axis=1)[:, None]
        dir_rows.append(dirs)
        if dr == 0.0:
            for th in sweep:
                rs = _static_radius(f, dirs @ lid_turn(th if which == "up" else -th).T, skin_only=True)
                col = np.minimum(col, np.where(rs > EYE_R, rs, np.inf))
    top = R_COV if which == "up" else R_COV_LO
    col = np.minimum(col - LID_CLEAR, top)
    col = np.array([col[max(0, i - 2):i + 3].min() for i in range(n_t)])
    col = np.convolve(np.pad(col, 3, mode="edge"), np.ones(7) / 7, mode="valid")
    col = np.maximum(col, EYE_R + 1e-4)                         # outside the globe
    if which == "up":
        col[t > 1] = EYE_R - 1e-4                               # past ex the upper lid goes inside the globe (the lower covers there)
    for (dpsi, dr), dirs in zip(rows, dir_rows):
        rc = col + dr
        if dr:
            rc = np.minimum(np.maximum(rc, R_COV_IN), col - 5e-5)       # the rolled edge clears the iris, behind the lid's front
        verts.append(dirs * rc[:, None])
    V = np.concatenate(verts)
    F = []
    for r in range(len(rows) - 1):
        for i in range(n_t - 1):
            p00, p01, p10, p11 = r * n_t + i, r * n_t + i + 1, (r + 1) * n_t + i, (r + 1) * n_t + i + 1
            F += [(p00, p01, p11), (p00, p11, p10)]
    F = np.array(F)
    fn = np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]])
    if np.median((fn * V[F].mean(1)).sum(1)) < 0:
        F = F[:, ::-1]
    return V, F, verts[2]


def iris_mesh(n_r=10, n_a=72):
    """the iris seen through the cornea: the cap a 10 mm sphere cuts from the globe, its rim the limbus (IRIS_D across), drawn finely
    (a sphere's own drawing is too coarse at the limbus), 0.15 mm proud of the globe; relative to the globe's centre, facing +x"""
    rl = IRIS_D / 2
    cap = .010
    xi = math.sqrt(EYE_R ** 2 - rl ** 2)
    di = xi - math.sqrt(cap ** 2 - rl ** 2)
    V = [np.array([di + cap + .00015, 0, 0])]
    for i in range(1, n_r + 1):
        r_ = rl * i / n_r
        x = di + math.sqrt(cap ** 2 - r_ ** 2) + .00015 * (1 - (i / n_r) ** 4)
        for j in range(n_a):
            ph = 2 * math.pi * j / n_a
            V.append(np.array([x, r_ * math.cos(ph), r_ * math.sin(ph)]))
    # a skirt tucked into the globe (hides the limbus's edge at a slant)
    for j in range(n_a):
        ph = 2 * math.pi * j / n_a
        V.append(np.array([xi - .0004, (rl + .0001) * math.cos(ph), (rl + .0001) * math.sin(ph)]))
    V = np.array(V)
    F = [(0, 1 + j, 1 + (j + 1) % n_a) for j in range(n_a)]
    for i in range(n_r):
        a0 = 1 + (i - 1) * n_a if i else None
        if i == 0:
            continue
        for j in range(n_a):
            j1 = (j + 1) % n_a
            F += [(1 + (i - 1) * n_a + j, 1 + i * n_a + j, 1 + i * n_a + j1), (1 + (i - 1) * n_a + j, 1 + i * n_a + j1, 1 + (i - 1) * n_a + j1)]
    last = 1 + (n_r - 1) * n_a
    sk = 1 + n_r * n_a
    for j in range(n_a):
        j1 = (j + 1) % n_a
        F += [(last + j, sk + j, sk + j1), (last + j, sk + j1, last + j1)]
    F = np.array(F)
    fn = np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]])
    if np.median(fn[:, 0]) < 0:
        F = F[:, ::-1]
    return V, F


# ---------------------------------------------------------------- her skin's shade of the room's indirect light (ambient occlusion)
# MuJoCo draws a light's ambient term (the room's indirect light, make_g1room.ROOM_INDIRECT of each light's diffuse) with no
# occlusion, so an eye socket, the creases beside the nose and under the lips would be lit as fully as her forehead. The share of
# a cosine-weighted hemisphere that leaves her head (AO) is computed from her own geometry (the sheet, her eyes, her hair) and
# baked, as renderers without indirect light bake it, into the part of her albedo the indirect light makes: at a surface lit at
# the mean cosine of a hemisphere of light directions (0.5) the indirect light is ROOM_INDIRECT / (ROOM_INDIRECT + 0.5) of the
# light it gets, so the albedo is scaled by 1 - share x (1 - AO). Never set: computed. The key and fill lamps' missing shadows are
# not faked (the eyes draw the sun's shadow only: body/sim/eyes.py).
AO_RAYS, AO_STEP, AO_REACH, AO_CELL = 48, .0008, .05, .0008      # rays per point, the march, its reach, the occupancy grid (m)
AO_TEX = 2048                   # the texture's size (an atlas of six box projections: split_charts)
AO_V = (.030, .280)             # the texture's height range (m)
MEAN_COS = 0.5


def ao_share(room_indirect):
    return room_indirect / (room_indirect + MEAN_COS)


# the texture's atlas: each face of the sheet drawn from the axis direction it faces most (a box projection: +x her face, +y and -y
# her sides and the nose's flanks, -z the undersides of the nose and chin, +z the crown, -x the nape), so no face is edge-on to its
# projection; a vertex shared by two charts is duplicated (the same place and normal)
ATLAS_BOX = ((-.090, .145), (-.085, .085), (.030, .290))
ATLAS = {0: ((1, 2), (0.00, 0.33), (0.00, 0.50)), 1: ((0, 2), (0.335, 0.665), (0.00, 0.50)), 2: ((0, 2), (0.67, 1.00), (0.00, 0.50)),
         3: ((0, 1), (0.00, 0.33), (0.505, 0.78)), 4: ((0, 1), (0.335, 0.665), (0.505, 0.78)), 5: ((1, 2), (0.67, 1.00), (0.505, 0.78))}
OVERFLOW_V, OVERFLOW_CELL = (0.785, 1.0), 10     # the faces a projection would draw over another surface: one cell of texels each
CHART_DIR = np.array([[1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, -1], [0, 0, 1], [-1, 0, 0]], float)


def uv_chart(V, chart):
    (i, j), (u0, u1), (v0, v1) = ATLAS[chart]
    fi = (V[:, i] - ATLAS_BOX[i][0]) / (ATLAS_BOX[i][1] - ATLAS_BOX[i][0])
    fj = (V[:, j] - ATLAS_BOX[j][0]) / (ATLAS_BOX[j][1] - ATLAS_BOX[j][0])
    return np.stack([u0 + (u1 - u0) * np.clip(fi, 0, 1), v0 + (v1 - v0) * np.clip(fj, 0, 1)], 1)


def split_charts(V, T, N, S=None):
    """each face assigned the chart it faces most; where two surfaces of one chart would share texels (a nostril's wall behind the
    nose's tip), the one behind gets its own cell in the overflow strip. A vertex used by two charts (or by an overflow face) is
    duplicated (the same place and normal). Returns V, T, N, UV, each vertex's source, and the overflow faces (their index in T)"""
    S = S or AO_TEX
    fn = np.cross(V[T[:, 1]] - V[T[:, 0]], V[T[:, 2]] - V[T[:, 0]])
    fn /= np.maximum(np.linalg.norm(fn, axis=1)[:, None], 1e-15)
    chart = np.argmax(fn @ CHART_DIR.T, axis=1)
    # the conflicts: per texel of a chart, the face furthest along the chart's direction owns it; faces more than 3 mm behind yield
    UVf = np.zeros((len(T), 3, 2))
    for c in ATLAS:
        sel = chart == c
        UVf[sel] = uv_chart(V[T[sel]].reshape(-1, 3), c).reshape(-1, 3, 2)
    depth = (V[T].mean(1) * CHART_DIR[chart]).sum(1)
    n_sub = 4
    bs = [(i / n_sub, j / n_sub) for i in range(n_sub + 1) for j in range(n_sub + 1 - i)]
    best = np.full(S * S, -np.inf)
    idx = []
    for a, b in bs:
        w = np.array([1 - a - b, a, b])
        uv_ = (UVf * w[None, :, None]).sum(1)
        k = np.clip((uv_[:, 1] * S).astype(int), 0, S - 1) * S + np.clip((uv_[:, 0] * S).astype(int), 0, S - 1)
        idx.append(k)
        np.maximum.at(best, k, depth)
    behind = np.zeros(len(T), bool)
    for k in idx:
        behind |= depth < best[k] - .003
    key = {}
    src, uvs = [], []
    T2 = np.empty_like(T)
    over = np.nonzero(behind)[0]
    ncol = S // OVERFLOW_CELL
    cell = {f_: i for i, f_ in enumerate(over)}
    for f_ in range(len(T)):
        if behind[f_]:
            i = cell[f_]
            u0 = (i % ncol) * OVERFLOW_CELL / S; v0 = OVERFLOW_V[0] + (i // ncol) * OVERFLOW_CELL / S
            corner = [(1.5, 1.5), (OVERFLOW_CELL - 1.5, 1.5), (1.5, OVERFLOW_CELL - 1.5)]
            for k in range(3):
                T2[f_, k] = len(src); src.append(int(T[f_, k]))
                uvs.append((u0 + corner[k][0] / S, v0 + corner[k][1] / S))
            continue
        c = int(chart[f_])
        for k in range(3):
            kk = (int(T[f_, k]), c)
            if kk not in key:
                key[kk] = len(src); src.append(kk[0]); uvs.append(tuple(uv_chart(V[kk[0]][None], c)[0]))
            T2[f_, k] = key[kk]
    if len(over) and OVERFLOW_V[0] + (len(over) // ncol + 1) * OVERFLOW_CELL / S > OVERFLOW_V[1]:
        raise RuntimeError(f"the texture's overflow strip is full ({len(over)} faces)")
    src = np.array(src)
    return V[src], T2, N[src], np.array(uvs), src, over


class Occupancy:
    """what blocks the indirect light: the head sheet's solid with the fissures open, the globes, her hair (the cap and the maker's
    shapes, as (centre, semi-axes, rotation))"""

    def __init__(self, hair=()):
        lo = np.array([-.09, -.095, .03]); hi = np.array([.15, .095, .30])
        self.lo, self.h = lo, AO_CELL
        xs = np.arange(lo[0], hi[0], AO_CELL); ys = np.arange(lo[1], hi[1], AO_CELL); zs = np.arange(lo[2], hi[2], AO_CELL)
        X, Y, Z = np.meshgrid(xs, ys, zs, indexing="ij")
        P = np.stack([X, Y, Z], -1).reshape(-1, 3)
        occ = np.concatenate([head_sdf(P[i:i + 400000]) < 0 for i in range(0, len(P), 400000)])
        C = EYE_C["L"]
        Pm = P.copy(); Pm[:, 1] = np.abs(P[:, 1])
        r = np.linalg.norm(Pm - C, axis=1)
        from matplotlib.path import Path as MPath
        inside = MPath(fissure_outline()).contains_points(Pm[:, 1:3])
        slot = inside & (Pm[:, 0] > C[0] - .004) & (r > EYE_R) & (r < .0175)
        occ = (occ & ~slot) | (r <= EYE_R)
        occ |= np.concatenate([in_hair(P[i:i + 400000]) for i in range(0, len(P), 400000)])
        for c, s_, R in hair:
            occ |= ((((P - np.asarray(c)) @ np.asarray(R)) / np.asarray(s_)) ** 2).sum(1) <= 1
        self.occ = occ.reshape(X.shape)

    def blocked(self, Q):
        ijk = np.floor((Q - self.lo) / self.h).astype(np.int64)
        ok = (ijk >= 0).all(-1) & (ijk < np.array(self.occ.shape)).all(-1)
        ijk = np.clip(ijk, 0, np.array(self.occ.shape) - 1)
        return ok & self.occ[ijk[..., 0], ijk[..., 1], ijk[..., 2]]


def _hemisphere(n, seed=0):
    rng = np.random.default_rng(seed)
    u1, u2 = rng.random(n), rng.random(n)
    return np.stack([np.sqrt(u1), np.sqrt(1 - u1) * np.cos(2 * np.pi * u2), np.sqrt(1 - u1) * np.sin(2 * np.pi * u2)], 1)


def ambient_occlusion(occ, P, Nrm, chunk=2000):
    """the share of a cosine-weighted hemisphere of directions about each normal that leaves her (0..1)"""
    local = _hemisphere(AO_RAYS)
    steps = np.arange(AO_STEP * 1.5, AO_REACH, AO_STEP)
    out = np.zeros(len(P))
    for s0 in range(0, len(P), chunk):
        p, n = P[s0:s0 + chunk], Nrm[s0:s0 + chunk]
        t1 = np.cross(n, np.where(np.abs(n[:, :1]) < .9, [[1.0, 0, 0]], [[0, 1.0, 0]]))
        t1 /= np.linalg.norm(t1, axis=1)[:, None]
        t2 = np.cross(n, t1)
        D = local[None, :, 0:1] * n[:, None, :] + local[None, :, 1:2] * t1[:, None, :] + local[None, :, 2:3] * t2[:, None, :]
        Q = (p + .0010 * n)[:, None, None, :] + D[:, :, None, :] * steps[None, None, :, None]
        out[s0:s0 + chunk] = 1 - occ.blocked(Q).any(-1).mean(1)
    return out


CHEEK_TINT = (.045, .065, .014, (.045, .135))   # the cheeks' warmth: green and blue lowered by these at their centre (y, z), a
                                                # Gaussian of this width (m) (ours: a real cheek's slight redness)


def skin_tint(V):
    """the sheet's per-vertex colour factor (r, g, b): 1, less green and blue over the cheeks"""
    dg, db, sig, (cy, cz) = CHEEK_TINT
    w = np.exp(-0.5 * (((np.abs(V[:, 1]) - cy) ** 2 + (V[:, 2] - cz) ** 2) / sig ** 2))
    return np.stack([np.ones(len(V)), 1 - dg * w, 1 - db * w], 1)


def smooth_on_mesh(val, T, iters):
    """a per-vertex value averaged with its mesh neighbours, iters times (the rays' noise out)"""
    E = np.concatenate([T[:, [0, 1]], T[:, [1, 2]], T[:, [2, 0]]])
    for _ in range(iters):
        s_ = np.zeros_like(val); c = np.zeros_like(val)
        np.add.at(s_, E[:, 0], val[E[:, 1]]); np.add.at(c, E[:, 0], 1)
        np.add.at(s_, E[:, 1], val[E[:, 0]]); np.add.at(c, E[:, 1], 1)
        val = 0.5 * val + 0.5 * s_ / np.maximum(c, 1)
    return val


def ao_texture(V, T, UV, ao, share, over=(), n_sub=8):
    """the sheet's shade (and the cheeks' warmth) rasterised into its atlas: each face sampled on a barycentric grid, its vertices'
    values interpolated, splatted; an overflow face fills its whole cell (its values carried to the cell's edge); the gaps filled
    from the nearest texel, lightly smoothed; RGB 0..255 (row 0 is v = 0)"""
    from scipy.ndimage import gaussian_filter, distance_transform_edt
    col = (1 - share * (1 - ao))[:, None] * skin_tint(V)
    bs = [(i / n_sub, j / n_sub) for i in range(n_sub + 1) for j in range(n_sub + 1 - i)]
    s = np.zeros((AO_TEX, AO_TEX, 3)); c = np.zeros((AO_TEX, AO_TEX))
    normal = np.ones(len(T), bool); normal[list(over)] = False
    Tn = T[normal]
    for a, b in bs:
        w = np.array([1 - a - b, a, b])
        uv_ = (UV[Tn] * w[None, :, None]).sum(1)
        vv = (col[Tn] * w[None, :, None]).sum(1)
        iu = np.clip((uv_[:, 0] * AO_TEX).astype(int), 0, AO_TEX - 1); iv = np.clip((uv_[:, 1] * AO_TEX).astype(int), 0, AO_TEX - 1)
        for k in range(3):
            np.add.at(s[..., k], (iv, iu), vv[:, k])
        np.add.at(c, (iv, iu), 1)
    img = np.where(c[..., None] > 0, s / np.maximum(c, 1)[..., None], 0)
    _, (ii, jj) = distance_transform_edt(c == 0, return_indices=True)
    img = img[ii, jj]
    n = OVERFLOW_CELL
    for f_ in over:                                     # each overflow cell: barycentric over the whole cell, clamped to its face
        t = T[f_]
        p = UV[t] * AO_TEX                              # texel coordinates of its three corners
        u0, v0 = int(np.floor(p[:, 0].min() - 1.5 + 1e-6)), int(np.floor(p[:, 1].min() - 1.5 + 1e-6))
        yy, xx = np.mgrid[v0:v0 + n, u0:u0 + n]
        q = np.stack([xx.ravel() + .5, yy.ravel() + .5], 1)
        M = np.array([p[1] - p[0], p[2] - p[0]]).T
        l12 = np.clip(np.linalg.solve(M, (q - p[0]).T).T, 0, 1)
        l12 /= np.maximum(l12.sum(1, keepdims=True), 1)
        w = np.concatenate([1 - l12.sum(1, keepdims=True), l12], 1)
        img[yy.ravel(), xx.ravel()] = w @ col[t]
    img = np.stack([gaussian_filter(img[..., k], 1.0) for k in range(3)], -1)
    for f_ in over:                                     # the cells unblurred (their neighbours are other faces)
        t = T[f_]
        p = UV[t] * AO_TEX
        u0, v0 = int(np.floor(p[:, 0].min() - 1.5 + 1e-6)), int(np.floor(p[:, 1].min() - 1.5 + 1e-6))
        yy, xx = np.mgrid[v0:v0 + n, u0:u0 + n]
        q = np.stack([xx.ravel() + .5, yy.ravel() + .5], 1)
        M = np.array([p[1] - p[0], p[2] - p[0]]).T
        l12 = np.clip(np.linalg.solve(M, (q - p[0]).T).T, 0, 1)
        l12 /= np.maximum(l12.sum(1, keepdims=True), 1)
        w = np.concatenate([1 - l12.sum(1, keepdims=True), l12], 1)
        img[yy.ravel(), xx.ravel()] = w @ col[t]
    return np.clip(np.round(img * 255), 0, 255).astype(np.uint8)


# ---------------------------------------------------------------- the frontal depth table (the brows, lips and cheeks lie on it)
FRONT_Y = np.arange(-.080, .0801, .001)
FRONT_Z = np.arange(.040, .2501, .001)
_FRONT = None


def front_table():
    """the sheet's front x on a 1 mm (y, z) grid (the build computes it; the scene loads it)"""
    global _FRONT
    if _FRONT is None:
        _FRONT = np.load(ASSETS / "front.npy").astype(float)
    return _FRONT


def head_surface_x(y, z):
    """the front of her face at (y, z) in her head's frame (m): the sheet's frontmost surface (bicubic on the 1 mm table)"""
    from scipy.ndimage import map_coordinates
    T = front_table()
    fy = (np.asarray(y, float) - FRONT_Y[0]) / .001; fz = (np.asarray(z, float) - FRONT_Z[0]) / .001
    v = map_coordinates(T, [np.atleast_1d(fz), np.atleast_1d(fy)], order=3, mode="nearest")
    return float(v[0]) if np.ndim(y) == 0 else v


def skin_frame(y, z):
    """the skin's point, outward normal and the tangent along +y at (y, z) (from the frontal table)"""
    e = .0005
    x = head_surface_x(y, z)
    dy = (head_surface_x(y + e, z) - head_surface_x(y - e, z)) / (2 * e)
    dz = (head_surface_x(y, z + e) - head_surface_x(y, z - e)) / (2 * e)
    n = np.array([1.0, -dy, -dz]); n /= np.linalg.norm(n)
    return np.array([x, y, z]), n


# ---------------------------------------------------------------- the moving features, drawn from graded face parameters
def _quat(R):
    from scipy.spatial.transform import Rotation as Rot
    x, y, z, w = Rot.from_matrix(R).as_quat()
    return np.array([w, x, y, z])


def _unit(v):
    v = np.asarray(v, float)
    return v / np.linalg.norm(v)


def _on_skin(y, z, along, half_len, half_h, depth, lift):
    """a thin ellipsoid lying on the skin at (y, z): its long axis along `along` (frontal (y, z) direction) in the skin's tangent
    plane, its thin axis the skin's normal; its centre `lift` off the skin (negative: sunk)"""
    p, n = skin_frame(y, z)
    a = np.array([0.0, along[0], along[1]])
    a = _unit(a - (a @ n) * n)
    b = np.cross(n, a)
    R = np.column_stack([n, a, b])
    return p + lift * n, _quat(R), (depth, half_len, half_h)


def _axis_x(v):
    x = _unit(v)
    z = np.array([0.0, 0.0, 1.0]); z = _unit(z - (z @ x) * x)
    return np.column_stack([x, np.cross(z, x), z])


BROW_PIECES = 8                 # each brow a chain of thin rigid strips lying on the skin, overlapping end to end (ours)
BROW_LIFT = (.00022, .00026)    # their height over the skin, alternating (a strip overlaps its neighbour's end; ours)
BROW_OVERLAP = .0025
BROW_MIN_LIFT = .00012          # no part of a moved strip lower than this over the skin (ours)
BROW_HEAD_IN = .0035            # the brow's head this far medial of en (ours: the brow full over the medial canthus)
_AL_Z = SN[2] + .005            # the alar base's height (the nose's ala centre, ours)


def brow_rest():
    """the left brow at rest: frontal y of its head and tail, and its lower margin z(y) and thickness (m). The lower margin over the
    canthi and the lid margin [G]; the top 25 mm over the pupil [M]; the head over the ala, the tail on the line from the ala through
    ex (the clinical convention), the head BROW_HEAD_IN medial of en; the thickness tapering from the head to the tail (ours)"""
    from scipy.interpolate import PchipInterpolator as Pchip, CubicSpline
    C = EYE_C["L"]
    ys = [EN[1]] + [C[1] + dy for dy, _ in BROW_OVER_LID] + [EX[1]]
    zlo = [EN[2] + BROW_OVER_EN] + [float(margin_yz((C[1] + dy - EN[1]) / (EX[1] - EN[1]), "up")[1]) + h for dy, h in BROW_OVER_LID] \
        + [EX[2] + BROW_OVER_EX]
    slope = (EX[1] - AL_AL / 2) / (EX[2] - _AL_Z)
    tail_z = EX[2] + BROW_OVER_EX - .0015
    y_tail = EX[1] + slope * (tail_z - EX[2])
    y_head = EN[1] - BROW_HEAD_IN
    ys = [y_head] + ys + [y_tail]; zlo = [zlo[0] + .0002] + zlo + [tail_z]
    lo = CubicSpline(ys, zlo, bc_type="natural")
    top_pupil = PUPIL_Z + BROW_TOP_OVER_PUPIL
    th = Pchip([y_head, C[1], EX[1], y_tail], [.0105, top_pupil - float(lo(C[1])), .0060, .0025])
    return y_head, y_tail, lo, th


def _brow_splits():
    y0, y1, lo, th = brow_rest()
    b = [y0 + (y1 - y0) * k / BROW_PIECES for k in range(BROW_PIECES + 1)]
    return y0, y1, lo, th, b


def _brow_mid(lo, th, y):
    return float(lo(y)) + float(th(y)) / 2


def _brow_along(lo, th, y, e=.0005):
    return _unit([2 * e, _brow_mid(lo, th, y + e) - _brow_mid(lo, th, y - e)])


def _skin_axes(y, z, along):
    p, n = skin_frame(y, z)
    a = np.array([0.0, along[0], along[1]])
    a = _unit(a - (a @ n) * n)
    return p, np.column_stack([n, a, np.cross(n, a)])


BROW_NS = 10


def brow_meshes(n_s=BROW_NS, n_t=6):
    """the left brow's strips: {k: (vertices in the strip's frame, faces)}; the frame at the strip's medial joint on the brow's midline
    on the skin at rest (x the skin's normal, y along the brow, z across it). Each strip a thin sheet over the skin, its rim tucked
    0.3 mm under it; the first strip's head rounded, the last strip's tail closed to a point"""
    y0, y1, lo, th, b = _brow_splits()
    out = {}
    for k in range(BROW_PIECES):
        ya = b[k]; yb = min(b[k + 1] + BROW_OVERLAP, y1)
        P0, R0 = _skin_axes(ya, _brow_mid(lo, th, ya), _brow_along(lo, th, ya))
        lift = BROW_LIFT[k % 2]
        rows = []
        for t in np.linspace(-0.08, 1.08, n_t + 2):
            row = []
            for s_ in np.linspace(0, 1, n_s):
                y = ya + (yb - ya) * s_
                h = float(th(y))
                w = (y - y0) / (y1 - y0)
                if w < .06:                                          # the head rounded
                    h *= .45 + .55 * math.sqrt(max(0.0, 1 - ((.06 - w) / .06) ** 2))
                if w > .9:                                           # the tail closing
                    h *= max(.15, 1 - (w - .9) / .1 * .85)
                z = _brow_mid(lo, th, y) + h * (np.clip(t, 0, 1) - .5)
                p, n = skin_frame(y, z)
                row.append(R0.T @ (p + n * (lift if 0 <= t <= 1 else -.0003) - P0))
            rows.append(row)
        V = np.array(rows).reshape(-1, 3)
        F = []
        for i in range(len(rows) - 1):
            for j in range(n_s - 1):
                a_, b_, c_, d_ = i * n_s + j, i * n_s + j + 1, (i + 1) * n_s + j, (i + 1) * n_s + j + 1
                F += [(a_, b_, d_), (a_, d_, c_)]
        F = np.array(F)
        fn = np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]])
        if np.median(fn[:, 0]) < 0:
            F = F[:, ::-1]
        out[k] = (V, F)
    return out


def _kabsch(A, B):
    """the rotation and translation that best carry points A onto points B (least squares)"""
    ca, cb = A.mean(0), B.mean(0)
    U, _, Vt = np.linalg.svd((A - ca).T @ (B - cb))
    D = np.diag([1.0, 1.0, np.sign(np.linalg.det(Vt.T @ U.T))])
    R = Vt.T @ D @ U.T
    return R, cb - R @ ca


def _brows(sd, bi, bo, bl, sm):
    """the brow's strips posed: the brow's rise dz along it (the head's, the middle's, the tail's: AU1, AU2, AU4, the smile) and AU4's
    draw-in; each point of a strip is carried to where its place on the brow goes, on the skin, and the strip takes the rigid pose that
    best fits its points' new places (least squares), lifted just clear of the skin where it would sink"""
    sg = 1 if sd == "L" else -1
    y0, y1, lo, th, b = _brow_splits()
    dz_head = .011 * bi - .0065 * bl + .002 * sm
    dz_mid = .5 * (.011 * bi + .009 * bo) - .004 * bl + .0015 * sm
    dz_tail = .009 * bo - .002 * bl + .001 * sm
    out = {}
    for k in range(BROW_PIECES):
        ya = b[k]
        P0, R0 = _skin_axes(ya, _brow_mid(lo, th, ya), _brow_along(lo, th, ya))
        v = rig()[f"brow{k}"]
        W0 = P0 + v @ R0.T                                        # the strip's points at rest (the left brow's)
        w = (W0[:, 1] - y0) / (y1 - y0)
        dz = np.interp(w, [0, .45, 1], [dz_head, dz_mid, dz_tail])
        dy = -.003 * bl * np.clip(1 - w, 0, 1) ** 2
        yt, zt = W0[:, 1] + dy, W0[:, 2] + dz
        lift = W0[:, 0] - head_surface_x(W0[:, 1], W0[:, 2])      # each point's height over the skin at rest, kept
        tgt = np.stack([head_surface_x(yt, zt) + lift, yt, zt], 1)
        R, t = _kabsch(W0, tgt)
        Wn = W0 @ R.T + t
        gap = Wn[:, 0] - head_surface_x(Wn[:, 1], Wn[:, 2])
        t = t + np.array([max(0.0, BROW_MIN_LIFT - float(gap.min())), 0, 0])
        P, Rf = R @ P0 + t, R @ R0
        if sd == "R":                                              # the mirror: the right strip's frame is the left's mirrored
            M = np.diag([1.0, -1.0, 1.0])
            P, Rf = M @ P, M @ Rf @ np.diag([1.0, 1.0, -1.0])
        out[f"brow{k}_{sd}"] = (P, _quat(Rf), None)
    return out


LIP_UP_N, LIP_LO_N = 20, 16     # the lips' segments (ours)
LIP_R = (.0060, .0064)          # the upper and lower lips' tube radius (ours: the upper stands 1.9 mm, the lower 2.1 mm off the skin)
LIP_DRAWN = (0.925, 0.98)       # the tube's chord for each lip over the skin's curve, so that the DRAWN vermilion is [F]'s height
                                # (measured on the drawn lips: tools/sim_face_measure.py, eyes 12)
MOUTH_CURVE = .0075             # the corners' lift at a full smile (the old face's; the born reading reads it)


def _mouth_upper(y, W, net, smile, frown, lift):
    u = np.minimum((y / W) ** 2, 1.0)
    return MOUTH_Z + net * MOUTH_CURVE * u - (.004 * smile - .002 * frown) * (1 - u) + lift * (1 - u)


def _tube(prefix, ys, zfun, hfun, radius):
    """a lip as a chain of capsules of one radius whose axis lies under the skin, deep where the vermilion is thin, so each shows a
    cap of the vermilion's height (a round lip: its fullest proud of the skin by radius - sqrt(radius^2 - h^2)), thinning to nothing
    at the corners; one radius, so the chain is one smooth tube"""
    out = {}
    A = []
    for y in ys:
        z = float(zfun(y)); h = min(float(hfun(y)), radius * .999)
        p, n = skin_frame(y, z)
        A.append(p - n * math.sqrt(radius * radius - h * h))
    for i in range(len(ys) - 1):
        a, b = A[i], A[i + 1]
        ax = b - a
        Rm = _axis_x(ax)
        out[f"{prefix}{i}"] = ((a + b) / 2, _quat(np.column_stack([Rm[:, 1], Rm[:, 2], Rm[:, 0]])), (radius, float(np.linalg.norm(ax)) / 2, 0.0))
    return out


def _lips(sm, fr, jw, rd, pr):
    net = sm - fr
    W = MOUTH_HALF_W * (1 + .12 * sm + .05 * pr - .30 * rd)
    sm_open = max(0.0, sm - .25) / .75
    lift = .0025 * jw + .004 * rd + .0012 * sm_open
    h_open = .0105 * sm_open + .017 * jw + .016 * rd
    hu = lambda y: LIP_DRAWN[0] * LS_ST / 2 * (1 - .4 * pr) * np.clip(1 - np.abs(y / W) ** 2.2, 0, 1) ** .55 + .0002
    hl = lambda y: LIP_DRAWN[1] * ST_LI / 2 * (1 - .4 * pr) * np.clip(1 - np.abs(y / W) ** 2.0, 0, 1) ** .6 + .0002
    zu = lambda y: _mouth_upper(y, W, net, sm, fr, lift)
    zl = lambda y: zu(y) - hu(y) - h_open * np.clip(1 - (y / W) ** 2, 0, 1) ** (.75 if rd > .2 else 1.0) - hl(y)
    out = _tube("mouth", np.linspace(-W, W, LIP_UP_N + 1), zu, hu, LIP_R[0])
    out.update(_tube("lip_lo", np.linspace(-W * .96, W * .96, LIP_LO_N + 1), zl, hl, LIP_R[1]))
    zu0 = zu(0.0)
    zl0 = zu0 - hu(0.0) * 2 - h_open
    oc = np.array([head_surface_x(0.0, (zu0 + zl0) / 2) - .0022, 0, (zu0 + zl0) / 2])
    shown = h_open > .0008
    out["mouth_open"] = (oc if shown else oc - [.02, 0, 0], _quat(np.eye(3)), (.0045, max(.001, W * (.80 + .15 * rd)), max(.001, h_open / 2 + .0012)))
    teeth_k = sm_open * (1 - rd) + .5 * max(0.0, jw - .35) * (1 - rd)
    out["teeth"] = (np.array([oc[0] + .0012, 0, zu0 - hu(0.0) * 2 + .0004 - .0012 * teeth_k]) if teeth_k > .02 else oc - [.02, 0, 0],
                    _quat(np.eye(3)), (.003, .001 + .016 * min(teeth_k, 1) * W / MOUTH_HALF_W, .001 + .0024 * min(teeth_k, 1)))
    return out


LASH_R = .00055                 # the upper lash line's half-thickness (ours)
LASH_SEG = 6
LASH_SPAN = (.06, .80)          # the lash line's reach along the margin from en (ours: short of the canthi)
_RIG = None


def rig():
    """the build's rest data the scene needs: the upper lid's margin row (the lash line's points), per eye relative to its centre"""
    global _RIG
    if _RIG is None:
        _RIG = dict(np.load(ASSETS / "rig.npz"))
    return _RIG


BLINK_INNER = (.02, .98)        # the share of the fissure (from en) a blink closes; its ends meet at the canthi anyway (ours)
_TH_GRID = np.radians(np.arange(0.0, 110.01, .25))
_TH_LO_GRID = np.radians(np.arange(-2.0, 40.01, .5))
_BLINK = {}


def _turned(P, grid):
    """points P (n x 3) turned by each lid turn of grid: (len(grid), n, 3)"""
    Rs = np.stack([_rodrigues(LID_AXIS, t) for t in grid])
    return np.einsum("kij,nj->kni", Rs, P)


def _lower_edge(th_lo):
    """the opening's lower edge seen from the front with the lower lid turned up th_lo: y (sorted), z, relative to the globe's centre
    (the drawn rim or the lower lid's margin, whichever is higher)"""
    u = np.linspace(0, 1, 201)
    y, z = margin_yz(u, "lo")
    t, a, psi, rho = _lid_frame("lo", 257)
    d = _from_axis(a, psi, rho)
    Q = (d / np.linalg.norm(d, axis=1)[:, None] * R_M) @ lid_turn(-th_lo).T
    o = np.argsort(Q[:, 1])
    ye = y - EYE_C["L"][1]
    return ye, np.maximum(z - EYE_C["L"][2], np.interp(ye, Q[o, 1], Q[o, 2]))


def _meet(P, drop):
    """for points P on the upper lid at rest (rel. the globe's centre): per lower lid turn of _TH_LO_GRID, the least upper turn (rad)
    at which each lies drop below the opening's lower edge (the grid's last where never): (len(_TH_LO_GRID), n)"""
    Q = _turned(P, _TH_GRID)
    out = []
    for tl in _TH_LO_GRID:
        ye, ze = _lower_edge(tl)
        ok = Q[..., 2] - np.interp(Q[..., 1], ye, ze) < -drop
        out.append(np.where(ok.any(0), _TH_GRID[np.argmax(ok, 0)], _TH_GRID[-1]))
    return np.array(out)


def closed_turn(th_lo):
    """the upper lid's turn (rad) that closes the fissure (its margin BLINK_OVERLAP under the lower edge over BLINK_INNER) with the
    lower lid turned th_lo"""
    if "closed" not in _BLINK:
        t, a, psi, rho = _lid_frame("up", 257)
        d = _from_axis(a, psi, rho)
        inner = (t > BLINK_INNER[0]) & (t < BLINK_INNER[1])
        _BLINK["closed"] = _meet((d / np.linalg.norm(d, axis=1)[:, None] * R_M)[inner], BLINK_OVERLAP).max(1)
    return float(np.interp(th_lo, _TH_LO_GRID, _BLINK["closed"]))


def lash_meet(th_lo):
    """each lash point's own turn (rad) at which it comes to rest on the opening's lower edge with the lower lid turned th_lo (the
    lashes stop on the lower lid; the upper lid's margin goes on behind it)"""
    if "lash" not in _BLINK:
        _BLINK["lash"] = _meet(rig()["lash"], -LASH_R)
    tab = _BLINK["lash"]
    j = np.interp(th_lo, _TH_LO_GRID, np.arange(len(_TH_LO_GRID)))
    j0 = int(min(max(math.floor(j), 0), len(_TH_LO_GRID) - 2))
    w = j - j0
    return (1 - w) * tab[j0] + w * tab[j0 + 1]


def lid_turns(lu, lt, ck, bk):
    """the upper lid's turn down and the lower's up (rad) for AU5/lowering (lu in [-1, 1]), AU7 (lt), AU6 (ck) and the blink (bk):
    the margins' heights over and under the pupil set the open turns; the blink carries the upper lid to its closed turn"""
    lo_m = -MRD2 + .0045 * ck + .002 * lt
    up_m = MRD1 + AU5_UP * max(lu, 0) - .005 * lt - .007 * max(-lu, 0)
    th_up, th_lo = lid_angles(up_m, lo_m)
    return th_up + max(closed_turn(th_lo) - th_up, 0.0) * bk, th_lo


def hinge_search(backs=range(0, 34, 3), tilts=range(-12, 16, 3)):
    """(the least closing turn (deg), back, tilt) of the hinge directions tried, best first (the lower lid at rest)"""
    global LID_AXIS, _E_FWD, _E_UP
    keep = LID_AXIS, _E_FWD, _E_UP
    out = []
    try:
        for b in backs:
            for tl in tilts:
                br, tr = math.radians(b), math.radians(tl)
                LID_AXIS = np.array([-math.sin(br) * math.cos(tr), math.cos(br) * math.cos(tr), math.sin(tr)])
                u = np.linspace(0, 1, 201)
                inner = (u > BLINK_INNER[0]) & (u < BLINK_INNER[1])
                y, z = margin_yz(u, "up")
                P = (np.stack([margin_x(y, z), y, z], 1) - EYE_C["L"])[inner]
                ye, ze = margin_yz(u, "lo")
                ye, ze = ye - EYE_C["L"][1], ze - EYE_C["L"][2]
                Q = _turned(P, _TH_GRID)
                ok = Q[..., 2] - np.interp(Q[..., 1], ye, ze) < -BLINK_OVERLAP
                need = np.where(ok.any(0), _TH_GRID[np.argmax(ok, 0)], np.inf)
                out.append((round(math.degrees(need.max()), 2), b, tl))
    finally:
        LID_AXIS, _E_FWD, _E_UP = keep
    return sorted(out)


def lid_range(which):
    """the turns (rad) a lid takes over every face state (each of lu, lt, ck, bk at its ends: the extremes are at the ends)"""
    th = [lid_turns(lu, lt, ck, bk) for lu in (-1, 0, 1) for lt in (0, 1) for ck in (0, 1) for bk in (0, 1)]
    th = np.array(th)[:, 0 if which == "up" else 1]
    return float(th.min()), float(th.max())


_LID_TABLE = {}


def _margin_height(which):
    """a lid's turn (rad, a grid) and its margin's height over the pupil (m) seen from the front at that turn (the margin row on
    the rim's sphere R_M, turned about the hinge, where it crosses the pupil's vertical), the rest height the drawn margin's"""
    if which not in _LID_TABLE:
        t, a, psi, rho = _lid_frame(which, 257)
        d = _from_axis(a, psi, rho)
        P = d / np.linalg.norm(d, axis=1)[:, None] * R_M
        th = np.radians(np.arange(-30.0, 75.01, 0.25))
        h = []
        for x in th:
            Q = P @ lid_turn(x if which == "up" else -x).T
            k = np.where((Q[:-1, 1] <= 0) & (Q[1:, 1] > 0) & (Q[:-1, 0] > 0))[0][0]
            w = -Q[k, 1] / (Q[k + 1, 1] - Q[k, 1])
            h.append(Q[k, 2] + w * (Q[k + 1, 2] - Q[k, 2]))
        h = np.array(h)
        h = h - np.interp(0.0, th, h) + (MRD1 if which == "up" else -MRD2)
        _LID_TABLE[which] = (th, h)
    return _LID_TABLE[which]


def lid_angles(up_m, lo_m):
    """the upper lid's turn down and the lower lid's turn up (rad) that put the margins at up_m over and lo_m under the pupil"""
    tu, hu = _margin_height("up")
    tl, hl = _margin_height("lo")
    return float(np.interp(-up_m, -hu, tu)), float(np.interp(lo_m, hl, tl))


def face_geoms(f, gaze_head=None, blink=None):
    """her face's moving parts for graded face parameters f (every key of parent_kin.FACE_NEUTRAL): {name: (pos, quat, size or None)}
    in her head's frame. The lids, the irises and the lash lines are meshes or capsules turned about the eyes' centres; the brows,
    the lips lie on her skin (the cheeks' tint is in her skin's texture). Size None: a mesh (the scene composes its compiled offset)"""
    c01 = lambda k: float(np.clip(f[k], 0, 1))
    sm, fr, ck = c01("smile"), c01("frown"), c01("cheek")
    bi, bo, bl = c01("brow_in"), c01("brow_out"), c01("brow_low")
    lu, lt = float(np.clip(f["lid_up"], -1, 1)), c01("lid_tight")
    jw, rd, pr = c01("jaw"), c01("round"), c01("press")
    bk = c01("blink") if blink is None else float(np.clip(max(blink, f["blink"]), 0, 1))
    out = _lips(sm, fr, jw, rd, pr)
    th_up, th_lo = lid_turns(lu, lt, ck, bk)
    lash = rig()["lash"]
    idx = np.linspace(0, len(lash) - 1, LASH_SEG + 1).round().astype(int)
    lash_th = np.minimum(th_up, lash_meet(th_lo)[idx])            # each lash point stops on the lower lid
    for sd, sg in (("L", 1), ("R", -1)):
        out.update(_brows(sd, bi, bo, bl, sm))
        C = EYE_C[sd]
        g = np.array([1.0, 0, 0]) if gaze_head is None else _unit(gaze_head[sd] if isinstance(gaze_head, dict) else gaze_head)
        yaw = float(np.clip(math.atan2(g[1], max(g[0], 1e-6)), -math.radians(35), math.radians(35)))
        pit = float(np.clip(math.atan2(g[2], math.hypot(g[0], g[1])), -math.radians(25), math.radians(25)))
        g = np.array([math.cos(pit) * math.cos(yaw), math.cos(pit) * math.sin(yaw), math.sin(pit)])
        Rg = _axis_x(g)
        out[f"iris_{sd}"] = (C.copy(), _quat(Rg), None)
        apex = IRIS_APEX
        out[f"pupil_{sd}"] = (C + (apex - PUPIL_SEAT) * g, _quat(Rg), (PUPIL_SEAT + 5e-5, PUPIL_D / 2, PUPIL_D / 2))
        Ru, Rl = lid_turn(th_up, sd), lid_turn(-th_lo, sd)
        out[f"lid_up_{sd}"] = (C.copy(), _quat(Ru), None)
        out[f"lid_lo_{sd}"] = (C.copy(), _quat(Rl), None)
        pts = lash[idx] * np.array([1, sg, 1])
        P = [C + lid_turn(lash_th[j], sd) @ pts[j] for j in range(LASH_SEG + 1)]
        for j in range(LASH_SEG):
            a, b = P[j], P[j + 1]
            ax = b - a
            R = _axis_x(ax)
            out[f"lash{j}_{sd}"] = ((a + b) / 2, _quat(np.column_stack([R[:, 1], R[:, 2], R[:, 0]])), (LASH_R, float(np.linalg.norm(ax)) / 2, 0))
    return out


IRIS_APEX = math.sqrt(EYE_R ** 2 - (IRIS_D / 2) ** 2) - math.sqrt(.010 ** 2 - (IRIS_D / 2) ** 2) + .010 + .00015
PUPIL_SEAT = .00015             # the pupil's disc (a flat ellipsoid) seated on the iris's cap: its centre this far behind the cap's apex,
                                # its front 0.05 mm proud of it, its rim in front of the cap (which falls 0.2 mm at the pupil's edge)


# ---------------------------------------------------------------- her hair's cap: a shell over her head above the hairline
# (the film's dark brown bob; ours: the style). Its inner surface is her head sheet; its thickness HAIR_T over her crown, thinning
# to HAIR_EDGE at the hairline over HAIR_TAPER; the hairline at the trichion over the forehead (63 mm over the nasion [F]), rising at
# the temples, down in front of and behind the ears to the nape (ours). The bob's sides, the nape and the bun are the maker's.
HAIR_T, HAIR_EDGE, HAIR_TAPER = .009, .0006, .014
HAIRLINE = ((0, 0.0), (22, 7.0), (40, -18.0), (62, -45.0), (95, -58.0), (125, -85.0), (180, -115.0))   # (|theta| deg, z - tr mm)
HAIR_TOP = .276


def hairline_z(th):
    a, dz = zip(*HAIRLINE)
    return TR_Z + np.interp(np.degrees(np.abs(th)), a, dz) * MM


def hair_mesh(n_th=180, n_z=40):
    """the hair cap: vertices, faces (a (theta, z) grid from the hairline to HAIR_TOP, a fan closing the crown)"""
    ths = np.linspace(-math.pi, math.pi, n_th, endpoint=False)
    V = []
    for j in range(n_z + 1):
        s_ = (j / n_z) ** 1.25
        z0 = hairline_z(ths)
        z = z0 + (HAIR_TOP - z0) * s_
        R = _radius(ths, z / MM) * MM
        dist = (z - z0)
        t = HAIR_EDGE + (HAIR_T - HAIR_EDGE) * np.clip(dist / HAIR_TAPER, 0, 1) ** .7
        r = R + t
        V.append(np.stack([AX_X + r * np.cos(ths), r * np.sin(ths), z], 1))
    V = np.concatenate(V + [np.array([[AX_X, 0, HAIR_TOP + .010]])])
    F = []
    for j in range(n_z):
        for i in range(n_th):
            a_, b_ = j * n_th + i, j * n_th + (i + 1) % n_th
            F += [(a_, b_, b_ + n_th), (a_, b_ + n_th, a_ + n_th)]
    top = len(V) - 1
    for i in range(n_th):
        F.append((n_z * n_th + i, n_z * n_th + (i + 1) % n_th, top))
    F = np.array(F)
    fn = np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]])
    c = V[F].mean(1) - np.array([AX_X, 0, .19])
    if np.median((fn * c).sum(1)) < 0:
        F = F[:, ::-1]
    return V, F


def in_hair(P):
    """whether points (n x 3, m) are inside the hair cap's shell"""
    th = np.arctan2(P[:, 1], P[:, 0] - AX_X)
    r = np.hypot(P[:, 0] - AX_X, P[:, 1])
    z = np.clip(P[:, 2], .02, .28)
    z0 = hairline_z(th)
    R = _radius(th, z / MM) * MM
    t = HAIR_EDGE + (HAIR_T - HAIR_EDGE) * np.clip((P[:, 2] - z0) / HAIR_TAPER, 0, 1) ** .7
    return (P[:, 2] >= z0) & (P[:, 2] <= HAIR_TOP + .01) & (r <= R + t)


# ---------------------------------------------------------------- her collision: the convex hulls of her own sheet's pieces
# MuJoCo collides a mesh through its convex hull, so her head is split into pieces, each the hull of a patch of her sheet (its
# points and the same points 12 mm in along the normal), a patch split until its hull stands at most COLLIDE_GAP in front of her
# drawn skin (a hull bridges a patch's hollows), so a hand stops within that of her face; her crown and the back of her head are
# the cranium's ellipsoid (under her hair).
COLLIDE_GAP = .004              # (m; ours)
COLLIDE_MIN = (math.radians(5), .006)      # the smallest piece (angle about the axis, height)
COLLIDE_DEPTH = .012


def _hull_gap(P, Nn, Q=None, Qn=None):
    """how far the hull of a piece's points (and the same points COLLIDE_DEPTH in) stands in front of the drawn skin: over its own
    points and any other skin points (Q, their normals Qn) it swallows, the largest distance from a point out along its normal to the
    hull's surface"""
    from scipy.spatial import ConvexHull
    H = ConvexHull(np.concatenate([P, P - COLLIDE_DEPTH * Nn]))
    eq = H.equations                                  # n . x + d <= 0 inside
    pts, nrm = (P, Nn) if Q is None else (np.concatenate([P, Q]), np.concatenate([Nn, Qn]))
    depth = -(pts @ eq[:, :3].T + eq[:, 3])            # each point's depth inside each facet's plane
    inside = depth.min(1) > 1e-6
    nd = nrm @ eq[:, :3].T
    with np.errstate(divide="ignore", invalid="ignore"):
        t = np.where(nd > 1e-6, depth / nd, np.inf)
    exit_ = np.min(t, axis=1)
    exit_ = np.where(inside | (np.arange(len(pts)) < len(P)), exit_, 0.0)
    return float(np.max(np.where(np.isfinite(exit_), exit_, 0.0)))


def collision_pieces(V, N, HV=None, th_range=(-math.radians(115), math.radians(115)), z_range=(.042, .262)):
    """the sheet's points split into pieces (theta, z cells, halved until each hull stands at most COLLIDE_GAP in front of any skin
    it covers, its own or its neighbours'), the nose split the same way apart from the face, and the hair cap's outer surface in
    pieces: [points]"""
    from scipy.spatial import ConvexHull
    th = np.arctan2(V[:, 1], V[:, 0] - AX_X)
    nose = (nose_sdf(V) < .0015) & (V[:, 0] > .095)
    front = (V[:, 0] > -.02) & ~nose                      # the fissures' rims in: a piece's hull closes over the eye (the globe behind)
    pieces = []
    grids = ((front, th_range, z_range, 6, 5), (nose, (-math.radians(20), math.radians(20)), (SN[2] - .006, N_PT[2] + .006), 2, 3))
    for member, (ta, tb_), (za, zb_), n_th, n_z in grids:    # the face, and the nose apart (its hulls never swallow a cheek)
        stack = []
        tb = np.linspace(ta, tb_, n_th + 1); zb = np.linspace(za, zb_, n_z + 1)
        for i in range(n_th):
            for j in range(n_z):
                stack.append((tb[i], tb[i + 1], zb[j], zb[j + 1]))
        while stack:
            t0, t1, z0, z1 = stack.pop()
            mt, mz = .004 / .08, .001
            sel = member & (th >= t0 - mt) & (th <= t1 + mt) & (V[:, 2] >= z0 - mz) & (V[:, 2] <= z1 + mz)
            if sel.sum() < 8:
                continue
            lo_, hi_ = V[sel].min(0) - .012, V[sel].max(0) + .012
            near = ~sel & (V >= lo_).all(1) & (V <= hi_).all(1) & (V[:, 0] > -.02)
            gap = _hull_gap(V[sel], N[sel], V[near], N[near])
            if gap > COLLIDE_GAP and (t1 - t0 > COLLIDE_MIN[0] or z1 - z0 > COLLIDE_MIN[1]):
                if (t1 - t0) * .08 >= (z1 - z0):
                    tm = (t0 + t1) / 2; stack += [(t0, tm, z0, z1), (tm, t1, z0, z1)]
                else:
                    zm = (z0 + z1) / 2; stack += [(t0, t1, z0, zm), (t0, t1, zm, z1)]
                continue
            P = np.concatenate([V[sel], V[sel] - COLLIDE_DEPTH * N[sel]])
            pieces.append(P[ConvexHull(P).vertices])
    if HV is not None:                                   # the hair cap: six sectors about the axis
        hth = np.arctan2(HV[:, 1], HV[:, 0] - AX_X)
        for k in range(6):
            a0 = -math.pi + k * math.pi / 3 - .05; a1 = a0 + math.pi / 3 + .1
            sel = ((hth >= a0) & (hth <= a1)) | (HV[:, 2] > HAIR_TOP)
            P = np.concatenate([HV[sel], np.array([[AX_X, 0, .19]])])
            pieces.append(P[ConvexHull(P).vertices])
    pieces.sort(key=lambda p: (round(float(p[:, 2].mean()), 4), round(float(np.arctan2(p[:, 1].mean(), p[:, 0].mean() - AX_X)), 4)))
    return pieces


def lip_surface(step=.0008):
    """points on her lips' drawn surface at rest (their capsules' outside where it stands off the sheet) and their normals"""
    from scipy.spatial.transform import Rotation as Rot
    neutral = dict(smile=0.0, frown=0.0, cheek=0.0, brow_in=0.0, brow_out=0.0, brow_low=0.0, lid_up=0.0, lid_tight=0.0, jaw=0.0,
                   round=0.0, press=0.0, blink=0.0, tilt=0.0)
    g = _lips(0.0, 0.0, 0.0, 0.0, 0.0)
    P, Nn = [], []
    for n, (c, q, sz) in g.items():
        if not n.startswith(("mouth", "lip_lo")) or n == "mouth_open":
            continue
        R = Rot.from_quat([q[1], q[2], q[3], q[0]]).as_matrix()
        r, hl = sz[0], sz[1]
        for t in np.arange(-hl, hl + 1e-9, step):
            for ph in np.linspace(0, 2 * math.pi, 48, endpoint=False):
                nl = np.array([math.cos(ph), math.sin(ph), 0.0])
                P.append(c + R @ (np.array([0, 0, t]) + r * nl)); Nn.append(R @ nl)
    P, Nn = np.array(P), np.array(Nn)
    keep = head_sdf(P) > .0002
    return P[keep], Nn[keep]


# ---------------------------------------------------------------- building the assets (the maker's; deterministic)
def write_msh(path, V, F=None, Nrm=None, T=None):
    """MuJoCo's binary mesh: the vertices rounded to 1 um, the normals to 1e-5, texture coordinates to 1e-6 (so the build's last
    bits never change the file)"""
    V = np.round(np.asarray(V, float), 6).astype(np.float32)
    Nrm = None if Nrm is None else np.round(np.asarray(Nrm, float), 5).astype(np.float32)
    T = None if T is None else np.round(np.asarray(T, float), 6).astype(np.float32)
    F = np.zeros((0, 3), np.int32) if F is None else np.asarray(F, np.int32)
    with open(path, "wb") as fh:
        fh.write(np.array([len(V), 0 if Nrm is None else len(V), 0 if T is None else len(V), len(F)], np.int32).tobytes())
        fh.write(V.tobytes())
        if Nrm is not None:
            fh.write(Nrm.tobytes())
        if T is not None:
            fh.write(T.tobytes())
        fh.write(F.tobytes())


def _clip_back(V, T, N):
    """drop the sheet under her hair (the back of her head above the nape, the crown): the hair and the cranium's solid cover it"""
    c = V[T].mean(1)
    keep = ~(((c[:, 0] < -.020) & (c[:, 2] > .110)) | (c[:, 2] > .262))
    T = T[keep]
    used = np.unique(T)
    remap = -np.ones(len(V), np.int64); remap[used] = np.arange(len(used))
    return V[used], remap[T], N[used]


def largest_component(V, T):
    """the faces of the largest connected piece of a mesh (a closed bubble inside her head, where the nose and the face's base leave
    a slit, is dropped: it is never seen, and a collision piece or a shade computed from it would be wrong)"""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    E = np.concatenate([T[:, [0, 1]], T[:, [1, 2]], T[:, [2, 0]]])
    A = coo_matrix((np.ones(len(E)), (E[:, 0], E[:, 1])), shape=(len(V), len(V)))
    n, lab = connected_components(A, directed=False)
    keep = np.argmax(np.bincount(lab[T[:, 0]]))
    return T[lab[T[:, 0]] == keep], n


def build_sheet():
    """the head sheet (vertices, faces, normals) with the fissures cut, the back under the hair clipped, one connected surface"""
    V, Q = surface_nets(head_sdf, SHEET_BOX[0], SHEET_BOX[1], SHEET_H)
    V = project(head_sdf, V)
    T = triangulate(V, Q)
    T, _ = largest_component(V, T)
    N = sdf_grad(head_sdf, V); N /= np.linalg.norm(N, axis=1)[:, None]
    V, T, N, _ = cut_eyes(head_sdf, V, T, N)
    return _clip_back(V, T, N)


def front_grid():
    Y, Z = np.meshgrid(FRONT_Y, FRONT_Z)
    x = np.empty(Y.size)
    y, z = Y.reshape(-1), Z.reshape(-1)
    for s in range(0, len(y), 4000):
        x[s:s + 4000] = front_x(head_sdf, y[s:s + 4000], z[s:s + 4000], lo=.0, hi=.14, step=1e-3)
    return x.reshape(Y.shape)


def _blocked_exact(Q, hair):
    """what blocks the indirect light, evaluated on the surfaces themselves (for the eye's small parts, finer than the grid): the
    sheet's solid with the fissures open, the globes, her hair"""
    occ = head_sdf(Q) < 0
    C = EYE_C["L"]
    Qm = Q.copy(); Qm[:, 1] = np.abs(Q[:, 1])
    r = np.linalg.norm(Qm - C, axis=1)
    from matplotlib.path import Path as MPath
    inside = MPath(fissure_outline()).contains_points(Qm[:, 1:3])
    slot = inside & (Qm[:, 0] > C[0] - .004) & (r > EYE_R) & (r < .0175)
    occ = (occ & ~slot) | (r <= EYE_R) | in_hair(Q)
    for c, s_, R in hair:
        occ |= ((((Q - np.asarray(c)) @ np.asarray(R)) / np.asarray(s_)) ** 2).sum(1) <= 1
    return occ


def part_ao(hair=()):
    """the eye's parts' shade of the room's indirect light: the mean AO over points on their visible fronts at rest (the globe in the
    fissure, the upper lid's band under the rim, the caruncle), marched against the surfaces themselves"""
    C = EYE_C["L"]
    pts = {"eye": [], "lid": [], "caruncle": []}
    for u in np.linspace(.1, .9, 9):
        y, zu = margin_yz(u, "up"); _, zl = margin_yz(u, "lo")
        for z in np.linspace(zl + .0008, zu - .0008, 4):
            q = EYE_R ** 2 - (y - C[1]) ** 2 - (z - C[2]) ** 2
            if q > 0:
                pts["eye"].append(np.array([C[0] + math.sqrt(q), y, z]))
        y2, z2 = margin_yz(u, "up", AU5_UP)
        for zz in np.linspace(zu + .0003, z2 - .0003, 2):
            q = R_COV ** 2 - (y - C[1]) ** 2 - (zz - C[2]) ** 2
            if q > 0:
                pts["lid"].append(np.array([C[0] + math.sqrt(q), y, zz]))
    pts["caruncle"] = [EN + CARUNCLE[0] + np.array([CARUNCLE[1][0], 0, dz]) for dz in (-.0006, 0, .0006)]
    local = _hemisphere(AO_RAYS)
    steps = np.arange(.0002, AO_REACH, .0003)
    out = {}
    for k, P in pts.items():
        P = np.array(P)
        n = P - C; n /= np.linalg.norm(n, axis=1)[:, None]
        if k == "caruncle":
            n = np.tile([[1.0, 0, 0]], (len(P), 1))
        vis = []
        for p, nn in zip(P, n):
            t1 = _unit(np.cross(nn, [1.0, 0, 0] if abs(nn[0]) < .9 else [0, 1.0, 0])); t2 = np.cross(nn, t1)
            D = local[:, :1] * nn + local[:, 1:2] * t1 + local[:, 2:3] * t2
            Q = (p + .00005 * nn)[None, None, :] + D[:, None, :] * steps[None, :, None]
            hit = _blocked_exact(Q.reshape(-1, 3), hair).reshape(len(D), len(steps)).any(1)
            vis.append(1 - hit.mean())
        out[k] = float(np.mean(vis))
    return out


def build_assets(tex_dir, hair=(), room_indirect=1 / 3):
    """write assets/parent/*.msh, front.npy, rig.npz and tex_dir/parent_face_ao.png; returns the eye parts' AO and a digest"""
    from PIL import Image
    ASSETS.mkdir(parents=True, exist_ok=True)
    V, T, N = build_sheet()
    set_sheet(V, T)                                            # the lids keep behind the drawn sheet around the eye
    occ = Occupancy(hair)
    ao = smooth_on_mesh(ambient_occlusion(occ, V, N), T, 4)
    share = ao_share(room_indirect)
    V2, T2, N2, UV, src, over = split_charts(V, T, N)
    Image.fromarray(ao_texture(V2, T2, UV, ao[src], share, over)).save(Path(tex_dir) / "parent_face_ao.png")
    write_msh(ASSETS / "face.msh", V2, T2, N2, UV)
    lash = None
    for which in ("up", "lo"):
        lv, lf, row = lid_mesh(head_sdf, which)
        ln = vertex_normals(lv, lf)
        write_msh(ASSETS / f"lid_{which}_L.msh", lv, lf, ln)
        write_msh(ASSETS / f"lid_{which}_R.msh", lv * [1, -1, 1], lf[:, ::-1], ln * [1, -1, 1])
        if which == "up":
            t = np.linspace(-0.12, 1.12, len(row))
            lash = row[(t >= LASH_SPAN[0]) & (t <= LASH_SPAN[1])]
            up_ = _E_UP - (lash / np.linalg.norm(lash, axis=1)[:, None] @ _E_UP)[:, None] * lash / np.linalg.norm(lash, axis=1)[:, None]
            lash = (lash * (1 + (.0002 + LASH_R) / np.linalg.norm(lash, axis=1))[:, None]
                    + LASH_R * up_ / np.linalg.norm(up_, axis=1)[:, None])   # standing out of the lid's front over the rim's edge (the
            # line shows along the margin), its lower edge at the margin
    hv, hf = hair_mesh()
    write_msh(ASSETS / "hair.msh", hv, hf, vertex_normals(hv, hf))
    iv, if_ = iris_mesh()
    write_msh(ASSETS / "iris.msh", iv, if_, vertex_normals(iv, if_))
    np.save(ASSETS / "front.npy", np.round(front_grid(), 7).astype(np.float32))
    global _FRONT
    _FRONT = None
    brows = brow_meshes()
    for k, (bv, bf) in brows.items():
        bn = vertex_normals(bv, bf)
        write_msh(ASSETS / f"brow{k}_L.msh", bv, bf, bn)
        write_msh(ASSETS / f"brow{k}_R.msh", bv * [1, 1, -1], bf[:, ::-1], bn * [1, 1, -1])   # the mirror: the frame's across-axis flips
    pa = part_ao(hair)
    tops = {f"brow{k}": np.round(bv.reshape(-1, BROW_NS, 3)[1:-1].reshape(-1, 3), 7) for k, (bv, bf) in brows.items()}
    np.savez(ASSETS / "rig.npz", lash=np.round(lash, 7), ao_eye=round(pa["eye"], 5), ao_lid=round(pa["lid"], 5),
             ao_caruncle=round(pa["caruncle"], 5), **tops)
    # her lips at rest stand ~2 mm off the sheet: their surface joins the sheet's points for the collision pieces (they move a few mm
    # with her expressions; the pieces cover them at rest)
    lp, ln = lip_surface()
    pieces = collision_pieces(np.concatenate([V, lp]), np.concatenate([N, ln]), hv)
    for f_ in ASSETS.glob("collide_*.msh"):
        f_.unlink()
    for i, P in enumerate(pieces):
        write_msh(ASSETS / f"collide_{i:02d}.msh", P)
    return {"parts_ao": pa, "share": share, "overflow_faces": int(len(over)), "pieces": len(pieces), "sheet": (len(V), len(T)),
            "ao_sheet": (float(ao.min()), float(np.median(ao)))}


def collide_count():
    return len(sorted(ASSETS.glob("collide_*.msh")))


# ---------------------------------------------------------------- the maker's XML (the head segment's geoms and the assets)
FACE_GEOM_NAMES = ([f"mouth{i}" for i in range(LIP_UP_N)] + [f"lip_lo{i}" for i in range(LIP_LO_N)] + ["mouth_open", "teeth"]
                   + [n for sd in ("L", "R") for n in ([f"brow{i}_{sd}" for i in range(BROW_PIECES)] + [f"iris_{sd}", f"pupil_{sd}",
                      f"lid_up_{sd}", f"lid_lo_{sd}"] + [f"lash{j}_{sd}" for j in range(LASH_SEG)])])
MESH_OF = {"iris_L": "parent_iris", "iris_R": "parent_iris", "lid_up_L": "parent_lid_up_L", "lid_up_R": "parent_lid_up_R",
           "lid_lo_L": "parent_lid_lo_L", "lid_lo_R": "parent_lid_lo_R"}
MESH_OF.update({f"brow{k}_{sd}": f"parent_brow{k}_{sd}" for k in range(BROW_PIECES) for sd in "LR"})
FACE_MATERIAL = dict(mouth="lips", lip_lo="lips", mouth_open="mouth_in", teeth="teeth", brow="brow", iris="iris", pupil="pupil",
                     lid_up="lid", lid_lo="lid", lash="lash")
CARUNCLE = (np.array([-.0015, .0018, 0.0]), (.0014, .0019, .0016))      # its offset from en (toward the globe), semi-axes (ours):
                                                                        # its front 0.1 mm behind the corner, its medial end at en
FORNIX = (np.array([-.003, -.0015, 0.0]), (.010, .0145, .0115))         # the conjunctiva behind the globe: offset from its centre


def material_of(name):
    base = name.rstrip("0123456789")
    base = base[:-2] if base.endswith(("_L", "_R")) else base
    base = base.rstrip("0123456789")
    return FACE_MATERIAL[base]


def asset_xml(rel):
    """the meshes (rel: a path in ASSETS -> the path the XML gives, from the compiler's meshdir) and the sheet's texture"""
    m = [f'<mesh name="parent_face" file="{rel(ASSETS / "face.msh")}" inertia="shell"/>',
         f'<mesh name="parent_hair_cap" file="{rel(ASSETS / "hair.msh")}" inertia="shell"/>',
         f'<mesh name="parent_iris" file="{rel(ASSETS / "iris.msh")}" inertia="shell"/>']
    m += [f'<mesh name="parent_lid_{w}_{sd}" file="{rel(ASSETS / f"lid_{w}_{sd}.msh")}" inertia="shell"/>' for w in ("up", "lo") for sd in "LR"]
    m += [f'<mesh name="parent_brow{k}_{sd}" file="{rel(ASSETS / f"brow{k}_{sd}.msh")}" inertia="shell"/>' for k in range(BROW_PIECES) for sd in "LR"]
    m += [f'<mesh name="parent_collide_{i:02d}" file="{rel(ASSETS / f"collide_{i:02d}.msh")}"/>' for i in range(collide_count())]
    m.append('<texture name="parent_face_ao" type="2d" file="parent_face_ao.png"/>')
    return m


def static_xml(decor, collide):
    """the head segment's still parts: the sheet, the globes, the caruncles, the conjunctivae, the collision pieces"""
    f = lambda *v: " ".join(f"{x:.6g}" for x in v)
    g = [f'<geom name="parent_face" type="mesh" mesh="parent_face" material="skin_face" {decor}/>',
         f'<geom name="parent_hair_cap" type="mesh" mesh="parent_hair_cap" material="hair" {decor}/>']
    for sd, sg in (("L", 1), ("R", -1)):
        C = EYE_C[sd]
        en = EN * [1, sg, 1]
        g.append(f'<geom name="parent_sclera_{sd}" type="sphere" size="{EYE_R}" pos="{f(*C)}" material="sclera" {decor}/>')
        g.append(f'<geom name="parent_caruncle_{sd}" type="ellipsoid" size="{f(*CARUNCLE[1])}" pos="{f(*(en + CARUNCLE[0] * [1, sg, 1]))}" '
                 f'material="caruncle" {decor}/>')
        g.append(f'<geom name="parent_fornix_{sd}" type="ellipsoid" size="{f(*FORNIX[1])}" pos="{f(*(C + FORNIX[0] * [1, sg, 1]))}" '
                 f'material="fornix" {decor}/>')
    for i in range(collide_count()):
        g.append(f'<geom name="parent_face_c{i:02d}" type="mesh" mesh="parent_collide_{i:02d}" rgba="0 0 0 0" group="3" {collide}/>')
    for sd in "LR":                                   # her eyes: the globes, their lids and lashes (a finger stops at the lashes' front)
        g.append(f'<geom name="parent_eye_c{sd}" type="sphere" size="{R_COV + .0002 + 2 * LASH_R:.6g}" pos="{f(*EYE_C[sd])}" rgba="0 0 0 0" group="3" '
                 f'{collide}/>')
    return g


def moving_xml(decor, neutral):
    """her face's moving parts at their neutral pose (the scene's set_parent moves them)"""
    f = lambda *v: " ".join(f"{x:.6g}" for x in v)
    g = []
    for n, (p, q, sz) in face_geoms(neutral).items():
        if n in MESH_OF:
            g.append(f'<geom name="parent_{n}" type="mesh" mesh="{MESH_OF[n]}" pos="{f(*p)}" quat="{f(*q)}" material="{material_of(n)}" {decor}/>')
        elif n.startswith(("lash", "mouth", "lip_lo")) and n != "mouth_open":
            g.append(f'<geom name="parent_{n}" type="capsule" pos="{f(*p)}" quat="{f(*q)}" size="{f(sz[0], sz[1])}" material="{material_of(n)}" {decor}/>')
        else:
            g.append(f'<geom name="parent_{n}" type="ellipsoid" pos="{f(*p)}" quat="{f(*q)}" size="{f(*sz)}" material="{material_of(n)}" {decor}/>')
    return g
