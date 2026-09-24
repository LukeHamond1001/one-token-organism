"""HER FACE MEASURED ON THE DRAWN GEOMETRY (docs/SIM_DESIGN.md 4.1; the W1 verifier's fourth round: the third round's report gave
the norms as built, but its eye opening was 22.0 mm wide where the norm is 30.7 and its inner eye corners 41.6 mm apart where it is
31.8, because only the declared constants were checked). An instrument, never the body; it writes nothing.

Everything here is read off what is drawn, by rays and by the drawn meshes' own vertices, in her head's frame, her face at rest:
  THE FISSURES  the drawn lid margins' corners (the sheet's rim on the opening's outline: where calipers go), and the opening as seen:
                rays from the front, from 20 and 40 deg
                to either side and 60 and 75 deg to the temple's side, and 15 deg above and below: every hit on an eye's contents
                (the globe, the iris, the pupil, the caruncle, the conjunctiva) is a point of the opening; the canthi are its extreme
                points along the canthal line (en medial, ex lateral), the fissure's height its vertical extent at the pupil (front)
  THE IRIS      the frontal extent of the iris's hits (the gaze ahead); THE PUPILS their drawn centres (IPD)
  THE MOUTH     the frontal extent of the lips' hits (ch-ch), the upper and lower vermilion's heights at the midline (ls-st, li-st)
  THE BROWS     their hits' lower margin over en, over the lid margin at the pupil and over ex, and their top over the pupil
  THE PROFILE   the midline's frontmost surface (the sheet): the nasion (the deepest point between the glabella and the nose), the
                pronasale (the nose's frontmost), the subnasale (the deepest point between the tip and the lip), the gnathion (the
                chin's lowest point in front); the trichion where her hair's cap meets the midline
  THE NOSE      al-al: the nose's widest drawn points at the alae
  THE HEAD      the sheet's widest point (eu-eu), its width at the zygion's level and at the eyes, the jaw's at the gonion's level
  THE RELIEF    Yaremchuk's depths: the brow's soft tissue, the cheek's prominence and the infraorbital rim against the cornea's apex
and each against its norm (parent_face.py's sources). COLLISION (--collision): rays at her face from the front and from 30-60 deg
to the sides, above and below (a hand coming at her): the first visible surface against the first of her collision shapes, how far
a hand would stop before her drawn face (+) or pass into it (-), by region. Run: nice -n 19 python3 tools/sim_face_measure.py [--collision]  (JSON on stdout)"""
import argparse
import json
import math
import os
import sys

import mujoco
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "body", "sim"))
import g1scene as G  # noqa: E402

kin = G.kin
F = kin.face
VISIBLE = np.array([1, 1, 1, 0, 0, 0], np.uint8)
COLLIDE = np.array([0, 0, 0, 1, 0, 0], np.uint8)
EYE_PARTS = ("sclera_", "iris_", "pupil_", "caruncle_", "fornix_")


class Face:
    """her head posed at birth (far from the G1), rays in her head's frame"""

    def __init__(self, scene=None):
        self.sc = scene or G.Scene()
        self.sc.birth()
        m, d = self.m, self.d = self.sc.m, self.sc.d
        mujoco.mj_forward(m, d)
        hb = m.body("parent_head").id
        self.p0, self.R = d.xpos[hb].copy(), d.xmat[hb].reshape(3, 3).copy()
        self.name = {g: (m.geom(g).name or "") for g in range(m.ngeom)}

    def ray(self, o_local, d_local, groups=VISIBLE):
        """(distance, geom name, hit point in the head frame) of a ray given in the head frame"""
        o = self.p0 + self.R @ np.asarray(o_local, float)
        dv = self.R @ (np.asarray(d_local, float) / np.linalg.norm(d_local))
        gid = np.array([-1], np.int32)
        dist = mujoco.mj_ray(self.m, self.d, o, dv, groups, 1, -1, gid)
        if gid[0] < 0:
            return None, "", None
        return dist, self.name[int(gid[0])], np.asarray(o_local, float) + dist * np.asarray(d_local, float) / np.linalg.norm(d_local)

    def is_eye(self, n, sd):
        return n.startswith(tuple("parent_" + p for p in EYE_PARTS)) and n.endswith(sd)

    def fissure_points(self, sd, step=2.5e-4):
        """points of an eye's opening seen from 7 directions: {view: [(y, z) of the ray's aim, hit point]}"""
        sg = 1 if sd == "L" else -1
        C = kin.EYE_C[sd]
        pts = []
        views = [(0, 0), (20, 0), (40, 0), (60, 0), (75, 0), (-20, 0), (-40, 0), (0, 15), (0, -15)]
        for yaw, pit in views:
            dvec = -np.array([math.cos(math.radians(pit)) * math.cos(math.radians(yaw)),
                              sg * math.cos(math.radians(pit)) * math.sin(math.radians(yaw)), math.sin(math.radians(pit))])
            u = np.cross(dvec, [0, 0, 1.0]); u /= np.linalg.norm(u); v = np.cross(u, dvec)
            for a in np.arange(-.020, .020, step):
                for b in np.arange(-.009, .009, step):
                    target = C + np.array([.012, 0, 0]) + a * u + b * v
                    dist, n, hp = self.ray(target - dvec * .1, dvec)
                    if n and self.is_eye(n, sd):
                        pts.append(hp)
        return np.array(pts)

    def frontal_extent(self, prefixes, y_range, z_range, step=2.5e-4):
        """the frontal (y, z) of rays from the front that hit geoms named with these prefixes"""
        out = []
        for y in np.arange(*y_range, step):
            for z in np.arange(*z_range, step):
                _, n, hp = self.ray([.25, y, z], [-1, 0, 0])
                if n.startswith(prefixes):
                    out.append((y, z, hp[0]))
        return np.array(out)

    def profile(self, z0=.040, z1=.250, step=2.5e-4):
        """the midline's frontmost drawn surface: (z, x) of the sheet or the lips or the hair at y = 0"""
        out = []
        for z in np.arange(z0, z1, step):
            _, n, hp = self.ray([.25, 0, z], [-1, 0, 0])
            out.append((z, hp[0] if hp is not None else np.nan, n))
        return out


def measure():
    f = Face()
    out = {}
    # the fissures
    can = {}
    for sd, sg in (("L", 1), ("R", -1)):
        P = f.fissure_points(sd)
        axis = F.EX - F.EN; axis = axis / np.linalg.norm(axis)
        axis = axis * np.array([1, sg, 1])
        t = P @ axis
        en, ex = P[np.argmin(t)], P[np.argmax(t)]
        front = f.frontal_extent(("parent_sclera_" + sd, "parent_iris_" + sd, "parent_pupil_" + sd),
                                 (sg * kin.EYE_C["L"][1] - .0006, sg * kin.EYE_C["L"][1] + .0006), (.150, .172))
        can[sd] = (en, ex)
        out[f"ex-en {sd}"] = (float(np.linalg.norm(ex - en)), F.EX_EN, .0006, "[F] ex-en")
        out[f"ps-pi {sd}"] = (float(np.ptp(front[:, 1])) if len(front) else 0.0, F.PS_PI, .0005, "[F] ps-pi, at the pupil")
    out["en-en"] = (float(np.linalg.norm(can["L"][0] - can["R"][0])), F.EN_EN, .0006, "[F] en-en")
    out["ex-ex"] = (float(np.linalg.norm(can["L"][1] - can["R"][1])), F.EX_EX, .0008, "[F] ex-ex")
    # the drawn canthi (where calipers go): the corners of the drawn lid margins, the sheet's rim around each opening (its vertices on
    # the opening's frontal outline, within 0.1 mm), their extreme points along the canthal line
    raw = np.fromfile(F.ASSETS / "face.msh", dtype=np.int32, count=4)
    Vs = np.fromfile(F.ASSETS / "face.msh", dtype=np.float32)[4:4 + 3 * raw[0]].reshape(-1, 3).astype(float)
    poly = F.fissure_outline(2000)
    dc = {}
    for sd, sg in (("L", 1), ("R", -1)):
        cand = Vs[(sg * Vs[:, 1] > 0) & (Vs[:, 0] > .07) & (np.abs(Vs[:, 2] - kin.PUPIL_Z) < .010)]
        yz = np.stack([sg * cand[:, 1], cand[:, 2]], 1)
        dist = np.min(np.linalg.norm(yz[:, None, :] - poly[None, :, :], axis=2), axis=1)
        rim = cand[dist < 1e-4]
        # the rim's front edge: where the rim rolls back into the socket, only its frontmost vertex at each place is the margin
        keyyz = np.round(rim[:, 1:3] / 5e-5).astype(np.int64)
        order = np.lexsort((-rim[:, 0], keyyz[:, 1], keyyz[:, 0]))
        first = np.ones(len(order), bool); first[1:] = (np.diff(keyyz[order], axis=0) != 0).any(1)
        rim = rim[order[first]]
        axis = (F.EX - F.EN) / np.linalg.norm(F.EX - F.EN) * np.array([1, sg, 1])
        t = rim @ axis
        dc[sd] = (rim[np.argmin(t)], rim[np.argmax(t)]) if len(rim) else (np.zeros(3), np.zeros(3))
        out[f"ex-en {sd} (drawn margins)"] = (float(np.linalg.norm(dc[sd][1] - dc[sd][0])), F.EX_EN, .0004, "[F] the drawn lid margins' corners")
    out["en-en (drawn margins)"] = (float(np.linalg.norm(dc["L"][0] - dc["R"][0])), F.EN_EN, .0004, "[F] the drawn lid margins' corners")
    out["ex-ex (drawn margins)"] = (float(np.linalg.norm(dc["L"][1] - dc["R"][1])), F.EX_EX, .0004, "[F] the drawn lid margins' corners")
    # the pupils and the iris
    m = f.m
    pc = {sd: f.R.T @ (f.d.geom_xpos[m.geom(f"parent_pupil_{sd}").id] - f.p0) for sd in "LR"}
    out["pupils apart"] = (float(np.linalg.norm(pc["L"] - pc["R"])), F.IPD, .0003, "[D]")
    ir = f.frontal_extent(("parent_iris_L",), (kin.EYE_C["L"][1] - .0068, kin.EYE_C["L"][1] + .0068), (kin.PUPIL_Z - .0002, kin.PUPIL_Z + .0002), 1e-4)
    out["iris across"] = (float(np.ptp(ir[:, 0])) + 1e-4 if len(ir) else 0.0, F.IRIS_D, .0003, "[Ru], the visible iris")
    # the mouth
    lips = f.frontal_extent(("parent_mouth", "parent_lip_lo"), (-.030, .030), (.088, .112), 2.5e-4)
    lips = lips[~np.array([False] * len(lips))]
    out["ch-ch"] = (float(np.ptp(lips[:, 0])), F.CH_CH, .0010, "[F] the mouth's width")
    up = f.frontal_extent(("parent_mouth",), (-.0002, .0002), (.095, .115), 1e-4)
    lo = f.frontal_extent(("parent_lip_lo",), (-.0002, .0002), (.085, .105), 1e-4)
    out["ls-st"] = (float(np.ptp(up[:, 1])) + 1e-4, F.LS_ST, .0006, "[F] the upper vermilion at the midline")
    out["li-st"] = (float(np.ptp(lo[:, 1])) + 1e-4, F.ST_LI, .0006, "[F] the lower vermilion at the midline")
    # the brows (the left)
    def brow_col(y):
        b = f.frontal_extent(("parent_brow",), (y - .0002, y + .0002), (.165, .195), 1e-4)
        return (float(b[:, 1].min()), float(b[:, 1].max())) if len(b) else (np.nan, np.nan)
    C = kin.EYE_C["L"]
    up_margin = float(F.margin_yz((C[1] - F.EN[1]) / (F.EX[1] - F.EN[1]), "up")[1])
    out["brow low over en"] = (brow_col(F.EN[1])[0] - can["L"][0][2], F.BROW_OVER_EN, .0010, "[G]")
    out["brow low over the lid at the pupil"] = (brow_col(C[1])[0] - up_margin, F.BROW_OVER_LID[1][1], .0010, "[G]")
    out["brow low over ex"] = (brow_col(F.EX[1])[0] - can["L"][1][2], F.BROW_OVER_EX, .0010, "[G]")
    out["brow top over the pupil"] = (brow_col(C[1])[1] - kin.PUPIL_Z, F.BROW_TOP_OVER_PUPIL, .0010, "[M]")
    # the profile
    prof = f.profile()
    z = np.array([p[0] for p in prof]); x = np.array([p[1] for p in prof]); nm = [p[2] for p in prof]
    sheet = np.array([n == "parent_face" for n in nm])
    def arg_in(lo_, hi_, fn, mask=sheet):
        sel = np.nonzero((z >= lo_) & (z <= hi_) & mask & np.isfinite(x))[0]
        return sel[fn(x[sel])]
    i_pn = arg_in(.115, .150, np.argmax)
    P2 = lambda i: np.array([x[i], z[i]])

    def deepest(i_a, i_b):
        """the profile's point between two others furthest behind the line joining them (a concavity's deepest point, the standard
        geometric reading of a soft-tissue landmark in a hollow)"""
        lo_, hi_ = sorted((i_a, i_b))
        idx = np.arange(lo_ + 1, hi_)
        idx = idx[np.isfinite(x[idx])]
        a, b = P2(i_a), P2(i_b)
        t = b - a; t = t / np.linalg.norm(t)
        dist = np.array([(P2(i) - a)[0] * t[1] - (P2(i) - a)[1] * t[0] for i in idx])
        return idx[np.argmax(np.abs(dist))]
    i_g = arg_in(.176, .196, np.argmax)                              # the glabella: the forehead's frontmost between the brows
    i_n = arg_in(z[i_g] - .025, z[i_g] - .004, np.argmin)           # the nasion: the frontonasal hollow's hindmost point
    i_ls = arg_in(z[i_pn] - .032, z[i_pn] - .014, np.argmax, np.isfinite(x))   # the upper lip's frontmost (labrale superius)
    i_sn = deepest(i_ls, i_pn)                                       # the subnasale: the hollow between the lip and the tip
    i_pog = arg_in(.060, .078, np.argmax)                            # the chin's frontmost (pogonion)
    i_c = arg_in(.040, .052, np.argmin)                              # where the chin's underside meets the neck
    # the gnathion: the chin's corner, the profile's point between the pogonion and the neck furthest out of the line joining them
    lo_, hi_ = sorted((i_c, i_pog))
    idx = np.arange(lo_ + 1, hi_)
    a, b = P2(i_pog), P2(i_c); t = (b - a) / np.linalg.norm(b - a)
    i_gn = idx[np.argmin([(P2(i) - a)[0] * t[1] - (P2(i) - a)[1] * t[0] for i in idx])]   # the convex side
    out["n-pn"] = (float(np.linalg.norm(P2(i_pn) - P2(i_n))), F.N_PN, .0015, "[F] the nose's length")
    out["n-sn"] = (float(np.linalg.norm(P2(i_sn) - P2(i_n))), F.N_SN, .0015, "[F] the nose's height")
    out["sn-pn"] = (float(np.linalg.norm(P2(i_pn) - P2(i_sn))), F.SN_PN, .0015, "[F] the tip's protrusion")
    out["n-gn"] = (float(np.linalg.norm(P2(i_gn) - P2(i_n))), F.N_GN, .0026, "[F] the face's height (0.5 SD)")
    hair = np.nonzero(np.array([n.startswith("parent_hair") or n == "parent_fringe" for n in nm]) & (z > .200))[0]
    i_tr = hair[np.argmin(z[hair])] if len(hair) else None
    if i_tr is not None:
        out["tr-n"] = (float(z[i_tr] - z[i_n]), F.TR_N, .0030, "[F] the forehead's height (0.5 SD): where her hair meets the midline")
    # the nose's width and the head (the sheet's own vertices)
    raw = np.fromfile(F.ASSETS / "face.msh", dtype=np.int32, count=4)
    V = np.fromfile(F.ASSETS / "face.msh", dtype=np.float32)[4:4 + 3 * raw[0]].reshape(-1, 3).astype(float)
    nose = (F.nose_sdf(V) < .0004) & (V[:, 0] > .095) & (np.abs(V[:, 2] - (F.SN[2] + .005)) < .006)
    out["al-al"] = (float(2 * np.abs(V[nose, 1]).max()), F.AL_AL, .0012, "[F] the nose's width")
    band = lambda z0, z1, xmin=-1.0, xmax=1.0: (V[:, 2] >= z0) & (V[:, 2] <= z1) & (V[:, 0] >= xmin) & (V[:, 0] <= xmax)
    out["eu-eu"] = (float(2 * np.abs(V[:, 1]).max()), F.HEAD_BREADTH, .0029, "[P] the head's breadth (0.5 SD)")
    out["zy-zy"] = (float(2 * np.abs(V[band(.136, .150, .020), 1]).max()), F.ZY_ZY, .0023, "[F] the face's width at the zygia, in front of the ears (0.5 SD)")
    out["go-go"] = (float(2 * np.abs(V[band(.072, .084, .005, .030), 1]).max()), F.GO_GO, .0030, "[F] the jaw's width at the gonia (0.5 SD)")
    out["head width at the eyes"] = (float(2 * np.abs(V[band(.156, .166), 1]).max()), F.HEAD_BREADTH, None, "no wider than eu-eu")
    apex = f.ray([.25, C[1], kin.PUPIL_Z], [-1, 0, 0])[2][0]
    fx = lambda y, zz: f.ray([.25, y, zz], [-1, 0, 0])[2][0]
    out["brow ahead of the cornea"] = (fx(C[1], .181) - apex, F.BROW_AHEAD, .0025, "[Y]")
    cheek = max(fx(y, zz) for y in np.arange(.030, .050, .002) for zz in np.arange(.128, .146, .002))
    out["cheek ahead of the cornea"] = (cheek - apex, F.CHEEK_AHEAD, .0025, "[Y]")
    out["infraorbital rim behind the cornea"] = (apex - fx(C[1], .146), F.INFRAORB_BEHIND, .0025, "[Y]")
    return out


REGIONS = {"forehead": ((.185, .240), (0, .055)), "brows and eyes": ((.148, .185), (0, .055)), "nose": ((.112, .172), (0, .018)),
           "cheeks": ((.110, .150), (.018, .060)), "mouth": ((.086, .114), (0, .030)), "chin and jaw": ((.050, .086), (0, .050))}
APPROACH = [(0, 0)] + [(a, e) for a in (-35, 0, 35) for e in (-30, 30)] + [(-35, 0), (35, 0), (-60, 0), (60, 0)]   # (azimuth, elevation) deg


def collision(step=.0015):
    """a hand coming at her face: rays from the front and from 30-60 deg to the sides, above and below, on a 1.5 mm grid over her face;
    where a ray meets her drawn face, the first visible surface's distance minus the first of her collision shapes' (mm; + the
    hand stops that far before her drawn face, - it passes that far into it), by region (where on her face the ray lands)"""
    f = Face()
    for g, n in f.name.items():                    # mj_ray skips invisible geoms: her collision shapes made opaque (the instrument's copy)
        if f.m.geom_group[g] == 3 and n.startswith("parent_"):
            f.m.geom_rgba[g, 3] = 1.0
    rows = []
    for az, el in APPROACH:
        d = -np.array([math.cos(math.radians(el)) * math.cos(math.radians(az)), math.cos(math.radians(el)) * math.sin(math.radians(az)),
                       math.sin(math.radians(el))])
        u = np.cross(d, [0, 0, 1.0]); u /= np.linalg.norm(u); v = np.cross(u, d)
        c = np.array([.09, 0, .15])
        for a in np.arange(-.075, .075, step):
            for b in np.arange(-.105, .095, step):
                o = c + a * u + b * v - d * .25
                A, na, hp = f.ray(o, d, VISIBLE)
                if A is None or not (na == "parent_face" or na.startswith(("parent_mouth", "parent_lip_lo", "parent_brow", "parent_lid",
                                                                          "parent_sclera", "parent_iris", "parent_pupil", "parent_lash",
                                                                          "parent_caruncle"))):
                    continue
                if not (.050 < hp[2] < .240 and hp[0] > .045):
                    continue
                B, nb, _ = f.ray(o, d, COLLIDE)
                if B is None or not nb.startswith("parent_"):
                    continue
                rows.append((hp, 1e3 * (A - B)))
    out = {}
    for reg, ((z0, z1), (y0, y1)) in REGIONS.items():
        g = np.array([r[1] for r in rows if z0 <= r[0][2] < z1 and y0 <= abs(r[0][1]) < y1])
        if len(g):
            out[reg] = {"n": int(len(g)), "min": round(float(g.min()), 2), "median": round(float(np.median(g)), 2),
                        "p95": round(float(np.percentile(g, 95)), 2), "max": round(float(g.max()), 2)}
    g = np.array([r[1] for r in rows])
    out["all"] = {"n": int(len(g)), "min": round(float(g.min()), 2), "median": round(float(np.median(g)), 2),
                  "p95": round(float(np.percentile(g, 95)), 2), "p99": round(float(np.percentile(g, 99)), 2),
                  "max": round(float(g.max()), 2), "into her face (< -0.5)": int((g < -.5).sum())}
    return out


LID_STATES = {"rest": {}, "blink": dict(blink=1.0), "half blink": dict(blink=.5), "AU5": dict(lid_up=1.0), "lowered": dict(lid_up=-1.0),
              "AU6": dict(cheek=1.0), "AU7": dict(lid_tight=1.0), "AU6 blink": dict(cheek=1.0, blink=1.0), "AU7 blink": dict(lid_tight=1.0, blink=1.0)}
EYE_SEEN = ("parent_sclera_L", "parent_iris_L", "parent_pupil_L")
LEAK_DEPTH = .00015             # a lid standing less than this out of her skin (along the line from the globe's centre) is level with
                                # it (where her skin meets the globe at the lateral corner the lids, just outside the globe, meet it
                                # too): half a pixel of the portrait camera at 0.52 m


def lids(step=5e-4, gazes=((0, 0), (15, 0), (-15, 0), (0, 12), (0, -12))):
    """the moving lids against her skin and her eye, in LID_STATES: rays at the left eye from the front and from 35 deg to either side
    on a step grid. THROUGH THE SKIN: a ray meets a lid or a lash first and, the lids and lashes taken away, her drawn sheet next,
    within 3 mm behind, and the lid's point stands more than LEAK_DEPTH out of her sheet along the line from the globe's centre,
    where the sheet is farther than parent_face.RIM_BAND from the lid margins (so the lid lies over her skin, not in the opening). THE EYE
    IN A BLINK: in the blinks, at the gazes given (yaw, pitch deg), a ray meets the globe, the iris or the pupil first. Returns
    {state: {"through skin": n, "eye seen": n, "rays": n, "where": [first few (y, z) mm of the skin leaks]}}"""
    from scipy.spatial import cKDTree
    f = Face()
    m = f.m
    rim = cKDTree(F.rim_curves())
    sheet = m.geom("parent_face").id
    C = kin.EYE_C["L"]
    hide = [m.material(n).id for n in ("lid", "lash")]
    alpha = m.mat_rgba[hide, 3].copy()
    out = {}
    for name, kw in LID_STATES.items():
        blink = kw.get("blink", 0) >= 1
        res = {"through skin": 0, "eye seen": 0, "eye seen from the front": 0, "eye seen from 35 deg": 0, "gazes": 0, "rays": 0, "where": []}
        for gz in (gazes if blink else gazes[:1]):
            res["gazes"] += 1
            pz = G.born_parent(); pz.expr = kin.face_params(**kw)
            if gz != (0, 0):                            # her eyes on a point 1 m along the gaze from her left eye
                yw, pt = math.radians(gz[0]), math.radians(gz[1])
                hp_, hR_ = kin.fk(pz)["head"]
                pz.gaze = hp_ + hR_ @ (kin.EYE_C["L"] + np.array([math.cos(pt) * math.cos(yw), math.cos(pt) * math.sin(yw), math.sin(pt)]))
            f.sc.set_parent(pz)
            mujoco.mj_forward(m, f.d)
            for az in (0, 35, -35):
                dv = -np.array([math.cos(math.radians(az)), math.sin(math.radians(az)), 0.0])
                for y in np.arange(.008, .052, step):
                    for z in np.arange(.148, .176, step):
                        o = np.array([.095, y, z]) - dv * .2
                        dist, n, hp = f.ray(o, dv)
                        if not n:
                            continue
                        res["rays"] += 1
                        if blink and n.startswith(EYE_SEEN):
                            res["eye seen"] += 1
                            res["eye seen from the front" if az == 0 else "eye seen from 35 deg"] += 1
                            res.setdefault("eye where", [])
                            if len(res["eye where"]) < 400:
                                res["eye where"].append([round(1e3 * hp[1], 1), round(1e3 * hp[2], 1), n, az, gz])
                        if not n.startswith(("parent_lid_up_L", "parent_lid_lo_L", "parent_lash")) or not n.endswith("_L"):
                            continue
                        m.mat_rgba[hide, 3] = 0
                        d2, n2, h2 = f.ray(o, dv)
                        m.mat_rgba[hide, 3] = alpha
                        if n2 != "parent_face" or d2 - dist > .003:
                            continue                    # behind the lid: the eye or the opening, not her skin
                        # how far the lid stands out of her skin: along the line from the globe's centre through the lid's point
                        u_ = (hp - C) / np.linalg.norm(hp - C)
                        rs = mujoco.mj_rayMesh(m, f.d, sheet, f.p0 + f.R @ C, f.R @ u_)
                        if rs < 0:
                            continue
                        depth = np.linalg.norm(hp - C) - rs
                        band = F.RIM_BAND + (2 * F.LASH_R if n.startswith("parent_lash") else 0.0)   # lashes stand out of the margin
                        if depth > LEAK_DEPTH and rim.query(C + rs * u_)[0] > band:
                            res["through skin"] += 1
                            res["deepest"] = max(res.get("deepest", 0.0), round(1e3 * depth, 3))
                            if len(res["where"]) < 6:
                                res["where"].append([round(1e3 * h2[1], 1), round(1e3 * h2[2], 1), n, az])
        out[name] = res
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--collision", action="store_true")
    a = ap.parse_args()
    res = {k: {"measured_mm": round(v[0] * 1e3, 2), "norm_mm": round(v[1] * 1e3, 2),
               "tol_mm": None if v[2] is None else round(v[2] * 1e3, 2), "source": v[3]} for k, v in measure().items()}
    if a.collision:
        res["collision"] = collision()
    print(json.dumps(res, indent=1))
