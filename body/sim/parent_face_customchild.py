"""THE CUSTOM CHILD'S PARENT FACE (kept for reference with body/sim/*_customchild.*; docs/SIM_DESIGN.md 15): the prototype's
cartoon face, its scalar expression path exactly as the custom child's room (livingroom_customchild.xml) was built with it. The G1
room's parent has the face of human proportions in parent_kin.py (the W1 verifier's third round, 2026-09-24); this copy keeps the
reference scene loading as it was."""
import math

import numpy as np

from parent_kin import frame, mjquat, unit

EYE_C = {"L": np.array([.098, .036, .172]), "R": np.array([.098, -.036, .172])}   # eyeball centres in the head frame

FACE_MOUTH_N = 7                # the mouth: 6 capsules through 7 points
MOUTH_HALF_W, MOUTH_Z, MOUTH_CURVE = .026, .118, .0075


def head_surface_x(y, z):
    """The front surface of the parent's head ellipsoid (centre (.015, 0, .16), semi-axes .098 .082 .106)."""
    q = 1 - (y / .084) ** 2 - ((z - .16) / .106) ** 2
    return .015 + .098 * math.sqrt(max(q, 0))


def face_geoms(expr, gaze_head=None, blink=0.0):
    """Local (pos, quat, size or None) of the parent's face geoms for an expression in [-1, 1] and a gaze direction
    given in the head frame (None: straight ahead). Names match the XML (parent_<name>).
    (The prototype's scalar face, as the custom child's room was built with it.)"""
    out = {}
    e = float(np.clip(expr, -1, 1))
    # mouth: a parabola whose corners rise for a smile and fall for a frown; the centre dips a little for a smile
    ys = np.linspace(-MOUTH_HALF_W * (1 + .12 * max(e, 0)), MOUTH_HALF_W * (1 + .12 * max(e, 0)), FACE_MOUTH_N)
    pts = []
    for y in ys:
        u = (y / MOUTH_HALF_W) ** 2
        z = MOUTH_Z + e * MOUTH_CURVE * u - (.004 * e if e > 0 else .002 * e) * (1 - u)
        pts.append(np.array([head_surface_x(y, z) + .0015, y, z]))
    for i in range(FACE_MOUTH_N - 1):
        a, b = pts[i], pts[i + 1]
        c = (a + b) / 2
        out[f"mouth{i}"] = (c, mjquat(frame([1, 0, 0], b - a)), (.0038, np.linalg.norm(b - a) / 2, 0))
    # open smile: a dark mouth opening and teeth that grow with the smile
    op = max(0.0, e - .25) / .75
    oc = np.array([head_surface_x(0, MOUTH_Z + .001) - .0012, 0, MOUTH_Z + .0012 + .0025 * op])
    out["mouth_open"] = (oc, mjquat(np.eye(3)), (.004, .001 + .022 * op, .001 + .0068 * op))
    out["teeth"] = (oc + [.001, 0, .0034 * op], mjquat(np.eye(3)), (.003, .001 + .016 * op, .001 + .0024 * op))
    # brows: raised and arched for a smile, lowered and drawn in (inner ends down) for a frown
    for sd, sg in (("L", 1), ("R", -1)):
        yb, zb = sg * .037, .199 + .004 * max(e, 0) - .004 * max(-e, 0)
        tilt = math.radians(8 * max(e, 0) - 16 * max(-e, 0))     # + lifts the inner end (a warm, open look); - draws it down
        a = np.array([0, yb - sg * .014, zb + math.sin(tilt) * .014])     # inner end
        b = np.array([0, yb + sg * .014, zb - math.sin(tilt) * .014 - .002])  # outer end, a slight arch
        a[0], b[0] = head_surface_x(a[1], a[2]) + .002, head_surface_x(b[1], b[2]) + .002
        c = (a + b) / 2
        out[f"brow_{sd}"] = (c, mjquat(frame([1, 0, 0], b - a)), (.0034, np.linalg.norm(b - a) / 2, 0))
        # the eye: iris and pupil slide over the white toward the gaze; lower lid rises with a smile
        ec = EYE_C[sd]
        g = np.array([1.0, 0, 0]) if gaze_head is None else unit(gaze_head[sd] if isinstance(gaze_head, dict) else gaze_head)
        oy = float(np.clip(g[1] / max(g[0], .2) * .016, -.0075, .0075))
        oz = float(np.clip(g[2] / max(g[0], .2) * .016, -.0065, .0055))
        out[f"iris_{sd}"] = (np.array([ec[0] + .0078, ec[1] + oy, ec[2] + oz]), mjquat(np.eye(3)), None)
        out[f"pupil_{sd}"] = (np.array([ec[0] + .0092, ec[1] + oy, ec[2] + oz]), mjquat(np.eye(3)), None)
        out[f"glint_{sd}"] = (np.array([ec[0] + .0104, ec[1] + oy + .003, ec[2] + oz + .0035]), mjquat(np.eye(3)), None)
        # lids: hidden inside the face at rest; the lower lid rises over the white in a smile (a warm squint), the
        # upper lid lowers a little in a frown and fully in a blink
        sm, fr = max(e, 0), max(-e, 0)
        lx = ec[0] + (.0004 if sm > .05 else -.014)
        out[f"lid_lo_{sd}"] = (np.array([lx, ec[1], ec[2] - .0185 + .0085 * sm]), mjquat(np.eye(3)), None)
        ux = ec[0] + (.0004 if (fr > .05 or blink > 0) else -.014)
        out[f"lid_up_{sd}"] = (np.array([ux, ec[1], ec[2] + .0175 - .005 * fr - .018 * blink]), mjquat(np.eye(3)), None)
    return out
