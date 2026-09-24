"""THE BORN FACE TEMPLATE ON THE PARENT'S FACE (docs/SIM_DESIGN.md 3.4, 3.7, 4.1, A1 and C3; the owner's decision of 2026-09-24,
made for him in the W1 verifier's third round: fix the world, not the detector; then measure the template on her face in the
fovea and the periphery at 0.3-2 m under the room's lights, and report it plainly). An instrument, never the body; it changes
neither the template nor her face.

The G1 lies as born. Her head (its mocap segment; the rest of her stands where birth put her) is put at a distance D from the left
eye along a direction in that eye's image, facing the eye and upright in its eyes (her crown toward the child's head, as when she
leans over it from its feet's side: the eye check's placement, tools/sim_eye_check.py), her face at its neutral expression, lit by
the room's own lights (no lamp at the eye), with the sun's shadow (the body's eyes). D runs 0.3-2.0 m; the direction is the window's
centre (yaw 0, pitch 0) for the fovea, and 20 deg off it (in the periphery, where she puts her face when she calls, A3) for the
periphery. For each view, the template (body/sim/eyes.py, its constants untouched) is read in each eye:
  FOVEA      the gaze aimed at her face's centre (the instrument's aim): the template on each eye's 32 px window, its best match
             anywhere in the window (what the event line reads) and whether it matches (r >= TEMPLATE_R and contrast >=
             TEMPLATE_CONTRAST); and whether the match is a DETECTION OF HER FACE (the W1 verifier's fourth round: an 8 px match on a
             33 px face is not one): centred within a third of her face's width of her face's centre, at a width within the bank's
             step (sqrt 2) of her face's in the template's own terms (her pupils' distance / 0.44: 140 mm); else a chance match. The
             control: the same window upside down (no face's layout) read by the same template;
  PERIPHERY  the template on each eye's 56 x 32 px periphery, its best match anywhere, and its best match centred on her face
             (within a third of her face's width of her face's centre in the image), with r and contrast, so a miss can be placed:
             too small for the template's smallest size (8 px), too little correlation, or too little contrast.
Also her whole body in one of the parent's acts (body/sim/g1acts.py): kneeling beside it, leaning over its chest and looking at its
eyes (attend). Under the room's midday light and the eye check's morning and dusk stand-ins for the sun (each with its indirect
light, the maker's ROOM_INDIRECT of its diffuse; the light is never changed for the template: 5.4). Run:
nice -n 19 python3 tools/sim_face_template.py [--lights midday,morning,dusk] [--expr neutral]  (JSON on stdout; nothing written)"""
import argparse
import json
import math
import os
import sys

import mujoco
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))
sys.path.insert(0, os.path.join(ROOT, "body", "sim"))
from body.sim import eyes as E  # noqa: E402
from body.sim import world as W  # noqa: E402
from sim_eye_check import LIGHTS  # noqa: E402
from make_g1room import ROOM_INDIRECT  # noqa: E402

G = W.G
kin = G.kin
DISTANCES = (0.3, 0.45, 0.6, 0.8, 1.0, 1.25, 1.5, 2.0)
PERIPH_OFF_DEG = 20.0


def template_maps(img):
    """the born template's r and contrast at every place and size of an image (as eyes.face_template computes them):
    [(width, height, r map, contrast map)]"""
    from numpy.lib.stride_tricks import sliding_window_view
    L = img.astype(float).mean(axis=-1) / 255.0
    out = []
    for w_, (h, n, K) in E.TEMPLATES.items():
        if h > L.shape[0] or w_ > L.shape[1]:
            continue
        win = sliding_window_view(L, (h, w_))
        rows, cols = win.shape[:2]
        P = win.reshape(rows * cols, h * w_)
        st, mean, m_skin, m_blob = (P @ K).T
        m2 = (P * P) @ K[:, 1]
        sd = np.sqrt(np.maximum(m2 - mean * mean, 0.0) * n)
        r = np.where(sd > 1e-6, st / np.maximum(sd, 1e-6), 0.0).reshape(rows, cols)
        con = ((m_skin - m_blob) / np.maximum(m_skin, 1e-6)).reshape(rows, cols)
        out.append((w_, h, r, con))
    return out


def best_near(img, cx, cy, radius):
    """the template's best match (by r, among places with the contrast; else by r) whose centre lies within `radius` px of (cx, cy)"""
    best = None
    for w_, h, r, con in template_maps(img):
        rows, cols = r.shape
        yy, xx = np.mgrid[0:rows, 0:cols]
        near = (xx + w_ / 2 - cx) ** 2 + (yy + h / 2 - cy) ** 2 <= radius ** 2
        if not near.any():
            continue
        for ok in (near & (con >= E.TEMPLATE_CONTRAST), near):
            if ok.any():
                k = np.argmax(np.where(ok, r, -np.inf))
                cand = (float(r.reshape(-1)[k]), float(con.reshape(-1)[k]), w_)
                if best is None or (cand[1] >= E.TEMPLATE_CONTRAST) > (best[1] >= E.TEMPLATE_CONTRAST) or \
                        ((cand[1] >= E.TEMPLATE_CONTRAST) == (best[1] >= E.TEMPLATE_CONTRAST) and cand[0] > best[0]):
                    best = cand
                break
    return best


# her face in the template's own terms: the template's eye blobs are 0.44 W apart (eyes.TEMPLATE_EYES), so her face's template width
# is her pupils' distance / 0.44 (140 mm), its height by the template's ellipse (FACE_FRONT_M's aspect), its centre 0.12 H under her
# pupils (the template's eyes sit 0.12 H over its centre)
FACE_W = kin.IPD / (2 * E.TEMPLATE_EYES[1][0])
FACE_H = FACE_W * E.FACE_FRONT_M[1] / E.FACE_FRONT_M[0]


def face_centre_local():
    """her face's centre in the head's frame, where the template's ellipse centre falls on her face"""
    z = kin.PUPIL_Z - E.TEMPLATE_EYES[1][1] * FACE_H
    return np.array([kin.head_surface_x(0, z), 0.0, z])


def detection(best, fp, x0=0.0, y0=0.0, pool=1.0):
    """whether a template match (r, contrast, width, col, row) is a detection of HER face: a match (r and contrast over the bars),
    centred within a third of her face's width of her face's centre, at a width within the bank's step (sqrt 2) of her face's (a match
    elsewhere or at another size is the template firing on something else: a chance match)"""
    if fp is None or not E.template_match(best):
        return False
    cx, cy, fw = (fp[0] / pool - x0), (fp[1] / pool - y0), fp[2] / pool
    return bool(math.hypot(best[3] - cx, best[4] - cy) <= fw / 3 and fw / math.sqrt(2) <= best[2] <= fw * math.sqrt(2))


def place_head(w, dist, yaw, pitch, expr):
    m, d = w.m, w.d
    camL = m.camera("eye_L").id
    head = m.body("parent_head").id
    hid = m.body_mocapid[head]
    R = d.cam_xmat[camL].reshape(3, 3)
    dirc = np.array([math.tan(yaw), math.tan(pitch), -1.0]); dirc /= np.linalg.norm(dirc)
    p = d.cam_xpos[camL] + R @ dirc * dist
    x = d.cam_xpos[camL] - p; x /= np.linalg.norm(x)
    z = np.array([-1.0, 0, 0]) - x * -x[0]; z /= np.linalg.norm(z)
    Rh = np.column_stack([x, np.cross(z, x), z])
    pose = G.born_parent(); pose.expr = expr
    w.scene.set_parent(pose)                                            # her face drawn (the expression), her body where birth put her
    d.mocap_pos[hid] = p - Rh @ face_centre_local()
    d.mocap_quat[hid] = kin.mjquat(Rh)
    mujoco.mj_forward(m, d)
    return p


def face_px(w, side, centre):
    """her face's centre and width in eye `side`'s native image (px), from the world (the instrument's)"""
    m, d = w.m, w.d
    pr = E.project(m, d, side, centre)
    if pr is None:
        return None
    return pr[0], pr[1], FACE_W * W.EYE_F_PX / pr[2]


def read_view(w, ey, centre, fovea_aim=True):
    m, d = w.m, w.d
    if fovea_aim:
        w.gaze = W.clamp_gaze(E.gaze_at(m, d, centre))
    else:
        w.gaze = np.zeros(3)
    seen = ey.see()
    tr = seen["truth"]
    out = {}
    for s in "LR":
        fp = face_px(w, s, centre)
        rec = {}
        if fovea_aim:
            b = tr["template_fovea"][s]
            x0, y0 = E.window_corner(s, w.gaze)
            flipped = E.face_template(tr["fovea"][s][::-1].copy())          # the control: the same pixels upside down (no face's layout)
            rec["fovea_best"] = {"r": round(b[0], 3) if np.isfinite(b[0]) else None, "contrast": round(b[1], 3), "width_px": b[2],
                                 "match": E.template_match(b), "her_face": detection(b, fp, x0, y0),
                                 "upside_down_r": round(flipped[0], 3) if np.isfinite(flipped[0]) else None}
            if fp is not None:
                nb = best_near(tr["fovea"][s], fp[0] - x0, fp[1] - y0, fp[2] / 3)
                rec["fovea_on_her_face"] = None if nb is None else {"r": round(nb[0], 3), "contrast": round(nb[1], 3), "width_px": nb[2]}
        b = tr["template_periphery"][s]
        rec["periphery_best"] = {"r": round(b[0], 3) if np.isfinite(b[0]) else None, "contrast": round(b[1], 3), "width_px": b[2],
                                 "match": E.template_match(b)}
        if fp is not None:
            px, py, pw = fp[0] / E.POOL, fp[1] / E.POOL, fp[2] / E.POOL
            rec["her_face_width_px"] = {"native": round(fp[2], 1), "periphery": round(pw, 1)}
            nb = best_near(tr["periphery"][s], px, py, max(pw / 3, 1.5))
            rec["periphery_on_her_face"] = None if nb is None else {"r": round(nb[0], 3), "contrast": round(nb[1], 3), "width_px": nb[2]}
            if b[2]:
                rec["periphery_best_is_her_face"] = bool(math.hypot(b[3] - px, b[4] - py) <= max(pw / 3, 1.5))
            rec["periphery_her_face"] = detection(b, fp, pool=E.POOL)
        out[s] = rec
    out["face_test"] = {s: tr["face_test"][s][1] or "passes" for s in "LR"}
    out["event_line_face_fovea"] = float(seen["face_fovea"][0]) if fovea_aim else None
    out["orienting_cue_face_periph"] = float(seen["face_periph"][0])
    return out


def pose_views(w, ey, expr):
    """her whole body: attend (kneel beside, lean over the chest, look at its eyes)"""
    import g1acts as GA
    m, d = w.m, w.d
    v = GA.G1View(m, d)
    at, yaw = GA.kneel_spot(v, "L")
    out = {}
    p, info = GA.attend(v, at, yaw, expr=expr)
    w.scene.set_parent(p); mujoco.mj_forward(m, d)
    hb = m.body("parent_head").id
    centre = d.xpos[hb] + d.xmat[hb].reshape(3, 3) @ face_centre_local()
    out["attend"] = dict(read_view(w, ey, centre), eye_distance_m=round(float(np.linalg.norm(centre - v.eyes)), 3), ok=bool(info["ok"]))
    return out


def main(lights=("midday", "morning", "dusk"), expr="neutral"):
    e = {"neutral": 0.0, "smile": 1.0}.get(expr, 0.0)
    w = W.G1World(seed=1)
    m, d = w.m, w.d
    ey = E.Eyes(w, shadows="sun")
    sun = m.light("sun").id
    sun0 = (m.light_dir[sun].copy(), m.light_diffuse[sun].copy())
    base = w.save_state()
    out = {"template": {"R": E.TEMPLATE_R, "contrast": E.TEMPLATE_CONTRAST, "widths_px": list(E.TEMPLATE_WIDTHS)}, "lights": {}}
    for L in lights:
        spec = LIGHTS[L]
        res = {"fovea": {}, "periphery": {}}
        for D in DISTANCES:
            for where, (yaw, pitch) in (("fovea", (0.0, 0.0)), ("periphery", (math.radians(PERIPH_OFF_DEG), 0.0))):
                w.load_state(base)
                if spec is None:
                    m.light_dir[sun], m.light_diffuse[sun] = sun0
                else:
                    m.light_dir[sun] = np.asarray(spec["dir"]) / np.linalg.norm(spec["dir"]); m.light_diffuse[sun] = spec["diffuse"]
                m.light_ambient[sun] = ROOM_INDIRECT * m.light_diffuse[sun]      # the room's indirect light follows its light (the maker's)
                centre = place_head(w, D, yaw, pitch, e)
                res[where][f"{D:g} m"] = read_view(w, ey, centre, fovea_aim=(where == "fovea"))
        w.load_state(base)
        if spec is not None:
            m.light_dir[sun] = np.asarray(spec["dir"]) / np.linalg.norm(spec["dir"]); m.light_diffuse[sun] = spec["diffuse"]
            m.light_ambient[sun] = ROOM_INDIRECT * m.light_diffuse[sun]
        res["poses"] = pose_views(w, ey, e)
        m.light_dir[sun], m.light_diffuse[sun] = sun0
        m.light_ambient[sun] = ROOM_INDIRECT * sun0[1]
        out["lights"][L] = res
    ey.close()
    # the plain count: in how many views did the template detect HER face (a match on her face at her face's size), and how many of
    # its matches were chance matches (elsewhere, or at another size)
    tally = {"fovea views": 0, "fovea detections of her face": 0, "fovea chance matches": 0, "periphery views": 0,
             "periphery detections of her face": 0, "periphery chance matches": 0}
    for L, res in out["lights"].items():
        for where in ("fovea", "periphery"):
            for D, v in res[where].items():
                for s in "LR":
                    tally[f"{where} views"] += 1
                    b = v[s].get("fovea_best" if where == "fovea" else "periphery_best", {})
                    her = v[s].get("fovea_best", {}).get("her_face") if where == "fovea" else v[s].get("periphery_her_face")
                    tally[f"{where} detections of her face"] += int(bool(her))
                    tally[f"{where} chance matches"] += int(bool(b.get("match")) and not her)
    out["tally"] = tally
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--lights", default="midday,morning,dusk")
    ap.add_argument("--expr", default="neutral")
    a = ap.parse_args()
    print(json.dumps(main(tuple(a.lights.split(",")), a.expr), indent=1))
