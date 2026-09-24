"""THE G1'S TWO EYES (docs/SIM_DESIGN.md 3.4, 3.7 and the decision log A1; the build plan's W3; the owner's decisions 4 and 9): a
colour stereo pair where the head's RealSense D435 imagers are (body/sim/g1scene.py adds the two cameras at load), each rendered
at its native 168 x 96 px once a tick, both into one offscreen buffer with one read-back, then split in software:
  PERIPHERY  the whole field (88 x 58 deg) averaged 3 x 3: 56 x 32 px (0.64 px a degree);
  FOVEA      a 32 x 32 px window of the native image (about 21 deg, 1.5 px a degree), placed by the world's gaze state (the gaze
             effector's conjugate yaw and pitch, the vergence setting the two windows apart; the VOR counter-turning it:
             body/sim/world.py). Nothing on the robot moves: the real G1 can run the same window on its own images.
No depth channel and no stereo algorithm: the body gets both eyes and learns what they share (the owner's decision 4).

THE RETINA'S CODE (the channels eye_p and eye_f, 3.4's table): each image is pooled into cells, 8 px cells of the periphery (7 x 4
per eye, each about 12.5 x 14.5 deg) and 4 px cells of the fovea (8 x 8 per eye), and each cell's mean colour is read by 6 fixed
channels: the retina's three opponent axes (luminance, red-green, blue-yellow; Hering; De Valois), each split into an ON and an
OFF half-wave (Kuffler; Schiller), against a fixed mid-grey: [L+, L-, RG+, RG-, BY+, BY-], L = mean(R, G, B) - 0.5, RG = R - G,
BY = B - (R + G) / 2, colours in 0-1. Fixed, not learned and not random: 2 x 168 numbers for the periphery, 2 x 384 for the
fovea, which the core's born code projects to d. The images themselves go to the frame's truth (the page, the stills, the eye
check), never to the body.

THE FACE TEST (A1; the reward's gate and the event line "a face in the fovea"): an eye sees the parent's face this tick when all
four hold: the parent's mouth point lies inside that eye's fovea window; a ray from the eye to it over the geoms the eyes render
hits nothing first (the child's own hand, a toy, the parent's hand or hair block it); the face is turned within 75 deg of the eye;
and its front (an ellipse 0.17 x 0.21 m) covers at least 20 fovea px x the cosine of the turn. Either eye counts (`face_seen`).
The reading of her expression, its 2 consecutive ticks and its 30-tick hold are the born reading's (the body's, P3/S5a).

SHADOWS. The room has a sun, a key spot and a fill. `shadows`: "sun" (the sun's shadow only: the body's eyes, 3.4), "all" or
"none". The eye check decides it on identity (C3): the fovea's code tells the ten toys and the face apart as well with the sun's
shadow as without (tools/sim_eye_check.py, 2026-09-24, 1,650 views at midday: a small nonlinear readout 0.865 with it and 0.864
without; the face 0.76 and 0.82; linear 0.62 and 0.64), so nothing in the check asks to drop it. It costs about 2.5x the render;
dropping it for speed is the owner's render choice (SIM_DESIGN.md 9 and B3), never ours."""
import hashlib
import math
import sys
import time
from pathlib import Path

import mujoco
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from body.sim import world as W  # noqa: E402

G = W.G
kin = G.kin
POOL = 3                        # the periphery: the native image averaged 3 x 3 (3.4; anatomy, ours)
CELL_P, CELL_F = 8, 4           # the retina's cells: 8 px of the periphery, 4 px of the fovea (3.4; anatomy, ours)
OPPONENT = 6                    # ON and OFF halves of luminance, red-green and blue-yellow (anatomy, ours)
FACE_TURN_DEG = 75.0            # the face test (A1; innate, ours)
FACE_FRONT_M = (0.17, 0.21)
FACE_MIN_PX = 20.0
RAY_SLACK_M = 0.02              # a ray's first hit within 2 cm of the mouth point is the face itself
EYE_GROUPS = np.array([1, 1, 1, 0, 0, 0], dtype=np.uint8)   # the eyes see groups 0-2: never collision proxies, sites or markers
EYE_P_SIZE = 2 * (G.EYE_W // POOL // CELL_P) * (G.EYE_H // POOL // CELL_P) * OPPONENT      # 336
EYE_F_SIZE = 2 * (W.FOVEA_PX // CELL_F) ** 2 * OPPONENT                                    # 768


def retina(img, cell):
    """the retina's code of an image (rows x cols x 3, uint8): cells of `cell` px, each cell's mean colour read by the 6 opponent
    ON/OFF channels; returns (rows / cell, cols / cell, 6) floats, flattened row by row by the caller"""
    h, w = img.shape[0] // cell * cell, img.shape[1] // cell * cell
    c = img[:h, :w].reshape(h // cell, cell, w // cell, cell, 3).mean(axis=(1, 3)) / 255.0
    lum = c.mean(axis=-1) - 0.5
    rg = c[..., 0] - c[..., 1]
    by = c[..., 2] - (c[..., 0] + c[..., 1]) / 2
    return np.stack([np.maximum(lum, 0), np.maximum(-lum, 0), np.maximum(rg, 0), np.maximum(-rg, 0),
                     np.maximum(by, 0), np.maximum(-by, 0)], axis=-1)


def periphery(img):
    """the periphery: the native image averaged POOL x POOL (uint8)"""
    h, w = img.shape[0] // POOL * POOL, img.shape[1] // POOL * POOL
    return img[:h, :w].reshape(h // POOL, POOL, w // POOL, POOL, 3).mean(axis=(1, 3)).round().astype(np.uint8)


def window_centre(side, gaze):
    """the fovea window's centre (column, row) in eye `side`'s native image for a gaze (yaw, pitch, vergence), in px"""
    yaw = gaze[0] + (gaze[2] / 2 if side == "L" else -gaze[2] / 2)
    return G.EYE_W / 2 + W.EYE_F_PX * math.tan(yaw), G.EYE_H / 2 - W.EYE_F_PX * math.tan(gaze[1])


def window_corner(side, gaze):
    """the window's top-left pixel (column, row), inside the image"""
    cx, cy = window_centre(side, gaze)
    half = W.FOVEA_PX / 2
    x0 = int(round(cx - half)); y0 = int(round(cy - half))
    return min(G.EYE_W - W.FOVEA_PX, max(0, x0)), min(G.EYE_H - W.FOVEA_PX, max(0, y0))


def fovea(img, side, gaze):
    x0, y0 = window_corner(side, gaze)
    return img[y0:y0 + W.FOVEA_PX, x0:x0 + W.FOVEA_PX]


def project(m, d, side, point):
    """a world point in eye `side`'s image: (column, row, depth along the axis) or None behind the eye"""
    c = m.camera(f"eye_{side}").id
    v = d.cam_xmat[c].reshape(3, 3).T @ (np.asarray(point, float) - d.cam_xpos[c])
    if v[2] >= 0:
        return None
    return G.EYE_W / 2 + W.EYE_F_PX * v[0] / -v[2], G.EYE_H / 2 - W.EYE_F_PX * v[1] / -v[2], -v[2]


def gaze_at(m, d, point):
    """the gaze (yaw, pitch, vergence) that puts a world point at both windows' centres (an instrument's aim; never the body's)"""
    out = []
    for side in "LR":
        c = m.camera(f"eye_{side}").id
        v = d.cam_xmat[c].reshape(3, 3).T @ (np.asarray(point, float) - d.cam_xpos[c])
        out.append((math.atan2(v[0], -v[2]), math.atan2(v[1], -v[2])))
    (yl, pl), (yr, pr) = out
    return np.array([(yl + yr) / 2, (pl + pr) / 2, yl - yr])


def mouth_point(m, d):
    """the parent's mouth point and her face's forward axis and centre, in the world (world truth: the face test's)"""
    b = m.body("parent_head").id
    R = d.xmat[b].reshape(3, 3)
    local = np.array([kin.head_surface_x(0, kin.MOUTH_Z) + .0015, 0.0, kin.MOUTH_Z])
    centre = np.array([kin.head_surface_x(0, 0.15), 0.0, 0.15])
    return d.xpos[b] + R @ local, R[:, 0].copy(), d.xpos[b] + R @ centre


def face_test(m, d, gaze):
    """A1's test for each eye: {side: (passes, why)}; why names the first condition that failed ("" when it passes)"""
    mouth, fwd, centre = mouth_point(m, d)
    out = {}
    for side in "LR":
        c = m.camera(f"eye_{side}").id
        eye = d.cam_xpos[c].copy()
        p = project(m, d, side, mouth)
        if p is None:
            out[side] = (False, "behind the eye"); continue
        x0, y0 = window_corner(side, gaze)
        if not (x0 <= p[0] < x0 + W.FOVEA_PX and y0 <= p[1] < y0 + W.FOVEA_PX):
            out[side] = (False, "not in the fovea"); continue
        to = mouth - eye
        dist = float(np.linalg.norm(to))
        gid = np.array([-1], dtype=np.int32)
        hit = mujoco.mj_ray(m, d, eye, to / dist, EYE_GROUPS, 1, -1, gid)
        if 0 <= hit < dist - RAY_SLACK_M:                              # something before the mouth (its own surface is at dist)
            out[side] = (False, "blocked"); continue
        to_eye = eye - centre
        cos_turn = float(fwd @ to_eye / np.linalg.norm(to_eye))
        if cos_turn < math.cos(math.radians(FACE_TURN_DEG)):
            out[side] = (False, "turned away"); continue
        fd = float(np.linalg.norm(to_eye))
        area = math.pi / 4 * (FACE_FRONT_M[0] * W.EYE_F_PX / fd) * (FACE_FRONT_M[1] * W.EYE_F_PX / fd) * cos_turn
        if area < FACE_MIN_PX:
            out[side] = (False, "too small"); continue
        out[side] = (True, "")
    return out


class Eyes:
    """both eyes over a G1World: `render()` the two native images, `see()` this tick's retina codes and face test (rendered again
    only when something they would see has moved: the state, the gaze, the scene's run-time fields), `close()` the GL context. Attaching sets the world's `eyes`, so its frames carry eye_p, eye_f and face_seen."""

    def __init__(self, world, shadows="sun", samples=4):
        from mujoco import gl_context
        self.world = world
        m = self.m = world.m
        self.w, self.h = G.EYE_W, G.EYE_H
        self.ctx = gl_context.GLContext(max(2 * self.w, 64), max(self.h, 64))
        self.ctx.make_current()
        room_samples = int(m.vis.quality.offsamples)
        m.vis.quality.offsamples = int(samples)                        # the eyes' multisampling, fixed in their own context
        try:
            self.con = mujoco.MjrContext(m, mujoco.mjtFontScale.mjFONTSCALE_100)
        finally:
            m.vis.quality.offsamples = room_samples
        mujoco.mjr_setBuffer(mujoco.mjtFramebuffer.mjFB_OFFSCREEN, self.con)
        self.scn = mujoco.MjvScene(m, maxgeom=5000)
        self.scn.flags[mujoco.mjtRndFlag.mjRND_REFLECTION] = False
        self.scn.flags[mujoco.mjtRndFlag.mjRND_SKYBOX] = False
        self.opt = G.eye_option()
        self.pert = mujoco.MjvPerturb()
        self.cams = {}
        for side in "LR":
            c = mujoco.MjvCamera(); c.type = mujoco.mjtCamera.mjCAMERA_FIXED; c.fixedcamid = m.camera(f"eye_{side}").id
            self.cams[side] = c
        self.buf = np.zeros((self.h, 2 * self.w, 3), np.uint8)
        self.set_shadows(shadows)
        self._cache = None
        self.timing = {"renders": 0, "render_s": 0.0}
        world.eyes = self

    def set_shadows(self, shadows):
        if shadows not in ("sun", "all", "none"):
            raise ValueError(f"shadows: 'sun', 'all' or 'none'; given {shadows!r}")
        self.shadows = shadows
        self.scn.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = shadows != "none"
        self._cache = None

    def render(self):
        """both eyes' native images {side: rows x cols x 3 uint8, row 0 at the top}, one read-back"""
        t0 = time.perf_counter()
        m, d = self.m, self.world.d
        self.ctx.make_current()
        cast = m.light_castshadow.copy()
        if self.shadows == "sun":
            for i in range(m.nlight):
                m.light_castshadow[i] = cast[i] if m.light(i).name == "sun" else 0
        try:
            for i, side in enumerate("LR"):
                mujoco.mjv_updateScene(m, d, self.opt, self.pert, self.cams[side], mujoco.mjtCatBit.mjCAT_ALL, self.scn)
                mujoco.mjr_render(mujoco.MjrRect(self.w * i, 0, self.w, self.h), self.scn, self.con)
            mujoco.mjr_readPixels(self.buf, None, mujoco.MjrRect(0, 0, 2 * self.w, self.h), self.con)
        finally:
            m.light_castshadow[:] = cast
        img = np.flipud(self.buf)
        self.timing["renders"] += 1; self.timing["render_s"] += time.perf_counter() - t0
        return {"L": img[:, :self.w].copy(), "R": img[:, self.w:].copy()}

    def see(self):
        """this tick's eyes: {"eye_p": 2 x 168, "eye_f": 2 x 384, "face_seen": [1 or 0], "truth": images and the face test}"""
        w = self.world
        m, d = self.m, w.d
        key = hashlib.blake2b(b"".join(np.ascontiguousarray(x).tobytes() for x in (
            d.qpos, d.mocap_pos, d.mocap_quat, w.gaze, m.geom_pos, m.geom_quat, m.geom_size, m.geom_rgba, m.light_active,
            m.light_diffuse, m.light_dir, m.light_pos, m.light_castshadow, m.mat_rgba, m.mat_emission))).digest()      # what the eyes would see: rendered again only if it moved
        if self._cache is None or self._cache[0] != key:
            imgs = self.render()
            per = {s: periphery(imgs[s]) for s in "LR"}
            fov = {s: fovea(imgs[s], s, w.gaze) for s in "LR"}
            eye_p = np.concatenate([retina(per[s], CELL_P).reshape(-1) for s in "LR"])
            eye_f = np.concatenate([retina(fov[s], CELL_F).reshape(-1) for s in "LR"])
            ft = face_test(self.m, w.d, w.gaze)
            seen = float(ft["L"][0] or ft["R"][0])
            truth = {"images": imgs, "periphery": per, "fovea": fov, "face_test": ft,
                     "windows": {s: window_corner(s, w.gaze) for s in "LR"}}
            self._cache = (key, eye_p, eye_f, seen, truth)
        _, eye_p, eye_f, seen, truth = self._cache
        return {"eye_p": eye_p.copy(), "eye_f": eye_f.copy(), "face_seen": np.array([seen]), "truth": truth}

    def close(self):
        if self.world.eyes is self:
            self.world.eyes = None
        self.con.free()
        self.ctx.free()
