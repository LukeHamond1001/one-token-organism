"""The G1's two eyes (docs/SIM_DESIGN.md section 3.4; from the 2026-09-24 prototype): a stereo pair of colour cameras where the head's D435 imagers are (see
g1scene.py), each rendered at its native size once a tick into one offscreen buffer and read back in one call, then
split in software into a PERIPHERY (the whole field, averaged 3 x 3) and a FOVEA (a 32 x 32 window of the native
image) whose position is an attention state moved by a gaze effector: conjugate (yaw, pitch, both windows together)
and vergence (the windows apart or together, so a near thing can be fixated by both). No depth channel; no stereo
algorithm: the body gets both eyes' images and learns what they share.

Gaze units: degrees in each camera's image (yaw to the image's right positive, pitch up positive), from the optical
axis. Vergence positive converges (the left window moves right, the right window left). Each window's centre is kept
inside its image, so the fovea's reach is the field minus half a window (about +-33 deg x +-18 deg)."""
import math
import sys
from pathlib import Path

import mujoco
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import g1scene as G  # noqa: E402


class Eyes:
    def __init__(self, m, w=G.EYE_W, h=G.EYE_H, pool=G.POOL, fovea=G.FOVEA, shadows=False, samples=4, ctx=None):
        from mujoco import gl_context
        self.m, self.w, self.h, self.pool, self.fov_px = m, w, h, pool, fovea
        m.vis.quality.offsamples = samples
        self.ctx = ctx or gl_context.GLContext(max(2 * w, 64), max(h, 64))
        self.ctx.make_current()
        self.con = mujoco.MjrContext(m, mujoco.mjtFontScale.mjFONTSCALE_100)
        mujoco.mjr_setBuffer(mujoco.mjtFramebuffer.mjFB_OFFSCREEN, self.con)
        self.scn = mujoco.MjvScene(m, maxgeom=4000)
        self.scn.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = shadows
        self.scn.flags[mujoco.mjtRndFlag.mjRND_REFLECTION] = False
        self.scn.flags[mujoco.mjtRndFlag.mjRND_SKYBOX] = False
        self.opt = G.eye_option()
        self.pert = mujoco.MjvPerturb()
        self.cams = {}
        for sd in "LR":
            c = mujoco.MjvCamera(); c.type = mujoco.mjtCamera.mjCAMERA_FIXED; c.fixedcamid = m.camera(f"eye_{sd}").id
            self.cams[sd] = c
        self.buf = np.zeros((h, 2 * w, 3), np.uint8)
        self.f = (h / 2) / math.tan(math.radians(m.cam_fovy[m.camera("eye_L").id]) / 2)   # focal length in px
        self.gaze = np.zeros(3)             # yaw, pitch, vergence (deg)

    def render(self, d):
        """Both eyes' native images (h x w x 3 each, row 0 at the top), one read-back."""
        for i, sd in enumerate("LR"):
            mujoco.mjv_updateScene(self.m, d, self.opt, self.pert, self.cams[sd], mujoco.mjtCatBit.mjCAT_ALL, self.scn)
            mujoco.mjr_render(mujoco.MjrRect(self.w * i, 0, self.w, self.h), self.scn, self.con)
        mujoco.mjr_readPixels(self.buf, None, mujoco.MjrRect(0, 0, 2 * self.w, self.h), self.con)
        img = np.flipud(self.buf)
        return {"L": img[:, :self.w], "R": img[:, self.w:]}

    def periphery(self, img):
        p = self.pool
        h, w = self.h // p * p, self.w // p * p
        return img[:h, :w].reshape(h // p, p, w // p, p, 3).mean((1, 3)).astype(np.uint8)

    def centre_px(self, sd, gaze=None):
        """The fovea window's centre (column, row) in eye sd's native image for a gaze (yaw, pitch, vergence)."""
        yaw, pitch, verg = self.gaze if gaze is None else gaze
        yaw = yaw + (verg / 2 if sd == "L" else -verg / 2)
        cx = self.w / 2 + self.f * math.tan(math.radians(yaw))
        cy = self.h / 2 - self.f * math.tan(math.radians(pitch))
        half = self.fov_px / 2
        return float(np.clip(cx, half, self.w - half)), float(np.clip(cy, half, self.h - half))

    def fovea(self, img, sd, gaze=None):
        cx, cy = self.centre_px(sd, gaze)
        x0, y0 = int(round(cx - self.fov_px / 2)), int(round(cy - self.fov_px / 2))
        return img[y0:y0 + self.fov_px, x0:x0 + self.fov_px]

    def tick(self, d):
        """What the body gets from its eyes this tick: per eye the periphery and the fovea."""
        imgs = self.render(d)
        return {sd: dict(periphery=self.periphery(imgs[sd]), fovea=self.fovea(imgs[sd], sd)) for sd in "LR"}, imgs

    # ---- instruments (world truth: for stills and tests, never for the body)
    def project(self, d, sd, point):
        """A world point's (yaw, pitch) in degrees in eye sd, and whether it lies in front of the camera."""
        c = self.m.camera(f"eye_{sd}").id
        R = d.cam_xmat[c].reshape(3, 3)
        v = R.T @ (np.asarray(point, float) - d.cam_xpos[c])
        if v[2] >= 0:
            return None
        return math.degrees(math.atan2(v[0], -v[2])), math.degrees(math.atan2(v[1], -v[2]))

    def gaze_at(self, d, point):
        """The gaze (yaw, pitch, vergence) that puts a world point at both fovea centres (conjugate = the mean of the two
        eyes' directions, vergence = their difference)."""
        a, b = self.project(d, "L", point), self.project(d, "R", point)
        if a is None or b is None:
            return None
        self.gaze = np.array([(a[0] + b[0]) / 2, (a[1] + b[1]) / 2, a[0] - b[0]])
        return self.gaze

    def close(self):
        self.con.free()
