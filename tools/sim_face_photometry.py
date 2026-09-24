"""HER FACE'S PHOTOMETRY AGAINST A REAL FACE'S (docs/SIM_DESIGN.md 4.1 and C3; the owner's decision of 2026-09-24, made for him in
the W1 verifier's third round: her face given the photometry real faces have at low spatial frequency). An instrument, never the
body. The source is Russell, Kramer and Jones 2017 (Adapt Hum Behav Physiol, table 2): in photographs of young women (19-31 y, no
cosmetics, neutral), the contrast (skin - feature) / (skin + feature) in CIE L* between each feature and an annulus of skin around it
is 0.152 for the eyes (the eye with the band of skin around its lashes), 0.126 for the brows and 0.092 for the lips. Measured here
the way the source measures it: her face (neutral) rendered from the front at 0.62 m (the head's portrait camera) under a uniform
light (ambient only: each pixel is the surface's albedo), the pixels read as sRGB and turned to
CIE L*, each feature's pixels (from a segmentation render: the eyes are the globe, the iris, the pupil, the highlight and the lash
line; the brows; the lips) averaged, and the skin's pixels (her face sheet and her lids) in an annulus of ANNULUS_MM around the
feature averaged. The source's photographs are lit from the front (a studio's), where the eye sockets are lit too, so the
contrasts are the surfaces' albedo: the sockets' baked occlusion of the room's indirect light (parent_kin.socket_occlusion, on the
eye's and the lids' materials) is taken off for the measure (`albedo=True`, the default) and stays in the scene. `calibrate` finds,
for each feature the colour that gives the source's contrast: the eyes by the sclera's grey (the iris, pupil and lashes kept dark);
the brows as her hair's colour over her skin, the share of the brow's area the hairs cover found; the lips as the prototype's lip
colour over her skin, its share found (so a feature keeps its hue and only its strength is set). The maker's base materials carry
them (body/sim/make_g1room.py MAT); the contrasts as the scene draws
them under the uniform light (the occlusion in) are reported beside (body/tests/test_sim_eyes.py checks both). Run:
nice -n 19 python3 tools/sim_face_photometry.py [--calibrate]  (JSON on stdout; nothing written)"""
import argparse
import json
import os
import sys

import mujoco
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "body", "sim"))
import g1scene as G  # noqa: E402

RUSSELL = {"eyes": 0.152, "brows": 0.126, "lips": 0.092}     # young women, no cosmetics (table 2)
ANNULUS_MM = 5.0                # the skin annulus's width around a feature (the source's figure 1 draws a band of about this)
RES = 700                       # px, the calibration render (the portrait camera's 30 deg at 0.62 m: about 2 px a mm)
FEATURES = {"eyes": ("sclera_", "iris_", "pupil_", "glint_", "lash_", "lash2_"), "brows": ("brow_", "brow2_"),
            "lips": ("mouth", "lip_lo")}
SKIN = ("parent_face", "parent_lid_up_", "parent_lid_lo_")
CAL_MATERIAL = {"eyes": ("sclera", "scale", None), "brows": ("brow", "over skin", (.24, .15, .10)),   # her hair's colour
                "lips": ("lips", "over skin", (.56, .24, .26))}                                     # the prototype's lip colour
SKIN_RGB = (.86, .66, .54)      # her skin (the maker's "skin")


def srgb_to_lstar(img):
    c = img.astype(float) / 255.0
    lin = np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)
    Y = lin @ np.array([0.2126, 0.7152, 0.0722])
    f = np.where(Y > (6 / 29) ** 3, np.cbrt(Y), Y / (3 * (6 / 29) ** 2) + 4 / 29)
    return 116 * f - 16


class Photometry:
    def __init__(self, scene=None, albedo=True):
        self.sc = scene or G.Scene()
        m, d = self.sc.m, self.sc.d
        self.ao = {}
        if albedo:                                                      # the sockets' baked occlusion off: the surfaces' albedo
            ao = G.kin.socket_occlusion()
            for part, names in G.kin.SOCKET_MATERIALS.items():
                for n in names:
                    m.mat_rgba[m.material(n).id, :3] /= ao[part]
                    self.ao[n] = ao[part]
        self.sc.birth()
        mujoco.mj_forward(m, d)
        self.r = mujoco.Renderer(m, RES, RES)
        self.seg = mujoco.Renderer(m, RES, RES)
        self.seg.enable_segmentation_rendering()
        names = {g: (m.geom(g).name or "") for g in range(m.ngeom)}
        self.feature_geoms = {k: {g for g, n in names.items() if n.startswith(tuple("parent_" + p for p in pre))} for k, pre in FEATURES.items()}
        self.skin_geoms = {g for g, n in names.items() if n == SKIN[0] or n.startswith(SKIN[1:])}
        cam = m.camera("parent_portrait")
        self.px_per_mm = RES / (2 * 0.62 * np.tan(np.radians(float(m.cam_fovy[cam.id]) / 2))) / 1000

    def render(self):
        m, d = self.sc.m, self.sc.d
        la, hl = m.light_active.copy(), (m.vis.headlight.active, m.vis.headlight.ambient.copy(), m.vis.headlight.diffuse.copy(),
                                         m.vis.headlight.specular.copy())
        try:
            m.light_active[:] = 0
            m.vis.headlight.active = 1; m.vis.headlight.ambient[:] = 1; m.vis.headlight.diffuse[:] = 0; m.vis.headlight.specular[:] = 0
            self.r.update_scene(d, camera="parent_portrait"); img = self.r.render()
            self.seg.update_scene(d, camera="parent_portrait"); s = self.seg.render()
        finally:
            m.light_active[:] = la
            m.vis.headlight.active, m.vis.headlight.ambient[:], m.vis.headlight.diffuse[:], m.vis.headlight.specular[:] = hl
        return img, s

    def contrasts(self):
        from scipy.ndimage import binary_dilation
        img, s = self.render()
        L = srgb_to_lstar(img)
        gid = np.where(s[..., 1] == int(mujoco.mjtObj.mjOBJ_GEOM), s[..., 0], -1)
        skin = np.isin(gid, list(self.skin_geoms))
        n = max(1, int(round(ANNULUS_MM * self.px_per_mm)))
        out = {}
        for k, geoms in self.feature_geoms.items():
            feat = np.isin(gid, list(geoms))
            ring = binary_dilation(feat, iterations=n) & ~feat & skin
            ls, lf = float(L[ring].mean()), float(L[feat].mean())
            out[k] = {"contrast": (ls - lf) / (ls + lf), "L_feature": lf, "L_skin": ls, "px_feature": int(feat.sum()), "px_ring": int(ring.sum())}
        return out

    def calibrate(self):
        """each feature's colour that gives the source's contrast (bisection): the sclera's grey scale; the brows' and lips' share
        of their colour over her skin"""
        m = self.sc.m
        found = {}
        for k, (mat, how, dark) in CAL_MATERIAL.items():
            mid = m.material(mat).id
            base = m.mat_rgba[mid].copy()
            if how == "scale":
                colour = lambda t: np.clip(base[:3] * t, 0, 1)
                lo, hi, lighter_up = 0.05, 1.0 / max(float(base[:3].max()), 1e-6), True
            else:
                colour = lambda t: t * np.asarray(dark) + (1 - t) * np.asarray(SKIN_RGB)
                lo, hi, lighter_up = 0.0, 1.0, False
            for _ in range(24):
                t = (lo + hi) / 2
                m.mat_rgba[mid, :3] = colour(t)
                c = self.contrasts()[k]["contrast"]
                too_dark = c > RUSSELL[k]
                if lighter_up:
                    lo, hi = (t, hi) if too_dark else (lo, t)
                else:
                    lo, hi = (lo, t) if too_dark else (t, hi)
            t = (lo + hi) / 2
            m.mat_rgba[mid, :3] = colour(t)
            found[k] = {"material": mat, how: round(t, 4), "rgba": [round(float(x), 4) for x in m.mat_rgba[mid]]}
        return found

    def close(self):
        self.r.close(); self.seg.close()


def main(calibrate=False):
    p = Photometry()
    out = {"source": RUSSELL, "albedo": p.contrasts()}
    if calibrate:
        out["calibrated (base albedo)"] = p.calibrate()
        out["albedo after"] = p.contrasts()
    p.close()
    q = Photometry(albedo=False)
    out["as drawn (the sockets' occlusion in)"] = q.contrasts()
    q.close()
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--calibrate", action="store_true")
    a = ap.parse_args()
    print(json.dumps(main(a.calibrate), indent=1, default=float))
