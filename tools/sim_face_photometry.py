"""HER FACE'S PHOTOMETRY AGAINST A REAL FACE'S (docs/SIM_DESIGN.md 4.1 and C3; the owner's decision of 2026-09-24, made for him in the
W1 verifier's third and fourth rounds: her face given the photometry real faces have at low spatial frequency). An instrument, never
the body; it writes nothing.

THE SOURCE. Russell, Kramer and Jones 2017 (Adapt Hum Behav Physiol 3:199-209, table 2; photographs from the FACES database, "taken
under identical lighting and exposure conditions in the same studio with the same camera"): in young women (19-31 y, no cosmetics,
neutral) the contrast (skin - feature) / (skin + feature) in CIE L* between each feature and an annulus of skin around it is 0.152 for
the eyes ("the eyes, including the band of skin around the eyelashes"), 0.126 for the brows and 0.092 for the lips.

MEASURED AS THE SOURCE MEASURES A PHOTOGRAPH (the W1 verifier's fourth round: the photographs are real faces, their own shading in):
her face (neutral) photographed from the front at 0.52 m (her head's portrait camera) under a studio's light: the room's lights
off, a frontal lamp at the camera (the headlight: a key along the view and a fill, STUDIO), exposed so her cheek's skin reads CIE L*
STUDIO_SKIN_L (a normally exposed portrait's skin; ours); everything her face carries is in: her own geometry's shading, the sheet's
shade of the room's indirect light (its texture) and the eye's parts' (their albedo). The pixels are read as sRGB and turned to
CIE L*. From a segmentation render at the same camera: THE EYE is the fissure's contents (the globe, the iris, the pupil, the
caruncle, the conjunctiva) and the upper lash line, grown by LASH_BAND_MM (the band of skin around the lashes: the lid margins); THE
BROWS their strips; THE LIPS their vermilion; each feature's skin annulus is ANNULUS_MM of skin around it (the source's figure 1
draws a band of about this; the scale in px per mm is taken at her face's own depth, 0.52 m, not the camera's frame origin). The
contrasts are the source's formula on the means. `calibrate` finds, for each feature, the albedo that gives the source's contrast:
the eyes by the iris's lightness (the prototype's brown, its hue kept: a pure white sclera could not bring the eye region to the
source's, 0.170, and a lash line light enough to would have left no lash line; the sclera kept at a sclera's off-white and the
lash line at half her hair's colour over her skin, both ours); the brows as her hair's colour over her skin, the share found; the
lips as the prototype's lip colour over her skin, the share found (so a feature keeps its hue and only its strength is found); the
maker's IRIS_RGB, BROW_RGB and LIPS_RGB carry them (body/sim/make_g1room.py; the iris's there without its baked shade) and
body/tests/test_sim_eyes.py eyes 13 checks them.
Run: nice -n 19 python3 tools/sim_face_photometry.py [--calibrate]  (JSON on stdout)"""
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

RUSSELL = {"eyes": 0.152, "brows": 0.126, "lips": 0.092}     # young women, no cosmetics (table 2)
ANNULUS_MM = 5.0
LASH_BAND_MM = 1.0              # the band of skin around the lashes (ours: the lid margin's width)
RES = 900                       # px (the portrait camera's 30 deg: about 3.3 px a mm at her face)
STUDIO = (0.35, 0.65)           # the studio lamp's ambient and diffuse shares (a frontal key and its fill; ours)
STUDIO_SKIN_L = 65.0            # her cheek's L* in the photograph (ours: a normally exposed portrait's skin)
FACE_DEPTH_M = 0.52             # the portrait camera's distance to her face's front (0.62 m from her head's frame origin, less the
                                # face's 0.10 m ahead of it)
FEATURES = {"eyes": ("sclera_", "iris_", "pupil_", "caruncle_", "fornix_", "lash"), "brows": ("brow",), "lips": ("mouth", "lip_lo")}
SKIN = ("parent_face", "parent_lid_up_", "parent_lid_lo_")
CAL_MATERIAL = {"eyes": ("iris", "scale", None),                     # the iris's lightness (its hue kept)
                "brows": ("brow", "over skin", (.24, .15, .10)),    # her hair's colour
                "lips": ("lips", "over skin", (.56, .24, .26))}     # the prototype's lip colour
SKIN_RGB = (.86, .66, .54)      # her skin (the maker's "skin")


def srgb_to_lstar(img):
    c = img.astype(float) / 255.0
    lin = np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)
    Y = lin @ np.array([0.2126, 0.7152, 0.0722])
    f = np.where(Y > (6 / 29) ** 3, np.cbrt(Y), Y / (3 * (6 / 29) ** 2) + 4 / 29)
    return 116 * f - 16


class Photometry:
    def __init__(self, scene=None):
        self.sc = scene or G.Scene()
        m, d = self.sc.m, self.sc.d
        self.sc.birth()
        mujoco.mj_forward(m, d)
        self.r = mujoco.Renderer(m, RES, RES)
        self.seg = mujoco.Renderer(m, RES, RES)
        self.seg.enable_segmentation_rendering()
        names = {g: (m.geom(g).name or "") for g in range(m.ngeom)}
        self.feature_geoms = {k: {g for g, n in names.items() if n.startswith(tuple("parent_" + p for p in pre))} for k, pre in FEATURES.items()}
        self.skin_geoms = {g for g, n in names.items() if n == SKIN[0] or n.startswith(SKIN[1:])}
        cam = m.camera("parent_portrait")
        self.px_per_mm = RES / (2 * FACE_DEPTH_M * math.tan(math.radians(float(m.cam_fovy[cam.id]) / 2))) / 1000
        self.gain = 1.0
        self.exposure()

    def render(self):
        m, d = self.sc.m, self.sc.d
        hl = m.vis.headlight
        la, old = m.light_active.copy(), (hl.active, hl.ambient.copy(), hl.diffuse.copy(), hl.specular.copy())
        try:
            m.light_active[:] = 0
            hl.active = 1; hl.ambient[:] = STUDIO[0] * self.gain; hl.diffuse[:] = STUDIO[1] * self.gain; hl.specular[:] = 0
            self.r.update_scene(d, camera="parent_portrait"); self.r.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = False
            img = self.r.render()
            self.seg.update_scene(d, camera="parent_portrait"); s = self.seg.render()
        finally:
            m.light_active[:] = la
            hl.active, hl.ambient[:], hl.diffuse[:], hl.specular[:] = old
        return img, s

    def cheek_L(self):
        """the L* of her cheeks' skin (the median over the skin pixels below the eyes and beside the nose)"""
        img, s = self.render()
        L = srgb_to_lstar(img)
        gid = np.where(s[..., 1] == int(mujoco.mjtObj.mjOBJ_GEOM), s[..., 0], -1)
        skin = np.isin(gid, list(self.skin_geoms))
        h, w = L.shape
        box = np.zeros_like(skin); box[int(h * .52):int(h * .62), int(w * .25):int(w * .75)] = True
        return float(np.median(L[skin & box]))

    def exposure(self):
        """the lamp's gain that gives the cheeks STUDIO_SKIN_L (bisection)"""
        lo, hi = 0.2, 3.0
        for _ in range(18):
            self.gain = (lo + hi) / 2
            if self.cheek_L() > STUDIO_SKIN_L:
                hi = self.gain
            else:
                lo = self.gain
        self.gain = (lo + hi) / 2

    def masks(self, s):
        from scipy.ndimage import binary_dilation
        gid = np.where(s[..., 1] == int(mujoco.mjtObj.mjOBJ_GEOM), s[..., 0], -1)
        skin = np.isin(gid, list(self.skin_geoms))
        feats = {k: np.isin(gid, list(g)) for k, g in self.feature_geoms.items()}
        any_feat = np.zeros_like(skin)
        for f in feats.values():
            any_feat |= f
        out = {}
        nb = max(1, int(round(LASH_BAND_MM * self.px_per_mm)))
        na = max(1, int(round(ANNULUS_MM * self.px_per_mm)))
        for k, f in feats.items():
            region = binary_dilation(f, iterations=nb) & (f | skin) if k == "eyes" else f
            ring = binary_dilation(region, iterations=na) & ~region & skin & ~binary_dilation(any_feat & ~region, iterations=2)
            out[k] = (region, ring)
        return out

    def contrasts(self):
        img, s = self.render()
        L = srgb_to_lstar(img)
        out = {}
        for k, (region, ring) in self.masks(s).items():
            ls, lf = float(L[ring].mean()), float(L[region].mean())
            out[k] = {"contrast": (ls - lf) / (ls + lf), "L_feature": lf, "L_skin": ls, "px_feature": int(region.sum()), "px_ring": int(ring.sum())}
        return out

    def calibrate(self):
        """each feature's albedo that gives the source's contrast (bisection): the sclera's grey scale; the brows' and lips' share of
        their colour over her skin"""
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
            for _ in range(22):
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
            found[k] = {"material": mat, how: round(t, 4), "rgba": [round(float(x), 4) for x in m.mat_rgba[mid]],
                        "note": "the albedo as the scene holds it (the eye's parts carry their shade: divide it out for the maker's MAT)"}
        return found

    def close(self):
        self.r.close(); self.seg.close()


def main(calibrate=False):
    p = Photometry()
    out = {"source": RUSSELL, "studio_gain": round(p.gain, 4), "cheek_L": round(p.cheek_L(), 2), "px_per_mm": round(p.px_per_mm, 3),
           "measured": p.contrasts()}
    if calibrate:
        out["calibrated"] = p.calibrate()
        out["after"] = p.contrasts()
    p.close()
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--calibrate", action="store_true")
    a = ap.parse_args()
    print(json.dumps(main(a.calibrate), indent=1, default=float))
