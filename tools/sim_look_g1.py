"""Stills of the G1 living room (body/sim/make_g1room.py) into video/sim_look_g1/ (from the 2026-09-24 prototype; the parent's poses are scripted by IK
for the picture, the G1's state comes from the physics: it was born on the mat and settled under its own servos).
Run: nice -n 19 python3 tools/sim_look_g1.py [room kneel show eyes] [--out=DIR]"""
import math
import sys
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "body" / "sim"))
import g1acts as GA  # noqa: E402
import g1eyes as GE  # noqa: E402
import g1scene as G  # noqa: E402
import parent_kin as kin  # noqa: E402

_ARGS = dict(a[2:].split("=", 1) for a in sys.argv[1:] if a.startswith("--") and "=" in a)
OUT = Path(_ARGS["out"]) if "out" in _ARGS else ROOT / "video" / "sim_look_g1"
OUT.mkdir(parents=True, exist_ok=True)
FONT = "/System/Library/Fonts/Avenir Next.ttc"
INK, SUB, PAPER = (40, 42, 48), (95, 97, 104), (247, 246, 243)
KNEEL = dict(side="L", along=.10, off=.78)       # clear of the G1's arm and the toys (checked: no parent contact)
SHOW = dict(lateral=-.12, side="L")               # the toy held a little toward her side: her face stays unblocked
                                                  # by her own arm in both of the G1's eyes (a ray test, tmp/t_showsearch.py)


def font(sz, bold=False):
    try:
        return ImageFont.truetype(FONT, sz, index=1 if bold else 0)
    except Exception:
        return ImageFont.load_default()


def free_cam(pos, target):
    c = mujoco.MjvCamera()
    c.type = mujoco.mjtCamera.mjCAMERA_FREE
    v = np.subtract(target, pos)
    c.lookat[:] = target
    c.distance = float(np.linalg.norm(v))
    c.azimuth = math.degrees(math.atan2(-v[1], -v[0])) + 180
    c.elevation = math.degrees(math.atan2(v[2], math.hypot(v[0], v[1])))
    return c


def render(w, cam, wd=1920, ht=1080, fovy=None, shadow=8192):
    m, d = w.m, w.d
    old = m.vis.global_.fovy
    m.vis.quality.shadowsize = shadow
    if fovy:
        m.vis.global_.fovy = fovy
    r = mujoco.Renderer(m, ht, wd)
    r.update_scene(d, cam)
    img = r.render().copy()
    r.close()
    m.vis.global_.fovy = old
    return img


def label(img, text, sub=None, xy=(24, 18), size=34, color=INK, bg=None):
    im = Image.fromarray(img) if isinstance(img, np.ndarray) else img
    dr = ImageDraw.Draw(im)
    f1 = font(size, True)
    if bg is not None:
        tw = max(dr.textlength(text, font=f1), dr.textlength(sub or "", font=font(int(size * .62))))
        dr.rounded_rectangle((xy[0] - 12, xy[1] - 8, xy[0] + tw + 14, xy[1] + size + 14 + (size * .8 if sub else 0)), 10, fill=bg)
    dr.text(xy, text, font=f1, fill=color)
    if sub:
        dr.text((xy[0], xy[1] + size + 6), sub, font=font(int(size * .62)), fill=SUB)
    return im


def face_point(pose):
    hp, hR = kin.fk(pose)["head"]
    return hp + hR @ np.array([.10, 0, .135])          # between the parent's eyes and mouth


def scene_kneel(expr=1.0, touch=True):
    w = G.World(); m, d = w.m, w.d
    w.birth()
    v = GA.G1View(m, d)
    at, yaw = GA.kneel_spot(v, **KNEEL)
    p, info = GA.attend(v, at, yaw, expr=expr, touch="R" if touch else None)
    w.set_parent(p); mujoco.mj_forward(m, d)
    return w, p, v, info


def scene_show(toy="duck", expr=1.0, dist=.40):
    """The parent holds a toy 40 cm in front of the G1's cameras, along their axis, and smiles."""
    w = G.World(); m, d = w.m, w.d
    w.birth()
    v = GA.G1View(m, d)
    at, yaw = GA.kneel_spot(v, **KNEEL)
    p, info = GA.show(v, at, yaw, dist=dist, expr=expr, **SHOW)
    w.set_parent(p); mujoco.mj_forward(m, d)
    tb = m.body(f"toy_{toy}").id
    jq = m.jnt_qposadr[m.body_jntadr[tb]]
    d.qpos[jq:jq + 3] = info["toy_at"]
    d.qpos[jq + 3:jq + 7] = kin.mjquat(_duck_R(v))
    mujoco.mj_forward(m, d)
    return w, p, v, info


def _duck_R(v):
    """The duck turned so the G1 sees it in profile: its beak (+x) to the image's right, its top (+z) up the image,
    its side (+y) toward the cameras."""
    x = v.eye_R[:, 0]                                     # image right
    y = -v.face_dir                                       # toward the cameras
    z = np.cross(x, y)
    return np.column_stack([x, y, z])


def still_room():
    w, p, v, _ = scene_kneel()
    img = render(w, free_cam((2.35, -2.35, 1.6), (-.2, -.35, .38)), fovy=50)
    Image.fromarray(img).save(OUT / "room.png")


def still_kneel():
    w, p, v, info = scene_kneel()
    img = render(w, free_cam((.95, -2.05, .95), (-.18, -.3, .33)), fovy=46)
    Image.fromarray(img).save(OUT / "kneel.png")
    return info


def still_show():
    w, p, v, info = scene_show()
    ta = info["toy_at"]
    img = render(w, free_cam((.95, -2.1, .9), (-.2, -.35, .35)), fovy=46)
    Image.fromarray(img).save(OUT / "show_toy.png")
    return {k: (np.round(v_, 3).tolist() if isinstance(v_, np.ndarray) else v_) for k, v_ in info.items()}


def still_eyes():
    """The G1's two eyes as the body gets them, while the parent shows it the duck: each eye's native render (168 x 96,
    88 x 58 deg), its periphery (56 x 32), and its fovea window (32 x 32, about 21 deg) placed on the duck by the gaze
    (conjugate + vergence), outlined on the native images."""
    w, p, v, info = scene_show()
    m, d = w.m, w.d
    eyes = GE.Eyes(m, shadows=True)
    sense, imgs = eyes.tick(d)
    gz = eyes.gaze_at(d, info["toy_at"])
    sense, imgs = eyes.tick(d)
    face_gz = eyes.project(d, "L", face_point(p))
    aud = render(w, free_cam((.95, -2.1, .9), (-.2, -.35, .35)), 640, 400, fovy=46, shadow=4096)
    eyes.close()
    W_, H_ = 1920, 1080
    canvas = Image.new("RGB", (W_, H_), PAPER)
    dr = ImageDraw.Draw(canvas)
    dr.text((60, 30), "What the G1's two eyes give its body", font=font(40, True), fill=INK)
    dr.text((60, 84), "A stereo pair where its head's depth camera sits, 5 cm apart. Per eye: a periphery (the whole field, "
                      "averaged 3 x 3) and a movable fovea (a 32 x 32 window, outlined).", font=font(22), fill=SUB)
    y0 = 140
    for i, sd in enumerate("LR"):
        x0 = 60 + i * 830
        big = Image.fromarray(imgs[sd]).resize((168 * 5, 96 * 5), Image.NEAREST)
        bd = ImageDraw.Draw(big)
        cx, cy = eyes.centre_px(sd)
        h = eyes.fov_px / 2
        bd.rectangle(((cx - h) * 5, (cy - h) * 5, (cx + h) * 5, (cy + h) * 5), outline=(255, 255, 255), width=4)
        canvas.paste(big.resize((800, 457)), (x0, y0))
        dr.text((x0, y0 + 464), f"{'left' if sd == 'L' else 'right'} eye: the native render, 168 x 96 px over 88 x 58 deg",
                font=font(21), fill=SUB)
    y1 = y0 + 520
    x = 60
    for sd in "LR":
        side = "left" if sd == "L" else "right"
        canvas.paste(Image.fromarray(sense[sd]["periphery"]).resize((280, 160), Image.NEAREST), (x, y1))
        canvas.paste(Image.fromarray(sense[sd]["periphery"]), (x, y1 + 172))
        dr.text((x, y1 + 212), f"{side} periphery, 56 x 32 px\n(enlarged; true size above)", font=font(18), fill=SUB)
        canvas.paste(Image.fromarray(sense[sd]["fovea"]).resize((160, 160), Image.NEAREST), (x + 300, y1))
        canvas.paste(Image.fromarray(sense[sd]["fovea"]), (x + 300, y1 + 172))
        dr.text((x + 300, y1 + 212), f"{side} fovea,\n32 x 32 px", font=font(18), fill=SUB)
        x += 300 + 160 + 70
    canvas.paste(Image.fromarray(aud).resize((480, 300)), (W_ - 540, y1))
    dr.text((W_ - 540, y1 + 306), "the same moment, from the room", font=font(20), fill=SUB)
    gz = np.where(np.abs(gz) < .05, 0.0, gz)
    dr.text((60, H_ - 64), f"gaze on the duck: yaw {gz[0]:.1f} deg, pitch {gz[1]:.1f} deg, vergence {gz[2]:.1f} deg (the duck "
                           f"{np.linalg.norm(info['toy_at'] - v.eyes):.2f} m away). Her face: yaw {face_gz[0]:.0f}, pitch {face_gz[1]:.0f} deg in the left eye.",
            font=font(23), fill=INK)
    canvas.save(OUT / "g1_eyes.png")
    return dict(gaze=np.round(gz, 2).tolist(), face_in_left_eye_deg=np.round(face_gz, 1).tolist())


if __name__ == "__main__":
    which = [a for a in sys.argv[1:] if not a.startswith("--")] or ["room", "kneel", "show", "eyes"]
    for k in which:
        print(k, globals()[f"still_{k}"]())
