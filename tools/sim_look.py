"""A first look at the high chair world (the design of 2026-09-23, docs/SIM_DESIGN.md 5.1 and 5.8 at commit e144afa): renders body/sim/highchair_onearm.xml (the one-arm reference) into video/sim_look/.

  hero.png        1920x1080 from 'hero': the two robots side by side across the table, both turned to the viewer, smiling
  overview.png    1920x1080 from 'overview' (three-quarter, behind the child): the parent smiling at the child, the child
                  looking at its face
  side.png        1920x1080 from 'side' (profile), the same moment
  faces.png       both robots' faces (the same body): smile, neutral, frown, set by each robot's 'face' actuator
  child_eye.png   what the child's eye gives the body, twice (looking at the cube; looking at the parent's smile):
                  the 64x64 periphery (RGB and inverse depth) and the 32x32 fovea, true size and enlarged (nearest neighbour)
  demo_clip.mp4   10 s at 30 fps, simulated: the parent pushes the cube toward the child with its own arm, the child's head
                  turns to watch, the parent looks up and smiles; the child's fovea is inset top right. Both robots move the
                  way the design's bodies do: once a 150 ms tick, each joint takes one step of its alphabet (arm +-0.27, +-0.09
                  or 0 rad; gaze +-0.2, +-0.07 or 0 rad) under the servo law (target = measured angle + step); the parent's
                  steps are chosen by inverse kinematics toward scripted mitten waypoints, the heads' toward scripted looks.

Nothing here is the body, the teacher or the babbler: every choice is scripted for the look. The physics is real (the cube
moves only because the parent's mitten pushes it). Run:  nice -n 19 python3 tools/sim_look.py [hero overview side faces eye clip]"""
import math
import subprocess
import sys
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from body.sim import make_highchair_onearm as H  # noqa: E402  (the scene's numbers and the robots' kinematics)

OUT = ROOT / "video" / "sim_look"
XML = ROOT / "body" / "sim" / "highchair_onearm.xml"
FONT = "/System/Library/Fonts/Avenir Next.ttc"
SMILE, FROWN = .5, -.5        # the mouth's half-angle in radians (the face actuator's range is +-0.6)
TICK, STEPS = .15, 75         # the body's tick: 75 physics steps of 2 ms
ARM_STEPS = (-.27, -.09, 0.0, .09, .27)    # the arm's per-joint alphabet, rad per tick (SIM_DESIGN 5.3)
GAZE_STEPS = (-.2, -.07, 0.0, .07, .2)     # the gaze's per-joint alphabet
FOV_DEG = math.degrees(2 * math.atan(math.tan(math.radians(H.EYE_FOVY / 2)) * 32 / 128))   # the fovea: 16.4 deg
INK, SUB, FAINT, PAPER, EDGE = (40, 42, 48), (80, 82, 90), (110, 112, 120), (247, 246, 243), (205, 205, 210)


def font(size, bold=False):
    try:
        return ImageFont.truetype(FONT, size, index=2 if bold else 5)
    except OSError:
        return ImageFont.load_default()


# ---------------------------------------------------------------- posing (kinematic, for the stills)
def setq(m, d, joint, v):
    d.qpos[m.jnt_qposadr[m.joint(joint).id]] = v
    try:
        d.ctrl[m.actuator(joint).id] = v
    except KeyError:
        pass


def getq(m, d, joint):
    return float(d.qpos[m.jnt_qposadr[m.joint(joint).id]])


def set_face(m, d, p, v, kinematic=True):
    """The face: v > 0 smiles, v < 0 frowns (the mouth's two outer capsules turn by +-v)."""
    d.ctrl[m.actuator(f"{p}face").id] = v
    if kinematic:
        setq(m, d, f"{p}mouth", v); setq(m, d, f"{p}mouth_R", -v)


def face_point(m, d, p):
    """The world point in the middle of robot p's face (between the eyes and the mouth)."""
    hid = m.body(f"{p}head").id
    return d.xpos[hid] + d.xmat[hid].reshape(3, 3) @ np.array([0, H.FACE_Y, 0])


def look(m, d, p, target):
    pan, tilt = H.head_lookat(p, target)
    setq(m, d, f"{p}pan", pan); setq(m, d, f"{p}tilt", tilt)


def face_to_face(m, d):
    for _ in range(3):   # each head's face moves as it turns, so settle the two look-ats together
        mujoco.mj_forward(m, d)
        look(m, d, "parent_", face_point(m, d, "child_")); look(m, d, "child_", face_point(m, d, "parent_"))
    mujoco.mj_forward(m, d)


def birth(m, d):
    mujoco.mj_resetDataKeyframe(m, d, m.key("birth").id)
    mujoco.mj_forward(m, d)


# ---------------------------------------------------------------- stills
def stills(m, d, which):
    r = mujoco.Renderer(m, 1080, 1920)
    for cam in which:
        birth(m, d)
        if cam == "hero":      # a portrait: both turn to the viewer and smile
            setq(m, d, "child_pan", -.70); setq(m, d, "child_tilt", -.10)
            setq(m, d, "parent_pan", .70); setq(m, d, "parent_tilt", -.10)
            set_face(m, d, "child_", SMILE); set_face(m, d, "parent_", SMILE)
        else:                  # the moment the design is built round: the child looks at the parent's face, the parent smiles
            face_to_face(m, d)
            set_face(m, d, "parent_", SMILE)
        mujoco.mj_forward(m, d)
        r.update_scene(d, camera=cam)
        Image.fromarray(r.render()).save(OUT / f"{cam}.png")
        print("wrote", OUT / f"{cam}.png")
    r.close()


def faces(m, d):
    T = 560
    r = mujoco.Renderer(m, T, T)
    opt = mujoco.MjvOption(); opt.geomgroup[2] = 0          # the print and the sideboard hidden: plain wall behind both rows
    rows = []
    for p in ("child_", "parent_"):
        row = []
        for v in (SMILE, 0.0, FROWN):
            birth(m, d)
            for j in ("pan", "tilt"):
                setq(m, d, f"{p}{j}", 0.0)
            set_face(m, d, p, v)
            mujoco.mj_forward(m, d)
            hid = m.body(f"{p}head").id
            fwd = d.xmat[hid].reshape(3, 3)[:, 1]
            cam = mujoco.MjvCamera()
            cam.lookat[:] = d.xpos[hid] + np.array([0, 0, -.012])
            cam.distance = .40
            cam.azimuth = math.degrees(math.atan2(-fwd[1], -fwd[0])) + 12
            cam.elevation = -6
            r.update_scene(d, camera=cam, scene_option=opt)
            row.append(r.render().copy())
        rows.append(row)
    r.close()
    pad, top, left = 24, 140, 150
    W = left + 3 * T + 4 * pad
    Hh = top + 2 * T + 3 * pad + 40
    sheet = Image.new("RGB", (W, Hh), PAPER)
    dr = ImageDraw.Draw(sheet)
    dr.text((left + pad, 26), "Both faces: smile, neutral, frown (one 'face' actuator each)", font=font(34, True), fill=INK)
    for j, lab in enumerate((f"smile  (face = +{SMILE})", "neutral  (0)", f"frown  ({FROWN})")):
        x = left + pad + j * (T + pad)
        dr.text((x + 8, top - 42), lab, font=font(26), fill=(70, 72, 80))
    for i, (row, lab) in enumerate(zip(rows, ("child", "parent"))):
        y = top + i * (T + pad)
        dr.text((22, y + T // 2 - 20), lab, font=font(34, True), fill=INK)
        for j, im in enumerate(row):
            sheet.paste(Image.fromarray(im), (left + pad + j * (T + pad), y))
    dr.text((left + pad, Hh - 52), "The parent's face is the child's reward, felt only while the child is looking at it. "
            "The child's face shows the reward it expects.", font=font(22), fill=(95, 97, 105))
    sheet.save(OUT / "faces.png")
    print("wrote", OUT / "faces.png")


def eye_frames(m, d):
    """One 128 px render from 'child_eye' -> (periphery RGB 64x64, inverse depth 64x64, fovea RGB 32x32, segmentation 32x32)."""
    rgb = mujoco.Renderer(m, 128, 128)
    dep = mujoco.Renderer(m, 128, 128); dep.enable_depth_rendering()
    seg = mujoco.Renderer(m, 128, 128); seg.enable_segmentation_rendering()
    rgb.update_scene(d, camera="child_eye"); im = rgb.render().astype(np.float32)
    dep.update_scene(d, camera="child_eye"); z = dep.render()
    seg.update_scene(d, camera="child_eye"); sg = seg.render()
    rgb.close(); dep.close(); seg.close()
    per = im.reshape(64, 2, 64, 2, 3).mean((1, 3))
    inv = np.clip(.3 / np.maximum(z, .05), 0, 1).reshape(64, 2, 64, 2).mean((1, 3))   # 1 at 0.3 m, capped
    fov = im[48:80, 48:80]
    return per.astype(np.uint8), (inv * 255).astype(np.uint8), fov.astype(np.uint8), sg[48:80, 48:80]


def parent_face_pixels(m, sg):
    """(face, head): fovea pixels showing the parent's face (the screen, its eyes and mouth) and its whole head (with frame and ears)."""
    geom = np.where(sg[..., 1] == int(mujoco.mjtObj.mjOBJ_GEOM), sg[..., 0], -1)
    face_ids = {m.geom(n).id for n in ("parent_screen", "parent_eye_L", "parent_eye_R", "parent_mouth_mid", "parent_mouth_L", "parent_mouth_R")}
    head_bodies = {m.body(n).id for n in ("parent_head", "parent_mouth_L", "parent_mouth_R")}
    head_ids = {g for g in range(m.ngeom) if m.geom_bodyid[g] in head_bodies}
    return int(np.isin(geom, list(face_ids)).sum()), int(np.isin(geom, list(head_ids)).sum())


def child_eye(m, d):
    rows = []
    birth(m, d)
    look(m, d, "child_", d.xpos[m.body("cube").id].copy())
    mujoco.mj_forward(m, d)
    rows.append(("looking at the cube", eye_frames(m, d)))
    face_to_face(m, d)
    set_face(m, d, "parent_", SMILE)
    mujoco.mj_forward(m, d)
    rows.append(("looking at the parent's face (smiling)", eye_frames(m, d)))
    face_px, head_px = parent_face_pixels(m, rows[1][1][3])
    print(f"  the parent's face fills {face_px} of the fovea's 1024 pixels (its whole head {head_px}); the 'facing' test is 20")

    big = 448                                   # periphery x7, fovea x14
    pad, left, top, rowh = 40, 40, 120, 448 + 150
    W = left + 64 + pad + big + pad + big + pad + 32 + pad + big + left + 30
    Hh = top + len(rows) * rowh + 70
    sheet = Image.new("RGB", (W, Hh), PAPER)
    dr = ImageDraw.Draw(sheet)
    dr.text((left, 26), "The child's eye, as the body gets it: one 128 px render (60°) per tick, split in two", font=font(34, True), fill=INK)

    def panel(im, x, y):
        sheet.paste(im, (x, y))
        dr.rectangle([x - 1, y - 1, x + im.width, y + im.height], outline=EDGE, width=1)

    for i, (lab, (per, inv, fov, _)) in enumerate(rows):
        y = top + i * rowh
        dr.text((left, y), lab, font=font(28, True), fill=(60, 62, 70))
        y += 50
        x = left
        panel(Image.fromarray(per), x, y); dr.text((x, y + 70), "true size", font=font(18), fill=FAINT)
        x += 64 + pad
        pim = Image.fromarray(per).resize((big, big), Image.NEAREST)
        pd = ImageDraw.Draw(pim)
        a, b = 24 * big // 64, 40 * big // 64     # the fovea: the centre 16 of the periphery's 64 pixels
        pd.rectangle([a, a, b, b], outline=(255, 255, 255), width=2)
        panel(pim, x, y)
        dr.text((x, y + big + 8), f"periphery: 64 x 64 RGB over 60° (shown x7)\nthe white box is where the fovea's {FOV_DEG:.1f}° fall",
                font=font(20), fill=SUB)
        x += big + pad
        panel(Image.fromarray(inv).convert("RGB").resize((big, big), Image.NEAREST), x, y)
        dr.text((x, y + big + 8), "periphery: inverse depth, capped (shown x7)\nnear is bright", font=font(20), fill=SUB)
        x += big + pad
        panel(Image.fromarray(fov), x, y); dr.text((x - 4, y + 38), "true\nsize", font=font(18), fill=FAINT)
        x += 32 + pad
        panel(Image.fromarray(fov).resize((big, big), Image.NEAREST), x, y)
        note = f"fovea: the centre 32 x 32 RGB over {FOV_DEG:.1f}° (shown x14)\nthe render's full resolution"
        if i == 1:
            note = (f"fovea: the centre 32 x 32 RGB over {FOV_DEG:.1f}° (shown x14)\n"
                    f"a smile counts only while the face is here\n(it fills {face_px} of 1024 pixels; the test is 20)")
        dr.text((x, y + big + 8), note, font=font(20), fill=SUB)
    dr.text((left, Hh - 50), "Each is coded by a born (fixed, random) retinotopic code of 8 x 8 cells x 6 colour mixes; world truth never enters the body.",
            font=font(22), fill=(95, 97, 105))
    sheet.save(OUT / "child_eye.png")
    print("wrote", OUT / "child_eye.png")


# ---------------------------------------------------------------- the clip (simulated)
def minjerk(a, b, s):
    s = min(max(s, 0.0), 1.0)
    return np.asarray(a) + (np.asarray(b) - np.asarray(a)) * (10 * s ** 3 - 15 * s ** 4 + 6 * s ** 5)


def path(keys, t):
    """keys: [(time, value)], min-jerk between successive keys, held outside."""
    if t <= keys[0][0]:
        return np.asarray(keys[0][1], float)
    for (t0, v0), (t1, v1) in zip(keys, keys[1:]):
        if t <= t1:
            return minjerk(v0, v1, (t - t0) / (t1 - t0))
    return np.asarray(keys[-1][1], float)


def servo_step(m, d, joint, want, alphabet):
    """The servo law with one act of the alphabet: target = measured angle + the step nearest the remaining error."""
    q = getq(m, d, joint)
    step = min(alphabet, key=lambda a: abs(want - q - a))
    lo, hi = m.actuator_ctrlrange[m.actuator(joint).id]
    d.ctrl[m.actuator(joint).id] = float(np.clip(q + step, lo, hi))
    return step


def free_cam(pos, at):
    """An MjvCamera at pos looking at at (for the camera's slow move)."""
    v = np.subtract(at, pos); dist = float(np.linalg.norm(v))
    cam = mujoco.MjvCamera(); cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat[:] = at; cam.distance = dist
    cam.azimuth = math.degrees(math.atan2(v[1], v[0])); cam.elevation = math.degrees(math.asin(v[2] / dist))
    return cam


def clip(m, d, seconds=10.0, fps=30):
    birth(m, d)
    face_to_face(m, d)
    cube = m.body("cube").id
    c0 = d.xpos[cube].copy()
    start = {k: d.xpos[m.body(k).id].copy() for k in H.OBJ_POS}
    half = .033 + H.MITTEN_R
    rest = np.array(H.BIRTH["parent_"]["mitten"])
    stage = c0[:2] + np.array([0, half + .03])            # behind the cube, on the parent's side
    push_to = c0[:2] + np.array([0, half - .13])          # the mitten ends 13 cm past first contact: about 12 cm of push
    mitten = [(0, rest), (1.0, rest), (2.6, stage), (3.0, stage), (6.4, push_to), (6.8, push_to), (8.2, rest)]   # then home
    SMILE_AT, P_UP, C_UP = 7.9, 6.6, 6.9
    captions = [(0.0, 3.0, "The parent robot reaches for the cube with its own arm"),
                (3.0, 6.6, "It pushes the cube toward the child. The child watches."),
                (6.6, 10.0, "The parent looks up and smiles. The child sees it.")]
    # the camera: a slow move in from the three-quarter view while the parent works, then over the child's shoulder
    cams = [((1.30, -.70, .60), (1.12, -.60, .52), (-.05, .03, .17), 34, 0.0, P_UP),
            ((.70, -1.10, .80), (.665, -1.035, .77), (-.05, .15, .19), 28, P_UP, seconds)]   # 'shoulder', moving in a little
    W, Hh = 1920, 1080
    r = mujoco.Renderer(m, Hh, W)
    eye = mujoco.Renderer(m, 128, 128)
    ff = subprocess.Popen(["nice", "-n", "19", "ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
                           "-s", f"{W}x{Hh}", "-r", str(fps), "-i", "-", "-c:v", "libx264", "-preset", "slow", "-crf", "18",
                           "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(OUT / "demo_clip.mp4")], stdin=subprocess.PIPE)
    cf, lf = font(30), font(22, True)
    log, gaze_steps, arm_steps = [], [], []
    tick = -1
    fovy0 = m.vis.global_.fovy
    for i in range(int(seconds * fps)):
        t = i / fps
        while d.time < t - 1e-9:
            k = int(d.time / TICK + 1e-9)
            if k != tick:            # a tick boundary: each robot chooses its acts for the next 150 ms
                tick = k; tt = k * TICK
                q1, q2 = H.arm_ik("parent_", path(mitten, tt + TICK), elbow=-1)
                arm_steps += [servo_step(m, d, "parent_shoulder", q1, ARM_STEPS), servo_step(m, d, "parent_elbow", q2, ARM_STEPS)]
                hand = d.xpos[m.body("parent_hand").id].copy(); hand[2] = .05
                cpos = d.xpos[cube].copy()
                p_target = cpos if .8 <= tt < P_UP else face_point(m, d, "child_")
                c_target = face_point(m, d, "parent_") if tt < 1.2 or tt >= C_UP else (hand if tt < 2.9 else cpos)
                for p, tgt in (("parent_", p_target), ("child_", c_target)):
                    pan, tilt = H.head_lookat(p, tgt)
                    gaze_steps += [servo_step(m, d, f"{p}pan", pan, GAZE_STEPS), servo_step(m, d, f"{p}tilt", tilt, GAZE_STEPS)]
                d.ctrl[m.actuator("parent_face").id] = SMILE if tt >= SMILE_AT - 1e-9 else 0.0
            mujoco.mj_step(m, d)
        # the camera for this frame
        (p0, p1, at, fovy, a, b) = next(c for c in cams if c[4] <= t < c[5])
        s = (t - a) / (b - a); s = s * s * (3 - 2 * s)
        m.vis.global_.fovy = fovy
        r.update_scene(d, camera=free_cam(np.add(p0, s * np.subtract(p1, p0)), at))
        frame = Image.fromarray(r.render())
        # the child's fovea, inset top right: the centre 32 x 32 of its eye's 128 px render, shown x8
        eye.update_scene(d, camera="child_eye")
        fov = Image.fromarray(eye.render()[48:80, 48:80]).resize((256, 256), Image.NEAREST)
        dr = ImageDraw.Draw(frame, "RGBA")
        fx, fy = W - 256 - 56, 56
        dr.rounded_rectangle([fx - 14, fy - 14, fx + 256 + 14, fy + 256 + 52], radius=14, fill=(20, 22, 28, 140))
        dr.rectangle([fx - 2, fy - 2, fx + 256 + 1, fy + 256 + 1], fill=(255, 255, 255, 255))
        frame.paste(fov, (fx, fy))
        lab = "the child's fovea"
        dr.text((fx + (256 - dr.textlength(lab, font=lf)) / 2, fy + 262), lab, font=lf, fill=(255, 255, 255, 255))
        cap = next((c for a, b, c in captions if a <= t < b), None)
        if cap:
            tw = dr.textlength(cap, font=cf)
            dr.rounded_rectangle([60, Hh - 110, 60 + tw + 48, Hh - 50], radius=14, fill=(20, 22, 28, 150))
            dr.text((84, Hh - 104), cap, font=cf, fill=(255, 255, 255, 255))
        ff.stdin.write(np.asarray(frame).tobytes())
        if i % 30 == 0:
            log.append((round(t, 1), np.round(d.xpos[cube][:2], 3).tolist(), np.round(d.xpos[m.body("parent_hand").id][:2], 3).tolist()))
        if SCRATCH and i % 15 == 0:                           # frames kept for checking (--frames=DIR)
            frame.save(SCRATCH / f"clip_{t:04.1f}.png")
    ff.stdin.close(); ff.wait(); r.close(); eye.close()
    m.vis.global_.fovy = fovy0
    moved = d.xpos[cube][:2] - c0[:2]
    print("wrote", OUT / "demo_clip.mp4", f"| the cube moved {100 * np.linalg.norm(moved):.1f} cm ({moved.round(3).tolist()})")
    for row in log:
        print("  t, cube xy, parent mitten xy:", row)
    print("  every object's displacement (cm):", {k: round(100 * float(np.linalg.norm(d.xpos[m.body(k).id] - v)), 1) for k, v in start.items()})
    print("  arm acts used:", {a: arm_steps.count(a) for a in ARM_STEPS}, "| gaze acts used:", {a: gaze_steps.count(a) for a in GAZE_STEPS})


SCRATCH = None


def main():
    global SCRATCH
    OUT.mkdir(parents=True, exist_ok=True)
    args = [a for a in sys.argv[1:] if not a.startswith("--frames=")]
    for a in sys.argv[1:]:
        if a.startswith("--frames="):
            SCRATCH = Path(a.split("=", 1)[1]); SCRATCH.mkdir(parents=True, exist_ok=True)
    want = set(args) or {"hero", "overview", "side", "faces", "eye", "clip"}
    m = mujoco.MjModel.from_xml_path(str(XML)); d = mujoco.MjData(m)
    if want & {"hero", "overview", "side"}:
        stills(m, d, [c for c in ("hero", "overview", "side") if c in want])
    if "faces" in want:
        faces(m, d)
    if "eye" in want:
        child_eye(m, d)
    if "clip" in want:
        clip(m, d)


if __name__ == "__main__":
    main()
