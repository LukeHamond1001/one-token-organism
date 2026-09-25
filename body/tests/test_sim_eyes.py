"""the G1's eyes (docs/SIM_DESIGN.md 3.4, 3.5, 3.7 and A1; the build plan's W3). Run: python3 -m body.tests.test_sim_eyes (at nice -n 19;
it renders, so it needs the Mac's GL). THE GAZE: the software fovea's effector moves both windows by its declared steps (conjugate
yaw and pitch, vergence), held in reach (each window inside its image, the vergence within the near point); its state and velocity
are the body channel's last six numbers. THE VOR: the window's ray counter-turned by the gyro's rotation, exactly the world-fixed
direction under any constant rotation (the slow phase); its QUICK PHASE (A23; the W1 verifier's third finding): a window carried to
its reach jumps back by half the reach in the direction of the turn, never sits pinned at the edge, in a synthetic turn and with the
trunk turned by the waist in the world; in the world, a far point fixated before the trunk turns stays fixated. THE EYES: two
native renders, the periphery 3 x 3 averaged, the fovea window where the gaze puts it (a ball aimed at fills its centre, and leaves
it when the gaze turns away), the retina's opponent ON/OFF code equal to its hand computation. THE FACE TEST (A1): the parent's
face in front and facing passes, and each of its four conditions fails it alone (out of the window, blocked by a toy, turned away,
too far); it stays in the truth, never in the body's channels (the W1 verifier's second finding). THE BORN FACE TEMPLATE: CONSPEC's
three dark blobs fire on a drawn face at every size of the bank and anywhere in the fovea, not on the same face with light blobs
(Farroni's polarity) nor on noise; the frame's face_fovea and face_periph are the template on the frame's own pixels, and the body's
channels are the sensors' alone. THE EXACT REPLAY with the eyes: every frame's eye codes bit for bit after a save and a restore, in
the same world and in a new one. NO LAMP AT THE EYES (the W1 verifier's fifth finding): the room has no headlight and the eyes refuse
one; what they see is rendered again when any light's term changes (the ambient, the specular, a light moved: its tenth finding);
with the room's lights off the eyes see the dark. THE PARENT'S FACE AS DRAWN (its first finding): at birth her face is drawn, so
in the fovea her mouth and her eyes are darker than her cheeks; the born template's reading of it is printed (C3 measures it).
HER FACE (the W1 verifier's third and fourth rounds, the owner's decision): of human proportions MEASURED ON THE DRAWN GEOMETRY, each
against its cited adult female norm; with a real face's photometry (Russell, Kramer and Jones 2017) as the source photographed its
faces, her shading in; her graded expressions working on the drawn face; the born template, unchanged, measured on her face at 0.3-2
m and written down, its matches told apart as detections of her face or chance matches (a finding, no bar); her collision her drawn
face (a hand stops at it); her face smooth, whole and clean (one sheet, its own normals, the lips one tube each, the brows on her skin,
the lids never through it)."""
import math
import os
import sys
import time

import mujoco
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))
sys.path.insert(0, os.path.join(ROOT, "body", "sim"))
from body.sim import eyes as E  # noqa: E402
from body.sim import world as W  # noqa: E402

G = W.G
kin = G.kin


def _world(extra=None, shadows="sun"):
    w = W.G1World(seed=1, extra=extra)
    return w, E.Eyes(w, shadows=shadows)


def _rot(axis, a):
    axis = np.asarray(axis, float) / np.linalg.norm(axis)
    K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return np.eye(3) + math.sin(a) * K + (1 - math.cos(a)) * K @ K


def _ray_angle(w, side, point):
    """the angle (deg) between eye `side`'s window-centre ray and the direction to a world point"""
    m, d = w.m, w.d
    c = m.camera(f"eye_{side}").id
    yaw = w.gaze[0] + (w.gaze[2] / 2 if side == "L" else -w.gaze[2] / 2)
    r = np.array([math.tan(yaw), math.tan(w.gaze[1]), -1.0]); r /= np.linalg.norm(r)
    rw = d.cam_xmat[c].reshape(3, 3) @ r
    to = np.asarray(point) - d.cam_xpos[c]; to /= np.linalg.norm(to)
    return math.degrees(math.acos(min(1.0, float(rw @ to))))


def test_the_gaze():
    """eyes 1: the gaze's acts move the windows by the declared steps; the gaze held in reach; the body channel carries it"""
    w = W.G1World(seed=1)
    assert W.EFFECTOR_NAMES[0] == "gaze" and W.EFFECTOR_FACTORS["gaze"] == [5, 5, 5] and W.EFFECTOR_REST["gaze"] == 62
    assert abs(math.degrees(W.GAZE_REACH_YAW) - 38.1) < 0.1 and abs(math.degrees(W.GAZE_REACH_PITCH) - 20.3) < 0.1
    assert abs(math.degrees(W.VERG_MAX) - 11.42) < 0.01
    w.apply({"gaze": W.act_flat([3, 1, 3])})                           # yaw +small, pitch -small, vergence +small
    assert np.allclose(w.gaze[1:], [-0.07, 0.03], atol=2e-3) and abs(w.gaze[0] - 0.07) < 2e-3, w.gaze   # (the VOR moves it a little)
    f = w.frame()
    assert f.obs["body"].shape == (242,) and np.array_equal(f.obs["body"][215:218], w.gaze)     # S5a: the anatomy's body channel
    assert np.allclose(f.obs["body"][218:221], w.gaze / 0.15, atol=0.02)
    for _ in range(10):
        w.apply({"gaze": W.act_flat([4, 4, 4])})
    assert abs(w.gaze[2] - W.VERG_MAX) < 1e-12 and abs(w.gaze[1] - W.GAZE_REACH_PITCH) < 0.01      # (the VOR's small turn within)
    ry = W.GAZE_REACH_YAW - W.VERG_MAX / 2                              # at the reach, or jumped back by the VOR's quick phase (a
    assert ry / 2 - 0.01 <= w.gaze[0] <= ry                              # head swaying on softer servos, A39), never past it
    for side in "LR":                                                   # both windows inside their images at the reach
        x0, y0 = E.window_corner(side, w.gaze)
        cx, cy = E.window_centre(side, w.gaze)
        assert abs(cx - (x0 + 16)) <= 0.5 and abs(cy - (y0 + 16)) <= 0.5, (side, cx, cy, x0, y0)
    try:
        w.apply({"gaze": 125})
    except ValueError:
        pass
    else:
        raise AssertionError("a gaze act out of range was taken")
    print(f"eyes 1: the gaze moves both windows by its steps (+-4 / +-11.5 deg, vergence +-1.7 / +-5.7), held within +-38.1 x +-20.3",
          f"deg and vergence 0-11.4 deg (the near point at 25 cm, A23), both windows inside their images at the reach; the body channel's",
          f"last six numbers are the gaze and its velocity")


def test_the_vor_exact():
    """eyes 2: the VOR's ray is exactly the world-fixed direction seen from the turned head, for any constant rotation"""
    rng = np.random.default_rng(0)
    worst = 0.0
    for _ in range(200):
        yaw, pitch = rng.uniform(-0.6, 0.6), rng.uniform(-0.3, 0.3)
        omega = rng.normal(0, 1.5, 3)                                   # rad/s in the camera frame
        n, dt = 75, 0.002
        y2, p2, _ = W.vor(yaw, pitch, np.tile(omega, (n, 1)), dt)
        r = np.array([math.tan(yaw), math.tan(pitch), -1.0]); r /= np.linalg.norm(r)
        a = float(np.linalg.norm(omega)) * n * dt
        R = _rot(omega, a)                                              # the head turned by omega over the tick
        rn = R.T @ r                                                    # the same world direction in the turned camera's frame
        want = (math.atan2(rn[0], -rn[2]), math.atan2(rn[1], -rn[2]))
        worst = max(worst, abs(y2 - want[0]), abs(p2 - want[1]))
    assert worst < 1e-9, worst
    assert np.allclose(W.vor(0.3, -0.1, np.zeros((75, 3)), 0.002)[:2], (0.3, -0.1), atol=1e-12, rtol=0)
    print(f"eyes 2: the VOR's window ray equals the world-fixed direction under 200 random head rotations to {worst:.1e} rad")


def test_the_vor_in_the_world():
    """eyes 3: a far point fixated, then the trunk turned by the waist: with the VOR the window stays on it; without, it slides"""
    w = W.G1World(seed=1)
    m, d = w.m, w.d
    c = m.camera("eye_L").id
    P = d.cam_xpos[c] + d.cam_xmat[c].reshape(3, 3) @ np.array([0.2, 0.1, -1.0]) * 3.0      # 3 m out, a little off the axis
    w.gaze = W.clamp_gaze(E.gaze_at(m, d, P))
    held = w.gaze.copy()
    R0 = d.cam_xmat[c].reshape(3, 3).copy()
    turn = W.act_flat([4, 2, 3])                                         # the waist: yaw +big, pitch +small
    for _ in range(10):                                                  # (Unitree's waist gain turns it slower: A39)
        w.apply({"waist": turn, "gaze": W.EFFECTOR_REST["gaze"]})
    err_vor = _ray_angle(w, "L", P)
    R1 = d.cam_xmat[c].reshape(3, 3)
    r = np.array([math.tan(held[0] + held[2] / 2), math.tan(held[1]), -1.0]); r /= np.linalg.norm(r)
    to = P - d.cam_xpos[c]; to /= np.linalg.norm(to)
    err_fixed = math.degrees(math.acos(min(1.0, float((R1 @ r) @ to))))
    turned = math.degrees(math.acos(min(1.0, (np.trace(R0.T @ R1) - 1) / 2)))
    assert turned > 4 and err_fixed > 1.5 and err_vor < 0.25 * err_fixed and err_vor < 3.0, (turned, err_fixed, err_vor)   # (the lying
    # trunk turned against the mat by Unitree's softer waist: A39)
    print(f"eyes 3: the trunk turned {turned:.0f} deg by the waist in 1.5 s: a window held fixed would be {err_fixed:.1f} deg off the",
          f"point 3 m away; with the VOR it is {err_vor:.2f} deg off (the rest: the eyes' translation and roll)")


def test_the_eyes_render():
    """eyes 4: two native renders; the periphery; the fovea where the gaze puts it; the retina's code by hand"""
    w, ey = _world()
    m, d = w.m, w.d
    f = w.frame()
    t = f.truth["eyes"]
    assert f.obs["eye_p"].shape == (E.EYE_P_SIZE,) == (336,) and f.obs["eye_f"].shape == (E.EYE_F_SIZE,) == (768,)
    for s in "LR":
        img = t["images"][s]
        assert img.shape == (96, 168, 3) and img.dtype == np.uint8 and img.std() > 5
        assert np.array_equal(t["periphery"][s], img.reshape(32, 3, 56, 3, 3).mean(axis=(1, 3)).round().astype(np.uint8))
    # the retina's code by hand, one cell
    per = t["periphery"]["L"].astype(float) / 255
    cell = per[8:16, 16:24].reshape(-1, 3).mean(0)
    L_, RG, BY = cell.mean() - 0.5, cell[0] - cell[1], cell[2] - (cell[0] + cell[1]) / 2
    want = [max(L_, 0), max(-L_, 0), max(RG, 0), max(-RG, 0), max(BY, 0), max(-BY, 0)]
    k = (1 * 7 + 2) * 6                                                  # row 1, column 2 of the left periphery's 4 x 7 cells
    assert np.allclose(f.obs["eye_p"][k:k + 6], want), (f.obs["eye_p"][k:k + 6], want)
    # the ball held 0.5 m out: aimed at, it fills the window's centre; the gaze turned away, it leaves
    c = m.camera("eye_L").id
    P = d.cam_xpos[c] + d.cam_xmat[c].reshape(3, 3) @ np.array([0.25, -0.1, -1.0]) * 0.5
    a = m.jnt_qposadr[m.body_jntadr[m.body("toy_ball").id]]
    d.qpos[a:a + 3] = P; mujoco.mj_forward(m, d)
    w.gaze = W.clamp_gaze(E.gaze_at(m, d, P))
    fov = w.frame().truth["eyes"]["fovea"]
    red = lambda im: float(np.mean((im[..., 0] > 20) & (im[..., 0].astype(float) > 0.6 * im.astype(float).sum(-1))))   # red-dominant (the room's own light)
    assert red(fov["L"][12:20, 12:20]) > 0.9 and red(fov["R"][12:20, 12:20]) > 0.9, (red(fov["L"][12:20, 12:20]), red(fov["R"][12:20, 12:20]))
    w.gaze = W.clamp_gaze(w.gaze + [-0.5, 0.2, 0])
    fov2 = w.frame().truth["eyes"]["fovea"]
    assert red(fov2["L"]) < 0.05, red(fov2["L"])
    ey.close()
    print(f"eyes 4: two 168 x 96 renders, the periphery 56 x 32, the retina's code (2 x 168 + 2 x 384) equal to its hand computation;",
          f"a ball 0.5 m out aimed at fills both windows' centres ({100 * red(fov['L'][12:20, 12:20]):.0f}% red), and leaves when the gaze",
          f"turns away; render {1e3 * ey.timing['render_s'] / ey.timing['renders']:.0f} ms a tick for both eyes (the sun's shadow)")


def _face_rig(w, dist=0.6, turn_deg=0.0):
    """the parent's head put `dist` out on the left eye's axis, facing it (turned by turn_deg about her own up axis)"""
    m, d = w.m, w.d
    c = m.camera("eye_L").id
    p = d.cam_xpos[c] + d.cam_xmat[c].reshape(3, 3) @ np.array([0, 0, -1.0]) * dist
    x = d.cam_xpos[c] - p; x /= np.linalg.norm(x)
    u = np.array([-1.0, 0, 0])                                          # her crown toward the child's head: leaning over its chest from
                                                                        # its feet's side, her face upright in its eyes
    z = u - x * float(x @ u); z /= np.linalg.norm(z)
    x = _rot(z, math.radians(turn_deg)) @ x
    Rh = np.column_stack([x, np.cross(z, x), z])
    local = np.array([kin.head_surface_x(0, kin.MOUTH_Z) + .0015, 0.0, kin.MOUTH_Z])
    w.scene.place_head(p - Rh @ local, Rh)                              # her head there, her body straight under it (a still)
    mujoco.mj_forward(m, d)
    mouth = E.mouth_point(m, d)[0]
    w.gaze = W.clamp_gaze(E.gaze_at(m, d, mouth))
    return mouth


def test_the_face_test():
    """eyes 5: A1: the parent's face in front and facing passes; each condition alone fails it"""
    w, ey = _world()
    m, d = w.m, w.d
    mouth = _face_rig(w, 0.6)
    f = w.frame()
    ft = f.truth["eyes"]["face_test"]
    assert ft["L"] == (True, "") and ft["R"] == (True, "") and "face_seen" not in f.obs, (ft, sorted(f.obs))
    w.gaze = W.clamp_gaze(w.gaze + [0.4, 0, 0])
    assert E.face_test(m, d, w.gaze)["L"] == (False, "not in the fovea") and w.frame().truth["eyes"]["face_test"]["L"][0] is False
    _face_rig(w, 0.6, turn_deg=80)
    assert E.face_test(m, d, w.gaze)["L"] == (False, "turned away")
    _face_rig(w, 0.6, turn_deg=88)                                      # the ray meets her own lip's corner 2 cm before the mouth
    assert E.face_test(m, d, w.gaze)["L"] == (False, "turned away")     # point: her mouth, not a block (C2 in the W1 fix 2)
    _face_rig(w, 3.4)                                                   # (the ceiling is 3.8 m out along this axis)
    assert E.face_test(m, d, w.gaze)["L"] == (False, "too small")
    _face_rig(w, 2.5)
    assert E.face_test(m, d, w.gaze)["L"] == (True, "")                  # 2.5 m facing: still seen (A1's range: about 3 m)
    mouth = _face_rig(w, 0.6)
    a = m.jnt_qposadr[m.body_jntadr[m.body("toy_block").id]]
    c = m.camera("eye_L").id
    d.qpos[a:a + 3] = d.cam_xpos[c] + (mouth - d.cam_xpos[c]) * 0.5; mujoco.mj_forward(m, d)
    assert E.face_test(m, d, w.gaze)["L"] == (False, "blocked")
    ey.close()
    print("eyes 5: the face test: the parent's face 0.6 m out and facing is seen by both eyes (in the truth only); out of the window,",
          "turned 80 deg, 3.4 m away or behind a block it is not (each condition alone; turned 88 deg, where the ray meets her own",
          "lip first, it is turned away, not blocked); at 2.5 m facing it is")


def test_the_vor_quick_phase():
    """eyes 7: A23: the VOR's quick phase. A synthetic steady turn: the window's slow phase carries it to its reach, where it jumps
    back by half the reach in the direction of the turn, and it never passes the reach; in the world, the trunk turned by the waist
    with the window near its edge: never pinned at the reach, the quick phases in the truth"""
    reach = (W.GAZE_REACH_YAW, W.GAZE_REACH_PITCH)
    for axis, k in ((np.array([0.0, 1.0, 0.0]), 0), (np.array([1.0, 0.0, 0.0]), 1)):
        for sgn in (1.0, -1.0):
            g = [0.0, 0.0]; quick = 0; jumps = []
            for _ in range(60 * 75):                                    # 60 ticks of a 1.2 rad/s turn, sample by sample
                prev = g[k]
                y, p, q = W.vor(g[0], g[1], (sgn * 1.2 * axis)[None, :], 0.002, reach=reach)
                g = [y, p]; quick += q
                assert abs(g[0]) <= reach[0] + 1e-12 and abs(g[1]) <= reach[1] + 1e-12, g
                if q:
                    jumps.append((prev, g[k]))
            y60, p60, q60 = W.vor(0.0, 0.0, np.tile(sgn * 1.2 * axis, (60 * 75, 1)), 0.002, reach=reach)
            assert q60 == quick and abs([y60, p60][k] - g[k]) < 1e-9, (q60, quick)       # a whole run in one call: the same
            assert quick >= 3, (axis, sgn, quick)
            drift = np.sign(jumps[0][0])                                # the slow phase drifts toward this edge
            for a_, b_ in jumps:                                        # each jump: from the edge back by half the reach (less a sample's drift)
                assert np.sign(b_ - a_) == -drift and abs(a_) > reach[k] - 0.003 and abs(abs(b_) - reach[k] / 2) < 0.003, (a_, b_)
    # in the world: the window near its edge, the trunk turned by the waist
    pinned, quicks, turned = 0, 0, 0.0
    for start in (+1, -1):
        w = W.G1World(seed=1)
        c = w.m.camera("eye_L").id
        R0 = w.d.cam_xmat[c].reshape(3, 3).copy()
        w.gaze = np.array([start * (W.GAZE_REACH_YAW - 0.02), 0.0, 0.0])
        for _ in range(16):                                              # (Unitree's waist gain: A39)
            w.apply({"waist": W.act_flat([4, 2, 2]), "gaze": W.EFFECTOR_REST["gaze"]})
            quicks += w.frame().truth["vor_quick"]
            pinned += int(abs(w.gaze[0]) >= W.GAZE_REACH_YAW - 1e-4)
        R1 = w.d.cam_xmat[c].reshape(3, 3)
        turned = max(turned, math.degrees(math.acos(min(1.0, (np.trace(R0.T @ R1) - 1) / 2))))
    assert turned > 5 and quicks >= 1 and pinned == 0, (turned, quicks, pinned)
    print(f"eyes 7: the VOR's quick phase: in a steady synthetic turn (yaw and pitch, both ways) the window jumps back by half its reach",
          f"in the direction of the turn and never passes it; in the world, the trunk turned {turned:.0f} deg by the waist with the window",
          f"at its edge: {quicks} quick phases, 0 of 32 ticks pinned at the reach")


def _drawn_face(size, cx, cy, polarity=1, bg=0.45, skin=0.75, dark=0.30, H=32, Wd=32):
    """a drawn face (not the parent's): a light ellipse, two eyes and a mouth darker (polarity 1) or lighter (-1) than it"""
    img = np.full((H, Wd), bg)
    yy, xx = np.mgrid[0:H, 0:Wd] + 0.5
    a, b = size / 2, size / 0.81 / 2
    ell = ((xx - cx) / a) ** 2 + ((yy - cy) / b) ** 2 <= 1
    img[ell] = skin
    blob = np.zeros_like(ell)
    for ex in (-0.21, 0.21):
        blob |= ((xx - cx - ex * size) ** 2 + (yy - cy + 0.1 * size / 0.81) ** 2) <= (0.11 * size) ** 2
    blob |= ((xx - cx) / (0.2 * size)) ** 2 + ((yy - cy - 0.27 * size / 0.81) / (0.06 * size / 0.81)) ** 2 <= 1
    img[blob & ell] = dark if polarity > 0 else 1.0
    return (np.repeat(img[..., None], 3, -1) * 255).round().astype(np.uint8)


def test_the_face_template():
    """eyes 8: the born face template (CONSPEC; the W1 verifier's second finding): it fires on a drawn face at each size of its bank
    and at places across the fovea, not on the same face with light blobs (Farroni's polarity) nor on noise; in the world the
    frame's face_fovea and face_periph are the template on the frame's own pixels, and the body's channels are the sensors' alone
    (A1's face test only in the truth). Its reading of the parent's face as she is drawn is printed (C3 measures it)"""
    for size in (8, 12, 16, 22, 28):
        for cx, cy in ((15, 16),) + (((11, 13), (20, 19)) if size <= 16 else ()):   # (the larger faces fill the fovea)
            b = E.face_template(_drawn_face(size, cx, cy))
            assert E.template_match(b), (size, cx, cy, b)
        assert not E.template_match(E.face_template(_drawn_face(size, 15, 16, polarity=-1))), size
    rng = np.random.default_rng(0)
    for _ in range(100):
        assert not E.template_match(E.face_template((rng.random((32, 32, 3)) * 255).astype(np.uint8)))
    w, ey = _world()
    reads = {}
    for dist in (0.45, 0.8, 1.5):
        _face_rig(w, dist)
        f = w.frame()
        assert set(f.obs) == {"body", "touch", "vestibular", "charge", "pain", "eye_p", "eye_f", "face_fovea", "face_periph", "imu_torso",
                              "words", "face", "ears", "sound_side", "onset_periph"}, sorted(f.obs)
        t = f.truth["eyes"]
        assert t["face_test"]["L"][0] and t["face_test"]["R"][0]
        for side in "LR":
            assert E.face_template(t["fovea"][side]) == t["template_fovea"][side]
            assert E.face_template(t["periphery"][side]) == t["template_periphery"][side]
        assert f.obs["face_fovea"][0] == float(E.template_match(t["template_fovea"]["L"]) or E.template_match(t["template_fovea"]["R"]))
        hits = [s_ for s_ in "LR" if E.template_match(t["template_periphery"][s_])]
        assert (f.obs["face_periph"][0] == 1.0) == bool(hits) and (hits or not f.obs["face_periph"].any())
        reads[dist] = (int(f.obs["face_fovea"][0]), round(float(t["template_fovea"]["L"][0]), 2), round(float(t["template_fovea"]["L"][1]), 2))
    t0 = time.perf_counter()
    for _ in range(10):
        for side in "LR":
            E.face_template(t["fovea"][side]); E.face_template(t["periphery"][side])
    ms = (time.perf_counter() - t0) / 10 * 1e3
    ey.close()
    print(f"eyes 8: the born template fires on a drawn face at 8-28 px anywhere in the fovea, not with light blobs nor on noise; the",
          f"frame's face_fovea and face_periph are the template on its own pixels, the channels the sensors' alone; {ms:.1f} ms a tick",
          f"for both eyes' fovea and periphery. On the parent's face as drawn, facing at 0.45 / 0.8 / 1.5 m (A1 passing):",
          f"face_fovea {[reads[k][0] for k in (0.45, 0.8, 1.5)]} (best r {[reads[k][1] for k in (0.45, 0.8, 1.5)]}): C3 measures it")


def test_exact_replay_with_the_eyes():
    """eyes 9: the eyes' codes bit for bit after a save and a restore, in the same world and in a new one (with the sun's shadow,
    the body's, and without)"""
    from sim_babble import Babbler
    for shadows in ("sun", "none"):
        w, ey = _world(shadows=shadows)
        b = Babbler(seed=3, p_rest=0.3)
        for _ in range(15):
            w.apply(b.acts())
        blob, bst = w.save_state(), b.state()
        f1 = []
        for _ in range(15):
            w.apply(b.acts()); f1.append(w.frame())
        for where in ("the same world", "a new world"):
            if where == "a new world":
                ey.close(); w, ey = _world(shadows=shadows)
            w.load_state(blob); b.load(bst)
            f2 = []
            for _ in range(15):
                w.apply(b.acts()); f2.append(w.frame())
            for x, y in zip(f1, f2):
                for k in ("eye_p", "eye_f", "face_fovea", "face_periph", "body", "touch", "vestibular"):
                    assert np.array_equal(x.obs[k], y.obs[k]), (shadows, where, x.tick, k)
        moved = float(np.abs(np.diff([x.obs["eye_f"] for x in f1], axis=0)).sum())
        assert moved > 1.0
        ey.close()
    print(f"eyes 9: 15 ticks of babble (the gaze among the effectors), a save, 15 more; restored in the same world and in a new one,",
          f"every frame's eye codes, the template's readings and the senses bit for bit, with the sun's shadow and without")


def test_no_lamp_at_the_eyes():
    """eyes 10: the room has no headlight and the eyes refuse one; each light term they see by (the ambient, the specular, a light
    moved) renders them again when it changes; with the room's lights off, they see the dark"""
    w, ey = _world()
    m = w.m
    assert m.vis.headlight.active == 0
    f0 = w.frame()
    img0 = f0.truth["eyes"]["images"]["L"].astype(float)
    m.vis.headlight.active = 1
    try:
        ey._cache = None
        w.frame()
    except ValueError as e:
        assert "headlight" in str(e)
    else:
        raise AssertionError("the eyes rendered with a lamp at the eye")
    finally:
        m.vis.headlight.active = 0
    sun = m.light("sun").id
    for field, val in (("light_ambient", m.light_ambient[sun] * 3), ("light_specular", m.light_specular[sun] + 0.5),
                       ("light_pos", m.light_pos[sun] + [0.5, 0, 0]), ("light_dir", [0.0, 0.0, -1.0])):
        keep = getattr(m, field)[sun].copy()
        r0 = ey.timing["renders"]
        getattr(m, field)[sun] = val
        w.frame()
        getattr(m, field)[sun] = keep
        assert ey.timing["renders"] == r0 + 1, field                    # the cache never serves a stale image
    assert np.array_equal(w.frame().truth["eyes"]["images"]["L"], f0.truth["eyes"]["images"]["L"])
    terms = {f: getattr(m, f).copy() for f in ("light_diffuse", "light_ambient", "light_specular")}
    for f in terms:
        getattr(m, f)[:] = 0.0
    glow = w.frame().truth["eyes"]["images"]["L"].astype(float)          # the lights dark: only the emissive surfaces
    emis = m.mat_emission.copy(); m.mat_emission[:] = 0.0
    dark = w.frame().truth["eyes"]["images"]["L"].astype(float)
    m.mat_emission[:] = emis
    for f, v in terms.items():
        getattr(m, f)[:] = v
    act = m.light_active.copy(); m.light_active[:] = 0
    unlit = w.frame().truth["eyes"]["images"]["L"].astype(float)         # every light switched off: MuJoCo draws the scene unlit
    m.light_active[:] = act
    ey.close()
    assert dark.max() == 0 and glow.mean() < img0.mean() and unlit.mean() > img0.mean(), (dark.max(), glow.mean(), unlit.mean(), img0.mean())
    print(f"eyes 10: no headlight, and the eyes refuse one; a change of any light's ambient, specular, place or direction renders",
          f"them again; with every light's terms at zero the eyes see only the emissive surfaces (mean {glow.mean():.1f} of 255 against",
          f"{img0.mean():.1f} lit: the ceiling's, the window's, the lamp's and the dock's, W5's night), with those too",
          f"at zero, black; every light switched off instead, MuJoCo draws the room unlit (mean {unlit.mean():.1f}): the night must",
          f"darken the lights, never switch them all off (for W5)")


def test_the_parents_face_as_drawn():
    """eyes 11: the parent's face drawn at birth (the W1 verifier's first finding: never drawn, her face had no mouth or brows,
    its irises hidden inside the whites and its lids at their placeholders): in both eyes' images at 0.45-0.8 m, facing, her mouth
    is drawn where the face test looks (the test passes in both eyes: its ray meets her lips) and each eye's region is dark against
    her cheeks; her lips' darkness is a real face's, eyes 13 (under the room's light a lit lip can read lighter than a shaded cheek);
    the born template's best reading of her face is printed (eyes 15 and C3 measure it)"""
    w, ey = _world()
    m, d = w.m, w.d
    hb = m.body("parent_head").id
    pts = {"mouth": np.array([kin.head_surface_x(0, kin.MOUTH_Z), 0.0, kin.MOUTH_Z]),
           "eye_L": kin.EYE_C["L"] + [0.009, 0, 0], "eye_R": kin.EYE_C["R"] + [0.009, 0, 0],
           "cheek_L": np.array([kin.head_surface_x(0.05, 0.14), 0.05, 0.14]),
           "cheek_R": np.array([kin.head_surface_x(-0.05, 0.14), -0.05, 0.14])}
    reads = {}
    for dist in (0.45, 0.6, 0.8):
        _face_rig(w, dist)
        f = w.frame()
        t = f.truth["eyes"]
        for side in "LR":
            img = t["images"][side].astype(float).mean(axis=-1)
            lum = {}
            for k, pl in pts.items():
                c, r_, _ = E.project(m, d, side, d.xpos[hb] + d.xmat[hb].reshape(3, 3) @ pl)
                ci, ri = int(c), int(r_)
                lum[k] = img[ri, ci] if not k.startswith("eye") else img[ri - 1:ri + 2, ci - 1:ci + 2].min()   # an eye: its iris
            cheek = min(lum["cheek_L"], lum["cheek_R"])
            assert t["face_test"][side] == (True, "") and lum["eye_L"] < 0.7 * cheek and lum["eye_R"] < 0.7 * cheek, (dist, side, lum)
        reads[dist] = (int(f.obs["face_fovea"][0]), round(max(t["template_fovea"]["L"][0], t["template_fovea"]["R"][0]), 2))
    ey.close()
    print(f"eyes 11: the parent's face drawn at birth: at 0.45 / 0.6 / 0.8 m her mouth where the face test finds it and her eyes dark;",
          f"the born template on her face: face_fovea {[reads[k][0] for k in (0.45, 0.6, 0.8)]}, best r",
          f"{[reads[k][1] for k in (0.45, 0.6, 0.8)]} (a match needs r {E.TEMPLATE_R}; C3 measures it; neither her face nor the",
          f"template is changed for it)")


def test_her_face_of_human_proportions():
    """eyes 12: HER FACE OF HUMAN PROPORTIONS, MEASURED ON THE DRAWN GEOMETRY (the owner's decision, made for him; the W1 verifier's
    fourth round: the third round's face declared the norms and did not build them: its eye opening 22.0 mm wide against 30.7, its
    inner eye corners 41.6 mm apart against 31.8, its head 168 mm wide; this test would have failed it on every one). Everything is
    read off what is drawn, by rays and by the drawn meshes' vertices (tools/sim_face_measure.py): the canthi as the drawn lid margins'
    corners (the sheet's rim on the opening's outline, where calipers go: within 0.4 mm) and as seen (the eye's contents seen from the
    front, the sides and above and below: their extreme points along the canthal line, within 1.5 mm, the corners' last half
    millimetre being too narrow to see into), the fissure's height at the pupil, the pupils, the iris, the mouth
    and the vermilion's heights, the brows over the canthi, the lid and the pupil, the midline profile's nasion, pronasale,
    subnasale and gnathion, the trichion at her hair, the nose's width, the head's and the jaw's and the face's widths, and
    Yaremchuk's depths, each against its cited norm (body/sim/parent_face.py's sources) within the stated tolerance"""
    import sim_face_measure as M
    res = M.measure()
    tol = {"ex-en L": .0015, "ex-en R": .0015, "en-en": .0015, "ex-ex": .0015}   # seen: the corners' last half millimetre is too
    # narrow to see into; the drawn lid margins' corners (where calipers go) are held to 0.4 mm
    bad = []
    for k, (v, want, t, src) in res.items():
        t = tol.get(k, t)
        if t is None:
            assert v <= want + 1e-9, (k, v, want)
        elif abs(v - want) > t:
            bad.append((k, round(v * 1e3, 2), round(want * 1e3, 2), round(t * 1e3, 2)))
    assert not bad, bad
    print("eyes 12: her face of human proportions, measured on the drawn geometry (mm; the norm):",
          ", ".join(f"{k} {1e3 * v:.1f} ({1e3 * want:.1f})" for k, (v, want, t, src) in res.items()))


def test_her_face_photometry():
    """eyes 13: HER FACE'S PHOTOMETRY (the owner's decision; Russell, Kramer and Jones 2017, table 2: young women, no cosmetics): her
    face photographed as the source's faces were (tools/sim_face_photometry.py: from the front under a studio's frontal lamp, exposed
    so her cheek reads L* 65, everything she carries in: her geometry's shading, her sheet's shade of the room's indirect light, the
    eye's parts'), each feature's CIE L* contrast against the skin around it is within 0.005 of a real face's: the eyes (the eye
    with the band of skin around its lashes) 0.152, the brows 0.126, the lips 0.092; no face material glows (her eyes' highlight is
    the room's lamps' reflection in her corneas, dark at night)"""
    from sim_face_photometry import Photometry, RUSSELL
    p = Photometry()
    c = p.contrasts()
    m = p.sc.m
    p.close()
    for k, want in RUSSELL.items():
        assert abs(c[k]["contrast"] - want) <= 0.005, (k, c[k], want)
    for n in ("skin_face", "sclera", "iris", "pupil", "lash", "lid", "caruncle", "fornix", "lips", "brow", "teeth", "mouth_in"):
        assert m.mat_emission[m.material(n).id] == 0, n
    assert not any((m.geom(g).name or "").startswith("parent_glint") for g in range(m.ngeom))
    print(f"eyes 13: her face's photometry (CIE L*, feature against the skin around it, as the source photographed its faces; a young",
          f"woman's): the eyes {c['eyes']['contrast']:.3f} (0.152), the brows {c['brows']['contrast']:.3f} (0.126), the lips",
          f"{c['lips']['contrast']:.3f} (0.092); no face material glows, no drawn highlight")


def _opening(f, sd="L"):
    """the eye's opening at the pupil's vertical (m, about the pupil): the lowest and highest heights where a ray from the front meets
    the eye's contents"""
    C = kin.EYE_C[sd]
    hits = []
    for y in (C[1] - .0004, C[1] + .0004):
        for z in np.arange(C[2] - .009, C[2] + .008, 1e-4):
            _, n, _ = f.ray([.25, y, z], [-1, 0, 0])
            if n.startswith(("parent_sclera_" + sd, "parent_iris_" + sd, "parent_pupil_" + sd)):
                hits.append(z - C[2])
    return (min(hits), max(hits)) if hits else (0.0, 0.0)


def test_her_expressions():
    """eyes 14: her graded face still moves (the owner's decision: keep her graded expressions working), measured on the drawn face:
    the smile lifts the mouth's corners and the frown drops them (the born reading, the corners' pull, unchanged: 2 x (smile -
    frown)); the blink closes the opening (no eye seen at the pupil's line); AU5 widens it, AU6 raises its lower edge; AU1 raises the
    brows' heads, AU4 lowers them; the jaw and the "oh" open the lips; the canthi never move (the lids turn about the canthal line)"""
    import sim_face_measure as M
    f = M.Face()
    sc = f.sc

    def pose(**kw):
        p = G.born_parent(); p.expr = kin.face_params(**kw); sc.set_parent(p); mujoco.mj_forward(sc.m, sc.d)
        hb = sc.m.body("parent_head").id
        f.p0, f.R = sc.d.xpos[hb].copy(), sc.d.xmat[hb].reshape(3, 3).copy()

    def geom(n):
        g = sc.m.geom("parent_" + n).id
        return f.R.T @ (sc.d.geom_xpos[g] - f.p0), f.R.T @ sc.d.geom_xmat[g].reshape(3, 3), sc.m.geom_size[g]

    def corners():
        out = []
        for n in ("mouth0", f"mouth{kin.face.LIP_UP_N - 1}"):
            c, R, sz = geom(n)
            out += [c - R[:, 2] * sz[1], c + R[:, 2] * sz[1]]
        return float(np.mean([min(out[:2], key=lambda p: -abs(p[1]))[2], min(out[2:], key=lambda p: -abs(p[1]))[2]]))

    pose()
    c0, op0, brow0 = corners(), _opening(f), geom("brow0_L")[0][2]
    lo0 = geom("lip_lo8")[0][2]
    pose(**kin.scalar_to_params(1.0)); c_sm = corners()
    pose(**kin.scalar_to_params(-1.0)); c_fr = corners()
    assert c_sm > c0 + .004 and c_fr < c0 - .003, (c0, c_sm, c_fr)
    assert kin.face_reading(kin.scalar_to_params(1.0)) == 2.0 and kin.face_reading(kin.scalar_to_params(-1.0)) == -2.0
    pose(blink=1.0); bl = _opening(f)
    assert bl == (0.0, 0.0), bl                                              # closed: no eye seen at the pupil's line
    pose(lid_up=1.0); wide = _opening(f)
    assert wide[1] > op0[1] + .0015, (wide, op0)
    pose(cheek=1.0); ck = _opening(f)
    assert ck[0] > op0[0] + .003, (ck, op0)
    pose(brow_in=1.0); b_in = geom("brow0_L")[0][2]
    pose(brow_low=1.0); b_lo = geom("brow0_L")[0][2]
    assert b_in > brow0 + .003 and b_lo < brow0 - .002, (brow0, b_in, b_lo)
    for k in ("jaw", "round"):
        pose(**{k: 1.0})
        assert geom("lip_lo8")[0][2] < lo0 - .008, k
    print(f"eyes 14: her expressions, on the drawn face: the smile lifts the corners {1e3 * (c_sm - c0):.1f} mm, the frown drops them",
          f"{1e3 * (c0 - c_fr):.1f} mm (the born reading +2 / -2); the opening at the pupil {1e3 * (op0[1] - op0[0]):.1f} mm at rest,",
          f"closed by the blink, AU5 raises its top {1e3 * (wide[1] - op0[1]):.1f} mm, AU6 its bottom {1e3 * (ck[0] - op0[0]):.1f} mm;",
          f"AU1 raises the brows' heads {1e3 * (b_in - brow0):.1f} mm, AU4 lowers them {1e3 * (brow0 - b_lo):.1f} mm; the jaw and the oh",
          f"open the lips")


def test_the_template_on_her_face():
    """eyes 15: THE BORN TEMPLATE ON HER REAL FACE, MEASURED (the owner's decision: fix the world, not the detector; measure it and
    report it plainly): the template's constants are the committed ones (its layout, sizes, r 0.5, contrast 0.1: never changed for
    her face); her head facing the left eye at 0.3-2 m, lit by the room's lights (no lamp at the eye), the face test passing in both
    eyes: the template's best reading in each fovea, whether it is a DETECTION OF HER FACE (the W1 verifier's fourth round: a match on
    her face, at her face's size: tools/sim_face_template.detection) or a chance match (anywhere else, or at another size), the same
    window upside down as the control, and its best reading on her face in each periphery; written down, no bar (whether it fires on
    her face is a finding for the design, SIM_DESIGN.md C3)"""
    assert (E.TEMPLATE_R, E.TEMPLATE_CONTRAST, E.TEMPLATE_WIDTHS, E.TEMPLATE_EYES, E.TEMPLATE_EYE_D, E.TEMPLATE_MOUTH) == \
        (0.5, 0.10, (8, 11, 16, 23), ((-0.22, 0.12), (0.22, 0.12)), 0.20, ((0.0, -0.25), (0.36, 0.10)))
    import sim_face_template as T
    w, ey = _world()
    base = w.save_state()
    rows = []
    for D in (0.3, 0.45, 0.6, 0.8, 1.0, 1.5, 2.0):
        w.load_state(base)
        c = T.place_head(w, D, 0.0, 0.0, 0.0)
        v = T.read_view(w, ey, c, fovea_aim=True)
        assert v["face_test"] == {"L": "passes", "R": "passes"}, (D, v["face_test"])
        best = max(v[s_]["fovea_best"]["r"] or -1 for s_ in "LR")
        her = sum(bool(v[s_]["fovea_best"]["her_face"]) for s_ in "LR")
        chance = sum(bool(v[s_]["fovea_best"]["match"]) and not v[s_]["fovea_best"]["her_face"] for s_ in "LR")
        flip = max(v[s_]["fovea_best"]["upside_down_r"] or -1 for s_ in "LR")
        peri = max((v[s_].get("periphery_on_her_face") or {"r": -1})["r"] for s_ in "LR")
        rows.append((D, best, her, chance, flip, peri, int(v["event_line_face_fovea"])))
    ey.close()
    print("eyes 15: the born template (unchanged) on her face of human proportions, the room's midday light, facing, both eyes:",
          "fovea best r / detections of her face / chance matches / upside-down r / periphery on her face r / the event line:",
          "; ".join(f"{D:g} m {b:.2f} / {h} / {c} / {fl:.2f} / {p:.2f} / {e}" for D, b, h, c, fl, p, e in rows),
          f"(a match needs r {E.TEMPLATE_R} and contrast {E.TEMPLATE_CONTRAST}; written down for the design, C3)")


def test_her_collision():
    """eyes 16: HER COLLISION IS HER DRAWN FACE (the W1 verifier's fourth round: her visible face stood 2-5 cm in front of her collision
    shape, 53 mm at the nose's tip, 55 at the chin, 34 at the lips, so a hand passed that far into her face before it touched her;
    this test would have failed it). A hand coming at her face: rays from the front and from 30-60 deg to the sides, above and below,
    on a 1.5 mm grid (tools/sim_face_measure.collision): where a ray meets her drawn face, the first of her collision shapes is met
    within 3 mm behind it at worst (her lashes and brows stand that proud of her skin at a slant), on fewer than 0.5% of rays more than
    0.5 mm behind; and at the median within 0.5 mm, 95% of rays within 5 mm in front of it (her hulls bridge her hollows)"""
    import sim_face_measure as M
    c = M.collision()
    a = c["all"]
    assert a["min"] > -3.0 and a["into her face (< -0.5)"] < .005 * a["n"], a
    assert abs(a["median"]) < 0.5 and a["p95"] < 5.0, a
    print(f"eyes 16: her collision is her drawn face: {a['n']} rays at her face from the front and 30-60 deg about; the collision met",
          f"at worst {-a['min']:.1f} mm behind the drawn surface ({a['into her face (< -0.5)']} rays more than 0.5 mm), median",
          f"{a['median']:+.2f} mm, 95% within {a['p95']:.1f} mm in front, 99% within {a['p99']:.1f} mm (by region: " +
          ", ".join(f"{k} {v['median']:+.1f} / {v['p95']:.1f}" for k, v in c.items() if k != "all") + " mm, median / p95)")


def test_her_look():
    """eyes 17: HER FACE IS SMOOTH, WHOLE AND CLEAN (the W1 verifier's fourth round: lumpy cheeks, brows standing off her forehead, a
    nose of a cylinder and balls, segmented lips, faceted patchy shading, bright specks): her head is one connected sheet whose
    normals are its own surface's (smooth shading: the drawn normals agree with the sheet's faces'); her nose is part of it; her lips
    are one smooth tube each (every segment of one radius, end to end); her brows lie on her skin in every expression (never
    sunk into it; at rest and in the smile within 0.6 mm of it, in the frown or a full brow raise or lowering within 1.2 mm: a rigid
    strip over the forehead's curve); her lids and lashes never show through her skin as they turn (the blink, AU5, AU6, AU7 and
    their mixtures) and a blink closes her eye (the W1 fix's own finding: a rigid lid turned about the frontal line opened the
    lateral corner in a blink, and one turned to the pupil's line alone left the medial half open); her head's solid stays inside
    her sheet; nothing on her face is white but her teeth and nothing glows (eyes 13)"""
    F = kin.face
    raw = np.fromfile(F.ASSETS / "face.msh", dtype=np.int32, count=4)
    nv = int(raw[0])
    buf = np.fromfile(F.ASSETS / "face.msh", dtype=np.float32)
    V = buf[4:4 + 3 * nv].reshape(-1, 3).astype(float)
    Nr = buf[4 + 3 * nv:4 + 6 * nv].reshape(-1, 3).astype(float)
    T = np.frombuffer(open(F.ASSETS / "face.msh", "rb").read()[16 + 4 * 8 * nv:], dtype=np.int32).reshape(-1, 3)
    comp = F.largest_component(V, T)[1]
    # vertices shared between the texture's charts are duplicates of one place: join them to count the sheet's pieces
    key = np.unique(np.round(V, 7), axis=0, return_inverse=True)[1].reshape(-1)
    Tj = key[T]
    n_parts = F.largest_component(np.zeros((key.max() + 1, 3)), Tj)[1]
    assert n_parts == 1, n_parts
    Vj = np.zeros((key.max() + 1, 3)); Vj[key] = V
    geo = F.vertex_normals(Vj, Tj)[key]                                      # the sheet's own normals, its charts' seams joined
    dots = (geo * Nr).sum(1)
    # the rolled rims of the openings and the nostrils' cut edges are creases, drawn with their faces' own normals
    rim = np.minimum(np.linalg.norm(V - kin.EYE_C["L"], axis=1), np.linalg.norm(V - kin.EYE_C["R"], axis=1)) < .0175
    nostril = (F.nose_sdf(V) < .003) & (V[:, 2] < F.SN[2] + .006)
    smooth = ~rim & ~nostril
    assert np.median(dots) > .999 and np.mean(dots[smooth] > .95) > .995, (np.median(dots), np.mean(dots[smooth] > .95))
    lips = F._lips(0.0, 0.0, 0.0, 0.0, 0.0)
    radii = {k[:5]: set() for k in lips}
    for k, (p_, q_, sz) in lips.items():
        if k.startswith(("mouth", "lip_lo")) and k != "mouth_open":
            radii[k[:5]].add(round(sz[0], 9))
    assert all(len(v) == 1 for k, v in radii.items() if k in ("mouth", "lip_l")), radii
    worst = {}
    for nm, kw, bar in (("rest", {}, .0006), ("smile", kin.scalar_to_params(1.0), .0006), ("frown", kin.scalar_to_params(-1.0), .0012),
                        ("AU1", dict(brow_in=1.0), .0012), ("AU2", dict(brow_out=1.0), .0012), ("AU4", dict(brow_low=1.0), .0012),
                        ("AU1+4", dict(brow_in=1.0, brow_low=1.0), .0012)):
        g = F.face_geoms(kin.face_params(**kw))
        lo_, hi_ = 1.0, 0.0
        for k in range(F.BROW_PIECES):
            P, q, _ = g[f"brow{k}_L"]
            R = np.zeros(9); mujoco.mju_quat2Mat(R, np.asarray(q, float)); R = R.reshape(3, 3)
            Wv = P + F.rig()[f"brow{k}"] @ R.T                          # the left brow's strips
            gap = Wv[:, 0] - F.head_surface_x(Wv[:, 1], Wv[:, 2])
            lo_, hi_ = min(lo_, float(gap.min())), max(hi_, float(gap.max()))
        worst[nm] = (lo_, hi_)
        assert lo_ >= F.BROW_MIN_LIFT - 1e-5 and hi_ <= bar, (nm, lo_, hi_)       # never sunk; at rest and in the smile
        # within 0.6 mm of the skin, in the frown or a full brow raise or lowering within 1.2 (a rigid strip over a curve)
    # the lids against her skin and her eye (tools/sim_face_measure.py lids): in nine lid states, rays at the left eye from the front
    # and from 35 deg to either side on a 0.5 mm grid: no lid or lash stands out of her skin (more than half a portrait pixel along
    # the globe's radius, away from the opening's rim; the lashes within their thickness of it); in the blinks, at five gazes, no
    # ray from the front meets the globe, the iris or the pupil, and from 35 deg the eye shows over less than 2 mm^2 a gaze (the
    # fissure's ends, where the lids meet at the canthi)
    import sim_face_measure as M
    lids = M.lids(step=5e-4)
    for st, r_ in lids.items():
        assert r_["through skin"] == 0, (st, r_["through skin"], r_["where"])
        assert r_["eye seen from the front"] == 0, (st, r_.get("eye where"))
        assert r_["eye seen from 35 deg"] * .25 / r_["gazes"] < 2.0, (st, r_["eye seen from 35 deg"])
    import make_g1room as MK
    (c_, r_) = MK.HEAD_SOLID
    u = np.random.default_rng(0).normal(size=(4000, 3)); u /= np.linalg.norm(u, axis=1)[:, None]
    Ps = np.asarray(c_) + u * np.asarray(r_)
    assert (F.head_sdf(Ps) < -.001).all()
    m = G.load_model()
    white = [m.material(i).name for i in range(m.nmat) if m.material(i).name in ("sclera", "iris", "pupil", "lash", "lid", "brow", "lips",
             "skin_face", "caruncle", "fornix") and (m.mat_rgba[i, :3] > .9).all()]
    assert not white, white
    print(f"eyes 17: her face smooth, whole and clean: one connected sheet ({len(T)} faces), its drawn normals its surface's (median",
          f"agreement {np.median(dots):.5f}); her lips one tube each (one radius a lip); her brows on her skin (mm over it: " +
          ", ".join(f"{k} {1e3 * a:.2f}-{1e3 * b:.2f}" for k, (a, b) in worst.items()) + f"); her lids never through her skin in",
          f"nine lid states, a blink closed from the front at five gazes (from 35 deg: " +
          ", ".join(f"{k} {.25 * v['eye seen from 35 deg'] / v['gazes']:.2f} mm^2" for k, v in lids.items() if v["gazes"] > 1) +
          "); her head's solid inside her sheet everywhere; nothing white but her teeth")


EYE_TESTS = [test_the_gaze, test_the_vor_exact, test_the_vor_in_the_world, test_the_eyes_render, test_the_face_test,
             test_the_vor_quick_phase, test_the_face_template, test_exact_replay_with_the_eyes, test_no_lamp_at_the_eyes,
             test_the_parents_face_as_drawn, test_her_face_of_human_proportions, test_her_face_photometry, test_her_expressions,
             test_the_template_on_her_face, test_her_collision, test_her_look]

if __name__ == "__main__":
    t0 = time.time(); failed = 0
    for t in EYE_TESTS:
        try:
            t()
        except AssertionError as e:
            failed += 1; print("FAIL", t.__name__, ":", e)
        except Exception as e:
            failed += 1; print("ERROR", t.__name__, ":", type(e).__name__, str(e)[:300])
    print(f"{len(EYE_TESTS) - failed}/{len(EYE_TESTS)} passed in {time.time() - t0:.0f}s")
    sys.exit(1 if failed else 0)
