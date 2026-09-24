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
HER FACE (the W1 verifier's third round, the owner's decision): of human proportions, each measure against its cited adult female
norm; with a real face's photometry (Russell, Kramer and Jones 2017) and her sockets' shade computed from her geometry; her graded
expressions working; the born template, unchanged, measured on her face at 0.3-2 m and written down (a finding, no bar)."""
import math
import os
import sys
import time

import mujoco
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))
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
    assert f.obs["body"].shape == (W.BODY_SIZE,) == (178,) and np.array_equal(f.obs["body"][172:175], w.gaze)
    assert np.allclose(f.obs["body"][175:], w.gaze / 0.15, atol=0.02)
    for _ in range(10):
        w.apply({"gaze": W.act_flat([4, 4, 4])})
    assert abs(w.gaze[2] - W.VERG_MAX) < 1e-12 and abs(w.gaze[1] - W.GAZE_REACH_PITCH) < 0.01      # (the VOR's small turn within)
    assert abs(w.gaze[0] - (W.GAZE_REACH_YAW - W.VERG_MAX / 2)) < 0.01 and w.gaze[0] <= W.GAZE_REACH_YAW - W.VERG_MAX / 2
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
    for _ in range(4):
        w.apply({"waist": turn, "gaze": W.EFFECTOR_REST["gaze"]})
    err_vor = _ray_angle(w, "L", P)
    R1 = d.cam_xmat[c].reshape(3, 3)
    r = np.array([math.tan(held[0] + held[2] / 2), math.tan(held[1]), -1.0]); r /= np.linalg.norm(r)
    to = P - d.cam_xpos[c]; to /= np.linalg.norm(to)
    err_fixed = math.degrees(math.acos(min(1.0, float((R1 @ r) @ to))))
    turned = math.degrees(math.acos(min(1.0, (np.trace(R0.T @ R1) - 1) / 2)))
    assert turned > 10 and err_fixed > 5 and err_vor < 0.25 * err_fixed and err_vor < 3.0, (turned, err_fixed, err_vor)
    print(f"eyes 3: the trunk turned {turned:.0f} deg by the waist in 0.6 s: a window held fixed would be {err_fixed:.1f} deg off the",
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
    hid = m.body_mocapid[m.body("parent_head").id]
    local = np.array([kin.head_surface_x(0, kin.MOUTH_Z) + .0015, 0.0, kin.MOUTH_Z])
    d.mocap_pos[hid] = p - Rh @ local
    d.mocap_quat[hid] = kin.mjquat(Rh)
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
        for _ in range(8):
            w.apply({"waist": W.act_flat([4, 2, 2]), "gaze": W.EFFECTOR_REST["gaze"]})
            quicks += w.frame().truth["vor_quick"]
            pinned += int(abs(w.gaze[0]) >= W.GAZE_REACH_YAW - 1e-4)
        R1 = w.d.cam_xmat[c].reshape(3, 3)
        turned = max(turned, math.degrees(math.acos(min(1.0, (np.trace(R0.T @ R1) - 1) / 2))))
    assert turned > 10 and quicks >= 1 and pinned == 0, (turned, quicks, pinned)
    print(f"eyes 7: the VOR's quick phase: in a steady synthetic turn (yaw and pitch, both ways) the window jumps back by half its reach",
          f"in the direction of the turn and never passes it; in the world, the trunk turned {turned:.0f} deg by the waist with the window",
          f"at its edge: {quicks} quick phases, 0 of 16 ticks pinned at the reach")


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
        assert set(f.obs) == {"body", "touch", "vestibular", "charge", "pain", "eye_p", "eye_f", "face_fovea",
                              "face_periph"}, sorted(f.obs)
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


def _limbus_d(g, side):
    """the visible iris's diameter (m) from the drawn geoms: the circle where the iris's cap sphere meets the globe"""
    (ci, _, si), (ec, _, se) = g[f"iris_{side}"], g[f"sclera_{side}"]
    dd = float(np.linalg.norm(np.asarray(ci) - np.asarray(ec)))
    R, Rc = se[0], si[0]
    x = (R * R - Rc * Rc + dd * dd) / (2 * dd)
    return 2 * math.sqrt(R * R - x * x)


def _margins(g, side):
    """the lids' margins over the pupil (m, about it): where the upper and the lower lid spheres leave the globe's front"""
    ec = kin.EYE_C[side]
    (cu, _, su), (cl, _, sl) = g[f"lid_up_{side}"], g[f"lid_lo_{side}"]
    up = kin.lid_margin_at(0.0, cu[2] - ec[2], su[0], ec[0] - cu[0], True)
    lo = kin.lid_margin_at(0.0, cl[2] - ec[2], sl[0], ec[0] - cl[0], False)
    return up, lo


def test_her_face_of_human_proportions():
    """eyes 12: HER FACE OF HUMAN PROPORTIONS (the owner's decision, made for him in the W1 verifier's third round; the verifier's
    first blocker: irises 19.6 mm, whites 27 mm tall, eyes 72 mm apart). Measured on her drawn face (neutral), each against its
    cited adult female norm (body/sim/parent_kin.py's sources): the pupils 61.7 mm apart [Dodgson]; the visible iris 11.7 mm across
    [Ruefer]; the eye fissure 10.9 mm tall [Farkas]; the mouth 50.2 mm wide [Farkas]; the brow's lower margin 18.5 mm [Gao] and its
    top 25 mm [McKinney] over the pupil; the eyes set in: the brow's soft tissue about 10 mm and the cheek's prominence about 2 mm
    in front of the cornea [Yaremchuk]. The prototype's face fails every one"""
    g = kin.face_geoms_graded(kin.FACE_NEUTRAL)
    ipd = float(np.linalg.norm(kin.EYE_C["L"] - kin.EYE_C["R"]))
    iris = [_limbus_d(g, s_) for s_ in "LR"]
    fissure = [u - l for u, l in (_margins(g, s_) for s_ in "LR")]
    c0, c1, c5, c4 = (np.asarray(g[f"mouth{i}"][0]) for i in (0, 1, 5, 4))
    mouth = float(np.linalg.norm((c5 + (c5 - c4) / 2) - (c0 - (c1 - c0) / 2)))     # corner to corner (the segments are evenly spaced)
    # the brow over the pupil: its first arc's midline and thickness where it crosses the pupil's line
    (b0, q0, sb) = g["brow_L"]
    Rb = np.zeros(9); mujoco.mju_quat2Mat(Rb, np.asarray(q0)); ax = Rb.reshape(3, 3)[:, 2]
    a_, b_ = np.asarray(b0) - ax * sb[1], np.asarray(b0) + ax * sb[1]
    t = (kin.EYE_C["L"][1] - a_[1]) / (b_[1] - a_[1])
    zmid = float(a_[2] + t * (b_[2] - a_[2])) - kin.PUPIL_Z
    brow_low, brow_top = zmid - sb[0], zmid + sb[0]
    brow_ahead = kin.head_surface_x(kin.IPD / 2, kin.Z_BROW) - kin.X_CORNEA
    cheek_ahead = max(kin.head_surface_x(y, z) for y in np.arange(.030, .055, .001) for z in np.arange(.125, .150, .001)) - kin.X_CORNEA
    checks = {"pupils apart": (ipd, .0617, .0005), "iris L": (iris[0], .0117, .0003), "iris R": (iris[1], .0117, .0003),
              "fissure L": (fissure[0], .0109, .0003), "fissure R": (fissure[1], .0109, .0003), "mouth wide": (mouth, .0502, .002),
              "brow low": (brow_low, .0185, .001), "brow top": (brow_top, .025, .001), "brow ahead": (brow_ahead, .010, .0025),
              "cheek ahead": (cheek_ahead, .002, .0025)}
    for k, (v, want, tol) in checks.items():
        assert abs(v - want) <= tol, (k, v, want)
    print("eyes 12: her face of human proportions (mm, measured on the drawn face; the norm):",
          ", ".join(f"{k} {1e3 * v:.1f} ({1e3 * want:.1f})" for k, (v, want, tol) in checks.items()))


def test_her_face_photometry():
    """eyes 13: HER FACE'S PHOTOMETRY (the owner's decision; the verifier's first blocker: her eye region read as light as her
    cheeks): rendered from the front under a uniform light and read in CIE L* as the source reads photographs (Russell, Kramer and
    Jones 2017: young women, no cosmetics), each feature against the skin around it is within 0.01 of a real face's: the eyes 0.152,
    the brows 0.126, the lips 0.092 (the prototype's: brows 0.49, lips 0.32); her sockets' shade of the room's indirect light is
    computed from her geometry (the eye's and the lids' materials are their albedo x parent_kin.socket_occlusion, exactly), and
    darkens the eye against its surround as drawn"""
    from sim_face_photometry import Photometry, RUSSELL
    p = Photometry()
    c = p.contrasts()
    p.close()
    q = Photometry(albedo=False)
    drawn = q.contrasts()
    m = q.sc.m
    q.close()
    for k, want in RUSSELL.items():
        assert abs(c[k]["contrast"] - want) <= 0.01, (k, c[k], want)
    sys.path.insert(0, os.path.join(ROOT, "body", "sim"))
    import make_g1room as M
    ao = kin.socket_occlusion()
    for part, names in kin.SOCKET_MATERIALS.items():
        assert 0.3 < ao[part] < 1.0, (part, ao)
        for n in names:
            base = np.array([float(x) for x in M.MAT[n]["rgba"].split()][:3])
            assert np.allclose(m.mat_rgba[m.material(n).id, :3], base * ao[part], atol=6e-4), (n, part)
    assert drawn["eyes"]["contrast"] > c["eyes"]["contrast"] + 0.02, (drawn["eyes"], c["eyes"])
    print(f"eyes 13: her face's photometry (CIE L*, feature against the skin around it; a young woman's): the eyes",
          f"{c['eyes']['contrast']:.3f} (0.152), the brows {c['brows']['contrast']:.3f} (0.126), the lips {c['lips']['contrast']:.3f}",
          f"(0.092); her sockets' occlusion of the room's indirect light, computed from her geometry: the eye {ao['eye']:.2f}, the",
          f"upper lid {ao['lid_up']:.2f}, the lower {ao['lid_lo']:.2f}, baked into their materials; as drawn the eyes read",
          f"{drawn['eyes']['contrast']:.3f} against their surround")


def test_her_expressions():
    """eyes 14: her graded face still moves (the owner's decision: keep her graded expressions working): a smile lifts the mouth's
    corners and a frown drops them (the born reading, the corners' pull, unchanged: 2 x (smile - frown)); the blink brings the upper
    lid down to the lower and covers the iris; AU5 widens the fissure; AU6 raises the lower lid; AU4 lowers the brows' heads; AU1
    raises them; the jaw and the "oh" open the lips"""
    n = kin.face_geoms_graded(kin.FACE_NEUTRAL)
    sm = kin.face_geoms_graded(kin.scalar_to_params(1.0))
    fr = kin.face_geoms_graded(kin.scalar_to_params(-1.0))
    corner = lambda g: (g["mouth0"][0][2] + g["mouth5"][0][2]) / 2
    assert corner(sm) > corner(n) + .004 and corner(fr) < corner(n) - .003
    assert kin.face_reading(kin.scalar_to_params(1.0)) == 2.0 and kin.face_reading(kin.scalar_to_params(-1.0)) == -2.0
    up0, lo0 = _margins(n, "L")
    bl = kin.face_geoms_graded(kin.face_params(blink=1.0))
    ub, lb = _margins(bl, "L")
    (cu, _, su) = bl["lid_up_L"]
    (ci, _, si) = bl["iris_L"]
    apex = np.asarray(ci) + np.array([si[0], 0, 0])
    assert ub - lb < .001 and np.linalg.norm(apex - np.asarray(cu)) < su[0], (ub, lb)
    wide = _margins(kin.face_geoms_graded(kin.face_params(lid_up=1.0)), "L")
    assert wide[0] > up0 + .0015
    ck = _margins(kin.face_geoms_graded(kin.face_params(cheek=1.0)), "L")
    assert ck[1] > lo0 + .003
    head = lambda g: g["brow_L"][0][2]
    assert head(kin.face_geoms_graded(kin.face_params(brow_low=1.0))) < head(n) - .002
    assert head(kin.face_geoms_graded(kin.face_params(brow_in=1.0))) > head(n) + .003
    for k in ("jaw", "round"):
        o = kin.face_geoms_graded(kin.face_params(**{k: 1.0}))
        assert o["lip_lo2"][0][2] < n["lip_lo2"][0][2] - .008, k
    print(f"eyes 14: her expressions: the smile lifts the corners {1e3 * (corner(sm) - corner(n)):.1f} mm, the frown drops them",
          f"{1e3 * (corner(n) - corner(fr)):.1f} mm (the born reading +2 / -2); the blink closes the fissure ({1e3 * (up0 - lo0):.1f} mm",
          f"open at rest) over the iris; AU5 opens it {1e3 * (wide[0] - up0):.1f} mm more, AU6 lifts the lower lid {1e3 * (ck[1] - lo0):.1f}",
          f"mm; AU4 and AU1 move the brows' heads; the jaw and the oh open the lips")


def test_the_template_on_her_face():
    """eyes 15: THE BORN TEMPLATE ON HER REAL FACE, MEASURED (the owner's decision: fix the world, not the detector; measure it and
    report it plainly): the template's constants are the committed ones (its layout, sizes, r 0.5, contrast 0.1: never changed for
    her face); her head facing the left eye at 0.3-2 m, lit by the room's lights (no lamp at the eye), the face test passing in both
    eyes' fovea: the template's best reading in each fovea and on her face in each periphery is written down, with no bar (whether
    it fires is a finding for the design, SIM_DESIGN.md C3)"""
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
        peri = max((v[s_].get("periphery_on_her_face") or {"r": -1})["r"] for s_ in "LR")
        rows.append((D, best, peri, int(v["event_line_face_fovea"])))
    ey.close()
    print("eyes 15: the born template (unchanged) on her face of human proportions, the room's midday light, facing: fovea best r /",
          "periphery on her face r / fired:", "; ".join(f"{D:g} m {b:.2f} / {p:.2f} / {f}" for D, b, p, f in rows),
          f"(a match needs r {E.TEMPLATE_R} and contrast {E.TEMPLATE_CONTRAST}; written down for the design, C3)")


EYE_TESTS = [test_the_gaze, test_the_vor_exact, test_the_vor_in_the_world, test_the_eyes_render, test_the_face_test,
             test_the_vor_quick_phase, test_the_face_template, test_exact_replay_with_the_eyes, test_no_lamp_at_the_eyes,
             test_the_parents_face_as_drawn, test_her_face_of_human_proportions, test_her_face_photometry, test_her_expressions,
             test_the_template_on_her_face]

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
