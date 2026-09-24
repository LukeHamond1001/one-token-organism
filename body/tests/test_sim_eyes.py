"""the G1's eyes (docs/SIM_DESIGN.md 3.4, 3.5, 3.7 and A1; the build plan's W3). Run: python3 -m body.tests.test_sim_eyes (at nice -n 19;
it renders, so it needs the Mac's GL). THE GAZE: the software fovea's effector moves both windows by its declared steps (conjugate
yaw and pitch, vergence), held in reach (each window inside its image, the vergence within the near point); its state and velocity
are the body channel's last six numbers. THE VOR: the window's ray counter-turned by the gyro's rotation, exactly the world-fixed
direction under any constant rotation; in the world, a far point fixated before the trunk turns stays fixated. THE EYES: two
native renders, the periphery 3 x 3 averaged, the fovea window where the gaze puts it (a ball aimed at fills its centre, and leaves
it when the gaze turns away), the retina's opponent ON/OFF code equal to its hand computation. THE FACE TEST (A1): the parent's
face in front and facing passes, and each of its four conditions fails it alone (out of the window, blocked by a toy, turned away,
too far). THE EXACT REPLAY with the eyes: every frame's eye codes bit for bit after a save and a restore, in the same world and in a
new one."""
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
        y2, p2 = W.vor(yaw, pitch, np.tile(omega, (n, 1)), dt)
        r = np.array([math.tan(yaw), math.tan(pitch), -1.0]); r /= np.linalg.norm(r)
        a = float(np.linalg.norm(omega)) * n * dt
        R = _rot(omega, a)                                              # the head turned by omega over the tick
        rn = R.T @ r                                                    # the same world direction in the turned camera's frame
        want = (math.atan2(rn[0], -rn[2]), math.atan2(rn[1], -rn[2]))
        worst = max(worst, abs(y2 - want[0]), abs(p2 - want[1]))
    assert worst < 1e-9, worst
    assert np.allclose(W.vor(0.3, -0.1, np.zeros((75, 3)), 0.002), (0.3, -0.1), atol=1e-12, rtol=0)
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
    red = lambda im: float(np.mean((im[..., 0].astype(int) - im[..., 1].astype(int) > 50) & (im[..., 0] > 80)))
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
    assert ft["L"] == (True, "") and ft["R"] == (True, "") and f.obs["face_seen"][0] == 1.0, ft
    w.gaze = W.clamp_gaze(w.gaze + [0.4, 0, 0])
    assert E.face_test(m, d, w.gaze)["L"] == (False, "not in the fovea") and w.frame().obs["face_seen"][0] == 0.0
    _face_rig(w, 0.6, turn_deg=80)
    assert E.face_test(m, d, w.gaze)["L"] == (False, "turned away")
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
    print("eyes 5: the face test: the parent's face 0.6 m out and facing is seen by both eyes (face_seen 1); out of the window, turned",
          "80 deg, 3.4 m away or behind a block it is not (each condition alone); at 2.5 m facing it is")


def test_exact_replay_with_the_eyes():
    """eyes 6: the eyes' codes bit for bit after a save and a restore, in the same world and in a new one (with the sun's shadow,
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
                for k in ("eye_p", "eye_f", "face_seen", "body", "touch", "vestibular"):
                    assert np.array_equal(x.obs[k], y.obs[k]), (shadows, where, x.tick, k)
        moved = float(np.abs(np.diff([x.obs["eye_f"] for x in f1], axis=0)).sum())
        assert moved > 1.0
        ey.close()
    print(f"eyes 6: 15 ticks of babble (the gaze among the effectors), a save, 15 more; restored in the same world and in a new one,",
          f"every frame's eye codes, face test and senses bit for bit, with the sun's shadow and without")


EYE_TESTS = [test_the_gaze, test_the_vor_exact, test_the_vor_in_the_world, test_the_eyes_render, test_the_face_test,
             test_exact_replay_with_the_eyes]

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
