"""the simulated world (docs/SIM_DESIGN.md 3.3-3.8, 5, 6 and 11: the build plan's W1; the G1 amendment of 2026-09-24). Run:
python3 -m body.tests.test_sim_world (at nice -n 19; about a minute on this Mac). THE SCENE: body/sim/g1room.xml is its maker's output
byte for byte, every path in it relative; the stock G1 file is byte for byte as committed (its sha256 pinned); the solver is the
one the G1 study fixed (the elliptic cone, multi-point CCD off) with impratio 10 against the soft contacts' creep; the parent's shapes
touch the G1; the mat's collision box is thick (a foot sphere can never pass its mid-plane); an act out of range moves nothing. THE
BODY'S LIMITS are Unitree's own (the URDF of this revision), where the Menagerie file differs on six joints. THE
SERVO LAW: each actuator's gains from its limit (kp = limit / 0.25 rad, the Dex3's / 0.1 rad; kv = 0.04 s x kp), weakness scaling the limits with the
charge, an act re-anchoring the targets at the measured angle plus its steps and a rest relaxing them to the measured angle at 3
ticks, bit for bit a hand-written replica of the law. BIRTH: on its back on the mat, the whole weight on the touch zones, no
self-contact at rest, deterministic. THE SENSES: joint sense, touch per zone, pain's 10 ms filter and its threshold from the body's
declared mass, the IMUs with the model's declared noise, the charge's drain and a charger in the palm. THE REFLEXES: each flexion
and closing sign measured on the G1's own geometry; the withdrawal on a limb's pain (a real blow: a weight dropped on the shin) for
2 ticks, none for the trunk; the grasp on a palm touch, not after the hand's own opening. THE EXACT REPLAY TEST: N ticks of babble,
a save, M more; restored (in the same world and in a new one), the M again: every frame and the final state bit for bit. THE
NIGHT: frozen, nothing seen or moved, the morning the same as never pausing. FAULTS: a tick MuJoCo cannot live raises WorldFault
and leaves the world where the tick began, no MuJoCo log file written. THE BABBLER: deterministic, its units and rests as declared.
THE WORLD IN THE CORE: a tiny body of the world's channels and its seven joint effectors (their reflexes declared) lives through the
core's world loop, every learning rate at 0."""
import hashlib
import math
import os
import sys
import tempfile
import time

import mujoco
import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)   # this tree's body, not a fixed one
sys.path.insert(0, os.path.join(ROOT, "tools"))
from body.sim import world as W  # noqa: E402
from body.sim import reflexes as R  # noqa: E402
from body.sim.world import G1World, WorldFault  # noqa: E402

G = W.G
SIM = os.path.join(ROOT, "body", "sim")
G1_FILE_SHA256 = "2d41a3c6783fbf9e17349608d9c2222d970163b1db315ce9f30c4a97631fc078"   # g1_with_hands.xml at dd8640e (MuJoCo Menagerie)
# Unitree's <limit effort> (N m) of unitree_ros robots/g1_description/g1_29dof_with_hand_rev_1_0.urdf, read 2026-09-24
URDF_EFFORT = dict(hip_pitch=88, hip_roll=139, hip_yaw=88, knee=139, ankle_pitch=35, ankle_roll=35, waist_yaw=88, waist_roll=35,
                   waist_pitch=35, shoulder_pitch=25, shoulder_roll=25, shoulder_yaw=25, elbow=25, wrist_roll=25, wrist_pitch=5,
                   wrist_yaw=5, hand_thumb_0=2.45, hand_thumb_1=1.4, hand_thumb_2=1.4, hand_index_0=1.4, hand_index_1=1.4,
                   hand_middle_0=1.4, hand_middle_1=1.4)


def _babbler(seed=5, p_rest=0.3):
    from sim_babble import Babbler
    return Babbler(seed=seed, p_rest=p_rest)


def _same_frame(a, b):
    if a.tick != b.tick or a.face != b.face or set(a.obs) != set(b.obs):
        return False
    if not all(np.array_equal(a.obs[k], b.obs[k]) for k in a.obs):
        return False
    for k in ("time", "pelvis", "torso", "touch_N", "pain_N", "peak_N", "drain", "fed", "ncon", "acts"):
        x, y = a.truth[k], b.truth[k]
        if not (np.array_equal(x, y) if isinstance(x, np.ndarray) else x == y):
            return False
    return all(np.array_equal(a.truth["toys"][t], b.truth["toys"][t]) for t in a.truth["toys"])


# ---------------------------------------------------------------- the scene
def test_the_scene():
    """world 1: the scene is its maker's output and loads from any checkout; the stock G1 is unchanged; the solver as fixed"""
    sys.path.insert(0, SIM)
    import make_g1room as M
    with open(os.path.join(SIM, "g1room.xml")) as f:
        committed = f.read()
    assert M.scene_xml(SIM) == committed, "g1room.xml is not the maker's output: re-run python3 body/sim/make_g1room.py"
    assert "/Users/" not in committed and "/private/" not in committed and 'file="assets/unitree_g1/g1_with_hands.xml"' in committed
    with open(G.G1_FILE, "rb") as f:
        assert hashlib.sha256(f.read()).hexdigest() == G1_FILE_SHA256, "the stock G1 file changed"
    w = G1World(seed=1)
    m = w.m
    assert (m.nu, m.nmocap, len(W.JOINTS), w.nz) == (43, 16, 43, 45)
    assert m.opt.cone == mujoco.mjtCone.mjCONE_ELLIPTIC and m.opt.disableflags & mujoco.mjtDisableBit.mjDSBL_MULTICCD
    assert m.opt.impratio == 10.0 and m.opt.integrator == mujoco.mjtIntegrator.mjINT_IMPLICITFAST and m.opt.timestep == 0.002
    assert abs(w.body_mass - 34.394) < 1e-3
    mg = m.geom("mat").id                                              # the mat's collision box: its top at 1.2 cm, sunk 10 cm below it
    assert abs(m.geom_pos[mg][2] + m.geom_size[mg][2] - 0.012) < 1e-9 and m.geom_size[mg][2] == 0.05
    try:
        w.apply({"arm_l": 5 ** 7})                                      # an act out of range moves nothing
    except ValueError:
        assert w.tick == 0 and w.save_state() == G1World(seed=1).save_state()
    else:
        raise AssertionError("an act out of range was taken")
    pg = m.geom("parent_hand_L").id
    assert m.geom_contype[pg] == 8 and m.geom_conaffinity[pg] == 1        # the parent's shapes touch the G1 (contype 1)
    assert all(m.light(i).name or not m.light_active[i] for i in range(m.nlight))   # the Menagerie scene light off
    assert [m.camera(f"eye_{s}").id >= 0 for s in "LR"] and [m.site(f"ear_{s}").id >= 0 for s in "LR"]
    print("world 1: g1room.xml is the maker's output (paths relative), the stock G1 file unchanged (sha256 pinned), 43 servos, 45 touch",
          "zones, the elliptic cone with multi-point CCD off and impratio 10, the parent's shapes touching the G1, the mat's collision box",
          "10 cm deep under its top, the eyes and ears added; an act out of range refused before anything moves")


def test_torque_limits_are_unitrees():
    """world 2: each joint's limit is Unitree's URDF effort; the Menagerie file agrees except on the ankles' and the waist's
    roll and pitch (50 there, 35 in Unitree's)"""
    w = G1World(seed=1)
    m = w.m
    stock = {}
    import xml.etree.ElementTree as ET
    for j in ET.parse(G.G1_FILE).getroot().iter("joint"):
        if j.get("actuatorfrcrange"):
            stock[j.get("name")] = float(j.get("actuatorfrcrange").split()[1])
    differ = []
    for k, jn in enumerate(W.JOINTS):
        base = jn[:-6].replace("left_", "").replace("right_", "")
        assert W.torque_limit(jn) == URDF_EFFORT[base] == w.tau_max[k], jn
        assert tuple(m.jnt_actfrcrange[w.jid[k]]) == (-URDF_EFFORT[base], URDF_EFFORT[base]), jn     # born full: h = 1
        if stock[jn] != URDF_EFFORT[base]:
            differ.append(jn)
    assert sorted(differ) == sorted(["left_ankle_pitch_joint", "left_ankle_roll_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
                                     "waist_roll_joint", "waist_pitch_joint"]), differ
    print(f"world 2: all 43 limits are Unitree's URDF efforts (88-139 N m legs, 25 arms, 5 wrists, 1.4-2.45 hands); the Menagerie file",
          f"differs on {len(differ)} (the ankles' and the waist's roll and pitch: 50 against 35), Unitree's kept")


# ---------------------------------------------------------------- the servo law
def _replica_tick(w, acts, h):
    """the servo law written out by hand on a world: weakness at the charge h, the acts' re-anchored targets, the rest's
    relaxation, 75 steps"""
    m, d = w.m, w.d
    lim = w.tau_max * (W.WEAK_FLOOR + (1 - W.WEAK_FLOOR) * h)
    m.jnt_actfrcrange[w.jid, 0] = -lim; m.jnt_actfrcrange[w.jid, 1] = lim
    rest = []
    q = d.qpos[w.qadr].copy()
    for name, js in G.EFFECTORS:
        sl = w.eff_slices[name]
        a = acts.get(name)
        if a is None or a == W.EFFECTOR_REST[name]:
            rest += list(range(sl.start, sl.stop)); continue
        for i, k in zip(range(sl.start, sl.stop), W.act_digits(a, len(js))):
            d.ctrl[w.aid[i]] = min(w.hi[i], max(w.lo[i], q[i] + W.SETTINGS[k]))
    alpha = 1 - math.exp(-0.002 / (3 * 0.150))
    for _ in range(75):
        for i in rest:
            d.ctrl[w.aid[i]] += alpha * (d.qpos[w.qadr[i]] - d.ctrl[w.aid[i]])
        mujoco.mj_step(m, d)
    mujoco.mj_forward(m, d)


def test_the_servo_law():
    """world 3: the gains from the limits; weakness; an act re-anchors, a rest relaxes, bit for bit the law written out by hand"""
    w = G1World(seed=1)
    m = w.m
    for k, a in enumerate(w.aid):
        kp = w.tau_max[k] / (0.1 if "_hand_" in W.JOINTS[k] else 0.25)
        assert m.actuator_gainprm[a, 0] == kp and m.actuator_biasprm[a, 1] == -kp and abs(m.actuator_biasprm[a, 2] + 0.04 * kp) < 1e-12
    for h, share in ((1.0, 1.0), (0.5, 0.65), (0.0, 0.3)):
        w.h = h
        assert np.allclose(w.limits_now(), w.tau_max * share)
    w.h = 1.0
    b = _babbler(3)
    acts = [b.acts() for _ in range(12)] + [{}] * 4 + [{"arm_l": R.flexion_act("arm_l")}] + [{}] * 3
    twin = G1World(seed=1)
    for i, a in enumerate(acts):
        q0 = w.d.qpos[w.qadr].copy(); h = w.h
        w.apply(a)
        _replica_tick(twin, a, h)
        assert np.array_equal(w.d.qpos, twin.d.qpos) and np.array_equal(w.d.ctrl, twin.d.ctrl) and np.array_equal(w.d.qvel, twin.d.qvel), i
        for name, js in G.EFFECTORS:                                   # an acting effector's targets: the measured angle + its steps
            sl = w.eff_slices[name]
            if a.get(name) not in (None, W.EFFECTOR_REST[name]):
                want = np.clip(q0[sl] + [W.SETTINGS[k] for k in W.act_digits(a[name], len(js))], w.lo[sl], w.hi[sl])
                assert np.array_equal(w.d.ctrl[w.aid[sl]], want), (i, name)
    # a raised arm at rest sinks under gravity: tone, no gravity compensation
    w2 = G1World(seed=1)
    lift = W.act_flat([0, 2, 2, 2, 2, 2, 2])                            # shoulder pitch -big: the arm up off the mat (supine)
    for _ in range(4):
        w2.apply({"arm_l": lift})
    z = lambda: float(w2.d.xpos[w2.m.body("left_wrist_yaw_link").id][2])
    z_up = z()
    for _ in range(20):
        w2.apply({})
    assert z() < z_up - 0.05, (z_up, z())
    print(f"world 3: kp = limit / 0.25 rad (the Dex3's / 0.1 rad), kv = 0.04 s x kp on every servo; limits x 0.3 at h 0, x 0.65 at h 0.5; 20 ticks of acts,",
          f"rests and a reflex's act bit for bit the law written out by hand; a raised hand at rest sank {100 * (z_up - z()):.0f} cm in 3 s")


# ---------------------------------------------------------------- birth and the senses
def test_birth_and_touch():
    """world 4: born on its back on the mat; its weight on the touch zones; no self-contact at rest; born the same every time"""
    w = G1World(seed=1)
    m, d = w.m, w.d
    tR = d.xmat[m.body("torso_link").id].reshape(3, 3)
    assert tR[2, 0] > 0.9, tR[:, 0]                                    # the chest's forward axis points up: supine
    assert tR[0, 2] < -0.9, tR[:, 2]                                   # the head toward -x
    assert abs(d.qpos[0] - G.BIRTH_XY[0]) < 0.1 and abs(d.qpos[1] - G.BIRTH_XY[1]) < 0.1
    g1 = w.scene.g1_set
    assert not [c for c in d.contact.geom if m.geom_bodyid[c[0]] in g1 and m.geom_bodyid[c[1]] in g1]
    f = w.frame()
    weight = w.body_mass * 9.81
    s = float(f.truth["touch_N"].sum())
    assert abs(s - weight) < 0.02 * weight, (s, weight)
    touched = [w.zones[i] for i in np.nonzero(f.truth["touch_N"] > 1.0)[0]]
    assert {"torso", "pelvis"} <= set(touched), touched
    assert np.allclose(f.obs["touch"][0::2], np.log1p(f.truth["touch_N"])) and not f.obs["pain"].any()
    assert G1World(seed=1).save_state() == w.save_state()
    for _ in range(20):
        w.apply({})
    s20 = float(w.frame().truth["touch_N"].sum())
    assert abs(s20 - weight) < 0.05 * weight, (s20, weight)
    print(f"world 4: born supine on the mat (head toward -x), no self-contact at rest, the touch zones carrying {s:.1f} N of its",
          f"{weight:.1f} N at birth and {s20:.1f} N after 3 s at rest ({len(touched)} zones touched: {', '.join(touched[:6])}...);",
          f"birth the same every time")


def test_joint_sense_and_vestibule():
    """world 5: joint sense (the scaled angle's sine and cosine, velocity, effort); the IMUs at rest with the declared noise"""
    w = G1World(seed=1)
    for _ in range(10):
        w.apply({})
    f = w.frame()
    b = f.obs["body"].reshape(43, 4)
    assert np.allclose(b[:, 0] ** 2 + b[:, 1] ** 2, 1.0) and (b[:, 1] >= -1e-3).all()   # within its range (a soft limit gives a little)
    q = w.d.qpos[w.qadr]
    ang = (q - (w.hi + w.lo) / 2) / ((w.hi - w.lo) / 2) * math.pi / 2
    assert np.allclose(b[:, 0], np.sin(ang)) and np.array_equal(b[:, 2], w.d.qvel[w.dof])
    assert np.allclose(b[:, 3], w.d.qfrc_actuator[w.dof] / w.m.jnt_actfrcrange[w.jid, 1]) and (np.abs(b[:, 3]) <= 1 + 1e-9).all()
    v = f.obs["vestibular"]
    acc_t, gyro_t, acc_p = v[0:3], v[6:9], v[12:15]
    assert abs(np.linalg.norm(acc_t) - 9.81) < 0.2 and abs(np.linalg.norm(acc_p) - 9.81) < 0.2, (acc_t, acc_p)
    assert np.abs(gyro_t).max() < 0.02
    # the noise drawn is the model's declared noise: the samples of a still tick scatter at 1e-2 (accelerometer), 5e-4 (gyro)
    imu = np.tile(w.d.sensordata[w.imu_adr], (20000, 1))
    x = imu + w.rng.standard_normal(imu.shape) * w.imu_noise
    assert np.allclose(x.std(axis=0), w.imu_noise, rtol=0.05), x.std(axis=0)
    assert list(w.imu_noise) == [1e-2] * 3 + [5e-4] * 3 + [1e-2] * 3 + [5e-4] * 3
    print(f"world 5: joint sense 43 x 4 (sin and cos of the angle over its range, velocity, effort within the present limit); at rest",
          f"the torso's accelerometer reads {np.linalg.norm(acc_t):.2f} m/s2 (gravity) and its gyro under 0.02 rad/s; the noise the",
          f"model declares (1e-2, 5e-4) drawn from the world's stream")


def test_pain():
    """world 6: F_pain from the declared mass; the tick's largest 10 ms mean, windows across the tick's start included; a real
    blow (10 kg dropped 0.6 m on the shin) hurts the leg's zones and withdraws the leg for 2 ticks, not the trunk"""
    w = G1World(seed=1)
    assert abs(w.f_pain - 3 * 34.394 * 9.81) < 0.2
    z = w.zones.index("left_knee")
    base = {k: v.copy() for k, v in w._sensed.items()}

    def felt(steps, carry=None):
        w._sensed = {k: v.copy() for k, v in base.items()}
        if carry is not None:
            w._sensed["carry"][:, z] = carry
        F = np.zeros((75, w.nz)); F[:len(steps), z] = steps
        w._sense_tick(F, np.zeros((75, 12)))
        return float(w._sensed["pain_force"][z])
    assert abs(felt([3000.0]) - 600.0) < 1e-9                          # one 2 ms spike of 3000 N: a 600 N mean, no pain
    assert felt([1100.0] * 5) == 1100.0 > w.f_pain                      # 10 ms at 1100 N: pain
    assert felt([1100.0] * 4) < w.f_pain
    assert felt([1100.0], carry=[1100.0] * 4) == 1100.0                 # a blow across the tick's start is felt
    # a real blow
    def rig(spec):
        b = spec.worldbody.add_body(name="rig_weight", pos=[2.0, 1.5, 0.3])
        b.add_freejoint()
        b.add_geom(name="rig_weight", type=mujoco.mjtGeom.mjGEOM_BOX, size=[.05, .05, .05], mass=10.0, contype=1, conaffinity=1)
    w = G1World(seed=1, extra=rig)
    m, d = w.m, w.d
    mid = (d.xpos[m.body("left_knee_link").id] + d.xpos[m.body("left_ankle_pitch_link").id]) / 2
    qa = m.jnt_qposadr[m.body_jntadr[m.body("rig_weight").id]]
    d.qpos[qa:qa + 3] = [mid[0], mid[1], mid[2] + 0.6]; d.qpos[qa + 3:qa + 7] = [1, 0, 0, 0]
    mujoco.mj_forward(m, d)
    rf = R.Reflexes(w.zones); st = {"leg_l": {}, "leg_r": {}, "arm_l": {}}
    hurt, wd = [], []
    for k in range(6):
        f = w.frame()
        acts = {}
        for limb in st:
            a = rf.withdrawal(f, limb, st[limb])
            if a is not None:
                acts[limb] = a; wd.append((k, limb))
        hurt.append(sorted(w.zones[i] for i in np.nonzero(f.obs["pain"])[0]))
        w.apply(acts)
    first = next(k for k, h in enumerate(hurt) if h)
    assert "left_knee" in hurt[first] and all(w.zones.index(zn) in rf.limb_zones["leg_l"] for zn in hurt[first]), hurt
    assert wd[:2] == [(first, "leg_l"), (first + 1, "leg_l")] and all(l_ == "leg_l" for _, l_ in wd), wd
    print(f"world 6: F_pain {w.f_pain:.1f} N (3 x the declared 34.394 kg); a 2 ms spike of 3000 N is a 600 N mean (no pain), 10 ms",
          f"at 1100 N hurts, a blow across the tick's start counts; 10 kg dropped 0.6 m on the shin hurt {hurt[first]} at tick {first}",
          f"and withdrew the left leg for {len(wd)} ticks (the trunk and the other limbs none)")


def test_the_charge():
    """world 7: the drain law; weakness follows the charge; a charger in the palm feeds 0.01 a tick"""
    w = G1World(seed=1)
    for _ in range(3):
        h0 = w.h
        w.apply(_babbler(4, 0.0).acts())
        f = w.frame()
        assert f.truth["drain"] > W.DRAIN_BASE and abs((h0 - w.h) - f.truth["drain"]) < 1e-15 and f.obs["charge"][1] == w.h - h0
    assert w.chargers.size == 0                                        # no bottle in the room yet (W3)
    w.h = 0.0; w.apply({})
    assert np.allclose(-w.m.jnt_actfrcrange[w.jid, 0], 0.3 * w.tau_max)
    palm = G1World(seed=1)
    pg = [g for g in range(palm.m.ngeom) if palm.zone_of_geom[g] == palm.zones.index("left_hand_palm")][0]
    pp = palm.d.geom_xpos[pg].copy()

    def rig(spec):
        b = spec.worldbody.add_body(name="rig_bottle", mocap=True, pos=pp.tolist())
        b.add_geom(name="bottle_rig", type=mujoco.mjtGeom.mjGEOM_SPHERE, size=[.02, 0, 0], contype=1, conaffinity=1)
    w = G1World(seed=1, extra=rig)
    w.h = 0.5; w.apply({}); f = w.frame()
    assert w.chargers.size == 1 and f.truth["fed"] and abs(w.h - (0.5 + W.FEED_RATE - f.truth["drain"])) < 1e-12
    print(f"world 7: the drain 4e-5 + 2e-3 x mean sum(tau^2)/sum(tau_max^2) a tick, exactly; at h 0 the limits are 0.3 of the",
          f"declared; a charger touching the palm fed +0.01 a tick")


def test_the_reflexes():
    """world 8: each flexion and closing sign measured on the G1 (at qpos0, standing); the withdrawal's two ticks; the grasp on a
    palm touch, not after the hand's own opening; a withdrawal's act flexes the elbow in the world"""
    w = G1World(seed=1)
    m, d = w.m, w.d
    mujoco.mj_resetData(m, d); mujoco.mj_kinematics(m, d)
    pos = lambda b: d.xpos[m.body(b).id].copy()

    def moved(joint, delta, fn):
        q0 = d.qpos.copy(); a = fn(); d.qpos[m.jnt_qposadr[m.joint(joint).id]] += delta; mujoco.mj_kinematics(m, d); b = fn()
        d.qpos[:] = q0; mujoco.mj_kinematics(m, d); return b - a
    for sd, s in (("left", "l"), ("right", "r")):
        fl = R.FLEXION[f"leg_{s}"]; fa = R.FLEXION[f"arm_{s}"]
        e = 0.1 * fa[f"{sd}_elbow_joint"]
        assert moved(f"{sd}_elbow_joint", e, lambda: np.linalg.norm(pos(f"{sd}_wrist_yaw_link") - pos(f"{sd}_shoulder_pitch_link"))) < -0.005
        d.qpos[m.jnt_qposadr[m.joint(f"{sd}_knee_joint").id]] = 0.5; mujoco.mj_kinematics(m, d)       # from a bent knee
        assert moved(f"{sd}_knee_joint", 0.1 * fl[f"{sd}_knee_joint"], lambda: np.linalg.norm(pos(f"{sd}_ankle_roll_link") - pos(f"{sd}_hip_pitch_link"))) < -0.005
        assert moved(f"{sd}_hip_pitch_joint", 0.1 * fl[f"{sd}_hip_pitch_joint"], lambda: pos(f"{sd}_knee_link")[0]) > 0.02    # the knee forward
        toe = [g for g in range(m.ngeom) if m.geom_bodyid[g] == m.body(f"{sd}_ankle_roll_link").id and m.geom_pos[g][0] > 0.1][0]
        heel = [g for g in range(m.ngeom) if m.geom_bodyid[g] == m.body(f"{sd}_ankle_roll_link").id and m.geom_pos[g][0] < 0][0]
        assert moved(f"{sd}_ankle_pitch_joint", 0.1 * fl[f"{sd}_ankle_pitch_joint"], lambda: d.geom_xpos[toe][2] - d.geom_xpos[heel][2]) > 0.01
        for j, sg in R.CLOSING[f"hand_{s}"].items():                   # each closing joint brings the thumb and the fingers together
            gap = lambda: np.linalg.norm(d.xipos[m.body(f"{sd}_hand_thumb_2_link").id] - (d.xipos[m.body(f"{sd}_hand_index_1_link").id] + d.xipos[m.body(f"{sd}_hand_middle_1_link").id]) / 2)
            lo, hi = m.jnt_range[m.joint(j).id]
            assert (lo < 0 < hi) or (sg > 0 and lo == 0) or (sg < 0 and hi == 0), (j, lo, hi)   # a finger opens at 0, closes into its range
            if j.endswith(("thumb_1_joint", "thumb_2_joint")):
                d.qpos[m.jnt_qposadr[m.joint(j).id]] = 0.3 * sg if lo < 0 < hi else 0.0
                mujoco.mj_kinematics(m, d)
                assert moved(j, 0.2 * sg, gap) < 0, j
    w = G1World(seed=1)
    rf = R.Reflexes(w.zones)
    f = w.frame()
    assert rf.withdrawal(f, "arm_l", {}) is None and rf.grasp(f, "hand_l", {}) is None
    st = {}
    f.obs["pain"][w.zones.index("left_hand_palm")] = 1.0                 # a hand's pain withdraws its arm
    seq = [rf.withdrawal(f, "arm_l", st)]
    f2 = w.frame(); seq += [rf.withdrawal(f2, "arm_l", st), rf.withdrawal(f2, "arm_l", st)]
    assert seq == [R.flexion_act("arm_l")] * 2 + [None], seq
    assert rf.withdrawal(f, "waist", {}) is None and rf.withdrawal(f, "arm_r", {}) is None
    f.obs["touch"][2 * w.zones.index("left_hand_palm")] = math.log1p(0.31)
    assert rf.grasp(f, "hand_l", {}) == R.closing_act("hand_l") and rf.grasp(f, "hand_r", {}) is None
    opening = W.act_flat([2, 1, 2, 2, 2, 2, 2])                           # the left thumb_1 opening a small step
    assert R.opens("hand_l", opening) and not R.opens("hand_l", R.closing_act("hand_l"))
    assert rf.grasp(f, "hand_l", {}, last_act=opening) is None and rf.grasp(f, "hand_l", {}, last_act=R.closing_act("hand_l")) is not None
    f.obs["touch"][2 * w.zones.index("left_hand_palm")] = math.log1p(0.29)
    assert rf.grasp(f, "hand_l", {}) is None
    el = lambda: float(w.d.qpos[w.qadr[W.JOINTS.index("left_elbow_joint")]])
    e0 = el()
    for a in seq:
        w.apply({"arm_l": a} if a is not None else {})
    assert el() < e0 - 0.3, (e0, el())
    print(f"world 8: every flexion sign (the elbow, the hip, the knee, the ankle's dorsiflexion) and closing sign measured on the",
          f"G1; the withdrawal two ticks on the limb in pain (a hand's pain withdraws its arm), none for the trunk; the grasp at a palm",
          f"touch of 0.3 N, not after the hand's own opening; the reflex's act flexed the elbow {e0 - el():.2f} rad in two ticks")


# ---------------------------------------------------------------- determinism
def test_exact_replay():
    """world 9: THE EXACT REPLAY TEST: N ticks of babble, save, M more; restored in the same world and in a new one, M again: every
    frame and the final state bit for bit; a world of another seed differs only through its stream"""
    N, M = 60, 60
    w = G1World(seed=1)
    b = _babbler(7, 0.3)
    for _ in range(N):
        w.apply(b.acts())
    blob, bst = w.save_state(), b.state()
    frames1 = [w.frame()]
    for _ in range(M):
        w.apply(b.acts()); frames1.append(w.frame())
    end1 = w.save_state()
    for where in ("the same world", "a new world"):
        w2 = w if where == "the same world" else G1World(seed=1)
        w2.load_state(blob); b.load(bst)
        frames2 = [w2.frame()]
        for _ in range(M):
            w2.apply(b.acts()); frames2.append(w2.frame())
        assert all(_same_frame(x, y) for x, y in zip(frames1, frames2)), where
        assert w2.save_state() == end1, where
    moved = float(np.linalg.norm(frames1[-1].truth["pelvis"][:2] - frames1[0].truth["pelvis"][:2]))
    w3 = G1World(seed=2)
    assert w3.save_state() != G1World(seed=1).save_state()             # the world's stream is the seed's
    w3.load_state(blob)                                                 # a save carries its stream: seed 2's world continues seed 1's life
    b.load(bst)
    for _ in range(M):
        w3.apply(b.acts())
    assert w3.save_state() == end1
    print(f"world 9: the exact replay: {N} ticks of babble, a save ({len(blob) / 1e3:.0f} KB), {M} more; restored in the same world and",
          f"in a new one, the {M} again bit for bit (every frame, the final state; the pelvis moved {100 * moved:.1f} cm); a save",
          f"carries the world's random stream")


def test_the_night():
    """world 10: the night freezes the world: nothing is seen or moved, and the morning is the same as never pausing"""
    w, twin = G1World(seed=1), G1World(seed=1)
    b1, b2 = _babbler(9), _babbler(9)
    for _ in range(20):
        w.apply(b1.acts()); twin.apply(b2.acts())
    before = w.save_state()
    w.pause()
    for bad in (lambda: w.frame(), lambda: w.apply({})):
        try:
            bad()
        except RuntimeError:
            continue
        raise AssertionError("the world was seen or moved at night")
    after = w.save_state(); w.resume()
    assert before != after and w.save_state() == before                # only the paused flag differed
    for _ in range(20):
        w.apply(b1.acts()); twin.apply(b2.acts())
    assert w.save_state() == twin.save_state() and _same_frame(w.frame(), twin.frame())
    print("world 10: the night: frozen (a frame or a move refused), the state unchanged, the morning bit for bit the day never paused")


def test_faults():
    """world 11: a tick MuJoCo cannot live raises WorldFault; the world stands where the tick began; no MUJOCO_LOG.TXT is written"""
    here = os.getcwd()
    with tempfile.TemporaryDirectory() as tmp:
        os.chdir(tmp)
        try:
            w = G1World(seed=1)
            for _ in range(3):
                w.apply({})
            reasons = []
            for bad in (np.nan, 1e11):
                w.d.qvel[10] = bad
                start = w.save_state()
                try:
                    w.apply({})
                except WorldFault as e:
                    reasons.append(e.reason)
                    assert e.tick == 3 == w.tick and w.save_state() == start
                else:
                    raise AssertionError("a bad state was lived")
                w.d.qvel[10] = 0.0
            w.apply({})
            assert w.tick == 4 and not os.path.exists(os.path.join(tmp, "MUJOCO_LOG.TXT"))
        finally:
            os.chdir(here)
    assert all("BADQVEL" in r for r in reasons), reasons
    print(f"world 11: a NaN and a huge velocity each raised WorldFault ({reasons[0][:60]}...), the world left where the tick began,",
          f"then lived on; MuJoCo's warnings kept in the world, no log file written")


def test_the_babbler():
    """world 12: the babbler: deterministic from its seed, its state round trip, acts in range, units of 1-8 ticks, rests as declared"""
    b1, b2 = _babbler(11, 0.6), _babbler(11, 0.6)
    s1 = [b1.acts() for _ in range(3000)]
    assert s1 == [b2.acts() for _ in range(3000)] and s1 != [_babbler(12, 0.6).acts() for _ in range(3000)]
    st = b1.state(); nxt = [b1.acts() for _ in range(50)]; b1.load(st)
    assert nxt == [b1.acts() for _ in range(50)]
    rests = 0; runs = []
    for n, fs in W.EFFECTOR_FACTORS.items():
        xs = [a[n] for a in s1]
        assert all(0 <= x < 5 ** len(fs) for x in xs)
        rests += sum(x == W.EFFECTOR_REST[n] for x in xs)
        k = 1
        for x, y in zip(xs, xs[1:]):
            if x == y:
                k += 1
            else:
                runs.append(k); k = 1
    share = rests / (3000 * len(W.EFFECTOR_FACTORS))
    assert 0.5 < share < 0.7 and min(runs) >= 1 and np.mean(runs) > 3, (share, np.mean(runs))
    print(f"world 12: the babbler: deterministic, its state round trip, acts in range, {100 * share:.0f}% rest ticks (p_rest 0.6),",
          f"acts held {np.mean(runs):.1f} ticks on average")


# ---------------------------------------------------------------- the world in the core
def test_the_world_in_the_core():
    """world 13: a tiny body of the world's channels (body, touch, vestibular, charge) and its seven joint effectors, each sensing its
    own joints, the limbs' reflexes declared, a pain source beside the diary's, lives 30 ticks through the core's world loop at every
    learning rate 0: each tick one frame and one apply, every effector's act reaching the world"""
    from tokenizers import Tokenizer
    from body.core.anatomy import Channel, Effector, LanguageAnatomy, RewardSource
    from body.core.world import WorldLoop
    from body.life import Life
    from body.model import Organs
    tok = Tokenizer.from_file("/Users/lukehamond/Projects/project/data/tok_char.json")
    w = G1World(seed=1)
    rf = R.Reflexes(w.zones)

    class Limb(Effector):
        def reflex(self, frame, life, state):
            if self.name in R.FLEXION:
                return rf.withdrawal(frame, self.name, state)
            if self.name in R.CLOSING:
                last = state.get("now", {}).get("world") if state.get("now") else None
                return rf.grasp(frame, self.name, state, last)
            return None

    class Pain(RewardSource):
        def felt(self, frame, life):
            p = frame.obs.get("pain")
            return -1.0 if p is not None and np.asarray(p).any() else None

    class SimBody(LanguageAnatomy):
        def __init__(self, tok, cfg=None):
            super().__init__(tok, cfg)
            self.channels += [Channel("body", "vector", 172, organ="body_in"), Channel("touch", "vector", 2 * w.nz, organ="touch_in"),
                              Channel("vestibular", "vector", 24, organ="vest_in"), Channel("charge", "vector", 2, organ="charge_in")]
            for name, js in G.EFFECTORS:
                sl = w.eff_slices[name]
                idx = [4 * j + c for j in range(sl.start, sl.stop) for c in range(4)]
                self.effectors.append(Limb(name, W.EFFECTOR_FACTORS[name], rest_id=W.EFFECTOR_REST[name], effort=0.01, sense="body",
                                           sense_idx=idx, inverse=name.startswith("arm"), inv_hidden=16))
            self.rewards.append(Pain("pain"))
    LR0 = dict(live_lr=0.0, value_lr=0.0, band_lr=0.0, night_lr=0.0, gate_lr=0.0, gate_adam_lr=0.0, vcrit_lr=0.0, actor_lr=0.0, face_lr=0.0)
    cfg = dict(LR0, wake_ticks=100000, wake_every=8, gate_every=8, write_floor=1e-30, gate_floor=0.5, fast_rls=0, stri_k=2, stri_m=64)
    torch.manual_seed(0)
    a = SimBody(tok, cfg).check()
    o = Organs(a.vocab, d=32, layers=1, heads=2, window=16, channels=a.channels, effectors=a.effectors, born_seed=0)
    for nm, n in (("body_in", 172), ("touch_in", 2 * w.nz), ("vest_in", 24), ("charge_in", 2)):
        setattr(o, nm, torch.nn.Linear(n, 32))                         # the channels' own maps (the sim's born encoders are a core step)
    t0 = time.time()
    L = Life(o, a, cfg=cfg, device="cpu", seed=0, world=w)
    applied = []
    real_apply = w.apply
    w.apply = lambda acts: (applied.append(dict(acts)), real_apply(acts))[1]
    run = WorldLoop(L)
    for _ in range(30):
        run.step()
    w.apply = real_apply
    assert (L.ticks, w.tick, len(applied)) == (30, 30, 30)
    names = set(W.EFFECTOR_NAMES)
    assert all(names <= set(x) for x in applied) and all(0 <= x[n] < 5 ** len(W.EFFECTOR_FACTORS[n]) for x in applied for n in names)
    moved = sum(int(x[n] != W.EFFECTOR_REST[n]) for x in applied for n in names)
    assert w.now is not None and w.now.tick == 29                       # the frame the last tick was lived on
    assert L.win and L.win[-1]["body"].shape[-1] == 172 and torch.allclose(L.win[-1]["body"].double(), torch.as_tensor(w.now.obs["body"]), atol=1e-6)
    print(f"world 13: a tiny body of the world's four channels and seven joint effectors (43 joints, the limbs' reflexes declared, a",
          f"pain source) lived 30 ticks through the core's world loop in {time.time() - t0:.1f} s: one frame and one apply a tick, every",
          f"effector's act in range at the world ({moved} acts, the rest rests), the joint sense in the window as the frame gave it")


WORLD_TESTS = [test_the_scene, test_torque_limits_are_unitrees, test_the_servo_law, test_birth_and_touch, test_joint_sense_and_vestibule,
               test_pain, test_the_charge, test_the_reflexes, test_exact_replay, test_the_night, test_faults, test_the_babbler,
               test_the_world_in_the_core]

if __name__ == "__main__":
    t0 = time.time(); failed = 0
    for t in WORLD_TESTS:
        try:
            t()
        except AssertionError as e:
            failed += 1; print("FAIL", t.__name__, ":", e)
        except Exception as e:
            failed += 1; print("ERROR", t.__name__, ":", type(e).__name__, str(e)[:300])
    print(f"{len(WORLD_TESTS) - failed}/{len(WORLD_TESTS)} passed in {time.time() - t0:.0f}s")
    sys.exit(1 if failed else 0)
