"""the simulated world (docs/SIM_DESIGN.md 3.3-3.8, 5, 6 and 11: the build plan's W1; the G1 amendment of 2026-09-24). Run:
python3 -m body.tests.test_sim_world (at nice -n 19; a minute or two on this Mac). THE SCENE: body/sim/g1room.xml is its maker's
output byte for byte, every path in it relative; the stock G1 file is byte for byte as committed (its sha256 pinned); the solver is
the one the G1 study fixed (the elliptic cone, multi-point CCD off) with impratio 10 against the soft contacts' creep, and MuJoCo's
auto-reset disabled (A18); the world's geoms and the toys at contact priority 2, so their surfaces decide the G1's contacts (C26);
the parent's shapes touch the G1; the mat's collision box is thick (a foot sphere can never pass its
mid-plane); an act out of range moves nothing; NO LAMP AT THE CHILD'S EYES (the room has no headlight; each room light carries the
room's indirect light as its ambient, a third of its diffuse); the cup at B1's default, 0.8. THE BODY'S LIMITS are the model's own (3.2), equal to 3.2's table. THE SERVO LAW:
each actuator's gains from its limit (kp = limit / 0.25 rad, the Dex3's / 0.1 rad; kv = 0.04 s x kp), weakness scaling the limits
with the charge, an act re-anchoring the targets at the measured angle plus its steps and a rest relaxing them to the measured angle
at 3 ticks, bit for bit a hand-written replica of the law. BIRTH: on its back on the mat, the whole weight on the touch zones, no
self-contact at rest, deterministic; the parent drawn (her face at its neutral expression, her hands shaped: the W1 verifier's first
finding). THE SENSES: joint sense, touch per zone, pain's 10 ms filter, its threshold from the body's declared mass and its skin
site (where on the link it hurts, in the link's frame), A12's blind spots exactly as written (the pairs pressing at rest: none as
born; the motor housings struck together felt on both zones: the W1 verifier's second finding), the IMUs with the model's declared
noise, the charge's drain and a charger in the palm. THE REFLEXES: the withdrawal's local sign (each stepped joint withdraws the
painful site, checked by finite differences on the G1's own geometry; a squeeze gives no direction; a hand's pain withdraws its
arm), for 2 ticks, none for the trunk; in the world, a withdrawal moves the site away; the grasp summed at the spinal cord into
the hand's own act of the same tick, and a fist closed on nothing counted in the truth (palm_own_N). C22 (the W1 verifier's third
finding): the housings struck together at the hip under babble, the withdrawal clears the pain by its second tick where rest does
not. LETTING GO (A11; the W1 verifier's first
finding): a ball kept in the palm, the hand closes on it at rest and opens by its own act while the ball still touches the palm.
THE EXACT REPLAY TEST: N ticks of babble with the withdrawal live and the sun moved (the model's light fields saved), a save, M
more; restored (in the same world and in a new one), the M again: every frame and the final state bit for bit. THE NIGHT: frozen, nothing seen or moved, the morning the same as never pausing. FAULTS (A18): a tick
MuJoCo cannot live raises WorldFault and leaves the world where the tick began: a bad state within the tick, a bad velocity left by
the tick's last step (which MuJoCo itself checks only at the next step), a stop in the tick's closing forward pass; no MuJoCo log
file written. THE BABBLER: deterministic, its units and rests as declared. THE WORLD IN THE CORE: a tiny body of the world's
channels (the body's 178: the 43 joints and the gaze), the gaze and the seven joint effectors (the limbs' withdrawal declared) lives
40 ticks through the core's world loop at every learning rate 0 with a ball kept in its left palm: the hand's gate draws on the
touched ticks, no reflex takes the hand's tick, and every own act that opens the hand reaches the world as it was drawn. The eyes
are body/tests/test_sim_eyes.py's."""
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
# SIM_DESIGN.md 3.2's table (N m; the model's own: MuJoCo Menagerie's g1_with_hands.xml)
DESIGN_LIMITS = dict(hip_pitch=88, hip_roll=139, hip_yaw=88, knee=139, ankle_pitch=50, ankle_roll=50, waist_yaw=88, waist_roll=50,
                     waist_pitch=50, shoulder_pitch=25, shoulder_roll=25, shoulder_yaw=25, elbow=25, wrist_roll=25, wrist_pitch=5,
                     wrist_yaw=5, hand_thumb_0=2.45, hand_thumb_1=1.4, hand_thumb_2=1.4, hand_index_0=1.4, hand_index_1=1.4,
                     hand_middle_0=1.4, hand_middle_1=1.4)
# Unitree's own sources differ from Menagerie and from each other (read 2026-09-24; body/sim/world.py's note; the W1 verifier's sixth
# finding): its URDF of this revision (unitree_ros g1_29dof_with_hand_rev_1_0.urdf, <limit effort>) on these four; its MJCF
# (unitree_mujoco unitree_robots/g1/g1_29dof.xml, the motors' ctrlrange) on the hip roll; its G1 page on the knee (the G1, the EDU).
# The design takes the model's (3.2, A21): these are Menagerie's, not "as Unitree publishes them"
URDF_DIFFERS = {"ankle_pitch": 35, "ankle_roll": 35, "waist_roll": 35, "waist_pitch": 35}
MJCF_DIFFERS = {"hip_roll": 88}
PAGE_KNEE = (90, 120)


def _babbler(seed=5, p_rest=0.3):
    from sim_babble import Babbler
    return Babbler(seed=seed, p_rest=p_rest)


def _same_frame(a, b):
    if a.tick != b.tick or a.face != b.face or set(a.obs) != set(b.obs):
        return False
    if not all(np.array_equal(a.obs[k], b.obs[k]) for k in a.obs):
        return False
    for k in ("time", "pelvis", "torso", "touch_N", "pain_N", "peak_N", "drain", "fed", "ncon", "acts", "spinal", "vor_quick", "gaze",
              "palm_own_N"):
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
    assert m.opt.disableflags & mujoco.mjtDisableBit.mjDSBL_AUTORESET                # A18: a bad state is never silently reset
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
    g1 = w.scene.g1_set                                                  # C26, A21: the world's geoms and the toys at contact priority 2
    col = [g for g in range(m.ngeom) if m.geom_contype[g] or m.geom_conaffinity[g]]
    worldish = [g for g in col if m.geom_bodyid[g] not in g1 and not (m.body(int(m.geom_bodyid[g])).name or "").startswith("parent")]
    assert len(worldish) >= 30 and all(m.geom_priority[g] == 2 for g in worldish)
    assert max(int(m.geom_priority[g]) for g in col if m.geom_bodyid[g] in g1) == 1                   # the G1's own, as shipped
    feet = [i for i in range(w.d.ncon) if {int(m.geom_bodyid[x]) for x in w.d.contact[i].geom} & {m.body("left_ankle_roll_link").id}
            and m.geom("mat").id in w.d.contact[i].geom]
    assert feet and all(w.d.contact[i].friction[0] == m.geom_friction[m.geom("mat").id][0] != 0.6 for i in feet)   # the mat decides
    assert all(m.light(i).name or not m.light_active[i] for i in range(m.nlight))   # the Menagerie scene light off
    assert [m.camera(f"eye_{s}").id >= 0 for s in "LR"] and [m.site(f"ear_{s}").id >= 0 for s in "LR"]
    # NO LAMP AT THE CHILD'S EYES (the W1 verifier's fifth finding): no headlight; the room's indirect light on its own lights
    hl = m.vis.headlight
    assert hl.active == 0 and not np.any(hl.ambient) and not np.any(hl.diffuse) and not np.any(hl.specular)
    lit = [i for i in range(m.nlight) if m.light_active[i]]
    assert sorted(m.light(i).name for i in lit) == ["fill", "key", "sun"]
    assert all(np.allclose(m.light_ambient[i], M.ROOM_INDIRECT * m.light_diffuse[i], atol=1e-5) and m.light_ambient[i].min() > 0 for i in lit)
    # the cup at B1's default (0.8: the owner's call; the W1 verifier's fourth finding)
    assert M.TOY_SCALE["cup"] == 0.8 and abs(m.geom_size[m.geom("cup").id][0] - 0.8 * 0.042) < 1e-9
    print("world 1: g1room.xml is the maker's output (paths relative), the stock G1 file unchanged (sha256 pinned), 43 servos, 45 touch",
          "zones, the elliptic cone with multi-point CCD off and impratio 10, MuJoCo's auto-reset off, the world's geoms and the toys at",
          "contact priority 2 (a foot on the mat takes the mat's friction, not the foot's 0.6), the parent's shapes touching the G1, the mat's collision box",
          "10 cm deep under its top, the eyes and ears added; an act out of range refused before anything moves; no headlight (no lamp at",
          "the child's eyes), each room light's ambient a third of its diffuse; the cup at 0.8 (B1's default)")


def test_torque_limits_are_the_models():
    """world 2: each joint's limit is the model's own (its actuatorfrcrange), equal to 3.2's table; nothing but weakness changes
    it at load"""
    w = G1World(seed=1)
    m = w.m
    stock = {}
    import xml.etree.ElementTree as ET
    for j in ET.parse(G.G1_FILE).getroot().iter("joint"):
        if j.get("actuatorfrcrange"):
            lo, hi = (float(x) for x in j.get("actuatorfrcrange").split())
            assert lo == -hi, j.get("name")
            stock[j.get("name")] = hi
    for k, jn in enumerate(W.JOINTS):
        base = jn[:-6].replace("left_", "").replace("right_", "")
        assert w.tau_max[k] == stock[jn] == DESIGN_LIMITS[base], jn
        assert tuple(m.jnt_actfrcrange[w.jid[k]]) == (-stock[jn], stock[jn]), jn          # born full: h = 1
    kp_ankle = w.kp[W.JOINTS.index("left_ankle_pitch_joint")]
    assert kp_ankle == 200.0                                             # 3.3: 200 N m per rad for the ankles
    assert all(DESIGN_LIMITS[k] != v for k, v in {**URDF_DIFFERS, **MJCF_DIFFERS}.items()) and DESIGN_LIMITS["knee"] not in PAGE_KNEE
    print(f"world 2: all 43 limits are the model's own (Menagerie: 88-139 N m legs, 50 ankles and the waist's roll and pitch, 25 arms,",
          f"5 wrists, 1.4-2.45 hands), 3.2's table; the ankles' kp {kp_ankle:.0f} N m per rad; Unitree's URDF differs on {len(URDF_DIFFERS)}",
          f"joint kinds (35 N m), its MJCF on the hip roll (88), its G1 page on the knee ({PAGE_KNEE[0]} / {PAGE_KNEE[1]} N m): flagged, the model's kept")


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
    """world 3: the gains from the limits; weakness; an act re-anchors, a rest relaxes, bit for bit the law written out by hand (the
    spinal cord off: the servo law alone)"""
    w = G1World(seed=1, spinal=False)
    m = w.m
    for k, a in enumerate(w.aid):
        kp = w.tau_max[k] / (0.1 if "_hand_" in W.JOINTS[k] else 0.25)
        assert m.actuator_gainprm[a, 0] == kp and m.actuator_biasprm[a, 1] == -kp and abs(m.actuator_biasprm[a, 2] + 0.04 * kp) < 1e-12
    for h, share in ((1.0, 1.0), (0.5, 0.65), (0.0, 0.3)):
        w.h = h
        assert np.allclose(w.limits_now(), w.tau_max * share)
    w.h = 1.0
    b = _babbler(3)
    acts = [b.acts() for _ in range(12)] + [{}] * 4 + [{"arm_l": W.act_flat([2, 2, 2, 0, 2, 2, 2])}] + [{}] * 3
    twin = G1World(seed=1, spinal=False)
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
          f"rests and an elbow's big step bit for bit the law written out by hand; a raised hand at rest sank {100 * (z_up - z()):.0f} cm in 3 s")


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
    # THE PARENT DRAWN AT BIRTH (the W1 verifier's first finding): her face's and hands' geoms where set_parent puts them, the face at
    # its neutral expression; the file's own places for them are placeholders (the mouth and brows inside her head at its origin)
    kin = G.kin
    face = kin.face_geoms(kin.scalar_to_params(0.0), None)
    for n, (pp, q, sz) in face.items():
        g = w.scene.face_ids[n]
        assert np.allclose(m.geom_pos[g], pp) and np.allclose(m.geom_quat[g], q), n
    for n in ("mouth0", "brow_L", "lip_lo0"):                          # on the face's front, not at the head's origin
        assert m.geom_pos[w.scene.face_ids[n]][0] > 0.09, n
    hid = m.body_mocapid[m.body("parent_head").id]
    want = kin.fk(G.born_parent())["head"][0]
    assert np.allclose(d.mocap_pos[hid], want) and w.scene.pose is not None and kin.face_reading(kin.scalar_to_params(0.0)) == 0.0
    for _ in range(20):
        w.apply({})
    s20 = float(w.frame().truth["touch_N"].sum())
    assert abs(s20 - weight) < 0.05 * weight, (s20, weight)
    print(f"world 4: born supine on the mat (head toward -x), no self-contact at rest, the touch zones carrying {s:.1f} N of its",
          f"{weight:.1f} N at birth and {s20:.1f} N after 3 s at rest ({len(touched)} zones touched: {', '.join(touched[:6])}...);",
          f"birth the same every time; the parent drawn at birth ({len(face)} face geoms at her neutral face, her head where the maker stands her)")


def test_joint_sense_and_vestibule():
    """world 5: joint sense (the scaled angle's sine and cosine, velocity, effort); the IMUs at rest with the declared noise"""
    w = G1World(seed=1)
    for _ in range(10):
        w.apply({})
    f = w.frame()
    assert f.obs["body"].shape == (W.BODY_SIZE,)
    b = f.obs["body"][:172].reshape(43, 4)
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
    blow (10 kg dropped 0.6 m on the shin) hurts the leg's zones, with a skin site on each zone in pain (on its link, in the
    link's frame) and none elsewhere, and the leg's withdrawal takes 2 ticks, not the trunk's"""
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
    rf = R.Reflexes(w); st = {"leg_l": {}, "leg_r": {}, "arm_l": {}}
    hurt, wd, frames = [], [], []
    for k in range(6):
        f = w.frame(); frames.append(f)
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
    # pain's skin site: on each zone in pain, on its own link (within the reach of the link's collision shapes), 0 elsewhere
    for f in frames:
        site = f.obs["pain_site"].reshape(w.nz, 6)
        pz = np.nonzero(f.obs["pain"])[0]
        assert all(np.any(site[z]) for z in pz) and not np.any(np.delete(site, pz, axis=0)), f.tick
        for z in pz:
            reach = max(float(np.linalg.norm(m.geom_pos[g])) + float(m.geom_rbound[g]) for g in range(m.ngeom) if w.zone_of_geom[g] == z)
            assert np.linalg.norm(site[z, :3]) <= reach + 1e-6 and np.linalg.norm(site[z, 3:]) <= 1 + 1e-9, (w.zones[z], site[z])
    zk = w.zones.index("left_knee")
    coh = float(np.linalg.norm(frames[first].obs["pain_site"].reshape(w.nz, 6)[zk, 3:]))
    print(f"world 6: F_pain {w.f_pain:.1f} N (3 x the declared 34.394 kg); a 2 ms spike of 3000 N is a 600 N mean (no pain), 10 ms",
          f"at 1100 N hurts, a blow across the tick's start counts; 10 kg dropped 0.6 m on the shin hurt {hurt[first]} at tick {first}",
          f"(the shin's skin site on its link, its pressing sites agreeing {coh:.2f}) and took the left leg for {len(wd)} ticks (the",
          f"trunk and the other limbs none)")


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


def _synthetic_pain(w, f, zone, u_local, length=1.0):
    """a copy of frame f with pain on one zone only, at a skin site on its link: the point along u (the link's own frame) on its
    first collision shape's bounding sphere, the outward normal u x length (1: one pressing direction; short: a squeeze)"""
    from body.core.world import Frame
    m = w.m
    z = w.zones.index(zone)
    g = [g for g in range(m.ngeom) if w.zone_of_geom[g] == z][0]
    u = np.asarray(u_local, float); u = u / np.linalg.norm(u)
    p = m.geom_pos[g] + u * m.geom_rbound[g]
    obs = dict(f.obs)
    pain = np.zeros(w.nz); pain[z] = 1.0
    site = np.zeros((w.nz, 6)); site[z] = np.concatenate([p, u * length])
    obs["pain"], obs["pain_site"] = pain, site.reshape(-1)
    return Frame(f.tick, obs, f.face, f.truth), z, p, u


def _fd_drive(w, limb, z, p_l, n_l, eps=1e-6):
    """each of the limb's joints' withdrawal of a skin site (m per rad along the inward normal), by finite differences on the
    world's own state (the G1 as it lies; the reflex's own kinematics never used)"""
    m = w.m
    dd = mujoco.MjData(m); dd.qpos[:] = w.d.qpos; mujoco.mj_kinematics(m, dd)
    b = int(w.zone_body[z])
    place = lambda: (dd.xpos[b] + dd.xmat[b].reshape(3, 3) @ p_l, dd.xmat[b].reshape(3, 3) @ n_l)
    p0, n0 = place(); n0 = n0 / np.linalg.norm(n0)
    out = []
    for j in dict(G.EFFECTORS)[limb]:
        a = m.jnt_qposadr[m.joint(j).id]; q = dd.qpos[a]
        dd.qpos[a] = q + eps; mujoco.mj_kinematics(m, dd); p1 = place()[0]
        dd.qpos[a] = q; mujoco.mj_kinematics(m, dd)
        out.append(-float((p1 - p0) @ n0) / eps)
    return np.array(out)


def test_the_reflexes():
    """world 8: THE WITHDRAWAL'S LOCAL SIGN (the W1 verifier's third finding; 3.7, C22): on the G1 as born, for skin sites on six
    sides of a shin, a thigh's housing (the hip's roll link), a forearm and a palm, the reflex's drive for each joint equals the
    site's withdrawal by finite differences on the world's own state; each joint at least half the strongest steps big the
    withdrawing way, the others hold; the opposite side flips every step; a squeeze (the sites' mean normal under half a unit)
    gives the rest; a hand's pain withdraws its arm; the trunk none; the act lasts two ticks; in the world, the act moves the site
    away. The grasp: on a palm touch, summed into the hand's own act of the same tick, not after the hand's own opening"""
    w = G1World(seed=1)
    m, d = w.m, w.d
    rf = R.Reflexes(w)
    f = w.frame()
    assert rf.withdrawal(f, "arm_l", {}) is None and not hasattr(rf, "grasp")      # the grasp is no core hook (it would take the tick)
    cases = [("leg_l", "left_knee"), ("leg_l", "left_hip_roll"), ("leg_r", "right_knee"), ("arm_l", "left_elbow"),
             ("arm_l", "left_hand_palm"), ("arm_r", "right_elbow")]
    axes = [np.eye(3)[i] * sg for i in range(3) for sg in (1, -1)]
    stepped = 0
    for limb, zone in cases:
        for u in axes:
            fz, z, p_l, n_l = _synthetic_pain(w, f, zone, u)
            e, used = rf.drive(fz, limb, [z])
            fd = _fd_drive(w, limb, z, p_l, n_l)
            assert used == 1 and np.allclose(e, fd, atol=1e-4), (zone, u, e, fd)
            dig = W.act_digits(rf.withdrawal_act(fz, limb, [z]), len(e))
            top = np.abs(e).max()
            for k, x in enumerate(e):
                if abs(x) >= R.RF_HALF * top:
                    assert dig[k] == (4 if x > 0 else 0) and fd[k] * (1 if x > 0 else -1) > 0, (zone, u, k, dig, fd)   # withdraws
                    stepped += 1
                else:
                    assert dig[k] == 2, (zone, u, k, dig)
            back = W.act_digits(rf.withdrawal_act(_synthetic_pain(w, f, zone, -u)[0], limb, [z]), len(e))
            assert back == [4 - x for x in dig], (zone, u, dig, back)                 # the other side: every step flipped
            squeeze = _synthetic_pain(w, f, zone, u, length=0.3)[0]
            assert rf.withdrawal_act(squeeze, limb, [z]) == W.EFFECTOR_REST[limb]      # pressed from opposite sides: no direction
    fz = _synthetic_pain(w, f, "left_hand_palm", [0, 0, 1])[0]
    assert rf.withdrawal(fz, "arm_l", {}) is not None and rf.withdrawal(fz, "arm_r", {}) is None     # a hand's pain: its arm's
    ft = _synthetic_pain(w, f, "pelvis", [0, 0, -1])[0]
    assert all(rf.withdrawal(ft, limb, {}) is None for limb in R.LIMBS) and rf.withdrawal(ft, "waist", {}) is None   # the trunk: none
    st = {}
    fz, z, p_l, n_l = _synthetic_pain(w, f, "left_elbow", [0, 1, 0])
    seq = [rf.withdrawal(fz, "arm_l", st), rf.withdrawal(f, "arm_l", st), rf.withdrawal(f, "arm_l", st)]
    act = rf.withdrawal_act(fz, "arm_l", [z])
    assert seq == [act, act, None] and act != W.EFFECTOR_REST["arm_l"], seq
    # in the world: the act, two ticks, moves the site along its inward normal
    b = int(w.zone_body[z])
    place = lambda: (d.xpos[b] + d.xmat[b].reshape(3, 3) @ p_l, d.xmat[b].reshape(3, 3) @ n_l)
    p0, n0 = place()
    for a in seq:
        w.apply({"arm_l": a} if a is not None else {})
    moved = float((place()[0] - p0) @ n0)
    assert moved < -0.02, moved
    # the grasp at the spinal cord: the hand's own act this tick, summed
    touched, light = math.log1p(0.31), math.log1p(0.29)
    rest_l = W.EFFECTOR_REST["hand_l"]
    assert R.grasp("hand_l", None, touched) == (R.closing_act("hand_l"), "grasp") == R.grasp("hand_l", rest_l, touched)
    assert R.grasp("hand_l", None, light) == (None, None) and R.grasp("hand_l", rest_l, light) == (rest_l, None)
    opening = W.act_flat([2, 1, 2, 2, 2, 2, 2])                           # the left thumb_1 opening a small step
    assert R.opens("hand_l", opening) and not R.opens("hand_l", R.closing_act("hand_l"))
    assert R.grasp("hand_l", opening, touched) == (opening, "overridden")  # the own act of the same tick opens it: the cortex overrides
    big_close = W.act_flat([2, 4, 4, 0, 0, 0, 0])                         # the own act closing by big steps: kept as it is
    assert R.grasp("hand_l", big_close, touched) == (big_close, "grasp")
    turn = W.act_flat([4, 2, 2, 2, 2, 2, 2])                              # the thumb's rotation (no closing sense): the own setting stays
    got = W.act_digits(R.grasp("hand_l", turn, touched)[0], 7)
    assert got[0] == 4 and got[1:] == W.act_digits(R.closing_act("hand_l"), 7)[1:], got
    assert R.grasp("hand_r", None, touched)[0] == R.closing_act("hand_r")
    # each closing sign closes the hand (measured on the G1 at qpos0): a finger opens at 0 and closes into its range; the thumb's two
    # flexing joints bring it toward the fingers
    mujoco.mj_resetData(m, d); mujoco.mj_kinematics(m, d)

    def shifted(joint, delta, fn):
        q0 = d.qpos.copy(); a = fn(); d.qpos[m.jnt_qposadr[m.joint(joint).id]] += delta; mujoco.mj_kinematics(m, d); b = fn()
        d.qpos[:] = q0; mujoco.mj_kinematics(m, d); return b - a
    for sd, s_ in (("left", "l"), ("right", "r")):
        gap = lambda: np.linalg.norm(d.xipos[m.body(f"{sd}_hand_thumb_2_link").id] - (d.xipos[m.body(f"{sd}_hand_index_1_link").id] + d.xipos[m.body(f"{sd}_hand_middle_1_link").id]) / 2)
        for j, sg in R.CLOSING[f"hand_{s_}"].items():
            lo, hi = m.jnt_range[m.joint(j).id]
            assert (lo < 0 < hi) or (sg > 0 and lo == 0) or (sg < 0 and hi == 0), (j, lo, hi)
            if j.endswith(("thumb_1_joint", "thumb_2_joint")):
                d.qpos[m.jnt_qposadr[m.joint(j).id]] = 0.3 * sg if lo < 0 < hi else 0.0
                mujoco.mj_kinematics(m, d)
                assert shifted(j, 0.2 * sg, gap) < 0, j
    print(f"world 8: the withdrawal's local sign on {len(cases)} links x 6 sides: every drive equal to the site's withdrawal by finite",
          f"differences, {stepped} joints stepped big the withdrawing way (at least half the strongest), the rest held; the other side",
          f"flips every step; a squeeze gives the rest; a hand's pain withdraws its arm, the trunk none; two ticks; in the world the",
          f"forearm's site moved {-100 * moved:.1f} cm away along its normal; the grasp at a palm touch of 0.3 N summed into the hand's",
          f"own act of the same tick (an opening act passes, a big closing one is kept)")


def _ball_in_palm(side="left", r=0.03):
    """a rig: a ball kept in the palm (a mocap sphere where the born palm's shape is), and its world"""
    ref = G1World(seed=1)
    pg = [g for g in range(ref.m.ngeom) if ref.zone_of_geom[g] == ref.zones.index(f"{side}_hand_palm")][0]
    pp = ref.d.geom_xpos[pg].copy()

    def rig(spec):
        b = spec.worldbody.add_body(name="rig_ball", mocap=True, pos=pp.tolist())
        b.add_geom(name="rig_ball", type=mujoco.mjtGeom.mjGEOM_SPHERE, size=[r, 0, 0], contype=1, conaffinity=1, rgba=[1, 0, 0, 1])
    return G1World(seed=1, extra=rig)


def _closure(w, hand):
    """how far a hand is closed: the sum over its closing joints of the angle times its closing sign (rad)"""
    return float(sum(sg * w.d.qpos[w.qadr[W.JOINTS.index(j)]] for j, sg in R.CLOSING[hand].items()))


def test_letting_go():
    """world 9: A11 (the W1 verifier's first finding): a ball kept in the left palm; at rest the grasp closes the hand on it every
    tick; the hand's own act opening it passes the spinal cord as it was sent (the grasp overridden) and the hand opens while the
    ball still touches the palm; resting again, it closes again. A FIST CLOSED ON NOTHING (the W1 verifier's eighth finding): the
    left hand closed by its own acts, then resting, keeps itself closed on its own fingers, and the truth's palm_own_N says so
    (all of the palm's force its own hand's), for W4 to count"""
    w = _ball_in_palm()
    z = w.zones.index("left_hand_palm")
    cl = R.CLOSING["hand_l"]
    opening = W.act_flat([2 if j not in cl else (0 if cl[j] > 0 else 4) for j in dict(G.EFFECTORS)["hand_l"]])   # every closing joint open, big
    assert R.opens("hand_l", opening)
    log = []
    for phase, act, n in (("rest", None, 8), ("open", opening, 8), ("rest again", None, 6)):
        c0 = _closure(w, "hand_l")
        for _ in range(n):
            f = w.frame()
            assert f.obs["touch"][2 * z] >= R.GRASP_LOG, (phase, f.truth["touch_N"][z])     # the ball in the palm all along
            w.apply({} if act is None else {"hand_l": act})
            g = w.frame().truth
            assert g["spinal"]["hand_l"] == ("overridden" if act is not None else "grasp"), (phase, g["spinal"])
            assert g["acts"]["hand_l"] == (act if act is not None else R.closing_act("hand_l")), (phase, g["acts"])
        log.append((phase, c0, _closure(w, "hand_l")))
    (_, a0, a1), (_, b0, b1), (_, c0, c1) = log
    assert a1 > a0 + 0.3 and b1 < b0 - 0.3 and c1 > c0 + 0.2, log
    ball_own = w.frame().truth["palm_own_N"]["hand_l"]
    # a fist closed on nothing
    fist = G1World(seed=1)
    close = W.act_flat([2 if j not in cl else (4 if cl[j] > 0 else 0) for j in dict(G.EFFECTORS)["hand_l"]])
    own = []
    for t in range(12):
        fist.apply({"hand_l": close} if t < 6 else {})
        g = fist.frame().truth
        if t >= 6:
            own.append((g["spinal"].get("hand_l"), g["touch_N"][z], g["palm_own_N"]["hand_l"]))
    assert all(ev == "grasp" and n_own >= R.GRASP_N for ev, n_all, n_own in own), own   # its own fingers alone would hold it shut
    alone = sum(abs(n_all - n_own) < 1e-9 for _, n_all, n_own in own)
    assert alone >= 3, own                                               # ticks on which all of the palm's force was its own
    print(f"world 9: a ball kept in the left palm: at rest the grasp closed the hand {a1 - a0:.2f} rad in 8 ticks; its own opening act",
          f"passed the spinal cord as sent (the grasp overridden, 8 of 8 ticks) and opened it {b0 - b1:.2f} rad with the ball still in the",
          f"palm; at rest again it closed {c1 - c0:.2f} rad: letting go is the hand's own act (A11); a fist closed on nothing kept",
          f"itself closed 6 of 6 resting ticks, its own fingers pressing the palm {own[-1][2]:.1f} N (palm_own_N; all of the palm's",
          f"force on {alone} of them; with the ball, {ball_own:.1f} N of its own)")


def test_blind_spots_are_a12s():
    """world 10: A12 as the design writes it (the W1 verifier's second finding): the skin's blind spots are the pairs pressing at
    rest, derived from the born state: none for the G1 as born. The motor housings struck together (the shoulder's roll link
    driven into the torso, 1 kN and more) are felt on both zones' touch and pain, and stay in collision; given a state where a pair
    presses at rest, the derivation lists exactly that pair and the world's touch drops it"""
    w = G1World(seed=1)
    m, d = w.m, w.d
    assert w.blind_pairs == [] and not w.blind.any()
    mujoco.mj_resetData(m, d)
    d.qpos[m.jnt_qposadr[m.joint("left_shoulder_roll_joint").id]] = -1.2              # the shoulder's roll link into the torso
    mujoco.mj_forward(m, d)
    tor, sr = m.body("torso_link").id, m.body("left_shoulder_roll_link").id
    pair = [i for i in range(d.ncon) if {int(m.geom_bodyid[d.contact[i].geom[0]]), int(m.geom_bodyid[d.contact[i].geom[1]])} == {tor, sr}]
    f_pair = sum(float(d.efc_force[d.contact[i].efc_address]) for i in pair)
    assert pair and f_pair > 1000, f_pair                                              # in collision, hard
    zt, zs = w.zones.index("torso"), w.zones.index("left_shoulder_roll")
    felt = w._zone_forces()
    assert felt[zs] >= f_pair - 1e-6 and felt[zt] >= f_pair - 1e-6, (felt[zs], felt[zt], f_pair)   # felt on both
    w._sense_tick(np.tile(felt, (W.STEPS_PER_TICK, 1)), np.zeros((W.STEPS_PER_TICK, 12)))
    pain = w._sensed["pain_force"] > w.f_pain
    assert pain[zs] == (felt[zs] > w.f_pain) and pain[zt], (felt[zs], felt[zt])       # and pain, past F_pain
    blind, pairs = W.rest_blind(m, d, w.scene.g1_set)                                  # a state taken as "at rest": what presses there
    g1 = w.scene.g1_set
    pressing, lost = set(), np.zeros(w.nz)
    for i in range(d.ncon):
        c = d.contact[i]
        a_, b_ = int(m.geom_bodyid[c.geom[0]]), int(m.geom_bodyid[c.geom[1]])
        if a_ in g1 and b_ in g1 and c.efc_address >= 0 and d.efc_force[c.efc_address] > 0:
            pressing.add(frozenset((m.body(a_).name, m.body(b_).name)))
    for i in range(d.ncon):                                             # every contact of a listed pair: its force leaves both zones
        c = d.contact[i]
        if c.efc_address >= 0 and frozenset((m.body(int(m.geom_bodyid[c.geom[0]])).name, m.body(int(m.geom_bodyid[c.geom[1]])).name)) in pressing:
            for gg in c.geom:
                lost[w.zone_of_geom[gg]] += float(d.efc_force[c.efc_address])
    assert {frozenset(x) for x in pairs} == pressing and frozenset(("torso_link", "left_shoulder_roll_link")) in pressing, pairs
    w.blind = blind
    unfelt = w._zone_forces()
    assert np.allclose(felt - unfelt, lost, rtol=1e-9, atol=1e-6) and lost[zs] >= f_pair - 1e-6, (felt - unfelt, lost)
    print(f"world 10: A12's blind spots are the pairs pressing at rest: none for the G1 as born; the shoulder's roll link driven",
          f"{f_pair:.0f} N into the torso is felt on both zones (touch and pain) and stays in collision; taking that pose as rest, the",
          f"derivation lists exactly the {len(pairs)} pairs pressing there (the torso with the shoulder's roll and yaw links and the",
          f"elbow) and touch drops exactly their force")


def test_withdrawal_c22():
    """world 16: C22 (the W1 verifier's third finding) on the pain babble meets most, the motor housings struck together at the hip:
    the left leg flexed and drawn in by its own big steps until the hip's roll link presses the pelvis past F_pain. From that saved
    state, two ticks three ways: the withdrawal (live), rest (tone) and pushing on. By the withdrawal's second tick the housings are
    apart (its force under F_pain and under rest's, which stays pressed near F_pain; pushing on presses harder): it drives the limb
    into no worse contact. Its first tick's force rises while the housing slides off: the elliptic cone's friction coupling (with
    the pair frictionless, the same first tick presses less than the onset did: diagnosed here, flagged for the design with C5
    and C22)"""
    w = G1World(seed=1)
    m = w.m
    z = w.zones.index("left_hip_roll")
    push = W.act_flat([0, 0, 2, 2, 2, 2])                               # hip flexion and adduction, big
    for _ in range(12):
        f = w.frame()
        if f.obs["pain"][z]:
            break
        w.apply({"leg_l": push})
    assert f.obs["pain"][z] and not f.obs["pain"][[w.zones.index(x) for x in w.zones if x.startswith("right_")]].any()
    blob, at = w.save_state(), float(f.truth["pain_N"][z])

    def two(form):
        w.load_state(blob)
        rf, st, out = R.Reflexes(w), {}, []
        for _ in range(2):
            g = w.frame()
            a = rf.withdrawal(g, "leg_l", st) if form == "withdraw" else W.EFFECTOR_REST["leg_l"] if form == "rest" else push
            w.apply({"leg_l": a}); out.append(float(w.frame().truth["pain_N"][z]))
        return out
    wd, rs, pu = two("withdraw"), two("rest"), two("push")
    assert wd[1] < w.f_pain and wd[1] < rs[1] and rs[1] > 0.8 * w.f_pain and pu[1] > at, (at, wd, rs, pu)
    # the first tick's rise: the cone's coupling of friction into the normal force while the housing slides off
    geoms = [g for g in range(m.ngeom) if w.zone_of_geom[g] in (z, w.zones.index("pelvis"))]
    cd = m.geom_condim[geoms].copy()
    m.geom_condim[geoms] = 1
    try:
        free = two("withdraw")
    finally:
        m.geom_condim[geoms] = cd
    assert free[0] < at < wd[0] and free[1] < w.f_pain, (at, wd, free)
    print(f"world 16: C22, the hip's housings struck together ({at:.0f} N on the roll link): the withdrawal's second tick {wd[1]:.0f} N",
          f"(apart), rest's {rs[1]:.0f} N (still pressed), pushing on {pu[1]:.0f} N; its first tick {wd[0]:.0f} N while the housing",
          f"slides off, {free[0]:.0f} N with the pair frictionless: the elliptic cone's friction coupling, not a push into the pelvis")


# ---------------------------------------------------------------- determinism
def test_exact_replay():
    """world 11: THE EXACT REPLAY TEST: N ticks of babble, save, M more; restored in the same world and in a new one, M again: every
    frame and the final state bit for bit; a world of another seed differs only through its stream"""
    N, M = 60, 60
    w = G1World(seed=1)
    b = _babbler(7, 0.3)
    sun = w.m.light("sun").id

    def live(world, n, frames=None, st=None):
        """n ticks of babble with the withdrawal live (its state carried in st), the frames kept"""
        rf = R.Reflexes(world)
        for _ in range(n):
            f = world.frame()
            if frames is not None:
                frames.append(f)
            acts = b.acts()
            for limb in R.LIMBS:
                a = rf.withdrawal(f, limb, st[limb])
                if a is not None:
                    acts[limb] = a
            world.apply(acts)
        if frames is not None:
            frames.append(world.frame())
    st1 = {limb: {} for limb in R.LIMBS}
    live(w, N, st=st1)
    w.m.light_dir[sun] = [0.3, -0.5, -0.81]; w.m.light_pos[sun] = [-1.0, 1.0, 3.0]    # the day's light moved (W5): saved with the world
    blob, bst, sst = w.save_state(), b.state(), {k: dict(v) for k, v in st1.items()}
    frames1 = []
    live(w, M, frames1, st1)
    end1 = w.save_state()
    for where in ("the same world", "a new world"):
        w2 = w if where == "the same world" else G1World(seed=1)
        if where == "the same world":
            w2.m.light_dir[sun] = [0.62, 0.18, -0.76]; w2.m.light_pos[sun] = [-2.0, -0.1, 3.0]
        w2.load_state(blob); b.load(bst)
        assert np.allclose(w2.m.light_dir[sun], [0.3, -0.5, -0.81]) and np.allclose(w2.m.light_pos[sun], [-1.0, 1.0, 3.0]), where
        frames2 = []
        live(w2, M, frames2, {k: dict(v) for k, v in sst.items()})
        assert all(_same_frame(x, y) for x, y in zip(frames1, frames2)) and len(frames1) == len(frames2), where
        assert w2.save_state() == end1, where
    pained = sum(bool(x.obs["pain"].any()) for x in frames1)
    moved = float(np.linalg.norm(frames1[-1].truth["pelvis"][:2] - frames1[0].truth["pelvis"][:2]))
    w3 = G1World(seed=2)
    assert w3.save_state() != G1World(seed=1).save_state()             # the world's stream is the seed's
    w3.load_state(blob)                                                 # a save carries its stream: seed 2's world continues seed 1's life
    b.load(bst)
    live(w3, M, None, {k: dict(v) for k, v in sst.items()})
    assert w3.save_state() == end1
    print(f"world 11: the exact replay: {N} ticks of babble with the withdrawal live, the sun moved, a save ({len(blob) / 1e3:.0f} KB),",
          f"{M} more ({pained} frames in pain); restored in the same world (its sun put back first) and in a new one, the {M} again",
          f"bit for bit (every frame, pain's sites included, the final state, the moved sun; the pelvis moved {100 * moved:.1f} cm); a",
          f"save carries the world's random stream")


def test_the_night():
    """world 12: the night freezes the world: nothing is seen or moved, and the morning is the same as never pausing"""
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
    print("world 12: the night: frozen (a frame or a move refused), the state unchanged, the morning bit for bit the day never paused")


def test_faults():
    """world 13: A18: MuJoCo's auto-reset is off; a tick MuJoCo cannot live raises WorldFault and leaves the world where the tick
    began: a bad state within the tick (NaN, a velocity past mjMAXVAL), a bad velocity left by the tick's LAST step (which MuJoCo
    itself would check only at the next tick's first step: the W1 verifier's finding), a stop in the tick's closing forward pass;
    the frame after the fault is the tick's start's; then the world lives on; no MUJOCO_LOG.TXT is written"""
    here = os.getcwd()
    real_step, real_fwd = mujoco.mj_step, mujoco.mj_forward
    with tempfile.TemporaryDirectory() as tmp:
        os.chdir(tmp)
        try:
            w = G1World(seed=1)
            assert w.m.opt.disableflags & mujoco.mjtDisableBit.mjDSBL_AUTORESET
            for _ in range(3):
                w.apply({})
            reasons = []
            for bad in (np.nan, 1e11):                                  # a bad state before the tick: its first step finds it
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
            assert w.tick == 4
            # the tick's last step leaves a velocity past MuJoCo's bound
            n = [0]

            def last_bad(m, d):
                real_step(m, d); n[0] += 1
                if n[0] == W.STEPS_PER_TICK:
                    d.qvel[10] = 1e11
            start, f0 = w.save_state(), w.frame()
            mujoco.mj_step = last_bad
            try:
                w.apply({})
            except WorldFault as e:
                reasons.append(e.reason)
            else:
                raise AssertionError("a bad velocity left by the tick's last step reached the frame")
            finally:
                mujoco.mj_step = real_step
            assert w.tick == 4 and w.save_state() == start and _same_frame(w.frame(), f0) and abs(w.frame().obs["body"]).max() < 1e3
            # a stop in the tick's closing forward pass
            k = [0]

            def stop_once(m, d):
                if k[0] == 0:
                    k[0] += 1
                    raise mujoco.FatalError("a stop, injected")
                return real_fwd(m, d)
            mujoco.mj_forward = stop_once
            try:
                w.apply({})
            except WorldFault as e:
                reasons.append(e.reason)
            else:
                raise AssertionError("a stop in the closing forward pass was lived")
            finally:
                mujoco.mj_forward = real_fwd
            assert w.tick == 4 and w.save_state() == start
            w.apply({})
            assert w.tick == 5 and not os.path.exists(os.path.join(tmp, "MUJOCO_LOG.TXT"))
        finally:
            mujoco.mj_step, mujoco.mj_forward = real_step, real_fwd
            os.chdir(here)
    assert "BADQVEL" in reasons[0] and ("BADQVEL" in reasons[1] or "MuJoCo stopped" in reasons[1]), reasons   # (no reset: the solver may stop)
    assert "qvel[10] = 1e+11" in reasons[2] and "a stop, injected" in reasons[3], reasons
    print(f"world 13: auto-reset off; a NaN and a huge velocity before a tick, a velocity of 1e11 left by a tick's last step, and a stop",
          f"in its closing forward pass each raised WorldFault ({reasons[2][:52]}...), the world and its frame left where the tick",
          f"began, then it lived on; no log file written")


def test_the_babbler():
    """world 14: the babbler: deterministic from its seed, its state round trip, acts in range, units of 1-8 ticks, rests as declared"""
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
    print(f"world 14: the babbler: deterministic, its state round trip, acts in range, {100 * share:.0f}% rest ticks (p_rest 0.6),",
          f"acts held {np.mean(runs):.1f} ticks on average")


# ---------------------------------------------------------------- the world in the core
def test_the_world_in_the_core():
    """world 15: a tiny body of the world's channels (body, touch, vestibular, charge), the gaze and its seven joint effectors, each
    sensing its own joints, the limbs' withdrawal declared (the hands declare no reflex: the grasp is the spinal cord's), a pain
    source beside the diary's, lives 40 ticks through the core's world loop at every learning rate 0 with a ball kept in its left
    palm (the rig keeps it there as the arm moves): each tick one frame and one apply, every effector's act reaching the world; THE W1 VERIFIER'S FIRST FINDING: no reflex
    takes the left hand's tick, its gate draws on the touched ticks, and each own act of it that opens the hand reaches the servos as
    it was drawn (the grasp overridden), while each other is summed with the grasp"""
    from tokenizers import Tokenizer
    from body.core.anatomy import Channel, Effector, LanguageAnatomy, RewardSource
    from body.core.world import WorldLoop
    from body.life import Life
    from body.model import Organs
    tok = Tokenizer.from_file("/Users/lukehamond/Projects/project/data/tok_char.json")
    w = _ball_in_palm()
    rf = R.Reflexes(w)

    class Limb(Effector):
        def reflex(self, frame, life, state):
            return rf.withdrawal(frame, self.name, state) if self.name in R.LIMBS else None

    class Pain(RewardSource):
        def felt(self, frame, life):
            p = frame.obs.get("pain")
            return -1.0 if p is not None and np.asarray(p).any() else None

    class SimBody(LanguageAnatomy):
        def __init__(self, tok, cfg=None):
            super().__init__(tok, cfg)
            self.channels += [Channel("body", "vector", W.BODY_SIZE, organ="body_in"), Channel("touch", "vector", 2 * w.nz, organ="touch_in"),
                              Channel("vestibular", "vector", 24, organ="vest_in"), Channel("charge", "vector", 2, organ="charge_in")]
            self.effectors.append(Limb("gaze", W.EFFECTOR_FACTORS["gaze"], rest_id=W.EFFECTOR_REST["gaze"], effort=0.01, sense="body",
                                       sense_idx=list(range(172, 178))))
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
    for nm, n in (("body_in", W.BODY_SIZE), ("touch_in", 2 * w.nz), ("vest_in", 24), ("charge_in", 2)):
        setattr(o, nm, torch.nn.Linear(n, 32))                         # the channels' own maps (the sim's born encoders are a core step)
    t0 = time.time()
    L = Life(o, a, cfg=cfg, device="cpu", seed=0, world=w)
    ih = [e.name for e in L.anatomy.effectors].index("hand_l")
    zp = w.zones.index("left_hand_palm")
    applied, lived = [], []
    real_apply = w.apply

    pg = [g for g in range(w.m.ngeom) if w.zone_of_geom[g] == zp][0]
    ball = w.m.body_mocapid[w.m.body("rig_ball").id]

    def apply(acts):
        applied.append(dict(acts)); now = dict(L.motor[ih - 1]["now"])
        touched = float(w.now.obs["touch"][2 * zp]) >= R.GRASP_LOG     # the frame the tick was lived on
        real_apply(acts)
        w.d.mocap_pos[ball] = w.d.geom_xpos[pg]                         # the rig keeps the ball in the palm as the arm moves
        lived.append((now, touched, w._last_acts["hand_l"], w._spinal.get("hand_l")))
    w.apply = apply
    run = WorldLoop(L)
    for _ in range(40):
        run.step()
    w.apply = real_apply
    assert (L.ticks, w.tick, len(applied)) == (40, 40, 40), (L.ticks, w.tick, len(applied))
    names = set(W.EFFECTOR_NAMES)
    assert all(names <= set(x) for x in applied) and all(0 <= x[n] < 5 ** len(W.EFFECTOR_FACTORS[n]) for x in applied for n in names)
    moved = sum(int(x[n] != W.EFFECTOR_REST[n]) for x in applied for n in names)
    assert w.now is not None and w.now.tick == 39, w.now                # the frame the last tick was lived on
    assert L.win and L.win[-1]["body"].shape[-1] == W.BODY_SIZE and torch.allclose(L.win[-1]["body"].double(), torch.as_tensor(w.now.obs["body"]), atol=1e-6)
    # the left hand, the ball in its palm
    assert all(t for _, t, _, _ in lived), [t for _, t, _, _ in lived]  # touched on every tick
    assert not any(n["reflex"] for n, _, _, _ in lived) and L.motor[ih - 1]["stops"]["reflex"] == 0, L.motor[ih - 1]["stops"]
    drew = sum(n["drew"] for n, _, _, _ in lived)
    opened = 0
    for n, _, got, ev in lived:
        own = n["act"]
        if R.opens("hand_l", own):
            opened += 1
            assert (got, ev) == (own, "overridden"), (own, got, ev)
        else:
            assert (got, ev) == R.grasp("hand_l", own, R.GRASP_LOG), (own, got, ev)
    assert drew >= 10 and opened >= 5, (drew, opened)
    print(f"world 15: a tiny body of the world's four channels, the gaze and seven joint effectors (43 joints, the limbs' withdrawal declared,",
          f"a pain source) lived 40 ticks through the core's world loop in {time.time() - t0:.1f} s with a ball kept in its left palm: one frame",
          f"and one apply a tick, every effector's act in range at the world ({moved} acts); the left hand's gate drew on {drew} of the 40",
          f"touched ticks (the old hook: 0), no reflex took its tick, its {opened} opening acts reached the servos as drawn, the others",
          f"summed with the grasp")


WORLD_TESTS = [test_the_scene, test_torque_limits_are_the_models, test_the_servo_law, test_birth_and_touch, test_joint_sense_and_vestibule,
               test_pain, test_the_charge, test_the_reflexes, test_letting_go, test_blind_spots_are_a12s, test_exact_replay, test_the_night,
               test_faults, test_the_babbler, test_the_world_in_the_core, test_withdrawal_c22]

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
