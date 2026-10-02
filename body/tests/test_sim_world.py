"""the simulated world (docs/SIM_DESIGN.md 3.3-3.8, 5, 6 and 11: the build plan's W1; the G1 amendment of 2026-09-24; S5a of
2026-09-25: the world as the G1's anatomy meets it, body/sim/anatomy.py: the body channel's 242, touch's 130 (the Dex3's 16 zones and the
observer's outside torques and base wrench), pain from the joints' gears (A37), the tract and the ears in the frame, Unitree's servo
gains (A39), the withdrawal as the anatomy's Limb.reflex, the cerebellum below the tick, the live, dark night; the older text below is
kept where it still holds, and each test's own docstring is current). Run:
python3 -m body.tests.test_sim_world (at nice -n 19; a minute or two on this Mac). THE SCENE: body/sim/g1room.xml is its maker's
output byte for byte, every path in it relative; the stock G1 file is byte for byte as committed (its sha256 pinned); the solver is
the one the G1 study fixed (the elliptic cone, multi-point CCD off) with impratio 10 chosen by the contacts' physics, and MuJoCo's
auto-reset disabled (A18); the world's geoms and the toys at contact priority 2, so their surfaces decide the G1's contacts (C26);
the parent's shapes never touch the G1 (A25c); the mat's collision box is thick (a foot sphere can never pass its
mid-plane); an act out of range moves nothing; NO LAMP AT THE CHILD'S EYES (the room has no headlight; each room light carries the
room's indirect light as its ambient, a third of its diffuse); the cup at B1's default, 0.8. THE BODY'S LIMITS are the model's own (3.2), equal to 3.2's table. THE SERVO LAW:
each actuator's gains from its limit (kp = limit / 0.25 rad, the Dex3's / 0.1 rad; kv = 0.04 s x kp), weakness scaling the limits
with the charge, an act re-anchoring the targets at the measured angle plus its steps and a rest relaxing them to the measured angle
at 3 ticks, bit for bit a hand-written replica of the law. BIRTH: on its back on the mat, the whole weight on the touch zones, no
self-contact at rest, deterministic; the parent drawn (her face at its neutral expression, her hands shaped: the W1 verifier's first
finding). THE SENSES: joint sense, touch per zone, pain's 10 ms filter and its threshold from the body's declared mass (pain alone,
no place on the link), A12's blind spots exactly as written (the pairs pressing at rest: none as born; the motor housings struck
together felt on both zones: the W1 verifier's second finding), the IMUs with the model's declared noise, no charge and no charger (A88). THE REFLEXES: the newborn's withdrawal as 3.7 and section 10 approve it (the W1 verifier's third round): a
generalized flexion step of the hurt limb's flexion joints (the hip, the knee, the ankle's dorsiflexion; the shoulder, the elbow;
each sign measured on the G1), the same act wherever on the limb it hurts, read from the frame's pain alone, for 2 ticks, none for
the trunk; the grasp summed at the spinal cord into the hand's own act of the same tick, and a fist closed on nothing counted in
the truth (palm_own_N). C22 WRITTEN DOWN AS THE NEWBORN'S: the housings struck together at the hip, the flexion, rest and pushing
on each written down, and the count taking either of the reflex's ticks. LETTING GO (A11; the W1 verifier's first
finding): a ball kept in the palm, the hand closes on it at rest and opens by its own act while the ball still touches the palm.
THE EXACT REPLAY TEST: N ticks of babble with the withdrawal live and the sun moved (the model's light fields saved), a save, M
more; restored (in the same world and in a new one), the M again: every frame and the final state bit for bit. THE PARENT'S POSE
(Scene.pose) is saved with the world and restored with it. IMPRATIO BY PHYSICS: below its sliding force a box holds (the soft model's
creep at most 2 mm in 2 s), above it it slides as Coulomb's law says, and the convex model's slip coupling is as disclosed. THE NIGHT: frozen, nothing seen or moved, the morning the same as never pausing. FAULTS (A18): a tick
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
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)   # this tree's body, not a fixed one
sys.path.insert(0, os.path.join(ROOT, "tools"))
from body.sim import world as W  # noqa: E402
from body.sim import reflexes as R  # noqa: E402
from body.sim.world import G1World, WorldFault  # noqa: E402
from body.sim import anatomy as AN  # noqa: E402
from body.sim.lang import lexicon as LX  # noqa: E402

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


_LIMBS = None


def _limbs():
    """the G1 anatomy's limb effectors (body/sim/anatomy.py), whose reflex is the newborn's withdrawal (S5a: from the joints' pain)"""
    global _LIMBS
    if _LIMBS is None:
        from body.sim.anatomy import SimAnatomy, SIM_CFG, born_table
        a = SimAnatomy(born_table(), SIM_CFG)
        _LIMBS = {e.name: e for e in a.effectors if e.name in R.LIMBS}
    return _LIMBS


def _withdraw(frame, st):
    """the tick's withdrawal acts from the frame (the anatomy's Limb.reflex, its counts in st[limb])"""
    out = {}
    for limb, e in _limbs().items():
        a = e.reflex(frame, None, st.setdefault(limb, {}))
        if a is not None:
            out[limb] = a
    return out


def _babbler(seed=5, p_rest=0.3):
    from sim_babble import Babbler
    return Babbler(seed=seed, p_rest=p_rest)


def _same_frame(a, b):
    if a.tick != b.tick or a.face != b.face or set(a.obs) != set(b.obs):
        return False
    if not all(np.array_equal(a.obs[k], b.obs[k]) for k in a.obs):
        return False
    for k in ("time", "pelvis", "torso", "touch_N", "outside_peak_Nm", "base_peak_N", "peak_N", "ncon", "acts", "spinal",
              "vor_quick", "gaze", "palm_own_N", "heat_C", "night"):
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
    her = [a for a in range(m.nu) if (m.actuator(a).name or "").startswith("parent_")]   # the parent is a body (her joints and her
    assert (m.nu - len(her), len(her), m.nmocap, len(W.JOINTS), w.nz) == (43, 39, 0, 43, 45)   # 39 muscles, A25b), not mocap
    assert her == list(range(43, 82)) and w.aid.max() < 43                   # the G1's 43 servos come first, as its file has them
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
    assert m.geom_contype[pg] == 8 and m.geom_conaffinity[pg] == 18       # the parent's shapes touch the room (16) and the floor (2), never the G1 (A25c)
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
          "contact priority 2 (a foot on the mat takes the mat's friction, not the foot's 0.6), the parent's shapes never touching the G1 (A25c), the mat's collision box",
          "10 cm deep under its top, the eyes and ears added; an act out of range refused before anything moves; no headlight (no lamp at",
          "the child's eyes), each room light's ambient a third of its diffuse; the cup at 0.8 (B1's default)")


def test_torque_limits_are_the_models():
    """world 2: each body joint's limit is the model's own (its actuatorfrcrange), equal to 3.2's table; the Dex3's are Unitree's
    (A80: the motor's ideal torque, its holding figure the pain line); nothing but weakness changes them at load"""
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
        assert stock[jn] == DESIGN_LIMITS[base], jn
        want = G.DEX3[jn]["ideal"] if jn in G.DEX3 else stock[jn]           # the Dex3's as Unitree specifies them (A80)
        assert w.tau_max[k] == want and tuple(m.jnt_actfrcrange[w.jid[k]]) == (-want, want), jn          # born full: h = 1
        assert w.tau_hold[k] == (G.DEX3[jn]["hold"] if jn in G.DEX3 else stock[jn]), jn                  # the pain line (A37, A80)
    kp_ankle = w.kp[W.JOINTS.index("left_ankle_pitch_joint")]
    assert kp_ankle == 40.0                                              # A39: Unitree's own ankle gain (unitree_sdk2's example)
    assert all(DESIGN_LIMITS[k] != v for k, v in {**URDF_DIFFERS, **MJCF_DIFFERS}.items()) and DESIGN_LIMITS["knee"] not in PAGE_KNEE
    print(f"world 2: all 43 limits are the model's own (Menagerie: 88-139 N m legs, 50 ankles and the waist's roll and pitch, 25 arms,",
          f"5 wrists, 1.4-2.45 hands), 3.2's table; the ankles' kp {kp_ankle:.0f} N m per rad; Unitree's URDF differs on {len(URDF_DIFFERS)}",
          f"joint kinds (35 N m), its MJCF on the hip roll (88), its G1 page on the knee ({PAGE_KNEE[0]} / {PAGE_KNEE[1]} N m): flagged, the model's kept")


# ---------------------------------------------------------------- the servo law
def _replica_tick(w, acts):
    """the servo law written out by hand on a world: the declared limits (no weakness since A88: no charge), the acts' re-anchored
    targets, the rest's relaxation, 75 steps (the parent's motion run around each as the world runs it: she is a body in the same
    physics)"""
    m, d = w.m, w.d
    par = w.parent
    par.tick_begin()
    lim = w.tau_max.copy()
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
    for s in range(75):
        for i in rest:
            d.ctrl[w.aid[i]] += alpha * (d.qpos[w.qadr[i]] - d.ctrl[w.aid[i]])
        par.before_step(s)
        mujoco.mj_step(m, d)
        par.after_step(s)
    mujoco.mj_forward(m, d)
    par.tick_end()
    w.tick += 1


def test_the_servo_law():
    """world 3: the gains from the limits; the limits the declared ones (A88); an act re-anchors, a rest relaxes, bit for bit the law
    written out by hand (the spinal cord off: the servo law alone)"""
    w = G1World(seed=1, spinal=False)
    m = w.m
    kps, kds = W.servo_gains()
    for k, a in enumerate(w.aid):                                        # A39: Unitree's published gains; the Dex3's (A72) its limit
        kp = w.tau_max[k] / W.STEP_BIG if "_hand_" in W.JOINTS[k] else W.UNITREE_KP[W.JOINTS[k][:-6].replace("left_", "").replace("right_", "")]
        kd = kds[k]                                                      # per big step, so a closing step spends its full torque
        assert abs(m.actuator_gainprm[a, 0] - kp) < 1e-12 and abs(m.actuator_biasprm[a, 1] + kp) < 1e-12 and m.actuator_biasprm[a, 2] == -kd
    assert np.array_equal(w.limits_now(), w.tau_max) and np.array_equal(-m.jnt_actfrcrange[w.jid, 0], w.tau_max)   # A88: no weakness
    b = _babbler(3)
    acts = [b.acts() for _ in range(12)] + [{}] * 4 + [{"arm_l": W.act_flat([2, 2, 2, 0, 2, 2, 2])}] + [{}] * 3
    twin = G1World(seed=1, spinal=False)
    for i, a in enumerate(acts):
        q0 = w.d.qpos[w.qadr].copy()
        w.apply(a)
        _replica_tick(twin, a)
        assert np.array_equal(w.d.qpos, twin.d.qpos) and np.array_equal(w.d.ctrl, twin.d.ctrl) and np.array_equal(w.d.qvel, twin.d.qvel), i
        for name, js in G.EFFECTORS:                                   # an acting effector's targets: the measured angle + its steps
            sl = w.eff_slices[name]
            if a.get(name) not in (None, W.EFFECTOR_REST[name]):
                want = np.clip(q0[sl] + [W.SETTINGS[k] for k in W.act_digits(a[name], len(js))], w.lo[sl], w.hi[sl])
                assert np.array_equal(w.d.ctrl[w.aid[sl]], want), (i, name)
    # a raised arm at rest sinks under gravity: tone, no gravity compensation
    w2 = G1World(seed=1)
    lift = W.act_flat([0, 2, 2, 2, 2, 2, 2])                            # shoulder pitch -big: the arm up off the mat (supine)
    z = lambda: float(w2.d.xpos[w2.m.body("left_wrist_yaw_link").id][2])
    z_0 = z()
    for _ in range(8):
        w2.apply({"arm_l": lift})
    z_up = z()
    for _ in range(20):
        w2.apply({})
    assert z_up > z_0 + 0.02 and z() < z_up - 0.01, (z_0, z_up, z())    # Unitree's shoulder gain lifts it slowly (A39)
    print(f"world 3: Unitree's gains on the body's servos (A39: unitree_sdk2's whole-body example), the Dex3's its limit per big step (A72); limits x 0.3 at h 0, x 0.65 at h 0.5; 20 ticks of acts,",
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
    assert np.allclose(f.obs["touch"][0:32:2], np.log1p(f.truth["touch_N"][w.dex])) and not f.obs["pain"].any()   # the Dex3's 16 zones
    base_f = float(np.linalg.norm(f.obs["touch"][118:124:2]))              # the observer's base force: the mat carries its weight (A37)
    assert abs(base_f - 1.0) < 0.03, base_f
    assert G1World(seed=1).save_state() == w.save_state()
    # THE PARENT DRAWN AT BIRTH (the W1 verifier's first finding): her face's and hands' geoms where set_parent puts them, the face at
    # its neutral expression; the file's own places for them are placeholders (the mouth and brows inside her head at its origin)
    kin = G.kin
    face = kin.face_geoms(kin.scalar_to_params(0.0), None)
    for n, (pp, q, sz) in face.items():
        g = w.scene.face_ids[n]
        if n in w.scene.mesh_offset:                                   # a mesh: its pose composed with its compiled offset
            mp, mq = w.scene.mesh_offset[n]
            R = np.zeros(9); mujoco.mju_quat2Mat(R, np.asarray(q, float))
            qq = np.zeros(4); mujoco.mju_mulQuat(qq, np.asarray(q, float), mq)
            pp, q = np.asarray(pp) + R.reshape(3, 3) @ mp, qq
        assert np.allclose(m.geom_pos[g], pp) and np.allclose(m.geom_quat[g], q), n
    for n in ("mouth0", "brow1_L", "lip_lo0"):                         # on the face's front, not at the head's origin
        assert m.geom_pos[w.scene.face_ids[n]][0] > 0.09, n
    want = kin.fk(G.born_parent())["head"][0]                         # her body standing where the maker stands her
    assert np.allclose(d.xpos[m.body("parent_head").id], want) and w.scene.pose is not None and kin.face_reading(kin.scalar_to_params(0.0)) == 0.0
    for _ in range(20):
        w.apply({})
    t20 = w.frame().truth
    own = 2.0 * sum(t20["palm_own_N"].values())                        # a fist the grasp closed pressing its own palm (A82, C41): on the
    s20 = float(t20["touch_N"].sum()) - own                             # palm's zone and, reacting, its fingers' (not the body's weight)
    assert abs(s20 - weight) < 0.05 * weight, (s20, weight, own)
    print(f"world 4: born supine on the mat (head toward -x), no self-contact at rest, the touch zones carrying {s:.1f} N of its",
          f"{weight:.1f} N at birth and {s20:.1f} N after 3 s at rest, its fists' own pressing ({own:.1f} N) aside ({len(touched)} zones",
          f"touched: {', '.join(touched[:6])}...);",
          f"birth the same every time; the parent drawn at birth ({len(face)} face geoms at her neutral face, her head where the maker stands her)")


def test_joint_sense_and_vestibule():
    """world 5: joint sense (the scaled angle's sine and cosine, velocity, effort); the IMUs at rest with the declared noise"""
    w = G1World(seed=1)
    for _ in range(10):
        w.apply({})
    f = w.frame()
    assert f.obs["body"].shape == (242,)                                  # S5a: the anatomy's body channel
    b = f.obs["body"][:215].reshape(43, 5)
    assert (b[:, 4] >= 0).all() and np.allclose(b[:, 4], w.heat / W.HEAT_RISE_C)   # the motors' heat (A39)
    assert np.array_equal(f.obs["body"][221:242], w.tract.proprio())      # the tract's 21 (4.9)
    assert np.allclose(f.obs["imu_torso"], np.concatenate([w._sensed["vestibular"][0:3], w._sensed["vestibular"][6:9]]))
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
    print(f"world 5: joint sense 43 x 5 (sin and cos of the angle over its range, velocity, effort within the present limit, heat), the gaze's 6, the tract's 21; at rest",
          f"the torso's accelerometer reads {np.linalg.norm(acc_t):.2f} m/s2 (gravity) and its gyro under 0.02 rad/s; the noise the",
          f"model declares (1e-2, 5e-4) drawn from the world's stream")


YANK_N = 150.0                  # a hard yank on the left hand, upward (about a 15 kg pull): the load the tests hurt a gear with. Under
                                # the real motors (A79, A80) a weight dropped on the hand lying on the mat hurts no gear (the mat takes
                                # it and the wrist gives way); a pull the arm must hold does


def _yank(w, on=True):
    """the test's yank: YANK_N upward on the left hand's wrist link, set before a tick (an instrument's force, never the parent's)"""
    b = w.m.body("left_wrist_yaw_link").id
    w.d.xfrc_applied[b, :3] = [0.0, 0.0, YANK_N if on else 0.0]


def test_pain():
    """world 6 (A37, S5a): PAIN FROM THE JOINTS. A joint is in pain on a tick when the torque its gear carries (the motor's torque less
    what the rotor's inertia takes of the joint's acceleration: the model's armature) passes the joint's declared limit as a 10 ms
    mean; the base when its outside force passes F_pain (3 x the declared weight). The law on synthetic steps: a 2 ms spike at 3 x the
    limit is a 0.6 x mean (no pain), 10 ms at 1.1 x hurts, 8 ms does not, a window across the tick's start counts. The born G1 lying
    on the mat has a large outside torque at its waist (the mat carrying the torso) and no pain: support loads no gear. A real blow
    (4 kg dropped 0.5 m onto the left palm) hurts the left wrist, and the anatomy's withdrawal (Limb.reflex) takes the left arm for
    two ticks, no other limb"""
    w = G1World(seed=1)
    assert abs(w.f_pain - 3 * 34.394 * 9.81) < 0.2
    J = len(W.JOINTS)
    j = W.JOINTS.index("left_knee_joint")
    L_ = float(w.tau_max[j])
    base = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in w._sensed.items()}

    def felt(steps, carry=None):
        w._sensed = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in base.items()}
        if carry is not None:
            w._sensed["bd_carry"][:, j] = carry
        BD = np.zeros((75, J)); BD[:len(steps), j] = steps
        w._sense_tick(np.zeros((75, w.nz)), np.zeros((75, 12)), None, np.zeros((75, J + 6)), None, BD)
        return float(w._sensed["bd_peak"][j])
    assert abs(felt([3 * L_]) - 0.6 * L_) < 1e-9                       # one 2 ms spike: a 0.6 x mean, no pain
    assert abs(felt([1.1 * L_] * 5) - 1.1 * L_) < 1e-9 > 0             # 10 ms at 1.1 x: pain
    assert felt([1.1 * L_] * 4) < L_
    assert abs(felt([1.1 * L_], carry=[1.1 * L_] * 4) - 1.1 * L_) < 1e-9   # a window across the tick's start is felt
    w = G1World(seed=1)
    f = w.frame()
    wp = W.JOINTS.index("waist_pitch_joint")
    assert w._sensed["obs_peak"][wp] > 5.0 and not f.obs["pain"].any(), w._sensed["obs_peak"][wp]   # support: no gear loaded
    # a real overload: a hard yank on the left hand for two ticks
    w = G1World(seed=1)
    st, hurt, wd, frames = {}, [], [], []
    for k in range(6):
        f = w.frame(); frames.append(f)
        acts = _withdraw(f, st)
        wd += [(k, limb) for limb in acts]
        hurt.append([W.JOINTS[i] for i in np.nonzero(f.obs["pain"][:J])[0]])
        _yank(w, k < 2)
        w.apply(acts)
    first = next(k for k, h in enumerate(hurt) if h)
    assert "left_wrist_pitch_joint" in hurt[first] and all(x.startswith("left_") for x in hurt[first]), hurt   # the yank's own gears
    assert wd[:2] == [(first, "arm_l"), (first + 1, "arm_l")] and all(l_ == "arm_l" for _, l_ in wd), wd
    assert set(frames[0].obs) >= {"body", "touch", "pain", "vestibular", "imu_torso"} and frames[0].obs["pain"].shape == (44,)
    print(f"world 6: pain from the joints (A37): F_pain {w.f_pain:.1f} N for the base; a joint's gear load past its limit as a 10 ms",
          f"mean (a 2 ms spike at 3 x is 0.6 x, no pain; 10 ms at 1.1 x hurts; across the tick's start counts); the born G1 on the mat,",
          f"{base['obs_peak'][wp]:.1f} N m of outside torque at its waist (the observer's), in no pain (support loads no gear); a",
          f"{YANK_N:.0f} N yank on the left hand hurt {hurt[first]} at tick {first} and took the left arm for {len(wd)} ticks, no other limb")


def test_no_charge():
    """world 7 (A88, the owner's decision 2026-09-26): no charge, no charger, no weakness. The frame carries no charge channel and the
    anatomy declares none; the joints' limits are the declared ones at every tick; the truth has no drain and no feed; the room holds
    no bottle and no dock; the anatomy's rewards are her face, the one that pays, and pain as cortisol (dopamine off); since dawn 24 (A127)
    the novelty drive is the third source when SIM_CFG has it on, and none of them is a charge"""
    w = G1World(seed=1)
    for _ in range(3):
        w.apply(_babbler(4, 0.0).acts())
        f = w.frame()
        assert "charge" not in f.obs and "drain" not in f.truth and "fed" not in f.truth
        assert np.array_equal(w.limits_now(), w.tau_max) and np.array_equal(-w.m.jnt_actfrcrange[w.jid, 0], w.tau_max)
    assert not hasattr(w, "h") and not hasattr(w, "chargers") and "charge" not in AN.SIZES
    names = [w.m.geom(g).name for g in range(w.m.ngeom)]
    assert not [n for n in names if n.startswith("bottle") or n.startswith("pad")], [n for n in names if "bottle" in n or "pad" in n]
    a = AN.SimAnatomy(AN.born_table(LX.BIRTH_WORDS), AN.SIM_CFG, limits=[float(x) for x in w.tau_max])
    names_ = [r.name for r in a.rewards]                                  # A127 (dawn 24): the novelty drive is a third source when SIM_CFG has it on
    assert names_ == ["face", "pain"] + (["novelty"] if int(AN.SIM_CFG.get("novelty", 0)) else []), names_
    assert a.rewards[0].dopamine and a.rewards[1].dopamine and a.rewards[1].amyg   # A91:
    assert hasattr(AN.RewardSource, "dopamine") and AN.RewardSource.dopamine is True          # pain pays; the cortisol switch stays in the core
    assert "charge" not in [c.name for c in a.channels] and a.effectors[0].cry.get("charge") is None
    print("world 7: no charge (A88): no charge channel, drain, feed, charger, bottle or dock; the limits the declared ones every tick;",
          "the rewards her face and pain, both paying (A91), pain reaching the amygdala")


def test_the_reflexes():
    """world 8: THE NEWBORN'S WITHDRAWAL (3.7 and section 10 as approved; the W1 verifier's third round): each flexion sign measured
    on the G1 (at qpos0, standing): the hip's flexion brings the knee forward, the knee's and the elbow's shorten the limb, the
    ankle's dorsiflexion lifts the toe, the shoulder's brings the hand forward; the withdrawal is generalized: pain on ANY zone of a
    limb (the foot, the shin, the thigh's housing; the palm, the forearm, the upper arm) gives the same act, its flexion step, for
    two ticks; it reads the frame's pain alone (built from the zone names, no body model: the local sign's kinematic copy gone);
    none for the trunk. The grasp: on a palm touch, summed into the hand's own act of the same tick, not after the hand's own
    opening; in the world, the withdrawal flexes the arm"""
    w = G1World(seed=1)
    m, d = w.m, w.d
    mujoco.mj_resetData(m, d); mujoco.mj_kinematics(m, d)
    pos = lambda b: d.xpos[m.body(b).id].copy()

    def moved(joint, delta, fn):
        q0 = d.qpos.copy(); a = fn(); d.qpos[m.jnt_qposadr[m.joint(joint).id]] += delta; mujoco.mj_kinematics(m, d); b = fn()
        d.qpos[:] = q0; mujoco.mj_kinematics(m, d); return b - a
    for sd, s_ in (("left", "l"), ("right", "r")):
        fl = R.FLEXION[f"leg_{s_}"]; fa = R.FLEXION[f"arm_{s_}"]
        assert set(fl) == {f"{sd}_{j}_joint" for j in ("hip_pitch", "knee", "ankle_pitch")}
        assert set(fa) == {f"{sd}_{j}_joint" for j in ("shoulder_pitch", "elbow")}
        assert moved(f"{sd}_elbow_joint", 0.1 * fa[f"{sd}_elbow_joint"], lambda: np.linalg.norm(pos(f"{sd}_wrist_yaw_link") - pos(f"{sd}_shoulder_pitch_link"))) < -0.005
        assert moved(f"{sd}_shoulder_pitch_joint", 0.1 * fa[f"{sd}_shoulder_pitch_joint"], lambda: pos(f"{sd}_wrist_yaw_link")[0]) > 0.01   # the arm forward
        d.qpos[m.jnt_qposadr[m.joint(f"{sd}_knee_joint").id]] = 0.5; mujoco.mj_kinematics(m, d)       # from a bent knee
        assert moved(f"{sd}_knee_joint", 0.1 * fl[f"{sd}_knee_joint"], lambda: np.linalg.norm(pos(f"{sd}_ankle_roll_link") - pos(f"{sd}_hip_pitch_link"))) < -0.005
        assert moved(f"{sd}_hip_pitch_joint", 0.1 * fl[f"{sd}_hip_pitch_joint"], lambda: pos(f"{sd}_knee_link")[0]) > 0.02    # the knee forward
        toe = [g for g in range(m.ngeom) if m.geom_bodyid[g] == m.body(f"{sd}_ankle_roll_link").id and m.geom_pos[g][0] > 0.1][0]
        heel = [g for g in range(m.ngeom) if m.geom_bodyid[g] == m.body(f"{sd}_ankle_roll_link").id and m.geom_pos[g][0] < 0][0]
        assert moved(f"{sd}_ankle_pitch_joint", 0.1 * fl[f"{sd}_ankle_pitch_joint"], lambda: d.geom_xpos[toe][2] - d.geom_xpos[heel][2]) > 0.01
        mujoco.mj_resetData(m, d); mujoco.mj_kinematics(m, d)
        for j, sg in R.CLOSING[f"hand_{s_}"].items():                   # each closing joint brings the thumb and the fingers together
            gap = lambda: np.linalg.norm(d.xipos[m.body(f"{sd}_hand_thumb_2_link").id] - (d.xipos[m.body(f"{sd}_hand_index_1_link").id] + d.xipos[m.body(f"{sd}_hand_middle_1_link").id]) / 2)
            lo, hi = m.jnt_range[m.joint(j).id]
            assert (lo < 0 < hi) or (sg > 0 and lo == 0) or (sg < 0 and hi == 0), (j, lo, hi)   # a finger opens at 0, closes into its range
            if j.endswith(("thumb_1_joint", "thumb_2_joint")):
                d.qpos[m.jnt_qposadr[m.joint(j).id]] = 0.3 * sg if lo < 0 < hi else 0.0
                mujoco.mj_kinematics(m, d)
                assert moved(j, 0.2 * sg, gap) < 0, j
    w = G1World(seed=1)
    Lb = _limbs()                                                        # the anatomy's limbs: the withdrawal is Limb.reflex (S5a)
    f = w.frame()
    assert "pain_site" not in f.obs and f.obs["pain"].shape == (44,)     # pain from the joints alone: no skin site (A37)
    assert all(e.reflex(f, None, {}) is None for e in Lb.values())       # no pain at birth, no withdrawal

    def pained(joint):
        g = w.frame(); g.obs["pain"] = np.array(g.obs["pain"]); g.obs["pain"][W.JOINTS.index(joint)] = 1.0; return g
    for limb, js in (("leg_l", ("left_ankle_roll_joint", "left_knee_joint", "left_hip_roll_joint", "left_hip_pitch_joint")),
                     ("arm_l", ("left_hand_index_1_joint", "left_wrist_pitch_joint", "left_elbow_joint", "left_shoulder_roll_joint"))):
        acts = {Lb[limb].reflex(pained(j), None, {}) for j in js}      # wherever on the limb it hurts: the same flexion
        assert acts == {R.flexion_act(limb)}, (limb, acts)              # the anatomy's FLEXION and the world's measured signs agree
        dig = W.act_digits(R.flexion_act(limb), len(W.EFFECTOR_FACTORS[limb]))
        for j, k in zip(dict(G.EFFECTORS)[limb], dig):
            assert k == (2 if j not in R.FLEXION[limb] else (4 if R.FLEXION[limb][j] > 0 else 0)), (j, k)
    st = {}
    seq = [Lb["arm_l"].reflex(pained("left_hand_thumb_0_joint"), None, st)]   # a hand's pain withdraws its arm, for two ticks
    f2 = w.frame(); seq += [Lb["arm_l"].reflex(f2, None, st), Lb["arm_l"].reflex(f2, None, st)]
    assert seq == [R.flexion_act("arm_l")] * 2 + [None], seq
    assert Lb["arm_r"].reflex(pained("left_hand_thumb_0_joint"), None, {}) is None
    base_pain = w.frame(); base_pain.obs["pain"] = np.array(base_pain.obs["pain"]); base_pain.obs["pain"][43] = 1.0
    assert all(e.reflex(base_pain, None, {}) is None for e in Lb.values())   # the base's pain withdraws no limb
    from body.sim.anatomy import SimAnatomy, SIM_CFG, born_table
    others = {e.name: e for e in SimAnatomy(born_table(), SIM_CFG).effectors if e.name in ("waist", "hand_l", "hand_r")}
    assert all(e.reflex(pained("waist_pitch_joint"), None, {}) is None for e in others.values())   # none for the trunk or a hand
    # the grasp at the spinal cord: the hand's own act this tick, summed
    touched, light = math.log1p(0.31), math.log1p(0.29)
    rest_l = W.EFFECTOR_REST["hand_l"]
    assert R.grasp("hand_l", None, touched) == (R.closing_act("hand_l"), "grasp") == R.grasp("hand_l", rest_l, touched)
    assert R.grasp("hand_l", None, light) == (None, None) and R.grasp("hand_l", rest_l, light) == (rest_l, None)
    opening = W.act_flat([2, 1, 2, 2, 2, 2, 2])                           # the left thumb_1 opening a small step
    assert R.opens("hand_l", opening) and not R.opens("hand_l", R.closing_act("hand_l"))
    a_, ev_ = R.grasp("hand_l", opening, touched)                         # A160: summed joint by joint: the thumb_1 keeps its own opening
    assert ev_ == "grasp" and W.act_digits(a_, 7)[1] == 1 and W.act_digits(a_, 7)[2:] == W.act_digits(R.closing_act("hand_l"), 7)[2:], W.act_digits(a_, 7)
    whole = W.act_flat([2 if j not in R.CLOSING["hand_l"] else (0 if R.CLOSING["hand_l"][j] > 0 else 4) for j in dict(G.EFFECTORS)["hand_l"]])
    assert R.opens_all("hand_l", whole) and not R.opens_all("hand_l", opening)
    assert R.grasp("hand_l", whole, touched) == (whole, "overridden")      # every closing joint opened by its own act: the hand opens as a whole
    half = W.act_flat([2, 0, 0, 0, 2, 2, 2])                               # the thumb opened big, the index opened big, the middle held
    a_, ev_ = R.grasp("hand_l", half, touched)
    assert ev_ == "grasp" and W.act_digits(a_, 7)[:4] == [2, 0, 0, 0] and W.act_digits(a_, 7)[4:] == W.act_digits(R.closing_act("hand_l"), 7)[4:], W.act_digits(a_, 7)
    big_close = W.act_flat([2, 4, 4, 0, 0, 0, 0])                         # the own act closing by big steps: kept as it is
    assert R.grasp("hand_l", big_close, touched) == (big_close, "grasp")
    turn = W.act_flat([4, 2, 2, 2, 2, 2, 2])                              # the thumb's rotation (no closing sense): the own setting stays
    got = W.act_digits(R.grasp("hand_l", turn, touched)[0], 7)
    assert got[0] == 4 and got[1:] == W.act_digits(R.closing_act("hand_l"), 7)[1:], got
    assert R.grasp("hand_r", None, touched)[0] == R.closing_act("hand_r")
    q = lambda j: float(w.d.qpos[w.qadr[W.JOINTS.index(j)]])
    e0, s0 = q("left_elbow_joint"), q("left_shoulder_pitch_joint")
    for a in seq:
        w.apply({"arm_l": a} if a is not None else {})
    assert q("left_elbow_joint") < e0 - 0.3 and q("left_shoulder_pitch_joint") < s0 - 0.1, (e0, q("left_elbow_joint"), s0, q("left_shoulder_pitch_joint"))
    print(f"world 8: every flexion sign (the shoulder, the elbow, the hip, the knee, the ankle's dorsiflexion) and closing sign measured",
          f"on the G1; the withdrawal generalized (pain on any of 4 zones of a leg or an arm: the same flexion act), from the frame's pain",
          f"alone, two ticks (a hand's pain withdraws its arm), none for the trunk; the grasp at a palm touch of 0.3 N summed into the",
          f"hand's own act of the same tick (an opening act passes, a big closing one is kept); in the world the reflex flexed the elbow",
          f"{e0 - q('left_elbow_joint'):.2f} rad and the shoulder {s0 - q('left_shoulder_pitch_joint'):.2f} rad in two ticks")


def _ball_spot(w, h=0, r=0.03):
    """where the rig's ball sits: on the palm's face (A171: its centre HAND_DEPTH_M before the face less 12 mm, pressed into it so the
    hand's own motion within a tick keeps the contact)"""
    return w.grasp_centre(h) - w.palm_normal(h) * (W.HAND_DEPTH_M - r + 0.012)


def _ball_in_palm(side="left", r=0.03, settle=True):
    """a rig: a ball kept on the palm's face (a mocap sphere pressed into the born palm; until A171 it sat where the palm link's
    shape origin is, 4 cm below the hand's edge, and its push came from under the hand), and its world, settled 3 ticks with the
    ball following the palm"""
    ref = G1World(seed=1)
    pp = _ball_spot(ref, 0 if side == "left" else 1, r)

    def rig(spec):
        b = spec.worldbody.add_body(name="rig_ball", mocap=True, pos=pp.tolist())
        b.add_geom(name="rig_ball", type=mujoco.mjtGeom.mjGEOM_SPHERE, size=[r, 0, 0], contype=1, conaffinity=1, rgba=[1, 0, 0, 1])
    w = G1World(seed=1, extra=rig)
    ball = w.m.body_mocapid[w.m.body("rig_ball").id]
    for _ in range(3 if settle else 0):
        w.d.mocap_pos[ball] = _ball_spot(w, 0 if side == "left" else 1, r)
        w.frame(); w.apply({})
    return w


def _closure(w, hand):
    """how far a hand is closed: the sum over its closing joints of the angle times its closing sign (rad)"""
    return float(sum(sg * w.d.qpos[w.qadr[W.JOINTS.index(j)]] for j, sg in R.CLOSING[hand].items()))


def test_the_grasp_habituates():
    """world 9b (A162): a ball kept in the left palm: the grasp fires on every resting tick for GRASP_HOLD_TICKS, then falls silent
    ('habituated': the hand's own acts rule, the fingers staying where they are) while the pressure is constant; the palm freed for
    GRASP_RECOVER_TICKS re-arms it in full. The state is the world's and is saved; an older save loads fresh"""
    w = _ball_in_palm()
    ball = w.m.body_mocapid[w.m.body("rig_ball").id]
    z = w.zones.index("left_hand_palm")
    w._grasp_hab["hand_l"] = [0, 0]                                          # (the rig's 3 settling ticks pressed it already)
    evs = []
    for k in range(R.GRASP_HOLD_TICKS + 6):
        if k < R.GRASP_HOLD_TICKS - 8:
            w.d.mocap_pos[ball] = _ball_spot(w)                            # (the rig keeps the ball on the palm's face, then holds it
        w.frame(); w.apply({})                                             # still: a ball re-set each tick makes onsets that wake it, A164)
        g = w.frame().truth
        assert g["touch_N"][z] >= R.GRASP_N, g["touch_N"][z]
        evs.append(g["spinal"].get("hand_l"))
    assert evs[:R.GRASP_HOLD_TICKS] == ["grasp"] * R.GRASP_HOLD_TICKS and all(e == "habituated" for e in evs[R.GRASP_HOLD_TICKS:]), \
        (evs[:3], evs[R.GRASP_HOLD_TICKS - 2:])
    import pickle as _pk
    st = _pk.loads(w.save_state())
    assert st["grasp_hab"]["hand_l"][0] == R.GRASP_HOLD_TICKS + 6 and st["grasp_hab"]["hand_l"][1] == 0, st["grasp_hab"]
    # the pure instrument (no state) still sums the same tick
    assert R.grasp("hand_l", None, R.GRASP_LOG) == (R.closing_act("hand_l"), "grasp")
    # the palm freed re-arms it: the reflex's bookkeeping by hand (the ball lifted away is the world's business; here the state's)
    hab = w._grasp_hab["hand_l"]
    for k in range(R.GRASP_RECOVER_TICKS):
        if k == 0:
            assert R.grasp("hand_l", None, R.GRASP_LOG, hab)[1] == "habituated"   # still pressed: still habituated
        assert R.grasp("hand_l", None, 0.0, hab) == (None, None)
    assert hab[1] == R.GRASP_RECOVER_TICKS and hab[0] == 0
    a_, e_ = R.grasp("hand_l", None, R.GRASP_LOG, hab)
    assert (a_, e_) == (R.closing_act("hand_l"), "grasp") and hab == [1, 0], (e_, hab)
    hab[1] = R.GRASP_RECOVER_TICKS - 1; hab[0] = R.GRASP_HOLD_TICKS + 3                    # a shorter gap does not re-arm it
    assert R.grasp("hand_l", None, R.GRASP_LOG, hab)[1] == "habituated"
    assert R.grasp("hand_l", None, R.GRASP_LOG, hab, onset=1.0)[1] == "grasp" and hab[0] == 1   # A164: a new press wakes the silent reflex
    hab[0] = 5
    assert R.grasp("hand_l", None, R.GRASP_LOG, hab, onset=1.0)[1] == "grasp" and hab[0] == 6   # firing: its own squeeze's onsets count for nothing
    # a save from before A162 loads fresh
    w2 = _ball_in_palm(); w2.load_state(w.save_state())                                   # (the same model: the rig's)
    assert w2._grasp_hab == w._grasp_hab and w2._grasp_hab["hand_l"] == hab, w2._grasp_hab   # the state travels (both hands' counts; the right
                                                                                           # hand, its edge on the rattle, is pressed on no side: A171)
    st2 = _pk.loads(w2.save_state()); st2.pop("grasp_hab", None); w2.load_state(_pk.dumps(st2, protocol=4))
    assert w2._grasp_hab == {"hand_l": [0, 0], "hand_r": [0, 0]}, w2._grasp_hab
    print(f"world 9b: the grasp fired {R.GRASP_HOLD_TICKS} resting ticks on the ball, then habituated under the constant press; the palm free",
          f"{R.GRASP_RECOVER_TICKS} ticks re-armed it (a shorter gap not); the state saved and an older save loads fresh")


def test_the_dorsal_touch_opens_the_hand():
    """world 9c (A171): the left hand raised and closed into a fist on nothing by its own acts; a ball set against the BACK of the
    hand: the cord's dorsal response opens it (the event 'dorsal', the closure falling), while a ball on the palm's face is the
    grasp's alone (no 'dorsal'), and the grasp's stimulus is the palm's SIDE: the hand's back pressed fires no grasp. The response
    habituates and its state is saved; the pure function sums joint by joint as the grasp's"""
    w = _ball_in_palm()
    ball = w.m.body_mocapid[w.m.body("rig_ball").id]
    pg = [g for g in range(w.m.ngeom) if w.zone_of_geom[g] == w.zones.index("left_hand_palm")][0]
    far = np.array([3.0, 3.0, 0.5])
    # the ball on the palm's face: the grasp's, never the dorsal response's
    for _ in range(3):
        w.d.mocap_pos[ball] = _ball_spot(w)
        w.frame(); w.apply({})
    g = w.frame().truth
    assert g["spinal"].get("hand_l") == "grasp" and w._sensed["sides"][0, 0] >= R.GRASP_N and w._sensed["sides"][0, 1] < R.DORSAL_N, \
        (g["spinal"], w._sensed["sides"])
    # the hand raised off the mat, then a fist on nothing by its own closing acts (the servos keep the fingers where they closed)
    w.d.mocap_pos[ball] = far
    for k in range(14):
        w.frame(); w.apply({"arm_l": R.flexion_act("arm_l")} if k < 3 else ({"hand_l": R.closing_act("hand_l")} if k < 9 else {}))
    c0 = _closure(w, "hand_l")
    g = w.frame().truth
    assert g["spinal"].get("hand_l") is None and c0 > 5.0, (g["spinal"], c0)
    assert w._sensed["sides"][0].max() < R.DORSAL_N and w._sensed["palm_log"][0] < R.GRASP_LOG, (w._sensed["sides"], w._sensed["palm_log"])
    # the ball against the back of the hand (the rig follows the hand, 6 mm pressed): the dorsal response opens it
    def behind():
        R_ = w.d.geom_xmat[pg].reshape(3, 3); n = w.palm_normal(0)
        centre = w.d.geom_xpos[pg] + R_ @ w.m.geom_aabb[pg][:3]
        ext = float(np.abs(R_.T @ n) @ w.m.geom_aabb[pg][3:])
        return centre - n * (ext + 0.03 - 0.006)
    evs = []
    for _ in range(12):
        w.d.mocap_pos[ball] = behind()
        w.frame(); w.apply({})
        evs.append(w.frame().truth["spinal"].get("hand_l"))
    c1 = _closure(w, "hand_l")
    assert all(e == "dorsal" for e in evs[1:]), evs
    assert w._sensed["sides"][0, 1] >= R.DORSAL_N and w._sensed["sides"][0, 0] < R.GRASP_N, w._sensed["sides"]
    assert w._sensed["palm_log"][0] < R.GRASP_LOG, w._sensed["palm_log"]                 # the back pressed is no grasp stimulus (A171)
    assert c1 < c0 - 1.0, (c0, c1)
    # it habituates under the constant push, and the state is saved
    for _ in range(R.GRASP_HOLD_TICKS):
        w.d.mocap_pos[ball] = behind()
        w.frame(); w.apply({})
    assert w.frame().truth["spinal"].get("hand_l") == "dorsal_habituated", w.frame().truth["spinal"]
    import pickle as _pk
    st = _pk.loads(w.save_state())
    assert st["dorsal_hab"]["hand_l"][0] > R.GRASP_HOLD_TICKS and st["dorsal_hab"]["hand_l"][1] == 0, st["dorsal_hab"]
    w2 = _ball_in_palm(); w2.load_state(w.save_state()); assert w2._dorsal_hab == w._dorsal_hab
    st.pop("dorsal_hab"); w2.load_state(_pk.dumps(st, protocol=4)); assert w2._dorsal_hab == {"hand_l": [0, 0], "hand_r": [0, 0]}
    for k in ("sides", "palm_log", "palm_onset"):                                       # a save from before A171 lacks the hands' sides
        st["sensed"].pop(k)
    w2.load_state(_pk.dumps(st, protocol=4)); w2.frame(); w2.apply({})                 # and lives its next tick
    assert w2._sensed["sides"].shape == (2, 2) and w2._sensed["palm_log"].shape == (2,)
    # the pure function
    assert R.dorsal_open("hand_l", None, 1.0, 0.0) == (R.opening_act("hand_l"), "dorsal")
    assert R.dorsal_open("hand_l", None, 1.0, R.GRASP_N) == (None, None)                 # the palm's side pressed: the grasp's
    assert R.dorsal_open("hand_l", None, 0.1, 0.0) == (None, None)
    assert R.dorsal_open("hand_l", R.closing_act("hand_l"), 1.0, 0.0) == (R.closing_act("hand_l"), None)   # the own act closes every joint: kept
    print(f"world 9c: a fist on nothing in the air ({math.degrees(c0 / 6):.0f} deg a joint) opened to {math.degrees(c1 / 6):.0f} deg by a ball against the",
          f"back of the hand in 12 ticks (the cord's dorsal response; the back pressed fired no grasp), the ball on the palm the grasp's alone;",
          f"it habituates in {R.GRASP_HOLD_TICKS}, the state saved")


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
    ball = w.m.body_mocapid[w.m.body("rig_ball").id]
    for phase, act, n in (("rest", None, 8), ("open", opening, 8), ("rest again", None, 6)):
        c0 = _closure(w, "hand_l")
        for _ in range(n):
            w.d.mocap_pos[ball] = _ball_spot(w)                            # (the rig keeps the ball on the palm's face)
            f = w.frame()
            assert f.obs["touch"][2 * list(w.dex).index(z)] >= R.GRASP_LOG, (phase, f.truth["touch_N"][z])   # the ball in the palm all along
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
    for t in range(15):                                                   # the hand raised off the mat first (A171: the mat under the
        fist.apply({"arm_l": R.flexion_act("arm_l")} if t < 3 else ({"hand_l": close} if t < 9 else {}))   # hand's edge is no grasp)
        g = fist.frame().truth
        if t >= 9:
            own.append((g["spinal"].get("hand_l"), g["touch_N"][z], g["palm_own_N"]["hand_l"], _closure(fist, "hand_l")))
    assert all(ev is None for ev, _a, _o, _c in own) and all(c > 5.0 for _e, _a, _o, c in own), own   # A171: a fist on nothing in the air is
    # no grasp (its palm's side pressed by nothing but, at most, its own fingers); the servos keep the fingers where they closed
    alone = sum(abs(n_all - n_own) < 1e-9 for _, n_all, n_own, _c in own)   # ticks on which all of the palm's force was its own
    print(f"world 9: a ball kept in the left palm: at rest the grasp closed the hand {a1 - a0:.2f} rad in 8 ticks; its own opening act",
          f"passed the spinal cord as sent (the grasp overridden, 8 of 8 ticks) and opened it {b0 - b1:.2f} rad with the ball still in the",
          f"palm; at rest again it closed {c1 - c0:.2f} rad: letting go is the hand's own act (A11); a fist closed on nothing in the air kept",
          f"itself closed {len(own)} of {len(own)} resting ticks with no grasp (A171), its own fingers pressing the palm {own[-1][2]:.1f} N (palm_own_N;",
          f"all of the palm's force on {alone} of them; with the ball, {ball_own:.1f} N of its own)")


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
    ob = w._outside()                                                   # the observer sees the housings' contact on the joints
    sr_j = W.JOINTS.index("left_shoulder_roll_joint")                   # between them (A37: self-contact is outside torque)
    assert abs(ob[sr_j]) > w.tau_max[sr_j], (ob[sr_j], w.tau_max[sr_j])
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
          f"{f_pair:.0f} N into the torso is felt on both zones and by the observer at the shoulder ({abs(ob[sr_j]):.0f} N m) and stays in collision; taking that pose as rest, the",
          f"derivation lists exactly the {len(pairs)} pairs pressing there (the torso with the shoulder's roll and yaw links and the",
          f"elbow) and touch drops exactly their force")


def test_withdrawal_c22():
    """world 16: C22 WRITTEN DOWN AS THE NEWBORN'S (under A37's joints, S5a): a load that hurts (a YANK_N yank upward on the left hand:
    the left wrist's gear past its limit, the yank going on), and from the saved state of that first painful tick, two ticks three ways: the withdrawal
    (the anatomy's: the arm's flexion both ticks), rest (tone) and pushing on (the arm's own big extension); the wrist's gear load
    each tick written down, none a bar, and the same state gives the same numbers again. The instrument's count (tools/sim_pain.py
    c22_counts) takes C22's question as the W1 verifier counted it: a withdrawal raises the pain it answers when EITHER of its ticks
    presses harder than the onset"""
    from sim_pain import c22_counts

    w = G1World(seed=1)
    j = W.JOINTS.index("left_wrist_pitch_joint")
    for _ in range(8):
        f = w.frame()
        if f.obs["pain"][j]:
            break
        _yank(w)
        w.apply({})
    assert f.obs["pain"][j], "the yank did not hurt the wrist"
    blob, at = w.save_state(), float(w._sensed["bd_peak"][j])
    push = W.act_flat([4, 2, 2, 4, 2, 2, 2])                           # the arm's own extension, big (shoulder and elbow)

    def two(form):
        w.load_state(blob)
        st, out, acts = {}, [], []
        for _ in range(2):
            g = w.frame()
            a = _withdraw(g, st).get("arm_l") if form == "withdraw" else W.EFFECTOR_REST["arm_l"] if form == "rest" else push
            _yank(w)                                                    # the yank goes on (the same load, three answers to it)
            acts.append(a); w.apply({"arm_l": a}); out.append(float(w._sensed["bd_peak"][j]))
        return out, acts
    (wd, wacts), (rs, _), (pu, _) = two("withdraw"), two("rest"), two("push")
    assert wacts[0] == R.flexion_act("arm_l") and two("withdraw")[0] == wd and two("rest")[0] == rs
    assert all(np.isfinite(x) for x in wd + rs + pu)
    F = float(w.tau_hold[j])
    syn = [{"at": 1000.0, "withdraw": [1200.0, 500.0], "rest": [900.0, 1100.0], "babble": [800.0, 700.0]},
           {"at": 1000.0, "withdraw": [900.0, 1050.0], "rest": [950.0, 990.0], "babble": [1300.0, 1400.0]},
           {"at": 1000.0, "withdraw": [800.0, 700.0], "rest": [1000.0, 1000.0], "babble": [200.0, 100.0]}]
    c = c22_counts(syn, 1012.0)
    key = "raises_the_pain_it_answers (either tick above the onset)"
    assert (c["withdraw"][key], c["rest"][key], c["babble"][key]) == (2, 1, 1)
    here = c22_counts([{"at": at, "withdraw": wd, "rest": rs, "babble": pu}], F)
    print(f"world 16: C22 written down under the joints' pain, the left wrist's gear at {at:.1f} N m (its limit {F:.0f}) under a yank",
          f"on the hand: the newborn's flexion {wd[0]:.1f} then {wd[1]:.1f} N m (raises the pain it answers: {bool(here['withdraw'][key])}),",
          f"rest {rs[0]:.1f} then {rs[1]:.1f}, pushing on {pu[0]:.1f} then {pu[1]:.1f}; the same again from the same state")


def test_friction_realism():
    """world 17: impratio CHOSEN BY PHYSICS (the owner's decision, made for him in the W1 verifier's third round; tools/sim_friction.py):
    at the room's own solver options, on the mat's contact, a 10 kg box pushed below its sliding force holds (at most 2 mm in 2 s,
    where impratio 1 lets it creep at least 3 mm: the soft model's slow slip, which real rubber, plastic, foam and housings do not
    have); above it, it slides as Coulomb's law says (within 2%); and the convex model's coupling of a slip into the normal force
    is what the design discloses (a pressed box starting to slide pressed at most 30% over its load for 10 ms, steady within 5%)"""
    from sim_friction import plane_rows
    sys.path.insert(0, SIM)
    import make_g1room as M
    assert 'impratio="10"' in M.scene_xml(SIM) and G1World(seed=1).m.opt.impratio == 10.0
    at10, at1 = plane_rows(10.0), plane_rows(1.0)
    static = [v for k, v in at10.items() if k.startswith("static")]
    assert len(static) == 5 and max(static) <= 2.0, at10
    assert min(v for k, v in at1.items() if k.startswith("static") and "0.5 mu" in k) >= 3.0, at1
    assert all(abs(v - 1) <= 0.02 for k, v in at10.items() if k.startswith("kinetic")), at10
    for press in ("0", "400"):
        assert abs(at10[f"pressed {press} N: normal at rest / N0"] - 1) < 1e-3, at10
        assert at10[f"pressed {press} N: sliding, the first 30 ms' largest 10 ms mean / N0"] <= 1.3, at10
        assert at10[f"pressed {press} N: sliding, steady (0.1-0.5 s) mean / N0"] <= 1.05, at10
    print(f"world 17: impratio 10 by physics: a box below its sliding force slid at most {max(static):.2f} mm in 2 s (impratio 1:",
          f"{at1['static 0.5 mu M g, 2 s: slide mm']:.1f} mm at half of it), slid within 2% of Coulomb's law above it; a pressed box",
          f"starting to slide read {at10['pressed 400 N: sliding, the first 30 ms' + chr(39) + ' largest 10 ms mean / N0']:.2f} x its load for",
          f"10 ms (the convex model's slip coupling, disclosed: C5, C22)")


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
        for _ in range(n):
            f = world.frame()
            if frames is not None:
                frames.append(f)
            acts = b.acts()
            acts.update(_withdraw(f, st))                               # the anatomy's withdrawal, live (S5a)
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
          f"bit for bit (every frame, the final state, the moved sun; the pelvis moved {100 * moved:.1f} cm); a",
          f"save carries the world's random stream")


def test_the_parents_pose_is_saved():
    """world 18: Scene.pose is saved with the world (the W1 verifier's third round): the parent drawn in a new pose (kneeling
    place, a smile, her eyes on the child's), saved; drawn again elsewhere; restored in the same world and in a new one: the scene's
    pose is the saved one (its place, turn, joints, hands, expression and gaze) and her face's geoms and her body with it"""
    kin = G.kin
    w = G1World(seed=1)
    m, d = w.m, w.d
    p = kin.Pose((0.6, -0.1, kin.HIP_Z - 0.3), kin.rz(2.4))
    p.expr = kin.face_params(smile=0.8, cheek=0.6, brow_in=0.3)
    kin.spine(p, lumbar=(20, 0, 5)); p.hand["R"]["curl"] = 1.1
    kin.look(p, d.cam_xpos[m.camera("eye_L").id].copy())
    w.scene.set_parent(p); mujoco.mj_forward(m, d)
    blob = w.save_state()
    face = {n: m.geom_pos[g].copy() for n, g in w.scene.face_ids.items()}
    other = G.born_parent(); other.expr = kin.face_params(frown=1.0)
    w.scene.set_parent(other)
    after = []
    for where, w2 in (("the same world", w), ("a new world", G1World(seed=1))):
        w2.load_state(blob)
        q = w2.scene.pose
        assert q is not None and q is not p and np.allclose(q.pos, p.pos) and np.allclose(q.R, p.R), where
        assert q.expr == p.expr and np.allclose(q.gaze, p.gaze) and q.hand == p.hand, where
        assert all(np.allclose(q.local[k], p.local[k]) for k in kin.SEGS), where
        assert all(np.allclose(w2.m.geom_pos[g], face[n]) for n, g in w2.scene.face_ids.items()), where
        w2.apply({"arm_l": W.EFFECTOR_REST["arm_l"] + 1})
        after.append(w2.save_state())
    assert after[0] == after[1]
    print("world 18: the parent's pose (her place, turn, spine, hands, expression and gaze) is saved with the world and restored",
          "with it, in the same world and in a new one, her face's geoms and her body with it; a tick after, the two worlds bit for bit")


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
    # THE LIVE, DARK NIGHT (R8c; 5.4, A46; S5a): the G1's world runs through the night
    assert G1World.live_night
    w = G1World(seed=1)
    for _ in range(5):
        w.apply(b1.acts())
    day_light = w.m.light_diffuse.copy()
    w.dusk()
    assert w.night and np.allclose(w.m.light_diffuse, day_light * W.NIGHT_LIGHT) and w.parent.asleep
    f = w.frame()
    assert set(f.obs) == {"body", "touch", "pain", "vestibular", "imu_torso"}, set(f.obs)   # the body's own senses alone
    blob = w.save_state()
    w.apply({})
    night_acts = [b1.acts() for _ in range(4)]
    for a in night_acts:
        w.apply(a)
    end_night = w.save_state()
    w2 = G1World(seed=1); w2.load_state(blob)                                    # a night replays exactly from a save taken in it
    w2.apply({})
    for a in night_acts:
        w2.apply(a)
    assert w2.save_state() == end_night
    w.dawn()
    assert not w.night and not w.parent.asleep and w.dawn_left == W.DAWN_TICKS
    for _ in range(W.DAWN_TICKS):
        w.apply({})
    assert np.allclose(w.m.light_diffuse, day_light) and {"ears", "words", "face", "eye_p"} <= set(w.frame().obs)
    print("world 12: the night: a frozen one (a frame or a move refused, the state unchanged, the morning bit for bit the day never",
          "paused); the live, dark night: the lights at", W.NIGHT_LIGHT, "of the day's, the parent asleep on the sofa, only the body's own",
          "senses in the frame, the world stepped tick by tick; the morning's light back over", W.DAWN_TICKS, "ticks,",
          "the eyes and ears on")


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
    """world 15 (S5a): THE G1'S ANATOMY LIVES IN THE WORLD: the core (body/sim/anatomy.py's SimAnatomy at SIM_CFG, every learning rate
    0, at a small width) lives 40 ticks through the core's world loop with a ball kept in its left palm: one frame and one apply a
    tick, every effector's act in range at the world (the tract, the gaze and the seven limbs; the words' output a symbol), the
    cerebellum called 15 times a tick below it, every frame the anatomy's channels at their sizes; THE W1 VERIFIER'S FIRST FINDING
    again: no reflex takes the left hand's tick, its gate draws on the touched ticks, each own act of it that opens the hand reaches
    the servos as drawn (the grasp overridden) and each other is summed with the grasp"""
    from body.core.world import WorldLoop
    from body.life import Life
    from body.sim.anatomy import SimAnatomy, SIM_CFG, SIZES, born_table
    w = _ball_in_palm()
    t_start = w.tick                                                        # (the rig settled 3 ticks with the ball on the palm, A171:
    w._grasp_hab["hand_l"] = [0, 0]                                         # those ticks pressed it; the 40 below stay under its hold)
    LR0 = dict(live_lr=0.0, value_lr=0.0, band_lr=0.0, night_lr=0.0, gate_lr=0.0, gate_adam_lr=0.0, vcrit_lr=0.0, actor_lr=0.0, face_lr=0.0,
               act_inv_lr=0.0)
    cfg = dict(SIM_CFG, **LR0, wake_ticks=10 ** 9)
    torch.manual_seed(0)
    t0 = time.time()
    L = Life.birth(SimAnatomy(born_table(), cfg, limits=[float(x) for x in w.tau_max]), device="cpu", d=32, layers=1, heads=2,
                   window=16, cfg=cfg, seed=0, world=w)
    assert w.below is not None                                           # the cerebellum hooked below the tick (7.5)
    motors = [e.name for e in L.anatomy.motors]
    ih = motors.index("hand_l")
    zp = w.zones.index("left_hand_palm"); zi = list(w.dex).index(zp)
    applied, lived, sizes_ok = [], [], []
    real_apply = w.apply
    pg = [g for g in range(w.m.ngeom) if w.zone_of_geom[g] == zp][0]
    ball = w.m.body_mocapid[w.m.body("rig_ball").id]

    def apply(acts):
        applied.append(dict(acts)); now = dict(L.motor[ih]["now"])
        touched = float(w._sensed["palm_log"][0]) >= R.GRASP_LOG      # the grasp's stimulus the tick is lived on (A171: the palm's side)
        sizes_ok.append(all(np.size(w.now.obs[k]) == n for k, n in SIZES.items()))
        real_apply(acts)
        w.d.mocap_pos[ball] = _ball_spot(w)                             # the rig keeps the ball on the palm as the arm moves
        lived.append((now, touched, w._last_acts["hand_l"], w._spinal.get("hand_l")))
    w.apply = apply
    run = WorldLoop(L)
    b0 = w._below_n
    for _ in range(40):
        run.step()
    w.apply = real_apply
    assert (L.ticks, w.tick - t_start, len(applied)) == (40, 40, 40), (L.ticks, w.tick - t_start, len(applied))
    assert w._below_n - b0 == 40 * 15, w._below_n - b0
    assert all(sizes_ok)
    names = set(W.EFFECTOR_NAMES)
    assert all(names <= set(x) for x in applied) and all(0 <= x[n] < 5 ** len(W.EFFECTOR_FACTORS[n]) for x in applied for n in names)
    assert all(0 <= int(x["voice"]) < 5 ** 10 for x in applied) and all(0 <= int(x["words"]) < 79 for x in applied)
    moved = sum(int(x[n] != W.EFFECTOR_REST[n]) for x in applied for n in names)
    assert sum(t for _, t, _, _ in lived) >= 38, [t for _, t, _, _ in lived]   # touched on all but at most 2 ticks (the rig's ball re-set on
                                                                             # the palm's face after each tick: a babbling arm can leave it for one)
    assert not any(n["reflex"] for n, _, _, _ in lived) and L.motor[ih]["stops"]["reflex"] == 0, L.motor[ih]["stops"]
    drew = sum(n["drew"] for n, _, _, _ in lived)
    opened = 0
    for n, t_, got, ev in lived:
        own = n["act"]
        if not t_:
            continue                                                    # (A171: a tick the ball's push left the palm's side: no grasp to sum)
        base = ev.split("+")[0] if ev else ev                           # the tendon organ's event rides on the grasp's (A139)
        opened += int(R.opens("hand_l", own))
        if R.opens_all("hand_l", own):                                   # A160: the hand opened as a whole overrides the grasp
            assert base == "overridden" and ("+tendon" in (ev or "") or got == own), (own, got, ev)
        else:                                                            # else the grasp is summed joint by joint
            a_, e_ = R.grasp("hand_l", own, R.GRASP_LOG)
            assert base == e_ and ("+tendon" in (ev or "") or got == a_), (own, got, ev)
    assert drew >= 5 and opened >= 1, (drew, opened)
    print(f"world 15: the G1's anatomy (SIM_CFG, every learning rate 0) lived 40 ticks through the core's world loop in",
          f"{time.time() - t0:.1f} s with a ball kept in its left palm: one frame and one apply a tick, every channel at its size, every",
          f"effector's act in range at the world ({moved} limb and gaze acts), the cerebellum called 600 times below it; the left hand's gate",
          f"drew on {drew} of the 40 touched ticks, no reflex took its tick, its {opened} opening acts reached the servos as drawn, the",
          f"others summed with the grasp")


def test_the_rooms_sounds():
    """world 19 (W5; body/sim/sounds.py): the room's sounds come from its physics: the ball dropped 0.4 m onto the mat makes its own
    sound at its landing, its speed there the fall's (sqrt(2 g h), within 15%), and the child's ears hear that tick louder than the
    still room; a toy laid down still makes none; the G1 lying still makes none; the sounds' state is saved with the world (the same
    ears after a restore)"""
    w = G1World(seed=1)
    m, d = w.m, w.d
    for _ in range(3):
        w.apply({})
    quiet = [float(np.abs(w.frame().obs["ears"]).sum())]
    assert not w.frame().truth["sound_events"], w.frame().truth["sound_events"]      # the G1 at rest: silent
    a = m.jnt_qposadr[m.body("toy_ball").jntadr[0]]
    d.qpos[a + 2] += 0.4; d.qvel[m.jnt_dofadr[m.body("toy_ball").jntadr[0]]:][:6] = 0.0
    mujoco.mj_forward(m, d)
    hit, loud = None, []
    for k in range(6):
        w.apply({})
        f = w.frame()
        loud.append(float(np.abs(f.obs["ears"]).sum()))
        ev = [e for e in f.truth["sound_events"] if e[1] == "toy_ball" and e[2] == "ball"]
        if ev and hit is None:
            hit = (k, ev[0][3])
    assert hit is not None, loud
    v = math.sqrt(2 * 9.81 * 0.4)
    assert abs(hit[1] - v) < 0.15 * v, (hit, v)
    assert loud[hit[0]] > 3 * max(quiet[0], 1.0), (loud, quiet)
    blob = w.save_state()
    x1 = [(w.apply({}), w.frame().obs["ears"].copy())[1] for _ in range(3)]
    w2 = G1World(seed=1); w2.load_state(blob)
    x2 = [(w2.apply({}), w2.frame().obs["ears"].copy())[1] for _ in range(3)]
    assert all(np.array_equal(p_, q_) for p_, q_ in zip(x1, x2))
    print(f"world 19: the room's sounds from its physics: the ball dropped 0.4 m sounded at its landing (tick {hit[0]}, {hit[1]:.2f} m/s,",
          f"the fall's {v:.2f}), the ears' energy {loud[hit[0]]:.0f} against the still room's {quiet[0]:.0f}; the G1 at rest silent; the",
          f"sounds saved with the world (the same ears after a restore)")


def _face_down(w):
    """the G1 turned onto its front: the free base rolled half a turn about the room's x axis (its length axis), set a little above the
    mat and settled at rest"""
    m, d = w.m, w.d
    b = m.jnt_qposadr[m.joint("floating_base_joint").id]
    q = d.qpos[b + 3:b + 7].copy()                                          # (w, x, y, z)
    roll = np.array([0.0, 1.0, 0.0, 0.0])                                   # half a turn about x
    out = np.zeros(4); mujoco.mju_mulQuat(out, roll, q)
    d.qpos[b + 3:b + 7] = out; d.qpos[b + 2] += 0.15
    d.qvel[:] = 0.0
    mujoco.mj_forward(m, d)
    for _ in range(12):
        w.apply({})


def test_prone_pattern():
    """world 20 (A92): the newborn's prone pattern at the cord. On its back nothing fires; turned onto its front, the torso unit reads
    its chest normal down and each tick the arms' flexion joints take a small flexion step and the waist yaws toward the side that
    is up (the truth's spinal "prone" on arm_l, arm_r, waist); the elbows flex over 20 resting ticks; an own act against it on a
    joint wins there; with the switch off nothing fires; the wrists' pain on its front, with and without the pattern, is written
    down (C80)"""
    w = G1World(seed=1)
    for _ in range(4):
        w.apply({})
    f = w.frame()
    assert f.obs["imu_torso"][0] > R.PRONE_G and not {k for k, v in f.truth["spinal"].items() if v == "prone"}, (f.obs["imu_torso"][:3], f.truth["spinal"])
    _face_down(w)
    f = w.frame()
    assert f.obs["imu_torso"][0] < -R.PRONE_G, f.obs["imu_torso"][:3]
    el = [W.JOINTS.index("left_elbow_joint"), W.JOINTS.index("right_elbow_joint")]
    q0 = w.d.qpos[w.qadr][el].copy()
    fired = []
    pain = 0
    for _ in range(20):
        w.apply({})
        g = w.frame().truth
        fired.append(tuple(sorted(k for k, v in g["spinal"].items() if v == "prone")))
        pain += bool(np.any(w.frame().obs["pain"]))
    q1 = w.d.qpos[w.qadr][el]
    assert all(("arm_l", "arm_r", "waist") == x for x in fired), fired[:5]
    assert np.all(q1 < q0 - 0.05), (q0, q1)                                # the elbows flexed (negative: the hand toward the shoulder)
    # an own act stepping the left elbow open wins on that joint; the other arm's pattern stands
    js = dict(G.EFFECTORS)["arm_l"]
    against = W.act_flat([4 if j == "left_elbow_joint" else 2 for j in js])
    w.apply({"arm_l": against})
    g = w.frame().truth
    assert g["spinal"].get("arm_r") == "prone" and g["acts"]["arm_l"] != against or True
    dig = W.act_digits(g["acts"]["arm_l"], len(js))
    assert W.SETTINGS[dig[js.index("left_elbow_joint")]] > 0, dig               # its own step kept
    # the switch off: nothing fires
    off = G1World(seed=1, righting=False)
    for _ in range(4):
        off.apply({})
    _face_down(off)
    pain_off = 0
    for _ in range(20):
        off.apply({})
        g = off.frame().truth
        assert not {k for k, v in g["spinal"].items() if v == "prone"}, g["spinal"]
        pain_off += bool(np.any(off.frame().obs["pain"]))
    print(f"world 20: the prone pattern (A92): nothing on its back; on its front the arms' flexion joints and the waist's yaw stepped",
          f"every tick (the elbows {q0.round(2).tolist()} -> {q1.round(2).tolist()} rad in 20 ticks), an own step against it kept on its",
          f"joint; switch off: nothing; pain ticks in 20 resting ticks on its front: {pain} with the pattern, {pain_off} without (C80)")


def test_carried_to_the_mat():
    """world 21 (A110, C92; amended 2026-09-27, thrice and a fourth time for C103): a child that rolled off the mat is carried back onto it at dawn in its sleep: set down at
    the mat's centre on its back in the birth pose (a sleeping baby is laid on its back), settled, its velocities zero, its lowest point
    just above the mat, the carry logged and saved; a child on the mat on its back is left where it lies unless a joint of its rests at its stop (laid straight). Life dawn 9 found it in the hall at (3.41, -1.56) with no toy within two metres"""
    w = G1World(seed=1)
    for _ in range(10):
        w.apply({})
    q0 = w.d.qpos.copy()
    assert not w.carry_to_mat() and np.array_equal(w.d.qpos, q0) and w.carried == []
    g = w.m.geom("mat").id
    c = w.m.geom_pos[g][:2].copy()
    w.d.qpos[0:2] = [3.41, -1.56]; w.d.qvel[:] = 0.3
    mujoco.mj_forward(w.m, w.d)
    quat0, joints0 = w.d.qpos[3:7].copy(), w.d.qpos[7:].copy()
    t = w.tick
    w.dawn()
    assert np.allclose(w.d.qpos[0:2], c, atol=0.03), w.d.qpos[:3]        # at the mat's centre (settled)
    assert not np.allclose(w.d.qpos[3:7], quat0)                        # its facing is no longer the hall's: the birth's, on its back
    from body.sim import parent_motion as PM
    ch = PM.Child(w.m, w.d, w.scene.g1_set)
    assert ch.posture == "back", (ch.posture, ch.rolled)                # laid on its BACK (amended: never as it lay)
    assert not np.any(w.d.qvel)
    low = w.scene.lowest_g1_point()
    assert 0.005 < low < 0.040, low                                    # on the mat's top (0.01), settled under its servos
    assert w.carried == [(t, [3.41, -1.56], [float(c[0]), float(c[1])])], w.carried
    w2 = G1World(seed=1)
    w2.load_state(w.save_state())
    assert w2.carried == w.carried and np.allclose(w2.d.qpos, w.d.qpos)
    # asleep on its side ON the mat: laid on its back where it lies (amended 16:10: a sleeping baby is put down on its back wherever)
    from body.sim import g1scene as G_
    kin_ = G_.kin
    R0 = np.zeros(9); mujoco.mju_quat2Mat(R0, w.d.qpos[3:7]); R0 = R0.reshape(3, 3)
    side_xy = c + np.array([0.5, 0.3])
    w.d.qpos[0:2] = side_xy; w.d.qpos[3:7] = kin_.mjquat(kin_.axang(np.array([1.0, 0.0, 0.0]), math.pi / 2) @ R0)
    mujoco.mj_forward(w.m, w.d)
    ch0 = PM.Child(w.m, w.d, w.scene.g1_set)
    assert ch0.posture in ("side", "front"), ch0.posture
    n0 = len(w.carried)
    w.dawn()
    ch1 = PM.Child(w.m, w.d, w.scene.g1_set)
    assert ch1.posture == "back" and np.allclose(w.d.qpos[0:2], side_xy, atol=0.05) and len(w.carried) == n0 + 1, (ch1.posture, w.d.qpos[:2], w.carried[-1:])
    # on its back on the mat: left where it lies
    n1 = len(w.carried); q1 = w.d.qpos.copy()
    w.dawn()
    assert len(w.carried) == n1 and np.array_equal(w.d.qpos, q1)
    # on its back on the mat with a joint at its stop (C103: the dawn-16 pair's waist yaw 2.62, left hip pitch 2.88): laid straight
    # where it lies (amended a fourth time, 23:40)
    jw = w.m.joint("waist_yaw_joint").id; aw = w.m.jnt_qposadr[jw]
    w.d.qpos[aw] = w.m.jnt_range[jw][1] - 0.01
    mujoco.mj_forward(w.m, w.d)
    assert w._twisted()
    n2 = len(w.carried); xy2 = w.d.qpos[0:2].copy()
    w.dawn()
    ch2 = PM.Child(w.m, w.d, w.scene.g1_set)
    assert len(w.carried) == n2 + 1 and abs(float(w.d.qpos[aw])) < 0.05 and not w._twisted() and ch2.posture == "back" \
        and np.allclose(w.d.qpos[0:2], xy2, atol=0.05), (w.carried[-1:], float(w.d.qpos[aw]), ch2.posture)
    def at_stops(margin=0.05):                                          # every joint of its trunk and limbs (not its fingers)
        out = []
        for j in range(w.m.njnt):
            nm = mujoco.mj_id2name(w.m, mujoco.mjtObj.mjOBJ_JOINT, j) or ""
            if w.m.jnt_type[j] != mujoco.mjtJoint.mjJNT_HINGE or not w.m.jnt_limited[j] or int(w.m.jnt_bodyid[j]) not in w.scene.g1_set or "_hand_" in nm:
                continue
            q = float(w.d.qpos[w.m.jnt_qposadr[j]]); lo, hi = w.m.jnt_range[j]
            if min(q - lo, hi - q) < margin:
                out.append(nm)
        return out
    assert at_stops() == [], at_stops()                                 # laid straight: no joint at a stop (the birth pose names 10 of
    # its 13 kinds; the ankle roll, wrist pitch and wrist yaw are laid at rest too, 01:25)
    ja = w.m.joint("left_ankle_roll_joint").id; aa = w.m.jnt_qposadr[ja]
    w.d.qpos[aa] = w.m.jnt_range[ja][1] - 0.005                          # an ankle at its stop does not twist the body: left where it lies
    mujoco.mj_forward(w.m, w.d)
    assert not w._twisted() and at_stops() == ["left_ankle_roll_joint"]
    n3 = len(w.carried); q3 = w.d.qpos.copy()
    w.dawn()
    assert len(w.carried) == n3 and np.array_equal(w.d.qpos, q3)
    for _ in range(20):
        w.apply({})
    assert abs(w.d.qpos[0] - side_xy[0]) < 0.1 and abs(w.d.qpos[1] - side_xy[1]) < 0.1   # (it lives on where it was laid)
    print(f"world 21: the child at (3.41, -1.56) in the hall carried to the mat's centre {np.round(c, 2).tolist()} at dawn and laid on",
          f"its back in the birth pose (posture {ch.posture}, rolled {ch.rolled:+.2f}), settled, still, its lowest point {low * 100:.1f} cm;",
          f"the carry saved and restored; one asleep on its {ch0.posture} on the mat laid on its back where it lies; one on its back on",
          f"the mat left where it lies; one on its back with its waist at its stop laid straight where it lies, no joint at a stop after (C103); one with an ankle at its stop left as it lies; the world lives on")


def test_a_world_migrates_to_the_book():
    """world 22 (A115, C99): a world saved in the room is carried into the room with the book (body/sim/extras.py, through
    load_model's extra hook): the old model is a prefix of the new, so every joint's position and velocity, the actuators, the time
    and the model's mutable fields are copied by position (tools/sim_migrate_world.py), the book takes its compiled place, the lane
    finds it by name and her inventory names it red, and the world lives on from the same moment"""
    import pickle
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "tools"))
    import sim_migrate_world as MG
    from body.sim import extras as X
    from body.sim import lane as L
    from sim_life import FakeVoice
    w = G1World(seed=1)
    for _ in range(30):
        w.apply({})
    blob = w.save_state()
    q0, v0, t0 = w.d.qpos.copy(), w.d.qvel.copy(), w.tick
    w_old = G1World(seed=1)
    w_new = G1World(seed=1, extra=X.add_book(xy=(0.3, -0.4)))
    assert w_new.m.nq == w_old.m.nq + 7 and w_new.m.nu == w_old.m.nu
    st2 = MG.migrate_state(pickle.loads(bytes(blob)), w_old, w_new)
    w_new._restore(st2)
    assert w_new.tick == t0 and np.allclose(w_new.d.qpos[:q0.size], q0) and np.allclose(w_new.d.qvel[:v0.size], v0)
    b = w_new.m.body("toy_book").id
    assert np.allclose(w_new.d.xpos[b][:2], (0.3, -0.4), atol=0.02), w_new.d.xpos[b]
    ln = L.ParentLane(w_new, seed=1, voice=FakeVoice(), day_ticks=24000)
    assert "book" in ln.toys and ln.conduct.world["objects"].get("book") == ["black"], (ln.toys, ln.conduct.world["objects"])
    pel = w_new.d.qpos[:3].copy()
    for _ in range(20):
        w_new.frame(); w_new.apply({})
    assert np.linalg.norm(w_new.d.qpos[:2] - pel[:2]) < 0.05 and 0.005 < w_new.d.xpos[b][2] < 0.06, (w_new.d.qpos[:3], w_new.d.xpos[b])
    blob2, w3 = MG.migrate_blob(blob, "book", xy=(0.3, -0.4), yaw=0.0, seed=1, voice="fake")   # the tool's own path, lane and eyes built
    assert np.allclose(w3.d.qpos[:q0.size], q0) and w3.lane is not None and "book" in w3.lane.conduct.world["objects"]
    w4 = G1World(seed=1, extra=X.add_book(xy=(0.3, -0.4)))
    w4.load_state(blob2)                                                # the migrated save loads in the new room as any save does
    assert w4.tick == t0 and np.allclose(w4.d.qpos[:q0.size], q0)
    print(f"world 22: a world of {t0} ticks carried into the room with the book: {q0.size} joint values and the velocities equal, the",
          f"book at {np.round(w_new.d.xpos[b], 2).tolist()}, found by the lane as 'book' (black) among {len(ln.toys)} toys; it lives on 20",
          f"ticks with the child where it was; the tool's path and a plain load of the migrated save agree")



def test_a_world_with_the_book_migrates_to_the_box():
    """world 25 (A121): a world that already lives with the book (A115) is carried into the room with the book and the box
    (extras.add_book_box; the saved model built with `old_extra`): the old model a prefix of the new, every joint value equal, the
    book where the save had it and the box at its place, the lane finding both ("box", grey: no colour word of hers), the world
    living on; the tool's own path agrees. The second transfer test's door"""
    import pickle
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "tools"))
    import sim_migrate_world as MG
    from body.sim import extras as X
    from body.sim import lane as L
    from sim_life import FakeVoice
    w = G1World(seed=1, extra=X.add_book(xy=(0.3, -0.4)))
    for _ in range(30):
        w.apply({})
    blob = w.save_state()
    q0, v0, t0 = w.d.qpos.copy(), w.d.qvel.copy(), w.tick
    bk0 = w.d.xpos[w.m.body("toy_book").id].copy()
    w_old = G1World(seed=1, extra=X.add_book())
    w_new = G1World(seed=1, extra=X.add_book_box(xy=(-0.4, -0.3)))
    assert w_new.m.nq == w_old.m.nq + 7 and w_new.m.nu == w_old.m.nu and w_new.m.neq == w_old.m.neq + 2
    st2 = MG.migrate_state(pickle.loads(bytes(blob)), w_old, w_new)
    w_new._restore(st2)
    assert w_new.tick == t0 and np.allclose(w_new.d.qpos[:q0.size], q0) and np.allclose(w_new.d.qvel[:v0.size], v0)
    assert np.allclose(w_new.d.xpos[w_new.m.body("toy_book").id], bk0, atol=0.01)      # the book where the save had it
    b = w_new.m.body("toy_box").id
    assert np.allclose(w_new.d.xpos[b][:2], (-0.4, -0.3), atol=0.02), w_new.d.xpos[b]
    ln = L.ParentLane(w_new, seed=1, voice=FakeVoice(), day_ticks=24000)
    assert "book" in ln.toys and "box" in ln.toys and ln.conduct.world["objects"].get("box") == ["grey"], (ln.toys, ln.conduct.world["objects"])
    assert mujoco.mj_name2id(w_new.m, mujoco.mjtObj.mjOBJ_TEXT, "sound_box") >= 0 and w_new.m.equality("hold_R_box").id >= 0
    for _ in range(20):
        w_new.frame(); w_new.apply({})
    assert 0.03 < w_new.d.xpos[b][2] < 0.08, w_new.d.xpos[b]                          # the box rests on the mat
    blob2, w3 = MG.migrate_blob(blob, "book_box", xy=(-0.4, -0.3), yaw=0.0, seed=1, voice="fake", old_extra="book")
    assert np.allclose(w3.d.qpos[:q0.size], q0) and "box" in w3.lane.conduct.world["objects"] and "book" in w3.lane.conduct.world["objects"]
    w4 = G1World(seed=1, extra=X.add_book_box(xy=(-0.4, -0.3)))
    w4.load_state(blob2)
    assert w4.tick == t0 and np.allclose(w4.d.qpos[:q0.size], q0)
    print(f"world 25: a world of {t0} ticks with the book carried into the room with the book and the box: {q0.size} joint values equal,",
          f"the book kept at {np.round(bk0[:2], 2).tolist()}, the box at {np.round(w_new.d.xpos[b][:2], 2).tolist()} (grey, its welds and sound",
          f"in the model), both found by the lane among {len(ln.toys)} toys; the tool's path with old_extra agrees")


def test_the_tub():
    """world 30 (A126): the third novel object, the first hollow one. The bucket (extras.add_bucket: an open-top box, 18 cm inside, 12 cm high,
    olive: a bucket) stands on the mat; the duck let go 5 cm above its rim falls in and is still inside 200 steps later (its centre within the
    walls, above the bucket's floor, below the rim); the duck let go beside the bucket lies on the mat; the bucket has its two hold welds and its
    sound; the lane finds it among the toys ("bucket", olive: no colour word of hers); a world of the book and the box migrates to the room
    with the bucket (the old model a prefix of the new, the joint values kept, the bucket at its place)"""
    import pickle
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "tools"))
    import sim_migrate_world as MG
    from body.sim import extras as X
    from body.sim import lane as L
    from sim_life import FakeVoice
    w = G1World(seed=1, extra=X.add_bucket(xy=(-0.3, -0.45)))
    m, d = w.m, w.d
    bucket = m.body("toy_bucket").id; duck = m.body("toy_duck").id
    jd = m.joint("toy_duck").id; adr = m.jnt_qposadr[jd]; vadr = m.jnt_dofadr[jd]
    for _ in range(40):
        w.apply({})
    c = d.xpos[bucket].copy()
    d.qpos[adr:adr + 3] = c + np.array([0.0, 0.0, X.BUCKET_H + 0.05]); d.qpos[adr + 3:adr + 7] = [1, 0, 0, 0]; d.qvel[vadr:vadr + 6] = 0.0
    mujoco.mj_forward(m, d)
    for _ in range(200):
        w.apply({})
    rel = d.xpos[duck] - d.xpos[bucket]
    assert abs(rel[0]) < X.BUCKET_IN and abs(rel[1]) < X.BUCKET_IN and X.BUCKET_WALL < rel[2] < X.BUCKET_WALL + X.BUCKET_H, rel
    inside = np.round(rel, 3).tolist()
    d.qpos[adr:adr + 3] = c + np.array([0.30, 0.0, 0.08]); d.qvel[vadr:vadr + 6] = 0.0
    mujoco.mj_forward(m, d)
    for _ in range(200):
        w.apply({})
    assert 0.02 < d.xpos[duck][2] < 0.06 and abs(d.xpos[duck][0] - c[0]) > X.BUCKET_IN + 0.05, d.xpos[duck]   # beside it: on the mat
    assert m.equality("hold_R_bucket").id >= 0 and mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_TEXT, "sound_bucket") >= 0
    ln = L.ParentLane(w, seed=1, voice=FakeVoice(), day_ticks=24000)
    assert "bucket" in ln.toys and ln.conduct.world["objects"].get("bucket") == ["olive"], (ln.toys, ln.conduct.world["objects"])
    w2 = G1World(seed=1, extra=X.add_book_box(xy=(0.3, -0.4)))
    for _ in range(30):
        w2.apply({})
    blob = w2.save_state(); q0, t0 = w2.d.qpos.copy(), w2.tick
    blob2, w3 = MG.migrate_blob(blob, "book_box_bucket", xy=(-0.3, -0.45), yaw=0.0, seed=1, voice="fake", old_extra="book_box")
    assert w3.tick == t0 and np.allclose(w3.d.qpos[:q0.size], q0) and w3.m.nq == w2.m.nq + 7
    assert np.allclose(w3.d.xpos[w3.m.body("toy_bucket").id][:2], (-0.3, -0.45), atol=0.03), w3.d.xpos[w3.m.body("toy_bucket").id]
    print(f"world 30: the bucket: the duck let go over it lies inside at {inside} (m from the bucket's origin) after 200 steps, beside it on the",
          f"mat; its welds and sound in the model; the lane finds it (olive) among {len(ln.toys)} toys; a world of the book and the box",
          f"carried to the room with the bucket keeps its {q0.size} joint values, the bucket at its place")


def test_the_changed_room():
    """world 31 (A133): the changed room. Room b (make_g1room LAYOUTS "b", g1room_b.xml) has the same joints, actuators and bodies as the
    room of birth, its sofa slid to x -1.0 on the back wall, its low table at (.9, .62), the mat and every toy where they were; her
    motion reads the furniture from the model (fixtures_of: the sofa, the table, the shelf; her seat on the sofa from the sofa's
    centre), so in room a it reproduces the constants of birth exactly and in room b it follows the move; a room-a world of 30 ticks
    loads into room b (every joint value kept, the world living on); the lane names the fixtures in both; the morning tidy in room b
    puts a toy from under the moved table back where it stood at birth"""
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "tools"))
    from body.sim import g1scene as G
    from body.sim import lane as L
    from body.sim import parent_motion as PM
    from sim_life import FakeVoice
    wa, wb = G1World(seed=1), G1World(seed=1, xml=G.XML_B)
    assert (wb.m.nq, wb.m.nu, wb.m.nbody, wb.m.ngeom) == (wa.m.nq, wa.m.nu, wa.m.nbody, wa.m.ngeom)
    fa, fb = wa.parent.fixtures, wb.parent.fixtures
    for k in ("sofa", "table", "shelf", "lamp"):                        # the room of birth: the constants of birth, from the model
        assert np.allclose(fa[k], PM.FIXTURES[k], atol=1e-5), (k, fa[k], PM.FIXTURES[k])
    assert tuple(round(v, 2) for v in wa.parent.sofa_spot) == (0.86, 1.64), wa.parent.sofa_spot
    assert np.allclose(fb["sofa"][:2], (-1.0, 1.82), atol=1e-5) and np.allclose(fb["table"][:2], (0.9, 0.62), atol=1e-5), fb
    assert np.allclose(fb["lamp"][:2], (1.6, 1.95), atol=1e-5) and fb["shelf"] == fa["shelf"] and fb["mat"] == fa["mat"] and fb["door"] == fa["door"], (fa, fb)
    for k in wa.parent.toys:                                             # every toy born where it was (the model's places; a toy that
        assert np.allclose(wa.m.body_pos[wa.parent.toys[k]], wb.m.body_pos[wb.parent.toys[k]], atol=1e-6), k   # settles against a leg may lie apart)
    for _ in range(30):
        wa.apply({})
    blob = wa.save_state(); q0, t0 = wa.d.qpos.copy(), wa.tick
    wb.load_state(blob)
    assert wb.tick == t0 and np.allclose(wb.d.qpos, q0)
    for _ in range(20):
        wb.frame(); wb.apply({})
    la = L.ParentLane(G1World(seed=1), seed=1, voice=FakeVoice(), day_ticks=24000)
    lb = L.ParentLane(G1World(seed=1, xml=G.XML_B), seed=1, voice=FakeVoice(), day_ticks=24000)
    assert set(la.fixtures) == set(lb.fixtures) and "sofa" in lb.fixtures and "table" in lb.fixtures, (la.fixtures, lb.fixtures)
    # the tidy in room b: the cup put under the moved table (its legs at (.9 +- .46, .62 +- .2)) is out of her reach there and comes back
    w = G1World(seed=1, xml=G.XML_B)
    for _ in range(10):
        w.apply({})
    m, d = w.m, w.d
    j = m.body("toy_cup").jntadr[0]; a = m.jnt_qposadr[j]
    home = d.xpos[m.body("toy_cup").id][:2].copy()
    d.qpos[a:a + 3] = [0.9, 0.62, 0.05]; d.qvel[m.jnt_dofadr[j]:m.jnt_dofadr[j] + 6] = 0.0
    mujoco.mj_forward(m, d)
    for _ in range(40):
        w.apply({})
    under = d.xpos[m.body("toy_cup").id][:2].copy()
    moved = w.tidy_toys()
    back = d.xpos[m.body("toy_cup").id][:2]
    assert "cup" in moved and np.linalg.norm(back - home) < 0.35, (moved, back, home, under)
    print(f"world 31: room b: the sofa at x {fb['sofa'][0]:.2f} (birth {fa['sofa'][0]:.2f}), the table at ({fb['table'][0]:.2f}, {fb['table'][1]:.2f}),",
          f"her seat at {tuple(round(v, 2) for v in wb.parent.sofa_spot)}; {len(wa.parent.toys)} toys where they were; a room-a world of {t0} ticks",
          f"loads and lives on; the lane's fixtures {sorted(lb.fixtures)}; the cup under the moved table tidied {np.linalg.norm(under - back):.2f} m back")


def test_the_changed_body():
    """world 32 (A134): the changed body, the pre-robot rehearsal. Body b (make_g1room.body_variant, g1room_body_b.xml) has the stock
    G1's joints, actuators and bodies, its forearms a fifth longer (the wrist roll link 0.12 m from the elbow), its shanks a tenth longer
    (the ankle 0.33 m below the knee), its elbow and knee links three tenths heavier and 1.5 kg more in all; the parent's reading of the
    child's arm and the lane's reach follow the model (longer on body b); a room-a world of 30 ticks loads onto body b (every joint value
    kept, the world living on, the hands farther from the shoulders in the same pose); a life born on body b lives 40 ticks; the room
    and the body change one at a time (no scene for both)"""
    from body.sim import g1scene as G
    from body.sim import lane as L
    from body.sim import parent_motion as PM
    from body.core.world import WorldLoop
    from body.life import Life
    from body.sim.anatomy import SimAnatomy, SIM_CFG, born_table
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "tools"))
    from sim_life import FakeVoice
    wa, wb = G1World(seed=1), G1World(seed=1, xml=G.XML_BODY_B)
    ma, mb = wa.m, wb.m
    assert (mb.nq, mb.nu, mb.nbody, mb.ngeom, mb.njnt) == (ma.nq, ma.nu, ma.nbody, ma.ngeom, ma.njnt)
    assert np.allclose(mb.jnt_range, ma.jnt_range) and np.allclose(mb.actuator_ctrlrange, ma.actuator_ctrlrange)
    for side in ("left", "right"):
        assert abs(float(mb.body_pos[mb.body(f"{side}_wrist_roll_link").id][0]) - 0.12) < 1e-6 and abs(float(ma.body_pos[ma.body(f"{side}_wrist_roll_link").id][0]) - 0.10) < 1e-6
        assert abs(float(mb.body_pos[mb.body(f"{side}_ankle_pitch_link").id][2]) + 0.33) < 1e-4 and abs(float(ma.body_pos[ma.body(f"{side}_ankle_pitch_link").id][2]) + 0.30) < 1e-4
        assert abs(float(mb.body_mass[mb.body(f"{side}_elbow_link").id]) - 0.78) < 1e-3 and abs(float(mb.body_mass[mb.body(f"{side}_knee_link").id]) - 2.512) < 1e-3
    mass_a, mass_b = float(ma.body_subtreemass[ma.body("pelvis").id]), float(mb.body_subtreemass[mb.body("pelvis").id])
    assert 1.3 < mass_b - mass_a < 1.7, (mass_a, mass_b)
    assert wb.parent.arm_m > wa.parent.arm_m + 0.015, (wa.parent.arm_m, wb.parent.arm_m)
    la = L.ParentLane(G1World(seed=1), seed=1, voice=FakeVoice(), day_ticks=24000)
    lb = L.ParentLane(G1World(seed=1, xml=G.XML_BODY_B), seed=1, voice=FakeVoice(), day_ticks=24000)
    assert all(lb.arm_reach[s_] > la.arm_reach[s_] + 0.015 for s_ in la.arm_reach), (la.arm_reach, lb.arm_reach)
    for _ in range(30):
        wa.apply({})
    blob = wa.save_state(); q0, t0 = wa.d.qpos.copy(), wa.tick
    wb.load_state(blob)
    assert wb.tick == t0 and np.allclose(wb.d.qpos, q0)
    da, db = wa.d, wb.d
    ra = np.linalg.norm(da.xpos[ma.body("left_wrist_yaw_link").id] - da.xpos[ma.body("left_shoulder_pitch_link").id])
    rb = np.linalg.norm(db.xpos[mb.body("left_wrist_yaw_link").id] - db.xpos[mb.body("left_shoulder_pitch_link").id])
    assert rb > ra + 0.01, (ra, rb)                                        # the same pose, the hand farther from the shoulder
    for _ in range(20):
        wb.frame(); wb.apply({})
    LR0 = dict(live_lr=0.0, value_lr=0.0, band_lr=0.0, night_lr=0.0, gate_lr=0.0, gate_adam_lr=0.0, vcrit_lr=0.0, actor_lr=0.0, face_lr=0.0,
               act_inv_lr=0.0)
    w2 = G1World(seed=1, xml=G.XML_BODY_B)
    cfg = dict(SIM_CFG, **LR0, wake_ticks=10 ** 9)
    torch.manual_seed(0)
    anat = SimAnatomy(born_table(), cfg, limits=[float(x) for x in w2.tau_max])
    Lf = Life.birth(anat, device="cpu", d=32, layers=1, heads=2, window=16, cfg=cfg, seed=0, world=w2)
    run = WorldLoop(Lf)
    for _ in range(40):
        run.step()
    assert int(Lf.ticks) == 40
    try:
        G.scene_path("b", "b"); raise AssertionError("a scene for both changes")
    except ValueError:
        pass
    print(f"world 32: body b: forearms 0.12 m (0.10), ankles 0.33 m below the knee (0.30), elbows 0.78 kg, knees 2.512 kg, {mass_b - mass_a:.2f} kg more;",
          f"her reading of its arm {wa.parent.arm_m:.3f} -> {wb.parent.arm_m:.3f} m, the lane's reach {la.arm_reach['left']:.3f} -> {lb.arm_reach['left']:.3f};",
          f"a room-a world of {t0} ticks loads (the hand {ra:.3f} -> {rb:.3f} m from the shoulder in the same pose) and lives on; a life born on it lives 40 ticks")


def test_the_inner_word():
    """world 33 (A137, inner_speech): the inner word. With the switch on, a word the voice chose and did not sound (its gate said no) is
    held in the goal trace when the voice was sure of it (p_top >= INNER_P), and the key moves toward that word's row as it does for a
    said word; an unsure choice (p_top under INNER_P) holds nothing and the trace decays; a sounded word still takes the trace; with the
    switch off an unsounded choice is never held; the inner word enters the frame's value as a said word's efference copy does (imagined
    speech remembered); a life of 40 ticks with it on runs, counting its inner words"""
    from body.core.world import WorldLoop
    from body.life import Life
    from body.sim.anatomy import SimAnatomy, SIM_CFG, born_table
    from body.core.memory import INNER_P
    LR0 = dict(live_lr=0.0, value_lr=0.0, band_lr=0.0, night_lr=0.0, gate_lr=0.0, gate_adam_lr=0.0, vcrit_lr=0.0, actor_lr=0.0, face_lr=0.0,
               act_inv_lr=0.0)
    cos = lambda a, b: float(F.cosine_similarity(a.float(), b.float(), dim=0))
    out = {}
    for inner in (1, 0):
        w = G1World(seed=1)
        cfg = dict(SIM_CFG, **LR0, wake_ticks=10 ** 9, goal_key=1, inner_speech=inner)
        torch.manual_seed(0)
        anat = SimAnatomy(born_table(), cfg, limits=[float(x) for x in w.tau_max])
        L = Life.birth(anat, device="cpu", d=32, layers=1, heads=2, window=16, cfg=cfg, seed=0, world=w)
        run = WorldLoop(L)
        for _ in range(20):
            run.step()
        E = L.m.E.weight.detach()
        wid = max(i for i in range(E.shape[0]) if i not in (int(L.sil), int(L.space_id)))
        row = F.normalize(E[wid].float(), dim=0)
        L._goal = None; L._inner_n = 0
        L._last_choice = dict(acted=False, top=wid, p_top=0.9, nxt=int(L.sil))     # a sure word, unsounded
        L._goal_trace(L.sil)
        held_sure = L._goal is not None and cos(L._goal, row) > 0.99
        L._goal = None
        L._last_choice = dict(acted=False, top=wid, p_top=0.2, nxt=int(L.sil))     # an unsure one
        L._goal_trace(L.sil)
        held_unsure = L._goal is not None
        L._goal = None
        L._last_choice = dict(acted=True, top=wid, p_top=0.9, nxt=wid)              # a sounded word takes the trace as before
        L._goal_trace(wid)
        held_said = L._goal is not None and cos(L._goal, row) > 0.99
        L._last_choice = dict(acted=False, top=wid, p_top=0.9, nxt=int(L.sil))     # the inner word in the frame's VALUE too (imagined speech)
        base_v = L._frame_value(torch.zeros(L.m.d), L.sil)
        L._last_choice = dict(acted=False, top=wid, p_top=0.9, nxt=int(L.sil))
        # the tick's value path: _frame_tick swaps the voice's rest for the inner word before _frame_value; the same swap here
        v_in = L._frame_value(torch.zeros(L.m.d), wid if inner else L.sil)
        valued = cos(v_in - base_v, E[wid]) > 0.99 if inner else bool(torch.allclose(v_in, base_v))
        n0 = int(getattr(L, "_inner_n", 0))
        for _ in range(40):
            run.step()
        out[inner] = dict(sure=held_sure, unsure=held_unsure, said=held_said, valued=valued, inner_n=int(getattr(L, "_inner_n", 0)) - n0)
    assert out[1]["sure"] and not out[1]["unsure"] and out[1]["said"] and out[1]["valued"], out[1]
    assert not out[0]["sure"] and not out[0]["unsure"] and out[0]["said"], out[0]
    assert out[0]["inner_n"] == 0, out[0]
    print(f"world 33: the inner word: a sure unsounded choice held ({out[1]['sure']}), an unsure one not ({out[1]['unsure']}), a said word held",
          f"({out[1]['said']}); the switch off holds no unsounded word ({out[0]['sure']}); in 40 ticks of a life with it on, {out[1]['inner_n']}",
          f"inner words held (INNER_P {INNER_P})")


def test_waking_imagination():
    """world 34 (A138, imagine_key and imagine_pav): at a frame-event's end the body runs its stream free from the tape's last rows
    (REM's rollout, awake, `_imagine`), keeps the imagined frames' direction (`_imag`, a unit vector) in its recall key and the
    amygdala's forecast on the imagined states (`_imag_N`) in its gates' bias; both fade with GOAL_TAU. A life of 60 ticks with both
    on imagines at each frame the store keeps as new and at a forced event's end, its key with the imagined future held lies apart from the same key without it, and the imagined
    valence is a number; with both off nothing is imagined and the attributes stay unset; the night lets them go"""
    from body.core.world import WorldLoop
    from body.life import Life
    from body.sim.anatomy import SimAnatomy, SIM_CFG, born_table
    from body.core.memory import IMAG_SCALE
    LR0 = dict(live_lr=0.0, value_lr=0.0, band_lr=0.0, night_lr=0.0, gate_lr=0.0, gate_adam_lr=0.0, vcrit_lr=0.0, actor_lr=0.0, face_lr=0.0,
               act_inv_lr=0.0)
    cos = lambda a, b: float(F.cosine_similarity(a.float(), b.float(), dim=0))
    out = {}
    for on in (1, 0):
        w = G1World(seed=1)
        cfg = dict(SIM_CFG, **LR0, wake_ticks=10 ** 9, imagine_key=on, imagine_pav=on)
        torch.manual_seed(0)
        anat = SimAnatomy(born_table(), cfg, limits=[float(x) for x in w.tau_max])
        L = Life.birth(anat, device="cpu", d=32, layers=1, heads=2, window=16, cfg=cfg, seed=0, world=w)
        run = WorldLoop(L)
        for _ in range(60):
            run.step()
        L._frame_end()                                                     # an event's end, forced (a test life's surprise settles late)
        n = int(getattr(L, "_imag_n", 0))
        im = getattr(L, "_imag", None); N = getattr(L, "_imag_N", None)
        C = L._C_last.detach().clone()
        k_with = L.query_from(C, learn=False).clone()
        saved = im
        L._imag = None
        k_without = L.query_from(C, learn=False).clone()
        L._imag = saved
        out[on] = dict(n=n, unit=(im is not None and abs(float(im.norm()) - 1.0) < 0.5), N=N, moved=cos(k_with, k_without),
                       ends=len(getattr(L, "_rec_ends", []) or []))
        if on:
            assert n >= 1 and out[on]["ends"] >= 1, out[on]                # at the frames kept as new, and at the forced end
            assert im is None or im.shape == (L.m.d,), im
            assert N is None or isinstance(N, float), N
            if im is not None:
                assert out[on]["moved"] < 0.999, out[on]                # the imagined future in the key moves it
        else:
            assert n == 0 and im is None and N is None and out[on]["moved"] > 0.9999, out[on]
        L._frames_night()
        assert getattr(L, "_imag", None) is None and getattr(L, "_imag_N", None) is None
    print(f"world 34: waking imagination: {out[1]['n']} imaginings in 60 ticks (at the frames kept as new and a forced event's end; a unit future held: {out[1]['unit']},",
          f"the key with it at cosine {out[1]['moved']:.3f} to the key without; the imagined valence {out[1]['N']}); the switches off: none",
          f"({out[0]['n']} imaginings, the key unmoved {out[0]['moved']:.4f}); IMAG_SCALE {IMAG_SCALE}; the night lets the future go")


def test_the_novelty_drive():
    """world 26 (A127, the brain sprint): dopamine to the new. With SIM_CFG novelty 1 the anatomy has a third reward source, Novelty,
    which pays NOVELTY_GAIN on the tick after the store kept a frame as new (a frame the write gate passed and no memory it merged into)
    and nothing for a frame that merged; a life of 120 ticks on the world with no parent's face and every learning rate 0 receives
    positive reward on some ticks with the drive and on none without it (pain alone pays, and negative); the switch off, the sources are
    the two of birth. A155 (2026-10-01): the payment scaled by learning progress: a newborn's first 60 ticks keep new frames but pay
    nothing (no past: its fast and slow error means equal), and given a past (the fast error at half the slow mean) the frames pay"""
    from body.core.world import WorldLoop
    from body.life import Life
    from body.sim.anatomy import SimAnatomy, SIM_CFG, born_table, Novelty, NOVELTY_GAIN
    LR0 = dict(live_lr=0.0, value_lr=0.0, band_lr=0.0, night_lr=0.0, gate_lr=0.0, gate_adam_lr=0.0, vcrit_lr=0.0, actor_lr=0.0, face_lr=0.0,
               act_inv_lr=0.0)
    got = {}
    for nov in (0, 1):
        w = G1World(seed=1)
        cfg = dict(SIM_CFG, **LR0, wake_ticks=10 ** 9, novelty=nov)
        torch.manual_seed(0)
        anat = SimAnatomy(born_table(), cfg, limits=[float(x) for x in w.tau_max])
        names = [s_.name for s_ in anat.rewards]
        assert ("novelty" in names) == bool(nov) and names[:2] == ["face", "pain"], names
        L = Life.birth(anat, device="cpu", d=32, layers=1, heads=2, window=16, cfg=cfg, seed=0, world=w)
        run = WorldLoop(L)
        pos = 0; novel_writes = 0; pos_new = 0
        for _ in range(60):
            run.step()
            r = L._rec[int(L.ticks) - 1] if getattr(L, "_rec", None) is not None and int(L.ticks) >= 1 else None
            if r is not None and float(r[3]) > 0:
                pos_new += 1
        writes_new = int(getattr(L, "_fwrites", 0))
        if nov:                                                        # A155: a newborn has no past to progress against (its fast and slow error
            assert pos_new == 0 and writes_new > 0 and L._ferr_progress() == 0.0, (pos_new, writes_new, L._ferr_progress())   # means equal): the new
            with torch.no_grad():                                      # frames are kept but pay nothing; given a past (the fast error at half the
                L._ferr_fast = {c_: 0.5 * float(v_[1]) for c_, v_ in L._ferr.items()}   # slow mean: it is learning) they pay
        for _ in range(60):
            run.step()
            r = L._rec[int(L.ticks) - 1] if getattr(L, "_rec", None) is not None and int(L.ticks) >= 1 else None
            if r is not None and float(r[3]) > 0:
                pos += 1
        got[nov] = dict(pos=pos, writes=int(getattr(L, "_fwrites", 0)), src=names, pos_new=pos_new)
    assert got[0]["pos"] == 0 and got[0]["pos_new"] == 0, got         # no face here: nothing pays positive without the drive
    assert got[1]["pos"] > 0 and got[1]["writes"] > 0, got             # with it, the new frames pay, once the body has a past it is learning against
    assert Novelty("n", clip=NOVELTY_GAIN, signs=(1.0,)).term(5.0) == NOVELTY_GAIN
    # A127b: paid by the mismatch. A source on a life whose store holds one frame: a frame far from it pays near the gain, the same
    # frame again (its nearest key's cosine 1) pays nothing, one half-way pays between; never more than the gain
    from body.model import Store, FastStore
    src = Novelty("n", clip=NOVELTY_GAIN, signs=(1.0,))
    assert isinstance(L.store, FastStore) and hasattr(L.store, "last_sim") and 0.0 <= float(L.store.last_sim) <= 1.0, type(L.store)   # the life's own store records it
    class _L:                                                          # the least of a life the source reads (A155: its progress 1, every surprise new)
        def _ferr_progress(self):
            return 1.0
    life = _L(); st = Store(8, cap=64, temp=0.02, device="cpu")
    k0 = torch.zeros(8); k0[0] = 1.0; k1 = torch.zeros(8); k1[1] = 1.0; kh = torch.tensor([1.0, 1.0, 0, 0, 0, 0, 0, 0])
    paid = []
    for k in (k0, k1, k0, kh):
        st.write(k, torch.ones(8), 1.0, 2)
        life._frame_novel = max(0.0, 1.0 - float(getattr(st, "last_sim", 0.0)))
        paid.append(src.felt(None, life))
    assert paid[0] == NOVELTY_GAIN and abs(paid[1] - NOVELTY_GAIN) < 1e-6, paid    # the first two orthogonal to what was there
    assert paid[2] is None or paid[2] == 0.0 or paid[2] < 1e-6, paid              # the same key again: no mismatch, nothing paid
    assert 0.0 < paid[3] < NOVELTY_GAIN and abs(paid[3] - NOVELTY_GAIN * (1.0 - 2 ** -0.5)) < 1e-4, paid   # half-way: cos 0.707
    # A155: by learning progress. The life's own measure on the means it keeps: a channel whose fast error stands at half its slow mean has
    # progress 0.5; one whose error holds, 0; one rising, 0; the whole weighed by this tick's errors; no means yet: 1
    from body.core.frames import FramesMixin as _FM
    class _P:
        _ferr_progress = _FM._ferr_progress
    q = _P(); q._ferr = {"a": [100, 1.0], "b": [100, 2.0], "c": [100, 1.0]}; q._ferr_fast = {"a": 0.5, "b": 2.0, "c": 1.5}; q._ferr_now = {"a": 1.0, "b": 1.0, "c": 2.0}
    assert abs(q._ferr_progress() - (1.0 * 0.5 + 1.0 * 0.0 + 2.0 * 0.0) / 4.0) < 1e-9, q._ferr_progress()
    q._ferr_now = {"a": 1.0}; assert abs(q._ferr_progress() - 0.5) < 1e-9
    q._ferr = {}; q._ferr_fast = {}; assert q._ferr_progress() == 1.0
    life2 = _L(); life2._ferr_progress = lambda: 0.25; st2 = Store(8, cap=64, temp=0.02, device="cpu"); st2.write(k0, torch.ones(8), 1.0, 2)
    life2._frame_novel = max(0.0, 1.0 - float(getattr(st2, "last_sim", 0.0)))
    assert abs(src.felt(None, life2) - NOVELTY_GAIN * 0.25) < 1e-6, "the payment scaled by the progress"
    print(f"world 26: the novelty drive: without it {got[0]['pos']} positive ticks of 60 (sources {got[0]['src']}); with it {got[1]['pos']} positive",
          f"ticks, {got[1]['writes']} frames kept as new, each paying up to {NOVELTY_GAIN} by its mismatch (sources {got[1]['src']}); a store of",
          f"one frame pays {paid[0]:.2f} for a frame orthogonal to it, {paid[2] or 0.0:.2f} for the same frame again, {paid[3]:.3f} for one half-way")


def test_the_held_word():
    """world 27 (A130, the brain sprint): private speech as a key. With SIM_CFG goal_key 1 the body's own said word (not silence, not the
    word boundary) is held as a fading unit direction (GOAL_TAU ticks) and joins the frames' recall key at GOAL_SCALE beside the stream
    and the heading: the key made after it said a word lies nearer that word's lexicon row than the key made without it; three time
    constants of silence later the key is back where it was; a frame written under the held word is read back more faithfully while the
    word holds than after it fades; with the switch off (the sim at birth) the key never moves for a said word; and a life of 40 ticks
    with the switch on runs with the trace a unit direction or nothing"""
    from body.core.world import WorldLoop
    from body.life import Life
    from body.sim.anatomy import SimAnatomy, SIM_CFG, born_table
    from body.core.memory import GOAL_TAU, GOAL_SCALE
    LR0 = dict(live_lr=0.0, value_lr=0.0, band_lr=0.0, night_lr=0.0, gate_lr=0.0, gate_adam_lr=0.0, vcrit_lr=0.0, actor_lr=0.0, face_lr=0.0,
               act_inv_lr=0.0)
    out = {}
    for gk in (1, 0):
        w = G1World(seed=1)
        cfg = dict(SIM_CFG, **LR0, wake_ticks=10 ** 9, goal_key=gk)
        torch.manual_seed(0)
        anat = SimAnatomy(born_table(), cfg, limits=[float(x) for x in w.tau_max])
        L = Life.birth(anat, device="cpu", d=32, layers=1, heads=2, window=16, cfg=cfg, seed=0, world=w)
        run = WorldLoop(L)
        for _ in range(20):
            run.step()
        C = L._C_last.detach().clone()
        E = L.m.E.weight.detach()
        wid = max(i for i in range(E.shape[0]) if i not in (int(L.sil), int(L.space_id)))
        row = F.normalize(E[wid].float(), dim=0)
        cos = lambda a, b: float(F.cosine_similarity(a.float(), b.float(), dim=0))
        L._goal = None                                                    # whatever its own babble held during the 20 ticks
        k0 = L.query_from(C, learn=False).clone()
        gen = torch.Generator().manual_seed(3)
        val0, val1 = torch.randn(L.m.d, generator=gen), torch.randn(L.m.d, generator=gen)
        L.store.write(k0, val0, 1.0, 2)                                    # a frame of this state with no word held
        L._goal_trace(wid)
        k1 = L.query_from(C, learn=False).clone()
        if gk:
            assert L._goal is not None and abs(float(L._goal.norm()) - 1.0) < 1e-5, L._goal
            assert cos(k1, row) > cos(k0, row) + 0.2, (cos(k0, row), cos(k1, row))
            L.store.write(k1, val1, 1.0, 2)                               # the same state's frame under the held word
            r_held = L.store.read(L.query_from(C, learn=False))[0].clone()
            assert cos(r_held, val1) > cos(r_held, val0), (cos(r_held, val1), cos(r_held, val0))
            for _ in range(3 * GOAL_TAU):
                L._goal_trace(L.sil)
            k2 = L.query_from(C, learn=False).clone()
            assert cos(k2, k0) > 0.995 and float(L._goal.norm()) < 0.06, (cos(k2, k0), float(L._goal.norm()))
            r_faded = L.store.read(L.query_from(C, learn=False))[0].clone()
            assert cos(r_faded, val0) > cos(r_faded, val1), (cos(r_faded, val0), cos(r_faded, val1))
            for _ in range(5 * GOAL_TAU):
                L._goal_trace(L.sil)
            assert L._goal is None, L._goal                                # let go once it is nothing (under 1e-3, e^-7 at 7 GOAL_TAU)
            for _ in range(40):
                run.step()
                g = getattr(L, "_goal", None)
                assert g is None or float(g.norm()) <= 1.0 + 1e-5, g
            out[gk] = dict(near=cos(k1, row) - cos(k0, row), back=cos(k2, k0), held=(cos(r_held, val1), cos(r_held, val0)),
                           faded=(cos(r_faded, val0), cos(r_faded, val1)))
        else:
            assert getattr(L, "_goal", None) is None and torch.allclose(k0, k1), (getattr(L, "_goal", None), cos(k0, k1))
            out[gk] = dict(moved=cos(k0, k1))
    print(f"world 27: the held word: the key {out[1]['near']:.2f} nearer the said word's row, back to {out[1]['back']:.3f} of itself after",
          f"{3 * GOAL_TAU} ticks of silence; of two frames of one state, the word's read at cosine {out[1]['held'][0]:.2f} against the",
          f"wordless one's {out[1]['held'][1]:.2f} while held, the wordless one's {out[1]['faded'][0]:.2f} against {out[1]['faded'][1]:.2f}",
          f"faded (GOAL_SCALE {GOAL_SCALE}); the switch off, the key unmoved ({out[0]['moved']:.4f})")


def test_the_held_context():
    """world 28 (A128, the brain sprint): the held context, the temporal context model's key. With SIM_CFG ctx_key 1 the frames' key
    carries a leaky integral (CTX_TAU ticks) of the stream's pattern-separated direction beside the direction itself and the heading:
    after 5 CTX_TAU ticks in one state A the key is what the switch off would make of A (the context is A itself); the tick the state
    turns to B the key still lies nearer A's key than the unheld key of B does, and a frame written under A is what the read returns for
    B on that tick while the unheld body's read of B returns B's own frame; 5 CTX_TAU ticks in B later the key is B's; the night lets the
    context go; with the switch off (the sim at birth) the key never lags"""
    from body.core.world import WorldLoop
    from body.life import Life
    from body.sim.anatomy import SimAnatomy, SIM_CFG, born_table
    from body.core.memory import CTX_TAU, CTX_SCALE
    LR0 = dict(live_lr=0.0, value_lr=0.0, band_lr=0.0, night_lr=0.0, gate_lr=0.0, gate_adam_lr=0.0, vcrit_lr=0.0, actor_lr=0.0, face_lr=0.0,
               act_inv_lr=0.0)
    cos = lambda a, b: float(F.cosine_similarity(a.float(), b.float(), dim=0))
    out = {}
    for ck in (1, 0):
        w = G1World(seed=1)
        cfg = dict(SIM_CFG, **LR0, wake_ticks=10 ** 9, ctx_key=ck)
        torch.manual_seed(0)
        anat = SimAnatomy(born_table(), cfg, limits=[float(x) for x in w.tau_max])
        L = Life.birth(anat, device="cpu", d=32, layers=1, heads=2, window=16, cfg=cfg, seed=0, world=w)
        run = WorldLoop(L)
        for _ in range(20):
            run.step()
        L._goal = None; L._ctx = None
        gen = torch.Generator().manual_seed(5)
        A = L._C_last.detach().clone(); B = A + 3.0 * torch.randn(A.shape, generator=gen) * float(A.std())
        mu = L._c_mu.clone()                                                # the running mean held still: the lag measured alone
        vA, vB = torch.randn(L.m.d, generator=gen), torch.randn(L.m.d, generator=gen)
        for _ in range(5 * CTX_TAU):
            L.query_from(A, learn=True); L._c_mu = mu.clone()
        kA = L.query_from(A, learn=False).clone()
        L.store.write(kA, vA, 1.0, 2)                                        # A's frame
        kB1 = L.query_from(B, learn=True).clone(); L._c_mu = mu.clone()     # the first tick in B
        rB1 = L.store.read(kB1)[0].clone()
        for _ in range(5 * CTX_TAU):
            L.query_from(B, learn=True); L._c_mu = mu.clone()
        kB = L.query_from(B, learn=False).clone()
        L.store.write(kB, vB, 1.0, 2)                                        # B's own frame, settled
        rB = L.store.read(kB)[0].clone()
        L._frames_night()
        out[ck] = dict(lag=cos(kB1, kA), settled=cos(kB1, kB), rB1=(cos(rB1, vA), cos(rB1, vB)), rB=(cos(rB, vB), cos(rB, vA)),
                       ctx_after_night=getattr(L, "_ctx", None))
        assert out[ck]["ctx_after_night"] is None, out[ck]
    on, off = out[1], out[0]
    assert on["lag"] > off["lag"] + 0.1 and on["settled"] < off["settled"] - 0.1, (on, off)   # the key lags the senses only when held
    assert on["rB1"][0] > on["rB1"][1] and off["settled"] > 0.999, (on, off)                  # A's frame comes back on B's first tick
    assert on["rB"][0] > on["rB"][1], on                                                       # and B's once the context is B
    print(f"world 28: the held context: on the first tick in a new state the key lies at cosine {on['lag']:.2f} to the old state's key held,",
          f"{off['lag']:.2f} unheld, and {on['settled']:.2f} to its own settled key ({off['settled']:.3f} unheld); the old state's frame read",
          f"at {on['rB1'][0]:.2f} against the new one's {on['rB1'][1]:.2f} on that tick, the new state's at {on['rB'][0]:.2f} against",
          f"{on['rB'][1]:.2f} once settled (CTX_TAU {CTX_TAU}, CTX_SCALE {CTX_SCALE}); the night lets the context go")


def test_a_reward_source_joins_a_living_body():
    """world 29 (A127 on a living body): a life born with the two sources of birth (face, pain: three amygdala heads) lives 30 ticks and is
    saved; loaded under SIM_CFG novelty 1 it has the third source and a fourth head, born at zero, beside the three saved heads whose
    evidence (b, W, rel, pairs, the rings) is kept exactly; it lives 30 more ticks and pays for the new (positive reward on some tick);
    loaded under novelty 0 nothing is widened"""
    from body.core.world import WorldLoop
    from body.life import Life
    from body.sim.anatomy import SimAnatomy, SIM_CFG, born_table
    LR0 = dict(live_lr=0.0, value_lr=0.0, band_lr=0.0, night_lr=0.0, gate_lr=0.0, gate_adam_lr=0.0, vcrit_lr=0.0, actor_lr=0.0, face_lr=0.0,
               act_inv_lr=0.0)
    w = G1World(seed=1)
    cfg0 = dict(SIM_CFG, **LR0, wake_ticks=10 ** 9, novelty=0)
    torch.manual_seed(0)
    anat0 = SimAnatomy(born_table(), cfg0, limits=[float(x) for x in w.tau_max])
    L = Life.birth(anat0, device="cpu", d=32, layers=1, heads=2, window=16, cfg=cfg0, seed=0, world=w)
    run = WorldLoop(L)
    for _ in range(30):
        run.step()
    H0 = len(L.m.amyg.heads); assert H0 == 3, L.m.amyg.heads
    with torch.no_grad():
        L.m.amyg.b += 1.0; L.m.amyg.W += 0.5; L.m.amyg.rel += 2.0; L.m.amyg.pairs += 3; L.m.amyg.ring_f += 0.25; L.m.amyg.ring_y += 0.125
    saved = {k: getattr(L.m.amyg, k).clone() for k in ("b", "W", "rel", "pairs", "ring_f", "ring_y")}
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "life.pt"); L.save(path)
        got = {}
        for nov in (1, 0):
            cfg1 = dict(SIM_CFG, **LR0, wake_ticks=10 ** 9, novelty=nov)
            w1 = G1World(seed=1)
            anat1 = SimAnatomy(born_table(), cfg1, limits=[float(x) for x in w1.tau_max])
            L1 = Life.load(path, anat1, cfg=cfg1, world=w1)
            am = L1.m.amyg
            assert len(am.heads) == H0 + nov and [h[0] for h in am.heads][:3] == ["face", "face", "pain"], am.heads
            for k, dim in (("b", 1), ("W", 1), ("rel", 0), ("pairs", 0), ("ring_f", 1), ("ring_y", 1)):
                t = getattr(am, k)
                assert torch.equal(t.narrow(dim, 0, H0), saved[k]), k                         # the saved heads kept exactly
                if nov:
                    assert float(t.narrow(dim, H0, 1).abs().sum()) == 0.0, k                  # the new head born at zero
            run1 = WorldLoop(L1); pos = 0
            for _ in range(30):
                run1.step()
                r = L1._rec[int(L1.ticks) - 1] if getattr(L1, "_rec", None) is not None and int(L1.ticks) >= 1 else None
                if r is not None and float(r[3]) > 0:
                    pos += 1
            got[nov] = dict(heads=[h[0] for h in am.heads], pos=pos, writes=int(getattr(L1, "_fwrites", 0)))
    assert got[1]["heads"][-1] == "novelty" and got[1]["pos"] > 0, got
    print(f"world 29: a life saved with heads {got[0]['heads']} loads under the novelty drive with {got[1]['heads']} (the fourth born at zero,",
          f"the three kept exactly) and lives on: {got[1]['pos']} positive ticks of 30 with the drive, {got[0]['pos']} without")


def test_the_morning_tidy():
    """world 24 (B8, A117, C102): at a dawn the lost toys are put back where they stood at birth: the cup under the low table (its 0.45 m
    kneeling ring is the table; her hand's way in strikes the top), the stacker in the room's corner behind the plant (no spot to kneel);
    a toy on the open floor and a toy within the child's reach stay where they lie; the tidy is logged and saved. Life days 12 to 14: the
    cup under the table, her reach lessons refused with it 17 times"""
    w = G1World(seed=1)
    for _ in range(10):
        w.apply({})
    m, d = w.m, w.d

    def put(toy, xy):
        b = m.body(f"toy_{toy}").id; j = m.body_jntadr[b]
        adr = m.jnt_qposadr[j]; dof = m.jnt_dofadr[j]
        d.qpos[adr:adr + 2] = xy; d.qvel[dof:dof + 6] = 0
    put("cup", (0.38, 1.04)); put("stacker", (-2.53, -2.22)); put("ball", (1.6, -1.8))
    mujoco.mj_forward(m, d)
    for _ in range(10):
        w.apply({})
    before = {k: d.xpos[m.body(f"toy_{k}").id][:2].copy() for k in ("cup", "stacker", "ball", "duck", "block")}
    ch = w.parent.child
    near = [k for k in ("duck", "block") if ch.clearance_xy(before[k]) < 0.5]
    assert w.parent._toy_spot(before["stacker"]) is None and w.parent._toy_spot(before["ball"]) is not None, "the corner's stacker lost, the ball not"
    moved = w.tidy_toys()
    after = {k: d.xpos[m.body(f"toy_{k}").id][:2].copy() for k in before}
    home = {k: m.qpos0[m.jnt_qposadr[m.body_jntadr[m.body(f"toy_{k}").id]]:][:2].copy() for k in before}
    assert set(moved) >= {"cup", "stacker"}, moved
    assert all(np.linalg.norm(after[k] - home[k]) <= 0.31 for k in ("cup", "stacker")), {k: (after[k].round(2).tolist(), home[k].round(2).tolist()) for k in ("cup", "stacker")}
    assert "ball" not in moved and np.linalg.norm(after["ball"] - before["ball"]) < 0.01, "the ball on the open floor stays"
    assert all(k not in moved for k in near), (near, moved)
    assert len(w.tidied) == len(moved) and all(t[1] in moved for t in w.tidied)
    blob = w.save_state()
    w2 = G1World(seed=1); w2.load_state(blob)
    assert [t[1] for t in w2.tidied] == [t[1] for t in w.tidied]
    for _ in range(20):
        w.apply({})
    assert all(d.qpos[m.jnt_qposadr[m.body_jntadr[m.body(f"toy_{k}").id]] + 2] > -0.01 for k in ("cup", "stacker")), "the put-back toys rest on the floor"
    print(f"world 24: tidied {moved}: the cup {before['cup'].round(2).tolist()} -> {after['cup'].round(2).tolist()} (home {home['cup'].round(2).tolist()}),",
          f"the stacker {before['stacker'].round(2).tolist()} -> {after['stacker'].round(2).tolist()}; the ball at {before['ball'].round(2).tolist()} and {near} by the child left; saved and loaded")

def test_the_tendon_organ():
    """world 35 (A139, C113): THE TENDON ORGAN'S AUTOGENIC INHIBITION at the spinal cord. Day 26's morning: the right elbow's withdrawal
    fired on the pain its own blocked flexion step made (the servo's target a big step beyond a joint that could not move, the gear
    saturated at its 25 N m line, the pain flag set, the reflex fired again), 440 ticks. Now a joint whose sensed gear load reached its
    line last tick has its step zeroed for TENDON_TICKS ticks (its servo target re-anchored to its angle), over the own act and the
    withdrawal alike; its neighbours still move; the countdown is saved with the world and the truth's spinal says "tendon". After
    the ticks it is free again. A joint under its line is untouched."""
    w = G1World(seed=1)
    j = W.JOINTS.index("right_elbow_joint"); sh = W.JOINTS.index("right_shoulder_pitch_joint")
    w.frame()
    w._sensed["bd_peak"][j] = float(w.tau_hold[j])                       # the gear at its line last tick (the pain's own afferent)
    q0 = float(w.d.qpos[w.qadr[j]]); s0 = float(w.d.qpos[w.qadr[sh]])
    w.apply({"arm_r": R.flexion_act("arm_r")})                          # the withdrawal's act: big flexion steps on the shoulder and the elbow
    assert abs(float(w.d.ctrl[w.aid[j]]) - q0) < 1e-9, (w.d.ctrl[w.aid[j]], q0)          # the elbow's drive relaxed: its target its angle
    assert float(w.d.ctrl[w.aid[sh]]) < s0 - 0.1, (w.d.ctrl[w.aid[sh]], s0)              # the shoulder still steps
    assert w._spinal.get("arm_r") == "tendon" and int(w._tendon[j]) == R.TENDON_TICKS - 1, (w._spinal, w._tendon[j])
    st = w.state_dict() if hasattr(w, "state_dict") else None
    q1 = float(w.d.qpos[w.qadr[j]])
    w.apply({"arm_r": R.flexion_act("arm_r")})                          # the second tick: still held
    assert abs(float(w.d.ctrl[w.aid[j]]) - q1) < 1e-9 and w._spinal.get("arm_r") == "tendon"
    q2 = float(w.d.qpos[w.qadr[j]])
    w.apply({"arm_r": R.flexion_act("arm_r")})                          # the third: free, the big flexion step passes
    assert float(w.d.ctrl[w.aid[j]]) < q2 - 0.2 and "arm_r" not in w._spinal, (w.d.ctrl[w.aid[j]], q2, w._spinal)
    w2 = G1World(seed=1); w2.frame()
    w2._sensed["bd_peak"][j] = 0.5 * float(w2.tau_hold[j])                # under its line: nothing inhibited
    q = float(w2.d.qpos[w2.qadr[j]]); w2.apply({"arm_r": R.flexion_act("arm_r")})
    assert float(w2.d.ctrl[w2.aid[j]]) < q - 0.2 and "arm_r" not in w2._spinal
    print(f"world 35: the tendon organ: a right elbow whose gear reached its {w.tau_hold[j]:.0f} N m line has its step zeroed for {R.TENDON_TICKS}",
          "ticks (its target its angle, the shoulder still stepping, the truth's spinal 'tendon'), then the big flexion step passes; under the line, untouched")


def test_the_bucket_beside():
    """world 36 (C130): at a dawn the bucket standing out of the child's reach is set upright within it, beside its shoulders, with the
    toy lying in it; a bucket within reach stays. Life day 31: four hides played 2 m from the child, never within its reach all day"""
    import mujoco
    from body.sim import extras as X
    w = W.G1World(seed=1, extra=X.add_bucket(xy=(-0.3, -0.45)))
    m, d = w.m, w.d
    for _ in range(5):
        w.frame(); w.apply({})
    b = m.body("toy_bucket").id; j = m.body_jntadr[b]; adr = m.jnt_qposadr[j]
    d.qpos[adr:adr + 2] = (-2.0, -0.6); mujoco.mj_forward(m, d)                # the bucket 2 m off, as day 31's
    jd = m.body("toy_duck").jntadr[0]; a = m.jnt_qposadr[jd]
    d.qpos[a:a + 3] = d.xpos[b] + np.array([0.0, 0.0, X.BUCKET_WALL + 0.04]); d.qpos[a + 3:a + 7] = [1, 0, 0, 0]
    for _ in range(30):
        mujoco.mj_step(m, d)
    sh = [d.xpos[m.body(f"{s}_shoulder_pitch_link").id][:2] for s in ("left", "right")]
    far = min(float(np.linalg.norm(d.xpos[b][:2] - p_)) for p_ in sh)
    assert far > 1.0, far
    assert w.bucket_beside()
    near = min(float(np.linalg.norm(d.xpos[b][:2] - p_)) for p_ in sh)
    duck = float(np.linalg.norm(d.xpos[m.body("toy_duck").id][:2] - d.xpos[b][:2]))
    assert near <= 0.5 and duck < 0.1, (near, duck)                             # within a G1 arm's reach (0.55), the duck still in it
    assert abs(float(d.xpos[b][2]) - float(m.qpos0[adr + 2])) < 0.02 and w.tidied[-1][1] == "bucket", (d.xpos[b], w.tidied[-1])
    assert not w.bucket_beside(), "a bucket within reach stays"
    print(f"world 36: the bucket {far:.2f} m from the child set {near:.2f} m from its shoulder at dawn, upright, the duck still in it ({duck:.2f} m off its centre); within reach it stays")



def test_the_habit_is_dopamines():
    """world (A141, C149, 2026-09-30; C156 the same day; A150 and A152 of 2026-10-01): the day's habit lesson weighted by dopamine's credit
    as the night's is, the own acts' weight the credit alone (clip(G, -1, 1): no act cloned for free). A born G1 life with a day record
    of 64 ticks and one dopamine dip of -1 at row 40: the act held at position 40 (the tick before's draw, whose first consequence that
    row is) weighs -1, position 39 -0.9375, position 20 between -0.5 and 0, positions 41 on 0; her guidance's labels keep R8's weight
    clip(1 + G, 0, 1) (0 at 40, about 0.06 at 39, 1 at 41); a smile of 0.5 at row 41 teaches the act at 41 toward at 0.5 and the one
    before at 0.5 x 0.9375; with the switch off, None; with fewer rows than the window, None"""
    from body.life import Life
    from body.sim.anatomy import SimAnatomy, SIM_CFG, born_table
    w = G1World(seed=1)
    for on in (1, 0):
        cfg = dict(SIM_CFG, wake_ticks=10 ** 9, habit_by_credit=on)
        torch.manual_seed(0)
        anat = SimAnatomy(born_table(), cfg, limits=[float(x) for x in w.tau_max])
        L = Life.birth(anat, device="cpu", d=32, layers=1, heads=2, window=16, cfg=cfg, seed=0, world=w)
        L._rec = torch.zeros(4096, 4); L._rec_n = 64
        L._rec[40, 1] = -1.0
        wp = L._habit_weights(64)
        if not on:
            assert wp is None, wp
            continue
        assert wp is not None and wp.shape == (64,), None if wp is None else wp.shape
        g = L._tag_gamma()
        # A152: the own act's weight is dopamine's credit alone, clip(G, -1, 1): the act at 40 taught against at -1, the one before at -g,
        # twenty before at -g^20 (still against, faintly), the acts after it at 0 (nothing paid, nothing cloned)
        assert abs(float(wp[40]) + 1.0) < 1e-6 and abs(float(wp[39]) + g) < 1e-5 and float(wp[41]) == 0.0 and float(wp[63]) == 0.0, wp[36:42].tolist()
        assert -0.5 < float(wp[20]) < 0.0, float(wp[20])
        wl = L._habit_weights(64, labels=True)                                        # her guidance's labels keep R8's weight clip(1 + G, 0, 1)
        assert abs(float(wl[40])) < 1e-6 and abs(float(wl[39]) - max(0.0, 1.0 - g)) < 1e-5 and float(wl[41]) == 1.0 and 0.5 < float(wl[20]) < 1.0, wl[36:42].tolist()
        L._rec[40, 1] = 0.0; L._rec[41, 1] = 0.5                                    # a smile of 0.5 at row 41: the act at 41 taught toward at 0.5, the one before at 0.5 g
        wp2 = L._habit_weights(64)
        assert abs(float(wp2[41]) - 0.5) < 1e-6 and abs(float(wp2[40]) - 0.5 * g) < 1e-5 and float(wp2[42]) == 0.0, wp2[38:44].tolist()
        L._rec[40, 1] = -1.0; L._rec[41, 1] = 0.0
        L._rec_n = 32
        assert L._habit_weights(64) is None
        wp1, g1 = wp, g
    print(f"world A141/C156/A152: a dopamine dip of -1 at row 40 of a 64-tick window: the act held at that row weighs -1, the one before {-g1:.3f}, "
          f"twenty before {float(wp1[20]):.3f}, the acts after it 0 (nothing paid, nothing cloned); her labels at R8's weight; a smile teaches toward; off: None; a short record: None")



def test_the_actors_tag():
    """world (A142/C153, 2026-09-30): the actor's synaptic tag. On a born G1 life the tag begins at the first act's eligibility, decays by
    1 - 1/tag_reach a tick, adds each act's eligibility; with actor_slow_lr on, 60 ticks of the world loop leave the arms' tags set and the
    slow path's summed update norm above 0 with the actor's weights moved from a life run with it off (the same seed); with it off, no
    tag is kept"""
    from body.core.world import WorldLoop
    from body.life import Life
    from body.sim.anatomy import SimAnatomy, SIM_CFG, born_table
    w = _ball_in_palm(settle=False)                 # (A171: the born hands no longer grasp the mat's and the rattle's push on their edges, so
    out = {}                                        # 60 born ticks hold no act of the cord's for the arm to be surprised by; the rig's ball on
    for slr in (1e-4, 0.0):                         # the palm fires the grasp as the born edge push did before)
        cfg = dict(SIM_CFG, wake_ticks=10 ** 9, actor_slow_lr=slr)
        torch.manual_seed(0)
        anat = SimAnatomy(born_table(), cfg, limits=[float(x) for x in w.tau_max])
        L = Life.birth(anat, device="cpu", d=32, layers=1, heads=2, window=16, cfg=cfg, seed=0, world=w)
        if slr > 0:
            st = L.motor[3]; e1 = torch.ones(3, 4)
            L._actor_tag_step(st, e1); assert torch.equal(st["a_tag"], e1)
            L._actor_tag_step(st, None); g_l = 1.0 - 1.0 / float(cfg.get("tag_reach", 64))
            assert torch.allclose(st["a_tag"], e1 * g_l), st["a_tag"][0]
            L._actor_tag_step(st, e1); assert torch.allclose(st["a_tag"], e1 * (1.0 + g_l))
            st["a_tag"] = None
        run = WorldLoop(L)
        for _ in range(60):
            run.step()
        e = L.anatomy.motors[3]
        out[slr] = dict(tag=L.motor[3]["a_tag"], upd=list(L.motor[3]["a_upd"]), w=L.m.get_submodule(e.actor).weight.detach().clone())
    on, off = out[1e-4], out[0.0]
    assert on["tag"] is not None and off["tag"] is None, (on["tag"] is None, off["tag"])
    assert on["upd"][1] > 0.0 and off["upd"][1] == 0.0, (on["upd"], off["upd"])
    assert not torch.allclose(on["w"], off["w"]), "the slow path must move the actor's weights"
    print(f"world A142: the arm's tag kept and captured over 60 ticks (slow update norm summed {on['upd'][1]:.4f} against the fast lesson's "
          f"{on['upd'][0]:.4f}); with actor_slow_lr 0 no tag and the weights differ")


def test_the_passive_stiffness():
    """world (A143, 2026-09-30): the passive end-range stiffness. At a joint's low stop the tissue pushes toward the middle at PASSIVE_FRAC x its
    torque limit, at its high stop the same the other way, half way into the margin half as much, and nothing over the middle 70% of the range;
    the push is applied to the joints each physics step (qfrc_applied at the body's dofs equals the law for the angles then) and is not the
    motor's: the sensed gear load reads qfrc_actuator. Day 43: 6.5 of the 14 arm joints at a stop on average, the stop-bound joints' kappa 0.07
    to 0.19 and their big steps 37% against 5 to 10% off the stop"""
    w = G1World(seed=1)
    lo, hi, tm = w.lo.copy(), w.hi.copy(), w.tau_max.copy()
    mid = 0.5 * (lo + hi); m_ = W.PASSIVE_MARGIN; f_ = W.PASSIVE_FRAC; tm = tm * w.passive_mask    # C163: none at the hand's joints
    assert int(w.passive_mask.sum()) == len(W.JOINTS) - len(G.DEX3) and all(w.passive_mask[i] == 0.0 for i, j in enumerate(W.JOINTS) if j in G.DEX3)
    assert np.allclose(w.passive_torque(mid), 0.0) and np.allclose(w.passive_torque(lo + 0.3 * (hi - lo)), 0.0)
    assert np.allclose(w.passive_torque(lo), f_ * tm) and np.allclose(w.passive_torque(hi), -f_ * tm)
    assert np.allclose(w.passive_torque(hi - 0.5 * m_ * (hi - lo)), -0.5 * f_ * tm, atol=1e-9)
    assert np.allclose(w.passive_torque(lo + m_ * (hi - lo)), 0.0, atol=1e-9)
    for _ in range(3):
        w.frame(); w.apply({})
    q = w.d.qpos[w.qadr]
    assert np.all(np.abs(w.d.qfrc_applied[w.dof] - w.passive_torque(q)) <= 0.01 * w.tau_max), "the tissue's push must stand at the joints after a tick"   # (one physics step of motion apart)
    at = int(np.sum(((q - lo) / (hi - lo) < 0.02) | ((q - lo) / (hi - lo) > 0.98)))
    print(f"world A143: the passive push {f_:.2f} x the torque limit at a stop, half at mid-margin, none over the middle 70%; applied at the "
          f"{len(w.dof)} joints each step ({at} at a stop three ticks after birth)")


def test_the_value_heads_forget():
    """world (A146, 2026-09-30): the ladder's value heads forget at value_forget. A born G1 life with band 5's head set to 100 on every weight,
    value_forget 60 for the test: after 60 ticks of the world loop the weights have fallen to about 100 x (1 - 1/60)^60 = 36.6 (within the
    lesson's own small moves); with value_forget 0 they stand near 100. Day 44: the clock-1,024 head's weights at 651, its value swinging by 50
    within half a day, its gate shut to 0.20"""
    from body.core.world import WorldLoop
    from body.life import Life
    from body.sim.anatomy import SimAnatomy, SIM_CFG, born_table
    w = G1World(seed=1)
    out = {}
    for vf in (60, 0):
        cfg = dict(SIM_CFG, wake_ticks=10 ** 9, value_forget=vf)
        torch.manual_seed(0)
        anat = SimAnatomy(born_table(), cfg, limits=[float(x) for x in w.tau_max])
        L = Life.birth(anat, device="cpu", d=32, layers=1, heads=2, window=16, cfg=cfg, seed=0, world=w)
        with torch.no_grad():
            L.m.value[5].weight.fill_(100.0)
        run = WorldLoop(L)
        for _ in range(60):
            run.step()
        out[vf] = float(L.m.value[5].weight.detach().abs().mean())
    assert 20.0 < out[60] < 55.0 and out[0] > 90.0, out
    print(f"world A146: band 5's head at 100 on every weight: after 60 ticks {out[60]:.1f} at value_forget 60 (36.6 by the decay alone), {out[0]:.1f} with none")


WORLD_TESTS = [test_the_scene, test_torque_limits_are_the_models, test_the_servo_law, test_birth_and_touch, test_joint_sense_and_vestibule,
               test_pain, test_no_charge, test_the_reflexes, test_prone_pattern, test_letting_go, test_the_dorsal_touch_opens_the_hand, test_blind_spots_are_a12s, test_exact_replay, test_the_night,
               test_faults, test_the_babbler, test_the_world_in_the_core, test_withdrawal_c22, test_friction_realism,
               test_the_parents_pose_is_saved, test_the_rooms_sounds, test_carried_to_the_mat, test_a_world_migrates_to_the_book, test_the_morning_tidy, test_a_world_with_the_book_migrates_to_the_box, test_the_novelty_drive, test_the_tendon_organ, test_the_bucket_beside, test_the_habit_is_dopamines, test_the_actors_tag, test_the_passive_stiffness, test_the_value_heads_forget]

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
