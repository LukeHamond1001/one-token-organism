"""Writes body/sim/highchair.xml and its floor texture: the high chair world of docs/SIM_DESIGN.md sections 5.1 and 5.8.

The child and the parent are ONE robot written twice by robot() below, so their bodies are identical by construction
(the same planar two-joint arm with a sticky mitten, the same pan/tilt head with a face, two ears, a charge light, the
same high chair). They differ only in the name prefix ('child_' / 'parent_'), the side of the table, and an accent
colour on the ears, collar and mitten cuff so a viewer can tell them apart.

Also here, for the build to reuse: the arm's inverse and forward kinematics (arm_ik, arm_fk), the head's aim (head_lookat),
the parent's guiding hold switched on and off from the current pose (guide), and the declared pain threshold (F_PAIN).

Edit the numbers here, then run:  python3 body/sim/make_highchair.py   (the XML is generated; do not hand-edit it)."""
import math
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent

# ---- the design's numbers (SIM_DESIGN.md 5.1, 5.8, 7; the rest from the measured scratch scene t9/t11) ----
TABLE_HX, TABLE_HY, TABLE_HZ = .40, .32, .02   # table 0.8 x 0.64 m, top surface at z = 0
RIM_H, RIM_T = .03, .016                      # the rim: 3 cm high, 1.6 cm thick (t9)
FLOOR_Z = -.72                                # a 72 cm table (the room's floor)
BASE_Y = .42        # each robot's head axis stands .42 m from the table's centre line (t9: head at y = -.42)
SHOULDER_Y = .12    # so each shoulder is at .30 m from the centre line (t9: shoulder at y = -.30)
ARM_Z = .08         # the arm's plane: 8 cm above the table
L1, L2 = .22, .19   # upper arm and forearm
HEAD_Z = .36        # head centre 0.36 m above the table
PAN_DEG = 60        # pan +-60 deg
TILT_DEG = (-75, 20)
SH_DEG, EL_DEG = 110, 150    # arm joint ranges (t9; not stated in the design)
MITTEN_R, MITTEN_DROP = .022, .052   # a downward capsule: bottom 6 mm above the table (t9)
EYE_UP = .016       # the eye camera sits on the face, .016 m above the tilt axis (between the drawn eyes)
EYE_FOVY = 60       # one 128 px render: periphery 64 px over 60 deg + fovea the centre 32 px (2 atan(tan 30 / 4) = 16.4 deg)
OBJ_MASS = .06      # 60 g each
PAD_XY, PAD_R = (.22, -.12), .045   # the charge pad: a flat magenta disc, radius 4.5 cm (t9/t11 place)
KEY_POS = (-1.35, .15, 1.85)         # the key light: high (below the 1.98 m ceiling), from the window side

# The four objects at the sizes t11 measured its 0.81 recognition on (t11d_contrast.py), which the design calls "1.3x":
# ball r .045 (1.25x), cube half .033 (1.3x), duck body 1.3x, cup r .036 half-height .036 (1.2x). Kept as measured.
OBJ_POS = {"ball": (.18, .0, 0), "cube": (-.07, .07, 0), "duck": (.07, -.12, -50), "cup": (-.21, -.07, 20)}

# ---- colours: saturated objects (t11), a pale grey table, a neutral room ----
MAT = dict(
    shell=dict(rgba=".95 .95 .93 1", specular=".28", shininess=".5"),
    joint=dict(rgba=".20 .21 .24 1", specular=".35", shininess=".5"),
    mitten=dict(rgba=".16 .17 .20 1", specular=".25", shininess=".4"),
    screen=dict(rgba=".07 .08 .10 1", specular=".7", shininess=".9"),
    glow=dict(rgba=".80 .95 1 1", emission=".9", specular="0"),
    charge=dict(rgba=".45 .95 .85 1", emission=".8", specular="0"),
    accent_child=dict(rgba=".98 .55 .38 1", specular=".35", shininess=".55"),
    accent_parent=dict(rgba=".24 .60 .68 1", specular=".35", shininess=".55"),
    table=dict(rgba=".72 .72 .74 1", specular=".15", shininess=".3", reflectance=".04"),   # pale grey as lit (about 215-240 of
    rim=dict(rgba=".78 .78 .80 1", specular=".1", shininess=".3"),                          # 255 in the eye), the rim a raised lip of it
    pad=dict(rgba=".90 .30 .90 1", emission=".15", specular=".2", shininess=".5"),
    ball=dict(rgba=".92 .12 .10 1", specular=".5", shininess=".7"),
    cube=dict(rgba=".10 .30 .98 1", specular=".4", shininess=".6"),
    duck=dict(rgba="1 .86 .10 1", specular=".4", shininess=".6"),
    beak=dict(rgba="1 .52 .08 1", specular=".3", shininess=".5"),
    black=dict(rgba=".05 .05 .06 1", specular=".6", shininess=".9"),
    cup=dict(rgba=".10 .74 .24 1", specular=".5", shininess=".7"),
    cup_in=dict(rgba=".05 .42 .13 1", specular=".2", shininess=".5"),
    wood=dict(rgba=".87 .76 .60 1", specular=".2", shininess=".35"),
    floor=dict(texture="floor_oak", texrepeat="1.6 1.6", specular=".15", shininess=".35", reflectance=".05"),
    wall=dict(rgba=".90 .88 .84 1", emission=".24", specular="0", shininess="0"),
    ceiling=dict(rgba=".95 .95 .94 1", emission=".4", specular="0"),
    trim=dict(rgba=".97 .97 .96 1", specular=".1", shininess=".3"),
    rug=dict(rgba=".78 .75 .70 1", specular="0"),
    sky=dict(texture="window_sky", emission=".75", specular="0"),
    art_a=dict(rgba=".93 .78 .62 1", specular="0"),
    art_b=dict(rgba=".70 .76 .86 1", specular="0"),
    art_c=dict(rgba=".96 .92 .84 1", specular="0"),
)

STATIC = 'contype="1" conaffinity="1"'   # table, rim, floor, objects, mittens, shoulder hubs and booms
LINK = STATIC                            # arm links touch everything solid, the other robot's links included (an arm's own
                                         # upper arm and forearm are parent and child bodies, so MuJoCo never pairs them)
DECOR = 'contype="0" conaffinity="0" density="0"'   # seen, never touched, no mass
ROOM = DECOR + ' group="2"'   # the room (walls, window, floor's look, print, sideboard): seen by every camera and eye (group 2 is
                              # drawn by default); its own group only so a portrait (tools/sim_look.py faces) can hide it


def f(*v):
    return " ".join(f"{x:.4g}" for x in v)


def lookat_xyaxes(pos, target):
    fwd = np.subtract(target, pos); fwd = fwd / np.linalg.norm(fwd)
    x = np.cross(fwd, [0, 0, 1]); x /= np.linalg.norm(x)
    y = np.cross(-fwd, x)
    return f(*x, *y)


# ---- kinematics (world coordinates), shared by the keyframe below and the scripts that pose or drive the robots ----
def to_local(p, xy):
    """World (x, y) -> robot p's base frame (the child's is the world shifted; the parent's is turned 180 deg)."""
    s = 1 if p == "child_" else -1
    return np.array([s * xy[0], s * xy[1] + BASE_Y])


def arm_ik(p, xy, elbow=1):
    """Joint angles (shoulder, elbow) in radians that put robot p's mitten centre over world (x, y); elbow=+1 bends the
    elbow to the robot's right side of the reach line, -1 to its left. Angles are measured from straight ahead, CCW from above."""
    lx, ly = to_local(p, xy) - np.array([0, SHOULDER_Y])
    r = float(np.clip(np.hypot(lx, ly), abs(L1 - L2) + 1e-3, L1 + L2 - 1e-3))
    q2 = elbow * math.acos(np.clip((r * r - L1 * L1 - L2 * L2) / (2 * L1 * L2), -1, 1))
    q1 = math.atan2(-lx, ly) - math.atan2(L2 * math.sin(q2), L1 + L2 * math.cos(q2))
    return q1, q2


def arm_fk(p, q1, q2):
    s = 1 if p == "child_" else -1
    u = lambda a: np.array([-math.sin(a), math.cos(a)])
    loc = np.array([0, SHOULDER_Y]) + L1 * u(q1) + L2 * u(q1 + q2)
    return np.array([s * loc[0], s * (loc[1] - BASE_Y)])


def head_lookat(p, target):
    """(pan, tilt) in radians that point robot p's eye camera at world point target (exact for this head's camera offset)."""
    lx, ly = to_local(p, target[:2])
    pan = math.atan2(-lx, ly)
    rho, h = math.hypot(lx, ly), target[2] - HEAD_Z
    tilt = math.atan2(h, rho) - math.asin(EYE_UP / math.hypot(rho, h))
    return (float(np.clip(pan, -math.radians(PAN_DEG), math.radians(PAN_DEG))),
            float(np.clip(tilt, math.radians(TILT_DEG[0]), math.radians(TILT_DEG[1]))))


# ---- the pain threshold (SIM_DESIGN 5.5, 5.6, 7): F_pain = k x the limb's own declared maximum steady force ----
SH_TAU, EL_TAU = 1.2, .8      # the servos' declared torque limits, N m (the forcerange below)
MITTEN_STALL_N = EL_TAU / L2  # the mitten's declared stall force: the elbow's limit over the forearm, 4.2 N, the force it holds
                              # sideways against (its weakest direction ranges 2.9 N at full reach, where the shoulder's
                              # 1.2 / 0.41 limits it, to 4.2 N folded); declared once here and written into the model file
F_PAIN_K = 5
F_PAIN = F_PAIN_K * MITTEN_STALL_N   # 21 N ("about 20 N"): a link or mitten force above this is pain

# ---- the parent's guiding hold on the child's forearm (SIM_DESIGN 5.1, 5.4) ----
GUIDE_AT = .15      # the hold point on the forearm: .15 m from the elbow (the wrist, 4 cm before the mitten); site child_forearm_hold


def guide(m, d, on=True):
    """Switch the parent's guiding hold on (or off). On writes the weld's offset from the current pose, so it holds the parent's
    hand and the child's forearm point where they are now (no yank), with torquescale 0: the point is held, the forearm's angle
    stays free. The offset lives in m.eq_data, so SimWorld's save and restore must carry that row with d.eq_active."""
    import mujoco
    e = m.equality("parent_guides_child").id
    if on:
        b1, b2 = m.eq_obj1id[e], m.eq_obj2id[e]
        anchor = np.array([0, GUIDE_AT, 0])                       # in body2 (the child's forearm)
        p = d.xpos[b2] + d.xmat[b2].reshape(3, 3) @ anchor         # that point in the world now
        q1inv = np.zeros(4); mujoco.mju_negQuat(q1inv, d.xquat[b1])
        rq = np.zeros(4); mujoco.mju_mulQuat(rq, q1inv, d.xquat[b2])
        m.eq_data[e, 0:3] = anchor
        m.eq_data[e, 3:6] = d.xmat[b1].reshape(3, 3).T @ (p - d.xpos[b1])   # the same point in body1 (the parent's hand)
        m.eq_data[e, 6:10] = rq
        m.eq_data[e, 10] = 0.0
    d.eq_active[e] = 1 if on else 0


# the birth pose: each arm folded to its own left (shoulder 77 deg, elbow -132 deg: inside both ranges), clear of the objects;
# the child looks at the table, the parent at the child; faces neutral
BIRTH = {"child_": dict(mitten=(-.06, -.141), elbow=-1, look=(-.02, .02, .03)),
         "parent_": dict(mitten=(.06, .141), elbow=-1, look=(0, -.30, .30))}


def keyframe(xml_path):
    import mujoco
    m = mujoco.MjModel.from_xml_path(str(xml_path)); d = mujoco.MjData(m)
    q = m.qpos0.copy(); ctrl = np.zeros(m.nu)
    for p, b in BIRTH.items():
        q1, q2 = arm_ik(p, b["mitten"], b["elbow"]); pan, tilt = head_lookat(p, b["look"])
        for j, v in ((f"{p}shoulder", q1), (f"{p}elbow", q2), (f"{p}pan", pan), (f"{p}tilt", tilt)):
            q[m.jnt_qposadr[m.joint(j).id]] = v; ctrl[m.actuator(j).id] = v
    d.qpos[:] = q; d.ctrl[:] = ctrl
    for _ in range(1000):              # 2 s: the objects settle on the table (the duck rocks onto its rest), the servos hold
        mujoco.mj_step(m, d)
    q = d.qpos.copy()
    for p, b in BIRTH.items():         # the robots exactly at their targets, everything still
        for j in ("shoulder", "elbow", "pan", "tilt"):
            q[m.jnt_qposadr[m.joint(p + j).id]] = ctrl[m.actuator(p + j).id]
        for j in ("mouth", "mouth_R"):
            q[m.jnt_qposadr[m.joint(p + j).id]] = 0.0
    d.qpos[:] = q; d.qvel[:] = 0; mujoco.mj_forward(m, d)
    hits = [(mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, c.geom1), mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, c.geom2))
            for c in d.contact[:d.ncon] if c.dist < 0]
    assert all("table" in a or "table" in b or "rim" in a or "rim" in b for a, b in hits), f"the birth pose interpenetrates: {hits}"
    return f'<key name="birth" qpos="{" ".join(f"{v:.6g}" for v in q)}" ctrl="{" ".join(f"{v:.6g}" for v in ctrl)}"/>'


# the body's shape (seen only; the arm, mitten, head joints and camera above are what the world and the design use)
SEAT_Z = -.105                         # the high chair's seat, 10.5 cm below the table top
TORSO_C, TORSO_R = (0, -.03, .07), (.125, .094, .175)   # an egg from the seat to the collar
HEAD_R = (.088, .074, .068)            # a rounded-box head, 17.6 cm wide
FACE_Y = .0795                         # the face's front plane (the screen's surface) in the head frame
EYE_UP_ = EYE_UP


def torso_front(z):
    """The torso's front surface (local y) at height z (an egg: an ellipsoid)."""
    cz = (z - TORSO_C[2]) / TORSO_R[2]
    return TORSO_C[1] + TORSO_R[1] * max(0.0, 1 - cz * cz) ** .5


def robot(p, accent, side):
    """One robot. p is the name prefix; side +1 puts it at y = -BASE_Y facing +y (the child), -1 at +BASE_Y facing -y."""
    y0 = -side * BASE_Y
    yaw = 0 if side > 0 else 180
    fz = FLOOR_Z + .012
    cl = .15                                # the charge light's height on the chest
    return f"""
    <!-- ===================== {p[:-1]}: the robot ===================== -->
    <body name="{p}base" pos="0 {y0:.4g} 0" euler="0 0 {yaw}">
      <!-- the high chair (seen only): seat, back, four splayed legs, two footrests -->
      <geom name="{p}seat" type="mesh" mesh="seat" pos="0 -.03 {SEAT_Z - .016:.4g}" material="wood" {DECOR}/>
      <geom name="{p}chair_back" type="mesh" mesh="chair_back" pos="0 -.15 {SEAT_Z + .15:.4g}" euler="-7 0 0" material="wood" {DECOR}/>
      <geom type="capsule" fromto=".115 .05 {SEAT_Z - .02:.4g} .20 .15 {fz:.4g}" size=".017" material="wood" {DECOR}/>
      <geom type="capsule" fromto="-.115 .05 {SEAT_Z - .02:.4g} -.20 .15 {fz:.4g}" size=".017" material="wood" {DECOR}/>
      <geom type="capsule" fromto=".115 -.11 {SEAT_Z - .02:.4g} .20 -.22 {fz:.4g}" size=".017" material="wood" {DECOR}/>
      <geom type="capsule" fromto="-.115 -.11 {SEAT_Z - .02:.4g} -.20 -.22 {fz:.4g}" size=".017" material="wood" {DECOR}/>
      <geom type="capsule" fromto="-.165 .105 -.47 .165 .105 -.47" size=".013" material="wood" {DECOR}/>
      <geom type="capsule" fromto="-.165 -.18 -.47 .165 -.18 -.47" size=".013" material="wood" {DECOR}/>
      <!-- the torso, and the charge light on the chest (its brightness is set from the charge at run time) -->
      <geom name="{p}torso" type="ellipsoid" pos="{f(*TORSO_C)}" size="{f(*TORSO_R)}" material="shell" {DECOR}/>
      <geom name="{p}charge_light" type="cylinder" pos="0 {torso_front(cl) + .002:.4g} {cl}" euler="-90 0 0" size=".015 .003" material="charge" {DECOR}/>
      <geom type="cylinder" pos="0 {torso_front(cl) - .001:.4g} {cl}" euler="-90 0 0" size=".021 .004" material="joint" {DECOR}/>
      <!-- the shoulder: a boom from the chest to the hub, in the arm's plane -->
      <geom name="{p}boom" type="capsule" fromto="0 {torso_front(ARM_Z) - .02:.4g} {ARM_Z} 0 {SHOULDER_Y} {ARM_Z}" size=".024" material="joint" {STATIC}/>
      <geom name="{p}hub" type="cylinder" pos="0 {SHOULDER_Y} {ARM_Z}" size=".032 .026" material="joint" {STATIC}/>
      <geom type="cylinder" pos="0 {SHOULDER_Y} {ARM_Z + .027:.4g}" size=".025 .002" material="{accent}" {DECOR}/>
      <!-- the arm: planar, two vertical hinges, links {L1} and {L2} m, {ARM_Z * 100:.0f} cm above the table -->
      <body name="{p}upper_arm" pos="0 {SHOULDER_Y} {ARM_Z}">
        <joint name="{p}shoulder" type="hinge" axis="0 0 1" range="{-SH_DEG} {SH_DEG}" damping=".1" armature=".002"/>
        <geom name="{p}upper_arm" type="capsule" fromto="0 0 0 0 {L1} 0" size=".018" mass=".3" material="shell" {LINK}/>
        <site name="{p}upper_arm_touch" type="capsule" fromto="0 0 0 0 {L1} 0" size=".020" group="4"/>
        <body name="{p}forearm" pos="0 {L1} 0">
          <joint name="{p}elbow" type="hinge" axis="0 0 1" range="{-EL_DEG} {EL_DEG}" damping=".05" armature=".002"/>
          <geom type="cylinder" size=".025 .023" material="joint" {DECOR}/>
          <geom name="{p}forearm" type="capsule" fromto="0 0 0 0 {L2} 0" size=".016" mass=".2" material="shell" {LINK}/>
          <site name="{p}forearm_touch" type="capsule" fromto="0 0 0 0 {L2} 0" size=".018" group="4"/>
          <site name="{p}forearm_hold" pos="0 {GUIDE_AT} 0" size=".006" group="4"/>
          <!-- the mitten: dark, downward, with the touch sensor and the adhesion grip -->
          <body name="{p}hand" pos="0 {L2} 0">
            <geom name="{p}mitten" type="capsule" fromto="0 0 0 0 0 {-MITTEN_DROP}" size="{MITTEN_R}" mass=".05" material="mitten" friction="1 .005 .0001" {STATIC}/>
            <geom type="cylinder" pos="0 0 .006" size=".0245 .006" material="{accent}" {DECOR}/>
            <site name="{p}mitten_touch" type="capsule" fromto="0 0 0 0 0 {-MITTEN_DROP}" size="{MITTEN_R + .002:.4g}" group="4"/>
          </body>
        </body>
      </body>
      <!-- the head: pan at the neck, tilt at the head's centre, {HEAD_Z} m above the table -->
      <geom name="{p}collar" type="mesh" mesh="collar" pos="0 {TORSO_C[1] / 2:.4g} {TORSO_C[2] + TORSO_R[2] - .004:.4g}" material="{accent}" {DECOR}/>
      <body name="{p}neck" pos="0 0 .27">
        <joint name="{p}pan" type="hinge" axis="0 0 1" range="{-PAN_DEG} {PAN_DEG}" damping=".05" armature=".0005"/>
        <geom type="cylinder" pos="0 0 .012" size=".028 .026" mass=".05" material="joint" contype="0" conaffinity="0"/>
        <body name="{p}head" pos="0 0 {HEAD_Z - .27:.4g}" gravcomp="1">
          <joint name="{p}tilt" type="hinge" axis="1 0 0" range="{TILT_DEG[0]} {TILT_DEG[1]}" damping=".05" armature=".0005"/>
          <geom name="{p}head" type="mesh" mesh="head" material="shell" mass=".2" contype="0" conaffinity="0"/>
          <!-- the face: a dark screen in a white frame, two lit eyes, a mouth whose curvature is the {p}face actuator
               (a fixed centre bar and two capsules hinged at its ends, turning up for a smile, down for a frown) -->
          <geom name="{p}bezel" type="mesh" mesh="bezel" pos="0 .066 .002" material="shell" {DECOR}/>
          <geom name="{p}screen" type="mesh" mesh="screen" pos="0 {FACE_Y - .0085:.4g} .002" material="screen" {DECOR}/>
          <geom name="{p}eye_L" type="ellipsoid" pos="-.029 {FACE_Y:.4g} {EYE_UP_}" size=".0115 .0028 .0155" material="glow" {DECOR}/>
          <geom name="{p}eye_R" type="ellipsoid" pos=".029 {FACE_Y:.4g} {EYE_UP_}" size=".0115 .0028 .0155" material="glow" {DECOR}/>
          <geom name="{p}mouth_mid" type="capsule" fromto="-.010 {FACE_Y + .0005:.4g} -.023 .010 {FACE_Y + .0005:.4g} -.023" size=".006" material="glow" {DECOR}/>
          <body name="{p}mouth_L" pos="-.010 {FACE_Y + .0005:.4g} -.023" gravcomp="1">
            <joint name="{p}mouth" type="hinge" axis="0 1 0" range="-35 35" damping=".0015" armature=".0001"/>
            <geom name="{p}mouth_L" type="capsule" fromto="0 0 0 -.019 0 0" size=".006" material="glow" mass=".001" contype="0" conaffinity="0"/>
          </body>
          <body name="{p}mouth_R" pos=".010 {FACE_Y + .0005:.4g} -.023" gravcomp="1">
            <joint name="{p}mouth_R" type="hinge" axis="0 1 0" range="-35 35" damping=".0015" armature=".0001"/>
            <geom name="{p}mouth_R" type="capsule" fromto="0 0 0 .019 0 0" size=".006" material="glow" mass=".001" contype="0" conaffinity="0"/>
          </body>
          <!-- two ears, left and right (the ear sites are where each ear hears from) -->
          <geom name="{p}ear_L" type="capsule" fromto="{-HEAD_R[0] + .001:.4g} 0 -.02 {-HEAD_R[0] + .001:.4g} 0 .02" size=".0105" material="{accent}" {DECOR}/>
          <geom name="{p}ear_R" type="capsule" fromto="{HEAD_R[0] - .001:.4g} 0 -.02 {HEAD_R[0] - .001:.4g} 0 .02" size=".0105" material="{accent}" {DECOR}/>
          <site name="{p}ear_L" pos="{-HEAD_R[0] - .011:.4g} 0 0" size=".006" group="4"/>
          <site name="{p}ear_R" pos="{HEAD_R[0] + .011:.4g} 0 0" size=".006" group="4"/>
          <!-- the eye: one camera between the drawn eyes; render 128 px -> periphery 64 px + fovea (centre 32 px) -->
          <camera name="{p}eye" pos="0 {FACE_Y + .002:.4g} {EYE_UP}" xyaxes="1 0 0 0 0 1" fovy="{EYE_FOVY}"/>
        </body>
      </body>
    </body>"""


def robot_actuators(p):
    sh, el = math.radians(SH_DEG), math.radians(EL_DEG)
    t0, t1 = (math.radians(a) for a in TILT_DEG)
    pan = math.radians(PAN_DEG)
    return f"""
    <!-- {p[:-1]}: position servos (the servo law sets target = measured angle + step each tick). Measured on this file 2026-09-24,
         from rest: an arm step of 0.27 or 0.09 rad is at 103% after one 150 ms tick (peak overshoot 4-6%); a gaze step of 0.2 or
         0.07 rad at 101-102% (overshoot under 2%); the face at 97% (overshoot 1.5%). The torque limits
         are the declared limits the charge drain is capped by and weakness scales ((0.3 + 0.7h) x forcerange);
         declared stall force at the mitten: 1.2 N m / 0.41 m = 2.9 N (shoulder), 0.8 N m / 0.19 m = 4.2 N (elbow) -->
    <position name="{p}shoulder" joint="{p}shoulder" kp="15" kv=".6" ctrlrange="{-sh:.4f} {sh:.4f}" forcelimited="true" forcerange="{-SH_TAU} {SH_TAU}"/>
    <position name="{p}elbow" joint="{p}elbow" kp="10" kv=".3" ctrlrange="{-el:.4f} {el:.4f}" forcelimited="true" forcerange="{-EL_TAU} {EL_TAU}"/>
    <adhesion name="{p}grip" body="{p}hand" ctrlrange="0 1" gain="3"/>
    <position name="{p}pan" joint="{p}pan" kp="1.5" kv=".02" ctrlrange="{-pan:.4f} {pan:.4f}"/>
    <position name="{p}tilt" joint="{p}tilt" kp="1.5" kv=".02" ctrlrange="{t0:.4f} {t1:.4f}"/>
    <position name="{p}face" joint="{p}mouth" kp=".12" kv=".005" ctrlrange="-.6 .6"/>"""


def robot_sensors(p):
    return f"""
    <touch name="{p}touch" site="{p}mitten_touch"/>
    <touch name="{p}upper_arm_touch" site="{p}upper_arm_touch"/><touch name="{p}forearm_touch" site="{p}forearm_touch"/>
    <jointpos name="{p}shoulder_pos" joint="{p}shoulder"/><jointpos name="{p}elbow_pos" joint="{p}elbow"/>
    <jointvel name="{p}shoulder_vel" joint="{p}shoulder"/><jointvel name="{p}elbow_vel" joint="{p}elbow"/>
    <jointpos name="{p}pan_pos" joint="{p}pan"/><jointpos name="{p}tilt_pos" joint="{p}tilt"/>
    <jointvel name="{p}pan_vel" joint="{p}pan"/><jointvel name="{p}tilt_vel" joint="{p}tilt"/>
    <actuatorfrc name="{p}shoulder_torque" actuator="{p}shoulder"/><actuatorfrc name="{p}elbow_torque" actuator="{p}elbow"/>"""


def objects():
    out = []
    for k, (x, y, yaw) in OBJ_POS.items():
        if k == "ball":
            g = f'<geom name="ball" type="sphere" size=".045" mass="{OBJ_MASS}" material="ball" friction=".8 .005 .0001" {STATIC}/>'
            z = .045
        elif k == "cube":
            g = f'<geom name="cube" type="box" size=".033 .033 .033" mass="{OBJ_MASS}" material="cube" friction=".8 .005 .0001" {STATIC}/>'
            z = .033
        elif k == "duck":   # a toy duck: body, flat base and head collide; beak, eyes and tail are seen only
            g = (f'<geom name="duck" type="ellipsoid" size=".046 .034 .03" mass=".043" material="duck" friction=".8 .005 .0001" {STATIC}/>'
                 f'<geom name="duck_base" type="cylinder" pos="0 0 -.0255" size=".02 .0045" mass=".004" material="duck" friction=".8 .005 .0001" {STATIC}/>'
                 f'<geom name="duck_head" type="sphere" pos=".026 0 .036" size=".022" mass=".013" material="duck" friction=".8 .005 .0001" {STATIC}/>'
                 f'<geom type="ellipsoid" pos=".05 0 .033" size=".014 .012 .0045" material="beak" {DECOR}/>'
                 f'<geom type="sphere" pos=".041 .0125 .044" size=".0038" material="black" {DECOR}/>'
                 f'<geom type="sphere" pos=".041 -.0125 .044" size=".0038" material="black" {DECOR}/>'
                 f'<geom type="ellipsoid" pos="-.043 0 .017" euler="0 -35 0" size=".016 .014 .007" material="duck" {DECOR}/>')
            z = .03
        else:               # a cup: the solid collides; the rim ring, the dark inside and the handle are seen only
            g = (f'<geom name="cup" type="cylinder" size=".036 .036" mass="{OBJ_MASS}" material="cup" friction=".8 .005 .0001" {STATIC}/>'
                 f'<geom type="cylinder" pos="0 0 .0362" size=".031 .0006" material="cup_in" {DECOR}/>'
                 f'<geom type="mesh" mesh="cup_rim" pos="0 0 .0352" material="cup" {DECOR}/>'
                 f'<geom type="mesh" mesh="cup_handle" pos=".04 0 -.002" euler="90 0 0" material="cup" {DECOR}/>')
            z = .036
        out.append(f'    <body name="{k}" pos="{x} {y} {z + .0005:.4g}" euler="0 0 {yaw}"><freejoint name="{k}"/>{g}</body>')
    return "\n".join(out)


def room():
    W, H = 2.2, 2.7                       # a 4.4 x 4.4 m room, 2.7 m high
    zc = FLOOR_Z + H / 2
    walls = []
    for nm, pos, size in (("wall_back", (0, W, zc), (W, .05, H / 2)), ("wall_front", (0, -W, zc), (W, .05, H / 2)),
                          ("wall_left", (-W, 0, zc), (.05, W, H / 2)), ("wall_right", (W, 0, zc), (.05, W, H / 2))):
        walls.append(f'<geom name="{nm}" type="box" pos="{f(*pos)}" size="{f(*size)}" material="wall" {ROOM}/>')
        sk = (size[0] - .04, .012, .05) if size[1] == .05 else (.012, size[1] - .04, .05)
        sp = (pos[0] - np.sign(pos[0]) * .055, pos[1] - np.sign(pos[1]) * .055, FLOOR_Z + .05)
        walls.append(f'<geom type="box" pos="{f(*sp)}" size="{f(*sk)}" material="trim" {ROOM}/>')
    # a window on the left wall (seen by the overview camera and by the eyes when they turn left)
    wx = -W + .06
    win = [f'<geom name="window" type="plane" pos="{wx} .15 .62" euler="0 90 0" size=".58 .62 .01" material="sky" {ROOM}/>']
    for yy, zz, sy, sz in ((.15, 1.22, .66, .035), (.15, .02, .66, .035), (-.49, .62, .035, .62), (.79, .62, .035, .62),
                           (.15, .62, .018, .58), (.15, .62, .62, .018)):
        win.append(f'<geom type="box" pos="{wx + .012:.4g} {yy} {zz}" size=".02 {sy} {sz}" material="trim" {ROOM}/>')
    win.append(f'<geom type="box" pos="{wx + .05:.4g} .15 -.01" size=".06 .72 .018" material="trim" {ROOM}/>')
    # a quiet print on the wall behind the parent, towards the window corner (soft, low-saturation, not an object colour)
    ay, ax, az = W - .06, -1.5, .40
    art = [f'<geom type="box" pos="{ax} {ay} {az}" size=".34 .012 .24" material="wood" {ROOM}/>',
           f'<geom type="box" pos="{ax} {ay - .008:.4g} {az}" size=".315 .012 .215" material="art_c" {ROOM}/>',
           f'<geom type="cylinder" pos="{ax - .10:.4g} {ay - .0245:.4g} {az + .04:.4g}" euler="90 0 0" size=".11 .004" material="art_a" {ROOM}/>',
           f'<geom type="cylinder" pos="{ax + .10:.4g} {ay - .027:.4g} {az - .055:.4g}" euler="90 0 0" size=".088 .004" material="art_b" {ROOM}/>']
    # a low sideboard behind the child, and a rug
    side = [f'<geom type="box" pos=".9 {-W + .25:.4g} {FLOOR_Z + .3:.4g}" size=".6 .2 .3" material="trim" {ROOM}/>',
            f'<geom type="box" pos=".9 {-W + .25:.4g} {FLOOR_Z + .605:.4g}" size=".62 .21 .012" material="wood" {ROOM}/>']
    return "\n    ".join(
        [f'<geom name="floor" type="plane" pos="0 0 {FLOOR_Z}" size="{W} {W} .1" material="floor" {STATIC} group="2"/>',
         f'<geom name="ceiling" type="box" pos="0 0 {FLOOR_Z + H + .025:.4g}" size="{W} {W} .025" material="ceiling" {ROOM}/>',
         f'<geom name="rug" type="box" pos="0 0 {FLOOR_Z + .003:.4g}" size="1.05 1.25 .003" material="rug" {ROOM}/>']
        + walls + win + art + side)


def table():
    legs = "\n    ".join(f'<geom type="cylinder" pos="{sx * .35:.4g} {sy * .26:.4g} {(FLOOR_Z - .04) / 2:.4g}" size=".026 {(-.04 - FLOOR_Z) / 2:.4g}" material="wood" {DECOR}/>'
                        for sx in (-1, 1) for sy in (-1, 1))
    rz = RIM_H / 2
    rim = "\n    ".join([
        f'<geom name="rim_far" type="box" pos="0 {TABLE_HY + RIM_T / 2 - .003:.4g} {rz}" size="{TABLE_HX + RIM_T:.4g} {RIM_T / 2} {rz}" material="rim" {STATIC}/>',
        f'<geom name="rim_near" type="box" pos="0 {-(TABLE_HY + RIM_T / 2 - .003):.4g} {rz}" size="{TABLE_HX + RIM_T:.4g} {RIM_T / 2} {rz}" material="rim" {STATIC}/>',
        f'<geom name="rim_right" type="box" pos="{TABLE_HX + RIM_T / 2 - .003:.4g} 0 {rz}" size="{RIM_T / 2} {TABLE_HY + RIM_T:.4g} {rz}" material="rim" {STATIC}/>',
        f'<geom name="rim_left" type="box" pos="{-(TABLE_HX + RIM_T / 2 - .003):.4g} 0 {rz}" size="{RIM_T / 2} {TABLE_HY + RIM_T:.4g} {rz}" material="rim" {STATIC}/>'])
    return f"""<geom name="table" type="box" pos="0 0 {-TABLE_HZ}" size="{TABLE_HX} {TABLE_HY} {TABLE_HZ}" material="table" friction=".6 .005 .0001" {STATIC}/>
    {rim}
    <geom type="box" pos="0 0 -.055" size="{TABLE_HX - .04:.4g} {TABLE_HY - .04:.4g} .015" material="wood" {DECOR}/>
    {legs}
    <!-- the charge pad: a flat magenta disc (no object shares its colour); it charges while the child's mitten is over it -->
    <geom name="pad" type="cylinder" pos="{PAD_XY[0]} {PAD_XY[1]} .0006" size="{PAD_R} .0006" material="pad" contype="0" conaffinity="0"/>
    <geom type="cylinder" pos="{PAD_XY[0]} {PAD_XY[1]} .0004" size="{PAD_R + .006:.4g} .0004" material="trim" contype="0" conaffinity="0"/>"""


def floor_texture(path):
    """Light oak planks, 1024 x 1024, deterministic."""
    from PIL import Image
    rng = np.random.default_rng(7)
    N, planks = 1024, 8
    img = np.zeros((N, N, 3), np.float32)
    base = np.array([.80, .66, .50])
    yy, xx = np.mgrid[0:N, 0:N].astype(np.float32)
    w = N // planks
    for i in range(planks):
        off = rng.integers(0, N)
        for seg in range(3):
            tone = base * rng.uniform(.9, 1.06) + rng.uniform(-.02, .02, 3)
            y0, y1 = (off + seg * N // 3) % N, (off + (seg + 1) * N // 3) % N
            rows = (yy >= y0) & (yy < y1) if y0 < y1 else (yy >= y0) | (yy < y1)
            m = rows & (xx >= i * w) & (xx < (i + 1) * w)
            grain = .035 * np.sin(xx * .9 + 6 * np.sin(yy * .011 + i) + rng.uniform(0, 6)) + .02 * np.sin(xx * 3.1 + yy * .05)
            img[m] = (tone[None, :] * (1 + grain[m, None]))
            img[m & (((yy - y0) % N) < 2)] *= .8
        img[:, i * w:i * w + 2] *= .78
    Image.fromarray(np.clip(img * 255, 0, 255).astype(np.uint8)).save(path)


def sky_texture(path):
    """A soft daylight sky for the window: pale blue above, lighter at the horizon, a faint far treeline."""
    from PIL import Image, ImageFilter
    H, W = 256, 256
    t = np.linspace(0, 1, H)[:, None, None]
    top, hor = np.array([.66, .80, .95]), np.array([.93, .96, .98])
    img = np.broadcast_to(top * (1 - t) + hor * t, (H, W, 3)).copy()
    rng = np.random.default_rng(3)
    x = np.arange(W)
    line = (H * .78 + 10 * np.sin(x / 23 + 1) + 6 * np.sin(x / 9 + rng.uniform(0, 6)) + 3 * np.sin(x / 4)).astype(int)
    for i in range(W):
        img[line[i]:, i] = np.array([.72, .80, .74])
    im = Image.fromarray((img * 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(2))
    im = im.transpose(Image.ROTATE_90)   # the window plane is turned 90 deg about y; this puts the sky up and the treeline down
    im.save(path)


def build():
    (HERE / "textures").mkdir(exist_ok=True)
    floor_texture(HERE / "textures" / "floor_oak.png")
    sky_texture(HERE / "textures" / "window_sky.png")
    mats = "\n    ".join(f'<material name="{k}" ' + " ".join(f'{a}="{v}"' for a, v in d.items()) + "/>" for k, d in MAT.items())
    overview_pos, overview_at = (1.30, -.70, .60), (-.05, .03, .17)     # three-quarter, behind the child's right shoulder
    side_pos, side_at = (1.62, 0, .44), (0, 0, .11)                    # profile, the window between them
    hero_pos, hero_at = (1.38, -.04, .40), (0, 0, .17)                 # a portrait of the two (they turn to it)
    shoulder_pos, shoulder_at = (.70, -1.10, .80), (-.05, .15, .19)    # over the child's shoulder: its head, the table, the parent's face
    xml = f"""<!-- GENERATED by body/sim/make_highchair.py: edit the numbers there and re-run; do not hand-edit.
     The high chair world (docs/SIM_DESIGN.md 5.1 and 5.8): a child robot and a parent robot with identical bodies across a
     pale grey rimmed table, four objects, a magenta charge pad, in a quiet room. MuJoCo 3.9; units m, kg, s; angles in degrees
     in this file (joint ranges, eulers), radians in ctrlrange. The table's top surface is z = 0; +y points from the child to the parent.
     Collision: everything solid is contype 1 / conaffinity 1: the table, rim, floor, objects, both arms (links and mittens) and the
     shoulder hubs and booms, so the two robots' arms touch each other and everything on the table; an arm's own upper arm and
     forearm are parent and child bodies and never pair. Heads, torsos, chairs and the room are seen only.
     Pain: F_pain = {F_PAIN_K} x the mitten's declared stall force {MITTEN_STALL_N:.2f} N = {F_PAIN:.1f} N (the custom numerics below). -->
<mujoco model="highchair">
  <compiler angle="degree" texturedir="textures"/>
  <option timestep="0.002" integrator="implicitfast"/>
  <statistic center="0 0 .1" extent="1.2"/>
  <visual>
    <global offwidth="1920" offheight="1080" fovy="45" azimuth="120" elevation="-25"/>
    <quality shadowsize="8192" offsamples="8"/>
    <headlight ambient=".30 .30 .31" diffuse=".21 .21 .21" specular=".03 .03 .03"/>
    <map znear=".004" zfar="30" shadowclip="1.6" shadowscale=".7"/>
  </visual>
  <asset>
    <texture name="floor_oak" type="2d" file="floor_oak.png"/>
    <texture name="window_sky" type="2d" file="window_sky.png"/>
    <texture name="skybox" type="skybox" builtin="gradient" rgb1=".92 .94 .97" rgb2=".75 .78 .82" width="256" height="256"/>
    {mats}
    <mesh name="head" builtin="supersphere" params="48 .42 .42" scale="{f(*HEAD_R)}"/>
    <mesh name="bezel" builtin="supersphere" params="32 .3 .3" scale=".075 .012 .059"/>
    <mesh name="screen" builtin="supersphere" params="32 .3 .3" scale=".068 .0085 .052"/>
    <mesh name="collar" builtin="supersphere" params="32 1 .35" scale=".043 .043 .013"/>
    <mesh name="seat" builtin="supersphere" params="24 .2 .2" scale=".15 .13 .016"/>
    <mesh name="chair_back" builtin="supersphere" params="24 .25 .25" scale=".14 .014 .15"/>
    <mesh name="cup_rim" builtin="supertorus" params="40 .12 1 1" scale=".0325 .0325 .0325"/>
    <mesh name="cup_handle" builtin="supertorus" params="30 .26 1 1" scale=".0165 .0165 .0165"/>
  </asset>
  <custom>
    <!-- the pain law's numbers (SIM_DESIGN 5.5, 7): the mitten's declared stall force = the elbow's torque limit / the forearm -->
    <numeric name="mitten_stall_force" data="{MITTEN_STALL_N:.4f}"/>
    <numeric name="f_pain_k" data="{F_PAIN_K}"/>
    <numeric name="f_pain" data="{F_PAIN:.4f}"/>
  </custom>
  <worldbody>
    <light name="key" type="spot" pos="{f(*KEY_POS)}" dir="{f(*(-np.array(KEY_POS) / np.linalg.norm(KEY_POS)))}" cutoff="32" exponent="1" castshadow="true" bulbradius=".12" diffuse=".52 .50 .47" specular=".22 .22 .22"/>
    <light name="top" type="spot" pos="0 0 1.8" dir="0 0 -1" cutoff="50" exponent="1" castshadow="false" diffuse=".16 .16 .16" specular=".05 .05 .05"/>
    <light name="fill" type="directional" pos="1.5 -1.2 1.5" dir="-1 .45 -.75" castshadow="false" diffuse=".46 .46 .48" specular="0 0 0"/>
    <camera name="overview" pos="{f(*overview_pos)}" xyaxes="{lookat_xyaxes(overview_pos, overview_at)}" fovy="34"/>
    <camera name="side" pos="{f(*side_pos)}" xyaxes="{lookat_xyaxes(side_pos, side_at)}" fovy="34"/>
    <camera name="hero" pos="{f(*hero_pos)}" xyaxes="{lookat_xyaxes(hero_pos, hero_at)}" fovy="36"/>
    <camera name="shoulder" pos="{f(*shoulder_pos)}" xyaxes="{lookat_xyaxes(shoulder_pos, shoulder_at)}" fovy="28"/>

    <!-- the room -->
    {room()}

    <!-- the table: 0.8 x 0.64 m, pale grey, a 3 cm rim -->
    {table()}

    <!-- the four objects: 60 g each, at the sizes t11 measured ("1.3x"), saturated colours -->
{objects()}
{robot("child_", "accent_child", +1)}
{robot("parent_", "accent_parent", -1)}
  </worldbody>

  <contact>
    <!-- the shoulder's hub and boom (on the fixed base) are solid for objects, but the upper arm turns inside the hub -->
    <exclude body1="child_base" body2="child_upper_arm"/>
    <exclude body1="parent_base" body2="parent_upper_arm"/>
  </contact>

  <equality>
    <!-- each mouth is one curvature: the right half mirrors the left (face ctrl > 0 smiles, < 0 frowns) -->
    <joint name="child_mouth_mirror" joint1="child_mouth_R" joint2="child_mouth" polycoef="0 -1 0 0 0" solref=".005 1"/>
    <joint name="parent_mouth_mirror" joint1="parent_mouth_R" joint2="parent_mouth" polycoef="0 -1 0 0 0" solref=".005 1"/>
    <!-- the parent's guiding hand on the child's forearm: a soft weld at the forearm's hold point (site child_forearm_hold,
         {GUIDE_AT} m from the elbow), off until the teacher guides. torquescale 0: it holds the point, never the forearm's angle.
         Switch it on only with make_highchair.guide(m, d), which writes the offset from the current pose (the compiled
         offset, from qpos0, is meaningless). Its constraint force is not a contact, so it never reads as touch or pain. -->
    <weld name="parent_guides_child" body1="parent_hand" body2="child_forearm" anchor="0 {GUIDE_AT} 0" torquescale="0" active="false" solref=".05 1"/>
  </equality>

  <actuator>{robot_actuators("child_")}
{robot_actuators("parent_")}
  </actuator>

  <sensor>{robot_sensors("child_")}
{robot_sensors("parent_")}
  </sensor>

  <keyframe/>
</mujoco>
"""
    out = HERE / "highchair.xml"
    out.write_text(xml.replace("<keyframe/>", ""))
    out.write_text(xml.replace("<keyframe/>", f"<keyframe>\n    {keyframe(out)}\n  </keyframe>"))
    return out


if __name__ == "__main__":
    print(build())
