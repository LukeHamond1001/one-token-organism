"""Loading the G1 living room and driving its kinematic parts (docs/SIM_DESIGN.md sections 3.1, 3.4, 5 and 15; the G1
amendment of 2026-09-24; from the G1 prototype of the same day). The world the body lives in is body/sim/world.py, built on
this scene; the parent's own motion (its plans, holds and caps) is the parent lane's (W2), which drives it through the
methods here.

THE G1 IS STOCK. g1room.xml includes body/sim/assets/unitree_g1/g1_with_hands.xml unchanged. This file adds the senses at
load time through MjSpec, as cameras and sites only (no geom, no mass, no joint, no actuator), where the real G1's sensors are:
  - the EYES: a stereo pair at the head's RealSense D435. Its pose is the Unitree URDF's d435_joint in torso_link's frame
    (unitree_ros, robots/g1_description/g1_29dof_with_hand_rev_1_0.urdf: xyz 0.0576235 0.01753 0.42987, pitch 0.8307767 rad,
    47.6 deg down), which by the D435's layout is its left imager (the depth origin); the right imager is 50 mm to the
    camera's right. Each eye is a colour pinhole camera (the real D435's two imagers are monochrome infrared with a colour
    camera 15 mm left of the left imager: a colour stereo pair is a sim choice, flagged). Field: 58 deg vertical and about
    88 deg horizontal (the D435's depth field is 87 x 58 deg). Each pinhole sits on its optical axis just outside the head
    shell (the lens behind the face's window), see EYE_PUSH. The eyes' render and their fovea are body/sim/eyes.py (W3).
  - the EARS: two sites on the head's left and right sides (the real G1 has a 4-microphone array whose positions are not in
    the model: flagged).
  - the VESTIBULE: the G1's own IMU sites and sensors (imu_in_torso, which moves with the head, and imu_in_pelvis), already
    in the model, with the model's own declared noise and ranges.
  - JOINT SENSE and TOUCH: read by body/sim/world.py from the joints, the actuators and the contacts.
The model file's own directional light (a Menagerie scene light, not part of the robot) is switched off at load. Nothing
here sets the G1's servo gains: the body's servo law does, in body/sim/world.py. The parent's face is the graded face, the
room's face of human proportions (parent_kin.py; the W1 verifier's third round): a scalar expression is drawn through
parent_kin.scalar_to_params, a dict of face parameters (the parent's feelings, parent_feel.py) directly; her face's moving meshes
(the lids, the irises) are posed through their compiled offsets (parent_face.py; the W1 verifier's fourth round)."""
import math
import sys
from pathlib import Path

import pathlib
import mujoco
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import parent_kin as kin  # noqa: E402

XML = HERE / "g1room.xml"
XML_B = HERE / "g1room_b.xml"                 # A133: the changed room (make_g1room.py --layout=b): the furniture moved, the mat and toys kept
ROOMS = {"a": XML, "b": XML_B}                # the runner's --room names
XML_BODY_B = HERE / "g1room_body_b.xml"       # A134: the room of birth with the changed body (make_g1room.py --body=b)
XML_DOOR = HERE / "g1room_door.xml"           # the door stage (make_g1room.py --door=1): room a with a door leaf and a second room behind it
SCENES = {("a", "a"): XML, ("b", "a"): XML_B, ("a", "b"): XML_BODY_B, ("door", "a"): XML_DOOR}   # (room, body) -> the scene; the runner's --room and --body


def scene_path(room="a", body="a"):
    """the scene for a room layout (A133) and a body (A134); the two changes are not combined (no such scene is written)"""
    try:
        return SCENES[(room, body)]
    except KeyError:
        raise ValueError(f"no scene for room {room!r} with body {body!r}: the room and the body change one at a time")
G1_FILE = HERE / "assets" / "unitree_g1" / "g1_with_hands.xml"

# ---------------------------------------------------------------- the G1's parts
LEGS = ["hip_pitch", "hip_roll", "hip_yaw", "knee", "ankle_pitch", "ankle_roll"]
ARM = ["shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow", "wrist_roll", "wrist_pitch", "wrist_yaw"]
HAND = ["hand_thumb_0", "hand_thumb_1", "hand_thumb_2", "hand_index_0", "hand_index_1", "hand_middle_0", "hand_middle_1"]
WAIST = ["waist_yaw", "waist_roll", "waist_pitch"]
# the body's joint effectors (SIM_DESIGN.md 3.5, for the G1: its waist in place of the neck and trunk, each arm, each Dex3
# hand, each leg; one effector per limb, the basal ganglia's parallel loops), in the order the anatomy declares them; joint 0
# of each is its act's most significant digit (body/core/anatomy.py, Effector)
EFFECTORS = (("waist", [f"{j}_joint" for j in WAIST]),
             ("arm_l", [f"left_{j}_joint" for j in ARM]), ("arm_r", [f"right_{j}_joint" for j in ARM]),
             ("hand_l", [f"left_{j}_joint" for j in HAND]), ("hand_r", [f"right_{j}_joint" for j in HAND]),
             ("leg_l", [f"left_{j}_joint" for j in LEGS]), ("leg_r", [f"right_{j}_joint" for j in LEGS]))

# ---------------------------------------------------------------- the senses (torso_link frame)
D435_POS = np.array([0.0576235, 0.01753, 0.42987])
D435_PITCH = 0.8307767239493009
STEREO_BASELINE = 0.050
EYE_PUSH = 0.003            # the D435 origin lies on the head shell (ray test: within 0.1 mm); each pinhole 3 mm in front of it
EYE_FOVY = 58.0             # degrees, vertical
EYE_W, EYE_H = 336, 192     # the native render per grey eye (A42): 3 px a degree at the centre; horizontal field 2 atan(336/192 tan 29
                            # deg) = 88.3 deg, as the D435's imagers' depth field (A38)
COL_W, COL_H = 238, 134     # the colour camera's render (A38; 3.4): 69.4 x 42.5 deg at about 3.4 px a degree
COL_FOVY = 42.5             # degrees, vertical (the D435's colour camera, OV2740: 69.4 x 42.5 deg)
COL_OFFSET = 0.015          # the colour camera 15 mm beside the left imager, on its outer side (RealSense's documentation gives 15 mm
                            # between the two centre-lines; which side and its exact axis are read from the datasheet before birth,
                            # C45; ours until then)
EAR_Y = 0.079               # the ear sites on the head's sides (its widest point is 0.078 m from the midline)
EAR_XZ = (0.005, 0.395)


def _ry(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def eye_frames():
    """Each eye's (pos, R) in torso_link's frame, the two grey imagers L and R and the colour camera C; R's columns are the MuJoCo
    camera's x (right), y (up), z (backward)."""
    Rd = _ry(D435_PITCH)                               # the d435 frame: x forward (optical axis), y left, z up
    Rcam = Rd @ np.column_stack([[0, -1, 0], [0, 0, 1], [-1, 0, 0]])
    fwd = Rd[:, 0]
    out = {}
    for sd, dy in (("L", 0.0), ("R", -STEREO_BASELINE), ("C", COL_OFFSET)):
        p = D435_POS + Rd @ np.array([0, dy, 0]) + fwd * EYE_PUSH
        out[sd] = (p, Rcam)
    return out


# THE MOTORS AS THE REAL ONES (A79, the lead's decision at S5a; A39, C46): each body joint's reflected rotor inertia (armature) and its
# friction loss, from Menagerie's MJX G1 (body/sim/assets/unitree_g1/g1_mjx.xml, the variant MuJoCo Playground transferred to the real
# robot: Zakka et al. 2025), in place of the file's one default for every joint (armature 0.01, friction loss 0.3: Menagerie's generic
# <default> in g1_with_hands.xml). Set on the compiled model, so the stock file stays byte for byte its commit. The Dex3's 14 finger
# joints keep the file's values: no source gives theirs yet (C46).
_A = dict(hip_big=0.025101925, hip=0.01017752004, ankle=0.00721945, arm=0.003609725, wrist=0.00425)
MOTORS = {}
for _sd in ("left", "right"):
    MOTORS.update({f"{_sd}_hip_pitch_joint": (_A["hip"], 0.1), f"{_sd}_hip_roll_joint": (_A["hip_big"], 0.1),
                   f"{_sd}_hip_yaw_joint": (_A["hip"], 0.1), f"{_sd}_knee_joint": (_A["hip_big"], 0.1),
                   f"{_sd}_ankle_pitch_joint": (_A["ankle"], 0.1), f"{_sd}_ankle_roll_joint": (_A["ankle"], 0.1),
                   f"{_sd}_shoulder_pitch_joint": (_A["arm"], 0.1), f"{_sd}_shoulder_roll_joint": (_A["arm"], 0.1),
                   f"{_sd}_shoulder_yaw_joint": (_A["arm"], 0.1), f"{_sd}_elbow_joint": (_A["arm"], 0.1),
                   f"{_sd}_wrist_roll_joint": (_A["arm"], 0.1), f"{_sd}_wrist_pitch_joint": (_A["wrist"], 0.1),
                   f"{_sd}_wrist_yaw_joint": (_A["wrist"], 0.1)})
MOTORS.update({"waist_yaw_joint": (_A["hip"], 0.1), "waist_roll_joint": (_A["ankle"], 0.1), "waist_pitch_joint": (_A["ankle"], 0.1)})

# THE DEX3-1 AS UNITREE SPECIFIES IT (A80, the lead's decision at S5a; Unitree's Dex3-1 page, read 2026-09-25): seven joints a hand,
# six driven by the F-1515-108 micro brushless joint (1:108; ideal 0.76 N m; 0.49 N m turning its own way; 1.37 N m held against a
# load turning it back; 23 rad/s) and the thumb's rotation (thumb_0, the one the file gives its larger limit) by the geared F-1515-214
# (1:214; 1.498; 0.86; 3.1; 11 rad/s). The file gave the six 1.4 N m, their holding figure, as their driving strength (about three
# times the real), and every joint armature 0.01. Here: the motor's ideal torque is its force range; its gear's friction, the
# ideal less the driving figure (0.27 and 0.638 N m), is the joint's friction loss, so a finger drives with 0.49 (0.86) as Unitree
# measured; the holding figure is the gear's load the joint takes before it back-drives: its pain line (A37, A73). The rotor's
# inertia is not published: ours, from the motor's size (a 1515 inner rotor about 7 mm across and 12 mm long, steel and magnet at
# 7.5 g/cm3: 2.1e-8 kg m2), reflected through its reduction and half again for the gearbox's first stage: 3.7e-4 and 1.4e-3 kg m2,
# until the datasheet gives it (C46). Its speed limits are the torque-speed envelope's, not yet modelled (A39).
DEX3_108 = dict(ideal=0.76, drive=0.49, hold=1.37, speed=23.0, ratio=108)
DEX3_214 = dict(ideal=1.498, drive=0.86, hold=3.1, speed=11.0, ratio=214)
DEX3_ROTOR = 2.1e-8 * 1.5                                              # kg m2 at the motor, the gearbox's first stage included (ours)
DEX3 = {}
for _sd in ("left", "right"):
    for _j in ("thumb_0", "thumb_1", "thumb_2", "index_0", "index_1", "middle_0", "middle_1"):
        DEX3[f"{_sd}_hand_{_j}_joint"] = DEX3_214 if _j == "thumb_0" else DEX3_108


# THE DOOR STAGE (the owner's word, 2026-10-04: generalization is a door in its one room opening onto a new room it enters itself):
# a leaf closes the doorway the hall had, and where the hall was a SECOND ROOM, 3.2 m by 3.6 m (x 2.7 to 5.9, y -3.3 to 0.3): its
# floor blue-green, its walls yellow, a round red rug, a low bench, an arch to pass under. Fixed geoms in a body added LAST, after
# the runner's extra toys (every body, geom, light and material of the saved life keeps its index: the saved state and model fields
# load as this model's first rows; world.load_state). The hall's three walls stay where the XML put them and the world moves them
# out to the second room's bounds (G1World._door_setup). An environment's change, disclosed. Sizes as make_g1room's room (ROOM_X 2.6,
# ROOM_H 2.6, the door y -1.95 to -1.05, 2.05 high); WORLD contact flags as make_g1room.WORLD
ROOM2 = dict(x0=2.70, x1=5.90, yc=-1.5, hy=1.8, door_h=2.05)
_W, _Hh = 2.6, 2.6


def _add_room2(spec):
    r = ROOM2; yc, X0, X1, hy, dh = r["yc"], r["x0"], r["x1"], r["hy"], r["door_h"]
    B = mujoco.mjtGeom.mjGEOM_BOX
    b = spec.worldbody.add_body(name="room2")
    def geom(name, type_, pos, size, material, world=True):
        g = b.add_geom(name=name, type=type_, pos=list(map(float, pos)), size=list(map(float, size)), material=material)
        if world:
            g.contype, g.conaffinity, g.priority = 17, 0, 2
        else:
            g.contype, g.conaffinity = 0, 0
        return g
    geom("door_leaf", B, (_W + .05, yc, dh / 2), (.02, .45 - .005, dh / 2 - .005), "door")
    geom("room2_floor", B, ((X0 + X1) / 2, yc, 0.0), ((X1 - X0) / 2, hy, .001), "floor2", world=False)
    geom("room2_ceiling", B, ((X0 + X1) / 2, yc, _Hh + .03), ((X1 - X0) / 2, hy, .03), "ceiling", world=False)
    geom("room2_rug", mujoco.mjtGeom.mjGEOM_CYLINDER, (_W + 1.5, yc, .003), (.7, .003, 0), "rug2", world=False)
    geom("room2_bench", B, (X1 - .25, yc + .9, .20), (.20, .60, .20), "bench2")
    for k_, yy in (("a", yc - 1.25), ("b", yc - .55)):                      # the arch: two posts and a beam, 0.7 m apart, by the far wall
        geom(f"room2_arch_{k_}", B, (X1 - .5, yy, .55), (.05, .05, .55), "arch2")
    geom("room2_arch_top", B, (X1 - .5, yc - .9, 1.15), (.05, .40, .05), "arch2")
    lt = b.add_light(name="room2_lamp", pos=[_W + 1.7, yc, 2.4], dir=[0, 0, -1])
    lt.diffuse[:] = (.55, .52, .46); lt.specular[:] = (0, 0, 0); lt.castshadow = False


# D2 (2026-10-06, the owner's word: a human home with other children showing examples): THE SIBLING, 'bo', a child-sized figure of
# capsules (a metre tall) the world moves each tick (G1World.sibling_tick): it walks a loop of the first room in the child's view,
# an example of walking before its eyes, the parent naming it. Visual only for now (no contact: it never knocks the child), in a
# body added last (the saved life's rows hold, as the second room's). Its parts' places and sizes: ours
SIB = dict(x0=-1.7, x1=1.7, y=1.25, speed=0.45, torso_z=0.62, torso_h=0.17, head_z=0.93, head_r=0.09, hip_z=0.44, leg_h=0.21, sho_z=0.78, arm_h=0.15, swing=0.45, rgba=(0.85, 0.55, 0.30, 1.0))


def _add_sibling(spec):
    b = spec.worldbody.add_body(name="sibling")
    def cap(name, half, r, pos):
        g = b.add_geom(name=name, type=mujoco.mjtGeom.mjGEOM_CAPSULE, size=[r, half, 0], pos=list(map(float, pos)))
        g.contype, g.conaffinity = 0, 0; g.rgba[:] = SIB["rgba"]
        return g
    cap("sib_torso", SIB["torso_h"], 0.09, (SIB["x0"], SIB["y"], SIB["torso_z"]))
    g = b.add_geom(name="sib_head", type=mujoco.mjtGeom.mjGEOM_SPHERE, size=[SIB["head_r"], 0, 0], pos=[SIB["x0"], SIB["y"], SIB["head_z"]])
    g.contype, g.conaffinity = 0, 0; g.rgba[:] = (0.93, 0.76, 0.62, 1.0)
    for sd, sg in (("l", 1), ("r", -1)):
        cap(f"sib_leg_{sd}", SIB["leg_h"], 0.045, (SIB["x0"], SIB["y"] + sg * 0.07, SIB["hip_z"] - SIB["leg_h"]))
        cap(f"sib_arm_{sd}", SIB["arm_h"], 0.035, (SIB["x0"], SIB["y"] + sg * 0.14, SIB["sho_z"] - SIB["arm_h"]))


def load_model(xml=XML, extra=None):
    """The world with the G1's senses added; extra(spec), if given, adds a test rig before compiling (instruments only,
    never the body)."""
    spec = mujoco.MjSpec.from_file(str(xml))
    torso = spec.body("torso_link")
    for sd, (p, R) in eye_frames().items():
        torso.add_camera(name=f"eye_{sd}", pos=p.tolist(), quat=kin.mjquat(R).tolist(), fovy=COL_FOVY if sd == "C" else EYE_FOVY)
    for sd, sg in (("L", 1), ("R", -1)):
        torso.add_site(name=f"ear_{sd}", pos=[EAR_XZ[0], sg * EAR_Y, EAR_XZ[1]], size=[.006, 0, 0], group=5)
    if extra is not None:
        extra(spec)
    if any(mt.name == "door" for mt in spec.materials):                    # THE DOOR STAGE's scene (make_g1room --door=1: its
        _add_room2(spec)                                                    # materials only): the second room's body, last of all
        _add_sibling(spec)                                                  # D2: the sibling figure, after it
    # A187 (2026-10-04, speed only): THE EYE'S OWN MESHES. Unitree's visual meshes carry 630,000 triangles (a finger link 30,000) and
    # the eyes drew them six times a tick (three views, each with the sun's shadow pass): 119 of a 436 ms tick on the day-78 copy.
    # Each visual geom (no contact, group 2) whose mesh has a decimated copy in assets_vis (tools/make_vis_meshes.py: 12% of the
    # faces, quadric decimation, kept in git) is drawn from the copy; the collision geoms keep Unitree's files, so the physics is
    # the same to the bit
    vis_dir = pathlib.Path(__file__).parent / "assets" / "unitree_g1" / "assets_vis"
    if vis_dir.is_dir():
        made = {}
        for g in spec.geoms:
            if g.type == mujoco.mjtGeom.mjGEOM_MESH and int(g.contype) == 0 and int(g.conaffinity) == 0 and g.meshname:
                src = spec.mesh(g.meshname)
                f = None if src is None or not src.file else vis_dir / pathlib.Path(src.file).name
                if f is not None and f.is_file():
                    if g.meshname not in made:
                        mv = spec.add_mesh(name=g.meshname + "_vis", file=str(f.resolve()))
                        mv.scale = src.scale; mv.refpos = src.refpos; mv.refquat = src.refquat
                        made[g.meshname] = mv.name
                    g.meshname = made[g.meshname]
    m = spec.compile()
    for j, (arm, fl) in MOTORS.items():                  # the motors as the real ones (A79): rotor inertia and friction per joint
        dof = m.jnt_dofadr[m.joint(j).id]
        m.dof_armature[dof] = arm
        m.dof_frictionloss[dof] = fl
    for j, mt in DEX3.items():                           # the Dex3's as Unitree specifies them (A80)
        jid = m.joint(j).id
        dof = m.jnt_dofadr[jid]
        m.jnt_actfrcrange[jid] = [-mt["ideal"], mt["ideal"]]
        m.dof_frictionloss[dof] = mt["ideal"] - mt["drive"]
        m.dof_armature[dof] = DEX3_ROTOR * mt["ratio"] ** 2
    for i in range(m.nlight):
        if m.light(i).name == "":                      # the Menagerie file's own scene light
            m.light_active[i] = 0
            m.light_castshadow[i] = 0
    return m


def g1_body_ids(m):
    """The G1's bodies: the pelvis and every body whose root is the pelvis."""
    root = m.body("pelvis").id
    return [b for b in range(m.nbody) if b == root or m.body_rootid[b] == root]


# the G1's birth pose (radians): on its back, arms a little out with the elbows a little bent, hips and knees a little
# flexed and turned out, hands open. (For the G1 the elbow is straight at about 1.28 and bent 73 deg at 0; hip pitch negative
# flexes the hip; the left hand's fingers close toward negative angles, the right's toward positive.)
BIRTH = dict(shoulder_pitch=0.0, shoulder_roll=0.35, shoulder_yaw=0.0, elbow=1.10, wrist_roll=0.0,
             hip_pitch=-0.35, hip_roll=0.12, hip_yaw=0.15, knee=0.55, ankle_pitch=-0.05)
BIRTH_XY = (-0.05, -0.62)       # the pelvis on the mat; the head toward -x, its left toward +y
BIRTH_SETTLE_S = 1.5            # settled under its servos holding the birth pose
# THE PARENT AT BIRTH: standing by the door where the maker stands her (make_g1room.py builds her body there at rest),
# her face drawn at its neutral expression, looking straight ahead, her hands at the Pose's default shape. The face's and the
# hands' geoms have no place of their own in the file (the file's are placeholders inside the head and the hands): only
# set_parent puts them where a face and hands are, so birth draws her (the W1 verifier's first finding: never drawn, her face had
# no mouth or brows and its lids and irises sat at their placeholders, and the born face template never fired on it).
PARENT_BIRTH = dict(pos=(1.9, -1.5, kin.HIP_Z), yaw=math.radians(150))


def born_parent():
    """the parent's pose at birth (kin.Pose: standing at PARENT_BIRTH, the neutral face, the default hands)"""
    return kin.Pose(PARENT_BIRTH["pos"], kin.rz(PARENT_BIRTH["yaw"]))


class Scene:
    """The loaded scene: the model (with the G1's senses), its data, and the drivers of what is posed from outside the physics
    (the parent's placement, face and hands; the welds; the G1's pose setters). The world's side only."""

    def __init__(self, xml=XML, extra=None):
        self.m = load_model(xml, extra)
        self.d = mujoco.MjData(self.m)
        m = self.m
        import parent_body as PB                       # her body: its joints, and the pose her planner makes written into them
        self.bmap = PB.BodyMap(m)
        self.gid = lambda n: m.geom(n).id
        self.face_ids = {n: self.gid(f"parent_{n}") for n in kin.FACE_GEOMS}     # her face's moving geoms (all of them: the room has them)
        # her moving meshes (the lids, the irises): MuJoCo stores a mesh about its own centre and axes and puts that offset in the
        # geom's pose, so a pose from parent_face (the mesh's authored frame) is composed with it
        self.mesh_offset = {}
        for n, g in self.face_ids.items():
            if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH:
                mid = m.geom_dataid[g]
                self.mesh_offset[n] = (m.mesh_pos[mid].copy(), m.mesh_quat[mid].copy())
        self.hand_ids = {sd: {n: self.gid(f"parent_{n}") for n in kin.hand_geoms(sd).keys()} for sd in ("L", "R")}
        for g in list(self.face_ids.values()) + [i for h in self.hand_ids.values() for i in h.values()]:
            m.geom_sameframe[g] = 0                    # moved at run time: MuJoCo's same-frame shortcut must be off for them
        self.g1_bodies = g1_body_ids(m)
        self.g1_set = set(self.g1_bodies)
        self.act = {m.actuator(i).name: i for i in range(m.nu)}
        self.g1_act = [i for i in range(m.nu) if not (m.actuator(i).name or "").startswith("parent_")]   # the G1's own servos
        self.pose = None
        self._hand_cache = {}

    # ---- the parent
    def set_parent(self, pose, body=True, face=True):
        """Draw the parent in `pose`: her body placed there at rest (her joints set to the pose, her velocities zero: a still, or
        an instrument's placement; unless body=False: her motion, body/sim/parent_motion.py, drives her body through the tick's
        physics steps and only her face and hands' shapes are drawn here), her face's moving geoms from its expression and gaze
        (unless face=False: drawn as they last were, when neither changed; a face costs about 24 ms to draw) and her hands'
        shapes."""
        m, d = self.m, self.d
        segs = kin.fk(pose)
        if body:
            b = self.bmap
            b.write_qpos(d.qpos, b.targets(segs))
            d.qvel[b.vadr] = 0.0
        hp, hR = segs["head"]
        gaze = None
        if pose.gaze is not None:
            gaze = {sd: hR.T @ (pose.gaze - (hp + hR @ kin.EYE_C[sd])) for sd in ("L", "R")}
        expr = pose.expr
        if not isinstance(expr, dict):
            expr = kin.scalar_to_params(expr)          # the old one-number expression, as graded parameters
        for n, (p, q, sz) in (kin.face_geoms_graded(expr, gaze).items() if face else ()):
            g = self.face_ids[n]
            if n in self.mesh_offset:                      # compose the mesh's compiled offset
                mp, mq = self.mesh_offset[n]
                R = np.zeros(9); mujoco.mju_quat2Mat(R, np.asarray(q, float))
                qq = np.zeros(4); mujoco.mju_mulQuat(qq, np.asarray(q, float), mq)
                m.geom_pos[g] = np.asarray(p) + R.reshape(3, 3) @ mp; m.geom_quat[g] = qq
                continue
            m.geom_pos[g] = p; m.geom_quat[g] = q
            if sz is not None:
                if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_CAPSULE:
                    m.geom_size[g, :2] = sz[:2]
                else:
                    m.geom_size[g] = sz
        for sd in ("L", "R"):
            h = pose.hand[sd]
            key = (sd, h["curl"], h["thumb"], h["index"])
            if key not in self._hand_cache:                # a hand's shape drawn once per shape (2.6 ms a hand otherwise)
                if len(self._hand_cache) > 512:
                    self._hand_cache.clear()
                self._hand_cache[key] = kin.hand_geoms(sd, h["curl"], h["thumb"], h["index"])
            for n, (p, q, hl) in self._hand_cache[key].items():
                g = self.hand_ids[sd][n]
                m.geom_pos[g] = p
                m.geom_quat[g] = q
                m.geom_size[g, 1] = hl
        self.pose = pose

    def weld(self, name, on=True, torquescale=None, anchor=(0, 0, 0)):
        """Switch a weld on from the current pose (no yank) or off; anchor: the held point in body2's frame."""
        m, d = self.m, self.d
        e = m.equality(name).id
        if on:
            b1, b2 = m.eq_obj1id[e], m.eq_obj2id[e]
            mujoco.mj_kinematics(m, d)
            anc = np.asarray(anchor, float)
            p2 = d.xpos[b2] + d.xmat[b2].reshape(3, 3) @ anc
            m.eq_data[e, 0:3] = anc
            m.eq_data[e, 3:6] = d.xmat[b1].reshape(3, 3).T @ (p2 - d.xpos[b1])
            q1inv = np.zeros(4); mujoco.mju_negQuat(q1inv, d.xquat[b1])
            rq = np.zeros(4); mujoco.mju_mulQuat(rq, q1inv, d.xquat[b2])
            m.eq_data[e, 6:10] = rq
            if torquescale is not None:
                m.eq_data[e, 10] = torquescale
        d.eq_active[e] = 1 if on else 0

    def weld_force(self, name):
        """The weld's constraint force on body2 (N), world frame, from the last step (efc rows of the equality)."""
        m, d = self.m, self.d
        e = m.equality(name).id
        rows = [i for i in range(d.nefc) if d.efc_type[i] == mujoco.mjtConstraint.mjCNSTR_EQUALITY and d.efc_id[i] == e]
        if not rows:
            return np.zeros(3)
        return d.efc_force[rows[:3]].copy()

    def place_head(self, pos, R):
        """AN INSTRUMENT'S STILL (the eye and face instruments): her head put at (pos, R) in the world, her body held straight
        under it in its rest pose (her joints at zero, her velocities zero), for a render or a ray, never lived (she is a body: her
        head cannot be put anywhere alone)"""
        m, d = self.m, self.d
        b = self.bmap
        up = sum((kin.OFFSET[s] for s in ("abdomen", "chest", "head")), np.zeros(3))   # pelvis to head at rest (her joints at zero)
        R = np.asarray(R, float)
        d.qpos[b.root_q:b.root_q + 3] = np.asarray(pos, float) - R @ up
        d.qpos[b.root_q + 3:b.root_q + 7] = kin.mjquat(R)
        d.qpos[b.ball_q] = np.array([1.0, 0.0, 0.0, 0.0])
        d.qpos[b.hinge_q] = 0.0
        d.qvel[b.vadr] = 0.0

    def parent_pose_now(self):
        """her segments as the physics has them now: {seg: (pos, R)} (world)"""
        m, d = self.m, self.d
        return {s: (d.xpos[b].copy(), d.xmat[b].reshape(3, 3).copy()) for s, b in self.bmap.seg_body.items()}

    def hand_proxy(self, side, on=True, forearm=False):
        """The parent's hand (and forearm) collision proxy on or off. Both bits: its conaffinity (2 | 16: the floor and the room, never
        the G1 since A25c) lets what carries those bits touch it, so zeroing contype alone would leave the proxy colliding."""
        for seg in ("hand", "forearm") if forearm else ("hand",):
            g = self.m.geom(f"parent_{seg}_{side}").id
            self.m.geom_contype[g] = 8 if on else 0
            self.m.geom_conaffinity[g] = 18 if on else 0                 # the room (16) and the floor (2), never the G1 (A25c)

    # ---- the G1
    def jq(self, joint):
        return self.m.jnt_qposadr[self.m.joint(joint).id]

    def jd(self, joint):
        return self.m.jnt_dofadr[self.m.joint(joint).id]

    def set_g1(self, joints, root=None, ctrl=True):
        """Set G1 joints from {joint name without _joint: rad}; names given once with side "" apply to both sides with
        mirroring for roll/yaw (the G1's left/right ranges mirror); others stay."""
        d = self.d
        for k, v in joints.items():
            names = [k] if k.startswith(("left_", "right_", "waist_")) else [f"left_{k}", f"right_{k}"]
            for n in names:
                val = v
                if n.startswith("right_") and not k.startswith("right_") and any(t in k for t in ("roll", "yaw", "thumb_1", "thumb_2", "index", "middle")):
                    val = -v
                d.qpos[self.jq(n + "_joint")] = val
        if root is not None:
            d.qpos[0:7] = root
        if ctrl:
            self.hold_ctrl()

    def hold_ctrl(self):
        """Every servo target of the G1's = the measured angle (her muscles, the parent's actuators, are her motion's: A25b)"""
        m, d = self.m, self.d
        for i in self.g1_act:
            d.ctrl[i] = d.qpos[m.jnt_qposadr[m.actuator_trnid[i, 0]]]

    def lowest_g1_point(self):
        """The lowest point of the G1's collision geoms (world z)."""
        m, d = self.m, self.d
        low = 1e9
        for g in range(m.ngeom):
            if m.geom_bodyid[g] in self.g1_set and m.geom_contype[g] and m.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH:
                mid = m.geom_dataid[g]
                V = m.mesh_vert[m.mesh_vertadr[mid]:m.mesh_vertadr[mid] + m.mesh_vertnum[mid]]
                W = V @ d.geom_xmat[g].reshape(3, 3).T + d.geom_xpos[g]
                low = min(low, float(W[:, 2].min()))
            elif m.geom_bodyid[g] in self.g1_set and m.geom_contype[g]:
                low = min(low, float(d.geom_xpos[g][2] - m.geom_rbound[g]))
        return low

    def place_on_mat(self, joints, R, xy, clearance=.004, settle_s=BIRTH_SETTLE_S, hold=True):
        """Pose the G1 (joints, root rotation R, pelvis at xy), lower it onto the mat, settle under its servos (targets held
        at the pose when hold, else following the measured angles)."""
        m, d = self.m, self.d
        mujoco.mj_resetData(m, d)
        self.set_g1(joints, root=np.r_[xy[0], xy[1], 1.0, kin.mjquat(R)])
        mujoco.mj_forward(m, d)
        d.qpos[2] -= self.lowest_g1_point() - (.012 + clearance)
        d.qvel[:] = 0
        mujoco.mj_forward(m, d)
        b = self.bmap                                                  # the parent is kept where she stands while the G1 settles
        her_q = d.qpos[b.qadr].copy()                                  # (no motion drives her yet: her body would fall)
        for _ in range(int(round(settle_s / m.opt.timestep))):
            if not hold:
                self.hold_ctrl()
            mujoco.mj_step(m, d)
            d.qpos[b.qadr] = her_q
            d.qvel[b.vadr] = 0.0
        return d

    def birth(self):
        """The birth state: the G1 on its back on the mat (head toward -x, its left toward +y), settled BIRTH_SETTLE_S under
        its servos holding the birth pose (under whatever servo law the model carries: the world sets the body's first);
        the toys settled; the parent where the maker stood it, by the door, drawn (born_parent: her face at its neutral
        expression, her hands shaped). Deterministic; computed, never cached."""
        m, d = self.m, self.d
        self.place_on_mat(BIRTH, kin.ry(-math.pi / 2), BIRTH_XY)
        self.set_parent(born_parent())
        d.qvel[:] = 0
        d.qacc_warmstart[:] = 0
        mujoco.mj_forward(m, d)


def eye_option():
    """What the eyes render: every visible geom (the room, the parent, the toys, the G1's own visual meshes), never collision
    proxies (3), sites (4) or hidden markers (5)."""
    opt = mujoco.MjvOption()
    for g in (3, 4, 5):
        opt.geomgroup[g] = 0
    return opt


# the G1 prototype's names (main's 1eb268b: body/sim/g1eyes.py, tools/sim_look_g1.py read them), kept so the prototype's stills
# still run against the built scene: its World is this Scene; its periphery pool and fovea size are the eyes' (body/sim/eyes.py)
World = Scene
POOL = 3
FOVEA = 32
