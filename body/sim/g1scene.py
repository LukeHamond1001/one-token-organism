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

import mujoco
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import parent_kin as kin  # noqa: E402

XML = HERE / "g1room.xml"
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
    m = spec.compile()
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
