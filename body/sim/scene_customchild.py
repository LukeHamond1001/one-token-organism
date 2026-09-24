"""Loading the CUSTOM CHILD's living room (kept for reference; the child is now the stock G1: g1scene.py) and driving its
kinematic parts: the parent's mocap segments and geoms, the child's screen face, the welds (from the 2026-09-24 prototype)."""
import math
import sys
from pathlib import Path

import mujoco
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import parent_kin as kin  # noqa: E402
import parent_face_customchild as face_cc  # noqa: E402  (the custom child's parent keeps the prototype's face)

XML = HERE / "livingroom_customchild.xml"

# the child's birth pose (degrees): supine, arms relaxed a little out from the sides with the elbows half bent, hips a
# little flexed and turned out, knees half bent (the resting posture of an infant on its back)
BIRTH_DEG = dict(shoulder_flex=15, shoulder_abd=55, shoulder_rot=60, elbow=85, pronation=0, wrist_flex=15,
                 hip_flex=35, hip_abd=35, hip_rot=35, knee=65, ankle_dorsi=0, ankle_inv=0)
BIRTH_XY, BIRTH_YAW = (0.05, -0.28), 0.0      # the pelvis on the mat; the head toward -x, its left side toward +y


EYE_HIDDEN_GROUPS = (3, 4, 5)   # never drawn in the child's eyes: collision proxies (3), sites (4), its own face (5)


def eye_option():
    """The render option for the child's eyes: everything but its own face and the invisible proxies."""
    opt = mujoco.MjvOption()
    for g in EYE_HIDDEN_GROUPS:
        opt.geomgroup[g] = 0
    return opt


class World:
    def __init__(self, xml=XML):
        self.m = mujoco.MjModel.from_xml_path(str(xml))
        self.d = mujoco.MjData(self.m)
        m = self.m
        self.mocap = {s: m.body_mocapid[m.body(f"parent_{s}").id] for s in kin.SEGS}
        self.gid = lambda n: m.geom(n).id
        self.face_ids = {}
        for n in face_cc.face_geoms(0.0).keys():
            self.face_ids[n] = self.gid(f"parent_{n}")
        self.hand_ids = {sd: {n: self.gid(f"parent_{n}") for n in kin.hand_geoms(sd).keys()} for sd in ("L", "R")}
        self.cface_ids = {n: self.gid(n) for n in kin.child_face_geoms(0).keys()}
        self.head_body = m.body("parent_head").id
        # these geoms are moved at run time: MuJoCo's compile-time "same frame as the body" shortcut must be off for them
        for g in list(self.face_ids.values()) + [i for h in self.hand_ids.values() for i in h.values()] + list(self.cface_ids.values()):
            m.geom_sameframe[g] = 0
        self.child_act = [m.actuator(i).name for i in range(m.nu)]
        self.pose = None

    # ---- the parent
    def set_parent(self, pose):
        m, d = self.m, self.d
        segs = kin.fk(pose)
        for s, (p, R) in segs.items():
            i = self.mocap[s]
            d.mocap_pos[i] = p
            d.mocap_quat[i] = kin.mjquat(R)
        hp, hR = segs["head"]
        gaze = None
        if pose.gaze is not None:
            gaze = {sd: hR.T @ (pose.gaze - (hp + hR @ face_cc.EYE_C[sd])) for sd in ("L", "R")}
        for n, (p, q, sz) in face_cc.face_geoms(pose.expr, gaze).items():
            g = self.face_ids[n]
            m.geom_pos[g] = p; m.geom_quat[g] = q
            if sz is not None:
                if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_CAPSULE:
                    m.geom_size[g, :2] = sz[:2]
                else:
                    m.geom_size[g] = sz
        for sd in ("L", "R"):
            h = pose.hand[sd]
            for n, (p, q, hl) in kin.hand_geoms(sd, h["curl"], h["thumb"], h["index"]).items():
                g = self.hand_ids[sd][n]
                m.geom_pos[g] = p if sd == "L" or True else p
                m.geom_quat[g] = q
                m.geom_size[g, 1] = hl
        self.pose = pose

    # ---- the child's face (drawn from its eye joints and an expression)
    def set_child_face(self, expr=0.0):
        m, d = self.m, self.d
        eyes = {sd: (d.qpos[m.jnt_qposadr[m.joint(f"child_{sd}_eye_yaw").id]], d.qpos[m.jnt_qposadr[m.joint(f"child_{sd}_eye_pitch").id]])
                for sd in ("L", "R")}
        for n, (p, q, sz) in kin.child_face_geoms(expr, eyes).items():
            g = self.cface_ids[n]
            m.geom_pos[g] = p; m.geom_quat[g] = q
            if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_CAPSULE:
                m.geom_size[g, :2] = sz[:2]
            else:
                m.geom_size[g] = sz

    # ---- welds: switched on from the current pose (no yank)
    def weld(self, name, on=True, torquescale=None, anchor=(0, 0, 0)):
        """Switch a weld on (from the current pose: no yank) or off. anchor: the held point in body2's frame (with
        torquescale 0 only that point is held; the body turns freely about it)."""
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

    def hand_proxy(self, side, on=True, forearm=False):
        """The parent's hand collision capsule on or off (off while the hand holds something through a weld); with
        forearm=True the forearm's too (a kinematic arm cannot yield when it brushes the child's arm)."""
        for seg in ("hand", "forearm") if forearm else ("hand",):
            g = self.m.geom(f"parent_{seg}_{side}").id
            self.m.geom_contype[g] = 8 if on else 0

    # ---- the child's joints
    def cj(self, name):
        return self.m.jnt_qposadr[self.m.joint(name).id]

    def child_pose_ctrl(self, deg, root=None):
        """Set the child's joints (and the servo targets) to a pose given in degrees by joint kind (both sides)."""
        m, d = self.m, self.d
        for i in range(m.nu):
            n = m.actuator(i).name
            if n.endswith("_fingers"):
                continue
            kind = n.split("_", 2)[-1] if n.startswith(("child_L_", "child_R_")) else n[len("child_"):]
            v = math.radians(deg.get(kind, 0.0))
            d.qpos[self.cj(n)] = v
            d.ctrl[i] = v
        if root is not None:
            d.qpos[self.cj("child_root"):self.cj("child_root") + 7] = root

    def hold_ctrl(self):
        """The servo law at rest: every target = the measured angle (fingers: the tendon's length)."""
        m, d = self.m, self.d
        for i in range(m.nu):
            if m.actuator_trntype[i] == mujoco.mjtTrn.mjTRN_JOINT:
                d.ctrl[i] = d.qpos[m.jnt_qposadr[m.actuator_trnid[i, 0]]]
            else:
                d.ctrl[i] = d.ten_length[m.actuator_trnid[i, 0]]

    def birth(self, settle_s=1.5, cache=True):
        """The birth state: the child supine on the mat (settled for settle_s under its servos), toys settled, the parent
        standing by the door. Cached to birth_state_customchild.npy."""
        m, d = self.m, self.d
        cp = HERE / "birth_state_customchild.npy"
        if cache and cp.exists() and cp.stat().st_mtime > XML.stat().st_mtime:
            st = np.load(cp)
            mujoco.mj_setState(m, d, st, mujoco.mjtState.mjSTATE_INTEGRATION)
            mujoco.mj_forward(m, d)
            return
        mujoco.mj_resetData(m, d)
        q = kin.ry(-math.pi / 2)                                              # face up, head toward -x
        quat = kin.mjquat(kin.rz(BIRTH_YAW) @ q)
        self.child_pose_ctrl(BIRTH_DEG, root=np.r_[BIRTH_XY[0], BIRTH_XY[1], .09, quat])
        mujoco.mj_forward(m, d)
        for _ in range(int(settle_s / m.opt.timestep)):
            mujoco.mj_step(m, d)
        d.qvel[:] = 0
        mujoco.mj_forward(m, d)
        n = mujoco.mj_stateSize(m, mujoco.mjtState.mjSTATE_INTEGRATION)
        st = np.zeros(n); mujoco.mj_getState(m, d, st, mujoco.mjtState.mjSTATE_INTEGRATION)
        np.save(cp, st)


def contacts(m, d, only_dist_below=0.0):
    out = []
    for c in d.contact[:d.ncon]:
        if c.dist < only_dist_below:
            out.append((mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, c.geom1), mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, c.geom2), round(float(c.dist), 4)))
    return out
