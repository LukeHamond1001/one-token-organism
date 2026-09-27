"""THE PARENT'S BODY (docs/SIM_DESIGN.md 4.1, 4.2, A4, A25, A25b; the lead's decisions of 2026-09-25: she is a body, and her trunk is
carried). The world's side only.

Her 16 segments are dynamic MuJoCo bodies in one tree (body/sim/make_g1room.py: a free pelvis; ball joints at the lumbar, thorax,
neck, shoulders, wrists, hips and ankles; hinges at the elbows, the forearms' pronation and the knees), with de Leva's female
masses at her body mass (parent_consts). Her planner (body/sim/parent_motion.py) makes the pose she means at each tick's end, as
parent_kin's segment frames; here that pose becomes her joints' targets and her pelvis's, and every physics step:

HER MUSCLES. Each axis of each joint has one MuJoCo actuator (make_g1room.parent_actuators) whose control range and force range are
her strength there, per direction (parent_consts.STRENGTH: a woman's). Her command is

  u = Kp (target - q) + Kd x target velocity + her tone + her posture's feedforward + her holds' effort

written as the actuator's control; MuJoCo clamps it to her strength, subtracts the actuator's own damping Kd x velocity, and clamps
the sum to her strength again, so her whole torque at a joint, damping included, never exceeds a woman's (the first physical build's
verifier found MuJoCo's joint damping outside the clip: 139 N m at her lumbar against its 64 / 103). Nothing else of hers acts on
her joints: no joint damping, no applied force (body/tests/test_sim_parent.py reads the actuators' forces every step).
  - Kp: each joint's strength over SAT_DEG (it spends its whole strength 20 deg off its plan), over STIFF_SAT_DEG (5 deg) at her
    spine's two joints and her neck, so her trunk and head follow the support (the lead's "a stiff neck, limited");
  - Kd: critical against the inertia the joint moves at her rest pose (MuJoCo's dof_M0), the actuator's own damping (its biasprm,
    set here at load), taken implicitly by MuJoCo's implicitfast while the force is inside its range;
  - her posture's feedforward: the torque that holds her own limbs and trunk against gravity as they are (MuJoCo's bias force on
    her joints: a person knows her own limbs' weight);
  - her tone: the error integrated over time (TONE_S) on each joint her plan holds still, within TONE_SHARE of its strength, so a
    posture she means is held against its load (a toy in her hand); a chain the child presses lets it go (she never builds up
    against it);
  - her holds' effort: each capped spring she holds the child by pushes the child's link with its force F at the held point, and
    her hand with -F at her grip (Newton), and her arm and trunk exert J^T F to carry it (parent_motion's holds).

HER SUPPORT (A25b: a person does not fall over; balance is the environment's business, never the child's). Her pelvis is pulled
toward the pelvis her plan means (its place and its turn, interpolated through the tick) by a spring and a damper, and her weight
is carried at her whole body's centre of mass (so carrying it twists her not at all); the force together is at most SUP_F_MAX and
never points down (her weight first: along the floor it keeps what the cap leaves), the torque at most SUP_T_MAX
(parent_consts: her weight and the child's greatest steady push; her two hips' strength). Plain numbers each step, applied as an
outside force on her pelvis (MuJoCo's xfrc_applied). So she cannot be toppled by the child and never presses down on anything
with more than her own weight; pushed along the floor past the cap, she gives way, carried. Her feet, knees and shins touch the
floor through contact always (no gait of hers is carried through the floor).

A CHAIN STOPPED (A4: "that segment stops"): a chain (each arm, or the rest of her) whose contact with the child passed its cap for
2 steps holds where it is for the rest of the tick: its joints' targets become their positions at that step, their target
velocities zero (for the rest of her, her pelvis's too: the support holds her where she is), so she presses no further.

EXACT DETERMINISM: pure numpy and plain floats on the physics state; the tick's targets are plain numbers in her motion's state."""
import math

import mujoco
import numpy as np

import parent_consts as K
import parent_kin as kin

TICK_S, STEPS = 0.150, 75
CHAINS = ("arm_L", "arm_R", "core")

# her joints in the scene, per segment (make_g1room.PARENT_JOINTS): (joint suffix, type)
BALLS = [("lumbar", "abdomen"), ("thorax", "chest"), ("neck", "head"), ("shoulder_L", "upper_arm_L"), ("wrist_L", "hand_L"),
         ("shoulder_R", "upper_arm_R"), ("wrist_R", "hand_R"), ("hip_L", "thigh_L"), ("ankle_L", "foot_L"), ("hip_R", "thigh_R"),
         ("ankle_R", "foot_R")]
HINGES = [("elbow_L", "forearm_L"), ("pron_L", "forearm_L"), ("elbow_R", "forearm_R"), ("pron_R", "forearm_R"),
          ("knee_L", "shin_L"), ("knee_R", "shin_R")]
STIFF = ("lumbar", "thorax", "neck")                                  # her spine and neck: STIFF_SAT_DEG (her trunk and head follow
                                                                       # the support)
JOINT_CHAIN = {**{j: "core" for j in ("lumbar", "thorax", "neck", "hip_L", "ankle_L", "hip_R", "ankle_R", "knee_L", "knee_R")},
               **{j: f"arm_{j[-1]}" for j in ("shoulder_L", "wrist_L", "shoulder_R", "wrist_R", "elbow_L", "pron_L", "elbow_R",
                                                "pron_R")}}


def _strength(joint):
    """(lo, hi) torque limits per dof of a joint, N m, in the joint's own frame (a ball's x, y, z; a hinge's axis): her strength
    (parent_consts.STRENGTH) mapped onto parent_kin's sign conventions (flexion about -y at the shoulder, hip and ankle, +y at the
    spine and neck; abduction about +x on the left; the right side mirrored in y, so its x and z swap their directions)"""
    S = K.STRENGTH
    base, sd = (joint[:-2], joint[-1]) if joint[-2] == "_" else (joint, "")
    L = sd != "R"
    if base == "shoulder":
        s = S["shoulder"]
        x = (-s["add"], s["abd"]) if L else (-s["abd"], s["add"])
        return [x, (-s["flex"], s["ext"]), (-s["rot"], s["rot"])]
    if base == "wrist":
        s = S["wrist"]
        x = (-s["flex"], s["ext"]) if L else (-s["ext"], s["flex"])
        return [x, (-s["dev"], s["dev"]), (-s["rot"], s["rot"])]
    if base == "hip":
        s = S["hip"]
        x = (-s["add"], s["abd"]) if L else (-s["abd"], s["add"])
        return [x, (-s["flex"], s["ext"]), (-s["rot"], s["rot"])]
    if base == "ankle":
        s = S["ankle"]
        return [(-s["inv"], s["inv"]), (-s["dorsi"], s["plantar"]), (-s["rot"], s["rot"])]
    if base == "neck":
        s = S["neck"]
        return [(-s["lat"], s["lat"]), (-s["ext"], s["flex"]), (-s["rot"], s["rot"])]
    if base in ("lumbar", "thorax"):
        s = S["trunk"]
        return [(-s["lat"], s["lat"]), (-s["ext"], s["flex"]), (-s["rot"], s["rot"])]
    if base == "elbow":
        return [(-S["elbow"]["ext"], S["elbow"]["flex"])]
    if base == "pron":
        return [(-S["pron"]["rot"], S["pron"]["rot"])]
    if base == "knee":                                                  # its axis +y, q its flexion: extension the negative torque
        return [(-S["knee"]["ext"], S["knee"]["flex"])]
    raise ValueError(joint)


def mat2quat(R):
    """rotation matrices (..., 3, 3) to MuJoCo quaternions (..., 4) [w, x, y, z], w >= 0"""
    R = np.asarray(R, float)
    q = np.empty(R.shape[:-2] + (4,))
    t = R[..., 0, 0] + R[..., 1, 1] + R[..., 2, 2]
    q[..., 0] = np.sqrt(np.maximum(0.0, 1.0 + t)) / 2
    q[..., 1] = np.sqrt(np.maximum(0.0, 1.0 + R[..., 0, 0] - R[..., 1, 1] - R[..., 2, 2])) / 2
    q[..., 2] = np.sqrt(np.maximum(0.0, 1.0 - R[..., 0, 0] + R[..., 1, 1] - R[..., 2, 2])) / 2
    q[..., 3] = np.sqrt(np.maximum(0.0, 1.0 - R[..., 0, 0] - R[..., 1, 1] + R[..., 2, 2])) / 2
    q[..., 1] = np.copysign(q[..., 1], R[..., 2, 1] - R[..., 1, 2])
    q[..., 2] = np.copysign(q[..., 2], R[..., 0, 2] - R[..., 2, 0])
    q[..., 3] = np.copysign(q[..., 3], R[..., 1, 0] - R[..., 0, 1])
    return q / np.linalg.norm(q, axis=-1, keepdims=True)


def quat2mat(q):
    q = np.asarray(q, float)
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    R = np.empty(q.shape[:-1] + (3, 3))
    R[..., 0, 0] = 1 - 2 * (y * y + z * z); R[..., 0, 1] = 2 * (x * y - w * z); R[..., 0, 2] = 2 * (x * z + w * y)
    R[..., 1, 0] = 2 * (x * y + w * z); R[..., 1, 1] = 1 - 2 * (x * x + z * z); R[..., 1, 2] = 2 * (y * z - w * x)
    R[..., 2, 0] = 2 * (x * z - w * y); R[..., 2, 1] = 2 * (y * z + w * x); R[..., 2, 2] = 1 - 2 * (x * x + y * y)
    return R


def qmul(a, b):
    aw, ax, ay, az = a[..., 0], a[..., 1], a[..., 2], a[..., 3]
    bw, bx, by, bz = b[..., 0], b[..., 1], b[..., 2], b[..., 3]
    return np.stack([aw * bw - ax * bx - ay * by - az * bz, aw * bx + ax * bw + ay * bz - az * by,
                     aw * by - ax * bz + ay * bw + az * bx, aw * bz + ax * by - ay * bx + az * bw], axis=-1)


def qconj(q):
    return q * np.array([1.0, -1.0, -1.0, -1.0])


def qlog(q):
    """the rotation vector (..., 3) of unit quaternions, the shorter way round"""
    q = np.where(q[..., :1] < 0, -q, q)
    v = q[..., 1:]
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    ang = 2 * np.arctan2(n, q[..., :1])
    scale = np.where(n > 1e-12, ang / np.where(n > 1e-12, n, 1.0), 2.0)
    return v * scale


def nlerp(q0, q1, a):
    q1 = np.where(np.sum(q0 * q1, axis=-1, keepdims=True) < 0, -q1, q1)
    q = q0 + a * (q1 - q0)
    return q / np.linalg.norm(q, axis=-1, keepdims=True)


class BodyMap:
    """her joints in a compiled scene: qpos and dof addresses, her strength and gains per dof, and the pose <-> joints maps"""

    def __init__(self, m):
        self.m = m
        jid = lambda n: m.joint(f"parent_{n}").id
        self.root_j = jid("root")
        self.root_q = int(m.jnt_qposadr[self.root_j]); self.root_d = int(m.jnt_dofadr[self.root_j])
        self.pelvis = m.body("parent_pelvis").id
        self.bodies = np.array([m.body(f"parent_{s}").id for s in kin.SEGS])
        self.seg_body = {s: m.body(f"parent_{s}").id for s in kin.SEGS}
        self.ball_q = np.array([[m.jnt_qposadr[jid(n)] + k for k in range(4)] for n, _ in BALLS])
        self.ball_d = np.array([[m.jnt_dofadr[jid(n)] + k for k in range(3)] for n, _ in BALLS])
        self.hinge_q = np.array([m.jnt_qposadr[jid(n)] for n, _ in HINGES])
        self.hinge_d = np.array([m.jnt_dofadr[jid(n)] for n, _ in HINGES])
        self.dofs = np.concatenate([self.ball_d.ravel(), self.hinge_d])        # her internal dofs, in the drive's order
        self.qadr = np.concatenate([np.arange(self.root_q, self.root_q + 7), self.ball_q.ravel(), self.hinge_q])   # all her qpos
        self.vadr = np.arange(self.root_d, self.root_d + 6).tolist() + self.dofs.tolist()
        self.vadr = np.array(sorted(self.vadr))
        lo, hi = [], []
        for n, _ in BALLS:
            for a, b in _strength(n):
                lo.append(a); hi.append(b)
        for n, _ in HINGES:
            (a, b), = _strength(n)
            lo.append(a); hi.append(b)
        self.lo = np.array(lo); self.hi = np.array(hi)
        sat = np.array([K.STIFF_SAT_DEG if n in STIFF else K.SAT_DEG for n, _ in BALLS for _k in range(3)]
                       + [K.STIFF_SAT_DEG if n in STIFF else K.SAT_DEG for n, _ in HINGES])
        self.kp = 0.5 * (self.hi - self.lo) / np.radians(sat)
        self.kd = 2.0 * K.DAMP_RATIO * np.sqrt(self.kp * m.dof_M0[self.dofs])
        # HER MUSCLES (make_g1room.parent_actuators): one actuator per dof in the drive's order, its force and control ranges her
        # strength, its damping Kd (set here: MuJoCo's dof_M0 is the compiled model's); no joint damping of hers
        names = [f"parent_{n}_{a}" for n, _ in BALLS for a in "xyz"] + [f"parent_{n}" for n, _ in HINGES]
        self.act = np.array([m.actuator(nm).id for nm in names])
        jd = [m.jnt_dofadr[int(m.actuator_trnid[a, 0])] for a in self.act]
        ax = [int(np.argmax(m.actuator_gear[a, :3])) if m.jnt_type[int(m.actuator_trnid[a, 0])] == mujoco.mjtJoint.mjJNT_BALL
              else 0 for a in self.act]
        if not np.array_equal(np.array(jd) + np.array(ax), self.dofs) or \
                not (np.array_equal(m.actuator_forcerange[self.act, 0], self.lo) and np.array_equal(m.actuator_forcerange[self.act, 1], self.hi)
                     and np.array_equal(m.actuator_ctrlrange[self.act, 0], self.lo) and np.array_equal(m.actuator_ctrlrange[self.act, 1], self.hi)
                     and m.actuator_forcelimited[self.act].all() and m.actuator_ctrllimited[self.act].all()):
            raise ValueError("her actuators are not her joints' axes at her strength (make_g1room.parent_actuators): regenerate g1room.xml")
        m.actuator_gainprm[self.act, :] = 0.0; m.actuator_gainprm[self.act, 0] = 1.0
        m.actuator_biasprm[self.act, :] = 0.0; m.actuator_biasprm[self.act, 2] = -self.kd
        m.dof_damping[self.dofs] = 0.0
        self.nb, self.nh = len(BALLS), len(HINGES)
        self.chain_of_dof = np.array([CHAINS.index(JOINT_CHAIN[n]) for n, _ in BALLS for _k in range(3)]
                                     + [CHAINS.index(JOINT_CHAIN[n]) for n, _ in HINGES])
        self.ball_chain = np.array([CHAINS.index(JOINT_CHAIN[n]) for n, _ in BALLS])
        self.hinge_chain = np.array([CHAINS.index(JOINT_CHAIN[n]) for n, _ in HINGES])
        self.arm_dof = np.array([JOINT_CHAIN[n] != "core" for n, _ in BALLS for _k in range(3)] + [JOINT_CHAIN[n] != "core" for n, _ in HINGES])
        legs = ("hip_", "knee_", "ankle_")
        self.leg_dof = np.array([n.startswith(legs) for n, _ in BALLS for _k in range(3)] + [n.startswith(legs) for n, _ in HINGES])
        self.hand_body = {sd: m.body(f"parent_hand_{sd}").id for sd in "LR"}

    # ---- the pose her planner makes -> her joints
    def targets(self, segs):
        """her joints' targets from parent_kin segment frames {seg: (pos, R)}: (root pos (3), root quat (4), ball quats (11, 4),
        hinge angles (6)). The elbow's rotation split into its flexion and the forearm's turn as parent_kin.arm_ik composes them,
        the knee's flexion as leg_ik's; the wrist and the ankle take what remains, so every segment's frame is kept"""
        R = {s: np.asarray(segs[s][1], float) for s in kin.SEGS}
        hinge = np.zeros(self.nh)
        par = {}
        for i, sd in enumerate("LR"):
            rel = R[f"upper_arm_{sd}"].T @ R[f"forearm_{sd}"]
            th = math.atan2(-rel[0, 2], rel[2, 2])
            M = kin.ry(th) @ rel
            psi = math.atan2(M[1, 0], M[0, 0])
            hinge[2 * i], hinge[2 * i + 1] = th, psi
            par[f"hand_{sd}"] = R[f"upper_arm_{sd}"] @ kin.ry(-th) @ kin.rz(psi)
            relk = R[f"thigh_{sd}"].T @ R[f"shin_{sd}"]
            tk = math.atan2(relk[0, 2], relk[2, 2])
            hinge[4 + i] = tk
            par[f"foot_{sd}"] = R[f"thigh_{sd}"] @ kin.ry(tk)
        rels = np.array([(par[s] if s in par else R[kin.SEG_PARENT[s]]).T @ R[s] for _, s in BALLS])
        return (np.asarray(segs["pelvis"][0], float).copy(), mat2quat(R["pelvis"]), mat2quat(rels), hinge)

    def write_qpos(self, qpos, tg):
        """her joints' targets written into a qpos vector (a scratch state, or the scene placing her)"""
        p, q, qb, h = tg
        qpos[self.root_q:self.root_q + 3] = p
        qpos[self.root_q + 3:self.root_q + 7] = q
        qpos[self.ball_q] = qb
        qpos[self.hinge_q] = h

    def read(self, d):
        """her joints as the physics has them now, in targets()' form"""
        return (d.qpos[self.root_q:self.root_q + 3].copy(), d.qpos[self.root_q + 3:self.root_q + 7].copy(),
                d.qpos[self.ball_q].copy(), d.qpos[self.hinge_q].copy())


def tg_plain(tg):
    return None if tg is None else [np.asarray(x, float).tolist() for x in tg]


def tg_unplain(s):
    return None if s is None else tuple(np.array(x, dtype=np.float64) for x in s)


class Drive:
    """her joints driven toward the tick's targets every physics step through her muscles, and her trunk carried by her support (see
    the module's doc). Its per-tick state (the tick's start and end targets, a stopped chain's hold, her tone) is plain numbers, in her
    motion's state"""

    def __init__(self, m, d, bmap):
        self.m, self.d, self.b = m, d, bmap
        self.t0 = self.t1 = None                                        # the tick's start and end targets
        self.vr = self.wr = self.vr_start = self.wr_start = None         # her pelvis's target velocities (this tick's, and its ramp's
        self.legs_rest = False                                          # start); her legs resting (set_tick)
        self.stop = [None, None, None]                                  # per chain: None, or the joints' positions it holds at
        self.effort = np.zeros(len(bmap.dofs))                          # her holds' effort this step (J^T F on her joints)
        self.tone = np.zeros(len(bmap.dofs))                            # her tone: each joint's error integrated (N m)
        self.ki = bmap.kp / K.TONE_S * m.opt.timestep                   # its gain a step ...
        self.tone_lo, self.tone_hi = K.TONE_SHARE * bmap.lo, K.TONE_SHARE * bmap.hi   # ... and its reach (a share of her strength)
        self.u = np.zeros(len(bmap.dofs))                               # her command on the last step (an instrument)
        self.sat = 0                                                    # steps on which some command was past her strength (instrument)
        self.sup = np.zeros(6)                                          # her support's force and torque on the last step (instrument)
        self.sup_sat = 0                                                # steps on which a cap of her support bound (instrument)
        self._e = np.zeros(len(bmap.dofs))
        self.weight = float(m.body_subtreemass[bmap.pelvis]) * float(-m.opt.gravity[2])
        names = [n for n, _ in BALLS]
        self.i_lumbar, self.i_thorax = names.index("lumbar"), names.index("thorax")   # her trunk's two joints above her pelvis
        self.abd, self.chest = bmap.seg_body["abdomen"], bmap.seg_body["chest"]
        self.sup_t = np.zeros((3, 3))                                   # her support's turning torque on her pelvis, abdomen, chest
        self.push = np.zeros(3)                                         # the child's push on her body (trunk, head, legs), the last step:
                                                                        # the sum of its contacts' forces on them (N, world frame)
        self.mass = float(m.body_subtreemass[bmap.pelvis])
        self.c_lin = 2.0 * K.SUP_ZETA * math.sqrt(K.SUP_K * self.mass)
        self.fh_max2 = K.SUP_F_MAX * K.SUP_F_MAX

    def state(self):
        return dict(t0=tg_plain(self.t0), t1=tg_plain(self.t1), stop=[tg_plain(x) for x in self.stop], tone=self.tone.tolist(),
                    legs_rest=bool(self.legs_rest), vr0=None if self.vr_start is None else self.vr_start.tolist(),
                    wr0=None if self.wr_start is None else self.wr_start.tolist(), push=self.push.tolist(),
                    vr=None if self.vr is None else np.asarray(self.vr, float).tolist(),
                    wr=None if self.wr is None else np.asarray(self.wr, float).tolist())   # the next tick's ramp starts from these, which
                                                                    # a chain held mid-tick zeroes (S5a: a replay saved then diverged)

    def load_state(self, s):
        t0, t1 = tg_unplain(s.get("t0")), tg_unplain(s.get("t1"))
        if t0 is not None and t1 is not None:
            self.set_tick(t0, t1, bool(s.get("legs_rest", False)), s.get("vr0"), s.get("wr0"))
        else:
            self.t0, self.t1 = t0, t1
        self.stop = [tg_unplain(x) for x in s.get("stop", [None, None, None])]
        self.tone = np.array(s.get("tone", np.zeros(len(self.b.dofs))), dtype=np.float64)
        self.push = np.array(s.get("push", np.zeros(3)), dtype=np.float64)
        if s.get("vr") is not None:
            self.vr = np.array(s["vr"], dtype=np.float64)
        if s.get("wr") is not None:
            self.wr = np.array(s["wr"], dtype=np.float64)

    def set_tick(self, t0, t1, legs_rest=False, v0=None, w0=None):
        """the tick's start and end targets (her plan at the last tick's end, and at this one's): her joints' and her pelvis's.
        legs_rest: her base is still (kneeling, standing or sitting where she is), so her legs rest where her carried pelvis puts
        them, at LEG_REST_SAT_DEG's stiffness (a planned leg held stiffly against the floor under a carried pelvis pressed it with a
        leg's strength: A25b's build)"""
        vr0 = self.vr if v0 is None else np.asarray(v0, float)          # the ramp's start: last tick's target velocities
        wr0 = self.wr if w0 is None else np.asarray(w0, float)
        self.t0, self.t1 = t0, t1
        self.legs_rest = bool(legs_rest)
        self.stop = [None, None, None]
        b = self.b
        self.kp = np.where(b.leg_dof, b.kp * (math.radians(K.SAT_DEG) / math.radians(K.LEG_REST_SAT_DEG)), b.kp) if legs_rest else b.kp
        a = ((np.arange(STEPS) + 1.0) / STEPS)[:, None]                  # each step's share of the tick (its end)
        q1b = nlerp(t0[2], t1[2], 1.0)
        self.QB = nlerp(t0[2][None], q1b[None], a[:, :, None])           # (steps, balls, 4): the balls' targets a step
        self.H = t0[3][None] + a * (t1[3] - t0[3])[None]                  # (steps, hinges)
        self.PR = t0[0][None] + a * (t1[0] - t0[0])[None]                 # (steps, 3): the pelvis's place
        self.QR = nlerp(t0[1][None], t1[1][None], a)                      # (steps, 4): its turn
        wb = qlog(qmul(qconj(t0[2]), q1b)) / TICK_S                       # the balls' target velocities (their own frames)
        wh = (t1[3] - t0[3]) / TICK_S
        wt = np.concatenate([np.repeat(np.linalg.norm(wb, axis=1), 3), np.abs(wh)])
        self.KI = np.where((wt <= K.TONE_STILL_RPS) & b.arm_dof, self.ki, 0.0)   # her tone settles on an arm's joint her plan holds
                                                                        # still (a toy in her hand); her trunk is carried and her legs
                                                                        # are only posed (wound up against the floor under a carried
                                                                        # pelvis, their tone pressed her knees into it: A25b's build)
        self.FF = np.concatenate([(b.kd[:3 * b.nb].reshape(-1, 3) * wb).ravel(), b.kd[3 * b.nb:] * wh])   # her damping's target
        self.vr = (t1[0] - t0[0]) / TICK_S                                                              # velocity (a tick's)
        self.wr = qlog(qmul(qconj(t0[1]), nlerp(t0[1], t1[1], 1.0))) / TICK_S   # the pelvis's (its own frame)
        # the support's target velocity ramps through the tick from last tick's to this one's (a step in it at the tick's start
        # would jolt her by the damper's whole gain times the step: 2,227 N per m/s)
        a1 = ((np.arange(STEPS) + 1.0) / STEPS)[:, None]
        self.vr_start = self.vr.copy() if vr0 is None else np.asarray(vr0, float).copy()
        self.wr_start = self.wr.copy() if wr0 is None else np.asarray(wr0, float).copy()
        self.VR = self.vr_start[None] + a1 * (self.vr - self.vr_start)[None]
        self.WR = self.wr_start[None] + a1 * (self.wr - self.wr_start)[None]
        self.acc = self.mass * (self.vr - self.vr_start) / TICK_S           # her planned acceleration carried (the ramp's: a person's
                                                                        # legs push her body where she means it to go)

    def hold_chain(self, c):
        """A4: chain c stops where it is for the rest of the tick: its targets become its joints' positions now, its target
        velocities zero, its tone let go (for the rest of her, her pelvis's target too: the support holds her where she is)"""
        if self.stop[c] is None:
            st = self.stop[c] = self.b.read(self.d)
            b = self.b
            sel = b.ball_chain == c
            self.QB[:, sel] = st[2][sel]
            selh = b.hinge_chain == c
            self.H[:, selh] = st[3][selh]
            self.FF[b.chain_of_dof == c] = 0.0
            self.KI[b.chain_of_dof == c] = 0.0
            if c == 2:
                self.PR[:] = st[0]; self.QR[:] = st[1]; self.vr = np.zeros(3); self.wr = np.zeros(3)
                self.VR[:] = 0.0; self.WR[:] = 0.0; self.acc = np.zeros(3)
        self.relax(c)

    def relax(self, c):
        """the child presses chain c: its tone is let go and builds no more this tick (she never builds up against it)"""
        sel = self.b.chain_of_dof == c
        self.KI[sel] = 0.0
        self.tone[sel] = 0.0

    def step(self, s):
        """her muscles' commands and her support's force for physics step s (before mj_step)"""
        d, b = self.d, self.b
        q = d.qpos[b.ball_q]; t = self.QB[s]                             # each ball's error, in its own frame: the rotation vector of
        w0, x0, y0, z0 = q[:, 0], q[:, 1], q[:, 2], q[:, 3]             # conj(q) t, the shorter way round (qlog(qmul(qconj(q), t)),
        w1, x1, y1, z1 = t[:, 0], t[:, 1], t[:, 2], t[:, 3]             # written out)
        rw = w0 * w1 + x0 * x1 + y0 * y1 + z0 * z1
        rx = w0 * x1 - x0 * w1 - y0 * z1 + z0 * y1
        ry = w0 * y1 + x0 * z1 - y0 * w1 - z0 * x1
        rz = w0 * z1 - x0 * y1 + y0 * x1 - z0 * w1
        sg = np.where(rw < 0.0, -1.0, 1.0)
        n = np.sqrt(rx * rx + ry * ry + rz * rz)
        big = n > 1e-12
        k = np.where(big, 2.0 * np.arctan2(n, rw * sg) / np.where(big, n, 1.0), 2.0) * sg
        e = self._e
        e[0:3 * b.nb:3] = rx * k; e[1:3 * b.nb:3] = ry * k; e[2:3 * b.nb:3] = rz * k
        e[3 * b.nb:] = self.H[s] - d.qpos[b.hinge_q]
        self.tone = np.clip(self.tone + self.KI * e, self.tone_lo, self.tone_hi)
        u = self.kp * e + self.FF + self.tone + d.qfrc_bias[b.dofs] + self.effort
        self.u = u
        if (u < b.lo).any() or (u > b.hi).any():
            self.sat += 1
        d.ctrl[b.act] = u                                               # MuJoCo: clamped to her strength, less her damping, clamped
        # HER SUPPORT (parent_consts.SUP_*): her weight at her centre of mass, a spring and a damper toward her plan's pelvis; the force
        # at most SUP_F_MAX and never down (her weight first), the torque at most SUP_T_MAX. Plain floats: a 3-vector's numpy call costs
        # more than its arithmetic
        qpos, qvel = d.qpos, d.qvel
        rq, rd = b.root_q, b.root_d
        pel = b.pelvis
        pr = self.PR[s]; vr = self.VR[s]
        kk, cc = K.SUP_K, self.c_lin
        ac = self.acc
        fx = float(ac[0]) + kk * (float(pr[0]) - float(qpos[rq])) + cc * (float(vr[0]) - float(qvel[rd]))
        fy = float(ac[1]) + kk * (float(pr[1]) - float(qpos[rq + 1])) + cc * (float(vr[1]) - float(qvel[rd + 1]))
        cz = float(ac[2]) + kk * (float(pr[2]) - float(qpos[rq + 2])) + cc * (float(vr[2]) - float(qvel[rd + 2]))
        # SHE GIVES WAY TO THE CHILD, CARRIED (A25b): while it pushes her body (its contacts on her trunk, head and legs past a resting
        # hand's weight, the last step), the support's correction never acts against that push: its part against the push is dropped,
        # so it never presses her back into the child, and pushed up by it she is never let down onto it (her weight stays carried:
        # the first carried build let her weight rest on a child that had lifted her, 612 N)
        px, py, pz = float(self.push[0]), float(self.push[1]), float(self.push[2])
        pn = math.sqrt(px * px + py * py + pz * pz)
        if pn > K.TOUCH_N:
            px, py, pz = px / pn, py / pn, pz / pn
            k_ = fx * px + fy * py + cz * pz
            if k_ < 0.0:
                fx -= k_ * px; fy -= k_ * py; cz -= k_ * pz
        # her weight is always carried (608 N, within the cap); the correction (fx, fy, cz) is scaled down, as a whole, as far as the
        # cap and "never down" ask: lam, the largest share in [0, 1] with |(fx, fy, W + lam cz)| <= SUP_F_MAX and W + lam cz >= 0
        W = self.weight
        lam = 1.0
        f2 = fx * fx + fy * fy + cz * cz
        if f2 > 0.0 and fx * fx + fy * fy + (W + cz) ** 2 > self.fh_max2:
            b_ = W * cz
            lam = (-b_ + math.sqrt(b_ * b_ - f2 * (W * W - self.fh_max2))) / f2
        if W + lam * cz < 0.0:
            lam = W / -cz
        capped = lam < 1.0
        fx *= lam; fy *= lam; fz = W + lam * cz
        # HER TRUNK CARRIED: her pelvis, abdomen and chest each turned toward the turn her plan means for it (her pelvis's, and her
        # spine's two joints' targets composed on it), by a spring and a damper; together at most SUP_T_MAX (the sum of the three
        # torques' sizes). Her spine's muscles do their part within her strength; the support carries what they cannot (a push on her
        # chest past her spine's strength, 103 N m, folded her back 126 deg on her carried pelvis, A25b's build)
        qp = self.QR[s]
        qa = qmul(qp, self.QB[s][self.i_lumbar]); qch = qmul(qa, self.QB[s][self.i_thorax])
        wt = self.WR[s]
        tot = 0.0
        for k_, (bid, qt) in enumerate(((pel, qp), (self.abd, qa), (self.chest, qch))):
            qc = d.xquat[bid]
            Rc = quat2mat(qc)
            er = Rc @ qlog(qmul(qconj(qc), qt))                         # its turn to her plan (world frame)
            w_ = d.cvel[bid][:3]                                        # its angular velocity (world frame: MuJoCo's cvel)
            self.sup_t[k_] = K.SUP_KR * er + K.SUP_CR * (Rc @ wt - w_)
            tot += math.sqrt(float(self.sup_t[k_] @ self.sup_t[k_]))
        if tot > K.SUP_T_MAX:
            self.sup_t *= K.SUP_T_MAX / tot; capped = True
        tq = self.sup_t[0]
        if capped:
            self.sup_sat += 1
        cx, cy, cz = (d.subtree_com[pel] - d.xipos[pel]).tolist()       # the force at her whole body's centre of mass (MuJoCo's
                                                                        # xfrc_applied acts at the pelvis's own: the lever between)
        sup = self.sup
        sup[0] = fx; sup[1] = fy; sup[2] = fz
        sup[3] = float(tq[0]) + cy * fz - cz * fy
        sup[4] = float(tq[1]) + cz * fx - cx * fz
        sup[5] = float(tq[2]) + cx * fy - cy * fx
        d.xfrc_applied[pel] = sup
        d.xfrc_applied[self.abd, 3:] = self.sup_t[1]; d.xfrc_applied[self.chest, 3:] = self.sup_t[2]
