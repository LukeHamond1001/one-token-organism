"""THE PARENT'S BODY (docs/SIM_DESIGN.md 4.1, 4.2, A4, A25; the lead's decision of 2026-09-25: she is a body). The world's side only.

Her 16 segments are dynamic MuJoCo bodies in one tree (body/sim/make_g1room.py: a free pelvis; ball joints at the lumbar, thorax,
neck, shoulders, wrists, hips and ankles; hinges at the elbows, the forearms' pronation and the knees), with de Leva's female
masses at her body mass (parent_consts). Her planner (body/sim/parent_motion.py) makes the pose she means at each tick's end, as
parent_kin's segment frames; here that pose becomes her joints' targets, and every physics step her joints are driven toward
the targets between the tick's start and end by torques a woman's muscles could give:

  tau = clip( Kp (target - q) + Kd (target velocity - velocity) + her tone + her posture's feedforward + her holds' effort,
              her strength )

  - Kp: each joint's strength over SAT_DEG (it spends its whole strength 20 deg off its plan); Kd critical against the inertia it
    moves at her rest pose (MuJoCo's dof_M0);
  - her posture's feedforward: the torque that holds her own limbs and trunk against gravity as they are (MuJoCo's bias force on
    her joints: a person knows her own limbs' weight);
  - her tone: the error integrated over time (TONE_S) on each joint her plan holds still, within its strength, so a posture she
    means is held against its load (her body resting on her heels, a toy in her hand), as a person's muscle tone settles; a chain
    the child presses lets it go and builds no more of it (she never builds up against it), but for her legs', which carry her
    own weight through it and go on settling it (the posture's feedforward is MuJoCo's bias force from her free pelvis, which
    holds no weight she kneels or stands on; let go, she sank onto the child);
  - her holds' effort: each capped spring she holds the child by pushes the child's link with its force F at the held point, and
    her hand with -F at her grip (Newton), and her arm and trunk exert J^T F to carry it (parent_motion's holds);
  - her strength: parent_consts.STRENGTH, per joint and direction (a woman's), binding all of it together.

Her balance (her whole body's strategies, which joint springs alone do not carry out) is a capped horizontal spring and a capped
turning spring on her pelvis toward her plan (parent_consts.BAL_*): still, never upward, so her weight is on the floor through
her feet, knees and shins, and never more than a person's footing holds against a push; while her gait is carried (walking,
turning, kneeling down, getting up, shuffling) the same springs stiff and her weight carried, her floor contact off (a disclosed
limit of this build: no walking controller; a residual force on the pelvis, as Yuan and Kitani 2020's residual force control).
Nothing here is the child's.

A CHAIN STOPPED (A4: "that segment stops"): a chain (each arm, or the rest of her) whose contact with the child passed its cap for
2 steps holds where it is for the rest of the tick: its joints' targets become their positions at that step, their target
velocities zero, so she presses no further; what the child's push does to her then is her body's give.

EXACT DETERMINISM: pure numpy on the physics state; the tick's targets are plain numbers in her motion's state."""
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
        self.kp = 0.5 * (self.hi - self.lo) / math.radians(K.SAT_DEG)
        self.kd = 2.0 * K.DAMP_RATIO * np.sqrt(self.kp * m.dof_M0[self.dofs])
        self.nb, self.nh = len(BALLS), len(HINGES)
        self.chain_of_dof = np.array([CHAINS.index(JOINT_CHAIN[n]) for n, _ in BALLS for _k in range(3)]
                                     + [CHAINS.index(JOINT_CHAIN[n]) for n, _ in HINGES])
        self.ball_chain = np.array([CHAINS.index(JOINT_CHAIN[n]) for n, _ in BALLS])
        self.hinge_chain = np.array([CHAINS.index(JOINT_CHAIN[n]) for n, _ in HINGES])
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


def _dead(e, r):
    """e with a dead band of radius r taken off its length"""
    n = float(np.linalg.norm(e))
    return e * (1.0 - r / n) if n > r else e * 0.0


def _cap(e, c):
    n = float(np.linalg.norm(e))
    return e * (c / n) if n > c else e


class Drive:
    """her joints driven toward the tick's targets every physics step (see the module's doc). Its per-tick state (the tick's start
    and end targets, a stopped chain's hold, her balance's support) is plain numbers, in her motion's state"""

    def __init__(self, m, d, bmap):
        self.m, self.d, self.b = m, d, bmap
        m.dof_damping[bmap.dofs] = bmap.kd                               # her joints' damping, integrated implicitly by MuJoCo's
                                                                        # implicitfast (an explicit one at her joints' own inertia
                                                                        # rang: the W2 rebuild's first try saturated every step)
        self.t0 = self.t1 = None                                        # the tick's start and end targets
        self.support = 0.0                                              # her balance's share of her gait this tick (0 still, 1 moving)
        self.dead = 1.0                                                 # its dead bands' share (1 resting, 0 standing)
        self.carry = 1.0                                                # its gait's springs (0 from a push's stop to the tick's end)
        self.stop = [None, None, None]                                  # per chain: None, or the joints' positions it holds at
        self.effort = np.zeros(len(bmap.dofs))                          # her holds' effort this step (J^T F on her joints)
        self.tone = np.zeros(len(bmap.dofs))                            # her tone: each joint's error integrated (N m)
        self.ki = bmap.kp / K.TONE_S * m.opt.timestep                   # its gain a step ...
        self.tone_lo, self.tone_hi = K.TONE_SHARE * bmap.lo, K.TONE_SHARE * bmap.hi   # ... and its reach (a share of her strength)
        self.tau = np.zeros(len(bmap.dofs))                             # the torques applied on the last step (an instrument)
        self.bal = np.zeros(6)
        self.sat = 0                                                    # steps on which some joint was at her strength (instrument)
        self.weight = float(m.body_subtreemass[bmap.pelvis]) * float(-m.opt.gravity[2])

    def state(self):
        return dict(t0=tg_plain(self.t0), t1=tg_plain(self.t1), stop=[tg_plain(x) for x in self.stop], support=self.support,
                    dead=self.dead, carry=self.carry, tone=self.tone.tolist())

    def load_state(self, s):
        t0, t1 = tg_unplain(s.get("t0")), tg_unplain(s.get("t1"))
        sup, dead, carry = float(s.get("support", 0.0)), float(s.get("dead", 1.0)), float(s.get("carry", 1.0))
        if t0 is not None and t1 is not None:
            self.set_tick(t0, t1, sup, dead, carry)
        else:
            self.t0, self.t1, self.support, self.dead, self.carry = t0, t1, sup, dead, carry
        self.stop = [tg_unplain(x) for x in s.get("stop", [None, None, None])]
        self.tone = np.array(s.get("tone", np.zeros(len(self.b.dofs))), dtype=np.float64)

    def set_tick(self, t0, t1, support, dead=1.0, carry=1.0):
        """the tick's start and end targets (her plan at the last tick's end, and at this one's), her balance's support (0 still, 1
        her gait carried), its dead bands' share (1 kneeling or sitting, where her body rests as the physics leaves it; 0 standing,
        where it balances on her feet) and carry (1; 0 from the step a push of the child's stops her body, for the rest of that
        tick: her gait's stiff springs let go, her weight still carried; the next tick her plan is where the child pushed her,
        backing off away from it)"""
        self.t0, self.t1, self.support, self.dead, self.carry = t0, t1, float(support), float(dead), float(carry)
        self.stop = [None, None, None]
        b = self.b
        a = ((np.arange(STEPS) + 1.0) / STEPS)[:, None]                  # each step's share of the tick (its end)
        q1b = nlerp(t0[2], t1[2], 1.0)
        self.QB = nlerp(t0[2][None], q1b[None], a[:, :, None])           # (steps, balls, 4): the balls' targets a step
        self.H = t0[3][None] + a * (t1[3] - t0[3])[None]                  # (steps, hinges)
        self.PR = t0[0][None] + a * (t1[0] - t0[0])[None]                 # (steps, 3): the pelvis's place
        self.QR = nlerp(t0[1][None], t1[1][None], a)                      # (steps, 4): its turn
        wb = qlog(qmul(qconj(t0[2]), q1b)) / TICK_S                       # the balls' target velocities (their own frames)
        wh = (t1[3] - t0[3]) / TICK_S
        wt = np.concatenate([np.repeat(np.linalg.norm(wb, axis=1), 3), np.abs(wh)])
        self.KI = np.where(wt <= K.TONE_STILL_RPS, self.ki, 0.0)          # her tone settles on a joint her plan holds still, and
        if self.support > 0.0:                                          # never while her gait is carried (her body and legs let it
            self.KI[b.chain_of_dof == 2] = 0.0                          # go: they rest on the floor again from scratch)
            self.tone[b.chain_of_dof == 2] = 0.0
        self.FF = np.concatenate([(b.kd[:3 * b.nb].reshape(-1, 3) * wb).ravel(), b.kd[3 * b.nb:] * wh])   # her damping's target
        self.vr = (t1[0] - t0[0]) / TICK_S                                                              # velocity (a tick's)
        self.wr = qlog(qmul(qconj(t0[1]), nlerp(t0[1], t1[1], 1.0))) / TICK_S   # the pelvis's (its own frame)

    def hold_chain(self, c):
        """A4: chain c stops where it is for the rest of the tick: its targets become its joints' positions now, its target
        velocities zero, its tone let go"""
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
        self.relax(c)
        if c == 2:
            self.carry = 0.0                                            # her gait's stiff springs off at once

    def relax(self, c):
        """the child presses chain c: its tone is let go and builds no more this tick (she never builds up against it), but for her
        legs' (her legs carry her weight on the floor through it, and go on settling it: the posture's feedforward, MuJoCo's bias
        force from her free pelvis, holds none of it; let go, her kneeling body sank 37 cm onto the child's shoulder, and held at
        what it was, it sank as she leaned on: the physical build's findings, 2026-09-25)"""
        sel = (self.b.chain_of_dof == c) & ~self.b.leg_dof
        self.KI[sel] = 0.0
        self.tone[sel] = 0.0

    def step(self, s):
        """the torques of physics step s (before mj_step)"""
        d, b = self.d, self.b
        eb = qlog(qmul(qconj(d.qpos[b.ball_q]), self.QB[s]))              # each ball's error, in its own frame
        e = np.concatenate([eb.ravel(), self.H[s] - d.qpos[b.hinge_q]])
        self.tone = np.clip(self.tone + self.KI * e, self.tone_lo, self.tone_hi)
        tau = b.kp * e + self.FF + self.tone + d.qfrc_bias[b.dofs] + self.effort   # her damping itself is MuJoCo's (implicit)
        cl = np.clip(tau, b.lo, b.hi)
        if not np.array_equal(cl, tau):
            self.sat += 1
        self.tau = cl
        d.qfrc_applied[b.dofs] = cl
        pr, qr, vr, wr = self.PR[s], self.QR[s], self.vr, self.wr
        # HER BALANCE (parent_consts.BAL_*): still, a horizontal spring and a turning spring past their dead bands, capped low and
        # never vertical (her weight on the floor); moving (her gait: walking, turning, kneeling down, shuffling), the same
        # springs stiff and her weight carried
        u = self.support
        uh = u * self.carry                                             # her gait's stiff springs (off while the child presses her)
        p = d.qpos[b.root_q:b.root_q + 3]; v = d.qvel[b.root_d:b.root_d + 3]
        e = pr - p
        eh = _dead(e[:2], (1.0 - uh) * self.dead * K.BAL_DEAD_M)
        kh = K.BAL_K_STILL + uh * (K.BAL_K_MOVE - K.BAL_K_STILL)
        ch = 2.0 * math.sqrt(kh * K.BODY_MASS_KG)
        fh = _cap(kh * eh + ch * (vr[:2] - v[:2]), K.BAL_F_STILL + uh * (K.BAL_F_MOVE - K.BAL_F_STILL))
        fz = u * (self.weight + float(np.clip(K.BAL_K_MOVE * e[2] + 2.0 * math.sqrt(K.BAL_K_MOVE * K.BODY_MASS_KG) * (vr[2] - v[2]),
                                              -K.BAL_Z_MOVE, K.BAL_Z_MOVE)))
        qc = d.qpos[b.root_q + 3:b.root_q + 7]
        Rc = quat2mat(qc)
        er = _dead(Rc @ qlog(qmul(qconj(qc), qr)), (1.0 - uh) * self.dead * math.radians(K.BAL_DEAD_DEG))   # its turn to its plan
        kr = K.BAL_KR_STILL + uh * (K.BAL_KR_MOVE - K.BAL_KR_STILL)
        w = Rc @ d.qvel[b.root_d + 3:b.root_d + 6]
        tq = _cap(kr * er + K.BAL_CR * (Rc @ wr - w), K.BAL_T_STILL + uh * (K.BAL_T_MOVE - K.BAL_T_STILL))
        self.bal = np.array([fh[0], fh[1], fz, tq[0], tq[1], tq[2]])
        d.xfrc_applied[b.pelvis] = self.bal
